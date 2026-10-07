"""V10 benchmark-review API — human ground truth for the answer benchmark.

Endpoints (under /api/knowledge-bases):

* GET  /{kb_id}/answer-benchmark/review-packet        — whole benchmark:
  identity, per-question state, completeness, approval gate, frozen versions
* GET  /{kb_id}/answer-benchmark/questions/{qid}       — one question with its
  actual corpus evidence, authored ground truth and full review history
* POST /{kb_id}/answer-benchmark/questions/{qid}/ground-truth — append a
  human-authored reference answer (append-only)
* GET  /{kb_id}/answer-benchmark/questions/{qid}/ground-truth — its history
* POST /{kb_id}/answer-benchmark/questions/{qid}/reviews — append a human
  review (append-only; an 'approved' outcome is refused unless every required
  dimension was assessed affirmatively)
* GET  /{kb_id}/answer-benchmark/questions/{qid}/reviews — full review history
* POST /{kb_id}/answer-benchmark/freeze                — freeze an APPROVED
  benchmark into a new immutable version
* GET  /{kb_id}/answer-benchmark/versions              — frozen versions
* GET  /{kb_id}/answer-benchmark/versions/{vid}        — one frozen version
* POST /{kb_id}/answer-benchmark/versions/{vid}/verify — tamper check

Nothing here can edit a stored record: every write is an INSERT. A correction
is a new row with `supersedes` set, so the history of what was claimed is never
rewritten.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_repo
from app.api.routes_answer_eval import _resolve_benchmark_path
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.services.answer_eval.benchmark import BenchmarkValidationError
from app.services.answer_eval.benchmark_review import QuestionReview
from app.services.answer_eval.ground_truth import (
    AnswerBenchmarkVersion,
    GroundTruthAnnotation,
    GroundTruthError,
    ReviewCompleteness,
    verify_answer_benchmark_version,
)
from app.services.answer_eval.review_workflow import (
    AnswerBenchmarkReviewService,
    BenchmarkIdentity,
    EvidenceItem,
    ReviewPacket,
    ReviewQuestionSummary,
    ReviewWorkflowError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["benchmark-review"])

#: The shipped Automobile answer benchmark, as a convenience default. It is
#: never modified by this API: reviews and annotations are stored in the
#: database and freezing creates a NEW in-database version.
DEFAULT_BENCHMARK_PATH = "benchmarks/answer-quality-automobile-v1.json"


class GroundTruthAuthorRequest(BaseModel):
    """Body for POST .../ground-truth. Everything is the reviewer's input."""

    benchmark_path: str = DEFAULT_BENCHMARK_PATH
    author: str = Field(
        min_length=1,
        description="Identity of the human authoring the reference answer. Required.",
    )
    expected_answer: str = Field(default="", description="The human-written reference answer.")
    key_points: list[str] = Field(
        default_factory=list,
        description="Facts a correct answer must state. Each becomes a scorable unit.",
    )
    acceptable_answer_elements: list[str] = Field(
        default_factory=list,
        description="Alternative phrasings a human judged acceptable.",
    )
    evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunks the author actually read. Empty = the benchmark's own "
        "required evidence for the question.",
    )
    method: str = Field(default="", description="How the answer was derived (recorded in provenance).")
    note: str = ""
    supersedes: str = Field(default="", description="Annotation id this one corrects.")


class GroundTruthAuthorResponse(BaseModel):
    annotation: GroundTruthAnnotation
    problems: list[str] = Field(default_factory=list)
    stored: bool = True


class QuestionReviewRequest(BaseModel):
    """Body for POST .../reviews (append-only)."""

    benchmark_path: str = DEFAULT_BENCHMARK_PATH
    reviewer: str = Field(min_length=1, description="Identity of the human reviewer. Required.")
    outcome: str | None = Field(
        default=None,
        description="One of in_review | approved | rejected | ambiguous | "
        "insufficient_evidence. None = derive from the dimension verdicts.",
    )
    verdicts: dict[str, str] = Field(
        default_factory=dict,
        description="Dimension -> verdict. Dimensions: reference_answer, key_points, "
        "acceptable_elements, ambiguity, answerability, evidence_sufficiency. "
        "Verdicts: ok, ok_with_note, needs_revision, wrong, ambiguous, unknown.",
    )
    notes: dict[str, str] = Field(default_factory=dict, description="Optional per-dimension notes.")
    evidence_checked: list[str] = Field(
        default_factory=list, description="Chunk ids / documents the reviewer actually read."
    )
    annotation_id: str = Field(
        default="", description="The authored annotation this review judges, when known."
    )
    answerability_verdict: str | None = None
    unresolved: list[str] = Field(default_factory=list)
    supersedes: str = Field(default="", description="Review id this one corrects.")


class QuestionDetail(BaseModel):
    identity: BenchmarkIdentity
    summary: ReviewQuestionSummary
    evidence: list[EvidenceItem] = Field(default_factory=list)
    annotation_history: list[GroundTruthAnnotation] = Field(default_factory=list)
    effective_annotation: GroundTruthAnnotation | None = None
    review_history: list[QuestionReview] = Field(default_factory=list)
    effective_review: QuestionReview | None = None


class FreezeRequest(BaseModel):
    benchmark_path: str = DEFAULT_BENCHMARK_PATH
    frozen_by: str = Field(min_length=1, description="Identity of the human freezing this version.")
    notes: list[str] = Field(default_factory=list)


class FreezeRefused(BaseModel):
    frozen: bool = False
    reasons: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class VersionVerifyResponse(BaseModel):
    version_id: str
    intact: bool
    problems: list[str] = Field(default_factory=list)


def _service(repo: Repository) -> AnswerBenchmarkReviewService:
    return AnswerBenchmarkReviewService(repo, get_settings())


def _kb_or_404(repo: Repository, kb_id: str) -> None:
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")


def _question_or_404(benchmark, question_id: str) -> None:
    if not any(q.question_id == question_id for q in benchmark.questions):
        raise HTTPException(
            404,
            f"question {question_id!r} is not in benchmark {benchmark.benchmark!r}",
        )


def _load_benchmark(repo: Repository, kb_id: str, path: str):
    """Resolve + load the benchmark, enforcing that it targets THIS knowledge base."""
    resolved = _resolve_benchmark_path(path)
    service = _service(repo)
    try:
        benchmark = service.load(resolved)
    except ReviewWorkflowError as exc:
        raise HTTPException(400, str(exc)) from exc
    if benchmark.kb_id != kb_id:
        raise HTTPException(
            400,
            f"benchmark targets knowledge base {benchmark.kb_id!r}, not {kb_id!r}; "
            f"an answer benchmark is always reviewed against its own corpus",
        )
    return benchmark, resolved


@router.get("/{kb_id}/answer-benchmark/review-packet", response_model=ReviewPacket)
def review_packet(
    kb_id: str,
    benchmark_path: str = Query(default=DEFAULT_BENCHMARK_PATH),
    repo: Repository = Depends(get_repo),
):
    """The whole review surface: states, completeness, gate, frozen versions."""
    _kb_or_404(repo, kb_id)
    benchmark, resolved = _load_benchmark(repo, kb_id, benchmark_path)
    return _service(repo).packet(benchmark, resolved)


@router.get(
    "/{kb_id}/answer-benchmark/questions/{question_id}",
    response_model=QuestionDetail,
)
def question_detail(
    kb_id: str,
    question_id: str,
    benchmark_path: str = Query(default=DEFAULT_BENCHMARK_PATH),
    repo: Repository = Depends(get_repo),
):
    """One question + the ACTUAL corpus evidence + full human input history."""
    _kb_or_404(repo, kb_id)
    benchmark, resolved = _load_benchmark(repo, kb_id, benchmark_path)
    service = _service(repo)
    packet = service.packet(benchmark, resolved)
    summary = next(
        (q for q in packet.questions if q.question_id == question_id), None
    )
    if summary is None:
        raise HTTPException(
            404, f"question {question_id!r} is not in benchmark {benchmark.benchmark!r}"
        )
    annotations = service.scoped_annotations(kb_id, benchmark)
    q_annotations = [a for a in annotations if a.question_id == question_id]
    from app.services.answer_eval.ground_truth import GroundTruthLog

    log = GroundTruthLog(q_annotations)
    reviews = [
        r
        for r in service.scoped_reviews(kb_id, benchmark)
        if r.question_id == question_id
    ]
    effective = None
    if reviews:
        superseded = {r.supersedes for r in reviews if r.supersedes}
        active = [r for r in reviews if r.review_id not in superseded]
        effective = active[-1] if active else None

    return QuestionDetail(
        identity=packet.identity,
        summary=summary,
        evidence=service.evidence(benchmark, question_id, kb_id=kb_id),
        annotation_history=q_annotations,
        effective_annotation=log.effective_for_question(question_id),
        review_history=reviews,
        effective_review=effective,
    )


@router.post(
    "/{kb_id}/answer-benchmark/questions/{question_id}/ground-truth",
    response_model=GroundTruthAuthorResponse,
    status_code=201,
)
def author_ground_truth(
    kb_id: str,
    question_id: str,
    payload: GroundTruthAuthorRequest,
    repo: Repository = Depends(get_repo),
):
    """Append a human-authored reference answer (draft until reviewed)."""
    _kb_or_404(repo, kb_id)
    benchmark, _resolved = _load_benchmark(repo, kb_id, payload.benchmark_path)
    _question_or_404(benchmark, question_id)
    try:
        annotation, problems = _service(repo).author_annotation(
            benchmark,
            question_id=question_id,
            author=payload.author,
            expected_answer=payload.expected_answer,
            key_points=payload.key_points,
            acceptable_answer_elements=payload.acceptable_answer_elements,
            evidence_chunk_ids=payload.evidence_chunk_ids,
            note=payload.note,
            method=payload.method,
            supersedes=payload.supersedes,
        )
    except (GroundTruthError, ReviewWorkflowError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return GroundTruthAuthorResponse(annotation=annotation, problems=problems)


@router.get(
    "/{kb_id}/answer-benchmark/questions/{question_id}/ground-truth",
    response_model=list[GroundTruthAnnotation],
)
def ground_truth_history(
    kb_id: str,
    question_id: str,
    benchmark_path: str = Query(default=DEFAULT_BENCHMARK_PATH),
    repo: Repository = Depends(get_repo),
):
    """All annotations for a question, oldest first (never latest-wins)."""
    _kb_or_404(repo, kb_id)
    benchmark, _resolved = _load_benchmark(repo, kb_id, benchmark_path)
    annotations = _service(repo).scoped_annotations(kb_id, benchmark)
    return [a for a in annotations if a.question_id == question_id]


@router.post(
    "/{kb_id}/answer-benchmark/questions/{question_id}/reviews",
    response_model=QuestionReview,
    status_code=201,
)
def create_question_review(
    kb_id: str,
    question_id: str,
    payload: QuestionReviewRequest,
    repo: Repository = Depends(get_repo),
):
    """Append ONE human review of one benchmark question."""
    _kb_or_404(repo, kb_id)
    benchmark, _resolved = _load_benchmark(repo, kb_id, payload.benchmark_path)
    _question_or_404(benchmark, question_id)
    try:
        return _service(repo).record_review(
            benchmark,
            question_id=question_id,
            reviewer=payload.reviewer,
            verdicts=payload.verdicts,
            outcome=payload.outcome,
            notes=payload.notes,
            evidence_checked=payload.evidence_checked,
            annotation_id=payload.annotation_id,
            answerability_verdict=payload.answerability_verdict,
            unresolved=payload.unresolved,
            supersedes=payload.supersedes,
        )
    except (GroundTruthError, ReviewWorkflowError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get(
    "/{kb_id}/answer-benchmark/questions/{question_id}/reviews",
    response_model=list[QuestionReview],
)
def question_review_history(
    kb_id: str,
    question_id: str,
    benchmark_path: str = Query(default=DEFAULT_BENCHMARK_PATH),
    repo: Repository = Depends(get_repo),
):
    """Full review history for one question, oldest first."""
    _kb_or_404(repo, kb_id)
    benchmark, _resolved = _load_benchmark(repo, kb_id, benchmark_path)
    reviews = _service(repo).scoped_reviews(kb_id, benchmark)
    return [r for r in reviews if r.question_id == question_id]


@router.post("/{kb_id}/answer-benchmark/freeze")
def freeze_benchmark(
    kb_id: str,
    payload: FreezeRequest,
    repo: Repository = Depends(get_repo),
):
    """Freeze an APPROVED benchmark into a new immutable version.

    Returns 409 with the gate's own reasons when the benchmark is not
    approvable, so the caller always learns WHY rather than receiving a bare
    refusal.
    """
    _kb_or_404(repo, kb_id)
    benchmark, _resolved = _load_benchmark(repo, kb_id, payload.benchmark_path)
    service = _service(repo)
    try:
        version = service.freeze(
            benchmark, frozen_by=payload.frozen_by, notes=payload.notes
        )
    except GroundTruthError as exc:
        _application, completeness, gate, _a, _r = service.evaluate(benchmark)
        raise HTTPException(
            409,
            detail={
                "message": str(exc),
                "reasons": gate.reasons,
                "counts": gate.counts,
                "completeness": completeness.model_dump(mode="json"),
            },
        ) from exc
    except (ReviewWorkflowError, BenchmarkValidationError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return version


@router.get("/{kb_id}/answer-benchmark/versions")
def list_versions(kb_id: str, repo: Repository = Depends(get_repo)):
    """Frozen answer-benchmark versions for this KB, newest first."""
    _kb_or_404(repo, kb_id)
    return _service(repo).version_summaries(kb_id)


@router.get("/{kb_id}/answer-benchmark/versions/{version_id}")
def get_version(
    kb_id: str, version_id: str, repo: Repository = Depends(get_repo)
):
    """One frozen version, with its embedded labels and review metadata."""
    _kb_or_404(repo, kb_id)
    version: AnswerBenchmarkVersion | None = repo.get_answer_benchmark_version(
        kb_id, version_id
    )
    if version is None:
        raise HTTPException(404, "Frozen benchmark version not found")
    return version


@router.post(
    "/{kb_id}/answer-benchmark/versions/{version_id}/verify",
    response_model=VersionVerifyResponse,
)
def verify_version(
    kb_id: str, version_id: str, repo: Repository = Depends(get_repo)
):
    """Recompute a frozen version's fingerprints and report tampering."""
    _kb_or_404(repo, kb_id)
    version: AnswerBenchmarkVersion | None = repo.get_answer_benchmark_version(
        kb_id, version_id
    )
    if version is None:
        raise HTTPException(404, "Frozen benchmark version not found")
    problems = verify_answer_benchmark_version(version)
    return VersionVerifyResponse(
        version_id=version_id, intact=not problems, problems=problems
    )


@router.get("/{kb_id}/answer-benchmark/completeness", response_model=ReviewCompleteness)
def completeness_only(
    kb_id: str,
    benchmark_path: str = Query(default=DEFAULT_BENCHMARK_PATH),
    repo: Repository = Depends(get_repo),
):
    """Review completeness alone (V10 STEP 3), for dashboards."""
    _kb_or_404(repo, kb_id)
    benchmark, _resolved = _load_benchmark(repo, kb_id, benchmark_path)
    application, completeness, _gate, _a, _r = _service(repo).evaluate(benchmark)
    if not application.states:  # pragma: no cover - defensive
        raise HTTPException(500, "state application produced no questions")
    return completeness


__all__ = [
    "DEFAULT_BENCHMARK_PATH",
    "GroundTruthAuthorRequest",
    "GroundTruthAuthorResponse",
    "FreezeRequest",
    "QuestionDetail",
    "QuestionReviewRequest",
    "VersionVerifyResponse",
    "router",
]
