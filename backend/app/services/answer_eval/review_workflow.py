"""V10 review workflow service — the orchestration between benchmark, corpus,
append-only human input, and immutable frozen versions.

The routers stay thin and the domain rules stay in `ground_truth.py`; this
module owns the three things neither of them should:

* **Scoping.** Reviews and annotations are keyed by
  (kb_id, benchmark_name, benchmark_fingerprint). Two benchmarks on one
  knowledge base never share review state, and an annotation written against
  different benchmark content is reported as STALE rather than silently
  applied.
* **Evidence assembly.** The reviewer must be able to read the ACTUAL corpus
  evidence before approving, so this module returns chunk text plus full
  provenance (document, source, page/slide/section, hash) — and reports a
  missing chunk explicitly instead of omitting it.
* **Persistence.** Appending annotations/reviews and freezing versions. No
  update or delete path exists for any of them.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.repositories.sqlite_repo import Repository
from app.services.answer_eval.benchmark import (
    AnswerBenchmark,
    BenchmarkValidationError,
    load_answer_benchmark,
)
from app.services.answer_eval.benchmark_review import QuestionReview
from app.services.answer_eval.ground_truth import (
    ApprovalGateResult,
    ApprovalPolicy,
    AnswerBenchmarkVersion,
    GroundTruthAnnotation,
    GroundTruthError,
    QuestionState,
    QuestionStateApplication,
    ReviewCompleteness,
    apply_question_states,
    build_answer_benchmark_version,
    build_author_provenance,
    compute_completeness,
    evaluate_approval_gate,
    validate_annotation,
)
from app.services.answer_eval.service import AnswerEvaluationService
from app.utils.ids import new_id

logger = logging.getLogger(__name__)


class ReviewWorkflowError(RuntimeError):
    """A review-workflow rule was violated (bad path, frozen version, …)."""


class BenchmarkIdentity(BaseModel):
    """Identity of the benchmark under review — resolved from the artifact."""

    name: str
    version: int
    kb_id: str
    kb_name: str = ""
    path: str = ""
    fingerprint: str = ""
    lifecycle: str = "draft"
    human_review: str = "pending"
    derived_from: str = ""
    question_count: int = 0
    authorship: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    """One piece of corpus evidence for a question (CENTER panel)."""

    chunk_id: str
    required: bool = True
    found: bool = False
    document_id: str | None = None
    text: str = ""
    source_title: str | None = None
    #: The source title recorded on the BENCHMARK's evidence label. Kept
    #: separate from the chunk's own field rather than merged, so a reviewer
    #: can see when the two disagree.
    label_source_title: str | None = None
    source_url: str | None = None
    source_type: str | None = None
    publisher: str | None = None
    document_title: str | None = None
    section: str | None = None
    section_path: str | None = None
    page: int | None = None
    slide: int | None = None
    slide_title: str | None = None
    content_hash: str = ""
    char_count: int = 0
    problems: list[str] = Field(default_factory=list)


class ReviewQuestionSummary(BaseModel):
    """One row of the review packet — everything the LEFT panel needs."""

    question_id: str
    question: str
    subdomain: str = ""
    answerability: str = "unknown"
    state: str = QuestionState.PENDING.value
    reasons: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    review_count: int = 0
    reviewers: list[str] = Field(default_factory=list)
    has_annotation: bool = False
    annotation_id: str = ""
    annotation_author: str = ""
    expected_answer: str = ""
    key_points: list[str] = Field(default_factory=list)
    acceptable_answer_elements: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    artifact_reference_answer: str = ""
    artifact_key_points: list[str] = Field(default_factory=list)


class ReviewPacket(BaseModel):
    """The whole review surface for one benchmark (V10 STEP 3/5)."""

    identity: BenchmarkIdentity
    questions: list[ReviewQuestionSummary] = Field(default_factory=list)
    completeness: ReviewCompleteness
    gate: ApprovalGateResult
    policy: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    versions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnswerBenchmarkReviewService:
    """Orchestrates the V10 benchmark-review workflow."""

    def __init__(
        self,
        repo: Repository,
        settings: Settings,
        evaluation_service: AnswerEvaluationService | None = None,
    ) -> None:
        self._repo = repo
        self._settings = settings
        self._eval = evaluation_service or AnswerEvaluationService(repo, settings)

    # -- loading ------------------------------------------------------------

    def load(self, benchmark_path: str) -> AnswerBenchmark:
        try:
            return load_answer_benchmark(benchmark_path)
        except BenchmarkValidationError as exc:
            raise ReviewWorkflowError(str(exc)) from exc

    def identity(self, benchmark: AnswerBenchmark, path: str = "") -> BenchmarkIdentity:
        return BenchmarkIdentity(
            name=benchmark.benchmark,
            version=benchmark.version,
            kb_id=benchmark.kb_id,
            kb_name=benchmark.kb_name,
            path=path,
            fingerprint=benchmark.fingerprint(),
            lifecycle=benchmark.lifecycle.value,
            human_review=benchmark.human_review.value,
            derived_from=benchmark.derived_from,
            question_count=benchmark.question_count,
            authorship=dict(benchmark.authorship or {}),
        )

    # -- scoped human input -------------------------------------------------

    def scoped_annotations(
        self, kb_id: str, benchmark: AnswerBenchmark
    ) -> list[GroundTruthAnnotation]:
        """Annotations for THIS benchmark, appended oldest-first.

        Scoped by benchmark name; a fingerprint mismatch is left in place and
        surfaced as STALE by `apply_question_states`, never silently dropped.
        """
        return self._repo.list_ground_truth_annotations(
            kb_id, benchmark_name=benchmark.benchmark
        )

    def scoped_reviews(
        self, kb_id: str, benchmark: AnswerBenchmark
    ) -> list[QuestionReview]:
        return self._repo.list_benchmark_question_reviews(
            kb_id, benchmark_name=benchmark.benchmark
        )

    def chunk_index(self, kb_id: str) -> dict[str, dict]:
        return self._eval.chunk_index(kb_id)

    # -- state / completeness / gate ---------------------------------------

    def evaluate(
        self, benchmark: AnswerBenchmark, kb_id: str | None = None
    ) -> tuple[
        QuestionStateApplication,
        ReviewCompleteness,
        ApprovalGateResult,
        list[GroundTruthAnnotation],
        list[QuestionReview],
    ]:
        kb = kb_id or benchmark.kb_id
        annotations = self.scoped_annotations(kb, benchmark)
        reviews = self.scoped_reviews(kb, benchmark)
        index = self.chunk_index(kb)
        application = apply_question_states(benchmark, annotations, reviews, index)
        completeness = compute_completeness(benchmark, annotations, reviews, application)
        gate = evaluate_approval_gate(benchmark, application, completeness)
        return application, completeness, gate, annotations, reviews

    def packet(self, benchmark: AnswerBenchmark, path: str = "") -> ReviewPacket:
        application, completeness, gate, annotations, reviews = self.evaluate(benchmark)
        by_question: dict[str, list[GroundTruthAnnotation]] = {}
        for a in annotations:
            by_question.setdefault(a.question_id, []).append(a)

        questions: list[ReviewQuestionSummary] = []
        for q in benchmark.questions:
            q_annotations = by_question.get(q.question_id, [])
            log_effective = None
            if q_annotations:
                from app.services.answer_eval.ground_truth import GroundTruthLog

                log_effective = GroundTruthLog(q_annotations).effective_for_question(
                    q.question_id
                )
            q_reviews = [r for r in reviews if r.question_id == q.question_id]
            questions.append(
                ReviewQuestionSummary(
                    question_id=q.question_id,
                    question=q.question,
                    subdomain=q.subdomain,
                    answerability=q.answerability.value,
                    state=application.states.get(q.question_id, QuestionState.PENDING.value),
                    reasons=application.reasons.get(q.question_id, []),
                    problems=application.problems.get(q.question_id, []),
                    review_count=len(q_reviews),
                    reviewers=sorted({r.reviewer for r in q_reviews}),
                    has_annotation=log_effective is not None,
                    annotation_id=log_effective.annotation_id if log_effective else "",
                    annotation_author=log_effective.author if log_effective else "",
                    expected_answer=(
                        log_effective.expected_answer if log_effective else ""
                    ),
                    key_points=log_effective.key_points if log_effective else [],
                    acceptable_answer_elements=(
                        log_effective.acceptable_answer_elements if log_effective else []
                    ),
                    evidence_chunk_ids=(
                        [e.chunk_id for e in log_effective.evidence] if log_effective else []
                    ),
                    artifact_reference_answer=q.expected_answer or "",
                    artifact_key_points=list(q.key_points),
                )
            )

        warnings: list[str] = []
        if completeness.total and completeness.approved == 0:
            warnings.append(
                "no question has been approved by a human reviewer yet; the "
                "benchmark is not usable as an evaluation instrument"
            )
        if completeness.pending == completeness.total:
            warnings.append(
                "every question is PENDING: this benchmark has never been reviewed"
            )
        if gate.excluded_question_ids:
            warnings.append(
                f"{len(gate.excluded_question_ids)} question(s) are explicitly "
                f"classified as non-scoring; they are excluded from aggregate "
                f"metrics by policy, not silently dropped"
            )
        return ReviewPacket(
            identity=self.identity(benchmark, path),
            questions=questions,
            completeness=completeness,
            gate=gate,
            policy=gate.policy,
            versions=self.version_summaries(benchmark.kb_id),
            warnings=warnings,
        )

    # -- evidence -----------------------------------------------------------

    def evidence(
        self,
        benchmark: AnswerBenchmark,
        question_id: str,
        kb_id: str | None = None,
    ) -> list[EvidenceItem]:
        """The ACTUAL corpus evidence for a question, with full provenance.

        Uses the question's `required_evidence` labels as the starting point —
        the reviewer can cite additional chunks when authoring — and reports a
        missing or provenance-broken chunk explicitly (a `problems` entry),
        never by dropping it.
        """
        kb = kb_id or benchmark.kb_id
        question = next(
            (q for q in benchmark.questions if q.question_id == question_id), None
        )
        if question is None:
            raise ReviewWorkflowError(
                f"question {question_id!r} is not in benchmark {benchmark.benchmark!r}"
            )
        ids = [e.chunk_id for e in question.required_evidence]
        chunks = {c.id: c for c in self._repo.get_chunks_by_ids(kb, ids)} if ids else {}
        items: list[EvidenceItem] = []
        for ev in question.required_evidence:
            chunk = chunks.get(ev.chunk_id)
            if chunk is None:
                items.append(
                    EvidenceItem(
                        chunk_id=ev.chunk_id,
                        required=ev.required,
                        found=False,
                        document_id=ev.document_id,
                        section=ev.section,
                        source_title=ev.source_title,
                        label_source_title=ev.source_title,
                        content_hash=ev.content_hash,
                        problems=[
                            "this chunk is not in the live corpus; a reference "
                            "answer cannot be grounded in it"
                        ],
                    )
                )
                continue
            problems: list[str] = []
            if ev.content_hash and chunk.content_hash and ev.content_hash != chunk.content_hash:
                problems.append(
                    "content hash differs from the benchmark label (the corpus "
                    "was re-chunked since the benchmark was written)"
                )
            if ev.document_id and chunk.document_id != ev.document_id:
                problems.append(
                    f"chunk currently belongs to document {chunk.document_id}, "
                    f"benchmark says {ev.document_id}"
                )
            items.append(
                EvidenceItem(
                    chunk_id=chunk.id,
                    required=ev.required,
                    found=True,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    source_title=chunk.source_title,
                    label_source_title=ev.source_title,
                    source_url=chunk.source_url,
                    source_type=chunk.source_type,
                    publisher=chunk.publisher,
                    document_title=chunk.document_title,
                    section=chunk.section,
                    section_path=chunk.section_path,
                    page=chunk.page,
                    slide=chunk.slide,
                    slide_title=chunk.slide_title,
                    content_hash=chunk.content_hash,
                    char_count=chunk.char_count,
                    problems=problems,
                )
            )
        return items

    # -- append-only human input -------------------------------------------

    def author_annotation(
        self,
        benchmark: AnswerBenchmark,
        *,
        question_id: str,
        author: str,
        expected_answer: str = "",
        key_points: list[str] | None = None,
        acceptable_answer_elements: list[str] | None = None,
        evidence_chunk_ids: list[str] | None = None,
        note: str = "",
        method: str = "",
        supersedes: str = "",
    ) -> tuple[GroundTruthAnnotation, list[str]]:
        """Append a reference answer authored by a named human.

        Returns the stored annotation plus its validation problems (empty when
        valid). The problems are informational: a draft with problems is
        STORED and reported, but it can never be approved while they hold —
        approval requires a validated, corpus-grounded annotation.
        """
        question = next(
            (q for q in benchmark.questions if q.question_id == question_id), None
        )
        if question is None:
            raise ReviewWorkflowError(
                f"question {question_id!r} is not in benchmark {benchmark.benchmark!r}"
            )
        if not (author or "").strip():
            raise ReviewWorkflowError(
                "an annotation must name its human author; an unattributed "
                "reference answer cannot be distinguished from an agent draft"
            )

        known = {e.chunk_id: e for e in question.required_evidence}
        evidence = []
        for chunk_id in evidence_chunk_ids or list(known):
            if chunk_id in known:
                evidence.append(known[chunk_id])
            else:
                # A reviewer may cite a chunk beyond the benchmark's original
                # selection; record it with the hash the corpus currently has.
                chunk = next(
                    iter(self._repo.get_chunks_by_ids(benchmark.kb_id, [chunk_id])), None
                )
                evidence.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": chunk.document_id if chunk else None,
                        "content_hash": chunk.content_hash if chunk else "",
                        "section": chunk.section if chunk else None,
                        "source_title": chunk.source_title if chunk else None,
                    }
                )
        from app.services.answer_eval.benchmark import RequiredEvidence

        evidence_models = [
            e if isinstance(e, RequiredEvidence) else RequiredEvidence(**e)
            for e in evidence
        ]

        provenance = [
            build_author_provenance(
                author=author.strip(),
                method=method.strip()
                or "authored via the V10 benchmark review workflow, reading the "
                "cited corpus chunks",
                source=",".join(e.chunk_id for e in evidence_models),
                justification=note.strip(),
                data={
                    "question_id": question_id,
                    "evidence_count": len(evidence_models),
                },
            )
        ]

        annotation = GroundTruthAnnotation(
            annotation_id=new_id("bgt"),
            kb_id=benchmark.kb_id,
            benchmark_name=benchmark.benchmark,
            benchmark_fingerprint=benchmark.fingerprint(),
            question_id=question_id,
            author=author.strip(),
            expected_answer=expected_answer or "",
            key_points=[p for p in (key_points or []) if p.strip()],
            acceptable_answer_elements=[
                e for e in (acceptable_answer_elements or []) if e.strip()
            ],
            evidence=evidence_models,
            provenance=provenance,
            supersedes=supersedes,
            note=note or "",
        )
        problems = list(validate_annotation(annotation))
        problems.extend(
            self._provenance_problems(annotation, benchmark.kb_id)
        )
        self._repo.create_ground_truth_annotation(annotation)
        return annotation, problems

    def record_review(
        self,
        benchmark: AnswerBenchmark,
        *,
        question_id: str,
        reviewer: str,
        verdicts: dict[str, str],
        outcome: str | None = None,
        notes: dict[str, str] | None = None,
        evidence_checked: list[str] | None = None,
        annotation_id: str = "",
        answerability_verdict: str | None = None,
        unresolved: list[str] | None = None,
        supersedes: str = "",
        review_id: str | None = None,
    ) -> QuestionReview:
        """Append ONE human review of one benchmark question.

        Strict on purpose:

        * a named reviewer is required;
        * dimension keys and verdicts must come from the closed vocabularies;
        * an explicit `approved` outcome is REFUSED unless every required
          dimension was assessed affirmatively — a reviewer cannot approve a
          question with an unassessed ambiguity dimension by accident.

        Everything is append-only: a correction is a new review with
        `supersedes` set.
        """
        from app.services.answer_eval.benchmark_review import (
            DimensionVerdict,
            ReviewDimension,
        )

        question = next(
            (q for q in benchmark.questions if q.question_id == question_id), None
        )
        if question is None:
            raise ReviewWorkflowError(
                f"question {question_id!r} is not in benchmark {benchmark.benchmark!r}"
            )
        if not (reviewer or "").strip():
            raise ReviewWorkflowError(
                "a review must name its reviewer; an unattributed approval is "
                "indistinguishable from an agent approving its own labels"
            )

        parsed_verdicts: dict[str, DimensionVerdict] = {}
        for key, value in (verdicts or {}).items():
            try:
                dimension = ReviewDimension(key)
            except ValueError as exc:
                raise ReviewWorkflowError(
                    f"unknown review dimension {key!r}; expected one of "
                    f"{[d.value for d in ReviewDimension]}"
                ) from exc
            try:
                parsed_verdicts[dimension.value] = DimensionVerdict(value)
            except ValueError as exc:
                raise ReviewWorkflowError(
                    f"unknown verdict {value!r} for dimension {key!r}; expected "
                    f"one of {[v.value for v in DimensionVerdict]}"
                ) from exc

        if outcome is not None:
            try:
                declared_outcome = QuestionState(outcome)
            except ValueError as exc:
                raise ReviewWorkflowError(
                    f"unknown outcome {outcome!r}; expected one of "
                    f"{[s.value for s in QuestionState]}"
                ) from exc
            if declared_outcome is QuestionState.PENDING:
                raise ReviewWorkflowError(
                    "'pending' is not a review outcome: it means no review was "
                    "recorded. Use in_review, approved, rejected, ambiguous or "
                    "insufficient_evidence."
                )
            outcome = declared_outcome.value

        review = QuestionReview(
            review_id=review_id or new_id("bqrev"),
            kb_id=benchmark.kb_id,
            benchmark_name=benchmark.benchmark,
            benchmark_fingerprint=benchmark.fingerprint(),
            question_id=question_id,
            reviewer=reviewer.strip(),
            verdicts=parsed_verdicts,
            notes=dict(notes or {}),
            evidence_checked=list(evidence_checked or []),
            answerability_verdict=answerability_verdict,
            outcome=outcome,
            annotation_id=annotation_id,
            unresolved=list(unresolved or []),
            supersedes=supersedes,
        )
        if outcome == QuestionState.APPROVED.value:
            blockers = review.blocks_approval()
            if blockers:
                raise ReviewWorkflowError(
                    "an 'approved' outcome requires every required dimension to "
                    "be assessed affirmatively; blocking: " + "; ".join(blockers)
                )
        self._repo.create_benchmark_question_review(review)
        return review

    def _provenance_problems(
        self, annotation: GroundTruthAnnotation, kb_id: str
    ) -> list[str]:
        from app.services.answer_eval.ground_truth import (
            validate_annotation_against_corpus,
        )

        return validate_annotation_against_corpus(annotation, self.chunk_index(kb_id))

    # -- versions -----------------------------------------------------------

    def version_summaries(self, kb_id: str) -> list[dict[str, Any]]:
        return [
            {
                "version_id": v.version_id,
                "version": v.version,
                "benchmark_name": v.benchmark_name,
                "benchmark_fingerprint": v.benchmark_fingerprint,
                "ground_truth_fingerprint": v.ground_truth_fingerprint,
                "artifact_fingerprint": v.artifact_fingerprint,
                "created_at": v.created_at,
                "frozen_by": v.frozen_by,
                "scoring_question_count": len(v.scoring_question_ids()),
                "excluded_question_count": len(v.excluded_question_ids()),
            }
            for v in self._repo.list_answer_benchmark_versions(kb_id)
        ]

    def freeze(
        self,
        benchmark: AnswerBenchmark,
        *,
        frozen_by: str,
        notes: list[str] | None = None,
    ) -> AnswerBenchmarkVersion:
        """Freeze an APPROVED benchmark into a new immutable version.

        Refuses when the approval gate has not passed; the gate's own reasons
        are raised so the caller can show exactly what is missing.
        """
        if not (frozen_by or "").strip():
            raise ReviewWorkflowError("a frozen version must name who froze it")
        application, completeness, gate, annotations, reviews = self.evaluate(benchmark)
        if not gate.approved:
            raise GroundTruthError(
                "refusing to freeze: the approval gate has not passed. "
                + " | ".join(gate.reasons)
            )
        next_version = 1 + max(
            [v.version for v in self._repo.list_answer_benchmark_versions(benchmark.kb_id)]
            or [0]
        )
        version = build_answer_benchmark_version(
            benchmark=benchmark,
            annotations=annotations,
            reviews=reviews,
            application=application,
            gate=gate,
            frozen_by=frozen_by.strip(),
            version=next_version,
            version_id=new_id("abv"),
            notes=notes or [],
        )
        self._repo.create_answer_benchmark_version(version)
        return version


__all__ = [
    "AnswerBenchmarkReviewService",
    "BenchmarkIdentity",
    "EvidenceItem",
    "ReviewPacket",
    "ReviewQuestionSummary",
    "ReviewWorkflowError",
]
