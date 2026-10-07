"""Answer-quality evaluation API (V8).

Endpoints (under /api/knowledge-bases):
* POST /{kb_id}/answer-evaluation/runs   — run an evaluation, store the record
* GET  /{kb_id}/answer-evaluation/runs   — list runs (newest first)
* GET  /{kb_id}/answer-evaluation/runs/{run_id} — full record incl. per-question
* GET  /{kb_id}/answer-evaluation/runs/{run_id}/questions/{qid} — one question
* GET  /{kb_id}/answer-evaluation/compare — compare two runs, refusing to
  compare different question subsets
* POST/GET /{kb_id}/answer-evaluation/runs/{run_id}/questions/{qid}/reviews —
  append-only human answer review (V8 STEP 4): POST adds one attributed
  review, GET returns the whole history oldest-first (never latest-wins)

The run response surfaces `warnings` and `unknown_metrics` prominently: a run
whose correctness metric is UNKNOWN must not look like a run that scored zero
on correctness.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.api.deps import get_repo
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.retrieval import RetrievalParams
from app.utils.ids import new_id
from app.services.answer_eval.review import AnswerReview, ReviewLabel, ReviewVerdict
from app.services.answer_eval.run import (
    AnswerEvaluationConfig,
    AnswerEvaluationRun,
    aggregate_results,
)
from app.services.answer_eval.service import (
    AnswerEvaluationError,
    AnswerEvaluationService,
    BenchmarkCorpusMismatch,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["answer-evaluation"])
#: Top-level (cross-KB) answer-evaluation endpoints (V8 STEP 10). Run
#: CREATION stays KB-scoped — a benchmark always belongs to one corpus — but
#: reading runs by id or across projects needs a path that does not force the
#: caller to know the KB first.
global_router = APIRouter(prefix="/api", tags=["answer-evaluation"])


class RunAnswerEvaluationRequest(BaseModel):
    """Body for POST /{kb_id}/answer-evaluation/runs.

    `benchmark_path` is a repo-relative path. Paths outside the repository are
    rejected: an evaluation must run against a versioned benchmark artifact, not
    an arbitrary file.
    """

    benchmark_path: str = Field(
        default="benchmarks/answer-quality-automobile-v1.json",
        description=(
            "Repository-relative path to a versioned answer-benchmark artifact, "
            "e.g. 'benchmarks/answer-quality-automobile-v1.json'. Paths that "
            "escape the repository are rejected. Ignored when "
            "`benchmark_version_id` is set."
        ),
    )
    benchmark_version_id: str = Field(
        default="",
        description="V10: score an IMMUTABLE frozen answer-benchmark version "
        "(created by the benchmark review workflow) instead of a file. The "
        "version's approval gate, approval policy and label fingerprints are "
        "verified before the run starts.",
    )
    strategy: str = Field(default="", description="dense | bm25 | hybrid | hybrid_reranked")
    retrieval_params: RetrievalParams | None = None
    answer_mode: str = "abstain_if_unsupported"
    question_ids: list[str] | None = None
    limit: int | None = None
    official: bool = Field(
        default=False,
        description="Mark this run as an OFFICIAL answer-quality result. Official "
        "runs are refused unless the benchmark lifecycle is 'frozen'; default "
        "false keeps development runs clearly labelled.",
    )
    evaluator: str = Field(
        default="deterministic",
        description="Which answer evaluator to use: 'deterministic' (offline "
        "default; correctness stays UNKNOWN), 'reference' (V10: correctness "
        "from a human-reviewed reference answer / key points under the "
        "published `reference-key-point-coverage-v1` policy; requires an "
        "approved or frozen benchmark), 'human' (correctness from stored human "
        "answer reviews; fresh answers have none and stay UNKNOWN), 'llm' "
        "(LLM-as-judge; refused with 400 when no provider is configured — "
        "never substituted silently).",
    )
    persist: bool = Field(
        default=True,
        description=(
            "Store the ANSWER-EVALUATION RUN for later inspection. This does NOT "
            "make the evaluation read-only: every question asked goes through the "
            "normal answering path, which persists an Answer row (and its retrieval "
            "run/trace) against the knowledge base. Set persist=false to avoid "
            "storing the run record only."
        ),
    )


class CompareRunsResponse(BaseModel):
    comparable: bool
    verdict: str = Field(
        description="COMPARABLE | INCONCLUSIVE | NOT_COMPARABLE. INCONCLUSIVE "
        "= subsets differ; differences cover the shared questions only.",
    )
    mode: str = Field(description="identical | intersection | no_overlap")
    reason: str
    shared_question_ids: list[str] = Field(default_factory=list)
    left: dict
    right: dict
    differences: dict[str, dict] = Field(default_factory=dict)


class CreateAnswerReviewRequest(BaseModel):
    """Body for POST .../questions/{qid}/reviews (V8 STEP 4).

    Append-only: submitting twice creates two reviews. There is no edit or
    delete endpoint, so a previous judgement can never be overwritten.
    """

    reviewer: str = Field(
        min_length=1,
        description="Identity of the human reviewer. Required: an unattributed "
        "review is not evidence of human oversight.",
    )
    verdict: ReviewVerdict
    labels: list[ReviewLabel] = Field(default_factory=list)
    notes: str = Field(default="", max_length=4000)
    answer_id: str = ""

    @field_validator("reviewer")
    @classmethod
    def _reviewer_not_blank(cls, v: str) -> str:
        value = (v or "").strip()
        if not value:
            raise ValueError("reviewer identity must not be blank")
        return value


def _resolve_benchmark_path(raw: str) -> str:
    """Resolve a caller-supplied benchmark path safely.

    Callers normally address benchmarks repo-relative (``benchmarks/foo.json``)
    because that is how a versioned artifact is named and referenced. Two things
    must hold:

    * the resolved path stays inside an allowed root (no ``..`` escaping it);
    * the file actually exists.

    Allowed roots are the repository (versioned benchmarks) and the configured
    ``DATA_DIR`` (per-installation benchmarks, and what the test harness uses).
    Anything else is refused, so this endpoint cannot be used to read arbitrary
    files off the machine. The roots are derived from package/settings locations
    rather than the process CWD, so the same relative path works no matter where
    the server was started from.
    """
    rel = (raw or "").replace("\\", "/").strip()
    if not rel:
        raise HTTPException(400, "benchmark_path is required")

    # backend/app/api/routes_answer_eval.py -> repository root
    repo_root = Path(__file__).resolve().parents[3]
    candidate = (repo_root / rel).resolve()

    allowed_roots = [repo_root]
    try:
        allowed_roots.append(Path(get_settings().data_dir).resolve())
    except Exception:  # pragma: no cover - settings must not break path resolution
        logger.debug("could not resolve data_dir for benchmark path guard", exc_info=True)

    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            break
        except ValueError:
            continue
    else:
        raise HTTPException(
            400,
            "benchmark_path must stay inside the repository or the configured "
            "data directory",
        ) from None

    if not candidate.is_file():
        raise HTTPException(
            400,
            f"benchmark file not found: {rel} (resolved to {candidate})",
        )
    return str(candidate)


def _service(repo: Repository) -> AnswerEvaluationService:
    return AnswerEvaluationService(repo, get_settings())


def _build_evaluator(kind: str, repo: Repository, kb_id: str):
    """Explicit evaluator selection. Non-deterministic kinds fail loudly."""
    key = (kind or "deterministic").strip().lower()
    if key in ("", "deterministic", "heuristic", "offline"):
        return None  # service default
    if key in ("reference", "reference-labels", "reviewed-reference"):
        from app.services.answer_eval.evaluator import ReferenceAnswerEvaluator

        return ReferenceAnswerEvaluator()
    if key in ("human", "human-reviews"):
        from app.services.answer_eval.evaluator import HumanAnswerEvaluator

        def reviews_provider(question_id: str, answer_id: str):
            return [
                r
                for r in repo.list_answer_reviews(kb_id, question_id=question_id)
                if not r.answer_id or r.answer_id == answer_id
            ]

        return HumanAnswerEvaluator(reviews_provider)
    if key in ("llm", "llm-judge"):
        from app.llm.provider import create_llm_provider
        from app.services.answer_eval.evaluator import LLMAnswerEvaluator

        try:
            provider = create_llm_provider(get_settings())
        except Exception as exc:
            raise HTTPException(
                400,
                f"LLM evaluator unavailable: {exc}. The LLM judge is never "
                f"silently replaced by the deterministic evaluator.",
            ) from exc
        return LLMAnswerEvaluator(provider)
    raise HTTPException(
        400,
        f"unknown evaluator {kind!r}; supported: deterministic, reference, human, llm",
    )


def _summary(run: AnswerEvaluationRun) -> dict:
    """Compact run view. Unknown metrics stay explicitly unknown."""
    agg = run.aggregate
    return {
        "id": run.id,
        "created_at": run.created_at,
        "benchmark_name": run.benchmark_name,
        "benchmark_version": run.benchmark_version,
        "benchmark_fingerprint": run.benchmark_fingerprint,
        "benchmark_lifecycle": run.benchmark_lifecycle,
        "benchmark_version_id": run.benchmark_version_id,
        "benchmark_artifact_fingerprint": run.benchmark_artifact_fingerprint,
        "official": run.official,
        "strategy": run.strategy,
        "retrieval_params": run.retrieval_params,
        "generator": run.generator,
        "model": run.model,
        "is_mock": run.is_mock,
        "evaluator": f"{run.evaluator_name} {run.evaluator_version}",
        "evaluator_is_model_based": run.evaluator_is_model_based,
        "evaluator_detail": run.evaluator_detail,
        "entailment_provider": run.entailment_provider,
        "entailment_is_model_based": run.entailment_is_model_based,
        "relevance_method": run.relevance_method,
        "relevance_weight_source": run.relevance_weight_source,
        "question_count": run.question_count,
        "answerable_count": run.answerable_count,
        "unanswerable_count": run.unanswerable_count,
        "failed_question_ids": run.failed_question_ids,
        "unknown_metrics": agg.unknown_metrics,
        "warnings": run.warnings,
        "aggregate": agg.model_dump(mode="json"),
    }


@router.post("/{kb_id}/answer-evaluation/runs")
def create_run(
    kb_id: str,
    payload: RunAnswerEvaluationRequest,
    repo: Repository = Depends(get_repo),
):
    """Run an answer-quality evaluation. Returns the full immutable record."""
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")

    # -- V10: an immutable frozen version, or a working benchmark file ------
    frozen_version = None
    frozen_benchmark = None
    if payload.benchmark_version_id:
        from app.services.answer_eval.ground_truth import (
            verify_answer_benchmark_version,
        )

        frozen_version = repo.get_answer_benchmark_version(
            kb_id, payload.benchmark_version_id
        )
        if frozen_version is None:
            raise HTTPException(
                404,
                "Frozen benchmark version not found for this knowledge base",
            )
        problems = verify_answer_benchmark_version(frozen_version)
        if problems:
            raise HTTPException(
                409,
                "frozen benchmark version failed its integrity check and cannot "
                "be used: " + "; ".join(problems),
            )
        frozen_benchmark = frozen_version.benchmark
        path = f"db:answer-benchmark-versions/{frozen_version.version_id}"
    else:
        path = _resolve_benchmark_path(payload.benchmark_path)

    evaluator = _build_evaluator(payload.evaluator, repo, kb_id)
    service = AnswerEvaluationService(repo, get_settings(), evaluator=evaluator) if evaluator else _service(repo)

    if getattr(evaluator, "name", "") == "reference-labels":
        # The reference evaluator measures correctness FROM human-reviewed
        # labels. It is therefore refused unless the instrument has actually
        # been through the review workflow, so a draft benchmark's labels can
        # never be scored as if a human had approved them.
        try:
            check_benchmark = frozen_benchmark or service.load_benchmark(path)
        except AnswerEvaluationError as exc:
            raise HTTPException(400, str(exc)) from exc
        if check_benchmark.lifecycle.value not in ("approved", "frozen"):
            raise HTTPException(
                400,
                f"the reference evaluator requires a human-reviewed benchmark: "
                f"lifecycle is {check_benchmark.lifecycle.value!r}. Complete the "
                f"V10 review workflow (author reference answers, review, freeze) "
                f"or use evaluator='deterministic'.",
            )
        if not any(
            q.key_points or (q.expected_answer or "").strip()
            for q in check_benchmark.questions
        ):
            raise HTTPException(
                400,
                "the reference evaluator requires human-reviewed reference "
                "answers or key points; this benchmark carries none.",
            )

    config = AnswerEvaluationConfig(
        benchmark_path=path,
        strategy=payload.strategy,
        retrieval_params=payload.retrieval_params,
        answer_mode=payload.answer_mode,
        question_ids=payload.question_ids,
        limit=payload.limit,
        official=payload.official,
        benchmark_version_id=(frozen_version.version_id if frozen_version else ""),
        benchmark_artifact_fingerprint=(
            frozen_version.artifact_fingerprint if frozen_version else ""
        ),
    )
    try:
        run = service.run(benchmark_path=path, config=config, benchmark=frozen_benchmark)
    except BenchmarkCorpusMismatch as exc:
        raise HTTPException(409, str(exc)) from exc
    except AnswerEvaluationError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # The benchmark owns its KB; a run against a different KB is a caller error.
    if run.kb_id != kb_id:
        raise HTTPException(
            400,
            f"benchmark targets knowledge base {run.kb_id!r}, not {kb_id!r}; "
            f"an answer benchmark is always evaluated against its own corpus",
        )

    if payload.persist:
        service.save_run(run)
    return run


@router.get("/{kb_id}/answer-evaluation/runs")
def list_runs(kb_id: str, limit: int = 50, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return [_summary(r) for r in _service(repo).list_runs(kb_id, limit=limit)]


@router.get("/{kb_id}/answer-evaluation/runs/{run_id}")
def get_run(kb_id: str, run_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    run = _service(repo).get_run(kb_id, run_id)
    if run is None:
        raise HTTPException(404, "Answer evaluation run not found")
    return run


@router.get("/{kb_id}/answer-evaluation/runs/{run_id}/questions/{question_id}")
def get_question(
    kb_id: str, run_id: str, question_id: str, repo: Repository = Depends(get_repo)
):
    """Per-question detail for failure analysis."""
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    run = _service(repo).get_run(kb_id, run_id)
    if run is None:
        raise HTTPException(404, "Answer evaluation run not found")
    for result in run.per_question:
        if result.question_id == question_id:
            return result
    raise HTTPException(404, f"Question {question_id!r} is not in this run")


@router.post(
    "/{kb_id}/answer-evaluation/runs/{run_id}/questions/{question_id}/reviews",
    status_code=201,
)
def create_answer_review(
    kb_id: str,
    run_id: str,
    question_id: str,
    payload: CreateAnswerReviewRequest,
    repo: Repository = Depends(get_repo),
):
    """Record ONE human review of one evaluated answer (append-only)."""
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    run = _service(repo).get_run(kb_id, run_id)
    if run is None:
        raise HTTPException(404, "Answer evaluation run not found")
    result = next(
        (r for r in run.per_question if r.question_id == question_id), None
    )
    if result is None:
        raise HTTPException(404, f"Question {question_id!r} is not in this run")

    review = AnswerReview(
        id=new_id("areview"),
        kb_id=kb_id,
        run_id=run_id,
        question_id=question_id,
        answer_id=payload.answer_id or result.answer_id,
        reviewer=payload.reviewer,
        verdict=payload.verdict,
        labels=payload.labels,
        notes=payload.notes,
    )
    try:
        repo.create_answer_review(review)
    except Exception as exc:
        raise HTTPException(500, f"could not store review: {exc}") from exc
    return review


@router.get(
    "/{kb_id}/answer-evaluation/runs/{run_id}/questions/{question_id}/reviews"
)
def list_answer_reviews(
    kb_id: str,
    run_id: str,
    question_id: str,
    limit: int = Query(default=500, le=1000),
    repo: Repository = Depends(get_repo),
):
    """Full review history for one question, oldest first.

    Never collapses to "latest wins": previous reviews are preserved and all
    returned, so disagreement between reviewers is visible.
    """
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    run = _service(repo).get_run(kb_id, run_id)
    if run is None:
        raise HTTPException(404, "Answer evaluation run not found")
    if not any(r.question_id == question_id for r in run.per_question):
        raise HTTPException(404, f"Question {question_id!r} is not in this run")
    return repo.list_answer_reviews(
        kb_id, run_id=run_id, question_id=question_id, limit=limit
    )


@router.post("/{kb_id}/answer-evaluation/runs/{run_id}/human-evaluation")
def create_human_evaluation(
    kb_id: str,
    run_id: str,
    persist: bool = Query(
        default=True,
        description="Store the derived human-evaluation run. The SOURCE run and "
        "the reviews are never modified either way.",
    ),
    repo: Repository = Depends(get_repo),
):
    """Derive a NEW run whose correctness comes from this run's human reviews.

    Reviews and the source run are read-only here: the derived run is a new
    immutable record with its own id, noting the run it came from. Questions
    without reviews keep correctness UNKNOWN.
    """
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    service = _service(repo)
    if service.get_run(kb_id, run_id) is None:
        raise HTTPException(404, "Answer evaluation run not found")
    try:
        derived = service.derive_human_run(kb_id, run_id)
    except AnswerEvaluationError as exc:
        raise HTTPException(400, str(exc)) from exc
    if persist:
        service.save_run(derived)
    return derived


@router.get("/{kb_id}/answer-evaluation/compare")
def compare_runs(
    kb_id: str,
    left: str = Query(..., description="left run id"),
    right: str = Query(..., description="right run id"),
    repo: Repository = Depends(get_repo),
):
    """Compare two runs, refusing to compare different question subsets.

    Comparing runs over different question sets would attribute a difference in
    question mix to the retrieval strategy, so that comparison is blocked rather
    than caveated.
    """
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    service = _service(repo)
    a = service.get_run(kb_id, left)
    b = service.get_run(kb_id, right)
    if a is None or b is None:
        raise HTTPException(404, "One or both answer evaluation runs were not found")

    ok, reason = a.is_comparable_with(b)
    if ok and a.relevance_method != b.relevance_method:
        reason += (
            "; NOTE: relevance was measured with different methods "
            f"({a.relevance_method or 'none'} vs {b.relevance_method or 'none'}), "
            "so question_answer_relevance is not directly comparable between "
            "these runs"
        )
    plan = a.compare_with(b)
    if plan.mode == "identical":
        plan.reason = reason  # keep the stricter is_comparable wording + note
        left_agg, right_agg = a.aggregate, b.aggregate
    elif plan.mode == "intersection":
        shared = set(plan.question_ids)
        left_agg = aggregate_results(
            [r for r in a.per_question if r.question_id in shared]
        )
        right_agg = aggregate_results(
            [r for r in b.per_question if r.question_id in shared]
        )
    else:
        left_agg = right_agg = None

    diffs: dict[str, dict] = {}
    if left_agg is not None and right_agg is not None:
        for name in (
            "citation_precision",
            "citation_recall",
            "evidence_support_rate",
            "unsupported_claim_rate",
            "unsupported_claim_ratio",
            "supported_claim_ratio",
            "fabricated_citation_rate",
            "unsupported_citation_rate",
            "retrieval_hit_rate",
            "key_point_recall",
            "expected_information_coverage",
            "reference_answer_similarity",
            "correctness",
            "question_answer_relevance",
            "relevance_failure_rate",
            "abstention_accuracy",
            "grounding_state_accuracy",
            "pass_rate",
        ):
            left_metric = getattr(left_agg, name, None)
            right_metric = getattr(right_agg, name, None)
            if left_metric is None or right_metric is None:
                continue
            diffs[name] = {
                "left": left_metric.model_dump(mode="json"),
                "right": right_metric.model_dump(mode="json"),
                # Difference is only meaningful when BOTH sides were measured.
                "delta": (
                    round(right_metric.value - left_metric.value, 6)
                    if left_metric.measured and right_metric.measured
                    and left_metric.value is not None and right_metric.value is not None
                    else None
                ),
            }
    return CompareRunsResponse(
        comparable=plan.comparable,
        verdict=plan.verdict,
        mode=plan.mode,
        reason=plan.reason,
        shared_question_ids=plan.question_ids,
        left=_summary(a),
        right=_summary(b),
        differences=diffs,
    )


# ---------------------------------------------------------------------------
# Top-level endpoints (V8 STEP 10): read-only access across knowledge bases.
# ---------------------------------------------------------------------------


@global_router.get("/answer-evaluation-runs")
def list_all_answer_evaluation_runs(
    limit: int = Query(default=100, le=500),
    kb_id: str | None = Query(default=None, description="optional KB filter"),
    repo: Repository = Depends(get_repo),
):
    """Answer-evaluation runs across all knowledge bases, newest first."""
    return [
        _summary(r)
        for r in repo.list_answer_evaluation_runs_all(kb_id=kb_id, limit=limit)
    ]


@global_router.get("/answer-evaluation-runs/{run_id}")
def get_answer_evaluation_run(run_id: str, repo: Repository = Depends(get_repo)):
    """One answer-evaluation run by its global id."""
    run = repo.get_answer_evaluation_run_by_id(run_id)
    if run is None:
        raise HTTPException(404, "Answer evaluation run not found")
    return run