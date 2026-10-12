"""Retrieval-diagnostics API (V12) and controlled-experiment API (V13).

Endpoints (under /api/knowledge-bases):

* GET  /{kb_id}/retrieval-diagnostics/benchmark — the retrieval benchmark as the
  service sees it: identity, fingerprint, per-question expected evidence, and the
  AUTHORSHIP/human-review provenance copied verbatim.
* GET  /{kb_id}/retrieval-diagnostics/evidence-audit — read-only availability
  audit for the benchmark's expected evidence against the live corpus AND index,
  without running any strategy. Answers "is this even a retrieval question, or is
  the evidence missing / unindexed?" before anything else is measured.
* GET  /{kb_id}/retrieval-diagnostics/strategies — what can actually be run, with
  each strategy's declared requirements.
* POST /{kb_id}/retrieval-diagnostics/runs — run N strategies over the benchmark,
  classify every question, compare strategies on matched questions and persist
  ONE append-only diagnostic run.
* GET  /{kb_id}/retrieval-diagnostics/runs — run summaries, newest first.
* GET  /{kb_id}/retrieval-diagnostics/runs/{rid} — the full stored report.
* GET  /{kb_id}/retrieval-diagnostics/runs/{rid}/questions/{qid} — one question
  under every strategy in the run, with expected-vs-retrieved placement.
* GET  /{kb_id}/retrieval-diagnostics/runs/{rid}/recommendations — the stored
  recommendations (evidence-backed hypotheses, never applied automatically).
* GET  /{kb_id}/retrieval-diagnostics/runs/{rid}/comparisons — strategy
  comparisons on matched questions, with their pairing verdicts.
* GET  /{kb_id}/retrieval-diagnostics/taxonomy — the published outcome vocabulary.

Read-only with respect to the knowledge base, the vector index and both benchmark
artifacts exposed here: no endpoint creates, deletes or reindexes anything, and
none of them writes a retrieval configuration. A diagnostic run only INSERTs its
own report row.

This module is deliberately a THIN HTTP layer. Every model it returns and every
rule it applies lives in ``services.answer_eval.retrieval_diagnostics`` and
``services.answer_eval.retrieval_diagnostic_runner`` so that the CLI, the API and
the tests cannot disagree about what a metric means.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_repo
from app.api.routes_answer_eval import _resolve_benchmark_path
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.services.answer_eval.retrieval_diagnostic_runner import (
    DEFAULT_RETRIEVAL_BENCHMARK_PATH,
    BenchmarkValidationError,
    RetrievalDiagnosticRunner,
    load_retrieval_benchmark,
)
from app.services.answer_eval.retrieval_diagnostics import (
    ALL_RETRIEVAL_CODES,
    CRITICAL_CODES,
    RetrievalDiagnosticsReport,
)
from app.services.answer_eval.retrieval_experiment_models import (
    MEANINGFUL_DELTA,
    ExperimentOutcome,
    RetrievalExperiment,
    RetrievalStrategyConfig,
)
from app.services.answer_eval.retrieval_experiment_runner import (
    ExperimentValidationError,
    RetrievalExperimentRunner,
)
from app.services.retrieval.service import KnowledgeBaseNotFound, RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["retrieval-diagnostics"])


#: The single sentence every recommendation payload carries. It names the
#: epistemic status of the advice: a recommendation is a HYPOTHESIS, and only a
#: controlled experiment turns it into a result.
RECOMMENDATION_NOTICE = (
    "Every entry below is an evidence-backed hypothesis, not an established "
    "finding: it names the failing questions and the measurements behind it, and "
    "proposed_config_change is always empty because nothing is applied "
    "automatically. A controlled experiment must be run and its paired result "
    "recorded before the change is treated as an improvement."
)

COMPARISON_NOTICE = (
    "Only questions BOTH strategies measured are paired. A comparison whose "
    "pairing verdict is not COMPARABLE must not be read as a strategy result."
)


class ExperimentArmRequest(BaseModel):
    """One arm of a controlled experiment."""

    label: str = Field(
        default="",
        description="Names the arm in the report. Two arms of the same strategy (dense "
        "at k=3 and k=10) must have distinct labels.",
    )
    strategy: str = Field(default="dense", description="A registered retrieval strategy")
    top_k: int = Field(default=10, ge=1, le=200, description="Evaluation cut-off")
    candidate_depth: int | None = Field(
        default=None, ge=1, le=500, description="Probe depth; defaults to max(top_k, 20)"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Strategy parameters under test"
    )
    rationale: str = Field(default="", description="Why this arm is being tried")


def _arm_config(arm: ExperimentArmRequest, role: str) -> RetrievalStrategyConfig:
    return RetrievalStrategyConfig(
        label=arm.label or arm.strategy,
        strategy=arm.strategy,
        top_k=arm.top_k,
        candidate_depth=arm.candidate_depth,
        params=dict(arm.params),
        role=role,
        rationale=arm.rationale,
    )


class ExperimentRunRequest(BaseModel):
    """A controlled retrieval experiment, declared before it is measured.

    ``hypothesis``, ``primary_metric``, ``guardrail_metrics`` and
    ``rejection_conditions`` are the DECLARATION. They are stored on the artifact
    before any arm runs, so the bar cannot be chosen after seeing the result.
    """

    name: str = Field(..., description="Human label; also names the artifact file")
    hypothesis: str = Field(
        ...,
        description="The falsifiable claim being tested, in terms of the failure it "
        "addresses and the change it proposes",
    )
    baseline: ExperimentArmRequest = Field(
        default_factory=lambda: ExperimentArmRequest(strategy="dense", top_k=10, label="dense_k10"),
        description="The measured baseline every candidate is compared against",
    )
    candidates: list[ExperimentArmRequest] = Field(
        default_factory=list, description="The configurations under test"
    )
    primary_metric: str = Field(
        default="mrr",
        description="The single metric the hypothesis is about. Must have a declared "
        "direction in the comparison module or the request is refused.",
    )
    guardrail_metrics: list[str] = Field(
        default_factory=lambda: ["recall_at_k"],
        description="Metrics that must NOT regress. A gain on the primary metric does "
        "not license a regression here.",
    )
    proposed_change: str = ""
    expected_effect: str = ""
    targeted_failure_code: str = Field(
        default="", description="The V12 diagnostic code this experiment targets"
    )
    targeted_question_ids: list[str] = Field(default_factory=list)
    rejection_conditions: list[str] = Field(default_factory=list)
    benchmark_path: str = DEFAULT_RETRIEVAL_BENCHMARK_PATH
    question_ids: list[str] = Field(
        default_factory=list,
        description="Restrict every arm to these questions. Empty = all. Arms always "
        "see the SAME set, so the comparison stays paired.",
    )
    persist: bool = Field(
        default=True, description="Store the artifact so the result can be re-read"
    )


class DiagnosticsRunRequest(BaseModel):
    """Body for POST .../retrieval-diagnostics/runs. All fields optional."""

    benchmark_path: str = Field(
        default=DEFAULT_RETRIEVAL_BENCHMARK_PATH,
        description="Repo-relative path to the FROZEN retrieval benchmark. Read only.",
    )
    strategies: list[str] = Field(
        default_factory=lambda: ["dense"],
        description="Retrieval strategies to run. A strategy that is unavailable is "
        "recorded as FAILED with its error — never silently replaced.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Evaluation cut-off. The same k is used for the metric cut-off and "
        "the retrieval window, so recall@k is measured at the depth that produced it.",
    )
    candidate_depth: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="How deep to probe candidates. Defaults to max(top_k, 20) so that "
        "'retrieved below the cut-off' stays distinguishable from 'not retrieved'. "
        "Must be >= top_k.",
    )
    question_ids: list[str] = Field(
        default_factory=list, description="Subset of benchmark question ids. Empty = all."
    )
    persist: bool = Field(
        default=True,
        description="Store the report as an append-only diagnostic run so it can be "
        "re-read and compared later.",
    )
    persist_retrieval_runs: bool = Field(
        default=False,
        description="Also write one retrieval-run observability row per question "
        "(off by default: probing a KB is not a user query).",
    )


# ---------------------------------------------------------------------------
# Read-only inspection
# ---------------------------------------------------------------------------


@router.get("/{kb_id}/retrieval-diagnostics/benchmark")
def get_benchmark_identity(
    kb_id: str,
    benchmark_path: str = Query(default=DEFAULT_RETRIEVAL_BENCHMARK_PATH),
    repo: Repository = Depends(get_repo),
) -> dict[str, Any]:
    """The retrieval benchmark's identity, fingerprint and expected evidence."""
    _require_kb(repo, kb_id)
    resolved = _resolve_benchmark_path(benchmark_path)
    try:
        bench = load_retrieval_benchmark(resolved)
    except BenchmarkValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "benchmark": bench.benchmark,
        "version": bench.version,
        "created_at": bench.created_at,
        "kb_id": bench.kb_id,
        "kb_name": bench.kb_name,
        "relevance_model": bench.relevance_model,
        "authorship": bench.authorship,
        "fingerprint": bench.fingerprint(),
        "path": resolved,
        "question_count": len(bench.questions),
        "matches_this_kb": (not bench.kb_id) or bench.kb_id == kb_id,
        "questions": [
            {
                "id": q.id,
                "subdomain": q.subdomain,
                "question": q.question,
                "expected_chunk_ids": q.expected_chunk_ids,
                "expected_document_ids": q.expected_document_ids,
                "provenance": q.provenance,
            }
            for q in bench.questions
        ],
        "note": (
            "This artifact is READ-ONLY here. Retrieval diagnostics never modify a "
            "benchmark; its sha256 is captured before and after every run."
        ),
    }


@router.get("/{kb_id}/retrieval-diagnostics/evidence-audit")
def get_evidence_audit(
    kb_id: str,
    benchmark_path: str = Query(default=DEFAULT_RETRIEVAL_BENCHMARK_PATH),
    repo: Repository = Depends(get_repo),
) -> dict[str, Any]:
    """Is the expected evidence actually retrievable? Nothing is run, nothing is written.

    Separates the three cases that look identical in a metric: evidence MISSING
    from the corpus, evidence present but NOT INDEXED, and evidence indexed and
    therefore genuinely available for a retriever to fail on.
    """
    _require_kb(repo, kb_id)
    resolved = _resolve_benchmark_path(benchmark_path)
    try:
        bench = load_retrieval_benchmark(resolved)
    except BenchmarkValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    runner = RetrievalDiagnosticRunner(repo, get_settings())
    try:
        return runner.evidence_audit(kb_id, bench, benchmark_path=resolved)
    except KnowledgeBaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{kb_id}/retrieval-diagnostics/strategies")
def list_strategies(
    kb_id: str, repo: Repository = Depends(get_repo)
) -> dict[str, Any]:
    """What the registry can actually build, so the UI never offers a ghost option."""
    _require_kb(repo, kb_id)
    service = RetrievalService(repo, get_settings())
    return {
        "strategies": service.available_strategies(),
        "note": (
            "A strategy listed here is registered; whether its model or reranker is "
            "actually available is only known when it runs, and a failure is recorded "
            "as FAILED rather than substituted."
        ),
    }


@router.get("/{kb_id}/retrieval-diagnostics/taxonomy")
def get_taxonomy(kb_id: str, repo: Repository = Depends(get_repo)) -> dict[str, Any]:
    """The published outcome vocabulary, so the UI never invents a code."""
    _require_kb(repo, kb_id)
    return {
        "codes": [c.value for c in ALL_RETRIEVAL_CODES],
        "critical_codes": sorted(c.value for c in CRITICAL_CODES),
        "not_emitted": {
            "LOW_PRECISION": (
                "part of the published vocabulary but never emitted: binary relevance "
                "over necessity-only labels makes precision@k a lower bound, so a low "
                "value is not evidence of a retrieval fault. It is still reported as a "
                "measured metric."
            )
        },
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@router.post(
    "/{kb_id}/retrieval-diagnostics/runs",
    response_model=RetrievalDiagnosticsReport,
    status_code=201,
)
def create_diagnostics_run(
    kb_id: str,
    payload: DiagnosticsRunRequest,
    repo: Repository = Depends(get_repo),
) -> RetrievalDiagnosticsReport:
    """Run retrieval diagnostics over the benchmark and persist one report."""
    _require_kb(repo, kb_id)
    resolved = _resolve_benchmark_path(payload.benchmark_path)
    runner = RetrievalDiagnosticRunner(repo, get_settings())
    try:
        return runner.run(
            kb_id,
            strategies=payload.strategies,
            top_k=payload.top_k,
            benchmark_path=resolved,
            question_ids=payload.question_ids,
            persist=payload.persist,
            persist_retrieval_runs=payload.persist_retrieval_runs,
            candidate_depth=payload.candidate_depth,
        )
    except BenchmarkValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KnowledgeBaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{kb_id}/retrieval-diagnostics/runs")
def list_diagnostics_runs(
    kb_id: str,
    benchmark_name: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    repo: Repository = Depends(get_repo),
) -> list[dict[str, Any]]:
    """Run summaries, newest first. Never fabricates a run that was not stored."""
    _require_kb(repo, kb_id)
    out: list[dict[str, Any]] = []
    for report in repo.list_retrieval_diagnostic_runs(
        kb_id, benchmark_name=benchmark_name, limit=limit
    ):
        out.append(
            {
                "id": report.id,
                "kb_id": report.kb_id,
                "kb_name": report.kb_name,
                "created_at": report.created_at,
                "benchmark_name": report.benchmark_name,
                "benchmark_fingerprint": report.benchmark_fingerprint,
                "top_k": report.top_k,
                "question_count": report.question_count,
                "matched_question_ids": report.matched_question_ids,
                "strategies": [
                    {
                        "strategy": s.strategy,
                        "status": s.status,
                        "failure_reason": s.failure_reason,
                        "top_k": s.top_k,
                        "candidate_depth": s.candidate_depth,
                        "config_fingerprint": s.config_fingerprint,
                        "code_counts": s.summary.code_counts,
                        "recall_at_k": s.summary.recall_at_k.model_dump(mode="json"),
                        "precision_at_k": s.summary.precision_at_k.model_dump(mode="json"),
                        "mrr": s.summary.mrr.model_dump(mode="json"),
                        "ndcg_at_k": s.summary.ndcg_at_k.model_dump(mode="json"),
                        "latency_ms": s.summary.latency_ms.model_dump(mode="json"),
                    }
                    for s in report.strategies
                ],
                "protected_kb_unchanged": report.integrity.protected_kb_unchanged,
                "benchmark_unchanged": report.integrity.benchmark_unchanged,
                "limitations": report.limitations,
            }
        )
    return out


@router.get(
    "/{kb_id}/retrieval-diagnostics/runs/{run_id}",
    response_model=RetrievalDiagnosticsReport,
)
def get_diagnostics_run(
    kb_id: str, run_id: str, repo: Repository = Depends(get_repo)
) -> RetrievalDiagnosticsReport:
    """The full stored report, exactly as it was written."""
    return _load_run(repo, kb_id, run_id)


@router.get("/{kb_id}/retrieval-diagnostics/runs/{run_id}/questions/{question_id}")
def get_question_diagnostics(
    kb_id: str,
    run_id: str,
    question_id: str,
    repo: Repository = Depends(get_repo),
) -> dict[str, Any]:
    """One question under every strategy in the run, plus expected-vs-retrieved."""
    report = _load_run(repo, kb_id, run_id)
    per_strategy: list[dict[str, Any]] = []
    known = False
    for strategy in report.strategies:
        diagnostic = strategy.question(question_id)
        if diagnostic is None:
            per_strategy.append(
                {
                    "strategy": strategy.strategy,
                    "status": strategy.status,
                    "failure_reason": strategy.failure_reason,
                    "note": "this question has no measurement in this strategy",
                }
            )
            continue
        known = True
        per_strategy.append(
            {
                "strategy": strategy.strategy,
                "status": strategy.status,
                "top_k": strategy.top_k,
                "candidate_depth": strategy.candidate_depth,
                "classification": diagnostic.classification.model_dump(mode="json"),
                "availability_rollup": diagnostic.availability_rollup.value,
                "availability": [
                    a.model_dump(mode="json") for a in diagnostic.availability
                ],
                **diagnostic.expected_vs_retrieved(),
                "retrieved": [
                    item.model_dump(mode="json") for item in diagnostic.evidence.retrieved
                ],
                "probe": diagnostic.probe.model_dump(mode="json"),
                "metrics": {
                    "recall_at_k": diagnostic.evidence.recall_at_k.model_dump(mode="json"),
                    "precision_at_k": diagnostic.evidence.precision_at_k.model_dump(
                        mode="json"
                    ),
                    "mrr": diagnostic.evidence.mrr.model_dump(mode="json"),
                    "ndcg_at_k": diagnostic.evidence.ndcg_at_k.model_dump(mode="json"),
                    "required_recall_at_depth": (
                        diagnostic.evidence.required_recall_at_depth.model_dump(mode="json")
                    ),
                },
                "latency_ms": diagnostic.latency_ms,
                "warnings": diagnostic.evidence.warnings,
            }
        )
    if not known:
        raise HTTPException(404, f"Question {question_id!r} is not in this run")
    return {
        "run_id": run_id,
        "question_id": question_id,
        "benchmark_name": report.benchmark_name,
        "benchmark_fingerprint": report.benchmark_fingerprint,
        "top_k": report.top_k,
        "strategies": per_strategy,
    }


@router.get("/{kb_id}/retrieval-diagnostics/runs/{run_id}/recommendations")
def get_recommendations(
    kb_id: str, run_id: str, repo: Repository = Depends(get_repo)
) -> dict[str, Any]:
    """Stored evidence-backed recommendations for one diagnostic run."""
    report = _load_run(repo, kb_id, run_id)
    recs = report.recommendations
    return {
        "run_id": run_id,
        "recommendations": [
            {
                "action": rec.action.value,
                "reason": rec.reason,
                "experiment_required": rec.experiment_required,
                "strength": rec.strength.value,
                "expected_effect": rec.expected_effect,
                "risk": rec.risk,
                "proposed_config_change": rec.proposed_config_change,
                "thresholds_used": rec.thresholds_used,
                "supporting_measurements": rec.supporting_measurements,
                "affected_question_ids": rec.affected_question_ids,
            }
            for rec in recs.recommendations
        ],
        "total_questions": recs.total_questions,
        "questions_with_ground_truth": recs.questions_with_ground_truth,
        "questions_analysed": recs.questions_analysed,
        "data_gaps": recs.data_gaps,
        "notes": recs.notes,
        "notice": RECOMMENDATION_NOTICE,
    }


@router.get("/{kb_id}/retrieval-diagnostics/runs/{run_id}/comparisons")
def get_comparisons(
    kb_id: str, run_id: str, repo: Repository = Depends(get_repo)
) -> dict[str, Any]:
    """Strategy comparisons, each carrying its pairing verdict and caveats."""
    report = _load_run(repo, kb_id, run_id)
    return {
        "run_id": run_id,
        "comparisons": [c.model_dump(mode="json") for c in report.comparisons],
        "notice": COMPARISON_NOTICE,
    }


# ---------------------------------------------------------------------------
# V13 controlled experiments
# ---------------------------------------------------------------------------


@router.get("/{kb_id}/retrieval-experiments/meta")
def get_experiment_meta(
    kb_id: str, repo: Repository = Depends(get_repo)
) -> dict[str, Any]:
    """The experiment vocabulary a client must not invent.

    Exposes the derived outcome values, the declared meaningful-delta bars and
    the set of metrics that can be compared, so the UI offers only real options.
    """
    _require_kb(repo, kb_id)
    return {
        "outcomes": [o.value for o in ExperimentOutcome],
        "meaningful_delta": dict(MEANINGFUL_DELTA),
        "comparison_method": "paired delta on matched questions (V9 comparison module)",
        "promotion_policy": (
            "An experiment NEVER changes the default retrieval configuration. "
            "promoted_to_default is always False on a stored artifact."
        ),
        "notice": (
            "Outcomes are derived from explicit rules over measured values. The "
            "primary metric and its bar are declared on the artifact BEFORE the arms "
            "are measured, so the bar cannot be chosen after seeing the result."
        ),
    }


@router.post(
    "/{kb_id}/retrieval-experiments",
    response_model=RetrievalExperiment,
    status_code=201,
)
def create_retrieval_experiment(
    kb_id: str,
    payload: ExperimentRunRequest,
    repo: Repository = Depends(get_repo),
) -> RetrievalExperiment:
    """Run a controlled retrieval experiment and persist its artifact.

    Read-only with respect to the index: every shipped strategy retrieves without
    writing, so no scratch collection is needed and the protected KB is not
    touched. The artifact records the before/after integrity measurements that
    back that claim.
    """
    _require_kb(repo, kb_id)
    resolved = _resolve_benchmark_path(payload.benchmark_path)
    runner = RetrievalExperimentRunner(repo, get_settings())
    try:
        return runner.run(
            kb_id,
            name=payload.name,
            hypothesis=payload.hypothesis,
            baseline=_arm_config(payload.baseline, "baseline"),
            candidates=[_arm_config(c, "candidate") for c in payload.candidates],
            primary_metric=payload.primary_metric,
            guardrail_metrics=payload.guardrail_metrics,
            proposed_change=payload.proposed_change,
            expected_effect=payload.expected_effect,
            targeted_failure_code=payload.targeted_failure_code,
            targeted_question_ids=payload.targeted_question_ids,
            rejection_conditions=payload.rejection_conditions,
            benchmark_path=resolved,
            question_ids=payload.question_ids,
            persist=payload.persist,
        )
    except (BenchmarkValidationError, ExperimentValidationError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except KnowledgeBaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{kb_id}/retrieval-experiments")
def list_retrieval_experiments(
    kb_id: str, repo: Repository = Depends(get_repo)
) -> list[dict[str, Any]]:
    """Experiment artifacts for this KB, newest first."""
    _require_kb(repo, kb_id)
    runner = RetrievalExperimentRunner(repo, get_settings())
    return [
        {
            "id": e.id,
            "name": e.name,
            "kb_id": e.kb_id,
            "created_at": e.created_at,
            "completed_at": e.completed_at,
            "hypothesis": e.hypothesis,
            "primary_metric": e.primary_metric,
            "guardrail_metrics": e.guardrail_metrics,
            "targeted_failure_code": e.targeted_failure_code,
            "targeted_question_ids": e.targeted_question_ids,
            "benchmark_name": e.benchmark_name,
            "benchmark_fingerprint": e.benchmark_fingerprint,
            "question_set_fingerprint": e.question_set_fingerprint,
            "matched_question_count": len(e.matched_question_ids),
            "baseline": e.baseline.model_dump(mode="json"),
            "candidates": [c.model_dump(mode="json") for c in e.candidates],
            "outcome": e.outcome.value,
            "outcome_explanation": e.outcome_explanation,
            "accepted_candidate": e.accepted_candidate,
            "promoted_to_default": e.promoted_to_default,
            "candidate_outcomes": [
                {
                    "label": c.label,
                    "outcome": c.outcome.value,
                    "acceptance": c.acceptance.value,
                    "pairing": c.pairing,
                    "primary_mean_delta": c.primary_mean_delta,
                    "paired_question_count": c.paired_question_count,
                }
                for c in e.candidate_outcomes
            ],
            "integrity": e.integrity,
            "errors": e.errors,
        }
        for e in runner.list_experiments(kb_id)
    ]


@router.get(
    "/{kb_id}/retrieval-experiments/{experiment_id}",
    response_model=RetrievalExperiment,
)
def get_retrieval_experiment(
    kb_id: str, experiment_id: str, repo: Repository = Depends(get_repo)
) -> RetrievalExperiment:
    """The full experiment artifact, exactly as it was written."""
    _require_kb(repo, kb_id)
    runner = RetrievalExperimentRunner(repo, get_settings())
    experiment = runner.get_experiment(kb_id, experiment_id)
    if experiment is None:
        raise HTTPException(404, "retrieval experiment not found")
    return experiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_kb(repo: Repository, kb_id: str) -> None:
    if repo.get_kb(kb_id) is None:
        raise HTTPException(404, "Knowledge base not found")


def _load_run(
    repo: Repository, kb_id: str, run_id: str
) -> RetrievalDiagnosticsReport:
    _require_kb(repo, kb_id)
    report = repo.get_retrieval_diagnostic_run(kb_id, run_id)
    if report is None:
        raise HTTPException(404, "Retrieval diagnostics run not found")
    return report
