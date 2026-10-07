"""Retrieval diagnostics (V9 Phase 3) — why did retrieval fail for this question?

Per benchmark question this module exposes the retrieval evidence a reviewer
needs: the query, the strategy and parameters that produced it, the ranked
chunks with their scores and provenance, which of them were selected for
generation, and whether the evidence the BENCHMARK says is necessary was
retrieved at all.

## The rule this module exists to enforce

Ground truth is frequently absent. The shipped Automobile benchmark labels
required evidence for its questions, but a question with no label, an unfrozen
benchmark, or a retrieval run scored against a different corpus has NO ground
truth for recall/MRR/nDCG. In every one of those cases the metric is **UNKNOWN**
(``Measured.measured is False``) with a reason — never ``0.0``.

Zero and unknown are different findings: "the right passage was ranked last"
and "we have no idea which passage is right" must not render the same, because
only the first is a retrieval bug. Every metric here therefore carries
``measured``, ``reason`` and ``sample_size``.

The metric arithmetic itself is NOT reimplemented: ``recall_at_k``,
``precision_at_k``, ``mrr`` and ``ndcg_at_k`` are imported from
``services.evaluation.metrics`` so retrieval and answer evaluation can never
disagree about what MRR means.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.services.evaluation.metrics import (
    average_metrics,
    mrr as compute_mrr,
    ndcg_at_k as compute_ndcg_at_k,
    precision_at_k as compute_precision_at_k,
    recall_at_k as compute_recall_at_k,
)


class DiagnosticMetric(BaseModel):
    """One diagnostic metric that may legitimately be unknown.

    Mirrors ``answer_eval.metrics.Measured`` for the retrieval side: the same
    three-field contract (value / measured / reason) so answer and retrieval
    diagnostics read the same way in the UI.
    """

    value: float | None = None
    measured: bool = True
    reason: str = ""
    sample_size: int | None = None

    @classmethod
    def unknown(cls, reason: str) -> "DiagnosticMetric":
        return cls(value=None, measured=False, reason=reason)

    @classmethod
    def of(cls, value: float, reason: str = "", sample_size: int | None = None) -> "DiagnosticMetric":
        return cls(value=round(float(value), 6), measured=True, reason=reason, sample_size=sample_size)


class RetrievedItem(BaseModel):
    """One retrieved chunk at one rank, with full provenance for inspection."""

    rank: int = Field(description="1-indexed rank in the final ranked list")
    chunk_id: str
    score: float | None = Field(
        default=None,
        description="Final score as produced by the retriever. None when the "
        "retriever did not report one — never coerced to 0.0",
    )
    score_breakdown: dict[str, Any] = Field(default_factory=dict)

    document_id: str | None = None
    source_id: str | None = None
    source_title: str | None = None
    source_url: str | None = None
    source_type: str | None = None
    publisher: str | None = None
    section: str | None = None
    page: int | None = None
    trust_score: float | None = None
    content_hash: str | None = None

    #: True when the benchmark names this chunk as NECESSARY evidence.
    is_required: bool = False
    #: True when this chunk was actually passed to the generator as evidence.
    selected_for_generation: bool = False


class QuestionDiagnostic(BaseModel):
    """Retrieval diagnostics for one benchmark question."""

    question_id: str
    question: str = ""
    subdomain: str = ""
    answerability: str = ""

    strategy: str = ""
    top_k: int | None = None
    #: The retrieval parameters as recorded with the run (persisted verbatim).
    retrieval_params: dict[str, Any] = Field(default_factory=dict)

    retrieved: list[RetrievedItem] = Field(default_factory=list)
    retrieval_depth: int = Field(
        default=0, description="len(retrieved) — how many chunks the retrieval actually returned"
    )

    # -- ground truth -------------------------------------------------------
    required_chunk_ids: list[str] = Field(default_factory=list)
    required_document_ids: list[str] = Field(default_factory=list)
    ground_truth_available: bool = Field(
        default=False,
        description="True only when the benchmark names required evidence for this "
        "question. Every metric below is UNKNOWN when this is False.",
    )
    #: Rank of the first required chunk, 1-indexed. None = not retrieved, or no
    #: ground truth. `ground_truth_available` distinguishes those two cases.
    first_relevant_rank: int | None = None
    required_retrieved: bool | None = Field(
        default=None,
        description="True/False when ground truth exists; None when it does not.",
    )

    # -- generation input ---------------------------------------------------
    evidence_selected_ids: list[str] = Field(default_factory=list)
    required_selected_ids: list[str] = Field(default_factory=list)
    required_missing_from_generation: list[str] = Field(
        default_factory=list,
        description="Required evidence that WAS retrieved but was NOT selected for "
        "generation — an evidence-selection fault, not a retrieval fault.",
    )

    # -- metrics (UNKNOWN when ground truth is unavailable) -----------------
    recall_at_k: DiagnosticMetric = Field(default_factory=lambda: DiagnosticMetric.unknown("not computed"))
    precision_at_k: DiagnosticMetric = Field(default_factory=lambda: DiagnosticMetric.unknown("not computed"))
    mrr: DiagnosticMetric = Field(default_factory=lambda: DiagnosticMetric.unknown("not computed"))
    ndcg_at_k: DiagnosticMetric = Field(default_factory=lambda: DiagnosticMetric.unknown("not computed"))
    #: How many required chunks appear anywhere in the retrieved list.
    required_recall_at_depth: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("not computed")
    )

    latency_ms: float | None = Field(
        default=None, description="Total retrieval latency for this question, when measured"
    )
    stage_timings: dict[str, float] = Field(default_factory=dict)
    trace_id: str = ""
    warnings: list[str] = Field(default_factory=list)


class DiagnosticSummary(BaseModel):
    """Aggregate over question diagnostics — measured-only means, explicit gaps."""

    question_count: int = 0
    ground_truth_question_count: int = 0
    no_ground_truth_question_count: int = 0
    no_result_question_ids: list[str] = Field(default_factory=list)

    recall_at_k: DiagnosticMetric = Field(default_factory=lambda: DiagnosticMetric.unknown("no questions"))
    precision_at_k: DiagnosticMetric = Field(default_factory=lambda: DiagnosticMetric.unknown("no questions"))
    mrr: DiagnosticMetric = Field(default_factory=lambda: DiagnosticMetric.unknown("no questions"))
    ndcg_at_k: DiagnosticMetric = Field(default_factory=lambda: DiagnosticMetric.unknown("no questions"))
    required_recall_at_depth: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("no questions")
    )
    first_relevant_rank: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("no questions")
    )
    latency_ms: DiagnosticMetric = Field(default_factory=lambda: DiagnosticMetric.unknown("no questions"))

    #: Distribution of first-relevant rank into the buckets the recommendation
    #: engine reads. Only populated for questions WITH ground truth.
    first_relevant_rank_histogram: dict[str, int] = Field(default_factory=dict)
    unknown_metrics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _metric_from_optional(
    value: float | None,
    *,
    unavailable_reason: str,
    measured_reason: str,
    sample_size: int,
) -> DiagnosticMetric:
    """Wrap an ``evaluation.metrics`` optional result, keeping UNKNOWN distinct."""
    if value is None:
        return DiagnosticMetric.unknown(unavailable_reason)
    return DiagnosticMetric.of(value, reason=measured_reason, sample_size=sample_size)


def build_question_diagnostic(
    *,
    question_id: str,
    question: str = "",
    subdomain: str = "",
    answerability: str = "",
    strategy: str = "",
    top_k: int | None = None,
    retrieval_params: dict[str, Any] | None = None,
    ranked_chunk_ids: list[str] = None,
    chunk_index: dict[str, dict[str, Any]] | None = None,
    required_chunk_ids: list[str] | None = None,
    required_document_ids: list[str] | None = None,
    evidence_selected_ids: list[str] | None = None,
    latency_ms: float | None = None,
    stage_timings: dict[str, float] | None = None,
    trace_id: str = "",
) -> QuestionDiagnostic:
    """Build diagnostics for one question from a retrieval run + the benchmark.

    ``chunk_index`` maps chunk_id -> provenance record (the same shape
    ``validate_against_corpus`` consumes: document_id, content_hash, section,
    source_title, ...). A chunk with no index entry is reported with unknown
    provenance rather than a fabricated one.
    """
    ranked = list(ranked_chunk_ids or [])
    index = chunk_index or {}
    required = list(required_chunk_ids or [])
    required_set = set(required)
    selected = list(evidence_selected_ids or [])
    selected_set = set(selected)

    # Ground truth exists only when the benchmark actually names required
    # evidence. An empty required list on an UNANSWERABLE question is a real
    # choice, not missing data — but it still yields no recall target.
    ground_truth = bool(required)

    retrieved: list[RetrievedItem] = []
    for rank, cid in enumerate(ranked, start=1):
        meta = index.get(cid, {}) or {}
        retrieved.append(
            RetrievedItem(
                rank=rank,
                chunk_id=cid,
                score=meta.get("score"),
                score_breakdown=dict(meta.get("score_breakdown") or {}),
                document_id=meta.get("document_id"),
                source_id=meta.get("source_id"),
                source_title=meta.get("source_title"),
                source_url=meta.get("source_url"),
                source_type=meta.get("source_type"),
                publisher=meta.get("publisher"),
                section=meta.get("section"),
                page=meta.get("page"),
                trust_score=meta.get("trust_score"),
                content_hash=meta.get("content_hash"),
                is_required=cid in required_set,
                selected_for_generation=cid in selected_set,
            )
        )

    k = top_k if top_k and top_k > 0 else len(ranked)
    no_gt_reason = (
        "the benchmark names no required evidence for this question, so there is "
        "no reference list to score against — this is unknown, not zero"
    )

    if not ground_truth:
        first_rank = None
        required_retrieved: bool | None = None
        recall = DiagnosticMetric.unknown(no_gt_reason)
        precision = DiagnosticMetric.unknown(no_gt_reason)
        mrr_metric = DiagnosticMetric.unknown(no_gt_reason)
        ndcg_metric = DiagnosticMetric.unknown(no_gt_reason)
        depth_recall = DiagnosticMetric.unknown(no_gt_reason)
    else:
        first_rank = None
        for rank, cid in enumerate(ranked, start=1):
            if cid in required_set:
                first_rank = rank
                break
        required_retrieved = first_rank is not None
        recall = _metric_from_optional(
            compute_recall_at_k(ranked, required_set, k),
            unavailable_reason=no_gt_reason,
            measured_reason=f"required chunks in top-{k} over all required chunks",
            sample_size=1,
        )
        precision = _metric_from_optional(
            compute_precision_at_k(ranked, required_set, k),
            unavailable_reason=no_gt_reason,
            measured_reason=f"required chunks in top-{k} over k",
            sample_size=1,
        )
        mrr_metric = _metric_from_optional(
            compute_mrr(ranked, required_set),
            unavailable_reason=no_gt_reason,
            measured_reason="reciprocal rank of the first required chunk",
            sample_size=1,
        )
        ndcg_metric = _metric_from_optional(
            compute_ndcg_at_k(ranked, required_set, k),
            unavailable_reason=no_gt_reason,
            measured_reason=f"binary-relevance NDCG@{k} over required evidence",
            sample_size=1,
        )
        # Depth-independent recall: did the required evidence enter the
        # candidate set at ANY rank? Distinguishes "too deep" from "absent".
        depth_recall = DiagnosticMetric.of(
            len(set(ranked) & required_set) / len(required_set),
            reason="required chunks present anywhere in the retrieved candidate set",
            sample_size=len(required_set),
        )

    required_missing_from_generation = sorted(required_set - selected_set) if ground_truth else []

    warnings: list[str] = []
    if not ranked:
        warnings.append(
            "retrieval returned no chunks for this question; every metric below is "
            "unknown rather than zero because there was nothing to rank"
        )
    if ground_truth and required_set and not required_retrieved:
        warnings.append(
            f"none of the {len(required_set)} required-evidence chunk(s) were "
            f"retrieved within depth {len(ranked)} — a retrieval or corpus gap"
        )
    if ground_truth and selected and required_missing_from_generation:
        retrieved_but_unselected = sorted(
            (required_set - selected_set) & set(ranked)
        )
        if retrieved_but_unselected:
            warnings.append(
                f"{len(retrieved_but_unselected)} required chunk(s) were retrieved "
                f"but not selected for generation ({', '.join(retrieved_but_unselected)}) "
                f"— an evidence-SELECTION fault, distinct from a retrieval miss"
            )

    return QuestionDiagnostic(
        question_id=question_id,
        question=question,
        subdomain=subdomain,
        answerability=answerability,
        strategy=strategy,
        top_k=top_k,
        retrieval_params=dict(retrieval_params or {}),
        retrieved=retrieved,
        retrieval_depth=len(retrieved),
        required_chunk_ids=sorted(required_set),
        required_document_ids=sorted(set(required_document_ids or [])),
        ground_truth_available=ground_truth,
        first_relevant_rank=first_rank,
        required_retrieved=required_retrieved,
        evidence_selected_ids=sorted(selected_set),
        required_selected_ids=sorted(required_set & selected_set),
        required_missing_from_generation=required_missing_from_generation,
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr_metric,
        ndcg_at_k=ndcg_metric,
        required_recall_at_depth=depth_recall,
        latency_ms=latency_ms,
        stage_timings=dict(stage_timings or {}),
        trace_id=trace_id,
        warnings=warnings,
    )


def _rank_bucket(rank: int | None) -> str:
    """Bucket a first-relevant rank for the recommendation engine."""
    if rank is None:
        return "not_retrieved"
    if rank <= 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_3"
    if rank <= 5:
        return "rank_4_5"
    if rank <= 10:
        return "rank_6_10"
    if rank <= 20:
        return "rank_11_20"
    return "rank_21_plus"


def summarise_diagnostics(diagnostics: list[QuestionDiagnostic]) -> DiagnosticSummary:
    """Aggregate diagnostics over measured values only; gaps stay explicit."""
    summary = DiagnosticSummary(question_count=len(diagnostics))
    if not diagnostics:
        summary.warnings.append("no question diagnostics supplied")
        return summary

    gt = [d for d in diagnostics if d.ground_truth_available]
    summary.ground_truth_question_count = len(gt)
    summary.no_ground_truth_question_count = len(diagnostics) - len(gt)
    summary.no_result_question_ids = sorted(d.question_id for d in diagnostics if not d.retrieved)

    def mean_metric(name: str, values: list[float | None], *, reason_if_empty: str) -> DiagnosticMetric:
        cleaned = [v for v in values if v is not None]
        if not cleaned:
            return DiagnosticMetric.unknown(reason_if_empty)
        averaged = average_metrics(cleaned)
        assert averaged is not None  # cleaned is non-empty
        out = DiagnosticMetric.of(
            averaged,
            reason=f"mean over {len(cleaned)} question(s) with measured {name}",
            sample_size=len(cleaned),
        )
        skipped = len(values) - len(cleaned)
        if skipped:
            out.reason += f"; {skipped} question(s) excluded as unmeasured (NOT counted as 0)"
        return out

    no_gt = "no question has ground-truth reference evidence"
    summary.recall_at_k = mean_metric("recall", [d.recall_at_k.value if d.recall_at_k.measured else None for d in diagnostics], reason_if_empty=no_gt)
    summary.precision_at_k = mean_metric("precision", [d.precision_at_k.value if d.precision_at_k.measured else None for d in diagnostics], reason_if_empty=no_gt)
    summary.mrr = mean_metric("mrr", [d.mrr.value if d.mrr.measured else None for d in diagnostics], reason_if_empty=no_gt)
    summary.ndcg_at_k = mean_metric("ndcg", [d.ndcg_at_k.value if d.ndcg_at_k.measured else None for d in diagnostics], reason_if_empty=no_gt)
    summary.required_recall_at_depth = mean_metric(
        "required_recall_at_depth",
        [d.required_recall_at_depth.value if d.required_recall_at_depth.measured else None for d in diagnostics],
        reason_if_empty=no_gt,
    )
    summary.first_relevant_rank = mean_metric(
        "first_relevant_rank",
        [float(d.first_relevant_rank) for d in gt if d.first_relevant_rank is not None] or [None],
        reason_if_empty="no question retrieved its required evidence",
    )
    summary.latency_ms = mean_metric(
        "latency_ms",
        [d.latency_ms for d in diagnostics],
        reason_if_empty="no retrieval latency was recorded for these questions",
    )

    histogram: dict[str, int] = {}
    for d in gt:
        bucket = _rank_bucket(d.first_relevant_rank)
        histogram[bucket] = histogram.get(bucket, 0) + 1
    summary.first_relevant_rank_histogram = dict(sorted(histogram.items()))

    for name, metric in (
        ("recall_at_k", summary.recall_at_k),
        ("precision_at_k", summary.precision_at_k),
        ("mrr", summary.mrr),
        ("ndcg_at_k", summary.ndcg_at_k),
        ("required_recall_at_depth", summary.required_recall_at_depth),
        ("first_relevant_rank", summary.first_relevant_rank),
        ("latency_ms", summary.latency_ms),
    ):
        if not metric.measured:
            summary.unknown_metrics.append(name)

    if summary.no_ground_truth_question_count:
        summary.warnings.append(
            f"{summary.no_ground_truth_question_count} of {summary.question_count} "
            f"question(s) have no benchmark reference evidence; their retrieval "
            f"metrics are UNKNOWN and are excluded from the means above rather "
            f"than counted as zero"
        )
    return summary


__all__ = [
    "DiagnosticMetric",
    "DiagnosticSummary",
    "QuestionDiagnostic",
    "RetrievedItem",
    "build_question_diagnostic",
    "summarise_diagnostics",
]
