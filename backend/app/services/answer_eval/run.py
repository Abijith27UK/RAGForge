"""Answer-evaluation runs: immutable, reproducible records (Phase 7/8).

A run answers "was this answer correct, supported, correctly cited, and
appropriately confident?" against a fixed question set. Three rules make the
numbers trustworthy:

1. **Runs are immutable.** Nothing overwrites an existing run; a new evaluation
   is a new row with its own id. Two runs over the same benchmark are directly
   comparable because the run records the benchmark fingerprint.
2. **The question subset is recorded explicitly.** Every run stores the exact
   `question_ids` it covered and the count, so a comparison across strategies can
   never silently mix different subsets. `comparable_with` refuses to treat runs
   over different question sets as comparable.
3. **Producers are recorded.** Generator/provider, evaluator name+version and
   the entailment provider are all stored. A metric without its producer is not
   reproducible, so this is not optional metadata.

Coverage (retrieval quality) and citation quality are kept as separate metrics.
A retrieval miss is reported as a retrieval miss and never folded into the
answering score.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.answer import AnswerStatus, GroundingState
from app.schemas.retrieval import RetrievalParams
from app.services.answer_eval.benchmark import AnswerBenchmark, AnswerBenchmarkQuestion
from app.services.answer_eval.evaluator import AnswerEvaluator
from app.services.answer_eval.metrics import (
    AnswerQualityAggregate,
    AnswerQualityResult,
    Measured,
    average_measured,
    grounding_states_compatible,
)
from app.services.answer_eval.relevance import METHOD_PLAIN

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnswerEvaluationConfig(BaseModel):
    """How to run an answer evaluation. Mirrors the retrieval evaluation config."""

    benchmark_path: str = ""
    question_ids: list[str] | None = Field(
        default=None, description="None = every question in the benchmark"
    )
    strategy: str = ""
    retrieval_params: RetrievalParams | None = None
    answer_mode: str = "abstain_if_unsupported"
    limit: int | None = None
    official: bool = Field(
        default=False,
        description="True only when the caller is producing an OFFICIAL result. "
        "Official runs require a FROZEN answer benchmark; development runs "
        "record official=false so draft numbers are never mistakable for them.",
    )
    # -- V10: evaluating a FROZEN, human-reviewed benchmark version -----------
    # When set, the run was scored against an immutable frozen version whose
    # labels a human approved. Recording the version id keeps two runs over
    # different frozen label sets from being treated as comparable.
    benchmark_version_id: str = ""
    benchmark_artifact_fingerprint: str = ""


class RunComparisonPlan(BaseModel):
    """How two runs may be compared (V8 STEP 11).

    Three outcomes, never collapsed:

    COMPARABLE     identical question sets and benchmark content; full
                   aggregate comparison is valid.
    INCONCLUSIVE   question subsets differ but overlap; differences are
                   computed on the SHARED questions only, and the verdict says
                   INCONCLUSIVE so a partial overlap is never presented as a
                   full-set comparison.
    NOT_COMPARABLE zero overlap, or the benchmark content itself changed;
                   no differences are reported at all.
    """

    verdict: str  # COMPARABLE | INCONCLUSIVE | NOT_COMPARABLE
    mode: str  # identical | intersection | no_overlap
    reason: str
    #: Question ids the comparison may use (shared ids for intersection).
    question_ids: list[str] = Field(default_factory=list)

    @property
    def comparable(self) -> bool:
        return self.verdict == "COMPARABLE"


class AnswerEvaluationRun(BaseModel):
    """Immutable observability record for one answer-quality evaluation."""

    id: str
    kb_id: str
    created_at: datetime = Field(default_factory=_utcnow)

    # -- what was evaluated -------------------------------------------------
    benchmark_name: str = ""
    benchmark_version: int = 1
    benchmark_fingerprint: str = ""
    benchmark_path: str = ""
    benchmark_lifecycle: str = ""
    #: V10: the frozen benchmark version this run scored, when applicable.
    benchmark_version_id: str = ""
    benchmark_artifact_fingerprint: str = ""
    official: bool = False
    kb_version: int | None = None
    corpus_version: str | None = None
    corpus_fingerprint: str | None = None

    # -- how it was produced ------------------------------------------------
    strategy: str = ""
    retrieval_params: dict[str, Any] = Field(default_factory=dict)
    retrieval_run_ids: list[str] = Field(default_factory=list)
    answer_mode: str = ""

    # -- producers (reproducibility) ----------------------------------------
    generator: str = ""
    model: str = ""
    is_mock: bool = False
    prompt_version: str = ""
    answerer_version: str = ""
    evaluator_name: str = ""
    evaluator_version: str = ""
    #: True when the evaluator is a model (LLM-as-judge). Persisted so a run
    #: whose correctness came from a model is never indistinguishable from a
    #: deterministic one (V8 criterion 13).
    evaluator_is_model_based: bool = False
    evaluator_detail: str = ""
    entailment_provider: str = ""
    entailment_is_model_based: bool = False
    #: Relevance method(s) used across the run's questions, and where the term
    #: weights came from ("uniform" = no corpus IDF statistics, plain fallback).
    relevance_method: str = ""
    relevance_weight_source: str = ""

    # -- question subset (comparability) -------------------------------------
    question_ids: list[str] = Field(default_factory=list)
    question_count: int = 0
    answerable_count: int = 0
    unanswerable_count: int = 0
    abstention_expected_count: int = 0
    subset_note: str = ""

    # -- results -------------------------------------------------------------
    aggregate: AnswerQualityAggregate = Field(default_factory=AnswerQualityAggregate)
    per_question: list[AnswerQualityResult] = Field(default_factory=list)
    failed_question_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def is_comparable_with(self, other: "AnswerEvaluationRun") -> tuple[bool, str]:
        """Can these two runs be compared side by side?"""
        if set(self.question_ids) != set(other.question_ids):
            only_self = sorted(set(self.question_ids) - set(other.question_ids))
            only_other = sorted(set(other.question_ids) - set(self.question_ids))
            return False, (
                f"different question subsets ({len(self.question_ids)} vs "
                f"{len(other.question_ids)}). Only in this run: {only_self[:5]}; "
                f"only in the other: {only_other[:5]}. Comparing these would "
                f"attribute a difference in question mix to the strategy."
            )
        if self.benchmark_version_id != other.benchmark_version_id:
            # V10: two runs over DIFFERENT frozen versions may share a content
            # fingerprint (labels are excluded from it), so the version id is
            # compared as well. A frozen run is never treated as comparable to
            # a draft run: they are different instruments.
            return False, (
                f"different benchmark version "
                f"({self.benchmark_version_id or 'working draft'} vs "
                f"{other.benchmark_version_id or 'working draft'})"
            )
        if self.benchmark_fingerprint != other.benchmark_fingerprint:
            return False, (
                f"different benchmark content (fingerprint {self.benchmark_fingerprint} "
                f"vs {other.benchmark_fingerprint})"
            )
        return True, "same question subset and benchmark content"

    def compare_with(self, other: "AnswerEvaluationRun") -> RunComparisonPlan:
        """Intersection-aware comparison plan (V8 STEP 11).

        `is_comparable_with` stays strict — it answers "are these the same
        experiment?". This method answers the follow-up question "what CAN be
        compared?" without ever letting a partial overlap masquerade as a
        full-set comparison: differing subsets yield an INCONCLUSIVE verdict
        over the shared questions only, and zero overlap yields nothing at
        all.
        """
        ok, reason = self.is_comparable_with(other)
        if ok:
            return RunComparisonPlan(
                verdict="COMPARABLE",
                mode="identical",
                reason=reason,
                question_ids=list(self.question_ids),
            )
        if self.benchmark_fingerprint != other.benchmark_fingerprint:
            shared_ids = set(self.question_ids) & set(other.question_ids)
            if not shared_ids:
                only_self = sorted(set(self.question_ids) - set(other.question_ids))
                only_other = sorted(set(other.question_ids) - set(self.question_ids))
                return RunComparisonPlan(
                    verdict="NOT_COMPARABLE",
                    mode="no_overlap",
                    reason=(
                        f"zero common questions between the runs; no metric can "
                        f"be compared. Only in this run: {only_self[:5]}; only "
                        f"in the other: {only_other[:5]}"
                    ),
                )
            return RunComparisonPlan(
                verdict="NOT_COMPARABLE",
                mode="no_overlap",
                reason=(
                    f"benchmark content differs (fingerprint "
                    f"{self.benchmark_fingerprint} vs {other.benchmark_fingerprint}); "
                    f"metrics from different benchmark versions are not "
                    f"comparable even when question ids coincide"
                ),
            )
        shared = sorted(set(self.question_ids) & set(other.question_ids))
        if not shared:
            only_self = sorted(set(self.question_ids) - set(other.question_ids))
            only_other = sorted(set(other.question_ids) - set(self.question_ids))
            return RunComparisonPlan(
                verdict="NOT_COMPARABLE",
                mode="no_overlap",
                reason=(
                    f"zero common questions between the runs; no metric can be "
                    f"compared. Only in this run: {only_self[:5]}; only in the "
                    f"other: {only_other[:5]}"
                ),
            )
        return RunComparisonPlan(
            verdict="INCONCLUSIVE",
            mode="intersection",
            reason=(
                f"question subsets differ ({len(self.question_ids)} vs "
                f"{len(other.question_ids)}); differences below are computed "
                f"on the {len(shared)} shared question(s) ONLY. This is "
                f"INCONCLUSIVE for the full sets — a coverage difference must "
                f"not be attributed to the strategy"
            ),
            question_ids=shared,
        )


def select_questions(
    benchmark: AnswerBenchmark, config: AnswerEvaluationConfig
) -> list[AnswerBenchmarkQuestion]:
    """Resolve the exact question subset, recording anything excluded."""
    wanted = set(config.question_ids) if config.question_ids else None
    selected = [q for q in benchmark.questions if wanted is None or q.question_id in wanted]
    if wanted:
        unknown = sorted(wanted - {q.question_id for q in benchmark.questions})
        if unknown:
            raise ValueError(
                f"question ids not present in benchmark {benchmark.benchmark!r}: {unknown[:10]}"
            )
    if config.limit is not None:
        selected = selected[: config.limit]
    return selected


def aggregate_results(results: list[AnswerQualityResult]) -> AnswerQualityAggregate:
    """Aggregate per-question results, keeping unknowns visible."""
    agg = AnswerQualityAggregate(
        question_count=len(results),
        answerable_count=sum(1 for r in results if r.answerability == "answerable"),
        unanswerable_count=sum(1 for r in results if r.answerability == "unanswerable"),
        abstention_expected_count=sum(1 for r in results if r.abstention_expected),
    )

    for name in (
        "citation_precision",
        "citation_recall",
        "citation_completeness",
        "evidence_support_rate",
        "unsupported_claim_rate",
        "contradiction_rate",
        "retrieval_hit_rate",
        "correctness",
        "key_point_recall",
        "expected_information_coverage",
        "reference_answer_similarity",
        "question_answer_relevance",
        "excess_information",
        "supported_claim_ratio",
        "partial_claim_ratio",
        "unsupported_claim_ratio",
        "fabricated_citation_rate",
        "unsupported_citation_rate",
    ):
        setattr(
            agg,
            name,
            average_measured(
                [getattr(r, name) for r in results],
                reason_if_empty=f"no question produced a measured {name}",
            ),
        )

    # -- relevance behaviour ---------------------------------------------------
    # Denominator is the questions where relevance APPLIES and was measured:
    # abstentions are excluded rather than counted as failures.
    relevance_cases = [r for r in results if r.relevance_passed is not None]
    if relevance_cases:
        failures = sum(1 for r in relevance_cases if r.relevance_passed is False)
        agg.relevance_failure_rate = Measured.of(
            failures / len(relevance_cases),
            reason="fraction of questions with a measured relevance verdict "
                   "below the threshold (abstentions excluded)",
            sample_size=len(relevance_cases),
        )
    else:
        agg.relevance_failure_rate = Measured.unknown(
            "no question produced a measured relevance verdict (abstentions, "
            "empty answers, or questions without content terms)"
        )

    # -- abstention behaviour ------------------------------------------------
    abstention_cases = [r for r in results if r.abstention_correct is not None]
    if abstention_cases:
        correct = sum(1 for r in abstention_cases if r.abstention_correct)
        agg.abstention_accuracy = Measured.of(
            correct / len(abstention_cases),
            reason="fraction of questions where the system abstained exactly when it should",
            sample_size=len(abstention_cases),
        )
    else:
        agg.abstention_accuracy = Measured.unknown(
            "no question has a human answerability determination, so abstention "
            "behaviour cannot be scored"
        )

    state_cases = [r for r in results if r.grounding_state_correct is not None]
    if state_cases:
        agg.grounding_state_accuracy = Measured.of(
            sum(1 for r in state_cases if r.grounding_state_correct) / len(state_cases),
            reason="fraction of questions whose grounding state met its expectation",
            sample_size=len(state_cases),
        )
    else:
        agg.grounding_state_accuracy = Measured.unknown("no grounding-state expectations recorded")

    fs = [r for r in results if r.false_supported]
    agg.false_supported_rate = Measured.of(
        len(fs) / len(results) if results else 0.0,
        reason="answered confidently without the required evidence",
        sample_size=len(results),
    )
    fu = [r for r in results if r.false_unsupported]
    agg.false_unsupported_rate = Measured.of(
        len(fu) / len(results) if results else 0.0,
        reason="refused to answer a question the corpus does contain",
        sample_size=len(results),
    )
    hallucinations = [
        r for r in results if r.false_supported or r.false_unsupported
    ]
    agg.hallucination_rate = Measured.of(
        len(hallucinations) / len(results) if results else 0.0,
        reason="answers that were confidently wrong about corpus coverage",
        sample_size=len(results),
    )

    agg.pass_rate = Measured.of(
        sum(1 for r in results if r.passed) / len(results) if results else 0.0,
        reason="fraction of questions passing citation + abstention checks",
        sample_size=len(results),
    )

    # -- distributions ---------------------------------------------------------
    dist: dict[str, int] = {}
    for r in results:
        dist[r.actual_grounding_state or "UNKNOWN"] = dist.get(r.actual_grounding_state or "UNKNOWN", 0) + 1
    agg.grounding_state_distribution = dist

    matrix: dict[str, dict[str, int]] = {}
    for r in results:
        exp = r.expected_grounding_state or "ANY_ACCEPTABLE"
        act = r.actual_grounding_state or "UNKNOWN"
        row = matrix.setdefault(exp, {})
        ok = "correct" if r.grounding_state_correct else "incorrect"
        row[f"{act} ({ok})"] = row.get(f"{act} ({ok})", 0) + 1
    agg.confusion_matrix = matrix

    agg.unknown_metrics = [
        name
        for name in (
            "correctness",
            "key_point_recall",
            "citation_precision",
            "citation_recall",
            "evidence_support_rate",
            "unsupported_claim_rate",
            "contradiction_rate",
            "retrieval_hit_rate",
            "citation_completeness",
            "expected_information_coverage",
            "reference_answer_similarity",
            "question_answer_relevance",
            "excess_information",
            "relevance_failure_rate",
            "supported_claim_ratio",
            "partial_claim_ratio",
            "unsupported_claim_ratio",
            "fabricated_citation_rate",
            "unsupported_citation_rate",
        )
        if not getattr(agg, name).measured
    ]
    if agg.unknown_metrics:
        agg.warnings.append(
            "metrics reported as UNKNOWN (not zero): " + ", ".join(agg.unknown_metrics)
        )
    if not any(r.correctness.measured for r in results):
        if any(r.reference_answer_similarity.measured for r in results):
            agg.warnings.append(
                "answer CORRECTNESS is UNKNOWN for this run: human reference "
                "answers exist and lexical similarity to them is reported as "
                "reference_answer_similarity, but correctness itself requires "
                "human review or an explicitly model-based judge."
            )
        else:
            agg.warnings.append(
                "answer CORRECTNESS was not measured: this benchmark has no "
                "human-authored reference answers. Groundedness and citation "
                "correctness are measured; factual correctness is UNKNOWN."
            )
    return agg


def build_run(
    *,
    kb_id: str,
    benchmark: AnswerBenchmark,
    benchmark_path: str,
    questions: list[AnswerBenchmarkQuestion],
    results: list[AnswerQualityResult],
    config: AnswerEvaluationConfig,
    retrieval_run_ids: list[str],
    generator: str,
    model: str,
    is_mock: bool,
    prompt_version: str,
    answerer_version: str,
    evaluator: AnswerEvaluator,
    run_id: str,
    corpus_version: str | None = None,
    corpus_fingerprint: str | None = None,
    notes: list[str] | None = None,
) -> AnswerEvaluationRun:
    """Assemble the immutable run record from per-question results."""
    from app.services.answer_eval.entailment import (
        HeuristicCitationEntailment,
    )

    entail = getattr(evaluator, "_entailment", None) or HeuristicCitationEntailment()
    subset_note = (
        "all benchmark questions"
        if not config.question_ids and not config.limit
        else f"explicit subset ({len(questions)} of {benchmark.question_count})"
    )
    # Relevance provenance travels with the run: a comparison between runs is
    # meaningless if one scored relevance with corpus IDF weights and the other
    # with the plain fallback.
    rel_methods = sorted({r.relevance_method for r in results if r.relevance_method})
    rel_sources = sorted({r.relevance_weight_source for r in results if r.relevance_weight_source})
    plain_count = sum(1 for r in results if r.relevance_method == METHOD_PLAIN)
    run = AnswerEvaluationRun(
        id=run_id,
        kb_id=kb_id,
        benchmark_name=benchmark.benchmark,
        benchmark_version=benchmark.version,
        benchmark_fingerprint=benchmark.fingerprint(),
        benchmark_path=benchmark_path,
        benchmark_lifecycle=benchmark.lifecycle.value,
        benchmark_version_id=config.benchmark_version_id,
        benchmark_artifact_fingerprint=config.benchmark_artifact_fingerprint,
        official=config.official,
        kb_version=None,
        corpus_version=corpus_version,
        corpus_fingerprint=corpus_fingerprint,
        strategy=config.strategy,
        retrieval_params=(config.retrieval_params.model_dump(mode="json")
                          if config.retrieval_params else {}),
        retrieval_run_ids=retrieval_run_ids,
        answer_mode=config.answer_mode,
        generator=generator,
        model=model,
        is_mock=is_mock,
        prompt_version=prompt_version,
        answerer_version=answerer_version,
        evaluator_name=evaluator.name,
        evaluator_version=evaluator.version,
        evaluator_is_model_based=getattr(evaluator, "is_model_based", False),
        evaluator_detail=getattr(evaluator, "detail", ""),
        entailment_provider=entail.name,
        entailment_is_model_based=entail.is_model_based,
        relevance_method=",".join(rel_methods),
        relevance_weight_source=",".join(rel_sources),
        question_ids=[q.question_id for q in questions],
        question_count=len(questions),
        answerable_count=sum(1 for q in questions if q.answerability.value == "answerable"),
        unanswerable_count=sum(1 for q in questions if q.answerability.value == "unanswerable"),
        abstention_expected_count=sum(1 for q in questions if q.abstention_required),
        subset_note=subset_note,
        aggregate=aggregate_results(results),
        per_question=results,
        failed_question_ids=[r.question_id for r in results if not r.passed],
        notes=notes or [],
    )
    if plain_count:
        run.warnings.append(
            f"relevance for {plain_count} question(s) used the PLAIN (non-IDF) "
            f"fallback because no corpus IDF statistics were available. Plain "
            f"coverage is measured to be non-discriminative for off-domain "
            f"answers, so its shortfalls are reported as warnings, not failures."
        )
    return run


__all__ = [
    "AnswerEvaluationConfig",
    "AnswerEvaluationRun",
    "RunComparisonPlan",
    "aggregate_results",
    "build_run",
    "select_questions",
]