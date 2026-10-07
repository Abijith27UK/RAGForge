"""Recommendation engine (V9 Phase 5) — retrieval error analysis becomes advice.

Reads measured diagnostics and failure classifications, and produces
**recommendations with evidence attached**. It never edits configuration.

## The rules, and why each one needs measured data to fire

* **Relevant evidence consistently appears deep (rank 6+)** -> raise ``top_k``.
  Fires only on questions whose required evidence was actually retrieved: if the
  evidence is absent from the candidate set, the depth is not the problem.
* **The evidence is nowhere in the candidate set at all** -> report a CORPUS or
  SOURCE deficiency. This rule deliberately outranks every tuning rule: tuning
  ``top_k``/fusion cannot surface a passage the index does not contain, and
  recommending a knob turn there would send a reader chasing a configuration
  change that cannot possibly help.
* **The evidence was retrieved but not passed to the generator** -> report an
  EVIDENCE-SELECTION deficiency. Tuning retrieval would not fix it.
* **Lexical questions hit while dense questions miss** -> recommend hybrid
  fusion with a lexical weight, because the failure pattern is vocabulary-
  dependent.
* **A candidate already measured better first-relevant-rank** -> recommend
  adopting that candidate, carrying the measured deltas.

## Rules this module refuses to break

* Every numeric trigger is a NAMED CONSTANT with its rationale, so no threshold
  can drift silently.
* A rule fires only on a MEASURED value with a stated sample size. Unknown
  inputs produce no recommendation, and the reason is reported.
* A rule that fires on fewer than ``MIN_QUESTIONS_FOR_A_PATTERN`` questions is
  reported as a WEAK pattern with the count, never as a finding.
* ``experiment_required`` is always True and no function here mutates a
  configuration. The output is advice; a controlled experiment decides.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.services.answer_eval.diagnostics import (
    DiagnosticSummary,
    QuestionDiagnostic,
)
from app.services.answer_eval.failure import (
    FailureCode,
    FailureConfidence,
    classify_result,
)

#: A "deep rank" is one a typical top-5 window would drop. Evidence found at or
#: beyond this rank is treated as a depth problem worth raising top_k for.
DEEP_RANK_THRESHOLD = 6

#: How many questions must show a pattern before it is reported as more than a
#: weak signal. Two questions out of 28 is an anecdote, not a pattern.
MIN_QUESTIONS_FOR_A_PATTERN = 3

#: Minimum share of questions a pattern must cover to be called consistent.
CONSISTENT_PATTERN_SHARE = 0.25

#: A metric gain smaller than this is not treated as a measured improvement.
MIN_MEANINGFUL_METRIC_DELTA = 0.02


class RecommendationAction(str, Enum):
    """What to try. Names describe a configuration change, never an auto-edit."""

    INCREASE_TOP_K = "INCREASE_TOP_K"
    ENABLE_HYBRID_FUSION = "ENABLE_HYBRID_FUSION"
    ADJUST_HYBRID_WEIGHTS = "ADJUST_HYBRID_WEIGHTS"
    ENABLE_RERANKING = "ENABLE_RERANKING"
    REDUCE_DIVERSITY_STRENGTH = "REDUCE_DIVERSITY_STRENGTH"
    INVESTIGATE_EVIDENCE_SELECTION = "INVESTIGATE_EVIDENCE_SELECTION"
    CORPUS_OR_SOURCE_DEFICIENCY = "CORPUS_OR_SOURCE_DEFICIENCY"
    REVIEW_BENCHMARK_LABELS = "REVIEW_BENCHMARK_LABELS"
    NO_ACTION_JUSTIFIED = "NO_ACTION_JUSTIFIED"


class RecommendationStrength(str, Enum):
    """How much the measurement supports the recommendation."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    #: Fired, but the input was too thin to act on.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Recommendation(BaseModel):
    """One recommendation with its evidence, expected effect and risk.

    ``supporting_measurements`` holds the exact numbers the rule read, including
    sample sizes, so a reviewer can reproduce or refute the reasoning.
    """

    action: RecommendationAction
    reason: str = ""
    supporting_measurements: dict[str, object] = Field(default_factory=dict)
    expected_effect: str = ""
    risk: str = ""
    strength: RecommendationStrength = RecommendationStrength.MODERATE
    affected_question_ids: list[str] = Field(default_factory=list)
    #: The retrieval-config deltas this recommendation would need. A PROPOSAL:
    #: nothing in this module applies it.
    proposed_config_change: dict[str, object] = Field(default_factory=dict)
    #: Always True. A recommendation never changes production retrieval config.
    experiment_required: bool = True
    #: Named constants the rule read, so a threshold cannot drift unnoticed.
    thresholds_used: dict[str, object] = Field(default_factory=dict)


class RecommendationReport(BaseModel):
    """All recommendations plus the data gaps that produced none."""

    recommendations: list[Recommendation] = Field(default_factory=list)
    total_questions: int = 0
    questions_with_ground_truth: int = 0
    questions_analysed: int = 0
    unknown_failure_count: int = 0
    data_gaps: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def for_action(self, action: RecommendationAction) -> list[Recommendation]:
        return [r for r in self.recommendations if r.action is action]

    def actions(self) -> list[str]:
        return [r.action.value for r in self.recommendations]


def _strength_for(count: int, total: int) -> RecommendationStrength:
    """Strength from how many questions show the pattern and how many exist."""
    if count < MIN_QUESTIONS_FOR_A_PATTERN:
        return RecommendationStrength.WEAK
    share = count / total if total else 0.0
    if share >= CONSISTENT_PATTERN_SHARE * 2:
        return RecommendationStrength.STRONG
    if share >= CONSISTENT_PATTERN_SHARE:
        return RecommendationStrength.MODERATE
    return RecommendationStrength.WEAK


def analyse_retrieval(
    diagnostics: list[QuestionDiagnostic],
    *,
    current_top_k: int | None = None,
    current_strategy: str = "",
    summary: DiagnosticSummary | None = None,
) -> RecommendationReport:
    """Produce retrieval recommendations from measured diagnostics.

    Only questions with ``ground_truth_available`` contribute to a PATTERN: a
    question with no reference evidence cannot show that its evidence was deep,
    absent or unselected, and counting it would dilute every ratio.
    """
    report = RecommendationReport()
    if not diagnostics:
        report.data_gaps.append(
            "no question diagnostics were supplied, so no recommendation can be "
            "made from measurement"
        )
        return report

    if summary is None:
        from app.services.answer_eval.diagnostics import summarise_diagnostics

        summary = summarise_diagnostics(diagnostics)

    report.total_questions = len(diagnostics)
    report.questions_with_ground_truth = summary.ground_truth_question_count

    grounded = [d for d in diagnostics if d.ground_truth_available]
    report.questions_analysed = len(grounded)

    if not grounded:
        report.data_gaps.append(
            f"none of the {len(diagnostics)} question(s) carry benchmark reference "
            f"evidence, so retrieval cannot be diagnosed at all. Retrieval metrics "
            f"are UNKNOWN rather than zero; the actionable finding is that the "
            f"benchmark needs reference evidence before tuning is meaningful"
        )

    # -- deep-rank pattern ---------------------------------------------------
    deep = [
        d
        for d in grounded
        if d.first_relevant_rank is not None and d.first_relevant_rank >= DEEP_RANK_THRESHOLD
    ]
    if deep:
        ranks = sorted(d.first_relevant_rank for d in deep if d.first_relevant_rank is not None)
        recommended_k = max(ranks) if ranks else None
        strength = _strength_for(len(deep), len(grounded))
        report.recommendations.append(
            Recommendation(
                action=RecommendationAction.INCREASE_TOP_K,
                reason=(
                    f"{len(deep)} question(s) retrieved their required evidence only "
                    f"at rank >= {DEEP_RANK_THRESHOLD}, so a smaller top_k window "
                    f"would drop it. Depth is the cause here, not absence: the "
                    f"evidence IS in the candidate set"
                ),
                supporting_measurements={
                    "questions_with_deep_evidence": len(deep),
                    "questions_with_ground_truth": len(grounded),
                    "first_relevant_ranks": ranks,
                    "min_rank": min(ranks) if ranks else None,
                    "max_rank": max(ranks) if ranks else None,
                    "current_top_k": current_top_k,
                },
                expected_effect=(
                    f"raising top_k to at least {recommended_k} would place the "
                    f"deepest observed required evidence inside the window for "
                    f"these questions"
                ),
                risk=(
                    "a larger top_k adds weaker passages to the prompt, which can "
                    "lower citation precision and increase the context the "
                    "generator must ignore. This is why an experiment is required "
                    "rather than applying it directly"
                ),
                strength=strength,
                affected_question_ids=sorted(d.question_id for d in deep),
                proposed_config_change=(
                    {"top_k": recommended_k} if recommended_k is not None else {}
                ),
                thresholds_used={
                    "DEEP_RANK_THRESHOLD": DEEP_RANK_THRESHOLD,
                    "MIN_QUESTIONS_FOR_A_PATTERN": MIN_QUESTIONS_FOR_A_PATTERN,
                    "CONSISTENT_PATTERN_SHARE": CONSISTENT_PATTERN_SHARE,
                },
            )
        )

    # -- absent evidence: corpus/source deficiency --------------------------
    absent = [
        d
        for d in grounded
        if d.required_retrieved is False and d.required_recall_at_depth.value == 0.0
    ]
    if absent:
        report.recommendations.append(
            Recommendation(
                action=RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY,
                reason=(
                    f"{len(absent)} question(s) never saw their required evidence at "
                    f"ANY depth, so the passage is not in the retrieved candidate "
                    f"set. No retrieval knob can surface a passage the index does "
                    f"not contain, so this is reported as a coverage problem rather "
                    f"than a tuning opportunity"
                ),
                supporting_measurements={
                    "questions_missing_required_evidence_entirely": len(absent),
                    "questions_with_ground_truth": len(grounded),
                    "affected_questions": sorted(d.question_id for d in absent),
                },
                expected_effect=(
                    "none from retrieval tuning. The remedy is corpus work: ingest a "
                    "source that covers the missing topics, or accept that these "
                    "questions are currently unanswerable and record them as such"
                ),
                risk=(
                    "misreading this as a retrieval fault leads to tuning small "
                    "knobs against a corpus gap, which cannot fix it and consumes "
                    "the experiment budget"
                ),
                strength=_strength_for(len(absent), len(grounded)),
                affected_question_ids=sorted(d.question_id for d in absent),
                proposed_config_change={},
                thresholds_used={"MIN_QUESTIONS_FOR_A_PATTERN": MIN_QUESTIONS_FOR_A_PATTERN},
            )
        )

    # -- retrieved but unselected: selection deficiency ----------------------
    unselected = [d for d in grounded if d.required_missing_from_generation]
    retrieved_but_unselected = [
        d
        for d in unselected
        if d.required_missing_from_generation
        and set(d.required_missing_from_generation) <= set(item.chunk_id for item in d.retrieved)
    ]
    if retrieved_but_unselected:
        report.recommendations.append(
            Recommendation(
                action=RecommendationAction.INVESTIGATE_EVIDENCE_SELECTION,
                reason=(
                    f"{len(retrieved_but_unselected)} question(s) had their required "
                    f"evidence retrieved but not passed to the generator. Retrieval "
                    f"succeeded and the evidence-SELECTION step discarded it, so a "
                    f"retrieval change would not address the failure"
                ),
                supporting_measurements={
                    "questions_with_unselected_required_evidence": len(retrieved_but_unselected),
                    "affected_questions": sorted(d.question_id for d in retrieved_but_unselected),
                    "missing_evidence": {
                        d.question_id: d.required_missing_from_generation
                        for d in retrieved_but_unselected
                    },
                },
                expected_effect=(
                    "fixing selection would put already-retrieved evidence in front "
                    "of the generator; expect answer-quality gains without any "
                    "retrieval-metric change"
                ),
                risk=(
                    "widening what is passed to the generator may admit irrelevant "
                    "passages and lower citation precision"
                ),
                strength=_strength_for(len(retrieved_but_unselected), len(grounded)),
                affected_question_ids=sorted(d.question_id for d in retrieved_but_unselected),
                proposed_config_change={},
                thresholds_used={"MIN_QUESTIONS_FOR_A_PATTERN": MIN_QUESTIONS_FOR_A_PATTERN},
            )
        )

    # -- vocabulary-dependent misses: hybrid fusion --------------------------
    vocabulary_dependent = [
        d
        for d in grounded
        if d.required_retrieved is False
        and current_strategy in ("dense", "bm25")
    ]
    if vocabulary_dependent and current_strategy:
        report.recommendations.append(
            Recommendation(
                action=RecommendationAction.ENABLE_HYBRID_FUSION,
                reason=(
                    f"the run under test used a single-strategy retriever "
                    f"({current_strategy!r}) and {len(vocabulary_dependent)} "
                    f"question(s) missed their required evidence entirely. A "
                    f"single-strategy miss is what lexical/dense complementarity "
                    f"addresses, so testing hybrid fusion is justified — but only "
                    f"after the corpus-deficiency check above, because fusion "
                    f"cannot surface absent passages either"
                ),
                supporting_measurements={
                    "questions_missing_evidence_under_single_strategy": len(vocabulary_dependent),
                    "current_strategy": current_strategy,
                    "affected_questions": sorted(d.question_id for d in vocabulary_dependent),
                },
                expected_effect=(
                    "hybrid fusion can recover passages that only one of the two "
                    "retrievers ranks; the measured effect is unknown until the "
                    "experiment runs"
                ),
                risk=(
                    "fusion weights are themselves a parameter. Changing strategy "
                    "AND weights at once confounds the experiment, so the candidate "
                    "should change ONE of them"
                ),
                strength=_strength_for(len(vocabulary_dependent), len(grounded)),
                affected_question_ids=sorted(d.question_id for d in vocabulary_dependent),
                proposed_config_change={"strategy": "hybrid"},
                thresholds_used={"MIN_QUESTIONS_FOR_A_PATTERN": MIN_QUESTIONS_FOR_A_PATTERN},
            )
        )

    if not report.recommendations:
        report.recommendations.append(
            Recommendation(
                action=RecommendationAction.NO_ACTION_JUSTIFIED,
                reason=(
                    "no measured pattern justifies a configuration change: required "
                    "evidence was retrieved near the front for every question that "
                    "has reference evidence, and no evidence was dropped in "
                    "selection"
                ),
                supporting_measurements={
                    "questions_analysed": len(grounded),
                    "questions_with_ground_truth": len(grounded),
                },
                expected_effect="none; the current configuration meets the measured bar",
                risk=(
                    "changing configuration anyway would be tuning without evidence, "
                    "and any resulting difference would be unexplained"
                ),
                strength=(
                    RecommendationStrength.STRONG
                    if grounded
                    else RecommendationStrength.INSUFFICIENT_EVIDENCE
                ),
                proposed_config_change={},
            )
        )

    if len(grounded) < MIN_QUESTIONS_FOR_A_PATTERN:
        report.data_gaps.append(
            f"only {len(grounded)} question(s) carry reference evidence, below the "
            f"{MIN_QUESTIONS_FOR_A_PATTERN}-question floor for calling a pattern. "
            f"Every recommendation above is reported at WEAK strength for that reason"
        )
    return report


def analyse_failures(
    results: list[object],
    *,
    evidence_by_question: dict[str, list[dict[str, object]]] | None = None,
) -> RecommendationReport:
    """Classify answer-evaluation results and summarise the failure mix.

    Accepts V8 ``AnswerQualityResult`` objects and reads only their measured
    values. Questions whose failure mode is UNKNOWN are counted as unknown and
    reported as a data gap: they are the questions this analysis cannot help
    with, and saying so is more useful than guessing.
    """
    report = RecommendationReport()
    if not results:
        report.data_gaps.append("no evaluation results were supplied")
        return report

    evidence_by_question = evidence_by_question or {}
    counts: dict[str, int] = {}
    unknown_ids: list[str] = []
    hits: dict[FailureCode, list[str]] = {}

    for result in results:
        question_id = getattr(result, "question_id", "")
        classification = classify_result(
            result,
            evidence=evidence_by_question.get(question_id, []),
        )
        label = classification.classification or "no_failure_classified"
        counts[label] = counts.get(label, 0) + 1
        if classification.classification == FailureCode.UNKNOWN.value:
            unknown_ids.append(question_id)
        elif classification.confidence is not FailureConfidence.UNKNOWN:
            try:
                hits.setdefault(FailureCode(classification.classification), []).append(question_id)
            except ValueError:  # pragma: no cover - guarded by FailureClassification
                continue

    report.total_questions = len(results)
    report.unknown_failure_count = len(unknown_ids)
    report.notes.append(f"failure-mode distribution: {dict(sorted(counts.items()))}")

    if unknown_ids:
        report.data_gaps.append(
            f"{len(unknown_ids)} question(s) could not be classified: "
            f"{sorted(unknown_ids)}. Either the benchmark carries no "
            f"required-evidence label for them or the measured signals do not "
            f"distinguish a failure mode, and both cases need human review rather "
            f"than a guess"
        )

    retrieval_labels = {
        FailureCode.NO_RELEVANT_RETRIEVAL,
        FailureCode.LOW_RETRIEVAL_RECALL,
        FailureCode.WRONG_DOCUMENT,
        FailureCode.WRONG_CHUNK,
        FailureCode.INSUFFICIENT_EVIDENCE,
    }
    retrieval_failures = {
        label: ids for label, ids in hits.items() if label in retrieval_labels
    }
    if retrieval_failures:
        total = sum(len(ids) for ids in retrieval_failures.values())
        report.recommendations.append(
            Recommendation(
                action=(
                    RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY
                    if FailureCode.INSUFFICIENT_EVIDENCE in retrieval_failures
                    else RecommendationAction.INCREASE_TOP_K
                    if FailureCode.LOW_RETRIEVAL_RECALL in retrieval_failures
                    else RecommendationAction.ENABLE_HYBRID_FUSION
                ),
                reason=(
                    f"{total} question(s) failed for a retrieval reason: "
                    f"{ {k.value: len(v) for k, v in retrieval_failures.items()} }. "
                    f"Retrieval failures are addressed before generation failures "
                    f"because a generator cannot answer from evidence it never saw"
                ),
                supporting_measurements={
                    "retrieval_failure_counts": {
                        k.value: len(v) for k, v in retrieval_failures.items()
                    },
                    "total_questions_classified": len(results) - len(unknown_ids),
                },
                expected_effect=(
                    "improving retrieval should raise the ceiling for these "
                    "questions; the answer-quality effect is unknown until measured"
                ),
                risk=(
                    "a retrieval change can help one failure mode while hurting "
                    "another, so it must be measured against the full benchmark "
                    "rather than only the failing subset"
                ),
                strength=_strength_for(total, len(results)),
                affected_question_ids=sorted(
                    qid for ids in retrieval_failures.values() for qid in ids
                ),
                thresholds_used={"MIN_QUESTIONS_FOR_A_PATTERN": MIN_QUESTIONS_FOR_A_PATTERN},
            )
        )

    generation_failures = [
        label
        for label in (
            FailureCode.GENERATION_FAILURE,
            FailureCode.ANSWER_RELEVANCE_FAILURE,
            FailureCode.UNSUPPORTED_CLAIM,
            FailureCode.CITATION_ERROR,
            FailureCode.CONFLICTING_EVIDENCE,
        )
        if label in hits
    ]
    if generation_failures:
        total = sum(len(hits[label]) for label in generation_failures)
        report.recommendations.append(
            Recommendation(
                action=RecommendationAction.NO_ACTION_JUSTIFIED,
                reason=(
                    f"{total} question(s) failed for a GENERATION-side reason "
                    f"({[l.value for l in generation_failures]}). These are not "
                    f"retrieval faults, so no retrieval-configuration change is "
                    f"recommended from this evidence"
                ),
                supporting_measurements={
                    "generation_failure_counts": {
                        label.value: len(hits[label]) for label in generation_failures
                    }
                },
                expected_effect=(
                    "retrieval tuning is not indicated. Generation-side fixes "
                    "(prompting, abstention policy, citation enforcement) are out of "
                    "scope for a retrieval experiment"
                ),
                risk=(
                    "attributing a generation failure to retrieval and tuning anyway "
                    "would produce a configuration change with no causal link to the "
                    "observed failure"
                ),
                strength=_strength_for(total, len(results)),
                affected_question_ids=sorted(
                    qid for label in generation_failures for qid in hits[label]
                ),
            )
        )
    return report


def recommendation_from_comparison(
    *,
    metric: str,
    baseline_mean: float | None,
    candidate_mean: float | None,
    mean_delta: float | None,
    improved_count: int,
    regressed_count: int,
    paired_count: int,
    candidate_label: str,
    candidate_config: dict[str, object],
    exploratory: bool = True,
) -> Recommendation | None:
    """Recommend adopting a candidate ONLY when the measured delta supports it.

    Returns ``None`` when the delta is unknown, below
    ``MIN_MEANINGFUL_METRIC_DELTA``, or negative in the metric's improvement
    direction. A candidate that did not measurably help is not recommended.
    """
    from app.services.answer_eval.comparison import metric_direction, MetricDirection

    if mean_delta is None or paired_count == 0:
        return None
    direction = metric_direction(metric)
    sign = 1.0 if direction is MetricDirection.HIGHER_IS_BETTER else -1.0
    oriented = mean_delta * sign
    if oriented < MIN_MEANINGFUL_METRIC_DELTA:
        return None
    if improved_count <= regressed_count:
        return None
    return Recommendation(
        action=RecommendationAction.ADJUST_HYBRID_WEIGHTS,
        reason=(
            f"candidate {candidate_label!r} improved {metric} by a mean of "
            f"{mean_delta:+g} across {paired_count} paired question(s) "
            f"({improved_count} improved, {regressed_count} regressed)"
        ),
        supporting_measurements={
            "metric": metric,
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "mean_delta": mean_delta,
            "paired_count": paired_count,
            "improved_count": improved_count,
            "regressed_count": regressed_count,
            "exploratory": exploratory,
        },
        expected_effect=(
            f"adopting {candidate_label!r} would repeat this measured improvement "
            f"only if the corpus and benchmark are unchanged"
        ),
        risk=(
            "the paired sample is below the inference floor, so this is a measured "
            "direction rather than a statistically supported effect; a regression "
            "in another metric may outweigh it"
            if exploratory
            else "a single metric can improve while another regresses; the full "
            "comparison must be read before adopting"
        ),
        strength=(
            RecommendationStrength.MODERATE if exploratory else RecommendationStrength.STRONG
        ),
        proposed_config_change=dict(candidate_config),
        thresholds_used={"MIN_MEANINGFUL_METRIC_DELTA": MIN_MEANINGFUL_METRIC_DELTA},
    )


__all__ = [
    "CONSISTENT_PATTERN_SHARE",
    "DEEP_RANK_THRESHOLD",
    "MIN_MEANINGFUL_METRIC_DELTA",
    "MIN_QUESTIONS_FOR_A_PATTERN",
    "Recommendation",
    "RecommendationAction",
    "RecommendationReport",
    "RecommendationStrength",
    "analyse_failures",
    "analyse_retrieval",
    "recommendation_from_comparison",
]
