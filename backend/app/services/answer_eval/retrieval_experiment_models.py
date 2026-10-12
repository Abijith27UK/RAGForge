"""V13 controlled-retrieval-experiment schemas and the explicit outcome rules.

This module is the single source of truth for

* the SHAPE of a persisted experiment (``RetrievalExperiment`` and friends), and
* the RULES that turn real measurements into an outcome.

It deliberately does NOT re-implement statistics. Pairing, per-metric deltas,
the sign test, the paired t-test and the accept/reject decision all come from
``services.answer_eval.comparison``, so an experiment and a V9/V12 comparison
can never disagree about what a delta means.

## What an experiment is, and is not

An experiment is a RECORD of a controlled comparison on matched questions. It is
not a search for a better number:

* the hypothesis, the primary metric, the guardrail metrics and the rejection
  conditions are all declared BEFORE the arms are run and are stored on the
  artifact;
* every arm is measured by the same retrieval engine as a V12 diagnostic, so the
  metric definitions are identical;
* an outcome is derived from explicit rules over measured values — nothing is
  inferred from a single metric, and nothing is promoted into the default
  production configuration automatically (``promoted_to_default`` stays False
  and can only be changed by an explicit, separate user decision).

## Metric direction

Direction is declared once, in ``comparison.METRIC_DIRECTIONS``. This module
references it rather than re-declaring it, and the experiment runner refuses to
summarise a metric whose direction is undeclared.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import BaseModel, Field

from app.services.answer_eval.comparison import (
    AcceptanceDecision,
    ExperimentComparison,
    MetricDirection,
    PairingVerdict,
    metric_direction,
    paired_significance,
)

#: How large a per-question mean delta must be, in the metric's own units, before
#: it is called "meaningful" rather than noise. Declared per metric and recorded
#: on every experiment so a reader can audit the bar that was used.
MEANINGFUL_DELTA: dict[str, float] = {
    "recall_at_k": 0.02,
    "required_recall_at_depth": 0.02,
    "mrr": 0.02,
    "ndcg_at_k": 0.02,
    "precision_at_k": 0.02,
    "first_relevant_rank": 0.5,
    "latency_ms": 50.0,
}

#: Metrics an experiment compares by default. Every one of them is direction-
#: declared in the comparison module; a name not declared there raises rather
#: than being compared in an assumed direction.
DEFAULT_COMPARISON_METRICS: tuple[str, ...] = (
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "first_relevant_rank",
    "required_recall_at_depth",
    "latency_ms",
)


class ExperimentOutcome(str, Enum):
    """The verdict of a controlled experiment. Derived, never asserted."""

    PENDING = "PENDING"
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NO_MEANINGFUL_IMPROVEMENT = "NO_MEANINGFUL_IMPROVEMENT"
    REGRESSION = "REGRESSION"
    IMPROVED = "IMPROVED"


class RetrievalStrategyConfig(BaseModel):
    """One arm of an experiment: a strategy plus the configuration under test."""

    label: str = ""
    strategy: str = "dense"
    top_k: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Evaluation cut-off for this arm. Also the retrieval window, so "
        "recall@k is measured at the depth that produced it.",
    )
    candidate_depth: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="Probe depth. None = max(top_k, the engine's default floor). Must "
        "be >= top_k; a probe shallower than the window would turn ranking failures "
        "into apparent recall failures.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy parameters under test (fusion weights, reranker, ...).",
    )
    role: str = Field(default="candidate", description="baseline | candidate")
    rationale: str = Field(
        default="", description="Why this arm is being tried, in the experiment's terms"
    )

    def to_arm(self) -> dict[str, Any]:
        """The dict ``RetrievalDiagnosticRunner.run_configs`` consumes."""
        return {
            "label": self.label or self.strategy,
            "strategy": self.strategy,
            "top_k": self.top_k,
            "candidate_depth": self.candidate_depth,
            "params": dict(self.params),
        }

    def fingerprint(self) -> str:
        payload = {
            "strategy": self.strategy,
            "top_k": self.top_k,
            "candidate_depth": self.candidate_depth,
            "params": self.params,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class MetricStats(BaseModel):
    """Descriptive statistics for ONE metric over the measured questions."""

    metric: str
    mean: float | None = None
    median: float | None = None
    min: float | None = None
    max: float | None = None
    sample_size: int = 0
    direction: str = Field(
        default="", description="higher_is_better | lower_is_better, from the shared map"
    )
    measured: bool = False
    reason: str = ""


class ExperimentPerQuestionResult(BaseModel):
    """One question's baseline vs candidate value for the PRIMARY metric."""

    question_id: str
    question: str = ""
    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = Field(default=None, description="candidate - baseline")
    direction: str = ""
    outcome: str = Field(default="UNKNOWN", description="IMPROVED | REGRESSED | UNCHANGED | UNKNOWN")
    note: str = ""


class ExperimentArm(BaseModel):
    """One measured arm, recorded exactly as it ran."""

    label: str
    strategy: str
    role: str = "candidate"
    config: RetrievalStrategyConfig
    status: str = Field(default="pending", description="completed | failed")
    failure_reason: str = ""
    top_k: int = 0
    candidate_depth: int = 0
    config_fingerprint: str = ""
    embedding_model: str = ""
    retrieval_backend: str = ""
    reranker_status: str = ""
    index_points: int | None = None
    duration_ms: float | None = None
    question_count: int = 0
    code_counts: dict[str, int] = Field(default_factory=dict)
    latency_ms: float | None = None
    metric_stats: dict[str, MetricStats] = Field(default_factory=dict)
    per_question: list[ExperimentPerQuestionResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CandidateOutcome(BaseModel):
    """One candidate's verdict, with the rules that produced it."""

    label: str
    strategy: str
    config_fingerprint: str = ""
    acceptance: AcceptanceDecision = Field(
        default=AcceptanceDecision.INCONCLUSIVE,
        description="Derived alongside the outcome by the declared rules: ACCEPT only "
        "when the primary metric cleared its bar AND no guardrail regressed by its bar; "
        "NOT_VALID when the comparison was not comparable.",
    )
    acceptance_reason: str = ""
    outcome: ExperimentOutcome = ExperimentOutcome.INCONCLUSIVE
    outcome_explanation: str = ""
    pairing: str = ""
    pairing_reason: str = ""
    primary_metric: str = ""
    primary_mean_delta: float | None = None
    primary_required_delta: float = 0.0
    primary_improved: bool = False
    paired_question_count: int = 0
    paired_wins: int | None = Field(
        default=None, description="Questions that improved on the primary metric"
    )
    paired_losses: int | None = Field(
        default=None, description="Questions that regressed on the primary metric"
    )
    primary_sign_test_p: float | None = Field(
        default=None,
        description="Exact sign-test p-value for the primary metric, recomputed from that "
        "metric's own pairing. The comparison object holds a FLAT list of tests, one per "
        "compared metric, so this is what makes the primary metric's test unambiguous.",
    )
    primary_sign_test_n: int = Field(
        default=0, description="Paired observations behind primary_sign_test_p"
    )
    guardrails_regressed: list[str] = Field(default_factory=list)
    guardrail_assessment: list[str] = Field(
        default_factory=list,
        description="Per-guardrail verdict against that metric's declared bar, including "
        "guardrails that were not paired",
    )
    rules_applied: list[str] = Field(default_factory=list)


class RetrievalExperiment(BaseModel):
    """An experiment artifact: hypothesis, arms, measurements, verdict.

    The artifact is JSON-serialisable and stored under the experiments directory
    next to the benchmarks, so the same integrity tooling can audit it and a
    result can never be attributed to a different benchmark or corpus.
    """

    id: str
    kb_id: str
    kb_name: str = ""
    name: str
    slug: str = ""

    # -- declared BEFORE running -------------------------------------------
    hypothesis: str = ""
    targeted_failure_code: str = ""
    targeted_question_ids: list[str] = Field(default_factory=list)
    proposed_change: str = ""
    expected_effect: str = ""
    primary_metric: str = ""
    guardrail_metrics: list[str] = Field(default_factory=list)
    comparison_method: str = "paired delta on matched questions (V9 comparison module)"
    rejection_conditions: list[str] = Field(default_factory=list)

    # -- instrument and corpus identity ------------------------------------
    baseline: RetrievalStrategyConfig
    candidates: list[RetrievalStrategyConfig] = Field(default_factory=list)
    benchmark_name: str = ""
    benchmark_fingerprint: str = ""
    benchmark_path: str = ""
    benchmark_sha256_before: str = ""
    benchmark_sha256_after: str = ""
    corpus_fingerprint: str | None = None
    corpus_version: str | None = None
    matched_question_ids: list[str] = Field(default_factory=list)
    question_set_fingerprint: str = Field(
        default="",
        description="Hash of the matched question ids and their expected evidence, so a "
        "re-run can prove it used the SAME questions.",
    )

    # -- instrument identity (V14) -----------------------------------------
    # Defaults keep every V13 artifact loadable: an artifact that does not say
    # it ran on the held-out instrument ran on the development one.
    instrument: str = Field(
        default="development",
        description="Which instrument produced these numbers: 'development' (the 28 "
        "questions a configuration was chosen on) or 'held-out' (questions neither "
        "authored nor selected with any retriever's behaviour in view).",
    )
    benchmark_lifecycle: str = Field(
        default="",
        description="The frozen instrument's tier when this ran: draft | provisional | "
        "validated. Empty for the development benchmark, which has no lifecycle gate.",
    )
    claim_strength: str = Field(
        default="",
        description="What this result is allowed to be called: 'confirmatory' only from "
        "a VALIDATED instrument, otherwise 'exploratory'.",
    )
    retests: str = Field(
        default="",
        description="For a held-out run: the name of the development experiment whose "
        "finding it retests. Recorded so a generalization claim is always traceable to "
        "the specific finding it is about.",
    )

    # -- results -----------------------------------------------------------
    status: str = "pending"
    arms: list[ExperimentArm] = Field(default_factory=list)
    comparisons: list[ExperimentComparison] = Field(default_factory=list)
    candidate_outcomes: list[CandidateOutcome] = Field(default_factory=list)
    outcome: ExperimentOutcome = ExperimentOutcome.PENDING
    outcome_explanation: str = ""
    accepted_candidate: str = ""
    promoted_to_default: bool = Field(
        default=False,
        description="Always False: an experiment never changes the production "
        "configuration. Promotion requires a separate, explicit user decision.",
    )

    integrity: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    created_at: str = ""
    completed_at: str | None = None
    created_by: str = ""

    # -- helpers -----------------------------------------------------------

    def arm(self, label: str) -> ExperimentArm | None:
        return next((a for a in self.arms if a.label == label), None)

    def comparison_for(self, candidate_label: str) -> ExperimentComparison | None:
        return next(
            (c for c in self.comparisons if c.candidate_label == candidate_label), None
        )

    def candidate_outcome(self, label: str) -> CandidateOutcome | None:
        return next((c for c in self.candidate_outcomes if c.label == label), None)

    def filename(self) -> str:
        return f"{self.slug or slugify(self.name)}.json"

    def path(self, experiments_dir: Path) -> Path:
        return experiments_dir / self.filename()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    """Stable file-name slug. Deterministic so a re-run overwrites its own file."""
    return re.sub(r"[^a-z0-9_\-]+", "_", (name or "").strip().lower()).strip("_") or "experiment"


def question_set_fingerprint(
    question_ids: list[str], expected_by_question: dict[str, list[str]]
) -> str:
    """Hash of the matched question set AND its expected evidence.

    Two runs whose fingerprint matches used the same questions with the same
    labels; a changed label moves it, so an experiment cannot be re-reported
    against a benchmark that has since been re-labelled.
    """
    payload = json.dumps(
        {
            "questions": sorted(question_ids),
            "expected": {q: sorted(expected_by_question.get(q, [])) for q in sorted(question_ids)},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def metric_stats(
    metric: str, series: dict[str, float | None], *, reason: str = ""
) -> MetricStats:
    """Descriptive stats over the MEASURED values of one metric.

    Unmeasured values are excluded and counted only through ``sample_size``: an
    unknown metric is never treated as zero, which would silently deflate every
    mean it appears in.
    """
    try:
        direction = metric_direction(metric)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    values = sorted(v for v in series.values() if v is not None)
    if not values:
        return MetricStats(
            metric=metric,
            measured=False,
            sample_size=0,
            direction=direction.value,
            reason=reason
            or "no question had a measured value for this metric; it is UNKNOWN, not 0",
        )
    return MetricStats(
        metric=metric,
        mean=round(sum(values) / len(values), 6),
        median=round(median(values), 6),
        min=round(min(values), 6),
        max=round(max(values), 6),
        sample_size=len(values),
        direction=direction.value,
        measured=True,
        reason=reason
        or f"mean over {len(values)} question(s) with a measured value; unmeasured "
        "questions were excluded, not counted as zero",
    )


def per_question_results(
    *,
    metric: str,
    baseline_series: dict[str, float | None],
    candidate_series: dict[str, float | None],
    questions_by_id: dict[str, str] | None = None,
) -> list[ExperimentPerQuestionResult]:
    """Pair the primary metric across questions, oriented by its direction."""
    direction = metric_direction(metric)
    sign = 1.0 if direction is MetricDirection.HIGHER_IS_BETTER else -1.0
    titles = questions_by_id or {}
    out: list[ExperimentPerQuestionResult] = []
    for qid in sorted(set(baseline_series) & set(candidate_series)):
        b, c = baseline_series[qid], candidate_series[qid]
        if b is None or c is None:
            out.append(
                ExperimentPerQuestionResult(
                    question_id=qid,
                    question=titles.get(qid, ""),
                    baseline=b,
                    candidate=c,
                    direction=direction.value,
                    outcome="UNKNOWN",
                    note="one side is unmeasured for this metric; excluded from the pairing",
                )
            )
            continue
        delta = round(c - b, 6)
        oriented = delta * sign
        if abs(oriented) < 1e-9:
            outcome = "UNCHANGED"
        elif oriented > 0:
            outcome = "IMPROVED"
        else:
            outcome = "REGRESSED"
        out.append(
            ExperimentPerQuestionResult(
                question_id=qid,
                question=titles.get(qid, ""),
                baseline=round(b, 6),
                candidate=round(c, 6),
                delta=delta,
                direction=direction.value,
                outcome=outcome,
                note=(
                    f"delta = {delta:+.6f} ({direction.value}); "
                    f"{'better' if outcome == 'IMPROVED' else 'worse' if outcome == 'REGRESSED' else 'no change'}"
                ),
            )
        )
    return out


def required_delta_for(metric: str) -> float:
    """The declared meaningful bar for one metric, in the metric's own units.

    Deliberately a single shared lookup rather than a per-experiment choice: a
    bar chosen after seeing the result would make the verdict unfalsifiable.
    """
    return float(MEANINGFUL_DELTA.get(metric, 0.02))


def decide_candidate_outcome(
    *,
    label: str,
    strategy: str,
    config_fingerprint: str,
    comparison: ExperimentComparison | None,
    primary_metric: str,
    required_delta: float,
    arm_status: str,
    arm_failure_reason: str = "",
    guardrail_metrics: list[str] | None = None,
) -> CandidateOutcome:
    """Derive ONE candidate's outcome from the comparison and the declared rules.

    The rules, in the order they are applied:

    1. the arm did not run            -> BLOCKED
    2. no comparison was produced     -> INCONCLUSIVE
    3. pairing is NOT_COMPARABLE      -> INCONCLUSIVE, acceptance NOT_VALID (a
                                         protocol mismatch is not a result)
    4. pairing is not COMPARABLE      -> INCONCLUSIVE (partial overlap is never
                                         presented as a full-set comparison)
    5. a declared guardrail regressed -> REGRESSION
    6. |mean primary delta| below the declared meaningful bar
                                      -> NO_MEANINGFUL_IMPROVEMENT
    7. the primary metric improved by at least the bar and no guardrail
       regressed                      -> IMPROVED
    8. anything else                  -> INCONCLUSIVE

    Every branch appends the rule it fired on to ``rules_applied``, so the verdict
    is auditable rather than asserted.

    ## Why guardrails use the declared bar, not raw inequality

    A guardrail fires only when its oriented mean delta is at least as large as
    that metric's declared meaningful bar. Comparing raw inequality instead would
    let ordinary run-to-run noise (a few milliseconds of latency, a rounding
    difference) reject a genuine improvement — on a 28-question benchmark that is
    not a safety property, it is a coin flip. The bar is the same one the primary
    metric is held to, so the standard is uniform and declared in advance.
    """
    guardrails = list(guardrail_metrics or [])
    outcome = CandidateOutcome(
        label=label,
        strategy=strategy,
        config_fingerprint=config_fingerprint,
        primary_metric=primary_metric,
        primary_required_delta=required_delta,
    )
    rules: list[str] = []

    def _finish(o: ExperimentOutcome, acceptance: AcceptanceDecision) -> CandidateOutcome:
        outcome.outcome = o
        outcome.acceptance = acceptance
        outcome.rules_applied = rules
        return outcome

    if arm_status != "completed":
        reason = arm_failure_reason or "the arm did not complete"
        rules.append("R1: the arm did not complete, so there is nothing to compare")
        outcome.acceptance_reason = f"BLOCKED: {reason}"
        outcome.outcome_explanation = (
            f"BLOCKED. The {strategy!r} arm did not run to completion: {reason}. "
            "No substitute measurement was produced."
        )
        return _finish(ExperimentOutcome.BLOCKED, AcceptanceDecision.INCONCLUSIVE)

    if comparison is None:
        rules.append("R2: no comparison object was produced for this candidate")
        outcome.acceptance_reason = "INCONCLUSIVE: no comparison was produced"
        outcome.outcome_explanation = (
            "INCONCLUSIVE. No paired comparison exists for this candidate, so no "
            "delta can be attributed to the change."
        )
        return _finish(ExperimentOutcome.INCONCLUSIVE, AcceptanceDecision.INCONCLUSIVE)

    outcome.pairing = comparison.pairing.value
    outcome.pairing_reason = comparison.pairing_reason
    primary_cmp = comparison.comparison_for(primary_metric)
    if primary_cmp is not None:
        outcome.paired_question_count = primary_cmp.paired_count
        outcome.primary_mean_delta = primary_cmp.mean_delta
        # The oriented win/loss counts come straight from the primary metric's own
        # comparison, which already classes every paired question by direction.
        outcome.paired_wins = primary_cmp.improved_count
        outcome.paired_losses = primary_cmp.regressed_count
        # The significance tests live in ONE flat list covering every compared
        # metric, so the primary metric's own test is recomputed rather than fished
        # out of that list by name — which would silently report some other
        # metric's sign test.
        sign_tests = [
            t for t in paired_significance(primary_cmp) if t.name == "exact_sign_test"
        ]
        if sign_tests:
            outcome.primary_sign_test_p = sign_tests[0].p_value
            outcome.primary_sign_test_n = sign_tests[0].n

    # A guardrail regresses only when its oriented mean delta reaches that
    # metric's own declared bar; see the docstring.
    regressed: list[str] = []
    guardrail_notes: list[str] = []
    for metric in guardrails:
        cmp = comparison.comparison_for(metric)
        if cmp is None or cmp.mean_delta is None:
            guardrail_notes.append(f"{metric}: not paired (treated as no evidence, not as a pass)")
            continue
        bar = required_delta_for(metric)
        sign = 1.0 if metric_direction(metric) is MetricDirection.HIGHER_IS_BETTER else -1.0
        oriented = cmp.mean_delta * sign
        if oriented <= -bar:
            regressed.append(metric)
            guardrail_notes.append(
                f"{metric}: REGRESSED by {oriented:+.4f} (bar {bar})"
            )
        else:
            guardrail_notes.append(
                f"{metric}: within tolerance ({oriented:+.4f}, bar {bar})"
            )
    outcome.guardrails_regressed = regressed
    outcome.guardrail_assessment = guardrail_notes

    if comparison.pairing is PairingVerdict.NOT_COMPARABLE:
        rules.append("R3: the comparison is NOT_COMPARABLE, so no delta is attributable")
        outcome.acceptance_reason = f"NOT_VALID: {comparison.pairing_reason}"
        outcome.outcome_explanation = (
            f"INCONCLUSIVE. The comparison is not valid: {comparison.pairing_reason}. "
            "No change is accepted or rejected on this evidence."
        )
        return _finish(ExperimentOutcome.INCONCLUSIVE, AcceptanceDecision.NOT_VALID)

    if comparison.pairing is not PairingVerdict.COMPARABLE:
        rules.append(
            "R4: the question sets only partially overlap, so the comparison is "
            "reported as INCONCLUSIVE rather than as a full-set result"
        )
        outcome.acceptance_reason = "INCONCLUSIVE: the question sets only partially overlap"
        outcome.outcome_explanation = (
            f"INCONCLUSIVE. {comparison.pairing_reason} Deltas are computed on the "
            "shared questions only."
        )
        return _finish(ExperimentOutcome.INCONCLUSIVE, AcceptanceDecision.INCONCLUSIVE)

    if primary_cmp is None:
        rules.append(
            f"R7: the declared primary metric {primary_metric!r} was not compared, "
            "so no direction can be established"
        )
        outcome.acceptance_reason = (
            f"INCONCLUSIVE: the declared primary metric {primary_metric!r} is absent "
            "from the comparison"
        )
        outcome.outcome_explanation = (
            f"INCONCLUSIVE. The declared primary metric {primary_metric!r} is absent "
            "from the comparison; no substitute metric was used in its place."
        )
        return _finish(ExperimentOutcome.INCONCLUSIVE, AcceptanceDecision.INCONCLUSIVE)

    direction = metric_direction(primary_metric)
    good_sign = 1.0 if direction is MetricDirection.HIGHER_IS_BETTER else -1.0
    mean_delta = primary_cmp.mean_delta
    oriented = None if mean_delta is None else mean_delta * good_sign
    outcome.primary_improved = bool(oriented is not None and oriented >= required_delta)

    if not primary_cmp.paired_count:
        rules.append("R2: zero questions were paired on the primary metric")
        outcome.acceptance_reason = "INCONCLUSIVE: no question was paired on the primary metric"
        outcome.outcome_explanation = (
            "INCONCLUSIVE. No question had a measured value on both sides of the "
            "primary metric."
        )
        return _finish(ExperimentOutcome.INCONCLUSIVE, AcceptanceDecision.INCONCLUSIVE)

    if regressed:
        rules.append(
            f"R5: declared guardrail metric(s) regressed by at least their declared "
            f"bar: {', '.join(regressed)}"
        )
        outcome.acceptance_reason = (
            f"REJECT: guardrail metric(s) regressed: {', '.join(regressed)}"
        )
        outcome.outcome_explanation = (
            f"REGRESSION. The primary metric moved the right way ({primary_metric} "
            f"mean delta {mean_delta:+.4f} over {outcome.paired_question_count} paired "
            f"question(s)) but a declared guardrail regressed by at least its "
            f"declared bar: {', '.join(regressed)}. A gain on one metric does not "
            "license a regression on a guardrail."
        )
        return _finish(ExperimentOutcome.REGRESSION, AcceptanceDecision.REJECT)

    if not outcome.primary_improved:
        rules.append(
            f"R6: the mean delta on {primary_metric} does not reach the declared "
            f"meaningful bar ({required_delta})"
        )
        outcome.acceptance_reason = (
            f"REJECT: {primary_metric} did not improve by the declared bar "
            f"({required_delta})"
        )
        shown = "unmeasured" if mean_delta is None else f"{mean_delta:+.4f}"
        outcome.outcome_explanation = (
            f"NO_MEANINGFUL_IMPROVEMENT. {primary_metric} mean delta = {shown} over "
            f"{outcome.paired_question_count} paired question(s), against a declared "
            f"meaningful bar of {required_delta}; "
            f"{outcome.paired_wins or 0} question(s) improved and "
            f"{outcome.paired_losses or 0} regressed. The hypothesis was not supported."
        )
        return _finish(ExperimentOutcome.NO_MEANINGFUL_IMPROVEMENT, AcceptanceDecision.REJECT)

    rules.append(
        f"R7: {primary_metric} improved by at least the declared bar "
        f"({required_delta}) and no declared guardrail regressed by its bar"
    )
    outcome.acceptance_reason = (
        f"ACCEPT: {primary_metric} improved by the declared bar and no guardrail "
        "regressed"
    )
    outcome.outcome_explanation = (
        f"IMPROVED. {primary_metric} mean delta = {mean_delta:+.4f} over "
        f"{outcome.paired_question_count} paired question(s) "
        f"({outcome.paired_wins or 0} improved / {outcome.paired_losses or 0} regressed), "
        f"against a declared meaningful bar of {required_delta}. "
        + (
            "The paired sample is below the inference floor, so this is a measured "
            "result on this benchmark and NOT a claim of statistical significance."
            if comparison.exploratory
            else ""
        )
    ).strip()
    return _finish(ExperimentOutcome.IMPROVED, AcceptanceDecision.ACCEPT)


def decide_experiment_outcome(
    candidate_outcomes: list[CandidateOutcome], *, errors: list[str] | None = None
) -> tuple[ExperimentOutcome, str, str]:
    """Roll the per-candidate verdicts into one experiment verdict.

    Returns ``(outcome, explanation, accepted_candidate)``. The experiment is
    IMPROVED only when at least one candidate is IMPROVED; if none is, the
    experiment reports the strongest signal that WAS observed (a regression if
    any candidate regressed, otherwise no-meaningful-improvement, otherwise
    inconclusive or blocked). Nothing is promoted.
    """
    errs = list(errors or [])
    if errs:
        return (
            ExperimentOutcome.BLOCKED,
            "BLOCKED. The experiment recorded runtime errors: " + "; ".join(errs),
            "",
        )
    if not candidate_outcomes:
        return (
            ExperimentOutcome.INCONCLUSIVE,
            "INCONCLUSIVE. No candidate arm was measured, so the hypothesis was not tested.",
            "",
        )

    improved = [c for c in candidate_outcomes if c.outcome is ExperimentOutcome.IMPROVED]
    if improved:
        best = max(improved, key=lambda c: (c.primary_mean_delta or 0.0, c.label))
        others = [c for c in candidate_outcomes if c is not best]
        parts = [
            f"IMPROVED: {best.label!r} met the declared bar on {best.primary_metric}",
            f"({best.primary_mean_delta:+.4f} over {best.paired_question_count} paired "
            f"question(s); required {best.primary_required_delta})",
        ]
        if others:
            parts.append(
                "other arms: "
                + "; ".join(
                    f"{c.label} -> {c.outcome.value}"
                    f"{f' ({c.primary_mean_delta:+.4f})' if c.primary_mean_delta is not None else ''}"
                    for c in others
                )
            )
        parts.append(
            "The configuration is NOT promoted to the default: promotion requires an "
            "explicit, separate decision."
        )
        return ExperimentOutcome.IMPROVED, " ".join(parts), best.label

    regressed = [c for c in candidate_outcomes if c.outcome is ExperimentOutcome.REGRESSION]
    if regressed:
        return (
            ExperimentOutcome.REGRESSION,
            "REGRESSION. "
            + "; ".join(f"{c.label}: {c.outcome_explanation}" for c in regressed),
            "",
        )
    no_gain = [
        c for c in candidate_outcomes if c.outcome is ExperimentOutcome.NO_MEANINGFUL_IMPROVEMENT
    ]
    if no_gain:
        return (
            ExperimentOutcome.NO_MEANINGFUL_IMPROVEMENT,
            "NO_MEANINGFUL_IMPROVEMENT. "
            + "; ".join(f"{c.label}: {c.outcome_explanation}" for c in no_gain),
            "",
        )
    blocked = [c for c in candidate_outcomes if c.outcome is ExperimentOutcome.BLOCKED]
    if blocked and len(blocked) == len(candidate_outcomes):
        return (
            ExperimentOutcome.BLOCKED,
            "BLOCKED. " + "; ".join(f"{c.label}: {c.outcome_explanation}" for c in blocked),
            "",
        )
    return (
        ExperimentOutcome.INCONCLUSIVE,
        "INCONCLUSIVE. "
        + "; ".join(f"{c.label}: {c.outcome_explanation}" for c in candidate_outcomes),
        "",
    )


__all__ = [
    "CandidateOutcome",
    "DEFAULT_COMPARISON_METRICS",
    "ExperimentArm",
    "ExperimentOutcome",
    "ExperimentPerQuestionResult",
    "MEANINGFUL_DELTA",
    "MetricStats",
    "RetrievalExperiment",
    "RetrievalStrategyConfig",
    "decide_candidate_outcome",
    "decide_experiment_outcome",
    "metric_stats",
    "now_iso",
    "per_question_results",
    "question_set_fingerprint",
    "slugify",
]
