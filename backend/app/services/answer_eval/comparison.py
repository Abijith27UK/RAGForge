"""Baseline vs candidate comparison (V9 Phases 6/7).

Answers one question: **did the candidate configuration actually improve
retrieval and answer quality against the same benchmark, corpus and evaluator?**

## Design rules this module follows

1. **No single magic score.** ``compare_suite`` returns one comparison PER
   METRIC. A collapsed headline number would let a large nDCG gain hide a
   hallucination-rate regression. The caller decides what matters.
2. **Paired, question-level, and explicit about gaps.** Only questions with a
   measured value on BOTH sides are paired. The rest are counted as
   ``unknown`` and excluded from every mean — never substituted with 0.0.
3. **Direction is declared, not assumed.** A lower unsupported-claim rate is an
   improvement while a lower recall is a regression, so every metric carries a
   ``MetricDirection`` from a registry. Guessing the direction is how an
   evaluation system reports an improvement that is actually a failure.
4. **UNKNOWN is a first-class outcome.** Too few paired observations, or no
   measured pair at all, yields ``UNKNOWN`` with a reason — not ``UNCHANGED``.
   "The candidate did not regress" and "we could not measure it" are different
   claims.
5. **Statistics at n=28 are labelled exploratory.** The exact sign test is
   assumption-light and reported as the primary test; the paired t-test uses a
   real Student's t distribution (regularized incomplete beta, no normal
   approximation) but is reported as secondary and marked exploratory below
   ``MIN_RELIABLE_PAIRED_N``. Nothing here claims significance.

Everything in this module is pure: given the same inputs it returns the same
output, and it never touches the filesystem or the network.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

#: Below this many paired questions a paired test is directional at best. The
#: shipped Automobile benchmark has 28 questions, so its results are ALWAYS
#: exploratory and every report must say so.
MIN_RELIABLE_PAIRED_N = 30

#: Below this many paired questions a test carries no information at all.
MIN_TESTABLE_PAIRED_N = 8


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


#: Direction for every metric this project compares. Declared once so no caller
#: can silently compare a "lower is better" metric as if higher were better.
METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    # retrieval quality
    "recall_at_k": MetricDirection.HIGHER_IS_BETTER,
    "precision_at_k": MetricDirection.HIGHER_IS_BETTER,
    "mrr": MetricDirection.HIGHER_IS_BETTER,
    "ndcg_at_k": MetricDirection.HIGHER_IS_BETTER,
    "required_recall_at_depth": MetricDirection.HIGHER_IS_BETTER,
    "retrieval_hit_rate": MetricDirection.HIGHER_IS_BETTER,
    # answer quality
    "question_answer_relevance": MetricDirection.HIGHER_IS_BETTER,
    "key_point_recall": MetricDirection.HIGHER_IS_BETTER,
    "reference_answer_similarity": MetricDirection.HIGHER_IS_BETTER,
    "citation_precision": MetricDirection.HIGHER_IS_BETTER,
    "citation_recall": MetricDirection.HIGHER_IS_BETTER,
    "citation_completeness": MetricDirection.HIGHER_IS_BETTER,
    "evidence_support_rate": MetricDirection.HIGHER_IS_BETTER,
    "supported_claim_ratio": MetricDirection.HIGHER_IS_BETTER,
    "grounding_state_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "abstention_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "correctness": MetricDirection.HIGHER_IS_BETTER,
    "pass_rate": MetricDirection.HIGHER_IS_BETTER,
    "latency_ms": MetricDirection.LOWER_IS_BETTER,
    "latency_total_ms": MetricDirection.LOWER_IS_BETTER,
    # answer pathologies: lower is better, and a rise is a regression
    "unsupported_claim_rate": MetricDirection.LOWER_IS_BETTER,
    "unsupported_citation_rate": MetricDirection.LOWER_IS_BETTER,
    "fabricated_citation_rate": MetricDirection.LOWER_IS_BETTER,
    "contradiction_rate": MetricDirection.LOWER_IS_BETTER,
    "hallucination_rate": MetricDirection.LOWER_IS_BETTER,
    "false_supported_rate": MetricDirection.LOWER_IS_BETTER,
    "false_unsupported_rate": MetricDirection.LOWER_IS_BETTER,
    "relevance_failure_rate": MetricDirection.LOWER_IS_BETTER,
    "excess_information": MetricDirection.LOWER_IS_BETTER,
    "first_relevant_rank": MetricDirection.LOWER_IS_BETTER,
}


def metric_direction(metric: str) -> MetricDirection:
    """Direction for a known metric; raises for an undeclared one.

    Refusing an unknown metric is deliberate: an undeclared direction would have
    to be assumed, and assuming it wrong inverts the verdict.
    """
    try:
        return METRIC_DIRECTIONS[metric]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"metric {metric!r} has no declared direction; add it to "
            f"METRIC_DIRECTIONS before comparing it (assuming a direction can "
            f"invert the improvement verdict)"
        ) from exc


class ComparisonOutcome(str, Enum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    UNCHANGED = "UNCHANGED"
    UNKNOWN = "UNKNOWN"


class PairingVerdict(str, Enum):
    COMPARABLE = "COMPARABLE"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class ProtocolMismatch(RuntimeError):
    """Two experiments were not run under the same protocol."""


class ExperimentProtocol(BaseModel):
    """The conditions two runs must share before a comparison is meaningful.

    Every field is recorded so a comparison can be audited: if the benchmark
    content, the corpus, the evaluator, the model or the answer mode differ, a
    delta between the two runs is not attributable to the retrieval change.
    """

    label: str
    benchmark_name: str = ""
    benchmark_fingerprint: str = ""
    corpus_fingerprint: str = ""
    evaluator_name: str = ""
    evaluator_version: str = ""
    evaluator_is_model_based: bool = False
    entailment_provider: str = ""
    generator: str = ""
    model: str = ""
    prompt_version: str = ""
    answer_mode: str = ""
    is_mock: bool = False
    kb_id: str = ""
    #: Retrieval configuration for this side, recorded verbatim.
    retrieval_params: dict[str, Any] = Field(default_factory=dict)
    config_fingerprint: str = ""

    def identity_fields(self) -> dict[str, str]:
        """The fields that MUST match for a valid comparison."""
        return {
            "kb_id": self.kb_id,
            "benchmark_name": self.benchmark_name,
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "generator": self.generator,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "answer_mode": self.answer_mode,
        }


def check_protocol(
    baseline: ExperimentProtocol,
    candidate: ExperimentProtocol,
) -> list[str]:
    """Return the identity fields that DIFFER between two protocols.

    Empty means the two are comparable. A different retrieval configuration is
    expected — that is the variable under test — so retrieval params and the
    config fingerprint are deliberately excluded from the check.
    """
    if baseline.is_mock != candidate.is_mock:
        # A mock generator answers from retrieved text verbatim; a real model
        # does not. Comparing them measures the generator, not the retrieval.
        return ["is_mock"]
    b = baseline.identity_fields()
    c = candidate.identity_fields()
    return sorted(k for k in b if b[k] != c[k])


def assert_same_protocol(baseline: ExperimentProtocol, candidate: ExperimentProtocol) -> None:
    """Raise when a comparison would be scientifically invalid."""
    differing = check_protocol(baseline, candidate)
    if differing:
        raise ProtocolMismatch(
            f"baseline ({baseline.label!r}) and candidate ({candidate.label!r}) "
            f"differ on {differing}; a delta between them is not attributable to "
            f"the retrieval change. Re-run under an identical protocol."
        )


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    maxit, eps, fpmin = 200, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) — implemented here so no undeclared scipy dependency is needed."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: int) -> float:
    """Exact two-sided p-value for Student's t with ``df`` degrees of freedom.

    Uses the incomplete beta identity ``P(|T| > t) = I_{df/(df+t^2)}(df/2, 1/2)``
    rather than a normal approximation, because a normal approximation at n=28
    understates p and would let a weak result be described as significant.
    """
    if df <= 0:
        return 1.0
    if math.isinf(t):
        return 0.0
    return max(0.0, min(1.0, regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))))


def exact_sign_test_p(positive: int, negative: int) -> float | None:
    """Exact two-sided sign-test p-value (binomial, p=0.5). None if no data.

    The sign test assumes only that each pair is independent and that under the
    null an improvement and a regression are equally likely. It is reported as
    the PRIMARY test because that assumption is far weaker than normality.
    """
    n = positive + negative
    if n == 0:
        return None
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


class PairedDelta(BaseModel):
    """One question's baseline and candidate values."""

    question_id: str
    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = Field(
        default=None,
        description="candidate - baseline, in the metric's own units. None when "
        "either side was unmeasured.",
    )
    outcome: ComparisonOutcome = ComparisonOutcome.UNKNOWN
    unknown_reason: str = ""


class SignificanceTest(BaseModel):
    """One statistical test result, always with its caveats attached."""

    name: str
    statistic: float | None = None
    p_value: float | None = None
    n: int = 0
    #: True when n is too small for the test to support a claim.
    exploratory: bool = True
    note: str = ""
    assumptions: str = ""


class MetricComparison(BaseModel):
    """Baseline vs candidate for ONE metric. Never merged with another metric."""

    metric: str
    direction: MetricDirection
    questions: int = 0
    paired_count: int = 0
    unknown_count: int = 0

    baseline_mean: float | None = None
    candidate_mean: float | None = None
    mean_delta: float | None = None
    median_delta: float | None = None
    min_delta: float | None = None
    max_delta: float | None = None

    improved_count: int = 0
    regressed_count: int = 0
    tie_count: int = 0
    outcome: ComparisonOutcome = ComparisonOutcome.UNKNOWN
    reason: str = ""

    #: Outcome counts broken down per question, so the UI can show which
    #: questions moved rather than only the aggregate.
    deltas: list[PairedDelta] = Field(default_factory=list)
    tie_epsilon: float = 0.0

    def changed_questions(self) -> list[PairedDelta]:
        """Questions that improved or regressed, in question-id order."""
        return [
            d
            for d in self.deltas
            if d.outcome in (ComparisonOutcome.IMPROVED, ComparisonOutcome.REGRESSED)
        ]


class ExperimentComparison(BaseModel):
    """The full baseline-vs-candidate result: per-metric, plus pairing validity."""

    baseline_label: str = ""
    candidate_label: str = ""
    baseline_config_fingerprint: str = ""
    candidate_config_fingerprint: str = ""
    benchmark_fingerprint: str = ""

    pairing: PairingVerdict = PairingVerdict.NOT_COMPARABLE
    pairing_reason: str = ""
    comparable_question_ids: list[str] = Field(default_factory=list)
    protocol_differences: list[str] = Field(default_factory=list)

    metric_comparisons: list[MetricComparison] = Field(default_factory=list)
    significance_tests: list[SignificanceTest] = Field(default_factory=list)

    improved_metric_count: int = 0
    regressed_metric_count: int = 0
    unchanged_metric_count: int = 0
    unknown_metric_count: int = 0

    exploratory: bool = Field(
        default=True,
        description="True whenever the paired sample is too small for inference. "
        "The shipped benchmark's 28 questions always set this.",
    )
    notes: list[str] = Field(default_factory=list)

    def comparison_for(self, metric: str) -> MetricComparison | None:
        for c in self.metric_comparisons:
            if c.metric == metric:
                return c
        return None

    def improved_metrics(self) -> list[str]:
        return [c.metric for c in self.metric_comparisons if c.outcome is ComparisonOutcome.IMPROVED]

    def regressed_metrics(self) -> list[str]:
        return [c.metric for c in self.metric_comparisons if c.outcome is ComparisonOutcome.REGRESSED]


def _pair_values(
    baseline: dict[str, float | None],
    candidate: dict[str, float | None],
) -> tuple[list[PairedDelta], list[str]]:
    """Pair the two sides on shared question ids. Returns (deltas, unpaired)."""
    unpaired = sorted(set(baseline) ^ set(candidate))
    deltas: list[PairedDelta] = []
    for qid in sorted(set(baseline) & set(candidate)):
        b, c = baseline[qid], candidate[qid]
        if b is None or c is None:
            missing = "baseline" if b is None else "candidate"
            deltas.append(
                PairedDelta(
                    question_id=qid,
                    baseline=b,
                    candidate=c,
                    delta=None,
                    outcome=ComparisonOutcome.UNKNOWN,
                    unknown_reason=f"{missing} value is unmeasured for this question",
                )
            )
        else:
            deltas.append(
                PairedDelta(question_id=qid, baseline=b, candidate=c, delta=round(c - b, 6))
            )
    return deltas, unpaired


def compare_metric(
    *,
    metric: str,
    baseline: dict[str, float | None],
    candidate: dict[str, float | None],
    tie_epsilon: float = 0.0,
) -> MetricComparison:
    """Pair one metric across questions and classify every question's delta.

    ``tie_epsilon`` sets how large a delta must be to count as a change. It
    defaults to 0.0 (any measured difference counts) because a non-zero epsilon
    is a judgement about what "no change" means and must be stated by the caller.
    """
    direction = metric_direction(metric)
    deltas, _unpaired = _pair_values(baseline, candidate)

    measured = [d for d in deltas if d.delta is not None]
    sign = 1.0 if direction is MetricDirection.HIGHER_IS_BETTER else -1.0

    improved = regressed = tie = 0
    for d in deltas:
        if d.delta is None:
            continue
        oriented = d.delta * sign
        if oriented > tie_epsilon:
            d.outcome = ComparisonOutcome.IMPROVED
            improved += 1
        elif oriented < -tie_epsilon:
            d.outcome = ComparisonOutcome.REGRESSED
            regressed += 1
        else:
            d.outcome = ComparisonOutcome.UNCHANGED
            tie += 1

    values = [d.delta for d in measured]
    baseline_values = [d.baseline for d in measured if d.baseline is not None]
    candidate_values = [d.candidate for d in measured if d.candidate is not None]

    def _mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 6) if vals else None

    def _median(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        mid = len(s) // 2
        if len(s) % 2:
            return round(s[mid], 6)
        return round((s[mid - 1] + s[mid]) / 2.0, 6)

    comparison = MetricComparison(
        metric=metric,
        direction=direction,
        questions=len(deltas),
        paired_count=len(values),
        unknown_count=len(deltas) - len(values),
        baseline_mean=_mean(baseline_values),
        candidate_mean=_mean(candidate_values),
        mean_delta=_mean(values),
        median_delta=_median(values),
        min_delta=round(min(values), 6) if values else None,
        max_delta=round(max(values), 6) if values else None,
        improved_count=improved,
        regressed_count=regressed,
        tie_count=tie,
        deltas=deltas,
        tie_epsilon=tie_epsilon,
    )

    if not values:
        comparison.outcome = ComparisonOutcome.UNKNOWN
        comparison.reason = (
            "no question has a measured value on both sides, so no change can be "
            "reported; this is UNKNOWN, not UNCHANGED"
        )
    elif improved == 0 and regressed == 0:
        comparison.outcome = ComparisonOutcome.UNKNOWN
        comparison.reason = (
            f"all {len(values)} paired question(s) are ties at epsilon="
            f"{tie_epsilon}; with every question unchanged there is no evidence "
            f"of a difference, which is not the same as proof of none"
        )
    elif improved > regressed:
        comparison.outcome = ComparisonOutcome.IMPROVED
        comparison.reason = (
            f"{improved} question(s) improved vs {regressed} regressed "
            f"(direction: {direction.value}, epsilon={tie_epsilon})"
        )
    elif regressed > improved:
        comparison.outcome = ComparisonOutcome.REGRESSED
        comparison.reason = (
            f"{regressed} question(s) regressed vs {improved} improved "
            f"(direction: {direction.value}, epsilon={tie_epsilon})"
        )
    else:
        # Equal counts in both directions: the mean decides, and when the mean
        # is zero the honest answer is that the metric did not move either way.
        if comparison.mean_delta is not None and abs(comparison.mean_delta) > tie_epsilon:
            oriented = comparison.mean_delta * sign
            comparison.outcome = (
                ComparisonOutcome.IMPROVED if oriented > 0 else ComparisonOutcome.REGRESSED
            )
            comparison.reason = (
                f"{improved} improved and {regressed} regressed, but the mean delta "
                f"{comparison.mean_delta:+g} decides the direction "
                f"(direction: {direction.value})"
            )
        else:
            comparison.outcome = ComparisonOutcome.UNKNOWN
            comparison.reason = (
                f"{improved} improved and {regressed} regressed with a mean delta of "
                f"{comparison.mean_delta!r}; the metric moved in both directions with "
                f"no net change, so the outcome is UNKNOWN"
            )
    return comparison


def paired_significance(
    comparison: MetricComparison,
) -> list[SignificanceTest]:
    """Sign test (primary) and paired t-test (secondary) for one metric.

    Both are labelled with their assumptions and marked exploratory when the
    paired sample is below ``MIN_RELIABLE_PAIRED_N``. Neither is presented as
    proof: at the shipped benchmark's size a p-value is directional at best.
    """
    deltas = [d.delta for d in comparison.deltas if d.delta is not None]
    n = len(deltas)
    sign = 1.0 if comparison.direction is MetricDirection.HIGHER_IS_BETTER else -1.0
    oriented = [d * sign for d in deltas]
    positive = sum(1 for v in oriented if v > 0)
    negative = sum(1 for v in oriented if v < 0)

    exploratory = n < MIN_RELIABLE_PAIRED_N
    tests: list[SignificanceTest] = []

    p_sign = exact_sign_test_p(positive, negative)
    tests.append(
        SignificanceTest(
            name="exact_sign_test",
            statistic=float(positive - negative) if p_sign is not None else None,
            p_value=round(p_sign, 6) if p_sign is not None else None,
            n=n,
            exploratory=exploratory,
            note=(
                "primary test: counts questions that improved vs regressed and "
                "ignores effect size. Ties are excluded"
            ),
            assumptions=(
                "each question is an independent pair; under the null an "
                "improvement and a regression are equally likely. No normality "
                "assumption"
            ),
        )
    )

    if n >= 2:
        mean = sum(oriented) / n
        var = sum((v - mean) ** 2 for v in oriented) / (n - 1)
        sd = math.sqrt(var)
        if sd > 0:
            t = mean / (sd / math.sqrt(n))
            p_t = student_t_two_sided_p(t, n - 1)
            tests.append(
                SignificanceTest(
                    name="paired_student_t",
                    statistic=round(t, 6),
                    p_value=round(p_t, 6),
                    n=n,
                    exploratory=exploratory,
                    note=(
                        "secondary test: uses effect size, so one large change can "
                        "dominate. Reported alongside the sign test, never instead"
                    ),
                    assumptions=(
                        "per-question deltas are independent and approximately "
                        "normally distributed — implausible for bounded metrics "
                        "at n<30, which is why it is secondary"
                    ),
                )
            )
    return tests


def compare_suite_explicit(
    *,
    baseline_label: str,
    candidate_label: str,
    baseline: ExperimentProtocol,
    candidate: ExperimentProtocol,
    baseline_metrics: dict[str, dict[str, float | None]],
    candidate_metrics: dict[str, dict[str, float | None]],
    tie_epsilon: float = 0.0,
    benchmark_fingerprint: str = "",
) -> ExperimentComparison:
    """Compare two suites of per-question metrics under an explicit protocol."""
    differing = check_protocol(baseline, candidate)

    shared_metrics = sorted(set(baseline_metrics) & set(candidate_metrics))
    only_baseline = sorted(set(baseline_metrics) - set(candidate_metrics))
    only_candidate = sorted(set(candidate_metrics) - set(baseline_metrics))

    result = ExperimentComparison(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_config_fingerprint=baseline.config_fingerprint,
        candidate_config_fingerprint=candidate.config_fingerprint,
        benchmark_fingerprint=benchmark_fingerprint or baseline.benchmark_fingerprint,
        protocol_differences=differing,
        exploratory=True,
    )

    # -- pairing validity ---------------------------------------------------
    all_qids = set()
    for d in list(baseline_metrics.values()) + list(candidate_metrics.values()):
        all_qids.update(d)
    paired_qids: set[str] = set()
    for metric in shared_metrics:
        paired_qids |= set(baseline_metrics[metric]) & set(candidate_metrics[metric])

    if differing:
        result.pairing = PairingVerdict.NOT_COMPARABLE
        result.pairing_reason = (
            f"protocol mismatch on {differing}: the benchmark, corpus, evaluator, "
            f"model or answer mode differs, so no delta is attributable to the "
            f"retrieval change. Metrics are still listed but must not be read as "
            f"a comparison"
        )
    elif not shared_metrics:
        result.pairing = PairingVerdict.NOT_COMPARABLE
        result.pairing_reason = "the two sides share no metric names"
    elif not paired_qids:
        result.pairing = PairingVerdict.NOT_COMPARABLE
        result.pairing_reason = (
            "the two sides share no question ids, so nothing can be paired"
        )
    elif all_qids and paired_qids == all_qids:
        result.pairing = PairingVerdict.COMPARABLE
        result.pairing_reason = (
            f"identical question sets ({len(paired_qids)} question(s)) and an "
            f"identical protocol; the full comparison is valid"
        )
    else:
        result.pairing = PairingVerdict.INCONCLUSIVE
        result.pairing_reason = (
            f"question subsets differ: {len(paired_qids)} of {len(all_qids)} "
            f"question(s) are shared. Deltas are computed on the SHARED questions "
            f"only and the verdict is INCONCLUSIVE so a partial overlap is never "
            f"presented as a full-set comparison"
        )

    result.comparable_question_ids = sorted(paired_qids)

    # -- per-metric comparison ---------------------------------------------
    for metric in shared_metrics:
        comparison = compare_metric(
            metric=metric,
            baseline=baseline_metrics[metric],
            candidate=candidate_metrics[metric],
            tie_epsilon=tie_epsilon,
        )
        result.metric_comparisons.append(comparison)
        result.significance_tests.extend(paired_significance(comparison))

    if only_baseline:
        result.notes.append(
            f"metrics present only in the baseline (not compared): {only_baseline}"
        )
    if only_candidate:
        result.notes.append(
            f"metrics present only in the candidate (not compared): {only_candidate}"
        )

    n_paired_max = max((c.paired_count for c in result.metric_comparisons), default=0)
    result.exploratory = n_paired_max < MIN_RELIABLE_PAIRED_N
    if result.exploratory:
        result.notes.append(
            f"largest paired sample is {n_paired_max} question(s), below the "
            f"{MIN_RELIABLE_PAIRED_N}-question floor for inference: every "
            f"statistical result here is EXPLORATORY and must not be reported as "
            f"significance"
        )
    if n_paired_max < MIN_TESTABLE_PAIRED_N:
        result.notes.append(
            f"only {n_paired_max} paired observation(s): no statistical test can "
            f"support a claim at this size"
        )
    if result.pairing is PairingVerdict.NOT_COMPARABLE:
        result.notes.append(
            "the comparison is NOT valid; per-metric outcomes are reported for "
            "inspection only and no accept/reject decision may rest on them"
        )

    result.improved_metric_count = sum(
        1 for c in result.metric_comparisons if c.outcome is ComparisonOutcome.IMPROVED
    )
    result.regressed_metric_count = sum(
        1 for c in result.metric_comparisons if c.outcome is ComparisonOutcome.REGRESSED
    )
    result.unchanged_metric_count = sum(
        1 for c in result.metric_comparisons if c.outcome is ComparisonOutcome.UNCHANGED
    )
    result.unknown_metric_count = sum(
        1 for c in result.metric_comparisons if c.outcome is ComparisonOutcome.UNKNOWN
    )
    return result


class AcceptanceDecision(str, Enum):
    """Whether a change should be accepted. Never inferred from one metric."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_VALID = "NOT_VALID"


class AcceptanceRationale(BaseModel):
    """A recorded accept/reject decision with the evidence behind it."""

    decision: AcceptanceDecision
    reason: str
    required_metrics_improved: list[str] = Field(default_factory=list)
    required_metrics_regressed: list[str] = Field(default_factory=list)
    guardrail_metrics_regressed: list[str] = Field(default_factory=list)
    invalid_because: list[str] = Field(default_factory=list)
    reviewed_by: str = ""


def decide_acceptance(
    comparison: ExperimentComparison,
    *,
    required_improvements: list[str],
    guardrail_metrics: list[str],
    reviewed_by: str = "",
) -> AcceptanceRationale:
    """Decide ACCEPT / REJECT / INCONCLUSIVE from declared criteria.

    The CALLER declares which metrics must improve and which must not regress.
    This function does not invent a pass mark: a change is accepted only when
    every required metric improved and no guardrail metric regressed. Anything
    else is REJECT or INCONCLUSIVE, and a not-comparable experiment can never be
    accepted.
    """
    rationale = AcceptanceRationale(
        decision=AcceptanceDecision.INCONCLUSIVE,
        reason="",
        reviewed_by=reviewed_by,
    )
    if comparison.pairing is PairingVerdict.NOT_COMPARABLE:
        rationale.decision = AcceptanceDecision.NOT_VALID
        rationale.invalid_because = [comparison.pairing_reason]
        rationale.reason = (
            "the experiment is not comparable, so no change can be accepted or "
            "rejected on this evidence"
        )
        return rationale

    for metric in required_improvements:
        c = comparison.comparison_for(metric)
        if c is None:
            rationale.required_metrics_improved.append(f"{metric} (not compared: missing)")
            continue
        if c.outcome is ComparisonOutcome.IMPROVED:
            rationale.required_metrics_improved.append(metric)
        else:
            rationale.required_metrics_improved.append(f"{metric} ({c.outcome.value.lower()})")

    for metric in guardrail_metrics:
        c = comparison.comparison_for(metric)
        if c is not None and c.outcome is ComparisonOutcome.REGRESSED:
            rationale.guardrail_metrics_regressed.append(metric)

    unmet = [m for m in rationale.required_metrics_improved if "(" in m]
    rationale.required_metrics_regressed = unmet

    if rationale.guardrail_metrics_regressed:
        rationale.decision = AcceptanceDecision.REJECT
        rationale.reason = (
            f"guardrail metric(s) regressed: "
            f"{', '.join(rationale.guardrail_metrics_regressed)}. A gain elsewhere "
            f"does not license a regression in a declared guardrail"
        )
    elif unmet:
        rationale.decision = AcceptanceDecision.REJECT
        rationale.reason = (
            f"required metric(s) did not improve: {', '.join(unmet)}"
        )
    else:
        rationale.decision = AcceptanceDecision.ACCEPT
        rationale.reason = (
            f"all {len(required_improvements)} required metric(s) improved and no "
            f"guardrail metric regressed. "
            + (
                "The paired sample is below the inference floor, so this acceptance "
                "rests on the recorded measurement, not on statistical significance."
                if comparison.exploratory
                else ""
            )
        ).strip()
    return rationale


__all__ = [
    "AcceptanceDecision",
    "AcceptanceRationale",
    "ComparisonOutcome",
    "ExperimentComparison",
    "ExperimentProtocol",
    "METRIC_DIRECTIONS",
    "MIN_RELIABLE_PAIRED_N",
    "MIN_TESTABLE_PAIRED_N",
    "MetricComparison",
    "MetricDirection",
    "PairedDelta",
    "PairingVerdict",
    "ProtocolMismatch",
    "SignificanceTest",
    "assert_same_protocol",
    "check_protocol",
    "compare_metric",
    "compare_suite_explicit",
    "decide_acceptance",
    "exact_sign_test_p",
    "metric_direction",
    "paired_significance",
    "regularized_incomplete_beta",
    "student_t_two_sided_p",
]
