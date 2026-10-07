"""V9 Phase 6/7 regression tests: baseline vs candidate comparison.

Pure unit tests. scipy is used ONLY as an oracle where it happens to be
installed; the module itself is stdlib-only and the oracle tests skip if scipy
is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.comparison import (  # noqa: E402
    AcceptanceDecision,
    ComparisonOutcome,
    ExperimentProtocol,
    MetricDirection,
    MIN_RELIABLE_PAIRED_N,
    PairingVerdict,
    ProtocolMismatch,
    assert_same_protocol,
    check_protocol,
    compare_metric,
    compare_suite_explicit,
    decide_acceptance,
    exact_sign_test_p,
    metric_direction,
    student_t_two_sided_p,
)


def _protocol(label: str = "base", **overrides) -> ExperimentProtocol:
    base = dict(
        label=label,
        kb_id="kb_test",
        benchmark_name="bench",
        benchmark_fingerprint="fp_bench",
        corpus_fingerprint="fp_corpus",
        evaluator_name="deterministic-evidence",
        evaluator_version="v8.3",
        generator="extractive-mock",
        model="",
        prompt_version="aej-v1",
        answer_mode="abstain_if_unsupported",
        is_mock=True,
    )
    base.update(overrides)
    return ExperimentProtocol(**base)


class TestMetricDirections:
    def test_known_metrics_have_directions(self):
        assert metric_direction("recall_at_k") is MetricDirection.HIGHER_IS_BETTER
        assert metric_direction("unsupported_claim_rate") is MetricDirection.LOWER_IS_BETTER
        assert metric_direction("first_relevant_rank") is MetricDirection.LOWER_IS_BETTER

    def test_undeclared_metric_is_refused(self):
        """Assuming a direction can invert the verdict, so it must raise."""
        with pytest.raises(KeyError, match="no declared direction"):
            metric_direction("made_up_metric")


class TestLowerIsBetterIsNotInverted:
    def test_unsupported_claim_rate_drop_is_an_improvement(self):
        c = compare_metric(
            metric="unsupported_claim_rate",
            baseline={"q1": 0.5, "q2": 0.4},
            candidate={"q1": 0.2, "q2": 0.1},
        )
        assert c.outcome is ComparisonOutcome.IMPROVED
        assert c.improved_count == 2
        assert c.mean_delta is not None and c.mean_delta < 0

    def test_unsupported_claim_rate_rise_is_a_regression(self):
        c = compare_metric(
            metric="unsupported_claim_rate",
            baseline={"q1": 0.1},
            candidate={"q1": 0.4},
        )
        assert c.outcome is ComparisonOutcome.REGRESSED

    def test_recall_rise_is_an_improvement(self):
        c = compare_metric(
            metric="recall_at_k",
            baseline={"q1": 0.2, "q2": 0.3},
            candidate={"q1": 0.6, "q2": 0.7},
        )
        assert c.outcome is ComparisonOutcome.IMPROVED


class TestUnknownIsNotUnchanged:
    def test_no_measured_pair_is_unknown_not_unchanged(self):
        c = compare_metric(
            metric="recall_at_k",
            baseline={"q1": None, "q2": None},
            candidate={"q1": None, "q2": None},
        )
        assert c.outcome is ComparisonOutcome.UNKNOWN
        assert c.paired_count == 0
        assert c.unknown_count == 2
        assert "UNKNOWN, not UNCHANGED" in c.reason
        assert c.mean_delta is None

    def test_missing_side_is_excluded_not_treated_as_zero(self):
        """A None on one side must not be scored as a 0.0 regression."""
        c = compare_metric(
            metric="recall_at_k",
            baseline={"q1": 0.5, "q2": None},
            candidate={"q1": 0.5, "q2": 0.9},
        )
        assert c.paired_count == 1
        assert c.unknown_count == 1
        assert c.mean_delta == 0.0
        unknown = [d for d in c.deltas if d.delta is None]
        assert unknown[0].question_id == "q2"
        assert "unmeasured" in unknown[0].unknown_reason

    def test_all_ties_is_unknown_not_improved(self):
        c = compare_metric(
            metric="mrr",
            baseline={"q1": 0.5, "q2": 0.5},
            candidate={"q1": 0.5, "q2": 0.5},
        )
        assert c.outcome is ComparisonOutcome.UNKNOWN
        assert c.tie_count == 2
        assert "not the same as proof of none" in c.reason

    def test_split_movement_with_zero_mean_is_unknown(self):
        """Equal wins and losses with no net movement is not an improvement."""
        c = compare_metric(
            metric="recall_at_k",
            baseline={"q1": 0.2, "q2": 0.8},
            candidate={"q1": 0.4, "q2": 0.6},
        )
        assert c.improved_count == 1
        assert c.regressed_count == 1
        assert c.mean_delta == 0.0
        assert c.outcome is ComparisonOutcome.UNKNOWN
        assert "no net change" in c.reason


class TestProtocolIdentity:
    def test_identical_protocols_are_comparable(self):
        assert check_protocol(_protocol(), _protocol("cand")) == []

    def test_different_corpus_is_a_mismatch(self):
        diff = check_protocol(_protocol(), _protocol("cand", corpus_fingerprint="other"))
        assert diff == ["corpus_fingerprint"]

    def test_mock_vs_real_generator_is_a_mismatch(self):
        diff = check_protocol(_protocol(), _protocol("cand", is_mock=False))
        assert diff == ["is_mock"]

    def test_assert_same_protocol_raises_with_reasons(self):
        with pytest.raises(ProtocolMismatch, match="corpus_fingerprint"):
            assert_same_protocol(_protocol(), _protocol("cand", corpus_fingerprint="other"))

    def test_retrieval_config_may_differ(self):
        """The retrieval config is the variable under test, not a mismatch."""
        base = _protocol("base", retrieval_params={"strategy": "dense", "top_k": 5})
        cand = _protocol("cand", retrieval_params={"strategy": "hybrid", "top_k": 10})
        assert check_protocol(base, cand) == []


class TestSuiteComparison:
    def _suite(self, recall_values: dict, extra: dict | None = None):
        out = {"recall_at_k": recall_values}
        if extra:
            out.update(extra)
        return out

    def test_full_overlap_is_comparable(self):
        res = compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            baseline_metrics=self._suite({"q1": 0.2, "q2": 0.3}),
            candidate_metrics=self._suite({"q1": 0.5, "q2": 0.6}),
        )
        assert res.pairing is PairingVerdict.COMPARABLE
        assert res.improved_metric_count == 1
        assert res.comparable_question_ids == ["q1", "q2"]

    def test_partial_overlap_is_inconclusive(self):
        res = compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            baseline_metrics=self._suite({"q1": 0.2, "q2": 0.3}),
            candidate_metrics=self._suite({"q1": 0.5, "q3": 0.6}),
        )
        assert res.pairing is PairingVerdict.INCONCLUSIVE
        assert res.comparable_question_ids == ["q1"]
        assert "SHARED questions" in res.pairing_reason

    def test_no_overlap_is_not_comparable(self):
        res = compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            baseline_metrics=self._suite({"q1": 0.2}),
            candidate_metrics=self._suite({"q9": 0.5}),
        )
        assert res.pairing is PairingVerdict.NOT_COMPARABLE
        assert res.comparable_question_ids == []

    def test_protocol_mismatch_blocks_the_comparison(self):
        res = compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid", corpus_fingerprint="different"),
            baseline_metrics=self._suite({"q1": 0.2}),
            candidate_metrics=self._suite({"q1": 0.9}),
        )
        assert res.pairing is PairingVerdict.NOT_COMPARABLE
        assert res.protocol_differences == ["corpus_fingerprint"]
        assert any("NOT valid" in n for n in res.notes)
        # The metric row is still computed for inspection, but is not a verdict.
        assert res.metric_comparisons[0].outcome is ComparisonOutcome.IMPROVED

    def test_no_single_magic_score_is_produced(self):
        """The result exposes per-metric rows and counts, never one blended number."""
        res = compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            baseline_metrics={
                "recall_at_k": {"q1": 0.2},
                "unsupported_claim_rate": {"q1": 0.1},
            },
            candidate_metrics={
                "recall_at_k": {"q1": 0.5},
                "unsupported_claim_rate": {"q1": 0.4},
            },
        )
        assert res.improved_metric_count == 1
        assert res.regressed_metric_count == 1
        assert not hasattr(res, "overall_score")

    def test_small_sample_is_marked_exploratory(self):
        res = compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            baseline_metrics=self._suite({f"q{i}": 0.1 for i in range(5)}),
            candidate_metrics=self._suite({f"q{i}": 0.5 for i in range(5)}),
        )
        assert res.exploratory is True
        assert any("EXPLORATORY" in n for n in res.notes)


class TestAcceptanceDecision:
    def _comparison(self, recall: dict, unsupported: dict, n: int = 10):
        return compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            baseline_metrics={
                "recall_at_k": {f"q{i}": 0.2 for i in range(n)},
                "unsupported_claim_rate": {f"q{i}": 0.1 for i in range(n)},
            },
            candidate_metrics={
                "recall_at_k": recall,
                "unsupported_claim_rate": unsupported,
            },
        )

    def test_required_improvement_with_clean_guardrails_is_accepted(self):
        comp = self._comparison(
            {f"q{i}": 0.6 for i in range(10)}, {f"q{i}": 0.1 for i in range(10)}
        )
        decision = decide_acceptance(
            comp,
            required_improvements=["recall_at_k"],
            guardrail_metrics=["unsupported_claim_rate"],
            reviewed_by="test",
        )
        assert decision.decision is AcceptanceDecision.ACCEPT

    def test_guardrail_regression_is_rejected_even_when_required_improves(self):
        comp = self._comparison(
            {f"q{i}": 0.9 for i in range(10)}, {f"q{i}": 0.5 for i in range(10)}
        )
        decision = decide_acceptance(
            comp,
            required_improvements=["recall_at_k"],
            guardrail_metrics=["unsupported_claim_rate"],
            reviewed_by="test",
        )
        assert decision.decision is AcceptanceDecision.REJECT
        assert decision.guardrail_metrics_regressed == ["unsupported_claim_rate"]

    def test_unmet_required_metric_is_rejected(self):
        comp = self._comparison(
            {f"q{i}": 0.2 for i in range(10)}, {f"q{i}": 0.1 for i in range(10)}
        )
        decision = decide_acceptance(
            comp,
            required_improvements=["recall_at_k"],
            guardrail_metrics=[],
            reviewed_by="test",
        )
        assert decision.decision is AcceptanceDecision.REJECT

    def test_not_comparable_can_never_be_accepted(self):
        comp = compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid", corpus_fingerprint="different"),
            baseline_metrics={"recall_at_k": {"q1": 0.2}},
            candidate_metrics={"recall_at_k": {"q1": 0.9}},
        )
        decision = decide_acceptance(
            comp, required_improvements=["recall_at_k"], guardrail_metrics=[]
        )
        assert decision.decision is AcceptanceDecision.NOT_VALID
        assert decision.invalid_because

    def test_acceptance_records_the_small_sample_caveat(self):
        comp = self._comparison(
            {f"q{i}": 0.9 for i in range(3)}, {f"q{i}": 0.1 for i in range(3)}, n=3
        )
        decision = decide_acceptance(
            comp,
            required_improvements=["recall_at_k"],
            guardrail_metrics=["unsupported_claim_rate"],
        )
        assert decision.decision is AcceptanceDecision.ACCEPT
        assert "not on statistical significance" in decision.reason

    def test_min_reliable_n_floor_is_thirty(self):
        assert MIN_RELIABLE_PAIRED_N == 30


class TestStatisticsAgainstScipyOracle:
    """Verify the stdlib implementations against scipy where it is installed."""

    @pytest.mark.parametrize("df", [1, 2, 5, 10, 20, 27, 50, 100])
    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0])
    def test_student_t_matches_scipy(self, df, t):
        stats = pytest.importorskip("scipy.stats")
        assert student_t_two_sided_p(t, df) == pytest.approx(
            float(2 * stats.t.sf(abs(t), df)), abs=1e-9
        )

    @pytest.mark.parametrize("positive", [0, 1, 5, 10, 14])
    @pytest.mark.parametrize("negative", [0, 1, 5, 10, 14])
    def test_sign_test_matches_scipy(self, positive, negative):
        stats = pytest.importorskip("scipy.stats")
        if positive + negative == 0:
            assert exact_sign_test_p(positive, negative) is None
            return
        expected = float(
            min(1.0, 2 * stats.binom.cdf(min(positive, negative), positive + negative, 0.5))
        )
        assert exact_sign_test_p(positive, negative) == pytest.approx(expected, abs=1e-12)

    def test_normal_approximation_would_have_been_anti_conservative(self):
        """Regression guard: the t p-value must NOT be a normal approximation.

        At df=1 the t distribution has far heavier tails than the normal, so a
        normal approximation understates p badly. Asserting a large gap keeps a
        future 'simplification' from silently reintroducing the overclaim.
        """
        t, df = 3.0, 1
        exact = student_t_two_sided_p(t, df)
        normal_approx = 2 * (1 - 0.5 * (1 + __import__("math").erf(t / __import__("math").sqrt(2))))
        assert exact > normal_approx * 2
