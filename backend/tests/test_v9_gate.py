"""V9 Phase 8 regression tests: off-domain gate measurement.

The headline tests use the REAL V8 measurement documented in
``services/answer_eval/relevance.py``: 286 stored on-domain answers from
``kb_f278c283c748`` plus exactly ONE known off-domain answer, compared under
plain term coverage and IDF-weighted coverage.

| method            | on-domain min | known off-domain |
|-------------------|---------------|------------------|
| plain coverage    | 0.333         | 0.400 (INVERTED) |
| IDF-weighted      | 0.300         | 0.260            |

These numbers pin the finding, so a future change that "fixes" the gate by
switching back to plain coverage would fail here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.gate import (  # noqa: E402
    HIGHER_MEANS_OFF_DOMAIN,
    MIN_MEDIAN_GAP,
    MIN_SAMPLES_PER_SIDE_FOR_A_RECOMMENDATION,
    GateStudyError,
    false_negatives,
    false_positives,
    gate_study,
    on_off_distribution,
)

#: The current unweighted gate threshold, and the shipped IDF threshold. The
#: shipped answer-relevance threshold is MIN_RELEVANCE_TO_PASS = 0.30.
SHIPPED_THRESHOLD = 0.30

#: Real measured values (relevance.py module docstring).
PLAIN_ON_DOMAIN_MIN = 0.333
PLAIN_OFF_DOMAIN = 0.400
IDF_ON_DOMAIN_MIN = 0.300
IDF_OFF_DOMAIN = 0.260


class TestRealV8Measurement:
    """The documented V8 finding, reproduced as a regression guard."""

    def test_plain_coverage_is_inverted_on_the_real_numbers(self):
        """Plain coverage scores the off-domain answer ABOVE the on-domain minimum."""
        scores = [PLAIN_ON_DOMAIN_MIN, PLAIN_OFF_DOMAIN]
        labels = ["on", "off"]
        # At the shipped threshold the off-domain answer is NOT rejected.
        fp = false_positives(scores, labels, SHIPPED_THRESHOLD)
        assert len(fp) == 1
        assert fp[0]["score"] == PLAIN_OFF_DOMAIN
        assert fp[0]["score"] >= PLAIN_ON_DOMAIN_MIN, (
            "the inversion: plain coverage ranked the off-domain answer higher "
            "than every on-domain answer"
        )

    def test_idf_weighting_rejects_the_known_off_domain_answer(self):
        scores = [IDF_ON_DOMAIN_MIN, IDF_OFF_DOMAIN]
        labels = ["on", "off"]
        assert false_positives(scores, labels, SHIPPED_THRESHOLD) == []
        assert false_negatives(scores, labels, SHIPPED_THRESHOLD) == []

    def test_idf_margin_is_thin_and_reported_as_such(self):
        """The separation is 0.04 on ONE negative sample; that must be visible."""
        margin = IDF_ON_DOMAIN_MIN - IDF_OFF_DOMAIN
        assert margin == pytest.approx(0.04)
        study = gate_study(
            scores=[IDF_ON_DOMAIN_MIN, IDF_OFF_DOMAIN],
            labels=["on", "off"],
            threshold=SHIPPED_THRESHOLD,
        )
        # n_off = 1 is far below the floor, so no threshold may be proposed.
        assert study["recommendation"] is None
        assert "insufficient samples" in study["reason"]
        assert "fits the sample rather than the problem" in study["reason"]
        assert study["sample_floor_per_side"] == MIN_SAMPLES_PER_SIDE_FOR_A_RECOMMENDATION

    def test_the_study_is_always_marked_exploratory(self):
        study = gate_study(
            scores=[IDF_ON_DOMAIN_MIN, IDF_OFF_DOMAIN],
            labels=["on", "off"],
            threshold=SHIPPED_THRESHOLD,
        )
        assert study["exploratory"] is True


class TestMedianIsATrueMedian:
    def test_even_sample_takes_the_mean_of_the_middle_two(self):
        dist = on_off_distribution([0.1, 0.2, 0.3, 0.4], ["on"] * 4)
        assert dist["on_median"] == pytest.approx(0.25), (
            "the upper-middle element (0.3) is NOT the median"
        )

    def test_odd_sample_takes_the_middle_element(self):
        dist = on_off_distribution([0.1, 0.2, 0.9], ["on"] * 3)
        assert dist["on_median"] == pytest.approx(0.2)

    def test_single_sample_median_is_that_sample(self):
        dist = on_off_distribution([0.26], ["off"])
        assert dist["off_median"] == pytest.approx(0.26)


class TestDistributionReporting:
    def test_distribution_reports_extremes_and_gap(self):
        dist = on_off_distribution(
            [0.30, 0.50, 0.20, 0.90], ["on", "on", "off", "off"]
        )
        assert dist["n_on"] == 2 and dist["n_off"] == 2
        assert dist["on_min"] == pytest.approx(0.30)
        assert dist["on_max"] == pytest.approx(0.50)
        assert dist["off_min"] == pytest.approx(0.20)
        assert dist["off_max"] == pytest.approx(0.90)
        # separation_gap = on_median - off_median = 0.40 - 0.55 = -0.15: the
        # off-domain group scores HIGHER, which is the inverted case.
        assert dist["separation_gap"] == pytest.approx(-0.15)
        assert dist["inverted"] is True
        assert dist["separable"] is False

    def test_overlapping_distributions_are_not_separable(self):
        dist = on_off_distribution(
            [0.30, 0.40, 0.32, 0.38], ["on", "on", "off", "off"]
        )
        assert dist["separation_gap"] == pytest.approx(0.0)
        assert dist["inverted"] is False
        assert dist["separable"] is False

    def test_empty_labels_produce_none_not_zero(self):
        dist = on_off_distribution([], [])
        assert dist["on_median"] is None
        assert dist["off_median"] is None
        assert dist["separation_gap"] is None
        assert dist["inverted"] is False
        assert dist["separable"] is False


class TestSampleFloor:
    def test_enough_samples_each_side_permits_a_recommendation(self):
        # On-domain HIGH, off-domain LOW: the real polarity of
        # `question_answer_relevance`.
        scores = [0.48, 0.50, 0.52, 0.54, 0.56, 0.10, 0.12, 0.14, 0.16, 0.18]
        labels = ["on"] * 5 + ["off"] * 5
        study = gate_study(scores=scores, labels=labels, threshold=SHIPPED_THRESHOLD)
        assert study["recommendation"] is not None
        assert study["recommendation"]["action"] == "threshold"
        # on median 0.52, off median 0.14 -> midpoint 0.33
        assert study["recommendation"]["new_threshold"] == pytest.approx(0.33)
        assert study["recommendation"]["exploratory"] is True
        assert "STARTING" in study["recommendation"]["rationale"]
        assert study["higher_means"] == "on_domain"

    def test_one_side_below_the_floor_yields_no_recommendation(self):
        scores = [0.40, 0.42, 0.44, 0.46, 0.48, 0.10]
        labels = ["on"] * 5 + ["off"]
        study = gate_study(scores=scores, labels=labels, threshold=SHIPPED_THRESHOLD)
        assert study["recommendation"] is None
        assert "insufficient samples" in study["reason"]

    def test_floor_is_five_per_side(self):
        assert MIN_SAMPLES_PER_SIDE_FOR_A_RECOMMENDATION == 5

    def test_gap_floor_is_declared(self):
        assert MIN_MEDIAN_GAP == 0.05

    def test_enough_samples_but_overlapping_advises_no_change(self):
        # on median 0.35, off median 0.33 -> gap 0.02, positive but below the
        # 0.05 floor: correctly ordered yet too close to separate.
        scores = [0.33, 0.34, 0.35, 0.36, 0.37, 0.31, 0.32, 0.33, 0.34, 0.35]
        labels = ["on"] * 5 + ["off"] * 5
        study = gate_study(scores=scores, labels=labels, threshold=SHIPPED_THRESHOLD)
        assert study["distribution"]["inverted"] is False
        assert study["recommendation"]["action"] == "no_change"
        assert "overlap" in study["recommendation"]["rationale"]

    def test_inverted_distribution_refuses_to_propose_a_threshold(self):
        """The plain-coverage defect: no threshold can fix a wrong ranking."""
        scores = [0.10, 0.12, 0.14, 0.16, 0.18, 0.48, 0.50, 0.52, 0.54, 0.56]
        labels = ["on"] * 5 + ["off"] * 5
        study = gate_study(scores=scores, labels=labels, threshold=SHIPPED_THRESHOLD)
        assert study["distribution"]["inverted"] is True
        assert study["recommendation"]["action"] == "no_threshold_possible"
        assert "INVERTED" in study["recommendation"]["rationale"]
        assert "the method itself must change" in study["recommendation"]["rationale"]

    def test_unknown_polarity_is_refused(self):
        """Assuming the polarity wrong inverts the advice, so it must be declared."""
        with pytest.raises(GateStudyError, match="polarity cannot be assumed"):
            gate_study(
                scores=[0.5, 0.1],
                labels=["on", "off"],
                threshold=SHIPPED_THRESHOLD,
                higher_means="whatever",
            )

    def test_off_domain_polarity_is_supported_explicitly(self):
        scores = [0.10, 0.12, 0.14, 0.16, 0.18, 0.48, 0.50, 0.52, 0.54, 0.56]
        labels = ["on"] * 5 + ["off"] * 5
        study = gate_study(
            scores=scores,
            labels=labels,
            threshold=SHIPPED_THRESHOLD,
            higher_means=HIGHER_MEANS_OFF_DOMAIN,
        )
        assert study["distribution"]["inverted"] is False
        assert study["recommendation"]["action"] == "threshold"


class TestCounts:
    def test_false_positives_and_negatives_are_counted_at_the_threshold(self):
        scores = [0.50, 0.20, 0.40, 0.10]
        labels = ["on", "on", "off", "off"]
        study = gate_study(scores=scores, labels=labels, threshold=SHIPPED_THRESHOLD)
        assert study["fp_count"] == 1  # the off-domain 0.40
        assert study["fn_count"] == 1  # the on-domain 0.20
        assert study["threshold"] == SHIPPED_THRESHOLD

    def test_length_mismatch_is_refused(self):
        with pytest.raises(GateStudyError, match="same length"):
            gate_study(scores=[0.1, 0.2], labels=["on"], threshold=SHIPPED_THRESHOLD)


class TestNoConfigIsModified:
    def test_study_returns_a_proposal_and_never_applies_it(self):
        """The study has no application path: the change needs a reviewer."""
        import inspect

        import app.services.answer_eval.gate as gate_module

        source = inspect.getsource(gate_module)
        for forbidden in ("setattr", "save_retrieval_config", "update_config", "open("):
            assert forbidden not in source, (
                f"the gate study must not modify configuration, but references {forbidden!r}"
            )
