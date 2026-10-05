"""V8 STEP 2/F regression tests — completeness from HUMAN labels only.

Completeness (key_point_recall / expected_information_coverage) is measured
ONLY when a human authored key points, and only as lexical coverage with a
published threshold. Reference-answer lexical similarity is reported under its
own name and correctness stays UNKNOWN even when a reference exists — a proxy
must not be promoted to a verdict.

Hermetic: no network, no Qdrant, no LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.answer import AnswerStatus, GateDecision, GroundingState  # noqa: E402
from app.services.answer_eval.benchmark import (  # noqa: E402
    Answerability, AnswerBenchmarkQuestion, ExpectedGroundingState,
    RequiredEvidence,
)
from app.services.answer_eval.evaluator import (  # noqa: E402
    KEY_POINT_COVERAGE_THRESHOLD, DeterministicAnswerEvaluator,
)
from app.services.answer_eval.run import aggregate_results  # noqa: E402

from tests.test_answer_eval_v8 import GOOD_EVIDENCE_TEXT, _answer, _claim, _evidence  # noqa: E402

EVAL = DeterministicAnswerEvaluator()
GOOD = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")

KEY_POINTS = [
    "The metacentric height GM must be positive for stable equilibrium",
    "Free surface effect reduces stability in a flooded compartment",
]


def _question(**overrides) -> AnswerBenchmarkQuestion:
    base = dict(
        question_id="q-001",
        question="What must the metacentric height GM be for stable equilibrium?",
        answerability=Answerability.ANSWERABLE,
        required_evidence=[
            RequiredEvidence(chunk_id="chk_good", document_id="doc_chk_good",
                             content_hash="hash_chk_good")
        ],
        expected_grounding_state=ExpectedGroundingState.ANY_ACCEPTABLE,
    )
    base.update(overrides)
    return AnswerBenchmarkQuestion(**base)


def _cited_answer(text: str) -> Answer:
    return _answer(
        text=text,
        claims=[_claim("The metacentric height GM must be positive for stable "
                       "equilibrium.", ["ev_0001"])],
        citations_for=[GOOD],
    )


FULL_ANSWER = (
    "The metacentric height GM must be positive for stable equilibrium of a "
    "floating body. Free surface effect reduces stability in a flooded "
    "compartment."
)


class TestKeyPointCoverage:
    def test_recall_is_measured_when_a_human_authored_key_points(self):
        r = EVAL.evaluate(_question(key_points=KEY_POINTS),
                          _cited_answer(FULL_ANSWER), [GOOD])
        assert r.key_point_recall.measured is True
        assert r.key_point_recall.value == 1.0
        assert r.key_point_recall.sample_size == 2
        assert "LEXICAL PROXY" in r.key_point_recall.reason

    def test_expected_information_coverage_is_the_same_measurement(self):
        r = EVAL.evaluate(_question(key_points=KEY_POINTS),
                          _cited_answer(FULL_ANSWER), [GOOD])
        assert r.expected_information_coverage.measured is True
        assert r.expected_information_coverage.value == r.key_point_recall.value

    def test_partial_coverage_counts_points_at_the_published_threshold(self):
        # covers point 1 (4/5 terms >= 0.6) but not point 2 (2/5 terms < 0.6)
        text = ("The metacentric height GM must be positive. Free surface "
                "effect is important.")
        r = EVAL.evaluate(_question(key_points=KEY_POINTS),
                          _cited_answer(text), [GOOD])
        assert r.key_point_recall.value == 0.5
        assert KEY_POINT_COVERAGE_THRESHOLD == 0.6

    def test_abstention_scores_zero_coverage_not_unknown(self):
        answer = _answer(
            text="I don't have enough evidence.",
            claims=[], citations=[],
            status=AnswerStatus.ABSTAINED,
            state=GroundingState.NO_RELEVANT_EVIDENCE,
            decision=GateDecision.ABSTAIN,
            sufficient=False,
        )
        r = EVAL.evaluate(_question(key_points=KEY_POINTS), answer, [GOOD])
        assert r.key_point_recall.measured is True
        assert r.key_point_recall.value == 0.0
        assert "no content" in r.key_point_recall.reason or "none of" in r.key_point_recall.reason

    def test_without_human_labels_both_stay_unknown(self):
        r = EVAL.evaluate(_question(), _cited_answer(FULL_ANSWER), [GOOD])
        assert r.key_point_recall.measured is False
        assert r.expected_information_coverage.measured is False
        assert r.key_point_recall.value is None


class TestReferenceAnswerSimilarity:
    REFERENCE = ("The metacentric height GM must be positive for stable "
                 "equilibrium of a floating body.")

    def test_similarity_is_measured_when_a_reference_exists(self):
        r = EVAL.evaluate(_question(expected_answer=self.REFERENCE),
                          _cited_answer(FULL_ANSWER), [GOOD])
        assert r.reference_answer_similarity.measured is True
        assert r.reference_answer_similarity.value is not None
        assert r.reference_answer_similarity.value > 0.9
        assert "NOT a correctness judgement" in r.reference_answer_similarity.reason

    def test_correctness_stays_unknown_even_with_a_reference(self):
        r = EVAL.evaluate(_question(expected_answer=self.REFERENCE),
                          _cited_answer(FULL_ANSWER), [GOOD])
        assert r.correctness.measured is False
        assert r.correctness.value is None
        assert "human review or an explicitly model-based judge" in r.correctness.reason

    def test_abstention_against_a_reference_scores_zero_similarity(self):
        answer = _answer(
            text="No evidence.",
            claims=[], citations=[],
            status=AnswerStatus.ABSTAINED,
            state=GroundingState.NO_RELEVANT_EVIDENCE,
            decision=GateDecision.ABSTAIN,
            sufficient=False,
        )
        r = EVAL.evaluate(_question(expected_answer=self.REFERENCE), answer, [GOOD])
        assert r.reference_answer_similarity.measured is True
        assert r.reference_answer_similarity.value == 0.0

    def test_no_reference_means_unknown_similarity(self):
        r = EVAL.evaluate(_question(), _cited_answer(FULL_ANSWER), [GOOD])
        assert r.reference_answer_similarity.measured is False
        assert "no human-authored reference answer" in r.reference_answer_similarity.reason


class TestAggregateCompleteness:
    def test_aggregates_and_warning_depend_on_label_presence(self):
        with_labels = EVAL.evaluate(
            _question(key_points=KEY_POINTS, expected_answer="reference"),
            _cited_answer(FULL_ANSWER), [GOOD])
        agg = aggregate_results([with_labels])
        assert agg.key_point_recall.value == 1.0
        assert agg.expected_information_coverage.value == 1.0
        assert agg.reference_answer_similarity.value is not None
        assert any("human reference answers exist" in w for w in agg.warnings)
        assert "key_point_recall" not in agg.unknown_metrics
        assert "correctness" in agg.unknown_metrics  # still unknown by design

    def test_no_labels_keeps_the_old_warning_and_unknowns(self):
        r = EVAL.evaluate(_question(), _cited_answer(FULL_ANSWER), [GOOD])
        agg = aggregate_results([r])
        assert any("no\nhuman-authored reference answers" in w or
                   "no human-authored reference answers" in w
                   for w in agg.warnings)
        for name in ("key_point_recall", "expected_information_coverage",
                     "reference_answer_similarity"):
            assert name in agg.unknown_metrics, name
