"""V10 STEP 9 — UNKNOWN → MEASURED correctness from human-reviewed labels.

Two things are pinned here:

1. The DEFAULT deterministic evaluator is unchanged: correctness stays UNKNOWN
   even when a reference answer exists (V8 semantics, V8 tests untouched).
2. `ReferenceAnswerEvaluator` — requested EXPLICITLY — measures correctness from
   a human-reviewed benchmark under the published
   `reference-key-point-coverage-v1` policy, and refuses (UNKNOWN) when the
   question has no reviewed labels.

Hermetic: no network, no Qdrant, no LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.answer import AnswerStatus, GateDecision, GroundingState  # noqa: E402
from app.services.answer_eval.benchmark import (  # noqa: E402
    Answerability,
    AnswerBenchmarkQuestion,
    ExpectedGroundingState,
    RequiredEvidence,
)
from app.services.answer_eval.evaluator import (  # noqa: E402
    DeterministicAnswerEvaluator,
    ReferenceAnswerEvaluator,
    create_answer_evaluator,
)
from app.services.answer_eval.run import aggregate_results  # noqa: E402

from tests.test_answer_eval_v8 import (  # noqa: E402
    GOOD_EVIDENCE_TEXT,
    _answer,
    _claim,
    _evidence,
)

GOOD = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
DEFAULT_EVAL = DeterministicAnswerEvaluator()
REFERENCE_EVAL = ReferenceAnswerEvaluator()

KEY_POINTS = [
    "The metacentric height GM must be positive for stable equilibrium",
    "Free surface effect reduces stability in a flooded compartment",
]
REFERENCE_ANSWER = (
    "The metacentric height GM must be positive for stable equilibrium of a "
    "floating body."
)
FULL_ANSWER = (
    "The metacentric height GM must be positive for stable equilibrium of a "
    "floating body. Free surface effect reduces stability in a flooded "
    "compartment."
)


def _question(**overrides) -> AnswerBenchmarkQuestion:
    base = dict(
        question_id="q-001",
        question="What must the metacentric height GM be for stable equilibrium?",
        answerability=Answerability.ANSWERABLE,
        required_evidence=[
            RequiredEvidence(
                chunk_id="chk_good",
                document_id="doc_chk_good",
                content_hash="hash_chk_good",
            )
        ],
        expected_grounding_state=ExpectedGroundingState.ANY_ACCEPTABLE,
    )
    base.update(overrides)
    return AnswerBenchmarkQuestion(**base)


def _cited_answer(text: str):
    return _answer(
        text=text,
        claims=[
            _claim(
                "The metacentric height GM must be positive for stable equilibrium.",
                ["ev_0001"],
            )
        ],
        citations_for=[GOOD],
    )


def _abstained_answer():
    return _answer(
        text="I don't have enough evidence.",
        claims=[],
        citations=[],
        status=AnswerStatus.ABSTAINED,
        state=GroundingState.NO_RELEVANT_EVIDENCE,
        decision=GateDecision.ABSTAIN,
        sufficient=False,
    )


class TestDefaultEvaluatorUnchanged:
    """V8's refusal is preserved: a proxy is not promoted to a verdict."""

    def test_correctness_is_unknown_even_with_reviewed_labels(self):
        r = DEFAULT_EVAL.evaluate(
            _question(key_points=KEY_POINTS, expected_answer=REFERENCE_ANSWER),
            _cited_answer(FULL_ANSWER),
            [GOOD],
        )
        assert r.correctness.measured is False
        assert r.correctness.value is None
        assert r.evaluator_name == "deterministic-evidence"

    def test_default_evaluator_still_measures_the_components(self):
        r = DEFAULT_EVAL.evaluate(
            _question(key_points=KEY_POINTS, expected_answer=REFERENCE_ANSWER),
            _cited_answer(FULL_ANSWER),
            [GOOD],
        )
        assert r.key_point_recall.measured is True
        assert r.reference_answer_similarity.measured is True


class TestReferenceEvaluatorUnlock:
    def test_no_labels_means_unknown_not_zero(self):
        r = REFERENCE_EVAL.evaluate(_question(), _cited_answer(FULL_ANSWER), [GOOD])
        assert r.correctness.measured is False
        assert r.correctness.value is None
        assert "V10 review workflow" in r.correctness.reason

    def test_reviewed_key_points_make_correctness_measurable(self):
        r = REFERENCE_EVAL.evaluate(
            _question(key_points=KEY_POINTS), _cited_answer(FULL_ANSWER), [GOOD]
        )
        assert r.correctness.measured is True
        assert r.correctness.value == 1.0
        assert r.correctness.sample_size == 2
        assert ReferenceAnswerEvaluator.CORRECTNESS_POLICY in r.correctness.reason
        assert "NOT semantic truth" in r.correctness.reason

    def test_partial_coverage_is_partial_credit_not_binary(self):
        text = (
            "The metacentric height GM must be positive. Free surface effect is "
            "important."
        )
        r = REFERENCE_EVAL.evaluate(
            _question(key_points=KEY_POINTS), _cited_answer(text), [GOOD]
        )
        assert r.correctness.measured is True
        assert r.correctness.value == 0.5

    def test_reference_answer_without_key_points_uses_reference_coverage(self):
        r = REFERENCE_EVAL.evaluate(
            _question(expected_answer=REFERENCE_ANSWER),
            _cited_answer(FULL_ANSWER),
            [GOOD],
        )
        assert r.correctness.measured is True
        assert r.correctness.value is not None and r.correctness.value > 0.9
        assert "no key points" in r.correctness.reason

    def test_no_answer_at_all_scores_zero_under_the_policy(self):
        r = REFERENCE_EVAL.evaluate(
            _question(key_points=KEY_POINTS),
            _abstained_answer(),
            [GOOD],
        )
        assert r.correctness.measured is True
        assert r.correctness.value == 0.0
        assert "produced no answer" in r.correctness.reason

    def test_abstention_expected_question_is_not_scored_for_correctness(self):
        r = REFERENCE_EVAL.evaluate(
            _question(key_points=KEY_POINTS, abstention_required=True),
            _abstained_answer(),
            [GOOD],
        )
        assert r.correctness.measured is False
        assert "abstention_accuracy" in r.correctness.reason

    def test_evaluator_identity_is_recorded_on_every_result(self):
        r = REFERENCE_EVAL.evaluate(
            _question(key_points=KEY_POINTS), _cited_answer(FULL_ANSWER), [GOOD]
        )
        assert r.evaluator_name == "reference-labels"
        assert r.evaluator_version == "v10.1"
        assert r.evaluator_is_model_based is False

    def test_factory_returns_the_reference_evaluator_only_when_asked(self):
        assert isinstance(create_answer_evaluator("reference"), ReferenceAnswerEvaluator)
        assert isinstance(
            create_answer_evaluator("deterministic"), DeterministicAnswerEvaluator
        )
        with pytest.raises(ValueError, match="reference"):
            create_answer_evaluator("nonsense")


class TestAggregation:
    def test_aggregate_correctness_moves_from_unknown_to_measured(self):
        unknown = aggregate_results(
            [REFERENCE_EVAL.evaluate(_question(), _cited_answer(FULL_ANSWER), [GOOD])]
        )
        assert unknown.correctness.measured is False
        assert "correctness" in unknown.unknown_metrics

        measured = aggregate_results(
            [
                REFERENCE_EVAL.evaluate(
                    _question(key_points=KEY_POINTS),
                    _cited_answer(FULL_ANSWER),
                    [GOOD],
                )
            ]
        )
        assert measured.correctness.measured is True
        assert measured.correctness.value == 1.0
        assert "correctness" not in measured.unknown_metrics
        assert measured.key_point_recall.measured is True

    def test_mixed_benchmark_keeps_unreviewed_questions_unknown(self):
        reviewed = REFERENCE_EVAL.evaluate(
            _question(question_id="q-reviewed", key_points=KEY_POINTS),
            _cited_answer(FULL_ANSWER),
            [GOOD],
        )
        unreviewed = REFERENCE_EVAL.evaluate(
            _question(question_id="q-unreviewed"),
            _cited_answer(FULL_ANSWER),
            [GOOD],
        )
        agg = aggregate_results([reviewed, unreviewed])
        # The mean is over MEASURED questions only; the unreviewed question is
        # excluded rather than being counted as a zero.
        assert agg.correctness.value == 1.0
        assert agg.correctness.sample_size == 1
