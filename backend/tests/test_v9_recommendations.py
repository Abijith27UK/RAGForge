"""V9 Phase 5 regression tests: the deterministic recommendation engine.

The property under test throughout: a recommendation fires only on MEASURED
evidence, and never recommends tuning retrieval to fix something tuning cannot
fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.diagnostics import build_question_diagnostic  # noqa: E402
from app.services.answer_eval.recommendations import (  # noqa: E402
    MIN_QUESTIONS_FOR_A_PATTERN,
    DEEP_RANK_THRESHOLD,
    Recommendation,
    RecommendationAction,
    RecommendationStrength,
    analyse_failures,
    analyse_retrieval,
    recommendation_from_comparison,
)

CHUNK_INDEX = {
    "chk_req": {"document_id": "doc_auto", "source_title": "Handbook", "content_hash": "h1"},
    "chk_other": {"document_id": "doc_tyres", "source_title": "Tyre Catalogue", "content_hash": "h2"},
}


def _diag(question_id: str, ranked: list[str], required: list[str], *, top_k: int = 5, selected=None):
    return build_question_diagnostic(
        question_id=question_id,
        question=f"question {question_id}",
        ranked_chunk_ids=ranked,
        chunk_index=CHUNK_INDEX,
        required_chunk_ids=required,
        evidence_selected_ids=selected if selected is not None else ranked,
        top_k=top_k,
        latency_ms=12.0,
    )


class TestDeepRankRecommendsTopK:
    def test_deep_evidence_triggers_increase_top_k(self):
        diagnostics = [
            _diag(f"q{i}", ["chk_other"] * 7 + ["chk_req"], ["chk_req"], top_k=5)
            for i in range(4)
        ]
        report = analyse_retrieval(diagnostics, current_top_k=5, current_strategy="dense")
        increase = report.for_action(RecommendationAction.INCREASE_TOP_K)
        assert len(increase) == 1
        rec = increase[0]
        assert rec.proposed_config_change["top_k"] == 8
        assert rec.supporting_measurements["questions_with_deep_evidence"] == 4
        assert rec.strength is not RecommendationStrength.WEAK
        assert rec.thresholds_used["DEEP_RANK_THRESHOLD"] == DEEP_RANK_THRESHOLD

    def test_rank_just_below_threshold_does_not_trigger_top_k(self):
        diagnostics = [
            _diag(f"q{i}", ["chk_other"] * 4 + ["chk_req"], ["chk_req"]) for i in range(5)
        ]
        report = analyse_retrieval(diagnostics, current_top_k=5, current_strategy="dense")
        assert report.for_action(RecommendationAction.INCREASE_TOP_K) == []

    def test_few_deep_questions_are_weak_not_a_finding(self):
        diagnostics = [
            _diag("q1", ["chk_other"] * 7 + ["chk_req"], ["chk_req"]),
            _diag("q2", ["chk_req"], ["chk_req"]),
        ]
        report = analyse_retrieval(diagnostics, current_top_k=5)
        recs = report.for_action(RecommendationAction.INCREASE_TOP_K)
        assert len(recs) == 1
        assert recs[0].strength is RecommendationStrength.WEAK


class TestAbsentEvidenceIsNotATuningProblem:
    def test_absent_evidence_recommends_corpus_deficiency(self):
        diagnostics = [
            _diag(f"q{i}", ["chk_other", "chk_other"], ["chk_req"]) for i in range(4)
        ]
        report = analyse_retrieval(diagnostics, current_top_k=5, current_strategy="dense")
        corpus = report.for_action(RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY)
        assert len(corpus) == 1
        assert corpus[0].proposed_config_change == {}, (
            "a corpus gap must not propose a retrieval configuration change"
        )
        assert "no retrieval knob" in corpus[0].reason.lower()

    def test_absent_evidence_and_deep_evidence_are_distinguished(self):
        """Deep-but-present must NOT be reported as absent."""
        deep = [_diag(f"q{i}", ["chk_other"] * 7 + ["chk_req"], ["chk_req"]) for i in range(4)]
        report = analyse_retrieval(deep, current_top_k=5)
        assert report.for_action(RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY) == []
        assert len(report.for_action(RecommendationAction.INCREASE_TOP_K)) == 1


class TestEvidenceSelectionIsSeparateFromRetrieval:
    def test_retrieved_but_unselected_recommends_selection_investigation(self):
        diagnostics = [
            _diag(f"q{i}", ["chk_req", "chk_other"], ["chk_req"], selected=["chk_other"])
            for i in range(4)
        ]
        report = analyse_retrieval(diagnostics, current_top_k=5)
        selection = report.for_action(RecommendationAction.INVESTIGATE_EVIDENCE_SELECTION)
        assert len(selection) == 1
        assert selection[0].proposed_config_change == {}
        assert "selection" in selection[0].reason.lower()


class TestNoGroundTruth:
    def test_no_reference_evidence_produces_a_data_gap_not_a_guess(self):
        diagnostics = [_diag(f"q{i}", ["chk_other"], []) for i in range(5)]
        report = analyse_retrieval(diagnostics)
        assert report.questions_with_ground_truth == 0
        assert any("needs reference evidence" in g for g in report.data_gaps)
        assert report.for_action(RecommendationAction.INCREASE_TOP_K) == []
        assert report.for_action(RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY) == []

    def test_no_diagnostics_at_all_is_reported(self):
        report = analyse_retrieval([])
        assert report.recommendations == []
        assert report.data_gaps == [
            "no question diagnostics were supplied, so no recommendation can be "
            "made from measurement"
        ]


class TestEveryRecommendationIsAnExperimentProposal:
    def test_experiment_required_is_always_true(self):
        diagnostics = [
            _diag(f"q{i}", ["chk_other"] * 7 + ["chk_req"], ["chk_req"]) for i in range(4)
        ]
        report = analyse_retrieval(diagnostics, current_top_k=5)
        assert report.recommendations
        for rec in report.recommendations:
            assert rec.experiment_required is True

    def test_recommendation_model_defaults_to_experiment_required(self):
        """A caller cannot construct a recommendation that skips the experiment."""
        rec = Recommendation(action=RecommendationAction.INCREASE_TOP_K)
        assert rec.experiment_required is True

    def test_no_action_justified_when_evidence_is_front_ranked(self):
        diagnostics = [_diag(f"q{i}", ["chk_req"], ["chk_req"]) for i in range(10)]
        report = analyse_retrieval(diagnostics, current_top_k=5)
        assert report.actions() == [RecommendationAction.NO_ACTION_JUSTIFIED.value]
        assert report.recommendations[0].strength is RecommendationStrength.STRONG


class TestPatternFloor:
    def test_below_the_pattern_floor_is_reported_as_a_data_gap(self):
        diagnostics = [_diag("q1", ["chk_req"], ["chk_req"])]
        report = analyse_retrieval(diagnostics)
        assert any(str(MIN_QUESTIONS_FOR_A_PATTERN) in g for g in report.data_gaps)

    def test_pattern_floor_is_three(self):
        assert MIN_QUESTIONS_FOR_A_PATTERN == 3


class TestFailureAnalysis:
    class _Result:
        """Minimal stand-in for a V8 AnswerQualityResult."""

        def __init__(self, question_id: str, **fields):
            self.question_id = question_id
            self.question = f"question {question_id}"
            self.answer_text = fields.pop("answer_text", "Some answer text here.")
            self.answerability = fields.pop("answerability", "answerable")
            self.retrieved_chunk_ids = fields.pop("retrieved_chunk_ids", ["chk_other"])
            self.retrieved_rank_of_required = fields.pop("retrieved_rank_of_required", None)
            self.required_chunk_ids = fields.pop("required_chunk_ids", ["chk_req"])
            self.relevance_passed = fields.pop("relevance_passed", None)
            self.abstention_performed = fields.pop("abstention_performed", False)
            self.actual_grounding_state = fields.pop("actual_grounding_state", "")
            self.answer_status = fields.pop("answer_status", "")
            self.claim_verdicts = fields.pop("claim_verdicts", [])
            self.required_evidence = fields.pop("required_evidence", [])
            for key, value in fields.items():
                setattr(self, key, value)

    def _metric(self, value):
        class _M:
            measured = value is not None
            measured_value = value

            def __init__(self, v):
                self.value = v

        return _M(value)

    def test_retrieval_failures_are_prioritised_over_generation(self):
        results = [
            self._Result(
                "q1",
                retrieval_hit_rate=self._metric(0.0),
                question_answer_relevance=self._metric(None),
                fabricated_citation_rate=self._metric(None),
                unsupported_citation_rate=self._metric(None),
            )
            for _ in range(4)
        ]
        report = analyse_failures(results)
        assert report.recommendations
        assert report.recommendations[0].action in {
            RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY,
            RecommendationAction.INCREASE_TOP_K,
            RecommendationAction.ENABLE_HYBRID_FUSION,
        }

    def test_unclassifiable_questions_are_counted_as_unknown_not_guessed(self):
        results = [
            self._Result(
                "q1",
                retrieval_hit_rate=self._metric(None),
                question_answer_relevance=self._metric(0.5),
                relevance_passed=True,
                fabricated_citation_rate=self._metric(None),
                unsupported_citation_rate=self._metric(None),
            )
        ]
        report = analyse_failures(results)
        assert report.unknown_failure_count == 1
        assert any("could not be classified" in g for g in report.data_gaps)

    def test_empty_results_are_reported(self):
        report = analyse_failures([])
        assert report.data_gaps == ["no evaluation results were supplied"]


class TestRecommendationFromComparison:
    def test_improvement_is_recommended(self):
        rec = recommendation_from_comparison(
            metric="recall_at_k",
            baseline_mean=0.30,
            candidate_mean=0.55,
            mean_delta=0.25,
            improved_count=10,
            regressed_count=1,
            paired_count=28,
            candidate_label="hybrid",
            candidate_config={"strategy": "hybrid"},
        )
        assert rec is not None
        assert rec.action is RecommendationAction.ADJUST_HYBRID_WEIGHTS
        assert rec.experiment_required is True

    def test_unknown_delta_is_not_recommended(self):
        assert (
            recommendation_from_comparison(
                metric="recall_at_k",
                baseline_mean=None,
                candidate_mean=None,
                mean_delta=None,
                improved_count=0,
                regressed_count=0,
                paired_count=0,
                candidate_label="hybrid",
                candidate_config={},
            )
            is None
        )

    def test_tiny_delta_is_not_recommended(self):
        assert (
            recommendation_from_comparison(
                metric="recall_at_k",
                baseline_mean=0.30,
                candidate_mean=0.305,
                mean_delta=0.005,
                improved_count=5,
                regressed_count=1,
                paired_count=28,
                candidate_label="hybrid",
                candidate_config={},
            )
            is None
        )

    def test_regression_is_not_recommended(self):
        assert (
            recommendation_from_comparison(
                metric="recall_at_k",
                baseline_mean=0.50,
                candidate_mean=0.30,
                mean_delta=-0.20,
                improved_count=1,
                regressed_count=10,
                paired_count=28,
                candidate_label="hybrid",
                candidate_config={},
            )
            is None
        )

    def test_lower_is_better_metric_drop_is_recommended(self):
        rec = recommendation_from_comparison(
            metric="unsupported_claim_rate",
            baseline_mean=0.30,
            candidate_mean=0.10,
            mean_delta=-0.20,
            improved_count=8,
            regressed_count=1,
            paired_count=28,
            candidate_label="hybrid",
            candidate_config={},
        )
        assert rec is not None

    def test_more_regressions_than_improvements_is_not_recommended(self):
        assert (
            recommendation_from_comparison(
                metric="recall_at_k",
                baseline_mean=0.30,
                candidate_mean=0.60,
                mean_delta=0.30,
                improved_count=2,
                regressed_count=5,
                paired_count=28,
                candidate_label="hybrid",
                candidate_config={},
            )
            is None
        ), "a positive mean with more regressions than improvements must not be recommended"
