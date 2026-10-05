"""V8 STEP 5/14 regression tests — evaluator abstractions.

Covered spec cases:
  * evaluator built without a provider (human AND llm) -> refused loudly
  * malformed LLM evaluator output -> correctness stays UNKNOWN, never guessed
  * evaluator unavailable at judge time -> graceful degradation, warning kept
  * LLM provider metadata (model + prompt version) persisted on the result
  * LLM judgement labelled MODEL-BASED with raw output stored for audit
  * human reviews -> correctness measured with reviewer attribution; answers
    with no matching review keep UNKNOWN

Hermetic: fake in-memory providers, no network, no Qdrant, no real LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.evaluator import (  # noqa: E402
    DeterministicAnswerEvaluator,
    HumanAnswerEvaluator,
    LLMAnswerEvaluator,
    create_answer_evaluator,
)
from app.services.answer_eval.review import (  # noqa: E402
    AnswerReview, ReviewLabel, ReviewVerdict,
)

from tests.test_answer_eval_v8 import GOOD_EVIDENCE_TEXT, _answer, _claim, _evidence, _question  # noqa: E402

GOOD = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")


def _good_answer():
    return _answer(
        claims=[_claim("The metacentric height GM must be positive for stable "
                       "equilibrium.", ["ev_0001"])],
        citations_for=[GOOD],
    )


def _review(reviewer: str, verdict: ReviewVerdict, answer_id: str = "ans_1",
            question_id: str = "q-001", **over) -> AnswerReview:
    return AnswerReview(
        id=over.get("id", "areview_1"),
        kb_id="kb_test",
        run_id="aerun_1",
        question_id=question_id,
        answer_id=answer_id,
        reviewer=reviewer,
        verdict=verdict,
        labels=over.get("labels", []),
        notes=over.get("notes", ""),
    )


class FakeLLM:
    model = "fake-deepseek-chat"

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error

    def generate_text(self, prompt: str) -> str:
        if self.error is not None:
            raise self.error
        return self.reply


GOOD_JUDGE = json.dumps({
    "correctness": 0.85,
    "confidence": 0.7,
    "reason": "matches the reference on the key point",
})


class TestHumanAnswerEvaluator:
    def test_requires_a_reviews_provider(self):
        with pytest.raises(ValueError, match="reviews_provider"):
            HumanAnswerEvaluator(None)

    def test_reviews_measure_correctness_with_attribution(self):
        reviews = [_review("ada@example.edu", ReviewVerdict.MOSTLY_CORRECT),
                   _review("ada@example.edu", ReviewVerdict.CORRECT,
                           id="areview_2")]
        ev = HumanAnswerEvaluator(lambda qid, aid: reviews)
        r = ev.evaluate(_question(), _good_answer(), [GOOD])
        assert r.correctness.measured is True
        assert r.correctness.value == 0.875  # (0.75 + 1.0) / 2
        assert r.correctness.sample_size == 2
        assert "ada@example.edu" in r.correctness.reason
        assert "policy mapping" in r.correctness.reason
        assert r.evaluator_name == "human-reviews"
        assert r.evaluator_is_model_based is False
        assert "ada@example.edu" in r.evaluator_detail
        # mechanical metrics still come from the deterministic base
        assert r.citation_precision.value == 1.0

    def test_no_reviews_leaves_correctness_unknown(self):
        ev = HumanAnswerEvaluator(lambda qid, aid: [])
        r = ev.evaluate(_question(), _good_answer(), [GOOD])
        assert r.correctness.measured is False
        assert "no human reviews" in r.correctness.reason

    def test_reviews_for_a_different_answer_are_ignored(self):
        reviews = [_review("ada", ReviewVerdict.INCORRECT, answer_id="ans_OTHER")]
        ev = HumanAnswerEvaluator(lambda qid, aid: reviews)  # provider ignores ids
        r = ev.evaluate(_question(), _good_answer(), [GOOD])
        assert r.correctness.measured is False
        assert "no human reviews" in r.correctness.reason


class TestLLMAnswerEvaluator:
    def test_requires_a_provider(self):
        with pytest.raises(ValueError, match="requires a provider"):
            LLMAnswerEvaluator(None)

    def test_valid_judgement_is_scored_and_labelled_model_based(self):
        ev = LLMAnswerEvaluator(FakeLLM(GOOD_JUDGE), model="deepseek-chat")
        r = ev.evaluate(_question(), _good_answer(), [GOOD])
        assert r.correctness.measured is True
        assert r.correctness.value == 0.85
        assert "LLM-as-judge" in r.correctness.reason
        assert "MODEL-BASED JUDGEMENT" in r.correctness.reason
        assert r.evaluator_is_model_based is True
        assert "deepseek-chat" in r.evaluator_detail
        assert "aej-v1" in r.evaluator_detail  # prompt version persisted
        assert r.judge_raw == GOOD_JUDGE        # raw output kept for audit
        assert any("not ground truth" in w for w in r.warnings)

    def test_json_wrapped_in_markdown_fences_is_still_parsed(self):
        reply = "Here you go:\n```json\n" + GOOD_JUDGE + "\n```"
        ev = LLMAnswerEvaluator(FakeLLM(reply), model="deepseek-chat")
        r = ev.evaluate(_question(), _good_answer(), [GOOD])
        assert r.correctness.measured is True
        assert r.correctness.value == 0.85

    def test_malformed_output_leaves_correctness_unknown(self):
        ev = LLMAnswerEvaluator(FakeLLM("the answer seems mostly fine"),
                                model="deepseek-chat")
        r = ev.evaluate(_question(), _good_answer(), [GOOD])
        assert r.correctness.measured is False
        assert r.correctness.value is None
        assert any("unparseable" in w for w in r.warnings)
        assert any("no verdict was guessed" in w for w in r.warnings)
        assert r.judge_raw == "the answer seems mostly fine"

    def test_out_of_range_score_is_rejected_not_clamped(self):
        ev = LLMAnswerEvaluator(
            FakeLLM(json.dumps({"correctness": 3.0, "reason": "great"})))
        r = ev.evaluate(_question(), _good_answer(), [GOOD])
        assert r.correctness.measured is False
        assert any("unparseable" in w for w in r.warnings)

    def test_provider_failure_degrades_gracefully(self):
        ev = LLMAnswerEvaluator(
            FakeLLM(error=RuntimeError("connection reset")),
            model="deepseek-chat")
        r = ev.evaluate(_question(), _good_answer(), [GOOD])  # must not raise
        assert r.correctness.measured is False
        assert any("LLM judge UNAVAILABLE" in w for w in r.warnings)
        assert r.evaluator_is_model_based is True  # identity is not erased

    def test_mechanical_metrics_survive_a_failed_judge(self):
        ev = LLMAnswerEvaluator(FakeLLM(error=RuntimeError("boom")))
        r = ev.evaluate(_question(), _good_answer(), [GOOD])
        assert r.citation_precision.value == 1.0
        assert r.passed is not None


class TestReproducibilityAndLabeling:
    """STEP 17 cases 16 (reproducibility) and 19 (mock generator labelling)."""

    def test_identical_inputs_produce_identical_metrics(self):
        ev = DeterministicAnswerEvaluator()
        a = ev.evaluate(_question(), _good_answer(), [GOOD])
        b = ev.evaluate(_question(), _good_answer(), [GOOD])
        dump_a = a.model_dump(mode="json")
        dump_b = b.model_dump(mode="json")
        # only the wall-clock timestamp may differ
        dump_a.pop("created_at")
        dump_b.pop("created_at")
        assert dump_a == dump_b, "the deterministic evaluator must be reproducible"

    def test_aggregate_is_reproducible(self):
        from app.services.answer_eval.run import aggregate_results

        ev = DeterministicAnswerEvaluator()
        r1 = ev.evaluate(_question(), _good_answer(), [GOOD])
        r2 = ev.evaluate(_question(), _good_answer(), [GOOD])
        agg_a = aggregate_results([r1]).model_dump(mode="json")
        agg_b = aggregate_results([r2]).model_dump(mode="json")
        assert agg_a == agg_b

    def test_mock_generator_flag_persists_on_the_run(self):
        from app.services.answer_eval.run import AnswerEvaluationRun

        run = AnswerEvaluationRun(
            id="aerun_mock", kb_id="kb_test",
            generator="llm", model="mock/mock-1", is_mock=True,
        )
        assert run.is_mock is True
        roundtrip = AnswerEvaluationRun.model_validate(run.model_dump(mode="json"))
        assert roundtrip.is_mock is True, "mock labelling must survive persistence"


class TestEvaluatorFactory:
    def test_unknown_kind_lists_supported_kinds(self):
        with pytest.raises(ValueError, match="deterministic, 'human'.*'llm'|human.*llm"):
            create_answer_evaluator("oracle")

    def test_llm_kind_requires_an_explicit_provider(self):
        with pytest.raises(ValueError, match="requires a provider"):
            create_answer_evaluator("llm")

    def test_human_kind_requires_reviews_provider(self):
        with pytest.raises(ValueError, match="reviews_provider"):
            create_answer_evaluator("human")

    def test_deterministic_is_the_default_and_not_model_based(self):
        ev = create_answer_evaluator()
        assert isinstance(ev, DeterministicAnswerEvaluator)
        assert ev.is_model_based is False

    def test_llm_kind_builds_a_model_based_evaluator(self):
        ev = create_answer_evaluator("llm", llm_provider=FakeLLM(GOOD_JUDGE),
                                     model="deepseek-chat")
        assert isinstance(ev, LLMAnswerEvaluator)
        assert ev.is_model_based is True
        assert "deepseek-chat" in ev.detail
