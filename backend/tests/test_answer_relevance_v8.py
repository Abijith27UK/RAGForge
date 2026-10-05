"""V8 STEP 8 regression tests — question/answer relevance.

The property under test: **groundedness cannot catch a grounded answer to the
wrong question**. An answer with perfect citations but off-topic text must be
detected — and detected honestly:

  * with corpus IDF weights, a relevance shortfall FAILS the answer;
  * with the plain fallback (no corpus statistics), the same shortfall is a
    WARNING, because plain coverage was measured to be non-discriminative;
  * abstentions are never scored for relevance — a refusal is not an
    irrelevant answer;
  * the method, its weight source, and close-call margins are recorded on
    every result so a lexical proxy is never mistaken for a model's judgement.

Hermetic: no network, no Qdrant, no LLM. Weights are injected as plain data.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.answer import AnswerStatus, GroundingState, GateDecision  # noqa: E402
from app.services.answer_eval.benchmark import (  # noqa: E402
    Answerability, AnswerBenchmarkQuestion, ExpectedGroundingState,
    RequiredEvidence,
)
from app.services.answer_eval.evaluator import DeterministicAnswerEvaluator  # noqa: E402
from app.services.answer_eval.relevance import (  # noqa: E402
    METHOD_IDF, METHOD_PLAIN, MIN_RELEVANCE_TO_PASS, TermWeights,
    compute_relevance, term_weights_from_lexical_index,
)
from app.services.answer_eval.run import aggregate_results  # noqa: E402

from tests.test_answer_eval_v8 import (  # noqa: E402
    GOOD_EVIDENCE_TEXT, UNRELATED_TEXT, _answer, _claim, _evidence,
)

# Corpus statistics where the metacentric terms are RARE (high IDF) and the
# cavitation question's terms are also rare — the off-topic answer shares none
# of them with its question.
WEIGHTS = TermWeights(
    n_docs=812,
    df={
        "causes": 400, "propeller": 90, "cavitation": 12,
        "metacentric": 20, "height": 300, "gm": 25,
        "stable": 60, "equilibrium": 18, "must": 700,
        "positive": 40, "righting": 15, "arm": 200,
        "floating": 30, "body": 400,
    },
    source="lexical-index:n_docs=812",
)

EV = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")


def _cavitation_question() -> AnswerBenchmarkQuestion:
    """A question whose subject the stored answer does NOT contain."""
    return AnswerBenchmarkQuestion(
        question_id="q-cav",
        question="What causes propeller cavitation?",
        answerability=Answerability.ANSWERABLE,
        required_evidence=[
            RequiredEvidence(chunk_id="chk_good", document_id="doc_chk_good",
                             content_hash="hash_chk_good")
        ],
        expected_grounding_state=ExpectedGroundingState.ANY_ACCEPTABLE,
    )


def _perfectly_cited_answer(text: str):
    """Citations and claims are correct; only the TEXT can be off-topic."""
    return _answer(
        text=text,
        claims=[_claim("The metacentric height GM must be positive for stable "
                       "equilibrium.", ["ev_0001"])],
        citations_for=[EV],
    )


class TestIrrelevantButGrounded:
    """Case 17 (spec list #11): irrelevant but grounded answer."""

    def test_idf_relevance_fails_an_off_topic_answer(self):
        ev = DeterministicAnswerEvaluator(weights=WEIGHTS)
        r = ev.evaluate(_cavitation_question(),
                        _perfectly_cited_answer(
                            "The metacentric height GM must be positive for "
                            "stable equilibrium of a floating body."),
                        [EV])
        # citations are perfect — relevance is the ONLY failure
        assert r.citation_precision.value == 1.0
        assert r.citation_recall.value == 1.0
        assert len(r.problems) == 1, r.problems
        assert "relevance" in r.problems[0]
        assert r.passed is False
        assert r.question_answer_relevance.measured is True
        assert r.question_answer_relevance.value is not None
        assert r.question_answer_relevance.value < MIN_RELEVANCE_TO_PASS
        assert r.relevance_passed is False
        assert r.relevance_method == METHOD_IDF

    def test_on_topic_answer_passes_the_same_check(self):
        ev = DeterministicAnswerEvaluator(weights=WEIGHTS)
        answer = _answer(
            text="Propeller cavitation occurs when the local pressure falls "
                 "below the vapour pressure of water.",
            claims=[_claim("The metacentric height GM must be positive.",
                           ["ev_0001"])],
            citations_for=[EV],
        )
        r = ev.evaluate(_cavitation_question(), answer, [EV])
        assert r.relevance_passed is True
        assert r.question_answer_relevance.value >= MIN_RELEVANCE_TO_PASS
        assert not any("relevance" in p for p in r.problems)

    def test_relevance_is_never_labelled_model_based(self):
        ev = DeterministicAnswerEvaluator(weights=WEIGHTS)
        r = ev.evaluate(_cavitation_question(),
                        _perfectly_cited_answer("Cavitation happens."), [EV])
        assert r.relevance_is_model_based is False
        assert r.question_answer_relevance.measured is True  # measured, not modelled
        assert "idf" in r.relevance_method


class TestPlainFallbackIsLabelledNotAuthoritative:
    def test_plain_shortfall_is_a_warning_not_a_failure(self):
        """Without corpus stats the same answer must NOT be failed.

        Plain coverage was measured to be non-discriminative (off-domain
        scored HIGHER than on-domain), so it cannot flip a verdict — but the
        shortfall must still be reported, not hidden.
        """
        ev = DeterministicAnswerEvaluator()  # no weights -> plain fallback
        r = ev.evaluate(_cavitation_question(),
                        _perfectly_cited_answer(
                            "The metacentric height GM must be positive for "
                            "stable equilibrium."),
                        [EV])
        assert r.relevance_method == METHOD_PLAIN
        assert r.relevance_passed is False          # measured...
        assert r.problems == [], r.problems         # ...but not a failure
        assert any("reported but NOT failed" in w for w in r.warnings)
        assert r.passed is True
        assert r.relevance_weight_source == "uniform"

    def test_idf_shortfall_carries_its_weight_source(self):
        ev = DeterministicAnswerEvaluator(weights=WEIGHTS)
        r = ev.evaluate(_cavitation_question(),
                        _perfectly_cited_answer("Cavitation."), [EV])
        assert r.relevance_weight_source == WEIGHTS.source

    def test_close_call_verdict_is_flagged(self):
        # Weights designed so the answer lands within the close-call margin of
        # the threshold (measured: value 0.300062 vs threshold 0.30).
        w = TermWeights(n_docs=1000, df={"causes": 500, "propeller": 500,
                                        "cavitation": 552}, source="test")
        rel = compute_relevance("What causes propeller cavitation?",
                                "Cavitation occurs.", w)
        assert rel.measured and rel.passes is True and rel.close_call is True
        ev = DeterministicAnswerEvaluator(weights=w)
        r = ev.evaluate(_cavitation_question(),
                        _perfectly_cited_answer("Cavitation occurs."), [EV])
        assert r.relevance_close_call is True
        assert any("close call" in w_ for w_ in r.warnings)


class TestAbstentionExemptFromRelevance:
    def _unanswerable(self) -> AnswerBenchmarkQuestion:
        return AnswerBenchmarkQuestion(
            question_id="q-unans",
            question="What is the manufacturer-specific torque tolerance?",
            answerability=Answerability.UNANSWERABLE,
            required_evidence=[],
            abstention_required=True,
            expected_grounding_state=ExpectedGroundingState.INSUFFICIENT_EVIDENCE,
        )

    def test_abstention_is_not_measured_for_relevance(self):
        ev = DeterministicAnswerEvaluator(weights=WEIGHTS)
        answer = _answer(
            text="I don't have enough evidence to answer this question.",
            claims=[], citations=[],
            status=AnswerStatus.ABSTAINED,
            state=GroundingState.NO_RELEVANT_EVIDENCE,
            decision=GateDecision.ABSTAIN,
            sufficient=False,
        )
        r = ev.evaluate(self._unanswerable(), answer, [])
        assert r.question_answer_relevance.measured is False
        assert r.question_answer_relevance.value is None
        assert r.relevance_passed is None
        assert r.relevance_method == METHOD_PLAIN  # default label, unused
        assert r.problems == [], r.problems
        assert r.passed, r.problems
        assert "not applicable to a refusal" in r.question_answer_relevance.reason or \
               "refusals/abstentions" in r.question_answer_relevance.reason

    def test_abstentions_do_not_count_in_the_failure_rate(self):
        ev = DeterministicAnswerEvaluator(weights=WEIGHTS)
        abstained = _answer(
            text="I don't have enough evidence.",
            claims=[], citations=[],
            status=AnswerStatus.ABSTAINED,
            state=GroundingState.INSUFFICIENT_EVIDENCE,
            decision=GateDecision.ABSTAIN,
            sufficient=False,
        )
        ok = ev.evaluate(
            AnswerBenchmarkQuestion(
                question_id="q-ok",
                question="What must the metacentric height GM be for stable "
                         "equilibrium?",
                answerability=Answerability.ANSWERABLE,
                required_evidence=[RequiredEvidence(
                    chunk_id="chk_good", document_id="doc_chk_good",
                    content_hash="hash_chk_good")],
                expected_grounding_state=ExpectedGroundingState.ANY_ACCEPTABLE,
            ),
            _perfectly_cited_answer(
                "The metacentric height GM must be positive for stable "
                "equilibrium."),
            [EV],
        )
        bad = ev.evaluate(_cavitation_question(),
                          _perfectly_cited_answer(
                              "The metacentric height GM must be positive."),
                          [EV])
        agg = aggregate_results([ok, bad, ev.evaluate(self._unanswerable(),
                                                      abstained, [])])
        # 1 failure over 2 MEASURED cases; the abstention is excluded
        assert agg.relevance_failure_rate.measured is True
        assert agg.relevance_failure_rate.value == 0.5
        assert agg.relevance_failure_rate.sample_size == 2

    def test_relevance_unknown_when_every_case_is_an_abstention(self):
        ev = DeterministicAnswerEvaluator(weights=WEIGHTS)
        answer = _answer(
            text="No evidence.",
            claims=[], citations=[],
            status=AnswerStatus.ABSTAINED,
            state=GroundingState.NO_RELEVANT_EVIDENCE,
            decision=GateDecision.ABSTAIN,
            sufficient=False,
        )
        r = ev.evaluate(self._unanswerable(), answer, [])
        agg = aggregate_results([r])
        assert agg.relevance_failure_rate.measured is False
        assert "relevance_failure_rate" in agg.unknown_metrics
        assert "question_answer_relevance" in agg.unknown_metrics


class TestRelevanceProvenance:
    def test_aggregate_averages_measured_relevance_only(self):
        ev = DeterministicAnswerEvaluator(weights=WEIGHTS)
        good_q = AnswerBenchmarkQuestion(
            question_id="q1",
            question="What must the metacentric height GM be for stable "
                     "equilibrium?",
            answerability=Answerability.ANSWERABLE,
            required_evidence=[RequiredEvidence(
                chunk_id="chk_good", document_id="doc_chk_good",
                content_hash="hash_chk_good")],
            expected_grounding_state=ExpectedGroundingState.ANY_ACCEPTABLE,
        )
        good = ev.evaluate(good_q, _perfectly_cited_answer(
            "The metacentric height GM must be positive for stable "
            "equilibrium."), [EV])
        bad = ev.evaluate(_cavitation_question(),
                          _perfectly_cited_answer("Unrelated text."), [EV])
        agg = aggregate_results([good, bad])
        assert agg.question_answer_relevance.measured is True
        assert agg.question_answer_relevance.sample_size == 2
        assert agg.excess_information.measured is True  # observation only

    def test_unrelated_text_constant_still_behaves(self):
        """Sanity: UNRELATED_TEXT shares no term with the metacentric question."""
        rel = compute_relevance(
            "What must the metacentric height GM be for stable equilibrium?",
            UNRELATED_TEXT, None,
        )
        assert rel.measured and rel.value == 0.0 and rel.passes is False


class TestTermWeightsLoading:
    def test_usable_payload_becomes_weights(self):
        w = term_weights_from_lexical_index(
            {"n_docs": 10, "df": {"a": 1, "b": 2}, "revision": 7})
        assert w is not None and w.available
        assert w.source.startswith("lexical-index:n_docs=10")

    def test_bad_payloads_return_none_not_garbage(self):
        assert term_weights_from_lexical_index(None) is None
        assert term_weights_from_lexical_index("nope") is None
        assert term_weights_from_lexical_index({"n_docs": 0, "df": {"a": 1}}) is None
        assert term_weights_from_lexical_index({"n_docs": 5, "df": {}}) is None
        assert term_weights_from_lexical_index({"n_docs": "x", "df": {"a": 1}}) is None
