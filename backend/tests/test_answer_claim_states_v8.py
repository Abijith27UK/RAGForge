"""V8 STEP 6/7 regression tests — claim-level states and citation metrics.

Covers the five-state claim vocabulary (SUPPORTED, PARTIALLY_SUPPORTED,
UNSUPPORTED, CONTRADICTED, UNVERIFIABLE), the claim-state ratios over ALL
claims, and the two STEP 7 citation metrics:

* fabricated_citation_rate — a reference that does not resolve to retrieved
  evidence (or fails provenance validation) cannot be verified; it is
  fabricated by construction at evaluation time;
* unsupported_citation_rate — citations on claims the entailment provider
  judged NOT_SUPPORTED, over JUDGED citations only (unknown judgements are
  excluded, never counted as supported).

Hermetic: no network, no Qdrant, no LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.answer import (  # noqa: E402
    AnswerStatus, Citation, GroundingState, GateDecision, SupportStatus,
)
from app.services.answer_eval.benchmark import (  # noqa: E402
    Answerability, AnswerBenchmarkQuestion, ExpectedGroundingState,
    RequiredEvidence,
)
from app.services.answer_eval.evaluator import DeterministicAnswerEvaluator  # noqa: E402
from app.services.answer_eval.metrics import CLAIM_EVALUATED_STATES  # noqa: E402
from app.services.answer_eval.run import aggregate_results  # noqa: E402

from tests.test_answer_eval_v8 import (  # noqa: E402
    GOOD_EVIDENCE_TEXT, _answer, _claim, _evidence,
)

EVAL = DeterministicAnswerEvaluator()

GOOD = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
SHORT = _evidence("chk_short", "Too short.", evidence_id="ev_0002")


def _question() -> AnswerBenchmarkQuestion:
    return AnswerBenchmarkQuestion(
        question_id="q-001",
        question="What must the metacentric height GM be for stable equilibrium?",
        answerability=Answerability.ANSWERABLE,
        required_evidence=[
            RequiredEvidence(chunk_id="chk_good", document_id="doc_chk_good",
                             content_hash="hash_chk_good")
        ],
        expected_grounding_state=ExpectedGroundingState.ANY_ACCEPTABLE,
    )


def _state_of(answer, evidence) -> str:
    r = EVAL.evaluate(_question(), answer, evidence)
    return r.claim_verdicts[0].evaluated_state


class TestFiveStateVocabulary:
    def test_supported_claim(self):
        state = _state_of(
            _answer(claims=[_claim(
                "The metacentric height GM must be positive for stable "
                "equilibrium.", ["ev_0001"])], citations_for=[GOOD]),
            [GOOD],
        )
        assert state == "SUPPORTED"
        assert state in CLAIM_EVALUATED_STATES

    def test_partially_supported_claim(self):
        claim = _claim(
            "The metacentric height GM must be positive for stable "
            "equilibrium.", ["ev_0001"])
        claim.support_status = SupportStatus.PARTIALLY_SUPPORTED
        state = _state_of(
            _answer(claims=[claim], citations_for=[GOOD]), [GOOD])
        assert state == "PARTIALLY_SUPPORTED"

    def test_uncited_factual_claim_is_unsupported(self):
        state = _state_of(
            _answer(claims=[_claim("Torque tolerance is 25 Nm.", [])],
                    citations=[]),
            [GOOD],
        )
        assert state == "UNSUPPORTED"

    def test_contradicted_claim_beats_the_generators_own_status(self):
        """Entailment wins: declared SUPPORTED, evidence says otherwise."""
        state = _state_of(
            _answer(claims=[_claim(
                "The metacentric height GM must be negative for stable "
                "equilibrium.", ["ev_0001"])], citations_for=[GOOD]),
            [GOOD],
        )
        assert state == "CONTRADICTED"

    def test_thin_evidence_is_unverifiable_not_supported(self):
        """Evidence the provider cannot judge must never become SUPPORTED."""
        state = _state_of(
            _answer(claims=[_claim(
                "The transverse metacentric height is derived from the "
                "righting arm.", ["ev_0002"])], citations_for=[SHORT]),
            [GOOD, SHORT],
        )
        assert state == "UNVERIFIABLE"


class TestClaimStateRatios:
    def _mixed_answer(self):
        c1 = _claim("The metacentric height GM must be positive for stable "
                    "equilibrium.", ["ev_0001"], "cl_sup")
        c2 = _claim("The metacentric height GM must be positive for stable "
                    "equilibrium.", ["ev_0001"], "cl_part")
        c2.support_status = SupportStatus.PARTIALLY_SUPPORTED
        c3 = _claim("The metacentric height GM must be negative for stable "
                    "equilibrium.", ["ev_0001"], "cl_contra")
        c4 = _claim("Torque tolerance is 25 Nm.", [], "cl_uncited")
        c5 = _claim("The transverse metacentric height is derived from the "
                    "righting arm.", ["ev_0002"], "cl_unver")
        return _answer(claims=[c1, c2, c3, c4, c5],
                       citations_for=[GOOD, SHORT])

    def test_ratios_bucket_every_claim_once(self):
        r = EVAL.evaluate(_question(), self._mixed_answer(), [GOOD, SHORT])
        states = [v.evaluated_state for v in r.claim_verdicts]
        assert sorted(set(states)) == sorted(CLAIM_EVALUATED_STATES)
        assert r.supported_claim_ratio.value == 0.2      # 1 of 5
        assert r.partial_claim_ratio.value == 0.2        # 1 of 5
        assert r.unsupported_claim_ratio.value == 0.4    # CONTRADICTED + uncited
        # the four buckets partition the claims: the gap is UNVERIFIABLE
        gap = 1.0 - (r.supported_claim_ratio.value
                     + r.partial_claim_ratio.value
                     + r.unsupported_claim_ratio.value)
        assert round(gap, 6) == 0.2
        assert r.supported_claim_ratio.sample_size == 5

    def test_unsupported_ratio_is_not_the_old_uncited_rate(self):
        """A contradicted claim is unsupported but NOT uncited."""
        answer = _answer(claims=[_claim(
            "The metacentric height GM must be negative for stable "
            "equilibrium.", ["ev_0001"])], citations_for=[GOOD])
        r = EVAL.evaluate(_question(), answer, [GOOD])
        assert r.unsupported_claim_ratio.value == 1.0
        assert r.unsupported_claim_rate.value == 0.0     # legacy: uncited only

    def test_no_claims_means_unknown_not_zero(self):
        r = EVAL.evaluate(_question(), _answer(claims=[], citations=[]), [GOOD])
        assert r.supported_claim_ratio.measured is False
        assert r.partial_claim_ratio.measured is False
        assert r.unsupported_claim_ratio.measured is False
        assert r.supported_claim_ratio.value is None


class TestFabricatedCitationRate:
    def test_unresolvable_reference_counts_as_fabricated(self):
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.",
                           ["ev_0001", "ev_9999"])],
            citations_for=[GOOD],
        )
        r = EVAL.evaluate(_question(), answer, [GOOD])
        # refs = {ev_0001, ev_9999}; ev_9999 was never retrieved
        assert r.fabricated_citation_rate.value == 0.5
        assert r.fabricated_citation_rate.sample_size == 2
        assert "unresolvable" in r.fabricated_citation_rate.reason

    def test_clean_answer_has_zero_fabrication(self):
        answer = _answer(claims=[_claim(
            "The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[GOOD])
        r = EVAL.evaluate(_question(), answer, [GOOD])
        assert r.fabricated_citation_rate.value == 0.0

    def test_provenance_invalid_citation_counts_as_fabricated(self):
        bad_citation = Citation(
            citation_id="cite_ev_0001",
            evidence_id="ev_0001",
            chunk_id="chk_good",
            document_id="doc_chk_good",
            source_title="Source chk_good",
            source_type="pdf",
            content_hash="hash_chk_good",
            snippet=GOOD_EVIDENCE_TEXT[:200],
            validation="invalid",
        )
        answer = _answer(claims=[_claim(
            "The metacentric height GM must be positive.", ["ev_0001"])],
            citations=[bad_citation])
        r = EVAL.evaluate(_question(), answer, [GOOD])
        assert r.fabricated_citation_rate.value == 1.0
        assert "provenance-invalid" in r.fabricated_citation_rate.reason

    def test_abstention_is_unknown_not_zero(self):
        answer = _answer(
            text="I don't have enough evidence.",
            claims=[], citations=[],
            status=AnswerStatus.ABSTAINED,
            state=GroundingState.NO_RELEVANT_EVIDENCE,
            decision=GateDecision.ABSTAIN,
            sufficient=False,
        )
        r = EVAL.evaluate(_question(), answer, [GOOD])
        assert r.fabricated_citation_rate.measured is False
        assert "not applicable" in r.fabricated_citation_rate.reason


class TestUnsupportedCitationRate:
    def test_citations_on_contradicted_claims_count(self):
        answer = _answer(claims=[_claim(
            "The metacentric height GM must be negative for stable "
            "equilibrium.", ["ev_0001"])], citations_for=[GOOD])
        r = EVAL.evaluate(_question(), answer, [GOOD])
        assert r.unsupported_citation_rate.value == 1.0
        assert r.unsupported_citation_rate.sample_size == 1

    def test_mixed_judgements_use_judged_denominator(self):
        c1 = _claim("The metacentric height GM must be positive for stable "
                    "equilibrium.", ["ev_0001"], "cl_ok")
        c2 = _claim("The metacentric height GM must be negative for stable "
                    "equilibrium.", ["ev_0001"], "cl_bad")
        c3 = _claim("The transverse metacentric height is derived from the "
                    "righting arm.", ["ev_0002"], "cl_unver")
        answer = _answer(claims=[c1, c2, c3], citations_for=[GOOD, SHORT])
        r = EVAL.evaluate(_question(), answer, [GOOD, SHORT])
        # judged refs: cl_ok(1) + cl_bad(1); cl_unver is UNKNOWN -> excluded
        assert r.unsupported_citation_rate.value == 0.5
        assert r.unsupported_citation_rate.sample_size == 2

    def test_unjudged_citations_are_unknown_not_supported(self):
        answer = _answer(claims=[_claim(
            "The transverse metacentric height is derived from the righting "
            "arm.", ["ev_0002"])], citations_for=[SHORT])
        r = EVAL.evaluate(_question(), answer, [GOOD, SHORT])
        assert r.unsupported_citation_rate.measured is False
        assert "not counted" in r.unsupported_citation_rate.reason or \
               "UNKNOWN" in r.unsupported_citation_rate.reason


class TestAggregateClaimMetrics:
    def test_new_metrics_are_aggregated_and_listed(self):
        good = EVAL.evaluate(_question(), _answer(
            claims=[_claim("The metacentric height GM must be positive for "
                           "stable equilibrium.", ["ev_0001"])],
            citations_for=[GOOD]), [GOOD])
        agg = aggregate_results([good])
        assert agg.supported_claim_ratio.value == 1.0
        assert agg.fabricated_citation_rate.value == 0.0
        assert agg.unsupported_citation_rate.value == 0.0
        assert agg.unknown_metrics  # correctness etc. still unknown

    def test_unknown_metrics_include_new_metrics_when_unmeasured(self):
        answer = _answer(
            text="No evidence available.",
            claims=[], citations=[],
            status=AnswerStatus.ABSTAINED,
            state=GroundingState.NO_RELEVANT_EVIDENCE,
            decision=GateDecision.ABSTAIN,
            sufficient=False,
        )
        r = EVAL.evaluate(_question(), answer, [GOOD])
        agg = aggregate_results([r])
        for name in ("supported_claim_ratio", "partial_claim_ratio",
                     "unsupported_claim_ratio", "fabricated_citation_rate",
                     "unsupported_citation_rate"):
            assert name in agg.unknown_metrics, name
