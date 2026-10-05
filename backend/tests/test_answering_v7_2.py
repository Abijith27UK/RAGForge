"""V7.2 unit tests: QueryTrace, EvidenceSelector, GroundingGate states,
FallbackAnswerGenerator honesty, citation fields, AnswerRun.

Deterministic and hermetic: no network, no API key, no Qdrant, no real LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.answer import (  # noqa: E402
    GATE_TO_GROUNDING,
    AnswerMode,
    AnswerPolicy,
    ConfidenceCategory,
    Evidence,
    EvidenceAssessment,
    EvidenceSet,
    GateDecision,
    GateSignal,
    GeneratedAnswer,
    GroundingState,
    QueryNature,
    QueryPlan,
    QueryTrace,
    QueryType,
)
from app.schemas.models import KnowledgeBase  # noqa: E402
from app.services.answering.evidence import (  # noqa: E402
    EvidenceSelector,
    ProvenancePreservingSelector,
    create_evidence_selector,
)
from app.services.answering.gate import (  # noqa: E402
    GroundingGate,
    HeuristicEvidenceGate,
    create_grounding_gate,
)
from app.services.answering.generator import (  # noqa: E402
    ExtractiveMockAnswerGenerator,
    FallbackAnswerGenerator,
    UnavailableAnswerGenerator,
)
from app.services.answering.query_processor import (  # noqa: E402
    PROCESSOR_VERSION,
    create_query_processor,
)

KB = KnowledgeBase(
    id="kb_test",
    name="Naval Architecture — Ship Stability",
    domain="Naval Architecture",
    purpose="study",
    target_audience="students",
)
PROC = create_query_processor()


def _plan(question: str, terms=None) -> QueryPlan:
    return QueryPlan(
        original_query=question,
        normalized_query=question,
        extracted_terms=terms if terms is not None else ["metacentric", "height", "gm"],
        query_type=QueryType.DEFINITION,
    )


def _evidence(n=2, *, score=0.8, doc_ids=None, content=None, source_type="pdf") -> EvidenceSet:
    doc_ids = doc_ids or [f"doc{i}" for i in range(n)]
    text = content or "The metacentric height GM must be positive for stable equilibrium."
    # Evidence ids follow the production format `ev_0001` (1-based), so tests
    # reference the same ids the pipeline really produces.
    items = [
        Evidence(
            evidence_id=f"ev_{i + 1:04d}",
            chunk_id=f"ch{i}",
            document_id=doc_ids[i],
            kb_id="kb_test",
            title=f"Ship Stability Notes {i}",
            source_type=source_type,
            content=text,
            retrieval_score=score,
            retrieval_strategy="dense",
            rank=i + 1,
            original_rank=i + 1,
            page=23 + i,
            section_path="Initial Stability",
            content_hash=f"hash{i}",
            trust_score=0.8,
        )
        for i in range(n)
    ]
    return EvidenceSet(items=items, strategy="dense")


# ---------------------------------------------------------------------------
# QueryTrace
# ---------------------------------------------------------------------------


class TestQueryTrace:
    def test_standalone_question_passes_through_unchanged(self):
        trace = PROC.process_with_trace("What is the metacentric height?", KB)
        assert trace.original_query == "What is the metacentric height?"
        assert trace.rewritten_query is None
        assert trace.subqueries == []
        assert trace.transformations == []
        assert trace.nature is QueryNature.KNOWLEDGE
        assert trace.processor == "heuristic-rules"
        assert trace.processor_version == PROCESSOR_VERSION

    def test_original_query_preserved_byte_for_byte(self):
        messy = "  What   is\tGM?  "
        trace = PROC.process_with_trace(messy, KB)
        assert trace.original_query == messy
        assert trace.normalized_query == "What is GM?"

    def test_timing_is_measured_not_estimated(self):
        trace = PROC.process_with_trace("What is GM?", KB)
        assert trace.timing_ms is not None
        assert trace.timing_ms >= 0

    @pytest.mark.parametrize(
        "question",
        ["hi", "Hello!", "thanks", "ok", "bye"],
    )
    def test_conversational_turns_detected(self, question):
        trace = PROC.process_with_trace(question, KB)
        assert trace.nature is QueryNature.CONVERSATIONAL

    @pytest.mark.parametrize("question", ["help", "what can you do", "who are you"])
    def test_non_knowledge_turns_detected(self, question):
        trace = PROC.process_with_trace(question, KB)
        assert trace.nature is QueryNature.NON_KNOWLEDGE

    def test_conversational_detection_does_not_fire_mid_question(self):
        """A real question containing 'thanks' is not a greeting."""
        trace = PROC.process_with_trace(
            "What is the metacentric height and thanks for explaining?", KB
        )
        assert trace.nature is not QueryNature.CONVERSATIONAL

    def test_underspecified_turn_detected(self):
        assert PROC.process_with_trace("and?", KB).nature is QueryNature.UNDERSPECIFIED

    def test_underspecified_does_not_reject_short_real_questions(self):
        """Short but specific questions must NOT be flagged underspecified."""
        for q in [
            "What is buoyancy?",
            "metacentric height?",
            "What is GM?",
            "Explain trim.",
        ]:
            assert PROC.process_with_trace(q, KB).nature is not QueryNature.UNDERSPECIFIED, q

    def test_deictic_follow_up_is_underspecified_without_domain_terms(self):
        trace = PROC.process_with_trace("What about the previous case?", KB)
        assert trace.nature is QueryNature.UNDERSPECIFIED

    def test_multi_hop_question_detected_and_flagged(self):
        trace = PROC.process_with_trace(
            "How does the center of gravity affect the metacentric height?", KB
        )
        assert trace.nature is QueryNature.MULTI_HOP
        assert any("multi-hop" in w.lower() for w in trace.warnings)

    def test_multi_hop_is_still_a_knowledge_question(self):
        trace = PROC.process_with_trace(
            "How does weight affect stability?", KB
        )
        assert trace.nature is QueryNature.MULTI_HOP
        assert trace.nature is not QueryNature.CONVERSATIONAL

    def test_multi_part_records_decomposition_not_implemented(self):
        trace = PROC.process_with_trace("What is GM and why does it matter?", KB)
        assert trace.subqueries == []
        assert any("decomposition" in w.lower() for w in trace.warnings)

    def test_expansion_is_opt_in_and_additive(self):
        off = PROC.process_with_trace("What is stability?", KB)
        assert off.expanded_terms == []
        assert off.transformations == []

    def test_expansion_never_replaces_the_question(self):
        trace = PROC.process_with_trace("What is stability?", KB, expand=True)
        # Even when expansion adds terms, the searched query is unchanged.
        assert trace.rewritten_query is None
        assert trace.original_query == "What is stability?"

    def test_passthrough_trace_when_processing_disabled(self):
        trace = PROC.passthrough_trace("What is GM?", reason="disabled by config")
        assert trace.enabled is False
        assert trace.transformations == []
        assert trace.rewritten_query is None
        assert trace.original_query == "What is GM?"
        assert any("disabled" in n for n in trace.notes)

    def test_base_trace_never_claims_a_rewrite(self):
        """The default ABC implementation must not imply transformations."""
        class Custom(PROC.__class__):
            name = "custom"

        trace = Custom().process_with_trace("What is GM?", KB)
        assert trace.rewritten_query is None
        assert trace.transformations == []


# ---------------------------------------------------------------------------
# EvidenceSelector
# ---------------------------------------------------------------------------


class TestEvidenceSelector:
    def test_selector_is_abstract(self):
        with pytest.raises(TypeError):
            EvidenceSelector()  # type: ignore[abstract]

    def test_default_selector_preserves_provenance(self):
        from app.schemas.models import RetrievalResponse, RetrievalResult

        response = RetrievalResponse(
            query="gm",
            top_k=2,
            strategy="dense",
            retrieval_run_id="rr_1",
            results=[
                RetrievalResult(
                    chunk_id="ch1", document_id="d1", text="GM must be positive.",
                    score=0.9, rank=1,
                    provenance={
                        "document_title": "Notes", "page": 12, "content_hash": "h1",
                        "source_type": "pdf", "trust_score": 0.7,
                    },
                )
            ],
        )
        selected = create_evidence_selector().select(response, kb_id="kb_test")
        assert len(selected.items) == 1
        item = selected.items[0]
        assert item.title == "Notes"
        assert item.page == 12
        assert item.content_hash == "h1"
        assert item.source_type == "pdf"
        assert item.trust_score == 0.7
        assert item.retrieval_run_id == "rr_1"

    def test_selector_does_not_invent_missing_fields(self):
        from app.schemas.models import RetrievalResponse, RetrievalResult

        response = RetrievalResponse(
            query="gm", top_k=1, strategy="dense",
            results=[RetrievalResult(chunk_id="ch1", document_id="d1",
                                     text="GM positive.", score=0.5, rank=1)],
        )
        item = create_evidence_selector().select(response, kb_id="kb_test").items[0]
        assert item.page is None
        assert item.slide is None
        assert item.publisher is None
        assert item.source_type is None

    def test_selector_records_that_no_score_floor_was_applied(self):
        from app.schemas.models import RetrievalResponse, RetrievalResult

        response = RetrievalResponse(
            query="gm", top_k=1, strategy="dense",
            results=[RetrievalResult(chunk_id="ch1", document_id="d1",
                                     text="unrelated text", score=0.01, rank=1)],
        )
        selected = ProvenancePreservingSelector().select(
            response, kb_id="kb_test", plan=_plan("gm")
        )
        assert len(selected.items) == 1, "a low score must NOT be filtered silently"
        assert any("no score floor" in n for n in selected.notes)

    def test_selector_name_and_version_recorded(self):
        assert create_evidence_selector().name
        assert create_evidence_selector().version


# ---------------------------------------------------------------------------
# GroundingGate states
# ---------------------------------------------------------------------------


class TestGroundingGateStates:
    gate = create_grounding_gate()

    def test_gate_is_a_grounding_gate(self):
        assert isinstance(self.gate, GroundingGate)
        assert isinstance(self.gate, HeuristicEvidenceGate)

    def test_all_five_states_exist(self):
        assert {s.value for s in GroundingState} == {
            "ANSWERED",
            "PARTIALLY_SUPPORTED",
            "INSUFFICIENT_EVIDENCE",
            "CONFLICTING_EVIDENCE",
            "NO_RELEVANT_EVIDENCE",
        }

    def test_every_decision_maps_to_a_state(self):
        for decision in GateDecision:
            assert decision in GATE_TO_GROUNDING

    def test_no_evidence_gives_no_relevant_evidence(self):
        assessment = self.gate.assess_with_trace(
            _plan("What is GM?"), EvidenceSet(strategy="dense")
        )
        assert assessment.grounding_state is GroundingState.NO_RELEVANT_EVIDENCE
        assert assessment.sufficient is False
        assert assessment.reason_code == "NO_EVIDENCE"

    def test_strong_evidence_gives_answered(self):
        assessment = self.gate.assess_with_trace(_plan("What is GM?"), _evidence(3))
        assert assessment.grounding_state is GroundingState.ANSWERED
        assert assessment.sufficient is True
        assert assessment.decision is GateDecision.ANSWER

    def test_conflicting_values_give_conflicting_evidence(self):
        items = _evidence(2, doc_ids=["docA", "docB"])
        items.items[0].content = "metacentric height = 1.2 m in the loaded condition."
        items.items[1].content = "metacentric height = 0.4 m in the loaded condition."
        assessment = self.gate.assess_with_trace(_plan("metacentric height value"), items)
        assert assessment.grounding_state is GroundingState.CONFLICTING_EVIDENCE
        assert any("conflicting" in n for n in assessment.notes)

    def test_conflicting_evidence_is_still_reported_when_answered(self):
        """The gate may answer AND flag the conflict — it must never hide it."""
        items = _evidence(2, doc_ids=["docA", "docB"])
        items.items[0].content = "metacentric height = 1.2 m metacentric height."
        items.items[1].content = "metacentric height = 0.4 m metacentric height."
        assessment = self.gate.assess_with_trace(_plan("metacentric height"), items)
        assert assessment.decision in (GateDecision.ANSWER, GateDecision.PARTIAL_ANSWER)
        assert assessment.grounding_state is GroundingState.CONFLICTING_EVIDENCE

    def test_partial_coverage_gives_partially_supported(self):
        items = _evidence(2, content="The metacentric height GM is a stability measure.")
        plan = _plan(
            "What is metacentric height and what is propeller cavitation?",
            terms=["metacentric", "height", "propeller", "cavitation"],
        )
        assessment = self.gate.assess_with_trace(plan, items)
        assert assessment.grounding_state is GroundingState.PARTIALLY_SUPPORTED
        assert assessment.decision is GateDecision.PARTIAL_ANSWER

    def test_low_alignment_gives_insufficient_evidence(self):
        items = _evidence(1, content="Propeller cavitation occurs at low pressure.")
        plan = _plan("What is the ISO 9001 audit procedure?",
                     terms=["iso", "audit", "procedure"])
        assessment = self.gate.assess_with_trace(plan, items)
        assert assessment.grounding_state is GroundingState.INSUFFICIENT_EVIDENCE
        assert assessment.sufficient is False

    def test_conversational_turn_is_not_judged_on_evidence(self):
        trace = PROC.process_with_trace("hello", KB)
        assessment = self.gate.assess_with_trace(_plan("hello"), _evidence(2), trace)
        assert assessment.grounding_state is GroundingState.INSUFFICIENT_EVIDENCE
        assert assessment.reason_code == "NON_KNOWLEDGE_TURN"
        assert assessment.signals == [], "evidence signals must not be computed"

    def test_underspecified_turn_still_uses_evidence_signals(self):
        """An underspecified question is still judged on real evidence."""
        trace = PROC.process_with_trace("and?", KB)
        assessment = self.gate.assess_with_trace(_plan("and?"), _evidence(2), trace)
        # The evidence signals must actually have run.
        assert assessment.signals
        assert assessment.grounding_state in (
            GroundingState.ANSWERED,
            GroundingState.PARTIALLY_SUPPORTED,
            GroundingState.INSUFFICIENT_EVIDENCE,
        )

    def test_every_assessment_carries_machine_readable_reasons(self):
        for ev in (_evidence(3), _evidence(1), EvidenceSet(strategy="dense")):
            assessment = self.gate.assess_with_trace(_plan("What is GM?"), ev)
            assert assessment.reason_code
            assert assessment.reason
            assert isinstance(assessment.signals, list)

    def test_state_is_not_a_fake_confidence_score(self):
        assessment = self.gate.assess_with_trace(_plan("What is GM?"), _evidence(3))
        assert assessment.confidence in (
            ConfidenceCategory.HIGH, ConfidenceCategory.MODERATE,
            ConfidenceCategory.LOW, ConfidenceCategory.NONE,
        )


# ---------------------------------------------------------------------------
# Generator honesty
# ---------------------------------------------------------------------------


class TestGeneratorHonesty:
    def test_fallback_generator_discloses_itself(self):
        preferred = UnavailableAnswerGenerator("llm", "LLM_API_KEY is required")
        fallback = ExtractiveMockAnswerGenerator()
        gen = FallbackAnswerGenerator(preferred, fallback, "LLM_API_KEY is required")
        assert gen.unavailable is True
        assert gen.is_mock is True
        assert "LLM_API_KEY" in gen.unavailable_reason

    def test_fallback_output_carries_a_warning_naming_the_reason(self):
        gen = FallbackAnswerGenerator(
            UnavailableAnswerGenerator("llm", "no key"), ExtractiveMockAnswerGenerator(), "no key"
        )
        out = gen.generate(_plan("What is GM?"), _evidence(1), AnswerPolicy(), "ok")
        assert any("FALLBACK" in w for w in out.warnings)
        assert any("no key" in w for w in out.warnings)
        assert any("mock" in w.lower() for w in out.warnings)

    def test_fallback_never_pretends_to_be_the_preferred_provider(self):
        gen = FallbackAnswerGenerator(
            UnavailableAnswerGenerator("llm", "no key"), ExtractiveMockAnswerGenerator(), "no key"
        )
        described = gen.describe()
        assert described["unavailable"] is True
        assert described["preferred_generator"] == "llm"
        assert described["fallback_generator"] == "extractive-mock"

    def test_mock_generator_labels_itself(self):
        gen = ExtractiveMockAnswerGenerator()
        assert gen.is_mock is True
        assert gen.unavailable is False
        out = gen.generate(_plan("What is GM?"), _evidence(1), AnswerPolicy(), "ok")
        assert any("NOT an LLM" in n for n in out.notes)

    def test_generator_describe_contains_no_secrets(self):
        gen = ExtractiveMockAnswerGenerator()
        described = gen.describe()
        assert set(described) == {
            "name", "version", "model", "is_mock", "unavailable", "unavailable_reason",
        }

    def test_mock_provider_quotes_the_relevant_chunk_not_a_stopword_match(self):
        """Regression: a stability question once quoted a propulsion sentence.

        The mock's selector used `len(term) > 3` substring matching, which dropped
        the critical acronym "GM" and kept the stop-word "when", so the only
        matching sentence came from an unrelated section. It must now select on
        the same stop-word-aware terms the query processor uses.
        """
        from app.services.answering.prompting import build_prompt

        evidence = EvidenceSet(
            items=[
                Evidence(
                    evidence_id="ev_0001", chunk_id="ch1", document_id="d1",
                    kb_id="kb_test", title="Stability", content=(
                        "The metacentric height GM must be positive for stable "
                        "equilibrium of a floating body."
                    ),
                    retrieval_score=0.50, retrieval_strategy="dense",
                    rank=1, original_rank=1, content_hash="h1",
                ),
                Evidence(
                    evidence_id="ev_0002", chunk_id="ch2", document_id="d2",
                    kb_id="kb_test", title="Propulsion", content=(
                        "Propeller cavitation occurs when the local pressure falls "
                        "below the vapour pressure of water."
                    ),
                    retrieval_score=0.07, retrieval_strategy="dense",
                    rank=2, original_rank=2, content_hash="h2",
                ),
            ],
            strategy="dense",
        )
        plan = _plan(
            "What happens to GM when the center of gravity rises?",
            terms=["happens", "gm", "center", "gravity", "rises"],
        )
        built = build_prompt(plan, evidence, AnswerPolicy(), "SUFFICIENT")

        from app.llm.provider import MockLLMProvider
        from app.services.answering.generator import AnswerDraftLLM

        draft = MockLLMProvider().generate_structured(
            built.user, AnswerDraftLLM, system=built.system
        )
        assert draft.claims, "expected a quoted sentence"
        assert draft.claims[0].citation_evidence_ids == ["ev_0001"], (
            "must cite the stability chunk, not the chunk that merely contains a "
            "stop-word from the question"
        )
        assert "metacentric" in draft.answer_text.lower()

    def test_mock_provider_abstains_when_no_sentence_is_relevant(self):
        from app.services.answering.prompting import build_prompt

        evidence = EvidenceSet(
            items=[Evidence(
                evidence_id="ev_0001", chunk_id="ch1", document_id="d1",
                kb_id="kb_test", title="Propulsion",
                content="Propeller cavitation occurs at low local pressure.",
                retrieval_score=0.5, retrieval_strategy="dense",
                rank=1, original_rank=1, content_hash="h1",
            )],
            strategy="dense",
        )
        plan = _plan("What is the ISO 9001 audit procedure?",
                     terms=["iso", "audit", "procedure"])
        built = build_prompt(plan, evidence, AnswerPolicy(), "INSUFFICIENT")
        from app.llm.provider import MockLLMProvider
        from app.services.answering.generator import AnswerDraftLLM

        draft = MockLLMProvider().generate_structured(
            built.user, AnswerDraftLLM, system=built.system
        )
        assert draft.abstain is True
        assert draft.abstention_reason


# ---------------------------------------------------------------------------
# Citation fields
# ---------------------------------------------------------------------------


class TestCitationFields:
    def _answer_for(self, evidence: EvidenceSet):
        from app.schemas.answer import ClaimDraft, ClaimType
        from app.services.answering.validation import validate_answer

        generated = GeneratedAnswer(
            text="The metacentric height GM must be positive.",
            claims=[
                ClaimDraft(
                    text="The metacentric height GM must be positive.",
                    claim_type=ClaimType.FACT,
                    citation_evidence_ids=["ev_0001"],
                )
            ],
        )
        assessment = EvidenceAssessment(
            sufficient=True, decision=GateDecision.ANSWER,
            grounding_state=GroundingState.ANSWERED,
            confidence=ConfidenceCategory.HIGH, reason_code="SUFFICIENT", reason="ok",
        )
        answer, _actions, _problems = validate_answer(
            generated, evidence, AnswerPolicy(mode=AnswerMode.GROUNDED), assessment,
            question="What is GM?", answer_id="ans_1", trace_id="atr_1",
            retrieval_run_id="rr_1", generator_name="t", generator_model="t",
            is_mock=True, kb_id="kb_test",
        )
        return answer

    def test_citation_carries_every_required_field(self):
        answer = self._answer_for(_evidence(1))
        assert answer.citations
        cite = answer.citations[0]
        assert cite.citation_id
        assert cite.evidence_id == "ev_0001"
        assert cite.document_id == "doc0"
        assert cite.chunk_id == "ch0"
        assert cite.source_title == "Ship Stability Notes 0"
        assert cite.source_type == "pdf"
        assert cite.page_number == 23
        assert cite.section_path == "Initial Stability"
        assert cite.content_hash == "hash0"

    def test_missing_page_stays_none_and_is_never_invented(self):
        evidence = _evidence(1)
        evidence.items[0].page = None
        evidence.items[0].slide = None
        answer = self._answer_for(evidence)
        cite = answer.citations[0]
        assert cite.page is None
        assert cite.page_number is None
        assert cite.slide is None
        assert cite.slide_number is None

    def test_slide_number_is_copied_when_present(self):
        evidence = _evidence(1, source_type="pptx")
        evidence.items[0].slide = 18
        answer = self._answer_for(evidence)
        cite = answer.citations[0]
        assert cite.slide == 18
        assert cite.slide_number == 18

    def test_source_title_falls_back_to_document_id_not_content(self):
        evidence = _evidence(1)
        evidence.items[0].title = ""
        answer = self._answer_for(evidence)
        assert answer.citations[0].source_title == "doc0"


# ---------------------------------------------------------------------------
# Case 3/4/5/6/7 from the spec's validation list
# ---------------------------------------------------------------------------


class TestValidationCases:
    def _run(self, generated, evidence, *, action="remove", decision=GateDecision.ANSWER):
        from app.services.answering.validation import validate_answer

        assessment = EvidenceAssessment(
            sufficient=decision is GateDecision.ANSWER,
            decision=decision,
            grounding_state=GATE_TO_GROUNDING[decision],
            confidence=ConfidenceCategory.HIGH if decision is GateDecision.ANSWER
            else ConfidenceCategory.NONE,
            reason_code="SUFFICIENT" if decision is GateDecision.ANSWER else "NO_EVIDENCE",
            reason="test reason",
        )
        return validate_answer(
            generated, evidence, AnswerPolicy(unsupported_claim_action=action), assessment,
            question="q", answer_id="a", trace_id="t", retrieval_run_id=None,
            generator_name="t", generator_model="t", is_mock=True, kb_id="kb_test",
        )

    def test_case_3_unsupported_claim_removed(self):
        from app.schemas.answer import ClaimDraft, ClaimType

        generated = GeneratedAnswer(
            text="GM is positive.",
            claims=[
                ClaimDraft(text="GM is positive.", claim_type=ClaimType.FACT,
                           citation_evidence_ids=["ev_0001"]),
                ClaimDraft(text="The vessel displaces 12000 tonnes.", claim_type=ClaimType.FACT,
                           citation_evidence_ids=[]),
            ],
        )
        answer, actions, problems = self._run(generated, _evidence(1))
        assert len(answer.claims) == 1
        assert any("remove_claim" in a for a in actions)

    def test_case_4_no_evidence_abstains(self):
        from app.schemas.answer import ClaimDraft, ClaimType

        generated = GeneratedAnswer(
            text="Something.",
            claims=[ClaimDraft(text="Something.", claim_type=ClaimType.FACT,
                               citation_evidence_ids=["ev_0001"])],
        )
        answer, actions, _ = self._run(generated, EvidenceSet(strategy="dense"))
        assert answer.status.value == "abstained"
        assert answer.citations == []

    def test_case_5_conflicting_evidence_reported_as_warning(self):
        from app.schemas.answer import ClaimDraft, ClaimType

        evidence = _evidence(1)
        assessment = EvidenceAssessment(
            sufficient=True, decision=GateDecision.ANSWER,
            grounding_state=GroundingState.CONFLICTING_EVIDENCE,
            confidence=ConfidenceCategory.MODERATE, reason_code="SUFFICIENT",
            reason="answered with conflict",
            notes=["conflicting evidence: 'gm' has conflicting values: 1.2 vs 0.4"],
        )
        from app.services.answering.validation import validate_answer

        generated = GeneratedAnswer(
            text="GM must be positive.",
            claims=[ClaimDraft(text="GM must be positive.", claim_type=ClaimType.FACT,
                               citation_evidence_ids=["ev_0001"])],
        )
        answer, _a, _p = validate_answer(
            generated, evidence, AnswerPolicy(), assessment,
            question="q", answer_id="a", trace_id="t", retrieval_run_id=None,
            generator_name="t", generator_model="t", is_mock=True, kb_id="kb_test",
        )
        assert any("disagree" in w for w in answer.warnings)

    def test_case_6_partial_evidence_gives_partial_status(self):
        from app.schemas.answer import ClaimDraft, ClaimType

        generated = GeneratedAnswer(
            text="GM must be positive.",
            claims=[ClaimDraft(text="GM must be positive.", claim_type=ClaimType.FACT,
                               citation_evidence_ids=["ev_0001"])],
        )
        answer, _a, _p = self._run(
            generated, _evidence(1), decision=GateDecision.PARTIAL_ANSWER
        )
        assert answer.status.value == "partial"

    def test_case_7_strong_evidence_gives_grounded_status(self):
        from app.schemas.answer import ClaimDraft, ClaimType

        generated = GeneratedAnswer(
            text="The metacentric height GM must be positive for stable equilibrium.",
            claims=[ClaimDraft(
                text="The metacentric height GM must be positive for stable equilibrium.",
                claim_type=ClaimType.FACT, citation_evidence_ids=["ev_0001"])],
        )
        answer, _a, _p = self._run(generated, _evidence(2))
        assert answer.status.value == "grounded"
        assert answer.confidence is ConfidenceCategory.HIGH

    def test_downgrade_action_records_a_downgrade_for_unsupported_claims(self):
        """A claim whose ONLY citation is fabricated is UNSUPPORTED, and the
        downgrade policy must record that it was downgraded rather than removed."""
        from app.schemas.answer import ClaimDraft, ClaimType

        generated = GeneratedAnswer(
            text="The vessel displaces 12000 tonnes at the design draft.",
            claims=[ClaimDraft(
                text="The vessel displaces 12000 tonnes at the design draft.",
                claim_type=ClaimType.NUMERICAL,
                citation_evidence_ids=["ev_0001", "ev_9999"])],
        )
        answer, actions, problems = self._run(generated, _evidence(1), action="downgrade")
        assert problems, "the fabricated citation must be reported"
        assert any("downgrade" in a for a in actions)
        assert len(answer.claims) == 1, "downgrade keeps the claim"
        assert answer.status.value == "partial", "a downgraded claim is not fully grounded"

    def test_weak_lexical_overlap_is_partially_supported_not_unsupported(self):
        """The lexical check is a HEURISTIC: weak overlap downgrades support to
        partially_supported and says so, rather than asserting unsupported."""
        from app.schemas.answer import ClaimDraft, ClaimType, SupportStatus

        generated = GeneratedAnswer(
            text="Something about unrelated topics entirely.",
            claims=[ClaimDraft(
                text="Something about unrelated topics entirely.",
                claim_type=ClaimType.FACT, citation_evidence_ids=["ev_0001"])],
        )
        answer, _actions, _problems = self._run(generated, _evidence(1))
        assert len(answer.claims) == 1
        claim = answer.claims[0]
        assert claim.support_status is SupportStatus.PARTIALLY_SUPPORTED
        assert claim.support_check == "lexical_overlap"
        assert "semantic entailment NOT PERFORMED" in claim.support_note

    def test_abstain_action_abstains_on_any_problem(self):
        from app.schemas.answer import ClaimDraft, ClaimType

        generated = GeneratedAnswer(
            text="Fabricated.",
            claims=[ClaimDraft(text="Fabricated.", claim_type=ClaimType.FACT,
                               citation_evidence_ids=["ev_9999"])],
        )
        answer, actions, _ = self._run(generated, _evidence(1), action="abstain")
        assert answer.status.value == "abstained"
        assert any("abstain" in a for a in actions)


# ---------------------------------------------------------------------------
# AnswerRun record
# ---------------------------------------------------------------------------


class TestAnswerRunRecord:
    def test_unmeasured_latency_defaults_to_none(self):
        from app.schemas.answer import AnswerRun

        run = AnswerRun(id="arun_1", kb_id="kb1", question="q")
        assert run.generation_ms is None
        assert run.retrieval_ms is None
        assert run.total_ms is None

    def test_run_records_grounding_reasons(self):
        from app.schemas.answer import AnswerRun

        run = AnswerRun(
            id="arun_1", kb_id="kb1", question="q",
            grounding_decision="ABSTAIN",
            grounding_state=GroundingState.INSUFFICIENT_EVIDENCE,
            grounding_reasons=["LOW_LEXICAL_ALIGNMENT", "not enough terms"],
            grounding_reason_code="LOW_LEXICAL_ALIGNMENT",
            grounding_sufficient=False,
        )
        assert run.grounding_reasons[0] == "LOW_LEXICAL_ALIGNMENT"
        assert run.grounding_state is GroundingState.INSUFFICIENT_EVIDENCE

    def test_run_carries_every_version_string(self):
        from app.schemas.answer import AnswerRun

        run = AnswerRun(
            id="arun_1", kb_id="kb1", question="q",
            prompt_version="v7.1", answerer_version="v7.2",
            query_processor="heuristic-rules", query_processor_version="v7.2",
            evidence_selector="provenance-preserving", grounding_gate="heuristic-rules",
            model="mock/deterministic-mock", provider="llm",
        )
        assert run.answerer_version == "v7.2"
        assert run.query_processor_version == "v7.2"
        assert run.evidence_selector == "provenance-preserving"


# ---------------------------------------------------------------------------
# Conversation memory isolation
# ---------------------------------------------------------------------------


class TestConversationIsolation:
    def test_message_used_as_knowledge_defaults_false(self):
        from app.schemas.answer import Message, MessageRole

        message = Message(
            id="msg_1", conversation_id="conv_1", kb_id="kb1",
            role=MessageRole.ASSISTANT, content="GM must be positive.",
        )
        assert message.used_as_knowledge is False

    def test_conversation_is_kb_scoped(self):
        from app.schemas.answer import Conversation

        conversation = Conversation(id="conv_1", kb_id="kb1")
        assert conversation.kb_id == "kb1"

    def test_reference_detection_is_conservative(self):
        from app.services.answering.chat import is_reference_turn

        assert is_reference_turn("what about that?")
        assert is_reference_turn("what about the free surface effect?")
        assert is_reference_turn("tell me more")
        # A real standalone question is NOT a reference turn.
        assert not is_reference_turn("What is the metacentric height?")
        assert not is_reference_turn("Explain propeller cavitation in detail.")
