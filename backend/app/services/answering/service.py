"""V7 — the grounded answer orchestration service.

Pipeline (each stage recorded in the AnswerTrace):

    question -> QueryPlan -> RetrievalService.run() -> EvidenceSet
             -> EvidenceGate.assess() -> AnswerGenerator.generate()
             -> validation pipeline -> Answer -> persisted with its trace

Rules enforced here:
* retrieval goes through the EXISTING RetrievalService/registry — there is no
  strategy-specific branch anywhere in this module;
* the gate decides BEFORE generation, so an insufficient-evidence question
  never reaches the generator in abstain mode (no LLM call to tempt a guess);
* the raw generator output is never returned to the client: only the validated
  `Answer` is, with the raw text preserved inside the trace;
* observability never breaks an answer: persistence failures are logged and
  surfaced as warnings, not raised.
"""
from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.answer import (
    Answer,
    AnswerMode,
    AnswerPolicy,
    AnswerTrace,
    EvidenceAssessment,
    QueryPlan,
    QueryTrace,
)
from app.schemas.models import DomainSpec, KnowledgeBase
from app.services.answering.evidence import (
    EvidenceSelector,
    build_evidence_set,
    create_evidence_selector,
)
from app.services.answering.gate import EvidenceGate, GroundingGate, create_evidence_gate
from app.services.answering.generator import (
    AnswerGenerationError,
    AnswerGenerator,
    create_answer_generator,
)
from app.services.answering.query_processor import QueryProcessor, create_query_processor
from app.services.answering.trace import AnswerTraceRecorder
from app.services.answering.validation import validate_answer
from app.services.retrieval.service import (
    KnowledgeBaseNotFound,
    RetrievalService,
)
from app.utils.ids import new_id

logger = logging.getLogger(__name__)

#: Answer modes that exist in V7. Others are future work and rejected loudly.
_SUPPORTED_MODES = {m.value for m in AnswerMode}


class AnswerError(RuntimeError):
    """Base error for the answering pipeline (HTTP 503 by default)."""


class AnswerConfigError(AnswerError):
    """Invalid caller input (HTTP 400)."""


class AnsweringService:
    def __init__(
        self,
        repo: Repository,
        settings: Settings | None = None,
        *,
        retrieval: RetrievalService | None = None,
        processor: QueryProcessor | None = None,
        gate: EvidenceGate | None = None,
        generator: AnswerGenerator | None = None,
        selector: EvidenceSelector | None = None,
    ) -> None:
        self._repo = repo
        self._settings = settings or get_settings()
        self._retrieval = retrieval or RetrievalService(repo, self._settings)
        self._processor = processor or create_query_processor()
        self._gate = gate or create_evidence_gate()
        self._selector = selector or create_evidence_selector()
        self._generator = generator  # resolved lazily so config errors stay local

    # -- helpers ------------------------------------------------------------

    def _get_generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = create_answer_generator(self._settings)
        return self._generator

    def _domain_spec(self, kb_id: str) -> DomainSpec | None:
        try:
            return self._repo.get_domain_spec(kb_id)
        except Exception:  # pragma: no cover - defensive
            return None

    # -- main entry ---------------------------------------------------------

    def answer(
        self,
        kb_id: str,
        question: str,
        *,
        mode: str | None = None,
        strategy: str | None = None,
        overrides: dict | None = None,
        policy: AnswerPolicy | None = None,
        expand_query: bool = False,
        query_trace: QueryTrace | None = None,
    ) -> tuple[Answer, AnswerTrace, EvidenceAssessment]:
        """Run the full pipeline. Returns (answer, trace, assessment).

        `query_trace`, when supplied, is the trace the CALLER already recorded
        (used by the chat service, which needs the trace before retrieval so it
        can resolve conversational references). Passing it in keeps one single
        query-processing pass per request instead of two that could disagree.

        Raises KnowledgeBaseNotFound (404), AnswerConfigError (400),
        RetrievalError/AnswerGenerationError (503) — mapped by the route.
        """
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")

        mode_key = (mode or AnswerMode.ABSTAIN_IF_UNSUPPORTED.value).strip().lower()
        if mode_key not in _SUPPORTED_MODES:
            raise AnswerConfigError(
                f"Unknown answer mode {mode!r}. Supported in V7: {sorted(_SUPPORTED_MODES)} "
                f"(concise/detailed/exam/teaching/engineering are NOT implemented yet)."
            )

        policy = policy or AnswerPolicy(mode=AnswerMode(mode_key))
        answer_id = new_id("ans")
        trace_id = new_id("atr")
        recorder = AnswerTraceRecorder(trace_id=trace_id, kb_id=kb_id, question=question)

        # -- Stage 1: query processing ------------------------------------
        spec = self._domain_spec(kb_id)
        if query_trace is not None:
            plan: QueryPlan = self._processor.process(question, kb, spec)
            recorder.stage(
                "query_processing",
                detail=f"processor={query_trace.processor} {query_trace.processor_version} (precomputed)",
                ms=query_trace.timing_ms,
            )
        else:
            with recorder.measure("query_processing", detail="heuristic rules, no LLM"):
                query_trace = self._processor.process_with_trace(
                    question, kb, spec, expand=expand_query
                )
                plan = self._processor.process(question, kb, spec)
        recorder.trace.query_plan = plan
        recorder.trace.query_trace = query_trace
        for w in query_trace.warnings:
            recorder.trace.generation_warnings.append(w)
        recorder.note(
            f"query type: {plan.query_type.value} ({plan.classification_method}); "
            f"nature={query_trace.nature.value}; "
            f"transformations={query_trace.transformations or 'none'}"
        )

        # -- Stage 2: retrieval (existing registry, no branching here) -----
        try:
            with recorder.measure("retrieval", detail=f"strategy={strategy or 'kb-config'}"):
                response = self._retrieval.run(
                    kb_id,
                    question,
                    strategy=strategy,
                    overrides=overrides,
                )
        except Exception:
            # measure() already recorded the error stage
            recorder.skip("evidence_assembly", "retrieval failed")
            recorder.skip("evidence_gate", "retrieval failed")
            recorder.skip("generation", "retrieval failed")
            recorder.skip("citation_validation", "retrieval failed")
            raise
        recorder.attach_retrieval(response)

        # -- Stage 3: evidence assembly -------------------------------------
        with recorder.measure("evidence_assembly", detail=f"selector={self._selector.name}"):
            evidence = self._selector.select(response, kb_id=kb_id, plan=plan)
        recorder.attach_evidence(evidence)

        # -- Stage 4: gate ----------------------------------------------------
        with recorder.measure("evidence_gate", detail=f"gate={self._gate.name}"):
            assessment = (
                self._gate.assess_with_trace(plan, evidence, query_trace)
                if isinstance(self._gate, GroundingGate)
                else self._gate.assess(plan, evidence)
            )
        recorder.trace.assessment = assessment
        recorder.note(
            f"gate: {assessment.decision.value} / {assessment.grounding_state.value} "
            f"({assessment.reason_code})"
        )

        generator = self._get_generator()

        # -- Stage 5: generation ----------------------------------------------
        # In ABSTAIN_IF_UNSUPPORTED mode an insufficient gate skips generation
        # entirely: no model is asked to answer a question we already know we
        # cannot ground. GROUNDED mode still generates for PARTIAL decisions.
        generated = None
        if assessment.decision.value == "ABSTAIN" and (
            policy.mode is AnswerMode.ABSTAIN_IF_UNSUPPORTED or not evidence.items
        ):
            recorder.skip(
                "generation",
                "evidence gate returned ABSTAIN; no model call was made "
                "(the abstention is deterministic)",
            )
        elif assessment.decision.value == "ASK_CLARIFICATION" and not evidence.items:
            recorder.skip("generation", "clarification required before generation")
        else:
            try:
                with recorder.measure("generation", detail=f"generator={generator.name}"):
                    generated = generator.generate(
                        plan, evidence, policy, assessment.reason
                    )
                recorder.attach_generated(generated)
            except AnswerGenerationError as exc:
                recorder.stage(
                    "generation",
                    status="error",
                    detail=f"generation failed: {exc}"[:300],
                )
                recorder.skip("citation_validation", "generation failed")
                answer = self._failure_answer(
                    kb_id=kb, question=question, answer_id=answer_id,
                    assessment=assessment, policy=policy, generator=generator,
                    reason=str(exc), response=response, recorder=recorder,
                )
                trace = self._persist(recorder, answer, kb_id, generator=generator)
                return answer, trace, assessment

        # -- Stage 6: validation --------------------------------------------
        if generated is None:
            # Build an abstention from the assessment without a generator call.
            from app.schemas.answer import GeneratedAnswer

            generated = GeneratedAnswer(
                text="",
                abstain_requested=True,
                abstention_reason=assessment.reason,
                notes=["no generation was performed"],
            )

        with recorder.measure("citation_validation", detail="deterministic checks"):
            answer, actions, problems = validate_answer(
                generated,
                evidence,
                policy,
                assessment,
                question=question,
                answer_id=answer_id,
                trace_id=trace_id,
                retrieval_run_id=response.retrieval_run_id,
                generator_name=generator.name,
                generator_model=getattr(generator, "model", generator.name),
                is_mock=generator.is_mock,
                kb_id=kb_id,
            )
        answer.kb_id = kb_id
        answer.answer_mode = policy.mode
        answer.prompt_version = policy.prompt_version
        # Disclose a degraded generator on the ANSWER itself, not only in logs.
        if getattr(generator, "unavailable", False):
            answer.warnings.append(
                "generator degraded: "
                f"{getattr(generator, 'unavailable_reason', 'preferred generator unavailable')}"
            )

        # -- Finalization ----------------------------------------------------
        recorder.stage("finalization", detail=f"status={answer.status.value}")
        trace = self._persist(
            recorder,
            answer,
            kb_id,
            generator=generator,
            validation_actions=actions,
            citation_problems=problems,
        )
        return answer, trace, assessment

    # -- failure / persistence helpers -------------------------------------

    def _failure_answer(self, *, kb, question, answer_id, assessment, policy,
                        generator, reason, response, recorder) -> Answer:
        from app.schemas.answer import AnswerStatus, ConfidenceCategory

        answer = Answer(
            answer_id=answer_id,
            kb_id=kb.id,
            question=question,
            status=AnswerStatus.GENERATION_FAILED,
            text=(
                "Answer generation failed, so no grounded answer is available. "
                "The retrieved evidence and its provenance are listed below."
            ),
            confidence=ConfidenceCategory.NONE,
            confidence_basis=f"generation failed: {reason}",
            generated_by=generator.name,
            model=getattr(generator, "model", generator.name),
            is_mock=generator.is_mock,
            answer_mode=policy.mode,
            retrieval_run_id=response.retrieval_run_id,
            warnings=[f"generation failed: {reason}"]
            + (
                [f"generator degraded: {generator.unavailable_reason}"]
                if getattr(generator, "unavailable", False)
                else []
            ),
            answer_trace_id=recorder.trace.id,
            assessment=assessment,
        )
        recorder.trace.raw_generated_text = ""
        return answer

    def _persist(
        self,
        recorder,
        answer: Answer,
        kb_id: str,
        *,
        generator,
        validation_actions: list[str] | None = None,
        citation_problems: list[dict] | None = None,
    ) -> AnswerTrace:
        trace = recorder.finalize(
            answer,
            validation_actions=validation_actions or [],
            citation_problems=citation_problems or [],
            generator=generator.name if generator else "",
            generator_model=getattr(generator, "model", "") if generator else "",
            is_mock=bool(getattr(generator, "is_mock", False)) if generator else False,
        )
        # answer carries the trace id for lookup
        answer.answer_trace_id = trace.id
        try:
            self._repo.create_answer(answer)
            self._repo.create_answer_trace(trace)
        except Exception as exc:  # observability must never break the response
            logger.warning("Could not persist answer/trace for %s: %s", kb_id, exc)
            answer.warnings.append(
                "answer could not be persisted to the answer history (see server logs)"
            )
        return trace

    # -- read APIs -----------------------------------------------------------

    def get_answer(self, kb_id: str, answer_id: str) -> Answer | None:
        return self._repo.get_answer(kb_id, answer_id)

    def get_trace(self, kb_id: str, trace_id: str) -> AnswerTrace | None:
        return self._repo.get_answer_trace(kb_id, trace_id)


__all__ = [
    "AnswerConfigError",
    "AnswerError",
    "AnsweringService",
]
