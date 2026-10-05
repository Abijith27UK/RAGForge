"""V7 Phase 13 — grounded chat: conversation memory over the answering pipeline.

This module adds the CHAT layer. It does not re-implement answering: every turn
is delegated to `AnsweringService.answer`, so chat and `/answer` can never drift
apart in behaviour, and an answer reached through chat is grounded exactly like
one reached directly.

Two things the chat layer adds, and nothing else:

1. CONVERSATION MEMORY (minimal, and deliberately NOT knowledge).
   A follow-up like "what about the previous case?" is unanswerable on its own.
   `resolve_reference` expands such a turn into a standalone question by
   borrowing terms from earlier USER turns. Rules that make this safe:

   * only USER messages are used — the assistant's own words are never fed back
     as if they were fact;
   * the borrowed terms are recorded in `Message.resolved_from`, so the
     transformation is auditable and the user can see their question was expanded;
   * the expanded question is still answered ONLY from retrieved KB evidence, or
     not at all. Conversation history can never become a source: it is not
     chunked, not embedded, not indexed, and `Message.used_as_knowledge` is
     permanently False.

2. OBSERVABILITY.
   An `AnswerRun` row records the measured timings, the grounding decision and
   its reasons, the selected evidence ids and every version string, so any answer
   can be reproduced and audited later. Timings that were not measured stay None.

Honesty rules carried over from the answering pipeline: no fabricated
confidence, no silent fallback, no answer without evidence.
"""
from __future__ import annotations

import logging
import re
import time

from app.config import Settings, get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.answer import (
    Answer,
    AnswerMode,
    AnswerPolicy,
    AnswerRun,
    AnswerStatus,
    AnswerTrace,
    ChatRequest,
    ChatResponse,
    Conversation,
    ConversationDetail,
    ConversationSummary,
    EvidenceAssessment,
    Message,
    MessageRole,
    QueryNature,
    QueryTrace,
)
from app.services.answering.service import (
    AnswerConfigError,
    AnsweringService,
)
from app.utils.ids import new_id

logger = logging.getLogger(__name__)

#: Version of the answerer as a whole (query processing + gate + generator +
#: validation wired together). Bump when the pipeline's behaviour changes.
ANSWERER_VERSION = "v7.2"

#: Follow-up phrasing that cannot stand alone: it refers to something in the
#: conversation. Matched at the START of the turn (or alone), so a full question
#: that merely contains "that" is not treated as a reference.
REFERENCE_RE = re.compile(
    r"^\s*(?:and|but|so|then|ok|okay|also)?\s*"
    r"(?:what|how)\s+about\b"
    r"|^\s*(?:why|how)\s*[?]?\s*$"
    r"|^\s*(?:tell\s+me\s+)?more\b"
    r"|^\s*(?:explain|elaborate)\s+(?:that|this|it|more)\b"
    r"|^\s*(?:and|but)\s+\w+\s*\??\s*$"
    r"|^\s*the\s+(?:previous|last|former|latter)\b",
    re.I,
)

#: Maximum prior turns consulted when resolving a reference. Bounded so a long
#: conversation cannot silently drag unrelated topics into a question.
MAX_HISTORY_TURNS = 3

#: Words too generic to carry a topic; excluded when borrowing terms.
_GENERIC = frozenset(
    {
        "what", "which", "when", "where", "about", "there", "their", "these",
        "those", "this", "that", "them", "then", "than", "with", "from", "have",
        "does", "doing", "case", "thing", "stuff", "more", "tell", "explain",
        "previous", "last", "former", "latter", "again", "same", "other",
    }
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _content_terms(text: str) -> list[str]:
    """Topic-bearing terms, order preserved, de-duplicated."""
    seen: dict[str, None] = {}
    for token in _WORD_RE.findall((text or "").lower()):
        if len(token) < 3 or token in _GENERIC:
            continue
        seen.setdefault(token, None)
    return list(seen)


def is_reference_turn(message: str) -> bool:
    """HEURISTIC: does this turn point at earlier conversation content?"""
    return bool(REFERENCE_RE.search(message or ""))


class ConversationNotFound(RuntimeError):
    """Requested conversation does not exist for this knowledge base (404)."""


class GroundedChatService:
    """Chat orchestration: conversation memory + answering + observability."""

    def __init__(
        self,
        repo: Repository,
        settings: Settings | None = None,
        *,
        answering: AnsweringService | None = None,
    ) -> None:
        self._repo = repo
        self._settings = settings or get_settings()
        self._answering = answering or AnsweringService(repo, self._settings)

    # -- conversations -------------------------------------------------------

    def create_conversation(self, kb_id: str, title: str = "") -> Conversation:
        if self._repo.get_kb(kb_id) is None:
            from app.services.retrieval.service import KnowledgeBaseNotFound

            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")
        conversation = Conversation(
            id=new_id("conv"),
            kb_id=kb_id,
            title=(title or "").strip()[:200],
        )
        self._repo.create_conversation(conversation)
        return conversation

    def list_conversations(self, kb_id: str, limit: int = 50) -> list[ConversationSummary]:
        return [
            ConversationSummary(
                id=c.id,
                kb_id=c.kb_id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
                message_count=c.message_count,
            )
            for c in self._repo.list_conversations(kb_id, limit=limit)
        ]

    def get_conversation(self, kb_id: str, conversation_id: str) -> ConversationDetail | None:
        conversation = self._repo.get_conversation(kb_id, conversation_id)
        if conversation is None:
            return None
        messages = self._repo.list_messages(conversation_id)
        return ConversationDetail(conversation=conversation, messages=messages)

    def delete_conversation(self, kb_id: str, conversation_id: str) -> bool:
        if self._repo.get_conversation(kb_id, conversation_id) is None:
            return False
        self._repo.delete_conversation(kb_id, conversation_id)
        return True

    # -- reference resolution ------------------------------------------------

    def resolve_reference(
        self, kb_id: str, conversation_id: str, message: str
    ) -> tuple[str, str | None]:
        """Return (question_to_answer, prior_message_id_used_or_None).

        Expansion happens ONLY for a detected reference turn AND only when a
        prior USER message exists to borrow from. Otherwise the message is
        returned byte-for-byte unchanged, so the common case is provably
        untransformed.

        The borrowed terms are appended as context; the original wording is
        preserved verbatim at the front of the resulting question.
        """
        if not is_reference_turn(message):
            return message, None

        prior_users = [
            m
            for m in self._repo.list_messages(conversation_id)
            if m.role is MessageRole.USER and m.content.strip()
        ]
        if not prior_users:
            return message, None

        source = prior_users[-1]
        terms = _content_terms(source.content)[:8]
        if not terms:
            return message, None

        # Borrow at most MAX_HISTORY_TURNS user turns' terms for a reference
        # that spans more than the immediately preceding turn.
        for extra in reversed(prior_users[-(MAX_HISTORY_TURNS + 1) : -1]):
            for term in _content_terms(extra.content):
                if term not in terms:
                    terms.append(term)
                if len(terms) >= 12:
                    break
            if len(terms) >= 12:
                break

        expanded = f"{message.strip()} [conversation context: {' '.join(terms)}]"
        return expanded, source.id

    # -- the turn ------------------------------------------------------------

    def chat(self, kb_id: str, payload: ChatRequest) -> ChatResponse:
        """Run one grounded chat turn end to end.

        Order matters and is deliberate:

        1. resolve the conversation (creating one when the caller did not supply
           an id) — so history is attached to the right thread;
        2. resolve conversational references, recording what was borrowed;
        3. store the USER message BEFORE answering, so a failure mid-answer still
           leaves an auditable record of what was asked;
        4. delegate to the answering pipeline (one query-processing pass, shared
           through `query_trace`);
        5. store the ASSISTANT message plus the AnswerRun record.
        """
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            from app.services.retrieval.service import KnowledgeBaseNotFound

            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")

        if payload.conversation_id:
            conversation = self._repo.get_conversation(kb_id, payload.conversation_id)
            if conversation is None:
                raise ConversationNotFound(
                    f"Conversation {payload.conversation_id!r} not found in knowledge base {kb_id!r}"
                )
        else:
            conversation = self.create_conversation(
                kb_id, title=(payload.message or "").strip()[:80]
            )

        started = time.perf_counter()

        # -- reference resolution ------------------------------------------
        question, resolved_from = self.resolve_reference(
            kb_id, conversation.id, payload.message
        )

        user_message = Message(
            id=new_id("msg"),
            conversation_id=conversation.id,
            kb_id=kb_id,
            role=MessageRole.USER,
            content=payload.message,
            resolved_from=resolved_from,
        )
        self._store_message(user_message)

        # -- answering (the real pipeline) ---------------------------------
        answer, trace, assessment = self._answering.answer(
            kb_id,
            question,
            mode=payload.answer_mode,
            strategy=payload.retrieval_strategy,
            overrides=dict(payload.retrieval_params or {}),
            policy=AnswerPolicy(mode=AnswerMode(_mode_key(payload.answer_mode))),
        )
        total_ms = round((time.perf_counter() - started) * 1000.0, 3)

        query_trace = trace.query_trace or _fallback_query_trace(payload.message, question)

        # -- assistant message + run record --------------------------------
        assistant_message = Message(
            id=new_id("msg"),
            conversation_id=conversation.id,
            kb_id=kb_id,
            role=MessageRole.ASSISTANT,
            content=answer.text,
            answer_id=answer.answer_id,
            answer_trace_id=trace.id,
            retrieval_run_id=answer.retrieval_run_id,
            grounding_state=assessment.grounding_state,
            citation_count=len(answer.citations),
        )
        run = self._build_run(
            kb_id=kb_id,
            conversation_id=conversation.id,
            user_message=user_message,
            assistant_message=assistant_message,
            answer=answer,
            trace=trace,
            assessment=assessment,
            query_trace=query_trace,
            strategy=payload.retrieval_strategy or "",
            retrieval_params=dict(payload.retrieval_params or {}),
            total_ms=total_ms,
        )
        assistant_message.answer_run_id = run.id

        self._store_message(assistant_message)
        self._update_conversation(conversation, assistant_message)

        run.message_id = user_message.id
        self._store_run(run)

        return ChatResponse(
            answer=answer.text,
            answer_id=answer.answer_id,
            status=answer.status,
            citations=answer.citations,
            claims=answer.claims,
            grounding=_grounding_payload(assessment),
            evidence=trace.evidence_items,
            retrieval_run_id=answer.retrieval_run_id,
            answer_run_id=run.id,
            answer_trace_id=trace.id,
            query_trace=query_trace,
            conversation_id=conversation.id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            generation={
                "generated_by": answer.generated_by,
                "model": answer.model,
                "is_mock": answer.is_mock,
                "prompt_version": answer.prompt_version,
                "answerer_version": ANSWERER_VERSION,
                "answer_mode": answer.answer_mode.value,
                "degraded": bool(getattr(self._answering._get_generator(), "unavailable", False)),
                "generator_status": self._generator_status(),
            },
            warnings=list(answer.warnings) + list(query_trace.warnings),
        )

    # -- read APIs -----------------------------------------------------------

    def get_run(self, kb_id: str, run_id: str) -> AnswerRun | None:
        return self._repo.get_answer_run(kb_id, run_id)

    def list_runs(self, kb_id: str, limit: int = 50) -> list[AnswerRun]:
        return self._repo.list_answer_runs(kb_id, limit=limit)

    # -- helpers -------------------------------------------------------------

    def _generator_status(self) -> str:
        """One honest word for how the answer was produced."""
        generator = self._answering._get_generator()
        if getattr(generator, "unavailable", False):
            return "fallback"
        if generator.is_mock:
            return "mock"
        return "live"

    def _build_run(
        self,
        *,
        kb_id: str,
        conversation_id: str,
        user_message: Message,
        assistant_message: Message,
        answer: Answer,
        trace: AnswerTrace,
        assessment: EvidenceAssessment,
        query_trace: QueryTrace,
        strategy: str,
        retrieval_params: dict,
        total_ms: float,
    ) -> AnswerRun:
        stages = {s.name: s for s in trace.stages}
        generator = self._answering._get_generator()

        def stage_ms(name: str) -> float | None:
            stage = stages.get(name)
            return stage.ms if stage is not None else None

        unsupported = sum(
            1 for c in answer.claims if c.support_status.value == "unsupported"
        )
        return AnswerRun(
            id=new_id("arun"),
            kb_id=kb_id,
            question=user_message.content,
            conversation_id=conversation_id,
            message_id=user_message.id,
            answer_id=answer.answer_id,
            answer_trace_id=trace.id,
            retrieval_run_id=answer.retrieval_run_id,
            strategy=strategy or trace.retrieval_strategy,
            retrieval_params=retrieval_params,
            selected_evidence_ids=list(trace.evidence_ids),
            evidence_count=assessment.evidence_count,
            document_count=assessment.document_count,
            grounding_decision=assessment.decision.value,
            grounding_state=assessment.grounding_state,
            grounding_reasons=[assessment.reason_code, assessment.reason],
            grounding_reason_code=assessment.reason_code,
            grounding_sufficient=assessment.sufficient,
            citation_count=len(answer.citations),
            claim_count=len(answer.claims),
            unsupported_claim_count=unsupported,
            query_processing_ms=stage_ms("query_processing"),
            retrieval_ms=stage_ms("retrieval"),
            evidence_selection_ms=stage_ms("evidence_assembly"),
            grounding_ms=stage_ms("evidence_gate"),
            generation_ms=stage_ms("generation"),
            citation_validation_ms=stage_ms("citation_validation"),
            total_ms=total_ms,
            model=answer.model,
            provider=answer.generated_by,
            is_mock=answer.is_mock,
            prompt_version=answer.prompt_version,
            answerer_version=ANSWERER_VERSION,
            query_processor=query_trace.processor,
            query_processor_version=query_trace.processor_version,
            evidence_selector=self._answering._selector.name,
            grounding_gate=self._answering._gate.name,
            answer_status=answer.status,
            warnings=list(answer.warnings),
            notes=[
                "conversation history was used for reference resolution only; it "
                "was never used as knowledge",
            ]
            if user_message.resolved_from
            else [],
        )

    def _store_message(self, message: Message) -> None:
        try:
            self._repo.create_message(message)
        except Exception as exc:  # chat must not fail because history could not persist
            logger.warning("Could not persist message for %s: %s", message.kb_id, exc)

    def _store_run(self, run: AnswerRun) -> None:
        try:
            self._repo.create_answer_run(run)
        except Exception as exc:
            logger.warning("Could not persist answer run for %s: %s", run.kb_id, exc)

    def _update_conversation(self, conversation: Conversation, assistant: Message) -> None:
        from datetime import datetime, timezone

        conversation.message_count = len(self._repo.list_messages(conversation.id))
        conversation.updated_at = datetime.now(timezone.utc)
        if not conversation.title:
            conversation.title = assistant.content.strip()[:80]
        try:
            self._repo.update_conversation(conversation)
        except Exception as exc:
            logger.warning("Could not update conversation %s: %s", conversation.id, exc)


def _mode_key(mode: str | None) -> str:
    key = (mode or AnswerMode.ABSTAIN_IF_UNSUPPORTED.value).strip().lower()
    if key not in {m.value for m in AnswerMode}:
        raise AnswerConfigError(
            f"Unknown answer mode {mode!r}. Supported: {sorted(m.value for m in AnswerMode)}"
        )
    return key


def _fallback_query_trace(original: str, question: str) -> QueryTrace:
    """Trace used only if the pipeline returned no trace (should not happen).

    States plainly that query processing was not observed rather than
    fabricating a classification.
    """
    return QueryTrace(
        original_query=original,
        normalized_query=question,
        rewritten_query=None,
        subqueries=[],
        processor="unavailable",
        processor_version="",
        nature=QueryNature.KNOWLEDGE,
        classification_method="not_observed",
        warnings=[
            "query processing trace was not returned by the answering pipeline; "
            "the query that was searched is shown in normalized_query"
        ],
        notes=["query trace unavailable"],
        enabled=False,
    )


def _grounding_payload(assessment: EvidenceAssessment) -> dict:
    """The `grounding` object the chat response exposes.

    Contains the state, the decision, the machine-readable reason code, the
    human reason, the measured counts and the per-signal measurements. It does
    NOT contain a blended confidence percentage, because there is no validated
    statistical basis for one.
    """
    return {
        "state": assessment.grounding_state.value,
        "decision": assessment.decision.value,
        "sufficient": assessment.sufficient,
        "confidence": assessment.confidence.value,
        "reason_code": assessment.reason_code,
        "reason": assessment.reason,
        "evidence_count": assessment.evidence_count,
        "document_count": assessment.document_count,
        "supporting_evidence_ids": list(assessment.supporting_evidence_ids),
        "unsupported_aspects": list(assessment.unsupported_aspects),
        "missing_information": list(assessment.missing_information),
        "recommended_action": assessment.recommended_action,
        "signals": [s.model_dump(mode="json") for s in assessment.signals],
        "notes": list(assessment.notes),
    }


__all__ = [
    "ANSWERER_VERSION",
    "ConversationNotFound",
    "GroundedChatService",
    "is_reference_turn",
]
