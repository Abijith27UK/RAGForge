"""V7 grounded-answer schemas (query plan, evidence, gate, answer, trace).

Design rules (AGENTS.md + V7 spec):
* Every stage of the answering pipeline has an explicit, validated shape so it
  can be inspected in the UI and reproduced from the stored trace.
* Heuristic results are LABELLED as heuristic; missing knowledge is `None` or an
  explicit status — never a fabricated confidence percentage.
* Nothing here imports from `app.schemas.models` at module level beyond what is
  strictly needed, and nothing imports this module in a cycle (same rule as
  `schemas/retrieval.py`: `models.py` may import from here).
* No secrets, API keys or environment values may ever be stored in these models.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Query processing (Phase 9)
# ---------------------------------------------------------------------------


class QueryType(str, Enum):
    """Heuristic query classification. The label describes the RULE that fired,
    not a statistically validated classifier."""

    FACTUAL = "factual"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    NUMERICAL = "numerical"
    MULTI_PART = "multi-part"
    EXPLANATION = "explanation"
    TROUBLESHOOTING = "troubleshooting"
    OTHER = "other"


class QueryPlan(BaseModel):
    """Structured, deterministic plan for one question.

    `original_query` is preserved byte-for-byte; everything else is derived.
    """

    original_query: str
    normalized_query: str
    detected_language: str | None = Field(
        default=None, description="Heuristic guess; None when not guessable"
    )
    query_type: QueryType = QueryType.OTHER
    classification_method: str = Field(
        default="heuristic_rules",
        description="Honest label: rule-based heuristic, NOT a trained classifier",
    )
    classification_signals: list[str] = Field(
        default_factory=list,
        description="Which rule(s) produced the classification (inspectable)",
    )
    extracted_terms: list[str] = Field(
        default_factory=list, description="Content terms from the question (stop-words removed)"
    )
    domain_terms: list[str] = Field(
        default_factory=list,
        description="Question terms matching the KB's domain specification",
    )
    retrieval_queries: list[str] = Field(
        default_factory=list,
        description="Query text(s) sent to retrieval; today always the original question",
    )
    filters: dict[str, Any] = Field(default_factory=dict)
    requested_answer_format: str = Field(
        default="standard", description="Caller-requested format; 'standard' in V7"
    )
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class QueryNature(str, Enum):
    """Coarse, deterministic classification of what KIND of question this is.

    This is deliberately separate from `QueryType` (which describes the answer
    shape). A conversational greeting is `NON_KNOWLEDGE` regardless of how it
    would otherwise be classified, and the gate uses that to avoid running
    retrieval for it.

    Every value is produced by an explicit, inspectable rule; none of them
    implies a trained classifier.
    """

    KNOWLEDGE = "knowledge"
    CONVERSATIONAL = "conversational"
    NON_KNOWLEDGE = "non_knowledge"
    UNDERSPECIFIED = "underspecified"
    MULTI_HOP = "multi_hop"


class QueryTrace(BaseModel):
    """Full, inspectable record of query processing for one question.

    Exists so that a caller can always answer: *what exactly was searched for,
    and did the system change my question?* Rules:

    * `original_query` is byte-for-byte what the caller sent.
    * `rewritten_query` is `None` (not an empty string) whenever no rewrite
      happened, so "we did not rewrite" and "we rewrote to nothing" can never
      be confused.
    * `subqueries` is empty unless decomposition actually ran; the reason it did
      not run is recorded in `notes`.
    * `timing_ms` is measured, never estimated.
    """

    original_query: str
    normalized_query: str
    rewritten_query: str | None = Field(
        default=None,
        description=(
            "The query actually sent to retrieval when it differs from the "
            "normalized question; None means the original passed through unchanged"
        ),
    )
    subqueries: list[str] = Field(
        default_factory=list,
        description="Additional retrieval queries produced by decomposition (empty when not run)",
    )
    processor: str = Field(default="heuristic-rules", description="Which QueryProcessor produced this")
    processor_version: str = Field(
        default="v7.2",
        description="Processor revision; bump when rule behaviour changes so stored traces stay interpretable",
    )
    query_type: QueryType = QueryType.OTHER
    nature: QueryNature = QueryNature.KNOWLEDGE
    classification_method: str = "heuristic_rules"
    classification_signals: list[str] = Field(default_factory=list)
    transformations: list[str] = Field(
        default_factory=list,
        description="Ordered list of transformations actually applied (empty = question passed through unchanged)",
    )
    expanded_terms: list[str] = Field(
        default_factory=list,
        description="Terms added by optional query expansion; empty when expansion did not run",
    )
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    timing_ms: float | None = Field(
        default=None, description="Measured query-processing latency; None when not measured"
    )
    enabled: bool = True
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Evidence (Phase 10)
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """One retrieved item that MAY be used to ground an answer.

    Built directly from a `RetrievalResult` — every provenance field the chunk
    carries is copied, never re-derived or re-labelled. An absent page number
    stays absent; a missing publisher stays missing.
    """

    evidence_id: str = Field(description="Unique within one answer; e.g. ev_0001")
    chunk_id: str
    document_id: str
    source_id: str | None = None
    kb_id: str = ""
    title: str = ""  # document title, falling back to source title
    source_type: str | None = Field(
        default=None, description="Source type recorded at ingestion; None when the source never recorded one"
    )
    content: str
    retrieval_score: float
    retrieval_strategy: str = ""
    rank: int  # final evidence rank (1-based)
    original_rank: int  # rank as returned by retrieval, before dedup
    page: int | None = None
    slide: int | None = None
    section: str | None = None
    section_path: str | None = None
    url: str | None = None
    publisher: str | None = None
    document_version: int | None = None
    content_hash: str = ""
    trust_score: float | None = Field(
        default=None, description="Source-quality trust score if the KB records one; None otherwise"
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="Full unmodified provenance dict from retrieval"
    )
    retrieval_run_id: str | None = None
    dedup_reason: str = Field(
        default="", description="Why an earlier duplicate of this item was dropped, if any"
    )
    overlap_fraction: float | None = Field(
        default=None, description="Measured text overlap with an earlier item when dedup applied"
    )


class EvidenceSet(BaseModel):
    """Ordered evidence for one answer, plus the dedup record."""

    items: list[Evidence] = Field(default_factory=list)
    dropped: list[Evidence] = Field(
        default_factory=list,
        description="Duplicates removed by dedup; retained for audit (not given to the generator)",
    )
    notes: list[str] = Field(default_factory=list)
    retrieval_run_id: str | None = None
    strategy: str = ""


# ---------------------------------------------------------------------------
# Evidence gate (Phase 11)
# ---------------------------------------------------------------------------


class GateDecision(str, Enum):
    """What the pipeline should do with this question + evidence."""

    ANSWER = "ANSWER"
    PARTIAL_ANSWER = "PARTIAL_ANSWER"
    ABSTAIN = "ABSTAIN"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"


class GroundingState(str, Enum):
    """The five externally-visible grounding outcomes (V7 spec Phase 12).

    This is the vocabulary the API and UI use. It maps ONTO `GateDecision`
    (which additionally distinguishes a request for clarification); the mapping
    is recorded in `EvidenceAssessment.grounding_state` so the two can never
    drift apart silently.

    There is no numeric confidence here on purpose: a single blended score
    would hide which signal failed.
    """

    ANSWERED = "ANSWERED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"


#: GateDecision -> GroundingState. Exhaustive; a new GateDecision without a
#: mapping is a programming error, not a runtime fallback.
GATE_TO_GROUNDING: dict[GateDecision, GroundingState] = {
    GateDecision.ANSWER: GroundingState.ANSWERED,
    GateDecision.PARTIAL_ANSWER: GroundingState.PARTIALLY_SUPPORTED,
    GateDecision.ABSTAIN: GroundingState.INSUFFICIENT_EVIDENCE,
    GateDecision.ASK_CLARIFICATION: GroundingState.INSUFFICIENT_EVIDENCE,
}


class ConfidenceCategory(str, Enum):
    """Categorical confidence — deliberately NOT a percentage. There is no
    validated statistical basis for a numeric grounding confidence in V7."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "none"


class GateSignal(BaseModel):
    """One measured input to the gate decision. `value` is a real measurement;
    `interpretation` is human-readable. Signals with no basis are recorded with
    `measured = False` and a `not_performed_reason` — never a made-up number."""

    name: str
    measured: bool = True
    value: float | None = None
    threshold: float | None = None
    passed: bool | None = None
    interpretation: str = ""
    not_performed_reason: str = ""


class EvidenceAssessment(BaseModel):
    """Result of the sufficiency / grounding gate (Phase 11)."""

    sufficient: bool
    decision: GateDecision
    grounding_state: GroundingState = Field(
        default=GroundingState.INSUFFICIENT_EVIDENCE,
        description="Externally-visible grounding outcome derived from `decision`",
    )
    confidence: ConfidenceCategory
    reason_code: str = Field(
        description="Machine-readable code, e.g. NO_EVIDENCE | LOW_ALIGNMENT | OK"
    )
    reason: str = Field(description="User-readable explanation of the decision")
    evidence_count: int = 0
    document_count: int = 0
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    unsupported_aspects: list[str] = Field(
        default_factory=list,
        description="Parts of the question the evidence does not cover",
    )
    missing_information: list[str] = Field(default_factory=list)
    recommended_action: str = ""
    signals: list[GateSignal] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Answer modes, claims, citations, answers (Phases 12, 13)
# ---------------------------------------------------------------------------


class AnswerMode(str, Enum):
    """V7 supports exactly two modes. Concise/detailed/exam/teaching/engineering
    are future extensions — NOT implemented, so not present as enum values."""

    GROUNDED = "grounded"
    ABSTAIN_IF_UNSUPPORTED = "abstain_if_unsupported"


class AnswerStatus(str, Enum):
    GROUNDED = "grounded"
    PARTIAL = "partial"
    ABSTAINED = "abstained"
    CLARIFICATION_REQUIRED = "clarification_required"
    GENERATION_FAILED = "generation_failed"


class ClaimType(str, Enum):
    FACT = "fact"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    NUMERICAL = "numerical"
    INFERENCE = "inference"
    MISSING_INFORMATION = "missing_information"
    ABSTENTION = "abstention"


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class Claim(BaseModel):
    """One atomic assertion in the answer, with its full evidence chain."""

    claim_id: str
    text: str
    claim_type: ClaimType = ClaimType.FACT
    citation_ids: list[str] = Field(
        default_factory=list, description="Citation ids supporting this claim"
    )
    evidence_ids: list[str] = Field(
        default_factory=list, description="Evidence ids this claim is grounded in"
    )
    support_status: SupportStatus = SupportStatus.UNSUPPORTED
    support_check: str = Field(
        default="not_performed",
        description=(
            "How support was checked: 'lexical_overlap' (HEURISTIC, deterministic) | "
            "'not_performed'. Semantic entailment is NOT IMPLEMENTED in V7 and must "
            "never be implied."
        ),
    )
    support_note: str = ""


class Citation(BaseModel):
    """A validated pointer from a claim to evidence and its provenance.

    Every provenance field is COPIED from the evidence item that was actually
    retrieved. A missing page/slide/section stays None — it is never filled in
    with a plausible-looking guess.
    """

    citation_id: str
    evidence_id: str
    chunk_id: str
    document_id: str
    source_title: str = ""
    source_type: str | None = None
    title: str = ""
    page: int | None = None
    page_number: int | None = Field(
        default=None, description="Same measurement as `page`, under the name the chat API exposes"
    )
    slide: int | None = None
    slide_number: int | None = Field(
        default=None, description="Same measurement as `slide`, under the name the chat API exposes"
    )
    section: str | None = None
    section_path: str | None = None
    content_hash: str = ""
    url: str | None = None
    publisher: str | None = None
    snippet: str = Field(default="", description="Short excerpt of the evidence actually used")
    validation: str = Field(
        default="pending",
        description=(
            "provenance_valid | invalid | pending. 'provenance_valid' means the id "
            "resolves to retrieved evidence with a complete provenance chain — it is "
            "NOT a claim that the citation is semantically correct."
        ),
    )
    validation_detail: str = ""


class CalculationStep(BaseModel):
    """A single deterministic arithmetic operation, recorded verbatim."""

    step_id: str
    operation: str  # add | subtract | multiply | divide
    inputs: dict[str, float] = Field(default_factory=dict)
    formula: str = ""
    result: float
    unit: str | None = None


class CalculationTrace(BaseModel):
    """Record of any arithmetic performed. Empty + performed=False when the
    answer contains numbers only as quoted values (the V7 default: numbers are
    PRESERVED, never silently recomputed)."""

    performed: bool = False
    steps: list[CalculationStep] = Field(default_factory=list)
    note: str = "No calculation performed; numerical values are quoted from evidence."


class ClaimDraft(BaseModel):
    """Generator output for one claim, BEFORE validation. `citation_evidence_ids`
    may contain ids the generator was NOT given — validation is what catches
    fabrication, so the raw draft must be preservable."""

    text: str
    claim_type: ClaimType = ClaimType.FACT
    citation_evidence_ids: list[str] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    """Raw generator output — never returned to the client unvalidated."""

    text: str = ""
    claims: list[ClaimDraft] = Field(default_factory=list)
    abstain_requested: bool = Field(
        default=False, description="Generator explicitly declined to answer"
    )
    abstention_reason: str = ""
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnswerPolicy(BaseModel):
    """Explicit rules the generator and validator must follow."""

    mode: AnswerMode = AnswerMode.ABSTAIN_IF_UNSUPPORTED
    allow_partial: bool = Field(
        default=True,
        description="Produce a partial answer (with missing parts stated) when the gate says PARTIAL_ANSWER",
    )
    unsupported_claim_action: str = Field(
        default="remove",
        description="remove | downgrade | abstain — applied by the validation pipeline",
    )
    regenerate_attempts: int = Field(
        default=0, ge=0, le=2, description="How many regeneration attempts are allowed on failure"
    )
    max_evidence_chars: int = Field(
        default=12000, ge=500, le=200000, description="Evidence context budget for the prompt"
    )
    prompt_version: str = "v7.1"


class Answer(BaseModel):
    """FINAL, validated answer. This is what the API returns and what the UI
    renders; the raw GeneratedAnswer lives only in the trace."""

    answer_id: str
    kb_id: str = ""
    question: str
    status: AnswerStatus
    text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: ConfidenceCategory = ConfidenceCategory.NONE
    confidence_basis: str = Field(
        default="", description="Human-readable basis for the categorical confidence"
    )
    generated_by: str = ""
    model: str = ""
    is_mock: bool = False
    prompt_version: str = ""
    answer_mode: AnswerMode = AnswerMode.ABSTAIN_IF_UNSUPPORTED
    retrieval_run_id: str | None = None
    answer_trace_id: str = ""
    assessment: EvidenceAssessment | None = None
    generation_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    calculation: CalculationTrace = Field(default_factory=CalculationTrace)
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Answer trace (audit trail)
# ---------------------------------------------------------------------------


class AnswerTraceStage(BaseModel):
    """One stage of the answering pipeline, with its REAL status.

    Reuses the honest status vocabulary of retrieval stages: a stage that did
    not run is 'skipped', never 'ok'.
    """

    name: str
    status: str = "ok"  # ok | skipped | unavailable | error
    detail: str = ""
    count: int | None = None
    ms: float | None = None


class AnswerTrace(BaseModel):
    """Complete, replayable audit trail for one answer.

    Contains no secrets: provider names/models are recorded, never keys,
    environment values or internal filesystem paths.
    """

    id: str
    kb_id: str
    answer_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    question: str
    answer_mode: AnswerMode = AnswerMode.ABSTAIN_IF_UNSUPPORTED
    # Stage 1: query processing
    query_plan: QueryPlan | None = None
    query_trace: QueryTrace | None = Field(
        default=None,
        description=(
            "What query processing actually did (rewrite/expansion/decomposition "
            "record, or explicit evidence that the question passed through unchanged)"
        ),
    )
    # Stage 2-3: retrieval + evidence
    retrieval_strategy: str = ""
    retrieval_run_id: str | None = None
    retrieval_params: dict[str, Any] | None = None
    retrieval_stages: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_items: list[Evidence] = Field(
        default_factory=list,
        description="Full evidence objects (deduplicated) so the trace can be replayed "
        "and the UI can render source cards without a second retrieval",
    )
    evidence_dropped: list[dict[str, Any]] = Field(default_factory=list)
    evidence_notes: list[str] = Field(default_factory=list)
    # Stage 4: gate
    assessment: EvidenceAssessment | None = None
    # Stage 5: generation
    generator: str = ""
    generator_model: str = ""
    is_mock: bool = False
    generation_notes: list[str] = Field(default_factory=list)
    generation_warnings: list[str] = Field(default_factory=list)
    raw_generated_text: str = Field(
        default="",
        description="Raw generator output BEFORE validation (so repairs are auditable)",
    )
    # Stage 6: validation
    validation_actions: list[str] = Field(
        default_factory=list, description="Recorded per-problem actions (remove/downgrade/abstain)"
    )
    citation_problems: list[dict[str, Any]] = Field(default_factory=list)
    # Final
    status: AnswerStatus | None = None
    stages: list[AnswerTraceStage] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    total_ms: float | None = None


# ---------------------------------------------------------------------------
# Answer run (observability record — Phase 13)
# ---------------------------------------------------------------------------


class AnswerRun(BaseModel):
    """Compact, immutable observability record for one answering execution.

    Deliberately smaller than `AnswerTrace`: this is the row you list, sort and
    aggregate. The full evidence text and raw generator output stay in the
    trace, reachable through `answer_trace_id`.

    Every latency field is a MEASURED duration in milliseconds or None. None
    means "not measured", never zero.
    """

    id: str
    kb_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    question: str
    conversation_id: str | None = None
    message_id: str | None = None
    answer_id: str = ""
    answer_trace_id: str = ""
    retrieval_run_id: str | None = None
    strategy: str = ""
    retrieval_params: dict[str, Any] = Field(default_factory=dict)
    # -- evidence -----------------------------------------------------------
    selected_evidence_ids: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    document_count: int = 0
    # -- grounding ----------------------------------------------------------
    grounding_decision: str = ""
    grounding_state: GroundingState = GroundingState.INSUFFICIENT_EVIDENCE
    grounding_reasons: list[str] = Field(
        default_factory=list,
        description="Machine-readable reason codes plus the human explanation, in that order",
    )
    grounding_reason_code: str = ""
    grounding_sufficient: bool = False
    # -- citations ----------------------------------------------------------
    citation_count: int = 0
    claim_count: int = 0
    unsupported_claim_count: int = 0
    # -- latencies (measured or None) ---------------------------------------
    query_processing_ms: float | None = None
    retrieval_ms: float | None = None
    evidence_selection_ms: float | None = None
    grounding_ms: float | None = None
    generation_ms: float | None = None
    citation_validation_ms: float | None = None
    total_ms: float | None = None
    # -- versions -----------------------------------------------------------
    model: str = ""
    provider: str = ""
    is_mock: bool = False
    prompt_version: str = ""
    answerer_version: str = ""
    query_processor: str = ""
    query_processor_version: str = ""
    evidence_selector: str = ""
    grounding_gate: str = ""
    answer_status: AnswerStatus | None = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Conversation memory (Phase 13 — minimal, deliberately NOT knowledge)
# ---------------------------------------------------------------------------


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """One turn in a conversation.

    CRITICAL: a message is NOT a knowledge source. It is stored separately from
    documents/chunks and is never embedded, indexed or retrieved from. Its only
    role is resolving references like "what about the previous case?" into a
    standalone question — the answer is still grounded in retrieved KB evidence
    or it is not produced at all.
    """

    id: str
    conversation_id: str
    kb_id: str
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=_utcnow)
    # -- assistant-only links (None on user messages) ------------------------
    answer_id: str | None = None
    answer_trace_id: str | None = None
    answer_run_id: str | None = None
    retrieval_run_id: str | None = None
    grounding_state: GroundingState | None = None
    citation_count: int | None = None
    # -- reference resolution record ----------------------------------------
    resolved_from: str | None = Field(
        default=None,
        description=(
            "The prior user message this turn's question was expanded from, when "
            "reference resolution rewrote the question; None when the message was "
            "used as written"
        ),
    )
    used_as_knowledge: bool = Field(
        default=False,
        description=(
            "Always False. Present so the UI and any audit can assert that "
            "conversation history was never promoted to knowledge."
        ),
    )


class Conversation(BaseModel):
    """A chat thread scoped to exactly one knowledge base.

    A conversation never crosses KBs: changing the knowledge base means a new
    conversation, so grounding can never be attributed to the wrong corpus.
    """

    id: str
    kb_id: str
    title: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    message_count: int = 0
    retrieval_strategy: str | None = None
    notes: list[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    """Compact listing for the conversation sidebar (no message bodies)."""

    id: str
    kb_id: str
    title: str = ""
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationDetail(BaseModel):
    """A conversation plus its messages, oldest first."""

    conversation: Conversation
    messages: list[Message] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Grounded chat (Phase 13)
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """`POST /knowledge-bases/{kb_id}/chat` body.

    `retrieval_strategy` and `retrieval_params` are optional; when omitted the
    KB's persisted retrieval configuration applies, exactly as in `/answer`.
    """

    message: str = Field(min_length=1, max_length=4000)
    retrieval_strategy: str | None = Field(
        default=None, description="dense | bm25 | hybrid | hybrid_reranked"
    )
    retrieval_params: dict[str, Any] | None = None
    conversation_id: str | None = None
    answer_mode: str = Field(
        default=AnswerMode.ABSTAIN_IF_UNSUPPORTED.value,
        description="grounded | abstain_if_unsupported",
    )


class ChatResponse(BaseModel):
    """Full, provenance-preserving chat response.

    Nothing about the retrieval or grounding process is hidden: the caller gets
    the evidence that was used, the reason the gate decided what it decided, and
    the query trace showing whether the question was transformed.
    """

    answer: str
    answer_id: str
    status: AnswerStatus
    citations: list[Citation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    grounding: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    retrieval_run_id: str | None = None
    answer_run_id: str = ""
    answer_trace_id: str = ""
    query_trace: QueryTrace
    conversation_id: str = ""
    user_message_id: str = ""
    assistant_message_id: str = ""
    generation: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)



