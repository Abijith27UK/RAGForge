"""Pydantic schemas: all RAGForge domain models.

Relationships:
    KnowledgeBase -> DomainSpec
    KnowledgeBase -> Sources -> Documents -> Chunks -> Qdrant vectors
    KnowledgeBase -> EvaluationQuestions -> EvaluationRuns
    KnowledgeBase -> BuildRuns
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class KBStatus(str, enum.Enum):
    DRAFT = "draft"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    BUILDING = "building"
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"


class SourceType(str, enum.Enum):
    WEB_PAGE = "web_page"
    PDF = "pdf"
    ARXIV = "arxiv"
    USER_UPLOAD = "user_upload"  # legacy alias kept for pre-V4 rows
    USER_PROVIDED = "user_provided"  # V4: a file/URL the user supplied
    PRESENTATION = "presentation"  # V4: ppt / pptx
    DOCUMENT = "document"  # V4: docx
    TEXT = "text"


class SourceMode(str, enum.Enum):
    """How a knowledge base obtains its knowledge (V4 Phase 1).

    A Knowledge Base is a PRODUCT artifact; an evaluation benchmark is a
    separate, OPTIONAL instrument. SourceMode therefore describes only where
    documents come from and never implies ground truth exists.

    EXTERNAL     - RAGForge discovers and scores authoritative public sources.
    USER_PROVIDED- The user uploads documents and/or supplies URLs.
    MIXED        - User documents plus externally discovered sources.
    """

    EXTERNAL = "external"
    USER_PROVIDED = "user_provided"
    MIXED = "mixed"


class DocumentStatus(str, enum.Enum):
    """Per-document lifecycle inside the document library (V4 Phase 5)."""

    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class SourceDecision(str, enum.Enum):
    PENDING = "PENDING"
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class QuestionStatus(str, enum.Enum):
    """Lifecycle of a benchmark question (V3 Phase A).

    DRAFT -> REVIEW -> APPROVED -> FROZEN. Editing rules:
    - DRAFT/REVIEW: content freely editable.
    - APPROVED: editing creates a new DRAFT revision (approved content preserved).
    - FROZEN: immutable; participates in frozen benchmark versions.
    """

    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    FROZEN = "FROZEN"


class GroundTruthStatus(str, enum.Enum):
    """Availability of human-reviewed ground truth for a knowledge base (V5).

    A DOMAIN DOES NOT AUTOMATICALLY HAVE GROUND TRUTH. Entering a domain name,
    analysing it, indexing a corpus, or reaching READY never produces ground
    truth. It only exists once humans author questions, review them, approve
    them and freeze them into a benchmark version.

    Until then it is ``NOT_AVAILABLE`` and no retrieval-quality metric may be
    reported for this knowledge base.
    """

    NOT_AVAILABLE = "NOT_AVAILABLE"
    DRAFT = "DRAFT"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    FROZEN = "FROZEN"


#: Sent with the API so the UI never has to invent this wording.
GROUND_TRUTH_UNAVAILABLE_EXPLANATION = (
    "This knowledge base has no human-reviewed ground truth, so no retrieval-quality "
    "metric (Recall@K, Precision@K, MRR, NDCG) can be reported for it. Corpus coverage "
    "and retrieval quality are different questions: the corpus may contain the "
    "information while nothing has verified that retrieval can find it. A domain name "
    "never produces ground truth — questions must be authored, reviewed and frozen."
)


class BenchmarkStatus(str, enum.Enum):
    """Status of a benchmark *version* (a set of questions)."""

    DRAFT = "DRAFT"
    REVIEWING = "REVIEWING"
    APPROVED = "APPROVED"
    FROZEN = "FROZEN"


class BuildStage(str, enum.Enum):
    DOMAIN_ANALYSIS = "domain_analysis"
    SOURCE_DISCOVERY = "source_discovery"
    INGESTION = "ingestion"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    EVALUATION = "evaluation"


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    domain: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=2000)
    target_audience: str = Field(min_length=1, max_length=500)
    depth: str = Field(default="intermediate", max_length=100)
    source_mode: SourceMode = Field(
        default=SourceMode.EXTERNAL,
        description=(
            "Where the knowledge comes from. EXTERNAL = discovered public sources; "
            "USER_PROVIDED = user uploads documents/URLs; MIXED = both. Ground truth "
            "is never required regardless of mode."
        ),
    )
    vector_backend: str = Field(
        default="qdrant", max_length=50,
        description="Vector store backend id ('qdrant'; experimental backends later)",
    )
    chunking_strategy: str = Field(
        default="section-aware", max_length=50,
        description="Chunking strategy: 'section-aware' | 'fixed-size'",
    )


class KnowledgeBase(BaseModel):
    id: str
    name: str
    domain: str
    purpose: str
    target_audience: str
    depth: str = "intermediate"
    status: KBStatus = KBStatus.DRAFT
    # --- V4: knowledge-source mode and build versioning ---
    source_mode: SourceMode = Field(
        default=SourceMode.EXTERNAL,
        description="How this KB obtains knowledge. Does NOT imply a benchmark exists.",
    )
    version: int = Field(
        default=1,
        description=(
            "Knowledge-base build version, incremented on every successful index. "
            "Historical experiment artifacts keep referencing their own kb_id/version."
        ),
    )
    last_build_at: datetime | None = Field(
        default=None, description="When the current index was last written"
    )
    # Identity of the embedding model the KB was indexed with (provider/model/
    # dimensions). Retrieval refuses to run against a mismatched model instead
    # of silently querying an incompatible vector space.
    embedding_identity: dict[str, Any] | None = None
    # --- V3: pluggable infrastructure identifiers ---
    vector_backend: str = Field(
        default="qdrant",
        description="Vector store backend id: 'qdrant' today; experimental backends later",
    )
    chunking_strategy: str = Field(
        default="section-aware",
        description="Chunking strategy name: 'section-aware' | 'fixed-size'",
    )
    chunking_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunking parameters (target_size, overlap) used for the last build",
    )
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class KBOverview(BaseModel):
    """Everything the KB overview screen needs, in one honest payload.

    A knowledge base is READY on indexing alone. `evaluation_status` reports
    "not_configured" when no benchmark exists, which is a VALID state — ground
    truth is optional by design (Benchmark = evaluation instrument,
    KnowledgeBase = product artifact).
    """

    kb: KnowledgeBase
    documents: int = 0
    documents_ready: int = 0
    documents_failed: int = 0
    user_provided_documents: int = 0
    external_documents: int = 0
    chunks: int = 0
    vectors: int | None = Field(
        default=None, description="Point count reported by the vector store; null when unreachable"
    )
    sources: int = 0
    sources_user_provided: int = 0
    sources_discovered: int = 0
    embedding_model: str = ""
    vector_backend: str = "qdrant"
    vector_store_status: str = "unknown"
    chunking_strategy: str = ""
    chunking_config: dict[str, Any] = Field(default_factory=dict)
    last_build_at: datetime | None = None
    last_build_status: str = ""
    version: int = 1
    build_status: str = "not_built"
    evaluation_status: str = Field(
        default="not_configured",
        description="not_configured | questions_only | benchmark_draft | benchmark_frozen | evaluated",
    )
    benchmark_versions: int = 0
    frozen_benchmark_versions: int = 0
    evaluation_questions: int = 0
    evaluation_runs: int = 0
    last_evaluation: dict[str, Any] | None = None
    evaluation_required: bool = Field(
        default=False,
        description="Always false: ground truth is optional and never blocks READY",
    )
    ground_truth_status: GroundTruthStatus = Field(
        default=GroundTruthStatus.NOT_AVAILABLE,
        description=(
            "Whether human-reviewed ground truth exists for this KB. NOT_AVAILABLE is "
            "the honest default for every new domain, including ones with large corpora."
        ),
    )
    ground_truth_explanation: str = Field(
        default=GROUND_TRUTH_UNAVAILABLE_EXPLANATION,
        description="Why the current ground-truth status is what it is",
    )
    ground_truth_question_count: int = 0


# ---------------------------------------------------------------------------
# Domain Analysis
# ---------------------------------------------------------------------------

class KnowledgeRequirement(BaseModel):
    area: str = Field(description="e.g. 'EV battery safety standards'")
    description: str = ""
    priority: str = Field(default="medium", description="high | medium | low")


class DomainSpec(BaseModel):
    kb_id: str
    domain: str
    description: str = ""
    subdomains: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    terminology: list[str] = Field(default_factory=list)
    knowledge_requirements: list[KnowledgeRequirement] = Field(default_factory=list)
    recommended_source_categories: list[str] = Field(default_factory=list)
    generated_by: str = Field(default="", description="LLM provider+model or 'mock'")
    is_mock: bool = Field(default=False, description="True if generated in dev/mock mode")
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class SourceCreate(BaseModel):
    url: str
    title: str | None = None
    source_type: SourceType = SourceType.WEB_PAGE
    publisher: str | None = None
    notes: str | None = None


class Source(BaseModel):
    id: str
    kb_id: str
    url: str
    title: str | None = None
    source_type: SourceType = SourceType.WEB_PAGE
    publisher: str | None = None
    discovered_via: str = Field(default="user", description="provider that found it")
    decision: SourceDecision = SourceDecision.PENDING
    trust_score: float | None = None
    quality: dict[str, Any] | None = Field(
        default=None, description="Full quality assessment: signals, reasons, warnings"
    )
    notes: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    # --- V4: user-provided provenance ---
    # A user-supplied file/URL must NOT be scored down for lacking public
    # authority signals (no website, no HTTP status, no publication date). The
    # user asserting relevance is recorded here and assessed by the separate
    # integrity model (app/services/source_quality/user_scorer.py).
    user_provided: bool = False
    provenance: str = Field(
        default="discovered",
        description="'discovered' (external discovery) | 'user_upload' (file) | 'user_url'",
    )
    file_name: str | None = None
    file_size: int | None = None
    integrity: dict[str, Any] | None = Field(
        default=None,
        description="User-file integrity assessment (validity, parser, duplication, extraction)",
    )


class SourceDecisionUpdate(BaseModel):
    decision: SourceDecision


class QualitySignals(BaseModel):
    # External-web signals (used by SourceQualityScorer).
    authority: float = Field(default=0.5, ge=0, le=1, description="Domain authority of publisher/URL")
    relevance: float = Field(default=0.5, ge=0, le=1, description="Match against domain spec")
    recency: float = Field(default=0.5, ge=0, le=1, description="Freshness of content")
    source_type: float = Field(default=0.5, ge=0, le=1, description="Trustworthiness of the type")
    accessibility: float = Field(default=0.5, ge=0, le=1, description="Fetchable / parseable")
    duplication: float = Field(default=1.0, ge=0, le=1, description="1 = unique, lower = duplicated")
    evidence_quality: float = Field(default=0.5, ge=0, le=1, description="Citations, standards refs, etc.")
    # --- V4 user-file integrity signals ---
    # Defaults are 1.0 ("not applicable, therefore not penalized") so a missing
    # signal can never lower a score. The user-file scorer sets these explicitly.
    file_validity: float = Field(default=1.0, ge=0, le=1, description="File is readable and not corrupt")
    content_extraction: float = Field(default=1.0, ge=0, le=1, description="Meaningful text was extracted")
    structure: float = Field(default=1.0, ge=0, le=1, description="Pages/slides/headings were recovered")
    user_relevance: float = Field(default=1.0, ge=0, le=1, description="Relevance asserted by the uploader")


class QualityAssessment(BaseModel):
    """Explainable source quality assessment. NOT an objective truth."""
    signals: QualitySignals
    weights: dict[str, float]
    score: float = Field(ge=0, le=1)
    decision: SourceDecision
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assessed_by: str = "heuristic-v1"


# ---------------------------------------------------------------------------
# Documents & Chunks
# ---------------------------------------------------------------------------

class Document(BaseModel):
    id: str
    kb_id: str
    source_id: str
    url: str
    title: str | None = None
    source_type: SourceType
    publisher: str | None = None
    file_path: str | None = None
    content_hash: str = ""
    text_length: int = 0
    page_count: int | None = None
    parse_error: str | None = None
    ingestion_timestamp: datetime = Field(default_factory=utcnow)
    # --- V4 document library ---
    status: DocumentStatus = Field(
        default=DocumentStatus.PARSED,
        description="uploaded | parsing | parsed | chunking | indexing | ready | failed",
    )
    user_provided: bool = False
    document_version: int = Field(
        default=1, description="Incremented when the file is replaced by a new version"
    )
    replaces_document_id: str | None = Field(
        default=None, description="Previous document version this one supersedes (kept in history)"
    )
    file_name: str | None = Field(default=None, description="Original file name as uploaded")
    file_size: int | None = Field(default=None, description="Original size in bytes")
    mime_type: str | None = None
    raw_file_path: str | None = Field(
        default=None,
        description="Path of the original binary/text file; file_path points at the parsed text",
    )
    parser: str | None = Field(default=None, description="Parser that produced the normalized text")
    slide_count: int | None = Field(default=None, description="PPTX slides (never inferred)")
    section_count: int | None = Field(default=None, description="DOCX heading sections (never inferred)")
    chunk_count: int = Field(default=0, description="Chunks currently indexed for this document")
    parse_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Original parser metadata (never discarded)"
    )
    indexed_at: datetime | None = None


class DocumentDetail(BaseModel):
    """Document + its source + library state, for the Document Library UI."""

    document: Document
    source: Source | None = None
    chunk_count: int = 0
    integrity: dict[str, Any] | None = None
    kb_version: int = 1
    retrieval_enabled: bool = False


class UploadedFileResult(BaseModel):
    file_name: str
    size_bytes: int = 0
    status: str = Field(
        default="uploaded",
        description="uploaded | duplicate | rejected | failed | indexed",
    )
    message: str = ""
    document: Document | None = None
    source: Source | None = None
    duplicate_of: str | None = Field(
        default=None, description="Existing document with the same content hash (never silently replaced)"
    )
    integrity: dict[str, Any] | None = None


class UploadResult(BaseModel):
    kb_id: str
    source_mode: SourceMode = SourceMode.USER_PROVIDED
    files: list[UploadedFileResult] = Field(default_factory=list)
    uploaded: int = 0
    duplicates: int = 0
    rejected: int = 0
    failed: int = 0
    indexed: bool = False
    indexing: dict[str, Any] | None = Field(
        default=None, description="Incremental index outcome when index=true"
    )


class Chunk(BaseModel):
    id: str
    document_id: str
    kb_id: str
    chunk_index: int
    text: str
    source_id: str | None = None
    # Provenance (first-class requirement)
    source_url: str | None = None
    source_title: str | None = None
    source_type: str | None = None
    publisher: str | None = None
    document_title: str | None = None
    section: str | None = None
    section_path: str | None = None
    page: int | None = None
    slide: int | None = None
    slide_title: str | None = None
    domain: str | None = None
    subdomain: str | None = None
    trust_score: float | None = None
    content_hash: str = ""
    ingestion_timestamp: datetime | None = None
    char_count: int = 0
    # --- V3: chunking provenance (strategy that produced this chunk) ---
    chunking_strategy: str | None = None
    chunking_config_version: str = "v1"
    chunking_config: dict[str, Any] = Field(
        default_factory=dict, description="Exact chunking parameters that produced this chunk"
    )
    # --- V4: document/build provenance ---
    document_version: int | None = None
    user_provided: bool = False
    kb_version: int | None = None


# ---------------------------------------------------------------------------
# Build runs
# ---------------------------------------------------------------------------

class StageStatus(BaseModel):
    stage: BuildStage
    status: str = Field(default="pending", description="pending | running | done | error | skipped")
    message: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    items_processed: int = 0


class BuildRun(BaseModel):
    id: str
    kb_id: str
    stages: list[StageStatus] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    status: str = Field(default="running", description="running | done | error")


# ---------------------------------------------------------------------------
# Embeddings / Retrieval / Evaluation
# ---------------------------------------------------------------------------

class EmbeddingConfig(BaseModel):
    provider: str = "sentence-transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int | None = None


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0, le=1)
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    provenance: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    top_k: int
    results: list[RetrievalResult] = Field(default_factory=list)
    embedding_model: str = ""
    retrieval_backend: str = "qdrant-dense"
    error: str | None = None


class EvaluationQuestion(BaseModel):
    id: str
    kb_id: str
    question: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(
        default_factory=list,
        description="Fallback relevance heuristic when chunk IDs are unknown",
    )
    generated_by: str = "manual"
    notes: str = ""
    # --- V3 Phase A: benchmark lifecycle metadata ---
    status: QuestionStatus = QuestionStatus.DRAFT
    author: str = ""
    reviewer: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    reviewed_at: datetime | None = None
    revision: int = 1
    supersedes: str | None = Field(
        default=None,
        description="Question ID this DRAFT revision replaces (an APPROVED ancestor)",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured ground-truth provenance: method, source_passages, justification",
    )


class EvaluationQuestionCreate(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    notes: str = Field(
        default="",
        max_length=2000,
        description="Ground-truth provenance: who authored it, from what evidence",
    )
    provenance: dict[str, Any] = Field(default_factory=dict)
    author: str = Field(default="", max_length=200)


class EvaluationRunConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)
    question_ids: list[str] = Field(default_factory=list, description="Empty = all")
    allow_keyword_fallback: bool = Field(
        default=False,
        description="False (strict): questions without explicit chunk/document ground truth "
        "are skipped and reported, never scored with the keyword heuristic. "
        "True (diagnostic): keyword-overlap questions are scored and disclosed.",
    )
    label: str = Field(
        default="",
        max_length=200,
        description="Free-text label for reproducibility, e.g. 'automobile-baseline-v1 k=5 strict'",
    )
    benchmark_version: str | None = Field(
        default=None,
        description=(
            "BenchmarkVersion id to score against. When set, the run MUST reference a "
            "FROZEN version and scores the immutable snapshot content, not live rows."
        ),
    )


class PerQuestionMetrics(BaseModel):
    question_id: str
    question: str
    # Chunk-level metrics (ground truth: expected_chunk_ids, or the transparent
    # keyword-overlap fallback when no explicit IDs are given).
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    ndcg: float | None = None
    num_relevant_found: int = 0
    num_relevant_total: int = 0
    # Document-level metrics (ground truth: expected_document_ids). Filled only
    # when document-level ground truth exists; never fabricated otherwise.
    doc_recall_at_k: float | None = None
    doc_precision_at_k: float | None = None
    doc_mrr: float | None = None
    doc_ndcg: float | None = None
    doc_num_relevant_found: int = 0
    doc_num_relevant_total: int = 0
    note: str = ""


class AggregateMetrics(BaseModel):
    # Chunk-basis aggregates (averaged only over questions that have them).
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    ndcg: float | None = None
    # Document-basis aggregates (averaged only over questions that have them).
    doc_recall_at_k: float | None = None
    doc_precision_at_k: float | None = None
    doc_mrr: float | None = None
    doc_ndcg: float | None = None
    questions_evaluated: int = 0
    # Integrity/reproducibility counts (how the aggregate was actually formed).
    questions_with_explicit_gt: int = 0
    questions_with_keyword_fallback: int = 0
    questions_skipped_no_gt: int = 0
    strict_mode: bool = False
    run_label: str = ""
    # Honest disclosure of how metrics were computed (e.g. how many questions
    # used the keyword-overlap fallback instead of explicit ground truth).
    notes: str = ""


class EvaluationRun(BaseModel):
    id: str
    kb_id: str
    config: EvaluationRunConfig
    aggregate: AggregateMetrics
    per_question: list[PerQuestionMetrics] = Field(default_factory=list)
    retrieval_backend: str = "qdrant-dense"
    embedding_model: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    # --- V3 Phase A: benchmark provenance for runs ---
    benchmark_version: str | None = Field(
        default=None,
        description="Snapshot id of the frozen benchmark this run scored, if any",
    )
    question_statuses: dict[str, str] | None = Field(
        default=None,
        description="question_id -> lifecycle status at run time (integrity disclosure)",
    )


# ---------------------------------------------------------------------------
# V3 Phase A: benchmark versions (frozen snapshots of evaluation questions)
# ---------------------------------------------------------------------------

class BenchmarkVersion(BaseModel):
    """An immutable snapshot of a KB's evaluation questions.

    Creation rules:
    - A snapshot of a KB whose questions are all APPROVED/FROZEN can be FROZEN
      immediately; snapshots containing DRAFT/REVIEW questions stay DRAFT.
    - FROZEN snapshots can never be deleted or modified (freeze protection).
    - Only FROZEN versions are usable for official experiments.
    """

    id: str
    kb_id: str
    version: str = Field(description="Human-readable, e.g. 'automobile-engineering-v1'")
    label: str = ""
    status: BenchmarkStatus = BenchmarkStatus.DRAFT
    question_ids: list[str] = Field(default_factory=list)
    questions_snapshot: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Deep copy of each question at freeze time (immutable audit copy)",
    )
    created_by: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    frozen_at: datetime | None = None
    notes: str = ""


class BenchmarkVersionCreate(BaseModel):
    version: str = Field(min_length=1, max_length=120)
    label: str = Field(default="", max_length=300)
    question_ids: list[str] = Field(default_factory=list, description="Empty = all questions")
    created_by: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=2000)


class QuestionStatusUpdate(BaseModel):
    status: QuestionStatus
    reviewer: str = Field(default="", max_length=200)


class QuestionRevision(BaseModel):
    """Edit payload; editing an APPROVED/FROZEN question creates a new DRAFT revision."""

    question: str | None = Field(default=None, min_length=1, max_length=1000)
    expected_chunk_ids: list[str] | None = None
    expected_document_ids: list[str] | None = None
    expected_keywords: list[str] | None = None
    notes: str | None = Field(default=None, max_length=2000)
    provenance: dict[str, Any] | None = None
