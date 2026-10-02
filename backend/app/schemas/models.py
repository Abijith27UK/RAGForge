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
    USER_UPLOAD = "user_upload"
    TEXT = "text"


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


class SourceDecisionUpdate(BaseModel):
    decision: SourceDecision


class QualitySignals(BaseModel):
    authority: float = Field(ge=0, le=1, description="Domain authority of publisher/URL")
    relevance: float = Field(ge=0, le=1, description="Match against domain spec")
    recency: float = Field(ge=0, le=1, description="Freshness of content")
    source_type: float = Field(ge=0, le=1, description="Trustworthiness of the type")
    accessibility: float = Field(ge=0, le=1, description="Fetchable / parseable")
    duplication: float = Field(ge=0, le=1, description="1 = unique, lower = duplicated")
    evidence_quality: float = Field(ge=0, le=1, description="Citations, standards refs, etc.")


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
    domain: str | None = None
    subdomain: str | None = None
    trust_score: float | None = None
    content_hash: str = ""
    ingestion_timestamp: datetime | None = None
    char_count: int = 0
    # --- V3: chunking provenance (strategy that produced this chunk) ---
    chunking_strategy: str | None = None
    chunking_config_version: str = "v1"


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
