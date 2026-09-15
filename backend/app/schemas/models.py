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
