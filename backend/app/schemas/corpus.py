"""Corpus-engineering schemas (V5): batches, manifest, integrity, versions.

Kept in a separate module from :mod:`app.schemas.models` because these models
describe *corpus state* rather than the original domain pipeline, and because
they are imported by several services that should not drag in the whole build
pipeline.

Honesty rules encoded here:

* Every count is either a real measured integer or the sentinel ``-1``
  ("unknown"). Nothing defaults to a number we did not measure.
* A failed document is a first-class persisted record, never a silent drop.
* Integrity findings name the exact document / chunk / vector they concern.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.models import utcnow

#: Sentinel for "we could not confirm this number". Never render as 0.
UNKNOWN = -1


def count_label(value: int) -> str:
    """Human label for a count that may be ``UNKNOWN``."""
    return "unknown" if value == UNKNOWN else str(value)


# ---------------------------------------------------------------------------
# Phase 1 — ingestion batches
# ---------------------------------------------------------------------------


class IngestionStage(str, enum.Enum):
    """Per-document lifecycle. Every transition is persisted."""

    QUEUED = "queued"
    VALIDATING = "validating"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Stages that mean "this document still has work to do".
INCOMPLETE_STAGES: frozenset[IngestionStage] = frozenset(
    {
        IngestionStage.QUEUED,
        IngestionStage.VALIDATING,
        IngestionStage.PARSING,
        IngestionStage.CHUNKING,
        IngestionStage.EMBEDDING,
        IngestionStage.INDEXING,
    }
)

TERMINAL_STAGES: frozenset[IngestionStage] = frozenset(
    {IngestionStage.COMPLETE, IngestionStage.FAILED, IngestionStage.CANCELLED}
)


class IngestionItemStatus(str, enum.Enum):
    """Outcome vocabulary for one file in a batch."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class IngestionItem(BaseModel):
    """One file inside one batch. The unit of resume and of failure isolation."""

    id: str
    batch_id: str
    kb_id: str
    #: Stable identity of the FILE across retries/resumes (not the Document id).
    item_key: str = Field(description="Deterministic identity of this upload slot")
    file_name: str
    normalized_file_name: str = ""
    size_bytes: int = 0
    mime_type: str | None = None
    content_hash: str | None = Field(
        default=None, description="Known as soon as the bytes are read"
    )
    status: IngestionItemStatus = IngestionItemStatus.PENDING
    stage: IngestionStage = IngestionStage.QUEUED
    #: 0.0 - 1.0. Measured from completed stages, never guessed.
    progress: float = 0.0
    error_code: str | None = None
    error_message: str | None = None
    attempts: int = 0
    #: The Document this item produced (None while pending/failed).
    document_id: str | None = None
    source_id: str | None = None
    #: Set when the content hash matched an existing document.
    duplicate_of: str | None = None
    parser_used: str | None = None
    chunk_count: int = 0
    vector_count: int = UNKNOWN
    warnings: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def is_incomplete(self) -> bool:
        return self.stage in INCOMPLETE_STAGES or self.status in {
            IngestionItemStatus.PENDING,
            IngestionItemStatus.PROCESSING,
        }


class IngestionBatch(BaseModel):
    """A persistent record of one bulk ingestion run."""

    id: str
    kb_id: str
    status: str = Field(default="pending", description="pending|running|complete|failed|cancelled")
    source_mode: str = "user_provided"
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    duplicate_items: int = 0
    rejected_items: int = 0
    #: Set when the batch finished indexing; drives corpus-version creation.
    chunker: str = "section-aware"
    target_size: int = 1200
    overlap: int = 150
    index_on_complete: bool = True
    #: True once a resume has run; used to make resume idempotent and visible.
    resumed_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    timings: dict[str, float] = Field(default_factory=dict)

    def progress(self, items: list[IngestionItem] | None = None) -> float:
        if self.total_items <= 0:
            return 0.0
        done = self.completed_items + self.failed_items + self.duplicate_items + self.rejected_items
        return round(min(1.0, done / self.total_items), 4)


class IngestionBatchDetail(BaseModel):
    batch: IngestionBatch
    items: list[IngestionItem]
    #: Items that a resume would pick up right now.
    resumable_items: int = 0


# ---------------------------------------------------------------------------
# Phase 2 — corpus manifest
# ---------------------------------------------------------------------------


class CorpusManifestEntry(BaseModel):
    """Everything RAGForge knows about ONE document in the corpus."""

    document_id: str
    file_name: str | None = None
    normalized_file_name: str = ""
    content_hash: str = ""
    document_version: int = 1
    source_mode: str = ""
    source_url: str | None = None
    source_id: str | None = None
    file_type: str = ""
    file_size: int | None = None
    parser: str | None = None
    parser_version: str | None = None
    page_count: int | None = None
    slide_count: int | None = None
    section_count: int | None = None
    text_length: int = 0
    chunk_count: int = 0
    vector_count: int = UNKNOWN
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    chunking_strategy: str | None = None
    chunking_config_hash: str | None = None
    indexed_at: datetime | None = None
    status: str = ""
    error_message: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class CorpusManifest(BaseModel):
    """'What exactly is inside this knowledge base?'"""

    kb_id: str
    kb_name: str
    kb_version: int = 1
    entries: list[CorpusManifestEntry] = Field(default_factory=list)
    summary: "CorpusSummary" = Field(default_factory=lambda: CorpusSummary())
    generated_at: datetime = Field(default_factory=utcnow)


class CorpusSummary(BaseModel):
    total_documents: int = 0
    completed_documents: int = 0
    failed_documents: int = 0
    processing_documents: int = 0
    total_pages: int = 0
    total_slides: int = 0
    total_sections: int = 0
    total_chunks: int = 0
    total_vectors: int = UNKNOWN
    total_corpus_size_bytes: int = 0
    duplicate_documents: int = 0
    stale_documents: int = 0
    orphan_vectors: int = UNKNOWN
    embedding_identity: str | None = None
    chunking_identity: str | None = None
    vector_backend: str | None = None
    last_successful_index_at: datetime | None = None
    #: True only when a real vector count was read from the store.
    vector_count_confirmed: bool = False


# ---------------------------------------------------------------------------
# Phase 3 — integrity
# ---------------------------------------------------------------------------


class IntegritySeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IntegrityCheck(str, enum.Enum):
    """The named checks. Report keys map 1:1 onto these values."""

    MISSING_VECTORS = "missing_vectors"
    ORPHAN_VECTORS = "orphan_vectors"
    STALE_VECTORS = "stale_vectors"
    EMBEDDING_MISMATCH = "embedding_mismatch"
    CHUNKING_MISMATCH = "chunking_mismatch"
    DUPLICATE_DOCUMENTS = "duplicate_documents"
    DUPLICATE_CHUNKS = "duplicate_chunks"
    FAILED_DOCUMENTS = "failed_documents"
    PARTIAL_DOCUMENTS = "partial_documents"
    PROVENANCE_GAPS = "provenance_gaps"
    BROKEN_SOURCE_REFERENCES = "broken_source_references"


class IntegrityFinding(BaseModel):
    """One concrete problem, always naming what it affects."""

    check: IntegrityCheck
    severity: IntegritySeverity = IntegritySeverity.WARNING
    message: str
    document_id: str | None = None
    chunk_id: str | None = None
    vector_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class IntegrityReport(BaseModel):
    """Read-only audit. This model NEVER repairs anything."""

    kb_id: str
    overall_status: str = Field(default="HEALTHY", description="HEALTHY|WARNING|ERROR")
    documents_checked: int = 0
    chunks_checked: int = 0
    vectors_checked: int = UNKNOWN
    findings: list[IntegrityFinding] = Field(default_factory=list)
    missing_vectors: list[IntegrityFinding] = Field(default_factory=list)
    orphan_vectors: list[IntegrityFinding] = Field(default_factory=list)
    stale_vectors: list[IntegrityFinding] = Field(default_factory=list)
    embedding_mismatches: list[IntegrityFinding] = Field(default_factory=list)
    chunking_mismatches: list[IntegrityFinding] = Field(default_factory=list)
    duplicate_documents: list[IntegrityFinding] = Field(default_factory=list)
    duplicate_chunks: list[IntegrityFinding] = Field(default_factory=list)
    failed_documents: list[IntegrityFinding] = Field(default_factory=list)
    partial_documents: list[IntegrityFinding] = Field(default_factory=list)
    provenance_gaps: list[IntegrityFinding] = Field(default_factory=list)
    broken_source_references: list[IntegrityFinding] = Field(default_factory=list)
    #: Findings whose count could not be confirmed because a service was down.
    unknown_counts: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
    duration_seconds: float = 0.0

    def counts(self) -> dict[str, int]:
        return {
            "missing_vectors": len(self.missing_vectors),
            "orphan_vectors": len(self.orphan_vectors),
            "stale_vectors": len(self.stale_vectors),
            "embedding_mismatches": len(self.embedding_mismatches),
            "chunking_mismatches": len(self.chunking_mismatches),
            "duplicate_documents": len(self.duplicate_documents),
            "duplicate_chunks": len(self.duplicate_chunks),
            "failed_documents": len(self.failed_documents),
            "partial_documents": len(self.partial_documents),
            "provenance_gaps": len(self.provenance_gaps),
            "broken_source_references": len(self.broken_source_references),
        }


# ---------------------------------------------------------------------------
# Phase 4 — safe repair
# ---------------------------------------------------------------------------


class RepairAction(str, enum.Enum):
    REINDEX_DOCUMENT = "reindex_document"
    REINDEX_FAILED_DOCUMENTS = "reindex_failed_documents"
    REMOVE_ORPHAN_VECTORS = "remove_orphan_vectors"
    REBUILD_DOCUMENT_VECTORS = "rebuild_document_vectors"
    REBUILD_KB = "rebuild_kb"
    RECOMPUTE_EMBEDDINGS = "recompute_embeddings"
    REINDEX_BATCH = "reindex_batch"


#: Actions that delete or overwrite data. These require explicit confirmation.
DESTRUCTIVE_ACTIONS: frozenset[RepairAction] = frozenset(
    {
        RepairAction.REMOVE_ORPHAN_VECTORS,
        RepairAction.REBUILD_KB,
        RepairAction.RECOMPUTE_EMBEDDINGS,
    }
)


class RepairPlan(BaseModel):
    """What a repair WOULD do. Produced before anything is changed."""

    action: RepairAction
    kb_id: str
    description: str
    affected_documents: list[str] = Field(default_factory=list)
    expected_chunks: int = UNKNOWN
    expected_vectors: int = UNKNOWN
    destructive: bool = False
    #: True when every expected count above was actually measured.
    counts_confirmed: bool = False


class RepairRequest(BaseModel):
    action: RepairAction
    document_ids: list[str] = Field(default_factory=list)
    batch_id: str | None = None
    #: Required for destructive actions. Must match `action` to prevent accidents.
    confirm_action: RepairAction | None = None
    chunker: str = "section-aware"
    target_size: int = 1200
    overlap: int = 150


class RepairResult(BaseModel):
    action: RepairAction
    kb_id: str
    ok: bool
    documents_affected: int = 0
    chunks_written: int = 0
    #: Confirmed post-conditions, or UNKNOWN when they cannot be verified.
    vectors_removed: int = UNKNOWN
    vectors_written: int = UNKNOWN
    removal_confirmed: bool = False
    message: str = ""
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None

    def vectors_removed_label(self) -> str:
        return count_label(self.vectors_removed)


# ---------------------------------------------------------------------------
# Phase 5/6 — corpus versions and diffs
# ---------------------------------------------------------------------------


class CorpusVersion(BaseModel):
    """A deterministic snapshot identity of a corpus at one moment."""

    id: str
    kb_id: str
    version: str
    fingerprint: str = Field(description="sha256 over canonicalized manifest metadata")
    document_count: int = 0
    chunk_count: int = 0
    vector_backend: str = ""
    embedding_identity: str = ""
    chunking_identity: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    note: str | None = None


class DocumentChange(BaseModel):
    document_id: str | None = None
    file_name: str | None = None
    old_content_hash: str | None = None
    new_content_hash: str | None = None
    old_chunk_count: int | None = None
    new_chunk_count: int | None = None
    vectors_invalidated: int = UNKNOWN
    vectors_created: int = UNKNOWN
    document_version: int | None = None


class CorpusDiff(BaseModel):
    """What changed between two corpus versions."""

    kb_id: str
    from_version: str | None = None
    to_version: str | None = None
    from_fingerprint: str | None = None
    to_fingerprint: str | None = None
    added: list[DocumentChange] = Field(default_factory=list)
    removed: list[DocumentChange] = Field(default_factory=list)
    changed: list[DocumentChange] = Field(default_factory=list)
    unchanged: list[DocumentChange] = Field(default_factory=list)
    #: True when the two fingerprints are equal (deterministic, order-independent).
    identical: bool = False


# ---------------------------------------------------------------------------
# Phase 7 — instrumentation
# ---------------------------------------------------------------------------


class PipelineTimings(BaseModel):
    """Measured, never estimated. Seconds."""

    upload: float = 0.0
    validation: float = 0.0
    parse: float = 0.0
    chunk: float = 0.0
    embed: float = 0.0
    index: float = 0.0
    total: float = 0.0
    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    bytes: int = 0

    def throughput(self) -> dict[str, float]:
        """Rates. A zero-duration stage yields 0.0 rather than a division error."""
        return {
            "documents_per_minute": round(self.documents * 60 / self.total, 2) if self.total > 0 else 0.0,
            "chunks_per_second": round(self.chunks / self.total, 2) if self.total > 0 else 0.0,
            "vectors_per_second": round(self.vectors / self.total, 2) if self.total > 0 else 0.0,
            "megabytes_per_second": round(self.bytes / (1024 * 1024) / self.total, 3) if self.total > 0 else 0.0,
        }


class ScaleBenchmarkResult(BaseModel):
    """One scale point of the corpus benchmark harness."""

    label: str
    documents: int
    chunks: int = 0
    bytes: int = 0
    timings: PipelineTimings = Field(default_factory=PipelineTimings)
    throughput: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


CorpusManifest.model_rebuild()