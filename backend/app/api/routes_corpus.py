"""Corpus Command Center API (V5 Phases 1-7).

Route naming follows the existing convention: everything is scoped under
``/api/knowledge-bases/{kb_id}/...`` alongside the document library.

Design rules enforced here:

* Bulk ingestion is persistent and resumable; a page refresh never loses state.
* Integrity scans are READ-ONLY. Repair is a separate, explicit endpoint and
  destructive repairs require ``confirm_action`` matching the action.
* No number is invented. Counts that cannot be measured are reported as
  ``-1`` / ``"unknown"``.
* This router performs NO destructive experiment against any pre-existing
  knowledge base; destructive repairs name their blast radius in a plan first.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import get_repo
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.corpus import (
    CorpusDiff,
    CorpusManifest,
    CorpusVersion,
    IntegrityReport,
    IngestionBatch,
    IngestionBatchDetail,
    RepairPlan,
    RepairRequest,
    RepairResult,
)
from app.schemas.models import Document, Source, SourceType
from app.services.corpus.batches import BatchError, IngestionBatchService, batch_timings
from app.services.corpus.fingerprint import (
    build_manifest,
    corpus_fingerprint,
    diff_corpus_versions,
    diff_manifests,
    snapshot_corpus_version,
)
from app.services.corpus.integrity import CorpusIntegrityService
from app.services.corpus.manifest import CorpusManifestService
from app.services.corpus.repair import CorpusRepairService, RepairError
from app.services.embeddings.provider import create_embedding_provider
from app.services.indexing.document_indexer import IndexingError, index_documents
from app.services.ingestion.ingestion import IngestionError, ingest_uploaded_bytes
from app.services.ingestion.parsers import supported_upload_extensions
from app.services.ingestion.upload import sanitize_file_name
from app.services.source_quality.user_scorer import assess_user_upload
# Resolve the backend through the MODULE at call time, not by binding the
# function at import time. The factory is the single extension point for vector
# backends, and resolving late means a registered/patched backend (including the
# hermetic in-memory store used by tests) is honoured instead of being bypassed.
from app.services.vector_store import factory as vector_factory
from app.utils.ids import new_id
from app.utils.text import sha256_bytes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["corpus"])

_EXTENSION_SOURCE_TYPE = {
    ".pdf": SourceType.PDF,
    ".pptx": SourceType.PRESENTATION, ".pptm": SourceType.PRESENTATION,
    ".ppt": SourceType.PRESENTATION,
    ".docx": SourceType.DOCUMENT, ".docm": SourceType.DOCUMENT,
    ".txt": SourceType.TEXT, ".md": SourceType.TEXT, ".markdown": SourceType.TEXT,
    ".html": SourceType.WEB_PAGE, ".htm": SourceType.WEB_PAGE,
}


class ResumeRequest(BaseModel):
    include_completed: bool = Field(
        default=False, description="Re-process documents that already completed (forces a rebuild)"
    )
    retry_failed: bool = Field(default=True, description="Retry documents that failed")
    index: bool = Field(default=True, description="Chunk/embed/index the resumed documents")


class SnapshotRequest(BaseModel):
    note: str | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _kb(repo: Repository, kb_id: str):
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


def _batch(repo: Repository, batch_id: str) -> IngestionBatch:
    batch = repo.get_ingestion_batch(batch_id)
    if batch is None:
        raise HTTPException(404, "Ingestion batch not found")
    return batch


def _service() -> IngestionBatchService:
    settings = get_settings()
    return IngestionBatchService(
        repo=None,  # type: ignore[arg-type]  # set per-request below
        max_bytes=settings.max_upload_bytes,
        upload_dir_for=settings.kb_upload_dir,
        allowed_extensions=supported_upload_extensions(),
    )


def _service_for(repo: Repository) -> IngestionBatchService:
    settings = get_settings()
    return IngestionBatchService(
        repo=repo,
        max_bytes=settings.max_upload_bytes,
        upload_dir_for=settings.kb_upload_dir,
        allowed_extensions=supported_upload_extensions(),
    )


def _source_builder(kb_id: str):
    def build(file_name: str, size: int, extension: str) -> Source:
        return Source(
            id=new_id("src"), kb_id=kb_id, url=f"upload://{file_name}", title=file_name,
            source_type=_EXTENSION_SOURCE_TYPE.get(extension, SourceType.USER_PROVIDED),
            discovered_via="user-upload", user_provided=True, provenance="user_upload",
            file_name=file_name, file_size=size,
        )
    return build


def _indexer(kb, chunker: str, target_size: int, overlap: int):
    """Bind the shared indexer with this batch's chunking configuration."""
    settings = get_settings()
    store = vector_factory.create_vector_store(settings, backend=kb.vector_backend)
    embedder = create_embedding_provider(settings)

    def run(documents: list[Document]):
        return index_documents(
            repo=run.repo, kb=kb, documents=documents, store=store, embedder=embedder,
            chunker_name=chunker, target_size=target_size, overlap=overlap,
            full_build=False, kb_version=kb.version + 1,
        ).as_dict()

    return run


# ---------------------------------------------------------------------------
# Phase 1 — ingestion batches
# ---------------------------------------------------------------------------

@router.post("/{kb_id}/ingestion-batches", response_model=IngestionBatchDetail, status_code=201)
async def create_ingestion_batch(
    kb_id: str,
    files: list[UploadFile] = File(..., description="Corpus files (designed for 50-200)"),
    index: bool = Form(default=True),
    chunker: str = Form(default="section-aware"),
    target_size: int = Form(default=1200),
    overlap: int = Form(default=150),
    repo: Repository = Depends(get_repo),
):
    """Create a persistent batch and ingest every file in it.

    Each file becomes an item row BEFORE any processing starts, so a crash
    mid-request still leaves a resumable record rather than silent data loss.
    A failure in one file never fails the batch.
    """
    kb = _kb(repo, kb_id)
    settings = get_settings()
    if len(files) > settings.max_upload_files_per_request:
        raise HTTPException(
            400,
            f"{len(files)} files exceeds the per-request limit of "
            f"{settings.max_upload_files_per_request}. Upload in several batches, or raise "
            "MAX_UPLOAD_FILES_PER_REQUEST.",
        )

    service = _service_for(repo)
    batch = service.create_batch(
        kb_id=kb_id,
        source_mode=kb.source_mode.value if hasattr(kb.source_mode, "value") else str(kb.source_mode),
        chunker=chunker, target_size=target_size, overlap=overlap,
        index_on_complete=index, total_items=len(files),
    )
    batch.status = "running"

    indexer = None
    if index:
        indexer = _indexer(kb, chunker, target_size, overlap)
        indexer.repo = repo  # type: ignore[attr-defined]

    builder = _source_builder(kb_id)
    for position, upload in enumerate(files):
        safe = sanitize_file_name(upload.filename or "upload")
        data = await upload.read()
        item = service.add_item(
            batch_id=batch.id, kb_id=kb_id, index=position, file_name=safe,
            size=len(data), content_hash=sha256_bytes(data) or None,
            mime_type=upload.content_type,
        )
        service.process_item(
            item, data=data, kb=kb, source_builder=builder,
            index_documents=indexer,
        )
        service._recompute_batch_counts(batch)

    batch = service.finish_batch(batch, kb)
    if index and batch.completed_items:
        try:
            manifest = build_manifest(repo, kb)
            snapshot_corpus_version(repo, kb, manifest, note=f"after batch {batch.id}")
        except Exception as exc:  # snapshotting must never lose the ingestion
            logger.warning("Could not snapshot corpus after batch %s: %s", batch.id, exc)

    detail = service.get_detail(batch)
    detail.batch.timings = {
        k: v for k, v in batch_timings(detail.items).model_dump().items()
        if isinstance(v, float)
    }
    return detail


@router.get("/{kb_id}/ingestion-batches", response_model=list[IngestionBatch])
def list_ingestion_batches(kb_id: str, repo: Repository = Depends(get_repo)):
    """Batch history. Survives a page refresh and a server restart."""
    _kb(repo, kb_id)
    return repo.list_ingestion_batches(kb_id)


@router.get("/{kb_id}/ingestion-batches/{batch_id}", response_model=IngestionBatchDetail)
def get_ingestion_batch(kb_id: str, batch_id: str, repo: Repository = Depends(get_repo)):
    kb = _kb(repo, kb_id)
    batch = _batch(repo, batch_id)
    if batch.kb_id != kb_id:
        raise HTTPException(404, "Ingestion batch not found for this knowledge base")
    return _service_for(repo).get_detail(batch)


@router.post("/{kb_id}/ingestion-batches/{batch_id}/resume", response_model=IngestionBatchDetail)
async def resume_ingestion_batch(
    kb_id: str, batch_id: str, payload: ResumeRequest, repo: Repository = Depends(get_repo),
):
    """Resume an interrupted or partially failed batch.

    Idempotent by construction: an item that already completed is skipped unless
    ``include_completed`` is set, and an item that already produced a Document
    reuses that Document rather than creating a second one — so a resume can
    neither duplicate vectors nor duplicate documents.
    """
    kb = _kb(repo, kb_id)
    batch = _batch(repo, batch_id)
    if batch.kb_id != kb_id:
        raise HTTPException(404, "Ingestion batch not found for this knowledge base")

    service = _service_for(repo)
    pending = service.resumable_items(
        batch, include_completed=payload.include_completed, retry_failed=payload.retry_failed,
    )
    if not pending:
        return service.get_detail(batch)

    indexer = None
    if payload.index:
        indexer = _indexer(kb, batch.chunker, batch.target_size, batch.overlap)
        indexer.repo = repo  # type: ignore[attr-defined]

    builder = _source_builder(kb_id)
    for item in pending:
        # Bytes live on disk; a resume re-reads them instead of asking the
        # client to re-upload, which is the whole point of persistence.
        if not item.document_id:
            raw = _stored_bytes_for(repo, kb_id, item)
            if raw is None:
                service._fail(item, "SOURCE_MISSING", "Original bytes are no longer on disk; re-upload this file.")
                continue
            service.process_item(
                item, data=raw, kb=kb, source_builder=builder, index_documents=indexer,
                force=payload.include_completed,
            )
        elif payload.include_completed:
            doc = repo.get_document(kb_id, item.document_id)
            if doc is not None and indexer is not None:
                try:
                    indexer([doc])
                except (IndexingError, IngestionError) as exc:
                    service._fail(item, "INDEX_FAILED", str(exc))
                    continue
                service._advance(item, item.stage, item.status)
        service._recompute_batch_counts(batch)

    batch.resumed_at = datetime.now(timezone.utc)
    batch.status = "running"
    repo.update_ingestion_batch(batch)
    batch = service.finish_batch(batch, kb)
    if batch.completed_items:
        try:
            snapshot_corpus_version(
                repo, kb, build_manifest(repo, kb), note=f"after resume of {batch.id}",
            )
        except Exception as exc:
            logger.warning("Could not snapshot corpus after resume: %s", exc)
    return service.get_detail(batch)


def _stored_bytes_for(repo: Repository, kb_id: str, item) -> bytes | None:
    """Re-read an item's original bytes from disk for a resume."""
    doc = repo.get_document(kb_id, item.document_id) if item.document_id else None
    raw_path = doc.raw_file_path if doc else None
    if not raw_path:
        return None
    try:
        return Path(raw_path).read_bytes()
    except OSError as exc:
        logger.warning("Could not re-read %s: %s", raw_path, exc)
        return None


# ---------------------------------------------------------------------------
# Phase 2 — corpus manifest
# ---------------------------------------------------------------------------

@router.get("/{kb_id}/corpus-manifest", response_model=CorpusManifest)
def corpus_manifest(kb_id: str, with_vectors: bool = True, repo: Repository = Depends(get_repo)):
    """'What exactly is inside this knowledge base?' — per document and summary."""
    kb = _kb(repo, kb_id)
    return CorpusManifestService(repo).build(kb, with_vector_count=with_vectors)


@router.get("/{kb_id}/corpus-summary")
def corpus_summary(kb_id: str, repo: Repository = Depends(get_repo)):
    kb = _kb(repo, kb_id)
    return CorpusManifestService(repo).build(kb).summary


# ---------------------------------------------------------------------------
# Phase 3 — integrity (READ-ONLY)
# ---------------------------------------------------------------------------

@router.get("/{kb_id}/corpus-integrity", response_model=IntegrityReport)
def corpus_integrity(kb_id: str, include_vectors: bool = True, repo: Repository = Depends(get_repo)):
    """Audit the corpus. This endpoint NEVER modifies anything."""
    kb = _kb(repo, kb_id)
    return CorpusIntegrityService(repo).scan(kb, include_vectors=include_vectors)


@router.get("/{kb_id}/corpus-health")
def corpus_health(kb_id: str, include_vectors: bool = True, repo: Repository = Depends(get_repo)):
    """Compact health strip for the Corpus Command Center."""
    kb = _kb(repo, kb_id)
    return CorpusIntegrityService(repo).summary(kb, include_vectors=include_vectors)


# ---------------------------------------------------------------------------
# Phase 4 — safe repair
# ---------------------------------------------------------------------------

@router.post("/{kb_id}/corpus-repair/plan", response_model=RepairPlan)
def plan_repair(kb_id: str, payload: RepairRequest, repo: Repository = Depends(get_repo)):
    """Describe a repair WITHOUT performing it. Shows the blast radius first."""
    kb = _kb(repo, kb_id)
    service = _repair_service(repo, kb, payload)
    report = None
    if payload.action.value == "remove_orphan_vectors":
        report = CorpusIntegrityService(repo).scan(kb)
    try:
        return service.plan(kb, payload.action, document_ids=payload.document_ids,
                            batch_id=payload.batch_id, report=report)
    except RepairError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{kb_id}/corpus-repair", response_model=RepairResult)
def run_repair(kb_id: str, payload: RepairRequest, repo: Repository = Depends(get_repo)):
    """Execute a repair. Destructive actions require an explicit confirmation."""
    kb = _kb(repo, kb_id)
    service = _repair_service(repo, kb, payload)
    report = None
    if payload.action.value == "remove_orphan_vectors":
        report = CorpusIntegrityService(repo).scan(kb)
    try:
        plan = service.plan(kb, payload.action, document_ids=payload.document_ids,
                            batch_id=payload.batch_id, report=report)
        return service.execute(
            kb, plan, confirm_action=payload.confirm_action,
            chunker=payload.chunker, target_size=payload.target_size, overlap=payload.overlap,
        )
    except (RepairError, IndexingError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _repair_service(repo: Repository, kb, payload: RepairRequest) -> CorpusRepairService:
    settings = get_settings()
    store = vector_factory.create_vector_store(settings, backend=kb.vector_backend)
    embedder = create_embedding_provider(settings)

    def run(documents: list[Document]):
        return index_documents(
            repo=repo, kb=kb, documents=documents, store=store, embedder=embedder,
            chunker_name=payload.chunker, target_size=payload.target_size,
            overlap=payload.overlap, full_build=payload.action.value == "rebuild_kb",
            kb_version=kb.version + 1,
        ).as_dict()

    return CorpusRepairService(repo=repo, index_documents=run, create_store=store)


# ---------------------------------------------------------------------------
# Phase 5/6 — versions and diffs
# ---------------------------------------------------------------------------

@router.get("/{kb_id}/corpus-versions", response_model=list[CorpusVersion])
def list_corpus_versions(kb_id: str, repo: Repository = Depends(get_repo)):
    kb = _kb(repo, kb_id)
    return repo.list_corpus_versions(kb_id)


@router.post("/{kb_id}/corpus-versions", response_model=CorpusVersion)
def snapshot_corpus(kb_id: str, payload: SnapshotRequest, repo: Repository = Depends(get_repo)):
    """Fingerprint the current corpus and record a version.

    Deterministic: snapshotting an unchanged corpus returns the EXISTING
    version rather than creating a duplicate.
    """
    kb = _kb(repo, kb_id)
    manifest = build_manifest(repo, kb)
    return snapshot_corpus_version(repo, kb, manifest, note=payload.note)


@router.get("/{kb_id}/corpus-versions/diff", response_model=CorpusDiff)
def diff_versions(kb_id: str, from_version: str | None = None, repo: Repository = Depends(get_repo)):
    """Diff a stored corpus version against the live corpus."""
    kb = _kb(repo, kb_id)
    old = repo.get_corpus_version(kb_id, from_version) if from_version else None
    if from_version and old is None:
        raise HTTPException(404, "Corpus version not found")
    manifest = build_manifest(repo, kb)
    return diff_corpus_versions(repo, kb_id, from_version=old, to_manifest=manifest, to_version=None)


@router.get("/{kb_id}/corpus-fingerprint")
def corpus_fingerprint_endpoint(kb_id: str, repo: Repository = Depends(get_repo)):
    """Current corpus fingerprint without persisting a version."""
    kb = _kb(repo, kb_id)
    manifest = build_manifest(repo, kb)
    return {
        "kb_id": kb_id,
        "fingerprint": corpus_fingerprint(manifest),
        "document_count": manifest.summary.total_documents,
        "chunk_count": manifest.summary.total_chunks,
        "embedding_identity": manifest.summary.embedding_identity,
        "chunking_identity": manifest.summary.chunking_identity,
    }