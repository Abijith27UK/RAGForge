"""Build pipeline routes: ingestion -> chunking -> embedding -> indexing."""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_repo
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.models import (
    BuildRun,
    BuildStage,
    Chunk,
    Document,
    KBStatus,
    Source,
    SourceDecision,
    StageStatus,
)
from app.services.chunking.chunker import get_chunker
from app.services.vector_store.factory import create_vector_store
from app.services.ingestion.ingestion import IngestionError, ingest_source
from app.services.vector_store.qdrant_store import QdrantVectorStore, VectorStoreError
from app.utils.ids import new_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["build"])

_QDRANT_NOT_RUNNING = (
    "Qdrant is not reachable. Start it (see README: qdrant/qdrant.exe) and try again."
)


def _qdrant_store():
    settings = get_settings()
    return QdrantVectorStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


class IngestRequest(BaseModel):
    source_ids: list[str] = Field(default_factory=list, description="Empty = all ACCEPTED sources")


class IndexRequest(BaseModel):
    chunker: str = "section-aware"
    target_size: int = Field(default=1200, ge=200, le=8000)
    overlap: int = Field(default=150, ge=0, le=1000)


@router.post("/{kb_id}/ingest", response_model=BuildRun)
def ingest(kb_id: str, payload: IngestRequest, repo: Repository = Depends(get_repo)):
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    sources = repo.list_sources(kb_id)
    if payload.source_ids:
        wanted = set(payload.source_ids)
        sources = [s for s in sources if s.id in wanted]
    else:  # default: all ACCEPTED sources
        sources = [s for s in sources if s.decision == SourceDecision.ACCEPT]
    if not sources:
        raise HTTPException(400, "No ACCEPTED sources to ingest. Accept sources first.")

    settings = get_settings()
    settings.documents_dir.mkdir(parents=True, exist_ok=True)

    run = BuildRun(
        id=new_id("build"),
        kb_id=kb_id,
        stages=[
            StageStatus(stage=BuildStage.INGESTION, status="running", started_at=datetime.now(timezone.utc))
        ],
    )
    repo.create_build_run(run)

    ingested, skipped, failed = 0, 0, 0
    for source in sources:
        try:
            doc = ingest_source(source, kb_id, settings.documents_dir)
            existing = repo.find_document_by_hash(kb_id, doc.content_hash)
            if existing:
                logger.info(
                    "Duplicate content (hash %s) from %s; keeping existing document %s",
                    doc.content_hash[:12], source.url, existing.id,
                )
                skipped += 1
                continue
            repo.create_document(doc)
            ingested += 1
        except IngestionError as exc:
            failed += 1
            logger.warning("Ingestion failed for %s: %s", source.url, exc)
        except Exception:
            failed += 1
            logger.exception("Unexpected ingestion error for %s", source.url)

    run.stages[0].status = "done" if (ingested or skipped) else "error"
    run.stages[0].items_processed = ingested
    run.stages[0].message = f"{ingested} ingested, {skipped} duplicate skipped, {failed} failed"
    run.stages[0].finished_at = datetime.now(timezone.utc)
    run.status = "done" if (ingested or skipped) else "error"
    run.finished_at = datetime.now(timezone.utc)
    repo.update_build_run(run)
    if ingested:
        kb.status = KBStatus.BUILDING
        repo.update_kb(kb)
    return run


@router.get("/{kb_id}/documents", response_model=list[Document])
def list_documents(kb_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return repo.list_documents(kb_id)


@router.get("/{kb_id}/chunks", response_model=list[Chunk])
def list_chunks(
    kb_id: str,
    document_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    repo: Repository = Depends(get_repo),
):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return repo.list_chunks(kb_id, document_id=document_id, limit=min(limit, 500), offset=offset)


@router.post("/{kb_id}/index", response_model=BuildRun)
def index_kb(kb_id: str, payload: IndexRequest, repo: Repository = Depends(get_repo)):
    """Chunk + embed + index all ingested documents into Qdrant.

    Stale-vector handling: vectors for every re-indexed document are deleted
    from Qdrant BEFORE new ones are upserted (re-chunking generates new chunk
    IDs, so upserting alone would leave orphaned points retrievable forever),
    and any leftover orphans (e.g. from deleted documents or older partial
    builds) are swept at the end via delete_orphaned_points.
    """
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    documents = repo.list_documents(kb_id)
    if not documents:
        raise HTTPException(400, "No documents ingested yet. Run ingestion first.")

    settings = get_settings()
    from app.services.embeddings.provider import EmbeddingError, create_embedding_provider

    try:
        embedder = create_embedding_provider(settings)
    except EmbeddingError as exc:
        raise HTTPException(503, str(exc)) from exc

    store = create_vector_store(settings, backend=kb.vector_backend)
    run = BuildRun(
        id=new_id("build"),
        kb_id=kb_id,
        stages=[
            StageStatus(stage=BuildStage.CHUNKING, status="running", started_at=datetime.now(timezone.utc)),
        ],
    )
    repo.create_build_run(run)

    try:
        chunker = get_chunker(payload.chunker)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    doc_texts: dict[str, str] = {}
    for doc in documents:
        if not doc.file_path:
            continue
        try:
            text = Path(doc.file_path).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read parsed text for %s: %s", doc.id, exc)
            continue
        doc_texts[doc.id] = text

    chunks: list[Chunk] = []
    for doc in documents:
        if doc.id not in doc_texts:
            continue
        source = repo.get_source(kb_id, doc.source_id)
        if source is None:
            logger.warning("Document %s references missing source %s; skipped", doc.id, doc.source_id)
            continue
        doc_chunks = chunker.chunk(
            document=doc,
            text=doc_texts[doc.id],
            source=source,
            target_size=payload.target_size,
            overlap=payload.overlap,
            domain=kb.domain,
        )
        chunks.extend(doc_chunks)
    if not chunks:
        raise HTTPException(400, "No chunks produced; check that parsed document text exists on disk.")

    run.stages[0].status = "done"
    run.stages[0].items_processed = len(chunks)
    run.stages[0].message = f"{len(chunks)} chunks from {len(doc_texts)} document(s)"
    run.stages[0].finished_at = datetime.now(timezone.utc)
    run.stages.append(
        StageStatus(stage=BuildStage.EMBEDDING, status="running", started_at=datetime.now(timezone.utc))
    )

    try:
        store.ensure_collection(kb_id, embedder.dimensions)
    except VectorStoreError as exc:
        raise HTTPException(503, f"{_QDRANT_NOT_RUNNING} ({exc})") from exc

    # Delete existing vectors for these documents BEFORE upserting new ones:
    # re-chunking generates new chunk IDs, so upserting alone would leave
    # orphaned points retrievable forever.
    try:
        store.delete_document_vectors(kb_id, list(doc_texts.keys()))
    except VectorStoreError as exc:
        run.stages[1].status = "error"
        run.stages[1].message = str(exc)
        run.status = "error"
        repo.update_build_run(run)
        raise HTTPException(503, f"{_QDRANT_NOT_RUNNING} ({exc})") from exc

    texts = [c.text for c in chunks]
    try:
        vectors = embedder.embed_texts(texts)
    except Exception as exc:
        run.stages[1].status = "error"
        run.stages[1].message = str(exc)
        run.status = "error"
        repo.update_build_run(run)
        raise HTTPException(503, f"Embedding failed: {exc}") from exc

    run.stages[1].status = "done"
    run.stages[1].items_processed = len(vectors)
    run.stages[1].message = f"{len(vectors)} vectors via {embedder.identity().describe()}"
    run.stages[1].finished_at = datetime.now(timezone.utc)
    run.stages.append(
        StageStatus(stage=BuildStage.INDEXING, status="running", started_at=datetime.now(timezone.utc))
    )
    try:
        store.upsert_chunks(kb_id, chunks, vectors)
    except VectorStoreError as exc:
        run.stages[2].status = "error"
        run.stages[2].message = str(exc)
        run.status = "error"
        repo.update_build_run(run)
        raise HTTPException(503, f"{_QDRANT_NOT_RUNNING} ({exc})") from exc

    # Persist chunks per-document AFTER successful upsert (SQLite then holds
    # exactly the chunks that are actually indexed in Qdrant).
    for doc_id in doc_texts:
        repo.delete_chunks_for_document(kb_id, doc_id)
    repo.create_chunks(chunks)

    # Sweep any remaining orphans (deleted documents, partial earlier builds,
    # or points whose document_id payload was missing).
    valid_chunk_ids = {c.id for c in chunks}
    try:
        swept = store.delete_orphaned_points(kb_id, valid_chunk_ids)
    except VectorStoreError as exc:
        logger.warning("Orphan sweep failed (non-fatal): %s", exc)
        swept = 0

    run.stages[2].status = "done"
    run.stages[2].items_processed = len(chunks)
    run.stages[2].message = f"{len(chunks)} indexed; {swept} stale vectors removed"
    run.stages[2].finished_at = datetime.now(timezone.utc)
    run.status = "done"
    run.finished_at = datetime.now(timezone.utc)
    repo.update_build_run(run)
    # Record WHICH embedding model produced the current vectors so retrieval
    # can refuse to run after a model change (no silent incompatible queries).
    kb.status = KBStatus.READY
    kb.embedding_identity = asdict(embedder.identity())
    # V3: record the chunking configuration that produced the current index.
    kb.chunking_strategy = chunker.name
    kb.chunking_config = {
        "target_size": payload.target_size,
        "overlap": payload.overlap,
        "config_version": "v1",
    }
    kb.updated_at = datetime.now(timezone.utc)
    repo.update_kb(kb)
    return run
