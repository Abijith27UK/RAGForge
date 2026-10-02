"""Build pipeline routes: ingestion -> chunking -> embedding -> indexing.

The chunk/embed/index work lives in app.services.indexing.document_indexer so
the full build here and the per-document incremental path in routes_documents
cannot drift apart in how stale vectors are handled.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone

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
from app.services.embeddings.provider import EmbeddingError, create_embedding_provider
from app.services.indexing.document_indexer import IndexingError, index_documents
from app.services.ingestion.ingestion import IngestionError, ingest_source
from app.services.vector_store.factory import create_vector_store
from app.utils.ids import new_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["build"])


class IngestRequest(BaseModel):
    source_ids: list[str] = Field(default_factory=list, description="Empty = all ACCEPTED sources")
    document_ids: list[str] = Field(
        default_factory=list,
        description="Restrict ingestion to documents whose source is in source_ids",
    )


class IndexRequest(BaseModel):
    chunker: str = "section-aware"
    target_size: int = Field(default=1200, ge=200, le=8000)
    overlap: int = Field(default=150, ge=0, le=1000)


@router.post("/{kb_id}/ingest", response_model=BuildRun)
def ingest(kb_id: str, payload: IngestRequest, repo: Repository = Depends(get_repo)):
    """Download + parse ACCEPTED EXTERNAL sources.

    USER_PROVIDED sources are already parsed at upload time and are therefore
    skipped here — they are (re)indexed through /documents/{id}/index.
    """
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    sources = repo.list_sources(kb_id)
    if payload.source_ids:
        wanted = set(payload.source_ids)
        sources = [s for s in sources if s.id in wanted]
    else:  # default: all ACCEPTED sources
        sources = [s for s in sources if s.decision == SourceDecision.ACCEPT]

    user_provided = [s for s in sources if s.user_provided]
    external = [s for s in sources if not s.user_provided]
    if user_provided and not external:
        raise HTTPException(
            400,
            "All selected sources are user-provided and were already parsed at upload time. "
            "Use POST /documents/{document_id}/index to index them.",
        )
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
    for source in external:
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
    """Full re-index: chunk + embed + index every READY/PARSED document.

    Stale-vector handling (unchanged contract, now implemented by the shared
    indexer): vectors for every re-indexed document are deleted from Qdrant
    BEFORE new ones are upserted — re-chunking generates new chunk IDs, so
    upserting alone would leave orphaned points retrievable forever — and any
    leftover orphans are swept at the end via delete_orphaned_points.
    """
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    documents = repo.list_documents(kb_id)
    if not documents:
        raise HTTPException(400, "No documents ingested yet. Upload files or run ingestion first.")

    settings = get_settings()
    try:
        embedder = create_embedding_provider(settings)
    except EmbeddingError as exc:
        raise HTTPException(503, str(exc)) from exc

    store = create_vector_store(settings, backend=kb.vector_backend)
    started = datetime.now(timezone.utc)
    run = BuildRun(
        id=new_id("build"),
        kb_id=kb_id,
        stages=[
            StageStatus(stage=BuildStage.CHUNKING, status="running", started_at=started),
            StageStatus(stage=BuildStage.EMBEDDING, status="pending"),
            StageStatus(stage=BuildStage.INDEXING, status="pending"),
        ],
    )
    repo.create_build_run(run)

    try:
        outcome = index_documents(
            repo=repo,
            kb=kb,
            documents=documents,
            store=store,
            embedder=embedder,
            chunker_name=payload.chunker,
            target_size=payload.target_size,
            overlap=payload.overlap,
            full_build=True,
            kb_version=kb.version + 1,
        )
    except ValueError as exc:  # unknown chunking strategy
        for stage in run.stages:
            stage.status = "error"
            stage.message = str(exc)
        run.status = "error"
        run.finished_at = datetime.now(timezone.utc)
        repo.update_build_run(run)
        raise HTTPException(422, str(exc)) from exc
    except IndexingError as exc:
        message = str(exc)
        for stage in run.stages:
            if stage.status == "running":
                stage.status = "error"
                stage.message = message
            stage.finished_at = datetime.now(timezone.utc)
        run.status = "error"
        run.finished_at = datetime.now(timezone.utc)
        repo.update_build_run(run)
        kb.status = KBStatus.ERROR
        repo.update_kb(kb)
        status = 503 if "Qdrant" in message or "vector" in message.lower() else 400
        raise HTTPException(status, message) from exc

    finished = datetime.now(timezone.utc)
    run.stages[0].status = "done"
    run.stages[0].items_processed = outcome.chunk_count
    run.stages[0].message = f"{outcome.chunk_count} chunks from {outcome.documents_indexed} document(s)"
    run.stages[0].finished_at = finished
    run.stages[1].status = "done"
    run.stages[1].items_processed = outcome.vectors_indexed
    run.stages[1].message = f"{outcome.vectors_indexed} vectors via {embedder.identity().describe()}"
    run.stages[1].finished_at = finished
    run.stages[2].status = "done"
    run.stages[2].items_processed = outcome.chunk_count
    run.stages[2].message = (
        f"{outcome.chunk_count} indexed; {outcome.stale_label()} stale vectors removed"
    )
    run.stages[2].finished_at = finished
    run.status = "done"
    run.finished_at = finished
    repo.update_build_run(run)

    # Record WHICH embedding model produced the current vectors so retrieval
    # can refuse to run after a model change (no silent incompatible queries).
    kb.status = KBStatus.READY
    kb.embedding_identity = asdict(embedder.identity())
    kb.chunking_strategy = payload.chunker
    kb.chunking_config = {
        "target_size": payload.target_size,
        "overlap": payload.overlap,
        "config_version": "v1",
    }
    kb.version += 1
    kb.last_build_at = finished
    kb.updated_at = finished
    repo.update_kb(kb)
    return run