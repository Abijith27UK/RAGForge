"""Document library routes (V4 Phases 2, 5, 6).

Knowledge Base -> Documents. This router owns the USER-PROVIDED workflow:
upload files, inspect them, rebuild them, index them incrementally and delete
them — always keeping SQLite and the vector store in agreement.

Privacy: uploaded bytes are parsed, chunked and embedded locally. Nothing here
calls an external LLM/API. `ALLOW_EXTERNAL_LLM_FOR_USER_DOCUMENTS` guards that
contract explicitly.
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
from app.schemas.models import (
    BuildRun,
    BuildStage,
    Chunk,
    Document,
    DocumentDetail,
    DocumentStatus,
    KBStatus,
    Source,
    SourceType,
    StageStatus,
    UploadedFileResult,
    UploadResult,
)
from app.services.embeddings.provider import EmbeddingError, create_embedding_provider
from app.services.indexing.document_indexer import (
    IndexingError,
    chunks_of_document,
    count_chunks_for_documents,
    index_documents,
)
from app.services.ingestion.ingestion import IngestionError, ingest_uploaded_bytes, reparse_document
from app.services.ingestion.parsers import supported_upload_extensions
from app.services.ingestion.upload import sanitize_file_name, validate_upload
from app.services.source_quality.user_scorer import assess_user_upload
from app.services.vector_store.factory import create_vector_store
from app.utils.ids import new_id
from app.utils.text import sha256_bytes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["documents"])

_QDRANT_NOT_RUNNING = (
    "Qdrant is not reachable. Start it (see README: qdrant/qdrant.exe) and try again."
)

#: Maps an accepted extension to the SourceType stored on the document/source.
_EXTENSION_SOURCE_TYPE = {
    ".pdf": SourceType.PDF,
    ".pptx": SourceType.PRESENTATION,
    ".pptm": SourceType.PRESENTATION,
    ".ppt": SourceType.PRESENTATION,
    ".docx": SourceType.DOCUMENT,
    ".docm": SourceType.DOCUMENT,
    ".txt": SourceType.TEXT,
    ".md": SourceType.TEXT,
    ".markdown": SourceType.TEXT,
    ".mdown": SourceType.TEXT,
    ".html": SourceType.WEB_PAGE,
    ".htm": SourceType.WEB_PAGE,
    ".xhtml": SourceType.WEB_PAGE,
}


class RebuildRequest(BaseModel):
    reindex: bool = Field(default=True, description="Re-chunk, re-embed and re-index after rebuilding")


class DocumentIndexRequest(BaseModel):
    chunker: str = "section-aware"
    target_size: int = Field(default=1200, ge=200, le=8000)
    overlap: int = Field(default=150, ge=0, le=1000)


def _kb_or_404(repo: Repository, kb_id: str):
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


def _document_or_404(repo: Repository, kb_id: str, document_id: str) -> Document:
    doc = repo.get_document(kb_id, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc


def _unit_count(doc: Document) -> int | None:
    return doc.page_count or doc.slide_count or doc.section_count or None


def _delete_vectors(kb_id: str, document_ids: list[str]) -> int:
    """Best-effort vector deletion that never hides the real failure reason."""
    if not document_ids:
        return 0
    try:
        store = create_vector_store(get_settings())
        return store.delete_document_vectors(kb_id, document_ids)
    except Exception as exc:
        logger.warning("Could not delete vectors for %s: %s", document_ids, exc)
        return 0


def _build_source(kb_id: str, file_name: str, size: int, extension: str) -> Source:
    """Create the USER_PROVIDED source that backs an uploaded file.

    `url` is a stable pseudo-URL (``upload://``) because a local file has no
    public address; it is never probed and never presented as a web source.
    """
    return Source(
        id=new_id("src"),
        kb_id=kb_id,
        url=f"upload://{file_name}",
        title=file_name,
        source_type=_EXTENSION_SOURCE_TYPE.get(extension, SourceType.USER_PROVIDED),
        publisher=None,
        discovered_via="user-upload",
        user_provided=True,
        provenance="user_upload",
        file_name=file_name,
        file_size=size,
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/{kb_id}/documents/upload", response_model=UploadResult, status_code=201)
async def upload_documents(
    kb_id: str,
    files: list[UploadFile] = File(..., description="One or more documents to ingest"),
    index: bool = Form(default=False, description="Parse + chunk + embed + index immediately"),
    chunker: str = Form(default="section-aware"),
    target_size: int = Form(default=1200),
    overlap: int = Form(default=150),
    repo: Repository = Depends(get_repo),
):
    """Upload user documents into the knowledge base.

    Every file is validated (name safety, size, magic bytes, container
    integrity) before anything is written to disk. Duplicates are reported and
    the existing document is kept — user documents are NEVER silently replaced.
    """
    kb = _kb_or_404(repo, kb_id)
    settings = get_settings()
    if len(files) > settings.max_upload_files_per_request:
        raise HTTPException(
            400,
            f"Too many files in one request ({len(files)}); "
            f"limit is {settings.max_upload_files_per_request}.",
        )

    result = UploadResult(kb_id=kb_id, source_mode=kb.source_mode)
    new_documents: list[Document] = []

    for upload in files:
        raw_name = upload.filename or "upload"
        safe_name = sanitize_file_name(raw_name)
        data = await upload.read()
        extension = Path(safe_name).suffix.lower()

        validation = validate_upload(
            safe_name, data, max_bytes=settings.max_upload_bytes,
            allowed_extensions=supported_upload_extensions(),
        )
        if not validation.ok:
            result.files.append(
                UploadedFileResult(
                    file_name=safe_name, size_bytes=len(data),
                    status="rejected", message=validation.error,
                )
            )
            result.rejected += 1
            continue

        content_hash = sha256_bytes(data)
        existing = repo.find_document_by_hash(kb_id, content_hash)
        if existing is not None:
            # Never replace silently: report the existing document and stop.
            source = repo.get_source(kb_id, existing.source_id)
            result.files.append(
                UploadedFileResult(
                    file_name=safe_name, size_bytes=len(data), status="duplicate",
                    message=(
                        "Identical content already exists in this knowledge base "
                        f"({existing.file_name or existing.id}); the existing document was kept."
                    ),
                    document=existing, source=source, duplicate_of=existing.id,
                )
            )
            result.duplicates += 1
            continue

        source = _build_source(kb_id, safe_name, len(data), extension)
        repo.create_source(source)

        try:
            doc = ingest_uploaded_bytes(
                kb_id=kb_id,
                source=source,
                upload_dir=settings.kb_upload_dir(kb_id),
                file_name=safe_name,
                data=data,
                mime_type=upload.content_type,
            )
        except IngestionError as exc:
            # Keep the failure visible in the library rather than silently dropping it.
            doc = Document(
                id=new_id("doc"),
                kb_id=kb_id,
                source_id=source.id,
                url=source.url,
                title=safe_name,
                source_type=source.source_type,
                file_path=None,
                content_hash=content_hash,
                text_length=0,
                parse_error=str(exc),
                status=DocumentStatus.FAILED,
                user_provided=True,
                file_name=safe_name,
                file_size=len(data),
                mime_type=upload.content_type,
                parser=validation.detected_format or extension.lstrip("."),
            )
            assessment = assess_user_upload(
                file_valid=True, parse_ok=False, extracted_chars=0, parse_error=str(exc)
            )
            source.integrity = assessment.model_dump(mode="json")
            source.quality = assessment.model_dump(mode="json")
            source.trust_score = assessment.score
            source.decision = assessment.decision
            repo.update_source(source)
            repo.create_document(doc)
            result.files.append(
                UploadedFileResult(
                    file_name=safe_name, size_bytes=len(data), status="failed",
                    message=str(exc), document=doc, source=source,
                    integrity=assessment.model_dump(mode="json"),
                )
            )
            result.failed += 1
            continue

        assessment = assess_user_upload(
            file_valid=True,
            parse_ok=True,
            extracted_chars=doc.text_length,
            unit_count=_unit_count(doc),
            warnings=validation.warnings,
        )
        source.trust_score = assessment.score
        source.decision = assessment.decision
        source.integrity = assessment.model_dump(mode="json")
        # The Sources page renders `quality` uniformly for both modes, so the
        # integrity assessment is exposed there too (assessed_by discloses which
        # model produced it).
        source.quality = assessment.model_dump(mode="json")
        repo.update_source(source)

        doc.status = (
            DocumentStatus.PARSED if assessment.decision.value != "REJECT" else DocumentStatus.FAILED
        )
        repo.create_document(doc)
        new_documents.append(doc)
        result.files.append(
            UploadedFileResult(
                file_name=safe_name, size_bytes=len(data), status="uploaded",
                message=f"Parsed {doc.text_length} characters"
                + (f" from {_unit_count(doc)} page(s)/slide(s)/section(s)" if _unit_count(doc) else ""),
                document=doc, source=source,
                integrity=assessment.model_dump(mode="json"),
            )
        )
        result.uploaded += 1

    if new_documents:
        kb.status = KBStatus.BUILDING
        kb.updated_at = datetime.now(timezone.utc)
        repo.update_kb(kb)

    if index and new_documents:
        try:
            outcome = _index_documents(
                repo=repo, kb=kb, documents=new_documents,
                chunker=chunker, target_size=target_size, overlap=overlap,
            )
            result.indexed = True
            result.indexing = outcome
            for f in result.files:
                if f.status == "uploaded" and f.document and f.document.id in outcome["document_ids"]:
                    f.status = "indexed"
        except (IndexingError, EmbeddingError, HTTPException) as exc:
            message = exc.detail if isinstance(exc, HTTPException) else str(exc)
            result.indexing = {"error": message, "documents_indexed": 0}
            result.indexed = False
            logger.warning("Incremental index after upload failed: %s", message)

    return result


# ---------------------------------------------------------------------------
# Indexing helper shared by upload/index/rebuild/replace
# ---------------------------------------------------------------------------

def _index_documents(
    *,
    repo: Repository,
    kb,
    documents: list[Document],
    chunker: str,
    target_size: int,
    overlap: int,
):
    settings = get_settings()
    try:
        embedder = create_embedding_provider(settings)
    except EmbeddingError as exc:
        raise HTTPException(503, str(exc)) from exc
    store = create_vector_store(settings, backend=kb.vector_backend)
    for doc in documents:
        doc.status = DocumentStatus.CHUNKING
        repo.update_document(doc)
    try:
        outcome = index_documents(
            repo=repo, kb=kb, documents=documents, store=store, embedder=embedder,
            chunker_name=chunker, target_size=target_size, overlap=overlap,
            full_build=False, kb_version=kb.version + 1,
        )
    except IndexingError as exc:
        for doc in documents:
            doc.status = DocumentStatus.FAILED
            doc.parse_error = str(exc)
            repo.update_document(doc)
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:  # unknown chunker
        raise HTTPException(422, str(exc)) from exc

    kb.version += 1
    kb.last_build_at = datetime.now(timezone.utc)
    kb.updated_at = kb.last_build_at
    repo.update_kb(kb)
    return outcome.as_dict()


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

@router.get("/{kb_id}/documents/{document_id}", response_model=DocumentDetail)
def inspect_document(kb_id: str, document_id: str, repo: Repository = Depends(get_repo)):
    kb = _kb_or_404(repo, kb_id)
    doc = _document_or_404(repo, kb_id, document_id)
    source = repo.get_source(kb_id, doc.source_id)
    counts = count_chunks_for_documents(repo, kb_id, [document_id])
    return DocumentDetail(
        document=doc,
        source=source,
        chunk_count=counts.get(document_id, doc.chunk_count),
        integrity=(source.integrity if source else None),
        kb_version=kb.version,
        retrieval_enabled=doc.status is DocumentStatus.READY and counts.get(document_id, 0) > 0,
    )


@router.get("/{kb_id}/documents/{document_id}/text")
def document_text(
    kb_id: str,
    document_id: str,
    offset: int = 0,
    limit: int = 20_000,
    repo: Repository = Depends(get_repo),
):
    """Extracted (normalized) text, for inspection. Never re-parsed here."""
    _kb_or_404(repo, kb_id)
    doc = _document_or_404(repo, kb_id, document_id)
    if not doc.file_path:
        raise HTTPException(404, "This document has no extracted text (it failed to parse).")
    try:
        full = Path(doc.file_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"Could not read extracted text: {exc}") from exc
    window = full[offset : offset + limit]
    return {
        "document_id": doc.id,
        "file_name": doc.file_name,
        "parser": doc.parser,
        "char_count": len(full),
        "offset": offset,
        "returned": len(window),
        "truncated": offset + limit < len(full),
        "text": window,
    }


@router.get("/{kb_id}/documents/{document_id}/chunks", response_model=list[Chunk])
def document_chunks(
    kb_id: str, document_id: str, limit: int = 500, repo: Repository = Depends(get_repo)
):
    _kb_or_404(repo, kb_id)
    _document_or_404(repo, kb_id, document_id)
    return chunks_of_document(repo, kb_id, document_id, limit=min(limit, 1000))


# ---------------------------------------------------------------------------
# Rebuild / re-index / replace
# ---------------------------------------------------------------------------

@router.post("/{kb_id}/documents/{document_id}/rebuild", response_model=BuildRun)
def rebuild_document(
    kb_id: str, document_id: str, payload: RebuildRequest, repo: Repository = Depends(get_repo)
):
    """Re-run the parser over the stored original bytes, then re-index.

    This never downloads or replaces anything: it only re-derives the normalized
    text (and therefore the chunks) from the file the user actually supplied.
    """
    kb = _kb_or_404(repo, kb_id)
    doc = _document_or_404(repo, kb_id, document_id)
    started = datetime.now(timezone.utc)
    run = BuildRun(
        id=new_id("build"), kb_id=kb_id,
        stages=[StageStatus(stage=BuildStage.INGESTION, status="running", started_at=started)],
    )
    repo.create_build_run(run)

    doc.status = DocumentStatus.PARSING
    repo.update_document(doc)
    try:
        text, meta = reparse_document(doc)
    except IngestionError as exc:
        doc.status = DocumentStatus.FAILED
        doc.parse_error = str(exc)
        repo.update_document(doc)
        run.stages[0].status = "error"
        run.stages[0].message = str(exc)
        run.stages[0].finished_at = datetime.now(timezone.utc)
        run.status = "error"
        run.finished_at = datetime.now(timezone.utc)
        repo.update_build_run(run)
        raise HTTPException(422, str(exc)) from exc

    doc.text_length = len(text)
    doc.page_count = meta.get("page_count", doc.page_count)
    doc.slide_count = meta.get("slide_count", doc.slide_count)
    doc.section_count = meta.get("section_count", doc.section_count)
    doc.parser = meta.get("format")
    doc.parse_metadata = {**doc.parse_metadata, **meta}
    doc.parse_error = None
    doc.status = DocumentStatus.PARSED
    doc.indexed_at = None
    repo.update_document(doc)

    run.stages[0].status = "done"
    run.stages[0].items_processed = 1
    run.stages[0].message = f"Re-parsed {len(text)} characters with parser '{doc.parser}'"
    run.stages[0].finished_at = datetime.now(timezone.utc)

    if payload.reindex:
        run.stages.append(
            StageStatus(stage=BuildStage.INDEXING, status="running", started_at=datetime.now(timezone.utc))
        )
        try:
            outcome = _index_documents(
                repo=repo, kb=kb, documents=[doc],
                chunker=kb.chunking_strategy or "section-aware",
                target_size=int(kb.chunking_config.get("target_size", 1200)),
                overlap=int(kb.chunking_config.get("overlap", 150)),
            )
        except HTTPException as exc:
            run.stages[1].status = "error"
            run.stages[1].message = str(exc.detail)
            run.status = "error"
            run.finished_at = datetime.now(timezone.utc)
            repo.update_build_run(run)
            raise
        run.stages[1].status = "done"
        run.stages[1].items_processed = outcome["chunk_count"]
        run.stages[1].message = (
            f"{outcome['chunk_count']} chunks; "
            f"{outcome['stale_vectors_removed_label']} stale vectors removed"
        )
        run.stages[1].finished_at = datetime.now(timezone.utc)

    run.status = "done"
    run.finished_at = datetime.now(timezone.utc)
    repo.update_build_run(run)
    kb.status = KBStatus.READY
    repo.update_kb(kb)
    return run


@router.post("/{kb_id}/documents/{document_id}/index", response_model=BuildRun)
def index_single_document(
    kb_id: str, document_id: str, payload: DocumentIndexRequest, repo: Repository = Depends(get_repo)
):
    """Incrementally re-index ONE document (chunk -> embed -> index).

    Other documents in the knowledge base are never re-embedded and their
    vectors are never touched.
    """
    kb = _kb_or_404(repo, kb_id)
    doc = _document_or_404(repo, kb_id, document_id)
    if doc.status is DocumentStatus.FAILED:
        raise HTTPException(422, "Document failed to parse; rebuild it before indexing.")

    started = datetime.now(timezone.utc)
    run = BuildRun(
        id=new_id("build"), kb_id=kb_id,
        stages=[
            StageStatus(stage=BuildStage.CHUNKING, status="running", started_at=started),
            StageStatus(stage=BuildStage.INDEXING, status="pending"),
        ],
    )
    repo.create_build_run(run)
    try:
        outcome = _index_documents(
            repo=repo, kb=kb, documents=[doc],
            chunker=payload.chunker, target_size=payload.target_size, overlap=payload.overlap,
        )
    except HTTPException as exc:
        for stage in run.stages:
            if stage.status == "running":
                stage.status = "error"
                stage.message = str(exc.detail)
                stage.finished_at = datetime.now(timezone.utc)
        run.status = "error"
        run.finished_at = datetime.now(timezone.utc)
        repo.update_build_run(run)
        raise

    run.stages[0].status = "done"
    run.stages[0].items_processed = outcome["chunk_count"]
    run.stages[0].message = f"{outcome['chunk_count']} chunks from 1 document"
    run.stages[0].finished_at = datetime.now(timezone.utc)
    run.stages[1].status = "done"
    run.stages[1].items_processed = outcome["vectors_indexed"]
    run.stages[1].message = (
        f"{outcome['vectors_indexed']} vectors indexed; "
        f"{outcome['stale_vectors_removed_label']} stale vectors removed"
    )
    run.stages[1].finished_at = datetime.now(timezone.utc)
    run.status = "done"
    run.finished_at = datetime.now(timezone.utc)
    repo.update_build_run(run)
    kb.status = KBStatus.READY
    kb.chunking_strategy = payload.chunker
    kb.chunking_config = {
        "target_size": payload.target_size, "overlap": payload.overlap, "config_version": "v1",
    }
    repo.update_kb(kb)
    return run


@router.post("/{kb_id}/documents/{document_id}/replace")
async def replace_document(
    kb_id: str,
    document_id: str,
    file: UploadFile = File(..., description="The new version of this document"),
    reason: str = Form(default=""),
    index: bool = Form(default=True),
    chunker: str = Form(default="section-aware"),
    target_size: int = Form(default=1200),
    overlap: int = Form(default=150),
    repo: Repository = Depends(get_repo),
):
    """Replace a document's file with a new version.

    Behaviour (Phase 6): the OLD document's vectors are deleted, the new file
    is ingested as a new document version that supersedes the old one, and the
    old document is retained as history (marked superseded) rather than being
    silently overwritten. Historical experiment artifacts are never destroyed.
    """
    kb = _kb_or_404(repo, kb_id)
    old = _document_or_404(repo, kb_id, document_id)
    settings = get_settings()

    safe_name = sanitize_file_name(file.filename or "upload")
    data = await file.read()
    extension = Path(safe_name).suffix.lower()
    validation = validate_upload(
        safe_name, data, max_bytes=settings.max_upload_bytes,
        allowed_extensions=supported_upload_extensions(),
    )
    if not validation.ok:
        raise HTTPException(400, validation.error)
    content_hash = sha256_bytes(data)
    if content_hash == old.content_hash:
        raise HTTPException(409, "The uploaded file is byte-identical to the current version.")

    conflicting = repo.find_document_by_hash(kb_id, content_hash)
    if conflicting is not None and conflicting.id != old.id:
        raise HTTPException(
            409,
            f"Identical content already exists in this knowledge base as document "
            f"{conflicting.id}. Remove it first if you really want this file here.",
        )

    source = _build_source(kb_id, safe_name, len(data), extension)
    repo.create_source(source)
    try:
        doc = ingest_uploaded_bytes(
            kb_id=kb_id, source=source, upload_dir=settings.kb_upload_dir(kb_id),
            file_name=safe_name, data=data, mime_type=file.content_type,
        )
    except IngestionError as exc:
        repo.delete_source(kb_id, source.id)
        raise HTTPException(422, f"Could not parse the replacement file: {exc}") from exc

    doc.document_version = (old.document_version or 1) + 1
    doc.replaces_document_id = old.id
    assessment = assess_user_upload(
        file_valid=True, parse_ok=True, extracted_chars=doc.text_length, unit_count=_unit_count(doc)
    )
    source.trust_score = assessment.score
    source.decision = assessment.decision
    source.integrity = assessment.model_dump(mode="json")
    source.quality = assessment.model_dump(mode="json")
    repo.update_source(source)
    repo.create_document(doc)

    # The superseded document stops being retrievable but stays in history.
    # Its vectors MUST go: an indexed document that stays in the vector store
    # would keep answering queries with content the user has replaced.
    _delete_vectors(kb_id, [old.id])
    old.status = DocumentStatus.FAILED
    old.parse_error = f"Superseded by document {doc.id} (version {doc.document_version})"
    old.chunk_count = 0
    old.indexed_at = None
    repo.update_document(old)

    indexing = None
    if index:
        try:
            indexing = _index_documents(
                repo=repo, kb=kb, documents=[doc],
                chunker=chunker, target_size=target_size, overlap=overlap,
            )
        except HTTPException as exc:
            return {"document": doc, "superseded": old.id, "indexing": {"error": exc.detail}}
    else:
        kb.status = KBStatus.BUILDING
        repo.update_kb(kb)

    return {
        "document": doc,
        "superseded": old.id,
        "reason": reason,
        "indexing": indexing,
    }


@router.delete("/{kb_id}/documents/{document_id}")
def delete_document(
    kb_id: str, document_id: str, delete_file: bool = True, repo: Repository = Depends(get_repo)
):
    """Remove a document, its chunks AND its vectors.

    Stale vectors are the whole point: deleting only the database rows would
    leave the chunks permanently retrievable from the vector store.
    """
    _kb_or_404(repo, kb_id)
    doc = _document_or_404(repo, kb_id, document_id)

    removed_vectors = 0
    vector_error = ""
    try:
        store = create_vector_store(get_settings())
        removed_vectors = store.delete_document_vectors(kb_id, [document_id])
    except Exception as exc:  # Qdrant may be offline; report it honestly
        vector_error = str(exc)
        logger.warning("Could not delete vectors for document %s: %s", document_id, exc)

    paths = [p for p in (doc.file_path, doc.raw_file_path) if p]
    repo.delete_document(kb_id, document_id)
    if delete_file:
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - best effort cleanup
                logger.warning("Could not remove file %s: %s", path, exc)

    # -1 == "the delete ran but the count could not be confirmed"; never render
    # that as "0 removed", which would claim the vectors are gone when we
    # cannot prove it.
    return {
        "deleted": document_id,
        "file_name": doc.file_name,
        "vectors_removed": removed_vectors,
        "vectors_removed_label": "unknown" if removed_vectors < 0 else str(removed_vectors),
        "vectors_removal_confirmed": removed_vectors >= 0 and not vector_error,
        "vector_store_error": vector_error or None,
    }


# ---------------------------------------------------------------------------
# Library summary
# ---------------------------------------------------------------------------

@router.get("/{kb_id}/document-library")
def document_library(kb_id: str, repo: Repository = Depends(get_repo)):
    """Compact library summary used by the Document Library page header."""
    _kb_or_404(repo, kb_id)
    documents = repo.list_documents(kb_id)
    counts = count_chunks_for_documents(repo, kb_id, [d.id for d in documents])
    by_status: dict[str, int] = {}
    for doc in documents:
        by_status[doc.status.value] = by_status.get(doc.status.value, 0) + 1
    return {
        "kb_id": kb_id,
        "documents": len(documents),
        "chunks": sum(counts.values()),
        "user_provided": sum(1 for d in documents if d.user_provided),
        "external": sum(1 for d in documents if not d.user_provided),
        "by_status": by_status,
        "supported_extensions": supported_upload_extensions(),
        "max_upload_bytes": get_settings().max_upload_bytes,
    }