"""Chunk -> embed -> index for a SET of documents, incrementally.

Both the full build (`POST /{kb_id}/index`) and the incremental path
(`POST /{kb_id}/documents/{doc_id}/index`) call this one function so they can
never drift apart in how stale vectors are handled.

Stale-vector contract (V4 Phase 6):
  1. Delete the vectors of the documents being (re)indexed BEFORE upserting.
     Re-chunking always generates new chunk IDs, so upserting alone would
     leave orphaned points retrievable forever.
  2. Replace the document's chunk rows only AFTER a successful upsert, so
     SQLite always holds exactly the chunks present in the vector store.
  3. `full_build=True` additionally sweeps any orphan left over from deleted
     documents or older partial builds. Incremental runs deliberately do NOT
     sweep: they must not pay the cost of scanning the whole collection, and
     they must never touch vectors belonging to documents they did not
     process.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.repositories.sqlite_repo import Repository
from app.schemas.models import Chunk, Document, DocumentStatus, KnowledgeBase, Source
from app.services.chunking.chunker import get_chunker
from app.services.embeddings.provider import EmbeddingProvider
from app.services.vector_store.qdrant_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class IndexingError(RuntimeError):
    """Recoverable indexing failure; callers map it to a user-facing message."""


@dataclass
class IndexOutcome:
    document_ids: list[str] = field(default_factory=list)
    documents_indexed: int = 0
    documents_skipped: list[dict] = field(default_factory=list)
    chunk_count: int = 0
    vectors_indexed: int = 0
    #: -1 means "deletion was performed but the count could not be confirmed".
    stale_vectors_removed: int = 0
    seconds: float = 0.0

    def stale_label(self) -> str:
        """Honest human label: never present an unconfirmed count as zero."""
        if self.stale_vectors_removed < 0:
            return "unknown"
        return str(self.stale_vectors_removed)

    def as_dict(self) -> dict:
        return {
            "document_ids": self.document_ids,
            "documents_indexed": self.documents_indexed,
            "documents_skipped": self.documents_skipped,
            "chunk_count": self.chunk_count,
            "vectors_indexed": self.vectors_indexed,
            "stale_vectors_removed": self.stale_vectors_removed,
            "stale_vectors_removed_label": self.stale_label(),
            "seconds": round(self.seconds, 3),
        }


def read_parsed_text(doc: Document) -> str | None:
    """Read a document's normalized text from its sidecar .parsed.txt file."""
    if not doc.file_path:
        return None
    try:
        return Path(doc.file_path).read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read parsed text for %s: %s", doc.id, exc)
        return None


def index_documents(
    *,
    repo: Repository,
    kb: KnowledgeBase,
    documents: list[Document],
    store: VectorStore,
    embedder: EmbeddingProvider,
    chunker_name: str,
    target_size: int,
    overlap: int,
    full_build: bool = False,
    kb_version: int | None = None,
) -> IndexOutcome:
    """Chunk, embed and index the given documents, replacing their old vectors.

    Raises IndexingError with a user-facing message; never leaves the vector
    store half-updated for the documents it did manage to process.
    """
    started = time.perf_counter()
    outcome = IndexOutcome()

    if not documents:
        raise IndexingError("No documents to index.")

    chunker = get_chunker(chunker_name)
    chunking_config = {
        "target_size": target_size,
        "overlap": overlap,
        "config_version": "v1",
    }

    # -- read + chunk ---------------------------------------------------
    chunks: list[Chunk] = []
    indexed_docs: list[Document] = []
    for doc in documents:
        text = read_parsed_text(doc)
        if not text:
            outcome.documents_skipped.append(
                {"document_id": doc.id, "reason": "parsed text not found on disk"}
            )
            continue
        source: Source | None = repo.get_source(kb.id, doc.source_id)
        if source is None:
            outcome.documents_skipped.append(
                {"document_id": doc.id, "reason": f"source {doc.source_id} no longer exists"}
            )
            continue
        try:
            doc_chunks = chunker.chunk(
                document=doc,
                text=text,
                source=source,
                target_size=target_size,
                overlap=overlap,
                domain=kb.domain,
                chunking_config=chunking_config,
                kb_version=kb_version if kb_version is not None else kb.version,
            )
        except Exception as exc:
            logger.exception("Chunking failed for document %s", doc.id)
            outcome.documents_skipped.append({"document_id": doc.id, "reason": f"chunking failed: {exc}"})
            continue
        chunks.extend(doc_chunks)
        indexed_docs.append(doc)

    if not chunks:
        detail = "; ".join(
            f"{s['document_id']}: {s['reason']}" for s in outcome.documents_skipped[:5]
        )
        raise IndexingError(
            "No chunks were produced for these documents. "
            + (f"Reasons: {detail}" if detail else "Check that parsed text exists on disk.")
        )

    # -- ensure collection before deleting anything ----------------------
    try:
        store.ensure_collection(kb.id, embedder.dimensions)
    except VectorStoreError as exc:
        raise IndexingError(str(exc)) from exc

    document_ids = [d.id for d in indexed_docs]
    outcome.document_ids = document_ids

    # -- delete stale vectors BEFORE upserting --------------------------
    try:
        outcome.stale_vectors_removed = store.delete_document_vectors(kb.id, document_ids)
    except (VectorStoreError, NotImplementedError) as exc:
        raise IndexingError(
            f"Cannot delete stale vectors for these documents: {exc}. "
            "Refusing to index, because re-chunking would leave orphaned points."
        ) from exc

    # -- embed + upsert -------------------------------------------------
    try:
        vectors = embedder.embed_texts([c.text for c in chunks])
    except Exception as exc:
        raise IndexingError(f"Embedding failed: {exc}") from exc

    try:
        store.upsert_chunks(kb.id, chunks, vectors)
    except VectorStoreError as exc:
        raise IndexingError(str(exc)) from exc

    # -- persist chunks only after a successful upsert -------------------
    for doc_id in document_ids:
        repo.delete_chunks_for_document(kb.id, doc_id)
    repo.create_chunks(chunks)

    now = datetime.now(timezone.utc)
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.document_id] = counts.get(c.document_id, 0) + 1
    for doc in indexed_docs:
        doc.chunk_count = counts.get(doc.id, 0)
        doc.status = DocumentStatus.READY
        doc.indexed_at = now
        repo.update_document(doc)

    if full_build:
        valid = {c.id for c in chunks}
        try:
            swept = store.delete_orphaned_points(kb.id, valid)
        except (VectorStoreError, NotImplementedError) as exc:
            logger.warning("Orphan sweep failed (non-fatal): %s", exc)
            swept = 0
        # -1 (unknown) must not cancel out a known count.
        if outcome.stale_vectors_removed >= 0 and swept >= 0:
            outcome.stale_vectors_removed += swept

    outcome.chunk_count = len(chunks)
    outcome.vectors_indexed = len(vectors)
    outcome.documents_indexed = len(indexed_docs)
    outcome.seconds = time.perf_counter() - started
    logger.info(
        "Indexed %d document(s) -> %d chunks into %s (%s stale vectors removed)",
        outcome.documents_indexed, outcome.chunk_count, kb.id, outcome.stale_label(),
    )
    return outcome


def chunker_names() -> list[str]:
    from app.services.chunking.chunker import CHUNKING_REGISTRY

    return CHUNKING_REGISTRY.names()


def count_chunks_for_documents(repo: Repository, kb_id: str, document_ids: list[str]) -> dict[str, int]:
    """Chunk counts per document, computed from the persisted chunk rows."""
    counts: dict[str, int] = {doc_id: 0 for doc_id in document_ids}
    if not document_ids:
        return counts
    for doc_id in document_ids:
        row = repo._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE kb_id = ? AND document_id = ?", (kb_id, doc_id)
        ).fetchone()
        counts[doc_id] = int(row["n"]) if row else 0
    return counts


def chunks_of_document(repo: Repository, kb_id: str, document_id: str, limit: int = 500) -> list[Chunk]:
    return repo.list_chunks(kb_id, document_id=document_id, limit=limit)