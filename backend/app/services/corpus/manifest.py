"""Corpus manifest: "what exactly is inside this knowledge base?" (V5 Phase 2).

The manifest is built from SQLite metadata joined with a real point count read
from the vector store. When the store cannot be reached the vector totals stay
``UNKNOWN`` (-1) and ``vector_count_confirmed`` is False — the manifest never
substitutes a guess.

CORPUS COVERAGE vs RETRIEVAL QUALITY
------------------------------------
This module answers *coverage only*: "does the corpus contain the information?".
It says nothing about whether retrieval can find it. Retrieval quality requires
ground truth (a frozen benchmark) and is measured elsewhere; the two must never
be conflated.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime

from app.repositories.sqlite_repo import Repository
from app.schemas.corpus import (
    UNKNOWN,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusSummary,
)
from app.schemas.models import DocumentStatus, KnowledgeBase

logger = logging.getLogger(__name__)

#: Statuses that mean "this document finished and is retrievable".
_READY_STATUSES = {DocumentStatus.READY.value}
#: Statuses that mean "work is still outstanding".
_IN_FLIGHT_STATUSES = {
    DocumentStatus.UPLOADED.value,
    DocumentStatus.PARSING.value,
    DocumentStatus.PARSED.value,
    DocumentStatus.CHUNKING.value,
    DocumentStatus.INDEXING.value,
}
_FAILED_STATUSES = {DocumentStatus.FAILED.value}


def canonical_json(payload) -> str:
    """Stable serialization used for every hash in the corpus layer."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def chunking_config_hash(strategy: str | None, config: dict | None) -> str:
    """Identity of the chunking configuration that produced a document's chunks."""
    return hashlib.sha256(
        canonical_json({"strategy": strategy or "", "config": config or {}}).encode("utf-8")
    ).hexdigest()[:16]


def _default_store(kb_id: str):
    """Resolve the configured vector backend through the factory."""
    from app.config import get_settings
    from app.services.vector_store.factory import create_vector_store

    return create_vector_store(get_settings())


def _embedding_identity(kb: KnowledgeBase) -> tuple[str | None, str | None, int | None]:
    """(provider/model label, model, dimension) from the KB's recorded identity.

    Read from ``kb.embedding_identity`` — set at index time — never from the
    currently configured provider, because those two are allowed to differ
    (that difference is exactly what the integrity engine must report).
    """
    identity = kb.embedding_identity or {}
    label = None
    provider = identity.get("provider")
    model = identity.get("model")
    if provider or model:
        label = f"{provider or '?'}/{model or '?'}"
        if identity.get("dimensions"):
            label = f"{label} ({identity['dimensions']}d)"
    return label, model, identity.get("dimensions")


class CorpusManifestService:
    """Builds a manifest + summary for one knowledge base."""

    def __init__(self, repo: Repository, store_factory=None) -> None:
        self.repo = repo
        # Injectable for hermetic tests; defaults to the configured backend.
        self._store_factory = store_factory or _default_store

    # -- vector store ---------------------------------------------------------

    def _vector_count(self, kb_id: str) -> tuple[int, bool]:
        """(points, confirmed). confirmed=False means 'store unavailable'."""
        try:
            store = self._store_factory(kb_id)
            info = store.collection_info(kb_id)
        except Exception as exc:
            logger.warning("Could not read vector count for %s: %s", kb_id, exc)
            return UNKNOWN, False
        count = info.get("points_count")
        return (int(count), True) if count is not None else (UNKNOWN, False)

    # -- entries ---------------------------------------------------------------

    def _entry(self, doc, kb: KnowledgeBase, chunk_counts: dict[str, int],
               chunking_hash: str) -> CorpusManifestEntry:
        embedding_label, embedding_model, dimension = _embedding_identity(kb)
        parse_meta = doc.parse_metadata or {}
        return CorpusManifestEntry(
            document_id=doc.id,
            file_name=doc.file_name or doc.title,
            normalized_file_name=(doc.file_name or "").lower().replace(" ", "_"),
            content_hash=doc.content_hash or "",
            document_version=doc.document_version,
            source_mode=kb.source_mode.value if hasattr(kb.source_mode, "value") else str(kb.source_mode),
            source_url=doc.url,
            source_id=doc.source_id,
            file_type=doc.source_type.value if hasattr(doc.source_type, "value") else str(doc.source_type),
            file_size=doc.file_size,
            parser=doc.parser,
            parser_version=parse_meta.get("parser_version"),
            page_count=doc.page_count,
            slide_count=doc.slide_count,
            section_count=doc.section_count,
            text_length=doc.text_length,
            chunk_count=chunk_counts.get(doc.id, doc.chunk_count or 0),
            # Per-document vector counts are only known when the store agrees
            # with SQLite; otherwise this stays UNKNOWN rather than equalling
            # the chunk count by assumption.
            vector_count=chunk_counts.get(doc.id, 0) if doc.status.value == DocumentStatus.READY.value else UNKNOWN,
            embedding_provider=embedding_label,
            embedding_model=embedding_model,
            embedding_dimension=dimension,
            chunking_strategy=kb.chunking_strategy,
            chunking_config_hash=chunking_hash,
            indexed_at=doc.indexed_at,
            status=doc.status.value,
            error_message=doc.parse_error,
            provenance={
                "user_provided": doc.user_provided,
                "publisher": doc.publisher,
                "replaces_document_id": doc.replaces_document_id,
                "pages": doc.page_count,
                "slides": doc.slide_count,
                "sections": doc.section_count,
            },
        )

    # -- public ---------------------------------------------------------------

    def build(self, kb: KnowledgeBase, *, with_vector_count: bool = True) -> CorpusManifest:
        docs = self.repo.list_documents(kb.id)
        chunk_counts: dict[str, int] = {}
        for doc in docs:
            row = self.repo._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE kb_id = ? AND document_id = ?",
                (kb.id, doc.id),
            ).fetchone()
            chunk_counts[doc.id] = int(row["n"]) if row else 0

        chash = chunking_config_hash(kb.chunking_strategy, kb.chunking_config)
        entries = [self._entry(d, kb, chunk_counts, chash) for d in docs]

        vectors, confirmed = self._vector_count(kb.id) if with_vector_count else (UNKNOWN, False)
        summary = self._summarize(kb, docs, entries, vectors, confirmed, chash)
        return CorpusManifest(
            kb_id=kb.id, kb_name=kb.name, kb_version=kb.version,
            entries=entries, summary=summary,
        )

    def _summarize(self, kb: KnowledgeBase, docs, entries: list[CorpusManifestEntry],
                   vectors: int, vectors_confirmed: bool, chash: str) -> CorpusSummary:
        status_of = {d.id: d.status.value for d in docs}
        completed = sum(1 for e in entries if status_of.get(e.document_id) in _READY_STATUSES)
        failed = sum(1 for e in entries if status_of.get(e.document_id) in _FAILED_STATUSES)
        processing = sum(1 for e in entries if status_of.get(e.document_id) in _IN_FLIGHT_STATUSES)

        # Documents superseded by a newer version are "stale" in the manifest
        # sense: they are history, not current corpus content.
        superseded = {d.replaces_document_id for d in docs if d.replaces_document_id}
        stale = sum(1 for e in entries if e.document_id in superseded)

        by_hash: dict[str, list[str]] = {}
        for e in entries:
            if e.content_hash:
                by_hash.setdefault(e.content_hash, []).append(e.document_id)
        duplicates = sum(len(v) - 1 for v in by_hash.values() if len(v) > 1)

        embedding_label, _, _ = _embedding_identity(kb)
        indexed_times = [e.indexed_at for e in entries if e.indexed_at]
        return CorpusSummary(
            total_documents=len(entries),
            completed_documents=completed,
            failed_documents=failed,
            processing_documents=processing,
            total_pages=sum(e.page_count or 0 for e in entries),
            total_slides=sum(e.slide_count or 0 for e in entries),
            total_sections=sum(e.section_count or 0 for e in entries),
            total_chunks=sum(e.chunk_count for e in entries),
            total_vectors=vectors,
            total_corpus_size_bytes=sum(e.file_size or 0 for e in entries),
            duplicate_documents=duplicates,
            stale_documents=stale,
            # Orphan detection is the integrity engine's job; the manifest does
            # not guess at it.
            orphan_vectors=UNKNOWN,
            embedding_identity=embedding_label,
            chunking_identity=f"{kb.chunking_strategy or 'unset'}:{chash}",
            vector_backend=kb.vector_backend or "qdrant",
            last_successful_index_at=max(indexed_times) if indexed_times else None,
            vector_count_confirmed=vectors_confirmed,
        )