"""CorpusIntegrityService: a READ-ONLY audit of a knowledge base (V5 Phase 3).

This service **never repairs anything**. It inspects and reports. Repair lives
in :mod:`app.services.corpus.repair` and must be invoked explicitly, so an
integrity scan can never be confused with a data-changing operation.

Checks implemented (A–K):

===  ==========================================================
A    Missing vectors — chunk metadata exists, vector does not
B    Orphan vectors  — vector exists, chunk metadata does not
C    Stale vectors   — vector belongs to a superseded document
     or a chunk_id that no longer exists
D    Embedding mismatch — vectors built with a different model
E    Chunking mismatch  — chunks built with a different config
F    Duplicate documents — identical content hashes
G    Duplicate chunks    — identical chunk content hashes
H    Failed documents    — never completed indexing
I    Partial documents   — parsing/chunking/indexing incomplete
J    Provenance gaps     — chunks missing expected provenance
K    Broken source refs  — document's source row is gone
===  ==========================================================

When the vector store cannot be reached, the vector-derived checks report
``vectors_checked = UNKNOWN`` and list their names in ``unknown_counts``. They do
NOT report zero findings, because "we could not look" is not "we found nothing".
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

from app.repositories.sqlite_repo import Repository
from app.schemas.corpus import (
    UNKNOWN,
    IntegrityCheck,
    IntegrityFinding,
    IntegrityReport,
    IntegritySeverity,
)
from app.schemas.models import KnowledgeBase
from app.services.corpus.manifest import CorpusManifestService, chunking_config_hash

logger = logging.getLogger(__name__)

#: Chunk provenance fields that must be populated for a retrievable chunk.
_REQUIRED_PROVENANCE = ("source_id", "document_id", "content_hash")


def _default_store(kb_id: str):
    """Resolve the KB's configured vector backend through the factory."""
    from app.config import get_settings
    from app.services.vector_store.factory import create_vector_store

    return create_vector_store(get_settings())


class CorpusIntegrityService:
    """Builds an :class:`IntegrityReport`. Never mutates corpus state."""

    def __init__(self, repo: Repository, store_factory=None) -> None:
        self.repo = repo
        # Injectable so tests never reach a live Qdrant by accident, and so a
        # caller can pin a specific backend for the duration of one audit.
        self._store_factory = store_factory or _default_store

    # -- vector store access ---------------------------------------------------

    def _scroll_vectors(self, kb_id: str) -> tuple[dict[str, dict] | None, list[str]]:
        """Return {chunk_id: payload} for every point, or None if unreachable."""
        try:
            store = self._store_factory(kb_id)
        except Exception as exc:
            logger.warning("Integrity: vector store unavailable: %s", exc)
            return None, ["vector store unavailable"]

        try:
            return store.all_point_payloads(kb_id), []
        except Exception as exc:
            logger.warning("Integrity: could not scroll %s: %s", kb_id, exc)
            return None, ["vector store unreachable during scan"]

    # -- checks ---------------------------------------------------------------

    def _check_vectors(
        self, report: IntegrityReport, kb: KnowledgeBase, chunks_by_id: dict,
        payloads: dict[str, dict], live_document_ids: set[str], expected_hash: str,
    ) -> None:
        """A: missing, B: orphan, C: stale, D: embedding mismatch."""
        seen_chunk_ids = set()

        for chunk in chunks_by_id.values():
            seen_chunk_ids.add(chunk.id)
            payload = payloads.get(chunk.id)
            if payload is None:
                report.missing_vectors.append(
                    IntegrityFinding(
                        check=IntegrityCheck.MISSING_VECTORS,
                        severity=IntegritySeverity.ERROR,
                        message=(
                            f"Chunk {chunk.id} (document {chunk.document_id}, "
                            f"index {chunk.chunk_index}) has no vector and is therefore "
                            "not retrievable."
                        ),
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                    )
                )
                continue

            # C: a vector whose document_id is no longer live is stale even if a
            # chunk row happens to share its id.
            payload_doc = payload.get("document_id")
            if payload_doc and payload_doc not in live_document_ids:
                report.stale_vectors.append(
                    IntegrityFinding(
                        check=IntegrityCheck.STALE_VECTORS,
                        severity=IntegritySeverity.ERROR,
                        message=(
                            f"Vector belongs to document {payload_doc}, which is no longer "
                            "in the corpus (superseded or deleted)."
                        ),
                        document_id=str(payload_doc),
                        chunk_id=chunk.id,
                        vector_id=str(payload.get("chunk_id")),
                    )
                )

            # D: embedding mismatch
            kb_emb = (kb.embedding_identity or {}).get("model")
            payload_model = payload.get("embedding_model") or payload.get("model")
            if kb_emb and payload_model and payload_model != kb_emb:
                report.embedding_mismatches.append(
                    IntegrityFinding(
                        check=IntegrityCheck.EMBEDDING_MISMATCH,
                        severity=IntegritySeverity.ERROR,
                        message=(
                            f"Vector was embedded with '{payload_model}' but the knowledge "
                            f"base currently uses '{kb_emb}'. Retrieval quality across "
                            "these vectors is not comparable."
                        ),
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        details={"vector_model": payload_model, "kb_model": kb_emb},
                    )
                )

            # J: provenance gaps on retrievable chunks
            missing = [f for f in _REQUIRED_PROVENANCE if not payload.get(f)]
            if missing:
                report.provenance_gaps.append(
                    IntegrityFinding(
                        check=IntegrityCheck.PROVENANCE_GAPS,
                        severity=IntegritySeverity.WARNING,
                        message=f"Vector payload is missing provenance: {', '.join(missing)}.",
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        details={"missing_fields": missing},
                    )
                )

        for chunk_id, payload in payloads.items():
            if chunk_id in seen_chunk_ids:
                continue
            # B: orphan. Either the chunk row is gone, or its document is gone.
            payload_doc = payload.get("document_id")
            report.orphan_vectors.append(
                IntegrityFinding(
                    check=IntegrityCheck.ORPHAN_VECTORS,
                    severity=IntegritySeverity.ERROR,
                    message=(
                        f"Vector for chunk {chunk_id} exists in the vector store but the "
                        "chunk metadata does not. It is retrievable but untraceable and "
                        "will never be cleaned up by re-indexing that document."
                    ),
                    document_id=str(payload_doc) if payload_doc else None,
                    chunk_id=chunk_id,
                    vector_id=str(payload.get("chunk_id", chunk_id)),
                    details={"document_present": payload_doc in live_document_ids if payload_doc else False},
                )
            )

    def _check_metadata(
        self, report: IntegrityReport, kb: KnowledgeBase, docs: list, chunks: list,
    ) -> None:
        """E, F, G, H, I, K — all decidable from SQLite alone."""
        live_ids = {d.id for d in docs}
        superseded = {d.replaces_document_id for d in docs if d.replaces_document_id}
        expected_hash = chunking_config_hash(kb.chunking_strategy, kb.chunking_config)

        # K: broken source references
        for doc in docs:
            if self.repo.get_source(kb.id, doc.source_id) is None:
                report.broken_source_references.append(
                    IntegrityFinding(
                        check=IntegrityCheck.BROKEN_SOURCE_REFERENCES,
                        severity=IntegritySeverity.ERROR,
                        message=(
                            f"Document references source {doc.source_id}, which no longer "
                            "exists. Provenance cannot be resolved for this document."
                        ),
                        document_id=doc.id,
                        details={"source_id": doc.source_id},
                    )
                )

        # H: failed documents
        for doc in docs:
            if doc.status.value == "failed":
                report.failed_documents.append(
                    IntegrityFinding(
                        check=IntegrityCheck.FAILED_DOCUMENTS,
                        severity=IntegritySeverity.ERROR,
                        message=f"Document never completed ingestion: {doc.parse_error or 'no error recorded'}.",
                        document_id=doc.id,
                        details={"file_name": doc.file_name, "error": doc.parse_error},
                    )
                )
            elif doc.id not in superseded and doc.status.value in {"uploaded", "parsing", "parsed", "chunking", "indexing"}:
                # I: partial — parsed but not retrievable, and not superseded.
                report.partial_documents.append(
                    IntegrityFinding(
                        check=IntegrityCheck.PARTIAL_DOCUMENTS,
                        severity=IntegritySeverity.WARNING,
                        message=(
                            f"Document is '{doc.status.value}' but is not indexed. Its content is "
                            "in the library yet invisible to retrieval."
                        ),
                        document_id=doc.id,
                        details={"file_name": doc.file_name, "status": doc.status.value},
                    )
                )

        # F: duplicate documents
        by_hash: dict[str, list[str]] = defaultdict(list)
        for doc in docs:
            if doc.content_hash:
                by_hash[doc.content_hash].append(doc.id)
        for content_hash, ids in by_hash.items():
            if len(ids) > 1:
                report.duplicate_documents.append(
                    IntegrityFinding(
                        check=IntegrityCheck.DUPLICATE_DOCUMENTS,
                        severity=IntegritySeverity.WARNING,
                        message=f"{len(ids)} documents share content hash {content_hash[:12]}.",
                        details={"content_hash": content_hash, "document_ids": sorted(ids)},
                    )
                )

        # G: duplicate chunks
        chunk_hashes: dict[str, list[str]] = defaultdict(list)
        for chunk in chunks:
            if chunk.content_hash:
                chunk_hashes[chunk.content_hash].append(chunk.id)
        for content_hash, ids in chunk_hashes.items():
            if len(ids) > 1:
                report.duplicate_chunks.append(
                    IntegrityFinding(
                        check=IntegrityCheck.DUPLICATE_CHUNKS,
                        severity=IntegritySeverity.INFO,
                        message=f"{len(ids)} chunks share content hash {content_hash[:12]}.",
                        details={"content_hash": content_hash, "chunk_ids": sorted(ids)[:20]},
                    )
                )

        # E: chunking mismatch
        by_doc: dict[str, list] = defaultdict(list)
        for chunk in chunks:
            by_doc[chunk.document_id].append(chunk)
        for doc_id, doc_chunks in by_doc.items():
            strategies = {c.chunking_strategy for c in doc_chunks if c.chunking_strategy}
            configs = {c.chunking_config.get("config_version") for c in doc_chunks if c.chunking_config}
            if len(strategies) > 1 or len(configs) > 1:
                report.chunking_mismatches.append(
                    IntegrityFinding(
                        check=IntegrityCheck.CHUNKING_MISMATCH,
                        severity=IntegritySeverity.WARNING,
                        message=(
                            f"Chunks of document {doc_id} were produced by more than one "
                            f"configuration: strategies={sorted(strategies)}, versions={sorted(configs)}."
                        ),
                        document_id=doc_id,
                        details={"strategies": sorted(strategies), "config_versions": sorted(configs)},
                    )
                )
            elif strategies and kb.chunking_strategy and strategies != {kb.chunking_strategy}:
                report.chunking_mismatches.append(
                    IntegrityFinding(
                        check=IntegrityCheck.CHUNKING_MISMATCH,
                        severity=IntegritySeverity.INFO,
                        message=(
                            f"Document {doc_id} was chunked with '{strategies.pop()}' while the "
                            f"knowledge base currently uses '{kb.chunking_strategy}'."
                        ),
                        document_id=doc_id,
                        details={"expected_chunking_hash": expected_hash},
                    )
                )

        # J: provenance gaps in metadata (no vector needed)
        for chunk in chunks:
            missing = [f for f in _REQUIRED_PROVENANCE if not getattr(chunk, f, None)]
            if missing:
                report.provenance_gaps.append(
                    IntegrityFinding(
                        check=IntegrityCheck.PROVENANCE_GAPS,
                        severity=IntegritySeverity.WARNING,
                        message=f"Chunk metadata is missing provenance: {', '.join(missing)}.",
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        details={"missing_fields": missing},
                    )
                )

    # -- public ---------------------------------------------------------------

    def _status(self, report: IntegrityReport) -> str:
        buckets = (
            [report.missing_vectors, report.orphan_vectors, report.stale_vectors,
             report.embedding_mismatches, report.failed_documents,
             report.broken_source_references],
            [report.chunking_mismatches, report.partial_documents, report.provenance_gaps,
             report.duplicate_documents, report.duplicate_chunks],
        )
        if any(buckets[0]):
            return "ERROR"
        if any(buckets[1]) or report.unknown_counts:
            return "WARNING"
        return "HEALTHY"

    def scan(self, kb: KnowledgeBase, *, include_vectors: bool = True) -> IntegrityReport:
        """Audit the corpus. Read-only: no document, chunk or vector is touched."""
        started = time.perf_counter()
        docs = self.repo.list_documents(kb.id)
        chunks = self.repo.list_chunks(kb.id, limit=100_000)
        report = IntegrityReport(
            kb_id=kb.id,
            documents_checked=len(docs),
            chunks_checked=len(chunks),
        )
        self._check_metadata(report, kb, docs, chunks)

        if include_vectors:
            live_ids = {d.id for d in docs}
            chunks_by_id = {c.id: c for c in chunks}
            payloads, problems = self._scroll_vectors(kb.id)
            if payloads is None:
                report.vectors_checked = UNKNOWN
                report.unknown_counts = [
                    "missing_vectors", "orphan_vectors", "stale_vectors",
                    "embedding_mismatches", "provenance_gaps (vector payload)",
                    *problems,
                ]
                logger.warning("Integrity: vector checks skipped for %s (%s)", kb.id, problems)
            else:
                report.vectors_checked = len(payloads)
                self._check_vectors(
                    report, kb, chunks_by_id, payloads, live_ids,
                    chunking_config_hash(kb.chunking_strategy, kb.chunking_config),
                )

        buckets = [
            report.missing_vectors, report.orphan_vectors, report.stale_vectors,
            report.embedding_mismatches, report.chunking_mismatches,
            report.duplicate_documents, report.duplicate_chunks,
            report.failed_documents, report.partial_documents,
            report.provenance_gaps, report.broken_source_references,
        ]
        for bucket in buckets:
            report.findings.extend(bucket)

        report.overall_status = self._status(report)
        report.generated_at = datetime.now(timezone.utc)
        report.duration_seconds = round(time.perf_counter() - started, 3)
        logger.info(
            "Integrity scan %s: %s (%d docs, %d chunks, %d findings)",
            kb.id, report.overall_status, len(docs), len(chunks), len(report.findings),
        )
        return report

    def summary(self, kb: KnowledgeBase, *, include_vectors: bool = True) -> dict:
        """Small payload for the Corpus Command Center health strip."""
        report = self.scan(kb, include_vectors=include_vectors)
        return {
            "overall_status": report.overall_status,
            "documents_checked": report.documents_checked,
            "chunks_checked": report.chunks_checked,
            "vectors_checked": report.vectors_checked,
            "counts": report.counts(),
            "unknown_counts": report.unknown_counts,
            "generated_at": report.generated_at.isoformat(),
        }