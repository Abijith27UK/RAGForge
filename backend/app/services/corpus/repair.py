"""Safe repair operations (V5 Phase 4).

Repair is ALWAYS explicit. Nothing in this module runs as a side effect of an
integrity scan, a manifest build, or a page load. Every operation:

1. produces a :class:`RepairPlan` first, so the caller can see exactly which
   documents/chunks/vectors are affected before anything changes;
2. requires ``confirm_action`` matching the requested action when destructive;
3. performs the smallest action that fixes the problem;
4. verifies the resulting state;
5. reports **confirmed** counts, or ``UNKNOWN`` when verification is impossible.

Deletion counts reuse the vector store's measured point-count delta, so a
repair never claims "0 removed" when it removed hundreds of points.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from app.repositories.sqlite_repo import Repository
from app.schemas.corpus import (
    UNKNOWN,
    DESTRUCTIVE_ACTIONS,
    RepairAction,
    RepairPlan,
    RepairResult,
)
from app.schemas.models import DocumentStatus, KnowledgeBase

logger = logging.getLogger(__name__)


class RepairError(RuntimeError):
    """Repair could not be planned or performed. Message is user-facing."""


def _default_store_factory(kb_id: str):
    from app.config import get_settings
    from app.services.vector_store.factory import create_vector_store

    return create_vector_store(get_settings())


class CorpusRepairService:
    """Explicit, confirmable repairs. Never invoked automatically."""

    def __init__(self, repo: Repository, index_documents, create_store=None,
                 store_factory=None) -> None:
        self.repo = repo
        # Injected so the repair path uses the SAME indexer as the API routes.
        # Contract: ``(documents, *, chunker, target_size, overlap) -> dict``.
        self._index_documents = index_documents
        self._create_store = create_store
        # Injectable so planning never reaches a live Qdrant in tests.
        self._store_factory = store_factory or _default_store_factory

    # -- planning --------------------------------------------------------------

    def plan(
        self, kb: KnowledgeBase, action: RepairAction, *,
        document_ids: list[str] | None = None, batch_id: str | None = None,
        report=None,
    ) -> RepairPlan:
        """Describe what the action would do. No mutation happens here."""
        docs = self.repo.list_documents(kb.id)
        destructive = action in DESTRUCTIVE_ACTIONS

        if action is RepairAction.REINDEX_DOCUMENT:
            targets = [d for d in docs if d.id in set(document_ids or [])]
            if not targets:
                raise RepairError("No matching documents to re-index.")
            return RepairPlan(
                action=action, kb_id=kb.id,
                description=f"Re-chunk, re-embed and re-index {len(targets)} document(s).",
                affected_documents=[d.id for d in targets],
                expected_chunks=sum(d.chunk_count or 0 for d in targets),
                expected_vectors=self._count_vectors(kb.id, [d.id for d in targets]),
                destructive=False, counts_confirmed=True,
            )

        if action is RepairAction.REINDEX_FAILED_DOCUMENTS:
            failed = [d for d in docs if d.status is DocumentStatus.FAILED]
            return RepairPlan(
                action=action, kb_id=kb.id,
                description=f"Retry {len(failed)} failed document(s).",
                affected_documents=[d.id for d in failed],
                expected_chunks=sum(d.chunk_count or 0 for d in failed),
                expected_vectors=UNKNOWN,
                destructive=False, counts_confirmed=False,
            )

        if action is RepairAction.REINDEX_BATCH:
            if not batch_id:
                raise RepairError("batch_id is required for reindex_batch.")
            items = self.repo.list_ingestion_items(batch_id)
            ids = [i.document_id for i in items if i.document_id]
            return RepairPlan(
                action=action, kb_id=kb.id,
                description=f"Re-index the {len(ids)} document(s) produced by batch {batch_id}.",
                affected_documents=ids,
                expected_chunks=UNKNOWN, expected_vectors=UNKNOWN,
                destructive=False, counts_confirmed=False,
            )

        if action is RepairAction.REMOVE_ORPHAN_VECTORS:
            orphans = report.orphan_vectors if report else []
            return RepairPlan(
                action=action, kb_id=kb.id,
                description=(
                    f"Delete {len(orphans)} orphan vector(s) that have no chunk metadata."
                    if orphans else "Delete orphan vectors (none currently detected)."
                ),
                affected_documents=sorted({f.document_id for f in orphans if f.document_id}),
                expected_vectors=len(orphans) if report else UNKNOWN,
                destructive=True,
                counts_confirmed=report is not None and report.vectors_checked != UNKNOWN,
            )

        if action is RepairAction.REBUILD_DOCUMENT_VECTORS:
            targets = [d for d in docs if d.id in set(document_ids or [])]
            if not targets:
                raise RepairError("No matching documents to rebuild.")
            expected = self._count_vectors(kb.id, [d.id for d in targets])
            return RepairPlan(
                action=action, kb_id=kb.id,
                description=(
                    f"Delete and recreate the vectors of {len(targets)} document(s), "
                    "discarding their existing chunks."
                ),
                affected_documents=[d.id for d in targets],
                expected_vectors=expected,
                destructive=True,
                counts_confirmed=expected != UNKNOWN,
            )

        if action is RepairAction.REBUILD_KB:
            expected = self._count_vectors(kb.id, [d.id for d in docs])
            return RepairPlan(
                action=action, kb_id=kb.id,
                description=(
                    f"Delete and re-index the entire corpus ({len(docs)} documents). "
                    "Existing chunks and vectors are discarded."
                ),
                affected_documents=[d.id for d in docs],
                expected_chunks=self.repo.count_chunks(kb.id),
                expected_vectors=expected,
                destructive=True, counts_confirmed=expected != UNKNOWN,
            )

        if action is RepairAction.RECOMPUTE_EMBEDDINGS:
            expected = self._count_vectors(kb.id, [d.id for d in docs])
            return RepairPlan(
                action=action, kb_id=kb.id,
                description=(
                    f"Re-embed and re-index all {len(docs)} document(s) with the currently "
                    "configured embedding model. Old vectors are removed first."
                ),
                affected_documents=[d.id for d in docs],
                expected_vectors=expected,
                destructive=True, counts_confirmed=expected != UNKNOWN,
            )

        raise RepairError(f"Unsupported repair action: {action}")

    def _count_vectors(self, kb_id: str, document_ids: list[str]) -> int:
        """Points attributable to these documents, or UNKNOWN if unmeasurable."""
        if not document_ids:
            return 0
        try:
            store = self._store_factory(kb_id)
            return store.count_points_for_documents(kb_id, document_ids)
        except Exception as exc:
            logger.warning("Could not count vectors for repair plan: %s", exc)
            return UNKNOWN

    # -- execution -------------------------------------------------------------

    def execute(
        self, kb: KnowledgeBase, plan: RepairPlan, *, confirm_action: RepairAction | None = None,
        chunker: str = "section-aware", target_size: int = 1200, overlap: int = 150,
    ) -> RepairResult:
        """Run a planned repair. Requires confirmation for destructive actions."""
        if plan.destructive and confirm_action != plan.action:
            raise RepairError(
                f"'{plan.action.value}' is destructive and must be confirmed by sending "
                f"confirm_action='{plan.action.value}'. Nothing was changed."
            )
        started = datetime.now(timezone.utc)
        result = RepairResult(action=plan.action, kb_id=kb.id, ok=False)

        try:
            if plan.action is RepairAction.REMOVE_ORPHAN_VECTORS:
                self._remove_orphans(kb, plan, result)
            else:
                self._reindex(kb, plan, result, chunker, target_size, overlap)
        except RepairError:
            raise
        except Exception as exc:
            logger.exception("Repair %s failed", plan.action)
            result.errors.append(str(exc))

        result.documents_affected = len(plan.affected_documents)
        result.started_at = started
        result.finished_at = datetime.now(timezone.utc)
        result.ok = not result.errors
        if result.ok:
            result.message = result.message or f"{plan.action.value} completed."
        logger.info(
            "Repair %s on %s: ok=%s docs=%d chunks=%d vectors_removed=%s",
            plan.action.value, kb.id, result.ok, result.documents_affected,
            result.chunks_written, result.vectors_removed_label(),
        )
        return result

    def _reindex(self, kb: KnowledgeBase, plan: RepairPlan, result: RepairResult,
                 chunker: str, target_size: int, overlap: int) -> None:
        docs = [self.repo.get_document(kb.id, did) for did in plan.affected_documents]
        # A FAILED document has no parsed text yet, so it must be re-parsed from
        # its stored bytes before it can be indexed.
        reparable = [d for d in docs if d is not None and d.status is DocumentStatus.FAILED]
        indexable = [d for d in docs if d is not None and d.status is not DocumentStatus.FAILED]
        if reparable:
            indexable.extend(self._reparse_failed(kb, reparable, result))

        if not indexable:
            # Nothing to do is not a failure. Say so plainly.
            result.message = "No indexable documents selected; nothing was changed."
            result.vectors_removed = 0
            result.removal_confirmed = True
            return
        outcome = self._index_documents(
            indexable, chunker=chunker, target_size=target_size, overlap=overlap,
        )
        result.chunks_written = int(outcome.get("chunk_count", 0))
        result.vectors_written = int(outcome.get("vectors_indexed", 0))
        result.vectors_removed = int(outcome.get("stale_vectors_removed", UNKNOWN))
        result.removal_confirmed = result.vectors_removed != UNKNOWN

    def _reparse_failed(self, kb: KnowledgeBase, docs: list, result: RepairResult) -> list:
        """Re-run the parser over each failed document's stored original bytes."""
        from app.services.ingestion.ingestion import IngestionError, reparse_document

        recovered: list = []
        for doc in docs:
            if not doc.raw_file_path or not Path(doc.raw_file_path).exists():
                result.errors.append(
                    f"{doc.file_name or doc.id}: original bytes are no longer on disk; "
                    "re-upload the document instead."
                )
                continue
            try:
                text, meta = reparse_document(doc)
            except IngestionError as exc:
                doc.parse_error = str(exc)
                self.repo.update_document(doc)
                result.errors.append(f"{doc.file_name or doc.id}: {exc}")
                continue
            doc.file_path = str(Path(doc.raw_file_path).with_suffix(
                Path(doc.raw_file_path).suffix + ".parsed.txt"
            ))
            doc.parse_metadata = dict(meta)
            doc.text_length = len(text)
            doc.parse_error = None
            doc.status = DocumentStatus.PARSED
            doc.page_count = meta.get("page_count")
            doc.slide_count = meta.get("slide_count")
            doc.section_count = meta.get("section_count")
            self.repo.update_document(doc)
            recovered.append(doc)
        return recovered

    def _remove_orphans(self, kb: KnowledgeBase, plan: RepairPlan, result: RepairResult) -> None:
        """Delete vectors whose chunk metadata is gone, then verify.

        Only point IDs whose chunk row is confirmed absent are deleted; anything
        we cannot prove is orphan is left alone.
        """
        from app.services.vector_store.qdrant_store import QdrantVectorStore
        from qdrant_client import models

        store = self._store_factory(kb.id)
        if not isinstance(store, QdrantVectorStore):
            raise RepairError("Orphan removal currently requires the Qdrant backend.")
        live = {c.id for c in self.repo.list_chunks(kb.id, limit=200_000)}
        stale_ids: list[str] = []
        offset = None
        while True:
            points, offset = store._client.scroll(
                collection_name=store._collection(kb.id), limit=512, offset=offset,
                with_payload=["chunk_id"], with_vectors=False,
            )
            for p in points:
                cid = (p.payload or {}).get("chunk_id")
                if cid and cid not in live:
                    stale_ids.append(str(cid))
            if offset is None:
                break
        if not stale_ids:
            result.message = "No orphan vectors were present; nothing deleted."
            result.vectors_removed = 0
            result.removal_confirmed = True
            return
        before = store._point_count(kb.id)
        int_ids = [
            int(__import__("hashlib").sha1(c.encode("utf-8")).hexdigest()[:15], 16)
            for c in stale_ids
        ]
        store._client.delete(
            collection_name=store._collection(kb.id),
            points_selector=models.PointIdsList(points=int_ids), wait=True,
        )
        after = store._point_count(kb.id)
        # Verified against the real point count, never the attempted count.
        result.vectors_removed = (
            max(0, before - after) if (before is not None and after is not None) else UNKNOWN
        )
        result.removal_confirmed = result.vectors_removed != UNKNOWN
        result.message = f"Removed {result.vectors_removed_label()} orphan vector(s)."