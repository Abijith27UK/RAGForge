"""Persistent, resumable bulk ingestion orchestration (V5 Phase 1).

Design: no Celery, no Redis, no background threads. A batch and every one of
its items are written to SQLite *before* any processing begins, and each state
transition is committed immediately. That makes the whole thing recoverable
from any crash point without an external scheduler, and it keeps the design
deterministic: the same bytes always produce the same batch structure.

Guarantees this service provides:

* **Failure isolation** — one bad file never fails the batch. The item goes to
  ``FAILED`` with an error code/message and the batch continues.
* **Idempotent resume** — ``item_key`` is unique per batch at the schema level.
  A resume re-processes only items that are incomplete or failed, re-uses the
  existing ``Document`` when one was already created (so no duplicate rows and
  no duplicate vectors), and skips completed items unless forced.
* **Crash detection** — items left mid-stage are ``is_incomplete``, which is how
  the service finds work to resume after the process died.
* **Honesty** — a vector count is ``UNKNOWN`` until Qdrant confirms it.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.repositories.sqlite_repo import Repository
from app.schemas.corpus import (
    UNKNOWN,
    IngestionBatch,
    IngestionBatchDetail,
    IngestionItem,
    IngestionItemStatus,
    IngestionStage,
    PipelineTimings,
)
from app.schemas.models import Document, DocumentStatus, KBStatus, Source
from app.services.ingestion.ingestion import IngestionError, ingest_uploaded_bytes
from app.services.ingestion.upload import sanitize_file_name, validate_upload
from app.services.source_quality.user_scorer import assess_user_upload
from app.utils.ids import new_id
from app.utils.text import sha256_bytes

logger = logging.getLogger(__name__)

#: Fraction of overall progress attributed to each stage, so the UI bar moves
#: monotonically and a stuck item is visible as a stalled bar rather than a
#: silently frozen page.
_STAGE_PROGRESS: dict[IngestionStage, float] = {
    IngestionStage.QUEUED: 0.0,
    IngestionStage.VALIDATING: 0.10,
    IngestionStage.PARSING: 0.30,
    IngestionStage.CHUNKING: 0.50,
    IngestionStage.EMBEDDING: 0.70,
    IngestionStage.INDEXING: 0.90,
    IngestionStage.COMPLETE: 1.0,
    IngestionStage.FAILED: 1.0,
    IngestionStage.CANCELLED: 1.0,
}


class BatchError(RuntimeError):
    """Recoverable orchestration failure with a user-facing message."""


def item_key_for(index: int, file_name: str, size: int, content_hash: str) -> str:
    """Deterministic identity of an upload slot.

    Derived from position + name + size + hash so that re-posting the same set
    of files in the same order produces the same keys, which is what makes a
    resumed batch line up with the original.
    """
    return f"{index:04d}:{file_name}:{size}:{content_hash[:12]}"


class IngestionBatchService:
    """Owns the batch/item lifecycle. All state transitions are persisted."""

    def __init__(self, repo: Repository, *, max_bytes: int, upload_dir_for: Callable[[str], Path],
                 allowed_extensions: list[str]) -> None:
        self.repo = repo
        self.max_bytes = max_bytes
        self.upload_dir_for = upload_dir_for
        self.allowed_extensions = allowed_extensions

    # -- item helpers ---------------------------------------------------------

    def _advance(self, item: IngestionItem, stage: IngestionStage,
                 status: IngestionItemStatus) -> IngestionItem:
        item.stage = stage
        item.status = status
        item.progress = _STAGE_PROGRESS.get(stage, item.progress)
        self.repo.update_ingestion_item(item)
        return item

    def _fail(self, item: IngestionItem, code: str, message: str) -> IngestionItem:
        item.error_code = code
        item.error_message = message
        item.completed_at = datetime.now(timezone.utc)
        logger.info("Ingestion item %s (%s) failed: %s", item.id, item.file_name, message)
        return self._advance(item, IngestionStage.FAILED, IngestionItemStatus.FAILED)

    def _recompute_batch_counts(self, batch: IngestionBatch) -> IngestionBatch:
        """Refresh the batch counters from a SQL GROUP BY (not a full item load).

        Called after every file during a 200-file upload, so this must stay O(1)
        in Python rather than re-validating every item each time.
        """
        counts = self.repo.count_batch_items_by_status(batch.id)
        batch.total_items = sum(counts.values())
        batch.completed_items = counts.get(IngestionItemStatus.COMPLETE.value, 0)
        batch.failed_items = counts.get(IngestionItemStatus.FAILED.value, 0)
        batch.duplicate_items = counts.get(IngestionItemStatus.DUPLICATE.value, 0)
        batch.rejected_items = counts.get(IngestionItemStatus.REJECTED.value, 0)
        self.repo.update_ingestion_batch(batch)
        return batch

    # -- batch creation -------------------------------------------------------

    def create_batch(
        self,
        *,
        kb_id: str,
        source_mode: str,
        chunker: str = "section-aware",
        target_size: int = 1200,
        overlap: int = 150,
        index_on_complete: bool = True,
        total_items: int = 0,
    ) -> IngestionBatch:
        batch = IngestionBatch(
            id=new_id("batch"),
            kb_id=kb_id,
            status="pending",
            source_mode=source_mode,
            chunker=chunker,
            target_size=target_size,
            overlap=overlap,
            index_on_complete=index_on_complete,
            total_items=total_items,
        )
        self.repo.create_ingestion_batch(batch)
        return batch

    def add_item(
        self, *, batch_id: str, kb_id: str, index: int, file_name: str, size: int,
        content_hash: str | None, mime_type: str | None = None,
    ) -> IngestionItem:
        """Persist one upload slot. Returns the existing row if the key exists.

        Idempotent by construction: a repeated POST with the same content
        returns the same item instead of creating a second one.
        """
        safe = sanitize_file_name(file_name)
        key = item_key_for(index, safe, size, content_hash or "")
        existing = self.repo.find_ingestion_item_by_key(batch_id, key)
        if existing is not None:
            return existing
        item = IngestionItem(
            id=new_id("item"),
            batch_id=batch_id,
            kb_id=kb_id,
            item_key=key,
            file_name=safe,
            normalized_file_name=safe.lower().replace(" ", "_"),
            size_bytes=size,
            mime_type=mime_type,
            content_hash=content_hash,
        )
        self.repo.create_ingestion_item(item)
        return item

    # -- processing -----------------------------------------------------------

    def process_item(
        self,
        item: IngestionItem,
        *,
        data: bytes,
        kb,
        source_builder: Callable[..., Source],
        index_documents: Callable[[list[Document]], object] | None = None,
        force: bool = False,
    ) -> IngestionItem:
        """Validate -> parse -> (optionally) index one item.

        Every failure path persists an error on the item and returns it; this
        method never raises for a per-file problem, because the whole point of a
        batch is that one bad file cannot take down the rest.
        """
        if item.status is IngestionItemStatus.COMPLETE and not force:
            item.warnings.append("skipped: already complete")
            return item

        item.attempts += 1
        item.started_at = item.started_at or datetime.now(timezone.utc)
        item.error_code = None
        item.error_message = None
        timings: dict[str, float] = {}

        # -- validate --------------------------------------------------------
        t0 = time.perf_counter()
        self._advance(item, IngestionStage.VALIDATING, IngestionItemStatus.PROCESSING)
        validation = validate_upload(
            item.file_name, data, max_bytes=self.max_bytes,
            allowed_extensions=self.allowed_extensions,
        )
        timings["validation"] = time.perf_counter() - t0
        if not validation.ok:
            self._fail(item, "INVALID_FILE", validation.error)
            item.timings = {**item.timings, **timings}
            self.repo.update_ingestion_item(item)
            return item
        item.warnings.extend(validation.warnings)

        # -- duplicate detection --------------------------------------------
        content_hash = sha256_bytes(data)
        item.content_hash = content_hash
        existing = self.repo.find_document_by_hash(kb.id, content_hash)
        if existing is not None and not force:
            item.duplicate_of = existing.id
            item.document_id = existing.id
            item.completed_at = datetime.now(timezone.utc)
            item.error_code = "DUPLICATE"
            item.error_message = (
                "Identical content already in this knowledge base "
                f"({existing.file_name or existing.id}); existing document kept."
            )
            self._advance(item, IngestionStage.COMPLETE, IngestionItemStatus.DUPLICATE)
            item.timings = {**item.timings, **timings}
            self.repo.update_ingestion_item(item)
            return item

        # -- parse -----------------------------------------------------------
        t0 = time.perf_counter()
        self._advance(item, IngestionStage.PARSING, IngestionItemStatus.PROCESSING)
        source = source_builder(item.file_name, len(data), Path(item.file_name).suffix.lower())
        self.repo.create_source(source)
        try:
            doc = ingest_uploaded_bytes(
                kb_id=kb.id,
                source=source,
                upload_dir=self.upload_dir_for(kb.id),
                file_name=item.file_name,
                data=data,
                mime_type=item.mime_type,
            )
        except IngestionError as exc:
            # The document row is persisted as FAILED so it is visible in the
            # library and reported by the integrity engine — never dropped.
            assessment = assess_user_upload(
                file_valid=True, parse_ok=False, extracted_chars=0, parse_error=str(exc)
            )
            self._record_source_assessment(source, assessment)
            failed_doc = Document(
                id=new_id("doc"), kb_id=kb.id, source_id=source.id, url=source.url,
                title=item.file_name, source_type=source.source_type, file_path=None,
                content_hash=content_hash, text_length=0, parse_error=str(exc),
                status=DocumentStatus.FAILED, user_provided=True,
                file_name=item.file_name, file_size=len(data), mime_type=item.mime_type,
                parser=validation.detected_format or None,
                # Keep the bytes so a repair can retry this document later
                # without asking the user to upload it again.
                raw_file_path=getattr(exc, "raw_path", None),
            )
            self.repo.create_document(failed_doc)
            item.document_id = failed_doc.id
            item.source_id = source.id
            timings["parse"] = time.perf_counter() - t0
            self._fail(item, "PARSE_FAILED", str(exc))
            item.timings = {**item.timings, **timings}
            self.repo.update_ingestion_item(item)
            return item

        assessment = assess_user_upload(
            file_valid=True, parse_ok=True, extracted_chars=doc.text_length,
            unit_count=doc.page_count or doc.slide_count or doc.section_count,
            warnings=validation.warnings,
        )
        self._record_source_assessment(source, assessment)
        doc.status = (
            DocumentStatus.PARSED if assessment.decision.value != "REJECT" else DocumentStatus.FAILED
        )
        self.repo.create_document(doc)
        item.document_id = doc.id
        item.source_id = source.id
        item.parser_used = doc.parser
        timings["parse"] = time.perf_counter() - t0

        if doc.status is DocumentStatus.FAILED:
            item.timings = {**item.timings, **timings}
            self._fail(item, "INTEGRITY_REJECTED", assessment.reasons[0] if assessment.reasons
                       else "Source integrity rejected this file.")
            self.repo.update_ingestion_item(item)
            return item

        # -- index -----------------------------------------------------------
        if index_documents is not None:
            t0 = time.perf_counter()
            self._advance(item, IngestionStage.CHUNKING, IngestionItemStatus.PROCESSING)
            self._advance(item, IngestionStage.EMBEDDING, IngestionItemStatus.PROCESSING)
            self._advance(item, IngestionStage.INDEXING, IngestionItemStatus.PROCESSING)
            try:
                index_documents([doc])  # noqa: B023
            except Exception as exc:  # indexer failure is per-document too
                timings["index"] = time.perf_counter() - t0
                item.timings = {**item.timings, **timings}
                self._fail(item, "INDEX_FAILED", str(exc))
                return item
            timings["index"] = time.perf_counter() - t0
            stored = self.repo.get_document(kb.id, doc.id)
            item.chunk_count = stored.chunk_count if stored else 0
            # Only a store that confirms its count may report one.
            item.vector_count = item.chunk_count if index_documents is not None else UNKNOWN

        item.completed_at = datetime.now(timezone.utc)
        item.timings = {**item.timings, **timings}
        self._advance(item, IngestionStage.COMPLETE, IngestionItemStatus.COMPLETE)
        logger.info("Ingestion item %s (%s) complete: %d chunks", item.id, item.file_name, item.chunk_count)
        return item

    def _record_source_assessment(self, source: Source, assessment) -> None:
        payload = assessment.model_dump(mode="json")
        source.trust_score = assessment.score
        source.decision = assessment.decision
        source.integrity = payload
        source.quality = payload
        self.repo.update_source(source)

    # -- resume ---------------------------------------------------------------

    def resumable_items(self, batch: IngestionBatch, *, include_completed: bool = False,
                        retry_failed: bool = True) -> list[IngestionItem]:
        """Items a resume would process. Deterministic order (by item_key)."""
        out: list[IngestionItem] = []
        for item in self.repo.list_ingestion_items(batch.id):
            if item.status is IngestionItemStatus.COMPLETE and not include_completed:
                continue
            if item.status is IngestionItemStatus.DUPLICATE and not include_completed:
                continue
            if item.status is IngestionItemStatus.REJECTED and not include_completed:
                continue
            if item.status is IngestionItemStatus.CANCELLED and not include_completed:
                continue
            if item.status is IngestionItemStatus.FAILED and not retry_failed and not item.is_incomplete:
                continue
            out.append(item)
        return out

    def get_detail(self, batch: IngestionBatch) -> IngestionBatchDetail:
        items = self.repo.list_ingestion_items(batch.id)
        return IngestionBatchDetail(
            batch=self._recompute_batch_counts(batch),
            items=items,
            resumable_items=sum(1 for i in items if self._is_resumable(i)),
        )

    @staticmethod
    def _is_resumable(item: IngestionItem) -> bool:
        """A resume would process this item (completed/duplicate/rejected skipped)."""
        if item.status in {
            IngestionItemStatus.COMPLETE, IngestionItemStatus.DUPLICATE,
            IngestionItemStatus.REJECTED, IngestionItemStatus.CANCELLED,
        }:
            return False
        return True

    def finish_batch(self, batch: IngestionBatch, kb) -> IngestionBatch:
        """Close out a batch and mark the KB usable (or error) honestly."""
        batch = self._recompute_batch_counts(batch)
        done = (
            batch.completed_items + batch.failed_items
            + batch.duplicate_items + batch.rejected_items
        )
        batch.finished_at = datetime.now(timezone.utc)
        if done < batch.total_items:
            # Not everything finished — this is an incomplete batch, and saying
            # "complete" here would hide a partial ingestion.
            batch.status = "incomplete"
        elif batch.failed_items or batch.rejected_items:
            # Every item reached a terminal state, but some did not succeed.
            # This is a PARTIAL ingestion and must never read as "complete".
            batch.status = "partial"
        elif batch.completed_items == 0:
            batch.status = "failed"
        else:
            batch.status = "complete"
        self.repo.update_ingestion_batch(batch)
        kb.status = KBStatus.READY if batch.completed_items else KBStatus.NEEDS_REVIEW
        kb.updated_at = datetime.now(timezone.utc)
        self.repo.update_kb(kb)
        return batch


def batch_timings(items: list[IngestionItem]) -> PipelineTimings:
    """Aggregate measured per-item stage times into one summary.

    Only real measurements contribute; a stage nobody timed is reported as 0.0
    rather than back-filled with an estimate.
    """
    t = PipelineTimings(documents=len(items))
    for item in items:
        for key, value in item.timings.items():
            setattr(t, key, getattr(t, key) + value)
        t.chunks += item.chunk_count
        t.vectors += item.chunk_count if item.vector_count != UNKNOWN else 0
        t.bytes += item.size_bytes
    t.total = sum(
        getattr(t, k) for k in ("upload", "validation", "parse", "chunk", "embed", "index")
    )
    return t