"""VectorStore abstraction + Qdrant implementation.

Qdrant is the primary vector database for the MVP. The interface exists so a
second backend can be added for experiments without touching services.
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.schemas.models import Chunk

logger = logging.getLogger(__name__)

# Qdrant filter/selectors batch size for deletes and upserts.
_DELETE_BATCH = 500


class VectorStoreError(RuntimeError):
    """Controlled error for any Qdrant availability/client failure.

    Routes map this to HTTP 503 with a user-facing message; raw httpx/socket
    errors must NEVER leak as HTTP 500 with a stack trace.
    """


def numeric_id(chunk_id: str) -> int:
    """Qdrant requires unsigned ints or UUIDs as point IDs; map deterministically."""
    return int(hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()[:15], 16)


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, kb_id: str, vector_size: int, distance: str = "cosine") -> None: ...

    @abstractmethod
    def upsert_chunks(self, kb_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    def search(
        self, kb_id: str, vector: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def collection_info(self, kb_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def delete_collection(self, kb_id: str) -> None: ...

    # The two methods below are REQUIRED by the factory contract (documented in
    # app/services/vector_store/factory.py) even though they are not abstract
    # here, so existing/simpler implementations keep working. Overriding them
    # is mandatory for any backend used with incremental ingestion: without
    # stale-vector deletion, replacing a document leaves orphaned points that
    # stay retrievable forever.
    def delete_document_vectors(self, kb_id: str, document_ids: list[str]) -> int:
        raise NotImplementedError

    def delete_orphaned_points(self, kb_id: str, valid_chunk_ids: set[str]) -> int:
        raise NotImplementedError

    def count_points_for_documents(self, kb_id: str, document_ids: list[str]) -> int:
        """Points whose payload ``document_id`` is in the given list.

        Used by the corpus repair planner to state how many vectors a repair
        will touch BEFORE it runs. Backends that cannot answer cheaply must
        raise rather than return 0, so the plan reports UNKNOWN instead of a
        fabricated zero.
        """
        raise NotImplementedError

    def all_point_payloads(self, kb_id: str) -> dict[str, dict[str, Any]]:
        """{chunk_id: payload} for every point in the collection.

        Required by the integrity engine. Raises when the collection cannot be
        read, so a scan reports UNKNOWN rather than "nothing is wrong".
        """
        raise NotImplementedError

    def fetch_vectors(self, kb_id: str, chunk_ids: list[str]) -> dict[str, list[float]]:
        """{chunk_id: vector} for the requested chunks (V6, used by MMR).

        Optional capability: backends that cannot return stored vectors raise
        NotImplementedError, and the diversity stage records MMR as unavailable
        instead of pretending diversification happened.
        """
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrant client wrapper. One collection per knowledge base."""

    def __init__(self, url: str, api_key: str | None = None) -> None:
        try:
            from qdrant_client import QdrantClient

            self._url = url
            self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        except Exception as exc:  # pragma: no cover - import-time failure
            raise VectorStoreError(f"Could not create Qdrant client: {exc}") from exc

    def _collection(self, kb_id: str) -> str:
        return f"kb_{kb_id}"

    def _wrap(self, exc: Exception, action: str) -> VectorStoreError:
        msg = str(exc)
        if isinstance(exc, KeyboardInterrupt):  # pragma: no cover
            raise exc
        lowered = msg.lower()
        if "connect" in lowered or "refused" in lowered or "timed out" in lowered or "unreachable" in lowered or "max retries" in lowered:
            return VectorStoreError(
                f"Qdrant is not reachable ({self._url}). Start it and try again. ({action}: {msg})"
            )
        return VectorStoreError(f"Qdrant error during {action}: {msg}")

    def ensure_collection(self, kb_id: str, vector_size: int, distance: str = "cosine") -> None:
        from qdrant_client import models

        try:
            existing = [c.name for c in self._client.get_collections().collections]
        except Exception as exc:
            raise self._wrap(exc, "listing collections") from exc
        name = self._collection(kb_id)
        if name in existing:
            try:
                info = self._client.get_collection(name)
            except Exception as exc:
                raise self._wrap(exc, "reading collection") from exc
            size = info.config.params.vectors.size  # type: ignore[union-attr]
            if size != vector_size:
                raise VectorStoreError(
                    f"Collection '{name}' exists with vector size {size}, but {vector_size} requested. "
                    "Delete the collection or use the same embedding model."
                )
            return
        try:
            self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
        except Exception as exc:
            raise self._wrap(exc, "creating collection") from exc
        logger.info("Created Qdrant collection %s (dim=%d)", name, vector_size)

    def upsert_chunks(self, kb_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client import models

        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        points = [
            models.PointStruct(
                id=numeric_id(c.id),
                vector=v,
                payload={
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "kb_id": c.kb_id,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "source_id": c.source_id,
                    "source_url": c.source_url,
                    "source_title": c.source_title,
                    "source_type": c.source_type,
                    "publisher": c.publisher,
                    "document_title": c.document_title,
                    "section": c.section,
                    "section_path": c.section_path,
                    "page": c.page,
                    "slide": c.slide,
                    "slide_title": c.slide_title,
                    "domain": c.domain,
                    "subdomain": c.subdomain,
                    "trust_score": c.trust_score,
                    "content_hash": c.content_hash,
                    "char_count": c.char_count,
                    "chunking_strategy": c.chunking_strategy,
                    "chunking_config": c.chunking_config,
                    "document_version": c.document_version,
                    "user_provided": c.user_provided,
                    "kb_version": c.kb_version,
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        B = 200
        try:
            for i in range(0, len(points), B):
                self._client.upsert(
                    collection_name=self._collection(kb_id), points=points[i : i + B], wait=True
                )
        except Exception as exc:
            raise self._wrap(exc, "upserting vectors") from exc
        logger.info("Upserted %d vectors into %s", len(points), self._collection(kb_id))

    def _point_count(self, kb_id: str) -> int | None:
        """Current point count, or None if the collection cannot be read.

        Used to confirm deletions independently of the server response shape.
        """
        try:
            return int(self._client.get_collection(self._collection(kb_id)).points_count or 0)
        except Exception:
            return None

    def count_points_for_documents(self, kb_id: str, document_ids: list[str]) -> int:
        """Count points matching a document filter. Raises if unmeasurable."""
        from qdrant_client import models

        if not document_ids:
            return 0
        try:
            result = self._client.count(
                collection_name=self._collection(kb_id),
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))]
                ),
                exact=True,
            )
        except Exception as exc:
            raise self._wrap(exc, "counting vectors for documents") from exc
        return int(result.count)

    def fetch_vectors(self, kb_id: str, chunk_ids: list[str]) -> dict[str, list[float]]:
        """Fetch stored vectors by chunk id (chunk_id -> numeric point id)."""
        from qdrant_client import models

        if not chunk_ids:
            return {}
        out: dict[str, list[float]] = {}
        try:
            for i in range(0, len(chunk_ids), 256):
                batch = chunk_ids[i : i + 256]
                ids = [numeric_id(c) for c in batch]
                records = self._client.retrieve(
                    collection_name=self._collection(kb_id),
                    ids=ids,
                    with_vectors=True,
                    with_payload=True,
                )
                for record in records:
                    payload = dict(record.payload or {})
                    chunk_id = payload.get("chunk_id")
                    vector = record.vector
                    if chunk_id and isinstance(vector, list):
                        out[str(chunk_id)] = [float(x) for x in vector]
        except Exception as exc:
            raise self._wrap(exc, "fetching stored vectors") from exc
        return out

    def all_point_payloads(self, kb_id: str) -> dict[str, dict[str, Any]]:
        """Scroll every point and return {chunk_id: payload}."""
        from qdrant_client import models

        payloads: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection(kb_id),
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = dict(p.payload or {})
                chunk_id = payload.get("chunk_id")
                if chunk_id:
                    payloads[str(chunk_id)] = payload
            if offset is None:
                break
        return payloads

    def _delete_by_document_filter(self, kb_id: str, document_ids: list[str]) -> int:
        """Delete by document filter and return the CONFIRMED number removed.

        Current qdrant-client versions return ``UpdateResult(operation_id, status)``
        with no deleted-count field, so reading a count off the response always
        yields 0 and every caller (build-run messages, the document-delete API)
        would report "0 stale vectors removed" even when hundreds were deleted.
        We therefore measure the before/after point-count delta ourselves.
        """
        from qdrant_client import models

        before = self._point_count(kb_id)
        try:
            self._client.delete(
                collection_name=self._collection(kb_id),
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", match=models.MatchAny(any=document_ids)
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            raise self._wrap(exc, "deleting old document vectors") from exc
        after = self._point_count(kb_id)
        if before is None or after is None:
            # Could not confirm: report -1 (unknown) rather than a fabricated 0.
            logger.warning(
                "Could not confirm the deletion count for %s; returning UNKNOWN (-1)",
                self._collection(kb_id),
            )
            return -1
        return max(0, before - after)

    def delete_document_vectors(self, kb_id: str, document_ids: list[str]) -> int:
        """Delete all vectors whose payload document_id is in the given list.

        Required before re-indexing: re-chunking generates NEW chunk IDs, so old
        points would otherwise be orphaned and retrievable forever.

        Returns the confirmed number of deleted points, measured from the
        collection's point-count delta. Returns -1 when that cannot be measured,
        so callers can report "unknown" instead of a fabricated zero.
        """
        if not document_ids:
            return 0
        deleted = 0
        for i in range(0, len(document_ids), _DELETE_BATCH):
            batch_deleted = self._delete_by_document_filter(
                kb_id, document_ids[i : i + _DELETE_BATCH]
            )
            if batch_deleted < 0:
                return -1
            deleted += batch_deleted
        if deleted:
            logger.info("Deleted %d stale vectors for %d document(s) in %s", deleted, len(document_ids), self._collection(kb_id))
        return deleted

    def delete_orphaned_points(self, kb_id: str, valid_chunk_ids: set[str]) -> int:
        """Delete points whose chunk_id is NOT in the given set.

        This catches stale vectors from ANY previous state (re-chunked documents,
        deleted documents, or points whose document_id payload is missing), not
        only those attributable to a known document list. It requires scrolling
        through all point IDs in the collection, which is cheap for MVP-scale
        corpora. Returns the number of deleted points.
        """
        from qdrant_client import models

        if not valid_chunk_ids:
            # Caller intentionally clears the collection contents.
            pass
        stale_ids: list[str] = []
        try:
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=self._collection(kb_id),
                    limit=256,
                    offset=offset,
                    with_payload=["chunk_id"],
                    with_vectors=False,
                )
                for p in points:
                    cid = (p.payload or {}).get("chunk_id")
                    if not cid or cid not in valid_chunk_ids:
                        stale_ids.append(cid or numeric_id_to_str(p.id))
                if offset is None:
                    break
        except Exception as exc:
            raise self._wrap(exc, "scanning for stale points") from exc
        if not stale_ids:
            return 0
        before = self._point_count(kb_id)
        deleted = 0
        try:
            for i in range(0, len(stale_ids), _DELETE_BATCH):
                batch = stale_ids[i : i + _DELETE_BATCH]
                # Points were created from numeric_id(chunk_id); map back.
                int_ids = [int(hashlib.sha1(c.encode("utf-8")).hexdigest()[:15], 16) for c in batch if isinstance(c, str)]
                self._client.delete(
                    collection_name=self._collection(kb_id),
                    points_selector=models.PointIdsList(points=int_ids),
                    wait=True,
                )
                deleted += len(int_ids)
        except Exception as exc:
            raise self._wrap(exc, "deleting stale points") from exc
        after = self._point_count(kb_id)
        # Prefer the measured delta; fall back to the attempted count only if the
        # collection could not be read (never invent a smaller number).
        confirmed = max(0, before - after) if (before is not None and after is not None) else deleted
        logger.info("Deleted %d stale/orphaned vectors from %s", confirmed, self._collection(kb_id))
        return confirmed

    def search(
        self, kb_id: str, vector: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        from qdrant_client import models

        qdrant_filter = None
        if filters:
            must = []
            for key, value in filters.items():
                if value is None:
                    continue
                if key in ("document_id", "source_id", "source_type", "publisher", "subdomain"):
                    must.append(
                        models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    )
                elif key == "user_provided" and isinstance(value, bool):
                    must.append(
                        models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    )
                elif key == "min_trust_score":
                    must.append(
                        models.FieldCondition(
                            key="trust_score", range=models.Range(gte=float(value))
                        )
                    )
            if must:
                qdrant_filter = models.Filter(must=must)

        try:
            result = self._client.query_points(
                collection_name=self._collection(kb_id),
                query=vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as exc:
            raise self._wrap(exc, "searching") from exc
        return [
            {
                "chunk_id": p.payload.get("chunk_id"),
                "score": p.score,
                "payload": dict(p.payload),
            }
            for p in result.points
        ]

    def collection_info(self, kb_id: str) -> dict[str, Any]:
        name = self._collection(kb_id)
        try:
            info = self._client.get_collection(name)
        except Exception as exc:
            raise VectorStoreError(f"Collection '{name}' not available: {exc}") from exc
        return {
            "name": name,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,  # type: ignore[union-attr]
            "status": str(info.status),
        }

    def delete_collection(self, kb_id: str) -> None:
        try:
            self._client.delete_collection(self._collection(kb_id))
        except Exception as exc:
            raise self._wrap(exc, "deleting collection") from exc
        logger.info("Deleted Qdrant collection %s", self._collection(kb_id))


def numeric_id_to_str(point_id: Any) -> str:
    """Best-effort reverse mapping for logging; numeric Qdrant IDs are one-way."""
    return f"point:{point_id}"
