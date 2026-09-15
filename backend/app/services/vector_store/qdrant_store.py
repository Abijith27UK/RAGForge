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
                    "source_url": c.source_url,
                    "source_title": c.source_title,
                    "source_type": c.source_type,
                    "publisher": c.publisher,
                    "document_title": c.document_title,
                    "section": c.section,
                    "section_path": c.section_path,
                    "page": c.page,
                    "domain": c.domain,
                    "subdomain": c.subdomain,
                    "trust_score": c.trust_score,
                    "content_hash": c.content_hash,
                    "char_count": c.char_count,
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

    def _delete_by_document_filter(self, kb_id: str, document_ids: list[str]) -> int:
        from qdrant_client import models

        try:
            result = self._client.delete(
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
        try:
            return len(result.operation.result)  # type: ignore[union-attr]
        except Exception:
            return 0

    def delete_document_vectors(self, kb_id: str, document_ids: list[str]) -> int:
        """Delete all vectors whose payload document_id is in the given list.

        Required before re-indexing: re-chunking generates NEW chunk IDs, so old
        points would otherwise be orphaned and retrievable forever.
        Returns the number of deleted point IDs (from the server response).
        """
        if not document_ids:
            return 0
        deleted = 0
        for i in range(0, len(document_ids), _DELETE_BATCH):
            deleted += self._delete_by_document_filter(kb_id, document_ids[i : i + _DELETE_BATCH])
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
        deleted = 0
        try:
            for i in range(0, len(stale_ids), _DELETE_BATCH):
                batch = stale_ids[i : i + _DELETE_BATCH]
                # Points were created from numeric_id(chunk_id); map back.
                int_ids = [int(hashlib.sha1(c.encode("utf-8")).hexdigest()[:15], 16) for c in batch if isinstance(c, str)]
                result = self._client.delete(
                    collection_name=self._collection(kb_id),
                    points_selector=models.PointIdsList(points=int_ids),
                    wait=True,
                )
                try:
                    deleted += len(result.operation.result)  # type: ignore[union-attr]
                except Exception:
                    deleted += len(batch)
        except Exception as exc:
            raise self._wrap(exc, "deleting stale points") from exc
        logger.info("Deleted %d stale/orphaned vectors from %s", deleted, self._collection(kb_id))
        return deleted

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
                if key in ("document_id", "source_type", "publisher", "subdomain"):
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
