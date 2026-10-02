"""Retrieval service: dense vector retrieval with full provenance.

Implemented today: DENSE only. BM25 / hybrid / reranking are NOT implemented and
are NOT advertised anywhere by this module.

Extension points (V4 Phase 12) are structural, not speculative features:
`Retriever` is the abstract surface every future strategy must implement and
`RETRIEVERS` is the registry future strategies register into. All strategies
produce the same normalized `RetrievalResponse`, so the API, the evaluator and
the UI never change shape when a strategy is added.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from app.schemas.models import RetrievalResponse, RetrievalResult
from app.services.embeddings.provider import EmbeddingIdentity, EmbeddingProvider
from app.services.vector_store.qdrant_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

#: Provenance keys copied from the vector-store payload into every result.
#: Everything needed for a future LLM layer to cite
#: KB -> Document -> Source -> Page/Slide/Section -> Chunk.
PROVENANCE_KEYS: tuple[str, ...] = (
    "document_id",
    "document_title",
    "document_version",
    "source_id",
    "source_url",
    "source_title",
    "source_type",
    "publisher",
    "section",
    "section_path",
    "page",
    "slide",
    "slide_title",
    "domain",
    "subdomain",
    "trust_score",
    "user_provided",
    "content_hash",
    "chunk_index",
    "chunking_strategy",
    "chunking_config",
    "kb_version",
)


class RetrievalError(RuntimeError):
    pass


class Retriever(ABC):
    """Normalized retrieval surface.

    Every strategy (dense today; BM25 / hybrid / reranked later) MUST return a
    `RetrievalResponse` with `retrieval_backend` naming itself honestly.
    """

    backend = "base"

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        identity: EmbeddingIdentity | None = None,
    ) -> None:
        self._embedder = embedding_provider
        self._store = vector_store
        self._identity = identity

    @abstractmethod
    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResponse:
        ...

    # -- shared helpers ----------------------------------------------------

    def _embedding_label(self) -> str:
        """Report the full identity (incl. dev-fallback labelling) honestly."""
        return self._identity.describe() if self._identity else self._embedder.model

    def _build_results(self, hits: list[dict[str, Any]], min_score: float) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            score = float(hit.get("score") or 0.0)
            if score < min_score:
                continue
            provenance = {k: payload.get(k) for k in PROVENANCE_KEYS if payload.get(k) is not None}
            results.append(
                RetrievalResult(
                    chunk_id=hit.get("chunk_id") or payload.get("chunk_id") or "unknown",
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    score=round(score, 4),
                    provenance=provenance,
                )
            )
        return results


class DenseRetriever(Retriever):
    """Dense-only retriever. Backend name is reported honestly in responses."""

    backend = "qdrant-dense"

    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResponse:
        try:
            query_vector = self._embedder.embed_texts([query])[0]
        except Exception as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc
        try:
            hits = self._store.search(kb_id, query_vector, top_k=top_k, filters=filters)
        except VectorStoreError as exc:
            raise RetrievalError(f"Vector store error: {exc}") from exc

        return RetrievalResponse(
            query=query,
            top_k=top_k,
            results=self._build_results(hits, min_score),
            embedding_model=self._embedding_label(),
            retrieval_backend=self.backend,
        )


# ---------------------------------------------------------------------------
# Registry (Phase 12 extension point)
# ---------------------------------------------------------------------------

RetrieverFactory = Callable[[EmbeddingProvider, VectorStore, EmbeddingIdentity | None], Retriever]

_RETRIEVERS: dict[str, RetrieverFactory] = {
    "dense": lambda e, s, i: DenseRetriever(e, s, i),
    # Alias kept so "qdrant-dense" (the reported backend name) also resolves.
    "qdrant-dense": lambda e, s, i: DenseRetriever(e, s, i),
}


def register_retriever(name: str, factory: RetrieverFactory) -> None:
    """Register a retrieval strategy (e.g. 'bm25', 'hybrid', 'reranked')."""
    _RETRIEVERS[name.strip().lower()] = factory


def available_retrievers() -> list[str]:
    return sorted(_RETRIEVERS)


def get_retriever(
    name: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    identity: EmbeddingIdentity | None = None,
) -> Retriever:
    factory = _RETRIEVERS.get((name or "dense").strip().lower())
    if factory is None:
        raise ValueError(f"Unknown retriever {name!r}. Available: {available_retrievers()}")
    return factory(embedding_provider, vector_store, identity)