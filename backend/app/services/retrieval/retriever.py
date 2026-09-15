"""Retrieval service: dense vector retrieval with full provenance.

Currently implemented: dense retrieval via Qdrant. BM25 / hybrid / reranking
are NOT implemented and are not advertised by this module.
"""
from __future__ import annotations

import logging
from typing import Any

from app.schemas.models import RetrievalResponse, RetrievalResult
from app.services.embeddings.provider import EmbeddingIdentity, EmbeddingProvider
from app.services.vector_store.qdrant_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    pass


class DenseRetriever:
    """Dense-only retriever. Backend name is reported honestly in responses."""

    backend = "qdrant-dense"

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        identity: EmbeddingIdentity | None = None,
    ) -> None:
        self._embedder = embedding_provider
        self._store = vector_store
        self._identity = identity

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

        results = [
            RetrievalResult(
                chunk_id=h["chunk_id"] or "unknown",
                document_id=h["payload"].get("document_id", ""),
                text=h["payload"].get("text", ""),
                score=round(float(h["score"]), 4),
                provenance={
                    k: h["payload"].get(k)
                    for k in (
                        "source_url",
                        "source_title",
                        "source_type",
                        "publisher",
                        "document_title",
                        "section",
                        "section_path",
                        "page",
                        "domain",
                        "subdomain",
                        "trust_score",
                        "content_hash",
                        "chunk_index",
                    )
                },
            )
            for h in hits
            if float(h["score"]) >= min_score
        ]
        # Report the full identity (incl. dev-fallback labelling) honestly.
        embedding_model = (
            self._identity.describe() if self._identity else self._embedder.model
        )
        return RetrievalResponse(
            query=query,
            top_k=top_k,
            results=results,
            embedding_model=embedding_model,
            retrieval_backend=self.backend,
        )
