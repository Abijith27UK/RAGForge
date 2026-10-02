"""V3 Step 6: vector-backend factory.

RAGForge reaches vectors only through the VectorStore abstraction
(app/services/vector_store/qdrant_store.py). This factory resolves a backend
id (stored per knowledge base as kb.vector_backend) to an implementation.

Today 'qdrant' is the only registered backend. Experimental backends
(e.g. TurboVec, planned Phase E) register here WITHOUT changing call sites,
and must implement the same abstract surface:

    ensure_collection / upsert_chunks / search / collection_info /
    delete_collection / delete_document_vectors / delete_orphaned_points
"""
from __future__ import annotations

from typing import Any, Callable

from app.config import Settings
from app.services.vector_store.qdrant_store import QdrantVectorStore, VectorStore

_BACKENDS: dict[str, Callable[[Settings], VectorStore]] = {
    "qdrant": lambda s: QdrantVectorStore(url=s.qdrant_url, api_key=s.qdrant_api_key),
}


def register_vector_backend(name: str, factory: Callable[[Settings], VectorStore]) -> None:
    """Register an experimental backend (e.g. 'turbovec' in Phase E)."""
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def create_vector_store(settings: Settings, backend: str | None = None) -> VectorStore:
    name = (backend or "qdrant").strip().lower()
    factory = _BACKENDS.get(name)
    if factory is None:
        raise ValueError(f"Unknown vector backend {name!r}. Available: {available_backends()}")
    return factory(settings)
