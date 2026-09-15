"""System routes: health + dependency status (Qdrant, embedding provider)."""
from __future__ import annotations

import logging

from fastapi import APIRouter

from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok", "app": "RAGForge API"}


@router.get("/status")
def status():
    """Honest dependency status: reports what is actually reachable/configured."""
    settings = get_settings()
    qdrant: dict = {"url": settings.qdrant_url, "reachable": False}
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=3)
        client.get_collections()
        qdrant["reachable"] = True
    except Exception as exc:
        qdrant["error"] = str(exc)[:200]

    embedding: dict = {"provider": settings.embedding_provider, "model": settings.embedding_model}
    llm: dict = {
        "provider": settings.llm_provider or "not configured (dev mock will be used)"
        if settings.allow_mock_llm
        else "not configured",
        "model": settings.llm_model or None,
        "allow_mock": settings.allow_mock_llm,
    }
    return {"qdrant": qdrant, "embedding": embedding, "llm": llm}
