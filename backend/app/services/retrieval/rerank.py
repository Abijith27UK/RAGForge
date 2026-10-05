"""Optional reranking (V6 Phase 6).

Contract: reranking is never silently faked. If the requested reranker cannot be
loaded (dependency missing, model not downloadable, no network), the retriever
returns the UNRERANKED order and reports ``status = unavailable_fallback`` with
the reason, plus a warning note on the response. A caller can therefore always
tell the difference between "reranked" and "we tried and could not".

The cross-encoder is loaded lazily and only when explicitly requested, because
loading it downloads/reads hundreds of MB. It shares the already-installed
`sentence-transformers` dependency (no new package).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from app.schemas.retrieval import RerankerStatus

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankResult:
    """Outcome of a rerank attempt. `order` is None when no reranking happened."""

    status: RerankerStatus
    model: str = ""
    order: list[str] | None = None
    scores: dict[str, float] = field(default_factory=dict)
    detail: str = ""

    @property
    def applied(self) -> bool:
        return self.status == RerankerStatus.APPLIED and self.order is not None


class Reranker(ABC):
    """Reranks (chunk_id, text) candidates against a query."""

    name: str = "base"
    model: str = ""

    def availability(self) -> tuple[bool, str]:
        """(available, reason). Must be cheap and must never raise."""
        return (True, "")

    @abstractmethod
    def rerank(self, query: str, items: Sequence[tuple[str, str]]) -> RerankResult: ...


class NoReranker(Reranker):
    """Explicit no-op. Reported as not_requested, not as a silent success."""

    name = "none"
    model = ""

    def rerank(self, query: str, items: Sequence[tuple[str, str]]) -> RerankResult:
        return RerankResult(
            status=RerankerStatus.NOT_REQUESTED, model="", detail="no reranker requested"
        )


class CrossEncoderReranker(Reranker):
    """sentence-transformers cross-encoder reranker (lazy, optional)."""

    name = "cross-encoder"

    def __init__(self, model: str = "") -> None:
        self.model = model or DEFAULT_CROSS_ENCODER_MODEL
        self._encoder: Any | None = None
        self._load_error: str | None = None

    def _load(self) -> Any | None:
        if self._encoder is not None:
            return self._encoder
        if self._load_error is not None:
            return None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # ImportError or a broken install
            self._load_error = (
                f"sentence-transformers CrossEncoder is unavailable ({exc}). "
                "Install backend/requirements-embeddings.txt to enable reranking."
            )
            logger.warning("Cross-encoder unavailable: %s", self._load_error)
            return None
        try:
            self._encoder = CrossEncoder(self.model)
        except Exception as exc:  # model download/network/invalid name
            self._load_error = (
                f"could not load cross-encoder '{self.model}' ({exc}). "
                "The model may be unavailable offline; retrieval continues without reranking."
            )
            logger.warning("Cross-encoder load failed: %s", self._load_error)
            return None
        return self._encoder

    def availability(self) -> tuple[bool, str]:
        if self._load() is None:
            return (False, self._load_error or "cross-encoder unavailable")
        return (True, "")

    def rerank(self, query: str, items: Sequence[tuple[str, str]]) -> RerankResult:
        encoder = self._load()
        if encoder is None:
            # Honest fallback: keep the incoming order, say why.
            return RerankResult(
                status=RerankerStatus.UNAVAILABLE_FALLBACK,
                model=self.model,
                detail=self._load_error or "cross-encoder unavailable",
            )
        if not items:
            return RerankResult(
                status=RerankerStatus.APPLIED, model=self.model, order=[], scores={},
                detail="no candidates to rerank",
            )
        pairs = [[query, text] for _, text in items]
        try:
            raw = encoder.predict(pairs)
        except Exception as exc:
            logger.warning("Cross-encoder prediction failed: %s", exc)
            return RerankResult(
                status=RerankerStatus.UNAVAILABLE_FALLBACK,
                model=self.model,
                detail=f"cross-encoder prediction failed: {exc}",
            )
        scores: dict[str, float] = {}
        for (chunk_id, _), value in zip(items, raw):
            try:
                scores[chunk_id] = float(value)
            except (TypeError, ValueError):
                continue
        order = [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        # Candidates whose score could not be read keep their incoming order at
        # the end rather than being dropped.
        for chunk_id, _ in items:
            if chunk_id not in scores:
                order.append(chunk_id)
        return RerankResult(
            status=RerankerStatus.APPLIED,
            model=self.model,
            order=order,
            scores=scores,
            detail=f"cross-encoder scored {len(scores)} candidate(s)",
        )


@lru_cache(maxsize=4)
def _cached_reranker(name: str, model: str) -> Reranker:
    """One instance per (name, model) so the model is loaded at most once."""
    if name == "cross-encoder":
        return CrossEncoderReranker(model)
    return NoReranker()


def get_reranker(name: str, model: str = "") -> Reranker:
    """Factory: 'none' | 'cross-encoder'. Unknown names raise (never guess)."""
    key = (name or "none").strip().lower()
    if key in ("", "none", "null", "off"):
        return _cached_reranker("none", "")
    if key == "cross-encoder":
        return _cached_reranker("cross-encoder", model or DEFAULT_CROSS_ENCODER_MODEL)
    raise ValueError(f"Unknown reranker {name!r}. Available: ['none', 'cross-encoder']")
