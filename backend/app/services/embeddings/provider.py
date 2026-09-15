"""EmbeddingProvider abstraction.

Primary: local SentenceTransformers (no API key, works offline after first download).
Optional: OpenAI-compatible embeddings.
Fallback (clearly labelled): deterministic hashing embeddings for tests/dev when
no model can be loaded — NEVER for real evaluation.
"""
from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Immutable description of an embedding configuration.

    Indexed corpora and queries MUST use the same identity; mismatches raise
    EmbeddingError instead of silently producing incompatible vectors.
    """

    provider: str
    model: str
    dimensions: int
    is_fallback: bool = False

    def describe(self) -> str:
        tag = " [DEV FALLBACK — not semantic]" if self.is_fallback else ""
        return f"{self.provider}/{self.model} ({self.dimensions}d){tag}"


class EmbeddingProvider(ABC):
    name: str = "base"
    model: str = "unknown"
    dimensions: int = 0
    is_fallback: bool = False

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence-transformers model. Downloads weights on first use."""

    name = "sentence-transformers"

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Run: pip install -r backend/requirements-embeddings.txt"
            ) from exc
        try:
            self._model = SentenceTransformer(model)
        except Exception as exc:
            raise EmbeddingError(f"Could not load embedding model '{model}': {exc}") from exc
        self.model = model
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(self.name, self.model, self.dimensions, self.is_fallback)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        return [e.tolist() for e in embeddings]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = "") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError(
                "The 'openai' package is not installed. "
                "Run: pip install -r requirements-openai.txt"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        # Determine dimensionality with a probe call.
        resp = self._client.embeddings.create(input=["probe"], model=model)
        self.dimensions = len(resp.data[0].embedding)

    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(self.name, self.model, self.dimensions, self.is_fallback)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = self._client.embeddings.create(input=batch, model=self.model)
            out.extend(d.embedding for d in resp.data)
        return out


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free fallback for tests/dev ONLY.

    Uses character n-gram hashing into a fixed-size vector with L2
    normalization. Clearly NOT semantic: it is labelled is_fallback=True so the
    UI/API can disclose it and real evaluation can refuse to use it.
    """

    name = "hashing-dev-fallback"
    model = "char-ngram-hashing-256"
    dimensions = 256
    is_fallback = True

    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(self.name, self.model, self.dimensions, self.is_fallback)

    def _hash_ngrams(self, text: str) -> list[float]:
        v = [0.0] * self.dimensions
        text = text.lower()
        tokens = text.split()
        grams = tokens + [text[i : i + 4] for i in range(0, max(0, len(text) - 3))]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimensions
            v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_ngrams(t) for t in texts]


@lru_cache(maxsize=8)
def _cached_provider(provider: str, model: str, api_key: str) -> EmbeddingProvider:
    """Process-level provider cache.

    SentenceTransformer model loading is expensive (hundreds of MB, seconds of
    startup); it must happen ONCE per process, not once per API request. Keyed
    on (provider, model, api_key) so a config change constructs a new instance.
    """
    if provider == "sentence-transformers":
        return SentenceTransformerProvider(model)
    if provider == "openai":
        return OpenAIEmbeddingProvider(api_key=api_key, model=model)
    if provider == "hashing-dev-fallback":
        return HashingEmbeddingProvider()
    raise EmbeddingError(f"Unknown EMBEDDING_PROVIDER '{provider}'")


def create_embedding_provider(
    settings: Settings,
    expected: EmbeddingIdentity | None = None,
) -> EmbeddingProvider:
    """Factory returning a CACHED provider for the configured backend.

    expected: if given (e.g. the identity a KB was originally indexed with),
    a mismatch in provider/model/dimensions raises EmbeddingError with a clear
    message instead of silently producing incompatible vectors.
    """
    provider_name = (settings.embedding_provider or "").strip().lower()
    model = (settings.embedding_model or "").strip()
    if provider_name == "openai":
        if not settings.embedding_api_key:
            raise EmbeddingError("EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=openai")
        model = model or "text-embedding-3-small"
    elif provider_name == "sentence-transformers" and not model:
        model = "sentence-transformers/all-MiniLM-L6-v2"
    prov = _cached_provider(provider_name, model, settings.embedding_api_key)
    if expected is not None:
        actual = prov.identity()
        if (
            actual.model != expected.model
            or actual.dimensions != expected.dimensions
            or actual.provider != expected.provider
        ):
            raise EmbeddingError(
                "Embedding model mismatch: this knowledge base was indexed with "
                f"{expected.describe()} but the current configuration is {actual.describe()}. "
                "Re-index the knowledge base with the current model, or restore the "
                "original EMBEDDING_PROVIDER/EMBEDDING_MODEL settings."
            )
    return prov
