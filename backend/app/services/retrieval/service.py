"""Retrieval orchestration (V6 Phases 2, 8, 18).

One place resolves a request into a validated `RetrievalParams`, builds the
configured strategy through the registry, runs it and records what happened.
The API routes, the evaluator and (later) the answering service all go through
this, so configuration resolution and observability cannot drift between callers.

Honesty rules enforced here:
* the embedding identity a KB was indexed with is verified before a vector query;
* a strategy that does not need embeddings never loads the embedding model;
* every run is recorded with its configuration, weights, corpus version and
  timings — an unmeasured field stays None rather than becoming 0;
* nothing is deleted or rebuilt here: the service is read-only on the corpus.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from app.config import Settings, get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.models import KnowledgeBase, RetrievalResponse
from app.schemas.retrieval import (
    RetrievalConfigRecord,
    RetrievalParams,
    RetrievalRun,
    RetrievalStage,
    RetrievalStrategy,
    RerankerStatus,
)
from app.services.retrieval.retriever import (
    RetrievalError,
    Retriever,
    available_retrievers,
    describe_retrievers,
    get_retriever,
    resolve_retriever_spec,
)
from app.services.vector_store import factory as vector_factory

logger = logging.getLogger(__name__)


class KnowledgeBaseNotFound(RuntimeError):
    pass


class RetrievalConfigError(RetrievalError):
    """Invalid or unknown retrieval configuration — a CLIENT error (HTTP 400).

    Kept distinct from RetrievalError so an unknown strategy name is never
    reported as a service outage. Configuration problems are the caller's to fix.
    """


class _NoEmbeddingProvider:
    """Placeholder for strategies that never embed anything (e.g. BM25).

    Deliberately loud if something tries to use it: a lexical-only strategy must
    not silently require a multi-hundred-MB model to answer a query.
    """

    name = "unused"
    model = ""
    dimensions = 0
    is_fallback = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RetrievalError(
            "This retrieval strategy does not use embeddings; a query needed one, "
            "which means the strategy was misconfigured."
        )


class RetrievalService:
    """Resolves configuration and runs retrieval strategies for a knowledge base."""

    def __init__(self, repo: Repository, settings: Settings | None = None) -> None:
        self._repo = repo
        self._settings = settings or get_settings()

    # -- configuration ------------------------------------------------------

    def get_config(self, kb_id: str) -> RetrievalConfigRecord | None:
        """Persisted configuration, or None when the KB has never been configured."""
        return self._repo.get_retrieval_config(kb_id)

    def save_config(
        self, kb_id: str, params: RetrievalParams, note: str = ""
    ) -> RetrievalConfigRecord:
        record = RetrievalConfigRecord(kb_id=kb_id, params=params, note=note)
        self._repo.save_retrieval_config(record)
        return record

    def resolve_params(
        self,
        kb_id: str,
        *,
        strategy: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> RetrievalParams:
        """Merge stored configuration, request overrides and defaults.

        Precedence: explicit request override > stored KB configuration > defaults.
        `None` values in `overrides` mean "not specified" and never clobber a
        stored value.
        """
        stored = self.get_config(kb_id)
        base: dict[str, Any] = stored.params.model_dump(mode="json") if stored else {}
        cleaned = {k: v for k, v in (overrides or {}).items() if v is not None}
        merged = {**base, **cleaned}
        if strategy:
            key = strategy.strip().lower()
            if key not in available_retrievers():
                raise RetrievalConfigError(
                    f"Unknown retrieval strategy {strategy!r}. "
                    f"Available: {available_retrievers()}"
                )
            merged["strategy"] = key
        try:
            return RetrievalParams(**merged)
        except Exception as exc:
            raise RetrievalConfigError(f"Invalid retrieval configuration: {exc}") from exc

    def available_strategies(self) -> list[dict[str, Any]]:
        return describe_retrievers()

    # -- execution ----------------------------------------------------------

    def build_retriever(
        self, kb: KnowledgeBase, params: RetrievalParams
    ) -> tuple[Retriever, list[str]]:
        """Build the configured strategy. Returns (retriever, notes)."""
        from app.services.embeddings.provider import (
            EmbeddingError,
            EmbeddingIdentity,
            create_embedding_provider,
        )

        notes: list[str] = []
        spec = resolve_retriever_spec(params.strategy.value)
        if spec.needs_embedding:
            expected = None
            if kb.embedding_identity:
                expected = EmbeddingIdentity(**kb.embedding_identity)
            try:
                embedder = create_embedding_provider(self._settings, expected=expected)
            except EmbeddingError as exc:
                raise RetrievalError(str(exc)) from exc
            identity = embedder.identity() if hasattr(embedder, "identity") else None
        else:
            embedder = _NoEmbeddingProvider()  # type: ignore[assignment]
            identity = None
            notes.append(
                "Lexical-only strategy: no embedding model was loaded, so the vector "
                "index and the query embedding are not involved in this result set."
            )
        store = vector_factory.create_vector_store(
            self._settings, backend=(kb.vector_backend or "qdrant")
        )
        retriever = get_retriever(
            spec.name,
            embedder,  # type: ignore[arg-type]
            store,
            identity,
            repo=self._repo,
            settings=self._settings,
        )
        return retriever, notes

    def run(
        self,
        kb_id: str,
        query: str,
        *,
        strategy: str | None = None,
        overrides: dict[str, Any] | None = None,
        persist_run: bool = True,
    ) -> RetrievalResponse:
        """Run retrieval and (by default) persist an observability record."""
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")
        params = self.resolve_params(kb_id, strategy=strategy, overrides=overrides)
        retriever, notes = self.build_retriever(kb, params)
        response = retriever.retrieve_with_params(kb_id, query, params)

        corpus_version, corpus_fingerprint = self._corpus_context(kb_id)
        response = response.model_copy(
            update={
                "strategy": params.strategy.value,
                "params": params,
                "corpus_version": corpus_version,
                "corpus_fingerprint": corpus_fingerprint,
                "notes": list(dict.fromkeys([*response.notes, *notes])),
            }
        )
        if not persist_run:
            return response
        run = self._record_run(kb_id, query, params, response)
        return response.model_copy(update={"retrieval_run_id": run.id})

    def _corpus_context(self, kb_id: str) -> tuple[str | None, str | None]:
        """Version + fingerprint of the corpus the vectors belong to, when known."""
        try:
            latest = self._repo.latest_corpus_version(kb_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read the corpus version for %s: %s", kb_id, exc)
            return (None, None)
        if latest is None:
            return (None, None)
        return (latest.version, latest.fingerprint)

    def _record_run(
        self, kb_id: str, query: str, params: RetrievalParams, response: RetrievalResponse
    ) -> RetrievalRun:
        dense_weight, lexical_weight = params.effective_weights()
        stages = list(response.stages)
        reranker_status = response.reranker.status if response.reranker else None
        run = RetrievalRun(
            id=f"rr_{uuid.uuid4().hex[:12]}",
            kb_id=kb_id,
            query=query,
            strategy=params.strategy.value,
            config_fingerprint=params.config_fingerprint(),
            params=params,
            applied_weights=(
                {"dense": round(dense_weight, 4), "bm25": round(lexical_weight, 4)}
                if params.strategy
                in (RetrievalStrategy.HYBRID, RetrievalStrategy.HYBRID_RERANKED)
                else {}
            ),
            normalization=params.normalization.value,
            rrf_k=params.rrf_k if params.fusion.value == "rrf" else None,
            embedding_model=response.embedding_model,
            vector_backend=response.retrieval_backend,
            reranker=params.reranker_name(),
            reranker_model=response.reranker.model if response.reranker else "",
            reranker_status=reranker_status or RerankerStatus.NOT_REQUESTED,
            corpus_version=response.corpus_version,
            corpus_fingerprint=response.corpus_fingerprint,
            candidate_counts=_candidate_counts(stages),
            result_count=len(response.results),
            timings=response.timings.model_copy() if response.timings else _empty_timings(),
            stages=stages,
            notes=response.notes,
        )
        try:
            self._repo.create_retrieval_run(run)
        except Exception as exc:  # observability must never break a query
            logger.warning("Could not persist retrieval run for %s: %s", kb_id, exc)
        return run

    def recent_runs(self, kb_id: str, limit: int = 50):
        return self._repo.list_retrieval_run_summaries(kb_id, limit=limit)

    def get_run(self, kb_id: str, run_id: str) -> RetrievalRun | None:
        return self._repo.get_retrieval_run(kb_id, run_id)

    def bm25_index_status(self, kb_id: str) -> dict[str, Any]:
        """Diagnostics for the lexical index, including explicit staleness.

        Read-only: it reports whether a persisted index matches the current corpus
        revision and never builds or invalidates anything.
        """
        from app.services.retrieval.bm25 import LexicalIndexStore

        return LexicalIndexStore(self._repo).status(kb_id)


def _candidate_counts(stages: list[RetrievalStage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stage in stages:
        if stage.count is not None:
            counts[stage.name] = stage.count
    return counts


def _empty_timings():
    from app.schemas.retrieval import RetrievalTimings

    return RetrievalTimings()


__all__ = [
    "KnowledgeBaseNotFound",
    "RetrievalConfigError",
    "RetrievalService",
]
