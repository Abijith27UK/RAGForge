"""Retrieval strategies + the retriever registry.

V1-V5 shipped a single dense strategy. V6 adds real BM25, hybrid fusion (weighted
and RRF) and optional reranking — all behind this ONE interface, so the API, the
evaluator, the CLI scripts and the UI never change shape when a strategy is added.

The registry is deliberately not hard-coded at call sites: `get_retriever()`
resolves a strategy name to an implementation, and the API records which one ran.

Backwards compatibility (important — V1-V5 callers must keep working):
* `Retriever.__init__(embedding_provider, vector_store, identity=None)` — the
  positional 3-arg form used by existing tests still works.
* `retrieve(kb_id, query, top_k=5, filters=None, min_score=0.0)` — still the
  abstract surface every strategy implements, still what the evaluator calls.
* `register_retriever(name, factory)` — still accepts a plain 3-arg factory.
* `available_retrievers()` / `get_retriever()` keep their signatures; `get_retriever`
  gained keyword-only `repo`/`settings` for strategies that need corpus access.

Built-in strategies register LAZILY (on first registry query) so importing this
module never triggers a heavy import or an import cycle.
"""
from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from app.schemas.models import Chunk, RetrievalResponse, RetrievalResult
from app.schemas.retrieval import (
    RetrievalParams,
    RetrievalStageStatus,
    RetrievalStrategy,
    ScoreBreakdown,
)
from app.services.embeddings.provider import EmbeddingIdentity, EmbeddingProvider
from app.services.retrieval.fusion import (
    DENSE,
    LEXICAL,
    Candidate,
    build_why,
)
from app.services.retrieval.trace import RetrievalTrace
from app.services.vector_store.qdrant_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

#: Provenance keys copied from the vector-store payload (or a Chunk row) into
#: every result. Everything needed to cite
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


# ---------------------------------------------------------------------------
# Provenance helpers (shared by every strategy so provenance cannot drift)
# ---------------------------------------------------------------------------

def provenance_from_mapping(source: dict[str, Any]) -> dict[str, Any]:
    """Extract the provenance subset from a payload/row mapping.

    Only present values are copied: an absent page number stays absent instead of
    becoming 0/None in the output.
    """
    return {k: source.get(k) for k in PROVENANCE_KEYS if source.get(k) is not None}


def provenance_from_chunk(chunk: Chunk) -> dict[str, Any]:
    """Same provenance contract, sourced from a SQLite chunk row (BM25 path)."""
    return provenance_from_mapping(chunk.model_dump())


def candidate_from_chunk(chunk: Chunk) -> Candidate:
    return Candidate(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text or "",
        provenance=provenance_from_chunk(chunk),
    )


def candidates_from_response(response: RetrievalResponse, source: str) -> list[Candidate]:
    """Convert a sub-retriever response into fusion candidates.

    Ranks are assigned from the response order, which is the order the
    sub-strategy ranked them in — never re-sorted here.
    """
    out: list[Candidate] = []
    for idx, result in enumerate(response.results, start=1):
        candidate = Candidate(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            text=result.text,
            provenance=dict(result.provenance),
        )
        candidate.raw_scores[source] = float(result.retrieval_score or result.score)
        candidate.ranks[source] = idx
        candidate.stages.append(source)
        out.append(candidate)
    return out


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def results_from_candidates(
    candidates: list[Candidate],
    *,
    method: str,
    min_score: float,
    top_k: int,
) -> list[RetrievalResult]:
    """Build the final, provenance-complete result list.

    `score` is the score this strategy ranks by; the raw per-source scores stay
    available in `score_breakdown` so nothing is lost in translation.
    """
    results: list[RetrievalResult] = []
    rank = 0
    for cand in candidates:
        if len(results) >= top_k:
            break
        score = cand.reported()
        if score is None or score < min_score:
            continue
        rank += 1
        # The "strategy score" is the fused score when fusion ran, otherwise the
        # raw score of the single source that produced this candidate.
        strategy_score = cand.fused_score
        if strategy_score is None:
            strategy_score = cand.raw_scores.get(DENSE)
        if strategy_score is None:
            strategy_score = cand.raw_scores.get(LEXICAL)
        results.append(
            RetrievalResult(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                text=cand.text,
                score=round(float(score), 4),
                provenance=dict(cand.provenance),
                rank=rank,
                retrieval_method=method,
                retrieval_score=_round(strategy_score),
                rerank_score=_round(cand.rerank_score),
                score_breakdown=ScoreBreakdown(
                    dense_score=_round(cand.raw_scores.get(DENSE)),
                    lexical_score=_round(cand.raw_scores.get(LEXICAL)),
                    normalized_dense=_round(cand.normalized.get(DENSE)),
                    normalized_lexical=_round(cand.normalized.get(LEXICAL)),
                    fused_score=_round(cand.fused_score),
                    rerank_score=_round(cand.rerank_score),
                    dense_contribution=_round(cand.contributions.get(DENSE)),
                    lexical_contribution=_round(cand.contributions.get(LEXICAL)),
                ),
                why=build_why(cand, method=method),
                stages=list(cand.stages),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Retriever interface
# ---------------------------------------------------------------------------

class Retriever(ABC):
    """Normalized retrieval surface.

    Every strategy returns a `RetrievalResponse` whose `retrieval_backend` names
    itself honestly and whose results carry full provenance.
    """

    backend = "base"
    strategy_name = "base"

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

    def retrieve_with_params(
        self, kb_id: str, query: str, params: RetrievalParams
    ) -> RetrievalResponse:
        """Canonical V6 entry point. Rich strategies override this.

        The default implementation maps the parameter object onto the classic
        5-argument surface, so any pre-V6 strategy keeps working unchanged and
        never receives a keyword it does not understand.
        """
        return self.retrieve(
            kb_id,
            query,
            top_k=params.top_k,
            filters=params.filters or None,
            min_score=params.min_score,
        )

    # -- shared helpers ----------------------------------------------------

    def _embedding_label(self) -> str:
        """Report the full identity (incl. dev-fallback labelling) honestly."""
        return self._identity.describe() if self._identity else self._embedder.model

    def _build_results(self, hits: list[dict[str, Any]], min_score: float) -> list[RetrievalResult]:
        """Legacy helper (store hits -> results). Kept for subclasses/tests."""
        candidates: list[Candidate] = []
        for idx, hit in enumerate(hits, start=1):
            payload = hit.get("payload") or {}
            candidate = Candidate(
                chunk_id=hit.get("chunk_id") or payload.get("chunk_id") or "unknown",
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                provenance=provenance_from_mapping(payload),
            )
            candidate.raw_scores[DENSE] = float(hit.get("score") or 0.0)
            candidate.ranks[DENSE] = idx
            candidate.stages.append(DENSE)
            candidates.append(candidate)
        return results_from_candidates(
            candidates, method=DENSE, min_score=min_score, top_k=len(candidates)
        )

    def corpus_names(self) -> list[str]:
        """Which corpus sources this strategy reads. Used by the UI/trace."""
        return [self.strategy_name]


def apply_diversity(
    candidates: list[Candidate],
    *,
    params: RetrievalParams,
    trace: RetrievalTrace,
    vector_lookup: dict[str, list[float]] | None = None,
    query_vector: list[float] | None = None,
) -> list[Candidate]:
    """Apply the configured diversification to an ordered candidate list.

    Returns the reordered list. Candidates over a document cap are demoted, never
    dropped, so a short corpus can still fill top_k. When MMR is requested but
    candidate vectors are unavailable the stage is recorded as unavailable and the
    document cap / relevance order is used instead.
    """
    from app.services.retrieval.diversity import cap_per_document, mmr_select

    if params.diversity == "none" and params.max_per_document is None:
        trace.skip("diversity", "disabled")
        return candidates
    order = [c.chunk_id for c in candidates]
    by_id = {c.chunk_id: c for c in candidates}
    document_of = {c.chunk_id: c.document_id for c in candidates}
    relevance = {c.chunk_id: (c.reported() or 0.0) for c in candidates}
    use_mmr = (
        params.diversity == "mmr" and bool(vector_lookup) and bool(query_vector)
    )
    if use_mmr:
        outcome = mmr_select(
            relevance,
            vector_lookup or {},
            lambda_=params.diversity_lambda,
            k=len(order),
            max_per_document=params.max_per_document,
            document_of=document_of,
        )
        new_order = outcome.kept + [cid for cid in order if cid not in set(outcome.kept)]
        trace.add_stage("diversity", detail=outcome.summary(), count=len(new_order))
    else:
        if params.diversity == "mmr":
            trace.add_stage(
                "diversity",
                status=RetrievalStageStatus.UNAVAILABLE,
                detail="MMR requested but candidate vectors are unavailable; "
                "used the relevance/document-cap order instead",
            )
        outcome = cap_per_document(order, document_of, params.max_per_document)
        new_order = outcome.kept
        for cid, reason in outcome.dropped.items():
            logger.debug("Diversity demoted %s: %s", cid, reason)
        trace.add_stage("diversity", detail=outcome.summary(), count=len(new_order))
    return [by_id[cid] for cid in new_order if cid in by_id]


def apply_min_score_and_cut(
    candidates: list[Candidate], *, min_score: float, top_k: int
) -> list[Candidate]:
    """Cut to top_k after the relevance floor, using the reported score."""
    kept: list[Candidate] = []
    for cand in candidates:
        score = cand.reported()
        if score is None or score < min_score:
            continue
        kept.append(cand)
        if len(kept) >= top_k:
            break
    return kept


class DenseRetriever(Retriever):
    """Dense-only retriever. Backend name is reported honestly in responses."""

    backend = "qdrant-dense"
    strategy_name = RetrievalStrategy.DENSE.value

    def candidate_pool(
        self,
        kb_id: str,
        query: str,
        *,
        candidate_k: int,
        filters: dict[str, Any] | None = None,
        trace: RetrievalTrace | None = None,
    ) -> tuple[list[Candidate], list[float]]:
        """Embed the query and return (raw dense candidate pool, query vector).

        The pool is ordered by the vector store's own similarity ranking; ranks are
        recorded so fusion can use them and the UI can explain them.
        """
        trace = trace or RetrievalTrace()
        try:
            with trace.measure("embed", detail="query embedding"):
                query_vector = self._embedder.embed_texts([query])[0]
        except Exception as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc
        try:
            with trace.measure("dense", detail="vector search"):
                hits = self._store.search(
                    kb_id, query_vector, top_k=candidate_k, filters=filters
                )
        except VectorStoreError as exc:
            raise RetrievalError(f"Vector store error: {exc}") from exc
        candidates: list[Candidate] = []
        for idx, hit in enumerate(hits, start=1):
            payload = hit.get("payload") or {}
            candidate = Candidate(
                chunk_id=hit.get("chunk_id") or payload.get("chunk_id") or "unknown",
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                provenance=provenance_from_mapping(payload),
            )
            candidate.raw_scores[DENSE] = float(hit.get("score") or 0.0)
            candidate.ranks[DENSE] = idx
            candidate.reported_score = candidate.raw_scores[DENSE]
            candidate.stages.append(DENSE)
            candidates.append(candidate)
        for stage in reversed(trace.stages):
            if stage.name == "dense":
                stage.count = len(candidates)
                break
        return candidates, query_vector

    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResponse:
        trace = RetrievalTrace()
        candidates, _ = self.candidate_pool(
            kb_id, query, candidate_k=top_k, filters=filters, trace=trace
        )
        for cand in candidates:
            cand.reported_score = cand.raw_scores.get(DENSE, 0.0)
        response = RetrievalResponse(
            query=query,
            top_k=top_k,
            results=results_from_candidates(
                candidates, method=DENSE, min_score=min_score, top_k=top_k
            ),
            embedding_model=self._embedding_label(),
            retrieval_backend=self.backend,
            strategy=self.strategy_name,
        )
        return trace.apply(response)

    def retrieve_with_params(
        self, kb_id: str, query: str, params: RetrievalParams
    ) -> RetrievalResponse:
        if params.diversity == "none" and params.max_per_document is None:
            # Preserve the exact classic behaviour (and cost) when nothing extra
            # was requested.
            return self.retrieve(
                kb_id,
                query,
                top_k=params.top_k,
                filters=params.filters or None,
                min_score=params.min_score,
            )
        trace = RetrievalTrace()
        candidate_k = params.resolved_candidate_k()
        candidates, _ = self.candidate_pool(
            kb_id, query, candidate_k=candidate_k, filters=params.filters or None, trace=trace
        )
        for cand in candidates:
            cand.normalized[DENSE] = cand.raw_scores.get(DENSE, 0.0)
            cand.reported_score = cand.raw_scores.get(DENSE, 0.0)
        candidates = apply_diversity(candidates, params=params, trace=trace)
        candidates = apply_min_score_and_cut(
            candidates, min_score=params.min_score, top_k=params.top_k
        )
        response = RetrievalResponse(
            query=query,
            top_k=params.top_k,
            results=results_from_candidates(
                candidates, method=DENSE, min_score=params.min_score, top_k=params.top_k
            ),
            embedding_model=self._embedding_label(),
            retrieval_backend=self.backend,
            strategy=self.strategy_name,
            params=params,
        )
        return trace.apply(response)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RetrieverFactory = Callable[..., Retriever]


@dataclass(frozen=True)
class RetrieverSpec:
    """A registered strategy: how to build it and what it needs."""

    name: str
    factory: RetrieverFactory
    description: str = ""
    requires_repo: bool = False
    needs_embedding: bool = True
    is_baseline: bool = False
    notes: str = ""
    aliases: tuple[str, ...] = field(default=())


_RETRIEVERS: dict[str, RetrieverSpec] = {}
_BUILTINS_LOADED = False


def register_retriever(
    name: str,
    factory: RetrieverFactory,
    *,
    description: str = "",
    requires_repo: bool = False,
    needs_embedding: bool = True,
    is_baseline: bool = False,
    notes: str = "",
    aliases: tuple[str, ...] = (),
) -> None:
    """Register (or replace) a retrieval strategy.

    A plain 3-argument factory — the V4 contract — is still accepted; the extra
    capabilities are opt-in keyword flags. Replacing a name replaces its whole
    spec, so a stub can override a built-in without inheriting its repo
    requirement.
    """
    key = (name or "").strip().lower()
    spec = RetrieverSpec(
        name=key,
        factory=factory,
        description=description,
        requires_repo=requires_repo,
        needs_embedding=needs_embedding,
        is_baseline=is_baseline,
        notes=notes,
        aliases=tuple(a.strip().lower() for a in aliases if a),
    )
    _RETRIEVERS[key] = spec
    for alias in spec.aliases:
        _RETRIEVERS[alias] = spec


def snapshot_retrievers() -> dict[str, RetrieverSpec]:
    """Copy of the current registry, for callers that temporarily override it.

    The registry is process-global, so a test (or an experiment) that swaps a
    built-in strategy MUST restore it afterwards or every later caller silently
    gets the substitute. Pair with `restore_retrievers` in a finally block.
    """
    _load_builtins()
    return dict(_RETRIEVERS)


def restore_retrievers(snapshot: dict[str, RetrieverSpec]) -> None:
    """Restore a registry captured by `snapshot_retrievers`, dropping anything
    registered in the meantime."""
    _RETRIEVERS.clear()
    _RETRIEVERS.update(snapshot)


def _load_builtins() -> None:
    """Register built-in strategies once, lazily (avoids import cycles)."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from app.services.retrieval.bm25 import Bm25Retriever
    from app.services.retrieval.hybrid import HybridRerankedRetriever, HybridRetriever

    register_retriever(
        "dense",
        lambda e, s, i, **_: DenseRetriever(e, s, i),
        description="Dense vector search (Qdrant + embeddings). Semantic, no lexical matching.",
        is_baseline=True,
        notes="Baseline strategy from V1-V5.",
        aliases=("qdrant-dense",),
    )
    register_retriever(
        "bm25",
        lambda e, s, i, *, repo=None, **_: Bm25Retriever(e, s, i, repo=repo),
        description="Lexical BM25 over the indexed chunks (real term weighting: k1/b).",
        requires_repo=True,
        needs_embedding=False,
        notes="Exact-term matching; strong for terminology, weak for paraphrase.",
    )
    register_retriever(
        "hybrid",
        lambda e, s, i, *, repo=None, **_: HybridRetriever(e, s, i, repo=repo),
        description="Dense + BM25 with configurable normalization and fusion (weighted or RRF).",
        requires_repo=True,
    )
    register_retriever(
        "hybrid_reranked",
        lambda e, s, i, *, repo=None, **_: HybridRerankedRetriever(e, s, i, repo=repo),
        description="Hybrid retrieval followed by an optional cross-encoder reranker.",
        requires_repo=True,
        notes="Falls back (and says so) when the reranker is unavailable.",
    )


def available_retrievers() -> list[str]:
    _load_builtins()
    return sorted(_RETRIEVERS)


def describe_retrievers() -> list[dict[str, Any]]:
    """Metadata for the UI: what each strategy is and what it needs."""
    _load_builtins()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for name in sorted(_RETRIEVERS):
        spec = _RETRIEVERS[name]
        if spec.name in seen:
            continue
        seen.add(spec.name)
        out.append(
            {
                "name": spec.name,
                "description": spec.description,
                "requires_repo": spec.requires_repo,
                "needs_embedding": spec.needs_embedding,
                "is_baseline": spec.is_baseline,
                "notes": spec.notes,
                "aliases": list(spec.aliases),
            }
        )
    return out


def resolve_retriever_spec(name: str) -> RetrieverSpec:
    _load_builtins()
    spec = _RETRIEVERS.get((name or "dense").strip().lower())
    if spec is None:
        raise ValueError(f"Unknown retriever {name!r}. Available: {available_retrievers()}")
    return spec


def get_retriever(
    name: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    identity: EmbeddingIdentity | None = None,
    *,
    repo: Any | None = None,
    settings: Any | None = None,
    **extra: Any,
) -> Retriever:
    """Build a strategy by name. Never guesses: an unknown name raises.

    Extra dependencies (repo/settings) are forwarded only to factories that
    declare them, so a legacy 3-argument factory — the V4 contract — keeps
    working verbatim.
    """
    spec = resolve_retriever_spec(name)
    if spec.requires_repo and repo is None:
        raise ValueError(
            f"Retriever {spec.name!r} needs corpus access (repository) but none was provided"
        )
    kwargs = _accepted_kwargs(spec.factory, repo=repo, settings=settings, extra=extra)
    return spec.factory(embedding_provider, vector_store, identity, **kwargs)


def _accepted_kwargs(
    factory: RetrieverFactory,
    *,
    repo: Any,
    settings: Any,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Keyword arguments a factory actually accepts (empty for 3-arg factories)."""
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # builtins/C callables: assume it takes none
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {"repo": repo, "settings": settings, **extra}
    accepted: dict[str, Any] = {}
    if "repo" in params:
        accepted["repo"] = repo
    if "settings" in params:
        accepted["settings"] = settings
    for key, value in extra.items():
        if key in params:
            accepted[key] = value
    return accepted
