"""Hybrid retrieval: dense + lexical, with explicit fusion and optional rerank.

Pipeline (every stage recorded on the response):

    query
      -> candidate retrieval     dense (vector) + lexical (BM25), candidate_k each
      -> normalization           min-max or rank, per candidate pool
      -> fusion                  weighted (default 0.65/0.35) or RRF
      -> diversification         optional (document cap / MMR)
      -> reranking               optional cross-encoder, honest fallback
      -> final top_k

Design decisions worth stating:

* Scores are normalized BEFORE fusion. Dense cosine and BM25 are not comparable,
  and blind addition would let whichever scale happens to be larger dominate.
* If one side returns no candidates (e.g. an empty lexical index), the weights are
  RESCALED to the surviving side and that is written into the response notes and
  into the run's `applied_weights`. Silently keeping 0.65/0.35 would cap every
  fused score at 0.65 and look like poor relevance.
* The reported `score` is the fused score (comparable, [0,1]). A reranker's raw
  score is reported separately as `rerank_score` and drives the ORDER only —
  cross-encoder logits are unbounded and must not be shown as a relevance
  percentage.
* Fusion never invents candidates: only chunks returned by a real retriever can
  appear in the output.
"""
from __future__ import annotations

import logging
from typing import Any

from app.repositories.sqlite_repo import Repository
from app.schemas.models import RetrievalResponse
from app.schemas.retrieval import (
    RerankerChoice,
    RerankerReport,
    RerankerStatus,
    RetrievalParams,
    RetrievalStageStatus,
    RetrievalStrategy,
)
from app.services.retrieval.bm25 import Bm25Retriever
from app.services.retrieval.fusion import (
    DENSE,
    LEXICAL,
    Candidate,
    merge_candidates,
    normalize_scores,
    order_by_score,
    reciprocal_rank_fusion,
    weighted_fuse,
)
from app.services.retrieval.rerank import get_reranker
from app.services.retrieval.retriever import (
    DenseRetriever,
    RetrievalError,
    Retriever,
    apply_diversity,
    apply_min_score_and_cut,
    candidates_from_response,
    results_from_candidates,
)
from app.services.retrieval.trace import RetrievalTrace

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """Dense + BM25 fusion with configurable normalization and fusion method."""

    backend = "hybrid"
    strategy_name = RetrievalStrategy.HYBRID.value
    #: Subclass hook: 'none' (plain hybrid) or 'cross-encoder'.
    default_reranker = RerankerChoice.NONE.value

    def __init__(
        self,
        embedding_provider: Any,
        vector_store: Any,
        identity: Any = None,
        *,
        repo: Repository | None = None,
        dense: Retriever | None = None,
        lexical: Retriever | None = None,
        reranker: Any | None = None,
        index_store: Any | None = None,
    ) -> None:
        super().__init__(embedding_provider, vector_store, identity)
        self._repo = repo
        self._dense: Retriever = dense or DenseRetriever(embedding_provider, vector_store, identity)
        if lexical is not None:
            self._lexical: Retriever | None = lexical
        elif repo is not None:
            self._lexical = Bm25Retriever(
                embedding_provider, vector_store, identity, repo=repo, index_store=index_store
            )
        else:
            self._lexical = None
        self._reranker_override = reranker

    # -- candidate retrieval ------------------------------------------------

    def _dense_pool(
        self, kb_id: str, query: str, candidate_k: int, params: RetrievalParams, trace: RetrievalTrace
    ) -> tuple[list[Candidate], list[float]]:
        filters = params.filters or None
        if isinstance(self._dense, DenseRetriever):
            return self._dense.candidate_pool(
                kb_id, query, candidate_k=candidate_k, filters=filters, trace=trace
            )
        response = self._dense.retrieve(kb_id, query, top_k=candidate_k, filters=filters)
        query_vector: list[float] = []
        if params.diversity == "mmr":
            try:
                query_vector = self._embedder.embed_texts([query])[0]
            except Exception as exc:  # MMR is optional; never fail the request for it
                trace.note(f"Could not embed the query for MMR: {exc}")
        return candidates_from_response(response, DENSE), query_vector

    def _lexical_pool(
        self, kb_id: str, query: str, candidate_k: int, params: RetrievalParams, trace: RetrievalTrace
    ) -> list[Candidate]:
        if self._lexical is None:
            raise RetrievalError(
                "Hybrid retrieval needs a lexical index: construct with repo=<Repository>"
            )
        filters = params.filters or None
        if isinstance(self._lexical, Bm25Retriever):
            return self._lexical.candidate_pool(
                kb_id,
                query,
                candidate_k=candidate_k,
                k1=params.bm25_k1,
                b=params.bm25_b,
                filters=filters,
                trace=trace,
            )
        response = self._lexical.retrieve(kb_id, query, top_k=candidate_k, filters=filters)
        return candidates_from_response(response, LEXICAL)

    # -- pipeline -----------------------------------------------------------

    def _run(self, kb_id: str, query: str, params: RetrievalParams) -> RetrievalResponse:
        trace = RetrievalTrace()
        candidate_k = params.resolved_candidate_k()
        dense_pool, query_vector = self._dense_pool(kb_id, query, candidate_k, params, trace)
        lexical_pool = self._lexical_pool(kb_id, query, candidate_k, params, trace)
        trace.add_stage(
            "candidates",
            detail=f"dense={len(dense_pool)} lexical={len(lexical_pool)}",
            count=len(dense_pool) + len(lexical_pool),
        )

        merged = merge_candidates([dense_pool, lexical_pool])
        candidates = list(merged.values())
        if not candidates:
            trace.note("No candidates were retrieved by either retriever.")
            response = RetrievalResponse(
                query=query,
                top_k=params.top_k,
                results=[],
                embedding_model=self._embedding_label(),
                retrieval_backend=self.backend,
                strategy=self.strategy_name,
                params=params,
                reranker=RerankerReport(
                    requested=params.reranker_name(),
                    model="",
                    status=RerankerStatus.NOT_REQUESTED,
                    detail="no candidates to rerank",
                ),
            )
            return trace.apply(response)

        # --- normalization + fusion ---
        dense_weight, lexical_weight, notes = self._resolve_weights(params, dense_pool, lexical_pool)
        for note in notes:
            trace.note(note)
        with trace.measure("fusion", detail=f"{params.fusion.value} fusion") as _:
            if params.fusion.value == "rrf":
                rankings = {
                    DENSE: [c.chunk_id for c in dense_pool],
                    LEXICAL: [c.chunk_id for c in lexical_pool],
                }
                fused = reciprocal_rank_fusion(
                    rankings,
                    rrf_k=params.rrf_k,
                    weights={DENSE: dense_weight, LEXICAL: lexical_weight},
                )
                for cand in candidates:
                    entry = fused.get(cand.chunk_id)
                    if entry is None:
                        continue
                    cand.fused_score = entry.get("fused")
                    cand.contributions[DENSE] = float(entry.get(f"{DENSE}_contribution", 0.0))
                    cand.contributions[LEXICAL] = float(entry.get(f"{LEXICAL}_contribution", 0.0))
                trace.note(
                    f"RRF fusion with k={params.rrf_k} and weights "
                    f"dense={dense_weight:.3f}, bm25={lexical_weight:.3f}."
                )
            else:
                dense_norm = normalize_scores(
                    {c.chunk_id: c.raw_scores.get(DENSE, 0.0) for c in dense_pool},
                    params.normalization,
                )
                lexical_norm = normalize_scores(
                    {c.chunk_id: c.raw_scores.get(LEXICAL, 0.0) for c in lexical_pool},
                    params.normalization,
                )
                fused_map = weighted_fuse(dense_norm, lexical_norm, dense_weight, lexical_weight)
                for cand in candidates:
                    if DENSE in cand.raw_scores:
                        cand.normalized[DENSE] = dense_norm.get(cand.chunk_id, 0.0)
                    if LEXICAL in cand.raw_scores:
                        cand.normalized[LEXICAL] = lexical_norm.get(cand.chunk_id, 0.0)
                    entry = fused_map.get(cand.chunk_id)
                    if entry is None:
                        continue
                    cand.fused_score = entry["fused"]
                    cand.contributions[DENSE] = entry["dense_contribution"]
                    cand.contributions[LEXICAL] = entry["lexical_contribution"]
                trace.note(
                    f"{params.normalization.value} normalization on each candidate pool, then "
                    f"weighted fusion (dense={dense_weight:.3f}, bm25={lexical_weight:.3f})."
                )
        candidates = [c for c in candidates if c.fused_score is not None]
        candidates.sort(key=lambda c: (-(c.fused_score or 0.0), c.chunk_id))
        for cand in candidates:
            cand.reported_score = cand.fused_score
            if "fusion" not in cand.stages:
                cand.stages.append("fusion")

        # --- diversity ---
        if params.diversity != "none" or params.max_per_document is not None:
            vector_lookup: dict[str, list[float]] = {}
            if params.diversity == "mmr":
                vector_lookup = self._candidate_vectors(kb_id, candidates, trace)
            candidates = apply_diversity(
                candidates,
                params=params,
                trace=trace,
                vector_lookup=vector_lookup,
                query_vector=query_vector,
            )
        else:
            trace.skip("diversity", "disabled")

        # --- reranking ---
        rerank_report, rerank_order = self._maybe_rerank(query, candidates, params, trace)
        if rerank_order is not None:
            index_of = {cid: i for i, cid in enumerate(rerank_order)}
            candidates.sort(key=lambda c: (index_of.get(c.chunk_id, len(index_of)), c.chunk_id))

        candidates = apply_min_score_and_cut(
            candidates, min_score=params.min_score, top_k=params.top_k
        )
        response = RetrievalResponse(
            query=query,
            top_k=params.top_k,
            results=results_from_candidates(
                candidates, method=self.strategy_name, min_score=0.0, top_k=params.top_k
            ),
            embedding_model=self._embedding_label(),
            retrieval_backend=self.backend,
            strategy=self.strategy_name,
            params=params,
            reranker=rerank_report,
        )
        return trace.apply(response)

    def _resolve_weights(
        self,
        params: RetrievalParams,
        dense_pool: list[Candidate],
        lexical_pool: list[Candidate],
    ) -> tuple[float, float, list[str]]:
        """Effective fusion weights after rescaling for an empty side."""
        dense_weight, lexical_weight = params.effective_weights()
        notes: list[str] = []
        if not lexical_pool and dense_pool:
            notes.append(
                "Lexical pool was empty (no chunks matched any query term or the BM25 index "
                "is empty); weights were rescaled to dense-only (1.0/0.0) so the fused score "
                "is not capped at the dense weight."
            )
            return 1.0, 0.0, notes
        if not dense_pool and lexical_pool:
            notes.append(
                "Dense pool was empty; weights were rescaled to lexical-only (0.0/1.0)."
            )
            return 0.0, 1.0, notes
        return dense_weight, lexical_weight, notes

    def _candidate_vectors(
        self, kb_id: str, candidates: list[Candidate], trace: RetrievalTrace
    ) -> dict[str, list[float]]:
        """Fetch stored vectors for MMR. Missing vectors are reported, not invented."""
        fetch = getattr(self._store, "fetch_vectors", None)
        if fetch is None:
            trace.note("Vector store cannot return chunk vectors; MMR is unavailable.")
            return {}
        try:
            vectors = fetch(kb_id, [c.chunk_id for c in candidates])
        except NotImplementedError:
            trace.note("Vector store cannot return chunk vectors; MMR is unavailable.")
            return {}
        except Exception as exc:
            trace.note(f"Could not load candidate vectors for MMR: {exc}")
            return {}
        return {cid: v for cid, v in (vectors or {}).items() if v}

    def _maybe_rerank(
        self,
        query: str,
        candidates: list[Candidate],
        params: RetrievalParams,
        trace: RetrievalTrace,
    ) -> tuple[RerankerReport, list[str] | None]:
        """Returns (report, new_order). `new_order` is None when nothing changed."""
        requested = params.reranker_name()
        if not params.reranking_requested():
            trace.skip("rerank", "not requested")
            return (
                RerankerReport(
                    requested=RerankerChoice.NONE.value,
                    model="",
                    status=RerankerStatus.NOT_REQUESTED,
                    candidate_k=len(candidates),
                    final_k=min(params.top_k, len(candidates)),
                    detail="no reranker requested",
                ),
                None,
            )
        model = params.reranker_model
        top_n = min(params.reranker_top_n or len(candidates), len(candidates))
        items = [(c.chunk_id, c.text) for c in candidates[:top_n]]
        try:
            reranker = self._reranker_override or get_reranker(requested, model)
        except ValueError as exc:
            trace.add_stage("rerank", status=RetrievalStageStatus.UNAVAILABLE, detail=str(exc))
            trace.note(f"Reranking was requested but UNAVAILABLE: {exc}")
            return (
                RerankerReport(
                    requested=requested,
                    model=model,
                    status=RerankerStatus.UNAVAILABLE_FALLBACK,
                    candidate_k=len(candidates),
                    reranked=0,
                    final_k=min(params.top_k, len(candidates)),
                    detail=str(exc),
                ),
                None,
            )
        # The model-loading call can be slow on first use: measure it separately.
        available, reason = reranker.availability()
        if not available:
            trace.add_stage("rerank", status=RetrievalStageStatus.UNAVAILABLE, detail=reason)
            trace.note(
                f"Reranking was requested but UNAVAILABLE: {reason} "
                "Results are the UNRERANKED hybrid fusion (order unchanged)."
            )
            return (
                RerankerReport(
                    requested=requested,
                    model=reranker.model,
                    status=RerankerStatus.UNAVAILABLE_FALLBACK,
                    candidate_k=len(candidates),
                    reranked=0,
                    final_k=min(params.top_k, len(candidates)),
                    detail=reason,
                ),
                None,
            )
        try:
            with trace.measure("rerank", detail=f"{reranker.name} over {len(items)} candidate(s)"):
                result = reranker.rerank(query, items)
        except Exception as exc:  # a reranker must never break retrieval
            logger.warning("Reranker raised: %s", exc)
            trace.add_stage("rerank", status=RetrievalStageStatus.UNAVAILABLE, detail=str(exc))
            trace.note(f"Reranker raised an unexpected error and was skipped: {exc}")
            return (
                RerankerReport(
                    requested=requested,
                    model=reranker.model,
                    status=RerankerStatus.UNAVAILABLE_FALLBACK,
                    candidate_k=len(candidates),
                    reranked=0,
                    final_k=min(params.top_k, len(candidates)),
                    detail=f"reranker raised: {exc}",
                ),
                None,
            )
        if not result.applied or result.order is None:
            detail = result.detail or "reranker did not apply"
            trace.add_stage("rerank", status=RetrievalStageStatus.UNAVAILABLE, detail=detail)
            trace.note(
                f"Reranking was requested but did not apply: {detail} "
                "Results are the UNRERANKED hybrid fusion (order unchanged)."
            )
            return (
                RerankerReport(
                    requested=requested,
                    model=result.model or reranker.model,
                    status=RerankerStatus.UNAVAILABLE_FALLBACK,
                    candidate_k=len(candidates),
                    reranked=0,
                    final_k=min(params.top_k, len(candidates)),
                    detail=detail,
                ),
                None,
            )
        for cand in candidates:
            if cand.chunk_id in result.scores:
                cand.rerank_score = result.scores[cand.chunk_id]
                if "rerank" not in cand.stages:
                    cand.stages.append("rerank")
        return (
            RerankerReport(
                requested=requested,
                model=result.model,
                status=RerankerStatus.APPLIED,
                candidate_k=len(candidates),
                reranked=len(result.scores),
                final_k=min(params.top_k, len(candidates)),
                detail=result.detail,
            ),
            list(result.order),
        )

    # -- public surface -----------------------------------------------------

    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResponse:
        params = RetrievalParams(
            strategy=RetrievalStrategy(self.strategy_name),
            top_k=top_k,
            filters=filters or {},
            min_score=min_score,
            reranker=RerankerChoice(self.default_reranker),
        )
        return self._run(kb_id, query, params)

    def retrieve_with_params(
        self, kb_id: str, query: str, params: RetrievalParams
    ) -> RetrievalResponse:
        effective = params
        if self.default_reranker == RerankerChoice.CROSS_ENCODER.value and (
            params.reranker == RerankerChoice.NONE
        ):
            # hybrid_reranked means "rerank"; the user does not have to repeat it.
            effective = params.model_copy(update={"reranker": RerankerChoice.CROSS_ENCODER})
        return self._run(kb_id, query, effective)

    def corpus_names(self) -> list[str]:
        return ["qdrant-vectors", "sqlite-chunks"]


class HybridRerankedRetriever(HybridRetriever):
    """Hybrid retrieval with reranking ON by default (honest fallback if unavailable)."""

    backend = "hybrid-reranked"
    strategy_name = RetrievalStrategy.HYBRID_RERANKED.value
    default_reranker = RerankerChoice.CROSS_ENCODER.value


__all__ = ["HybridRetriever", "HybridRerankedRetriever", "order_by_score"]
