"""BM25 retriever over the indexed corpus (V6 Phase 3).

Where the index comes from
--------------------------
Chunk text already lives in SQLite (`chunks` table). The BM25 statistics are
derived from it rather than duplicating corpus text into a second store:

    chunks          -> durable source of truth for text (+ provenance)
    lexical_indexes -> persisted BM25 statistics (postings / df / doc lengths)
    corpus_revision -> bumped on every chunk write

Staleness detection is therefore EXACT for the SQLite repository: the persisted
index records the corpus revision it was built from, so a chunk insert/delete is
detected in O(1) and the index is rebuilt before it is used. Rebuilding is
reported on the response (`bm25_index` stage + a note), never hidden.

For an arbitrary VectorStore that cannot report a corpus revision, the index is
still built and cached **in memory per process**, and the response states that
staleness could not be checked — a capability gap is surfaced, not asserted away.

The index deliberately does NOT store chunk text or provenance. Top-k candidates
are hydrated from the store in one batched call, so the provenance returned by
BM25 is identical in shape to the dense path.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.repositories.sqlite_repo import Repository
from app.schemas.models import RetrievalResponse
from app.schemas.retrieval import (
    RetrievalParams,
    RetrievalStageStatus,
    RetrievalStrategy,
)
from app.services.retrieval.fusion import LEXICAL, Candidate, min_max_normalize
from app.services.retrieval.lexical import TOKENIZER_VERSION, LexicalIndex, tokenize
from app.services.retrieval.retriever import (
    Retriever,
    candidate_from_chunk,
    results_from_candidates,
)
from app.services.retrieval.trace import RetrievalTrace

logger = logging.getLogger(__name__)

#: How many scored ids to hydrate before filtering: filters need provenance
#: (document/source), which the index intentionally does not store.
_HYDRATE_OVERSAMPLE = 3
_HYDRATE_MIN = 50


@dataclass(frozen=True)
class LexicalIndexStatus:
    """Provenance of the index used for one query (honest, never inferred)."""

    source: str = "built"  # memory | persisted | built
    revision: int | None = None
    current_revision: int | None = None
    rebuilt: bool = False
    rebuilt_because: str = ""
    persisted: bool = False
    staleness_check: str = "revision"  # revision | unavailable
    documents: int = 0
    terms: int = 0
    tokenizer_version: str = TOKENIZER_VERSION

    def describe(self) -> str:
        bits = [
            f"{self.documents} chunk(s)",
            f"{self.terms} term(s)",
            f"source={self.source}",
        ]
        if self.staleness_check == "revision":
            bits.append(f"revision={self.revision}")
            if self.rebuilt:
                bits.append(f"rebuilt:{self.rebuilt_because}")
        else:
            bits.append("staleness check UNAVAILABLE (store cannot report a corpus revision)")
        return "; ".join(bits)


@dataclass
class _CacheEntry:
    index: LexicalIndex
    revision: int
    persisted: bool = False


class LexicalIndexStore:
    """Builds, caches and (when possible) persists BM25 indexes per corpus.

    `store` is a VectorStore-like object: it must expose
    `list_chunk_lexical_rows(kb_id)`; `corpus_revision`, `bump_corpus_revision`,
    `get_lexical_index` and `save_lexical_index` are used when present.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._lock = threading.RLock()
        self._cache: dict[str, _CacheEntry] = {}

    # -- public API ---------------------------------------------------------

    def get(self, kb_id: str) -> tuple[LexicalIndex, LexicalIndexStatus]:
        revision, measurable = self._revision_of(kb_id)
        with self._lock:
            cached = self._cache.get(kb_id)
        if cached is not None and (not measurable or cached.revision == revision):
            return cached.index, LexicalIndexStatus(
                source="memory",
                revision=cached.revision,
                current_revision=revision,
                persisted=cached.persisted,
                staleness_check="revision" if measurable else "unavailable",
                documents=cached.index.n_docs,
                terms=cached.index.term_count,
            )
        # A persisted index from an earlier process can be reused when it is not
        # stale and was built with the current tokenizer.
        loaded = self._load_persisted(kb_id, revision, measurable)
        if loaded is not None:
            index, status = loaded
            with self._lock:
                self._cache[kb_id] = _CacheEntry(index, revision, persisted=True)
            return index, status
        return self._build(kb_id, revision, measurable)

    def invalidate(self, kb_id: str) -> None:
        with self._lock:
            self._cache.pop(kb_id, None)

    def status(self, kb_id: str) -> dict[str, Any]:
        """Read-only status (never builds an index). Used by diagnostics/UI."""
        revision, measurable = self._revision_of(kb_id)
        with self._lock:
            cached = self._cache.get(kb_id)
        if cached is not None:
            return {
                "source": "memory",
                "revision": cached.revision,
                "current_revision": revision if measurable else None,
                "stale": bool(measurable and cached.revision != revision),
                "staleness_check": "revision" if measurable else "unavailable",
                "documents": cached.index.n_docs,
                "terms": cached.index.term_count,
                "tokenizer_version": cached.index.tokenizer_version,
            }
        getter = getattr(self._store, "get_lexical_index", None)
        if getter is None:
            return {
                "source": "none",
                "revision": None,
                "current_revision": revision if measurable else None,
                "stale": None,
                "staleness_check": "revision" if measurable else "unavailable",
                "documents": None,
                "terms": None,
                "tokenizer_version": None,
            }
        try:
            row = getter(kb_id)
        except Exception as exc:
            logger.warning("Could not read the persisted lexical index for %s: %s", kb_id, exc)
            row = None
        if row is None:
            return {
                "source": "none",
                "revision": None,
                "current_revision": revision if measurable else None,
                "stale": None,
                "staleness_check": "revision" if measurable else "unavailable",
                "documents": None,
                "terms": None,
                "tokenizer_version": None,
            }
        stored_revision, payload_json = row
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return {
                "source": "persisted",
                "revision": stored_revision,
                "current_revision": revision if measurable else None,
                "stale": True,
                "staleness_check": "revision" if measurable else "unavailable",
                "documents": None,
                "terms": None,
                "tokenizer_version": None,
            }
        return {
            "source": "persisted",
            "revision": stored_revision,
            "current_revision": revision if measurable else None,
            "stale": bool(measurable and stored_revision != revision),
            "staleness_check": "revision" if measurable else "unavailable",
            "documents": payload.get("n_docs"),
            "terms": len(payload.get("postings") or {}),
            "tokenizer_version": payload.get("tokenizer_version"),
        }

    # -- internals ----------------------------------------------------------

    def _revision_of(self, kb_id: str) -> tuple[int, bool]:
        """(revision, measurable). `measurable=False` means staleness is unknown."""
        fn = getattr(self._store, "corpus_revision", None)
        if fn is None:
            return (0, False)
        try:
            return (int(fn(kb_id)), True)
        except Exception as exc:
            logger.warning("Could not read the corpus revision for %s: %s", kb_id, exc)
            return (0, False)

    def _load_persisted(
        self, kb_id: str, revision: int, measurable: bool
    ) -> tuple[LexicalIndex, LexicalIndexStatus] | None:
        getter = getattr(self._store, "get_lexical_index", None)
        if getter is None:
            return None
        try:
            row = getter(kb_id)
        except Exception as exc:
            logger.warning("Could not read the persisted lexical index for %s: %s", kb_id, exc)
            return None
        if row is None:
            return None
        stored_revision, payload_json = row
        if measurable and stored_revision != revision:
            logger.info(
                "Persisted BM25 index for %s is stale (revision %s != %s); rebuilding",
                kb_id,
                stored_revision,
                revision,
            )
            return None
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            logger.warning("Persisted BM25 index for %s is not valid JSON (%s)", kb_id, exc)
            return None
        index = LexicalIndex.from_payload(payload)
        if not index.matches_tokenizer():
            logger.info(
                "Persisted BM25 index for %s used tokenizer %r; rebuilding with %r",
                kb_id,
                index.tokenizer_version,
                TOKENIZER_VERSION,
            )
            return None
        return index, LexicalIndexStatus(
            source="persisted",
            revision=stored_revision,
            current_revision=revision if measurable else None,
            persisted=True,
            staleness_check="revision" if measurable else "unavailable",
            documents=index.n_docs,
            terms=index.term_count,
            tokenizer_version=index.tokenizer_version,
        )

    def _build(
        self, kb_id: str, revision: int, measurable: bool
    ) -> tuple[LexicalIndex, LexicalIndexStatus]:
        rows = self._store.list_chunk_lexical_rows(kb_id)
        index = LexicalIndex.build((row["chunk_id"], row["text"]) for row in rows)
        persisted = False
        saver = getattr(self._store, "save_lexical_index", None)
        if saver is not None and measurable:
            try:
                saver(
                    kb_id,
                    revision,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(index.to_payload(), separators=(",", ":")),
                )
                persisted = True
            except Exception as exc:  # persistence must never break retrieval
                logger.warning("Could not persist the BM25 index for %s: %s", kb_id, exc)
        status = LexicalIndexStatus(
            source="built",
            revision=revision if measurable else None,
            current_revision=revision if measurable else None,
            rebuilt=True,
            rebuilt_because=(
                "no usable index for the current corpus revision"
                if measurable
                else "staleness cannot be checked with this store"
            ),
            persisted=persisted,
            staleness_check="revision" if measurable else "unavailable",
            documents=index.n_docs,
            terms=index.term_count,
        )
        with self._lock:
            self._cache[kb_id] = _CacheEntry(index, revision, persisted=persisted)
        return index, status


class Bm25Retriever(Retriever):
    """Lexical BM25 retrieval with real term weighting and exact staleness checks."""

    backend = "sqlite-bm25"
    strategy_name = RetrievalStrategy.BM25.value

    def __init__(
        self,
        embedding_provider: Any,
        vector_store: Any,
        identity: Any = None,
        *,
        repo: Repository | None = None,
        index_store: LexicalIndexStore | None = None,
    ) -> None:
        super().__init__(embedding_provider, vector_store, identity)
        self._repo = repo
        self._indexes = index_store or LexicalIndexStore(repo or vector_store)

    def corpus_names(self) -> list[str]:
        return ["chunks"]

    def index_status(self, kb_id: str) -> dict[str, Any]:
        """Read-only BM25 index diagnostics, including an explicit staleness flag."""
        return self._indexes.status(kb_id)

    def candidate_pool(
        self,
        kb_id: str,
        query: str,
        *,
        candidate_k: int,
        k1: float = 1.2,
        b: float = 0.75,
        filters: dict[str, Any] | None = None,
        trace: RetrievalTrace | None = None,
    ) -> list[Candidate]:
        """Score the corpus, hydrate the top ids, return the candidate pool.

        Scores here are RAW BM25 (unbounded). Normalization happens in the fusion
        layer or in `retrieve_with_params` — never inside the scorer.
        """
        trace = trace or RetrievalTrace()
        index, status = self._indexes.get(kb_id)
        if status.staleness_check == "unavailable":
            trace.note(
                "BM25 staleness could not be verified: this store does not report a "
                "corpus revision, so the lexical index may lag the corpus."
            )
        if status.rebuilt:
            trace.note(f"BM25 index rebuilt before scoring ({status.rebuilt_because}).")
        if index.is_empty:
            trace.add_stage(
                "lexical",
                status=RetrievalStageStatus.OK,
                detail=status.describe() + "; corpus has no indexed chunks",
                count=0,
            )
            trace.note("BM25 index is empty: no chunks are indexed for this knowledge base.")
            return []
        tokens = tokenize(query)
        if not tokens:
            trace.add_stage(
                "lexical",
                detail=status.describe() + "; query produced no lexical tokens",
                count=0,
            )
            return []
        hydrate_k = max(candidate_k * _HYDRATE_OVERSAMPLE, _HYDRATE_MIN)
        scored = index.search(query, k1=k1, b=b, top_k=hydrate_k)
        if not scored:
            trace.add_stage("lexical", detail=status.describe(), count=0)
            return []
        raw_by_id = dict(scored)
        chunks = self._fetch_chunks(kb_id, list(raw_by_id))
        allowed_documents = _filter_values(filters, "document_id")
        allowed_sources = _filter_values(filters, "source_id")
        candidates: list[Candidate] = []
        for chunk in chunks:
            if allowed_documents and chunk.document_id not in allowed_documents:
                continue
            if allowed_sources and chunk.source_id not in allowed_sources:
                continue
            candidate = candidate_from_chunk(chunk)
            candidate.raw_scores[LEXICAL] = float(raw_by_id.get(chunk.id, 0.0))
            candidate.stages.append(LEXICAL)
            candidates.append(candidate)
        # The fetch does not preserve BM25 order: restore it explicitly and
        # deterministically (score desc, then chunk id asc).
        candidates.sort(key=lambda c: (-c.raw_scores[LEXICAL], c.chunk_id))
        for idx, candidate in enumerate(candidates, start=1):
            candidate.ranks[LEXICAL] = idx
        candidates = candidates[:candidate_k]
        trace.add_stage(
            "lexical",
            detail=(
                f"BM25 k1={k1} b={b}; {status.describe()}; "
                f"{len(scored)} scored, {len(candidates)} kept after filters"
            ),
            count=len(candidates),
        )
        return candidates

    def _fetch_chunks(self, kb_id: str, chunk_ids: list[str]) -> list[Any]:
        if self._repo is not None:
            return self._repo.get_chunks_by_ids(kb_id, chunk_ids)
        getter = getattr(self._store, "get_chunks_by_ids", None)
        if getter is None:
            raise NotImplementedError(
                "BM25 needs to hydrate chunk text/provenance: construct with repo=<Repository> "
                "or use a store that implements get_chunks_by_ids()."
            )
        return getter(kb_id, chunk_ids)

    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResponse:
        params = RetrievalParams(
            strategy=RetrievalStrategy.BM25,
            top_k=top_k,
            filters=filters or {},
            min_score=min_score,
        )
        return self.retrieve_with_params(kb_id, query, params)

    def retrieve_with_params(
        self, kb_id: str, query: str, params: RetrievalParams
    ) -> RetrievalResponse:
        trace = RetrievalTrace()
        candidate_k = params.resolved_candidate_k()
        candidates = self.candidate_pool(
            kb_id,
            query,
            candidate_k=candidate_k,
            k1=params.bm25_k1,
            b=params.bm25_b,
            filters=params.filters or None,
            trace=trace,
        )
        # Raw BM25 is unbounded, so a [0,1] relevance floor needs a common scale.
        # The normalization is computed over this pool and reported; raw scores stay
        # visible in score_breakdown.
        raw_scores = {c.chunk_id: c.raw_scores[LEXICAL] for c in candidates}
        normalized = min_max_normalize(raw_scores)
        for cand in candidates:
            cand.normalized[LEXICAL] = normalized.get(cand.chunk_id, 0.0)
            cand.reported_score = cand.normalized[LEXICAL]
        survivors = [
            c
            for c in candidates
            if c.reported_score is not None and c.reported_score >= params.min_score
        ][: params.top_k]
        if params.min_score > 0:
            trace.note(
                "min_score was applied to the within-pool normalized BM25 score: raw BM25 "
                "is unbounded and corpus-dependent, so raw thresholds are not comparable. "
                "Raw scores remain available in score_breakdown."
            )
        response = RetrievalResponse(
            query=query,
            top_k=params.top_k,
            results=results_from_candidates(
                survivors, method="bm25", min_score=0.0, top_k=params.top_k
            ),
            embedding_model=self._embedding_label(),
            retrieval_backend=self.backend,
            strategy=self.strategy_name,
            params=params,
        )
        return trace.apply(response)


def _filter_values(filters: dict[str, Any] | None, key: str) -> set[str]:
    """Read a filter value that may be a scalar or a list."""
    if not filters:
        return set()
    value = filters.get(key)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return {str(value)}


__all__ = ["Bm25Retriever", "LexicalIndexStatus", "LexicalIndexStore"]
