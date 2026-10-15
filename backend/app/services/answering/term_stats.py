"""V20 — corpus term statistics for the evidence gate.

Why this module exists
----------------------
Before V20 the evidence gate measured `lexical_alignment` as the *unweighted*
fraction of question content terms found anywhere in the retrieved text. That
metric treats a corpus-ubiquitous word as equivalent to a discriminating one, so
an off-domain question whose generic words happen to occur in a motor-industry
corpus cleared the abstention floor and was answered:

    "How many moons does Jupiter have?"
    terms = {jupiter, many, moons};  only "many" occurs in the corpus
    alignment = 1/3 = 0.333  >=  MIN_ALIGNMENT_TO_PARTIAL (0.15)
    -> ANSWER, status=grounded, 1 citation, on an Automobile knowledge base

The gate could not tell "the corpus discusses this" from "one common English word
from the question appears somewhere in the text". Fixing that needs one piece of
information the gate never had: **how common each question term is in the corpus
being searched.**

That information already exists in this project. `lexical_indexes` stores the
BM25 statistics (df, doc lengths, avgdl, n_docs) built from the chunk table, and
`app/services/retrieval/lexical.py` defines the Lucene-style IDF the retriever
already uses. This module exposes those stored statistics to the gate through a
narrow, read-only interface — no new dependency, no model, no second corpus scan.

Honesty rules
-------------
* **Read-only.** This never builds, persists or repairs an index. Retrieval owns
  that. If no usable index was persisted, or the persisted one is stale relative
  to the corpus revision, statistics are reported ``available=False`` with the
  reason, and the gate records the check as NOT PERFORMED rather than guessing.
* **No inferred statistics.** A term the corpus never contains has ``df == 0``,
  which is reported as *absent*, not as a made-up IDF. The gate treats "absent
  from the corpus" as its own named case for exactly that reason.
* **Failures are surfaced, not swallowed.** A store that raises, or a corrupt
  payload, yields ``available=False`` plus the reason, so a missing signal is
  always visible in the gate's recorded signals.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from app.services.retrieval.lexical import LexicalIndex, TOKENIZER_VERSION, idf

logger = logging.getLogger(__name__)

#: Above this many chunks, term statistics are NOT built on the answering path.
#: The statistics come from one read of the chunk table, which is cheap at this
#: project's scale — the real Automobile corpus (812 chunks, 548 K characters)
#: measures ~82 ms including the SQLite read — but it is still a corpus scan, so
#: it is bounded and refused with a stated reason rather than paid on a request.
MAX_TERM_STATS_CHUNKS = 50_000


@dataclass(frozen=True)
class CorpusTermStats:
    """Document-frequency statistics for one corpus.

    ``available=False`` means the gate must not use IDF-weighted reasoning at
    all; ``reason`` says why, and is copied into the gate's recorded signals.
    """

    available: bool
    source: str = "none"  # persisted | unavailable
    n_docs: int = 0
    df: Mapping[str, int] = field(default_factory=dict)
    tokenizer_version: str = ""
    revision: int | None = None
    reason: str = ""

    # -- lookups -----------------------------------------------------------

    def document_frequency(self, term: str) -> int | None:
        """Number of chunks containing `term`, or None when stats are unusable.

        A term absent from the corpus returns 0 — an exact measurement, not a
        default. Callers must not treat None and 0 alike.
        """
        if not self.available:
            return None
        return int(self.df.get(term, 0))

    def is_absent_from_corpus(self, term: str) -> bool:
        """True only when the corpus provably does not contain `term`."""
        if not self.available:
            return False
        return self.df.get(term, 0) == 0

    def inverse_document_frequency(self, term: str) -> float | None:
        """BM25 (Lucene) IDF for `term`, or None when stats are unusable.

        Uses the SAME function the BM25 retriever scores with, so a gate
        threshold and a retrieval score cannot disagree about term rarity.
        Terms absent from the corpus have no measurable df, so their IDF is not
        fabricated here: callers get None and must handle absence explicitly.
        """
        if not self.available:
            return None
        df = self.df.get(term)
        if df is None:
            return None
        return idf(self.n_docs, int(df))


@runtime_checkable
class TermStatsProvider(Protocol):
    """Supplies corpus statistics for a knowledge base."""

    name: str

    def stats(self, kb_id: str) -> CorpusTermStats: ...


class NullTermStatsProvider:
    """Always-unavailable provider: the gate's informativeness rule is skipped.

    Used where no corpus statistics exist (unit tests that do not model a
    lexical index, stores without one). The gate records the skip explicitly, so
    behaviour without statistics is visible rather than silently different.
    """

    name = "none"

    def __init__(self, reason: str = "no term-statistics provider is configured") -> None:
        self._reason = reason

    def stats(self, kb_id: str) -> CorpusTermStats:  # noqa: ARG002 - protocol shape
        return CorpusTermStats(available=False, source="unavailable", reason=self._reason)


class LexicalIndexStatsProvider:
    """Corpus term statistics from the persisted BM25 index, or built read-only.

    Two sources, in this order:

    1. the PERSISTED index (`lexical_indexes`), when it is present, matches the
       current corpus revision and was built with the current tokenizer. This is
       the O(1) path for any corpus BM25 has already been run on.
    2. a read-only IN-MEMORY build from the chunk table, when NO index is
       persisted and the store can list chunk text. This matters because a fresh
       knowledge base that has only ever used dense retrieval has no persisted
       lexical index, and without this fallback the gate's scale-free rule would
       silently be unavailable on exactly the corpora a new user builds.

    **It never writes.** The in-memory build is cached per process (keyed by the
    corpus revision) and is deliberately NOT persisted — persisting is retrieval's
    job (`LexicalIndexStore`), not the answering path's.
    """

    name = "lexical-index"

    def __init__(self, store: Any, max_chunks: int = MAX_TERM_STATS_CHUNKS) -> None:
        self._store = store
        self._max_chunks = int(max_chunks)
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[int | None, CorpusTermStats]] = {}

    # -- public API ---------------------------------------------------------

    def stats(self, kb_id: str) -> CorpusTermStats:
        persisted = self._from_persisted(kb_id)
        if persisted is not None:
            # A persisted index EXISTS. Use it when usable; when it is stale or
            # unusable, say so instead of quietly substituting different numbers.
            return persisted
        return self._from_chunk_table(kb_id)

    # -- source 1: the persisted index -------------------------------------

    def _from_persisted(self, kb_id: str) -> CorpusTermStats | None:
        """Persisted statistics, or None when nothing has been persisted."""
        getter = getattr(self._store, "get_lexical_index", None)
        if getter is None:
            return None
        try:
            row = getter(kb_id)
        except Exception as exc:  # noqa: BLE001 - a missing signal is reported, not raised
            logger.warning("Could not read the lexical index for %s: %s", kb_id, exc)
            return CorpusTermStats(
                available=False,
                source="unavailable",
                reason=f"reading the persisted lexical index failed ({type(exc).__name__})",
            )
        if row is None:
            return None
        stored_revision, payload_json = row
        current, measurable = self._revision_of(kb_id)
        if measurable and current != stored_revision:
            return CorpusTermStats(
                available=False,
                source="unavailable",
                revision=stored_revision,
                reason=(
                    f"the persisted lexical index describes corpus revision "
                    f"{stored_revision} but the corpus is at {current}; term "
                    f"frequencies would describe the wrong corpus"
                ),
            )
        try:
            payload = json.loads(payload_json)
            index = LexicalIndex.from_payload(payload)
        except Exception as exc:  # noqa: BLE001 - a corrupt index is a reported gap
            logger.warning("Persisted lexical index for %s is unusable: %s", kb_id, exc)
            return CorpusTermStats(
                available=False,
                source="unavailable",
                revision=stored_revision,
                reason=f"the persisted lexical index could not be parsed ({type(exc).__name__})",
            )
        if not index.matches_tokenizer():
            return CorpusTermStats(
                available=False,
                source="unavailable",
                revision=stored_revision,
                tokenizer_version=index.tokenizer_version,
                reason=(
                    f"the persisted lexical index was built with tokenizer "
                    f"{index.tokenizer_version!r} but this build uses {TOKENIZER_VERSION!r}"
                ),
            )
        if index.n_docs == 0:
            return CorpusTermStats(
                available=False,
                source="unavailable",
                revision=stored_revision,
                reason="the corpus has no indexed chunks, so term frequencies are empty",
            )
        return CorpusTermStats(
            available=True,
            source="persisted",
            n_docs=index.n_docs,
            df=index.df,
            tokenizer_version=index.tokenizer_version,
            revision=stored_revision,
        )

    # -- source 2: a read-only build from the chunk table ------------------

    def _from_chunk_table(self, kb_id: str) -> CorpusTermStats:
        lister = getattr(self._store, "list_chunk_lexical_rows", None)
        if lister is None:
            return CorpusTermStats(
                available=False,
                source="unavailable",
                reason=(
                    "no lexical index has been persisted for this corpus and this "
                    "store cannot list chunk text, so term frequencies are unknown"
                ),
            )
        revision, _measurable = self._revision_of(kb_id)
        with self._lock:
            cached = self._cache.get(kb_id)
        if cached is not None and cached[0] == revision:
            return cached[1]
        try:
            rows = lister(kb_id)
        except Exception as exc:  # noqa: BLE001 - report the gap, do not crash answering
            logger.warning("Could not list chunk text for %s: %s", kb_id, exc)
            return CorpusTermStats(
                available=False,
                source="unavailable",
                reason=f"reading chunk text for term statistics failed ({type(exc).__name__})",
            )
        if not rows:
            return CorpusTermStats(
                available=False,
                source="unavailable",
                reason="the corpus has no chunks, so term frequencies are empty",
            )
        if len(rows) > self._max_chunks:
            return CorpusTermStats(
                available=False,
                source="unavailable",
                reason=(
                    f"the corpus has {len(rows)} chunks, above the {self._max_chunks}-chunk "
                    "bound on building term statistics during an answer; run a lexical "
                    "retrieval first to persist an index"
                ),
            )
        index = LexicalIndex.build(
            (str(r["chunk_id"]), str(r.get("text") or "")) for r in rows
        )
        stats = CorpusTermStats(
            available=index.n_docs > 0,
            source="built-read-only",
            n_docs=index.n_docs,
            df=index.df if index.n_docs else {},
            tokenizer_version=index.tokenizer_version,
            revision=revision,
            reason="" if index.n_docs else "the corpus has no indexed chunks",
        )
        with self._lock:
            self._cache[kb_id] = (revision, stats)
        return stats

    def _revision_of(self, kb_id: str) -> tuple[int, bool]:
        fn = getattr(self._store, "corpus_revision", None)
        if fn is None:
            return (0, False)
        try:
            return (int(fn(kb_id)), True)
        except Exception as exc:  # noqa: BLE001 - unknown staleness is reported
            logger.warning("Could not read the corpus revision for %s: %s", kb_id, exc)
            return (0, False)


def create_term_stats_provider(store: Any | None) -> TermStatsProvider:
    """Factory: a lexical-index provider when a store exists, else an explicit none."""
    if store is None:
        return NullTermStatsProvider()
    return LexicalIndexStatsProvider(store)


__all__ = [
    "CorpusTermStats",
    "LexicalIndexStatsProvider",
    "MAX_TERM_STATS_CHUNKS",
    "NullTermStatsProvider",
    "TermStatsProvider",
    "create_term_stats_provider",
]
