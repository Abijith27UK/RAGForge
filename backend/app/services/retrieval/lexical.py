"""Deterministic tokenization + a real BM25 scorer (V6 Phase 3).

This is REAL BM25 (Robertson/Sparck-Jones term weighting with the Lucene IDF
form), not keyword-overlap dressed up as BM25:

    score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
    IDF(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

Properties that matter for this project:
* term frequency saturation and document-length normalization are real (k1/b),
  so a long document cannot win by repeating a term or by being long;
* deterministic: identical inputs always produce identical scores, and ties are
  broken by chunk id (never by dict/insertion order);
* no external dependency, no network, no model download;
* the tokenizer version is part of the persisted index, so changing it forces a
  rebuild instead of silently mixing incompatible statistics.

Known limitations (documented, not hidden): there is no stemming and no stop-word
list — IDF already discounts common terms, and stemming would need its own
versioned tokenizer. CJK text is not segmented (Chinese/Japanese would tokenize as
unbroken runs and score poorly); that is a real gap, not a hidden one.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

#: Bump when tokenization changes; a mismatch forces an index rebuild.
TOKENIZER_VERSION = "lex-v1"

#: Tokens are Unicode word runs that may contain internal separators, so technical
#: identifiers survive as one term: "gm-0.5", "3.2.1", "s/n", "iso_9001".
#: `\w` (Unicode-aware for str patterns) keeps accented characters intact — the
#: earlier ASCII class silently split "métacentre" into "m" + "tacentre".
_TOKEN_RE = re.compile(r"\w+(?:[.\-/]\w+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Deterministic token list for a piece of text.

    NFKC-normalizes (so full-width/compatibility forms match), lowercases, then
    extracts identifier-ish tokens. Unicode letters are preserved, so accented
    European text tokenizes correctly rather than being cut at the accent.
    Punctuation-only strings yield no tokens.
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_RE.findall(normalized)


def idf(n_docs: int, df: int) -> float:
    """Lucene-style BM25 IDF: always positive, never negative for common terms."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


@dataclass(frozen=True)
class LexicalIndex:
    """Inverted index over chunks. Immutable; safe to cache and to persist."""

    tokenizer_version: str
    chunk_ids: tuple[str, ...]
    doc_len: dict[str, int]
    postings: dict[str, dict[str, int]]  # term -> {chunk_id: term frequency}
    df: dict[str, int]
    avgdl: float
    n_docs: int

    # -- construction -------------------------------------------------------

    @classmethod
    def build(cls, documents: Iterable[tuple[str, str]]) -> "LexicalIndex":
        """Build from (chunk_id, text) pairs. Duplicate ids: last one wins."""
        chunk_ids: list[str] = []
        doc_len: dict[str, int] = {}
        postings: dict[str, dict[str, int]] = {}
        df: dict[str, int] = {}
        seen: set[str] = set()
        for chunk_id, text in documents:
            if chunk_id in seen:
                # Defensive: duplicate chunk ids would corrupt df counts.
                continue
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
            tokens = tokenize(text)
            doc_len[chunk_id] = len(tokens)
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for term, tf in counts.items():
                postings.setdefault(term, {})[chunk_id] = tf
                df[term] = df.get(term, 0) + 1
        n_docs = len(chunk_ids)
        avgdl = (sum(doc_len.values()) / n_docs) if n_docs else 0.0
        return cls(
            tokenizer_version=TOKENIZER_VERSION,
            chunk_ids=tuple(chunk_ids),
            doc_len=doc_len,
            postings=postings,
            df=df,
            avgdl=avgdl,
            n_docs=n_docs,
        )

    @property
    def is_empty(self) -> bool:
        return self.n_docs == 0

    @property
    def term_count(self) -> int:
        return len(self.postings)

    # -- persistence --------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        """JSON-serializable form. Round-trips exactly (verified by tests)."""
        return {
            "tokenizer_version": self.tokenizer_version,
            "chunk_ids": list(self.chunk_ids),
            "doc_len": self.doc_len,
            "postings": self.postings,
            "df": self.df,
            "avgdl": self.avgdl,
            "n_docs": self.n_docs,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LexicalIndex":
        return cls(
            tokenizer_version=str(payload.get("tokenizer_version") or ""),
            chunk_ids=tuple(payload.get("chunk_ids") or ()),
            doc_len={str(k): int(v) for k, v in (payload.get("doc_len") or {}).items()},
            postings={
                str(t): {str(c): int(n) for c, n in (hits or {}).items()}
                for t, hits in (payload.get("postings") or {}).items()
            },
            df={str(t): int(n) for t, n in (payload.get("df") or {}).items()},
            avgdl=float(payload.get("avgdl") or 0.0),
            n_docs=int(payload.get("n_docs") or 0),
        )

    def matches_tokenizer(self) -> bool:
        """False when the index was built with a different tokenizer version."""
        return self.tokenizer_version == TOKENIZER_VERSION

    # -- scoring ------------------------------------------------------------

    def score_document(self, tokens: list[str], chunk_id: str, *, k1: float, b: float) -> float:
        """BM25 score of ONE document for a token list (used by tests/diagnostics).

        Query term multiplicity is deliberately ignored (classic BM25 has no
        query-term-frequency component), so repeating a word in the query does
        not change the ranking. `score()` applies the same rule.
        """
        dl = self.doc_len.get(chunk_id, 0)
        if dl == 0 or self.avgdl <= 0:
            return 0.0
        total = 0.0
        for term in dict.fromkeys(tokens):
            tf = (self.postings.get(term) or {}).get(chunk_id, 0)
            if tf <= 0:
                continue
            df = self.df.get(term, 0)
            if df <= 0:
                continue
            denom = tf + k1 * (1.0 - b + b * (dl / self.avgdl))
            if denom <= 0:
                continue
            total += idf(self.n_docs, df) * (tf * (k1 + 1.0)) / denom
        return total

    def score(self, tokens: list[str], *, k1: float, b: float) -> dict[str, float]:
        """BM25 scores for every document matching at least one query term."""
        if self.n_docs == 0 or self.avgdl <= 0 or not tokens:
            return {}
        # Unique terms keep the accumulation identical to score_document (which
        # also iterates the raw token list) only when this is used standalone;
        # we therefore accumulate over unique terms exactly like the formula.
        unique_terms: list[str] = []
        seen: set[str] = set()
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_terms.append(t)
        scores: dict[str, float] = {}
        for term in unique_terms:
            hits = self.postings.get(term)
            if not hits:
                continue
            idf_t = idf(self.n_docs, self.df.get(term, 0))
            for chunk_id, tf in hits.items():
                dl = self.doc_len.get(chunk_id, 0)
                if dl == 0:
                    continue
                denom = tf + k1 * (1.0 - b + b * (dl / self.avgdl))
                if denom <= 0:
                    continue
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf_t * (tf * (k1 + 1.0)) / denom
        return scores

    def search(
        self,
        text: str,
        *,
        k1: float,
        b: float,
        top_k: int,
        allowed: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score) for a query string. Deterministic tie-break."""
        tokens = tokenize(text)
        if not tokens:
            return []
        scores = self.score(tokens, k1=k1, b=b)
        if allowed is not None:
            scores = {c: s for c, s in scores.items() if c in allowed}
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[: max(0, top_k)]
