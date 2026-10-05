"""Candidate model + score normalization + fusion (V6 Phases 4 & 5).

Why this module exists: dense cosine similarity and BM25 live on different,
incomparable scales (cosine ∈ [−1, 1] but practically [0, 1]; BM25 is unbounded
and corpus-dependent). Combining raw numbers would silently let one retriever
dominate. Every fusion path therefore normalizes FIRST, and the method used is
recorded with the run.

Pure functions only: no IO, no model calls, no randomness. Ties are broken by
chunk id so results are reproducible across runs and processes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from app.schemas.retrieval import NormalizationMethod

#: Names of the two candidate sources. Used as keys in score dictionaries so the
#: breakdown shown in the UI is unambiguous.
DENSE = "dense"
LEXICAL = "lexical"


@dataclass
class Candidate:
    """One retrievable chunk as it flows through the retrieval pipeline."""

    chunk_id: str
    document_id: str = ""
    text: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    #: raw score reported by each source that retrieved this chunk
    raw_scores: dict[str, float] = field(default_factory=dict)
    #: 1-based rank within each source's own result list
    ranks: dict[str, int] = field(default_factory=dict)
    #: normalized score per source (filled by normalize_*)
    normalized: dict[str, float] = field(default_factory=dict)
    #: weighted contribution of each source to the fused score
    contributions: dict[str, float] = field(default_factory=dict)
    fused_score: float | None = None
    rerank_score: float | None = None
    #: the score this strategy reports to the user. Set explicitly by the strategy
    #: (dense cosine / normalized BM25 / fused score) so a raw unbounded score can
    #: never leak into a field that is documented as comparable.
    reported_score: float | None = None
    #: pipeline stages this candidate survived, e.g. ["dense", "fusion", "rerank"]
    stages: list[str] = field(default_factory=list)

    def provenance_size(self) -> int:
        return len(self.provenance)

    def reported(self) -> float | None:
        """Reported score, falling back through the measured sources."""
        if self.reported_score is not None:
            return self.reported_score
        if self.fused_score is not None:
            return self.fused_score
        if DENSE in self.raw_scores:
            return self.raw_scores[DENSE]
        if LEXICAL in self.raw_scores:
            return self.raw_scores[LEXICAL]
        return None


def merge_candidates(groups: Iterable[list[Candidate]]) -> dict[str, Candidate]:
    """Union candidates from several retrievers, keyed by chunk id.

    Provenance is merged field-by-field; the richer provenance wins per field so
    a lexical hit (from SQLite) and a dense hit (from the vector payload) always
    produce one result carrying every known field. Measurement wins over absence:
    a field already present is never overwritten by a missing value.
    """
    merged: dict[str, Candidate] = {}
    for group in groups:
        for cand in group:
            existing = merged.get(cand.chunk_id)
            if existing is None:
                merged[cand.chunk_id] = Candidate(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    text=cand.text,
                    provenance=dict(cand.provenance),
                    raw_scores=dict(cand.raw_scores),
                    ranks=dict(cand.ranks),
                    stages=list(cand.stages),
                )
                continue
            if not existing.document_id and cand.document_id:
                existing.document_id = cand.document_id
            if not existing.text and cand.text:
                existing.text = cand.text
            for key, value in cand.provenance.items():
                if value is not None and existing.provenance.get(key) is None:
                    existing.provenance[key] = value
            existing.raw_scores.update(cand.raw_scores)
            existing.ranks.update(cand.ranks)
            for stage in cand.stages:
                if stage not in existing.stages:
                    existing.stages.append(stage)
    return merged


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Map scores onto [0, 1] using the observed min/max of this candidate pool.

    Degenerate pools (single candidate, or all scores identical) map to 1.0:
    every candidate is equally (best-)scored within the pool, which is honest —
    nothing in the data distinguishes them. The pool is recorded with the run, so
    the normalization is always reversible in interpretation.
    """
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo <= 0:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Rank-based normalization: best candidate 1.0, worst 1/n. Scale-free."""
    if not scores:
        return {}
    order = order_by_score(scores)
    n = len(order)
    if n == 1:
        return {order[0]: 1.0}
    return {cid: (n - idx) / n for idx, cid in enumerate(order)}


def normalize_scores(
    scores: dict[str, float], method: NormalizationMethod | str = NormalizationMethod.MIN_MAX
) -> dict[str, float]:
    key = method.value if isinstance(method, NormalizationMethod) else str(method)
    if key == NormalizationMethod.RANK.value:
        return rank_normalize(scores)
    return min_max_normalize(scores)


def order_by_score(scores: dict[str, float]) -> list[str]:
    """Deterministic ordering: score desc, then chunk id asc."""
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def weighted_fuse(
    dense_scores: dict[str, float],
    lexical_scores: dict[str, float],
    dense_weight: float,
    lexical_weight: float,
) -> dict[str, dict[str, float]]:
    """Weighted fusion over NORMALIZED scores.

    Returns {chunk_id: {"fused": ..., "dense_contribution": ..., "lexical_contribution": ...}}.
    A candidate present in only one list contributes 0 for the other side —
    recorded explicitly (contribution = 0.0) rather than hidden, because "found
    by one retriever only" is materially different from "found by both".
    """
    out: dict[str, dict[str, float]] = {}
    for cid, d in dense_scores.items():
        out[cid] = {"fused": dense_weight * d, "dense_contribution": dense_weight * d,
                    "lexical_contribution": 0.0}
    for cid, lex in lexical_scores.items():
        entry = out.setdefault(cid, {"fused": 0.0, "dense_contribution": 0.0,
                                     "lexical_contribution": 0.0})
        entry["fused"] += lexical_weight * lex
        entry["lexical_contribution"] = lexical_weight * lex
    return out


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    *,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """RRF(d) = Σ_i w_i / (rrf_k + rank_i(d)), ranks 1-based.

    Weights default to 1.0 per source. Contributions are returned per source so
    the UI can show exactly which list and rank produced the fused score.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    out: dict[str, dict[str, float]] = {}
    for source, ids in rankings.items():
        w = 1.0 if not weights else float(weights.get(source, 1.0))
        for idx, cid in enumerate(ids, start=1):
            contribution = w / (rrf_k + idx)
            entry = out.setdefault(cid, {"fused": 0.0})
            entry["fused"] += contribution
            entry[f"{source}_contribution"] = contribution
            entry[f"{source}_rank"] = float(idx)
    return out


def contributions_for(candidate: Candidate, source: str) -> float | None:
    """Normalized score of `source` for a candidate, or None when absent."""
    return candidate.normalized.get(source)


def build_why(candidate: Candidate, *, method: str) -> str:
    """Deterministic, human-readable explanation of why a chunk was returned.

    Built from measured values only — no LLM, no invented justification.
    """
    parts: list[str] = []
    if DENSE in candidate.raw_scores:
        rank = candidate.ranks.get(DENSE)
        raw = candidate.raw_scores[DENSE]
        parts.append(
            f"dense rank {rank} (cosine {raw:.4f})" if rank else f"dense cosine {raw:.4f}"
        )
    if LEXICAL in candidate.raw_scores:
        rank = candidate.ranks.get(LEXICAL)
        raw = candidate.raw_scores[LEXICAL]
        parts.append(
            f"lexical rank {rank} (BM25 {raw:.4f})" if rank else f"BM25 {raw:.4f}"
        )
    if not parts:
        return f"selected by {method}"
    detail = " + ".join(parts)
    if candidate.fused_score is not None and candidate.fused_score != candidate.raw_scores.get(DENSE):
        detail += f" -> fused {candidate.fused_score:.4f}"
    if candidate.reported_score is not None and candidate.fused_score is None:
        detail += f" -> reported {candidate.reported_score:.4f}"
    if candidate.rerank_score is not None:
        detail += f" -> reranked {candidate.rerank_score:.4f}"
    return detail
