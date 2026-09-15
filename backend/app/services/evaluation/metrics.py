"""Retrieval evaluation metrics — pure, deterministic, unit-testable.

These functions compute REAL metrics from a ranked result list and a set of
relevant item IDs. They never fabricate values: if nothing relevant is known,
the metric is None (unknown), not zero.
"""
from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    """Recall@K = |relevant ∩ top-k| / |relevant|. None if no relevant known."""
    if not relevant:
        return None
    top = ranked[:k]
    return len(set(top) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    """Precision@K = |relevant ∩ top-k| / k. None if no relevant known."""
    if not relevant:
        return None
    top = ranked[:k]
    return len(set(top) & relevant) / k if k > 0 else None


def mrr(ranked: list[str], relevant: set[str]) -> float | None:
    """Mean Reciprocal Rank for a single query: 1/rank of first relevant hit."""
    if not relevant:
        return None
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    """Binary-relevance NDCG@K. None if no relevant known."""
    if not relevant:
        return None
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(ranked[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_metrics(values: list[float | None]) -> float | None:
    """Mean over non-None values; None if every value is None (unknown)."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)
