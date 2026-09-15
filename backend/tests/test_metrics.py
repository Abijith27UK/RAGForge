"""Evaluation metrics tests — verify the math is correct."""
from __future__ import annotations

import pytest

from app.services.evaluation.metrics import (
    average_metrics,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RANKED = ["c1", "c2", "c3", "c4", "c5"]
RELEVANT = {"c2", "c4"}


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.5  # c2 found, c4 not
    assert recall_at_k(RANKED, RELEVANT, 4) == 1.0
    assert recall_at_k(RANKED, set(), 5) is None


def test_precision_at_k():
    assert precision_at_k(RANKED, RELEVANT, 4) == 0.5  # c2, c4 of top-4
    assert precision_at_k(RANKED, RELEVANT, 1) == 0.0


def test_mrr():
    assert mrr(RANKED, RELEVANT) == 0.5  # first relevant at rank 2
    assert mrr(["c9", "c8"], RELEVANT) == 0.0
    assert mrr(RANKED, set()) is None


def test_ndcg_at_k():
    log2 = __import__("math").log2
    idcg = 1.0 + 1.0 / log2(3)  # two relevant items, ideal positions
    # k=2: only c2 (rank 2) is inside top-2
    assert ndcg_at_k(RANKED, RELEVANT, 2) == pytest.approx((1.0 / log2(3)) / idcg)
    # k=5: c2 at rank 2, c4 at rank 4
    dcg = 1.0 / log2(3) + 1.0 / log2(5)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, set(), 5) is None


def test_perfect_ranking_scores_one():
    ranked = ["c2", "c4", "c1", "c3"]
    assert ndcg_at_k(ranked, RELEVANT, 4) == pytest.approx(1.0)
    assert mrr(ranked, RELEVANT) == 1.0
    assert recall_at_k(ranked, RELEVANT, 2) == 1.0


def test_average_metrics_handles_none():
    assert average_metrics([0.5, None, 1.0]) == pytest.approx(0.75)
    assert average_metrics([None, None]) is None
    assert average_metrics([]) is None
