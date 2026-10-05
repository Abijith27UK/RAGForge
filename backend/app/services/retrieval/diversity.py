"""Transparent diversification: document cap + MMR (V6 Phase 7).

Problem being solved: the top-k can be ten near-identical chunks from one page,
which produces a technically-correct but useless evidence set (and an answer that
cites one document ten times).

Two mechanisms, both configurable and both OFF by default because relevance must
not be sacrificed silently:

* ``cap_per_document`` — keeps at most N chunks per document, filling the
  remaining slots with the next best candidates from other documents. Needs no
  vectors, always available, fully deterministic.
* ``mmr`` — Maximal Marginal Relevance (Carbonell & Goldstein, 1998):
  ``argmax λ·rel(d) − (1−λ)·max_{s∈S} sim(d, s)``. Needs candidate vectors; when
  a vector is missing the candidate is scored with similarity 0 (neutral) and the
  missing count is reported, never hidden.

Both return the kept ordering AND the reason each dropped candidate was dropped,
so the Retrieval Lab can explain the selection deterministically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DiversityOutcome:
    kept: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    method: str = "none"
    lambda_: float | None = None
    max_per_document: int | None = None
    missing_vectors: int = 0

    def summary(self) -> str:
        if self.method == "none":
            return "diversity disabled"
        if self.method == "document_cap":
            return (
                f"document cap: max {self.max_per_document} chunk(s) per document; "
                f"{len(self.dropped)} candidate(s) demoted"
            )
        detail = f"MMR (lambda={self.lambda_}); {len(self.dropped)} candidate(s) demoted"
        if self.missing_vectors:
            detail += f"; {self.missing_vectors} candidate(s) had no stored vector"
        return detail


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity; 0.0 for zero-norm vectors (undefined -> neutral)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def cap_per_document(
    order: list[str],
    document_of: dict[str, str],
    max_per_document: int | None,
) -> DiversityOutcome:
    """Enforce a per-document cap while preserving the original relevance order.

    Candidates over the cap are demoted (not discarded): they are appended after
    the kept ones so a short corpus can still fill top_k. This keeps the answer
    set complete while making the *first* k results diverse.
    """
    outcome = DiversityOutcome(method="document_cap", max_per_document=max_per_document)
    if max_per_document is None:
        outcome.kept = list(order)
        return outcome
    counts: dict[str, int] = {}
    kept: list[str] = []
    demoted: list[str] = []
    for chunk_id in order:
        doc = document_of.get(chunk_id) or ""
        used = counts.get(doc, 0)
        if doc and used >= max_per_document:
            demoted.append(chunk_id)
            outcome.dropped[chunk_id] = (
                f"document {doc} already has {used} chunk(s) selected (cap {max_per_document})"
            )
        else:
            counts[doc] = used + 1
            kept.append(chunk_id)
    outcome.kept = kept + demoted
    return outcome


def mmr_select(
    relevance: dict[str, float],
    vectors: dict[str, list[float]],
    *,
    lambda_: float,
    k: int,
    max_per_document: int | None = None,
    document_of: dict[str, str] | None = None,
) -> DiversityOutcome:
    """Maximal Marginal Relevance selection over candidates with vectors.

    ``relevance`` must already be normalized to [0, 1] (fusion output). Cosine
    similarity is mapped from [−1, 1] to [0, 1] before combining so both terms
    share a scale; the mapping is stated here rather than hidden.
    """
    outcome = DiversityOutcome(
        method="mmr", lambda_=lambda_, max_per_document=max_per_document
    )
    if k <= 0 or not relevance:
        return outcome
    # Deterministic starting order: relevance desc, then id asc.
    pool = [cid for cid, _ in sorted(relevance.items(), key=lambda kv: (-kv[1], kv[0]))]
    doc_counts: dict[str, int] = {}
    selected: list[str] = []
    remaining = list(pool)
    while remaining and len(selected) < k:
        best_id: str | None = None
        best_score = -math.inf
        for cid in remaining:
            doc = (document_of or {}).get(cid, "")
            if (
                max_per_document is not None
                and doc
                and doc_counts.get(doc, 0) >= max_per_document
            ):
                continue
            rel = relevance.get(cid, 0.0)
            redundancy = 0.0
            if selected:
                sims: list[float] = []
                for chosen in selected:
                    v1 = vectors.get(cid)
                    v2 = vectors.get(chosen)
                    if v1 is None or v2 is None:
                        sims.append(0.0)
                    else:
                        sims.append((cosine_similarity(v1, v2) + 1.0) / 2.0)
                redundancy = max(sims) if sims else 0.0
            score = lambda_ * rel - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_id = cid
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
        outcome.kept.append(best_id)
        doc = (document_of or {}).get(best_id, "")
        if doc:
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
    for cid in remaining:
        outcome.dropped[cid] = "below MMR selection cut"
    outcome.missing_vectors = sum(1 for cid in pool if cid not in vectors)
    return outcome
