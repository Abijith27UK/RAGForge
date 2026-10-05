"""Phase 10 — evidence assembly.

Converts a `RetrievalResponse` (V6) into the `EvidenceSet` the rest of the
answering pipeline consumes, WITHOUT losing provenance:

* every field of the chunk's provenance dict is copied verbatim;
* the original retrieval rank is preserved alongside the final evidence rank;
* deduplication records WHY an item was dropped, and dropped items are kept in
  `EvidenceSet.dropped` for audit rather than silently vanishing.

Dedup rules (deterministic, no LLM):
1. same chunk_id            -> duplicate (hybrid/rerank can surface it twice)
2. same content_hash        -> duplicate content
3. same document + text containment >= threshold -> near-duplicate;
   the HIGHER-scoring item is kept, the other is dropped with the measured
   overlap recorded.

Rule 3 only fires on near-identical text (containment >= 0.9 by default):
partially overlapping chunks from the same document are legitimate context and
are NOT removed. Different documents with similar text are never deduped —
independent sources agreeing is a signal the gate wants to see.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.schemas.answer import Evidence, EvidenceSet, QueryPlan
from app.schemas.models import RetrievalResponse, RetrievalResult

logger = logging.getLogger(__name__)

#: Minimum containment (|intersection| / min(|a|, |b|)) to call two chunks
#: near-duplicates of the same document.
DEFAULT_CONTAINMENT_THRESHOLD = 0.9

_TOKEN_SPLIT = " "


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t}


def containment(a: str, b: str) -> float:
    """Fraction of the smaller token set contained in the larger.

    Deterministic and cheap; deliberately not fuzzy (no embeddings here — this
    is the deterministic half of the pipeline).
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    smaller, larger = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return len(smaller & larger) / len(smaller)


def _field(result: RetrievalResult, key: str) -> object | None:
    """Provenance value lookup: provenance dict first, never inventing values."""
    value = result.provenance.get(key)
    return value if value is not None else None


def result_to_evidence(
    result: RetrievalResult,
    *,
    index: int,
    kb_id: str,
    strategy: str,
    retrieval_run_id: str | None,
) -> Evidence:
    """Build one Evidence from one retrieval result, preserving all provenance."""
    prov = dict(result.provenance)
    # Carry the V6 score provenance along so downstream signals (e.g. hybrid
    # retrieval agreement) can see where the score came from. It is copied,
    # never re-derived.
    breakdown = result.score_breakdown.model_dump(exclude_none=True) if result.score_breakdown else {}
    if breakdown:
        prov.setdefault("score_breakdown", breakdown)
    title = str(prov.get("document_title") or prov.get("source_title") or "")
    original_rank = result.rank if result.rank is not None else index
    return Evidence(
        evidence_id=f"ev_{index:04d}",
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        source_id=(str(prov["source_id"]) if prov.get("source_id") is not None else None),
        kb_id=(str(prov["kb_id"]) if prov.get("kb_id") is not None else kb_id),
        title=title,
        source_type=(
            str(prov["source_type"]) if prov.get("source_type") is not None else None
        ),
        content=result.text or "",
        retrieval_score=float(result.score),
        retrieval_strategy=result.retrieval_method or strategy,
        rank=index,
        original_rank=original_rank,
        page=int(prov["page"]) if prov.get("page") is not None else None,
        slide=int(prov["slide"]) if prov.get("slide") is not None else None,
        section=str(prov["section"]) if prov.get("section") is not None else None,
        section_path=str(prov["section_path"]) if prov.get("section_path") is not None else None,
        url=str(prov["source_url"]) if prov.get("source_url") is not None else None,
        publisher=str(prov["publisher"]) if prov.get("publisher") is not None else None,
        document_version=(
            int(prov["document_version"]) if prov.get("document_version") is not None else None
        ),
        content_hash=str(prov.get("content_hash") or ""),
        trust_score=(
            float(prov["trust_score"]) if prov.get("trust_score") is not None else None
        ),
        provenance=prov,
        retrieval_run_id=retrieval_run_id,
    )


def deduplicate(
    candidates: list[Evidence],
    *,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> tuple[list[Evidence], list[Evidence], list[str]]:
    """Return (kept, dropped, notes). Keeps the stronger item of each duplicate
    pair; the loser records the reason and the measured overlap."""
    kept: list[Evidence] = []
    dropped: list[Evidence] = []
    notes: list[str] = []

    for cand in candidates:
        reason: str | None = None
        overlap: float | None = None
        for k in kept:
            if cand.chunk_id == k.chunk_id:
                reason = f"duplicate chunk_id {cand.chunk_id} (also retrieved as {k.evidence_id})"
                break
            if cand.content_hash and cand.content_hash == k.content_hash:
                reason = (
                    f"duplicate content_hash {cand.content_hash[:12]}… "
                    f"(kept {k.evidence_id})"
                )
                break
            if cand.document_id == k.document_id:
                c = containment(cand.content, k.content)
                if c >= containment_threshold:
                    reason = (
                        f"near-duplicate of {k.evidence_id} in the same document "
                        f"(containment {c:.2f} >= {containment_threshold})"
                    )
                    overlap = round(c, 4)
                    break
        if reason is None:
            kept.append(cand)
            continue
        # Decide which of the pair survives: higher retrieval score wins; ties
        # keep the item that ranked better in retrieval.
        existing_idx = next(
            i for i, k in enumerate(kept)
            if k.chunk_id == cand.chunk_id
            or (cand.content_hash and k.content_hash == cand.content_hash)
            or (
                k.document_id == cand.document_id
                and containment(cand.content, k.content) >= containment_threshold
            )
        )
        existing = kept[existing_idx]
        if cand.retrieval_score > existing.retrieval_score:
            cand.dedup_reason = reason
            cand.overlap_fraction = overlap
            existing.dedup_reason = (
                f"superseded by {cand.evidence_id} which scored higher; "
                f"both refer to the same content"
            )
            existing.overlap_fraction = overlap
            kept[existing_idx] = cand
            dropped.append(existing)
            notes.append(
                f"evidence dedup: {existing.evidence_id} replaced by {cand.evidence_id} "
                f"(higher retrieval score for identical content)"
            )
        else:
            cand.dedup_reason = reason
            cand.overlap_fraction = overlap
            dropped.append(cand)
            notes.append(f"evidence dedup: dropped {cand.evidence_id} — {reason}")
    # Re-rank survivors by retrieval score so `rank` reflects the final order,
    # while `original_rank` still shows where retrieval placed them.
    kept.sort(key=lambda e: (-e.retrieval_score, e.original_rank))
    for i, item in enumerate(kept, start=1):
        item.rank = i
    return kept, dropped, notes


class EvidenceSelector(ABC):
    """Chooses and normalizes the evidence an answer may be grounded in.

    This is the ONLY component the answering pipeline lets near retrieval
    results, which is what keeps the generator from ever seeing "the whole
    database": whatever `select` returns is the complete universe of citable
    text.

    The interface is deliberately narrow — one retrieval response plus the
    query plan in, one `EvidenceSet` out — so alternative selection policies
    (score floors, per-document caps, MMR over evidence, learned rerankers) can
    be swapped in without touching the gate, the generator or the validator.
    """

    name: str = "base"
    version: str = "v7.2"

    @abstractmethod
    def select(
        self,
        response: RetrievalResponse,
        *,
        kb_id: str,
        plan: QueryPlan | None = None,
    ) -> EvidenceSet: ...


class ProvenancePreservingSelector(EvidenceSelector):
    """Default selector: copy every provenance field, then deduplicate.

    Behaviour is IDENTICAL to the module-level `build_evidence_set` it wraps, so
    introducing this class cannot change an existing answer. It exists so the
    pipeline depends on an interface rather than on a module function.

    What it deliberately does NOT do (stated, not silently skipped):
    * no score floor — a low-scoring chunk is kept and the GATE judges whether
      the evidence is sufficient, so the decision is explained in one place;
    * no per-document cap — document diversity is a gate signal, not a selector
      policy, in v7.2.
    """

    name = "provenance-preserving"
    version = "v7.2"

    def __init__(self, containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD) -> None:
        self.containment_threshold = containment_threshold

    def select(
        self,
        response: RetrievalResponse,
        *,
        kb_id: str,
        plan: QueryPlan | None = None,
    ) -> EvidenceSet:
        evidence = build_evidence_set(
            response, kb_id=kb_id, containment_threshold=self.containment_threshold
        )
        if plan is not None:
            evidence.notes.append(
                f"evidence selector '{self.name}' {self.version}: kept "
                f"{len(evidence.items)}/{len(evidence.items) + len(evidence.dropped)} "
                f"candidate(s) after dedup; no score floor or document cap was applied"
            )
        return evidence


def create_evidence_selector(
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> EvidenceSelector:
    """Factory. One implementation today; the ABC is the extension point."""
    return ProvenancePreservingSelector(containment_threshold=containment_threshold)


def build_evidence_set(
    response: RetrievalResponse,
    *,
    kb_id: str,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> EvidenceSet:
    """Assemble the auditable EvidenceSet for one retrieval response."""
    candidates = [
        result_to_evidence(
            result,
            index=i,
            kb_id=kb_id,
            strategy=response.strategy,
            retrieval_run_id=response.retrieval_run_id,
        )
        for i, result in enumerate(response.results, start=1)
    ]
    kept, dropped, notes = deduplicate(candidates, containment_threshold=containment_threshold)
    notes = [
        f"assembled {len(candidates)} candidate(s) from strategy "
        f"'{response.strategy}' (retrieval_run_id={response.retrieval_run_id or 'none'})",
        *notes,
    ]
    if not candidates:
        notes.append("retrieval returned no results; evidence set is empty")
    return EvidenceSet(
        items=kept,
        dropped=dropped,
        notes=notes,
        retrieval_run_id=response.retrieval_run_id,
        strategy=response.strategy,
    )


__all__ = [
    "DEFAULT_CONTAINMENT_THRESHOLD",
    "EvidenceSelector",
    "ProvenancePreservingSelector",
    "build_evidence_set",
    "containment",
    "create_evidence_selector",
    "deduplicate",
    "result_to_evidence",
]
