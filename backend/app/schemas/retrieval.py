"""V6 retrieval schemas: strategy selection, fusion/rerank parameters, run record.

Design rules (AGENTS.md + V6 spec):
* Every knob that changes retrieval behaviour is an explicit, validated field.
  Nothing about retrieval is hard-coded in business logic.
* Parameters are PERSISTED with the KB (see `RetrievalConfigRecord`) and RECORDED
  with every run (`RetrievalRun`), so a result can always be explained by the
  configuration that produced it.
* Scores are reported as measured. A missing threshold/score is `None`, never a
  fabricated number.

This module deliberately imports NOTHING from `app.schemas.models` so that
`models.py` can import it (response models embed these types) without a cycle.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RetrievalStrategy(str, Enum):
    """Selectable retrieval strategies. Names are the registry keys."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANKED = "hybrid_reranked"


class FusionMethod(str, Enum):
    """How dense and lexical candidates are combined."""

    WEIGHTED = "weighted"
    RRF = "rrf"


class NormalizationMethod(str, Enum):
    """How raw scores are mapped onto a common scale before fusion.

    Dense cosine similarities and BM25 scores are NOT comparable (BM25 is
    unbounded). Fusion therefore always normalizes first; the chosen method is
    recorded with every run.
    """

    MIN_MAX = "min_max"
    RANK = "rank"


class RerankerChoice(str, Enum):
    NONE = "none"
    CROSS_ENCODER = "cross-encoder"


class RerankerStatus(str, Enum):
    """Reported honestly: reranking is never silently claimed."""

    NOT_REQUESTED = "not_requested"
    APPLIED = "applied"
    UNAVAILABLE_FALLBACK = "unavailable_fallback"


class RetrievalStageStatus(str, Enum):
    """Status of one retrieval stage.

    Named distinctly from `schemas.models.StageStatus` (build-run stages) so the
    two can never be confused when imported into the same module.
    """

    OK = "ok"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


#: BM25 defaults (Lucene/Robertson standard values).
DEFAULT_BM25_K1 = 1.2
DEFAULT_BM25_B = 0.75
#: Reciprocal Rank Fusion constant from the RRF paper.
DEFAULT_RRF_K = 60
#: Default fusion weights (V6 spec). Configurable, never hard-coded downstream.
DEFAULT_DENSE_WEIGHT = 0.65
DEFAULT_BM25_WEIGHT = 0.35
#: Candidate pool multiplier when the caller does not state candidate_k.
CANDIDATE_POOL_FACTOR = 4
CANDIDATE_POOL_MIN = 20
#: Default reranker model (only loaded when explicitly requested).
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RetrievalParams(BaseModel):
    """Complete, validated description of one retrieval configuration."""

    strategy: RetrievalStrategy = RetrievalStrategy.DENSE

    # --- result sizing ---
    top_k: int = Field(default=5, ge=1, le=100, description="final number of results")
    candidate_k: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="per-strategy candidate pool before fusion/rerank; "
        "None = max(top_k * 4, 20)",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="relevance floor applied to the reported (normalized) score",
    )

    # --- fusion ---
    fusion: FusionMethod = FusionMethod.WEIGHTED
    dense_weight: float = Field(default=DEFAULT_DENSE_WEIGHT, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=DEFAULT_BM25_WEIGHT, ge=0.0, le=1.0)
    rrf_k: int = Field(default=DEFAULT_RRF_K, ge=1, le=1000)
    normalization: NormalizationMethod = NormalizationMethod.MIN_MAX

    # --- lexical ---
    bm25_k1: float = Field(default=DEFAULT_BM25_K1, ge=0.0, le=10.0)
    bm25_b: float = Field(default=DEFAULT_BM25_B, ge=0.0, le=1.0)

    # --- reranking ---
    reranker: RerankerChoice = RerankerChoice.NONE
    reranker_model: str = Field(default="", max_length=200)
    reranker_top_n: int | None = Field(
        default=None, ge=1, le=1000, description="rerank only the first N candidates"
    )

    # --- diversity ---
    diversity: str = Field(
        default="none",
        description="'none' | 'mmr' | 'document_cap' (transparent, configurable)",
    )
    diversity_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR lambda")
    max_per_document: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="document-diversification cap; None = unlimited",
    )

    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("diversity")
    @classmethod
    def _known_diversity(cls, v: str) -> str:
        allowed = {"none", "mmr", "document_cap"}
        key = (v or "none").strip().lower()
        if key not in allowed:
            raise ValueError(f"diversity must be one of {sorted(allowed)} (got {v!r})")
        return key

    @model_validator(mode="after")
    def _weights_nonzero(self) -> "RetrievalParams":
        if self.strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.HYBRID_RERANKED):
            if self.dense_weight + self.bm25_weight <= 0:
                raise ValueError("hybrid retrieval requires dense_weight + bm25_weight > 0")
        return self

    # -- helpers ------------------------------------------------------------

    def resolved_candidate_k(self) -> int:
        if self.candidate_k is not None:
            return max(self.candidate_k, self.top_k)
        return max(self.top_k * CANDIDATE_POOL_FACTOR, CANDIDATE_POOL_MIN)

    def effective_weights(self) -> tuple[float, float]:
        """(dense_weight, bm25_weight) normalized to sum to 1.0.

        Callers may pass weights that do not sum to 1 (e.g. 1.0/1.0); the ratio
        is preserved and the APPLIED values are recorded with the run, so the
        configuration is never ambiguous.
        """
        total = self.dense_weight + self.bm25_weight
        if total <= 0:
            return (1.0, 0.0)
        return (self.dense_weight / total, self.bm25_weight / total)

    def reranking_requested(self) -> bool:
        if self.reranker == RerankerChoice.CROSS_ENCODER:
            return True
        return self.strategy == RetrievalStrategy.HYBRID_RERANKED

    def reranker_name(self) -> str:
        """Effective reranker: hybrid_reranked implies cross-encoder."""
        if self.reranker == RerankerChoice.CROSS_ENCODER:
            return RerankerChoice.CROSS_ENCODER.value
        if self.strategy == RetrievalStrategy.HYBRID_RERANKED:
            return RerankerChoice.CROSS_ENCODER.value
        return RerankerChoice.NONE.value

    def config_fingerprint(self) -> str:
        """Deterministic hash of this configuration (for experiment artifacts)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class RetrievalConfigRecord(BaseModel):
    """Persisted per-KB retrieval configuration (Phase 8)."""

    kb_id: str
    params: RetrievalParams = Field(default_factory=RetrievalParams)
    note: str = ""
    updated_at: datetime = Field(default_factory=_utcnow)


class RetrievalTimings(BaseModel):
    """Measured stage latencies in milliseconds. Unmeasured stages stay None."""

    embed_ms: float | None = None
    dense_ms: float | None = None
    lexical_ms: float | None = None
    fusion_ms: float | None = None
    rerank_ms: float | None = None
    diversity_ms: float | None = None
    total_ms: float | None = None


class RetrievalStage(BaseModel):
    """One stage of the retrieval pipeline, with its REAL status."""

    name: str
    status: RetrievalStageStatus = RetrievalStageStatus.OK
    detail: str = ""
    count: int | None = None
    ms: float | None = None


class RerankerReport(BaseModel):
    """What the reranker actually did (or honestly did not)."""

    requested: str = RerankerChoice.NONE.value
    model: str = ""
    status: RerankerStatus = RerankerStatus.NOT_REQUESTED
    candidate_k: int | None = None
    reranked: int | None = None
    final_k: int | None = None
    detail: str = ""

    @property
    def degraded(self) -> bool:
        return self.status == RerankerStatus.UNAVAILABLE_FALLBACK


class ScoreBreakdown(BaseModel):
    """Per-candidate score provenance. Only measured values are populated."""

    dense_score: float | None = None
    lexical_score: float | None = None
    normalized_dense: float | None = None
    normalized_lexical: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    dense_contribution: float | None = None
    lexical_contribution: float | None = None


class RetrievalRun(BaseModel):
    """Immutable observability record for one retrieval execution (Phase 18)."""

    id: str
    kb_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    query: str
    strategy: str
    config_fingerprint: str = ""
    params: RetrievalParams = Field(default_factory=RetrievalParams)
    applied_weights: dict[str, float] = Field(default_factory=dict)
    normalization: str = ""
    rrf_k: int | None = None
    embedding_model: str = ""
    vector_backend: str = ""
    reranker: str = RerankerChoice.NONE.value
    reranker_model: str = ""
    reranker_status: RerankerStatus = RerankerStatus.NOT_REQUESTED
    corpus_version: str | None = None
    corpus_fingerprint: str | None = None
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    result_count: int = 0
    timings: RetrievalTimings = Field(default_factory=RetrievalTimings)
    stages: list[RetrievalStage] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RetrievalRunSummary(BaseModel):
    """Compact listing of past runs (list endpoint; full record by id)."""

    id: str
    kb_id: str
    created_at: datetime
    query: str
    strategy: str
    result_count: int
    total_ms: float | None = None
    reranker_status: RerankerStatus = RerankerStatus.NOT_REQUESTED
