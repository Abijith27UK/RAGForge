"""Retrieval trace builder (V6 Phase 18 groundwork).

Every retrieval strategy records what actually happened — which stages ran, which
were skipped or unavailable, how long each took, and any warnings. Nothing is
inferred: an unmeasured stage has ``ms=None`` and a stage that did not run is
recorded as skipped, never as ok.

The trace is attached to the response and (via the run record) persisted, so a
result can always be explained after the fact.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from app.schemas.models import RetrievalResponse
from app.schemas.retrieval import (
    RetrievalStage,
    RetrievalStageStatus,
    RetrievalTimings,
)

#: stage name -> RetrievalTimings field
TIMING_FIELDS: dict[str, str] = {
    "embed": "embed_ms",
    "dense": "dense_ms",
    "lexical": "lexical_ms",
    "fusion": "fusion_ms",
    "rerank": "rerank_ms",
    "diversity": "diversity_ms",
}


class RetrievalTrace:
    """Mutable per-request trace. Cheap: no IO unless the caller persists it."""

    def __init__(self) -> None:
        self.stages: list[RetrievalStage] = []
        self.notes: list[str] = []
        self.timings = RetrievalTimings()
        self._started = time.perf_counter()

    # -- recording ----------------------------------------------------------

    def add_stage(
        self,
        name: str,
        *,
        status: RetrievalStageStatus = RetrievalStageStatus.OK,
        detail: str = "",
        count: int | None = None,
        ms: float | None = None,
    ) -> None:
        self.stages.append(
            RetrievalStage(name=name, status=status, detail=detail, count=count, ms=ms)
        )
        if ms is not None:
            field = TIMING_FIELDS.get(name)
            if field:
                setattr(self.timings, field, round(ms, 3))

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    @contextmanager
    def measure(
        self,
        name: str,
        *,
        detail: str = "",
        status: RetrievalStageStatus = RetrievalStageStatus.OK,
    ) -> Iterator[None]:
        """Time a block and record one stage; records ERROR on exception."""
        started = time.perf_counter()
        try:
            yield
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.add_stage(
                name,
                status=RetrievalStageStatus.ERROR,
                detail=detail or "stage raised",
                ms=elapsed,
            )
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        self.add_stage(name, status=status, detail=detail, ms=elapsed)

    def skip(self, name: str, detail: str = "") -> None:
        self.add_stage(name, status=RetrievalStageStatus.SKIPPED, detail=detail)

    # -- output -------------------------------------------------------------

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000.0, 3)

    def apply(self, response: RetrievalResponse) -> RetrievalResponse:
        """Attach this trace to a response (returns a copy; never mutates input)."""
        self.timings.total_ms = self.elapsed_ms()
        return response.model_copy(
            update={
                "stages": list(self.stages),
                "timings": self.timings,
                "notes": list(dict.fromkeys(self.notes)),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.model_dump(mode="json") for s in self.stages],
            "timings": self.timings.model_dump(mode="json"),
            "notes": list(self.notes),
        }
