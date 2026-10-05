"""Answer trace builder (V7 Phase: AnswerTrace).

Mirrors `app.services.retrieval.trace.RetrievalTrace`: a cheap, mutable
per-request recorder that stages the audit trail for one answer. Rules:

* a stage that did not run is recorded as 'skipped', never as 'ok';
* an unmeasured duration stays None;
* the raw generator output is preserved (truncated to a sane cap) so a
  repair/downgrade/abstain decision can be audited against what the model
  actually produced;
* no secrets: provider name/model only, never keys or env values.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from app.schemas.answer import (
    Answer,
    AnswerTrace,
    AnswerTraceStage,
    EvidenceSet,
    GeneratedAnswer,
)

#: Raw generator text kept in the trace (chars). Enough to audit, not enough
#: to bloat SQLite rows unboundedly.
RAW_TEXT_CAP = 8000

#: Canonical stage order, so the UI always renders the pipeline left-to-right
#: even when a stage was skipped.
STAGE_ORDER = [
    "query_processing",
    "retrieval",
    "evidence_assembly",
    "evidence_gate",
    "generation",
    "citation_validation",
    "finalization",
]


class AnswerTraceRecorder:
    """Mutable per-request trace. No IO until the caller persists it."""

    def __init__(self, *, trace_id: str, kb_id: str, question: str) -> None:
        self.trace = AnswerTrace(id=trace_id, kb_id=kb_id, answer_id="", question=question)
        self._started = time.perf_counter()
        self._recorded: set[str] = set()

    # -- recording -----------------------------------------------------------

    def stage(
        self,
        name: str,
        *,
        status: str = "ok",
        detail: str = "",
        count: int | None = None,
        ms: float | None = None,
    ) -> None:
        self.trace.stages.append(
            AnswerTraceStage(name=name, status=status, detail=detail, count=count, ms=ms)
        )
        self._recorded.add(name)

    @contextmanager
    def measure(self, name: str, *, detail: str = "") -> Iterator[None]:
        """Time a block and record one stage; records 'error' on exception."""
        started = time.perf_counter()
        try:
            yield
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.stage(name, status="error", detail=detail or "stage raised", ms=elapsed)
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        self.stage(name, status="ok", detail=detail, ms=elapsed)

    def skip(self, name: str, detail: str = "") -> None:
        self.stage(name, status="skipped", detail=detail)

    def note(self, message: str) -> None:
        if message and message not in self.trace.notes:
            self.trace.notes.append(message)

    # -- attach data ---------------------------------------------------------

    def attach_retrieval(self, response: Any) -> None:
        """Copy strategy/config/stage facts from a RetrievalResponse."""
        self.trace.retrieval_strategy = getattr(response, "strategy", "") or ""
        self.trace.retrieval_run_id = getattr(response, "retrieval_run_id", None)
        params = getattr(response, "params", None)
        if params is not None:
            self.trace.retrieval_params = params.model_dump(mode="json")
        stages = getattr(response, "stages", []) or []
        self.trace.retrieval_stages = [s.model_dump(mode="json") for s in stages]

    def attach_evidence(self, evidence: EvidenceSet) -> None:
        self.trace.evidence_ids = [e.evidence_id for e in evidence.items]
        self.trace.evidence_items = list(evidence.items)
        self.trace.evidence_dropped = [
            {
                "evidence_id": d.evidence_id,
                "chunk_id": d.chunk_id,
                "reason": d.dedup_reason,
                "original_rank": d.original_rank,
            }
            for d in evidence.dropped
        ]
        self.trace.evidence_notes = list(evidence.notes)

    def attach_generated(self, generated: GeneratedAnswer) -> None:
        self.trace.generation_notes = list(generated.notes)
        self.trace.generation_warnings = list(generated.warnings)
        text = (generated.text or "")[:RAW_TEXT_CAP]
        if generated.abstain_requested:
            text = f"[generator requested abstention] {generated.abstention_reason}\n{text}"
        self.trace.raw_generated_text = text

    def finalize(
        self,
        answer: Answer,
        *,
        validation_actions: list[str],
        citation_problems: list[dict],
        generator: str,
        generator_model: str,
        is_mock: bool,
    ) -> AnswerTrace:
        """Fill the canonical stage list (missing stages marked 'skipped'),
        stamp the answer id and total duration, and return the trace."""
        self.trace.answer_id = answer.answer_id
        self.trace.status = answer.status
        self.trace.validation_actions = list(validation_actions)
        self.trace.citation_problems = list(citation_problems)
        self.trace.generator = generator
        self.trace.generator_model = generator_model
        self.trace.is_mock = is_mock

        recorded = {s.name for s in self.trace.stages}
        complete: list[AnswerTraceStage] = []
        for name in STAGE_ORDER:
            if name in recorded:
                complete.extend(s for s in self.trace.stages if s.name == name)
            else:
                complete.append(
                    AnswerTraceStage(name=name, status="skipped", detail="stage did not run")
                )
        # Preserve any out-of-order extras (shouldn't happen, but never drop data).
        extras = [s for s in self.trace.stages if s.name not in set(STAGE_ORDER)]
        self.trace.stages = complete + extras
        self.trace.total_ms = round((time.perf_counter() - self._started) * 1000.0, 3)
        return self.trace


__all__ = ["AnswerTraceRecorder", "RAW_TEXT_CAP", "STAGE_ORDER"]
