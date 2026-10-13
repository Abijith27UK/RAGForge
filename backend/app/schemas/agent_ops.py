"""Agent operation records: the lifecycle of a controlled, agent-requested action.

V18 lets an MCP client ask RAGForge to change things (create a knowledge base,
ingest sources, index a corpus, add documents). Every such request is recorded
here so that:

* the caller always has a stable id to poll (``get_operation_status``);
* a **plan** is distinguishable from an **execution** — a plan row is
  ``proposed`` and authorizes nothing;
* a **confirmation** is bound to ONE action and ONE exact parameter set, is
  single-use, and expires, so a stale or edited plan cannot be executed;
* what actually happened is auditable after the fact (status, parameters,
  result summary, warnings).

Lifecycle (the same six stages the documents use):

===  ==========================  ==========================================
0    REQUEST                     the client calls a tool
1    PLAN                        an ``AgentOperation`` row, status proposed
2    USER CONFIRMATION           ``confirm_plan`` moves it to confirmed and
                                 issues a single-use ``confirmation_id``
3    EXECUTION                   ``begin_execution`` refuses unless the
                                 parameters match the confirmed fingerprint;
                                 the row goes running -> done | error
4    VALIDATION                  ``validate_knowledge_base`` reads the real
                                 index (separate, read-only)
5    REPORT                      ``get_operation_status`` returns this row
                                 joined with the underlying run record
===  ==========================  ==========================================

The row is stored in SQLite next to the rest of RAGForge's metadata, so an
operator can audit agent-driven changes with the same tools they use for the
rest of the application. It is NOT a job runner: nothing here executes work in
the background — the MCP tool performs the action synchronously and writes the
outcome back to its own row.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OperationStatus(str, enum.Enum):
    """Where an operation row is in the lifecycle.

    ``PROPOSED``      - planned, authorizes nothing
    ``CONFIRMED``     - a user confirmation was recorded for exactly these
                        parameters; it can be executed once, until it expires
    ``RUNNING``       - execution started (the confirmation is already consumed)
    ``DONE``          - execution finished; ``result_summary`` says what happened
    ``ERROR``         - execution failed; ``warnings`` carries the real reason
    ``EXPIRED``       - was confirmed but the confirmation window passed unused
    """

    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    EXPIRED = "expired"


class AgentOperation(BaseModel):
    """One planned or executed action requested by an agent client."""

    id: str
    #: "plan" for a plan row; otherwise the tool name the row authorizes,
    #: e.g. "ingest_sources".
    kind: str
    status: OperationStatus = OperationStatus.PROPOSED
    #: None for knowledge-base creation (the KB does not exist yet).
    kb_id: str | None = None
    #: The plan row this action was proposed under, if any.
    plan_id: str | None = None
    action: str = ""
    #: The exact parameters this row authorizes. Stored verbatim so the
    #: confirmed set can be shown to a reviewer and compared at execution time.
    parameters: dict[str, Any] = Field(default_factory=dict)
    #: sha256 over the canonical JSON of `parameters`.
    fingerprint: str = ""
    #: Set when a confirmation is recorded; the value the caller must pass back.
    confirmation_id: str | None = None
    confirmed_at: datetime | None = None
    confirmation_note: str = ""
    #: Set when execution begins. A consumed confirmation cannot be reused.
    consumed_at: datetime | None = None
    starts_at: datetime | None = None
    expires_at: datetime | None = None
    finished_at: datetime | None = None
    #: What the real service returned, summarised (ids, counts, statuses).
    result_summary: dict[str, Any] = Field(default_factory=dict)
    #: Honest caveats: partial failures, unavailable dependencies, mock/labelled
    #: generators. Never a success claim that the evidence does not support.
    warnings: list[str] = Field(default_factory=list)
    error_code: str = ""
    error_message: str = ""
    #: Who asked. "mcp" today; kept explicit so a future surface cannot
    #: silently masquerade as this one.
    origin: str = "mcp"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def is_executable(self) -> bool:
        return self.status == OperationStatus.CONFIRMED and self.consumed_at is None

    def expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or utcnow()) >= self.expires_at


__all__ = ["AgentOperation", "OperationStatus", "utcnow"]
