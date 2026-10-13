"""Operation lifecycle for agent-requested changes (V18).

This module owns exactly one responsibility: making "a plan is not a
confirmation" enforceable.

    plan      -> an ``AgentOperation`` row with status=proposed. Authorizes NOTHING.
    confirm   -> records that ONE action with ONE exact parameter set was
                 approved by the user. Issues a single-use confirmation id that
                 expires.
    execute   -> refuses unless the caller presents that confirmation AND the
                 parameters it is about to run match the confirmed fingerprint
                 exactly. The confirmation is consumed BEFORE the work starts,
                 so a retry after a crash cannot run twice.

Honest limitation (also stated in docs/mcp-agentic-workflow.md): MCP over stdio
cannot verify that a human pressed "yes". The server enforces everything it CAN
verify — that the action is one a read-only plan proposed, that the parameters
are byte-for-byte the confirmed ones, that the confirmation is used at most once
and within its window. Human approval itself is the host client's job; the
recommended client behaviour is to require user approval before ``confirm_plan``
is ever called.

Nothing here executes a RAGForge action: the MCP tools do that with the existing
services and write the outcome back through :meth:`complete`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import timedelta
from typing import Any

from app.mcp.errors import (
    ConfirmationError,
    ConfirmationExpiredError,
    ConfirmationRequiredError,
    InvalidArgumentError,
    NotFoundError,
)
from app.repositories.sqlite_repo import Repository
from app.schemas.agent_ops import AgentOperation, OperationStatus, utcnow
from app.utils.ids import new_id

logger = logging.getLogger(__name__)

#: A confirmation is valid for this long. Long enough for a human to read a plan
#: and answer, short enough that an abandoned confirmation does not sit around
#: authorizing work hours later.
DEFAULT_CONFIRMATION_TTL = timedelta(minutes=30)

#: Action names this layer can authorize. Anything else is refused, so a tool
#: cannot invent a new consequential action without touching this allow-list.
CONSEQUENTIAL_ACTIONS = (
    "create_knowledge_base",
    "ingest_sources",
    "build_knowledge_base",
    "add_user_documents",
)


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: same parameters -> same string, always.

    Keys are sorted and separators fixed so a fingerprint cannot depend on
    dictionary insertion order (which would make confirmations flaky rather than
    strict).
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def fingerprint(parameters: Any) -> str:
    """sha256 over the canonical form of a parameter set."""
    return hashlib.sha256(canonical_json(parameters).encode("utf-8")).hexdigest()


class OperationService:
    """Reads and writes agent operation rows; enforces the lifecycle rules."""

    def __init__(self, repo: Repository, ttl: timedelta | None = None) -> None:
        self.repo = repo
        self.ttl = ttl or DEFAULT_CONFIRMATION_TTL

    # -- planning --------------------------------------------------------------

    def create_plan(
        self,
        *,
        kb_id: str | None,
        parameters: dict[str, Any],
        summary: str = "",
        warnings: list[str] | None = None,
    ) -> AgentOperation:
        """Record a plan. It authorizes nothing until it is confirmed."""
        op = AgentOperation(
            id=new_id("plan"),
            kind="plan",
            status=OperationStatus.PROPOSED,
            kb_id=kb_id,
            parameters=parameters,
            fingerprint=fingerprint(parameters),
            result_summary={"summary": summary} if summary else {},
            warnings=list(warnings or []),
        )
        self.repo.create_mcp_operation(op)
        logger.info("MCP plan %s proposed (kb=%s)", op.id, kb_id or "new")
        return op

    def propose_action(
        self,
        *,
        plan_id: str,
        action: str,
        kb_id: str | None,
        parameters: dict[str, Any],
        summary: str = "",
        warnings: list[str] | None = None,
    ) -> AgentOperation:
        """Attach one confirmable action to a plan, with its exact parameters."""
        if action not in CONSEQUENTIAL_ACTIONS:
            raise InvalidArgumentError(
                f"{action!r} is not a consequential action this server can authorize.",
                hint=f"Authorizable actions: {', '.join(CONSEQUENTIAL_ACTIONS)}.",
            )
        op = AgentOperation(
            id=new_id("op"),
            kind=action,
            action=action,
            status=OperationStatus.PROPOSED,
            kb_id=kb_id,
            plan_id=plan_id,
            parameters=parameters,
            fingerprint=fingerprint(parameters),
            result_summary={"summary": summary} if summary else {},
            warnings=list(warnings or []),
        )
        self.repo.create_mcp_operation(op)
        return op

    # -- confirmation ----------------------------------------------------------

    def confirm(
        self,
        *,
        plan_id: str,
        action: str,
        parameters_fingerprint: str,
        note: str = "",
    ) -> tuple[AgentOperation, AgentOperation]:
        """Record a user confirmation for ONE proposed action.

        Refuses when the action is not attached to this plan, when the
        fingerprint does not match a proposed action (i.e. the parameters were
        changed after planning), or when the action was already confirmed.
        """
        plan = self.repo.get_mcp_operation(plan_id)
        if plan is None:
            raise NotFoundError(
                f"Plan {plan_id!r} does not exist.",
                hint="Call plan_knowledge_base (or plan_source_intake) to create one first.",
            )
        if plan.kind != "plan":
            raise InvalidArgumentError(
                f"{plan_id!r} is not a plan (kind={plan.kind!r}).",
                hint="Pass the plan_id returned by a planning tool.",
            )

        candidates = [
            op
            for op in self.repo.list_mcp_operations(plan_id=plan_id, limit=200)
            if op.action == action
        ]
        if not candidates:
            raise InvalidArgumentError(
                f"Plan {plan_id} does not propose a {action!r} action.",
                hint=(
                    "Available in this plan: "
                    + (", ".join(sorted({op.action for op in self.repo.list_mcp_operations(plan_id=plan_id, limit=200)}) ) or "none")
                ),
            )

        wanted = parameters_fingerprint.strip()
        match = next((op for op in candidates if op.fingerprint == wanted), None)
        if match is None:
            raise ConfirmationError(
                f"The fingerprint {wanted[:12]!r} does not match any {action} this plan proposes.",
                hint=(
                    "Confirm one of the plan's fingerprints exactly; a changed parameter set "
                    "is a different operation and must be planned again. Proposed: "
                    + ", ".join(op.fingerprint[:12] for op in candidates)
                ),
            )
        if match.status == OperationStatus.CONFIRMED:
            raise ConfirmationError(
                f"{action} was already confirmed with confirmation id {match.confirmation_id}.",
                hint="Execute the existing confirmation, or plan the change again.",
            )
        if match.status != OperationStatus.PROPOSED:
            raise ConfirmationError(
                f"{action} is {match.status.value}, so it cannot be confirmed again.",
                hint="Create a new plan for a new authorization.",
            )

        now = utcnow()
        match.status = OperationStatus.CONFIRMED
        match.confirmation_id = new_id("conf")
        match.confirmed_at = now
        match.confirmation_note = note
        match.expires_at = now + self.ttl
        match.updated_at = now
        self.repo.update_mcp_operation(match)
        logger.info(
            "MCP confirmation %s recorded for %s (plan=%s, kb=%s)",
            match.confirmation_id, action, plan_id, match.kb_id,
        )
        return plan, match

    # -- execution -------------------------------------------------------------

    def check(
        self, *, confirmation_id: str, action: str, parameters: dict[str, Any]
    ) -> AgentOperation:
        """Validate a confirmation WITHOUT consuming it.

        A tool calls this before doing any cheap validation of its own, so that a
        forged, reused, expired or mismatched confirmation is always reported as
        such — even when the arguments would also fail for another reason. The
        confirmation is only consumed by :meth:`begin_execution`.
        """
        if not confirmation_id:
            raise ConfirmationRequiredError(
                f"{action} changes RAGForge state, so it needs a recorded user confirmation.",
                hint=(
                    "Call plan_knowledge_base to get the exact parameters, have the user approve "
                    "them, then call confirm_plan with the plan's fingerprint."
                ),
            )
        op = self.repo.find_mcp_operation_by_confirmation(confirmation_id)
        if op is None:
            raise ConfirmationError(
                f"Confirmation {confirmation_id!r} is not known to this workspace.",
                hint="Confirmations are issued by confirm_plan and are specific to one action.",
            )
        if op.action != action:
            raise ConfirmationError(
                f"Confirmation {confirmation_id} authorizes {op.action!r}, not {action!r}.",
                hint="Ask for a confirmation of the action you intend to run.",
            )
        if op.consumed_at is not None or op.status in (
            OperationStatus.RUNNING,
            OperationStatus.DONE,
            OperationStatus.ERROR,
        ):
            raise ConfirmationError(
                f"Confirmation {confirmation_id} was already used "
                f"({op.consumed_at.isoformat() if op.consumed_at else op.status.value}).",
                hint="A confirmation is single-use: plan and confirm the change again to repeat it.",
            )
        if op.status == OperationStatus.EXPIRED or op.expired():
            op.status = OperationStatus.EXPIRED
            op.updated_at = utcnow()
            self.repo.update_mcp_operation(op)
            raise ConfirmationExpiredError(
                f"Confirmation {confirmation_id} expired at "
                f"{op.expires_at.isoformat() if op.expires_at else 'unknown'}.",
                hint="Plan the change again; a fresh confirmation is needed.",
            )
        if op.status != OperationStatus.CONFIRMED:
            raise ConfirmationError(
                f"Confirmation {confirmation_id} is {op.status.value}, not confirmed.",
                hint="Confirmations are created by confirm_plan.",
            )

        actual = fingerprint(parameters)
        if actual != op.fingerprint:
            raise ConfirmationError(
                f"The parameters of this call do not match what was confirmed for {action}.",
                hint=(
                    f"Confirmed fingerprint {op.fingerprint[:12]}, call fingerprint {actual[:12]}. "
                    "Re-plan the change, re-confirm it, and run it with the confirmed parameters."
                ),
            )
        return op

    def begin_execution(
        self, *, confirmation_id: str, action: str, parameters: dict[str, Any]
    ) -> AgentOperation:
        """Validate AND consume a confirmation. Returns the operation row.

        Raises before any work happens when anything does not line up, so a
        refused call has no side effects at all. Consuming here (rather than after
        the work) is what makes the confirmation single-use even if the process
        dies mid-execution.
        """
        op = self.check(confirmation_id=confirmation_id, action=action, parameters=parameters)
        now = utcnow()
        op.status = OperationStatus.RUNNING
        op.consumed_at = now
        op.starts_at = now
        op.updated_at = now
        self.repo.update_mcp_operation(op)
        return op

    def complete(
        self,
        op: AgentOperation,
        *,
        result_summary: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> AgentOperation:
        """Record a successful execution against its own operation row."""
        now = utcnow()
        op.status = OperationStatus.DONE
        op.finished_at = now
        op.updated_at = now
        if result_summary:
            op.result_summary = {**op.result_summary, **result_summary}
        if warnings:
            op.warnings = list(op.warnings) + [w for w in warnings if w]
        self.repo.update_mcp_operation(op)
        return op

    def fail(self, op: AgentOperation, *, code: str, message: str) -> AgentOperation:
        """Record a failed execution. The row keeps the real reason."""
        now = utcnow()
        op.status = OperationStatus.ERROR
        op.finished_at = now
        op.updated_at = now
        op.error_code = code
        op.error_message = message
        self.repo.update_mcp_operation(op)
        return op

    # -- reporting -------------------------------------------------------------

    def get(self, operation_id: str) -> AgentOperation:
        op = self.repo.get_mcp_operation(operation_id)
        if op is None:
            raise NotFoundError(
                f"Operation {operation_id!r} does not exist.",
                hint="Operation ids come from the planning and execution tools.",
            )
        if op.status == OperationStatus.CONFIRMED and op.expired():
            op.status = OperationStatus.EXPIRED
            op.updated_at = utcnow()
            self.repo.update_mcp_operation(op)
        return op

    def for_kb(self, kb_id: str, limit: int = 50) -> list[AgentOperation]:
        return self.repo.list_mcp_operations(kb_id=kb_id, limit=limit)


__all__ = [
    "OperationService",
    "CONSEQUENTIAL_ACTIONS",
    "DEFAULT_CONFIRMATION_TTL",
    "canonical_json",
    "fingerprint",
]
