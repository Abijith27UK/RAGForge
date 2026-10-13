"""Error taxonomy for the RAGForge MCP server.

Every failure a client can cause has ONE named error carrying a message that is
safe to show an external AI client:

* no stack traces, no filesystem paths, no environment values;
* the message says what was wrong AND what the caller can do instead;
* `code` is stable so a client can branch on it without parsing prose.

These are raised inside tool functions. The MCP SDK turns a raised exception into
a tool result with `is_error=true`, which is what an MCP client expects; the
`payload()` form exists for tests and for any caller that wants the same
structure without going through the protocol.
"""
from __future__ import annotations

from typing import Any


class MCPToolError(RuntimeError):
    """Base class: a tool failure that is SAFE to return to a client."""

    code = "TOOL_ERROR"
    #: What the caller should do about it. Kept short and imperative.
    default_hint = ""

    def __init__(self, message: str, *, hint: str = "", **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint or self.default_hint
        self.details = details

    def payload(self) -> dict[str, Any]:
        """Structured form of the failure (used in tests and error results)."""
        out: dict[str, Any] = {"error": True, "code": self.code, "message": self.message}
        if self.hint:
            out["hint"] = self.hint
        if self.details:
            out["details"] = self.details
        return out


class KnowledgeBaseNotFoundError(MCPToolError):
    code = "KB_NOT_FOUND"
    default_hint = "Call list_knowledge_bases to see the available knowledge base ids."


class NotFoundError(MCPToolError):
    code = "NOT_FOUND"
    default_hint = "Check the identifier; ids are returned by the other tools."


class InvalidArgumentError(MCPToolError):
    code = "INVALID_ARGUMENT"
    default_hint = "Correct the argument and retry; ranges are documented in the tool schema."


class CapabilityUnavailableError(MCPToolError):
    """A real capability of RAGForge that cannot run right now.

    Used when a dependency (vector store, embedding model, generator) is missing
    or a KB is not indexed. It is NEVER a fallback: the requested work is not
    silently done a different way.
    """

    code = "CAPABILITY_UNAVAILABLE"
    default_hint = "Fix the reported dependency, or call get_knowledge_base_status for the current state."


class LifecycleBlockedError(MCPToolError):
    """A lifecycle gate (draft/review/approved/frozen) refuses this run."""

    code = "LIFECYCLE_BLOCKED"
    default_hint = (
        "Official runs require a FROZEN benchmark version; see the reported "
        "version states and complete the existing review workflow."
    )


class ConfirmationRequiredError(MCPToolError):
    """A state-changing action was called without a recorded confirmation."""

    code = "CONFIRMATION_REQUIRED"
    default_hint = (
        "Plan the change, have the user approve the exact parameters, then call "
        "confirm_plan and pass the returned confirmation id."
    )


class ConfirmationError(MCPToolError):
    """A confirmation exists but does not authorize THIS call.

    Used for an unknown id, a used/expired id, an action mismatch, or a
    parameter mismatch — the confirmation is bound to one exact parameter set,
    so any difference is a different operation.
    """

    code = "CONFIRMATION_INVALID"
    default_hint = (
        "Confirmations are single-use and parameter-bound. Re-plan the change and "
        "confirm the parameters you intend to run."
    )


class ConfirmationExpiredError(ConfirmationError):
    code = "CONFIRMATION_EXPIRED"
    default_hint = "Plan and confirm the change again; an expired confirmation authorizes nothing."


class DuplicateOperationError(MCPToolError):
    """The same operation appears to have been requested twice."""

    code = "DUPLICATE_OPERATION"
    default_hint = "Reuse the existing artifact named in the details, or change the parameters."


__all__ = [
    "MCPToolError",
    "KnowledgeBaseNotFoundError",
    "NotFoundError",
    "InvalidArgumentError",
    "CapabilityUnavailableError",
    "LifecycleBlockedError",
    "ConfirmationRequiredError",
    "ConfirmationError",
    "ConfirmationExpiredError",
    "DuplicateOperationError",
]
