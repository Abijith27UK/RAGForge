"""Hard limits for MCP tool calls.

An MCP client is an untrusted caller: it decides how much text to ask for, how
many chunks to request and how large a retrieval window to open. Every one of
those numbers is bounded here, in ONE place, and the bound is reported back to
the caller (`limits_applied`) so a truncation is never silent.

The limits are deliberately conservative: an MCP tool result is fed into another
model's context window, so returning an entire corpus would be both a cost and a
data-exposure problem, not a feature.
"""
from __future__ import annotations

from app.mcp.errors import InvalidArgumentError

#: Retrieval window a client may request.
DEFAULT_TOP_K = 5
MAX_TOP_K = 50

#: Question/query length.
MAX_QUERY_CHARS = 2000

#: Evidence-by-id lookups.
DEFAULT_EVIDENCE_CHUNKS = 10
MAX_EVIDENCE_CHUNKS = 50

#: Characters of ONE chunk's text returned to the client.
MAX_TEXT_CHARS = 4000

#: Chunks whose text is included with a grounded answer.
MAX_ANSWER_EVIDENCE = 10

#: Questions scored in one evaluation run.
MAX_EVALUATION_QUESTIONS = 200

#: Claim/citation lists returned with an answer (the pipeline may produce more;
#: the count is always reported, so a truncated list is visible as a number).
MAX_ANSWER_CLAIMS = 100


def check_top_k(top_k: int | None) -> int:
    """Validate a requested retrieval window."""
    if top_k is None:
        return DEFAULT_TOP_K
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise InvalidArgumentError(
            f"top_k must be an integer between 1 and {MAX_TOP_K}; got {top_k!r}."
        )
    if top_k < 1 or top_k > MAX_TOP_K:
        raise InvalidArgumentError(
            f"top_k must be between 1 and {MAX_TOP_K}; got {top_k}.",
            hint=f"Ask for at most {MAX_TOP_K} chunks per call.",
        )
    return top_k


def check_query(query: str) -> str:
    """Validate a search query / question."""
    if not isinstance(query, str):
        raise InvalidArgumentError("query must be a string.")
    cleaned = query.strip()
    if not cleaned:
        raise InvalidArgumentError("query must not be empty.")
    if len(cleaned) > MAX_QUERY_CHARS:
        raise InvalidArgumentError(
            f"query is {len(cleaned)} characters; the maximum is {MAX_QUERY_CHARS}.",
            hint="Shorten the question or split it into separate calls.",
        )
    return cleaned


def check_max_chunks(value: int | None, *, default: int = DEFAULT_EVIDENCE_CHUNKS) -> int:
    """Validate a requested evidence-chunk count."""
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidArgumentError("max_chunks must be an integer.")
    if value < 1 or value > MAX_EVIDENCE_CHUNKS:
        raise InvalidArgumentError(
            f"max_chunks must be between 1 and {MAX_EVIDENCE_CHUNKS}; got {value}."
        )
    return value


def truncate_text(text: str, limit: int = MAX_TEXT_CHARS) -> tuple[str, bool]:
    """Return (text, truncated). A truncation is reported, never hidden."""
    if text is None:
        return "", False
    if len(text) <= limit:
        return text, False
    return text[:limit], True


__all__ = [
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "MAX_QUERY_CHARS",
    "DEFAULT_EVIDENCE_CHUNKS",
    "MAX_EVIDENCE_CHUNKS",
    "MAX_TEXT_CHARS",
    "MAX_ANSWER_EVIDENCE",
    "MAX_EVALUATION_QUESTIONS",
    "MAX_ANSWER_CLAIMS",
    "check_top_k",
    "check_query",
    "check_max_chunks",
    "truncate_text",
]
