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


#: V18: how much material an agent may hand over in ONE call. Chosen so a
#: legitimate lecture-notes handover fits, while a bulk corpus dump does not:
#: a corpus is built with the HTTP upload workflow, not through a tool result.
MAX_USER_DOCUMENTS = 20
MAX_USER_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_USER_DOCUMENTS_TOTAL_BYTES = 24 * 1024 * 1024

#: V18: URLs a plan may validate at once.
MAX_PLAN_URLS = 50

#: V18: sources one discovery call may register.
MAX_DISCOVERY_LIMIT = 20


def check_documents(documents: list[dict] | None) -> list[dict]:
    """Validate the shape and total size of a caller-supplied document list."""
    items = list(documents or [])
    if not items:
        return []
    if len(items) > MAX_USER_DOCUMENTS:
        raise InvalidArgumentError(
            f"{len(items)} documents in one call; the maximum is {MAX_USER_DOCUMENTS}.",
            hint="Split the handover into several calls and confirm each one.",
        )
    total = 0
    for item in items:
        if not isinstance(item, dict):
            raise InvalidArgumentError("Each document must be an object with file_name and content.")
        text = item.get("content_text")
        encoded = item.get("content_base64")
        size = len(text.encode("utf-8")) if isinstance(text, str) else 0
        if isinstance(encoded, str):
            size = (len(encoded) * 3) // 4
        if size > MAX_USER_DOCUMENT_BYTES:
            raise InvalidArgumentError(
                f"Document {item.get('file_name')!r} is about {size} bytes; the per-document cap is "
                f"{MAX_USER_DOCUMENT_BYTES} bytes.",
                hint="Use the HTTP upload workflow for large files.",
            )
        total += size
    if total > MAX_USER_DOCUMENTS_TOTAL_BYTES:
        raise InvalidArgumentError(
            f"The documents total about {total} bytes; the cap per call is "
            f"{MAX_USER_DOCUMENTS_TOTAL_BYTES} bytes.",
            hint="Split the handover into several calls, each with its own confirmation.",
        )
    return items


def check_urls(urls: list[str] | None) -> list[str]:
    """Validate a list of URLs a plan is asked to consider (not fetched here)."""
    items = [str(u).strip() for u in (urls or []) if str(u).strip()]
    if len(items) > MAX_PLAN_URLS:
        raise InvalidArgumentError(
            f"{len(items)} URLs in one call; the maximum is {MAX_PLAN_URLS}.",
            hint="Plan a smaller batch, or register sources in the UI/API.",
        )
    return items



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
    "MAX_USER_DOCUMENTS",
    "MAX_USER_DOCUMENT_BYTES",
    "MAX_USER_DOCUMENTS_TOTAL_BYTES",
    "MAX_PLAN_URLS",
    "MAX_DISCOVERY_LIMIT",
    "check_top_k",
    "check_query",
    "check_max_chunks",
    "check_documents",
    "check_urls",
    "truncate_text",
]
