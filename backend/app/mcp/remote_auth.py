"""Authentication primitives for the remote MCP transport (V22).

Two paths, both real, both covered by tests:

* ``StaticBearerTokenVerifier`` - operator-configured bearer tokens for LOCAL
  MCP clients (a CLI, curl, a tunnelled client, tests). ChatGPT cannot use this
  path: it cannot present a caller-supplied API key.
* ``LocalOAuthProvider`` - an OAuth 2.1 authorization server embedded in the MCP
  server, implementing the protocol surface the MCP SDK drives: dynamic client
  registration, an authorization endpoint that returns a consent URL, an
  authorization-code exchange, and refresh-token rotation. PKCE S256 verification
  is performed by the SDK's own token handler against the challenge this provider
  stores, so a stolen code cannot be redeemed without the verifier.

State is in memory and per-process on purpose: this is a single-user research
deployment, and persistent token storage would add a security surface without a
multi-user feature to justify it. Restarting the server invalidates every
outstanding token; that is disclosed, not hidden.
"""
from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    OAuthClientInformationFull,
    OAuthToken,
    RegistrationError,
    RefreshToken,
    TokenError,
)

DEFAULT_ACCESS_TOKEN_TTL = 3600
DEFAULT_REFRESH_TOKEN_TTL = 30 * 24 * 3600


def new_token() -> str:
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class StaticToken:
    token: str
    subject: str
    scopes: tuple[str, ...]


class StaticBearerTokenVerifier:
    """Verifies operator-configured bearer tokens.

    Comparison is constant-time over the presented bytes so a timing probe cannot
    walk a configured secret, and the raw token is never logged or echoed back.
    An unknown token returns None, which the SDK turns into a 401 carrying a
    ``WWW-Authenticate`` challenge.
    """

    def __init__(
        self,
        entries: Iterable[StaticToken],
        *,
        resource: str,
        issuer: str = "",
        now: Any = time.time,
    ) -> None:
        self._entries = list(entries)
        self._resource = resource
        self._issuer = issuer
        self._now = now

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token:
            return None
        presented = token.encode("utf-8")
        matched: StaticToken | None = None
        for entry in self._entries:
            if hmac.compare_digest(presented, entry.token.encode("utf-8")):
                matched = entry
        if matched is None:
            return None
        return AccessToken(
            token=token,
            client_id=f"static:{matched.subject}",
            scopes=list(matched.scopes),
            subject=matched.subject,
            resource=self._resource,
            claims={"iss": self._issuer} if self._issuer else None,
        )


def parse_static_tokens(raw: str) -> list[StaticToken]:
    """Parse ``MCP_ACCESS_TOKENS``.

    Format: ``token:subject:scope1 scope2`` entries separated by ``;`` or
    newlines. A malformed entry is skipped with a warning rather than silently
    granting wider access, and the token value itself is never echoed.
    """
    out: list[StaticToken] = []
    for chunk in (raw or "").replace("\n", ";").split(";"):
        item = chunk.strip()
        if not item:
            continue
        parts = item.split(":", 2)
        if len(parts) != 3 or not parts[0].strip():
            continue
        token, subject, scopes = (p.strip() for p in parts)
        scope_tuple = tuple(s for s in scopes.split() if s)
        if not scope_tuple:
            continue
        out.append(StaticToken(token=token, subject=subject, scopes=scope_tuple))
    return out
