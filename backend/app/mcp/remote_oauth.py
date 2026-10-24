"""Embedded OAuth 2.1 authorization server for the remote MCP transport (V22).

ChatGPT's custom-connector flow is OAuth 2.1: authorization code + PKCE S256,
RFC 9728 protected-resource metadata and RFC 8707 resource indicators. It will
not present a static API key. So a RAGForge that wants to be reachable from
ChatGPT must BE an authorization server (or delegate to one).

This is the "be an authorization server" option, implemented against the
protocol surface the MCP Python SDK drives:

* ``/register``        dynamic client registration (SDK route, our storage)
* ``/authorize``       authorization request -> consent URL -> redirect with code
* ``/token``           authorization code -> access + refresh (PKCE verified by
                       the SDK handler against the challenge we stored)
* refresh tokens       rotated on every use, single-use, revocable

Deliberate scope: one local user, in-memory state, consent that requires an
explicit POST (auto-approve is available for a local-only deployment and is
OFF by default). See docs/remote-mcp-chatgpt.md for what this does and does not
get you in ChatGPT.
"""
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
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

from app.mcp.remote_auth import (
    DEFAULT_ACCESS_TOKEN_TTL,
    DEFAULT_REFRESH_TOKEN_TTL,
    new_token,
)

logger = logging.getLogger("ragforge.mcp.remote.oauth")


@dataclass
class _Client:
    info: OAuthClientInformationFull
    created_at: float = field(default_factory=time.time)


@dataclass
class _PendingAuthorization:
    client_id: str
    params: AuthorizationParams
    created_at: float


class LocalOAuthProvider(OAuthAuthorizationServerProvider):
    """A single-user OAuth 2.1 authorization server held in memory.

    ``subject`` is the identity every issued token carries; it is fixed at
    construction and is never taken from the client, so a client cannot ask to
    be someone else. ``resource`` is the RFC 8707 indicator: tokens are minted
    for exactly one MCP resource URL and the SDK's bearer middleware rejects a
    token presented at a different one.
    """

    def __init__(
        self,
        *,
        issuer_url: str,
        resource_url: str,
        subject: str = "local",
        auto_approve: bool = False,
        access_token_ttl: int = DEFAULT_ACCESS_TOKEN_TTL,
        refresh_token_ttl: int = DEFAULT_REFRESH_TOKEN_TTL,
        scopes: Sequence[str] = (),
        now: Any = time.time,
    ) -> None:
        self._issuer = issuer_url.rstrip("/")
        self._resource = resource_url
        self._subject = subject
        self._auto_approve = auto_approve
        self._access_ttl = access_token_ttl
        self._refresh_ttl = refresh_token_ttl
        self._scopes = tuple(scopes)
        self._now = now
        self._clients: dict[str, _Client] = {}
        self._pending: dict[str, _PendingAuthorization] = {}
        self._codes: dict[str, AuthorizationCode] = {}
        self._refresh: dict[str, RefreshToken] = {}
        self._access: dict[str, AccessToken] = {}
        # Consent requests a human has not answered yet, exposed by the health
        # route so an operator can see that a browser tab is waiting.
        self.pending_consents: dict[str, _PendingAuthorization] = {}

    @property
    def resource_url(self) -> str:
        return self._resource

    @property
    def issuer_url(self) -> str:
        return self._issuer

    # -- clients -----------------------------------------------------------
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        client = self._clients.get(client_id)
        return client.info if client else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = _Client(info=client_info)
        logger.info("registered MCP OAuth client %s", client_info.client_id)

    # -- authorization ------------------------------------------------------
    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        code = new_token()
        self._pending[code] = _PendingAuthorization(
            client_id=client.client_id, params=params, created_at=self._now()
        )
        if self._auto_approve:
            return await self._complete(code)
        self.pending_consents[code] = self._pending[code]
        query = urlencode({"request": code, "client_id": client.client_id})
        return f"{self._issuer}/consent?{query}"

    async def _complete(self, request_id: str) -> str:
        pending = self._pending.pop(request_id, None)
        self.pending_consents.pop(request_id, None)
        if pending is None:
            raise AuthorizeError(error="invalid_request", error_description="unknown request")
        code = new_token()
        self._codes[code] = AuthorizationCode(
            code=code,
            scopes=list(pending.params.scopes or self._scopes),
            expires_at=self._now() + 300,
            client_id=pending.client_id,
            code_challenge=pending.params.code_challenge,
            redirect_uri=pending.params.redirect_uri,
            resource=pending.params.resource or self._resource,
            subject=self._subject,
        )
        query = urlencode(
            {"code": code, "state": pending.params.state or "", "resource": self._resource}
        )
        sep = "&" if "?" in str(pending.params.redirect_uri) else "?"
        return f"{str(pending.params.redirect_uri).rstrip('/')}{sep}{query}"

    async def approve(self, request_id: str) -> str:
        """Complete a pending authorization after a human consented."""
        return await self._complete(request_id)

    async def deny(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
        self.pending_consents.pop(request_id, None)

    # -- tokens -------------------------------------------------------------
    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        code = self._codes.get(authorization_code)
        if code is None or code.client_id != client.client_id:
            return None
        if code.expires_at <= self._now():
            self._codes.pop(authorization_code, None)
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        self._codes.pop(authorization_code.code, None)
        return self._mint(client_id=client.client_id, scopes=authorization_code.scopes)

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        stored = self._refresh.pop(refresh_token, None)
        if stored is None or stored.client_id != client.client_id:
            return None
        if stored.expires_at is not None and stored.expires_at <= time.time():
            return None
        return stored

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        return self._mint(client_id=client.client_id, scopes=scopes or refresh_token.scopes)

    async def revoke_token(self, token: str) -> None:
        """Best-effort revocation: drop the token from every table it could be."""
        self._access.pop(token, None)
        self._refresh.pop(token, None)
        self._codes.pop(token, None)

    def _mint(self, *, client_id: str, scopes: Iterable[str]) -> OAuthToken:
        scope_list = [s for s in scopes if s]
        access = AccessToken(
            token=new_token(),
            client_id=client_id,
            scopes=scope_list,
            expires_at=self._now() + self._access_ttl,
            resource=self._resource,
            subject=self._subject,
            claims={"iss": self._issuer},
        )
        refresh = RefreshToken(
            token=new_token(),
            client_id=client_id,
            scopes=scope_list,
            expires_at=int(self._now() + self._refresh_ttl),
            resource=self._resource,
            subject=self._subject,
        )
        self._access[access.token] = access
        self._refresh[refresh.token] = refresh
        return OAuthToken(
            access_token=access.token,
            token_type="Bearer",
            expires_in=self._access_ttl,
            refresh_token=refresh.token,
            scope=" ".join(scope_list),
        )
