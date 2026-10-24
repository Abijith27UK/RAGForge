"""Remote MCP transport for RAGForge (V22): Streamable HTTP over the official SDK.

Why this module exists
----------------------
The V17 server speaks **stdio**. A stdio server is launched BY the MCP client, so
it is only reachable by a program the local user started — exactly the security
boundary RAGForge wants on one machine, and exactly why it cannot serve ChatGPT.
To be reachable from a remote host an MCP server must expose an HTTP endpoint.

What ChatGPT actually requires (verified against the current OpenAI docs, cited in
docs/remote-mcp-chatgpt.md):
* a remote MCP endpoint speaking **Streamable HTTP** (SSE is also accepted);
* an **HTTPS** URL the browser can reach (a tunnel is allowed for custom
  connections; a public URL is required for plugin-directory submission);
* for anything private, **OAuth 2.1**: authorization code + **PKCE S256**,
  RFC 9728 protected-resource metadata, RFC 8707 `resource` indicators, and a
  client-registration method (dynamic registration, CIMD, or a pre-registered
  client). ChatGPT cannot present a static API key or bearer token of its own,
  so a static-token path is NOT usable from ChatGPT — it is provided here only
  for local/CLI MCP clients and for tests.

What this module does
---------------------
It reuses the V17 server OBJECT (`app.mcp.server.create_server`) so the tool list,
schemas, confirmation gates, provenance and grounding behaviour cannot drift
between stdio and HTTP. Nothing here re-implements retrieval or answering.

Security properties it actually enforces:
* every request must carry a valid Bearer token (401 + `WWW-Authenticate` with a
  protected-resource metadata pointer when it does not);
* tokens are resolved to a **principal** (subject + scopes) and every tool call
  that names a knowledge base is checked against the KB access policy — a token
  issued for another subject cannot read a KB it does not own;
* Host/Origin are validated (DNS-rebinding protection) and request bodies are
  size-limited;
* failures return MCP error objects, never tracebacks, filesystem paths, secrets
  or environment values;
* a `/healthz` route reports liveness only, with no configuration detail.

Deliberate limits: single-process, in-memory OAuth state (an access/refresh token
is lost on restart), one local user. See docs/remote-mcp-chatgpt.md.
"""
from __future__ import annotations

import hmac
import logging
import secrets
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from app.config import Settings, get_settings
from app.mcp.remote_auth import StaticBearerTokenVerifier, StaticToken, parse_static_tokens
from app.mcp.remote_oauth import LocalOAuthProvider

logger = logging.getLogger("ragforge.mcp.remote")

#: Scope that may see and use every knowledge base, regardless of ownership.
ADMIN_SCOPE = "ragforge:admin"
#: Scope required to run read-only tools (search / evidence / answer).
READ_SCOPE = "ragforge:read"
#: Scope required for build-side tools (create, ingest, index).
WRITE_SCOPE = "ragforge:write"

ALL_SCOPES = (ADMIN_SCOPE, READ_SCOPE, WRITE_SCOPE)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Principal:
    """The identity a token represents. Resolved from the TOKEN, never from a
    tool argument: a client cannot ask to be someone else."""

    subject: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    client_id: str = ""

    def has_scope(self, scope: str) -> bool:
        return ADMIN_SCOPE in self.scopes or scope in self.scopes

    def is_admin(self) -> bool:
        return ADMIN_SCOPE in self.scopes


#: Set for the duration of one request by the auth middleware. `None` means the
#: transport is running WITHOUT authentication (local stdio-style deployments),
#: which is only ever the case when the operator disabled auth explicitly.
_principal: ContextVar[Principal | None] = ContextVar("ragforge_mcp_principal", default=None)


def current_principal() -> Principal | None:
    return _principal.get()


class AccessDenied(PermissionError):
    """Raised when a principal may not touch a knowledge base."""


class KBAccessPolicy:
    """Who may use which knowledge base.

    Rules, in order:
      1. no principal (auth disabled)        -> allowed (single-user local mode)
      2. `ragforge:admin` in the token scopes -> allowed
      3. a KB that declares an `owner`        -> only that owner
      4. a KB without an owner                -> allowed (treated as shared)

    Rule 4 is what keeps every existing knowledge base usable after upgrading an
    install that has no owners at all; a KB becomes private the moment it records
    one. The `owner` field lives in the KB's own metadata, so no schema migration
    is required to start using it.
    """

    def __init__(self, repo: Any) -> None:
        self._repo = repo

    def owner_of(self, kb_id: str) -> str | None:
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            return None
        return getattr(kb, "owner", None)

    def is_allowed(self, principal: Principal | None, kb_id: str) -> bool:
        if principal is None or principal.is_admin():
            return True
        owner = self.owner_of(kb_id)
        if owner is None:
            return True
        return owner == principal.subject

    def check(self, principal: Principal | None, kb_id: str, *, tool: str = "") -> None:
        if self.is_allowed(principal, kb_id):
            return
        who = principal.subject if principal else "anonymous"
        raise AccessDenied(
            f"Knowledge base {kb_id} belongs to another user and cannot be used by "
            f"'{who}'" + (f" (tool: {tool})" if tool else "")
        )

    def filter_ids(self, principal: Principal | None, kb_ids: Iterable[str]) -> list[str]:
        return [kb for kb in kb_ids if self.is_allowed(principal, kb)]


# ---------------------------------------------------------------------------
# Scoped bridge: every tool call that names a KB is checked
# ---------------------------------------------------------------------------

#: Tool argument that carries a knowledge-base id, in the V17 tool schemas.
_KB_ARG = "kb_id"


class ScopedBridge:
    """Wraps `RAGForgeBridge` so no tool can reach a KB the caller may not use.

    The 20+ MCP tools all take their knowledge base as a ``kb_id`` argument and
    all go through one object, so interception belongs here rather than in each
    tool: a new tool that forgets to check inherits the check for free, and a
    tool that names no KB (``list_knowledge_bases``) is handled explicitly.
    """

    def __init__(self, bridge: Any, policy: KBAccessPolicy) -> None:
        self._bridge = bridge
        self._policy = policy

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._bridge, name)
        if not callable(attribute):
            return attribute

        def guarded(*args: Any, **kwargs: Any) -> Any:
            kb_id = kwargs.get(_KB_ARG)
            if kb_id is None and args and isinstance(args[0], str):
                kb_id = args[0]
            if kb_id is not None:
                self._policy.check(current_principal(), str(kb_id), tool=name)
            return attribute(*args, **kwargs)

        guarded.__name__ = name
        guarded.__doc__ = attribute.__doc__
        return guarded

    def list_knowledge_bases(self) -> Any:
        """List only the knowledge bases this principal may use.

        Returning every KB id and name to a principal that cannot open them
        would leak another user's corpus metadata, so the list is filtered
        rather than annotated.
        """
        payload = self._bridge.list_knowledge_bases()
        principal = current_principal()
        if principal is None or principal.is_admin():
            return payload
        kept = [
            kb
            for kb in payload.knowledge_bases
            if self._policy.is_allowed(principal, kb.id)
        ]
        if len(kept) != len(payload.knowledge_bases):
            notes = list(payload.notes) + [
                f"{len(payload.knowledge_bases) - len(kept)} knowledge base(s) are not "
                "visible to this token and were omitted"
            ]
            payload = payload.model_copy(update={"knowledge_bases": kept, "total": len(kept), "notes": notes})
        return payload


# ---------------------------------------------------------------------------
# Request middleware: resolve the principal once per HTTP request
# ---------------------------------------------------------------------------


def _principal_from_scope(scope: Mapping[str, Any]) -> Principal | None:
    """Build the principal from the ASGI scope's authenticated user.

    The SDK's bearer middleware stores an ``AuthenticatedUser`` in ``scope['user']``
    with the verified ``AccessToken`` attached. Nothing else may set it.
    """
    user = scope.get("user")
    token = getattr(user, "access_token", None)
    if token is None:
        return None
    subject = token.subject or getattr(user, "display_name", "") or "unknown"
    return Principal(
        subject=str(subject),
        scopes=frozenset(token.scopes or ()),
        client_id=str(getattr(token, "client_id", "")),
    )


class PrincipalMiddleware:
    """ASGI middleware that publishes the verified principal for one request.

    Tool calls run inside the request task (and anyio propagates contextvars into
    the worker thread that runs a sync tool), so the contextvar is the reliable
    way for the scope check to see WHO is calling without changing 20 tool
    signatures.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    @property
    def routes(self) -> Any:
        """Delegate to the wrapped app so tooling/tests can still introspect it."""
        return getattr(self._app, "routes", [])

    async def __call__(self, scope: Mapping[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        token = _principal_from_scope(scope)
        ctx = _principal.set(token)
        try:
            await self._app(scope, receive, send)
        finally:
            _principal.reset(ctx)


# ---------------------------------------------------------------------------
# Server construction
# ---------------------------------------------------------------------------

def parse_tokens(settings: Settings) -> tuple[list[StaticToken], list[str]]:
    """(configured tokens, configuration warnings). Never echoes a token value."""
    entries = parse_static_tokens(settings.mcp_access_tokens or "")
    warnings: list[str] = []
    for index, entry in enumerate(entries):
        if len(entry.token) < 16:
            warnings.append(
                f"MCP_ACCESS_TOKENS entry {index + 1} is shorter than 16 characters; "
                "it will be accepted but is not a strong secret"
            )
    return entries, warnings


def resolve_resource_urls(settings: Settings) -> tuple[str, str]:
    """(resource_url, issuer_url).

    The resource URL is the RFC 8707 indicator: the URL clients must name when
    they authorize, and the audience every token is bound to. It defaults to the
    local endpoint so a development install works with no configuration.
    """
    origin = settings.mcp_issuer_url or f"http://127.0.0.1:{settings.mcp_remote_port}"
    resource = settings.mcp_resource_server_url or f"{origin.rstrip('/')}/mcp"
    return resource.rstrip("/"), origin.rstrip("/")


def create_remote_server(
    bridge: Any | None = None,
    settings: Settings | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the MCP server object for Streamable HTTP.

    Reuses `app.mcp.server.create_server` so the tool surface is IDENTICAL to the
    stdio server; only authentication and the transport differ. Returns
    ``(server, info)`` where ``info`` describes the auth configuration for the
    health route and the logs (never secrets).
    """
    from mcp.server.auth.settings import AuthSettings

    from app.mcp.bridge import RAGForgeBridge
    from app.mcp.server import create_server

    settings = settings or get_settings()
    resource, issuer = resolve_resource_urls(settings)
    real_bridge = bridge or RAGForgeBridge()
    policy = KBAccessPolicy(real_bridge.repo)
    scoped = ScopedBridge(real_bridge, policy)

    tokens, warnings = parse_tokens(settings)
    oauth: LocalOAuthProvider | None = None
    verifier: Any | None = None
    auth_settings: Any | None = None

    if settings.mcp_allow_oauth:
        oauth = LocalOAuthProvider(
            issuer_url=issuer,
            resource_url=resource,
            subject=settings.mcp_oauth_subject,
            auto_approve=settings.mcp_oauth_auto_approve,
            scopes=("ragforge:read", "ragforge:write"),
        )

    if tokens or oauth is not None:
        class _Verifier:
            """Accepts a static token or an OAuth-issued token for this resource."""

            def __init__(self) -> None:
                self._static = StaticBearerTokenVerifier(tokens, resource=resource, issuer=issuer)

            async def verify_token(self, token: str) -> Any:
                result = await self._static.verify_token(token)
                if result is not None:
                    return result
                if oauth is None:
                    return None
                access = oauth._access.get(token)
                if access is None or access.resource != resource:
                    return None
                if access.expires_at is not None and access.expires_at <= oauth._now():
                    oauth._access.pop(token, None)
                    return None
                return access

        verifier = _Verifier()
        auth_settings = AuthSettings(
            issuer_url=issuer,  # type: ignore[arg-type]
            resource_server_url=resource,  # type: ignore[arg-type]
            required_scopes=[],
            validate_token_resource=True,
        )

    server = create_server(
        bridge=scoped,
        token_verifier=verifier,
        auth=auth_settings,
        auth_server_provider=oauth,
    )
    # Kept on the server so the consent route (and tests) can reach it without a
    # second global.
    server._oauth_provider = oauth

    info = {
        "resource_url": resource,
        "issuer_url": issuer,
        "authentication": "oauth" if oauth is not None else ("static-token" if tokens else "disabled"),
        "static_tokens": len(tokens),
        "oauth_auto_approve": bool(oauth is not None and oauth._auto_approve),
        "warnings": warnings,
    }
    return server, info


# ---------------------------------------------------------------------------
# ASGI application
# ---------------------------------------------------------------------------

HEALTH_FIELDS = ("status", "transport", "version", "authentication")


def create_remote_app(
    bridge: Any | None = None,
    settings: Settings | None = None,
) -> Any:
    """Build the Streamable HTTP ASGI application.

    Shape of the app, outermost first:

        PrincipalMiddleware      publish the verified principal per request
        SDK auth middleware      401 + WWW-Authenticate for a bad/absent token
        SDK transport security   Host/Origin validation (DNS-rebinding defence)
        SDK request-size limit   body cap before any parsing
        Starlette routes         /healthz, /consent (human approval), /mcp
    """
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse

    settings = settings or get_settings()
    server, info = create_remote_server(bridge=bridge, settings=settings)

    mcp_path = "/mcp"
    app = server.streamable_http_app(
        streamable_http_path=mcp_path,
        max_request_body_size=settings.mcp_max_request_bytes,
        host=settings.mcp_remote_host,
        transport_security=_transport_security(settings, info["issuer_url"]),
    )
    oauth: LocalOAuthProvider | None = getattr(server, "_oauth_provider", None)

    async def health(_: Request) -> JSONResponse:
        """Liveness only: no hostnames, no paths, no counts, no secrets."""
        return JSONResponse({key: (info["authentication"] if key == "authentication" else
                                   ("ok" if key == "status" else
                                    ("streamable-http" if key == "transport" else "0.1.0")))
                             for key in HEALTH_FIELDS})

    async def consent(request: Request):
        """Human approval page for the OAuth flow (auto-approve deployments skip it)."""
        if oauth is None:
            return PlainTextResponse("OAuth is not enabled on this server", status_code=404)
        request_id = request.query_params.get("request", "")
        if not request_id or request_id not in oauth.pending_consents:
            return PlainTextResponse("Unknown or already-completed request", status_code=404)
        if request.method == "GET":
            pending = oauth.pending_consents[request_id]
            body = (
                "<!doctype html><title>RAGForge MCP authorization</title>"
                "<p>An MCP client is requesting access to this RAGForge.</p>"
                f"<p>Client: {pending.client_id}<br>Scopes: "
                f"{' '.join(pending.params.scopes or ()) or '(none requested)'}</p>"
                f'<form method="post"><input type="hidden" name="request" value="{request_id}">'
                '<button name="decision" value="approve">Approve</button> '
                '<button name="decision" value="deny">Deny</button></form>'
            )
            return PlainTextResponse(body, media_type="text/html")
        form = await request.form()
        if form.get("decision") != "approve":
            await oauth.deny(request_id)
            return PlainTextResponse("Authorization denied. You can close this tab.")
        location = await oauth.approve(request_id)
        return RedirectResponse(location, status_code=302)

    app.router.add_route("/healthz", health, methods=["GET"])
    app.router.add_route("/consent", consent, methods=["GET", "POST"])
    # The SDK app mounts /mcp already; keep the app's own lifespan intact.
    return PrincipalMiddleware(app)


def _transport_security(settings: Settings, issuer_url: str) -> Any:
    """Host/Origin allow-list. Empty configuration means loopback only."""
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = [h.strip() for h in (settings.mcp_allowed_hosts or "").split(",") if h.strip()]
    origins = [o.strip() for o in (settings.mcp_allowed_origins or "").split(",") if o.strip()]
    if not hosts:
        hosts = [f"127.0.0.1:{settings.mcp_remote_port}", f"localhost:{settings.mcp_remote_port}"]
    if not origins and settings.mcp_oauth_auto_approve:
        origins = [issuer_url]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the remote MCP server (Streamable HTTP) on the configured bind."""
    import uvicorn

    from app.mcp.server import configure_mcp_logging

    configure_mcp_logging()
    settings = get_settings()
    app = create_remote_app(settings=settings)
    logger.info(
        "RAGForge remote MCP listening on http://%s:%s/mcp",
        settings.mcp_remote_host,
        settings.mcp_remote_port,
    )
    uvicorn.run(app, host=settings.mcp_remote_host, port=settings.mcp_remote_port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by hand
    raise SystemExit(main())


__all__ = [
    "AccessDenied",
    "KBAccessPolicy",
    "Principal",
    "PrincipalMiddleware",
    "create_remote_app",
    "create_remote_server",
    "current_principal",
    "main",
    "parse_tokens",
    "resolve_resource_urls",
]
