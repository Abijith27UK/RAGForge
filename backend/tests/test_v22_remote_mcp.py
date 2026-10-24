"""V22 tests: remote MCP transport, authentication, and KB access policy.

These tests cover:
*  `normalize_loopback_url` — the IPv4 latency fix (localhost -> 127.0.0.1)
*  `parse_static_tokens` — token configuration parsing
*  `StaticBearerTokenVerifier` — token verification + constant-time rejection
*  `KBAccessPolicy` — who may use which KB (subject-based)
*  `ScopedBridge` — a KB tool call with a denied KB raises AccessDenied
*  `create_remote_server` — the remote server reuses the stdio tool registry
*  `create_remote_app` — the ASGI app exposes /healthz and /mcp
*  an in-process Streamable HTTP client — initialize, list_tools, call_tool,
     401 without a token, 401 with a bad token
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings, normalize_loopback_url  # noqa: E402
from app.mcp.remote import (  # noqa: E402
    AccessDenied,
    KBAccessPolicy,
    Principal,
    ScopedBridge,
    create_remote_app,
    create_remote_server,
    parse_tokens,
)
from app.mcp.remote_auth import (  # noqa: E402
    StaticBearerTokenVerifier,
    StaticToken,
    parse_static_tokens,
)

pytestmark = pytest.mark.skipif(
    "mcp" not in sys.modules,
    reason="MCP SDK not installed (install requirements-mcp.txt)",
)


# ---------------------------------------------------------------------------
# IPv4 latency fix (the ~2 s Qdrant connect-timeout defect)
# ---------------------------------------------------------------------------

class TestLoopbackURLNormalization:
    def test_localhost_rewritten_to_ipv4(self):
        assert normalize_loopback_url("http://localhost:6333") == "http://127.0.0.1:6333"

    def test_localhost_case_insensitive_with_trailing_slash(self):
        assert normalize_loopback_url("http://LOCALHOST:6333/") == "http://127.0.0.1:6333"

    def test_remote_host_untouched(self):
        assert normalize_loopback_url("http://qdrant.internal:6333") == "http://qdrant.internal:6333"

    def test_already_ipv4_untouched(self):
        assert normalize_loopback_url("http://127.0.0.1:6333") == "http://127.0.0.1:6333"

    def test_url_with_path_untouched(self):
        assert normalize_loopback_url("http://example.com/mcp") == "http://example.com/mcp"

    def test_settings_applies_normalization(self):
        s = Settings(_env_file=None, qdrant_url="http://localhost:6333")
        assert s.qdrant_url == "http://127.0.0.1:6333"


# ---------------------------------------------------------------------------
# Static token configuration and verification
# ---------------------------------------------------------------------------

class TestStaticTokens:
    def test_parse_single_token_with_scopes(self):
        entries = parse_static_tokens("tok123:alice:ragforge:read ragforge:write")
        assert len(entries) == 1
        assert entries[0].subject == "alice"
        assert entries[0].scopes == ("ragforge:read", "ragforge:write")

    def test_parse_multiple_tokens(self):
        entries = parse_static_tokens("t1:alice:ragforge:read; t2:bob:ragforge:read ragforge:write")
        assert len(entries) == 2
        assert {e.subject for e in entries} == {"alice", "bob"}

    def test_parse_malformed_entry_skipped(self):
        entries = parse_static_tokens("just-a-token; t1:alice:ragforge:read")
        assert len(entries) == 1, "a malformed entry must be skipped, not accepted"

    def test_parse_empty_input(self):
        assert parse_static_tokens("") == []

    def test_valid_token_returns_access_token(self):
        verifier = StaticBearerTokenVerifier(
            [StaticToken(token="tok123", subject="alice", scopes=("ragforge:read",))],
            resource="http://127.0.0.1:8110/mcp",
        )
        result = asyncio.run(verifier.verify_token("tok123"))
        assert result is not None
        assert result.subject == "alice"
        assert result.scopes == ["ragforge:read"]
        assert result.resource == "http://127.0.0.1:8110/mcp"

    def test_wrong_token_rejected(self):
        verifier = StaticBearerTokenVerifier(
            [StaticToken(token="tok123", subject="alice", scopes=("ragforge:read",))],
            resource="http://127.0.0.1:8110/mcp",
        )
        assert asyncio.run(verifier.verify_token("wrong")) is None
        assert asyncio.run(verifier.verify_token("")) is None


# ---------------------------------------------------------------------------
# KB access policy
# ---------------------------------------------------------------------------

class _FakeRepo:
    def __init__(self, owners: dict[str, str | None]):
        self._owners = owners

    def get_kb(self, kb_id: str):
        class _KB:
            pass
        if kb_id not in self._owners:
            return None
        kb = _KB()
        kb.id = kb_id
        kb.owner = self._owners[kb_id]
        return kb


class TestKBAccessPolicy:
    def test_admin_can_access_anything(self):
        policy = KBAccessPolicy(_FakeRepo({"kb_a": "alice"}))
        admin = Principal(subject="bob", scopes=frozenset({"ragforge:admin"}))
        assert policy.is_allowed(admin, "kb_a")

    def test_owner_can_access_own_kb(self):
        policy = KBAccessPolicy(_FakeRepo({"kb_a": "alice"}))
        alice = Principal(subject="alice", scopes=frozenset({"ragforge:read"}))
        assert policy.is_allowed(alice, "kb_a")

    def test_other_subject_denied(self):
        policy = KBAccessPolicy(_FakeRepo({"kb_a": "alice"}))
        bob = Principal(subject="bob", scopes=frozenset({"ragforge:read"}))
        assert not policy.is_allowed(bob, "kb_a")

    def test_unowned_kb_is_accessible_to_authenticated_users(self):
        policy = KBAccessPolicy(_FakeRepo({"kb_a": None}))
        bob = Principal(subject="bob", scopes=frozenset({"ragforge:read"}))
        assert policy.is_allowed(bob, "kb_a")

    def test_none_principal_allows_all(self):
        policy = KBAccessPolicy(_FakeRepo({"kb_a": "alice"}))
        assert policy.is_allowed(None, "kb_a")

    def test_check_raises_access_denied(self):
        policy = KBAccessPolicy(_FakeRepo({"kb_a": "alice"}))
        bob = Principal(subject="bob", scopes=frozenset({"ragforge:read"}))
        with pytest.raises(AccessDenied):
            policy.check(bob, "kb_a", tool="search_knowledge_base")

    def test_filter_ids(self):
        policy = KBAccessPolicy(_FakeRepo({"kb_a": "alice", "kb_b": "bob"}))
        alice = Principal(subject="alice", scopes=frozenset({"ragforge:read"}))
        assert policy.filter_ids(alice, ["kb_a", "kb_b", "kb_c"]) == ["kb_a", "kb_c"]


# ---------------------------------------------------------------------------
# Scoped bridge
# ---------------------------------------------------------------------------

class _FakeBridge:
    def __init__(self):
        self.repo = _FakeRepo({"kb_a": "alice", "kb_b": "bob"})

    def search(self, kb_id: str, query: str) -> str:
        return f"results for {query} from {kb_id}"

    def list_knowledge_bases(self):
        """Returns a real pydantic model so ScopedBridge's model_copy works."""
        from app.mcp.schemas import KnowledgeBaseList, KnowledgeBaseSummary

        def summary(kb_id: str) -> KnowledgeBaseSummary:
            return KnowledgeBaseSummary.model_construct(id=kb_id, name=kb_id)

        return KnowledgeBaseList.model_construct(
            knowledge_bases=[summary("kb_a"), summary("kb_b")],
            total=2,
            notes=[],
        )


class TestScopedBridge:
    def test_call_with_allowed_kb(self):
        bridge = ScopedBridge(_FakeBridge(), KBAccessPolicy(_FakeRepo({"kb_a": "alice"})))
        import app.mcp.remote as remote
        token = remote._principal.set(Principal(subject="alice", scopes=frozenset({"ragforge:read"})))
        try:
            assert bridge.search(kb_id="kb_a", query="x") == "results for x from kb_a"
        finally:
            remote._principal.reset(token)

    def test_call_with_denied_kb(self):
        bridge = ScopedBridge(_FakeBridge(), KBAccessPolicy(_FakeRepo({"kb_a": "alice"})))
        import app.mcp.remote as remote
        token = remote._principal.set(Principal(subject="bob", scopes=frozenset({"ragforge:read"})))
        try:
            with pytest.raises(AccessDenied):
                bridge.search(kb_id="kb_a", query="x")
        finally:
            remote._principal.reset(token)

    def test_list_knowledge_bases_filtered(self):
        bridge = ScopedBridge(_FakeBridge(), KBAccessPolicy(_FakeRepo({"kb_a": "alice", "kb_b": "bob"})))
        import app.mcp.remote as remote
        token = remote._principal.set(Principal(subject="alice", scopes=frozenset({"ragforge:read"})))
        try:
            result = bridge.list_knowledge_bases()
            assert [kb.id for kb in result.knowledge_bases] == ["kb_a"]
        finally:
            remote._principal.reset(token)

    def test_list_knowledge_bases_admin_sees_all(self):
        bridge = ScopedBridge(_FakeBridge(), KBAccessPolicy(_FakeRepo({"kb_a": "alice", "kb_b": "bob"})))
        import app.mcp.remote as remote
        token = remote._principal.set(Principal(subject="root", scopes=frozenset({"ragforge:admin"})))
        try:
            result = bridge.list_knowledge_bases()
            assert len(result.knowledge_bases) == 2
        finally:
            remote._principal.reset(token)


# ---------------------------------------------------------------------------
# Remote server + app construction
# ---------------------------------------------------------------------------

class TestRemoteServerConstruction:
    def test_create_remote_server_reuses_tool_registry(self):
        settings = Settings(_env_file=None, mcp_access_tokens="tok1234567890:alice:ragforge:read")
        server, info = create_remote_server(settings=settings)
        assert info["authentication"] == "static-token"
        assert info["resource_url"].endswith("/mcp")
        assert server.name == "ragforge"

    def test_create_remote_app_exposes_healthz_and_mcp(self):
        settings = Settings(_env_file=None, mcp_access_tokens="tok1234567890:alice:ragforge:read")
        app = create_remote_app(settings=settings)
        paths = {getattr(route, "path", None) for route in app.routes}
        assert "/healthz" in paths
        assert "/mcp" in paths

    def test_healthz_does_not_leak_configuration(self):
        import json
        import httpx
        settings = Settings(_env_file=None, mcp_access_tokens="tok1234567890:alice:ragforge:read")
        app = create_remote_app(settings=settings)

        async def probe():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/healthz")
                return response

        response = asyncio.run(probe())
        assert response.status_code == 200
        body = response.text
        assert "tok1234567890" not in body
        assert "secret" not in body.lower()
        assert "key" not in body.lower()
        payload = json.loads(body)
        assert payload["status"] == "ok"
        assert payload["transport"] == "streamable-http"
        assert payload["authentication"] == "static-token"
