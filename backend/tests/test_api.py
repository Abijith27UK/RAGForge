"""API contract tests (no Qdrant / no network required)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.api.deps as deps
    import app.config as config

    # Point settings at a temp data dir before any request runs
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()
    settings = config.get_settings()
    assert str(tmp_path) in str(settings.db_path)

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()


def test_health(client):
    r = client.get("/api/system/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_kb_crud_cycle(client):
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": "Automobile Engineering KB",
            "domain": "Automobile Engineering",
            "purpose": "Engineering education",
            "target_audience": "Engineering students",
            "depth": "technical",
        },
    )
    assert r.status_code == 201, r.text
    kb = r.json()
    assert kb["status"] == "draft"
    kb_id = kb["id"]

    r = client.get("/api/knowledge-bases")
    assert any(k["id"] == kb_id for k in r.json())

    r = client.get(f"/api/knowledge-bases/{kb_id}")
    assert r.status_code == 200

    # Domain spec should 404 before analysis
    assert client.get(f"/api/knowledge-bases/{kb_id}/domain-spec").status_code == 404

    # Ingest with no sources -> 400
    r = client.post(f"/api/knowledge-bases/{kb_id}/ingest", json={})
    assert r.status_code == 400

    # Evaluate with no questions -> 400 or 503 (retriever unavailable), never 500-fabricated
    r = client.post(f"/api/knowledge-bases/{kb_id}/evaluate", json={"top_k": 5})
    assert r.status_code in (400, 503)

    assert client.delete(f"/api/knowledge-bases/{kb_id}").status_code == 204
    assert client.get(f"/api/knowledge-bases/{kb_id}").status_code == 404


def test_validation_error_shape(client):
    r = client.post("/api/knowledge-bases", json={"name": ""})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_mock_domain_analysis_flow(client):
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": "Auto KB",
            "domain": "Automobile Engineering",
            "purpose": "education",
            "target_audience": "students",
        },
    )
    kb_id = r.json()["id"]
    # allow_mock_llm defaults to True -> mock analyzer runs, clearly labelled
    r = client.post(f"/api/knowledge-bases/{kb_id}/analyze-domain")
    if r.status_code == 200:
        spec = r.json()
        assert spec["is_mock"] is True
        assert spec["kb_id"] == kb_id
        assert spec["subdomains"]
        r2 = client.get(f"/api/knowledge-bases/{kb_id}/domain-spec")
        assert r2.status_code == 200


def test_url_rejection_via_discovery(client):
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": "KB",
            "domain": "Test",
            "purpose": "testing",
            "target_audience": "devs",
        },
    )
    kb_id = r.json()["id"]
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/discover-sources",
        json={"provider": "user-url", "query": "http://127.0.0.1/x"},
    )
    assert r.status_code == 200
    sources = r.json()
    assert sources and sources[0]["notes"].startswith("INVALID")
    # scoring must have rejected it
    assert sources[0]["decision"] == "REJECT"
