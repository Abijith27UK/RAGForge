"""V6 API contract tests: retrieval strategies, configuration, runs, evaluation.

Hermetic like the V4/V5 API tests: temporary DATA_DIR, the vector factory stubbed
with a REAL in-memory cosine store, the hashing embedding provider, and no network.
Indexing runs through the real upload → chunk → embed → index path, so the
retrieval results under test are produced from genuinely indexed chunks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

STABILITY = """# Hydrostatics and Stability

The metacentric height GM must be positive for stable equilibrium of a floating body.
Free surface effect reduces the effective metacentric height in a flooded compartment.
A tender ship has a small metacentric height and a long natural roll period.

# Propulsion

Propeller cavitation occurs when the local pressure falls below the vapour pressure of water.
The advance ratio relates the advance per revolution to the propeller diameter.
Wake fraction describes the velocity deficit of water entering the propeller disc.
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    import app.api.deps as deps
    import app.config as config

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing-dev-fallback")
    monkeypatch.setenv("EMBEDDING_MODEL", "char-ngram-hashing-256")
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()

    from tests.test_retrieval_integration import InMemoryVectorStore
    from app.services.embeddings.provider import HashingEmbeddingProvider

    store = InMemoryVectorStore()
    embedder = HashingEmbeddingProvider()

    import app.services.vector_store.factory as factory_mod
    import app.api.routes_build as routes_build
    import app.api.routes_documents as routes_documents
    import app.services.embeddings.provider as provider_module

    def fake_create_vector_store(settings, backend=None):
        return store

    def fake_create_embedder(settings, expected=None):
        return embedder

    monkeypatch.setattr(factory_mod, "create_vector_store", fake_create_vector_store)
    monkeypatch.setattr(routes_build, "create_vector_store", fake_create_vector_store)
    monkeypatch.setattr(routes_documents, "create_vector_store", fake_create_vector_store)
    monkeypatch.setattr(provider_module, "create_embedding_provider", fake_create_embedder)
    monkeypatch.setattr(routes_build, "create_embedding_provider", fake_create_embedder)
    monkeypatch.setattr(routes_documents, "create_embedding_provider", fake_create_embedder)

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, store
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()


@pytest.fixture
def client(env):
    return env[0]


@pytest.fixture
def kb(client):
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": "Naval Architecture — IITM",
            "domain": "Naval Architecture",
            "purpose": "Lecture corpus for stability and propulsion",
            "target_audience": "Undergraduate students",
            "source_mode": "user_provided",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def indexed_kb(client, kb):
    """Upload + index one real document through the real pipeline."""
    kb_id = kb["id"]
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/documents/upload",
        files=[("files", ("naval-notes.md", STABILITY.encode("utf-8"), "text/markdown"))],
        data={"index": "true", "chunker": "section-aware", "target_size": "400", "overlap": "40"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["indexed"] is True, r.text
    return kb


# ---------------------------------------------------------------------------
# Strategy registry surface
# ---------------------------------------------------------------------------

def test_retrieval_strategies_endpoint_lists_the_registry(client, kb):
    r = client.get(f"/api/knowledge-bases/{kb['id']}/retrieval-strategies")
    assert r.status_code == 200, r.text
    specs = {s["name"]: s for s in r.json()}
    assert {"dense", "bm25", "hybrid", "hybrid_reranked"} <= set(specs)
    assert specs["dense"]["is_baseline"] is True
    assert specs["bm25"]["requires_repo"] is True
    assert specs["bm25"]["needs_embedding"] is False
    assert specs["dense"]["aliases"] == ["qdrant-dense"]
    assert specs["hybrid_reranked"]["notes"].startswith("Falls back")


def test_retrieval_strategies_requires_an_existing_kb(client):
    r = client.get("/api/knowledge-bases/kb_missing/retrieval-strategies")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Configuration persistence
# ---------------------------------------------------------------------------

def test_retrieval_config_defaults_are_labelled_as_defaults(client, kb):
    r = client.get(f"/api/knowledge-bases/{kb['id']}/retrieval-config")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["params"]["strategy"] == "dense"
    assert body["params"]["top_k"] == 5
    assert "defaults" in body["note"]


def test_retrieval_config_round_trip(client, kb):
    kb_id = kb["id"]
    payload = {
        "params": {
            "strategy": "hybrid",
            "top_k": 7,
            "dense_weight": 0.5,
            "bm25_weight": 0.5,
            "fusion": "rrf",
            "rrf_k": 42,
            "bm25_k1": 1.5,
            "bm25_b": 0.6,
        },
        "note": "hybrid RRF exploration",
    }
    r = client.put(f"/api/knowledge-bases/{kb_id}/retrieval-config", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["note"] == "hybrid RRF exploration"

    stored = client.get(f"/api/knowledge-bases/{kb_id}/retrieval-config").json()
    assert stored["params"]["strategy"] == "hybrid"
    assert stored["params"]["fusion"] == "rrf"
    assert stored["params"]["rrf_k"] == 42
    assert stored["params"]["bm25_k1"] == 1.5
    assert stored["note"] == "hybrid RRF exploration"


def test_invalid_retrieval_config_is_rejected(client, kb):
    r = client.put(
        f"/api/knowledge-bases/{kb['id']}/retrieval-config",
        json={"params": {"strategy": "hybrid", "dense_weight": 0.0, "bm25_weight": 0.0}},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Retrieval runs
# ---------------------------------------------------------------------------

def test_dense_retrieve_keeps_the_v5_response_shape(indexed_kb, client):
    kb_id = indexed_kb["id"]
    r = client.post(f"/api/knowledge-bases/{kb_id}/retrieve", json={"query": "metacentric height", "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["retrieval_backend"] == "qdrant-dense"
    assert body["strategy"] == "dense"
    assert body["results"], "the indexed corpus must be retrievable"
    assert body["retrieval_run_id"], "every run must be recorded for observability"
    first = body["results"][0]
    assert first["rank"] == 1
    assert first["retrieval_method"] == "dense"
    assert first["provenance"]["content_hash"]
    assert first["score_breakdown"]["dense_score"] is not None
    assert first["why"]


def test_bm25_retrieve_returns_provenance_and_ignores_the_vector_store(indexed_kb, client):
    kb_id = indexed_kb["id"]
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/retrieve",
        json={"query": "propeller cavitation", "top_k": 3, "strategy": "bm25"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["strategy"] == "bm25"
    assert body["retrieval_backend"] == "sqlite-bm25"
    assert body["results"]
    assert any("cavitation" in res["text"] for res in body["results"])
    assert body["results"][0]["score_breakdown"]["lexical_score"] > 0
    # Lexical-only retrieval must not claim an embedding model was involved.
    assert any("no embedding model was loaded" in note for note in body["notes"])


def test_hybrid_retrieve_reports_fusion_configuration(indexed_kb, client):
    kb_id = indexed_kb["id"]
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/retrieve",
        json={
            "query": "metacentric height stability",
            "strategy": "hybrid",
            "config": {"top_k": 3, "fusion": "weighted", "candidate_k": 8},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["strategy"] == "hybrid"
    assert body["params"]["dense_weight"] == 0.65
    assert body["params"]["bm25_weight"] == 0.35
    assert any("weighted fusion" in note for note in body["notes"])
    assert any(res["score_breakdown"]["fused_score"] is not None for res in body["results"])


def test_hybrid_reranked_falls_back_honestly_when_the_model_is_unavailable(
    indexed_kb, client, monkeypatch
):
    """A requested-but-unavailable reranker must be REPORTED, never faked.

    The reranker is forced unavailable instead of letting the test download a real
    cross-encoder: the property under test is the honest fallback contract, and the
    "applied" path is covered deterministically in test_retrieval_v6.py.
    """
    import app.services.retrieval.hybrid as hybrid_module
    from app.services.retrieval.rerank import Reranker, RerankerStatus

    class Unavailable(Reranker):
        name = "cross-encoder"
        model = "cross-encoder/not-cached"

        def availability(self):
            return (False, "model not cached and offline")

        def rerank(self, query, items):  # pragma: no cover - must never be called
            raise AssertionError("rerank must not run when the reranker is unavailable")

    monkeypatch.setattr(hybrid_module, "get_reranker", lambda name, model: Unavailable())
    kb_id = indexed_kb["id"]
    baseline = client.post(
        f"/api/knowledge-bases/{kb_id}/retrieve",
        json={"query": "advance ratio", "strategy": "hybrid", "config": {"top_k": 3}},
    ).json()
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/retrieve",
        json={"query": "advance ratio", "strategy": "hybrid_reranked", "config": {"top_k": 3}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["strategy"] == "hybrid_reranked"
    reranker = body["reranker"]
    assert reranker["requested"] == "cross-encoder"
    assert reranker["status"] == RerankerStatus.UNAVAILABLE_FALLBACK.value
    assert reranker["reranked"] == 0
    assert any("UNAVAILABLE" in note for note in body["notes"])
    # The results are the unreranked hybrid fusion, not a silently different list.
    assert [res["chunk_id"] for res in body["results"]] == [
        res["chunk_id"] for res in baseline["results"]
    ]



def test_unknown_strategy_is_a_client_error(indexed_kb, client):
    r = client.post(
        f"/api/knowledge-bases/{indexed_kb['id']}/retrieve",
        json={"query": "x", "strategy": "teleport"},
    )
    assert r.status_code in (400, 422)


def test_retrieve_on_a_missing_kb_is_404(client):
    r = client.post("/api/knowledge-bases/kb_missing/retrieve", json={"query": "x"})
    assert r.status_code == 404


def test_runs_are_listed_and_retrievable_by_id(indexed_kb, client):
    kb_id = indexed_kb["id"]
    first = client.post(
        f"/api/knowledge-bases/{kb_id}/retrieve",
        json={"query": "metacentric height", "strategy": "bm25", "top_k": 2},
    ).json()
    run_id = first["retrieval_run_id"]

    listed = client.get(f"/api/knowledge-bases/{kb_id}/retrieval-runs").json()
    assert any(run["id"] == run_id for run in listed)
    summary = next(run for run in listed if run["id"] == run_id)
    assert summary["strategy"] == "bm25"
    assert summary["result_count"] == len(first["results"])
    assert summary["total_ms"] is not None

    full = client.get(f"/api/knowledge-bases/{kb_id}/retrieval-runs/{run_id}")
    assert full.status_code == 200, full.text
    body = full.json()
    assert body["query"] == "metacentric height"
    assert body["params"]["top_k"] == 2
    assert body["config_fingerprint"]
    assert body["candidate_counts"].get("lexical") is not None
    assert body["reranker_status"] == "not_requested"

    missing = client.get(f"/api/knowledge-bases/{kb_id}/retrieval-runs/rr_missing")
    assert missing.status_code == 404


def test_bm25_index_status_endpoint_reports_staleness(indexed_kb, client):
    kb_id = indexed_kb["id"]
    before = client.get(f"/api/knowledge-bases/{kb_id}/bm25-index").json()
    assert before["source"] == "none"
    assert before["stale"] is None

    client.post(
        f"/api/knowledge-bases/{kb_id}/retrieve",
        json={"query": "cavitation", "strategy": "bm25"},
    )
    after = client.get(f"/api/knowledge-bases/{kb_id}/bm25-index").json()
    assert after["stale"] is False
    assert after["documents"] and after["documents"] > 0
    assert after["staleness_check"] == "revision"

    # Re-indexing the document changes the corpus -> the index reports itself stale.
    docs = client.get(f"/api/knowledge-bases/{kb_id}/documents").json()
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/documents/{docs[0]['id']}/index", json={}
    )
    assert r.status_code == 200, r.text
    stale = client.get(f"/api/knowledge-bases/{kb_id}/bm25-index").json()
    assert stale["stale"] is True

    # ... and the next query rebuilds it and reports that it did.
    rebuilt = client.post(
        f"/api/knowledge-bases/{kb_id}/retrieve",
        json={"query": "cavitation", "strategy": "bm25"},
    ).json()
    assert any("rebuilt" in note for note in rebuilt["notes"])
    assert client.get(f"/api/knowledge-bases/{kb_id}/bm25-index").json()["stale"] is False


# ---------------------------------------------------------------------------
# Evaluation over a selected strategy (real ground truth only)
# ---------------------------------------------------------------------------

def test_evaluation_scores_the_selected_strategy_with_real_ground_truth(indexed_kb, client):
    kb_id = indexed_kb["id"]
    chunks = client.get(f"/api/knowledge-bases/{kb_id}/chunks?limit=100").json()
    cavitation = next(c for c in chunks if "cavitation" in c["text"])
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions",
        json={
            "question": "When does propeller cavitation occur?",
            "expected_chunk_ids": [cavitation["id"]],
            "author": "test",
            "notes": "ground truth authored for the V6 API test",
        },
    )
    assert r.status_code == 201, r.text

    run = client.post(
        f"/api/knowledge-bases/{kb_id}/evaluate",
        json={"top_k": 3, "strategy": "bm25", "label": "v6 bm25 k=3"},
    )
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["retrieval_strategy"] == "bm25"
    assert body["retrieval_params"]["strategy"] == "bm25"
    assert body["aggregate"]["questions_with_explicit_gt"] == 1
    assert body["aggregate"]["recall_at_k"] is not None
    assert body["aggregate"]["mrr"] is not None
    assert body["retrieval_backend"] == "sqlite-bm25"
    assert body["per_question"][0]["recall_at_k"] is not None


def test_evaluation_without_ground_truth_still_reports_nothing(indexed_kb, client):
    kb_id = indexed_kb["id"]
    run = client.post(f"/api/knowledge-bases/{kb_id}/evaluate", json={"top_k": 3})
    assert run.status_code == 400, run.text
    assert "No evaluation questions" in run.json()["detail"]


def test_evaluation_is_reproducible_for_the_same_strategy(indexed_kb, client):
    kb_id = indexed_kb["id"]
    chunks = client.get(f"/api/knowledge-bases/{kb_id}/chunks?limit=100").json()
    target = next(c for c in chunks if "metacentric" in c["text"])
    client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions",
        json={"question": "metacentric height", "expected_chunk_ids": [target["id"]]},
    )
    payload = {"top_k": 3, "strategy": "dense", "label": "reproducibility check"}
    first = client.post(f"/api/knowledge-bases/{kb_id}/evaluate", json=payload).json()
    second = client.post(f"/api/knowledge-bases/{kb_id}/evaluate", json=payload).json()
    assert first["aggregate"]["recall_at_k"] == second["aggregate"]["recall_at_k"]
    assert first["aggregate"]["mrr"] == second["aggregate"]["mrr"]
    assert first["aggregate"]["ndcg"] == second["aggregate"]["ndcg"]
