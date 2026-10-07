"""V10 benchmark-review API tests.

Hermetic: temporary DATA_DIR, in-memory vector store, hashing embeddings, mock
LLM. A real document is indexed through the real upload → chunk → embed → index
path, so the evidence a reviewer reads is genuinely indexed corpus content and
the scratch benchmark references REAL chunk ids.

The benchmark used here is written to the temp DATA_DIR — never the shipped
`benchmarks/answer-quality-automobile-v1.json`, which must stay byte-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

NAVAL = """# Hydrostatics and Stability

The metacentric height GM must be positive for stable equilibrium of a floating body.
Free surface effect reduces the effective metacentric height in a flooded compartment.
A tender ship has a small metacentric height and a long natural roll period.

# Propulsion

Propeller cavitation occurs when the local pressure falls below the vapour pressure of water.
The advance ratio relates the advance per revolution to the propeller diameter.
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    import app.api.deps as deps
    import app.config as config

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing-dev-fallback")
    monkeypatch.setenv("EMBEDDING_MODEL", "char-ngram-hashing-256")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()

    from tests.test_retrieval_integration import InMemoryVectorStore
    from app.services.embeddings.provider import HashingEmbeddingProvider

    store = InMemoryVectorStore()
    embedder = HashingEmbeddingProvider()

    import app.api.routes_build as routes_build
    import app.api.routes_documents as routes_documents
    import app.services.embeddings.provider as provider_module
    import app.services.vector_store.factory as factory_mod

    def fake_store(settings, backend=None):
        return store

    def fake_embedder(settings, expected=None):
        return embedder

    monkeypatch.setattr(factory_mod, "create_vector_store", fake_store)
    monkeypatch.setattr(routes_build, "create_vector_store", fake_store)
    monkeypatch.setattr(routes_documents, "create_vector_store", fake_store)
    monkeypatch.setattr(provider_module, "create_embedding_provider", fake_embedder)
    monkeypatch.setattr(routes_build, "create_embedding_provider", fake_embedder)
    monkeypatch.setattr(routes_documents, "create_embedding_provider", fake_embedder)

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, tmp_path
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()


@pytest.fixture
def client(env):
    return env[0]


def _make_kb(client, name="Naval — V10 review"):
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": name,
            "domain": "Naval Architecture",
            "purpose": "V10 benchmark review tests",
            "target_audience": "students",
            "source_mode": "user_provided",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def _index(client, kb_id):
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/documents/upload",
        files=[("files", ("naval.md", NAVAL.encode("utf-8"), "text/markdown"))],
        data={
            "index": "true",
            "chunker": "section-aware",
            "target_size": "400",
            "overlap": "40",
        },
    )
    assert r.status_code == 201, r.text


@pytest.fixture
def indexed_kb(client):
    kb = _make_kb(client)
    _index(client, kb["id"])
    return kb


def _chunk_ids(kb_id: str) -> list[str]:
    import app.api.deps as deps

    repo = deps.get_repo()
    return [c.id for c in repo.list_chunks(kb_id)]


def _write_benchmark(
    tmp_path: Path,
    kb_id: str,
    chunk_ids: list[str],
    name="scratch-v10-review",
    *,
    chunk_source_kb_id: str | None = None,
) -> str:
    """Write a scratch benchmark whose evidence points at REAL indexed chunks.

    `chunk_source_kb_id` lets a test attach real chunk metadata to a benchmark
    that CLAIMS a different knowledge base (the mismatch case).
    """
    import app.api.deps as deps

    repo = deps.get_repo()
    found = {
        c.id: c
        for c in repo.get_chunks_by_ids(chunk_source_kb_id or kb_id, chunk_ids)
    }
    evidence = []
    for cid in chunk_ids:
        c = found.get(cid)
        assert c is not None, f"chunk {cid} missing"
        evidence.append(
            {
                "chunk_id": cid,
                "document_id": c.document_id,
                "content_hash": c.content_hash,
                "section": c.section,
                "source_title": c.source_title,
                "required": True,
            }
        )
    benchmark = {
        "benchmark": name,
        "version": 1,
        "created_at": "2026-01-01",
        "kb_id": kb_id,
        "human_review": "pending",
        "questions": [
            {
                "question_id": "nav-001",
                "question": "What must the metacentric height GM be for stable equilibrium?",
                "subdomain": "stability",
                "answerability": "answerable",
                "required_evidence": evidence,
                "citation_requirements": {
                    "require_at_least_one_citation": True,
                    "must_cite_all_required_evidence": True,
                    "must_not_cite_irrelevant_chunks": True,
                },
                "expected_grounding_state": "ANY_ACCEPTABLE",
                "abstention_required": False,
            }
        ],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    return str(path)


ALL_VERDICTS = {
    "reference_answer": "ok",
    "key_points": "ok",
    "ambiguity": "ok",
    "answerability": "ok",
    "evidence_sufficiency": "ok",
}


def _author(client, kb_id, path, question_id="nav-001", **overrides):
    body = {
        "benchmark_path": path,
        "author": "ada-reviewer",
        "expected_answer": (
            "The metacentric height GM must be positive for stable equilibrium "
            "of a floating body."
        ),
        "key_points": [
            "metacentric height GM must be positive for stable equilibrium",
            "free surface effect reduces stability in a flooded compartment",
        ],
        "acceptable_answer_elements": ["GM greater than zero"],
        "note": "read the stability chunk",
    }
    body.update(overrides)
    return client.post(
        f"/api/knowledge-bases/{kb_id}/answer-benchmark/questions/{question_id}/ground-truth",
        json=body,
    )


def _review(client, kb_id, path, question_id="nav-001", **overrides):
    body = {
        "benchmark_path": path,
        "reviewer": "ada-reviewer",
        "outcome": "approved",
        "verdicts": dict(ALL_VERDICTS),
        "evidence_checked": [],
    }
    body.update(overrides)
    return client.post(
        f"/api/knowledge-bases/{kb_id}/answer-benchmark/questions/{question_id}/reviews",
        json=body,
    )


class TestReviewPacket:
    def test_unreviewed_benchmark_reports_pending_and_unknown(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/review-packet",
            params={"benchmark_path": path},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["identity"]["name"] == "scratch-v10-review"
        assert body["identity"]["lifecycle"] == "draft"
        assert body["completeness"]["total"] == 1
        assert body["completeness"]["pending"] == 1
        assert body["completeness"]["approved"] == 0
        # missing information must not become zero
        assert body["completeness"]["reference_answer_coverage"]["measured"] is False
        assert body["completeness"]["reference_answer_coverage"]["value"] is None
        assert body["gate"]["approved"] is False
        assert body["questions"][0]["state"] == "pending"
        assert any("no question has been approved" in w for w in body["warnings"])

    def test_benchmark_for_another_kb_is_rejected(self, client, indexed_kb, env):
        other = _make_kb(client, "other kb")
        # Structurally valid benchmark, but it belongs to a DIFFERENT corpus.
        path = _write_benchmark(
            env[1],
            other["id"],
            _chunk_ids(indexed_kb["id"])[:1],
            chunk_source_kb_id=indexed_kb["id"],
        )
        r = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/review-packet",
            params={"benchmark_path": path},
        )
        assert r.status_code == 400
        assert "targets knowledge base" in r.json()["detail"]

    def test_path_traversal_is_rejected(self, client, indexed_kb):
        r = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/review-packet",
            params={"benchmark_path": "../../etc/passwd"},
        )
        assert r.status_code == 400


class TestQuestionDetail:
    def test_evidence_contains_real_chunk_text_and_provenance(self, client, indexed_kb, env):
        chunk_ids = _chunk_ids(indexed_kb["id"])[:1]
        path = _write_benchmark(env[1], indexed_kb["id"], chunk_ids)
        r = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/questions/nav-001",
            params={"benchmark_path": path},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["summary"]["state"] == "pending"
        assert len(body["evidence"]) == 1
        ev = body["evidence"][0]
        assert ev["found"] is True
        assert ev["text"], "the reviewer must be able to read the actual chunk text"
        assert ev["document_id"]
        assert ev["content_hash"]
        assert ev["problems"] == []

    def test_unknown_question_is_404(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/questions/nope",
            params={"benchmark_path": path},
        )
        assert r.status_code == 404


class TestAuthoring:
    def test_authoring_stores_a_reference_answer_and_reports_problems(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(client, indexed_kb["id"], path)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["problems"] == []
        annotation = body["annotation"]
        assert annotation["author"] == "ada-reviewer"
        assert annotation["expected_answer"].startswith("The metacentric height")
        assert len(annotation["key_points"]) == 2
        assert annotation["provenance"][0]["tags"] == ["human_review"]
        assert annotation["provenance"][0]["data"]["reviewer"] == "ada-reviewer"
        # A draft is IN_REVIEW, never approved on its own.
        packet = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/review-packet",
            params={"benchmark_path": path},
        ).json()
        assert packet["questions"][0]["state"] == "in_review"

    def test_authoring_without_evidence_records_a_problem(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(client, indexed_kb["id"], path, evidence_chunk_ids=["chk_does_not_exist"])
        assert r.status_code == 201, r.text
        problems = r.json()["problems"]
        assert any("not in the corpus" in p for p in problems)

    def test_blank_author_is_refused(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(client, indexed_kb["id"], path, author="   ")
        assert r.status_code in (400, 422)

    def test_authoring_for_an_unknown_question_is_404(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(client, indexed_kb["id"], path, question_id="nope")
        assert r.status_code == 404

    def test_history_is_append_only_and_keeps_every_version(self, client, indexed_kb, env):
        chunk_ids = _chunk_ids(indexed_kb["id"])
        path = _write_benchmark(env[1], indexed_kb["id"], chunk_ids[:1])
        first = _author(client, indexed_kb["id"], path).json()["annotation"]
        second = _author(
            client,
            indexed_kb["id"],
            path,
            expected_answer="Corrected reference answer.",
            supersedes=first["annotation_id"],
        ).json()["annotation"]
        history = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/questions/nav-001/ground-truth",
            params={"benchmark_path": path},
        ).json()
        assert len(history) == 2
        assert history[0]["expected_answer"] != history[1]["expected_answer"]
        assert second["annotation_id"] != first["annotation_id"]


class TestReviewRecording:
    def test_approval_requires_every_required_dimension(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        incomplete = {k: v for k, v in ALL_VERDICTS.items() if k != "ambiguity"}
        r = _review(client, indexed_kb["id"], path, verdicts=incomplete)
        assert r.status_code == 400
        assert "ambiguity" in r.json()["detail"]

    def test_unknown_dimension_and_verdict_are_refused(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        assert _review(
            client, indexed_kb["id"], path, verdicts={"made_up": "ok"}
        ).status_code == 400
        assert _review(
            client, indexed_kb["id"], path, verdicts={"ambiguity": "fine"}
        ).status_code == 400

    def test_pending_is_not_a_review_outcome(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        r = _review(client, indexed_kb["id"], path, outcome="pending")
        assert r.status_code == 400
        assert "pending" in r.json()["detail"]

    def test_review_history_is_append_only(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        first = _review(client, indexed_kb["id"], path, outcome="in_review", verdicts={})
        assert first.status_code == 201, first.text
        second = _review(
            client,
            indexed_kb["id"],
            path,
            supersedes=first.json()["review_id"],
        )
        assert second.status_code == 201, second.text
        history = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/questions/nav-001/reviews",
            params={"benchmark_path": path},
        ).json()
        assert len(history) == 2

    def test_ambiguous_outcome_is_recorded_not_approved(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        r = _review(
            client,
            indexed_kb["id"],
            path,
            outcome="ambiguous",
            verdicts={"ambiguity": "ambiguous"},
        )
        assert r.status_code == 201, r.text
        packet = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/review-packet",
            params={"benchmark_path": path},
        ).json()
        assert packet["questions"][0]["state"] == "ambiguous"
        assert packet["completeness"]["ambiguous"] == 1
        assert packet["completeness"]["approved"] == 0


class TestFreezeGate:
    def test_unapproved_benchmark_cannot_freeze_and_explains_why(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/freeze",
            json={"benchmark_path": path, "frozen_by": "ada-reviewer"},
        )
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail["counts"]["pending"] == 1
        assert any("pending" in reason for reason in detail["reasons"])

    def test_authoring_alone_is_not_approval(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/freeze",
            json={"benchmark_path": path, "frozen_by": "ada-reviewer"},
        )
        assert r.status_code == 409
        assert r.json()["detail"]["counts"]["in_review"] == 1

    def test_rejected_question_blocks_freeze(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        _review(client, indexed_kb["id"], path, outcome="rejected", verdicts={"reference_answer": "wrong"})
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/freeze",
            json={"benchmark_path": path, "frozen_by": "ada-reviewer"},
        )
        assert r.status_code == 409
        assert r.json()["detail"]["counts"]["rejected"] == 1

    def test_approved_question_freezes_into_an_immutable_version(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        assert _review(client, indexed_kb["id"], path).status_code == 201

        frozen = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/freeze",
            json={"benchmark_path": path, "frozen_by": "ada-reviewer", "notes": ["first reviewed set"]},
        )
        assert frozen.status_code == 200, frozen.text
        body = frozen.json()
        assert body["version"] == 1
        assert body["benchmark"]["lifecycle"] == "frozen"
        assert body["benchmark"]["human_review"] == "human_reviewed"
        assert body["ground_truth_fingerprint"]
        assert body["artifact_fingerprint"]
        assert body["gate"]["approved"] is True
        assert body["frozen_by"] == "ada-reviewer"

        verify = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/versions/{body['version_id']}/verify"
        )
        assert verify.status_code == 200, verify.text
        assert verify.json()["intact"] is True

        versions = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/versions"
        ).json()
        assert [v["version"] for v in versions] == [1]
        assert versions[0]["scoring_question_count"] == 1

    def test_no_route_can_edit_a_stored_record(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        annotation = _author(client, indexed_kb["id"], path).json()["annotation"]
        base = f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/questions/nav-001"
        assert client.put(f"{base}/ground-truth", json={}).status_code == 405
        assert client.delete(f"{base}/ground-truth").status_code == 405
        assert client.put(f"{base}/reviews", json={}).status_code == 405
        assert annotation["annotation_id"]  # stored append-only


class TestEvaluationFromFrozenVersion:
    def _frozen(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        assert _review(client, indexed_kb["id"], path).status_code == 201
        frozen = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/freeze",
            json={"benchmark_path": path, "frozen_by": "ada-reviewer"},
        )
        assert frozen.status_code == 200, frozen.text
        return frozen.json(), path

    def test_reference_evaluator_unlocks_correctness_on_the_frozen_version(
        self, client, indexed_kb, env
    ):
        version, _path = self._frozen(client, indexed_kb, env)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={
                "benchmark_version_id": version["version_id"],
                "evaluator": "reference",
                "official": True,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["benchmark_version_id"] == version["version_id"]
        assert body["benchmark_artifact_fingerprint"] == version["artifact_fingerprint"]
        assert body["benchmark_lifecycle"] == "frozen"
        assert body["official"] is True
        assert body["evaluator_name"] == "reference-labels"
        assert body["aggregate"]["correctness"]["measured"] is True
        assert body["aggregate"]["correctness"]["value"] is not None
        assert body["aggregate"]["key_point_recall"]["measured"] is True
        assert "correctness" not in body["aggregate"]["unknown_metrics"]

    def test_default_evaluator_on_the_same_version_stays_unknown(
        self, client, indexed_kb, env
    ):
        version, _path = self._frozen(client, indexed_kb, env)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_version_id": version["version_id"]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["evaluator_name"] == "deterministic-evidence"
        assert body["aggregate"]["correctness"]["measured"] is False
        assert body["aggregate"]["reference_answer_similarity"]["measured"] is True

    def test_reference_evaluator_refuses_a_draft_benchmark(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, "evaluator": "reference"},
        )
        assert r.status_code == 400
        assert "human-reviewed benchmark" in r.json()["detail"]

    def test_unknown_frozen_version_is_404(self, client, indexed_kb):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_version_id": "abv_does_not_exist"},
        )
        assert r.status_code == 404
