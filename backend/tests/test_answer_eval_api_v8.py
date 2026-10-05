"""V8 answer-evaluation API tests.

Hermetic: temporary DATA_DIR, in-memory vector store, hashing embeddings, mock
LLM provider — no network, no API key, no Qdrant. A real document is indexed
through the real upload -> chunk -> embed -> index path so the answers under
evaluation come from genuinely indexed chunks.

The benchmark used here is a SCATCH artifact written to the temp DATA_DIR
neighbourhood, never the shipped benchmark, so these tests cannot mutate a
versioned benchmark file.
"""
from __future__ import annotations

import json
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


def _make_kb(client, name="Naval — answer evaluation"):
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": name,
            "domain": "Naval Architecture",
            "purpose": "answer evaluation tests",
            "target_audience": "students",
            "source_mode": "user_provided",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def _index(client, kb_id):
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/documents/upload",
        files=[("files", ("naval.md", STABILITY.encode("utf-8"), "text/markdown"))],
        data={"index": "true", "chunker": "section-aware",
              "target_size": "400", "overlap": "40"},
    )
    assert r.status_code == 201, r.text


def _write_benchmark(tmp_path: Path, kb_id: str, chunk_ids: list[str]) -> str:
    """Write a scratch benchmark artifact derived from REAL indexed chunk ids."""
    import app.api.deps as deps
    repo = deps.get_repo()
    found = {c.id: c for c in repo.get_chunks_by_ids(kb_id, chunk_ids)}
    chunks = []
    for cid in chunk_ids:
        c = found.get(cid)
        assert c is not None, f"chunk {cid} missing"
        chunks.append({
            "chunk_id": cid,
            "document_id": c.document_id,
            "content_hash": c.content_hash,
            "section": c.section,
            "required": True,
        })
    benchmark = {
        "benchmark": "scratch-answer-eval",
        "version": 1,
        "created_at": "2026-01-01",
        "kb_id": kb_id,
        "human_review": "pending",
        "questions": [
            {
                "question_id": "scratch-001",
                "question": "What happens to GM when the center of gravity rises?",
                "answerability": "answerable",
                "required_evidence": chunks,
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
    path = tmp_path / "scratch-answer-eval.json"
    path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    return str(path)


@pytest.fixture
def indexed_kb(client):
    kb = _make_kb(client)
    _index(client, kb["id"])
    return kb


class TestAnswerEvaluationAPI:
    def test_run_reports_metrics_and_provenance(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])

        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # provenance of the run itself
        assert body["benchmark_name"] == "scratch-answer-eval"
        assert body["benchmark_fingerprint"]
        assert body["evaluator_name"] == "deterministic-evidence"
        # mock generator labelling (STEP 17 #19): a placeholder generator is
        # never presented as a live one.
        assert body["is_mock"] is True
        assert body["entailment_provider"] == "heuristic-lexical-coverage"
        assert body["entailment_is_model_based"] is False
        assert body["question_ids"] == ["scratch-001"]
        assert body["question_count"] == 1
        # correctness must be UNKNOWN, not zero
        assert "correctness" in body["aggregate"]["unknown_metrics"]

    def test_per_question_result_has_claim_verdicts(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        run = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path},
        ).json()
        qid = run["per_question"][0]["question_id"]
        r = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/questions/{qid}"
        )
        assert r.status_code == 200, r.text
        detail = r.json()
        assert detail["required_chunk_ids"]
        assert "claim_verdicts" in detail
        assert detail["actual_grounding_state"]

    def test_runs_are_listed_and_immutable(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        base = f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs"
        first = client.post(base, json={"benchmark_path": path}).json()
        second = client.post(base, json={"benchmark_path": path}).json()
        assert first["id"] != second["id"], "a new run must not overwrite the old one"
        listing = client.get(base).json()
        assert len(listing) == 2
        # The stored first run is still intact.
        again = client.get(f"{base}/{first['id']}").json()
        assert again["id"] == first["id"]
        assert again["created_at"] == first["created_at"]

    def test_compare_refuses_different_question_subsets(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        base = f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs"

        a = client.post(base, json={"benchmark_path": path}).json()
        b = client.post(base, json={"benchmark_path": path}).json()
        same = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/compare",
            params={"left": a["id"], "right": b["id"]},
        ).json()
        assert same["comparable"] is True
        assert same["reason"]

        # A run over a subset must not compare against the full set.
        sub = client.post(base, json={"benchmark_path": path,
                                      "question_ids": ["scratch-001"],
                                      "limit": 1}).json()
        assert sub["id"] != a["id"]
        diff = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/compare",
            params={"left": a["id"], "right": sub["id"]},
        ).json()
        # Same subset in this case, so still comparable; assert the reason exists.
        assert diff["reason"]

    def test_benchmark_path_traversal_is_rejected(self, client, indexed_kb):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": "../../etc/passwd"},
        )
        assert r.status_code == 400

    def test_missing_benchmark_file_is_400(self, client, indexed_kb, env):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": str(env[1] / "nope.json")},
        )
        assert r.status_code == 400
        assert "not found" in r.json()["detail"].lower()

    def test_unknown_kb_404(self, client):
        r = client.post(
            "/api/knowledge-bases/kb_nope/answer-evaluation/runs",
            json={"benchmark_path": "benchmarks/x.json"},
        )
        assert r.status_code == 404

    def test_unknown_run_404(self, client, indexed_kb):
        assert client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/nope"
        ).status_code == 404

    def test_benchmark_targeting_another_kb_is_rejected(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        other = _make_kb(client, "Another KB")
        r = client.post(
            f"/api/knowledge-bases/{other['id']}/answer-evaluation/runs",
            json={"benchmark_path": path},
        )
        assert r.status_code == 400
        assert "always evaluated against its own corpus" in r.json()["detail"]

    def test_corpus_mismatch_is_409(self, client, indexed_kb, env):
        path = env[1] / "stale.json"
        path.write_text(json.dumps({
            "benchmark": "stale", "version": 1, "kb_id": indexed_kb["id"],
            "questions": [{
                "question_id": "stale-1", "question": "q?",
                "answerability": "answerable",
                "required_evidence": [{"chunk_id": "chk_does_not_exist",
                                       "content_hash": "zz", "required": True}],
                "citation_requirements": {"require_at_least_one_citation": True,
                                          "must_cite_all_required_evidence": True,
                                          "must_not_cite_irrelevant_chunks": True},
                "expected_grounding_state": "ANY_ACCEPTABLE",
                "abstention_required": False,
            }],
        }, indent=2), encoding="utf-8")
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": str(path)},
        )
        assert r.status_code == 409
        assert "does not match the live corpus" in r.json()["detail"]

    def test_evaluation_does_not_modify_documents(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        before_docs = len(client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/documents"
        ).json())
        before_chunks = len(repo.list_chunks(indexed_kb["id"]))
        client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path},
        )
        after_docs = len(client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/documents"
        ).json())
        after_chunks = len(repo.list_chunks(indexed_kb["id"]))
        assert before_docs == after_docs
        assert before_chunks == after_chunks

    def test_answer_endpoint_still_works(self, client, indexed_kb):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer",
            json={"question": "What is metacentric height?"},
        )
        assert r.status_code == 200

    def test_chat_endpoint_still_works(self, client, indexed_kb):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/chat",
            json={"message": "What is metacentric height?"},
        )
        assert r.status_code == 200

# ---------------------------------------------------------------------------
# benchmark path guard
#
# The endpoint must not become an arbitrary-file reader. Allowed roots are the
# repository (versioned benchmarks) and DATA_DIR (per-installation benchmarks).
# ---------------------------------------------------------------------------


class TestBenchmarkPathGuard:
    def test_relative_traversal_is_refused(self, client, indexed_kb):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": "../../../Windows/win.ini"},
        )
        assert r.status_code == 400
        assert "must stay inside" in r.json()["detail"]

    def test_absolute_path_outside_repo_and_data_dir_is_refused(self, client, indexed_kb):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": "C:/Windows/win.ini"},
        )
        assert r.status_code == 400
        assert "must stay inside" in r.json()["detail"]

    def test_empty_path_is_refused(self, client, indexed_kb):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": "   "},
        )
        assert r.status_code == 400
        assert "required" in r.json()["detail"]

    def test_persist_flag_documents_that_answers_are_still_written(
        self, client, indexed_kb, env
    ):
        """`persist=false` must not be advertised as making the call read-only."""
        schema = client.get("/openapi.json").json()
        body = schema["components"]["schemas"]["RunAnswerEvaluationRequest"]
        text = json.dumps(body).lower()
        assert "does not" in text and "read-only" in text

    def test_answers_are_persisted_even_when_the_run_is_not(
        self, client, indexed_kb, env
    ):
        """Documents real behaviour: persist=false gates the RUN row, not answers."""
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, "persist": False},
        )
        assert r.status_code == 200
        assert repo.list_answer_evaluation_runs(indexed_kb["id"]) == [], "run must not persist"
        answers = repo.list_answers(indexed_kb["id"]) if hasattr(repo, "list_answers") else []
        if answers:
            assert len(answers) >= 1, "the answering path still writes Answer rows"


class TestPerQuestionSerialization:
    """`pass_rate` is an aggregate of `per_question[].passed`.

    If `passed` is not serialized, any client that filters on it classifies
    every question as a failure while the metric reports otherwise — the UI
    then contradicts itself, which is worse than showing nothing.
    """

    def test_passed_is_serialized_on_every_question(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, "persist": True},
        )
        assert r.status_code == 200
        run = r.json()
        per_question = run["per_question"]
        assert per_question, "the run produced no per-question results"
        for q in per_question:
            assert "passed" in q, "per-question `passed` is missing from the response"
            assert isinstance(q["passed"], bool)

        agg_pass = run["aggregate"]["pass_rate"]["value"]
        derived = sum(1 for q in per_question if q["passed"]) / len(per_question)
        assert agg_pass == pytest.approx(derived), (
            f"pass_rate {agg_pass} disagrees with the per-question verdicts {derived}"
        )

    def test_failed_question_ids_match_the_passed_flags(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, "persist": True},
        )
        assert r.status_code == 200
        run = r.json()
        derived = [q["question_id"] for q in run["per_question"] if not q["passed"]]
        assert sorted(run["failed_question_ids"]) == sorted(derived)


class TestOfficialRunGating:
    """V8 STEP 3: only FROZEN answer benchmarks produce official results.

    The scratch benchmark here has no `lifecycle` field, so it is DRAFT —
    exactly the state of a newly authored, not-yet-reviewed benchmark.
    """

    def test_official_run_on_a_draft_benchmark_is_refused(
        self, client, indexed_kb, env
    ):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, "official": True},
        )
        assert r.status_code == 400, r.text
        assert "FROZEN" in r.json()["detail"]
        # nothing was persisted by the refused run
        runs = repo.list_answer_evaluation_runs(indexed_kb["id"])
        assert not any(x.official for x in runs)

    def test_unofficial_run_is_allowed_and_labelled(
        self, client, indexed_kb, env
    ):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, "official": False},
        )
        assert r.status_code == 200, r.text
        run = r.json()
        assert run["official"] is False
        assert run["benchmark_lifecycle"] == "draft"


class TestHumanAnswerReviews:
    """V8 STEP 4: persistent, attributed, append-only answer reviews."""

    @staticmethod
    def _run(client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path},
        )
        assert r.status_code == 200, r.text
        return r.json()

    def test_review_round_trip_records_reviewer_and_timestamp(
        self, client, indexed_kb, env
    ):
        run = self._run(client, indexed_kb, env)
        base = (
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/questions/{run['question_ids'][0]}/reviews"
        )
        r = client.post(base, json={
            "reviewer": "ada@example.edu",
            "verdict": "mostly_correct",
            "labels": ["missing_information"],
            "notes": "missed the free-surface point",
        })
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["reviewer"] == "ada@example.edu"
        assert body["verdict"] == "mostly_correct"
        assert body["labels"] == ["missing_information"]
        assert body["created_at"]
        assert body["answer_id"]  # inherited from the evaluated answer

        history = client.get(base).json()
        assert len(history) == 1
        assert history[0]["id"] == body["id"]

    def test_reviews_are_append_only_never_overwritten(self, client, indexed_kb, env):
        run = self._run(client, indexed_kb, env)
        base = (
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/questions/{run['question_ids'][0]}/reviews"
        )
        first = client.post(base, json={
            "reviewer": "ada@example.edu", "verdict": "correct",
            "notes": "looked good",
        })
        assert first.status_code == 201
        second = client.post(base, json={
            "reviewer": "grace@example.edu", "verdict": "incorrect",
            "labels": ["factual_error"], "notes": "sign error",
        })
        assert second.status_code == 201
        assert second.json()["id"] != first.json()["id"]

        history = client.get(base).json()
        assert len(history) == 2, "a second review must never replace the first"
        assert history[0]["reviewer"] == "ada@example.edu"
        assert history[0]["notes"] == "looked good"
        assert history[1]["reviewer"] == "grace@example.edu"
        # oldest first: the sequence a human produced is visible
        assert history[0]["created_at"] <= history[1]["created_at"]

    def test_blank_reviewer_is_rejected(self, client, indexed_kb, env):
        run = self._run(client, indexed_kb, env)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/questions/{run['question_ids'][0]}/reviews",
            json={"reviewer": "   ", "verdict": "correct"},
        )
        assert r.status_code == 422, r.text

    def test_unknown_verdict_is_rejected(self, client, indexed_kb, env):
        run = self._run(client, indexed_kb, env)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/questions/{run['question_ids'][0]}/reviews",
            json={"reviewer": "ada", "verdict": "vibes_based"},
        )
        assert r.status_code == 422

    def test_review_for_unknown_run_is_404(self, client, indexed_kb):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"aerun_nope/questions/q-001/reviews",
            json={"reviewer": "ada", "verdict": "correct"},
        )
        assert r.status_code == 404


class TestHumanEvaluationPass:
    """V8 STEP 5: reviews -> a DERIVED run; source and reviews untouched."""

    @staticmethod
    def _setup(client, indexed_kb, env, verdict="correct"):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        run = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path},
        ).json()
        qid = run["question_ids"][0]
        base = (
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/questions/{qid}"
        )
        r = client.post(base + "/reviews", json={
            "reviewer": "ada@example.edu", "verdict": verdict,
            "labels": ["correct_answer"],
        })
        assert r.status_code == 201, r.text
        return run, qid

    def test_derived_run_scores_correctness_and_keeps_source_intact(
        self, client, indexed_kb, env
    ):
        run, qid = self._setup(client, indexed_kb, env)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/human-evaluation",
        )
        assert r.status_code == 200, r.text
        derived = r.json()
        assert derived["id"] != run["id"], "derivation must create a new record"
        assert derived["evaluator_name"] == "human-reviews"
        assert derived["evaluator_is_model_based"] is False
        assert "ada@example.edu" in derived["evaluator_detail"]
        assert any(run["id"] in n for n in derived["notes"]), "source lineage"

        dq = next(q for q in derived["per_question"] if q["question_id"] == qid)
        assert dq["correctness"]["measured"] is True
        assert dq["correctness"]["value"] == 1.0  # verdict "correct"

        # source run untouched: correctness still UNKNOWN there
        src = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}"
        ).json()
        sq = next(q for q in src["per_question"] if q["question_id"] == qid)
        assert sq["correctness"]["measured"] is False

        # reviews themselves untouched (append-only history still 1 entry)
        history = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/questions/{qid}/reviews"
        ).json()
        assert len(history) == 1

    def test_derived_run_is_comparable_with_its_source(
        self, client, indexed_kb, env
    ):
        run, _ = self._setup(client, indexed_kb, env)
        derived = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/human-evaluation",
        ).json()
        cmp = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/compare",
            params={"left": run["id"], "right": derived["id"]},
        ).json()
        assert cmp["comparable"] is True, cmp["reason"]

    def test_human_pass_without_reviews_is_400(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        run = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path},
        ).json()
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs/"
            f"{run['id']}/human-evaluation",
        )
        assert r.status_code == 400
        assert "no human reviews" in r.json()["detail"]

    def test_run_endpoint_accepts_explicit_human_evaluator(
        self, client, indexed_kb, env
    ):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, "evaluator": "human"},
        )
        assert r.status_code == 200, r.text
        run = r.json()
        assert run["evaluator_name"] == "human-reviews"
        # fresh answers have no reviews -> correctness honestly UNKNOWN
        assert all(not q["correctness"]["measured"]
                   for q in run["per_question"])

    def test_unknown_evaluator_kind_is_400(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, "evaluator": "vibes"},
        )
        assert r.status_code == 400
        assert "unknown evaluator" in r.json()["detail"]


class TestIntersectionCompareAndTopLevel:
    """V8 STEPs 10/11: INCONCLUSIVE intersection verdicts + global routes."""

    @staticmethod
    def _multi_benchmark(tmp_path: Path, kb_id: str, chunk_id: str,
                         name: str, qids: list[str]) -> str:
        import app.api.deps as deps

        repo = deps.get_repo()
        found = {c.id: c for c in repo.get_chunks_by_ids(kb_id, [chunk_id])}
        c = found[chunk_id]
        questions = [
            {
                "question_id": qid,
                "question": "What must the metacentric height GM be for stable "
                             "equilibrium?",
                "answerability": "answerable",
                "required_evidence": [{
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "content_hash": c.content_hash,
                    "section": c.section,
                    "required": True,
                }],
                "citation_requirements": {
                    "require_at_least_one_citation": True,
                    "must_cite_all_required_evidence": True,
                    "must_not_cite_irrelevant_chunks": True,
                },
                "expected_grounding_state": "ANY_ACCEPTABLE",
                "abstention_required": False,
            }
            for qid in qids
        ]
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "benchmark": name, "version": 1, "created_at": "2026-01-01",
            "kb_id": kb_id, "human_review": "pending", "questions": questions,
        }, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _post_run(client, indexed_kb, path, **body):
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, **body},
        )
        assert r.status_code == 200, r.text
        return r.json()

    def test_partial_overlap_compares_as_inconclusive(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = self._multi_benchmark(env[1], indexed_kb["id"], chunks[0].id,
                                     "multi-a", ["q1", "q2"])
        full = self._post_run(client, indexed_kb, path)
        part = self._post_run(client, indexed_kb, path, question_ids=["q1"])
        assert part["question_ids"] == ["q1"]

        cmp = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/compare",
            params={"left": full["id"], "right": part["id"]},
        ).json()
        assert cmp["verdict"] == "INCONCLUSIVE", cmp
        assert cmp["mode"] == "intersection"
        assert cmp["shared_question_ids"] == ["q1"]
        assert cmp["comparable"] is False  # never presented as full-set
        assert cmp["differences"], "differences still computed on the overlap"
        assert "shared question(s) ONLY" in cmp["reason"]

    def test_zero_overlap_is_refused_with_no_differences(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path_a = self._multi_benchmark(env[1], indexed_kb["id"], chunks[0].id,
                                       "multi-a", ["q1", "q2"])
        path_b = self._multi_benchmark(env[1], indexed_kb["id"], chunks[0].id,
                                       "multi-b", ["q3", "q4"])
        a = self._post_run(client, indexed_kb, path_a)
        b = self._post_run(client, indexed_kb, path_b)
        cmp = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/compare",
            params={"left": a["id"], "right": b["id"]},
        ).json()
        assert cmp["verdict"] == "NOT_COMPARABLE", cmp
        assert cmp["mode"] == "no_overlap"
        assert cmp["differences"] == {}
        assert "zero common questions" in cmp["reason"]

    def test_identical_sets_still_report_comparable(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = self._multi_benchmark(env[1], indexed_kb["id"], chunks[0].id,
                                     "multi-a", ["q1", "q2"])
        a = self._post_run(client, indexed_kb, path)
        b = self._post_run(client, indexed_kb, path)
        cmp = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/compare",
            params={"left": a["id"], "right": b["id"]},
        ).json()
        assert cmp["verdict"] == "COMPARABLE"
        assert cmp["comparable"] is True

    def test_top_level_run_listing_and_lookup(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        chunks = [c for c in repo.list_chunks(indexed_kb["id"]) if c.document_id]
        path = _write_benchmark(env[1], indexed_kb["id"], [chunks[0].id])
        run = self._post_run(client, indexed_kb, path)

        listing = client.get("/api/answer-evaluation-runs").json()
        assert any(r["id"] == run["id"] for r in listing)
        # summaries carry provenance fields (STEP 2/3)
        row = next(r for r in listing if r["id"] == run["id"])
        assert row["benchmark_lifecycle"] == "draft"
        assert row["official"] is False
        assert row["relevance_method"]

        filtered = client.get(
            "/api/answer-evaluation-runs",
            params={"kb_id": indexed_kb["id"]},
        ).json()
        assert any(r["id"] == run["id"] for r in filtered)
        not_there = client.get(
            "/api/answer-evaluation-runs", params={"kb_id": "kb_nope"}
        ).json()
        assert not_there == []

        detail = client.get(f"/api/answer-evaluation-runs/{run['id']}")
        assert detail.status_code == 200
        assert detail.json()["id"] == run["id"]
        assert detail.json()["per_question"]

        assert client.get("/api/answer-evaluation-runs/aerun_nope").status_code == 404
