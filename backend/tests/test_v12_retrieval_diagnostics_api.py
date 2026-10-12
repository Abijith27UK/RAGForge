"""V12 retrieval-diagnostics API tests.

Hermetic: temporary DATA_DIR, in-memory vector store, hashing embeddings, mock
LLM. A real document is indexed through the real upload -> chunk -> embed -> index
path, so the benchmark's expected chunk ids are REAL ids from a REAL corpus.

The tests that matter most are the integrity ones: a diagnostics run must not
touch the knowledge base, must not touch the frozen benchmark artifacts, and must
record a strategy that cannot run as FAILED rather than substituting another one.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from tests.test_v10_review_api import (  # noqa: E402,F401
    NAVAL,
    client,
    env,
    indexed_kb,
)

RUNS = "/api/knowledge-bases/{kb}/retrieval-diagnostics/runs"
AUDIT = "/api/knowledge-bases/{kb}/retrieval-diagnostics/evidence-audit"
BENCH = "/api/knowledge-bases/{kb}/retrieval-diagnostics/benchmark"
TAXONOMY = "/api/knowledge-bases/{kb}/retrieval-diagnostics/taxonomy"

SHIPPED_RETRIEVAL_BENCHMARK = BACKEND_DIR.parent / "benchmarks" / (
    "automobile-engineering-baseline-v1.json"
)


def _chunks(client, kb_id):
    r = client.get(f"/api/knowledge-bases/{kb_id}/chunks", params={"limit": 500})
    assert r.status_code == 200, r.text
    body = r.json()
    return body["chunks"] if isinstance(body, dict) else body


def _write_benchmark(tmp_dir: Path, kb_id: str, questions: list[dict]) -> str:
    """Scratch retrieval benchmark inside the temp DATA_DIR — never the shipped one."""
    payload = {
        "benchmark": "naval-retrieval-diagnostics-v1",
        "version": 1,
        "created_at": "2026-10-10",
        "kb_id": kb_id,
        "kb_name": "scratch",
        "authorship": {"method": "test", "human_review": "PENDING"},
        "relevance_model": "binary",
        "questions": questions,
    }
    path = Path(tmp_dir) / "v12-retrieval-benchmark.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _benchmark_questions(client, kb_id) -> list[dict]:
    """Three questions whose expected evidence really is in the indexed corpus."""
    chunks = _chunks(client, kb_id)
    stability = next(c for c in chunks if "metacentric" in c["text"])
    cavitation = next(c for c in chunks if "cavitation" in c["text"])
    return [
        {
            "id": "nav-001",
            "subdomain": "stability",
            "question": "What must the metacentric height GM be for stable equilibrium?",
            "expected_chunk_ids": [stability["id"]],
            "expected_document_ids": [stability["document_id"]],
            "expected_keywords": [],
            "provenance": "test fixture",
        },
        {
            "id": "nav-002",
            "subdomain": "propulsion",
            "question": "When does propeller cavitation occur?",
            "expected_chunk_ids": [cavitation["id"]],
            "expected_document_ids": [cavitation["document_id"]],
            "expected_keywords": [],
            "provenance": "test fixture",
        },
        {
            "id": "nav-003",
            "subdomain": "stability",
            "question": "How does free surface effect change the metacentric height?",
            "expected_chunk_ids": [stability["id"]],
            "expected_document_ids": [stability["document_id"]],
            "expected_keywords": [],
            "provenance": "test fixture",
        },
    ]


class TestEvidenceAudit:
    def test_audit_resolves_expected_evidence_without_running_anything(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        r = client.get(AUDIT.format(kb=kb_id), params={"benchmark_path": path})
        assert r.status_code == 200, r.text
        audit = r.json()
        assert audit["question_count"] == 3
        assert audit["index_probe"] == "verified"
        assert audit["availability_counts"] == {"all_indexed": 3}
        assert "fully resolvable" in audit["verdict"]
        assert all(
            q["availability_rollup"] == "all_indexed" for q in audit["questions"]
        )
        # The audit must not create a run row.
        assert client.get(RUNS.format(kb=kb_id)).json() == []

    def test_expected_evidence_that_is_not_in_the_corpus_is_reported_as_missing(
        self, client, indexed_kb, env
    ):
        kb_id = indexed_kb["id"]
        questions = _benchmark_questions(client, kb_id)
        questions[0]["expected_chunk_ids"] = ["chk_does_not_exist"]
        path = _write_benchmark(env[1], kb_id, questions)
        audit = client.get(AUDIT.format(kb=kb_id), params={"benchmark_path": path}).json()
        assert audit["availability_counts"]["missing_from_corpus"] == 1
        assert "EVIDENCE GAPS FOUND" in audit["verdict"]

    def test_benchmark_identity_is_exposed_with_its_provenance(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        body = client.get(BENCH.format(kb=kb_id), params={"benchmark_path": path}).json()
        assert body["benchmark"] == "naval-retrieval-diagnostics-v1"
        assert body["fingerprint"]
        assert body["authorship"]["human_review"] == "PENDING"
        assert body["matches_this_kb"] is True
        assert len(body["questions"]) == 3


class TestDiagnosticsRun:
    def _run(self, client, kb_id, path, **overrides):
        payload = {
            "benchmark_path": path,
            "strategies": ["dense"],
            "top_k": 10,
            "persist": True,
        }
        payload.update(overrides)
        return client.post(RUNS.format(kb=kb_id), json=payload)

    def test_run_produces_measured_metrics_and_classifications(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        r = self._run(client, kb_id, path)
        assert r.status_code == 201, r.text
        report = r.json()

        assert report["question_count"] == 3
        assert report["benchmark_fingerprint"]
        assert report["top_k"] == 10
        strategy = report["strategies"][0]
        assert strategy["strategy"] == "dense"
        assert strategy["status"] == "completed", strategy["failure_reason"]
        assert strategy["config_fingerprint"]
        # Candidates must be probed deeper than the evaluated cut-off, otherwise
        # "retrieved below k" and "not retrieved at all" are indistinguishable.
        assert strategy["candidate_depth"] >= strategy["top_k"]
        assert strategy["retrieval_params"]["top_k"] == strategy["candidate_depth"]

        codes = {q["question_id"]: q["classification"]["code"] for q in strategy["questions"]}
        assert set(codes) == {"nav-001", "nav-002", "nav-003"}
        assert all(
            code
            in {
                "RETRIEVAL_SUCCESS",
                "RANKING_FAILURE",
                "LOW_RECALL",
                "NO_RELEVANT_EVIDENCE",
                "RETRIEVAL_FAILURE",
                "EXPECTED_EVIDENCE_UNAVAILABLE",
                "CORPUS_OR_SOURCE_DEFICIENCY",
                "UNKNOWN",
            }
            for code in codes.values()
        )
        # Every classification carries the rule that produced it and its evidence.
        for q in strategy["questions"]:
            assert q["classification"]["rule"]
            assert q["classification"]["evidence"]
            assert q["classification"]["reason"]

        summary = strategy["summary"]
        assert summary["recall_at_k"]["measured"] is True
        assert summary["recall_at_k"]["value"] is not None
        assert summary["mrr"]["measured"] is True
        assert summary["ground_truth_question_count"] == 3
        assert summary["latency_ms"]["measured"] is True
        assert sum(summary["code_counts"].values()) == 3
        assert any("LOW_PRECISION" in w for w in summary["warnings"])

        integrity = report["integrity"]
        assert integrity["protected_kb_unchanged"] is True
        assert integrity["benchmark_unchanged"] is True
        assert integrity["retrieval_runs_persisted"] == 0

    def test_expected_vs_retrieved_is_inspectable_per_question(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        questions = _benchmark_questions(client, kb_id)
        path = _write_benchmark(env[1], kb_id, questions)
        report = self._run(client, kb_id, path).json()
        run_id = report["id"]
        r = client.get(f"{RUNS.format(kb=kb_id)}/{run_id}/questions/nav-001")
        assert r.status_code == 200, r.text
        body = r.json()
        strategy = body["strategies"][0]
        assert set(strategy["expected_chunk_ids"]) == set(questions[0]["expected_chunk_ids"])
        assert strategy["rank_of_expected"], "the expected chunk's rank must be reported"
        assert strategy["retrieved"], "the retrieved items must be inspectable"
        expected_id = questions[0]["expected_chunk_ids"][0]
        assert expected_id in strategy["retrieved_chunk_ids"]
        assert strategy["availability"][0]["status"] == "indexed"
        assert strategy["availability"][0]["retrievable"] is True
        assert strategy["availability_by_chunk"][expected_id] == "indexed"
        assert strategy["metrics"]["recall_at_k"]["measured"] is True

    def test_run_is_persisted_append_only_and_re_readable(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        first = self._run(client, kb_id, path).json()
        second = self._run(client, kb_id, path).json()
        assert first["id"] != second["id"]

        listed = client.get(RUNS.format(kb=kb_id)).json()
        assert {r["id"] for r in listed} == {first["id"], second["id"]}
        assert all(r["benchmark_name"] == "naval-retrieval-diagnostics-v1" for r in listed)
        assert all(r["protected_kb_unchanged"] is True for r in listed)

        fetched = client.get(f"{RUNS.format(kb=kb_id)}/{first['id']}").json()
        assert fetched["id"] == first["id"]
        assert fetched["created_at"] == first["created_at"]
        assert fetched["strategies"][0]["strategy"] == "dense"

    def test_recommendations_cite_measurements_and_require_an_experiment(
        self, client, indexed_kb, env
    ):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        report = self._run(client, kb_id, path).json()
        r = client.get(f"{RUNS.format(kb=kb_id)}/{report['id']}/recommendations")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["recommendations"], "a run must always state its conclusion"
        for rec in body["recommendations"]:
            assert rec["experiment_required"] is True
            assert rec["proposed_config_change"] == {}, (
                "a recommendation must never carry an applied configuration change"
            )
            assert rec["supporting_measurements"] is not None
            assert rec["affected_question_ids"] is not None
        assert body["data_gaps"] is not None
        assert "hypothesis" in body["notice"]

    def test_question_without_reference_evidence_is_unknown_not_a_failure(
        self, client, indexed_kb, env
    ):
        kb_id = indexed_kb["id"]
        questions = _benchmark_questions(client, kb_id)
        questions[0]["expected_chunk_ids"] = []
        questions[0]["expected_document_ids"] = []
        path = _write_benchmark(env[1], kb_id, questions)
        report = self._run(client, kb_id, path).json()
        strategy = report["strategies"][0]
        first = next(q for q in strategy["questions"] if q["question_id"] == "nav-001")
        assert first["classification"]["code"] == "UNKNOWN"
        assert first["evidence"]["recall_at_k"]["measured"] is False
        assert first["evidence"]["recall_at_k"]["value"] is None
        assert strategy["summary"]["no_ground_truth_question_count"] == 1
        assert "unknown, not zero" in first["evidence"]["recall_at_k"]["reason"]

    def test_unavailable_strategy_is_recorded_as_failed_not_substituted(
        self, client, indexed_kb, env
    ):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        r = self._run(client, kb_id, path, strategies=["not_a_strategy"])
        assert r.status_code == 201, r.text
        strategy = r.json()["strategies"][0]
        assert strategy["strategy"] == "not_a_strategy"
        assert strategy["status"] == "failed"
        assert strategy["failure_reason"]
        assert strategy["questions"] == []
        assert strategy["summary"]["recall_at_k"]["measured"] is False

    def test_a_candidate_window_shallower_than_k_is_refused(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        r = self._run(client, kb_id, path, top_k=10, candidate_depth=3)
        assert r.status_code == 400
        assert "candidate_depth" in r.json()["detail"]

    def test_benchmark_for_another_kb_is_refused(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        questions = _benchmark_questions(client, kb_id)
        path = _write_benchmark(env[1], "kb_somewhere_else", questions)
        r = self._run(client, kb_id, path)
        assert r.status_code == 400
        assert "authored for knowledge base" in r.json()["detail"]

    def test_taxonomy_is_published_and_low_precision_explained(self, client, indexed_kb):
        body = client.get(TAXONOMY.format(kb=indexed_kb["id"])).json()
        assert "RETRIEVAL_SUCCESS" in body["codes"]
        assert "UNKNOWN" in body["codes"]
        assert "LOW_PRECISION" in body["codes"]
        assert "LOW_PRECISION" in body["not_emitted"]
        assert "lower bound" in body["not_emitted"]["LOW_PRECISION"]


class TestStrategyComparisonThroughApi:
    def test_two_strategies_are_compared_on_matched_questions(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        r = client.post(
            RUNS.format(kb=kb_id),
            json={
                "benchmark_path": path,
                "strategies": ["dense", "bm25"],
                "top_k": 10,
                "persist": True,
            },
        )
        assert r.status_code == 201, r.text
        report = r.json()
        assert [s["strategy"] for s in report["strategies"]] == ["dense", "bm25"]
        assert report["comparisons"], "two completed strategies must be compared"
        comparison = report["comparisons"][0]
        assert comparison["baseline_label"] == "dense"
        assert comparison["candidate_label"] == "bm25"
        assert comparison["pairing"] in ("COMPARABLE", "INCONCLUSIVE", "NOT_COMPARABLE")
        assert comparison["comparable_question_ids"]
        assert any(c["metric"] == "mrr" for c in comparison["metric_comparisons"])
        assert comparison["notes"], "a comparison must carry its caveats"

        body = client.get(f"{RUNS.format(kb=kb_id)}/{report['id']}/comparisons").json()
        assert body["comparisons"][0]["pairing"] == comparison["pairing"]
        assert any("BOTH strategies measured" in n for n in body["comparisons"][0]["notes"])


class TestIntegrity:
    def test_diagnostics_never_mutate_the_corpus_index_or_benchmark(
        self, client, indexed_kb, env
    ):
        import app.api.deps as deps

        repo = deps.get_repo()
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))

        chunks_before = sorted(c["id"] for c in _chunks(client, kb_id))
        docs_before = sorted(d["id"] for d in client.get(f"/api/knowledge-bases/{kb_id}/documents").json())
        store = None
        from tests.test_retrieval_integration import InMemoryVectorStore

        # The env fixture wired one shared fake store; find it via the factory.
        import app.services.vector_store.factory as factory_mod

        store = factory_mod.create_vector_store(deps.get_settings())
        assert isinstance(store, InMemoryVectorStore)
        points_before = json.dumps(sorted(store.points.keys()))

        report = client.post(
            RUNS.format(kb=kb_id),
            json={"benchmark_path": path, "strategies": ["dense"], "top_k": 10},
        ).json()

        assert report["integrity"]["protected_kb_unchanged"] is True
        assert sorted(c["id"] for c in _chunks(client, kb_id)) == chunks_before
        assert sorted(
            d["id"] for d in client.get(f"/api/knowledge-bases/{kb_id}/documents").json()
        ) == docs_before
        assert json.dumps(sorted(store.points.keys())) == points_before
        assert repo.count_chunks(kb_id) == len(chunks_before)
        assert repo.corpus_revision(kb_id) == report["integrity"]["corpus_revision_before"]

    def test_shipped_frozen_benchmarks_are_untouched_by_a_run(self, client, indexed_kb, env):
        before = hashlib.sha256(SHIPPED_RETRIEVAL_BENCHMARK.read_bytes()).hexdigest()
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        report = client.post(
            RUNS.format(kb=kb_id),
            json={"benchmark_path": path, "strategies": ["dense"], "top_k": 10},
        ).json()
        after = hashlib.sha256(SHIPPED_RETRIEVAL_BENCHMARK.read_bytes()).hexdigest()
        assert before == after
        # The scratch benchmark, too, is byte-identical after the run.
        scratch = Path(path)
        assert report["integrity"]["benchmark_sha256_before"] == report["integrity"][
            "benchmark_sha256_after"
        ]
        assert hashlib.sha256(scratch.read_bytes()).hexdigest() == report["integrity"][
            "benchmark_sha256_before"
        ]

    def test_a_run_does_not_persist_retrieval_history_by_default(self, client, indexed_kb, env):
        import app.api.deps as deps

        repo = deps.get_repo()
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        before = len(client.get(f"/api/knowledge-bases/{kb_id}/retrieval-runs").json())
        client.post(
            RUNS.format(kb=kb_id),
            json={"benchmark_path": path, "strategies": ["dense"], "top_k": 10},
        )
        after = len(client.get(f"/api/knowledge-bases/{kb_id}/retrieval-runs").json())
        assert after == before
        assert repo.count_chunks(kb_id) == len(_chunks(client, kb_id))

    def test_bad_benchmark_paths_are_refused(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        outside = client.post(
            RUNS.format(kb=kb_id),
            json={"benchmark_path": "../outside.json", "strategies": ["dense"]},
        )
        assert outside.status_code in (400, 404)
        missing = client.get(AUDIT.format(kb=kb_id), params={"benchmark_path": "benchmarks/nope.json"})
        assert missing.status_code in (400, 404)
        # A path OUTSIDE the allowed roots must be refused, not read: this endpoint
        # must not become an arbitrary-file-read on the server.
        outside_abs = client.get(
            AUDIT.format(kb=kb_id),
            params={"benchmark_path": "C:/Windows/win.ini"},
        )
        assert outside_abs.status_code in (400, 404)
