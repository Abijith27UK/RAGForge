"""V14 held-out validation API, end to end on a hermetic corpus.

Hermetic: temporary DATA_DIR, in-memory vector store, hashing embeddings. A real
document is indexed through the real upload -> chunk -> embed -> index path, so
the evidence a reviewer reads is genuinely indexed content and the candidate
labels reference REAL chunk ids.

Two properties are measured rather than asserted:

* the API writes ONLY the two held-out artifacts, and only where settings point
  them. The shipped candidate file and the frozen instruments are hashed before
  and after this module runs, so "the tests did not touch the research
  instrument" is a fact about the run;
* before a freeze, a decision is a recorded opinion that can still be revised;
  after a freeze, the artifact is immutable and both a re-review and a
  changed-content freeze are refused.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.heldout_benchmark import (  # noqa: E402
    CandidateStatus,
    HeldOutCandidate,
    HeldOutCandidateSet,
    HeldOutLifecycle,
    load_heldout_benchmark,
    save_candidate_set,
)

REPO_ROOT = BACKEND_DIR.parent
SHIPPED = (
    REPO_ROOT / "benchmarks" / "automobile-engineering-heldout-v1-candidates.json",
    REPO_ROOT / "benchmarks" / "automobile-engineering-heldout-v1.json",
    REPO_ROOT / "benchmarks" / "automobile-engineering-baseline-v1.json",
    REPO_ROOT / "benchmarks" / "answer-quality-automobile-v1.json",
)

NAVAL = """# Hydrostatics and Stability

The metacentric height GM must be positive for stable equilibrium of a floating body.
Free surface effect reduces the effective metacentric height in a flooded compartment.

# Propulsion

Propeller cavitation occurs when the local pressure falls below the vapour pressure of water.
The advance ratio relates the advance per revolution to the propeller diameter.

# Resistance

Froude number scaling compares model and full-scale wave resistance.
Skin friction dominates the resistance of a long slender hull at low speed.

# Seakeeping

Added resistance in waves increases the required shaft power in a seaway.
Roll natural period is set by the transverse metacentric height and the radius of gyration.

# Structures

Longitudinal bending moment peaks amidships in a sagging or hogging condition.
A longitudinal bulkhead increases the torsional stiffness of an open-deck ship.

# Manoeuvring

The rudder area ratio determines the turning-circle diameter of a displacement hull.
A ship with a large block coefficient is typically directionally unstable.
"""

PROTOCOL = {
    "method": "test authoring",
    "question_selection": "read the indexed chunk text; no retriever was run",
    "evidence_selection": "one necessary chunk per question",
    "review_policy": "a named human decides before a validated freeze",
}


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


@pytest.fixture(scope="module")
def shipped_hashes():
    return {p.name: _sha(p) for p in SHIPPED}


@pytest.fixture
def v14_env(tmp_path, monkeypatch):
    """Hermetic app with the held-out artifacts redirected into tmp_path."""
    import app.api.deps as deps
    import app.config as config

    candidate_path = tmp_path / "heldout-candidates.json"
    benchmark_path = tmp_path / "heldout-frozen.json"

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing-dev-fallback")
    monkeypatch.setenv("EMBEDDING_MODEL", "char-ngram-hashing-256")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    # THE load-bearing part: without these, every endpoint below would read and
    # write the shipped research instrument.
    monkeypatch.setenv("HELDOUT_CANDIDATE_PATH", str(candidate_path))
    monkeypatch.setenv("HELDOUT_BENCHMARK_PATH", str(benchmark_path))
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()

    from app.services.embeddings.provider import HashingEmbeddingProvider
    from tests.test_retrieval_integration import InMemoryVectorStore

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

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, tmp_path, candidate_path, benchmark_path
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()


def _make_kb(client, name="Naval — V14 held-out") -> dict:
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": name,
            "domain": "Naval Architecture",
            "purpose": "V14 held-out validation tests",
            "target_audience": "students",
            "source_mode": "user_provided",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def _index(client, kb_id: str) -> None:
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/documents/upload",
        files=[("files", ("naval.md", NAVAL.encode("utf-8"), "text/markdown"))],
        data={"index": "true", "chunker": "section-aware", "target_size": "300", "overlap": "30"},
    )
    assert r.status_code == 201, r.text


def _author_from_corpus(candidate_path: Path, kb_id: str, n: int = 4) -> list[str]:
    """Write a candidate set whose labels are REAL chunk ids of the temp corpus.

    Quotes are taken verbatim from the indexed chunk text, so the validator's
    only machine-checkable sufficiency test actually passes for the right reason.
    """
    from app.api.deps import get_repo

    repo = get_repo()
    chunks = [c for c in repo.list_chunks(kb_id, limit=500) if len(c.text) > 40]
    assert len(chunks) >= n, "the corpus did not produce enough chunks for this test"
    chosen = chunks[:n]

    candidates = []
    for i, chunk in enumerate(chosen):
        first_sentence = chunk.text.replace("\n", " ").strip().split(".")[0].strip()
        candidates.append(
            HeldOutCandidate(
                id=f"naval-hout-{i + 1:03d}",
                subdomain="hydrostatics" if i < 2 else "propulsion",
                question=f"Which statement does the corpus make in chunk {i + 1}?",
                expected_chunk_ids=[chunk.id],
                expected_document_ids=[chunk.document_id],
                evidence_quote=first_sentence,
                coverage_rationale="adds naval content absent from the development set",
                provenance=f"Chunk {chunk.id} ({chunk.document_id}) read in full at test time",
                draft_origin="agent-drafted from the chunk text; no retrieval",
            )
        )
    candidate_set = HeldOutCandidateSet(
        benchmark="heldout-test-v1",
        version=1,
        kb_id=kb_id,
        kb_name="Naval — V14 held-out",
        protocol=dict(PROTOCOL),
        authorship={"agent_authored": True, "human_review": "PENDING"},
        candidates=candidates,
    )
    save_candidate_set(candidate_set, candidate_path)
    return [c.id for c in candidates]


@pytest.fixture
def heldout_kb(v14_env):
    """An indexed KB plus a candidate set authored from that same corpus."""
    client, tmp_path, candidate_path, benchmark_path = v14_env
    kb = _make_kb(client)
    _index(client, kb["id"])
    ids = _author_from_corpus(candidate_path, kb["id"])
    return client, kb, ids, candidate_path, benchmark_path


class TestArtifactIntegrity:
    def test_the_tests_did_not_touch_the_shipped_artifacts(self, shipped_hashes):
        for path in SHIPPED:
            assert _sha(path) == shipped_hashes[path.name], (
                f"{path} changed during the test run; the held-out tooling must "
                "never write the shipped research artifacts"
            )

    def test_the_shipped_frozen_instrument_can_only_be_a_sealed_human_artifact(
        self, shipped_hashes
    ):
        """A test may never create the shipped instrument; a human may.

        The original form of this test asserted the file did not exist, which was
        true only while nobody had frozen anything. It exists now because a human
        sealed it through the API, so the property worth pinning is that whatever
        sits at that path is a SEALED, ATTRIBUTED artifact and is byte-identical
        before and after this module runs. A stray draft written by a test would
        have no ``frozen_by`` and would change the hash.
        """
        path = REPO_ROOT / "benchmarks" / "automobile-engineering-heldout-v1.json"
        if not path.exists():
            assert shipped_hashes[path.name] == ""
            pytest.skip("no shipped frozen instrument at this path yet")
        frozen = load_heldout_benchmark(path)
        assert frozen.frozen_by.strip(), "an unattributed freeze cannot be audited"
        assert frozen.lifecycle is not HeldOutLifecycle.DRAFT
        assert frozen.review["reviewers"], "a sealed instrument names its reviewers"
        assert _sha(path) == shipped_hashes[path.name]


class TestArtifactPathResolution:
    """Where a RELATIVE held-out path points, independently of the process CWD.

    Not a detail. Freezing CREATES the artifact, and the first version of the
    resolver fell back to the process CWD in exactly that case, so a freeze
    triggered from the API (server started in ``backend/``) sealed the instrument
    at ``backend/benchmarks/`` while every document names ``benchmarks/`` at the
    repository root. An artifact whose location depends on the directory a server
    happened to be started in is one no reader or script can rely on.
    """

    def test_a_relative_path_resolves_under_the_repo_root_even_if_it_is_missing(
        self, tmp_path, monkeypatch
    ):
        from app.services.answer_eval.heldout_service import resolve_artifact

        monkeypatch.chdir(tmp_path)
        target = resolve_artifact("benchmarks/not-created-yet.json")
        assert target == REPO_ROOT / "benchmarks" / "not-created-yet.json"
        assert not target.exists()

    def test_a_relative_path_is_not_read_from_the_cwd(
        self, tmp_path, monkeypatch
    ):
        from app.services.answer_eval.heldout_service import resolve_artifact

        # A same-named file next to the CWD must not shadow the repo-root one.
        (tmp_path / "benchmarks").mkdir()
        decoy = tmp_path / "benchmarks" / "automobile-engineering-baseline-v1.json"
        decoy.write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        resolved = resolve_artifact("benchmarks/automobile-engineering-baseline-v1.json")
        assert resolved == REPO_ROOT / "benchmarks" / "automobile-engineering-baseline-v1.json"
        assert resolved != decoy

    def test_an_absolute_path_is_passed_through(self, tmp_path):
        from app.services.answer_eval.heldout_service import resolve_artifact

        given = tmp_path / "explicit.json"
        assert resolve_artifact(given) == given

    def test_the_service_defaults_point_into_the_repository_benchmarks_directory(
        self, repo
    ):
        from app.services.answer_eval.heldout_service import HeldOutService

        service = HeldOutService(repo)
        for path in (service.candidate_path, service.benchmark_path, service.dev_benchmark_path):
            assert path.parent == REPO_ROOT / "benchmarks", (
                f"{path} is not under the repository benchmarks/ directory"
            )


class TestReadEndpoints:
    def test_status_reports_draft_before_any_freeze(self, heldout_kb):
        client, kb, ids, _, _ = heldout_kb
        r = client.get(f"/api/knowledge-bases/{kb['id']}/held-out/benchmark")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["lifecycle"] == "draft"
        assert body["claim_strength"] == "exploratory"
        assert body["surviving_count"] == len(ids)
        assert body["frozen"] is None
        assert body["validation"]["errors"] == [], body["validation"]["errors"]
        assert body["review"]["complete"] is False
        # Separation is measured against the REAL development benchmark.
        assert body["validation"]["separation"]["development_benchmark_questions"] == 28
        assert (
            body["validation"]["separation"]["candidates_sharing_expected_chunk_with_development"]
            == 0
        )

    def test_unknown_knowledge_base_is_a_404(self, heldout_kb):
        client, _, _, _, _ = heldout_kb
        r = client.get("/api/knowledge-bases/kb_missing/held-out/benchmark")
        assert r.status_code == 404

    def test_review_view_shows_the_actual_chunk_text(self, heldout_kb):
        """A reviewer must read the corpus, not a summary of it."""
        client, kb, ids, _, _ = heldout_kb
        r = client.get(f"/api/knowledge-bases/{kb['id']}/held-out/review-view")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["question_count"] == len(ids)
        first = body["questions"][0]
        assert first["evidence"], "the evidence chunk text must be present"
        ev = first["evidence"][0]
        assert ev["resolves_in_corpus"] is True
        assert ev["present_in_index"] is True
        assert len(ev["text"]) > 40
        # The verbatim quote really is in the chunk the reviewer is shown.
        assert first["evidence_quote"].lower() in ev["text"].replace("\n", " ").lower()
        assert first["needs_review"] is True
        assert first["review"] is None

    def test_config_flags_match_the_gate(self, heldout_kb):
        client, kb, _, _, _ = heldout_kb
        body = client.get(f"/api/knowledge-bases/{kb['id']}/held-out/config").json()
        assert body["can_review"] is True
        assert body["can_freeze"] is True
        assert body["can_run"] is False, "a draft instrument is not runnable"
        assert body["supports_confirmatory_claim"] is False

    def test_experiments_endpoint_is_empty_before_any_run(self, heldout_kb):
        client, kb, _, _, _ = heldout_kb
        r = client.get(f"/api/knowledge-bases/{kb['id']}/held-out/experiments")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["experiments"] == []
        assert body["development_references"] == []
        assert body["heldout"] is None
        assert "EXPLORATORY" in body["notice"]


class TestReview:
    def test_a_decision_is_recorded_with_its_reviewer(self, heldout_kb):
        client, kb, ids, _, _ = heldout_kb
        r = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/review",
            json={
                "decisions": [
                    {
                        "question_id": ids[0],
                        "decision": "approve",
                        "reviewer": "A. Reviewer",
                        "notes": "evidence checked",
                    }
                ]
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["review"]["candidates_reviewed"] == 1
        assert body["review"]["reviewers"] == ["A. Reviewer"]

        view = client.get(f"/api/knowledge-bases/{kb['id']}/held-out/review-view").json()
        reviewed = {q["id"]: q for q in view["questions"]}[ids[0]]
        assert reviewed["status"] == "approved"
        assert reviewed["review"]["reviewer"] == "A. Reviewer"
        assert reviewed["needs_review"] is False

    def test_an_unknown_question_id_is_a_400(self, heldout_kb):
        client, kb, _, _, _ = heldout_kb
        r = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/review",
            json={
                "decisions": [
                    {"question_id": "nope-999", "decision": "approve", "reviewer": "A. Reviewer"}
                ]
            },
        )
        assert r.status_code == 400
        assert "unknown candidate id" in r.text

    def test_an_anonymous_decision_is_rejected(self, heldout_kb):
        client, kb, ids, _, _ = heldout_kb
        r = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/review",
            json={"decisions": [{"question_id": ids[0], "decision": "approve", "reviewer": " "}]},
        )
        assert r.status_code == 422

    def test_an_edit_can_replace_the_question_text(self, heldout_kb):
        client, kb, ids, _, _ = heldout_kb
        r = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/review",
            json={
                "decisions": [
                    {
                        "question_id": ids[1],
                        "decision": "edit",
                        "reviewer": "A. Reviewer",
                        "question": "What does the corpus state about free surface effect?",
                    }
                ]
            },
        )
        assert r.status_code == 200, r.text
        view = client.get(f"/api/knowledge-bases/{kb['id']}/held-out/review-view").json()
        edited = {q["id"]: q for q in view["questions"]}[ids[1]]
        assert edited["question"].startswith("What does the corpus state")
        assert edited["review"]["previous_question"].startswith("Which statement")

    def test_a_rejected_question_keeps_its_record_and_leaves_the_set(self, heldout_kb):
        client, kb, ids, _, _ = heldout_kb
        r = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/review",
            json={
                "decisions": [
                    {
                        "question_id": ids[3],
                        "decision": "reject",
                        "reviewer": "A. Reviewer",
                        "notes": "the chunk does not answer this",
                    }
                ]
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["review"]["candidates_rejected"] == 1
        assert body["review"]["candidates_surviving"] == len(ids) - 1


class TestExperimentGate:
    """The runner's refusal to let an unreviewed instrument produce a validated
    claim. These run NO retrieval: every case must be rejected on its declaration
    before a single query is issued."""

    def _runner(self):
        from app.api.deps import get_repo
        from app.config import get_settings
        from app.services.answer_eval.retrieval_experiment_runner import (
            RetrievalExperimentRunner,
        )

        return RetrievalExperimentRunner(get_repo(), get_settings())

    def _args(self, kb_id: str, benchmark_path: Path) -> dict:
        from app.services.answer_eval.retrieval_experiment_models import (
            RetrievalStrategyConfig,
        )

        return {
            "name": "gate test",
            "hypothesis": "a hypothesis is required to run anything",
            "baseline": RetrievalStrategyConfig(label="b", strategy="dense", top_k=3),
            "candidates": [RetrievalStrategyConfig(label="c", strategy="dense", top_k=5)],
            "primary_metric": "recall_at_k",
            "benchmark_path": benchmark_path,
            "persist": False,
        }

    def test_a_held_out_run_that_does_not_declare_the_tier_is_refused(self, heldout_kb):
        from app.services.answer_eval.retrieval_experiment_runner import (
            ExperimentValidationError,
        )

        client, kb, _, _, benchmark_path = heldout_kb
        with pytest.raises(ExperimentValidationError) as exc:
            self._runner().run(
                kb["id"], instrument="held-out", **self._args(kb["id"], benchmark_path)
            )
        assert "tier" in str(exc.value)

    def test_a_confirmatory_claim_from_a_provisional_instrument_is_refused(
        self, heldout_kb
    ):
        from app.services.answer_eval.retrieval_experiment_runner import (
            ExperimentValidationError,
        )

        client, kb, _, _, benchmark_path = heldout_kb
        with pytest.raises(ExperimentValidationError) as exc:
            self._runner().run(
                kb["id"],
                instrument="held-out",
                benchmark_lifecycle="provisional",
                claim_strength="confirmatory",
                **self._args(kb["id"], benchmark_path),
            )
        assert "confirmatory" in str(exc.value)

    def test_an_invented_instrument_name_is_refused(self, heldout_kb):
        from app.services.answer_eval.retrieval_experiment_runner import (
            ExperimentValidationError,
        )

        client, kb, _, _, benchmark_path = heldout_kb
        with pytest.raises(ExperimentValidationError) as exc:
            self._runner().run(
                kb["id"], instrument="nearly-held-out", **self._args(kb["id"], benchmark_path)
            )
        assert "unknown instrument" in str(exc.value)


class TestFreeze:
    def _review_all(self, client, kb_id, ids, reviewer="A. Reviewer"):
        r = client.post(
            f"/api/knowledge-bases/{kb_id}/held-out/review",
            json={
                "decisions": [
                    {"question_id": q, "decision": "approve", "reviewer": reviewer} for q in ids
                ]
            },
        )
        assert r.status_code == 200, r.text

    def test_freezing_without_any_review_seals_as_provisional(self, heldout_kb):
        client, kb, ids, _, benchmark_path = heldout_kb
        r = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/freeze",
            json={"frozen_by": "A. Reviewer"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["lifecycle"] == "provisional"
        assert body["claim_strength"] == "exploratory"
        assert body["fingerprint"]
        assert benchmark_path.exists()

        # The frozen file is loadable by the RETRIEVAL loader the engine uses.
        from app.services.answer_eval.retrieval_diagnostic_runner import (
            load_retrieval_benchmark,
        )

        bench = load_retrieval_benchmark(benchmark_path)
        assert len(bench.questions) == len(ids)
        assert bench.relevance_model == "binary"
        assert bench.kb_id == kb["id"]

    def test_freezing_after_reviewing_everything_seals_as_validated(self, heldout_kb):
        client, kb, ids, _, _ = heldout_kb
        self._review_all(client, kb["id"], ids)
        r = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/freeze",
            json={"frozen_by": "A. Reviewer"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["lifecycle"] == "validated"
        assert body["claim_strength"] == "confirmatory"
        assert body["benchmark"]["review"]["complete"] is True
        assert body["benchmark"]["authorship"]["agent_authored"] is True
        assert body["benchmark"]["authorship"]["human_review"].startswith("COMPLETE")

    def test_freezing_must_name_who_is_sealing_it(self, heldout_kb):
        client, kb, _, _, _ = heldout_kb
        r = client.post(f"/api/knowledge-bases/{kb['id']}/held-out/freeze", json={"frozen_by": "  "})
        assert r.status_code == 400

    def test_a_review_after_the_freeze_is_refused(self, heldout_kb):
        client, kb, ids, _, _ = heldout_kb
        client.post(f"/api/knowledge-bases/{kb['id']}/held-out/freeze", json={"frozen_by": "R"})
        r = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/review",
            json={
                "decisions": [
                    {"question_id": ids[0], "decision": "approve", "reviewer": "A. Reviewer"}
                ]
            },
        )
        assert r.status_code == 400
        assert "no longer blind" in r.text, (
            "the refusal must explain that a post-hoc review cannot strengthen the tier"
        )

    def test_the_frozen_instrument_is_immutable_but_re_freeze_is_idempotent(
        self, heldout_kb
    ):
        client, kb, ids, candidate_path, benchmark_path = heldout_kb
        first = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/freeze", json={"frozen_by": "R"}
        )
        assert first.status_code == 201

        # Same content: a re-run is a no-op, not an error. It may rewrite
        # metadata (the freeze timestamp), which is why immutability here is about
        # the artifact's IDENTITY — its fingerprint — rather than its bytes.
        again = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/freeze", json={"frozen_by": "R"}
        )
        assert again.status_code == 201
        assert again.json()["fingerprint"] == first.json()["fingerprint"]
        before = benchmark_path.read_bytes()

        # Different content under the same path: refused, and refused WITHOUT
        # writing. A rejected freeze that still overwrote the file would be worse
        # than no check at all.
        changed = json.loads(candidate_path.read_text(encoding="utf-8"))
        changed["candidates"][0]["question"] = "A completely different question about drag?"
        candidate_path.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
        conflict = client.post(
            f"/api/knowledge-bases/{kb['id']}/held-out/freeze", json={"frozen_by": "R"}
        )
        assert conflict.status_code == 400
        assert "immutable" in conflict.text
        assert benchmark_path.read_bytes() == before, (
            "a refused freeze must leave the frozen artifact byte-identical"
        )

    def test_freezing_is_refused_while_validation_errors_exist(self, heldout_kb):
        """A frozen artifact with unverifiable labels would make every result from
        it unverifiable, so the gate stops at the freeze rather than at the read."""
        client, kb, _, candidate_path, benchmark_path = heldout_kb
        data = json.loads(candidate_path.read_text(encoding="utf-8"))
        data["candidates"][0]["evidence_quote"] = "a sentence that is not in that chunk"
        candidate_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

        r = client.post(f"/api/knowledge-bases/{kb['id']}/held-out/freeze", json={"frozen_by": "R"})
        assert r.status_code == 400
        assert "validation" in r.text.lower()
        assert not benchmark_path.exists()

    def test_the_status_after_a_freeze_reports_the_tier_and_the_gate(
        self, heldout_kb
    ):
        client, kb, _, _, _ = heldout_kb
        client.post(f"/api/knowledge-bases/{kb['id']}/held-out/freeze", json={"frozen_by": "R"})

        status = client.get(f"/api/knowledge-bases/{kb['id']}/held-out/benchmark").json()
        assert status["lifecycle"] == "provisional"
        assert status["frozen"]["can_be_run"] is True
        assert status["frozen"]["supports_confirmatory_claim"] is False
        assert status["frozen"]["frozen_by"] == "R"

        config = client.get(f"/api/knowledge-bases/{kb['id']}/held-out/config").json()
        assert config["can_review"] is False
        assert config["can_freeze"] is False
        assert config["can_run"] is True
        assert config["supports_confirmatory_claim"] is False
