"""V5 API contract tests for the Corpus Command Center endpoints.

Same hermetic style as the V4 API tests: a temporary DATA_DIR, and the vector
store stubbed at the factory boundary with a REAL in-memory cosine store — never
a fabricated success. No test touches a live Qdrant or a pre-existing KB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from tests.test_user_ingestion import make_docx, make_pptx  # noqa: E402

LONG_BODY = (
    "# Hydrostatics\n\nBuoyancy and displacement relate through Archimedes' principle "
    "for a vessel of given volume and mean draft. The waterplane area enters the "
    "displacement equation directly. The block coefficient links volume, draft and "
    "the midship section. " * 3
)


def txt_doc(name: str, marker: str) -> tuple[str, bytes]:
    return (name, (f"# {name}\n\n{marker}\n\n{LONG_BODY}").encode("utf-8"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A client with the vector factory pointed at the in-memory store."""
    import app.api.deps as deps
    import app.config as config

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing-dev-fallback")
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()

    from tests.test_retrieval_integration import InMemoryVectorStore

    store = InMemoryVectorStore()

    import app.services.vector_store.factory as factory_mod

    monkeypatch.setattr(
        factory_mod, "create_vector_store",
        lambda settings, backend=None: store,
    )
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, store
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()


@pytest.fixture
def kb(client, env):
    c, _ = env
    r = c.post("/api/knowledge-bases", json={
        "name": "Naval Architecture — IITM",
        "domain": "Naval Architecture",
        "purpose": "Lecture corpus for naval architecture students",
        "target_audience": "Undergraduate students",
        "source_mode": "user_provided",
    })
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def client(env):
    c, _ = env
    return c


def make_batch(client, kb_id, files, **form):
    """POST a batch. Every file must go under the `files` field name."""
    body = {"index": "true", **form}
    multipart = [("files", (name, content, "application/octet-stream")) for name, content in files]
    return client.post(
        f"/api/knowledge-bases/{kb_id}/ingestion-batches", files=multipart, data=body,
    )


# ---------------------------------------------------------------------------
# Phase 1 — batch endpoints
# ---------------------------------------------------------------------------

def test_create_batch_persists_every_file(client, kb):
    files = [txt_doc(f"lecture_{i}.txt", f"Marker {i}") for i in range(5)]
    r = make_batch(client, kb["id"], files)
    assert r.status_code == 201, r.text
    detail = r.json()
    assert detail["batch"]["total_items"] == 5
    assert len(detail["items"]) == 5
    assert detail["batch"]["completed_items"] == 5
    assert all(i["status"] == "complete" for i in detail["items"])
    assert all(i["progress"] == 1.0 for i in detail["items"])


def test_batch_state_survives_a_new_request(client, kb):
    """A page refresh reads the same batch back from SQLite."""
    files = [txt_doc(f"f{i}.txt", f"M{i}") for i in range(3)]
    created = make_batch(client, kb["id"], files).json()
    batch_id = created["batch"]["id"]

    listed = client.get(f"/api/knowledge-bases/{kb['id']}/ingestion-batches").json()
    assert batch_id in [b["id"] for b in listed]

    fetched = client.get(f"/api/knowledge-bases/{kb['id']}/ingestion-batches/{batch_id}").json()
    assert fetched["batch"]["id"] == batch_id
    assert len(fetched["items"]) == 3
    assert fetched["resumable_items"] == 0


def test_one_bad_file_does_not_fail_the_batch_endpoint(client, kb):
    files = [
        txt_doc("good.txt", "Good"),
        ("broken.pptx", b"this is definitely not a zip container"),
        txt_doc("good2.txt", "Good2"),
    ]
    r = make_batch(client, kb["id"], files)
    assert r.status_code == 201
    detail = r.json()
    assert detail["batch"]["completed_items"] == 2
    assert detail["batch"]["failed_items"] == 1
    assert detail["batch"]["status"] == "partial", "a partial batch must not read as complete"
    failed = [i for i in detail["items"] if i["status"] == "failed"][0]
    assert failed["error_code"] == "INVALID_FILE"
    assert failed["error_message"]


def test_duplicate_upload_is_reported_not_reingested(client, kb):
    files = [txt_doc("same.txt", "Identical")]
    first = make_batch(client, kb["id"], files).json()
    assert first["batch"]["completed_items"] == 1

    second = make_batch(client, kb["id"], files).json()
    assert second["batch"]["duplicate_items"] == 1
    dup = [i for i in second["items"] if i["status"] == "duplicate"][0]
    assert dup["duplicate_of"] == first["items"][0]["document_id"]


def test_resume_is_idempotent(client, kb, env):
    _, store = env
    files = [txt_doc(f"f{i}.txt", f"M{i}") for i in range(3)]
    created = make_batch(client, kb["id"], files).json()
    batch_id = created["batch"]["id"]

    docs_before = client.get(f"/api/knowledge-bases/{kb['id']}/overview").json()["documents"]
    points_before = len(store.points)

    r = client.post(
        f"/api/knowledge-bases/{kb['id']}/ingestion-batches/{batch_id}/resume",
        json={"include_completed": False, "retry_failed": True, "index": True},
    )
    assert r.status_code == 200, r.text
    assert r.json()["batch"]["completed_items"] == 3

    overview = client.get(f"/api/knowledge-bases/{kb['id']}/overview").json()
    assert overview["documents"] == docs_before, "resume must not duplicate documents"
    assert len(store.points) == points_before, "resume must not duplicate vectors"


def test_batch_rejects_a_batch_of_unknown_kb(client):
    r = make_batch(client, "kb_does_not_exist", [txt_doc("a.txt", "a")])
    assert r.status_code == 404


def test_get_batch_from_another_kb_is_404(client, kb):
    files = [txt_doc("a.txt", "a")]
    created = make_batch(client, kb["id"], files).json()
    other = client.post("/api/knowledge-bases", json={
        "name": "Other", "domain": "d", "purpose": "p", "target_audience": "a",
    }).json()
    r = client.get(f"/api/knowledge-bases/{other['id']}/ingestion-batches/{created['batch']['id']}")
    assert r.status_code == 404


def test_100_document_batch(client, kb):
    """A 100-file batch is accounted for exactly."""
    files = [txt_doc(f"lecture_{i:03d}.txt", f"Marker {i}") for i in range(100)]
    r = make_batch(client, kb["id"], files)
    assert r.status_code == 201, r.text
    detail = r.json()
    assert detail["batch"]["total_items"] == 100
    assert detail["batch"]["completed_items"] == 100
    assert detail["batch"]["failed_items"] == 0
    assert len(detail["items"]) == 100
    assert len({i["item_key"] for i in detail["items"]}) == 100, "item keys must be unique"


# ---------------------------------------------------------------------------
# Phase 2 — manifest
# ---------------------------------------------------------------------------

def test_corpus_manifest_and_summary(client, kb, env):
    _, store = env
    files = [txt_doc(f"f{i}.txt", f"M{i}") for i in range(3)]
    make_batch(client, kb["id"], files)

    r = client.get(f"/api/knowledge-bases/{kb['id']}/corpus-manifest")
    assert r.status_code == 200
    manifest = r.json()
    assert manifest["summary"]["total_documents"] == 3
    assert manifest["summary"]["total_chunks"] > 0
    assert manifest["summary"]["vector_count_confirmed"] is True
    entry = manifest["entries"][0]
    assert entry["content_hash"]
    assert entry["provenance"]["user_provided"] is True

    s = client.get(f"/api/knowledge-bases/{kb['id']}/corpus-summary").json()
    assert s["total_documents"] == 3
    assert s["vector_count_confirmed"] is True


def test_manifest_on_empty_corpus_is_honest(client, kb):
    manifest = client.get(f"/api/knowledge-bases/{kb['id']}/corpus-manifest").json()
    assert manifest["summary"]["total_documents"] == 0
    assert manifest["summary"]["total_chunks"] == 0


# ---------------------------------------------------------------------------
# Phase 3 — integrity (read-only)
# ---------------------------------------------------------------------------

def test_corpus_integrity_endpoint(client, kb):
    files = [txt_doc(f"f{i}.txt", f"M{i}") for i in range(2)]
    make_batch(client, kb["id"], files)
    r = client.get(f"/api/knowledge-bases/{kb['id']}/corpus-integrity")
    assert r.status_code == 200
    report = r.json()
    assert report["documents_checked"] == 2
    assert report["overall_status"] in {"HEALTHY", "WARNING", "ERROR"}
    # Every declared bucket must exist in the payload (frontend contract).
    for key in ("missing_vectors", "orphan_vectors", "stale_vectors",
                "embedding_mismatches", "chunking_mismatches", "duplicate_documents",
                "duplicate_chunks", "failed_documents", "partial_documents",
                "provenance_gaps", "broken_source_references"):
        assert key in report


def test_integrity_scan_does_not_modify_anything(client, kb, env):
    _, store = env
    make_batch(client, kb["id"], [txt_doc(f"f{i}.txt", f"M{i}") for i in range(3)])
    before_docs = client.get(f"/api/knowledge-bases/{kb['id']}/overview").json()["documents"]
    before_chunks = client.get(f"/api/knowledge-bases/{kb['id']}/overview").json()["chunks"]
    before_points = len(store.points)

    client.get(f"/api/knowledge-bases/{kb['id']}/corpus-integrity")

    after = client.get(f"/api/knowledge-bases/{kb['id']}/overview").json()
    assert after["documents"] == before_docs
    assert after["chunks"] == before_chunks
    assert len(store.points) == before_points


def test_corpus_health_summary(client, kb):
    make_batch(client, kb["id"], [txt_doc("a.txt", "A")])
    r = client.get(f"/api/knowledge-bases/{kb['id']}/corpus-health")
    assert r.status_code == 200
    body = r.json()
    assert body["overall_status"] in {"HEALTHY", "WARNING", "ERROR"}
    assert "orphan_vectors" in body["counts"]


# ---------------------------------------------------------------------------
# Phase 4 — repair
# ---------------------------------------------------------------------------

def test_repair_plan_does_not_change_anything(client, kb):
    make_batch(client, kb["id"], [txt_doc("a.txt", "A")])
    r = client.post(
        f"/api/knowledge-bases/{kb['id']}/corpus-repair/plan",
        json={"action": "rebuild_kb"},
    )
    assert r.status_code == 200, r.text
    plan = r.json()
    assert plan["destructive"] is True
    assert plan["description"]
    overview = client.get(f"/api/knowledge-bases/{kb['id']}/overview").json()
    assert overview["documents"] == 1


def test_destructive_repair_without_confirmation_is_refused(client, kb):
    make_batch(client, kb["id"], [txt_doc("a.txt", "A")])
    r = client.post(
        f"/api/knowledge-bases/{kb['id']}/corpus-repair",
        json={"action": "rebuild_kb"},
    )
    assert r.status_code == 400
    assert "confirm_action" in r.json()["detail"]


def test_reindex_failed_documents_repair_runs(client, kb):
    files = [txt_doc("a.txt", "A"), ("broken.pptx", b"not a zip")]
    make_batch(client, kb["id"], files)
    r = client.post(
        f"/api/knowledge-bases/{kb['id']}/corpus-repair",
        json={"action": "reindex_failed_documents"},
    )
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["ok"] is True


def test_repair_unknown_action_is_rejected(client, kb):
    r = client.post(
        f"/api/knowledge-bases/{kb['id']}/corpus-repair",
        json={"action": "not_a_real_action"},
    )
    assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Phase 5/6 — versions, fingerprints
# ---------------------------------------------------------------------------

def test_fingerprint_endpoint_is_stable(client, kb):
    make_batch(client, kb["id"], [txt_doc("a.txt", "A"), txt_doc("b.txt", "B")])
    first = client.get(f"/api/knowledge-bases/{kb['id']}/corpus-fingerprint").json()
    second = client.get(f"/api/knowledge-bases/{kb['id']}/corpus-fingerprint").json()
    assert first["fingerprint"] == second["fingerprint"]
    assert first["document_count"] == 2


def test_snapshot_unchanged_corpus_does_not_create_a_new_version(client, kb):
    make_batch(client, kb["id"], [txt_doc("a.txt", "A")])
    v1 = client.post(f"/api/knowledge-bases/{kb['id']}/corpus-versions", json={}).json()
    v2 = client.post(f"/api/knowledge-bases/{kb['id']}/corpus-versions", json={}).json()
    assert v1["id"] == v2["id"]
    assert len(client.get(f"/api/knowledge-bases/{kb['id']}/corpus-versions").json()) == 1


def test_adding_a_document_creates_v2(client, kb):
    make_batch(client, kb["id"], [txt_doc("a.txt", "A")])
    v1 = client.post(f"/api/knowledge-bases/{kb['id']}/corpus-versions", json={}).json()
    make_batch(client, kb["id"], [txt_doc("b.txt", "B")])
    v2 = client.post(f"/api/knowledge-bases/{kb['id']}/corpus-versions", json={}).json()
    assert v1["version"] == "v1"
    assert v2["version"] == "v2"
    assert v1["fingerprint"] != v2["fingerprint"]


def test_version_diff_endpoint(client, kb):
    make_batch(client, kb["id"], [txt_doc("a.txt", "A")])
    v1 = client.post(f"/api/knowledge-bases/{kb['id']}/corpus-versions", json={}).json()
    r = client.get(f"/api/knowledge-bases/{kb['id']}/corpus-versions/diff",
                    params={"from_version": v1["id"]})
    assert r.status_code == 200, r.text
    diff = r.json()
    assert "added" in diff and "changed" in diff and "removed" in diff


# ---------------------------------------------------------------------------
# Phase 11 — ground truth honesty
# ---------------------------------------------------------------------------

def test_new_domain_reports_ground_truth_not_available(client, kb):
    """A domain name alone must never produce ground truth."""
    make_batch(client, kb["id"], [txt_doc(f"f{i}.txt", f"M{i}") for i in range(3)])
    overview = client.get(f"/api/knowledge-bases/{kb['id']}/overview").json()
    assert overview["ground_truth_status"] == "NOT_AVAILABLE"
    assert overview["ground_truth_question_count"] == 0
    assert "domain" in overview["ground_truth_explanation"].lower()
    assert overview["evaluation_required"] is False


def test_ground_truth_stays_not_available_with_a_large_corpus(client, kb):
    make_batch(client, kb["id"], [txt_doc(f"f{i}.txt", f"M{i}") for i in range(5)])
    overview = client.get(f"/api/knowledge-bases/{kb['id']}/overview").json()
    assert overview["documents"] == 5
    assert overview["ground_truth_status"] == "NOT_AVAILABLE", (
        "corpus size does not create ground truth"
    )


# ---------------------------------------------------------------------------
# Multi-format ingestion through the batch endpoint
# ---------------------------------------------------------------------------

def test_batch_accepts_mixed_formats(client, kb):
    # Bodies must be long enough to survive the chunker's MIN_CHUNK_CHARS.
    pptx_body = [
        f"Slide {n} explains {topic} in detail for naval architecture students. " * 3
        for n, topic in enumerate(["free surface effect", "metacentric height", "trim"], start=1)
    ]
    docx_body = " ".join(
        f"Section {n} on {topic} develops the theory and its practical use aboard a vessel."
        for n, topic in enumerate(["resistance", "propulsion", "stability"], start=1)
    ) * 3
    files = [
        txt_doc("notes.txt", "Plain notes"),
        ("deck.pptx", make_pptx([
            ("Free Surface Effect", [pptx_body[0]]),
            ("Metacentric Height", [pptx_body[1]]),
            ("Trim", [pptx_body[2]]),
        ])),
        ("report.docx", make_docx([
            ("heading", "Resistance"), ("body", docx_body),
            ("heading", "Propulsion"), ("body", docx_body),
        ])),
        ("readme.md", ("# Readme\n\nMarkdown lecture notes.\n\n" + LONG_BODY).encode("utf-8")),
    ]
    r = make_batch(client, kb["id"], files)
    assert r.status_code == 201, r.text
    detail = r.json()
    assert detail["batch"]["total_items"] == 4
    assert detail["batch"]["failed_items"] == 0, detail["items"]
    assert detail["batch"]["completed_items"] == 4
    parsers = {i["parser_used"] for i in detail["items"]}
    assert len(parsers) >= 3, f"expected several parsers, got {parsers}"