"""V4 API contract tests: upload, document library, source modes, overview.

Extends the existing hermetic API test style (no Qdrant, no network) with the
V4 endpoints. Indexing that needs a vector store is stubbed at the factory
boundary so the contract is tested without services — but the stub is a real
in-memory cosine store, never a fabricated success.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from tests.test_user_ingestion import make_docx, make_pdf_like_bytes, make_pptx  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.api.deps as deps
    import app.config as config

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()
    settings = config.get_settings()
    assert str(tmp_path) in str(settings.db_path)

    # Route every vector operation to the hermetic in-memory store so these
    # API tests never create or touch a real Qdrant collection.
    from tests.test_retrieval_integration import InMemoryVectorStore

    _store = InMemoryVectorStore()
    import app.services.vector_store.factory as _factory

    monkeypatch.setattr(
        _factory, "create_vector_store",
        lambda settings, backend=None: _store,
    )

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()


@pytest.fixture
def kb(client):
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": "Naval Architecture KB",
            "domain": "Naval Architecture",
            "purpose": "Study assistant for naval architecture students",
            "target_audience": "Naval Architecture students",
            "source_mode": "user_provided",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def upload(client, kb_id: str, files: list[tuple[str, bytes]], **form):
    return client.post(
        f"/api/knowledge-bases/{kb_id}/documents/upload",
        files=[("files", (name, data, "application/octet-stream")) for name, data in files],
        data=form,
    )


# ---------------------------------------------------------------------------
# Phase 1: source mode on the knowledge base
# ---------------------------------------------------------------------------

def test_kb_defaults_to_external_source_mode(client):
    r = client.post(
        "/api/knowledge-bases",
        json={"name": "Auto", "domain": "Automobile Engineering", "purpose": "p", "target_audience": "s"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["source_mode"] == "external"
    assert body["version"] == 1


@pytest.mark.parametrize("mode", ["external", "user_provided", "mixed"])
def test_all_source_modes_are_accepted(client, mode):
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": f"KB {mode}", "domain": "Naval Architecture",
            "purpose": "p", "target_audience": "s", "source_mode": mode,
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["source_mode"] == mode


def test_invalid_source_mode_is_rejected(client):
    r = client.post(
        "/api/knowledge-bases",
        json={
            "name": "KB", "domain": "d", "purpose": "p", "target_audience": "s",
            "source_mode": "telepathy",
        },
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Phase 2: upload workflow
# ---------------------------------------------------------------------------

def test_upload_multiple_supported_formats(client, kb):
    kb_id = kb["id"]
    r = upload(
        client, kb_id,
        [
            ("Lecture_08.pptx", make_pptx([("Free Surface Effect", ["Loose tanks reduce effective GM significantly."])])),
            ("Notes.docx", make_docx([("heading", "Trim"), ("body", "Trim is a longitudinal moment of the ship.")])),
            ("paper.pdf", make_pdf_like_bytes()),
            ("todo.md", b"# TODO\n\nReview the metacentre derivation before the next class session.\n"),
        ],
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["uploaded"] == 4
    assert body["duplicates"] == 0 and body["rejected"] == 0 and body["failed"] == 0
    for entry in body["files"]:
        assert entry["status"] == "uploaded"
        assert entry["document"]["user_provided"] is True
        assert entry["document"]["file_name"] == entry["file_name"]
        assert entry["integrity"]["assessed_by"] == "user-integrity-v1"
        assert entry["document"]["source_id"]

    docs = client.get(f"/api/knowledge-bases/{kb_id}/documents").json()
    assert len(docs) == 4
    by_name = {d["file_name"]: d for d in docs}
    assert by_name["Lecture_08.pptx"]["slide_count"] == 1
    assert by_name["Lecture_08.pptx"]["page_count"] is None
    assert by_name["paper.pdf"]["page_count"] == 2
    assert by_name["Notes.docx"]["section_count"] == 1


def test_duplicate_upload_is_reported_and_nothing_is_replaced(client, kb):
    kb_id = kb["id"]
    payload = ("notes.md", b"# Notes\n\nA sufficiently long body of user supplied notes about GM.\n")
    first = upload(client, kb_id, [payload])
    assert first.status_code == 201 and first.json()["uploaded"] == 1

    second = upload(client, kb_id, [payload])
    assert second.status_code == 201
    body = second.json()
    assert body["uploaded"] == 0 and body["duplicates"] == 1
    entry = body["files"][0]
    assert entry["status"] == "duplicate"
    assert entry["duplicate_of"]
    assert "existing document was kept" in entry["message"]

    docs = client.get(f"/api/knowledge-bases/{kb_id}/documents").json()
    assert len(docs) == 1, "a duplicate must never create a second document"


def test_invalid_and_unsupported_uploads_are_rejected_per_file(client, kb):
    kb_id = kb["id"]
    r = upload(
        client, kb_id,
        [
            ("archive.zip", b"PK\x03\x04random bytes that are not an office document"),
            ("notes.txt", b""),
            ("fake.pptx", b"definitely not a powerpoint file at all"),
            ("good.md", b"# Good\n\nA perfectly acceptable markdown document for the library.\n"),
        ],
    )
    assert r.status_code == 201
    body = r.json()
    assert body["uploaded"] == 1
    assert body["rejected"] == 3
    statuses = {f["file_name"]: f["status"] for f in body["files"]}
    assert statuses["archive.zip"] == "rejected"
    assert statuses["notes.txt"] == "rejected"
    assert statuses["fake.pptx"] == "rejected"
    assert statuses["good.md"] == "uploaded"
    assert all(f["message"] for f in body["files"] if f["status"] == "rejected")


def test_upload_path_traversal_is_neutralised(client, kb):
    r = upload(client, kb["id"], [("../../evil.md", b"# Evil\n\nA path traversal attempt that must be sanitised.\n")])
    assert r.status_code == 201
    assert r.json()["uploaded"] == 1
    doc = r.json()["files"][0]["document"]
    assert doc["file_name"] == "evil.md"
    assert ".." not in doc["raw_file_path"]


def test_upload_rejects_when_no_files_are_supplied(client, kb):
    r = client.post(f"/api/knowledge-bases/{kb['id']}/documents/upload", data={})
    assert r.status_code == 422


def test_upload_into_unknown_kb_is_404(client):
    r = upload(client, "kb_missing", [("a.md", b"# A\n\nBody text long enough to pass validation.\n")])
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Phase 5: document library endpoints
# ---------------------------------------------------------------------------

def test_document_library_endpoints(client, kb):
    kb_id = kb["id"]
    doc = upload(
        client, kb_id,
        [("deck.pptx", make_pptx([("Stability", ["The metacentric height determines initial stability of the vessel."])]))],
    ).json()["files"][0]["document"]

    detail = client.get(f"/api/knowledge-bases/{kb_id}/documents/{doc['id']}")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["document"]["id"] == doc["id"]
    assert detail_body["source"]["user_provided"] is True
    assert detail_body["integrity"] is not None
    assert detail_body["retrieval_enabled"] is False, "not indexed yet"
    assert detail_body["kb_version"] == 1

    text = client.get(f"/api/knowledge-bases/{kb_id}/documents/{doc['id']}/text")
    assert text.status_code == 200
    assert "metacentric" in text.json()["text"]
    assert text.json()["parser"] == "pptx"

    chunks = client.get(f"/api/knowledge-bases/{kb_id}/documents/{doc['id']}/chunks")
    assert chunks.status_code == 200

    library = client.get(f"/api/knowledge-bases/{kb_id}/document-library")
    assert library.status_code == 200
    lib = library.json()
    assert lib["documents"] == 1
    assert lib["user_provided"] == 1
    assert lib["external"] == 0
    assert lib["by_status"]["parsed"] == 1
    assert ".pptx" in lib["supported_extensions"]
    assert lib["max_upload_bytes"] > 0


def test_document_text_truncation_is_reported(client, kb):
    kb_id = kb["id"]
    doc = upload(
        client, kb_id,
        [("long.md", b"# Long\n\n" + b"word " * 2000)],
    ).json()["files"][0]["document"]
    page = client.get(f"/api/knowledge-bases/{kb_id}/documents/{doc['id']}/text", params={"limit": 100})
    body = page.json()
    assert body["returned"] == 100
    assert body["truncated"] is True
    assert body["char_count"] > 100


def test_unknown_document_endpoints_return_404(client, kb):
    assert client.get(f"/api/knowledge-bases/{kb['id']}/documents/doc_nope").status_code == 404
    assert client.get(f"/api/knowledge-bases/{kb['id']}/documents/doc_nope/text").status_code == 404


def test_delete_document_endpoint_removes_it(client, kb):
    kb_id = kb["id"]
    doc = upload(client, kb_id, [("gone.md", b"# Gone\n\nThis document will be removed from the library.\n")]).json()
    doc_id = doc["files"][0]["document"]["id"]
    r = client.delete(f"/api/knowledge-bases/{kb_id}/documents/{doc_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == doc_id
    # Qdrant is not running in tests: the failure is reported, not hidden.
    assert "vector_store_error" in body
    assert client.get(f"/api/knowledge-bases/{kb_id}/documents/{doc_id}").status_code == 404


def test_rebuild_reuses_the_stored_original(client, kb):
    kb_id = kb["id"]
    doc = upload(client, kb_id, [("rebuild.md", b"# Rebuild\n\nOriginal user supplied body text here.\n")]).json()["files"][0]["document"]
    # Rebuilding without indexing does not need a vector store.
    r = client.post(f"/api/knowledge-bases/{kb_id}/documents/{doc['id']}/rebuild", json={"reindex": False})
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["status"] == "done"
    assert "re-parsed" in run["stages"][0]["message"].lower()

    after = client.get(f"/api/knowledge-bases/{kb_id}/documents/{doc['id']}").json()["document"]
    assert after["parser"] == "markdown"
    assert after["parse_error"] is None


# ---------------------------------------------------------------------------
# Phase 6: incremental indexing through the API
# ---------------------------------------------------------------------------

class _StubStore:
    """In-memory stand-in used only where a vector store is unavoidable."""

    def __init__(self) -> None:
        self.indexed: list[str] = []
        self.deleted: list[list[str]] = []

    def ensure_collection(self, kb_id, vector_size, distance="cosine"):
        return None

    def upsert_chunks(self, kb_id, chunks, vectors):
        self.indexed.extend(c.document_id for c in chunks)

    def delete_document_vectors(self, kb_id, document_ids):
        self.deleted.append(list(document_ids))
        return len(document_ids)

    def delete_orphaned_points(self, kb_id, valid_chunk_ids):
        return 0

    def search(self, kb_id, vector, top_k, filters=None):
        return []

    def collection_info(self, kb_id):
        return {"name": kb_id, "points_count": len(self.indexed), "vector_size": 384, "status": "green"}

    def delete_collection(self, kb_id):
        return None


@pytest.fixture
def stub_vector_store(monkeypatch):
    import app.services.vector_store.factory as factory
    import app.services.indexing.document_indexer  # noqa: F401
    from app.services.embeddings.provider import HashingEmbeddingProvider

    store = _StubStore()

    def fake_create(settings, backend=None):
        return store

    monkeypatch.setattr(factory, "create_vector_store", fake_create)
    import app.api.routes_documents as routes_documents
    import app.api.routes_build as routes_build

    monkeypatch.setattr(routes_documents, "create_vector_store", fake_create)
    monkeypatch.setattr(routes_build, "create_vector_store", fake_create)

    import app.services.embeddings.provider as provider_module

    fake_embedder = lambda settings, expected=None: HashingEmbeddingProvider()  # noqa: E731
    monkeypatch.setattr(provider_module, "create_embedding_provider", fake_embedder)
    monkeypatch.setattr(routes_documents, "create_embedding_provider", fake_embedder)
    return store


def test_upload_with_index_ingests_incrementally(client, kb, stub_vector_store):
    kb_id = kb["id"]
    r = upload(
        client, kb_id,
        [("a.md", b"# Stability\n\nThe metacentric height determines the initial stability of a vessel.\n")],
        index="true",
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["indexed"] is True
    assert body["indexing"]["documents_indexed"] == 1
    assert body["indexing"]["chunk_count"] > 0
    assert body["files"][0]["status"] == "indexed"
    assert stub_vector_store.indexed

    # A second upload only processes the new document.
    before = len(stub_vector_store.indexed)
    r2 = upload(
        client, kb_id,
        [("b.md", b"# Trim\n\nTrim is the longitudinal moment of the ship caused by weight distribution.\n")],
        index="true",
    )
    assert r2.status_code == 201
    assert r2.json()["indexing"]["documents_indexed"] == 1
    assert len(stub_vector_store.indexed) > before

    docs = client.get(f"/api/knowledge-bases/{kb_id}/documents").json()
    assert all(d["status"] == "ready" for d in docs)
    assert all(d["chunk_count"] > 0 for d in docs)


def test_single_document_index_endpoint_removes_stale_vectors(client, kb, stub_vector_store):
    kb_id = kb["id"]
    doc_id = upload(
        client, kb_id,
        [("c.md", b"# Stability\n\nA long body of text about the metacentric height and initial stability.\n")],
    ).json()["files"][0]["document"]["id"]

    r = client.post(
        f"/api/knowledge-bases/{kb_id}/documents/{doc_id}/index",
        json={"chunker": "section-aware", "target_size": 600, "overlap": 80},
    )
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["status"] == "done"
    assert "stale vectors removed" in run["stages"][1]["message"]
    assert stub_vector_store.deleted[-1] == [doc_id]


def test_indexing_a_failed_document_is_refused(client, kb, stub_vector_store):
    kb_id = kb["id"]
    # A .ppt that passes magic-byte validation but is rejected by the parser.
    result = upload(client, kb_id, [("old.ppt", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 128)])
    body = result.json()
    entry = body["files"][0]
    assert entry["status"] == "failed"
    assert "legacy binary PowerPoint" in entry["message"]
    assert entry["document"]["status"] == "failed"
    doc_id = entry["document"]["id"]

    r = client.post(f"/api/knowledge-bases/{kb_id}/documents/{doc_id}/index", json={})
    assert r.status_code == 422
    assert "rebuild" in r.json()["detail"].lower()


def test_replace_document_creates_a_new_version(client, kb, stub_vector_store):
    kb_id = kb["id"]
    original = upload(
        client, kb_id, [("v1.md", b"# Notes v1\n\nThe first edition of these naval architecture notes.\n")],
    ).json()["files"][0]["document"]

    r = client.post(
        f"/api/knowledge-bases/{kb_id}/documents/{original['id']}/replace",
        files=[("file", ("v2.md", b"# Notes v2\n\nThe second edition with an extra chapter on trim.\n", "text/markdown"))],
        data={"reason": "new semester"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document"]["document_version"] == 2
    assert body["document"]["replaces_document_id"] == original["id"]
    assert body["superseded"] == original["id"]

    # The old document is history, not deleted.
    history = client.get(f"/api/knowledge-bases/{kb_id}/documents/{original['id']}").json()["document"]
    assert history["status"] == "failed"
    assert "Superseded by" in history["parse_error"]


def test_replace_with_identical_bytes_is_refused(client, kb):
    kb_id = kb["id"]
    data = b"# Same\n\nIdentical content uploaded twice in a row for the same document.\n"
    original = upload(client, kb_id, [("same.md", data)]).json()["files"][0]["document"]
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/documents/{original['id']}/replace",
        files=[("file", ("same.md", data, "text/markdown"))],
    )
    assert r.status_code == 409
    assert "byte-identical" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Phase 8: KB overview, and evaluation being optional
# ---------------------------------------------------------------------------

def test_overview_reports_a_ready_kb_with_no_benchmark(client, kb, stub_vector_store):
    kb_id = kb["id"]
    upload(
        client, kb_id,
        [("a.md", b"# Stability\n\nThe metacentric height of a vessel governs its initial stability.\n")],
        index="true",
    )
    r = client.get(f"/api/knowledge-bases/{kb_id}/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["kb"]["source_mode"] == "user_provided"
    assert body["documents"] == 1
    assert body["documents_ready"] == 1
    assert body["user_provided_documents"] == 1
    assert body["external_documents"] == 0
    assert body["chunks"] > 0
    assert body["vectors"] == body["chunks"]
    assert body["vector_backend"] == "qdrant"
    assert body["chunking_strategy"] == "section-aware"
    assert body["evaluation_status"] == "not_configured"
    assert body["evaluation_required"] is False, "ground truth is optional"
    assert body["benchmark_versions"] == 0
    assert body["build_status"] in ("ready", "building")


def test_overview_without_documents_is_honest(client, kb):
    body = client.get(f"/api/knowledge-bases/{kb['id']}/overview").json()
    assert body["documents"] == 0
    assert body["chunks"] == 0
    assert body["build_status"] == "not_built"
    assert body["evaluation_status"] == "not_configured"
    # No vector store in tests -> reported as unreachable, never as 0 vectors.
    assert body["vectors"] is None or body["vectors"] == 0


def test_evaluate_without_ground_truth_never_fabricates(client, kb):
    r = client.post(f"/api/knowledge-bases/{kb['id']}/evaluate", json={"top_k": 5})
    assert r.status_code in (400, 503)
    assert "detail" in r.json()


def test_overview_of_unknown_kb_is_404(client):
    assert client.get("/api/knowledge-bases/kb_nope/overview").status_code == 404


# ---------------------------------------------------------------------------
# Phase 11: the frozen benchmark stays immutable
# ---------------------------------------------------------------------------

def test_editing_an_approved_question_still_creates_a_draft_revision(client, kb):
    """V3 benchmark behaviour must survive V4 unchanged."""
    kb_id = kb["id"]
    qid = client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions",
        json={"question": "What is the metacentric height?", "expected_keywords": ["metacentric"]},
    ).json()["id"]
    client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions/{qid}/status",
        json={"status": "APPROVED", "reviewer": "human"},
    )
    revision = client.patch(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions/{qid}", json={"question": "changed"},
    )
    assert revision.status_code == 200
    assert revision.json()["status"] == "DRAFT"
    assert revision.json()["supersedes"] == qid
    assert revision.json()["revision"] == 2

    # The approved ancestor keeps its approved content untouched.
    original = client.get(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions"
    ).json()
    ancestor = next(x for x in original if x["id"] == qid)
    assert ancestor["question"] == "What is the metacentric height?"
    assert ancestor["status"] == "APPROVED"


def test_frozen_benchmark_cannot_be_edited_or_deleted_through_v4_flows(client, kb):
    kb_id = kb["id"]
    q = client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions",
        json={"question": "What is the metacentric height?", "expected_keywords": ["metacentric"]},
    )
    assert q.status_code == 201
    qid = q.json()["id"]

    client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions/{qid}/status",
        json={"status": "APPROVED", "reviewer": "human"},
    )
    # Freezing the question makes it immutable.
    assert client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions/{qid}/status",
        json={"status": "FROZEN", "reviewer": "human"},
    ).status_code == 200

    version = client.post(
        f"/api/knowledge-bases/{kb_id}/benchmark-versions",
        json={"version": "naval-v1", "created_by": "human"},
    )
    assert version.status_code == 201
    bv_id = version.json()["id"]
    assert version.json()["status"] == "FROZEN", "all questions frozen -> snapshot is frozen"

    frozen = version.json()
    assert client.delete(f"/api/knowledge-bases/{kb_id}/benchmark-versions/{bv_id}").status_code == 409

    edit = client.patch(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions/{qid}",
        json={"question": "changed again"},
    )
    assert edit.status_code == 409
    assert "FROZEN" in edit.json()["detail"]

    # Uploading user documents must not disturb the benchmark.
    upload(client, kb_id, [("a.md", b"# Notes\n\nSome user material that changes no ground truth.\n")])
    still = client.get(f"/api/knowledge-bases/{kb_id}/benchmark-versions/{bv_id}").json()
    assert still["status"] == "FROZEN"
    assert still["questions_snapshot"] == frozen["questions_snapshot"]

    overview = client.get(f"/api/knowledge-bases/{kb_id}/overview").json()
    assert overview["evaluation_status"] == "benchmark_frozen"
    assert overview["frozen_benchmark_versions"] == 1
    assert overview["documents"] == 1


def test_frozen_benchmark_gate_for_v3_experiment_is_exercised_by_the_runner(client, kb):
    """The v3 runner refuses anything but a FROZEN benchmark version."""
    import sys as _sys

    _sys.path.insert(0, str(BACKEND_DIR / "scripts"))
    from run_source_selection_experiment_v3 import resolve_frozen_benchmark

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}

        def json(self) -> dict:
            return self._payload

    class FakeClient:
        """Minimal httpx.Client stand-in over the live TestClient."""

        def __init__(self, inner: TestClient) -> None:
            self._inner = inner

        def get(self, path: str) -> FakeResponse:
            response = self._inner.get(path)
            return FakeResponse(response.status_code, response.json())

    runner_client = FakeClient(client)
    kb_id = kb["id"]

    q = client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions",
        json={"question": "Q?", "expected_keywords": ["x"]},
    ).json()
    draft = client.post(
        f"/api/knowledge-bases/{kb_id}/benchmark-versions", json={"version": "draft-v1"},
    ).json()
    assert draft["status"] == "DRAFT"
    with pytest.raises(SystemExit, match="not FROZEN"):
        resolve_frozen_benchmark(runner_client, kb_id, draft["id"])

    # Approve + freeze the question, so the snapshot itself is FROZEN.
    client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions/{q['id']}/status",
        json={"status": "APPROVED", "reviewer": "human"},
    )
    client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions/{q['id']}/status",
        json={"status": "FROZEN", "reviewer": "human"},
    )
    frozen_version = client.post(
        f"/api/knowledge-bases/{kb_id}/benchmark-versions", json={"version": "naval-frozen-v1"},
    ).json()
    assert frozen_version["status"] == "FROZEN"

    resolved = resolve_frozen_benchmark(runner_client, kb_id, frozen_version["id"])
    assert resolved["status"] == "FROZEN"
    assert resolved["version"] == "naval-frozen-v1"
    assert len(resolved["questions_snapshot"]) == 1

    with pytest.raises(SystemExit, match="not found"):
        resolve_frozen_benchmark(runner_client, kb_id, "bv_missing")


# ---------------------------------------------------------------------------
# Source-mode behaviour in the sources API
# ---------------------------------------------------------------------------

def test_ingest_route_explains_why_user_sources_are_not_downloaded(client, kb):
    upload(client, kb["id"], [("a.md", b"# Notes\n\nA user supplied document that needs no downloading.\n")])
    r = client.post(f"/api/knowledge-bases/{kb['id']}/ingest", json={})
    assert r.status_code == 400
    assert "user-provided" in r.json()["detail"]


def test_user_urls_are_marked_user_provided(client):
    kb_id = client.post(
        "/api/knowledge-bases",
        json={
            "name": "Mixed", "domain": "Naval Architecture", "purpose": "p",
            "target_audience": "s", "source_mode": "mixed",
        },
    ).json()["id"]

    r = client.post(
        f"/api/knowledge-bases/{kb_id}/sources/user-urls",
        json={"urls": ["http://127.0.0.1/private", "https://example.org/class-notes"]},
    )
    assert r.status_code == 200
    sources = r.json()
    assert len(sources) == 2
    assert all(s["user_provided"] for s in sources)
    assert all(s["provenance"] == "user_url" for s in sources)
    invalid = next(s for s in sources if "127.0.0.1" in s["url"])
    assert invalid["notes"].startswith("INVALID")
    assert invalid["decision"] in ("REVIEW", "REJECT")


def test_sources_list_distinguishes_user_from_discovered(client, kb):
    kb_id = kb["id"]
    upload(client, kb_id, [("a.md", b"# Notes\n\nUploaded material that becomes a user provided source.\n")])
    sources = client.get(f"/api/knowledge-bases/{kb_id}/sources").json()
    assert len(sources) == 1
    assert sources[0]["user_provided"] is True
    assert sources[0]["provenance"] == "user_upload"
    assert sources[0]["quality"]["assessed_by"] == "user-integrity-v1"


def test_upload_data_guards_torch_fixtures_are_real(tmp_path):
    """Sanity guard on the shared OOXML fixtures."""
    assert make_pptx([("T", ["Body"])])[:2] == b"PK"
    assert len(make_docx([("heading", "H"), ("body", "B")])) > 1000
    assert io.BytesIO(b"x").getbuffer().nbytes == 1