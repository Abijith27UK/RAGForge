"""Corpus-engineering tests: bulk ingestion, manifest, integrity, repair,
fingerprints and diffs (V5 Phases 1-6, 12).

Everything here runs against a throwaway SQLite database, a temporary upload
directory and the hermetic in-memory vector store. No test touches a real
Qdrant collection, a pre-existing knowledge base, or anything under
``benchmarks/``.

The theme throughout: the system must be honest about what it knows. A count
that was not measured is UNKNOWN, not zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.schemas.corpus import (  # noqa: E402
    UNKNOWN,
    CorpusVersion,
    IngestionItemStatus,
    IngestionStage,
    RepairAction,
    RepairPlan,
)
from app.schemas.models import (  # noqa: E402
    Chunk,
    Document,
    DocumentStatus,
    KnowledgeBase,
    Source,
    SourceType,
)
from app.services.corpus.batches import IngestionBatchService, item_key_for  # noqa: E402
from app.services.corpus.fingerprint import (  # noqa: E402
    corpus_fingerprint,
    diff_manifests,
    snapshot_corpus_version,
)
from app.services.corpus.integrity import CorpusIntegrityService  # noqa: E402
from app.services.corpus.manifest import CorpusManifestService, chunking_config_hash  # noqa: E402
from app.services.corpus.repair import CorpusRepairService, RepairError  # noqa: E402
from app.services.embeddings.provider import HashingEmbeddingProvider  # noqa: E402
from app.services.indexing.document_indexer import index_documents  # noqa: E402
from app.utils.ids import new_id  # noqa: E402
from tests.test_retrieval_integration import InMemoryVectorStore  # noqa: E402
from tests.test_user_ingestion import make_pptx  # noqa: E402

KB_ID = "kb_corpus"

#: Long enough to produce real chunks (MIN_CHUNK_CHARS is 60).
BODY = (
    "Hydrostatics establishes the relation between a vessel's displaced volume, "
    "its mean draft and the waterplane area. Archimedes' principle underpins every "
    "stability calculation performed on the vessel. The block coefficient enters "
    "directly into the displacement equation used by the loading computer. "
)


def long_text(seed: int) -> str:
    return f"# Section {seed}\n\n" + BODY + f" Unique marker token for document {seed}." * 3


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    """A KB + repo + in-memory store, with ``seed`` documents already indexed."""
    repo = Repository(str(tmp_path / "corpus.db"))
    kb = KnowledgeBase(
        id=KB_ID, name="Naval Architecture", domain="Naval Architecture",
        purpose="Course material", target_audience="students", depth="technical",
    )
    repo.create_kb(kb)
    store = InMemoryVectorStore()
    return repo, kb, store, tmp_path


def seed_document(repo: Repository, kb: KnowledgeBase, store, seed: int, tmp_path: Path, *,
                  status: DocumentStatus = DocumentStatus.READY) -> Document:
    """Create one real document, parse it and index it through the shared path."""
    source = Source(
        id=new_id("src"), kb_id=kb.id, url=f"upload://doc{seed}.txt", title=f"doc{seed}.txt",
        source_type=SourceType.TEXT, user_provided=True, provenance="user_upload",
        file_name=f"doc{seed}.txt", file_size=len(long_text(seed)),
    )
    repo.create_source(source)
    raw = corpus_tmp(tmp_path, seed)
    doc = Document(
        id=new_id("doc"), kb_id=kb.id, source_id=source.id, url=source.url,
        title=f"doc{seed}.txt", source_type=SourceType.TEXT,
        file_path=str(raw.with_suffix(".parsed.txt")), content_hash=f"hash{seed}",
        text_length=len(long_text(seed)), file_name=f"doc{seed}.txt",
        file_size=len(long_text(seed)), mime_type="text/plain", parser="text",
        raw_file_path=str(raw), user_provided=True, status=DocumentStatus.PARSED,
    )
    raw.with_suffix(".parsed.txt").write_text(long_text(seed), encoding="utf-8")
    repo.create_document(doc)
    index_documents(
        repo=repo, kb=kb, documents=[doc], store=store,
        embedder=HashingEmbeddingProvider(), chunker_name="section-aware",
        target_size=600, overlap=80,
    )
    stored = repo.get_document(kb.id, doc.id)
    if status is not DocumentStatus.READY:
        stored.status = status
        repo.update_document(stored)
        return stored
    return stored


def corpus_tmp(tmp_path: Path, seed: int) -> Path:
    p = Path(tmp_path) / f"raw{seed}.txt"
    p.write_text(long_text(seed), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Phase 1 — batch persistence and lifecycle
# ---------------------------------------------------------------------------

def _service(repo, tmp_path) -> IngestionBatchService:
    return IngestionBatchService(
        repo=repo, max_bytes=10 * 1024 * 1024,
        upload_dir_for=lambda kb_id: Path(tmp_path) / kb_id,
        allowed_extensions=[".txt", ".md", ".pdf", ".pptx", ".docx"],
    )


def _source_builder(kb_id: str):
    def build(file_name: str, size: int, extension: str) -> Source:
        return Source(
            id=new_id("src"), kb_id=kb_id, url=f"upload://{file_name}", title=file_name,
            source_type=SourceType.TEXT, discovered_via="user-upload",
            user_provided=True, provenance="user_upload",
            file_name=file_name, file_size=size,
        )
    return build


def _store_for(store):
    return lambda kb_id: store


def _indexer(repo, kb, store):
    def run(documents, *, chunker="section-aware", target_size=600, overlap=80):
        return index_documents(
            repo=repo, kb=kb, documents=documents, store=store,
            embedder=HashingEmbeddingProvider(), chunker_name=chunker,
            target_size=target_size, overlap=overlap,
        ).as_dict()
    return run


def test_batch_item_keys_are_deterministic():
    a = item_key_for(3, "Lecture_05.pdf", 1000, "abcdef0123456789")
    b = item_key_for(3, "Lecture_05.pdf", 1000, "abcdef0123456789")
    assert a == b
    assert a != item_key_for(4, "Lecture_05.pdf", 1000, "abcdef0123456789")
    assert a != item_key_for(3, "Other.pdf", 1000, "abcdef0123456789")


def test_fifty_file_batch_creates_one_item_per_file(corpus):
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=50)
    builder = _source_builder(kb.id)

    for i in range(50):
        item = svc.add_item(
            batch_id=batch.id, kb_id=kb.id, index=i,
            file_name=f"lecture_{i:03d}.txt", size=100, content_hash=f"h{i:03d}",
        )
        svc.process_item(item, data=long_text(i).encode(), kb=kb,
                         source_builder=builder, index_documents=_indexer(repo, kb, store))

    detail = svc.get_detail(repo.get_ingestion_batch(batch.id))
    assert detail.batch.total_items == 50
    assert detail.batch.completed_items == 50, "every file must be accounted for"
    assert repo.count_documents(kb.id) == 50
    assert sum(1 for i in detail.items if i.stage is IngestionStage.COMPLETE) == 50
    assert all(i.progress == 1.0 for i in detail.items)


def test_batch_state_survives_a_fresh_service_instance(corpus):
    """The batch is in SQLite, not in memory — a page refresh cannot lose it."""
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=2)
    builder = _source_builder(kb.id)
    for i in range(2):
        item = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=i,
                            file_name=f"f{i}.txt", size=10, content_hash=f"h{i}")
        svc.process_item(item, data=long_text(i).encode(), kb=kb,
                         source_builder=builder, index_documents=None)

    # A brand new service (as after a process restart) reads the same state.
    fresh = _service(repo, tmp_path)
    reloaded = repo.get_ingestion_batch(batch.id)
    assert reloaded is not None
    detail = fresh.get_detail(reloaded)
    assert len(detail.items) == 2
    assert detail.resumable_items == 0, "both completed; nothing to resume"


def test_one_bad_file_does_not_fail_the_batch(corpus):
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=3)
    builder = _source_builder(kb.id)

    good_a = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=0,
                          file_name="a.txt", size=10, content_hash="ha")
    svc.process_item(good_a, data=long_text(1).encode(), kb=kb,
                     source_builder=builder, index_documents=None)

    bad = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=1,
                       file_name="broken.pptx", size=10, content_hash="hb")
    svc.process_item(bad, data=b"not a zip at all", kb=kb,
                     source_builder=builder, index_documents=None)

    good_b = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=2,
                          file_name="b.txt", size=10, content_hash="hc")
    svc.process_item(good_b, data=long_text(2).encode(), kb=kb,
                     source_builder=builder, index_documents=None)

    batch = svc.finish_batch(repo.get_ingestion_batch(batch.id), kb)
    detail = svc.get_detail(batch)
    assert detail.batch.completed_items == 2
    assert detail.batch.failed_items == 1
    assert bad.status is IngestionItemStatus.FAILED
    assert bad.error_code == "INVALID_FILE"
    assert bad.error_message, "a failure must always carry a reason"
    # Every item reached a terminal state, but one of three never succeeded:
    # that is a PARTIAL batch and must never read as "complete".
    assert detail.batch.status == "partial"
    assert 2 == detail.batch.total_items - detail.batch.failed_items


def test_unreadable_pptx_is_recorded_not_dropped(corpus):
    """A corrupt Office file is rejected at validation and clearly reported.

    It never becomes a Document (nothing was parsed), but the batch item
    persists with a reason so it cannot silently disappear.
    """
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=1)
    item = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=0,
                        file_name="lecture.pptx", size=20, content_hash="hx")
    svc.process_item(item, data=b"PK\x03\x04 definitely broken", kb=kb,
                     source_builder=_source_builder(kb.id), index_documents=None)
    assert item.status is IngestionItemStatus.FAILED
    assert item.error_code == "INVALID_FILE"
    assert "ZIP" in (item.error_message or "")
    assert repo.count_documents(kb.id) == 0, "a rejected file must not become a document"
    # The batch item is still there — that is what makes the failure visible.
    assert len(svc.get_detail(repo.get_ingestion_batch(batch.id)).items) == 1


def test_duplicate_content_is_detected_and_existing_kept(corpus):
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    builder = _source_builder(kb.id)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=2)

    first = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=0,
                         file_name="a.txt", size=10, content_hash="same")
    svc.process_item(first, data=long_text(9).encode(), kb=kb,
                     source_builder=builder, index_documents=None)
    original_doc_id = first.document_id

    second = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=1,
                          file_name="copy.txt", size=10, content_hash="same")
    svc.process_item(second, data=long_text(9).encode(), kb=kb,
                     source_builder=builder, index_documents=None)

    assert second.status is IngestionItemStatus.DUPLICATE
    assert second.duplicate_of == original_doc_id
    assert repo.count_documents(kb.id) == 1, "a duplicate must not create a second document"


def test_resume_is_idempotent_and_skips_completed(corpus):
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    builder = _source_builder(kb.id)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=3)
    for i in range(3):
        item = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=i,
                            file_name=f"f{i}.txt", size=10, content_hash=f"h{i}")
        svc.process_item(item, data=long_text(i).encode(), kb=kb,
                         source_builder=builder, index_documents=_indexer(repo, kb, store))

    docs_before = repo.count_documents(kb.id)
    points_before = len(store.points)

    # Simulate an interruption: every item already COMPLETE.
    pending = svc.resumable_items(repo.get_ingestion_batch(batch.id))
    assert pending == []

    # Running resume changes nothing.
    for item in svc.resumable_items(repo.get_ingestion_batch(batch.id), include_completed=False):
        svc.process_item(item, data=long_text(0).encode(), kb=kb,
                         source_builder=builder, index_documents=_indexer(repo, kb, store),
                         force=False)

    assert repo.count_documents(kb.id) == docs_before, "resume must not duplicate documents"
    assert len(store.points) == points_before, "resume must not duplicate vectors"


def test_resume_retries_only_failed_and_incomplete_items(corpus):
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    builder = _source_builder(kb.id)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=3)

    done = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=0,
                        file_name="ok.txt", size=10, content_hash="h0")
    svc.process_item(done, data=long_text(0).encode(), kb=kb,
                     source_builder=builder, index_documents=None)

    crashed = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=1,
                           file_name="mid.txt", size=10, content_hash="h1")
    crashed.stage = IngestionStage.EMBEDDING   # left mid-flight by a crash
    crashed.status = IngestionItemStatus.PROCESSING
    repo.update_ingestion_item(crashed)

    failed = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=2,
                          file_name="bad.txt", size=10, content_hash="h2")
    svc.process_item(failed, data=b"", kb=kb, source_builder=builder,
                     index_documents=None)

    pending = svc.resumable_items(repo.get_ingestion_batch(batch.id))
    keys = {i.item_key for i in pending}
    assert crashed.item_key in keys, "an item left mid-stage must be resumable"
    assert failed.item_key in keys, "a failed item must be retried"
    assert done.item_key not in keys, "a completed item must be skipped"
    assert crashed.is_incomplete


def test_item_insert_is_idempotent_for_the_same_key(corpus):
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=1)
    a = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=0,
                     file_name="x.txt", size=5, content_hash="hx")
    b = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=0,
                     file_name="x.txt", size=5, content_hash="hx")
    assert a.id == b.id
    assert len(svc.repo.list_ingestion_items(batch.id)) == 1


def test_stage_progress_is_monotonic(corpus):
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=1)
    item = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=0,
                        file_name="p.txt", size=10, content_hash="hp")
    seen = [item.progress]
    svc.process_item(item, data=long_text(3).encode(), kb=kb,
                     source_builder=_source_builder(kb.id),
                     index_documents=_indexer(repo, kb, store))
    seen.append(item.progress)
    assert seen == sorted(seen), "progress must never go backwards"
    assert seen[-1] == 1.0


# ---------------------------------------------------------------------------
# Phase 2 — corpus manifest
# ---------------------------------------------------------------------------

def test_manifest_lists_every_document_with_provenance(corpus):
    repo, kb, store, tmp_path = corpus
    for i in range(3):
        seed_document(repo, kb, store, i, tmp_path)
    manifest = CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False)
    assert manifest.summary.total_documents == 3
    entry = manifest.entries[0]
    for field in ("document_id", "content_hash", "document_version", "parser",
                  "chunk_count", "status", "provenance"):
        assert getattr(entry, field) is not None or field == "error_message"
    assert entry.provenance["user_provided"] is True
    assert manifest.summary.total_chunks > 0


def test_manifest_reports_unknown_vectors_when_store_is_unreadable(corpus, monkeypatch):
    repo, kb, store, tmp_path = corpus
    seed_document(repo, kb, store, 0, tmp_path)

    import app.services.corpus.manifest as manifest_mod

    class DeadStore:
        def collection_info(self, kb_id):
            raise RuntimeError("Qdrant is not reachable")

    monkeypatch.setattr(
        "app.services.vector_store.factory.create_vector_store", lambda *a, **k: DeadStore()
    )
    manifest = CorpusManifestService(repo).build(kb)
    assert manifest.summary.total_vectors == UNKNOWN, (
        "an unreachable vector store must report UNKNOWN, never 0"
    )
    assert manifest.summary.vector_count_confirmed is False


def test_manifest_counts_duplicates_and_superseded_documents(corpus):
    repo, kb, store, tmp_path = corpus
    doc = seed_document(repo, kb, store, 0, tmp_path)
    twin = seed_document(repo, kb, store, 1, tmp_path)
    twin.content_hash = doc.content_hash
    repo.update_document(twin)
    doc.replaces_document_id = None
    # Mark doc as superseded so the manifest can distinguish history from content.
    newer = seed_document(repo, kb, store, 2, tmp_path)
    doc.replaces_document_id = None
    doc.document_version = 1
    repo.update_document(doc)
    newer.replaces_document_id = doc.id
    repo.update_document(newer)

    manifest = CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False)
    assert manifest.summary.duplicate_documents >= 1
    assert manifest.summary.stale_documents >= 1


# ---------------------------------------------------------------------------
# Phase 3 — integrity engine
# ---------------------------------------------------------------------------

def test_healthy_corpus_reports_healthy(corpus):
    repo, kb, store, tmp_path = corpus
    for i in range(2):
        seed_document(repo, kb, store, i, tmp_path)
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert report.overall_status in {"HEALTHY", "WARNING"}
    assert report.orphan_vectors == []
    assert report.missing_vectors == []
    assert report.stale_vectors == []
    assert report.documents_checked == 2


def test_missing_vector_is_detected(corpus):
    repo, kb, store, tmp_path = corpus
    seed_document(repo, kb, store, 0, tmp_path)
    store.points.clear()   # metadata survives, vectors vanish
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert len(report.missing_vectors) == 1
    assert report.missing_vectors[0].document_id is not None
    assert report.missing_vectors[0].chunk_id is not None
    assert report.overall_status == "ERROR"


def test_orphan_vector_is_detected(corpus):
    repo, kb, store, tmp_path = corpus
    doc = seed_document(repo, kb, store, 0, tmp_path)
    # Delete the chunk metadata but leave the point in the store.
    repo.delete_chunks_for_document(kb.id, doc.id)
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert len(report.orphan_vectors) == 1
    assert report.orphan_vectors[0].document_id == doc.id
    assert report.overall_status == "ERROR"


def test_stale_vector_from_superseded_document_is_detected(corpus):
    repo, kb, store, tmp_path = corpus
    old = seed_document(repo, kb, store, 0, tmp_path)
    newer = seed_document(repo, kb, store, 1, tmp_path)
    newer.replaces_document_id = old.id
    repo.update_document(newer)
    # Remove the old document row but keep its points in the store.
    repo._execute("DELETE FROM documents WHERE kb_id = ? AND id = ?", (kb.id, old.id))
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert report.stale_vectors or report.orphan_vectors
    assert any(f.document_id == old.id for f in report.stale_vectors + report.orphan_vectors)


def test_embedding_mismatch_is_detected(corpus):
    repo, kb, store, tmp_path = corpus
    seed_document(repo, kb, store, 0, tmp_path)
    kb.embedding_identity = {"provider": "sentence-transformers", "model": "a-different-model"}
    repo.update_kb(kb)
    for point in store.points.values():
        point["payload"]["embedding_model"] = "some-other-model"
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert len(report.embedding_mismatches) == 1
    assert report.overall_status == "ERROR"


def test_duplicate_documents_and_chunks_are_detected(corpus):
    repo, kb, store, tmp_path = corpus
    a = seed_document(repo, kb, store, 0, tmp_path)
    b = seed_document(repo, kb, store, 1, tmp_path)
    b.content_hash = a.content_hash
    repo.update_document(b)
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert len(report.duplicate_documents) == 1
    assert a.content_hash in report.duplicate_documents[0].details["content_hash"]


def test_failed_and_partial_documents_are_detected(corpus):
    repo, kb, store, tmp_path = corpus
    failed = seed_document(repo, kb, store, 0, tmp_path, status=DocumentStatus.FAILED)
    failed.parse_error = "scanner produced no text"
    repo.update_document(failed)
    partial = seed_document(repo, kb, store, 1, tmp_path, status=DocumentStatus.PARSED)
    assert partial is not None
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert any(f.document_id == failed.id for f in report.failed_documents)
    assert any(f.document_id == partial.id for f in report.partial_documents)
    assert report.overall_status == "ERROR"


def test_provenance_gap_is_detected(corpus):
    repo, kb, store, tmp_path = corpus
    seed_document(repo, kb, store, 0, tmp_path)
    for point in store.points.values():
        point["payload"]["content_hash"] = ""
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert report.provenance_gaps
    assert any("content_hash" in f.message for f in report.provenance_gaps)


def test_broken_source_reference_is_detected(corpus):
    repo, kb, store, tmp_path = corpus
    doc = seed_document(repo, kb, store, 0, tmp_path)
    repo.delete_source(kb.id, doc.source_id)
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert any(f.document_id == doc.id for f in report.broken_source_references)
    assert report.overall_status == "ERROR"


def test_chunking_mismatch_is_detected(corpus):
    repo, kb, store, tmp_path = corpus
    doc = seed_document(repo, kb, store, 0, tmp_path)
    # Re-chunk this document with a different strategy, leaving the mixed set.
    index_documents(
        repo=repo, kb=kb, documents=[doc], store=store,
        embedder=HashingEmbeddingProvider(), chunker_name="fixed-size",
        target_size=600, overlap=80,
    )
    chunks = repo.list_chunks(kb.id, document_id=doc.id)
    strategies = {c.chunking_strategy for c in chunks}
    assert "fixed-size" in strategies
    report = CorpusIntegrityService(repo, _store_for(store)).scan(kb)
    assert report.chunking_mismatches


def test_integrity_scan_is_read_only(corpus):
    """A scan must never repair. Prove it by counting before and after."""
    repo, kb, store, tmp_path = corpus
    for i in range(3):
        seed_document(repo, kb, store, i, tmp_path)
    repo.delete_chunks_for_document(kb.id, repo.list_documents(kb.id)[0].id)

    docs_before = repo.count_documents(kb.id)
    chunks_before = repo.count_chunks(kb.id)
    points_before = len(store.points)

    CorpusIntegrityService(repo, _store_for(store)).scan(kb)

    assert repo.count_documents(kb.id) == docs_before
    assert repo.count_chunks(kb.id) == chunks_before
    assert len(store.points) == points_before, "integrity must not delete vectors"


def test_integrity_reports_unknown_when_store_is_unreadable(corpus, monkeypatch):
    repo, kb, store, tmp_path = corpus
    seed_document(repo, kb, store, 0, tmp_path)

    class DeadStore:
        def all_point_payloads(self, kb_id):
            raise RuntimeError("Qdrant is not reachable")

    monkeypatch.setattr(
        "app.services.vector_store.factory.create_vector_store", lambda *a, **k: DeadStore()
    )
    report = CorpusIntegrityService(repo).scan(kb)
    assert report.vectors_checked == UNKNOWN
    assert "orphan_vectors" in report.unknown_counts
    # Metadata-only checks still ran and still report.
    assert report.documents_checked == 1
    assert report.overall_status == "WARNING", "not HEALTHY: we could not check vectors"


# ---------------------------------------------------------------------------
# Phase 4 — safe repair
# ---------------------------------------------------------------------------

def test_destructive_repair_requires_explicit_confirmation(corpus):
    repo, kb, store, tmp_path = corpus
    seed_document(repo, kb, store, 0, tmp_path)
    svc = CorpusRepairService(
        repo=repo, index_documents=_indexer(repo, kb, store),
        create_store=store, store_factory=_store_for(store),
    )
    plan = svc.plan(kb, RepairAction.REBUILD_KB)
    assert plan.destructive is True
    assert plan.affected_documents
    with pytest.raises(RepairError) as exc:
        svc.execute(kb, plan)
    assert "confirm_action" in str(exc.value)
    # Nothing changed.
    assert repo.count_documents(kb.id) == 1


def test_reindex_document_repair_replaces_vectors(corpus):
    repo, kb, store, tmp_path = corpus
    doc = seed_document(repo, kb, store, 0, tmp_path)
    points_before = len(store.points)
    svc = CorpusRepairService(
        repo=repo, index_documents=_indexer(repo, kb, store),
        create_store=store, store_factory=_store_for(store),
    )
    plan = svc.plan(kb, RepairAction.REINDEX_DOCUMENT, document_ids=[doc.id])
    assert plan.destructive is False
    result = svc.execute(kb, plan)
    assert result.ok
    assert result.chunks_written > 0
    assert result.vectors_removed > 0, "re-index must remove the old vectors"
    assert len(store.points) == points_before, "replace, never duplicate"


def test_repair_plan_reports_unknown_counts_instead_of_zero(corpus):
    repo, kb, store, tmp_path = corpus
    doc = seed_document(repo, kb, store, 0, tmp_path)

    class Unmeasurable:
        """A backend that cannot cheaply answer a count — must not fake zero."""

        def count_points_for_documents(self, kb_id, document_ids):
            raise NotImplementedError

    svc = CorpusRepairService(
        repo=repo, index_documents=_indexer(repo, kb, store),
        create_store=store, store_factory=lambda kb_id: Unmeasurable(),
    )
    plan = svc.plan(kb, RepairAction.REBUILD_DOCUMENT_VECTORS, document_ids=[doc.id])
    assert plan.expected_vectors == UNKNOWN
    assert plan.counts_confirmed is False


def test_reindex_failed_documents_plans_only_failed(corpus):
    repo, kb, store, tmp_path = corpus
    ok = seed_document(repo, kb, store, 0, tmp_path)
    bad = seed_document(repo, kb, store, 1, tmp_path, status=DocumentStatus.FAILED)
    svc = CorpusRepairService(
        repo=repo, index_documents=_indexer(repo, kb, store),
        create_store=store, store_factory=_store_for(store),
    )
    plan = svc.plan(kb, RepairAction.REINDEX_FAILED_DOCUMENTS)
    assert plan.affected_documents == [bad.id]
    assert ok.id not in plan.affected_documents


# ---------------------------------------------------------------------------
# Phase 5/6 — fingerprint, versions, diffs
# ---------------------------------------------------------------------------

def test_fingerprint_is_deterministic_for_identical_corpora(corpus, tmp_path):
    results = []
    for seed in range(2):
        repo = Repository(str(tmp_path / f"fp{seed}.db"))
        kb = KnowledgeBase(id=f"kb_fp{seed}", name="x", domain="d", purpose="p",
                           target_audience="a", depth="technical")
        repo.create_kb(kb)
        store = InMemoryVectorStore()
        for i in range(2):
            seed_document(repo, kb, store, i, tmp_path)
        manifest = CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False)
        results.append(corpus_fingerprint(manifest))
    assert results[0] == results[1], "same documents must give the same fingerprint"


def test_fingerprint_changes_when_content_changes(corpus):
    repo, kb, store, tmp_path = corpus
    doc = seed_document(repo, kb, store, 0, tmp_path)
    before = corpus_fingerprint(CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False))
    doc.content_hash = "a-totally-different-hash"
    repo.update_document(doc)
    after = corpus_fingerprint(CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False))
    assert before != after


def test_fingerprint_is_order_independent(corpus, tmp_path):
    repo1 = Repository(str(tmp_path / "o1.db"))
    repo2 = Repository(str(tmp_path / "o2.db"))
    prints = []
    for i, repo in enumerate((repo1, repo2)):
        kb = KnowledgeBase(id=f"kb_o{i}", name="x", domain="d", purpose="p",
                           target_audience="a", depth="technical")
        repo.create_kb(kb)
        store = InMemoryVectorStore()
        order = [0, 1] if i == 0 else [1, 0]
        for n in order:
            seed_document(repo, kb, store, n, tmp_path)
        prints.append(corpus_fingerprint(CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False)))
    assert prints[0] == prints[1]


def test_snapshot_reuses_version_for_unchanged_corpus(corpus):
    repo, kb, store, tmp_path = corpus
    seed_document(repo, kb, store, 0, tmp_path)
    manifest = CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False)
    first = snapshot_corpus_version(repo, kb, manifest)
    second = snapshot_corpus_version(repo, kb, manifest)
    assert first.id == second.id, "an unchanged corpus must not grow the history"
    assert len(repo.list_corpus_versions(kb.id)) == 1


def test_snapshot_creates_a_new_version_when_the_corpus_changes(corpus):
    repo, kb, store, tmp_path = corpus
    seed_document(repo, kb, store, 0, tmp_path)
    snapshot_corpus_version(repo, kb, CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False))
    seed_document(repo, kb, store, 1, tmp_path)
    second = snapshot_corpus_version(repo, kb, CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False))
    assert second.version == "v2"
    assert len(repo.list_corpus_versions(kb.id)) == 2


def test_corpus_version_records_identity(corpus):
    repo, kb, store, tmp_path = corpus
    kb.embedding_identity = {"provider": "sentence-transformers", "model": "MiniLM", "dimensions": 384}
    repo.update_kb(kb)
    seed_document(repo, kb, store, 0, tmp_path)
    version = snapshot_corpus_version(repo, kb, CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False))
    assert version.fingerprint
    assert version.document_count == 1
    assert version.embedding_identity != "unknown"
    assert version.chunking_identity != "unknown"


def test_diff_reports_added_removed_changed_and_unchanged(corpus):
    """A diff must be computed against the same corpus, mutated in place."""
    repo, kb, store, tmp_path = corpus
    manifest_service = CorpusManifestService(repo, _store_for(store))
    keep = seed_document(repo, kb, store, 0, tmp_path)
    change = seed_document(repo, kb, store, 1, tmp_path)
    drop = seed_document(repo, kb, store, 2, tmp_path)
    old_manifest = manifest_service.build(kb)

    # 1. change a document's content
    change.content_hash = "a-revised-content-hash"
    repo.update_document(change)
    # 2. delete a document
    repo.delete_document(kb.id, drop.id)
    # 3. add a document
    seed_document(repo, kb, store, 3, tmp_path)

    new_manifest = manifest_service.build(kb)
    diff = diff_manifests(old_manifest, new_manifest)

    assert [c.document_id for c in diff.changed] == [change.id]
    assert [c.document_id for c in diff.removed] == [drop.id]
    assert [c.document_id for c in diff.unchanged] == [keep.id]
    assert len(diff.added) == 1
    assert diff.identical is False

    changed = diff.changed[0]
    assert changed.old_content_hash != changed.new_content_hash
    assert changed.old_chunk_count is not None and changed.new_chunk_count is not None
    assert changed.vectors_invalidated >= 0
    assert changed.vectors_created >= 0


def test_diff_of_identical_corpora_is_identical(corpus):
    repo, kb, store, tmp_path = corpus
    for i in range(2):
        seed_document(repo, kb, store, i, tmp_path)
    manifest = CorpusManifestService(repo, _store_for(store)).build(kb, with_vector_count=False)
    diff = diff_manifests(manifest, manifest)
    assert diff.identical is True
    assert not diff.added and not diff.removed and not diff.changed


def test_chunking_config_hash_is_stable_and_sensitive():
    a = chunking_config_hash("section-aware", {"target_size": 1200, "overlap": 150})
    b = chunking_config_hash("section-aware", {"overlap": 150, "target_size": 1200})
    assert a == b, "key order must not matter"
    assert a != chunking_config_hash("section-aware", {"target_size": 900, "overlap": 150})
    assert a != chunking_config_hash("fixed-size", {"target_size": 1200, "overlap": 150})

def test_failed_document_keeps_its_bytes_so_it_can_be_retried(corpus):
    """A parse failure must not lose the file: retry reads it from disk."""
    repo, kb, store, tmp_path = corpus
    svc = _service(repo, tmp_path)
    batch = svc.create_batch(kb_id=kb.id, source_mode="user_provided", total_items=1)
    item = svc.add_item(batch_id=batch.id, kb_id=kb.id, index=0,
                        file_name="empty.docx", size=8, content_hash="hx")

    # A ZIP that passes validation (it does contain word/document.xml) but
    # breaks the parser: the failure happens AFTER the bytes are stored.
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<this-is-not-valid-xml")
    svc.process_item(item, data=buf.getvalue(), kb=kb,
                     source_builder=_source_builder(kb.id), index_documents=None)

    assert item.status is IngestionItemStatus.FAILED
    assert item.error_code == "PARSE_FAILED"
    doc = repo.list_documents(kb.id)[0]
    assert doc.status is DocumentStatus.FAILED
    assert doc.raw_file_path, "the original bytes must be retained for a retry"
    assert Path(doc.raw_file_path).exists()
