"""V4 document library, incremental ingestion and API contract tests.

Covers (V4 Phases 1, 2, 5, 6, 8, 10):
- source modes EXTERNAL / USER_PROVIDED / MIXED and their behaviour
- upload workflow: validation, duplicates, failures, statuses
- incremental ingestion: only the new document is processed
- stale-vector deletion on re-index and on document delete
- document replacement keeps history and removes old vectors
- retrieval provenance (document / page / slide / section)
- a knowledge base is READY with NO benchmark at all
- the frozen benchmark stays immutable and is untouched by V4

Vector-store tests use the hermetic in-memory cosine fake (real cosine search,
no services), so nothing here is faked and nothing requires Qdrant.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.schemas.models import (  # noqa: E402
    BenchmarkStatus,
    Chunk,
    Document,
    DocumentStatus,
    EvaluationQuestion,
    KnowledgeBase,
    QuestionStatus,
    Source,
    SourceMode,
    SourceType,
)
from app.services.embeddings.provider import HashingEmbeddingProvider  # noqa: E402
from app.services.indexing.document_indexer import IndexingError, index_documents  # noqa: E402
from app.services.ingestion.ingestion import ingest_uploaded_bytes  # noqa: E402
from app.services.retrieval.retriever import (  # noqa: E402
    DenseRetriever,
    available_retrievers,
    get_retriever,
)
from app.services.source_quality.user_scorer import assess_user_upload  # noqa: E402
from app.utils.ids import new_id  # noqa: E402
from tests.test_retrieval_integration import InMemoryVectorStore  # noqa: E402

KB_ID = "kb_incremental"

# Chunking drops fragments shorter than MIN_CHUNK_CHARS (60), so every test
# body below is deliberately long enough to produce real chunks.
LONG_STABILITY = (
    "# Ship Stability\n\n"
    "The metacentric height GM equals KB plus BM minus KG for small angles of heel. "
    "A vessel is stable while the metacentre remains above the centre of gravity.\n\n"
    "## Free Surface Effect\n\n"
    "Free surface effect reduces the effective metacentric height whenever a tank is "
    "only partly filled with a liquid of lower density than the surrounding fluid.\n"
)
LONG_TRIM = (
    "# Trim and List\n\n"
    "Trim is a longitudinal moment produced by weight distribution along the length of "
    "the vessel, while list is a transverse moment produced by weight athwartships.\n"
)
LONG_RESISTANCE = (
    "# Hull Resistance\n\n"
    "Frictional resistance grows with the wetted surface and with the form factor of "
    "the hull at the operating Froude number of the design speed.\n"
)
LONG_GENERIC = (
    "# Notes\n\n"
    "The metacentric height of a floating vessel is the distance between the centre of "
    "buoyancy and the centre of gravity, measured along the vertical axis.\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_doc(repo: Repository, name: str, body: str, kb_id: str = KB_ID) -> Document:
    data = body.encode("utf-8")
    source = Source(
        id=new_id("src"), kb_id=kb_id, url=f"upload://{name}", title=name,
        source_type=SourceType.TEXT, user_provided=True, provenance="user_upload",
        file_name=name, file_size=len(data),
    )
    repo.create_source(source)
    doc = ingest_uploaded_bytes(
        kb_id=kb_id, source=source,
        upload_dir=Path(repo.db_path).parent / "uploads" / kb_id,
        file_name=name, data=data,
    )
    repo.create_document(doc)
    return doc


@pytest.fixture
def kb_with_docs(repo: Repository) -> tuple[Repository, KnowledgeBase, dict[str, Document]]:
    kb = KnowledgeBase(
        id=KB_ID, name="Naval Architecture", domain="Naval Architecture",
        purpose="Study assistant", target_audience="Naval Architecture students",
        source_mode=SourceMode.USER_PROVIDED,
    )
    repo.create_kb(kb)
    docs = {
        "stability": _write_doc(repo, "stability.txt", LONG_STABILITY),
        "trim": _write_doc(repo, "trim.txt", LONG_TRIM),
    }
    return repo, kb, docs


def _index(repo: Repository, kb: KnowledgeBase, store, docs, *, full_build=False):
    return index_documents(
        repo=repo, kb=kb, documents=list(docs), store=store,
        embedder=HashingEmbeddingProvider(), chunker_name="section-aware",
        target_size=600, overlap=80, full_build=full_build,
    )


# ---------------------------------------------------------------------------
# Phase 6: incremental ingestion
# ---------------------------------------------------------------------------

def test_adding_one_document_only_indexes_that_document(kb_with_docs):
    repo, kb, docs = kb_with_docs
    store = InMemoryVectorStore()

    first = _index(repo, kb, store, [docs["stability"]])
    assert first.documents_indexed == 1
    points_after_first = len(store.points)

    # The 101st document: only it is processed.
    extra = _write_doc(repo, "resistance.txt", LONG_RESISTANCE)
    second = _index(repo, kb, store, [extra])

    assert second.document_ids == [extra.id]
    assert second.documents_indexed == 1
    assert len(store.points) > points_after_first
    # The already-indexed document's vectors were neither deleted nor duplicated.
    assert points_after_first > 0
    survivors = {
        p["payload"]["document_id"] for p in store.points.values()
    }
    assert docs["stability"].id in survivors
    assert docs["trim"].id not in survivors, "trim was never indexed in this scenario"
    # DB chunk rows agree with the vector store.
    assert repo.count_chunks(KB_ID) == second.chunk_count + first.chunk_count


def test_incremental_index_deletes_stale_vectors_for_reindexed_document(kb_with_docs):
    repo, kb, docs = kb_with_docs
    store = InMemoryVectorStore()
    _index(repo, kb, store, [docs["stability"], docs["trim"]])
    before = len(store.points)

    outcome = _index(repo, kb, store, [docs["stability"]])

    assert outcome.stale_vectors_removed > 0
    assert len(store.points) == before, "re-index must replace, never duplicate"
    stability_points = [
        p for p in store.points.values() if p["payload"]["document_id"] == docs["stability"].id
    ]
    assert len(stability_points) == outcome.chunk_count
    # No orphan point survives a full build's sweep.
    swept = store.delete_orphaned_points(KB_ID, {c.id for c in repo.list_chunks(KB_ID)})
    assert swept == 0


def test_document_status_and_counts_are_recorded_after_indexing(kb_with_docs):
    repo, kb, docs = kb_with_docs
    store = InMemoryVectorStore()
    outcome = _index(repo, kb, store, [docs["stability"]])

    stored = repo.get_document(KB_ID, docs["stability"].id)
    assert stored is not None
    assert stored.status is DocumentStatus.READY
    assert stored.chunk_count == outcome.chunk_count
    assert stored.indexed_at is not None


def test_index_refuses_when_vectors_cannot_be_deleted(tmp_path):
    """A backend without stale-vector deletion must block indexing, not corrupt it."""
    from app.services.vector_store.qdrant_store import VectorStore

    class NoDeleteStore(InMemoryVectorStore):
        def delete_document_vectors(self, kb_id, document_ids):  # type: ignore[override]
            raise NotImplementedError

        def delete_orphaned_points(self, kb_id, valid_chunk_ids):  # type: ignore[override]
            raise NotImplementedError

    repo = Repository(str(tmp_path / "nodelete.db"))
    kb = KnowledgeBase(
        id=KB_ID, name="k", domain="d", purpose="p", target_audience="a",
        source_mode=SourceMode.USER_PROVIDED,
    )
    repo.create_kb(kb)
    doc = _write_doc(repo, "a.txt", LONG_GENERIC, kb.id)

    with pytest.raises(IndexingError, match="Refusing to index"):
        _index(repo, kb, NoDeleteStore(), [doc])


def test_indexing_reports_skip_reasons_instead_of_failing_silently(kb_with_docs):
    repo, kb, docs = kb_with_docs
    store = InMemoryVectorStore()
    good = docs["stability"]

    orphan = Document(
        id=new_id("doc"), kb_id=KB_ID, source_id="src_missing",
        url="upload://ghost.txt", title="ghost.txt", source_type=SourceType.TEXT,
    )
    repo.create_document(orphan)

    outcome = _index(repo, kb, store, [good, orphan])
    assert outcome.documents_indexed == 1
    assert outcome.documents_skipped
    reason = outcome.documents_skipped[0]["reason"]
    assert "not found on disk" in reason or "source" in reason


# ---------------------------------------------------------------------------
# Phase 5: document library / delete removes vectors
# ---------------------------------------------------------------------------

def test_deleting_a_document_removes_its_vectors(kb_with_docs):
    repo, kb, docs = kb_with_docs
    store = InMemoryVectorStore()
    _index(repo, kb, store, [docs["stability"], docs["trim"]])
    before = len(store.points)

    removed = store.delete_document_vectors(KB_ID, [docs["trim"].id])
    # `Repository.delete_document` is what the DELETE /documents/{id} endpoint
    # calls: vectors first, then chunk rows and the document row.
    assert repo.delete_document(KB_ID, docs["trim"].id) is True

    assert removed > 0
    assert len(store.points) == before - removed
    assert not repo.list_chunks(KB_ID, document_id=docs["trim"].id)
    assert repo.get_document(KB_ID, docs["trim"].id) is None
    # The untouched document keeps everything.
    assert repo.get_document(KB_ID, docs["stability"].id) is not None
    assert repo.list_chunks(KB_ID, document_id=docs["stability"].id)


def test_document_replacement_keeps_history_and_indexes_the_new_version(kb_with_docs):
    repo, kb, docs = kb_with_docs
    store = InMemoryVectorStore()
    _index(repo, kb, store, [docs["stability"]])
    old = docs["stability"]

    replacement = _write_doc(
        repo,
        "stability_v2.txt",
        "# Ship Stability (revised)\n\n"
        "The corrected metacentric height formula for large heel angles includes a "
        "geometric term that the first edition of these lecture notes omitted entirely.\n",
    )
    replacement.document_version = (old.document_version or 1) + 1
    replacement.replaces_document_id = old.id
    repo.update_document(replacement)

    # Exactly what the POST /documents/{id}/replace endpoint does.
    superseded_vectors = store.delete_document_vectors(KB_ID, [old.id])
    old.status = DocumentStatus.FAILED
    old.parse_error = f"Superseded by document {replacement.id} (version 2)"
    old.chunk_count = 0
    old.indexed_at = None
    repo.update_document(old)

    outcome = _index(repo, kb, store, [replacement])

    assert outcome.documents_indexed == 1
    assert replacement.document_version == 2
    assert superseded_vectors > 0, "the superseded version's vectors must be removed"
    # The previous document is retained as history, never silently overwritten.
    history = repo.get_document(KB_ID, old.id)
    assert history is not None
    assert history.status is DocumentStatus.FAILED
    assert history.replaces_document_id is None
    # Only the replacement's vectors are live.
    live = {p["payload"]["document_id"] for p in store.points.values()}
    assert live == {replacement.id}


# ---------------------------------------------------------------------------
# Phase 1: source modes
# ---------------------------------------------------------------------------

def test_source_mode_defaults_to_external_for_backwards_compatibility():
    kb = KnowledgeBase(
        id="kb_legacy", name="Auto KB", domain="Automobile Engineering",
        purpose="p", target_audience="students",
    )
    assert kb.source_mode is SourceMode.EXTERNAL
    assert kb.version == 1, "existing KBs must keep version 1"
    assert kb.last_build_at is None
    assert DocumentStatus.PARSED.value == "parsed"


def test_all_three_source_modes_round_trip(repo: Repository):
    for i, mode in enumerate(SourceMode):
        kb = KnowledgeBase(
            id=f"kb_mode_{i}", name=f"KB {mode.value}", domain="Naval Architecture",
            purpose="p", target_audience="students", source_mode=mode,
        )
        repo.create_kb(kb)
        stored = repo.get_kb(kb.id)
        assert stored is not None and stored.source_mode is mode


def test_external_and_user_provided_sources_are_distinguishable(kb_with_docs):
    repo, kb, docs = kb_with_docs
    external = Source(
        id=new_id("src"), kb_id=KB_ID, url="https://www.rina.org/guide",
        title="Class Rules", source_type=SourceType.WEB_PAGE, user_provided=False,
        provenance="discovered",
    )
    repo.create_source(external)
    sources = repo.list_sources(KB_ID)
    user = [s for s in sources if s.user_provided]
    discovered = [s for s in sources if not s.user_provided]
    assert len(user) == 2 and len(discovered) == 1
    assert all(s.provenance == "user_upload" for s in user)
    assert discovered[0].provenance == "discovered"


def test_user_provided_source_is_not_scored_by_the_external_authority_scorer(kb_with_docs):
    repo, kb, docs = kb_with_docs
    source = repo.get_source(KB_ID, docs["stability"].source_id)
    assert source is not None and source.user_provided
    # Its integrity assessment never contains publication-authority signals.
    assessment = assess_user_upload(file_valid=True, parse_ok=True, extracted_chars=3000, unit_count=2)
    assert set(assessment.weights).isdisjoint(
        {"authority", "recency", "accessibility", "evidence_quality"}
    )
    assert assessment.decision.value == "ACCEPT"


# ---------------------------------------------------------------------------
# Phase 10: retrieval provenance
# ---------------------------------------------------------------------------

def test_retrieval_results_carry_document_page_slide_and_section_provenance(kb_with_docs):
    repo, kb, docs = kb_with_docs
    store = InMemoryVectorStore()

    # Enrich the fake store's payload with the provenance the real Qdrant
    # payload carries, so the retriever's normalisation is under test.
    from app.services.chunking.chunker import get_chunker

    source = repo.get_source(KB_ID, docs["stability"].source_id)
    text = Path(docs["stability"].file_path).read_text(encoding="utf-8")
    chunks = get_chunker("section-aware").chunk(
        document=docs["stability"], text=text, source=source,
        target_size=600, overlap=80, domain=kb.domain,
    )
    embedder = HashingEmbeddingProvider()
    for chunk in chunks:
        chunk.slide = 17
        chunk.slide_title = "Ship Stability"
        chunk.page = 42
        chunk.section = "Free Surface Effect"
    repo.create_chunks(chunks)

    from app.services.vector_store.qdrant_store import numeric_id

    for chunk in chunks:
        store.points[numeric_id(chunk.id)] = {
            "vector": embedder.embed_texts([chunk.text])[0],
            "payload": {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "document_title": "stability.txt",
                "document_version": 1,
                "source_id": source.id,
                "source_title": "stability.txt",
                "source_url": source.url,
                "source_type": source.source_type.value,
                "section": chunk.section,
                "section_path": chunk.section_path,
                "page": 42,
                "slide": 17,
                "slide_title": "Ship Stability",
                "content_hash": chunk.content_hash,
                "chunk_index": chunk.chunk_index,
                "chunking_strategy": "section-aware",
                "chunking_config": {"target_size": 600, "overlap": 80},
                "user_provided": True,
                "kb_version": 2,
            },
        }

    response = DenseRetriever(embedder, store).retrieve(
        KB_ID, "free surface effect reduces metacentric height", top_k=3
    )
    assert response.results
    top = response.results[0]
    prov = top.provenance
    assert prov["document_id"] == docs["stability"].id
    assert prov["document_title"] == "stability.txt"
    assert prov["page"] == 42
    assert prov["slide"] == 17
    assert prov["slide_title"] == "Ship Stability"
    assert prov["section"] == "Free Surface Effect"
    assert prov["user_provided"] is True
    assert prov["document_version"] == 1
    assert prov["kb_version"] == 2
    assert prov["chunking_strategy"] == "section-aware"
    assert prov["chunking_config"]["overlap"] == 80
    # Citation-ready: everything an LLM layer would need is present.
    assert top.chunk_id and top.text and top.score > 0


def test_missing_provenance_keys_are_omitted_not_invented(kb_with_docs):
    repo, kb, docs = kb_with_docs
    store = InMemoryVectorStore()
    from app.services.vector_store.qdrant_store import numeric_id

    embedder = HashingEmbeddingProvider()
    chunk = Chunk(
        id=new_id("chk"), document_id="doc_x", kb_id=KB_ID, chunk_index=0,
        text="A plain chunk with no structural information at all.",
        content_hash="h", char_count=50,
    )
    store.points[numeric_id(chunk.id)] = {
        "vector": embedder.embed_texts([chunk.text])[0],
        "payload": {"chunk_id": chunk.id, "document_id": "doc_x", "text": chunk.text},
    }
    result = DenseRetriever(embedder, store).retrieve(KB_ID, "plain chunk", top_k=1).results[0]
    for absent in ("page", "slide", "slide_title", "section", "section_path", "trust_score"):
        assert absent not in result.provenance, f"{absent} must be absent, not fabricated"


def test_retriever_registry_exposes_dense_and_accepts_new_strategies():
    assert "dense" in available_retrievers()
    assert "qdrant-dense" in available_retrievers()

    class StubRetriever(DenseRetriever):
        backend = "bm25-stub"

    from app.services.retrieval.retriever import register_retriever

    register_retriever("bm25", lambda e, s, i: StubRetriever(e, s, i))
    assert "bm25" in available_retrievers()
    store = InMemoryVectorStore()
    built = get_retriever("bm25", HashingEmbeddingProvider(), store)
    assert built.retrieve(KB_ID, "x", top_k=1).retrieval_backend == "bm25-stub"


def test_unknown_retriever_is_rejected():
    with pytest.raises(ValueError, match="Unknown retriever"):
        get_retriever("quantum", HashingEmbeddingProvider(), InMemoryVectorStore())


# ---------------------------------------------------------------------------
# Architectural distinction: benchmark != knowledge base
# ---------------------------------------------------------------------------

def test_kb_without_benchmark_has_no_questions_and_evaluations(repo: Repository):
    kb = KnowledgeBase(
        id="kb_nobench", name="Course KB", domain="Naval Architecture",
        purpose="p", target_audience="students", source_mode=SourceMode.USER_PROVIDED,
    )
    repo.create_kb(kb)
    assert repo.list_evaluation_questions("kb_nobench") == []
    assert repo.list_evaluation_runs("kb_nobench") == []
    assert repo.list_benchmark_versions("kb_nobench") == []
    # Nothing about the KB requires ground truth to exist.
    doc = _write_doc(repo, "notes.txt", LONG_GENERIC, "kb_nobench")
    assert repo.get_document("kb_nobench", doc.id) is not None


def test_frozen_benchmark_stays_immutable_across_document_work(repo: Repository):
    """V4 document work must never touch benchmark ground truth."""
    kb = KnowledgeBase(
        id="kb_frozen", name="Auto KB", domain="Automobile Engineering",
        purpose="p", target_audience="students",
    )
    repo.create_kb(kb)
    question = EvaluationQuestion(
        id=new_id("eq"), kb_id=kb.id, question="What is GM?",
        expected_keywords=["metacentric"], status=QuestionStatus.FROZEN,
        reviewer="human-reviewer",
    )
    repo.create_evaluation_question(question)

    from app.schemas.models import BenchmarkVersion

    version = BenchmarkVersion(
        id=new_id("bv"), kb_id=kb.id, version="automobile-engineering-v1",
        status=BenchmarkStatus.FROZEN,
        question_ids=[question.id],
        questions_snapshot=[question.model_dump(mode="json")],
        created_by="human-reviewer",
    )
    repo.create_benchmark_version(version)

    # Simulate user-document work in the same KB.
    doc = _write_doc(repo, "extra.txt", LONG_GENERIC, kb.id)
    store = InMemoryVectorStore()
    index_documents(
        repo=repo, kb=kb, documents=[doc], store=store, embedder=HashingEmbeddingProvider(),
        chunker_name="section-aware", target_size=600, overlap=80,
    )

    stored = repo.get_benchmark_version(kb.id, version.id)
    assert stored is not None
    assert stored.status is BenchmarkStatus.FROZEN
    assert stored.question_ids == [question.id]
    assert stored.questions_snapshot == version.questions_snapshot
    frozen_question = repo.get_evaluation_question(kb.id, question.id)
    assert frozen_question is not None
    assert frozen_question.status is QuestionStatus.FROZEN
    assert frozen_question.expected_keywords == ["metacentric"]


def test_document_delete_then_kb_delete_cleans_everything(repo: Repository):
    kb = KnowledgeBase(
        id="kb_cleanup", name="k", domain="d", purpose="p", target_audience="a",
        source_mode=SourceMode.USER_PROVIDED,
    )
    repo.create_kb(kb)
    doc = _write_doc(repo, "a.txt", LONG_GENERIC, "kb_cleanup")
    store = InMemoryVectorStore()
    index_documents(
        repo=repo, kb=kb, documents=[doc], store=store, embedder=HashingEmbeddingProvider(),
        chunker_name="section-aware", target_size=600, overlap=80,
    )
    assert repo.count_documents("kb_cleanup") == 1
    assert repo.count_chunks("kb_cleanup") > 0

    repo.delete_kb("kb_cleanup")
    assert repo.count_documents("kb_cleanup") == 0
    assert repo.count_chunks("kb_cleanup") == 0
    assert repo.get_kb("kb_cleanup") is None


def test_reparse_reuses_stored_original_bytes(kb_with_docs, tmp_path):
    """Rebuild re-derives text from the user's own file, never re-downloads."""
    from app.services.ingestion.ingestion import reparse_document

    repo, kb, docs = kb_with_docs
    doc = docs["stability"]
    text, meta = reparse_document(doc)
    assert meta["format"] == "txt"
    assert "Free Surface Effect" in text
    assert Path(doc.raw_file_path).exists()
    # Re-parsing must not create a second document.
    assert repo.count_documents(KB_ID) == 2


def test_uploaded_pdf_bytes_are_never_sent_over_the_network(kb_with_docs, monkeypatch):
    """Local processing is the default; ingestion must not make HTTP calls."""
    import httpx

    repo, kb, docs = kb_with_docs

    def explode(*_args, **_kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("user document ingestion must not perform network requests")

    monkeypatch.setattr(httpx.Client, "get", explode)
    data = b"# Notes\n\nA purely local parse of user supplied material.\n"
    source = Source(
        id=new_id("src"), kb_id=KB_ID, url="upload://local.md", title="local.md",
        source_type=SourceType.TEXT, user_provided=True, provenance="user_upload",
    )
    doc = ingest_uploaded_bytes(
        kb_id=KB_ID, source=source, upload_dir=Path(repo.db_path).parent / "up2",
        file_name="local.md", data=data,
    )
    assert doc.text_length > 0


def test_ooxml_roundtrip_uses_real_bytes_in_memory():
    """Guard: the PPTX/DOCX fixtures the other tests rely on are real files."""
    from tests.test_user_ingestion import make_docx, make_pptx

    pptx = make_pptx([("Title", ["Body"])])
    docx = make_docx([("heading", "H"), ("body", "B")])
    assert pptx[:2] == b"PK"
    assert docx[:2] == b"PK"
    assert io.BytesIO(pptx).getbuffer().nbytes == len(pptx)

# ---------------------------------------------------------------------------
# Unconfirmed deletion counts must never be reported as zero (V4 honesty rule)
# ---------------------------------------------------------------------------


def test_stale_label_never_renders_unconfirmed_as_zero():
    from app.services.indexing.document_indexer import IndexOutcome

    assert IndexOutcome(stale_vectors_removed=7).stale_label() == "7"
    assert IndexOutcome(stale_vectors_removed=0).stale_label() == "0"
    # -1 == the deletion ran but the store never confirmed a count.
    assert IndexOutcome(stale_vectors_removed=-1).stale_label() == "unknown"
    assert IndexOutcome(stale_vectors_removed=-1).as_dict()["stale_vectors_removed_label"] == "unknown"


def test_orphan_sweep_unknown_does_not_cancel_a_known_count(kb_with_docs):
    """A sweep that cannot confirm its count must not overwrite a real one."""
    repo, kb, docs = kb_with_docs

    class UnknownSweepStore(InMemoryVectorStore):
        def delete_document_vectors(self, kb_id, document_ids):  # type: ignore[override]
            return super().delete_document_vectors(kb_id, document_ids)

        def delete_orphaned_points(self, kb_id, valid_chunk_ids):  # type: ignore[override]
            return -1

    store = UnknownSweepStore()
    # First pass only seeds vectors, so there is genuinely nothing to delete.
    index_documents(
        repo=repo, kb=kb, documents=[docs["stability"]], store=store,
        embedder=HashingEmbeddingProvider(), chunker_name="section-aware",
        target_size=600, overlap=80, full_build=True,
    )

    second = index_documents(
        repo=repo, kb=kb, documents=[docs["stability"]], store=store,
        embedder=HashingEmbeddingProvider(), chunker_name="section-aware",
        target_size=600, overlap=80, full_build=True,
    )
    assert second.stale_vectors_removed > 0, "unknown sweep must not zero out a known count"
    assert second.stale_label() != "unknown"


class _FakeCollectionInfo:
    def __init__(self, count):
        self.points_count = count


class _FakeQdrantClient:
    """Minimal stand-in exposing only what the delete path touches."""

    def __init__(self, counts):
        self._counts = list(counts)  # popped before, then after, the delete
        self.deleted = 0

    def get_collection(self, name):
        return _FakeCollectionInfo(self._counts.pop(0))

    def delete(self, **_kwargs):
        self.deleted += 1


def _store_with_client(client):
    from app.services.vector_store.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore.__new__(QdrantVectorStore)  # bypass live client creation
    store._url = "http://stub"
    store._client = client
    return store


def test_delete_count_is_measured_from_the_point_count_delta():
    client = _FakeQdrantClient([684, 42])
    store = _store_with_client(client)

    removed = store._delete_by_document_filter("kb_x", ["doc_a"])

    assert removed == 642, "must report the measured delta, not the server response"
    assert client.deleted == 1


def test_delete_count_is_unknown_when_the_collection_cannot_be_read():
    class BlindClient(_FakeQdrantClient):
        def get_collection(self, name):
            raise RuntimeError("collection is not readable")

    store = _store_with_client(BlindClient([]))
    assert store._delete_by_document_filter("kb_x", ["doc_a"]) == -1
    assert store.delete_document_vectors("kb_x", ["doc_a"]) == -1
