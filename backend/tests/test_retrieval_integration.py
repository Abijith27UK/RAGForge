"""Retrieval + indexing integration tests.

Two layers:
1. Hermetic: an in-memory VectorStore fake with real cosine search, driven by
   the deterministic hashing embedding provider. Always runs, no services.
2. Live: the same scenario against real Qdrant on localhost:6333 when it is
   reachable; skips with an honest reason otherwise (never fakes a pass).
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from app.repositories.sqlite_repo import Repository
from app.schemas.models import (
    Chunk,
    Document,
    EvaluationQuestion,
    EvaluationRunConfig,
    Source,
    SourceType,
)
from app.services.embeddings.provider import HashingEmbeddingProvider
from app.services.evaluation.evaluator import Evaluator
from app.services.retrieval.retriever import DenseRetriever
from app.services.vector_store.qdrant_store import (
    QdrantVectorStore,
    VectorStore,
    VectorStoreError,
    numeric_id,
)
from app.utils.ids import new_id


# ---------------------------------------------------------------------------
# In-memory vector store fake (real cosine similarity, no services)
# ---------------------------------------------------------------------------

class InMemoryVectorStore(VectorStore):
    """Cosine-similarity fake of the Qdrant store for hermetic tests."""

    backend_name = "in-memory-cosine"

    def __init__(self) -> None:
        self.points: dict[int, dict[str, Any]] = {}

    def ensure_collection(self, kb_id: str, vector_size: int, distance: str = "cosine") -> None:
        return None

    def upsert_chunks(self, kb_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for c, v in zip(chunks, vectors):
            self.points[numeric_id(c.id)] = {
                "vector": v,
                # Mirrors the REAL Qdrant payload (same keys as
                # QdrantVectorStore.upsert_chunks) so dense-path provenance is as
                # rich here as in production. A thinner fake would let provenance
                # regressions pass the test suite unnoticed.
                "payload": {
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "kb_id": c.kb_id,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "source_id": c.source_id,
                    "source_url": c.source_url,
                    "source_title": c.source_title,
                    "source_type": c.source_type,
                    "publisher": c.publisher,
                    "document_title": c.document_title,
                    "section": c.section,
                    "section_path": c.section_path,
                    "page": c.page,
                    "slide": c.slide,
                    "slide_title": c.slide_title,
                    "domain": c.domain,
                    "subdomain": c.subdomain,
                    "trust_score": c.trust_score,
                    "content_hash": c.content_hash,
                    "char_count": c.char_count,
                    "chunking_strategy": c.chunking_strategy,
                    "chunking_config": c.chunking_config,
                    "document_version": c.document_version,
                    "user_provided": c.user_provided,
                    "kb_version": c.kb_version,
                },
            }

    def delete_document_vectors(self, kb_id: str, document_ids: list[str]) -> int:
        ids = set(document_ids)
        stale = [pid for pid, p in self.points.items() if p["payload"]["document_id"] in ids]
        for pid in stale:
            del self.points[pid]
        return len(stale)

    def delete_orphaned_points(self, kb_id: str, valid_chunk_ids: set[str]) -> int:
        stale = [pid for pid, p in self.points.items() if p["payload"].get("chunk_id") not in valid_chunk_ids]
        for pid in stale:
            del self.points[pid]
        return len(stale)

    def search(self, kb_id: str, vector: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return dot / (na * nb)

        scored = [(cosine(vector, p["vector"]), p) for p in self.points.values()]
        scored.sort(key=lambda t: -t[0])
        return [
            {"chunk_id": p["payload"]["chunk_id"], "score": s, "payload": dict(p["payload"])}
            for s, p in scored[:top_k]
        ]

    def collection_info(self, kb_id: str) -> dict[str, Any]:
        return {"name": kb_id, "points_count": len(self.points), "vector_size": 0, "status": "green"}

    def count_points_for_documents(self, kb_id: str, document_ids: list[str]) -> int:
        ids = set(document_ids)
        return sum(1 for p in self.points.values() if p["payload"].get("document_id") in ids)

    def all_point_payloads(self, kb_id: str) -> dict[str, dict[str, Any]]:
        return {
            str(p["payload"]["chunk_id"]): dict(p["payload"])
            for p in self.points.values()
            if p["payload"].get("chunk_id")
        }

    def fetch_vectors(self, kb_id: str, chunk_ids: list[str]) -> dict[str, list[float]]:
        """Stored vectors by chunk id (mirrors the Qdrant capability; V6 MMR)."""
        wanted = set(chunk_ids)
        return {
            str(p["payload"]["chunk_id"]): list(p["vector"])
            for p in self.points.values()
            if p["payload"].get("chunk_id") in wanted
        }

    def delete_collection(self, kb_id: str) -> None:
        self.points.clear()


# ---------------------------------------------------------------------------
# Fixtures: one KB with two documents, three chunks each
# ---------------------------------------------------------------------------

TEXTS = {
    "docA": [
        "The battery management system monitors cell voltage and temperature.",
        "Regenerative braking recovers kinetic energy during deceleration.",
        "Lithium-ion cells degrade faster at high state of charge.",
    ],
    "docB": [
        "MacPherson strut suspension is common in compact passenger cars.",
        "Double wishbone suspension offers better camber control in corners.",
        "Anti-roll bars reduce body roll during cornering maneuvers.",
    ],
}


@pytest.fixture
def indexed_env(repo: Repository):
    """Repository with two ingested documents and chunks upserted into the fake store."""
    kb_id = "kb_itest"
    embedder = HashingEmbeddingProvider()
    store = InMemoryVectorStore()
    all_chunks: list[Chunk] = []
    for doc_name, texts in TEXTS.items():
        source = Source(id=new_id("src"), kb_id=kb_id, url=f"https://example.com/{doc_name}", title=doc_name, source_type=SourceType.TEXT, trust_score=0.9)
        repo.create_source(source)
        doc = Document(id=new_id("doc"), kb_id=kb_id, source_id=source.id, url=source.url, title=doc_name, source_type=SourceType.TEXT, content_hash=f"hash-{doc_name}")
        repo.create_document(doc)
        chunks = [
            Chunk(id=new_id("chk"), document_id=doc.id, kb_id=kb_id, chunk_index=i, text=t, content_hash=f"h{i}", char_count=len(t))
            for i, t in enumerate(texts)
        ]
        repo.create_chunks(chunks)
        vectors = embedder.embed_texts([c.text for c in chunks])
        store.upsert_chunks(kb_id, chunks, vectors)
        all_chunks.extend(chunks)
    retriever = DenseRetriever(embedder, store)
    return repo, retriever, store, kb_id


# ---------------------------------------------------------------------------
# Hermetic tests
# ---------------------------------------------------------------------------

def test_retrieval_returns_relevant_chunk_first(indexed_env):
    repo, retriever, store, kb_id = indexed_env
    resp = retriever.retrieve(kb_id, "battery management system cell voltage", top_k=3)
    assert resp.results, "must return results"
    assert resp.retrieval_backend == "qdrant-dense"
    assert "battery" in resp.results[0].text.lower()
    assert resp.results[0].score >= resp.results[-1].score


def test_chunk_level_evaluation_metrics(indexed_env):
    repo, retriever, store, kb_id = indexed_env
    # Ground truth: the exact chunk that discusses regenerative braking.
    chunks = repo.list_chunks(kb_id)
    target = next(c for c in chunks if "regenerative braking" in c.text.lower())
    q = EvaluationQuestion(
        id=new_id("eq"), kb_id=kb_id, question="How does the vehicle recover braking energy?",
        expected_chunk_ids=[target.id], generated_by="manual",
    )
    repo.create_evaluation_question(q)
    run = Evaluator(retriever, repo).run_evaluation(kb_id, EvaluationRunConfig(top_k=3))
    assert run.aggregate.recall_at_k == 1.0
    assert run.aggregate.questions_evaluated == 1
    assert run.per_question[0].num_relevant_found == 1


def test_document_level_evaluation_metrics(indexed_env):
    repo, retriever, store, kb_id = indexed_env
    docs = repo.list_documents(kb_id)
    doc_b = next(d for d in docs if d.title == "docB")
    q = EvaluationQuestion(
        id=new_id("eq"), kb_id=kb_id, question="suspension geometry and camber control",
        expected_document_ids=[doc_b.id], generated_by="manual",
    )
    repo.create_evaluation_question(q)
    run = Evaluator(retriever, repo).run_evaluation(kb_id, EvaluationRunConfig(top_k=3))
    p = run.per_question[0]
    assert p.doc_recall_at_k is not None, "doc ground truth must yield doc metrics"
    assert p.doc_num_relevant_total == 1
    # top-3 are all docB chunks for a suspension query with this fake store
    assert p.doc_recall_at_k == 1.0
    assert run.aggregate.doc_recall_at_k == 1.0


def test_no_ground_truth_question_is_skipped_not_fabricated(indexed_env):
    repo, retriever, store, kb_id = indexed_env
    q = EvaluationQuestion(id=new_id("eq"), kb_id=kb_id, question="unanswerable?", generated_by="manual")
    repo.create_evaluation_question(q)
    run = Evaluator(retriever, repo).run_evaluation(kb_id, EvaluationRunConfig(top_k=3))
    assert run.aggregate.recall_at_k is None, "no ground truth -> metric must be None, not 0"
    assert run.aggregate.questions_evaluated == 0
    assert "skipped" in run.per_question[0].note


def test_keyword_fallback_excluded_from_strict_headline_metrics(indexed_env):
    repo, retriever, store, kb_id = indexed_env
    q = EvaluationQuestion(
        id=new_id("eq"), kb_id=kb_id, question="energy recovery",
        expected_keywords=["regenerative"], generated_by="manual",
    )
    repo.create_evaluation_question(q)
    # Diagnostic run (opt-in): keyword heuristic is scored and disclosed.
    run = Evaluator(retriever, repo).run_evaluation(
        kb_id, EvaluationRunConfig(top_k=3, allow_keyword_fallback=True, label="diag")
    )
    p = run.per_question[0]
    assert "keyword-overlap" in p.note
    # A relevant chunk exists in the corpus but may not be retrieved; recall is
    # computed only over retrieved chunks, so it can never be negative-faked.
    assert p.recall_at_k is not None and 0.0 <= p.recall_at_k <= 1.0
    assert p.num_relevant_total == p.num_relevant_found  # fallback universe = retrieved
    assert run.aggregate.questions_with_keyword_fallback == 1
    assert run.aggregate.strict_mode is False
    assert run.aggregate.run_label == "diag"
    assert "diagnostic" in run.aggregate.notes.lower()


def test_strict_run_skips_keyword_questions_and_reports_counts(indexed_env):
    """Strict (headline) run: keyword-only questions excluded, counts disclosed."""
    repo, retriever, store, kb_id = indexed_env
    chunks = repo.list_chunks(kb_id)
    target = next(c for c in chunks if "battery" in c.text.lower())
    q_explicit = EvaluationQuestion(
        id=new_id("eq"), kb_id=kb_id, question="explicit: battery management",
        expected_chunk_ids=[target.id], generated_by="manual", notes="unit-test ground truth",
    )
    q_keyword = EvaluationQuestion(
        id=new_id("eq"), kb_id=kb_id, question="keyword-only: energy recovery",
        expected_keywords=["regenerative"], generated_by="manual",
    )
    repo.create_evaluation_question(q_explicit)
    repo.create_evaluation_question(q_keyword)
    run = Evaluator(retriever, repo).run_evaluation(kb_id, EvaluationRunConfig(top_k=3, label="strict-test"))
    by_id = {m.question_id: m for m in run.per_question}
    assert by_id[q_keyword.id].recall_at_k is None
    assert "DIAGNOSTIC-ONLY" in by_id[q_keyword.id].note
    assert by_id[q_explicit.id].recall_at_k is not None
    agg = run.aggregate
    assert agg.questions_with_explicit_gt == 1
    assert agg.questions_with_keyword_fallback == 0
    assert agg.questions_skipped_no_gt == 1
    assert agg.strict_mode is True
    assert agg.run_label == "strict-test"
    assert "skipped" in agg.notes.lower()


def test_stale_vectors_are_replaced_on_reindex(indexed_env):
    repo, retriever, store, kb_id = indexed_env
    before = len(store.points)
    # Re-chunk docA with different chunk IDs (as the real pipeline does).
    doc_a = next(d for d in repo.list_documents(kb_id) if d.title == "docA")
    new_chunks = [
        Chunk(id=new_id("chk"), document_id=doc_a.id, kb_id=kb_id, chunk_index=i, text=t, content_hash=f"nh{i}", char_count=len(t))
        for i, t in enumerate(TEXTS["docA"])
    ]
    store.delete_document_vectors(kb_id, [doc_a.id])
    repo.delete_chunks_for_document(kb_id, doc_a.id)
    embedder = HashingEmbeddingProvider()
    store.upsert_chunks(kb_id, new_chunks, embedder.embed_texts([c.text for c in new_chunks]))
    repo.create_chunks(new_chunks)
    # Old docA point IDs must be gone: total unchanged, no orphaned IDs remain.
    assert len(store.points) == before, "stale vectors must be deleted, not duplicated"
    swept = store.delete_orphaned_points(kb_id, {c.id for c in repo.list_chunks(kb_id)})
    assert swept == 0, "no orphans may remain after reindex"


def test_embedding_identity_mismatch_is_detected(repo, tmp_path):
    """The provider factory must REFUSE to serve a different model than the one
    a KB was indexed with (mismatch -> EmbeddingError, never silent vectors)."""
    import pytest as _pytest

    from app.config import Settings
    from app.services.embeddings.provider import EmbeddingError, EmbeddingIdentity, create_embedding_provider

    settings = Settings(data_dir=tmp_path, embedding_provider="hashing-dev-fallback")
    with _pytest.raises(EmbeddingError, match="mismatch"):
        create_embedding_provider(
            settings,
            expected=EmbeddingIdentity(provider="sentence-transformers", model="x", dimensions=384),
        )  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Live Qdrant tests (skip honestly when Qdrant is not reachable)
# ---------------------------------------------------------------------------

def _qdrant_available() -> QdrantVectorStore | None:
    try:
        store = QdrantVectorStore(url="http://localhost:6333")
        store._client.get_collections()
        return store
    except Exception:
        return None


@pytest.fixture
def live_store():
    store = _qdrant_available()
    if store is None:
        pytest.skip("Qdrant not reachable on localhost:6333; live integration test skipped (not faked)")
    yield store
    try:
        store.delete_collection("kb_live_itest")
    except Exception:
        pass


def test_live_qdrant_index_search_and_stale_cleanup(live_store):
    store: QdrantVectorStore = live_store
    kb_id = "live_itest"
    embedder = HashingEmbeddingProvider()
    chunks_v1 = [
        Chunk(id=new_id("chk"), document_id="doc_live", kb_id=kb_id, chunk_index=0, text="battery management system monitors voltage", content_hash="h1", char_count=45)
    ]
    store.ensure_collection(kb_id, embedder.dimensions)
    store.upsert_chunks(kb_id, chunks_v1, embedder.embed_texts([c.text for c in chunks_v1]))
    info = store.collection_info(kb_id)
    assert info["points_count"] == 1

    # Re-index with a NEW chunk id for the same document (real pipeline behaviour).
    chunks_v2 = [
        Chunk(id=new_id("chk"), document_id="doc_live", kb_id=kb_id, chunk_index=0, text="battery management system monitors voltage", content_hash="h2", char_count=45)
    ]
    store.delete_document_vectors(kb_id, ["doc_live"])
    store.upsert_chunks(kb_id, chunks_v2, embedder.embed_texts([c.text for c in chunks_v2]))
    assert store.collection_info(kb_id)["points_count"] == 1, "old vector must be gone after re-index"

    hits = store.search(kb_id, embedder.embed_texts(["battery voltage monitoring"])[0], top_k=3)
    assert len(hits) == 1
    assert hits[0]["payload"]["chunk_id"] == chunks_v2[0].id

    # Orphan sweep: mark nothing valid -> everything is stale.
    swept = store.delete_orphaned_points(kb_id, set())
    assert swept == 1
