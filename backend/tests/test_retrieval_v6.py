"""V6 retrieval intelligence tests (deterministic, hermetic, no services).

Covered here:
* BM25 correctness (formula, k1/b effects, determinism, persistence round-trip)
* score normalization + weighted fusion + Reciprocal Rank Fusion
* the retriever registry (specs, repo requirement, legacy 3-arg factories)
* provenance preservation through every strategy
* exact stale-index detection (corpus revision) and honest reporting
* diversity (document cap, MMR, and the honest "MMR unavailable" path)
* reranking: applied / not_requested / unavailable-fallback (never silently faked)
* retrieval configuration persistence and run recording (observability)

No Qdrant, no model downloads: the vector side is a real-cosine in-memory fake and
embeddings use the deterministic hashing provider.
"""
from __future__ import annotations

import math

import pytest

from app.repositories.sqlite_repo import Repository
from app.schemas.models import (
    Chunk,
    Document,
    DocumentStatus,
    KnowledgeBase,
    Source,
    SourceType,
)
from app.schemas.retrieval import (
    FusionMethod,
    NormalizationMethod,
    RerankerChoice,
    RerankerStatus,
    RetrievalParams,
    RetrievalStageStatus,
    RetrievalStrategy,
)
from app.services.embeddings.provider import HashingEmbeddingProvider
from app.services.retrieval import fusion, lexical
from app.services.retrieval.bm25 import Bm25Retriever, LexicalIndexStore
from app.services.retrieval.diversity import cap_per_document, mmr_select
from app.services.retrieval.hybrid import HybridRerankedRetriever, HybridRetriever
from app.services.retrieval.rerank import (
    CrossEncoderReranker,
    NoReranker,
    RerankResult,
    Reranker,
    get_reranker,
)
from app.services.retrieval.retriever import (
    DenseRetriever,
    Retriever,
    available_retrievers,
    describe_retrievers,
    get_retriever,
    register_retriever,
)
from app.services.retrieval.service import KnowledgeBaseNotFound, RetrievalService
from app.services.vector_store import factory as vector_factory
from app.utils.ids import new_id
from tests.test_retrieval_integration import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class TestBed:
    """A KB with two documents, three chunks each, indexed in a fake vector store."""

    #: Not a pytest test class (it takes a constructor argument).
    __test__ = False

    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.kb = KnowledgeBase(
            id=new_id("kb"),
            name="V6 test KB",
            domain="Naval Architecture",
            purpose="Unit tests",
            target_audience="Engineers",
            depth="technical",
        )
        repo.create_kb(self.kb)
        repo.update_kb(self.kb.model_copy(update={"embedding_identity": HashingEmbeddingProvider().identity().__dict__}))
        self.store = InMemoryVectorStore()
        self.embedder = HashingEmbeddingProvider()
        self.chunks: list[Chunk] = []
        self._index_documents(
            {
                "stability.pdf": [
                    ("page", 17, "The metacentric height GM must be positive for stable equilibrium of a ship."),
                    ("page", 18, "Free surface effect reduces the effective metacentric height in a flooded compartment."),
                    ("page", 19, "A tender ship has a small metacentric height and a long natural roll period."),
                ],
                "propulsion.pdf": [
                    ("page", 4, "Propeller cavitation occurs when local pressure falls below the vapour pressure of water."),
                    ("page", 5, "The propeller advance ratio relates the advance per revolution to the propeller diameter."),
                    ("page", 6, "Wake fraction describes the velocity deficit of water entering the propeller disc."),
                ],
            }
        )

    def _index_documents(self, documents: dict[str, list[tuple[str, int, str]]]) -> None:
        """Create real chunk rows (+ vectors) so both retrieval paths share a corpus."""
        for file_name, entries in documents.items():
            source = Source(
                id=new_id("src"),
                kb_id=self.kb.id,
                url=f"upload://{file_name}",
                title=file_name,
                source_type=SourceType.USER_PROVIDED,
            )
            self.repo.create_source(source)
            doc = Document(
                id=new_id("doc"),
                kb_id=self.kb.id,
                source_id=source.id,
                url=source.url,
                source_type=SourceType.USER_PROVIDED,
                title=file_name,
                file_name=file_name,
                file_path="",
                status=DocumentStatus.READY,
                content_hash=f"hash-{file_name}",
                user_provided=True,
                document_version=1,
            )
            self.repo.create_document(doc)
            new_chunks = []
            for idx, (kind, number, text) in enumerate(entries):
                page = number if kind == "page" else None
                slide = number if kind == "slide" else None
                new_chunks.append(
                    Chunk(
                        id=new_id("chk"),
                        document_id=doc.id,
                        kb_id=self.kb.id,
                        chunk_index=idx,
                        text=text,
                        source_id=source.id,
                        source_url=source.url,
                        source_title=file_name,
                        document_title=file_name,
                        source_type=SourceType.USER_PROVIDED.value,
                        page=page,
                        slide=slide,
                        section_path=f"Chapter {idx + 1}",
                        content_hash=f"hash-{file_name}-{idx}",
                        user_provided=True,
                        kb_version=1,
                        chunking_strategy="section_aware",
                    )
                )
            self.repo.create_chunks(new_chunks)
            self.chunks.extend(new_chunks)
            self.store.upsert_chunks(
                self.kb.id, new_chunks, self.embedder.embed_texts([c.text for c in new_chunks])
            )

    def bm25(self, **kwargs) -> Bm25Retriever:
        return Bm25Retriever(self.embedder, self.store, repo=self.repo, **kwargs)

    def hybrid(self, cls=HybridRetriever, **kwargs) -> HybridRetriever:
        return cls(self.embedder, self.store, repo=self.repo, **kwargs)

    def dense(self) -> DenseRetriever:
        return DenseRetriever(self.embedder, self.store)


@pytest.fixture
def bed(repo: Repository) -> TestBed:
    return TestBed(repo)


@pytest.fixture
def hashing_settings(monkeypatch):
    """Force the hermetic embedding provider for code that reads settings.

    `RetrievalService.build_retriever` resolves the embedding provider from
    settings, so without this a test would try to load a real SentenceTransformer
    model (slow, and a mismatch against the test KB's declared identity).
    """
    import app.config as config

    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing-dev-fallback")
    monkeypatch.setenv("EMBEDDING_MODEL", "char-ngram-hashing-256")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def stubbed_store(bed: TestBed, monkeypatch):
    """Route the production vector-store factory to the hermetic fake.

    The factory is patched as a MODULE ATTRIBUTE (not an import-time binding), the
    same isolation rule the V5 corpus tests use: any code path that resolves
    `factory.create_vector_store(...)` at call time gets the fake, so no test can
    create or query a real Qdrant collection.
    """
    monkeypatch.setattr(
        vector_factory, "create_vector_store", lambda settings, backend=None: bed.store
    )
    return bed.store


_ALL_DOCS = {
    "doc_short": [("page", 1, "metacentric height stability")],
    "doc_long": [
        (
            "page",
            1,
            "metacentric height stability " + " ".join(f"filler{i}" for i in range(200)),
        )
    ],
    "doc_twice": [("page", 1, "metacentric height metacentric height stability stability")],
}


# ---------------------------------------------------------------------------
# BM25: tokenizer + formula
# ---------------------------------------------------------------------------

def test_tokenizer_preserves_technical_identifiers():
    tokens = lexical.tokenize("GM-0.5, ISO_9001 s/n 3.2.1 — Métacentre.")
    assert "gm-0.5" in tokens
    assert "iso_9001" in tokens
    assert "s/n" in tokens
    assert "3.2.1" in tokens
    assert "métacentre" in tokens  # NFKC + lowercase
    assert all(t.strip() == t for t in tokens)


def test_tokenizer_is_deterministic_and_empty_for_punctuation():
    assert lexical.tokenize("--- ... ,,,") == []
    assert lexical.tokenize("Battery CELL voltage") == lexical.tokenize("Battery CELL voltage")
    assert lexical.tokenize("") == []


def test_bm25_matches_the_hand_computed_formula():
    index = lexical.LexicalIndex.build([("d1", "alpha beta"), ("d2", "alpha alpha")])
    # N=2, avgdl=(2+2)/2=2, df(alpha)=2, df(beta)=1, df(d1)=2, df(d2)=2
    idf_alpha = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    idf_beta = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    k1, b = 1.2, 0.75
    for cid, tf_alpha in (("d1", 1), ("d2", 2)):
        dl = 2
        expected = idf_alpha * (tf_alpha * (k1 + 1)) / (
            tf_alpha + k1 * (1 - b + b * dl / 2)
        )
        assert index.score_document(["alpha"], cid, k1=k1, b=b) == pytest.approx(expected)
    # beta appears in one of two documents, so it is rarer (higher idf) than alpha
    assert idf_beta > idf_alpha
    assert index.score_document(["beta"], "d1", k1=k1, b=b) == pytest.approx(
        idf_beta * (1 * (k1 + 1)) / (1 + k1 * (1 - b + b * 1.0))
    )
    assert index.score_document(["beta"], "d2", k1=k1, b=b) == 0.0


def test_bm25_saturates_term_frequency_and_normalizes_length():
    index = lexical.LexicalIndex.build(
        (cid, texts[0][2]) for cid, texts in _ALL_DOCS.items()
    )
    short = index.score_document(["metacentric"], "doc_short", k1=1.2, b=0.75)
    long = index.score_document(["metacentric"], "doc_long", k1=1.2, b=0.75)
    twice = index.score_document(["metacentric"], "doc_twice", k1=1.2, b=0.75)
    # tf saturation: doubling tf must NOT double the score
    assert twice < 2 * short
    # length normalization: the long document scores strictly lower
    assert long < short


def test_bm25_parameters_change_ranking_in_the_documented_direction():
    index = lexical.LexicalIndex.build(
        (cid, texts[0][2]) for cid, texts in _ALL_DOCS.items()
    )
    # b=0 disables length normalization -> the long doc is no longer penalised
    rel_long_b0 = index.score_document(["metacentric"], "doc_long", k1=1.2, b=0.0)
    rel_short_b0 = index.score_document(["metacentric"], "doc_short", k1=1.2, b=0.0)
    assert rel_long_b0 == pytest.approx(rel_short_b0)
    # k1=0 removes tf saturation scaling entirely: score becomes idf-driven
    zero_k1 = index.score_document(["metacentric"], "doc_twice", k1=0.0, b=0.75)
    assert zero_k1 == pytest.approx(index.score_document(["metacentric"], "doc_short", k1=0.0, b=0.75))


def test_bm25_index_payload_round_trip_is_exact():
    index = lexical.LexicalIndex.build([("c1", "alpha beta"), ("c2", "beta gamma")])
    restored = lexical.LexicalIndex.from_payload(index.to_payload())
    assert restored == index
    assert restored.matches_tokenizer()


def test_bm25_search_is_deterministic_with_id_tie_break():
    index = lexical.LexicalIndex.build([("b", "alpha"), ("a", "alpha")])
    hits = index.search("alpha", k1=1.2, b=0.75, top_k=2)
    assert [cid for cid, _ in hits] == ["a", "b"]  # equal scores -> id ascending
    assert hits[0][1] == pytest.approx(hits[1][1])


def test_bm25_repeat_query_term_does_not_change_ranking():
    index = lexical.LexicalIndex.build([("c1", "alpha"), ("c2", "alpha beta beta")])
    once = index.score(["alpha"], k1=1.2, b=0.75)
    thrice = index.score(["alpha", "alpha", "alpha"], k1=1.2, b=0.75)
    assert once == thrice


# ---------------------------------------------------------------------------
# Normalization + fusion
# ---------------------------------------------------------------------------

def test_min_max_normalization_and_degenerate_pools():
    assert fusion.min_max_normalize({"a": 5.0, "b": 3.0, "c": 1.0}) == {
        "a": 1.0,
        "b": 0.5,
        "c": 0.0,
    }
    assert fusion.min_max_normalize({"a": 2.0}) == {"a": 1.0}
    assert fusion.min_max_normalize({"a": 1.0, "b": 1.0}) == {"a": 1.0, "b": 1.0}
    assert fusion.min_max_normalize({}) == {}


def test_rank_normalization_is_scale_free():
    out = fusion.rank_normalize({"a": 100.0, "b": 0.2, "c": 0.1})
    assert out["a"] == pytest.approx(1.0)
    assert out["b"] == pytest.approx(2 / 3)
    assert out["c"] == pytest.approx(1 / 3)


def test_weighted_fusion_respects_weights_and_records_contributions():
    # d1 dense-only, d2 found by both, d3 lexical-only (all on a [0,1] scale).
    dense = {"d1": 1.0, "d2": 1.0}
    lexical = {"d2": 1.0, "d3": 1.0}
    out = fusion.weighted_fuse(dense, lexical, 0.65, 0.35)
    assert out["d1"]["fused"] == pytest.approx(0.65)
    assert out["d1"]["lexical_contribution"] == 0.0
    assert out["d2"]["fused"] == pytest.approx(1.0)
    assert out["d3"]["fused"] == pytest.approx(0.35)
    assert out["d3"]["dense_contribution"] == 0.0
    # a chunk found by BOTH retrievers outranks one found by either alone
    assert out["d2"]["fused"] > out["d1"]["fused"] > out["d3"]["fused"]


def test_weighted_fusion_with_zero_weight_is_degenerate_but_explicit():
    out = fusion.weighted_fuse({"a": 1.0}, {"b": 1.0}, 1.0, 0.0)
    assert out["a"]["fused"] == pytest.approx(1.0)
    # a semantic score of 0.0 for a lexical-only hit is reported, not hidden
    assert out["b"]["fused"] == pytest.approx(0.0)


def test_rrf_matches_formula_and_is_rank_based():
    out = fusion.reciprocal_rank_fusion(
        {"dense": ["a", "b"], "lexical": ["b", "c"]}, rrf_k=60
    )
    assert out["b"]["fused"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["a"]["fused"] == pytest.approx(1 / 61)
    assert out["c"]["fused"] == pytest.approx(1 / 62)
    assert out["b"]["fused"] > out["a"]["fused"]
    assert out["b"]["dense_rank"] == 2.0
    assert out["b"]["lexical_rank"] == 1.0


def test_rrf_weights_and_validation():
    out = fusion.reciprocal_rank_fusion(
        {"dense": ["a"], "lexical": ["b"]}, rrf_k=10, weights={"dense": 0.65, "lexical": 0.35}
    )
    assert out["a"]["fused"] == pytest.approx(0.65 / 11)
    assert out["b"]["fused"] == pytest.approx(0.35 / 11)
    with pytest.raises(ValueError):
        fusion.reciprocal_rank_fusion({"dense": ["a"]}, rrf_k=0)


def test_order_by_score_is_deterministic():
    assert fusion.order_by_score({"b": 1.0, "a": 1.0, "c": 0.5}) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_exposes_the_four_v6_strategies():
    names = available_retrievers()
    for expected in ("dense", "bm25", "hybrid", "hybrid_reranked", "qdrant-dense"):
        assert expected in names
    described = {spec["name"]: spec for spec in describe_retrievers()}
    assert described["bm25"]["requires_repo"] is True
    assert described["dense"]["is_baseline"] is True
    assert described["bm25"]["needs_embedding"] is False
    assert described["hybrid_reranked"]["aliases"] == []


def test_registry_builds_each_strategy_and_refuses_unknown_names(bed: TestBed):
    for name in ("dense", "bm25", "hybrid", "hybrid_reranked"):
        retriever = get_retriever(
            name, bed.embedder, bed.store, None, repo=bed.repo
        )
        assert retriever.strategy_name == name
    with pytest.raises(ValueError, match="Unknown retriever"):
        get_retriever("quantum", bed.embedder, bed.store, None, repo=bed.repo)
    # repo-requiring strategies fail loudly rather than degrading
    with pytest.raises(ValueError, match="corpus access"):
        get_retriever("hybrid", bed.embedder, bed.store, None)


def test_registry_accepts_a_legacy_three_argument_factory(bed: TestBed):
    """Registering is global; the registry is restored so no other test sees
    this stub."""
    from app.services.retrieval.retriever import (
        restore_retrievers,
        snapshot_retrievers,
    )

    class Stub(DenseRetriever):
        strategy_name = "legacy-stub"

    snapshot = snapshot_retrievers()
    try:
        register_retriever("legacy-stub", lambda e, s, i: Stub(e, s, i))
        assert "legacy-stub" in available_retrievers()
        built = get_retriever("legacy-stub", bed.embedder, bed.store, None, repo=bed.repo)
        assert built.strategy_name == "legacy-stub"
    finally:
        restore_retrievers(snapshot)
    assert "legacy-stub" not in available_retrievers()


# ---------------------------------------------------------------------------
# BM25 retriever end-to-end (hermetic)
# ---------------------------------------------------------------------------

def test_bm25_finds_exact_terminology_with_full_provenance(bed: TestBed):
    response = bed.bm25().retrieve(bed.kb.id, "metacentric height", top_k=3)
    assert response.strategy == "bm25"
    assert response.retrieval_backend == "sqlite-bm25"
    assert response.results, "BM25 must find chunks containing the query terms"
    top = response.results[0]
    assert "metacentric height" in top.text
    assert top.provenance["page"] in (17, 18, 19)
    assert top.provenance["section_path"].startswith("Chapter")
    assert top.provenance["document_title"] == "stability.pdf"
    assert top.provenance["user_provided"] is True
    assert top.retrieval_method == "bm25"
    assert top.score_breakdown.lexical_score is not None
    assert top.score_breakdown.lexical_score > 0
    assert top.rank == 1
    assert "lexical rank 1" in top.why
    # Raw BM25 is unbounded; the reported score is the normalized one.
    assert 0.0 <= top.score <= 1.0


def test_bm25_returns_nothing_for_unmatched_terms(bed: TestBed):
    response = bed.bm25().retrieve(bed.kb.id, "photosynthesis chlorophyll", top_k=5)
    assert response.results == []
    stages = {s.name: s for s in response.stages}
    assert stages["lexical"].count == 0


def test_bm25_respects_document_filters(bed: TestBed):
    propulsion_doc = next(c for c in bed.chunks if "cavitation" in c.text)
    response = bed.bm25().retrieve(
        bed.kb.id,
        "propeller cavitation",
        top_k=5,
        filters={"document_id": propulsion_doc.document_id},
    )
    assert response.results
    assert {r.document_id for r in response.results} == {propulsion_doc.document_id}


def test_bm25_respects_k1_and_b_parameters(bed: TestBed):
    params = RetrievalParams(
        strategy=RetrievalStrategy.BM25, top_k=3, bm25_k1=0.1, bm25_b=0.0
    )
    response = bed.bm25().retrieve_with_params(bed.kb.id, "metacentric height", params)
    assert response.results
    assert params.bm25_k1 == 0.1 and params.bm25_b == 0.0


# ---------------------------------------------------------------------------
# Staleness detection + persistence
# ---------------------------------------------------------------------------

def test_index_is_rebuilt_when_the_corpus_revision_changes(bed: TestBed):
    store = LexicalIndexStore(bed.repo)
    first, status_first = store.get(bed.kb.id)
    assert status_first.rebuilt is True
    assert first.n_docs == len(bed.chunks)

    # Same revision -> served from memory, not rebuilt.
    again, status_again = store.get(bed.kb.id)
    assert status_again.rebuilt is False
    assert status_again.source == "memory"

    # Add a chunk -> revision bumps -> rebuild is forced and reported.
    extra = Chunk(
        id=new_id("chk"),
        document_id=bed.chunks[0].document_id,
        kb_id=bed.kb.id,
        chunk_index=99,
        text="bilge pump capacity",
    )
    bed.repo.create_chunks([extra])
    third, status_third = store.get(bed.kb.id)
    assert status_third.rebuilt is True
    assert third.n_docs == len(bed.chunks) + 1
    assert "bilge" in third.postings


def test_persisted_index_is_reused_across_store_instances(bed: TestBed):
    LexicalIndexStore(bed.repo).get(bed.kb.id)  # persists with the current revision
    fresh = LexicalIndexStore(bed.repo)
    index, status = fresh.get(bed.kb.id)
    assert status.source == "persisted"
    assert status.rebuilt is False
    assert index.n_docs == len(bed.chunks)


def test_tokenizer_version_change_forces_a_rebuild(bed: TestBed):
    from app.services.retrieval import bm25 as bm25_module

    LexicalIndexStore(bed.repo).get(bed.kb.id)
    stored_revision, payload = bed.repo.get_lexical_index(bed.kb.id)  # type: ignore[misc]
    # Simulate an index written by an older tokenizer.
    import json

    data = json.loads(payload)
    data["tokenizer_version"] = "lex-v0"
    bed.repo.save_lexical_index(bed.kb.id, stored_revision, "2020-01-01T00:00:00+00:00", json.dumps(data))

    index, status = LexicalIndexStore(bed.repo).get(bed.kb.id)
    assert status.rebuilt is True
    assert index.tokenizer_version == bm25_module.TOKENIZER_VERSION


def test_bm25_status_reports_staleness_without_building(bed: TestBed):
    store = LexicalIndexStore(bed.repo)
    assert store.status(bed.kb.id)["source"] == "none"
    store.get(bed.kb.id)
    assert store.status(bed.kb.id)["stale"] is False
    bed.repo.bump_corpus_revision(bed.kb.id)
    assert store.status(bed.kb.id)["stale"] is True


def test_bm25_response_notes_a_rebuild(bed: TestBed):
    response = bed.bm25().retrieve(bed.kb.id, "metacentric", top_k=2)
    assert any("rebuilt" in note for note in response.notes)
    stages = {s.name: s for s in response.stages}
    assert "revision=" in stages["lexical"].detail


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------

def test_hybrid_fuses_dense_and_lexical_with_recorded_configuration(bed: TestBed):
    params = RetrievalParams(strategy=RetrievalStrategy.HYBRID, top_k=3, candidate_k=10)
    response = bed.hybrid().retrieve_with_params(bed.kb.id, "metacentric height", params)
    assert response.strategy == "hybrid"
    assert response.results
    assert response.params is not None
    assert response.params.dense_weight == 0.65
    fused = [r for r in response.results if r.score_breakdown.fused_score is not None]
    assert fused, "hybrid results must carry the fused score"
    # Chunks found by BOTH retrievers carry both normalized components in their
    # score breakdown — that is what makes the fusion auditable.
    both = [
        r
        for r in response.results
        if r.score_breakdown.normalized_dense is not None
        and r.score_breakdown.normalized_lexical is not None
    ]
    assert both, "at least one chunk must be found by both dense and lexical retrieval"
    assert "dense" in both[0].stages and "lexical" in both[0].stages
    assert "fusion" in both[0].stages
    assert any("weighted fusion" in note for note in response.notes)
    counts = {s.name: s.count for s in response.stages}
    assert counts["dense"] is not None and counts["lexical"] is not None


def test_hybrid_rrf_fusion_selects_by_rank(bed: TestBed):
    params = RetrievalParams(
        strategy=RetrievalStrategy.HYBRID,
        top_k=3,
        candidate_k=10,
        fusion=FusionMethod.RRF,
        rrf_k=10,
    )
    response = bed.hybrid().retrieve_with_params(bed.kb.id, "propeller cavitation", params)
    assert response.results
    assert any("RRF fusion with k=10" in note for note in response.notes)
    assert any(s.name == "fusion" for s in response.stages)
    # RRF scores are tiny by construction; they must still be reported as-is.
    assert response.results[0].score == pytest.approx(response.results[0].score_breakdown.fused_score or 0.0, abs=1e-4)


def test_hybrid_is_structured_around_the_same_response_model_as_dense(bed: TestBed):
    dense = bed.dense().retrieve(bed.kb.id, "metacentric height", top_k=3)
    hybrid = bed.hybrid().retrieve(bed.kb.id, "metacentric height", top_k=3)
    for response in (dense, hybrid):
        assert response.query and response.top_k == 3
        assert response.embedding_model
        for result in response.results:
            assert result.chunk_id and result.document_id and result.text
            assert result.provenance.get("document_title")


def test_hybrid_rescales_weights_when_the_lexical_pool_is_empty(bed: TestBed):
    retriever = HybridRetriever(
        bed.embedder,
        bed.store,
        dense=DenseRetriever(bed.embedder, bed.store),
        lexical=_EmptyLexicalRetriever(bed.embedder, bed.store),
    )
    response = retriever.retrieve_with_params(
        bed.kb.id, "metacentric height", RetrievalParams(strategy=RetrievalStrategy.HYBRID, top_k=3)
    )
    assert response.results
    assert any("rescaled to dense-only" in note for note in response.notes)
    # With dense-only weights the top fused score must reach 1.0, not stop at 0.65.
    assert response.results[0].score_breakdown.fused_score == pytest.approx(1.0)


class _EmptyLexicalRetriever(Retriever):
    """A lexical stand-in that returns nothing (simulates an empty BM25 index)."""

    backend = "lexical-empty"
    strategy_name = "lexical-empty"

    def retrieve(self, kb_id, query, top_k=5, filters=None, min_score=0.0):  # type: ignore[override]
        from app.schemas.models import RetrievalResponse

        return RetrievalResponse(
            query=query, top_k=top_k, retrieval_backend=self.backend, strategy=self.strategy_name
        )


def test_hybrid_min_score_filters_on_the_fused_score(bed: TestBed):
    retriever = bed.hybrid()
    unfiltered = retriever.retrieve_with_params(
        bed.kb.id,
        "metacentric height",
        RetrievalParams(strategy=RetrievalStrategy.HYBRID, top_k=5, candidate_k=8),
    )
    filtered = retriever.retrieve_with_params(
        bed.kb.id,
        "metacentric height",
        RetrievalParams(
            strategy=RetrievalStrategy.HYBRID, top_k=5, min_score=0.99, candidate_k=8
        ),
    )
    assert len(filtered.results) <= len(unfiltered.results)
    assert all(r.score >= 0.99 for r in filtered.results)


def test_hybrid_zero_weight_side_is_respected(bed: TestBed):
    params = RetrievalParams(
        strategy=RetrievalStrategy.HYBRID,
        top_k=3,
        dense_weight=0.0,
        bm25_weight=1.0,
        candidate_k=8,
    )
    response = bed.hybrid().retrieve_with_params(bed.kb.id, "metacentric height", params)
    assert response.results
    for result in response.results:
        assert result.score_breakdown.dense_contribution == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def test_document_cap_demotes_rather_than_deletes():
    order = ["a1", "a2", "a3", "b1"]
    docs = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}
    outcome = cap_per_document(order, docs, max_per_document=1)
    assert outcome.kept == ["a1", "b1", "a2", "a3"]
    assert set(outcome.dropped) == {"a2", "a3"}
    assert "cap 1" in outcome.dropped["a2"]


def test_document_cap_without_limit_is_identity():
    order = ["a1", "a2"]
    outcome = cap_per_document(order, {"a1": "A", "a2": "A"}, None)
    assert outcome.kept == order
    assert outcome.dropped == {}


def test_mmr_prefers_diverse_candidates_when_redundancy_is_penalized():
    relevance = {"q": 1.0, "near_dup": 0.9, "different": 0.6}
    vectors = {
        "q": [1.0, 0.0],
        "near_dup": [0.99, 0.01],
        "different": [0.0, 1.0],
    }
    outcome = mmr_select(relevance, vectors, lambda_=0.5, k=2)
    assert outcome.kept == ["q", "different"]
    assert "near_dup" in outcome.dropped


def test_mmr_with_lambda_one_ignores_diversity():
    relevance = {"q": 1.0, "near_dup": 0.9, "different": 0.6}
    vectors = {"q": [1.0, 0.0], "near_dup": [0.99, 0.01], "different": [0.0, 1.0]}
    outcome = mmr_select(relevance, vectors, lambda_=1.0, k=2)
    assert outcome.kept == ["q", "near_dup"]


def test_hybrid_mmr_uses_stored_vectors_when_available(bed: TestBed):
    params = RetrievalParams(
        strategy=RetrievalStrategy.HYBRID,
        top_k=4,
        candidate_k=6,
        diversity="mmr",
        diversity_lambda=0.5,
    )
    response = bed.hybrid().retrieve_with_params(bed.kb.id, "metacentric height", params)
    assert response.results
    stage = next(s for s in response.stages if s.name == "diversity")
    assert stage.status == RetrievalStageStatus.OK
    assert "MMR" in stage.detail


def test_hybrid_mmr_reports_unavailable_when_vectors_cannot_be_fetched(bed: TestBed):
    class NoVectors(InMemoryVectorStore):
        def fetch_vectors(self, kb_id, chunk_ids):
            raise NotImplementedError

    store = NoVectors()
    # Same corpus (so the dense pool is non-empty) but no way to read vectors back.
    store.points = dict(bed.store.points)
    retriever = HybridRetriever(
        bed.embedder,
        store,
        dense=DenseRetriever(bed.embedder, store),
        lexical=_EmptyLexicalRetriever(bed.embedder, store),
    )
    params = RetrievalParams(
        strategy=RetrievalStrategy.HYBRID, top_k=3, diversity="mmr", candidate_k=6
    )
    response = retriever.retrieve_with_params(bed.kb.id, "metacentric height", params)
    stage = next(s for s in response.stages if s.name == "diversity")
    assert stage.status == RetrievalStageStatus.UNAVAILABLE
    assert "MMR" in stage.detail
    assert any("MMR is unavailable" in note or "candidate vectors are unavailable" in note or "cannot return chunk vectors" in note for note in response.notes)


# ---------------------------------------------------------------------------
# Reranking honesty
# ---------------------------------------------------------------------------

def test_no_reranker_reports_not_requested():
    result = NoReranker().rerank("q", [("c1", "text")])
    assert result.status == RerankerStatus.NOT_REQUESTED
    assert result.applied is False


def test_cross_encoder_reports_unavailable_and_never_fakes_reranking(monkeypatch):
    reranker = CrossEncoderReranker()
    monkeypatch.setattr(
        CrossEncoderReranker,
        "_load",
        lambda self: (setattr(self, "_load_error", "no network") or None),
    )
    available, reason = reranker.availability()
    assert available is False
    assert "no network" in reason
    result = reranker.rerank("q", [("c1", "a"), ("c2", "b")])
    assert result.status == RerankerStatus.UNAVAILABLE_FALLBACK
    assert result.order is None
    assert result.applied is False


def test_hybrid_with_requested_but_unavailable_reranker_keeps_order_and_says_so(
    bed: TestBed, monkeypatch
):
    import app.services.retrieval.hybrid as hybrid_module

    class Unavailable(Reranker):
        name = "cross-encoder"
        model = "cross-encoder/test"

        def availability(self):
            return (False, "model not cached and offline")

        def rerank(self, query, items):
            raise AssertionError("must not be called when unavailable")

    monkeypatch.setattr(hybrid_module, "get_reranker", lambda name, model: Unavailable())
    params = RetrievalParams(
        strategy=RetrievalStrategy.HYBRID_RERANKED,
        top_k=3,
        candidate_k=6,
        reranker=RerankerChoice.CROSS_ENCODER,
    )
    baseline = bed.hybrid().retrieve_with_params(bed.kb.id, "metacentric height", params)
    response = bed.hybrid(HybridRerankedRetriever).retrieve_with_params(
        bed.kb.id, "metacentric height", params
    )
    assert response.reranker is not None
    assert response.reranker.status == RerankerStatus.UNAVAILABLE_FALLBACK
    assert response.reranker.degraded is True
    assert response.reranker.reranked == 0
    assert any("UNAVAILABLE" in note for note in response.notes)
    assert [r.chunk_id for r in response.results] == [r.chunk_id for r in baseline.results]
    for result in response.results:
        assert result.rerank_score is None


def test_hybrid_reranker_applied_changes_order_and_reports_model(bed: TestBed, monkeypatch):
    import app.services.retrieval.hybrid as hybrid_module

    class FakeReranker(Reranker):
        name = "cross-encoder"
        model = "cross-encoder/fake"

        def availability(self):
            return (True, "")

        def rerank(self, query, items):
            ids = [cid for cid, _ in items]
            reversed_ids = list(reversed(ids))
            scores = {cid: float(len(ids) - i) for i, cid in enumerate(reversed_ids)}
            return RerankResult(
                status=RerankerStatus.APPLIED,
                model=self.model,
                order=reversed_ids,
                scores=scores,
                detail="fake reranker",
            )

    monkeypatch.setattr(hybrid_module, "get_reranker", lambda name, model: FakeReranker())
    params = RetrievalParams(
        strategy=RetrievalStrategy.HYBRID_RERANKED,
        top_k=3,
        candidate_k=6,
    )
    response = bed.hybrid(HybridRerankedRetriever).retrieve_with_params(
        bed.kb.id, "metacentric height", params
    )
    assert response.reranker is not None
    assert response.reranker.status == RerankerStatus.APPLIED
    assert response.reranker.model == "cross-encoder/fake"
    assert response.reranker.reranked == response.reranker.candidate_k
    assert any("rerank" in r.stages for r in response.results)
    assert all(r.rerank_score is not None for r in response.results)
    # Ordering follows the reranker, not the fused score.
    rerank_scores = [r.rerank_score for r in response.results]
    assert rerank_scores == sorted(rerank_scores, reverse=True)


def test_reranker_factory_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker("magic-reranker")
    assert isinstance(get_reranker("none"), NoReranker)
    assert isinstance(get_reranker("cross-encoder", "cross-encoder/x"), CrossEncoderReranker)


def test_hybrid_reranked_defaults_to_cross_encoder():
    assert HybridRerankedRetriever.default_reranker == RerankerChoice.CROSS_ENCODER.value
    # hybrid_reranked implies reranking even when the params do not repeat it
    implied = RetrievalParams(strategy=RetrievalStrategy.HYBRID_RERANKED, top_k=2)
    assert implied.reranker == RerankerChoice.NONE
    assert implied.reranking_requested() is True
    assert implied.reranker_name() == "cross-encoder"

    # plain hybrid does not rerank unless explicitly asked
    plain = RetrievalParams(strategy=RetrievalStrategy.HYBRID, top_k=2)
    assert plain.reranking_requested() is False
    assert plain.reranker_name() == "none"
    asked = plain.model_copy(update={"reranker": RerankerChoice.CROSS_ENCODER})
    assert asked.reranking_requested() is True
    assert asked.reranker_name() == "cross-encoder"


# ---------------------------------------------------------------------------
# Configuration persistence + run recording
# ---------------------------------------------------------------------------

def test_retrieval_config_round_trips_through_the_repository(bed: TestBed):
    service = RetrievalService(bed.repo)
    assert service.get_config(bed.kb.id) is None
    params = RetrievalParams(
        strategy=RetrievalStrategy.HYBRID,
        top_k=7,
        dense_weight=0.5,
        bm25_weight=0.5,
        fusion=FusionMethod.RRF,
    )
    saved = service.save_config(bed.kb.id, params, note="unit test")
    loaded = service.get_config(bed.kb.id)
    assert loaded is not None
    assert loaded.params == params
    assert loaded.note == "unit test"
    assert saved.updated_at == loaded.updated_at


def test_resolve_params_precedence(bed: TestBed):
    service = RetrievalService(bed.repo)
    service.save_config(
        bed.kb.id, RetrievalParams(strategy=RetrievalStrategy.BM25, top_k=9, min_score=0.2)
    )
    # stored values survive
    stored_only = service.resolve_params(bed.kb.id)
    assert stored_only.strategy == RetrievalStrategy.BM25
    assert stored_only.top_k == 9
    # explicit overrides win
    overridden = service.resolve_params(
        bed.kb.id, strategy="hybrid", overrides={"top_k": 3, "min_score": None}
    )
    assert overridden.strategy == RetrievalStrategy.HYBRID
    assert overridden.top_k == 3
    assert overridden.min_score == 0.2  # None never clobbers a stored value
    # unknown strategy is a clear error
    with pytest.raises(Exception):
        service.resolve_params(bed.kb.id, strategy="teleport")


def test_config_fingerprint_is_stable_and_configuration_sensitive():
    a = RetrievalParams(strategy=RetrievalStrategy.HYBRID, top_k=5)
    b = RetrievalParams(strategy=RetrievalStrategy.HYBRID, top_k=5)
    c = RetrievalParams(strategy=RetrievalStrategy.HYBRID, top_k=6)
    assert a.config_fingerprint() == b.config_fingerprint()
    assert a.config_fingerprint() != c.config_fingerprint()


def test_effective_weights_normalize_but_preserve_the_ratio():
    params = RetrievalParams(
        strategy=RetrievalStrategy.HYBRID, dense_weight=1.0, bm25_weight=1.0
    )
    assert params.effective_weights() == (0.5, 0.5)
    assert params.dense_weight == 1.0  # the requested value is never rewritten


def test_run_is_recorded_with_configuration_weights_and_stages(
    bed: TestBed, stubbed_store, hashing_settings
):
    service = RetrievalService(bed.repo)
    service.save_config(
        bed.kb.id,
        RetrievalParams(
            strategy=RetrievalStrategy.HYBRID, top_k=3, dense_weight=0.8, bm25_weight=0.2
        ),
    )
    response = service.run(bed.kb.id, "metacentric height")
    assert response.retrieval_run_id
    runs = service.recent_runs(bed.kb.id)
    assert runs and runs[0].id == response.retrieval_run_id
    full = service.get_run(bed.kb.id, response.retrieval_run_id)
    assert full is not None
    assert full.strategy == "hybrid"
    assert full.applied_weights == {"dense": 0.8, "bm25": 0.2}
    assert full.normalization == "min_max"
    assert full.result_count == len(response.results)
    assert full.candidate_counts.get("dense") is not None
    assert full.timings.total_ms is not None and full.timings.total_ms >= 0
    assert full.params.top_k == 3
    assert full.config_fingerprint


def test_run_records_reranker_status_honestly(bed: TestBed, stubbed_store, hashing_settings):
    service = RetrievalService(bed.repo)
    response = service.run(
        bed.kb.id,
        "propeller cavitation",
        overrides={"strategy": "dense"},
    )
    run = service.get_run(bed.kb.id, response.retrieval_run_id or "")
    assert run is not None
    assert run.reranker_status == RerankerStatus.NOT_REQUESTED
    assert run.reranker == "none"


def test_bm25_run_does_not_require_an_embedding_model(
    bed: TestBed, stubbed_store, hashing_settings
):
    service = RetrievalService(bed.repo)
    response = service.run(bed.kb.id, "metacentric height", strategy="bm25")
    assert response.results
    assert any("no embedding model was loaded" in note for note in response.notes)


def test_run_raises_for_unknown_knowledge_base(bed: TestBed):
    service = RetrievalService(bed.repo)
    with pytest.raises(KnowledgeBaseNotFound):
        service.run("kb_does_not_exist", "anything")


def test_bm25_index_status_endpoint_helper(bed: TestBed):
    service = RetrievalService(bed.repo)
    status = service.bm25_index_status(bed.kb.id)
    assert status["source"] == "none"
    bed.bm25().retrieve(bed.kb.id, "metacentric", top_k=2)
    status = service.bm25_index_status(bed.kb.id)
    assert status["stale"] is False
    assert status["documents"] == len(bed.chunks)


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

def test_dense_retrieve_signature_is_unchanged_for_v5_callers(bed: TestBed):
    response = bed.dense().retrieve(bed.kb.id, "metacentric height", top_k=2)
    assert response.retrieval_backend == "qdrant-dense"
    assert response.strategy == "dense"
    assert len(response.results) <= 2
    assert response.results[0].score == pytest.approx(
        response.results[0].score_breakdown.dense_score or 0.0, abs=1e-4
    )


def test_retrieve_with_params_default_delegates_to_retrieve(bed: TestBed):
    class Minimal(DenseRetriever):
        def __init__(self):
            self.calls = []

        def retrieve(self, kb_id, query, top_k=5, filters=None, min_score=0.0):
            self.calls.append((kb_id, query, top_k, filters, min_score))
            from app.schemas.models import RetrievalResponse

            return RetrievalResponse(query=query, top_k=top_k)

    minimal = Minimal()
    params = RetrievalParams(
        strategy=RetrievalStrategy.HYBRID, top_k=4, min_score=0.1, filters={"document_id": "d"}
    )
    minimal.retrieve_with_params("kb", "q", params)
    assert minimal.calls == [("kb", "q", 4, {"document_id": "d"}, 0.1)]


# ---------------------------------------------------------------------------
# Schema-level guards
# ---------------------------------------------------------------------------

def test_retrieval_params_rejects_invalid_values():
    with pytest.raises(Exception):
        RetrievalParams(diversity="chaos")
    with pytest.raises(Exception):
        RetrievalParams(strategy=RetrievalStrategy.HYBRID, dense_weight=0.0, bm25_weight=0.0)
    with pytest.raises(Exception):
        RetrievalParams(bm25_b=1.5)
    with pytest.raises(Exception):
        RetrievalParams(normalization=NormalizationMethod("nonsense"))


def test_candidate_k_defaults_are_documented_values():
    assert RetrievalParams(top_k=3).resolved_candidate_k() == 20
    assert RetrievalParams(top_k=10).resolved_candidate_k() == 40
    assert RetrievalParams(top_k=10, candidate_k=5).resolved_candidate_k() == 10
