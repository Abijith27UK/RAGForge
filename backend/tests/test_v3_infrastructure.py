"""V3 Step 4/5/6 tests: seeded determinism contract, chunking registry, vector-backend factory.

Phase B (v3 experiment runner) and Phase D (pluggable vector store) rely on:
- deterministic seeded randomness for corpus selection (multi-seed reproducibility)
- a registry-resolved ChunkingStrategy surface
- a registry-resolved VectorStore surface with qdrant always available
"""
from __future__ import annotations

import random

import pytest

from app.schemas.models import Document, Source, SourceType
from app.services.chunking.chunker import (
    CHUNKING_REGISTRY,
    FixedSizeChunker,
    SectionAwareChunker,
    get_chunker,
)
from app.services.vector_store.factory import (
    available_backends,
    create_vector_store,
    register_vector_backend,
)
from app.config import get_settings


# ---------------------------------------------------------------------------
# Phase B: multi-seed determinism contract used by the v3 runner
# ---------------------------------------------------------------------------

def _seeded_sample(population: list[str], n: int, seed: int) -> list[str]:
    """The exact selection primitive the v3 runner uses (must stay deterministic)."""
    rng = random.Random(seed)
    return sorted(rng.sample(population, n))


def test_seeded_selection_is_deterministic_across_repeats():
    pool = [f"src_{i:03d}" for i in range(36)]
    a = _seeded_sample(pool, 11, 20260915)
    b = _seeded_sample(pool, 11, 20260915)
    assert a == b
    assert len(a) == 11 and len(set(a)) == 11


def test_seeded_selection_differs_across_the_five_v3_seeds():
    """Every official v3 seed must produce a usable selection; seeds are not all identical."""
    pool = [f"src_{i:03d}" for i in range(36)]
    picks = {tuple(_seeded_sample(pool, 8, seed)) for seed in (20260915, 20260916, 20260917, 20260918, 20260919)}
    assert len(picks) == 5, "five official seeds must yield five distinct corpora"


# ---------------------------------------------------------------------------
# Step 5: chunking strategy interface + registry
# ---------------------------------------------------------------------------

def _doc_and_source():
    doc = Document(id="doc_reg", kb_id="kb_reg", url="https://example.com/x",
                   title="Regenerative braking", content_hash="h" * 64,
                   source_id="src_reg", source_type=SourceType.WEB_PAGE)
    src = Source(id="src_reg", kb_id="kb_reg", url="https://example.com/x",
                 title="Regenerative braking", source_type=SourceType.WEB_PAGE)
    return doc, src


TEXT = (
    "## Braking\n" + ("Regenerative braking recovers kinetic energy. " * 40) + "\n"
    "## Powertrain\n" + ("The electric motor drives the wheels through a gearbox. " * 40)
)


def test_registry_resolves_both_strategies_and_rejects_unknown():
    assert set(CHUNKING_REGISTRY.names()) >= {"section-aware", "fixed-size"}
    assert isinstance(get_chunker("fixed-size"), FixedSizeChunker)
    assert isinstance(get_chunker("section-aware"), SectionAwareChunker)
    with pytest.raises(ValueError):
        get_chunker("semantic-quantum")


def test_same_document_and_config_produce_identical_content_hashes():
    """Reproducibility rule: identical doc + config => identical chunk content hashes."""
    doc, src = _doc_and_source()
    c1 = [c.content_hash for c in SectionAwareChunker().chunk(doc, TEXT, src, target_size=400, overlap=60)]
    c2 = [c.content_hash for c in SectionAwareChunker().chunk(doc, TEXT, src, target_size=400, overlap=60)]
    c3 = [c.content_hash for c in FixedSizeChunker().chunk(doc, TEXT, src, target_size=400, overlap=60)]
    assert c1 == c2
    assert c1 != c3, "different strategies must produce different chunk sets"
    # Strategy name is stamped on every chunk for provenance.
    c_all = SectionAwareChunker().chunk(doc, TEXT, src, target_size=400, overlap=60)
    assert all(c.chunking_strategy == "section-aware" for c in c_all)


def test_strategy_config_round_trips_on_kb_payload():
    from app.schemas.models import KnowledgeBase, KnowledgeBaseCreate

    payload = KnowledgeBaseCreate(
        name="n", domain="d", purpose="p", target_audience="a",
        vector_backend="qdrant", chunking_strategy="fixed-size",
    )
    assert payload.chunking_strategy == "fixed-size"
    kb = KnowledgeBase(**payload.model_dump(), id="kb_x")
    assert kb.chunking_strategy == "fixed-size" and kb.vector_backend == "qdrant"
    # Defaults stay the current product default.
    default = KnowledgeBaseCreate(name="n", domain="d", purpose="p", target_audience="a")
    assert default.chunking_strategy == "section-aware"


# ---------------------------------------------------------------------------
# Step 6: vector-backend factory
# ---------------------------------------------------------------------------

def test_qdrant_backend_always_available_and_unknown_rejected():
    assert "qdrant" in available_backends()
    with pytest.raises(ValueError):
        create_vector_store(get_settings(), "turbovec")  # Phase E: not registered yet


def test_experimental_backend_registration_extends_factory_without_touching_qdrant():
    class _Stub:
        def __init__(self, settings):
            self.settings = settings

    register_vector_backend("unit-test-backend", lambda s: _Stub(s))
    try:
        store = create_vector_store(get_settings(), "unit-test-backend")
        assert isinstance(store, _Stub)
        assert "qdrant" in available_backends()  # untouched
    finally:
        # remove the test registration to keep other tests hermetic
        from app.services.vector_store import factory as f
        f._BACKENDS.pop("unit-test-backend", None)
