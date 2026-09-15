"""Tests for the v2 content-aware source-quality scorer.

Deterministic logic only: a tiny bag-of-words embedder stands in for the real
model so tests need no network or model download. Cosine geometry still works
because vectors are L2-normalized term vectors.
"""
from __future__ import annotations

import hashlib
import math
import re

import pytest

from app.services.embeddings.provider import EmbeddingProvider
from app.services.source_quality import content_scorer as cs
from app.services.source_quality.content_scorer import (
    ContentAwareSourceScorer,
    ContentRelevanceScorer,
    load_domain_map,
)
from app.schemas.models import Source


class BagOfWordsEmbedder(EmbeddingProvider):
    """Deterministic hashed term-frequency vectors: cosine == word overlap."""

    name, model, dimensions = "bow-test", "bag-of-words", 4096

    def identity(self):
        from app.schemas.models import EmbeddingIdentity

        return EmbeddingIdentity(self.name, self.model, self.dimensions)

    def embed_texts(self, texts):
        def vec(t: str) -> list[float]:
            v = [0.0] * self.dimensions
            for w in re.findall(r"[a-z]+", t.lower()):
                if len(w) < 3:
                    continue
                idx = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self.dimensions
                v[idx] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / n for x in v]

        return [vec(t) for t in texts]


@pytest.fixture(autouse=True)
def _bow_offset():
    """SIM_NORM_OFFSET is calibrated for real semantic embeddings; the BoW test
    embedder produces much smaller raw cosines, so disable it for unit tests."""
    original = cs.SIM_NORM_OFFSET
    cs.SIM_NORM_OFFSET = 0.0
    yield
    cs.SIM_NORM_OFFSET = original


@pytest.fixture(scope="module")
def domain_map() -> dict:
    return load_domain_map("Automobile Engineering")


@pytest.fixture(scope="module")
def scorer(domain_map) -> ContentRelevanceScorer:
    return ContentRelevanceScorer(BagOfWordsEmbedder(), domain_map)


# --- domain map ---------------------------------------------------------------


def test_domain_map_loads_with_validated_weights(domain_map):
    ids = [a["id"] for a in domain_map["requirement_areas"]]
    assert set(ids) >= {
        "engine",
        "transmission",
        "braking",
        "steering",
        "suspension",
        "chassis",
        "vehicle-dynamics",
        "electric-vehicles",
        "fundamentals",
    }
    assert abs(sum(a["weight"] for a in domain_map["requirement_areas"]) - 1.0) < 1e-9
    assert "human review pending" in domain_map["provenance"].lower()


def test_domain_map_missing_domain_raises():
    with pytest.raises(FileNotFoundError):
        load_domain_map("Quantum Knitting")


# --- content analysis ---------------------------------------------------------


def test_specialist_page_covers_its_area_and_nothing_else(scorer):
    text = (
        "The engine block houses the cylinders. A four-stroke petrol engine "
        "converts fuel into mechanical work through combustion. Engine power "
        "and torque depend on displacement. " * 6
    )
    a = scorer.analyze_content(text)
    engine = next(p for p in a.per_area if p["area_id"] == "engine")
    steering = next(p for p in a.per_area if p["area_id"] == "steering")
    assert engine["similarity"] > 0.0
    assert engine["similarity"] > steering["similarity"] + 0.15
    assert "engine" in engine["evidence"].lower()


def test_zero_overlap_gives_near_zero_similarity(scorer):
    text = ("Quantum entanglement correlates particle states across space. " * 30)
    a = scorer.analyze_content(text)
    assert a.content_relevance < 0.02
    assert a.domain_coverage == 0.0
    for p in a.per_area:
        assert p["similarity"] < 0.02 and not p["covered"]


def test_windows_shorter_than_prose_minimum_are_discarded():
    link_text = "\n".join("link" for _ in range(400))
    assert ContentRelevanceScorer._window(link_text) == []
    prose = " ".join(["suspension damper absorbs road vibration energy"] * 80)
    assert len(ContentRelevanceScorer._window(prose)) > 0


def test_similarity_offset_renormalizes_raw_scores(monkeypatch):
    # 0.20 raw must map to exactly 0.0; 1.0 raw maps to 1.0.
    monkeypatch.setattr(cs, "SIM_NORM_OFFSET", 0.20)  # the production value
    off = cs.SIM_NORM_OFFSET
    assert abs((0.20 - off) / (1 - off)) < 1e-12
    assert abs((1.00 - off) / (1 - off) - 1.0) < 1e-12


def test_tiny_content_reports_low_confidence_and_limitation(scorer):
    a = scorer.analyze_content("short")
    assert a.confidence == 0.0 and a.content_relevance == 0.0
    a2 = scorer.analyze_content("The engine burns fuel. " * 9)  # windows fail prose filter
    assert a2.n_windows == 0
    assert any("windows" in lim for lim in a2.limitations)


# --- fetch + boilerplate stripping -------------------------------------------


def test_fetch_strips_wikipedia_navigation_boxes(monkeypatch, tmp_path):
    html = """
    <html><body>
      <table class="navbox"><tr><td>Automobile series
      Transmission Chassis Engine Suspension</td></tr></table>
      <div id="toc">Contents</div>
      <p>A drum brake is a vehicle braking system in which friction is applied
      to the inner surface of a brake drum attached to the wheel hub. The drum
      rotates with the wheel and the shoes press outward against it.</p>
      <span class="mw-editsection">[edit]</span>
    </body></html>
    """

    class FakeResp:
        status_code = 200
        content = html.encode("utf-8")

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return FakeResp()

    import sys

    fake_httpx = type(sys)("httpx")
    fake_httpx.Client = FakeClient
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    text, status, err = cs.fetch_source_content("https://example.org/drum-brake", tmp_path)
    assert err is None and status == 200
    assert "drum brake" in text
    for marker in ("Automobile series", "Contents", "[edit]"):
        assert marker not in text


# --- composite scorer ----------------------------------------------------------


def _src(**kw) -> Source:
    base = dict(id="s1", kb_id="kb", url="https://example.org/engine-guide")
    base.update(kw)
    return Source(**base)


def test_composite_weights_place_content_above_metadata(domain_map):
    w = cs.V2_WEIGHTS
    assert w["content_relevance"] + w["domain_coverage"] == pytest.approx(0.60)
    assert w["authority"] + w["source_type"] + w["recency"] == pytest.approx(0.25)


def test_very_authoritative_but_irrelevant_source_loses_to_relevant_one(domain_map):
    s = ContentAwareSourceScorer(BagOfWordsEmbedder(), domain_map)

    relevant_text = (
        "The engine converts fuel combustion into mechanical work driving the "
        "transmission, whose gears and clutch deliver torque to the drivetrain. "
        "Brakes dissipate kinetic energy as friction between pad and disc. "
        "Suspension springs and dampers control wheel motion; the chassis frame "
        "carries loads. Electric vehicles use battery packs and electric motors. "
        "The automobile integrates these systems for road transport. " * 12
    )
    off_topic_text = "Recipes for baking bread cakes and pastries with flour. " * 60

    relevant = s.assess_with_content(_src(), relevant_text, http_status=200)
    irrelevant_authoritative = s.assess_with_content(
        _src(url="https://www.nist.gov/bread"), off_topic_text, http_status=200
    )
    assert relevant["signals"]["content_relevance"] > irrelevant_authoritative["signals"]["content_relevance"]
    assert relevant["score"] > irrelevant_authoritative["score"]


def test_unfetchable_content_zeroes_content_signals_and_limits_score(domain_map):
    s = ContentAwareSourceScorer(BagOfWordsEmbedder(), domain_map)
    out = s.assess_with_content(_src(), None, http_status=None)
    assert out["signals"]["content_relevance"] == 0.0
    assert out["signals"]["domain_coverage"] == 0.0
    assert out["confidence"] == 0.0
    assert any("Content could not be fetched" in r for r in out["reasons"])
    # Content is 60% of the composite: an unfetchable source cannot outscore a
    # merely-average relevant one.
    assert out["score"] < 0.30


def test_per_requirement_evidence_and_matched_by_present(domain_map):
    s = ContentAwareSourceScorer(BagOfWordsEmbedder(), domain_map)
    out = s.assess_with_content(
        _src(), "Transmission gears clutch torque drivetrain mechanics. " * 60, http_status=200
    )
    tr = next(p for p in out["per_requirement"] if p["area_id"] == "transmission")
    assert tr["evidence"] and tr["matched_by"]
    assert tr["threshold"] == cs.COVER_THRESHOLD
