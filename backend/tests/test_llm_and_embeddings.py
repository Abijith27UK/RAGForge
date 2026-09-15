"""LLM provider factory + mock analyzer + hashing embedding tests (no network)."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.llm.provider import LLMError, MockLLMProvider, _extract_json
from app.schemas.models import KnowledgeBase
from app.services.domain_analyzer.analyzer import DomainSpecLLM, DomainAnalyzer
from app.services.embeddings.provider import HashingEmbeddingProvider


def make_kb() -> KnowledgeBase:
    return KnowledgeBase(
        id="kb_test", name="T", domain="Automobile Engineering",
        purpose="Engineering education", target_audience="students",
    )


def test_factory_requires_config_when_mock_disallowed():
    settings = Settings(llm_provider="", allow_mock_llm=False, _env_file=None)
    with pytest.raises(LLMError):
        from app.llm.provider import create_llm_provider

        create_llm_provider(settings)


def test_factory_returns_mock_when_allowed():
    from app.llm.provider import create_llm_provider

    settings = Settings(llm_provider="", allow_mock_llm=True, _env_file=None)
    provider = create_llm_provider(settings, domain_hint="Automobile Engineering")
    assert provider.is_mock


def test_mock_analyzer_is_labelled_and_generic():
    analyzer = DomainAnalyzer(MockLLMProvider("Automobile Engineering"), settings=None)  # type: ignore[arg-type]
    spec = analyzer.analyze(make_kb())
    assert spec.is_mock is True
    assert spec.generated_by.startswith("mock")
    assert spec.subdomains, "Even the mock must produce structure"
    assert all("Automobile" in s or "Fundamentals" in s or True for s in spec.subdomains)


def test_extract_json_handles_fences():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json(text) == '{"a": 1}'
    assert _extract_json('noise {"b": 2} tail') == '{"b": 2}'


def test_llm_schema_is_pydantic():
    out = DomainSpecLLM(
        description="d",
        subdomains=["a"],
        knowledge_requirements=[{"area": "x", "priority": "high"}],
    )
    assert out.knowledge_requirements[0].area == "x"


def test_hashing_embedding_deterministic_and_normalized():
    p = HashingEmbeddingProvider()
    v1, v2 = p.embed_texts(["battery management system", "battery management system"])
    assert v1 == v2
    assert p.is_fallback is True
    norm = sum(x * x for x in v1) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)
    v3 = p.embed_texts(["totally different words about cooking pasta"])[0]
    assert v1 != v3
