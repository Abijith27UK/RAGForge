"""DomainSpec / schema validation tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.models import (
    DomainSpec,
    KnowledgeBaseCreate,
    RetrievalRequest,
    Source,
    SourceDecision,
)


def test_kb_create_requires_fields():
    with pytest.raises(ValidationError):
        KnowledgeBaseCreate(name="", domain="x", purpose="y", target_audience="z")


def test_kb_create_ok():
    kb = KnowledgeBaseCreate(
        name="Auto KB", domain="Automobile Engineering",
        purpose="education", target_audience="students",
    )
    assert kb.depth == "intermediate"


def test_retrieval_request_bounds():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="hello", top_k=0)
    with pytest.raises(ValidationError):
        RetrievalRequest(query="", top_k=5)
    r = RetrievalRequest(query="battery management system", top_k=50)
    assert r.top_k == 50


def test_domain_spec_defaults():
    spec = DomainSpec(kb_id="kb_1", domain="Automobile Engineering")
    assert spec.subdomains == []
    assert spec.is_mock is False


def test_source_decision_enum():
    assert SourceDecision("ACCEPT") is SourceDecision.ACCEPT
    with pytest.raises(ValueError):
        SourceDecision("MAYBE")
