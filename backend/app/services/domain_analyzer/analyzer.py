"""Domain Analyzer: turns a high-level domain + use case into a structured DomainSpec.

The analyzer is generic: it does not hard-code any domain. The LLM provider
abstraction supplies the intelligence; this service handles prompt construction,
validation, and persistence concerns.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.config import Settings
from app.llm.provider import LLMError, LLMProvider
from app.schemas.models import DomainSpec, KnowledgeBase, KnowledgeRequirement

logger = logging.getLogger(__name__)


class DomainSpecLLM(BaseModel):
    """Schema the LLM must produce; validated with Pydantic before use."""

    description: str = ""
    subdomains: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    terminology: list[str] = Field(default_factory=list)
    knowledge_requirements: list[KnowledgeRequirement] = Field(default_factory=list)
    recommended_source_categories: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = (
    "You are a knowledge-engineering assistant that designs RAG knowledge bases. "
    "You produce factual, well-structured domain analyses. Respond only with JSON."
)


def build_analysis_prompt(kb: KnowledgeBase) -> str:
    return (
        f"Analyze the following domain for the purpose of building a retrieval-augmented "
        f"generation (RAG) knowledge base.\n\n"
        f"Domain: {kb.domain}\n"
        f"Intended use case / purpose: {kb.purpose}\n"
        f"Target audience: {kb.target_audience}\n"
        f"Desired depth: {kb.depth}\n\n"
        "Produce:\n"
        "1. A one-paragraph description of the domain as relevant to this use case.\n"
        "2. 5-12 subdomains covering the important areas of the domain.\n"
        "3. 10-25 key concepts a retrieval corpus should cover.\n"
        "4. Important entities (systems, components, organizations, standards, tools).\n"
        "5. Domain-specific terminology a retrieval system should handle.\n"
        "6. Knowledge requirements: what kinds of knowledge the corpus must contain, "
        "each with an area name, description, and priority (high|medium|low).\n"
        "7. Recommended source categories (e.g. 'university lecture notes', 'SAE standards', "
        "'peer-reviewed journals') that would authoritatively cover this domain."
    )


class DomainAnalyzer:
    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    @property
    def is_mock(self) -> bool:
        return self._provider.is_mock

    def analyze(self, kb: KnowledgeBase) -> DomainSpec:
        prompt = build_analysis_prompt(kb)
        result = self._provider.generate_structured(prompt, DomainSpecLLM, system=SYSTEM_PROMPT)
        spec = DomainSpec(
            kb_id=kb.id,
            domain=kb.domain,
            description=result.description,
            subdomains=result.subdomains,
            key_concepts=result.key_concepts,
            entities=result.entities,
            terminology=result.terminology,
            knowledge_requirements=result.knowledge_requirements,
            recommended_source_categories=result.recommended_source_categories,
            generated_by=f"{self._provider.name}:{self._provider.model}",
            is_mock=self._provider.is_mock,
        )
        logger.info("Domain analysis for KB %s produced %d subdomains", kb.id, len(spec.subdomains))
        return spec


def analyze_domain(kb: KnowledgeBase, settings: Settings) -> tuple[DomainSpec, bool]:
    """Convenience helper: create provider from settings and analyze.

    Returns (spec, used_mock). Raises LLMError with an actionable message when
    configuration is missing and mock mode is disallowed.
    """
    from app.llm.provider import create_llm_provider

    try:
        provider = create_llm_provider(settings, domain_hint=kb.domain)
    except LLMError:
        raise
    analyzer = DomainAnalyzer(provider, settings)
    return analyzer.analyze(kb), provider.is_mock
