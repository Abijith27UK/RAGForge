"""LLM provider abstraction.

The rest of the codebase depends only on the LLMProvider protocol; concrete
providers are selected via configuration. Providers must return *structured*
output so services can validate with Pydantic.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when an LLM provider fails (missing key, network, bad output)."""


class LLMProvider(ABC):
    """Interface for LLM providers returning structured Pydantic output."""

    name: str = "base"
    model: str = "unknown"
    is_mock: bool = False

    @abstractmethod
    def generate_structured(self, prompt: str, schema: type[T], system: str = "") -> T:
        """Generate a response validated against the given Pydantic schema."""


def _extract_json(text: str) -> str:
    """Extract a JSON object from an LLM response (handles ```json fences)."""
    text = text.strip()
    if text.startswith("```"):
        # strip fence line and trailing fence
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError("LLM response contained no JSON object")
    return text[start : end + 1]


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible chat APIs (OpenAI, vLLM, LM Studio, Ollama's
    OpenAI endpoint, etc.). Uses the `openai` package when available."""

    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = "") -> None:
        try:
            from openai import OpenAI  # optional dependency
        except ImportError as exc:
            raise LLMError(
                "The 'openai' package is not installed. "
                "Run: pip install -r backend/requirements-openai.txt"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model

    def generate_structured(self, prompt: str, schema: type[T], system: str = "") -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\nRespond with a single JSON object that validates "
                    f"against this JSON Schema:\n{schema_json}"
                ),
            }
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
            )
        except Exception as exc:  # network, auth, rate limit...
            logger.exception("LLM call failed")
            raise LLMError(f"LLM call failed: {exc}") from exc
        text = resp.choices[0].message.content or ""
        try:
            data = json.loads(_extract_json(text))
        except (json.JSONDecodeError, LLMError) as exc:
            raise LLMError(f"Could not parse LLM JSON output: {exc}") from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMError(f"LLM output failed schema validation: {exc}") from exc


class MockLLMProvider(LLMProvider):
    """Deterministic development-mode provider. Outputs are clearly labelled as
    mock via `is_mock = True` and `generated_by` fields downstream. NEVER used
    to fabricate quality scores or evaluation metrics."""

    name = "mock"
    model = "mock-1"
    is_mock = True

    def __init__(self, domain_hint: str = "") -> None:
        self.domain_hint = domain_hint

    def generate_structured(self, prompt: str, schema: type[T], system: str = "") -> T:
        if schema.__name__ == "AnswerDraftLLM":
            return self._answer_draft(prompt, schema)  # type: ignore[return-value]
        if schema.__name__ != "DomainSpecLLM":
            raise LLMError(
                f"Mock provider only supports domain analysis and answer drafts "
                f"(got {schema.__name__})"
            )
        # Build a generic, honest, deterministic spec from the domain hint.
        d = self.domain_hint or "the domain"
        domain = d
        return schema(  # type: ignore[call-arg]
            description=f"[DEV MOCK] Placeholder analysis for {domain}. Configure an LLM provider for real analysis.",
            subdomains=[f"{domain}: Fundamentals", f"{domain}: Core Systems", f"{domain}: Applications", f"{domain}: Standards and Safety"],
            key_concepts=[f"{domain} fundamentals", f"{domain} design", f"{domain} analysis", f"{domain} best practices"],
            entities=[f"key {domain} components", "standards bodies", "typical tools"],
            terminology=[f"{domain} terminology"],
            knowledge_requirements=[{"area": f"Core {domain} theory", "description": "Textbooks and lecture material", "priority": "high"}],
            recommended_source_categories=["University lecture notes (.edu)", "Standards organizations", "Peer-reviewed papers", "Technical reference books"],
        )

    def _answer_draft(self, prompt: str, schema: type[T]) -> T:
        """Deterministic answer draft parsed from the REAL prompt built by
        `app.services.answering.prompting`. Exercises prompt construction and
        structured round-trip without a network call. Clearly a mock: it quotes
        evidence sentences and never synthesizes.

        Sentence selection uses the SAME stop-word-aware term extractor as query
        processing, and ranks sentences by how many question terms they contain
        rather than taking the first substring hit. The earlier
        `len(term) > 3` substring approach was actively harmful: it dropped
        short-but-critical engineering acronyms (GM, KG, LCG) and kept stop-words
        like "when", so "What happens to GM when the centre of gravity rises?"
        quoted a sentence containing the word "when" from an unrelated section.
        """
        import re as _re

        from app.schemas.answer import ClaimDraft, ClaimType
        from app.services.answering.query_processor import extract_terms

        m = _re.search(r"<AVAILABLE_EVIDENCE_IDS>\s*([^<]+?)\s*</AVAILABLE_EVIDENCE_IDS>", prompt, _re.S)
        ids = [i.strip() for i in m.group(1).split(",") if i.strip()] if m else []
        ev = _re.search(r"<EVIDENCE>([\s\S]*?)</EVIDENCE>", prompt)
        claims: list[ClaimDraft] = []
        if ev:
            qmatch = _re.search(r"<USER_QUESTION>([\s\S]*?)</USER_QUESTION>", prompt)
            question = qmatch.group(1) if qmatch else ""
            terms = extract_terms(question)
            for block in _re.finditer(r"\[(ev_\d+)\]([^\n]*)\n([\s\S]*?)(?=\n\[ev_|\Z)", ev.group(1)):
                eid, _title, body = block.group(1), block.group(2), block.group(3)
                if eid not in ids:
                    continue
                best: tuple[int, int, str] | None = None
                for idx, sent in enumerate(_re.split(r"(?<=[.!?])\s+", body.strip())):
                    s = sent.strip()
                    if len(s) < 20:
                        continue
                    low = s.lower()
                    hits = sum(1 for t in terms if t and t in low)
                    if hits == 0:
                        continue
                    # Most question-term hits first; original order breaks ties.
                    candidate = (-hits, idx, s)
                    if best is None or candidate < best:
                        best = candidate
                if best is not None:
                    claims.append(ClaimDraft(text=best[2], claim_type=ClaimType.FACT,
                                             citation_evidence_ids=[eid]))
                if len(claims) >= 3:
                    break
        if not claims:
            return schema(answer_text="", claims=[], abstain=True,  # type: ignore[call-arg]
                          abstention_reason="[DEV MOCK] No evidence sentence matched the question.")
        return schema(  # type: ignore[call-arg]
            answer_text=" ".join(c.text for c in claims),
            claims=claims,
            abstain=False,
            abstention_reason="",
        )


def create_llm_provider(settings: Settings, domain_hint: str = "") -> LLMProvider:
    """Factory selecting the provider configured in settings."""
    provider = (settings.llm_provider or "").strip().lower()
    if provider == "mock":
        return MockLLMProvider(domain_hint)
    if provider in ("openai", "anthropic", "gemini", ""):
        if provider == "":
            if settings.allow_mock_llm:
                logger.warning(
                    "No LLM_PROVIDER configured; using DEV MOCK provider. "
                    "Outputs are placeholders, not real analysis."
                )
                return MockLLMProvider(domain_hint)
            raise LLMError(
                "No LLM provider configured. Set LLM_PROVIDER (openai|anthropic|gemini|mock), "
                "LLM_MODEL and LLM_API_KEY in backend/.env"
            )
        if not settings.llm_api_key and provider != "mock":
            raise LLMError(f"LLM_API_KEY is required for provider '{provider}'")
        if provider == "openai":
            return OpenAICompatibleProvider(
                api_key=settings.llm_api_key,
                model=settings.llm_model or "gpt-4o-mini",
                base_url=settings.llm_base_url,
            )
        # anthropic/gemini: structured via OpenAI-compat is not guaranteed; keep for future
        raise LLMError(
            f"Provider '{provider}' is declared but not yet implemented. "
            "Use 'openai' (or any OpenAI-compatible endpoint via LLM_BASE_URL) or 'mock'."
        )
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'")
