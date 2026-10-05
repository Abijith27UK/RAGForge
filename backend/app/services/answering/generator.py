"""Phase 12 — grounded answer generation behind a provider-agnostic interface.

Three pieces:
* `AnswerGenerator` ABC — the only thing the pipeline depends on.
* `LLMAnswerGenerator` — wraps ANY `LLMProvider` (OpenAI-compatible today,
  others via the same interface). The provider is injected, never imported by
  name, so core logic is not coupled to one vendor.
* `ExtractiveMockAnswerGenerator` — deterministic, LLM-free extractive mock.
  It selects sentences from the evidence that share terms with the question and
  cites the evidence they came from. REQUIRED for hermetic tests: no network,
  no API key. It is labelled `is_mock = True` and never used to fabricate
  quality claims.

The generator receives ONLY: the question, the structured evidence (already
wrapped as untrusted data by `prompting.build_prompt`), the answer policy and
the gate's reason. It never sees provider secrets.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from app.llm.provider import LLMError, LLMProvider
from app.schemas.answer import (
    AnswerPolicy,
    ClaimDraft,
    ClaimType,
    EvidenceSet,
    GeneratedAnswer,
    QueryPlan,
)
from app.services.answering.prompting import build_prompt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnswerGenerationError(RuntimeError):
    """Raised when a generator fails outright (mapped to GENERATION_FAILED)."""


#: Structured schema the LLM must return. Kept flat so every provider can emit it.
class AnswerDraftLLM(BaseModel):
    answer_text: str = ""
    claims: list[ClaimDraft] = Field(default_factory=list)
    abstain: bool = False
    abstention_reason: str = ""


class AnswerGenerator(ABC):
    """Produces a raw, UNVALIDATED answer from selected evidence.

    Contract every implementation must honour:
    * it receives only the evidence it was given — never a database handle, a
      repository, a vector store or a file path;
    * it never sees provider secrets (keys live in the provider, not here);
    * its output is a DRAFT: citation ids it invents are the validator's
      problem to catch, and must be preserved rather than pre-filtered.

    `unavailable` / `unavailable_reason` exist so a degraded generator can say
    so out loud. A generator that could not run is NEVER allowed to look like a
    generator that ran.
    """

    name: str = "base"
    version: str = "v7.2"
    is_mock: bool = False
    unavailable: bool = False
    unavailable_reason: str = ""

    @abstractmethod
    def generate(
        self,
        plan: QueryPlan,
        evidence: EvidenceSet,
        policy: AnswerPolicy,
        assessment_reason: str,
    ) -> GeneratedAnswer: ...

    def describe(self) -> dict[str, str | bool]:
        """Honest, secret-free description of what produced an answer."""
        return {
            "name": self.name,
            "version": self.version,
            "model": str(getattr(self, "model", "")),
            "is_mock": self.is_mock,
            "unavailable": self.unavailable,
            "unavailable_reason": self.unavailable_reason,
        }


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _pick_sentences(question_terms: list[str], content: str, limit: int = 3) -> list[str]:
    """Deterministic sentence selection: most question-term hits first,
    original order preserved as a tie-break."""
    scored: list[tuple[int, int, str]] = []
    for idx, sent in enumerate(_SENTENCE_SPLIT.split(content.strip())):
        s = sent.strip()
        if len(s) < 20:
            continue
        low = s.lower()
        hits = sum(1 for t in question_terms if t and t in low)
        if hits:
            scored.append((-hits, idx, s))
    scored.sort()
    return [s for _, _, s in scored[:limit]]


class ExtractiveMockAnswerGenerator(AnswerGenerator):
    """Deterministic extractive mock (NO LLM, no network, no key).

    Honesty: the output is a set of evidence sentences quoted back with their
    citations — NOT a synthesized answer. `is_mock = True` and the generation
    notes say so explicitly so no UI can present it as model reasoning.
    """

    name = "extractive-mock"
    version = "v7.2"
    is_mock = True

    def generate(
        self,
        plan: QueryPlan,
        evidence: EvidenceSet,
        policy: AnswerPolicy,
        assessment_reason: str,
    ) -> GeneratedAnswer:
        notes = [
            "generated_by=extractive-mock: sentences quoted from retrieved evidence, "
            "selected by deterministic question-term overlap. NOT an LLM synthesis."
        ]
        if not evidence.items:
            return GeneratedAnswer(
                text="",
                abstain_requested=True,
                abstention_reason="No evidence was available to ground an answer.",
                notes=notes,
            )
        terms = plan.extracted_terms
        claims: list[ClaimDraft] = []
        quoted: list[str] = []
        for item in evidence.items:
            for sent in _pick_sentences(terms, item.content, limit=2):
                if sent in quoted:
                    continue
                quoted.append(sent)
                claims.append(
                    ClaimDraft(
                        text=sent,
                        claim_type=ClaimType.FACT,
                        citation_evidence_ids=[item.evidence_id],
                    )
                )
                if len(claims) >= 5:
                    break
            if len(claims) >= 5:
                break
        if not claims:
            return GeneratedAnswer(
                text="",
                abstain_requested=True,
                abstention_reason=(
                    "No sentence in the retrieved evidence shares enough terms with "
                    "the question to be quoted as a grounded answer."
                ),
                notes=notes,
            )
        text = " ".join(c.text for c in claims)
        return GeneratedAnswer(
            text=text,
            claims=claims,
            notes=notes,
            warnings=[
                "extractive mock generator: answer text quotes evidence sentences; "
                "no language-model synthesis was performed"
            ],
        )


class LLMAnswerGenerator(AnswerGenerator):
    """Grounded generation through the existing LLMProvider abstraction."""

    name = "llm"
    version = "v7.2"
    is_mock = False  # instance-level override below reflects the provider

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        # Honest labelling: if the underlying provider is the deterministic
        # mock, this generator's output is a mock too and must say so.
        self.is_mock = bool(getattr(provider, "is_mock", False))

    @property
    def model(self) -> str:
        return f"{self._provider.name}/{self._provider.model}"

    def generate(
        self,
        plan: QueryPlan,
        evidence: EvidenceSet,
        policy: AnswerPolicy,
        assessment_reason: str,
    ) -> GeneratedAnswer:
        built = build_prompt(plan, evidence, policy, assessment_reason)
        try:
            draft: AnswerDraftLLM = self._provider.generate_structured(
                built.user, AnswerDraftLLM, system=built.system
            )
        except LLMError as exc:
            raise AnswerGenerationError(str(exc)) from exc
        notes = [f"generated_by={self.model}"]
        if built.truncated:
            notes.append("evidence was truncated to the policy context budget")
        return GeneratedAnswer(
            text=draft.answer_text or "",
            claims=list(draft.claims),
            abstain_requested=bool(draft.abstain),
            abstention_reason=draft.abstention_reason or "",
            notes=notes,
            warnings=list(built.warnings),
        )


class UnavailableAnswerGenerator(AnswerGenerator):
    """Marker for a generator that could not be constructed at all.

    It is never called to produce text — `FallbackAnswerGenerator` reports it as
    the generator that was WANTED, so an audit can say "you asked for openai and
    got the extractive mock" instead of quietly losing that fact.
    """

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.unavailable = True
        self.unavailable_reason = reason

    def generate(
        self,
        plan: QueryPlan,
        evidence: EvidenceSet,
        policy: AnswerPolicy,
        assessment_reason: str,
    ) -> GeneratedAnswer:  # pragma: no cover - defensive; never reached
        raise AnswerGenerationError(
            f"generator {self.name!r} is unavailable: {self.unavailable_reason}"
        )


class FallbackAnswerGenerator(AnswerGenerator):
    """Wraps a preferred generator and an explicit fallback.

    Exists to satisfy one rule from the V7 spec: *never silently fall back from
    a real LLM to a mock*. When the preferred generator cannot be used, this
    class:

    1. records `unavailable = True` and a reason on ITSELF, so any caller that
       asks "did the real model run?" gets the truth;
    2. marks every answer it produces with a warning that names the fallback and
       the reason, so the answer object itself carries the disclosure;
    3. never pretends the fallback output came from the preferred provider.

    The disclosure is a WARNING, not an error: the request still succeeds, it
    just succeeds visibly degraded.
    """

    name = "fallback"
    version = "v7.2"

    def __init__(self, preferred: AnswerGenerator, fallback: AnswerGenerator, reason: str) -> None:
        self._preferred = preferred
        self._fallback = fallback
        self.is_mock = fallback.is_mock
        self.unavailable = True
        self.unavailable_reason = reason

    @property
    def model(self) -> str:
        return str(getattr(self._fallback, "model", self._fallback.name))

    def describe(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "version": self.version,
            "model": self.model,
            "is_mock": self.is_mock,
            "unavailable": True,
            "unavailable_reason": self.unavailable_reason,
            "preferred_generator": self._preferred.name,
            "fallback_generator": self._fallback.name,
        }

    def generate(
        self,
        plan: QueryPlan,
        evidence: EvidenceSet,
        policy: AnswerPolicy,
        assessment_reason: str,
    ) -> GeneratedAnswer:
        result = self._fallback.generate(plan, evidence, policy, assessment_reason)
        result.warnings.append(
            f"FALLBACK GENERATOR: the configured generator '{self._preferred.name}' "
            f"was unavailable ({self.unavailable_reason}); this answer was produced "
            f"by '{self._fallback.name}'"
            + (" — a deterministic mock, NOT a language model" if self._fallback.is_mock else "")
        )
        result.notes.append(f"generator_fallback_from={self._preferred.name}")
        return result


def create_answer_generator(settings, provider: LLMProvider | None = None) -> AnswerGenerator:
    """Choose the generator from configuration.

    Provider resolution reuses the EXISTING `create_llm_provider` contract:
    * LLM_PROVIDER=mock    -> LLMAnswerGenerator(MockLLMProvider): the REAL
      prompt-building and structured-output path runs, with no network/key.
      This is what the tests use.
    * LLM_PROVIDER=openai (+key) -> LLMAnswerGenerator(OpenAICompatibleProvider)
    * nothing usable       -> a FallbackAnswerGenerator wrapping the deterministic
      extractive mock. The fallback is DISCLOSED (generator.unavailable=True +
      an answer-level warning naming the reason) rather than silent.

    Note the difference between the two mock paths, which the answer records
    distinguish: LLM_PROVIDER=mock means the prompt path genuinely ran against a
    deterministic stand-in, while a fallback means the configured provider could
    not be constructed at all.
    """
    if provider is not None:
        return LLMAnswerGenerator(provider)
    from app.llm.provider import create_llm_provider

    try:
        return LLMAnswerGenerator(create_llm_provider(settings))
    except LLMError as exc:
        reason = str(exc)
        logger.warning(
            "No usable LLM provider for answering (%s); falling back to the "
            "deterministic extractive mock. This is DISCLOSED on the answer.",
            reason,
        )
        return FallbackAnswerGenerator(
            preferred=UnavailableAnswerGenerator("llm", reason),
            fallback=ExtractiveMockAnswerGenerator(),
            reason=reason,
        )


__all__ = [
    "AnswerDraftLLM",
    "AnswerGenerationError",
    "AnswerGenerator",
    "ExtractiveMockAnswerGenerator",
    "FallbackAnswerGenerator",
    "LLMAnswerGenerator",
    "UnavailableAnswerGenerator",
    "create_answer_generator",
]
