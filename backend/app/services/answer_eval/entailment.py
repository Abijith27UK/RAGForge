"""Citation entailment: does the cited chunk actually support the claim?

`docs/grounding-and-citations.md` is explicit that the runtime pipeline performs
lexical-overlap support checking and that **semantic entailment is NOT
IMPLEMENTED**. That honesty has to survive into evaluation: the evaluator needs
to judge citation support too, and if it silently used a fuzzy string match and
called it "support", every downstream metric would inherit the overstatement.

So support is expressed as a three-valued judgement, never a boolean guess:

    SUPPORTED     the evidence was checked and supports the claim
    NOT_SUPPORTED the evidence was checked and does not support it
    UNKNOWN       support could not be determined by this provider

`UNKNOWN` is a real outcome, not an error. A provider that cannot decide must say
so; the default heuristic returns UNKNOWN rather than dressing up a heuristic as
semantics.

Provider availability is explicit:

* `HeuristicCitationEntailment` — deterministic, offline, default. Returns
  SUPPORTED / NOT_SUPPORTED from term coverage with a published threshold, and
  UNKNOWN when the evidence is too thin to judge either way.
* `LLMCitationEntailment` — optional, behind this abstraction, never the default
  product path. If its provider is unavailable it raises rather than silently
  degrading, so a run can never attribute LLM-judged metrics to a heuristic.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SupportJudgement(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"


class CitationEntailmentResult(BaseModel):
    """One provider's verdict for one (claim, evidence) pair."""

    judgement: SupportJudgement = SupportJudgement.UNKNOWN
    method: str = Field(
        default="",
        description="How the verdict was reached; surfaced so a heuristic is never mistaken for an NLI model",
    )
    detail: str = ""
    coverage: float | None = Field(
        default=None,
        description="Measured term coverage when the provider computes one",
    )
    provider_name: str = ""
    is_model_based: bool = False

    @property
    def decided(self) -> bool:
        return self.judgement is not SupportJudgement.UNKNOWN


class CitationEntailmentEvaluator(ABC):
    """Decides whether a piece of evidence supports a claim.

    Implementations must:
    * be honest about their method in `method`;
    * return UNKNOWN rather than guessing when they cannot decide;
    * never raise for an ordinary "no" — only for a provider failure, which the
      caller must surface rather than swallow.
    """

    name: str = "base"
    version: str = "1"
    is_model_based: bool = False

    @abstractmethod
    def evaluate(self, claim_text: str, evidence_text: str) -> CitationEntailmentResult: ...

    def describe(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "version": self.version,
            "is_model_based": self.is_model_based,
        }


#: Content terms below this length carry no discriminative signal.
_MIN_TERM_LEN = 3


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", (text or "").lower()) if len(t) >= _MIN_TERM_LEN]


#: Antonym pairs used for polarity conflict detection.
#:
#: Pure term-overlap cannot tell "must be positive" from "must be negative": the
#: claim shares nearly every term with the evidence, so a coverage-only judge
#: would call a CONTRADICTED claim SUPPORTED. Detecting that deterministically
#: matters more than it might seem — an evaluator that cannot flag a flipped sign
#: cannot support the claim that RAGForge knows when it does not know.
#:
#: This is a HEURISTIC over a small explicit pair list, labelled as such. It
#: only fires when the claim and the evidence each use OPPOSITE members of a
#: pair about the same sentence; it is not a general negation parser.
ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("positive", "negative"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("increasing", "decreasing"),
    ("higher", "lower"),
    ("greater", "less"),
    ("more", "less"),
    ("larger", "smaller"),
    ("larger", "smaller"),
    ("maximum", "minimum"),
    ("max", "min"),
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("allow", "prevent"),
    ("allows", "prevents"),
    ("safe", "unsafe"),
    ("stable", "unstable"),
    ("rise", "fall"),
    ("rises", "falls"),
    ("above", "below"),
    ("exceeds", "falls short"),
    ("true", "false"),
    ("required", "prohibited"),
    ("necessary", "unnecessary"),
    ("sufficient", "insufficient"),
    ("hot", "cold"),
    ("clockwise", "counterclockwise"),
    ("forward", "backward"),
    ("clockwise", "anticlockwise"),
)


def polarity_conflict(claim_text: str, evidence_text: str) -> tuple[bool, str]:
    """Do the claim and the evidence assert OPPOSITE polarity about something?

    Returns (conflict, explanation). Deterministic: it looks for the same
    antonym pair with one member in the claim and the other in the evidence,
    while requiring enough shared context that the two are talking about the
    same thing.
    """
    claim_low = (claim_text or "").lower()
    ev_low = (evidence_text or "").lower()
    claim_terms = set(_terms(claim_low))
    ev_terms = set(_terms(ev_low))

    # Require genuine topical overlap before treating a polarity difference as a
    # contradiction; unrelated sentences can easily contain opposite words.
    shared = claim_terms & ev_terms
    if len(shared) < 2:
        return False, ""

    for a, b in ANTONYM_PAIRS:
        a, b = a.lower(), b.lower()
        in_claim_a, in_claim_b = a in claim_low, b in claim_low
        in_ev_a, in_ev_b = a in ev_low, b in ev_low
        if in_claim_a and in_ev_b and not in_claim_b and not in_ev_a:
            return True, (
                f"claim asserts '{a}' where the evidence asserts '{b}' "
                f"(deterministic antonym check)"
            )
        if in_claim_b and in_ev_a and not in_claim_a and not in_ev_b:
            return True, (
                f"claim asserts '{b}' where the evidence asserts '{a}' "
                f"(deterministic antonym check)"
            )
    return False, ""


class HeuristicCitationEntailment(CitationEntailmentEvaluator):
    """Deterministic, offline term-coverage provider (the default).

    It publishes its own threshold and labels itself `lexical_coverage` so no
    caller can mistake it for semantic entailment. Where the evidence is too
    short or the claim has no content terms, it returns UNKNOWN rather than
    forcing a verdict.
    """

    name = "heuristic-lexical-coverage"
    version = "1"
    is_model_based = False

    #: Fraction of claim content terms that must appear in the evidence.
    SUPPORT_THRESHOLD = 0.6
    #: Below this the evidence is treated as contradicting the claim.
    CONTRADICT_THRESHOLD = 0.2
    #: Evidence shorter than this cannot be judged reliably.
    MIN_EVIDENCE_CHARS = 40

    def evaluate(self, claim_text: str, evidence_text: str) -> CitationEntailmentResult:
        claim_terms = _terms(claim_text)
        evidence = (evidence_text or "").strip()

        if not claim_terms:
            return CitationEntailmentResult(
                judgement=SupportJudgement.UNKNOWN,
                method=self.name,
                detail="claim has no content terms to check",
                provider_name=self.name,
            )
        if len(evidence) < self.MIN_EVIDENCE_CHARS:
            return CitationEntailmentResult(
                judgement=SupportJudgement.UNKNOWN,
                method=self.name,
                detail=(
                    f"evidence is only {len(evidence)} characters; too short to "
                    f"judge support reliably"
                ),
                provider_name=self.name,
            )

        low = evidence.lower()
        hits = sum(1 for t in claim_terms if t in low)
        coverage = hits / len(claim_terms)

        # A polarity conflict overrides coverage: a high-overlap claim that
        # reverses the evidence's direction is CONTRADICTED, not supported.
        conflict, conflict_detail = polarity_conflict(claim_text, evidence)
        if conflict:
            return CitationEntailmentResult(
                judgement=SupportJudgement.NOT_SUPPORTED,
                method=self.name,
                detail=(
                    f"{conflict_detail}; term coverage was {coverage:.0%} but the "
                    f"claim reverses the evidence's direction"
                ),
                coverage=round(coverage, 4),
                provider_name=self.name,
            )

        if coverage >= self.SUPPORT_THRESHOLD:
            judgement = SupportJudgement.SUPPORTED
            detail = (
                f"{hits}/{len(claim_terms)} claim terms appear in the evidence "
                f"({coverage:.0%}); LEXICAL COVERAGE ONLY, not semantic entailment"
            )
        elif coverage <= self.CONTRADICT_THRESHOLD:
            judgement = SupportJudgement.NOT_SUPPORTED
            detail = (
                f"only {hits}/{len(claim_terms)} claim terms appear in the evidence "
                f"({coverage:.0%}); LEXICAL COVERAGE ONLY"
            )
        else:
            judgement = SupportJudgement.UNKNOWN
            detail = (
                f"{hits}/{len(claim_terms)} claim terms appear ({coverage:.0%}), "
                f"which is ambiguous between support and non-support; semantic "
                f"entailment NOT PERFORMED"
            )

        return CitationEntailmentResult(
            judgement=judgement,
            method=self.name,
            detail=detail,
            coverage=round(coverage, 4),
            provider_name=self.name,
        )


class LLMCitationEntailment(CitationEntailmentEvaluator):
    """Optional model-based entailment judge, behind the same interface.

    NOT the default product path and not used by any test. The provider is
    injected; if it is missing the constructor raises so a caller can never
    quietly label heuristic results as model-based.
    """

    name = "llm-entailment"
    version = "1"
    is_model_based = True

    def __init__(self, provider, model: str = "") -> None:
        if provider is None:
            raise ValueError(
                "LLMCitationEntailment requires a provider; use "
                "HeuristicCitationEntailment for the offline default path"
            )
        self._provider = provider
        self._model = model or getattr(provider, "model", "unknown")

    def evaluate(self, claim_text: str, evidence_text: str) -> CitationEntailmentResult:
        from app.llm.provider import LLMError

        prompt = (
            "Decide whether the EVIDENCE supports the CLAIM. Answer with exactly one "
            "word: SUPPORTED, NOT_SUPPORTED, or UNKNOWN.\n\n"
            f"<CLAIM>{claim_text}</CLAIM>\n"
            f"<EVIDENCE>{evidence_text}</EVIDENCE>"
        )
        try:
            text = self._provider.generate_text(prompt)
        except LLMError as exc:
            # Surfaced, never swallowed: a failed judge must not be recorded as
            # a negative verdict.
            raise RuntimeError(f"LLM entailment provider failed: {exc}") from exc

        verdict = (text or "").strip().upper()
        mapping = {
            "SUPPORTED": SupportJudgement.SUPPORTED,
            "NOT_SUPPORTED": SupportJudgement.NOT_SUPPORTED,
            "UNKNOWN": SupportJudgement.UNKNOWN,
        }
        judgement = mapping.get(verdict, SupportJudgement.UNKNOWN)
        return CitationEntailmentResult(
            judgement=judgement,
            method=f"{self.name}:{self._model}",
            detail=f"model verdict: {verdict or 'unparseable'}",
            provider_name=self._model,
        )


def create_entailment_evaluator(kind: str = "heuristic", provider=None) -> CitationEntailmentEvaluator:
    """Factory. `heuristic` is offline and deterministic; `llm` requires a
    provider and fails loudly rather than degrading silently."""
    key = (kind or "heuristic").strip().lower()
    if key in ("heuristic", "lexical", "offline"):
        return HeuristicCitationEntailment()
    if key == "llm":
        return LLMCitationEntailment(provider)
    raise ValueError(
        f"Unknown citation entailment evaluator {kind!r}. "
        f"Supported: 'heuristic' (offline default), 'llm' (requires an explicit provider)."
    )


__all__ = [
    "CitationEntailmentEvaluator",
    "CitationEntailmentResult",
    "HeuristicCitationEntailment",
    "LLMCitationEntailment",
    "SupportJudgement",
    "create_entailment_evaluator",
    "polarity_conflict",
]