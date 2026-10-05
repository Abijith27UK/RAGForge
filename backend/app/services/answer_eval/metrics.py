"""Answer-quality metrics: schemas with an explicit UNKNOWN state.

Every metric is a `Measured` triple: `value`, `measured`, `reason`. A metric
that could not be computed is `measured=False` with a reason — it is never
silently 0.0, because "we could not check this" and "this scored zero" are
completely different findings and collapsing them is how evaluation systems end
up reporting confident nonsense.

`final_score` is provided for ranking convenience only. It is computed from the
submetrics that were actually MEASURED, and `final_score_computable` is False
whenever a headline submetric is unknown, so a partial run can never look like a
complete one.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator

from app.schemas.answer import GroundingState
from app.services.answer_eval.entailment import SupportJudgement


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Measured(BaseModel):
    """A single metric value that may legitimately be unknown."""

    value: float | None = None
    measured: bool = True
    reason: str = Field(default="", description="Why it is unknown, or how it was measured")
    sample_size: int | None = None

    @classmethod
    def unknown(cls, reason: str) -> "Measured":
        return cls(value=None, measured=False, reason=reason)

    @classmethod
    def of(cls, value: float, reason: str = "", sample_size: int | None = None) -> "Measured":
        return cls(value=round(float(value), 6), measured=True, reason=reason, sample_size=sample_size)


class ClaimCitationVerdict(BaseModel):
    """Per-claim citation audit — the unit that makes citation errors visible."""

    claim_id: str
    claim_text: str
    support_status: str
    #: The evaluator's five-state verdict for this claim (V8 STEP 6):
    #: SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | CONTRADICTED |
    #: UNVERIFIABLE. Derived from the claim's declared status, the entailment
    #: provider's judgement, and whether the claim cites anything at all —
    #: never from the claim's own say-so.
    evaluated_state: str = "UNVERIFIABLE"
    cited_evidence_ids: list[str] = Field(default_factory=list)
    resolved_chunk_ids: list[str] = Field(default_factory=list)
    #: Chunk ids that are REQUIRED for this question but were NOT cited here.
    missing_required_chunk_ids: list[str] = Field(default_factory=list)
    #: Cited chunks that the question did not require.
    irrelevant_chunk_ids: list[str] = Field(default_factory=list)
    entailment: SupportJudgement = SupportJudgement.UNKNOWN
    entailment_detail: str = ""
    entailment_method: str = ""
    #: True when at least one cited evidence item was judged NOT_SUPPORTED.
    has_invalid_citation: bool = False
    #: True when the claim asserts fact with no citation at all.
    uncited: bool = False
    problems: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.uncited


class AnswerQualityResult(BaseModel):
    """Per-question answer-quality result. Every submetric stays inspectable."""

    question_id: str
    question: str
    answerability: str
    answer_id: str = ""
    answer_trace_id: str = ""
    answer_run_id: str = ""
    answer_status: str = ""
    answer_text: str = ""

    # -- headline submetrics (never collapsed into one another) -------------
    citation_precision: Measured = Field(default_factory=lambda: Measured.unknown("not evaluated"))
    citation_recall: Measured = Field(default_factory=lambda: Measured.unknown("not evaluated"))
    evidence_support_rate: Measured = Field(default_factory=lambda: Measured.unknown("not evaluated"))
    unsupported_claim_rate: Measured = Field(default_factory=lambda: Measured.unknown("not evaluated"))
    contradiction_rate: Measured = Field(default_factory=lambda: Measured.unknown("not evaluated"))
    retrieval_hit_rate: Measured = Field(default_factory=lambda: Measured.unknown("not evaluated"))
    citation_completeness: Measured = Field(default_factory=lambda: Measured.unknown("not evaluated"))
    key_point_recall: Measured = Field(default_factory=lambda: Measured.unknown("not evaluated"))
    #: Spec name for completeness (V8 STEP 2/F): the SAME measurement as
    #: `key_point_recall`, exposed under the vocabulary the spec defines so
    #: neither name can drift from the other.
    expected_information_coverage: Measured = Field(
        default_factory=lambda: Measured.unknown("not evaluated")
    )
    #: Lexical similarity between the answer and a HUMAN reference answer,
    #: reported SEPARATELY from correctness: a lexical proxy is not a
    #: correctness judgement, and calling it one would overstate it.
    reference_answer_similarity: Measured = Field(
        default_factory=lambda: Measured.unknown("not evaluated")
    )
    correctness: Measured = Field(default_factory=lambda: Measured.unknown("not evaluated"))
    #: How much of the question's subject the answer text actually contains.
    #: Kept SEPARATE from groundedness: a perfectly grounded answer about the
    #: wrong subject scores low here and cannot hide behind its citations.
    question_answer_relevance: Measured = Field(
        default_factory=lambda: Measured.unknown("not evaluated")
    )
    #: Fraction of answer sentences sharing no question content term.
    #: OBSERVATION ONLY — never a failure, because background sentences
    #: legitimately explain context without repeating the question's words.
    excess_information: Measured = Field(
        default_factory=lambda: Measured.unknown("not evaluated")
    )
    # -- claim-state ratios (V8 STEP 6/7) ----------------------------------
    #: Fraction of claims the evaluator state = SUPPORTED. Denominator is ALL
    #: claims, so UNVERIFIABLE claims pull this down rather than vanish.
    supported_claim_ratio: Measured = Field(
        default_factory=lambda: Measured.unknown("not evaluated")
    )
    partial_claim_ratio: Measured = Field(
        default_factory=lambda: Measured.unknown("not evaluated")
    )
    #: UNSUPPORTED (asserts fact, cites nothing) + CONTRADICTED (its own
    #: evidence does not support it), over ALL claims. Deliberately distinct
    #: from `unsupported_claim_rate`, which counts ONLY uncited claims.
    unsupported_claim_ratio: Measured = Field(
        default_factory=lambda: Measured.unknown("not evaluated")
    )
    # -- citation-level fabrication / support (V8 STEP 7) ------------------
    #: Distinct evidence references that do not resolve to the retrieved
    #: evidence set, or whose provenance validation says 'invalid', over all
    #: distinct references the answer made. A reference that cannot be
    #: resolved to retrieved evidence is fabricated BY CONSTRUCTION at
    #: evaluation time: it points at something that was never produced.
    fabricated_citation_rate: Measured = Field(
        default_factory=lambda: Measured.unknown("not evaluated")
    )
    #: Citations attached to claims judged NOT_SUPPORTED, over citations that
    #: received a judgement at all. UNKNOWN-judged citations are excluded and
    #: their count reported via sample_size — 'we could not judge' is not
    #: 'supported'.
    unsupported_citation_rate: Measured = Field(
        default_factory=lambda: Measured.unknown("not evaluated")
    )

    # -- behavioural correctness --------------------------------------------
    abstention_expected: bool = False
    abstention_performed: bool = False
    abstention_correct: bool | None = None
    expected_grounding_state: str = "ANY_ACCEPTABLE"
    actual_grounding_state: str = ""
    grounding_state_correct: bool | None = None
    false_supported: bool = False
    false_unsupported: bool = False

    # -- raw detail ---------------------------------------------------------
    claim_verdicts: list[ClaimCitationVerdict] = Field(default_factory=list)
    required_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_rank_of_required: int | None = None
    warnings: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)

    final_score: float | None = None
    final_score_computable: bool = False
    evaluator_version: str = ""
    evaluator_name: str = ""
    #: True when the EVALUATOR itself is a model (LLM-as-judge). Distinct from
    #: `entailment_is_model_based` (claim-level support) and
    #: `relevance_is_model_based` (relevance): every model-based judgement on
    #: this result is flagged independently so criterion 13 holds per metric.
    evaluator_is_model_based: bool = False
    #: Which model/reviewers produced this evaluation (e.g.
    #: "deepseek-chat (prompt aej-v1)" or "reviewers: ada, grace").
    evaluator_detail: str = ""
    #: Raw LLM-judge output, kept verbatim for audit when the evaluator is
    #: model-based. Empty for deterministic/human evaluation.
    judge_raw: str = ""
    entailment_provider: str = ""
    entailment_is_model_based: bool = False
    #: Which relevance method produced `question_answer_relevance`, plus whether
    #: that method is model-based (it never is by default) and whether the
    #: pass/fail verdict sat within a thin margin of the threshold.
    relevance_method: str = ""
    relevance_is_model_based: bool = False
    relevance_close_call: bool = False
    #: Where the term weights came from ("uniform" = no corpus statistics, so
    #: the weaker plain method was used).
    relevance_weight_source: str = ""
    #: True/False when relevance was measured, None when it was not applicable
    #: (abstention, empty answer, or a question with no content terms).
    relevance_passed: bool | None = None

    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _score_requires_measured_inputs(self) -> "AnswerQualityResult":
        """`final_score` may only be computed when its inputs were measured."""
        if self.final_score is not None and not self.final_score_computable:
            raise ValueError(
                "final_score was supplied while final_score_computable is False; "
                "a score must never accompany an unmeasured input"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Pass requires citations correct AND abstention behaviour correct.

        Serialized deliberately. The aggregate exposes `pass_rate` while this
        exposes the per-question verdict, and a UI that cannot see `passed`
        will silently classify every question as a failure.
        """
        if self.abstention_correct is False:
            return False
        if self.grounding_state_correct is False and not self.false_supported:
            return False
        return not self.problems


class AnswerQualityAggregate(BaseModel):
    """Aggregate over a set of per-question results.

    Every aggregate reports how many questions it is based on, so a metric
    computed over 3 questions is never displayed as if it covered 28.
    """

    question_count: int = 0
    answerable_count: int = 0
    unanswerable_count: int = 0
    abstention_expected_count: int = 0

    citation_precision: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    citation_recall: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    evidence_support_rate: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    unsupported_claim_rate: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    contradiction_rate: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    retrieval_hit_rate: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    citation_completeness: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    correctness: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    key_point_recall: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    expected_information_coverage: Measured = Field(
        default_factory=lambda: Measured.unknown("no questions"),
        description="Human-authored key points covered by the answers — the "
        "spec's name for completeness (same measurement as key_point_recall)",
    )
    reference_answer_similarity: Measured = Field(
        default_factory=lambda: Measured.unknown("no questions"),
        description="Lexical similarity to human reference answers — NOT "
        "correctness",
    )
    question_answer_relevance: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    excess_information: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    supported_claim_ratio: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    partial_claim_ratio: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    unsupported_claim_ratio: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))
    fabricated_citation_rate: Measured = Field(
        default_factory=lambda: Measured.unknown("no questions"),
        description="Evidence references that do not resolve to retrieved "
        "evidence (or are provenance-invalid) — fabricated citations",
    )
    unsupported_citation_rate: Measured = Field(
        default_factory=lambda: Measured.unknown("no questions"),
        description="Citations on claims the entailment provider judged "
        "NOT_SUPPORTED, over judged citations only",
    )

    # -- behavioural --------------------------------------------------------
    relevance_failure_rate: Measured = Field(
        default_factory=lambda: Measured.unknown("no questions"),
        description="Answers that did not reach the relevance threshold — "
        "grounded but not about the question",
    )
    abstention_accuracy: Measured = Field(default_factory=lambda: Measured.unknown("no abstention cases"))
    grounding_state_accuracy: Measured = Field(default_factory=lambda: Measured.unknown("no expectations"))
    false_supported_rate: Measured = Field(default_factory=lambda: Measured.unknown("no cases"))
    false_unsupported_rate: Measured = Field(default_factory=lambda: Measured.unknown("no cases"))
    hallucination_rate: Measured = Field(
        default_factory=lambda: Measured.unknown("no cases"),
        description="Answerable-in-corpus question answered confidently without the required evidence",
    )

    pass_rate: Measured = Field(default_factory=lambda: Measured.unknown("no questions"))

    grounding_state_distribution: dict[str, int] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    unknown_metrics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


#: The five claim states an evaluation can assign (V8 STEP 6). Kept explicit
#: so the vocabulary cannot drift between evaluator, aggregate and UI.
CLAIM_EVALUATED_STATES = (
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "UNVERIFIABLE",
)

#: Which aggregate metrics gate `final_score`. Kept explicit so adding a metric
#: cannot silently change the headline.
SCORE_INPUTS = (
    "citation_precision",
    "citation_recall",
    "evidence_support_rate",
)


def average_measured(values: list[Measured], *, reason_if_empty: str) -> Measured:
    """Mean over measured values only; unknown if none were measured.

    Unknown inputs are excluded rather than treated as 0.0, and the exclusion
    count is reported so a partially-measured average is never mistaken for a
    complete one.
    """
    usable = [m.value for m in values if m.measured and m.value is not None]
    if not usable:
        return Measured.unknown(reason_if_empty)
    out = Measured.of(sum(usable) / len(usable), reason="mean over measured values", sample_size=len(usable))
    skipped = len(values) - len(usable)
    if skipped:
        out.reason += f"; {skipped} question(s) excluded as unmeasured"
    return out


def compute_final_score(result: AnswerQualityResult) -> tuple[float | None, bool]:
    """Weighted headline score, or (None, False) when inputs are unmeasured."""
    weights = {
        "citation_precision": 0.4,
        "citation_recall": 0.3,
        "evidence_support_rate": 0.3,
    }
    total = 0.0
    for name, weight in weights.items():
        metric: Measured = getattr(result, name)
        if not metric.measured or metric.value is None:
            return None, False
        total += weight * metric.value
    return round(total, 6), True


def grounding_states_compatible(expected: str, actual: str) -> bool:
    """Is the actual grounding state acceptable for this expectation?"""
    if expected == "ANY_ACCEPTABLE":
        return True
    if expected == actual:
        return True
    # A partially supported answer is acceptable when the reviewer expected a
    # full answer but the corpus genuinely only covers part of it; the reverse
    # (expecting partial, getting full) is also acceptable only when the answer
    # still cited the required evidence. Callers enforce the citation half.
    if expected == GroundingState.ANSWERED.value and actual == GroundingState.PARTIALLY_SUPPORTED.value:
        return True
    if expected == GroundingState.PARTIALLY_SUPPORTED.value and actual == GroundingState.ANSWERED.value:
        return True
    # Both "insufficient" and "no relevant evidence" mean the same thing to a
    # user: the system refused to answer.
    insufficient = {GroundingState.INSUFFICIENT_EVIDENCE.value, GroundingState.NO_RELEVANT_EVIDENCE.value}
    if expected in insufficient and actual in insufficient:
        return True
    return False


__all__ = [
    "AnswerQualityAggregate",
    "AnswerQualityResult",
    "CLAIM_EVALUATED_STATES",
    "ClaimCitationVerdict",
    "GroundingState",
    "Measured",
    "SCORE_INPUTS",
    "average_measured",
    "compute_final_score",
    "grounding_states_compatible",
]