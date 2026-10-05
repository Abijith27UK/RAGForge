"""Human answer review: persistent, append-only, attributed (V8 STEP 4).

A review is a human's judgement about ONE generated answer for ONE benchmark
question in ONE evaluation run. Three rules:

1. **Append-only.** There is no update or delete path. A later review is a new
   row with its own id, reviewer and timestamp, so earlier judgements can
   never be silently overwritten. `list_answer_reviews` returns the whole
   history in creation order.
2. **Attributed.** An empty reviewer is rejected: a review with no identity is
   not evidence of human oversight and must not be stored as if it were.
3. **Structured.** The verdict (how correct?) and labels (what is wrong?) are
   closed vocabularies, so reviews can be aggregated without free-text
   parsing. Free-text notes exist too, but never *instead of* the structure.

The verdict/label vocabulary follows the V8 spec exactly: Correct, Mostly
correct, Partially correct, Incorrect, Should have abstained, Correctly
abstained; labels for factual error, unsupported claim, missing information,
wrong citation, irrelevant evidence, contradiction, incomplete answer,
excessive answer, correct answer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReviewVerdict(str, Enum):
    """The reviewer's overall judgement of the generated answer."""

    CORRECT = "correct"
    MOSTLY_CORRECT = "mostly_correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    SHOULD_HAVE_ABSTAINED = "should_have_abstained"
    CORRECTLY_ABSTAINED = "correctly_abstained"


#: Verdicts that affirm the answer (used by the human evaluator to derive a
#: correctness signal without inventing one).
AFFIRMATIVE_VERDICTS = frozenset(
    {ReviewVerdict.CORRECT, ReviewVerdict.CORRECTLY_ABSTAINED}
)
PARTIAL_VERDICTS = frozenset(
    {ReviewVerdict.MOSTLY_CORRECT, ReviewVerdict.PARTIALLY_CORRECT}
)


class ReviewLabel(str, Enum):
    """Structured issue labels a reviewer can attach alongside the verdict."""

    FACTUAL_ERROR = "factual_error"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    MISSING_INFORMATION = "missing_information"
    WRONG_CITATION = "wrong_citation"
    IRRELEVANT_EVIDENCE = "irrelevant_evidence"
    CONTRADICTION = "contradiction"
    INCOMPLETE_ANSWER = "incomplete_answer"
    EXCESSIVE_ANSWER = "excessive_answer"
    CORRECT_ANSWER = "correct_answer"


#: Ordinal convention for folding verdicts into a numeric correctness value.
#: This is a POLICY choice, not a measurement: the raw verdicts stay on every
#: review record, and any aggregate derived from them reports how many reviews
#: it averaged. Equal spacing (0.75 for "mostly") is arbitrary but published;
#: a different convention would be a documented change, not a silent one.
VERDICT_SCORES: dict[ReviewVerdict, float] = {
    ReviewVerdict.CORRECT: 1.0,
    ReviewVerdict.CORRECTLY_ABSTAINED: 1.0,
    ReviewVerdict.MOSTLY_CORRECT: 0.75,
    ReviewVerdict.PARTIALLY_CORRECT: 0.5,
    ReviewVerdict.INCORRECT: 0.0,
    ReviewVerdict.SHOULD_HAVE_ABSTAINED: 0.0,
}


def verdict_score(verdict: ReviewVerdict) -> float:
    return VERDICT_SCORES[verdict]


class AnswerReview(BaseModel):
    """One immutable human review of one evaluated answer."""

    id: str
    kb_id: str
    run_id: str = Field(description="The answer-evaluation run this review judges")
    question_id: str
    answer_id: str = Field(
        default="",
        description="The generated answer reviewed, when one exists "
        "(an abstention review may leave this empty)",
    )
    reviewer: str = Field(
        min_length=1,
        description="Identity of the human reviewer. Required: an unattributed "
        "review is not evidence of human oversight.",
    )
    verdict: ReviewVerdict
    labels: list[ReviewLabel] = Field(
        default_factory=list,
        description="Structured issue labels; empty is allowed (verdict alone "
        "may suffice) but labels are never invented for the reviewer",
    )
    notes: str = Field(default="", max_length=4000)
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("reviewer")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        value = (v or "").strip()
        if not value:
            raise ValueError("reviewer identity must not be blank")
        return value


__all__ = [
    "AFFIRMATIVE_VERDICTS",
    "AnswerReview",
    "PARTIAL_VERDICTS",
    "ReviewLabel",
    "ReviewVerdict",
    "VERDICT_SCORES",
    "verdict_score",
]
