"""Benchmark review workflow (V9 Phase 1) — append-only human review.

The problem this module exists for: the shipped Automobile answer benchmark was
authored by an AGENT that read real indexed chunk texts and selected reference
evidence. That selection is real work and real provenance, but it is NOT human
review, and V8 recorded it honestly as ``human_review=PENDING`` with 0 of 28
questions approved.

V9's first priority is making those 28 questions reviewable *systematically*
without ever inventing a human label. This module provides:

* ``QuestionReview``   — one reviewer's structured verdict on one question,
  covering each dimension the V9 spec names.
* ``QuestionReviewLog``— an APPEND-ONLY in-memory log. Appending is the only
  mutation; a duplicate review id raises instead of overwriting. A later review
  of the same question/dimension supersedes an earlier one while both remain
  readable, so the history cannot be rewritten.
* ``derive_review_status`` — the review_status a set of reviews justifies.
* ``apply_review_status``  — sets ``review_status`` on the questions it is given
  and reports what it could NOT justify.

## What this module refuses to do

* It never marks a question ``approved`` without a review record that says so.
* It never fills in a missing dimension with an assumed verdict. An unassessed
  dimension is ``UNKNOWN`` and is reported as unassessed.
* It never mutates a frozen benchmark artifact. ``apply_review_status`` works on
  an in-memory model and the caller decides where (if anywhere) to persist it;
  writing to a frozen file is refused by
  ``assert_not_frozen_artifact``.
* ``approved`` requires EVERY required dimension to be assessed AND to be
  affirmative. A review that flags ambiguity or insufficient evidence cannot
  produce an approval, because those are exactly the labels that must not be
  laundered into ground truth.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.services.answer_eval.benchmark import (
    AnswerBenchmark,
    AnswerBenchmarkQuestion,
    BenchmarkValidationError,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BenchmarkReviewError(RuntimeError):
    """A review workflow rule was violated."""


class ReviewDimension(str, Enum):
    """The dimensions a reviewer must assess (V9 Phase 1).

    Every dimension is stated separately because collapsing them into one
    "looks fine" verdict is how an ambiguous question with weak evidence ends up
    approved.
    """

    #: Is the reference answer factually correct and complete?
    REFERENCE_ANSWER = "reference_answer"
    #: Are the key points correct — do they name what a correct answer must say?
    KEY_POINTS = "key_points"
    #: Are the acceptable answer elements genuinely acceptable?
    ACCEPTABLE_ELEMENTS = "acceptable_elements"
    #: Is the question ambiguous, or does it admit several defensible readings?
    AMBIGUITY = "ambiguity"
    #: Is the question actually answerable from the indexed corpus?
    ANSWERABILITY = "answerability"
    #: Is the selected required evidence sufficient to answer the question?
    EVIDENCE_SUFFICIENCY = "evidence_sufficiency"


#: Dimensions that must be assessed for a question to be eligible for approval.
#:
#: ``AMBIGUITY`` is on this list deliberately. A question nobody assessed for
#: ambiguity is precisely the question that gets silently approved with two
#: defensible readings, and the review is the only place that can catch it.
#: ``ACCEPTABLE_ELEMENTS`` is deliberately NOT here: many legitimate questions
#: have no acceptable-alternative list, so its absence is a fact about the
#: question rather than an unreviewed dimension.
REQUIRED_DIMENSIONS: tuple[ReviewDimension, ...] = (
    ReviewDimension.ANSWERABILITY,
    ReviewDimension.EVIDENCE_SUFFICIENCY,
    ReviewDimension.REFERENCE_ANSWER,
    ReviewDimension.KEY_POINTS,
    ReviewDimension.AMBIGUITY,
)


class DimensionVerdict(str, Enum):
    """A reviewer's verdict on ONE dimension.

    ``UNKNOWN`` is a first-class answer: it records that a human looked and could
    not determine the dimension. It is never converted into ``OK``.
    """

    OK = "ok"
    OK_WITH_NOTE = "ok_with_note"
    NEEDS_REVISION = "needs_revision"
    WRONG = "wrong"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


#: Verdicts that permit approval of a question.
AFFIRMATIVE_VERDICTS: frozenset[DimensionVerdict] = frozenset(
    {DimensionVerdict.OK, DimensionVerdict.OK_WITH_NOTE}
)


class QuestionReview(BaseModel):
    """One append-only structured review of one benchmark question."""

    review_id: str = Field(description="Unique id. A duplicate id is refused, never merged.")
    kb_id: str = ""
    benchmark_name: str = ""
    benchmark_fingerprint: str = Field(
        default="",
        description="Fingerprint of the benchmark content this review was made against. "
        "A review whose fingerprint no longer matches the benchmark is STALE and "
        "cannot be treated as approval.",
    )
    question_id: str
    reviewer: str = Field(description="Identity of the human reviewer. Required.")
    created_at: datetime = Field(default_factory=_utcnow)

    verdicts: dict[str, DimensionVerdict] = Field(
        default_factory=dict,
        description="Dimension value -> verdict. A missing dimension is UNASSESSED, "
        "never assumed to be fine.",
    )
    #: Free-text notes per dimension, keyed by the same dimension values.
    notes: dict[str, str] = Field(default_factory=dict)
    #: Where the reviewer verified the label (URL, chunk id, document id). Kept
    #: so an approval can be traced to something the reviewer actually read.
    evidence_checked: list[str] = Field(default_factory=list)
    #: The reviewer's own answerability determination, when it differs from the
    #: artifact. None = not stated, which is NOT the same as agreeing.
    answerability_verdict: str | None = None
    #: V10 addition: the reviewer's explicit per-question OUTCOME. One of
    #: pending | in_review | approved | rejected | ambiguous |
    #: insufficient_evidence. None = not chosen, in which case the V9
    #: dimension-based derivation decides. Optional and defaulting to None so
    #: every V9 review record still loads unchanged.
    outcome: str | None = Field(
        default=None,
        description="Explicit per-question outcome (V10). None = derived from the "
        "dimension verdicts instead.",
    )
    #: V10 addition: the annotation (reference answer) this review judges.
    #: Empty = not recorded (a V9-style review of the question as a whole).
    annotation_id: str = ""
    #: Questions the reviewer found the artifact does not address.
    unresolved: list[str] = Field(default_factory=list)
    #: Review id this one replaces, when correcting an earlier review.
    supersedes: str = ""

    @model_validator(mode="after")
    def _reviewer_required(self) -> "QuestionReview":
        if not self.reviewer.strip():
            raise BenchmarkReviewError(
                "a review must name a reviewer; an unattributed approval is "
                "indistinguishable from an agent approving its own labels"
            )
        return self

    def assessed_dimensions(self) -> set[ReviewDimension]:
        out: set[ReviewDimension] = set()
        for key in self.verdicts:
            try:
                out.add(ReviewDimension(key))
            except ValueError:
                continue
        return out

    def blocks_approval(self) -> list[str]:
        """Reasons this review CANNOT support an approval. Empty means it can.

        A blocking reason is a non-affirmative verdict on a required dimension,
        or a required dimension the reviewer did not assess at all.
        """
        blockers: list[str] = []
        for dim in REQUIRED_DIMENSIONS:
            verdict = self.verdicts.get(dim.value)
            if verdict is None:
                blockers.append(f"{dim.value}: not assessed")
            elif verdict not in AFFIRMATIVE_VERDICTS:
                blockers.append(f"{dim.value}: {verdict.value}")
        return blockers

    def is_approval(self) -> bool:
        """True only when every required dimension was assessed and affirmative.

        Deliberately strict: this is the single predicate standing between an
        agent-authored label and a claim of human ground truth.
        """
        return not self.blocks_approval()

    def model_fingerprint(self) -> str:
        payload = {
            "kb_id": self.kb_id,
            "benchmark_name": self.benchmark_name,
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "question_id": self.question_id,
            "reviewer": self.reviewer,
            "verdicts": {k: v.value for k, v in sorted(self.verdicts.items())},
            "notes": self.notes,
            "evidence_checked": self.evidence_checked,
            "answerability_verdict": self.answerability_verdict,
            "outcome": self.outcome,
            "annotation_id": self.annotation_id,
            "unresolved": self.unresolved,
            "supersedes": self.supersedes,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]


class QuestionReviewLog:
    """Append-only collection of ``QuestionReview`` records.

    The ONLY mutation is ``append``. There is no update, no delete, and no
    "replace" — a correction is a NEW review carrying ``supersedes``, so the
    record that a label was once claimed differently survives.
    """

    def __init__(self, reviews: list[QuestionReview] | None = None) -> None:
        self._reviews: dict[str, QuestionReview] = {}
        self._order: list[str] = []
        for review in reviews or []:
            self.append(review)

    def append(self, review: QuestionReview) -> QuestionReview:
        """Add a review. A duplicate review_id raises; it is never merged."""
        if review.review_id in self._reviews:
            raise BenchmarkReviewError(
                f"review {review.review_id!r} already exists; reviews are append-only. "
                f"Record a correction as a new review with `supersedes` set."
            )
        self._reviews[review.review_id] = review
        self._order.append(review.review_id)
        return review

    def all(self) -> list[QuestionReview]:
        """Every review, in append order."""
        return [self._reviews[rid] for rid in self._order]

    def for_question(self, question_id: str) -> list[QuestionReview]:
        return [r for r in self.all() if r.question_id == question_id]

    def effective_for_question(self, question_id: str) -> QuestionReview | None:
        """The latest review for a question, ignoring superseded ones.

        A superseded review is dropped from the *current* picture but stays in
        ``all()`` — history is preserved, only the active claim changes.
        """
        reviews = self.for_question(question_id)
        superseded = {r.supersedes for r in reviews if r.supersedes}
        active = [r for r in reviews if r.review_id not in superseded]
        return active[-1] if active else None

    def reviewers(self) -> list[str]:
        return sorted({r.reviewer for r in self.all()})

    def question_ids(self) -> list[str]:
        return sorted({r.question_id for r in self.all()})


class ReviewStatus(str, Enum):
    """Per-question review status, consistent with the benchmark vocabulary."""

    PENDING = "pending"
    REVIEWED = "reviewed"
    APPROVED = "approved"


def derive_review_status(
    reviews: list[QuestionReview],
    *,
    benchmark_fingerprint: str = "",
) -> ReviewStatus:
    """The status a set of reviews justifies for one question.

    * no review                       -> PENDING
    * review(s) that cannot approve   -> REVIEWED (a human looked; approval is
      blocked by a flagged dimension)
    * an approval review              -> APPROVED

    Reviews whose ``benchmark_fingerprint`` disagrees with the benchmark being
    assessed are ignored entirely: a review of different content says nothing
    about this content.
    """
    relevant = [
        r
        for r in reviews
        if not benchmark_fingerprint
        or not r.benchmark_fingerprint
        or r.benchmark_fingerprint == benchmark_fingerprint
    ]
    if not relevant:
        return ReviewStatus.PENDING
    latest = relevant[-1]
    if latest.is_approval():
        return ReviewStatus.APPROVED
    return ReviewStatus.REVIEWED


class ReviewApplication(BaseModel):
    """Outcome of applying reviews to a benchmark, per question."""

    statuses: dict[str, str] = Field(default_factory=dict)
    approved_count: int = 0
    reviewed_count: int = 0
    pending_count: int = 0
    stale_review_question_ids: list[str] = Field(default_factory=list)
    blockers: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def all_approved(self) -> bool:
        return self.pending_count == 0 and self.reviewed_count == 0


def apply_review_status(
    benchmark: AnswerBenchmark,
    reviews: list[QuestionReview],
) -> ReviewApplication:
    """Set ``review_status`` on each question from the reviews that justify it.

    Mutates the benchmark model IN MEMORY and returns what was applied. It does
    not touch the filesystem: persisting a reviewed benchmark is the caller's
    decision and must never overwrite a frozen artifact (see
    ``assert_not_frozen_artifact``).

    A question that was already ``approved`` in the artifact but has NO approval
    review is DOWNGRADED to pending. Carrying an unbacked approval forward is
    exactly the failure mode this module exists to prevent.
    """
    fp = benchmark.fingerprint()
    by_question: dict[str, list[QuestionReview]] = {}
    stale: list[str] = []
    for r in reviews:
        if r.benchmark_fingerprint and r.benchmark_fingerprint != fp:
            stale.append(r.question_id)
            continue
        by_question.setdefault(r.question_id, []).append(r)

    result = ReviewApplication(stale_review_question_ids=sorted(set(stale)))
    for question in benchmark.questions:
        q_reviews = by_question.get(question.question_id, [])
        status = derive_review_status(q_reviews)
        question.review_status = status.value
        result.statuses[question.question_id] = status.value
        if status is ReviewStatus.APPROVED:
            result.approved_count += 1
        elif status is ReviewStatus.REVIEWED:
            result.reviewed_count += 1
            latest = q_reviews[-1]
            result.blockers[question.question_id] = latest.blocks_approval()
        else:
            result.pending_count += 1
    return result


#: Files that must never be rewritten by the review workflow.
FROZEN_ARTIFACT_MARKERS: tuple[str, ...] = ("-results.json", "-frozen")


def assert_not_frozen_artifact(path: str | Path) -> None:
    """Refuse to write to a frozen artifact.

    Frozen artifacts are the historical record V9 is measured against. Rewriting
    one would make every result that referenced it unreproducible.
    """
    name = Path(path).name
    if name in FROZEN_BENCHMARK_FILES:
        raise BenchmarkValidationError(
            f"{name} is a FROZEN historical artifact and must never be rewritten; "
            f"record new results under a new file name"
        )
    for marker in FROZEN_ARTIFACT_MARKERS:
        if marker in name:
            raise BenchmarkValidationError(
                f"{name} looks like a frozen result artifact ({marker!r}); "
                f"must never be rewritten"
            )


#: Benchmark artifacts that are part of the V8 record and must stay byte-stable.
FROZEN_BENCHMARK_FILES: frozenset[str] = frozenset(
    {
        "answer-evaluation-v1.json",
        "answer-evaluation-v1-results.json",
        "answer-quality-automobile-v1.json",
        "automobile-engineering-baseline-v1.json",
        "automobile-engineering-baseline-v1-results.json",
        "source-selection-experiment-v1.json",
        "source-selection-experiment-v1-results.json",
        "source-selection-experiment-v2.json",
        "source-selection-experiment-v2-results.json",
        "source-selection-experiment-v2-paired-analysis.json",
        "source-selection-experiment-v3.json",
        "source-quality-v2-pool-report.json",
        "v2-pool-report.json",
    }
)


def unreviewed_questions(benchmark: AnswerBenchmark) -> list[str]:
    """Question ids that are not approved — the honest gap, never a pass."""
    return [
        q.question_id
        for q in benchmark.questions
        if (q.review_status or "").strip().lower() != ReviewStatus.APPROVED.value
    ]


def review_coverage(
    benchmark: AnswerBenchmark,
    reviews: list[QuestionReview],
) -> dict[str, Any]:
    """Measure review coverage: how much of the benchmark is actually reviewed.

    Reports the numbers rather than a verdict, including how many questions have
    NO review at all — the state V9 must make visible rather than paper over.
    """
    reviewed_ids = {r.question_id for r in reviews}
    question_ids = {q.question_id for q in benchmark.questions}
    assessed_dimensions: dict[str, int] = {d.value: 0 for d in ReviewDimension}
    for r in reviews:
        for dim in r.assessed_dimensions():
            assessed_dimensions[dim.value] += 1
    approvals = sum(1 for r in reviews if r.is_approval())
    return {
        "question_count": len(question_ids),
        "reviewed_question_count": len(reviewed_ids & question_ids),
        "unreviewed_question_count": len(question_ids - reviewed_ids),
        "unreviewed_question_ids": sorted(question_ids - reviewed_ids),
        "review_count": len(reviews),
        "approval_review_count": approvals,
        "blocked_review_count": len(reviews) - approvals,
        "reviewers": sorted({r.reviewer for r in reviews}),
        "assessed_dimension_counts": assessed_dimensions,
        "unassessed_dimensions": [
            d.value for d in ReviewDimension if assessed_dimensions[d.value] == 0
        ],
    }


__all__ = [
    "AFFIRMATIVE_VERDICTS",
    "BenchmarkReviewError",
    "DimensionVerdict",
    "FROZEN_BENCHMARK_FILES",
    "QuestionReview",
    "QuestionReviewLog",
    "REQUIRED_DIMENSIONS",
    "ReviewApplication",
    "ReviewDimension",
    "ReviewStatus",
    "apply_review_status",
    "assert_not_frozen_artifact",
    "derive_review_status",
    "review_coverage",
    "unreviewed_questions",
]
