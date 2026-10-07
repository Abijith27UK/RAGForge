"""V10 — Human ground truth for the answer benchmark.

The problem this module exists for: the shipped Automobile answer benchmark
(`benchmarks/answer-quality-automobile-v1.json`) carries 28 questions whose
reference evidence was selected by an agent reading real indexed chunk text.
That is real work and real provenance, but it is NOT human ground truth, and
V9 made the gap visible: 28/28 questions unreviewed, `correctness` and
`key_point_recall` UNKNOWN.

This module is the V10 workflow that lets a HUMAN reviewer close that gap:

```
QUESTION → CORPUS EVIDENCE → REFERENCE ANSWER → KEY POINTS
        → ACCEPTABLE ELEMENTS → PROVENANCE → HUMAN REVIEW
        → APPROVED BENCHMARK → FROZEN BENCHMARK VERSION
```

## What this module refuses to do

* It never turns an authoring draft into `approved` without an affirmative
  review from a named human reviewer.
* It never treats a missing artefact as a zero. A coverage rate with no
  denominator is `Measured.unknown(...)`, never `0.0`.
* It never approves a reference answer whose evidence does not validate
  against the live corpus (missing chunk, changed content hash, wrong
  document/knowledge base).
* It never mutates a frozen benchmark version. Freezing creates a NEW
  immutable record; a correction is a new version.
* It never fabricates a human label. The `author` on an annotation and the
  `reviewer` on a review are both required and both recorded.

## Relationship to V9's `benchmark_review.py`

`benchmark_review.py` owns the append-only review *record* (dimension
verdicts, notes, supersedes). This module owns the *ground truth* the review
is about (reference answer, key points, acceptable elements, evidence,
provenance) and the workflow on top of it (states, completeness, approval,
freeze). Where a concept already exists — `ReviewDimension`,
`DimensionVerdict`, `QuestionReview`, `Provenance` — it is imported verbatim
rather than re-declared.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.answer_eval.benchmark import (
    AnswerBenchmark,
    AnswerBenchmarkLifecycle,
    BenchmarkValidationError,
    HumanReviewStatus,
    RequiredEvidence,
)
from app.services.answer_eval.benchmark_review import (
    DimensionVerdict,
    QuestionReview,
    ReviewDimension,
)
from app.services.answer_eval.metrics import Measured
from app.services.answer_eval.provenance import (
    Provenance,
    ProvenanceKind,
    ProvenanceLevel,
    ProvenanceTag,
    human_review_provenance,
    validate_provenance,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GroundTruthError(RuntimeError):
    """A ground-truth workflow rule was violated."""


class QuestionState(str, Enum):
    """Per-question review state (V10 STEP 2).

    Deliberately separate from the benchmark-level lifecycle (draft / review /
    approved / frozen): a benchmark can be in REVIEW while individual questions
    are already approved, and a question can be AMBIGUOUS while the benchmark
    is DRAFT. Overloading one enum for both was explicitly forbidden by the
    V10 specification.

    Only APPROVED questions are scored by the reference-based evaluation
    policy. AMBIGUOUS and INSUFFICIENT_EVIDENCE are honest, explicitly
    classified non-scoring outcomes — never silent exclusions. REJECTED is not
    terminal: a rejected question must be re-authored (or explicitly
    classified as ambiguous / insufficient evidence) before the gate passes.
    """

    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GroundTruthAnnotation(BaseModel):
    """One append-only authored ground-truth record for one question.

    This is what a HUMAN writes after reading the actual corpus evidence:
    a reference answer, the key points a correct answer must state, and
    acceptable alternative elements. The evidence list records which chunks
    the reviewer actually read, so provenance validation can check the
    reference answer is grounded in the live corpus.

    An annotation is a DRAFT until a review approves it. It is never
    presented as ground truth on its own.
    """

    annotation_id: str = Field(description="Unique id. A duplicate id is refused, never merged.")
    kb_id: str
    benchmark_name: str = ""
    benchmark_fingerprint: str = Field(
        default="",
        description="Fingerprint of the benchmark content this annotation was written "
        "against. An annotation whose fingerprint no longer matches the benchmark is "
        "STALE and cannot be approved.",
    )
    question_id: str
    author: str = Field(description="Identity of the human who authored this. Required.")
    created_at: datetime = Field(default_factory=_utcnow)

    expected_answer: str = Field(
        default="",
        description="The reference answer. Written by the named human from the cited "
        "corpus evidence. NEVER generated by an LLM, never copied from a RAG answer.",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Facts a correct answer must state. Each becomes a scorable unit "
        "for key_point_recall.",
    )
    acceptable_answer_elements: list[str] = Field(
        default_factory=list,
        description="Alternative phrasings/contents a human judged acceptable. "
        "Recorded for audit and surfaced to reviewers; not a scoring input in "
        "policy v1.",
    )
    evidence: list[RequiredEvidence] = Field(
        default_factory=list,
        description="The corpus chunks the author actually read. Required: an "
        "annotation grounded in nothing is indistinguishable from a guess.",
    )
    provenance: list[Provenance] = Field(
        default_factory=list,
        description="How and from where this annotation was produced. Must contain at "
        "least one HUMAN_REVIEW-tagged block naming the author.",
    )
    supersedes: str = Field(
        default="", description="Annotation id this one replaces, when correcting."
    )
    note: str = ""

    @field_validator("author")
    @classmethod
    def _author_required(cls, v: str) -> str:
        value = (v or "").strip()
        if not value:
            raise GroundTruthError(
                "an annotation must name its human author; an unattributed "
                "reference answer cannot be distinguished from an agent draft"
            )
        return value

    def model_fingerprint(self) -> str:
        payload = {
            "kb_id": self.kb_id,
            "benchmark_name": self.benchmark_name,
            "question_id": self.question_id,
            "author": self.author,
            "expected_answer": self.expected_answer,
            "key_points": list(self.key_points),
            "acceptable_answer_elements": list(self.acceptable_answer_elements),
            "evidence": [e.chunk_id for e in self.evidence],
            "supersedes": self.supersedes,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]


def validate_annotation(annotation: GroundTruthAnnotation) -> list[str]:
    """Structural problems with an annotation. Empty means structurally valid.

    Checks only what the artifact itself can establish. Corpus consistency is
    ``validate_annotation_against_corpus``; approval semantics are the gate's
    job.
    """
    problems: list[str] = []
    if not annotation.expected_answer.strip() and not annotation.key_points:
        problems.append(
            "annotation states neither a reference answer nor key points; there "
            "is nothing a reviewer could approve"
        )
    for i, point in enumerate(annotation.key_points):
        if not point.strip():
            problems.append(f"key_points[{i}] is blank")
    for i, element in enumerate(annotation.acceptable_answer_elements):
        if not element.strip():
            problems.append(f"acceptable_answer_elements[{i}] is blank")
    if not annotation.evidence:
        problems.append(
            "annotation cites no corpus evidence; a reference answer must be "
            "written from the actual corpus, and the chunks it was read from "
            "must be recorded"
        )
    if not annotation.provenance:
        problems.append("annotation carries no provenance block")
    else:
        problems.extend(validate_provenance(annotation.provenance))
        if not any(b.is_human_reviewed() for b in annotation.provenance):
            problems.append(
                "annotation has no HUMAN_REVIEW-tagged provenance block; an "
                "annotation that does not record human authorship cannot be "
                "treated as human ground truth"
            )
    return problems


def validate_annotation_against_corpus(
    annotation: GroundTruthAnnotation,
    chunk_index: dict[str, dict[str, Any]],
) -> list[str]:
    """Provenance problems: is this annotation traceable to the REAL corpus?

    Checks each cited chunk: exists, belongs to the annotation's knowledge
    base, belongs to the recorded document, and still has the recorded content
    hash. A re-chunked corpus changes hashes, which would silently invalidate
    the reference answer's grounding — so it is a failure, not a warning.
    """
    problems: list[str] = []
    for ev in annotation.evidence:
        found = chunk_index.get(ev.chunk_id)
        if found is None:
            problems.append(
                f"cited chunk {ev.chunk_id} is not in the corpus of this "
                f"knowledge base"
            )
            continue
        if found.get("kb_id") and annotation.kb_id and found["kb_id"] != annotation.kb_id:
            problems.append(
                f"cited chunk {ev.chunk_id} belongs to knowledge base "
                f"{found['kb_id']}, not {annotation.kb_id}"
            )
        if ev.content_hash and found.get("content_hash") and (
            ev.content_hash != found["content_hash"]
        ):
            problems.append(
                f"cited chunk {ev.chunk_id} content hash differs (corpus "
                f"re-chunked since the annotation was written)"
            )
        if ev.document_id and found.get("document_id") and (
            ev.document_id != found["document_id"]
        ):
            problems.append(
                f"cited chunk {ev.chunk_id} is now in document "
                f"{found['document_id']}, annotation says {ev.document_id}"
            )
    return problems


def build_author_provenance(
    *,
    author: str,
    method: str,
    source: str = "",
    justification: str = "",
    data: dict[str, Any] | None = None,
) -> Provenance:
    """Provenance for a human-authored annotation, naming the author."""
    return human_review_provenance(
        kind=ProvenanceKind.REFERENCE_ANSWER,
        reviewer=author,
        method=method,
        source=source,
        justification=justification,
        level=ProvenanceLevel.ORIGINAL,
        data=data,
    )


def assert_provenance_tags_honest(annotation: GroundTruthAnnotation) -> None:
    """Refuse provenance that claims human review without naming a human."""
    for block in annotation.provenance:
        if ProvenanceTag.HUMAN_REVIEW in block.tags:
            reviewer = str(block.data.get("reviewer", "")).strip()
            if not reviewer:
                raise GroundTruthError(
                    "a HUMAN_REVIEW provenance block must name the reviewer"
                )


class GroundTruthLog:
    """Append-only collection of annotations (V10 STEP 6).

    The ONLY mutation is ``append``. A correction is a NEW annotation carrying
    ``supersedes``; the record that a label was once claimed differently
    survives. A duplicate id raises rather than merging.
    """

    def __init__(self, annotations: list[GroundTruthAnnotation] | None = None) -> None:
        self._annotations: dict[str, GroundTruthAnnotation] = {}
        self._order: list[str] = []
        for annotation in annotations or []:
            self.append(annotation)

    def append(self, annotation: GroundTruthAnnotation) -> GroundTruthAnnotation:
        if annotation.annotation_id in self._annotations:
            raise GroundTruthError(
                f"annotation {annotation.annotation_id!r} already exists; ground "
                f"truth is append-only. Record a correction as a new annotation "
                f"with `supersedes` set."
            )
        self._annotations[annotation.annotation_id] = annotation
        self._order.append(annotation.annotation_id)
        return annotation

    def all(self) -> list[GroundTruthAnnotation]:
        return [self._annotations[aid] for aid in self._order]

    def for_question(self, question_id: str) -> list[GroundTruthAnnotation]:
        return [a for a in self.all() if a.question_id == question_id]

    def effective_for_question(self, question_id: str) -> GroundTruthAnnotation | None:
        """Latest annotation for a question, ignoring superseded ones."""
        items = self.for_question(question_id)
        superseded = {a.supersedes for a in items if a.supersedes}
        active = [a for a in items if a.annotation_id not in superseded]
        return active[-1] if active else None

    def authors(self) -> list[str]:
        return sorted({a.author for a in self.all()})


#: State -> the value written to `AnswerBenchmarkQuestion.review_status`.
#: The benchmark's closed vocabulary gained these values additively in V10 so
#: a refused or ambiguous question is recorded as such rather than left to
#: look unreviewed.
STATE_TO_REVIEW_STATUS: dict[QuestionState, str] = {
    QuestionState.PENDING: "pending",
    QuestionState.IN_REVIEW: "in_review",
    QuestionState.APPROVED: "approved",
    QuestionState.REJECTED: "rejected",
    QuestionState.AMBIGUOUS: "ambiguous",
    QuestionState.INSUFFICIENT_EVIDENCE: "insufficient_evidence",
}

#: Review outcomes that map directly onto a question state.
_OUTCOME_STATES: dict[QuestionState, QuestionState] = {
    QuestionState.REJECTED: QuestionState.REJECTED,
    QuestionState.AMBIGUOUS: QuestionState.AMBIGUOUS,
    QuestionState.INSUFFICIENT_EVIDENCE: QuestionState.INSUFFICIENT_EVIDENCE,
    QuestionState.IN_REVIEW: QuestionState.IN_REVIEW,
}


def effective_review(reviews: list[QuestionReview]) -> QuestionReview | None:
    """Latest review, ignoring superseded ones."""
    if not reviews:
        return None
    superseded = {r.supersedes for r in reviews if r.supersedes}
    active = [r for r in reviews if r.review_id not in superseded]
    return active[-1] if active else None


def derive_question_state(
    *,
    annotation: GroundTruthAnnotation | None,
    annotation_problems: list[str] | None = None,
    reviews: list[QuestionReview] | None = None,
    benchmark_fingerprint: str = "",
) -> tuple[QuestionState, list[str]]:
    """The state a question's annotation + reviews justify, and WHY.

    The precedence is deliberate:

    1. A reviewer's explicit terminal outcome (rejected / ambiguous /
       insufficient evidence) wins — the human said so.
    2. Otherwise an approval needs a valid annotation AND an affirmative
       review on every required dimension. A review that claims approval
       without a valid annotation is downgraded to IN_REVIEW, never approved.
    3. Otherwise, an annotation with no review is IN_REVIEW (a draft awaiting
       a human verdict) and nothing at all is PENDING.
    """
    problems = list(annotation_problems or [])
    relevant = [
        r
        for r in (reviews or [])
        if not benchmark_fingerprint
        or not r.benchmark_fingerprint
        or r.benchmark_fingerprint == benchmark_fingerprint
    ]
    latest = effective_review(relevant)

    if latest is None:
        if annotation is None:
            return QuestionState.PENDING, [
                "no reference answer has been authored and no reviewer has "
                "recorded a verdict"
            ]
        return QuestionState.IN_REVIEW, [
            "a reference answer is drafted but no reviewer has recorded a verdict"
        ]

    outcome = getattr(latest, "outcome", None)
    outcome_state: QuestionState | None = None
    if outcome:
        try:
            outcome_state = QuestionState(outcome)
        except ValueError:
            outcome_state = None
    if outcome_state in _OUTCOME_STATES:
        return _OUTCOME_STATES[outcome_state], [
            f"reviewer {latest.reviewer!r} recorded outcome "
            f"{outcome_state.value}"
        ]

    # An affirmative-or-explicit-approval review: approval still requires a
    # valid annotation and no unassessed/negative required dimension.
    if annotation is None:
        return QuestionState.IN_REVIEW, [
            f"reviewer {latest.reviewer!r} recorded no terminal outcome and no "
            f"reference answer was authored"
        ]
    if problems:
        return QuestionState.IN_REVIEW, [
            "an approving review exists but the annotation is not valid: "
            + "; ".join(problems)
        ]
    blockers = latest.blocks_approval()
    if blockers:
        return QuestionState.IN_REVIEW, [
            f"reviewer {latest.reviewer!r} did not pass every required "
            f"dimension: " + "; ".join(blockers)
        ]
    return QuestionState.APPROVED, [
        f"reviewer {latest.reviewer!r} assessed every required dimension "
        f"affirmatively and the annotation validates against the corpus"
    ]


class QuestionStateApplication(BaseModel):
    """Outcome of applying the V10 states to a benchmark (in memory)."""

    states: dict[str, str] = Field(default_factory=dict)
    reasons: dict[str, list[str]] = Field(default_factory=dict)
    problems: dict[str, list[str]] = Field(default_factory=dict)
    authored_annotation_ids: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)

    def ids_in_state(self, state: QuestionState) -> list[str]:
        return sorted(
            qid for qid, value in self.states.items() if value == state.value
        )

    @property
    def approved_ids(self) -> list[str]:
        return self.ids_in_state(QuestionState.APPROVED)


def apply_question_states(
    benchmark: AnswerBenchmark,
    annotations: list[GroundTruthAnnotation],
    reviews: list[QuestionReview],
    chunk_index: dict[str, dict[str, Any]],
) -> QuestionStateApplication:
    """Set each question's ``review_status`` from its annotation + reviews.

    Mutates the benchmark model IN MEMORY (labels are copied onto APPROVED
    questions so the evaluator can score them) and returns the full state map.
    Nothing is persisted here; persisting is the caller's decision, and
    writing to a frozen artifact is refused by ``assert_not_frozen_artifact``.

    Labels are copied onto a question ONLY when it is APPROVED. An in-review
    draft must not silently become scorable ground truth.
    """
    fp = benchmark.fingerprint()
    by_question: dict[str, list[GroundTruthAnnotation]] = {}
    by_review: dict[str, list[QuestionReview]] = {}
    for a in annotations:
        by_question.setdefault(a.question_id, []).append(a)
    for r in reviews:
        by_review.setdefault(r.question_id, []).append(r)

    application = QuestionStateApplication()
    for question in benchmark.questions:
        q_annotations = by_question.get(question.question_id, [])
        log = GroundTruthLog(q_annotations)
        effective = log.effective_for_question(question.question_id)
        problems: list[str] = []
        if effective is not None:
            problems.extend(validate_annotation(effective))
            problems.extend(validate_annotation_against_corpus(effective, chunk_index))
            if (
                effective.benchmark_fingerprint
                and effective.benchmark_fingerprint != fp
            ):
                problems.append(
                    "annotation was authored against a different benchmark "
                    f"content (fingerprint {effective.benchmark_fingerprint} vs {fp}); "
                    "it is STALE and must be re-authored"
                )

        state, reasons = derive_question_state(
            annotation=effective,
            annotation_problems=problems,
            reviews=by_review.get(question.question_id, []),
            benchmark_fingerprint=fp,
        )
        question.review_status = STATE_TO_REVIEW_STATUS[state]
        application.states[question.question_id] = state.value
        application.reasons[question.question_id] = reasons
        if problems:
            application.problems[question.question_id] = problems
        if effective is not None:
            application.authored_annotation_ids[question.question_id] = (
                effective.annotation_id
            )
            if state is QuestionState.APPROVED:
                question.expected_answer = effective.expected_answer.strip() or None
                question.key_points = list(effective.key_points)
                question.acceptable_answer_elements = list(
                    effective.acceptable_answer_elements
                )

    for state in QuestionState:
        application.counts[state.value] = len(application.ids_in_state(state))
    return application


class ReviewCompleteness(BaseModel):
    """V10 STEP 3 — how complete the human review actually is.

    Counts are real counts (0 approved is a real 0). RATES are `Measured`:
    a coverage rate with no denominator is UNKNOWN, never 0.0.
    """

    benchmark_name: str = ""
    benchmark_fingerprint: str = ""
    total: int = 0
    pending: int = 0
    in_review: int = 0
    approved: int = 0
    rejected: int = 0
    ambiguous: int = 0
    insufficient_evidence: int = 0

    authored_reference_answer_count: int = 0
    authored_key_point_count: int = 0
    authored_acceptable_element_count: int = 0
    provenance_valid_count: int = 0
    provenance_invalid_count: int = 0
    review_count: int = 0
    reviewers: list[str] = Field(default_factory=list)

    key_point_coverage: Measured = Field(
        default_factory=lambda: Measured.unknown("not computed")
    )
    acceptable_element_coverage: Measured = Field(
        default_factory=lambda: Measured.unknown("not computed")
    )
    provenance_coverage: Measured = Field(
        default_factory=lambda: Measured.unknown("not computed")
    )
    reference_answer_coverage: Measured = Field(
        default_factory=lambda: Measured.unknown("not computed")
    )
    unknown_metrics: list[str] = Field(default_factory=list)

    @property
    def reviewed(self) -> int:
        """Questions with an actual reviewer verdict (not merely authored)."""
        return self.approved + self.rejected + self.ambiguous + self.insufficient_evidence


def compute_completeness(
    benchmark: AnswerBenchmark,
    annotations: list[GroundTruthAnnotation],
    reviews: list[QuestionReview],
    application: QuestionStateApplication,
) -> ReviewCompleteness:
    """Measure review completeness over the WHOLE benchmark (V10 STEP 3).

    Every question is counted in exactly one state bucket, so the buckets sum
    to the total and no question can disappear from the report.
    """
    total = len(benchmark.questions)
    completeness = ReviewCompleteness(
        benchmark_name=benchmark.benchmark,
        benchmark_fingerprint=benchmark.fingerprint(),
        total=total,
        pending=application.counts.get(QuestionState.PENDING.value, 0),
        in_review=application.counts.get(QuestionState.IN_REVIEW.value, 0),
        approved=application.counts.get(QuestionState.APPROVED.value, 0),
        rejected=application.counts.get(QuestionState.REJECTED.value, 0),
        ambiguous=application.counts.get(QuestionState.AMBIGUOUS.value, 0),
        insufficient_evidence=application.counts.get(
            QuestionState.INSUFFICIENT_EVIDENCE.value, 0
        ),
        review_count=len(reviews),
        reviewers=sorted({r.reviewer for r in reviews}),
    )

    log = GroundTruthLog(annotations)
    authored: list[GroundTruthAnnotation] = []
    for question in benchmark.questions:
        effective = log.effective_for_question(question.question_id)
        if effective is None:
            continue
        authored.append(effective)
        if effective.expected_answer.strip():
            completeness.authored_reference_answer_count += 1
        if effective.key_points:
            completeness.authored_key_point_count += 1
        if effective.acceptable_answer_elements:
            completeness.authored_acceptable_element_count += 1
        problems = application.problems.get(question.question_id, [])
        if problems:
            completeness.provenance_invalid_count += 1
        else:
            # `application.problems` also carries structural problems, so a
            # clean question is only provenance-valid when the annotation's
            # own structural validation passed too.
            structural = validate_annotation(effective)
            if structural:
                completeness.provenance_invalid_count += 1
            else:
                completeness.provenance_valid_count += 1

    def rate(count: int, denominator: int, what: str) -> Measured:
        if denominator == 0:
            return Measured.unknown(
                f"no question has an authored annotation, so {what} has no "
                f"denominator — this is unknown, not zero"
            )
        return Measured.of(
            count / denominator,
            reason=f"{count}/{denominator} authored question(s) with {what}",
            sample_size=denominator,
        )

    n_authored = len(authored)
    completeness.reference_answer_coverage = rate(
        completeness.authored_reference_answer_count, n_authored, "a reference answer"
    )
    completeness.key_point_coverage = rate(
        completeness.authored_key_point_count, n_authored, "key points"
    )
    completeness.acceptable_element_coverage = rate(
        completeness.authored_acceptable_element_count,
        n_authored,
        "acceptable answer elements",
    )
    completeness.provenance_coverage = rate(
        completeness.provenance_valid_count, n_authored, "corpus-valid provenance"
    )
    completeness.unknown_metrics = [
        name
        for name in (
            "reference_answer_coverage",
            "key_point_coverage",
            "acceptable_element_coverage",
            "provenance_coverage",
        )
        if not getattr(completeness, name).measured
    ]
    return completeness


class ApprovalPolicy(BaseModel):
    """The documented rules under which the benchmark may be APPROVED.

    Published in full on every gate result and inside every frozen version, so
    a benchmark's approval can always be re-explained from the record rather
    than from memory.
    """

    policy_version: str = "v10-answer-benchmark-approval-v1"
    scoring_states: list[str] = Field(
        default_factory=lambda: [QuestionState.APPROVED.value],
        description="Only questions in these states receive human ground truth "
        "and are scored by the reference-based policy.",
    )
    permitted_non_scoring_states: list[str] = Field(
        default_factory=lambda: [
            QuestionState.AMBIGUOUS.value,
            QuestionState.INSUFFICIENT_EVIDENCE.value,
        ],
        description="States a reviewer may leave unscored. They are counted, "
        "named, and excluded LOUDLY — never silently dropped.",
    )
    blocking_states: list[str] = Field(
        default_factory=lambda: [
            QuestionState.PENDING.value,
            QuestionState.IN_REVIEW.value,
            QuestionState.REJECTED.value,
        ],
        description="States that refuse approval: unresolved work (pending / "
        "in review) or a question a human refused (rejected).",
    )
    minimum_approved_questions: int = 1
    require_key_points: bool = Field(
        default=True,
        description="Every scored question must carry at least one human key point, "
        "because key_point_recall is the metric this workflow exists to unlock. "
        "A question without key points cannot be scored for completeness.",
    )
    require_reference_answer: bool = True
    require_provenance_valid: bool = True


class ApprovalGateResult(BaseModel):
    """Why the benchmark may (or may not) be approved, in full detail."""

    approved: bool
    policy: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    evaluated_at: datetime = Field(default_factory=_utcnow)
    counts: dict[str, int] = Field(default_factory=dict)
    approved_question_ids: list[str] = Field(default_factory=list)
    excluded_question_ids: list[str] = Field(default_factory=list)
    blocking: dict[str, list[str]] = Field(default_factory=dict)
    problems: dict[str, list[str]] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


def evaluate_approval_gate(
    benchmark: AnswerBenchmark,
    application: QuestionStateApplication,
    completeness: ReviewCompleteness,
    *,
    policy: ApprovalPolicy | None = None,
) -> ApprovalGateResult:
    """The V10 STEP 7 approval gate.

    Approval is NOT "every question was opened". It requires every question to
    be in a terminal state a human explicitly chose:

    * APPROVED  — scored; the annotation must be valid, provenance-valid, and
                  carry the labels the policy requires;
    * AMBIGUOUS / INSUFFICIENT_EVIDENCE — explicitly classified, counted and
                  excluded from scoring;
    * PENDING / IN_REVIEW / REJECTED — BLOCK approval. A question nobody
                  finished, or a human refused, cannot be papered over.

    The result always explains itself: `reasons` states the policy outcome,
    `blocking` names every blocking question, and `problems` carries the
    per-question reason.
    """
    policy = policy or ApprovalPolicy()
    result = ApprovalGateResult(
        approved=False,
        policy=policy,
        counts=dict(application.counts),
    )
    by_id = {q.question_id: q for q in benchmark.questions}

    for qid, state_value in sorted(application.states.items()):
        if state_value in policy.blocking_states:
            result.blocking.setdefault(state_value, []).append(qid)
        elif state_value in policy.permitted_non_scoring_states:
            result.excluded_question_ids.append(qid)
        elif state_value in policy.scoring_states:
            result.approved_question_ids.append(qid)

    # -- per-question requirements for SCORED questions ---------------------
    for qid in result.approved_question_ids:
        question = by_id.get(qid)
        if question is None:
            result.problems[qid] = ["question id is not in the benchmark"]
            continue
        problems = list(application.problems.get(qid, []))
        if policy.require_reference_answer and not (
            question.expected_answer and question.expected_answer.strip()
        ):
            problems.append(
                "no reference answer is attached to the approved question"
            )
        if policy.require_key_points and not question.key_points:
            problems.append(
                "no key points are attached to the approved question; "
                "key_point_recall could not be measured for it"
            )
        if problems:
            result.problems[qid] = problems

    blocking_total = sum(len(ids) for ids in result.blocking.values())
    scoring_with_problems = sorted(result.problems)

    # -- explain the outcome -------------------------------------------------
    result.reasons.append(
        f"policy {policy.policy_version}: scored states "
        f"{policy.scoring_states}; permitted non-scoring states "
        f"{policy.permitted_non_scoring_states}; blocking states "
        f"{policy.blocking_states}"
    )
    if blocking_total:
        for state, ids in sorted(result.blocking.items()):
            shown = ", ".join(ids[:5]) + (" …" if len(ids) > 5 else "")
            result.reasons.append(
                f"BLOCKED by {len(ids)} question(s) in state {state!r}: {shown}. "
                f"Unresolved or refused questions must be re-authored or "
                f"explicitly classified as ambiguous / insufficient evidence "
                f"before the benchmark can be approved."
            )
    if scoring_with_problems:
        shown = ", ".join(scoring_with_problems[:5]) + (
            " …" if len(scoring_with_problems) > 5 else ""
        )
        result.reasons.append(
            f"BLOCKED by {len(scoring_with_problems)} approved question(s) that do "
            f"not satisfy the policy: {shown}. See `problems` for the reason on "
            f"each."
        )
    if len(result.approved_question_ids) < policy.minimum_approved_questions:
        result.reasons.append(
            f"BLOCKED: {len(result.approved_question_ids)} approved question(s), "
            f"below the policy minimum of {policy.minimum_approved_questions}"
        )

    approved = (
        blocking_total == 0
        and not scoring_with_problems
        and len(result.approved_question_ids) >= policy.minimum_approved_questions
    )
    result.approved = approved
    if approved:
        result.reasons.append(
            f"APPROVED: {len(result.approved_question_ids)} question(s) carry "
            f"reviewed ground truth and will be scored; "
            f"{len(result.excluded_question_ids)} question(s) are explicitly "
            f"classified as non-scoring ({', '.join(result.excluded_question_ids[:5])}"
            + (" …" if len(result.excluded_question_ids) > 5 else "")
            + ") and are excluded from aggregate metrics BY POLICY, recorded in "
            "this result and in any frozen version."
        )
    return result


def ground_truth_fingerprint(
    benchmark: AnswerBenchmark, annotations: list[GroundTruthAnnotation]
) -> str:
    """Content hash of the LABELS only (reference answers + key points).

    `AnswerBenchmark.fingerprint()` deliberately excludes labels so review
    metadata does not change it (V8). That makes it the wrong hash for
    versioning frozen ground truth: two versions differing only in their
    human labels would collide. This fingerprint closes that gap.
    """
    log = GroundTruthLog(annotations)
    payload = []
    for q in benchmark.questions:
        effective = log.effective_for_question(q.question_id)
        payload.append(
            {
                "question_id": q.question_id,
                "review_status": q.review_status or "",
                # The labels ON THE BENCHMARK (what an evaluation would read)…
                "benchmark_expected_answer": q.expected_answer or "",
                "benchmark_key_points": list(q.key_points),
                "benchmark_acceptable_elements": list(q.acceptable_answer_elements),
                # …and the annotations they were promoted from. Hashing both
                # detects an edit to either side, and a divergence between them.
                "annotation_expected_answer": effective.expected_answer if effective else "",
                "annotation_key_points": effective.key_points if effective else [],
                "annotation_acceptable_elements": (
                    effective.acceptable_answer_elements if effective else []
                ),
            }
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


class AnswerBenchmarkVersion(BaseModel):
    """An IMMUTABLE frozen answer benchmark version (V10 STEP 8).

    Contains everything needed to re-run an evaluation and re-explain the
    approval: the frozen benchmark (with embedded labels), the annotations
    they came from, the reviews that approved them, and the gate result with
    the exact policy used.

    There is no update path. A correction is a NEW version; an existing one is
    only ever read. `verify_answer_benchmark_version` recomputes the recorded
    fingerprints and reports tampering.
    """

    version_id: str
    kb_id: str
    version: int = Field(ge=1, description="1-based sequence for this KB")
    benchmark_name: str
    benchmark_fingerprint: str = Field(
        description="AnswerBenchmark.fingerprint() of the embedded benchmark"
    )
    ground_truth_fingerprint: str = Field(
        description="Hash of the labels (reference answers + key points)"
    )
    artifact_fingerprint: str = Field(
        description="Hash of this whole record's content; recomputed by verify"
    )
    created_at: datetime = Field(default_factory=_utcnow)
    frozen_by: str = Field(description="Identity of the human who froze this version")
    gate: ApprovalGateResult
    benchmark: AnswerBenchmark
    annotations: list[GroundTruthAnnotation] = Field(default_factory=list)
    reviews: list[QuestionReview] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("frozen_by")
    @classmethod
    def _frozen_by_required(cls, v: str) -> str:
        value = (v or "").strip()
        if not value:
            raise GroundTruthError("a frozen version must name who froze it")
        return value

    def scoring_question_ids(self) -> list[str]:
        return sorted(
            q.question_id
            for q in self.benchmark.questions
            if (q.review_status or "") == QuestionState.APPROVED.value
        )

    def excluded_question_ids(self) -> list[str]:
        allowed = set(self.gate.policy.permitted_non_scoring_states)
        return sorted(
            q.question_id
            for q in self.benchmark.questions
            if (q.review_status or "") in allowed
        )


def _artifact_payload(version: AnswerBenchmarkVersion) -> dict[str, Any]:
    """The deterministic content of a frozen version (excluding its own hash)."""
    return {
        "version_id": version.version_id,
        "kb_id": version.kb_id,
        "version": version.version,
        "benchmark_name": version.benchmark_name,
        "benchmark_fingerprint": version.benchmark_fingerprint,
        "ground_truth_fingerprint": version.ground_truth_fingerprint,
        "created_at": version.created_at.isoformat(),
        "frozen_by": version.frozen_by,
        "gate": version.gate.model_dump(mode="json"),
        "benchmark": version.benchmark.model_dump(mode="json"),
        "annotations": [a.model_dump(mode="json") for a in version.annotations],
        "reviews": [r.model_dump(mode="json") for r in version.reviews],
        "notes": list(version.notes),
    }


def compute_artifact_fingerprint(version: AnswerBenchmarkVersion) -> str:
    payload = _artifact_payload(version)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def verify_answer_benchmark_version(version: AnswerBenchmarkVersion) -> list[str]:
    """Tamper check for a frozen version. Empty means intact.

    A frozen benchmark edited in place is worse than no frozen benchmark,
    because every result computed from it silently becomes unattributable.
    """
    problems: list[str] = []
    recomputed = compute_artifact_fingerprint(version)
    if recomputed != version.artifact_fingerprint:
        problems.append(
            f"artifact fingerprint mismatch: recorded "
            f"{version.artifact_fingerprint}, recomputed {recomputed} — this "
            f"frozen version has been modified since it was created"
        )
    if version.benchmark.lifecycle is not AnswerBenchmarkLifecycle.FROZEN:
        problems.append("the embedded benchmark is not marked FROZEN")
    if not version.gate.approved:
        problems.append("the embedded gate result does not record approval")
    recorded = ground_truth_fingerprint(version.benchmark, version.annotations)
    if recorded != version.ground_truth_fingerprint:
        problems.append(
            "ground-truth fingerprint mismatch: the embedded labels do not "
            "match the recorded label hash"
        )
    return problems


def build_answer_benchmark_version(
    *,
    benchmark: AnswerBenchmark,
    annotations: list[GroundTruthAnnotation],
    reviews: list[QuestionReview],
    application: QuestionStateApplication,
    gate: ApprovalGateResult,
    frozen_by: str,
    version: int,
    version_id: str,
    notes: list[str] | None = None,
) -> AnswerBenchmarkVersion:
    """Freeze an APPROVED benchmark into an immutable version record.

    Raises when the gate has not passed: freezing an unapproved benchmark is
    exactly how a draft becomes an official-looking artifact. The embedded
    benchmark is a deep copy marked FROZEN, so later edits to the working
    model cannot reach into the frozen record.
    """
    if not gate.approved:
        raise BenchmarkValidationError(
            "refusing to freeze: the approval gate has not passed. Reasons: "
            + " | ".join(gate.reasons)
        )
    embedded = benchmark.model_copy(deep=True)
    embedded.lifecycle = AnswerBenchmarkLifecycle.FROZEN
    embedded.human_review = HumanReviewStatus.HUMAN_REVIEWED
    embedded.evaluator_contract = {
        **(embedded.evaluator_contract or {}),
        "frozen_from": benchmark.benchmark,
        "approval_policy": gate.policy.policy_version,
    }

    record = AnswerBenchmarkVersion(
        version_id=version_id,
        kb_id=benchmark.kb_id,
        version=version,
        benchmark_name=benchmark.benchmark,
        benchmark_fingerprint=embedded.fingerprint(),
        ground_truth_fingerprint=ground_truth_fingerprint(embedded, annotations),
        artifact_fingerprint="",
        frozen_by=frozen_by,
        gate=gate,
        benchmark=embedded,
        annotations=list(annotations),
        reviews=list(reviews),
        notes=list(notes or []),
    )
    record.artifact_fingerprint = compute_artifact_fingerprint(record)
    return record


__all__ = [
    "ApprovalGateResult",
    "ApprovalPolicy",
    "AnswerBenchmarkVersion",
    "GroundTruthAnnotation",
    "GroundTruthError",
    "GroundTruthLog",
    "QuestionState",
    "QuestionStateApplication",
    "ReviewCompleteness",
    "STATE_TO_REVIEW_STATUS",
    "apply_question_states",
    "assert_provenance_tags_honest",
    "build_answer_benchmark_version",
    "build_author_provenance",
    "compute_artifact_fingerprint",
    "compute_completeness",
    "derive_question_state",
    "effective_review",
    "evaluate_approval_gate",
    "ground_truth_fingerprint",
    "validate_annotation",
    "validate_annotation_against_corpus",
    "verify_answer_benchmark_version",
]
