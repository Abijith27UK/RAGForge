"""Answer-benchmark schemas and loader.

DESIGN DECISION — why there is no "expected answer text" field.

The frozen retrieval benchmark (`benchmarks/automobile-engineering-baseline-v1.json`)
contains 28 human-reviewed questions with `expected_chunk_ids`, but **no expected
answers, no key points, no answerability labels and no citation requirements**.
Its own `authorship.human_review` field reads `PENDING`. So the only ground truth
that genuinely exists today is *which chunk answers the question*.

This module therefore builds an answer benchmark whose ground truth is
**evidence-based, not answer-based**:

* `required_evidence` — the chunk(s) that contain the answer, inherited from the
  frozen benchmark and re-validated against the live corpus at load time.
* `answerability` — derived ONLY from whether the reviewer selected an answering
  chunk. A question with a selected chunk is answerable; unanswerable cases must
  be authored explicitly by a human and are flagged as such.
* `citation_requirements` — mechanically derivable from `required_evidence`
  (the answer as a whole must cite the chunks marked necessary; extra retrieved
  context is reported, not failed).

Metrics that REQUIRE a human-written answer (correctness against a reference
answer, key-point recall beyond the evidence) are therefore reported as
`UNKNOWN` with a reason, never guessed. See `docs/answer-benchmark-design.md`.

Everything here is validated on load: a benchmark that references a chunk that no
longer exists, or that claims human review that did not happen, is rejected.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Answerability(str, Enum):
    """Whether the corpus is expected to contain the answer.

    `answerable`   — the reviewer selected the chunk(s) that answer the question.
    `unanswerable` — a human asserted the corpus does NOT contain the answer, and
                     the correct behaviour is to abstain.
    `unknown`      — no human determination is recorded. Treated as
                     `answerable` for aggregation but reported distinctly; never
                     silently folded into either class.
    """

    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"
    UNKNOWN = "unknown"


class HumanReviewStatus(str, Enum):
    """Honest provenance of the labels in this artifact.

    `PENDING` is the normal state for agent-authored selections and must not be
    described as "human-reviewed" anywhere.
    """

    HUMAN_REVIEWED = "human_reviewed"
    PENDING = "pending"


class AnswerBenchmarkLifecycle(str, Enum):
    """Where an answer benchmark sits in its authoring lifecycle (V8 STEP 3).

    draft    — being authored; may contain agent-derived evidence labels only.
               A NEW DOMAIN ALWAYS STARTS HERE: no new domain ever receives
               answer ground truth automatically.
    approved — a human reviewed the labels that exist and accepted them.
    frozen   — content sealed. ONLY frozen answer benchmarks may be used for
               OFFICIAL answer-quality experiments (see
               `require_official_benchmark`).

    This is deliberately separate from `HumanReviewStatus`, which records
    whether human review happened at all: lifecycle is the gate on official
    use, provenance is the claim about authorship. Mixing them would let a
    provenance field silently become a permission field.
    """

    DRAFT = "draft"
    APPROVED = "approved"
    FROZEN = "frozen"


class ExpectedGroundingState(str, Enum):
    """The grounding outcome a correct system should reach.

    This is a *behavioural expectation* derived from answerability, NOT a
    prediction of what the current system will do. `ANY_ACCEPTABLE` is used when
    several states are legitimately correct for the question.
    """

    ANSWERED = "ANSWERED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
    ANY_ACCEPTABLE = "ANY_ACCEPTABLE"


class RequiredEvidence(BaseModel):
    """One chunk that the human determined answers the question.

    SEMANTICS — `required` is a NECESSITY label, not an exhaustiveness label.
    It means "a correct answer must use this chunk". It does NOT mean "this is
    the only chunk the answer is allowed to cite". A human picking one canonical
    passage out of a Wikipedia article is not asserting that the surrounding
    retrieved context is irrelevant.

    This distinction is load-bearing for two metrics:

    * citation_recall / completeness — legitimate and strict, because required
      chunks really are necessary.
    * citation_precision — a LOWER BOUND, because any additional retrieved
      context the answerer cites counts against it even when that context is
      genuinely on-topic.

    Getting this backwards is how a benchmark starts punishing answers for being
    well-sourced, so it is stated here rather than left implicit.
    """

    chunk_id: str
    document_id: str | None = None
    content_hash: str = Field(
        default="",
        description="Recorded so a re-chunked corpus is detected as a mismatch rather than silently scored",
    )
    section: str | None = None
    source_title: str | None = None
    required: bool = Field(
        default=True,
        description="True = necessary evidence a correct answer must cite. "
        "False = acceptable supporting context, not necessary.",
    )


class CitationRequirements(BaseModel):
    """What a correct citation set must look like for this question.

    Both requirements are evaluated at ANSWER level, not per claim — see the
    semantics note on `RequiredEvidence`.
    """

    require_at_least_one_citation: bool = True
    must_cite_all_required_evidence: bool = Field(
        default=True,
        description="Enforced per answer: every chunk marked required must appear in the "
        "answer's citation set. Strict and legitimate, because these chunks are "
        "necessary to answer the question.",
    )
    must_not_cite_irrelevant_chunks: bool = Field(
        default=True,
        description="Reported as an observation, never as a failure. The benchmark does "
        "not establish that an unlabelled chunk is irrelevant; it only records which "
        "evidence is necessary. Its effect is already captured by citation_precision.",
    )
    note: str = ""


class AnswerBenchmarkQuestion(BaseModel):
    """One question in an answer-quality benchmark."""

    question_id: str
    question: str
    subdomain: str = ""
    answerability: Answerability = Answerability.UNKNOWN
    required_evidence: list[RequiredEvidence] = Field(default_factory=list)
    citation_requirements: CitationRequirements = Field(default_factory=CitationRequirements)

    # -- NOT populated from the frozen benchmark (see module docstring) --------
    expected_answer: str | None = Field(
        default=None,
        description=(
            "Reference answer. NULL unless a human wrote one. Never generated by an LLM "
            "and never presented as ground truth without human authorship."
        ),
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Human-authored required factual points. Empty when no human has authored them.",
    )
    acceptable_answer_elements: list[str] = Field(
        default_factory=list,
        description="Alternative phrasings a human judged acceptable. Empty when not authored.",
    )
    numerical_tolerance: float | None = Field(
        default=None, description="Absolute tolerance for numeric checks, set by a human for that question"
    )
    requires_units: bool = False

    expected_grounding_state: ExpectedGroundingState = ExpectedGroundingState.ANY_ACCEPTABLE
    abstention_required: bool = False

    provenance: str = ""
    reviewer_metadata: dict[str, Any] = Field(default_factory=dict)

    # -- optional human review metadata (V8 STEP 3) ---------------------------
    # All optional and defaulting to None/"": an artifact written before these
    # fields existed loads unchanged, and their ABSENCE is recorded as absence
    # — never backfilled with an assumed value.
    difficulty: str | None = Field(
        default=None,
        description="Human-assigned difficulty: easy | medium | hard. None = not assigned.",
    )
    reviewer: str | None = Field(
        default=None,
        description="Identity of the human who reviewed this question's labels. "
        "None = no human reviewer recorded (do not describe as human-reviewed).",
    )
    review_status: str | None = Field(
        default=None,
        description="draft | reviewed | approved. None = not recorded.",
    )

    @field_validator("difficulty", "review_status")
    @classmethod
    def _closed_vocabulary(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        value = (v or "").strip().lower()
        if not value:
            return None
        allowed = (
            {"easy", "medium", "hard"}
            if info.field_name == "difficulty"
            else {"draft", "reviewed", "approved"}
        )
        if value not in allowed:
            raise ValueError(
                f"{info.field_name} must be one of {sorted(allowed)} (or null); "
                f"got {v!r}"
            )
        return value

    @model_validator(mode="after")
    def _consistency(self) -> "AnswerBenchmarkQuestion":
        if self.answerability is Answerability.UNANSWERABLE and self.required_evidence:
            raise ValueError(
                "an unanswerable question cannot require evidence: if evidence is "
                "required the question is answerable"
            )
        if self.answerability is Answerability.UNANSWERABLE and not self.abstention_required:
            raise ValueError(
                "an unanswerable question must set abstention_required=True"
            )
        if self.answerability is Answerability.ANSWERABLE and not self.required_evidence:
            raise ValueError(
                "an answerable question must name the required evidence chunk(s)"
            )
        return self


class AnswerBenchmark(BaseModel):
    """A versioned, immutable answer-quality benchmark artifact."""

    benchmark: str
    version: int = 1
    created_at: str = ""
    kb_id: str
    kb_name: str = ""
    derived_from: str = Field(
        default="",
        description="The retrieval benchmark this was derived from, when applicable",
    )
    authorship: dict[str, Any] = Field(default_factory=dict)
    human_review: HumanReviewStatus = HumanReviewStatus.PENDING
    lifecycle: AnswerBenchmarkLifecycle = Field(
        default=AnswerBenchmarkLifecycle.DRAFT,
        description="Authoring lifecycle. Defaults to DRAFT: an artifact that "
        "does not say it is frozen is not frozen. Only FROZEN answer "
        "benchmarks may be used for official experiments.",
    )
    evaluator_contract: dict[str, Any] = Field(
        default_factory=dict,
        description="Which metrics are computable from this artifact's ground truth",
    )
    questions: list[AnswerBenchmarkQuestion] = Field(default_factory=list)

    @field_validator("benchmark")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError("benchmark name must not be empty")
        return v

    # -- summary helpers ----------------------------------------------------

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def answerable_count(self) -> int:
        return sum(1 for q in self.questions if q.answerability is Answerability.ANSWERABLE)

    @property
    def unanswerable_count(self) -> int:
        return sum(1 for q in self.questions if q.answerability is Answerability.UNANSWERABLE)

    @property
    def unknown_count(self) -> int:
        return sum(1 for q in self.questions if q.answerability is Answerability.UNKNOWN)

    def question_ids(self) -> list[str]:
        return [q.question_id for q in self.questions]

    def fingerprint(self) -> str:
        """Stable content hash, recorded on every run so results are attributable."""
        import hashlib

        payload = json.dumps(
            {
                "benchmark": self.benchmark,
                "version": self.version,
                "kb_id": self.kb_id,
                "questions": [
                    {
                        "id": q.question_id,
                        "question": q.question,
                        "answerability": q.answerability.value,
                        "evidence": sorted(e.chunk_id for e in q.required_evidence),
                    }
                    for q in self.questions
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class BenchmarkValidationError(ValueError):
    """The benchmark is not trustworthy as written."""


def require_official_benchmark(benchmark: AnswerBenchmark) -> None:
    """Gate for OFFICIAL answer-quality experiments (V8 STEP 3).

    Raises `BenchmarkValidationError` unless the benchmark lifecycle is FROZEN.
    Development runs (`official=false`) are unrestricted and simply recorded as
    non-official, so a draft benchmark can be exercised without its numbers
    ever being mistakable for an official result.
    """
    if benchmark.lifecycle is not AnswerBenchmarkLifecycle.FROZEN:
        raise BenchmarkValidationError(
            f"official answer-quality experiments require a FROZEN benchmark; "
            f"{benchmark.benchmark!r} is {benchmark.lifecycle.value!r}. Set "
            f"official=false for development runs, or advance the benchmark "
            f"through review (draft -> approved -> frozen) first."
        )


def load_answer_benchmark(path: str | Path) -> AnswerBenchmark:
    """Load and structurally validate a benchmark file."""
    p = Path(path)
    if not p.exists():
        raise BenchmarkValidationError(f"benchmark file not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkValidationError(f"benchmark file is not valid JSON: {exc}") from exc
    try:
        return AnswerBenchmark.model_validate(raw)
    except Exception as exc:
        raise BenchmarkValidationError(f"benchmark failed validation: {exc}") from exc


def validate_against_corpus(
    benchmark: AnswerBenchmark,
    chunk_index: dict[str, dict[str, Any]],
) -> list[str]:
    """Verify every required chunk exists and its content hash still matches.

    Returns a list of human-readable problems; empty means the benchmark is
    consistent with the live corpus. A re-chunked corpus changes content hashes,
    which would silently invalidate citation scoring — so this is mandatory
    before any run.
    """
    problems: list[str] = []
    for q in benchmark.questions:
        if not q.required_evidence:
            continue
        for ev in q.required_evidence:
            found = chunk_index.get(ev.chunk_id)
            if found is None:
                problems.append(
                    f"{q.question_id}: required chunk {ev.chunk_id} is not in the corpus"
                )
                continue
            if ev.content_hash and found.get("content_hash") and (
                ev.content_hash != found["content_hash"]
            ):
                problems.append(
                    f"{q.question_id}: required chunk {ev.chunk_id} content hash differs "
                    f"(corpus re-chunked since the benchmark was written) — citation "
                    f"scoring would be invalid"
                )
            if ev.document_id and found.get("document_id") and ev.document_id != found["document_id"]:
                problems.append(
                    f"{q.question_id}: required chunk {ev.chunk_id} is now in document "
                    f"{found['document_id']}, benchmark says {ev.document_id}"
                )
    return problems


__all__ = [
    "AnswerBenchmark",
    "AnswerBenchmarkLifecycle",
    "AnswerBenchmarkQuestion",
    "Answerability",
    "BenchmarkValidationError",
    "CitationRequirements",
    "ExpectedGroundingState",
    "HumanReviewStatus",
    "RequiredEvidence",
    "load_answer_benchmark",
    "require_official_benchmark",
    "validate_against_corpus",
]