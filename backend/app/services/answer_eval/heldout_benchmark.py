"""Held-out retrieval benchmark — authoring, review and the freeze gate (V14).

## Why this module exists

V13 produced four exploratory verdicts on the 28-question development benchmark.
Every one of them is open to the same objection: the SAME 28 questions were used
to choose the candidate configurations, so a configuration that scores well is
indistinguishable from a configuration fitted to those questions. Nothing in the
repository could answer "do these findings generalize?".

Answering it needs a second instrument, and a second instrument is only worth
having if it is genuinely separate. Three things make it separate, and all three
are MECHANICAL here rather than asserted in prose:

1. **Separate artifact, separate loader.** The held-out set is its own file with
   its own identity, version and fingerprint. ``load_heldout_benchmark`` reads
   it; the development benchmark is never read or written by anything in this
   module.
2. **Separation is measured, not promised.** ``heldout_validation`` computes the
   overlap between the two sets — shared expected chunks, nearest development
   question, paraphrase similarity — and reports it per candidate. A held-out
   question that reuses a development chunk is an ERROR, not a warning.
3. **The label set is sealed with its review state.** A frozen held-out artifact
   records WHICH questions a human actually checked and which ones were only
   machine-drafted, and the claim strength available is derived from that record.
   That is the whole point of the two-tier lifecycle below.

## The two-tier lifecycle, and why one tier is not enough

The answer benchmark (``benchmark.py``) uses draft -> approved -> frozen and
refuses to run anything official below FROZEN. That is the right shape here too,
but it collapses two different questions that a held-out set must keep apart:

* Has the CONTENT stopped changing, so a result computed from it stays
  reproducible?  -> ``lifecycle``: DRAFT / PROVISIONAL / VALIDATED.
* May a HUMAN-REVIEWED result be claimed from it?  -> ``claim_strength``:
  EXPLORATORY / CONFIRMATORY.

A set can be sealed (reproducible) while its labels are agent-authored and
unreviewed (not claimable). Refusing to run at all in that state would be
needlessly destructive — the measurements are still informative — while running
it and calling the result "validation" would be fabrication. So the artifact
carries the tier, every experiment records the tier it ran under, and
``require_heldout_benchmark`` refuses to let a PROVISIONAL set support a
confirmatory claim.

THE TIER IS SEALED AT FREEZE TIME. ``freeze`` computes it from the review records
that exist AT THAT MOMENT and writes it into the file. There is deliberately no
"upgrade the tier" entry point: raising the tier after seeing held-out results
would let a reviewer who knows the answer bless the labels that produced it,
which is exactly the taint the tier exists to prevent. A re-review produces a new
version and therefore a new instrument.

## Honesty rules baked into the models

* ``authorship.agent_authored`` and ``human_review`` are separate fields. A set
  drafted by an agent and reviewed by a human is both, and must say both.
* A rejected candidate is KEPT in the candidate set with its decision. Deleting
  it would make the rejection rate unreportable, and a high rejection rate is a
  real finding about the drafting method.
* ``protocol`` is mandatory: an artifact that cannot say how its questions were
  chosen cannot support an independence claim.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

#: The V13 development benchmark. Read by the VALIDATOR (to measure overlap) and
#: never written by anything. Named here so the held-out artifact can record what
#: it was held out FROM, which is what makes "independent" checkable.
DEFAULT_DEV_BENCHMARK_PATH = "benchmarks/automobile-engineering-baseline-v1.json"

#: The mutable authoring artifact: candidates plus their review records.
DEFAULT_HELDOUT_CANDIDATE_PATH = "benchmarks/automobile-engineering-heldout-v1-candidates.json"

#: The frozen instrument. Written once; changing it requires a new version.
DEFAULT_HELDOUT_BENCHMARK_PATH = "benchmarks/automobile-engineering-heldout-v1.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HeldOutValidationError(ValueError):
    """The held-out artifact cannot be trusted, or the requested action is unsafe."""


class HeldOutLifecycle(str, Enum):
    """How far the held-out set has advanced through authoring and review.

    DRAFT       — being authored or reviewed. NOT runnable: a set whose content
                  can still change cannot produce a reproducible number.
    PROVISIONAL — sealed and reproducible, but at least one surviving question
                  has no human decision recorded. Exploratory runs only.
    VALIDATED   — sealed, and every surviving question carries a human decision.
                  Confirmatory claims are permitted.
    """

    DRAFT = "draft"
    PROVISIONAL = "provisional"
    VALIDATED = "validated"


class ClaimStrength(str, Enum):
    """What a result measured on this instrument is allowed to be called."""

    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


class CandidateStatus(str, Enum):
    """Where one candidate question stands. `draft` is the default and is never
    treated as reviewed."""

    DRAFT = "draft"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"

    @property
    def survives(self) -> bool:
        """Only non-rejected candidates enter the frozen instrument."""
        return self is not CandidateStatus.REJECTED

    @property
    def is_reviewed(self) -> bool:
        """A human decided on this question. `draft` means nobody did."""
        return self is not CandidateStatus.DRAFT


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class QueryType(str, Enum):
    """The kind of retrieval behaviour a question exercises.

    Recorded so the held-out set's composition is inspectable rather than
    implicit, and so a difference between the two sets can be seen instead of
    blamed on the corpus.
    """

    DEFINITION = "definition"
    MECHANISM = "mechanism"
    COMPARISON = "comparison"
    ENUMERATION = "enumeration"
    TERMINOLOGY = "terminology"
    QUANTITATIVE = "quantitative"


class Difficulty(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class HeldOutReviewRecord(BaseModel):
    """One append-only human decision about one candidate.

    The record preserves WHAT the reviewer changed, not merely that they looked:
    a decision plus the pre-edit values is what makes "edited" auditable later.
    """

    decision: ReviewDecision
    reviewer: str
    reviewed_at: str = Field(default_factory=now_iso)
    notes: str = ""
    #: The values as they stood BEFORE an edit, so the change is reconstructible.
    previous_question: str = ""
    previous_chunk_ids: list[str] = Field(default_factory=list)
    previous_subdomain: str = ""
    #: True when the reviewer acted on results they had already seen. This is not
    #: forbidden, but it must be visible: a label chosen with knowledge of the
    #: outcome is weaker evidence than one chosen blind.
    results_seen: bool = False

    @field_validator("reviewer")
    @classmethod
    def _named_reviewer(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError(
                "a review decision must name the reviewer; an anonymous decision "
                "cannot be audited and is not treated as human review"
            )
        return v.strip()


class HeldOutReviewRequest(BaseModel):
    """API/CLI input for one decision. `question_id` targets one candidate."""

    question_id: str
    decision: ReviewDecision
    reviewer: str
    notes: str = ""
    #: Replacement values. Required for EDIT, ignored otherwise.
    question: str | None = None
    expected_chunk_ids: list[str] | None = None
    subdomain: str | None = None
    results_seen: bool = False

    @model_validator(mode="after")
    def _edit_must_change_something(self) -> "HeldOutReviewRequest":
        if self.decision is ReviewDecision.EDIT and not any(
            [
                (self.question or "").strip(),
                self.expected_chunk_ids,
                (self.subdomain or "").strip(),
            ]
        ):
            raise ValueError(
                "an EDIT decision must supply at least one replacement value "
                "(question, expected_chunk_ids or subdomain); otherwise there is "
                "nothing that distinguishes it from an approval"
            )
        return self

    @field_validator("reviewer")
    @classmethod
    def _named_reviewer(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError("a review decision must name the reviewer")
        return v.strip()


class HeldOutCandidate(BaseModel):
    """One draft held-out question and the evidence that answers it.

    The evidence label is a NECESSITY label, exactly as in the development
    benchmark: "a correct answer must use this chunk". It is not a claim that the
    chunk is the only relevant one, which is why precision@k is a lower bound
    there and here alike.
    """

    id: str
    subdomain: str = ""
    question: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    #: The verbatim sentence(s) from the evidence chunk that answer the question.
    #: The validator checks these appear in the chunk text, so "the evidence is
    #: sufficient" is a verified claim rather than the author's opinion.
    evidence_quote: str = ""
    #: WHY this question belongs in a held-out set rather than the development one.
    coverage_rationale: str = ""
    query_type: QueryType = QueryType.DEFINITION
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    #: Human-readable provenance: which document/section the evidence came from.
    provenance: str = ""
    #: How the draft was produced. Recorded per question so the drafting method is
    #: auditable at question granularity, not only once for the whole file.
    draft_origin: str = ""
    status: CandidateStatus = CandidateStatus.DRAFT
    review: HeldOutReviewRecord | None = None

    @model_validator(mode="after")
    def _rejected_needs_a_review(self) -> "HeldOutCandidate":
        if self.status is CandidateStatus.REJECTED and self.review is None:
            raise ValueError(
                f"candidate {self.id!r} is rejected but carries no review record; a "
                "rejection without a recorded human decision is a silent deletion"
            )
        if self.status.is_reviewed and self.review is None:
            raise ValueError(
                f"candidate {self.id!r} is {self.status.value!r} but carries no review "
                "record; a reviewed status must be backed by the decision that set it"
            )
        return self


class HeldOutCandidateSet(BaseModel):
    """The mutable authoring artifact: candidates + review records + protocol."""

    benchmark: str
    version: int = 1
    created_at: str = Field(default_factory=now_iso)
    kb_id: str = ""
    kb_name: str = ""
    #: The benchmark this set is held out FROM. Naming it is what makes
    #: "independent of the development set" a checkable statement.
    derived_from: str = DEFAULT_DEV_BENCHMARK_PATH
    #: The documented creation protocol. Mandatory; see `_protocol_is_mandatory`.
    protocol: dict[str, Any] = Field(default_factory=dict)
    authorship: dict[str, Any] = Field(default_factory=dict)
    #: The retrieval benchmark whose SCHEMA the frozen artifact follows, so the
    #: existing diagnostics engine can score it without a single change.
    relevance_model: str = "binary"
    candidates: list[HeldOutCandidate] = Field(default_factory=list)

    @field_validator("protocol")
    @classmethod
    def _protocol_is_mandatory(cls, v: dict[str, Any]) -> dict[str, Any]:
        missing = [
            key
            for key in ("method", "question_selection", "evidence_selection", "review_policy")
            if not str(v.get(key, "")).strip()
        ]
        if missing:
            raise ValueError(
                "the creation protocol must document "
                + ", ".join(missing)
                + "; an artifact that cannot say how its questions were chosen "
                "cannot support an independence claim"
            )
        return v

    # -- summaries ----------------------------------------------------------

    @property
    def surviving(self) -> list[HeldOutCandidate]:
        return [c for c in self.candidates if c.status.survives]

    @property
    def rejected(self) -> list[HeldOutCandidate]:
        return [c for c in self.candidates if not c.status.survives]

    def review_completeness(self) -> dict[str, Any]:
        """Counts that make the review state reportable without interpretation."""
        total = len(self.candidates)
        surviving = self.surviving
        reviewed = [c for c in surviving if c.status.is_reviewed]
        unreviewed = [c.id for c in surviving if not c.status.is_reviewed]
        return {
            "candidates_total": total,
            "candidates_surviving": len(surviving),
            "candidates_rejected": total - len(surviving),
            "candidates_reviewed": len(reviewed),
            "candidates_unreviewed": len(unreviewed),
            "unreviewed_question_ids": unreviewed,
            "reviewers": sorted({c.review.reviewer for c in self.candidates if c.review}),
            "complete": bool(surviving) and not unreviewed,
        }

    def subdomain_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.surviving:
            counts[c.subdomain or "(unspecified)"] = counts.get(c.subdomain or "(unspecified)", 0) + 1
        return dict(sorted(counts.items()))

    def query_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.surviving:
            counts[c.query_type.value] = counts.get(c.query_type.value, 0) + 1
        return dict(sorted(counts.items()))

    def difficulty_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.surviving:
            counts[c.difficulty.value] = counts.get(c.difficulty.value, 0) + 1
        return dict(sorted(counts.items()))

    def fingerprint(self) -> str:
        """Stable content hash of what the instrument MEASURES.

        Only surviving questions count, so:

        * recording a review decision that does not change any label (an
          approval) does NOT move the fingerprint — the instrument is unchanged;
        * editing a question or its evidence DOES move it;
        * rejecting a question DOES move it, and correctly so: the question is
          removed from the instrument, so the instrument that measures 25
          questions is a different instrument from the one that measured 26.

        Whitespace-only differences do not move it. A changed question, id,
        expected chunk or expected document does.
        """
        payload = json.dumps(
            {
                "benchmark": self.benchmark,
                "version": self.version,
                "kb_id": self.kb_id,
                "questions": [
                    {
                        "id": c.id,
                        "question": c.question,
                        "expected_chunk_ids": sorted(c.expected_chunk_ids),
                        "expected_document_ids": sorted(c.expected_document_ids),
                    }
                    for c in self.surviving
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class HeldOutBenchmark(BaseModel):
    """The FROZEN held-out instrument.

    Field-for-field compatible with the retrieval-benchmark schema that
    ``load_retrieval_benchmark`` validates (the extra fields are ignored by that
    loader), so the V12/V13 measurement engine scores it unchanged. The extra
    fields are the ones that make the artifact's provenance and claim strength
    machine-readable.
    """

    benchmark: str
    version: int = 1
    created_at: str = Field(default_factory=now_iso)
    kb_id: str = ""
    kb_name: str = ""
    derived_from: str = DEFAULT_DEV_BENCHMARK_PATH
    authorship: dict[str, Any] = Field(default_factory=dict)
    relevance_model: str = "binary"
    questions: list[dict[str, Any]] = Field(default_factory=list)

    # -- held-out specific ---------------------------------------------------
    lifecycle: HeldOutLifecycle = HeldOutLifecycle.DRAFT
    claim_strength: ClaimStrength = ClaimStrength.EXPLORATORY
    frozen_at: str = ""
    frozen_by: str = ""
    #: The candidate-set fingerprint this instrument was frozen FROM, so the
    #: artifact can be traced back to the reviewable drafts.
    source_fingerprint: str = ""
    review: dict[str, Any] = Field(default_factory=dict)
    protocol: dict[str, Any] = Field(default_factory=dict)
    composition: dict[str, Any] = Field(default_factory=dict)
    rejected_question_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def question_count(self) -> int:
        return len(self.questions)

    def question_ids(self) -> list[str]:
        return [q.get("id", "") for q in self.questions]

    def fingerprint(self) -> str:
        """Identical in ALGORITHM to the retrieval benchmark's, so the same
        numbers appear whether the artifact is read by the retrieval loader or by
        this one. A fingerprint that changed shape between loaders would make two
        artifacts of the same instrument look different."""
        payload = json.dumps(
            {
                "benchmark": self.benchmark,
                "version": self.version,
                "kb_id": self.kb_id,
                "questions": [
                    {
                        "id": q.get("id", ""),
                        "question": q.get("question", ""),
                        "expected_chunk_ids": sorted(q.get("expected_chunk_ids") or []),
                        "expected_document_ids": sorted(q.get("expected_document_ids") or []),
                    }
                    for q in self.questions
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------
# lifecycle rules
# ----------------------------------------------------------------------------


def claim_strength_for(lifecycle: HeldOutLifecycle) -> ClaimStrength:
    """The ONLY mapping from freeze tier to allowed claim.

    Kept as a function so no caller can invent a stronger claim than its tier.
    """
    return (
        ClaimStrength.CONFIRMATORY
        if lifecycle is HeldOutLifecycle.VALIDATED
        else ClaimStrength.EXPLORATORY
    )


def require_runnable(benchmark: HeldOutBenchmark) -> None:
    """Refuse to run anything against a DRAFT instrument.

    DRAFT means the content may still change, so a metric computed from it could
    not be reproduced. Sealing it as PROVISIONAL is always available and costs
    nothing but honesty, so this is a formatting gate rather than a blocker.
    """
    if benchmark.lifecycle is HeldOutLifecycle.DRAFT:
        raise HeldOutValidationError(
            f"held-out benchmark {benchmark.benchmark!r} is DRAFT: its content can "
            "still change, so no reproducible number can be computed from it. Seal "
            "it first (freeze as PROVISIONAL if human review has not happened yet)."
        )


def require_confirmatory(benchmark: HeldOutBenchmark) -> None:
    """Refuse to call a result a validation unless the instrument was reviewed.

    This is the gate the request's "do not claim successful independent validation
    if the questions remain unreviewed" rule reduces to in code.
    """
    if benchmark.claim_strength is not ClaimStrength.CONFIRMATORY:
        unreviewed = benchmark.review.get("unreviewed_question_ids") or []
        shown = ", ".join(unreviewed[:5])
        more = f" (+{len(unreviewed) - 5} more)" if len(unreviewed) > 5 else ""
        raise HeldOutValidationError(
            f"held-out benchmark {benchmark.benchmark!r} is {benchmark.lifecycle.value} "
            f"and therefore supports {benchmark.claim_strength.value} results only. "
            f"{len(unreviewed)} question(s) carry no human decision: {shown}{more}. A "
            "confirmatory generalization claim requires a VALIDATED instrument, which "
            "must be frozen AFTER the review — a later review cannot upgrade this one, "
            "because a reviewer who has seen the results is no longer blind to them."
        )


# ----------------------------------------------------------------------------
# load / save
# ----------------------------------------------------------------------------


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise HeldOutValidationError(f"file not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HeldOutValidationError(f"{p} is not valid JSON: {exc}") from exc


def load_candidate_set(path: str | Path) -> HeldOutCandidateSet:
    """Load the mutable candidate artifact."""
    raw = _read_json(path)
    try:
        return HeldOutCandidateSet.model_validate(raw)
    except Exception as exc:
        raise HeldOutValidationError(f"candidate set failed validation: {exc}") from exc


def save_candidate_set(candidates: HeldOutCandidateSet, path: str | Path) -> Path:
    """Write the candidate artifact. Pretty-printed and newline-terminated so a
    review diff is readable in git."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(candidates.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return p


def load_heldout_benchmark(path: str | Path) -> HeldOutBenchmark:
    """Load a frozen held-out instrument."""
    raw = _read_json(path)
    try:
        return HeldOutBenchmark.model_validate(raw)
    except Exception as exc:
        raise HeldOutValidationError(f"held-out benchmark failed validation: {exc}") from exc


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes; "" when it cannot be read.

    Re-declared here rather than imported so this module has no dependency on the
    retrieval runner (the runner imports THIS module's neighbours, and a cycle
    would be introduced by importing it back).
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


# ----------------------------------------------------------------------------
# review
# ----------------------------------------------------------------------------


def apply_review(
    candidates: HeldOutCandidateSet,
    decisions: list[HeldOutReviewRequest],
) -> HeldOutCandidateSet:
    """Apply human decisions and return the updated candidate set.

    Rules that keep the record honest:

    * An unknown question id is an ERROR. A decision that silently matched nothing
      would let a reviewer believe they had reviewed more than they did.
    * Editing a REJECTED candidate is an ERROR; rejecting is terminal within a
      version, because un-rejecting after seeing results is the same taint as
      raising the tier.
    * The reviewer's name is mandatory and is stored per question.
    """
    by_id = {c.id: c for c in candidates.candidates}
    unknown = [d.question_id for d in decisions if d.question_id not in by_id]
    if unknown:
        raise HeldOutValidationError(
            f"review decision(s) name unknown candidate id(s): {', '.join(sorted(set(unknown)))}"
        )

    updated: list[HeldOutCandidate] = []
    for cand in candidates.candidates:
        # The last decision for a question wins; decisions are applied in order so
        # a reviewer can correct themselves within one request.
        relevant = [d for d in decisions if d.question_id == cand.id]
        if not relevant:
            updated.append(cand)
            continue
        for decision in relevant:
            if cand.status is CandidateStatus.REJECTED and decision.decision is not ReviewDecision.REJECT:
                raise HeldOutValidationError(
                    f"candidate {cand.id!r} was rejected and cannot be re-reviewed in "
                    "this version: un-rejecting after results are known is the same "
                    "taint the freeze tier exists to prevent. Create a new version."
                )
            record = HeldOutReviewRecord(
                decision=decision.decision,
                reviewer=decision.reviewer,
                notes=decision.notes,
                previous_question=cand.question,
                previous_chunk_ids=list(cand.expected_chunk_ids),
                previous_subdomain=cand.subdomain,
                results_seen=decision.results_seen,
            )
            if decision.decision is ReviewDecision.REJECT:
                cand = cand.model_copy(
                    update={"status": CandidateStatus.REJECTED, "review": record}
                )
            elif decision.decision is ReviewDecision.APPROVE:
                cand = cand.model_copy(
                    update={"status": CandidateStatus.APPROVED, "review": record}
                )
            else:
                cand = cand.model_copy(
                    update={
                        "status": CandidateStatus.EDITED,
                        "review": record,
                        "question": (decision.question or cand.question).strip(),
                        "expected_chunk_ids": (
                            decision.expected_chunk_ids
                            if decision.expected_chunk_ids is not None
                            else cand.expected_chunk_ids
                        ),
                        "subdomain": (decision.subdomain or cand.subdomain).strip(),
                    }
                )
        updated.append(cand)

    refreshed = candidates.model_copy(update={"candidates": updated})
    # Cheap but load-bearing: a candidate set that no longer validates (a validator
    # rule enforced at model level) must not be written back to disk as if it did.
    return HeldOutCandidateSet.model_validate(refreshed.model_dump(mode="json"))


# ----------------------------------------------------------------------------
# freeze
# ----------------------------------------------------------------------------


def freeze(
    candidates: HeldOutCandidateSet,
    *,
    path: str | Path = DEFAULT_HELDOUT_BENCHMARK_PATH,
    frozen_by: str,
    validation: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    require_clean_validation: bool = True,
) -> HeldOutBenchmark:
    """Seal the candidate set into an immutable instrument.

    The tier is computed HERE, from the review records that exist now, and written
    into the file. Two consequences worth stating:

    * A set with even one unreviewed surviving question can only freeze as
      PROVISIONAL, which caps every result from it at EXPLORATORY. That is not a
      failure — it is the honest description of an agent-drafted set.
    * An existing frozen artifact is never overwritten with different content. A
      changed instrument needs a new version and a new filename, so a result can
      never be silently re-pointed at a benchmark it did not use.
    """
    if not (frozen_by or "").strip():
        raise HeldOutValidationError("freezing must name who sealed the artifact")

    surviving = candidates.surviving
    if not surviving:
        raise HeldOutValidationError(
            "refusing to freeze a held-out benchmark with no surviving questions: "
            "there would be nothing to measure"
        )
    unreviewed = [c.id for c in surviving if not c.status.is_reviewed]
    lifecycle = HeldOutLifecycle.PROVISIONAL if unreviewed else HeldOutLifecycle.VALIDATED

    errors: list[str] = list((validation or {}).get("errors") or [])
    if errors and require_clean_validation:
        raise HeldOutValidationError(
            f"refusing to freeze: the candidate set has {len(errors)} validation "
            f"error(s) that must be resolved or the affected questions rejected. "
            f"First: {errors[0]}"
        )

    questions = [
        {
            "id": c.id,
            "subdomain": c.subdomain,
            "question": c.question,
            "expected_chunk_ids": list(c.expected_chunk_ids),
            "expected_document_ids": list(c.expected_document_ids),
            "expected_keywords": list(c.expected_keywords),
            "provenance": c.provenance,
            # Held-out extras; the retrieval loader ignores unknown fields, so
            # these travel with the question without breaking the engine.
            "coverage_rationale": c.coverage_rationale,
            "query_type": c.query_type.value,
            "difficulty": c.difficulty.value,
            "draft_origin": c.draft_origin,
            "review_status": c.status.value,
        }
        for c in surviving
    ]

    completeness = candidates.review_completeness()
    benchmark = HeldOutBenchmark(
        benchmark=candidates.benchmark,
        version=candidates.version,
        created_at=candidates.created_at,
        kb_id=candidates.kb_id,
        kb_name=candidates.kb_name,
        derived_from=candidates.derived_from,
        authorship={
            **candidates.authorship,
            # Both statements, always: agent-drafted AND human-reviewed are
            # independent facts and the artifact must not imply one from the other.
            "agent_authored": True,
            "human_review": (
                "COMPLETE — every surviving question carries a human decision"
                if not unreviewed
                else f"PENDING — {len(unreviewed)} question(s) carry no human decision"
            ),
            "frozen_by": frozen_by.strip(),
        },
        relevance_model=candidates.relevance_model,
        questions=questions,
        lifecycle=lifecycle,
        claim_strength=claim_strength_for(lifecycle),
        frozen_at=now_iso(),
        frozen_by=frozen_by.strip(),
        source_fingerprint=candidates.fingerprint(),
        review=completeness,
        protocol=candidates.protocol,
        composition={
            "question_count": len(surviving),
            "subdomains": candidates.subdomain_counts(),
            "query_types": candidates.query_type_counts(),
            "difficulty": candidates.difficulty_counts(),
            "rejected": completeness["candidates_rejected"],
            "rejected_question_ids": [c.id for c in candidates.rejected],
        },
        rejected_question_ids=[c.id for c in candidates.rejected],
        notes=list(notes or []),
    )

    target = Path(path)
    if target.exists():
        existing = load_heldout_benchmark(target)
        if existing.fingerprint() != benchmark.fingerprint():
            raise HeldOutValidationError(
                f"{target} already holds a frozen held-out benchmark "
                f"({existing.fingerprint()}) whose content differs from this freeze "
                f"({benchmark.fingerprint()}). A frozen instrument is immutable: "
                "write a new version file instead of editing this one, so no result "
                "can be re-pointed at a benchmark it did not use."
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(benchmark.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return benchmark


__all__ = [
    "CandidateStatus",
    "ClaimStrength",
    "DEFAULT_DEV_BENCHMARK_PATH",
    "DEFAULT_HELDOUT_BENCHMARK_PATH",
    "DEFAULT_HELDOUT_CANDIDATE_PATH",
    "Difficulty",
    "HeldOutBenchmark",
    "HeldOutCandidate",
    "HeldOutCandidateSet",
    "HeldOutLifecycle",
    "HeldOutReviewRecord",
    "HeldOutReviewRequest",
    "HeldOutValidationError",
    "QueryType",
    "ReviewDecision",
    "apply_review",
    "claim_strength_for",
    "file_sha256",
    "freeze",
    "load_candidate_set",
    "load_heldout_benchmark",
    "now_iso",
    "require_confirmatory",
    "require_runnable",
    "save_candidate_set",
]
