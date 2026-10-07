"""Failure taxonomy for answer-quality evaluation (V9 Phase 2).

Every evaluated question may be classified with **one dominant failure mode**
where evidence permits. Classifications are DETERMINISTIC, transparent and
attributed: they never claim semantic truth and they prefer UNKNOWN over a
guess.

The taxonomy exists so a reviewer can answer "why did this answer fail?" with a
fixed, inspectable code plus the exact evidence that produced it. It is not a
substitute for human judgment.

Label vocabulary (closed set):

    NO_RELEVANT_RETRIEVAL
    LOW_RETRIEVAL_RECALL
    WRONG_DOCUMENT
    WRONG_CHUNK
    INSUFFICIENT_EVIDENCE
    CONFLICTING_EVIDENCE
    CITATION_ERROR
    UNSUPPORTED_CLAIM
    ANSWER_RELEVANCE_FAILURE
    GENERATION_FAILURE
    UNKNOWN

Each classification carries:
    classification   the closed-set code above, or "" when nothing failed
    confidence       "deterministic" | "partial" | "unknown"
    evidence         the exact signals used, as human-readable strings
    reason           human-readable explanation

## Why the rules are ordered this way

The rules are ordered so that a **failure the pipeline itself stated** always
outranks a failure this module **infers**, and so that a retrieval fault
outranks an answer-symptom. Precedence:

1. ``GENERATION_FAILED`` answer status  -> GENERATION_FAILURE (stated)
2. retrieval ran, nothing required hit -> NO_RELEVANT_RETRIEVAL
3. stated ``INSUFFICIENT_EVIDENCE``, or retrieval returned nothing at all while
   the question is answerable from the corpus -> INSUFFICIENT_EVIDENCE
4. required evidence retrieved but deep -> LOW_RETRIEVAL_RECALL
5. required evidence retrieved but outranked -> WRONG_CHUNK
6. evidence drawn from the wrong documents -> WRONG_DOCUMENT
7. a claim contradicted by its own citations -> CONFLICTING_EVIDENCE
8. a claim asserted with no support -> UNSUPPORTED_CLAIM
9. a citation that resolves to nothing retrieved -> CITATION_ERROR
10. answer on the wrong subject -> ANSWER_RELEVANCE_FAILURE
11. empty answer with no stated status -> GENERATION_FAILURE (inferred)
12. otherwise -> UNKNOWN

## Ground truth is required before blaming retrieval

``retrieval_hit_rate`` is UNKNOWN (``None``) whenever the benchmark carries no
required-evidence label for a question (see
``services.answer_eval.evaluator``). A ``None`` hit rate therefore means "we
have no reference evidence", and this module will NOT classify a retrieval
fault from it. That is why the no-ground-truth cases fall through to UNKNOWN
instead of being blamed on retrieval: missing ground truth is not evidence of a
retrieval failure.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.services.answer_eval.metrics import AnswerQualityResult


class FailureCode(str, Enum):
    """Closed set of failure labels.

    Adding a label is a deliberate breaking change because the UI and the
    persisted experiment records key on this set.
    """

    NO_RELEVANT_RETRIEVAL = "NO_RELEVANT_RETRIEVAL"
    LOW_RETRIEVAL_RECALL = "LOW_RETRIEVAL_RECALL"
    WRONG_DOCUMENT = "WRONG_DOCUMENT"
    WRONG_CHUNK = "WRONG_CHUNK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    CITATION_ERROR = "CITATION_ERROR"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    ANSWER_RELEVANCE_FAILURE = "ANSWER_RELEVANCE_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    UNKNOWN = "UNKNOWN"


#: How much a classification can be trusted. Never a numeric confidence:
#: a rule either fired on a stated signal, on an inferred signal, or not at all.
class FailureConfidence(str, Enum):
    #: Fired on a value the pipeline or the benchmark states directly.
    DETERMINISTIC = "deterministic"
    #: Fired on an inference (e.g. empty answer text with no status).
    PARTIAL = "partial"
    #: Nothing fired; the failure mode could not be determined.
    UNKNOWN = "unknown"


#: Every published label, in deterministic order of precedence.
ALL_CODES: tuple[FailureCode, ...] = (
    FailureCode.NO_RELEVANT_RETRIEVAL,
    FailureCode.LOW_RETRIEVAL_RECALL,
    FailureCode.WRONG_DOCUMENT,
    FailureCode.WRONG_CHUNK,
    FailureCode.INSUFFICIENT_EVIDENCE,
    FailureCode.CONFLICTING_EVIDENCE,
    FailureCode.CITATION_ERROR,
    FailureCode.UNSUPPORTED_CLAIM,
    FailureCode.ANSWER_RELEVANCE_FAILURE,
    FailureCode.GENERATION_FAILURE,
    FailureCode.UNKNOWN,
)

#: Ranks at or below this are considered "the front of the retrieved list".
#:
#: This is a CONVENTION, not a measured truth, and it is named so it cannot be
#: silently changed: a required chunk inside the front window that still lost to
#: another chunk is reported as WRONG_CHUNK (ranking problem), while a required
#: chunk beyond it is reported as LOW_RETRIEVAL_RECALL (depth problem) because a
#: smaller `top_k` window would have dropped it entirely. Both are retrieval
#: faults; the distinction only directs which knob to try.
FRONT_RANK_WINDOW = 3


class FailureClassification(BaseModel):
    """A single, auditable failure classification for one question.

    ``classification`` is "" when nothing failed, or the dominant failure label
    when evidence supports one. ``evidence`` holds the exact signals that
    produced the classification so a reviewer can trace every decision.
    """

    classification: str = Field(
        default="",
        description="Closed-set failure code, or '' when no failure was classified",
    )
    confidence: FailureConfidence = FailureConfidence.UNKNOWN
    evidence: list[str] = Field(
        default_factory=list, description="The exact signals that produced this label"
    )
    reason: str = ""

    @model_validator(mode="after")
    def _check_code(self) -> "FailureClassification":
        if self.classification and self.classification not in {
            c.value for c in ALL_CODES
        }:
            raise ValueError(
                f"unknown failure code {self.classification!r}; "
                f"allowed: {[c.value for c in ALL_CODES]}"
            )
        return self

    @property
    def failed(self) -> bool:
        return bool(self.classification)


def _measured_value(metric: Any) -> float | None:
    """Read a V8 ``Measured`` (or a plain number) as 'measured value or None'."""
    if metric is None:
        return None
    value = getattr(metric, "value", metric)
    measured = getattr(metric, "measured", True)
    if not measured:
        return None
    return value


def _verdict_field(verdict: Any, name: str, default: Any = None) -> Any:
    if isinstance(verdict, dict):
        return verdict.get(name, default)
    return getattr(verdict, name, default)


def _claim_verdicts_as_dicts(verdicts: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in verdicts or []:
        if isinstance(v, dict):
            out.append(v)
        elif hasattr(v, "model_dump"):
            out.append(v.model_dump())
        else:
            out.append(
                {
                    "evaluated_state": getattr(v, "evaluated_state", ""),
                    "has_invalid_citation": getattr(v, "has_invalid_citation", False),
                    "uncited": getattr(v, "uncited", False),
                    "entailment": str(getattr(v, "entailment", "")),
                }
            )
    return out


def classify_failure(
    *,
    question: str,
    answer_text: str,
    evidence: list[dict[str, Any]],
    retrieved_chunk_ids: list[str],
    retrieved_rank_of_required: int | None,
    required_chunk_ids: list[str],
    retrieval_hit_rate: float | None,
    question_answer_relevance: float | None,
    relevance_passed: bool | None,
    claim_verdicts: list[dict[str, Any]],
    fabrication_rate: float | None,
    unsupported_citation_rate: float | None,
    answerability: str,
    answer_status: str = "",
    grounding_state: str = "",
    abstained: bool = False,
) -> FailureClassification:
    """Deterministic, evidence-driven failure classification.

    Every argument is either a value the benchmark states or a value V8's
    evaluator measured. Arguments left as ``None`` mean UNMEASURED — they never
    count as zero and never justify a classification.

    See the module docstring for the precedence order and why it is that order.
    """

    evidence = list(evidence or [])
    claim_verdicts = _claim_verdicts_as_dicts(claim_verdicts)
    retrieved = list(retrieved_chunk_ids or [])
    required = list(required_chunk_ids or [])
    required_set = set(required)
    text = (answer_text or "").strip()

    # ------------------------------------------------------------------
    # 1. GENERATION_FAILURE — the pipeline itself said generation failed.
    # ------------------------------------------------------------------
    if answer_status == "generation_failed":
        return FailureClassification(
            classification=FailureCode.GENERATION_FAILURE,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=["answer_status=generation_failed (reported by the pipeline)"],
            reason="the pipeline reported that generation itself failed.",
        )

    # ------------------------------------------------------------------
    # 2. NO_RELEVANT_RETRIEVAL — retrieval ran, ground truth exists, and none
    #    of the required evidence came back.
    # ------------------------------------------------------------------
    if retrieved and retrieval_hit_rate == 0.0 and not evidence:
        return FailureClassification(
            classification=FailureCode.NO_RELEVANT_RETRIEVAL,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"retrieved {len(retrieved)} chunk(s); none matched the "
                f"{len(required)} required-evidence chunk(s)",
                "retrieval_hit_rate=0.0 (measured, required evidence labelled)",
            ],
            reason=(
                "retrieval surfaced no required passage: none of the evidence "
                "the benchmark identifies as necessary was returned at any rank."
            ),
        )

    # ------------------------------------------------------------------
    # 3. INSUFFICIENT_EVIDENCE — stated by the pipeline's gate, or retrieval
    #    returned nothing at all for a question the corpus can answer.
    # ------------------------------------------------------------------
    if grounding_state == "INSUFFICIENT_EVIDENCE":
        return FailureClassification(
            classification=FailureCode.INSUFFICIENT_EVIDENCE,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                "grounding_state=INSUFFICIENT_EVIDENCE (reported by the gate)"
            ],
            reason="the evidence gate found no passage sufficient to answer.",
        )
    if (
        answerability in ("answerable", "unknown")
        and text != ""
        and not retrieved
        and retrieval_hit_rate == 0.0
        and required_set
        and not evidence
    ):
        return FailureClassification(
            classification=FailureCode.INSUFFICIENT_EVIDENCE,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                "retrieval returned no candidate chunks at all",
                f"{len(required)} required-evidence chunk(s) were labelled but "
                "never entered the candidate set",
            ],
            reason=(
                "retrieval produced an empty candidate set, so the generator had "
                "no passage to ground an answer in — a corpus/retrieval supply "
                "gap rather than a ranking miss."
            ),
        )

    # ------------------------------------------------------------------
    # 4. LOW_RETRIEVAL_RECALL — the required evidence WAS retrieved, but
    #    beyond the front window, so a smaller top_k would have dropped it.
    # ------------------------------------------------------------------
    if (
        retrieved_rank_of_required is not None
        and retrieved_rank_of_required > FRONT_RANK_WINDOW
    ):
        return FailureClassification(
            classification=FailureCode.LOW_RETRIEVAL_RECALL,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"required chunk at rank {retrieved_rank_of_required} "
                f"(front window = {FRONT_RANK_WINDOW})",
                "retrieval_hit_rate=1.0 — the evidence was retrieved, just too deep",
            ],
            reason=(
                "the required evidence was retrieved, but at a rank beyond the "
                "front window, so any top_k smaller than that rank would miss it."
            ),
        )

    # ------------------------------------------------------------------
    # 5. WRONG_CHUNK — the required chunk was near the top but lost to
    #    another chunk, so the top result misled the generator.
    # ------------------------------------------------------------------
    if (
        retrieved_rank_of_required is not None
        and 1 < retrieved_rank_of_required <= FRONT_RANK_WINDOW
    ):
        top = retrieved[0] if retrieved else "?"
        return FailureClassification(
            classification=FailureCode.WRONG_CHUNK,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"required chunk at rank {retrieved_rank_of_required} "
                f"(front window = {FRONT_RANK_WINDOW})",
                f"top-ranked chunk = {top}",
            ],
            reason=(
                "the required chunk was inside the front window but was outranked "
                "by a chunk that is not the evidence a correct answer needs."
            ),
        )

    # ------------------------------------------------------------------
    # 6. WRONG_DOCUMENT — ground truth exists, the answer used evidence, and
    #    that evidence came from documents other than the required ones.
    # ------------------------------------------------------------------
    required_docs = {
        e.get("document_id") for e in evidence if e.get("required") and e.get("document_id")
    }
    used_docs = {
        e.get("document_id")
        for e in evidence
        if not e.get("required") and e.get("document_id")
    }
    if (
        required_docs
        and used_docs
        and required_docs.isdisjoint(used_docs)
        and retrieval_hit_rate == 0.0
    ):
        return FailureClassification(
            classification=FailureCode.WRONG_DOCUMENT,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"evidence used came from document(s) {sorted(used_docs)}",
                f"required evidence lives in document(s) {sorted(required_docs)}",
            ],
            reason=(
                "the retrieved evidence came from documents unrelated to the "
                "question's subject, while the documents that can answer it were "
                "never surfaced."
            ),
        )

    # ------------------------------------------------------------------
    # 7. CONFLICTING_EVIDENCE — a claim the answer made is contradicted by the
    #    passage it cites, so the retrieved material disagrees with itself.
    # ------------------------------------------------------------------
    contradicted = [
        v for v in claim_verdicts if v.get("evaluated_state") == "CONTRADICTED"
    ]
    if contradicted:
        return FailureClassification(
            classification=FailureCode.CONFLICTING_EVIDENCE,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"{len(contradicted)} claim(s) judged CONTRADICTED by the "
                "evidence they cite"
            ],
            reason=(
                "at least one cited passage contradicts the claim drawn from it, "
                "so the evidence set is internally inconsistent about the answer."
            ),
        )

    # ------------------------------------------------------------------
    # 8. UNSUPPORTED_CLAIM — the answer asserted something with no support.
    # ------------------------------------------------------------------
    unsupported = [
        v for v in claim_verdicts if v.get("evaluated_state") == "UNSUPPORTED"
    ]
    if unsupported:
        return FailureClassification(
            classification=FailureCode.UNSUPPORTED_CLAIM,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"{len(unsupported)} claim(s) judged UNSUPPORTED (asserted fact, "
                "no supporting citation)"
            ],
            reason=(
                "the answer states facts that its own citations do not support, "
                "so it is not grounded in the retrieved evidence."
            ),
        )

    # ------------------------------------------------------------------
    # 9. CITATION_ERROR — a citation points at something that was never
    #    retrieved, or whose provenance does not validate.
    # ------------------------------------------------------------------
    invalid_citations = [
        v
        for v in claim_verdicts
        if v.get("has_invalid_citation") or v.get("uncited")
    ]
    fabricated = fabrication_rate is not None and fabrication_rate > 0.0
    if invalid_citations or fabricated:
        signals: list[str] = []
        if invalid_citations:
            signals.append(
                f"{len(invalid_citations)} claim(s) carry an unresolvable citation "
                "or assert fact with no citation"
            )
        if fabricated:
            signals.append(f"fabricated_citation_rate={fabrication_rate}")
        return FailureClassification(
            classification=FailureCode.CITATION_ERROR,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=signals,
            reason=(
                "the answer cites evidence that was never retrieved, or asserts "
                "facts with no citation at all, so its provenance is broken."
            ),
        )

    # ------------------------------------------------------------------
    # 10. ANSWER_RELEVANCE_FAILURE — grounded, but about the wrong subject.
    # ------------------------------------------------------------------
    if relevance_passed is False:
        return FailureClassification(
            classification=FailureCode.ANSWER_RELEVANCE_FAILURE,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"question/answer relevance={question_answer_relevance} "
                "(below the pass threshold)"
            ],
            reason=(
                "the answer does not actually address the question asked: its "
                "content terms do not overlap the question's subject."
            ),
        )

    # ------------------------------------------------------------------
    # 11. GENERATION_FAILURE (inferred) — no text and no legitimation for it.
    # ------------------------------------------------------------------
    if not text and not abstained and answer_status != "abstained":
        return FailureClassification(
            classification=FailureCode.GENERATION_FAILURE,
            confidence=FailureConfidence.PARTIAL,
            evidence=[
                "answer_text is empty and the pipeline reported no "
                "generation_failed status",
            ],
            reason=(
                "the generator produced no text. Inferred from the empty answer "
                "alone, so a reviewer should confirm it was not an intended "
                "abstention."
            ),
        )

    # ------------------------------------------------------------------
    # 12. UNKNOWN — prefer honest uncertainty over a forced verdict.
    # ------------------------------------------------------------------
    return FailureClassification(
        classification=FailureCode.UNKNOWN,
        confidence=FailureConfidence.UNKNOWN,
        evidence=[
            f"answer_text length={len(text)}",
            f"required_chunks={len(required)}, retrieved_chunks={len(retrieved)}",
            f"retrieval_hit_rate={retrieval_hit_rate!r}",
            f"ground_truth_label_present={bool(required)}",
        ],
        reason=(
            "no deterministic rule fired on measured evidence: either the "
            "benchmark carries no required-evidence label for this question or "
            "the signals available do not distinguish between failure modes. "
            "A human reviewer should decide."
        ),
    )


def classify_result(
    result: AnswerQualityResult,
    *,
    evidence: list[dict[str, Any]] | None = None,
    answer_status: str = "",
    grounding_state: str = "",
    abstained: bool = False,
) -> FailureClassification:
    """Classify a V8 ``AnswerQualityResult`` using only its measured values.

    This is the integration point: it reads the same metrics the UI displays and
    never invents a signal. A metric the evaluator left UNKNOWN arrives here as
    ``None`` and cannot justify a classification.
    """

    return classify_failure(
        question=getattr(result, "question", ""),
        answer_text=result.answer_text,
        evidence=list(evidence or []),
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        retrieved_rank_of_required=result.retrieved_rank_of_required,
        required_chunk_ids=result.required_chunk_ids,
        retrieval_hit_rate=_measured_value(result.retrieval_hit_rate),
        question_answer_relevance=_measured_value(result.question_answer_relevance),
        relevance_passed=result.relevance_passed,
        claim_verdicts=[
            v.model_dump() if hasattr(v, "model_dump") else dict(v)
            for v in result.claim_verdicts
        ],
        fabrication_rate=_measured_value(result.fabricated_citation_rate),
        unsupported_citation_rate=_measured_value(result.unsupported_citation_rate),
        answerability=result.answerability,
        answer_status=answer_status or result.answer_status,
        grounding_state=grounding_state or result.actual_grounding_state,
        abstained=abstained or result.abstention_performed,
    )


__all__ = [
    "ALL_CODES",
    "FRONT_RANK_WINDOW",
    "FailureClassification",
    "FailureCode",
    "FailureConfidence",
    "classify_failure",
    "classify_result",
]
