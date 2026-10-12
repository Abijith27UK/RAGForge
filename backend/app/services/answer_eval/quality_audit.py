"""V11 pre-freeze benchmark quality audit — findings, never silent fixes.

WHY THIS EXISTS
---------------
V10 can freeze whatever a human approves. Before freezing the official answer
benchmark we need the answer to "is this benchmark actually sound?", and the
dangerous failure mode is a benchmark that *looks* complete: 28 questions,
28 approvals, a green gate, and a reference answer that no corpus text
supports. That benchmark would produce a confident correctness number that
means nothing.

So this module inspects the artifact and the live corpus and reports problems.
It is deliberately READ-ONLY: it has no write path to the benchmark, the
annotations, or Qdrant. A finding is a recommendation for a human, never an
edit. The V10 rule that a benchmark problem is preserved in the audit trail
rather than silently repaired applies to everything here.

WHICH CHECKS RUN, AND WHICH DELIBERATELY DO NOT
-----------------------------------------------
Every check either establishes something from the stored artifacts (evidence
present? hashes match? are two questions near-identical? does an authored key
point have support?) or reports that it could not run, with its reason. Nothing
returns a pass by default. In particular the annotation-dependent checks report
`no_annotation` before any human has authored anything, rather than reporting
"0 problems" and implying the reference answers were verified — the exact
substitution of "not checked" for "checked and fine" that this project refuses.

The audit is also explicit that it CANNOT decide meaning. It can prove a cited
chunk is missing, that a question's terms appear nowhere in its evidence, or
that two questions are nearly the same string. It cannot prove a question is
ambiguous or that a reference answer is wrong; it flags the ones a human must
look at and says so in `limitations`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.services.answer_eval.benchmark import AnswerBenchmark
from app.services.answer_eval.drafting import MIN_QUESTION_COVERAGE
from app.services.answer_eval.entailment import (
    HeuristicCitationEntailment,
    SupportJudgement,
)
from app.services.answer_eval.ground_truth import GroundTruthAnnotation
from app.services.answer_eval.relevance import TermWeights
from app.services.answering.query_processor import extract_terms

#: A chunk shorter than this cannot plausibly carry an answer paragraph. Set
#: well below the corpus's real chunk sizes, so it fires only on genuinely
#: degenerate content (a stray title, a nav fragment) rather than on short
#: but legitimate chunks.
MIN_EVIDENCE_CHARS = 80

#: Jaccard overlap of question content terms above which two questions are
#: reported as near-duplicates. High on purpose: two genuinely distinct
#: questions in the same subdomain share plenty of vocabulary.
NEAR_DUPLICATE_JACCARD = 0.60

#: Containment above which a shorter question is reported as covered by a
#: longer one (catches "What is X?" duplicated inside "What is X and how does
#: it work?").
NEAR_DUPLICATE_CONTAINMENT = 0.80

#: Question shorter than this is reported as under-specified.
MIN_QUESTION_CHARS = 25

_REFERENCE_DANGLING = (
    "the above",
    "the following",
    "as mentioned",
    "this document",
    "the previous section",
    "described above",
    "mentioned earlier",
    "figure ",
    "table ",
)
_COMPOUND_RE = (
    (" and what ", "and what"),
    (" and how ", "and how"),
    (" and why ", "and why"),
    (" and which ", "and which"),
    (" then what ", "then what"),
)


class Severity(str, Enum):
    ERROR = "error"  # blocks freezing: the benchmark cannot be trusted as-is
    WARNING = "warning"  # a human decision is required before freezing
    INFO = "info"  # recorded for the reviewer; not a defect


class RecommendedAction(str, Enum):
    """What the auditor suggests a human do. Suggestions only — never applied."""

    NONE = "none"
    KEEP_ANSWERABLE = "keep_answerable"
    FIX_EVIDENCE_LINK = "fix_evidence_link"
    REWRITE_QUESTION = "rewrite_question"
    CLASSIFY_AMBIGUOUS = "classify_ambiguous"
    CLASSIFY_INSUFFICIENT_EVIDENCE = "classify_insufficient_evidence"
    REVIEW_QUESTION_EVIDENCE_MATCH = "review_question_evidence_match"
    AUTHOR_MISSING_LABELS = "author_missing_labels"
    RESOLVE_DUPLICATE = "resolve_duplicate"


class Finding(BaseModel):
    """One reported problem. `code` is stable so it can be asserted in tests."""

    code: str
    severity: Severity
    message: str
    question_id: str = ""
    also_question_id: str = Field(
        default="", description="Second question for pairwise findings (near-duplicates)."
    )
    evidence_chunk_id: str = ""
    action: RecommendedAction = RecommendedAction.NONE
    detail: dict[str, Any] = Field(default_factory=dict)


class QuestionAudit(BaseModel):
    question_id: str
    question: str
    subdomain: str = ""
    answerability: str = "unknown"
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    has_annotation: bool = False
    key_points_checked: int = 0
    key_points_supported: int = 0
    key_points_unsupported: list[str] = Field(default_factory=list)
    question_terms: list[str] = Field(default_factory=list)
    uncovered_terms: list[str] = Field(default_factory=list)
    coverage: float | None = Field(
        default=None,
        description="IDF-weighted coverage of the question's content terms by its own "
        "pinned evidence. None when corpus statistics were unavailable — never a "
        "substituted 0.",
    )
    recommended_action: RecommendedAction = RecommendedAction.NONE
    findings: list[Finding] = Field(default_factory=list)


class BenchmarkQualityAudit(BaseModel):
    """The read-only report. Nothing here modifies the benchmark."""

    kb_id: str
    benchmark_name: str
    benchmark_fingerprint: str = ""
    benchmark_path: str = ""
    audited_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    audited_by: str = "RAGForge V11 benchmark quality audit (deterministic, read-only)"

    questions: list[QuestionAudit] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)

    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    ready_to_freeze: bool = False
    verdict: str = ""

    checks_run: list[str] = Field(default_factory=list)
    checks_not_run: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


#: Named checks, so `checks_run` / `checks_not_run` are auditable.
CHECKS = (
    "evidence_present_in_corpus",
    "evidence_content_hash_matches",
    "evidence_document_matches",
    "evidence_text_non_trivial",
    "answerability_supported_by_evidence",
    "question_terms_covered_by_evidence",
    "question_wording_heuristics",
    "near_duplicate_questions",
    "citation_requirement_consistency",
    "authored_key_points_supported_by_evidence",
    "annotation_reference_answer_grounded",
    "shared_required_evidence",
    "evidence_source_title_recorded",
)


def _severity_rank(severity: Severity) -> int:
    return {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}[severity]


def _overlap(a: set[str], b: set[str]) -> tuple[float, float]:
    """(Jaccard, containment of the smaller set in the larger)."""
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    union = len(a | b)
    jaccard = inter / union if union else 0.0
    containment = inter / min(len(a), len(b))
    return jaccard, containment


def _wording_findings(question_id: str, question: str, terms: list[str]) -> list[Finding]:
    """Deterministic wording heuristics. These flag; they do not decide.

    A finding here means "a human should read this question again", NOT "this
    question is ambiguous" — no deterministic rule can establish meaning, and
    the report says so in `limitations`.
    """
    findings: list[Finding] = []
    text = (question or "").strip()
    low = f" {text.lower()} "

    if len(text) < MIN_QUESTION_CHARS:
        findings.append(
            Finding(
                code="QUESTION_UNDER_SPECIFIED",
                severity=Severity.WARNING,
                question_id=question_id,
                message=(
                    f"question is only {len(text)} characters; too short to fix what a "
                    f"correct answer must contain"
                ),
                action=RecommendedAction.REWRITE_QUESTION,
                detail={"length": len(text), "floor": MIN_QUESTION_CHARS},
            )
        )

    if text.count("?") > 1:
        findings.append(
            Finding(
                code="QUESTION_MULTIPLE_INTERROGATIVES",
                severity=Severity.WARNING,
                question_id=question_id,
                message=(
                    f"question contains {text.count('?')} question marks; an evaluation "
                    f"item should ask one thing unless the reference answer is authored to "
                    f"cover every part"
                ),
                action=RecommendedAction.REWRITE_QUESTION,
            )
        )

    for needle, label in _COMPOUND_RE:
        if needle in low:
            findings.append(
                Finding(
                    code="QUESTION_COMPOUND",
                    severity=Severity.INFO,
                    question_id=question_id,
                    message=(
                        f"question asks more than one thing ('{label.strip()}'); the "
                        f"reference answer and key points must cover BOTH parts or the "
                        f"wrong key point will be recorded as missing"
                    ),
                    detail={"connector": label.strip()},
                )
            )
            break

    for dangling in _REFERENCE_DANGLING:
        if dangling in low:
            findings.append(
                Finding(
                    code="QUESTION_DANGLING_REFERENCE",
                    severity=Severity.WARNING,
                    question_id=question_id,
                    message=(
                        f"question refers to '{dangling.strip()}', which has no antecedent in "
                        f"a standalone benchmark item"
                    ),
                    action=RecommendedAction.REWRITE_QUESTION,
                    detail={"phrase": dangling.strip()},
                )
            )

    if len(terms) < 2:
        findings.append(
            Finding(
                code="QUESTION_TOO_FEW_CONTENT_TERMS",
                severity=Severity.WARNING,
                question_id=question_id,
                message=(
                    f"question has {len(terms)} content term(s) after stop-word removal; "
                    f"retrieval and term-coverage metrics cannot discriminate for it"
                ),
                action=RecommendedAction.REWRITE_QUESTION,
                detail={"terms": terms},
            )
        )

    return findings


def run_quality_audit(
    *,
    benchmark: AnswerBenchmark,
    evidence_texts: dict[str, str],
    evidence_meta: dict[str, dict[str, Any]] | None = None,
    annotations: list[GroundTruthAnnotation] | None = None,
    weights: TermWeights | None = None,
    benchmark_path: str = "",
    benchmark_fingerprint: str = "",
) -> BenchmarkQualityAudit:
    """Run every check that CAN run, and say which could not.

    Pure and offline: the caller supplies live-corpus chunk texts, the KB's
    corpus term statistics (optional but load-bearing for the coverage check),
    and current annotations. Read-only by construction — there is no parameter
    that lets a caller mutate the benchmark from here.
    """
    meta = evidence_meta or {}
    by_question: dict[str, list[GroundTruthAnnotation]] = {}
    for annotation in annotations or []:
        by_question.setdefault(annotation.question_id, []).append(annotation)

    entailment = HeuristicCitationEntailment()
    per_question: list[QuestionAudit] = []
    findings: list[Finding] = []
    checks_run: set[str] = set()
    checks_not_run: dict[str, str] = {}
    title_gaps: list[tuple[str, str, str]] = []

    effective_annotations: dict[str, GroundTruthAnnotation] = {}
    for annotation in sorted(
        annotations or [], key=lambda a: (a.created_at, a.annotation_id)
    ):
        effective_annotations[annotation.question_id] = annotation

    questions = list(benchmark.questions or [])

    for question in questions:
        qid = question.question_id
        q_audit = QuestionAudit(
            question_id=qid,
            question=question.question,
            subdomain=question.subdomain,
            answerability=str(getattr(question.answerability, "value", question.answerability)),
            evidence_chunk_ids=[e.chunk_id for e in question.required_evidence],
        )
        q_findings: list[Finding] = []

        # -- evidence integrity ------------------------------------------------
        checks_run.add("evidence_present_in_corpus")
        usable: list[str] = []
        for ev in question.required_evidence:
            info = meta.get(ev.chunk_id, {})
            text = evidence_texts.get(ev.chunk_id, "")
            if not text.strip():
                q_findings.append(
                    Finding(
                        code="EVIDENCE_MISSING_FROM_CORPUS",
                        severity=Severity.ERROR,
                        question_id=qid,
                        evidence_chunk_id=ev.chunk_id,
                        message=(
                            f"required evidence chunk {ev.chunk_id} has no text in the live "
                            f"corpus; no reference answer can be grounded in it and no "
                            f"answer can cite it"
                        ),
                        action=RecommendedAction.FIX_EVIDENCE_LINK,
                    )
                )
                continue
            usable.append(ev.chunk_id)

            if info.get("content_hash") and ev.content_hash and (
                info["content_hash"] != ev.content_hash
            ):
                q_findings.append(
                    Finding(
                        code="EVIDENCE_HASH_MISMATCH",
                        severity=Severity.ERROR,
                        question_id=qid,
                        evidence_chunk_id=ev.chunk_id,
                        message=(
                            "the live corpus chunk's content hash differs from the benchmark "
                            "label: the corpus was re-chunked, so anything authored against "
                            "the old text must be re-read"
                        ),
                        action=RecommendedAction.FIX_EVIDENCE_LINK,
                        detail={
                            "benchmark_hash": ev.content_hash,
                            "corpus_hash": info["content_hash"],
                        },
                    )
                )
                checks_run.add("evidence_content_hash_matches")
            if info.get("document_id") and ev.document_id and (
                info["document_id"] != ev.document_id
            ):
                q_findings.append(
                    Finding(
                        code="EVIDENCE_DOCUMENT_MISMATCH",
                        severity=Severity.ERROR,
                        question_id=qid,
                        evidence_chunk_id=ev.chunk_id,
                        message=(
                            f"chunk now belongs to document {info['document_id']}, benchmark "
                            f"says {ev.document_id}"
                        ),
                        action=RecommendedAction.FIX_EVIDENCE_LINK,
                    )
                )

            if len(text) < MIN_EVIDENCE_CHARS:
                q_findings.append(
                    Finding(
                        code="EVIDENCE_TEXT_TRIVIAL",
                        severity=Severity.WARNING,
                        question_id=qid,
                        evidence_chunk_id=ev.chunk_id,
                        message=(
                            f"required evidence is only {len(text)} characters; too short to "
                            f"support a reference answer for this question"
                        ),
                        action=RecommendedAction.CLASSIFY_INSUFFICIENT_EVIDENCE,
                        detail={"chars": len(text), "floor": MIN_EVIDENCE_CHARS},
                    )
                )

            # Display-provenance discrepancy: the benchmark's label records a source
            # title, the live chunk does not (or records a different one). The
            # load-bearing provenance — chunk id, document id, content hash — is
            # checked above and still holds, so this is not an integrity failure; it
            # is a real gap in what a reviewer can see, and it is reported rather
            # than smoothed over. Where the whole corpus is affected it is raised
            # once at benchmark level instead of once per chunk.
            checks_run.add("evidence_source_title_recorded")
            label_title = (ev.source_title or "").strip()
            chunk_title = str(info.get("source_title") or "").strip()
            if label_title and not chunk_title:
                title_gaps.append((qid, ev.chunk_id, label_title))
            elif label_title and chunk_title and label_title != chunk_title:
                q_findings.append(
                    Finding(
                        code="EVIDENCE_SOURCE_TITLE_DIFFERS",
                        severity=Severity.INFO,
                        question_id=qid,
                        evidence_chunk_id=ev.chunk_id,
                        message=(
                            f"benchmark label records source title {label_title!r} but the live "
                            f"chunk records {chunk_title!r}; the chunk id and content hash "
                            f"agree, so this is a display-metadata discrepancy"
                        ),
                        detail={"label": label_title, "corpus": chunk_title},
                    )
                )

        if not question.required_evidence:
            q_findings.append(
                Finding(
                    code="NO_REQUIRED_EVIDENCE",
                    severity=Severity.ERROR,
                    question_id=qid,
                    message=(
                        "question pins no required evidence; answerability cannot be "
                        "established and citation_recall is undefined for it"
                    ),
                    action=RecommendedAction.CLASSIFY_INSUFFICIENT_EVIDENCE,
                )
            )

        # -- answerability vs. what the corpus actually holds -------------------
        checks_run.add("answerability_supported_by_evidence")
        if not usable and question.required_evidence:
            q_findings.append(
                Finding(
                    code="ANSWERABILITY_UNSUPPORTED",
                    severity=Severity.ERROR,
                    question_id=qid,
                    message=(
                        f"question is labelled '{q_audit.answerability}' but NONE of its "
                        f"required evidence is present in the corpus; it must be "
                        f"re-classified rather than silently scored as a failure"
                    ),
                    action=RecommendedAction.CLASSIFY_INSUFFICIENT_EVIDENCE,
                )
            )

        # -- question-term coverage -------------------------------------------
        #
        # Two very different things hide behind "a question term is not in the
        # evidence": the question's own phrasing (a synonym or a function word —
        # 'make', 'up', 'type', 'use' are all uncovered on real questions here)
        # and a genuine mismatch between the question and the passage pinned to
        # it. Raw term presence cannot tell them apart, so it is reported as an
        # OBSERVATION, and severity comes from IDF-weighted coverage of the
        # question's content terms — the rare, subject-naming words dominate and
        # the generic ones barely move it. Plain unweighted coverage is NOT used
        # as a fallback: the project measured it as non-discriminative for
        # relevance (see relevance.py), so where no corpus statistics exist this
        # check reports that it did not run.
        terms = extract_terms(question.question or "")
        union = " ".join(evidence_texts.get(cid, "") for cid in usable)
        union_terms = set(extract_terms(union)) if union else set()
        uncovered = [t for t in terms if t not in union_terms]
        if weights is not None and weights.available:
            uncovered = sorted(uncovered, key=lambda t: (-weights.idf(t), t))
        q_audit.question_terms = terms
        q_audit.uncovered_terms = uncovered

        weighted = weights is not None and weights.available and bool(usable)
        if weighted:
            checks_run.add("question_terms_covered_by_evidence")
            total = sum(weights.idf(t) for t in terms)
            covered_weight = sum(weights.idf(t) for t in terms if t in union_terms)
            coverage = (covered_weight / total) if total > 0 else 0.0
            q_audit.coverage = round(coverage, 6)
            # The observation is recorded for EVERY question with absent terms,
            # above or below the line, so the reader always sees which terms are
            # missing and how each is weighted. The line only decides whether the
            # question is additionally escalated for a human.
            if usable and uncovered:
                q_findings.append(
                    Finding(
                        code="QUESTION_TERMS_ABSENT_FROM_EVIDENCE",
                        severity=Severity.INFO,
                        question_id=qid,
                        message=(
                            f"terms {uncovered[:6]} do not appear literally in the pinned "
                            f"evidence ({uncovered[0]!r} weighs most heavily of them); weighted "
                            f"coverage is {coverage:.0%}. Recorded as an observation, not a "
                            f"defect: these are frequently the question's own phrasing rather "
                            f"than missing knowledge — but annotation wording must not assume "
                            f"knowledge absent from the cited evidence."
                        ),
                        detail={
                            "coverage": round(coverage, 6),
                            "uncovered_by_idf": uncovered[:10],
                        },
                    )
                )
            if usable and coverage < MIN_QUESTION_COVERAGE:
                q_findings.append(
                    Finding(
                        code="EVIDENCE_QUESTION_COVERAGE_LOW",
                        severity=Severity.WARNING,
                        question_id=qid,
                        message=(
                            f"the question's pinned evidence covers only {coverage:.0%} of its "
                            f"IDF-weighted content terms (triage line "
                            f"{MIN_QUESTION_COVERAGE:.0%}); most significant terms not present: "
                            f"{', '.join(uncovered[:6])}. The question as written may not be "
                            f"answered by the evidence pinned to it — CHECK BY READING, because "
                            f"a question can use a different word for something the evidence "
                            f"states plainly"
                        ),
                        action=RecommendedAction.REVIEW_QUESTION_EVIDENCE_MATCH,
                        detail={
                            "coverage": round(coverage, 6),
                            "threshold": MIN_QUESTION_COVERAGE,
                            "uncovered_by_idf": uncovered[:10],
                        },
                    )
                )
        else:
            checks_not_run.setdefault(
                "question_terms_covered_by_evidence",
                (
                    "no corpus term statistics are available for this knowledge base; plain "
                    "unweighted coverage is documented as non-discriminative, so evidence "
                    "coverage of the question was NOT assessed "
                    f"(uncovered terms observed: {uncovered[:8]})"
                )
                if uncovered
                else "no corpus term statistics are available for this knowledge base",
            )
            if usable and uncovered:
                q_findings.append(
                    Finding(
                        code="QUESTION_TERMS_ABSENT_FROM_EVIDENCE",
                        severity=Severity.INFO,
                        question_id=qid,
                        message=(
                            f"terms {uncovered[:6]} do not appear literally in the pinned "
                            f"evidence. Coverage was NOT weighted (no corpus statistics), so "
                            f"this audit cannot say whether the evidence answers the question."
                        ),
                        detail={"uncovered": uncovered[:10]},
                    )
                )

        # -- wording heuristics ------------------------------------------------
        checks_run.add("question_wording_heuristics")
        q_findings.extend(_wording_findings(qid, question.question, terms))

        # -- citation requirement consistency ---------------------------------
        checks_run.add("citation_requirement_consistency")
        req = question.citation_requirements
        if req.must_cite_all_required_evidence and not req.require_at_least_one_citation:
            q_findings.append(
                Finding(
                    code="CITATION_REQUIREMENTS_CONTRADICTORY",
                    severity=Severity.ERROR,
                    question_id=qid,
                    message=(
                        "citations are not required but every required chunk must be cited; "
                        "these cannot both hold"
                    ),
                    action=RecommendedAction.FIX_EVIDENCE_LINK,
                )
            )

        # -- authored labels: only when they exist -----------------------------
        annotation = effective_annotations.get(qid)
        q_audit.has_annotation = annotation is not None
        if annotation is None:
            q_findings.append(
                Finding(
                    code="ANSWER_LABEL_NOT_AUTHORED",
                    severity=Severity.WARNING,
                    question_id=qid,
                    message=(
                        "no reference answer has been authored for this question, so "
                        "correctness and key-point recall are UNKNOWN for it (not zero)"
                    ),
                    action=RecommendedAction.AUTHOR_MISSING_LABELS,
                )
            )
            for check in (
                "authored_key_points_supported_by_evidence",
                "annotation_reference_answer_grounded",
            ):
                checks_not_run.setdefault(check, "no annotations exist yet")
        else:
            checks_run.update(
                {"authored_key_points_supported_by_evidence", "annotation_reference_answer_grounded"}
            )
            cited = {
                ev.chunk_id: evidence_texts.get(ev.chunk_id, "")
                for ev in annotation.evidence
            }
            cited_text = " ".join(t for t in cited.values() if t)
            for point in annotation.key_points:
                q_audit.key_points_checked += 1
                verdict = (
                    entailment.evaluate(point, cited_text)
                    if cited_text
                    else None
                )
                if verdict is not None and verdict.judgement is SupportJudgement.SUPPORTED:
                    q_audit.key_points_supported += 1
                else:
                    q_audit.key_points_unsupported.append(point)
                    detail = verdict.detail if verdict is not None else "annotation cites no evidence text"
                    q_findings.append(
                        Finding(
                            code="KEY_POINT_UNSUPPORTED_BY_CITED_EVIDENCE",
                            severity=Severity.ERROR,
                            question_id=qid,
                            message=(
                                f"authored key point has no support in the evidence the "
                                f"annotation cites: {point[:160]!r} ({detail})"
                            ),
                            action=RecommendedAction.FIX_EVIDENCE_LINK,
                            detail={"key_point": point, "judgement": str(verdict.judgement) if verdict else "no_evidence"},
                        )
                    )
            if annotation.expected_answer.strip():
                verdict = entailment.evaluate(annotation.expected_answer, cited_text) if cited_text else None
                if verdict is None or verdict.judgement is not SupportJudgement.SUPPORTED:
                    q_findings.append(
                        Finding(
                            code="REFERENCE_ANSWER_UNSUPPORTED_BY_CITED_EVIDENCE",
                            severity=Severity.ERROR,
                            question_id=qid,
                            message=(
                                "the authored reference answer is not supported by the "
                                "evidence the annotation cites, so it may encode outside "
                                "knowledge: "
                                + (verdict.detail if verdict else "annotation cites no evidence text")
                            ),
                            action=RecommendedAction.FIX_EVIDENCE_LINK,
                        )
                    )

        q_findings.sort(key=lambda f: (_severity_rank(f.severity), f.code))
        q_audit.findings = q_findings
        q_audit.recommended_action = _recommend(q_findings, q_audit.has_annotation)
        per_question.append(q_audit)
        findings.extend(q_findings)

    # `per_question` is complete here; the pairwise pass below may add to each
    # question's findings and recomputes its recommendation.

    # -- pairwise near-duplicate detection ------------------------------------
    checks_run.add("near_duplicate_questions")
    by_id = {qa.question_id: qa for qa in per_question}
    term_sets = {q.question_id: set(extract_terms(q.question or "")) for q in questions}
    for i, first in enumerate(questions):
        for second in questions[i + 1 :]:
            jaccard, containment = _overlap(
                term_sets[first.question_id], term_sets[second.question_id]
            )
            if jaccard >= NEAR_DUPLICATE_JACCARD or containment >= NEAR_DUPLICATE_CONTAINMENT:
                finding = Finding(
                    code="NEAR_DUPLICATE_QUESTIONS",
                    severity=Severity.WARNING,
                    question_id=first.question_id,
                    also_question_id=second.question_id,
                    message=(
                        f"questions overlap heavily (jaccard {jaccard:.2f}, containment "
                        f"{containment:.2f}); one may be a duplicate of the other, which "
                        f"would silently weight the benchmark toward that fact"
                    ),
                    action=RecommendedAction.RESOLVE_DUPLICATE,
                    detail={"jaccard": round(jaccard, 4), "containment": round(containment, 4)},
                )
                findings.append(finding)
                # Attach it to BOTH questions as well, so a reviewer looking at one
                # question sees who it is suspected of duplicating — and so the
                # per-question recommendation can reflect it.
                for qid in (first.question_id, second.question_id):
                    entry = by_id.get(qid)
                    if entry is not None:
                        entry.findings.append(finding)

    # Recommendations are derived from each question's full finding list, which
    # the pairwise pass may have added to, so they are computed once here.
    for entry in per_question:
        entry.findings.sort(key=lambda f: (_severity_rank(f.severity), f.code))
        entry.recommended_action = _recommend(entry.findings, entry.has_annotation)

    # -- shared evidence sets -------------------------------------------------
    #
    # In a small corpus several questions legitimately resolve to the same
    # passage, so this is INFO, never a defect. It is reported because it has a
    # concrete consequence for the reader: if one chunk is the required evidence
    # for several questions, those questions' scores are not independent
    # observations, and a failure on that chunk depresses all of them at once.
    # That is a fact a human should know before treating an aggregate as broad
    # evidence of quality.
    checks_run.add("shared_required_evidence")
    groups: dict[tuple[str, ...], list[str]] = {}
    for question in questions:
        key = tuple(sorted(e.chunk_id for e in question.required_evidence))
        if key:
            groups.setdefault(key, []).append(question.question_id)
    for key, qids in sorted(groups.items()):
        if len(qids) < 2:
            continue
        findings.append(
            Finding(
                code="SHARED_REQUIRED_EVIDENCE",
                severity=Severity.INFO,
                question_id=qids[0],
                also_question_id=qids[1] if len(qids) > 1 else "",
                message=(
                    f"{len(qids)} questions pin the same evidence ({', '.join(key)}): "
                    f"{', '.join(qids)}. Their scores are NOT independent — a failure "
                    f"rooted in this passage affects all of them"
                ),
                detail={"chunk_ids": list(key), "question_ids": qids},
            )
        )

    # -- systematic display-provenance gap ----------------------------------
    if title_gaps:
        affected = sorted({qid for qid, _cid, _t in title_gaps})
        findings.append(
            Finding(
                code="EVIDENCE_SOURCE_TITLE_NOT_IN_CORPUS",
                severity=Severity.WARNING,
                question_id="",
                message=(
                    f"{len(title_gaps)} of {sum(len(q.required_evidence) for q in questions)} "
                    f"required-evidence slots ({len(affected)} questions) carry a source title "
                    f"on the BENCHMARK label that the live corpus chunk does not record "
                    f"(chunk.source_title is empty). The chunk id, document id and content "
                    f"hash all still agree, so citations remain valid, but any view that "
                    f"reads the title from the CHUNK will show a blank or a mismatch."
                ),
                action=RecommendedAction.FIX_EVIDENCE_LINK,
                detail={
                    "affected_question_ids": affected,
                    "example": {
                        "question_id": title_gaps[0][0],
                        "chunk_id": title_gaps[0][1],
                        "label_title": title_gaps[0][2],
                    },
                },
            )
        )

    errors = [f for f in findings if f.severity is Severity.ERROR]
    warnings = [f for f in findings if f.severity is Severity.WARNING]
    infos = [f for f in findings if f.severity is Severity.INFO]

    authored = len(effective_annotations)
    ready = not errors and authored == len(questions)

    if errors:
        verdict = (
            f"NOT READY: {len(errors)} error-level finding(s) must be resolved by a human "
            f"before this benchmark can be frozen. The audit does not repair them."
        )
    elif authored < len(questions):
        verdict = (
            f"STRUCTURALLY SOUND BUT INCOMPLETE: {len(questions) - authored} of "
            f"{len(questions)} questions have no authored reference answer, so correctness "
            f"is UNKNOWN. Authoring is the remaining work."
        )
    elif warnings:
        verdict = (
            f"READY SUBJECT TO {len(warnings)} WARNING(S): no error-level finding remains; a "
            f"human should work through the warnings before freezing."
        )
    else:
        verdict = "READY: every check that can run passed. Freezing is the human's decision."

    limitations = [
        "This audit is deterministic and lexical. It cannot establish MEANING: it cannot "
        "prove a question is ambiguous, that a reference answer is factually correct, or "
        "that two questions test different knowledge.",
        "Key-point support uses lexical term coverage (see entailment.py), not semantic "
        "entailment. A UNKNOWN judgement is reported as unsupported so that 'not checked' "
        "is never presented as 'fine'.",
        f"The coverage triage line ({MIN_QUESTION_COVERAGE:.0%}, IDF-weighted) is a "
        "judgement calibrated on the shipped 28-question benchmark, not a validated "
        "classifier. Its low end contains questions that genuinely look mis-pinned, but a "
        "question above the line is NOT thereby proven to be answerable by its evidence.",
        "Uncovered question terms are reported as OBSERVATIONS. Several are normally the "
        "question's own phrasing ('make', 'up', 'type', 'use'), so term absence alone is "
        "not evidence of a corpus gap and is never scored as one.",
        "Near-duplicate detection compares question TERMS only; two differently-worded "
        "questions testing the same fact are not detected.",
        "The audit reads the live corpus and the stored annotations. It never writes to "
        "either, so its findings must be acted on by a human.",
    ]

    return BenchmarkQualityAudit(
        kb_id=benchmark.kb_id,
        benchmark_name=benchmark.benchmark,
        benchmark_fingerprint=benchmark_fingerprint,
        benchmark_path=benchmark_path,
        questions=per_question,
        findings=findings,
        error_count=len(errors),
        warning_count=len(warnings),
        info_count=len(infos),
        ready_to_freeze=ready,
        verdict=verdict,
        checks_run=sorted(checks_run),
        checks_not_run=[f"{check} — {reason}" for check, reason in sorted(checks_not_run.items())],
        limitations=limitations,
    )


def _recommend(findings: list[Finding], has_annotation: bool) -> RecommendedAction:
    """One suggested action per question, derived from its own findings.

    Ordered by what most likely unblocks the reviewer. Suggestions only: the
    audit never changes a question's state.
    """
    codes = {f.code for f in findings}
    if codes & {"EVIDENCE_MISSING_FROM_CORPUS", "ANSWERABILITY_UNSUPPORTED", "NO_REQUIRED_EVIDENCE"}:
        return RecommendedAction.CLASSIFY_INSUFFICIENT_EVIDENCE
    if codes & {"EVIDENCE_HASH_MISMATCH", "EVIDENCE_DOCUMENT_MISMATCH"}:
        return RecommendedAction.FIX_EVIDENCE_LINK
    if "EVIDENCE_TEXT_TRIVIAL" in codes:
        return RecommendedAction.CLASSIFY_INSUFFICIENT_EVIDENCE
    if codes & {"KEY_POINT_UNSUPPORTED_BY_CITED_EVIDENCE", "REFERENCE_ANSWER_UNSUPPORTED_BY_CITED_EVIDENCE"}:
        return RecommendedAction.FIX_EVIDENCE_LINK
    if "EVIDENCE_QUESTION_COVERAGE_LOW" in codes:
        return RecommendedAction.REVIEW_QUESTION_EVIDENCE_MATCH
    if "NEAR_DUPLICATE_QUESTIONS" in codes:
        return RecommendedAction.RESOLVE_DUPLICATE
    if codes & {"QUESTION_MULTIPLE_INTERROGATIVES", "QUESTION_DANGLING_REFERENCE", "QUESTION_UNDER_SPECIFIED", "QUESTION_TOO_FEW_CONTENT_TERMS"}:
        return RecommendedAction.REWRITE_QUESTION
    if not has_annotation:
        return RecommendedAction.AUTHOR_MISSING_LABELS
    return RecommendedAction.NONE


__all__ = [
    "MIN_EVIDENCE_CHARS",
    "NEAR_DUPLICATE_CONTAINMENT",
    "NEAR_DUPLICATE_JACCARD",
    "BenchmarkQualityAudit",
    "Finding",
    "QuestionAudit",
    "RecommendedAction",
    "Severity",
    "run_quality_audit",
]
