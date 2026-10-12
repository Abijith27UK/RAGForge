"""Held-out candidate validation — separation, evidence and label sufficiency (V14).

## What this module is for

An agent can draft 24 plausible-looking questions in a minute. The reason that is
dangerous is not that the questions look wrong — it is that the failure modes are
invisible without a check:

* A "new" question whose expected chunk is one the development benchmark already
  uses. Nothing about the wording reveals this, and the resulting "held-out"
  result would be a re-measurement of data the configuration was chosen on.
* A question that is a paraphrase of a development question with a different id.
  The request warns about exactly this: different ids are not independence.
* An evidence label pointing at a chunk that no longer exists, or exists in the
  corpus but has no vector, so every retriever "fails" on a question that was
  never answerable in the first place. Distinguishing these two is the difference
  between a corpus gap and a retrieval failure.
* An evidence label that is *insufficient*: the chunk is real and relevant but
  does not actually contain the answer. This is the subtlest one, and it is why
  every candidate carries a verbatim ``evidence_quote`` — the validator checks the
  quote appears in the named chunk, which converts "the evidence is sufficient"
  from the author's opinion into a text-verifiable claim.

## Severity is not decoration

``error`` means the artifact must not be frozen as written: overlap with the
development set, unresolvable evidence, a fabricated quote. ``warning`` means it
can freeze but the reader should know (a near-paraphrase, an unspecified
subdomain, a question nobody has reviewed). ``info`` records state.

Only findings attached to SURVIVING candidates can block a freeze. A rejected
candidate's problems are reported for the audit trail but cannot hold the
instrument hostage — otherwise rejecting a bad question would not remove it.

## Errors reported for rejected candidates

Kept, not suppressed. A rejection rate is a real finding about the drafting
method, and the reasons are the evidence for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from app.services.answer_eval.heldout_benchmark import (
    DEFAULT_DEV_BENCHMARK_PATH,
    CandidateStatus,
    HeldOutCandidate,
    HeldOutCandidateSet,
    HeldOutValidationError,
    load_candidate_set,
)

#: Below this token-set Jaccard similarity, two questions are not considered
#: paraphrases of each other. Chosen conservatively low: a false warning costs a
#: sentence of reading, a false negative costs an independence claim.
PARAPHRASE_TOKEN_JACCARD = 0.60

#: Character 4-gram Jaccard catches reordered/near-verbatim restatements that the
#: token measure underrates. Also deliberately low.
PARAPHRASE_CHAR_JACCARD = 0.55

_STOPWORDS = frozenset(
    """
    a an the is are was were be been being of in on at to for from by with without
    and or but not no does do did how what which who whom whose when where why
    that this these those it its as into than then there here their they them
    between during about after before over under out up down off
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalize_tokens(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed, for paraphrase comparison.

    Stopwords are dropped because a question like "What is an engine?" and "What
    is a brake?" share almost all of their *words*; the content words are what
    decides whether two questions test the same knowledge.
    """
    words = _TOKEN_RE.findall((text or "").lower())
    return [w for w in words if w not in _STOPWORDS]


def normalize_text(text: str) -> str:
    """Whitespace-collapsed lowercase text, for verbatim-quote containment.

    Chunk text is extracted from HTML and is full of "\n\n" and stray spaces, so
    a byte-exact substring test would fail on formatting rather than on content.
    """
    return _WS_RE.sub(" ", (text or "").replace("\u00a0", " ").strip().lower())


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(normalize_tokens(a)), set(normalize_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def char_ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    def grams(text: str) -> set[str]:
        s = normalize_text(text)
        return {s[i : i + n] for i in range(max(len(s) - n + 1, 0))} or ({s} if s else set())

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


class Finding(BaseModel):
    """One validation finding."""

    severity: str  # "error" | "warning" | "info"
    code: str
    detail: str
    action: str
    question_id: str = ""
    #: Whether this finding belongs to a candidate that will enter the instrument.
    applies_to_frozen: bool = True


@dataclass
class EvidenceProbe:
    """The KB-scoped facts the validator needs, resolved once by the caller.

    `index_chunk_ids` is deliberately allowed to be None. "We could not read the
    index" and "the evidence is not in the index" are different statements, and a
    validator that collapsed them would manufacture retrieval failures.
    """

    chunk_rows: dict[str, Any] = field(default_factory=dict)
    document_ids: set[str] = field(default_factory=set)
    index_chunk_ids: set[str] | None = None
    index_note: str = ""

    @property
    def index_verified(self) -> bool:
        return self.index_chunk_ids is not None


class QuestionValidation(BaseModel):
    """The per-question verdict, so the UI can show one row per candidate."""

    question_id: str
    status: str
    subdomain: str = ""
    question: str = ""
    expected_chunk_ids: list[str] = Field(default_factory=list)
    evidence_resolved: int = 0
    evidence_total: int = 0
    evidence_in_index: int | None = None
    needs_review: bool = True
    findings: list[Finding] = Field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(f.severity == "error" and f.applies_to_frozen for f in self.findings)


class HeldOutValidationReport(BaseModel):
    """Everything the validator established, and everything it could not."""

    benchmark: str
    kb_id: str = ""
    candidate_fingerprint: str = ""
    question_count: int = 0
    surviving_count: int = 0
    rejected_count: int = 0
    checks_run: list[str] = Field(default_factory=list)
    checks_not_run: list[dict[str, str]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: EVERY finding, structured, including the SET-LEVEL ones (duplicate ids,
    #: duplicate question text) that belong to no single question. Errors and
    #: warnings are the same findings rendered as sentences; this list is what a
    #: UI or a test can filter without parsing prose.
    findings: list[Finding] = Field(default_factory=list)
    questions: list[QuestionValidation] = Field(default_factory=list)
    separation: dict[str, Any] = Field(default_factory=dict)
    composition: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    verdict: str = ""
    limitations: list[str] = Field(default_factory=list)

    @property
    def freezeable(self) -> bool:
        return not self.errors


# ----------------------------------------------------------------------------
# separation from the development benchmark
# ----------------------------------------------------------------------------


def _load_dev_questions(path: str | Any) -> list[dict[str, Any]]:
    """Read the development benchmark read-only.

    Imported lazily from the retrieval runner so this module stays importable in
    tests that never touch a repository, and so there is exactly one reader of the
    development benchmark in the codebase.
    """
    from app.services.answer_eval.retrieval_diagnostic_runner import (
        load_retrieval_benchmark,
        resolve_benchmark_path,
    )

    bench = load_retrieval_benchmark(resolve_benchmark_path(path))
    return [
        {
            "id": q.id,
            "subdomain": q.subdomain,
            "question": q.question,
            "expected_chunk_ids": list(q.expected_chunk_ids),
            "expected_document_ids": list(q.expected_document_ids),
        }
        for q in bench.questions
    ]


def measure_separation(
    candidates: list[HeldOutCandidate],
    dev_questions: list[dict[str, Any]],
    *,
    token_threshold: float = PARAPHRASE_TOKEN_JACCARD,
    char_threshold: float = PARAPHRASE_CHAR_JACCARD,
) -> dict[str, Any]:
    """Measure how separate the held-out questions are from the development ones.

    Returns, per surviving candidate, the shared expected chunk ids, the nearest
    development question and both similarity scores. The point is that
    "independent" becomes a number a reader can check, including when it is bad.
    """
    dev_chunk_owner: dict[str, list[str]] = {}
    for dq in dev_questions:
        for cid in dq["expected_chunk_ids"]:
            dev_chunk_owner.setdefault(cid, []).append(dq["id"])
    dev_documents = sorted(
        {d for dq in dev_questions for d in dq["expected_document_ids"]}
    )

    per_question: dict[str, Any] = {}
    max_token = 0.0
    max_char = 0.0
    shared_any = 0
    for cand in candidates:
        shared = sorted(cid for cid in cand.expected_chunk_ids if cid in dev_chunk_owner)
        if shared:
            shared_any += 1
        nearest_id = ""
        nearest_token = 0.0
        nearest_char = 0.0
        nearest_question = ""
        for dq in dev_questions:
            t = token_jaccard(cand.question, dq["question"])
            c = char_ngram_jaccard(cand.question, dq["question"])
            if max(t, c) > max(nearest_token, nearest_char):
                nearest_id, nearest_token, nearest_char = dq["id"], t, c
                nearest_question = dq["question"]
        max_token = max(max_token, nearest_token)
        max_char = max(max_char, nearest_char)
        per_question[cand.id] = {
            "shared_expected_chunk_ids": shared,
            "shared_with_development_questions": sorted(
                {q for cid in shared for q in dev_chunk_owner[cid]}
            ),
            "nearest_development_question_id": nearest_id,
            "nearest_development_question": nearest_question,
            "token_jaccard": round(nearest_token, 4),
            "char_ngram_jaccard": round(nearest_char, 4),
            "paraphrase_suspected": (
                nearest_token >= token_threshold or nearest_char >= char_threshold
            ),
        }

    return {
        "development_benchmark_questions": len(dev_questions),
        "development_benchmark_chunk_ids": len(dev_chunk_owner),
        "development_document_ids": dev_documents,
        "thresholds": {"token_jaccard": token_threshold, "char_ngram_jaccard": char_threshold},
        "candidates_sharing_expected_chunk_with_development": shared_any,
        "max_token_jaccard_vs_development": round(max_token, 4),
        "max_char_ngram_jaccard_vs_development": round(max_char, 4),
        "per_question": per_question,
        # Stated as a limitation, not hidden: these are lexical measures. They can
        # show that two questions reuse the same evidence or nearly the same words;
        # they cannot prove two differently-worded questions test different knowledge.
        "note": (
            "Lexical separation only. A low similarity score does not prove a "
            "question tests different knowledge, and a zero shared-chunk count does "
            "not prove the corpus region is new. Both measures make overlap "
            "detectable; neither makes independence provable."
        ),
    }


# ----------------------------------------------------------------------------
# the validator
# ----------------------------------------------------------------------------


def validate_candidates(
    candidates: HeldOutCandidateSet,
    probe: EvidenceProbe,
    *,
    dev_questions: list[dict[str, Any]] | None = None,
    dev_benchmark_path: str = DEFAULT_DEV_BENCHMARK_PATH,
) -> HeldOutValidationReport:
    """Validate every candidate against the corpus, the index and the dev set."""
    findings: list[Finding] = []
    checks_run: list[str] = []
    checks_not_run: list[dict[str, str]] = []
    limitations: list[str] = []

    if dev_questions is None:
        dev_questions = _load_dev_questions(dev_benchmark_path)
    checks_run.append("separation.vs_development_benchmark")
    separation = measure_separation(candidates.surviving, dev_questions)

    # -- set-level checks ---------------------------------------------------
    seen_ids: dict[str, int] = {}
    for cand in candidates.candidates:
        seen_ids[cand.id] = seen_ids.get(cand.id, 0) + 1
    for qid, count in sorted(seen_ids.items()):
        if count > 1:
            findings.append(
                Finding(
                    severity="error",
                    code="DUPLICATE_QUESTION_ID",
                    question_id=qid,
                    detail=f"question id {qid!r} appears {count} times",
                    action="give every question a unique id before freezing",
                )
            )
    checks_run.append("uniqueness.question_ids")

    normalized_texts: dict[str, str] = {}
    for cand in candidates.surviving:
        key = " ".join(normalize_tokens(cand.question))
        owner = normalized_texts.get(key)
        if owner and owner != cand.id:
            findings.append(
                Finding(
                    severity="error",
                    code="DUPLICATE_QUESTION_TEXT",
                    question_id=cand.id,
                    detail=(
                        f"question {cand.id!r} is word-for-word identical (after "
                        f"normalization) to {owner!r}"
                    ),
                    action="keep one of the two, or rewrite one to test different knowledge",
                )
            )
        normalized_texts[key] = cand.id
    checks_run.append("uniqueness.question_text")

    if probe.index_verified:
        checks_run.append("evidence.present_in_index")
    else:
        checks_not_run.append(
            {
                "check": "evidence.present_in_index",
                "reason": probe.index_note
                or "the vector index could not be enumerated for this run",
            }
        )
        limitations.append(
            "index presence of the expected evidence was NOT verified: a question "
            "reported as resolvable here might still have no vector, in which case "
            "every retriever would fail on it for a reason unrelated to retrieval"
        )
    checks_run.append("evidence.present_in_corpus")
    checks_run.append("evidence.quote_matches_chunk")
    checks_run.append("labels.follow_benchmark_schema")
    checks_run.append("provenance.complete")
    checks_run.append("review.state_recorded")

    # -- per-question -------------------------------------------------------
    per_question: list[QuestionValidation] = []
    for cand in candidates.candidates:
        q_findings: list[Finding] = []
        survives = cand.status.survives

        def add(severity: str, code: str, detail: str, action: str) -> None:
            q_findings.append(
                Finding(
                    severity=severity,
                    code=code,
                    detail=detail,
                    action=action,
                    question_id=cand.id,
                    applies_to_frozen=survives,
                )
            )

        # 1. Label schema: binary relevance needs at least one necessary chunk.
        if not cand.expected_chunk_ids:
            add(
                "error",
                "MISSING_EXPECTED_CHUNK",
                "the question carries no expected chunk id, so no retrieval metric "
                "can be computed from it",
                "select the chunk that answers the question, or reject the question",
            )

        # 2. Evidence resolution, distinguishing corpus absence from index absence.
        in_corpus = [cid for cid in cand.expected_chunk_ids if cid in probe.chunk_rows]
        missing_corpus = [cid for cid in cand.expected_chunk_ids if cid not in probe.chunk_rows]
        for cid in missing_corpus:
            add(
                "error",
                "EVIDENCE_MISSING_FROM_CORPUS",
                f"expected chunk {cid} is not a chunk of this knowledge base",
                "re-select the evidence from the live corpus, or reject the question",
            )
        in_index: int | None = None
        if probe.index_verified:
            in_index = sum(1 for cid in in_corpus if cid in (probe.index_chunk_ids or set()))
            for cid in in_corpus:
                if cid not in (probe.index_chunk_ids or set()):
                    add(
                        "error",
                        "EVIDENCE_MISSING_FROM_INDEX",
                        f"expected chunk {cid} is in the corpus but has no vector in "
                        "the index, so no retriever can return it",
                        "re-index the document, or reject the question as unanswerable "
                        "by retrieval in the current state of the KB",
                    )

        # 3. Expected documents must resolve and must agree with the chunks.
        for doc_id in cand.expected_document_ids:
            if doc_id not in probe.document_ids:
                add(
                    "error",
                    "EXPECTED_DOCUMENT_MISSING",
                    f"expected document {doc_id} is not in this knowledge base",
                    "correct or remove the document id",
                )
        for cid in in_corpus:
            row = probe.chunk_rows[cid]
            row_doc = getattr(row, "document_id", None)
            if cand.expected_document_ids and row_doc not in cand.expected_document_ids:
                add(
                    "error",
                    "CHUNK_DOCUMENT_MISMATCH",
                    f"expected chunk {cid} belongs to document {row_doc}, which is not "
                    f"among the question's expected_document_ids "
                    f"({', '.join(cand.expected_document_ids)})",
                    "add the chunk's real document id, or re-select the evidence",
                )

        # 4. Evidence sufficiency: the quoted answering sentence must be IN the chunk.
        if cand.evidence_quote.strip():
            quote = normalize_text(cand.evidence_quote)
            # The quote may legitimately span several chunks when the answer is
            # split; matching any single expected chunk is sufficient.
            if not any(
                quote in normalize_text(getattr(probe.chunk_rows[cid], "text", ""))
                for cid in in_corpus
            ):
                add(
                    "error",
                    "EVIDENCE_QUOTE_NOT_FOUND_IN_CHUNK",
                    "the recorded evidence_quote does not appear in any of the named "
                    "chunks, so the label's claim that the chunk answers the question "
                    "is unverified",
                    "quote the answering sentence verbatim, or re-select the chunk",
                )
        else:
            add(
                "warning",
                "EVIDENCE_QUOTE_MISSING",
                "no verbatim evidence quote is recorded, so sufficiency of the label "
                "cannot be verified from text",
                "record the answering sentence; this is the only machine check that "
                "the chunk actually answers the question",
            )

        # 5. Provenance and metadata.
        if not cand.provenance.strip():
            add(
                "error",
                "PROVENANCE_MISSING",
                "no provenance recorded",
                "record the document, section and selection date",
            )
        if not cand.draft_origin.strip():
            add(
                "warning",
                "DRAFT_ORIGIN_MISSING",
                "no drafting origin recorded",
                "record how the draft was produced so the method is auditable per question",
            )
        if not cand.coverage_rationale.strip():
            add(
                "warning",
                "COVERAGE_RATIONALE_MISSING",
                "no coverage rationale recorded",
                "state which concept this question adds that the development set does not cover",
            )
        if not cand.subdomain.strip():
            add(
                "warning",
                "SUBDOMAIN_UNSPECIFIED",
                "no subdomain recorded",
                "assign a subdomain so composition is comparable with the development set",
            )

        # 6. Separation findings.
        sep = separation["per_question"].get(cand.id, {})
        shared = sep.get("shared_expected_chunk_ids") or []
        if shared:
            add(
                "error",
                "SHARED_EVIDENCE_WITH_DEVELOPMENT",
                "expected chunk(s) "
                + ", ".join(shared)
                + " are also expected evidence for development question(s) "
                + ", ".join(sep.get("shared_with_development_questions") or [])
                + "; reusing the evidence makes this question a re-measurement of the "
                "data the candidate configurations were chosen on",
                "select different evidence, or drop the question",
            )
        if sep.get("paraphrase_suspected"):
            add(
                "warning",
                "PARAPHRASE_SUSPECTED",
                f"nearest development question {sep.get('nearest_development_question_id')!r} "
                f"scores token_jaccard={sep.get('token_jaccard')} and "
                f"char_ngram_jaccard={sep.get('char_ngram_jaccard')}, above the "
                "paraphrase thresholds",
                "confirm this tests different knowledge, or rewrite it",
            )

        # 7. Review state — reported, never assumed.
        if survives and not cand.status.is_reviewed:
            add(
                "info",
                "REVIEW_PENDING",
                "no human decision recorded",
                "approve, edit or reject it in the review view; until then the "
                "instrument can only freeze as PROVISIONAL (exploratory results)",
            )
        if not survives:
            add(
                "info",
                "REJECTED",
                f"rejected by {cand.review.reviewer if cand.review else 'nobody'}"
                + (f": {cand.review.notes}" if cand.review and cand.review.notes else ""),
                "no action; kept in the artifact so the rejection rate stays reportable",
            )

        per_question.append(
            QuestionValidation(
                question_id=cand.id,
                status=cand.status.value,
                subdomain=cand.subdomain,
                question=cand.question,
                expected_chunk_ids=list(cand.expected_chunk_ids),
                evidence_resolved=len(in_corpus),
                evidence_total=len(cand.expected_chunk_ids),
                evidence_in_index=in_index,
                needs_review=survives and not cand.status.is_reviewed,
                findings=q_findings,
            )
        )
        findings.extend(q_findings)

    errors = [f"{f.question_id}: {f.detail}" for f in findings if f.severity == "error" and f.applies_to_frozen]
    warnings = [
        f"{f.question_id}: {f.detail}" for f in findings if f.severity == "warning" and f.applies_to_frozen
    ]
    rejected_errors = [
        f"{f.question_id}: {f.detail}"
        for f in findings
        if f.severity == "error" and not f.applies_to_frozen
    ]

    completeness = candidates.review_completeness()
    failing = [q.question_id for q in per_question if q.has_error]
    if errors:
        verdict = (
            f"NOT FREEZEABLE — {len(errors)} error(s) across {len(failing)} question(s). "
            "Fix the evidence or reject the affected questions."
        )
    elif not completeness["complete"]:
        verdict = (
            f"FREEZEABLE AS PROVISIONAL — no errors, but {completeness['candidates_unreviewed']} "
            "question(s) carry no human decision, which caps every result at EXPLORATORY."
        )
    else:
        verdict = (
            "FREEZEABLE AS VALIDATED — no errors and every surviving question carries "
            "a human decision."
        )

    if rejected_errors:
        limitations.append(
            f"{len(rejected_errors)} validation error(s) belong to REJECTED candidates "
            "and therefore do not block the freeze; they are reported because the "
            "rejection rate is itself a finding about the drafting method"
        )

    return HeldOutValidationReport(
        benchmark=candidates.benchmark,
        kb_id=candidates.kb_id,
        candidate_fingerprint=candidates.fingerprint(),
        question_count=len(candidates.candidates),
        surviving_count=len(candidates.surviving),
        rejected_count=len(candidates.rejected),
        checks_run=checks_run,
        checks_not_run=checks_not_run,
        errors=errors,
        warnings=warnings,
        findings=findings,
        questions=per_question,
        separation=separation,
        composition={
            "subdomains": candidates.subdomain_counts(),
            "query_types": candidates.query_type_counts(),
            "difficulty": candidates.difficulty_counts(),
            "documents_covered": sorted(
                {d for c in candidates.surviving for d in c.expected_document_ids}
            ),
            "documents_not_covered_by_development": sorted(
                set(d for c in candidates.surviving for d in c.expected_document_ids)
                - set(separation["development_document_ids"])
            ),
        },
        review=completeness,
        verdict=verdict,
        limitations=limitations,
    )


def probe_evidence(
    repo: Any,
    kb_id: str,
    candidates: HeldOutCandidateSet,
    *,
    backend: str | None = None,
    vector_store_factory: Any | None = None,
) -> EvidenceProbe:
    """Resolve the corpus, document and index facts the validator needs.

    Read-only. The index enumeration is attempted through the same factory the
    diagnostics engine uses, so "the index could not be read" is reported exactly
    the way it is reported during a run.
    """
    expected_ids = sorted({c for cand in candidates.candidates for c in cand.expected_chunk_ids})
    chunk_rows = {ch.id: ch for ch in repo.get_chunks_by_ids(kb_id, expected_ids)}
    document_ids = {d.id for d in repo.list_documents(kb_id)}

    index_chunk_ids: set[str] | None = None
    index_note = ""
    try:
        from app.services.vector_store import factory as vector_factory

        factory = vector_store_factory or vector_factory
        store = factory.create_vector_store(_settings_for(repo), backend=backend)
        index_chunk_ids = set(store.all_point_payloads(kb_id).keys())
    except NotImplementedError:
        index_note = (
            "the configured vector backend cannot enumerate points, so index presence "
            "of the expected evidence is UNVERIFIED"
        )
    except Exception as exc:  # network/collection errors are not absence
        index_note = (
            f"the vector index could not be read ({exc}), so index presence of the "
            "expected evidence is UNVERIFIED"
        )

    return EvidenceProbe(
        chunk_rows=chunk_rows,
        document_ids=document_ids,
        index_chunk_ids=index_chunk_ids,
        index_note=index_note,
    )


def _settings_for(repo: Any) -> Any:
    """Best-effort settings handle for the vector-store factory.

    The repository does not carry settings, so the configured defaults are used.
    Tests that need a different store inject `vector_store_factory` instead.
    """
    from app.config import get_settings

    return get_settings()


__all__ = [
    "EvidenceProbe",
    "Finding",
    "HeldOutValidationReport",
    "PARAPHRASE_CHAR_JACCARD",
    "PARAPHRASE_TOKEN_JACCARD",
    "QuestionValidation",
    "char_ngram_jaccard",
    "load_candidate_set",
    "measure_separation",
    "normalize_text",
    "normalize_tokens",
    "probe_evidence",
    "token_jaccard",
    "validate_candidates",
]
