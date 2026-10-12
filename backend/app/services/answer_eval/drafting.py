"""V11 corpus-grounded DRAFT assist for human ground-truth authoring.

WHY THIS EXISTS
---------------
V10 built the whole human-ground-truth workflow (authoring, review, approval
gate, immutable freeze, reference evaluator) and it works — but the official
answer benchmark still has 0/28 questions reviewed, so correctness is UNKNOWN.
The bottleneck is no longer infrastructure: it is that a human has to read
28 evidence dossiers and write 28 reference answers plus their key points from
a blank form.

This module reduces that work without touching the integrity rule that made
V10 trustworthy. It does NOT author ground truth. It produces a **DRAFT**:
a machine-assembled starting point built ONLY from verbatim sentences that
already exist in the question's required corpus evidence.

THE ONE RULE
------------
A draft may never contain a fact that is not literally present in the cited
chunk text. That is not a convention here, it is enforced structurally:
`_assert_verbatim` raises `DraftingError` if any key point or any part of the
assembled reference answer is not a verbatim substring of the chunk it cites.
An extractive generator cannot hallucinate, and if someone later swaps in a
generative one, this guard fails loudly instead of silently shipping invented
facts into a benchmark.

WHAT THIS IS NOT
----------------
* NOT ground truth. Every suggestion carries `is_draft = True`, a conspicuous
  `label`, `requires_human_review = True`, and a caveat saying so. The name is
  `DraftSuggestion`, deliberately not `GroundTruthAnnotation`; the two are
  different types, stored in different tables, and nothing in
  `ground_truth.py` can read a suggestion. There is no code path from a
  suggestion to an APPROVED state — approval requires a named human to POST an
  annotation through the V10 authoring endpoint.
* NOT an LLM. `method` names the deterministic algorithm and `generator`
  names the tool, so a draft is reproducible from the same corpus and its
  origin can never be mistaken for a human's. No provider, no key, no network.
* NOT a scoring input. The evaluator never sees a suggestion. It only exists to
  be reviewed, edited, and explicitly re-submitted by a human as their own work
  — or discarded.

HONEST FAILURE
--------------
When the question's content terms are not covered by the evidence, this says so
instead of padding the draft: `status = INSUFFICIENT_EVIDENCE` with the
uncovered terms listed. That is the signal a human needs in order to classify a
question `insufficient_evidence` rather than invent an answer from outside
knowledge. Same for `PARTIAL`: coverage exists but some question terms never
appear in the evidence at all.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.services.answer_eval.benchmark import AnswerBenchmarkQuestion, RequiredEvidence
from app.services.answer_eval.relevance import TermWeights
from app.services.answering.query_processor import extract_terms
from app.utils.ids import new_id

#: Machine-readable identity of the algorithm. Recorded on every suggestion so a
#: draft can always be traced back to the exact procedure that produced it.
METHOD_EXTRACTIVE = "extractive-verbatim-sentence-selection-v1"

#: Identity of the generator. Never blank: an unattributed draft is
#: indistinguishable from a human's, which is the failure mode this whole
#: module is designed to prevent.
GENERATOR = "RAGForge V11 draft assist (deterministic extractive; no LLM, no network)"

#: Shown verbatim in the UI and stored on every suggestion.
DRAFT_LABEL = "DRAFT — MACHINE-GENERATED SUGGESTION, NOT HUMAN GROUND TRUTH"

#: Below this IDF-weighted question-term coverage a sentence is not a candidate
#: at all. A published policy constant rather than a tuned knob: it exists to
#: stop an unrelated sentence being offered as the "answer" just because it
#: shares one common word with the question.
MIN_SELECT_COVERAGE = 0.20

#: Deterministic structural bonus added when a sentence has the shape the
#: question is asking for (see `_question_kind`). Small on purpose: it breaks
#: ties between two sentences that cover the question equally well, and is
#: never enough to select a sentence that fails `MIN_SELECT_COVERAGE`.
STRUCTURE_BONUS = 0.10

#: Maximum key points offered. The benchmark's questions each pin one or two
#: required chunks, so this is a ceiling for authoring convenience, not a claim
#: about how many facts a complete answer needs.
MAX_KEY_POINTS = 4

#: IDF-weighted share of the question's content terms that the pinned evidence
#: must contain before a draft is called complete. This is a TRIAGE threshold for
#: directing human attention, not a validated classifier, and it is published
#: with its measured basis rather than presented as a policy derived from
#: theory.
#:
#: MEASURED on the shipped 28-question benchmark (corpus IDF from the KB's own
#: BM25 lexical index, n_docs=812): the IDF-weighted coverage of each question's
#: own pinned evidence, sorted, is
#:   0.048, 0.111, 0.338, 0.390, 0.425, 0.472, 0.477, 0.482, 0.488, 0.500,
#:   0.503, 0.504, 0.562, 0.613, 0.640, 0.698, 0.715, 0.734, 0.738, 0.767,
#:   0.777, 0.872, 0.889, 0.935, 1.000, 1.000, 1.000, 1.000.
#: The low end contains questions that genuinely look mis-pinned (auto-eng-024
#: asks about roll/pitch/yaw while its evidence never says pitch or yaw;
#: auto-eng-028's evidence does not contain 'recovers' or 'wasted' at all), and
#: 0.45 draws a tight line under them: 5 of 28 questions, listed for a human.
#: Higher values start flagging questions whose only 'missing' terms are
#: function words, which is why it is not raised. The margin here is a judgement
#: on one small benchmark, and the report says so.
MIN_QUESTION_COVERAGE = 0.45

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_QUOTED_RE = re.compile(r"['\u2018\u2019]([^'\u2018\u2019]{15,})['\u2018\u2019]")
_WS_RE = re.compile(r"\s+")

#: The corpus text is parsed web/Wikipedia content and carries layout artefacts:
#: line breaks land mid-phrase and punctuation is left on its own line ("...form
#: of\nenergy\n."). Collapsing whitespace alone would then produce "energy ." —
#: which is NOT equal to the same sentence as a human reads it, so the
#: quote-agreement signal reported a false mismatch on 23 of 28 questions.
#: Joining a space to the punctuation it precedes repairs the layout artefact and
#: is applied identically to both sides of every comparison, so it cannot hide a
#: real textual difference.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?%)\]])")


class DraftingError(RuntimeError):
    """A draft invariant was violated (chiefly: non-verbatim content)."""


class DraftStatus(str, Enum):
    """How much of the question the evidence could actually support."""

    #: Every question content term appears somewhere in the evidence and at
    #: least one sentence was selected.
    DRAFT_READY = "draft_ready"
    #: A usable sentence was found, but the evidence does not cover the question
    #: as written (measured by IDF-weighted coverage where corpus statistics
    #: exist). The draft is a starting point only.
    PARTIAL = "partial"
    #: No sentence in the evidence covers enough of the question. No draft is
    #: offered; the human writes it or classifies the question as
    #: insufficient_evidence.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DraftKeyPoint(BaseModel):
    """One candidate key point: a verbatim corpus sentence, with its origin.

    `text` is guaranteed by `_assert_verbatim` to be an exact substring of the
    chunk named by `source_chunk_id`. A reviewer can therefore verify any key
    point by searching the evidence panel for it.
    """

    text: str
    source_chunk_id: str
    source_sentence_index: int = Field(
        description="0-based index of this sentence within the cited chunk's text."
    )
    score: float = 0.0
    coverage: float = Field(
        default=0.0,
        description="IDF-weighted share of the question's content terms present in this sentence.",
    )
    method: str = METHOD_EXTRACTIVE


class DraftEvidence(BaseModel):
    """The corpus evidence the draft was assembled from (verbatim origin)."""

    chunk_id: str
    found: bool = False
    document_id: str | None = None
    source_title: str | None = None
    section: str | None = None
    content_hash: str = ""
    char_count: int = 0
    sentence_count: int = 0
    problems: list[str] = Field(default_factory=list)


class DraftSuggestion(BaseModel):
    """A machine-generated starting point for ONE question. Never ground truth.

    Stored append-only in its own table (`benchmark_draft_suggestions`), scoped
    by (kb_id, benchmark_name, benchmark_fingerprint, question_id) exactly like
    annotations and reviews, so a draft is never applied across benchmarks.
    """

    suggestion_id: str
    kb_id: str
    benchmark_name: str = ""
    benchmark_fingerprint: str = ""
    question_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # -- provenance of the DRAFT ITSELF (not of the answer) -------------------
    method: str = METHOD_EXTRACTIVE
    generator: str = GENERATOR
    is_draft: bool = True
    label: str = DRAFT_LABEL
    requires_human_review: bool = True

    status: DraftStatus = DraftStatus.INSUFFICIENT_EVIDENCE
    reason: str = ""
    caveats: list[str] = Field(default_factory=list)

    question: str = ""
    subdomain: str = ""

    expected_answer_draft: str = Field(
        default="",
        description="Verbatim sentences joined in evidence order. Empty when nothing qualified.",
    )
    key_points_draft: list[DraftKeyPoint] = Field(default_factory=list)
    evidence: list[DraftEvidence] = Field(default_factory=list)

    question_terms: list[str] = Field(default_factory=list)
    covered_terms: list[str] = Field(default_factory=list)
    uncovered_terms: list[str] = Field(
        default_factory=list,
        description="Question content terms absent from ALL of the evidence, ordered by "
        "corpus IDF (most significant first). NOT a verdict: several of these are "
        "function words or the question's own phrasing ('make', 'up', 'type', 'use'), "
        "which is exactly why severity is decided by weighted coverage instead.",
    )
    coverage: float = 0.0
    coverage_measured: bool = Field(
        default=False,
        description="False when the KB has no corpus statistics: plain unweighted coverage "
        "is documented as non-discriminative (see relevance.py), so it is reported as "
        "unavailable rather than used as if it meant something.",
    )
    coverage_method: str = ""

    benchmark_quote: str = Field(
        default="",
        description="The sentence a human previously recorded in the artifact's provenance.",
    )
    agrees_with_benchmark_quote: bool | None = Field(
        default=None,
        description="Did the deterministic selector independently reproduce the sentence "
        "the human picked earlier? True is a sanity signal, False is a prompt to look "
        "closely, None means the artifact recorded no quote to compare against.",
    )


# ---------------------------------------------------------------------------
# deterministic selection
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    """Whitespace-normalised view of text: the form used for ALL comparisons.

    Deliberately not a cleanup of the stored text — the corpus keeps its layout
    artefacts, and V10's content hashes depend on it. This is only how two
    strings are compared and how a draft is presented for reading.
    """
    collapsed = _WS_RE.sub(" ", (text or "").strip())
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", collapsed)


def _sentences(text: str) -> list[str]:
    """Simple deterministic sentence split (same heuristic as relevance.py).

    Deliberately NOT a parser. A chunk with unusual punctuation can split badly,
    and the consequence is only a less useful draft — every emitted sentence is
    still verbatim, so a bad split can never produce an untrue key point.
    """
    parts = [p.strip() for p in _SENTENCE_RE.split((text or "").strip())]
    return [p for p in parts if p]


def _coverage(terms: list[str], text: str, weights: TermWeights | None) -> float:
    """IDF-weighted share of `terms` present in `text` (0..1).

    Mirrors `relevance.compute_relevance` so a draft's coverage number means the
    same thing as the evaluator's relevance number. Falls back to plain uniform
    coverage — and reports that it did, via the caller's `method` — when the KB
    has no corpus statistics.
    """
    if not terms:
        return 0.0
    present = set(extract_terms(text))
    if weights is not None and weights.available:
        total = sum(weights.idf(t) for t in terms)
        if total <= 0:
            return 0.0
        return sum(weights.idf(t) for t in terms if t in present) / total
    return len([t for t in terms if t in present]) / len(terms)


class _QuestionKind(str, Enum):
    DEFINITION = "definition"
    COMPARISON = "comparison"
    MECHANISM = "mechanism"
    ENUMERATION = "enumeration"
    OTHER = "other"


def _question_kind(question: str) -> _QuestionKind:
    """Deterministic question-type guess, used only for a small tie-break bonus."""
    low = f" {question.lower()} "
    if re.search(r"\bwhat (is|are|does) \b", low) or re.search(r"\bdefine\b", low):
        return _QuestionKind.DEFINITION
    if any(k in low for k in (" differ", " difference", " versus ", " vs ", " compared", "instead of", "rather than")):
        return _QuestionKind.COMPARISON
    if re.search(r"\bhow (do|does|can|are|is) \b", low):
        return _QuestionKind.MECHANISM
    if any(
        k in low
        for k in (
            " which ",
            " what four ",
            " factors",
            " phenomena",
            " attributes",
            " states ",
            " stages ",
            " types ",
            " list ",
        )
    ):
        return _QuestionKind.ENUMERATION
    return _QuestionKind.OTHER


_KIND_CUES: dict[_QuestionKind, re.Pattern[str]] = {
    _QuestionKind.DEFINITION: re.compile(r"\b(is|are|refers to|means|describes|defined as)\b", re.I),
    _QuestionKind.COMPARISON: re.compile(
        r"\b(while|whereas|unlike|compared|than|instead|but|differ)\b", re.I
    ),
    _QuestionKind.MECHANISM: re.compile(r"\b(by|through|via|using|allows|enables|converts|transmits)\b", re.I),
    _QuestionKind.ENUMERATION: re.compile(r",|\band\b|\bor\b", re.I),
    _QuestionKind.OTHER: re.compile(r"$^"),  # never matches
}


class _Candidate(BaseModel):
    chunk_id: str
    text: str
    index: int
    coverage: float
    score: float
    structured: bool = False


def _candidates(
    chunk_ids: list[str],
    texts: dict[str, str],
    question_terms: list[str],
    kind: _QuestionKind,
    weights: TermWeights | None,
) -> list[_Candidate]:
    """Every sentence of every cited chunk, scored; best first within a chunk."""
    cue = _KIND_CUES[kind]
    out: list[_Candidate] = []
    for chunk_id in chunk_ids:
        for i, sentence in enumerate(_sentences(texts.get(chunk_id, ""))):
            coverage = _coverage(question_terms, sentence, weights)
            if coverage < MIN_SELECT_COVERAGE:
                continue
            structured = bool(cue.search(sentence))
            out.append(
                _Candidate(
                    chunk_id=chunk_id,
                    # Presented whitespace-normalised: the sentences come from
                    # parsed web content and would otherwise carry the source's
                    # line breaks into a reference answer a human has to read.
                    # `_assert_verbatim` still proves each one exists in the cited
                    # chunk, so this is presentation, not paraphrase.
                    text=_norm(sentence),
                    index=i,
                    coverage=round(coverage, 6),
                    score=round(min(1.0, coverage + (STRUCTURE_BONUS if structured else 0.0)), 6),
                    structured=structured,
                )
            )
    return out


def _select(candidates: list[_Candidate], chunk_ids: list[str]) -> list[_Candidate]:
    """One best sentence per cited chunk, verbatim-deduplicated, capped.

    One per chunk (rather than global top-k) is deliberate: the benchmark marks
    each cited chunk as NECESSARY to the answer, so a draft that quietly ignores
    one of them would look complete while leaving a required key point out.
    Every chunk that yields nothing is visible through `evidence[].sentence_count`
    and the caller's `reason`, not silently dropped.
    """
    selected: list[_Candidate] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        best = next(
            (c for c in sorted(candidates, key=lambda c: (-c.score, c.index)) if c.chunk_id == chunk_id),
            None,
        )
        if best is None:
            continue
        key = _norm(best.text).lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(best)
        if len(selected) >= MAX_KEY_POINTS:
            break
    return selected


def _assert_verbatim(value: str, chunk_text: str, what: str) -> None:
    """Refuse any draft content that is not literally in the chunk it cites.

    This is the module's central guarantee: a draft may only contain text that
    already exists in the cited evidence, so an extractive draft cannot invent
    a fact, and a future generative generator fails loudly here instead of
    quietly shipping invented facts into a benchmark.

    Whitespace is normalised before comparison (chunk text contains layout
    newlines), which is the only difference permitted between a draft span and
    the corpus bytes it came from.
    """
    needle = _norm(value)
    if not needle:
        raise DraftingError(f"{what} is empty; an empty draft must be reported, never stored")
    if needle not in _norm(chunk_text):
        raise DraftingError(
            f"{what} is not verbatim text of the chunk it cites: {value[:120]!r}. "
            f"A draft may only contain sentences that exist in the cited evidence."
        )


def _assert_answer_composition(
    answer: str, selected: list[_Candidate], sources: dict[str, str]
) -> None:
    """The assembled answer must BE the selected sentences, and nothing else.

    A multi-chunk answer is a join of sentences from different chunks, so it is
    not a substring of any single chunk — checking the join as one blob would
    either fail on legitimate drafts or, worse, pass by accident. Instead the
    composition is asserted exactly: every selected sentence is verbatim in the
    chunk it names, and the answer equals their concatenation in order. Nothing
    can be added to a draft answer without this failing.
    """
    expected = " ".join(c.text for c in selected)
    if _norm(answer) != _norm(expected):
        raise DraftingError(
            "the draft reference answer is not exactly the concatenation of its "
            "cited sentences; a draft must not add text of its own"
        )
    for c in selected:
        _assert_verbatim(c.text, sources.get(c.chunk_id, ""), f"key point from {c.chunk_id}")


def _benchmark_quote(provenance: str) -> str:
    """Pull the human-recorded supporting sentence out of an artifact's provenance.

    The shipped benchmark stores e.g. ``Chunk chk_x (Title, section Y): 'An
    engine or motor is a machine...' Manually read and selected 2026-09-15.``
    Recovering it lets the draft report whether the deterministic selector
    independently landed on the sentence the human chose — a free, real sanity
    check that needs no new human input.
    """
    match = _QUOTED_RE.search(provenance or "")
    return _norm(match.group(1)) if match else ""


def build_draft_suggestion(
    *,
    kb_id: str,
    question: AnswerBenchmarkQuestion,
    evidence_texts: dict[str, str],
    evidence_meta: dict[str, dict[str, Any]] | None = None,
    weights: TermWeights | None = None,
    benchmark_name: str = "",
    benchmark_fingerprint: str = "",
    generator: str = GENERATOR,
    now: datetime | None = None,
) -> DraftSuggestion:
    """Assemble a DRAFT for one question from its real evidence text.

    Pure and offline: the caller supplies the chunk texts (from the live corpus,
    so provenance problems are detectable) and optional corpus term statistics.
    Returns a `DraftSuggestion` — a DRAFT, never an annotation.
    """
    meta = evidence_meta or {}
    required: list[RequiredEvidence] = list(question.required_evidence or [])
    chunk_ids = [e.chunk_id for e in required]

    # Only chunks whose text we actually hold may be quoted.
    usable_ids = [cid for cid in chunk_ids if _norm(evidence_texts.get(cid, ""))]
    sources = {cid: evidence_texts[cid] for cid in usable_ids}

    question_terms = extract_terms(question.question or "")
    union_text = " ".join(evidence_texts.get(cid, "") for cid in usable_ids)
    covered = [t for t in question_terms if t in set(extract_terms(union_text))]
    uncovered = [t for t in question_terms if t not in set(covered)]
    # Most significant first, so the human reads the terms that actually matter
    # (a missing 'roll'/'pitch'/'yaw' outranks a missing 'instead').
    if weights is not None and weights.available:
        uncovered = sorted(uncovered, key=lambda t: (-weights.idf(t), t))
    coverage_measured = bool(weights is not None and weights.available and usable_ids)
    coverage = _coverage(question_terms, union_text, weights) if usable_ids else 0.0

    kind = _question_kind(question.question or "")
    candidates = _candidates(usable_ids, evidence_texts, question_terms, kind, weights)
    selected = _select(candidates, usable_ids)

    if selected:
        # Structural guarantee, re-checked rather than assumed.
        _assert_answer_composition(
            " ".join(c.text for c in selected), selected, sources
        )

    if not usable_ids:
        status = DraftStatus.INSUFFICIENT_EVIDENCE
        reason = (
            "none of this question's required evidence is present in the live corpus, "
            "so no draft can be assembled from it"
        )
    elif not selected:
        status = DraftStatus.INSUFFICIENT_EVIDENCE
        reason = (
            f"no sentence in the cited evidence covers at least "
            f"{MIN_SELECT_COVERAGE:.0%} of the question's content terms "
            f"({len(question_terms)} terms, {len(covered)} present somewhere in the "
            f"evidence); the corpus text does not state an answer in a form this "
            f"deterministic selector can quote"
        )
    elif coverage_measured and coverage < MIN_QUESTION_COVERAGE:
        status = DraftStatus.PARTIAL
        reason = (
            f"the cited evidence covers only {coverage:.0%} of the question's "
            f"IDF-weighted content terms (below the {MIN_QUESTION_COVERAGE:.0%} triage "
            f"line); terms not present, most significant first: "
            f"{', '.join(uncovered[:8])}{' …' if len(uncovered) > 8 else ''}. A human must "
            f"decide whether the question as written is answered by its pinned evidence"
        )
    elif not coverage_measured and uncovered:
        status = DraftStatus.PARTIAL
        reason = (
            f"{len(covered)}/{len(question_terms)} question terms appear in the evidence "
            f"and {len(selected)} verbatim sentence(s) were selected. Corpus statistics are "
            f"unavailable for this knowledge base, so completeness was judged on plain term "
            f"presence — a weaker signal that is documented as non-discriminative; terms "
            f"not present: {', '.join(uncovered[:8])}"
        )
    else:
        status = DraftStatus.DRAFT_READY
        reason = (
            f"the cited evidence covers {coverage:.0%} of the question's IDF-weighted "
            f"content terms and {len(selected)} verbatim sentence(s) were selected"
            if coverage_measured
            else f"all {len(question_terms)} question content terms appear in the evidence "
            f"and {len(selected)} verbatim sentence(s) were selected"
        )

    evidence_rows: list[DraftEvidence] = []
    for ev in required:
        text = evidence_texts.get(ev.chunk_id, "")
        info = meta.get(ev.chunk_id, {})
        evidence_rows.append(
            DraftEvidence(
                chunk_id=ev.chunk_id,
                found=bool(_norm(text)),
                document_id=info.get("document_id") or ev.document_id,
                source_title=info.get("source_title") or ev.source_title,
                section=info.get("section") or ev.section,
                content_hash=str(info.get("content_hash") or ev.content_hash or ""),
                char_count=len(text or ""),
                sentence_count=len(_sentences(text)),
                problems=list(info.get("problems") or []),
            )
        )

    quote = _benchmark_quote(question.provenance or "")
    agrees: bool | None = None
    if quote and selected:
        chosen = {_norm(c.text).lower() for c in selected}
        agrees = any(quote.lower() in c or c in quote.lower() for c in chosen)
    elif quote:
        agrees = False

    caveats = [
        "This is a MACHINE-GENERATED DRAFT assembled from verbatim sentences of the cited "
        "corpus evidence. It is NOT human ground truth and must never be approved as-is.",
        "To become ground truth it must be read, corrected as needed, and explicitly "
        "submitted by a named human through the authoring endpoint; only then is it eligible "
        "for review and approval.",
        "The draft contains only text that already exists in the evidence — it adds no "
        "outside knowledge and no paraphrase, so it can also be incomplete in wording.",
    ]
    if status is DraftStatus.PARTIAL and uncovered:
        caveats.append(
            f"Terms of the question that appear nowhere in the cited evidence, most "
            f"significant first: {', '.join(uncovered[:10])}. Several of these are usually the "
            f"question's own phrasing (synonyms or function words) rather than a real gap — "
            f"check the evidence before treating any of them as missing knowledge, and do "
            f"not supply them from outside knowledge."
        )
    if not coverage_measured:
        caveats.append(
            "This knowledge base has no corpus term statistics, so coverage could not be "
            "weighted; plain term coverage is documented as non-discriminative and this "
            "draft's completeness signal is therefore weaker than usual."
        )
    if status is DraftStatus.INSUFFICIENT_EVIDENCE:
        caveats.append(
            "No draft was produced. Do not infer an answer: classify the question as "
            "insufficient_evidence (or ambiguous) unless you can cite additional corpus chunks."
        )
    if agrees is False and quote:
        caveats.append(
            "The selector's sentence differs from the sentence recorded in the benchmark's "
            "provenance for this question — compare both before deciding which is right."
        )

    return DraftSuggestion(
        suggestion_id=new_id("dft"),
        kb_id=kb_id,
        benchmark_name=benchmark_name,
        benchmark_fingerprint=benchmark_fingerprint,
        question_id=question.question_id,
        created_at=now or datetime.now(timezone.utc),
        method=METHOD_EXTRACTIVE,
        generator=generator,
        status=status,
        reason=reason,
        caveats=caveats,
        question=question.question,
        subdomain=question.subdomain,
        expected_answer_draft=" ".join(c.text for c in selected),
        key_points_draft=[
            DraftKeyPoint(
                text=c.text,
                source_chunk_id=c.chunk_id,
                source_sentence_index=c.index,
                score=c.score,
                coverage=c.coverage,
            )
            for c in selected
        ],
        evidence=evidence_rows,
        question_terms=question_terms,
        covered_terms=covered,
        uncovered_terms=uncovered,
        coverage=round(float(coverage), 6),
        coverage_measured=coverage_measured,
        coverage_method=(
            "idf-weighted-question-term-coverage"
            if coverage_measured
            else ("unavailable: no corpus statistics for this knowledge base")
        ),
        benchmark_quote=quote,
        agrees_with_benchmark_quote=agrees,
    )


def summarise_drafts(suggestions: list[DraftSuggestion]) -> dict[str, Any]:
    """Small honest roll-up for the batch endpoint and the UI banner.

    Counts only what was actually produced — no projected or assumed values.
    """
    counts = {s.value: 0 for s in DraftStatus}
    for s in suggestions:
        counts[s.status.value] += 1
    return {
        "total": len(suggestions),
        "by_status": counts,
        "with_draft": sum(1 for s in suggestions if s.expected_answer_draft),
        "key_points_offered": sum(len(s.key_points_draft) for s in suggestions),
        "agreeing_with_benchmark_quote": sum(
            1 for s in suggestions if s.agrees_with_benchmark_quote is True
        ),
        "disagreeing_with_benchmark_quote": sum(
            1 for s in suggestions if s.agrees_with_benchmark_quote is False
        ),
        "note": (
            "Drafts are machine-generated starting points, not ground truth. Counts describe "
            "the drafts themselves; they say nothing about answer quality."
        ),
    }


__all__ = [
    "DRAFT_LABEL",
    "GENERATOR",
    "MAX_KEY_POINTS",
    "METHOD_EXTRACTIVE",
    "MIN_SELECT_COVERAGE",
    "STRUCTURE_BONUS",
    "DraftEvidence",
    "DraftKeyPoint",
    "DraftStatus",
    "DraftSuggestion",
    "DraftingError",
    "build_draft_suggestion",
    "summarise_drafts",
]
