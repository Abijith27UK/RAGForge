"""Phase 11 — evidence sufficiency / grounding gate.

The gate answers ONE question: may this question be answered from this evidence,
and if not, what should happen instead? It is deterministic (no LLM) and every
decision is explained by recorded `GateSignal`s.

Signals used (all measured, all disclosed):
* evidence_count / document_count   — how much came back, from how many docs
* score distribution                — top/mean retrieval scores (strategy scale,
                                      recorded with the strategy name)
* evidence relevance floor (V20)    — top reported similarity vs an absolute
                                      floor. Makes the top-score signal decisive
                                      instead of merely recorded. NOT APPLICABLE,
                                      with the reason, for strategies that
                                      normalize inside the candidate pool
* corpus vocabulary fit (V20)       — how many of the question's terms occur in
                                      the corpus at all, and whether the corpus
                                      holds any rare term from it. Distinguishes
                                      "the corpus discusses this" from "one
                                      common word matched". Unmeasured when no
                                      corpus term statistics exist, and said so
* independent-source agreement      — distinct documents containing the same
                                      key term
* provenance completeness           — how much of the provenance chain exists
* trust                             — mean recorded trust score, when available
* retrieval agreement               — dense/lexical pool agreement when the
                                      strategy is hybrid; NOT PERFORMED for
                                      single-pool strategies (stated, not faked)

Honesty rules:
* A high retrieval score is NOT sufficient evidence by itself.
* `confidence` is categorical (high/moderate/low/none), never a fabricated
  percentage.
* Sub-aspect coverage is checked by deterministic term matching — labelled
  heuristic in the reason text where it drives the decision.
* V20: a term that merely OCCURS in the evidence is not treated as coverage of
  the question. The gate asks whether the question's distinctive terms are there.
  When corpus term statistics are unavailable the check is recorded as NOT
  PERFORMED (with the reason) instead of being silently skipped.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas.answer import (
    GATE_TO_GROUNDING,
    ConfidenceCategory,
    EvidenceAssessment,
    EvidenceSet,
    GateDecision,
    GateSignal,
    GroundingState,
    QueryPlan,
    QueryTrace,
)
from app.services.answering.query_processor import QueryNature, extract_terms
from app.services.answering.term_stats import (
    CorpusTermStats,
    TermStatsProvider,
    create_term_stats_provider,
)

# Thresholds: explicit constants so tests and docs can reference them.
MIN_ALIGNMENT_TO_ANSWER = 0.34      # >= of question terms found in evidence
MIN_ALIGNMENT_TO_PARTIAL = 0.15     # below this we do not answer at all
ASPECT_MIN_COVERAGE = 0.5           # a clause counts as covered only when a
                                    # STRICT MAJORITY of its terms appear in
                                    # evidence (prefer understating)
TOP_SCORE_MODERATE = 0.35           # dense/cosine-ish scale; see score notes
INDEPENDENT_SOURCES_STRONG = 2      # documents agreeing on a key term
TRUST_LOW = 0.3                     # mean recorded trust below this = low-trust evidence

#: V20 — floor on the TOP reported similarity below which nothing retrieved is
#: topically related to the question, so no answer can be grounded in it.
#:
#: Measured, not invented. On the real Automobile corpus (`kb_f278c283c748`, 812
#: chunks, `all-MiniLM-L6-v2`, strategy `dense`) the two bands do not overlap:
#:
#:     12 supported in-domain questions : top cosine 0.4365 .. 0.6946
#:      6 off-domain questions          : top cosine 0.2317 .. 0.3411
#:
#: 0.35 falls in the gap (0.3411 < 0.35 < 0.4365) and is deliberately NOT a new
#: number: it is `TOP_SCORE_MODERATE`, the gate's own "moderate relevance" mark
#: since V7, which the gate recorded in a signal (`score_distribution.passed`)
#: and then never consulted. V20 makes that existing signal decisive.
#: Reproduce the measurement with:
#:     python -m scripts.v20_calibrate_relevance --kb <kb_id>
#: CAVEAT (documented, not hidden): a cosine scale is embedding-model dependent,
#: so this floor is calibrated PER embedding model. It is applied only to
#: strategies that report an unnormalized similarity — see
#: `_RAW_SIMILARITY_STRATEGIES`; for pool-normalized strategies the gate records
#: the check as NOT APPLICABLE instead of applying a threshold that would mean
#: something else there.
MIN_RELEVANCE_TO_ANSWER = TOP_SCORE_MODERATE

#: V20 — share of a question's content terms that must be absent from the corpus
#: ENTIRELY (df == 0 across every indexed chunk) before the gate reports the topic
#: as outside the corpus. Set to a plain majority: a question built mostly from
#: words that occur in none of 812 chunks of technical text is not a question this
#: corpus can answer, and "absent from the corpus" is a measurement, not a guess.
MIN_ABSENT_TERM_SHARE = 0.5

#: V20 — a question term is DISTINCTIVE when its BM25 (Lucene) IDF in the corpus
#: being searched is at least this. IDF comes from the very same function the
#: BM25 retriever scores with (`services/retrieval/lexical.idf`), so the gate and
#: retrieval cannot disagree about how rare a term is.
#:
#: Calibration on the real Automobile corpus (812 indexed chunks) — measurements:
#:   corpus-ubiquitous English:  "vehicle" 1.108  "engine" 1.860  "many" 2.566
#:                               "steering" 2.534 "brake" 2.550  "suspension" 2.142
#:   topic-bearing vocabulary:   "torque" 3.076  "clutch" 3.502  "order" 3.960
#:                               "cylinder" 4.561 "turbocharger" 5.448 "firing" 5.784
#:   never in the corpus:        "jupiter", "moons", "drone", "lyrebird"
#: KNOWN LIMITATION (measured, and the reason this is used only to decide whether
#: the corpus has ANY rare handle on the question): rare is not the same as
#: topical. Question-framing words that technical prose rarely uses also land
#: above this line — "specification" 6.295, "purpose" 5.448, "difference" 4.686,
#: "work" 3.316, "components" 3.131 — and "explain"/"describe" do not occur in
#: the corpus at all. A rule that abstained on such words alone was measured to
#: abstain on "How do brakes work?" and "Explain the steering system.", both of
#: which this corpus answers. That is why V20 tests absent-term SHARE plus the
#: absence of any rare corpus term, and never IDF alone.
MIN_DISTINCTIVE_IDF = 3.0

#: Strategies whose reported `retrieval_score` is an UNNORMALIZED similarity, so
#: that `MIN_RELEVANCE_TO_ANSWER` is meaningful for them. `bm25`, `hybrid` and
#: `hybrid_reranked` min-max normalize inside the candidate pool (their top hit
#: is ~1.0 by construction), so an absolute floor is NOT comparable there and the
#: gate says so rather than pretending the check ran.
_RAW_SIMILARITY_STRATEGIES = frozenset({"dense"})

#: Pattern for "term = number" style facts (engineering documents are full of
#: them) used by the HEURISTIC numerical-conflict detector.
_VALUE_RE = re.compile(r"\b([A-Za-z][A-Za-z ]{1,28}?)\s*[=:]\s*(\d+(?:\.\d+)?)\b")


def detect_numerical_conflicts(items: list) -> list[str]:
    """HEURISTIC: same quantity term with different values across documents.

    Deterministic string/number comparison only — it cannot catch prose-form
    disagreements, which is stated wherever the result is displayed.
    """
    seen: dict[str, tuple[str, str]] = {}  # normalized term -> (value, evidence_id)
    conflicts: list[str] = []
    for item in items:
        for term, value in _VALUE_RE.findall(item.content):
            key = " ".join(term.lower().split())
            if len(key) < 3:
                continue
            prior = seen.get(key)
            if prior is None:
                seen[key] = (value, item.evidence_id)
            elif prior[0] != value:
                other_doc = next(
                    (e.document_id for e in items if e.evidence_id == prior[1]), ""
                )
                if item.document_id != other_doc:
                    conflicts.append(
                        f"'{key}' has conflicting values: {prior[0]} ({prior[1]}) "
                        f"vs {value} ({item.evidence_id})"
                    )
    return conflicts

_SPLIT_RE = re.compile(r"\s+(?:and|also|versus|vs\.?|;)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class _VocabularyFit:
    """V20 result: how much of the question the corpus can even represent.

    ``measured=False`` is a first-class outcome — it carries the reason the check
    could not be run (no corpus statistics, no kb id, no question terms) so the
    gate never implies a check it did not perform.
    """

    measured: bool
    not_performed_reason: str = ""
    terms: tuple[str, ...] = ()
    absent_from_corpus: tuple[str, ...] = ()
    corpus_present_distinctive: tuple[str, ...] = ()
    absent_share: float = 0.0
    n_docs: int = 0
    stats_source: str = ""


class EvidenceGate(ABC):
    """Decides whether evidence may support an answer, and why.

    Kept as the pipeline-facing interface (one method, `assess`) so existing
    callers and tests are unaffected. `GroundingGate` below is the same
    capability expressed with the V7 five-state vocabulary.

    `kb_id` is optional and keyword-only: it identifies the corpus the evidence
    came from, which the distinctive-term check needs in order to read that
    corpus's term statistics. Callers that do not supply it get the check
    recorded as NOT PERFORMED rather than an invented result.
    """

    name: str = "base"

    @abstractmethod
    def assess(
        self, plan: QueryPlan, evidence: EvidenceSet, *, kb_id: str = ""
    ) -> EvidenceAssessment: ...


class GroundingGate(EvidenceGate):
    """EvidenceGate that reports one of the five explicit grounding states.

    The states exist so the API and UI never have to invent a label from a
    number:

    * ANSWERED               — evidence supports the question; answer it.
    * PARTIALLY_SUPPORTED    — evidence supports part of the question only.
    * INSUFFICIENT_EVIDENCE  — something was retrieved but it does not ground
                               an answer (or the turn was not a question).
    * CONFLICTING_EVIDENCE   — sources disagree on a quantity; answer, but say
                               so, and never silently pick a side.
    * NO_RELEVANT_EVIDENCE   — retrieval returned nothing usable at all.

    `assess` fills `EvidenceAssessment.grounding_state` via the exhaustive
    GATE_TO_GROUNDING map, then refines it with the CONFLICTING_EVIDENCE and
    NO_RELEVANT_EVIDENCE cases that the coarser GateDecision cannot express.
    Subclasses implement `assess` exactly as before; they inherit the state
    derivation for free and stay deterministic.
    """

    version: str = "v7.2"

    def assess_with_trace(
        self,
        plan: QueryPlan,
        evidence: EvidenceSet,
        trace: QueryTrace | None = None,
        *,
        kb_id: str = "",
    ) -> EvidenceAssessment:
        """Assess, then attach the grounding state and the query-trace context.

        Conversational and non-knowledge turns short-circuit WITHOUT consulting
        the evidence: a greeting has no evidence requirement, so asking whether
        its evidence was sufficient would be a category error, and reporting
        "the corpus does not cover this" would be misleading.

        UNDERSPECIFIED and MULTI_HOP are deliberately NOT short-circuited. Both
        are still knowledge questions, so the real evidence signals drive the
        decision — which is what keeps an empty knowledge base reporting the
        honest NO_EVIDENCE reason rather than a classification artefact.
        """
        if trace is not None and trace.nature in (
            QueryNature.CONVERSATIONAL,
            QueryNature.NON_KNOWLEDGE,
        ):
            no_evidence = not evidence.items
            return EvidenceAssessment(
                sufficient=False,
                decision=GateDecision.ASK_CLARIFICATION,
                grounding_state=(
                    GroundingState.NO_RELEVANT_EVIDENCE
                    if no_evidence
                    else GroundingState.INSUFFICIENT_EVIDENCE
                ),
                confidence=ConfidenceCategory.NONE,
                reason_code="NO_RELEVANT_EVIDENCE" if no_evidence else "NON_KNOWLEDGE_TURN",
                reason=(
                    f"The message was classified as {trace.nature.value} rather than a "
                    "knowledge question, so there is nothing to ground an answer in. "
                    + (
                        "No evidence was retrieved."
                        if no_evidence
                        else "Ask a specific question about this knowledge base."
                    )
                ),
                evidence_count=len(evidence.items),
                document_count=len({e.document_id for e in evidence.items}),
                supporting_evidence_ids=[e.evidence_id for e in evidence.items],
                unsupported_aspects=[trace.normalized_query] if trace.normalized_query else [],
                missing_information=["a knowledge question about this knowledge base"],
                recommended_action="ask_clarification",
                signals=[],
                notes=[
                    f"grounding gate: {trace.nature.value} turn; the evidence "
                    "sufficiency rules were not applied"
                ],
            )

        assessment = self.assess(plan, evidence, kb_id=kb_id)
        return self._with_grounding_state(assessment, evidence)

    def _with_grounding_state(
        self, assessment: EvidenceAssessment, evidence: EvidenceSet
    ) -> EvidenceAssessment:
        """Derive the five-state outcome from the recorded decision + signals.

        Refinements, in priority order:
        1. Nothing retrieved (or nothing survived selection) -> NO_RELEVANT_EVIDENCE.
        2. Conflicting numerical values detected across documents ->
           CONFLICTING_EVIDENCE, even when the gate chose to answer. The answer
           may still be produced; the state exists so the caller is told.
        3. Otherwise the decision maps through GATE_TO_GROUNDING.
        """
        assessment.grounding_state = GATE_TO_GROUNDING[assessment.decision]

        if not evidence.items or assessment.evidence_count == 0:
            assessment.grounding_state = GroundingState.NO_RELEVANT_EVIDENCE
            return assessment

        conflict = next(
            (
                s
                for s in assessment.signals
                if s.name == "conflicting_values" and s.value and s.value > 0
            ),
            None,
        )
        if conflict is not None:
            assessment.grounding_state = GroundingState.CONFLICTING_EVIDENCE
        return assessment


def _clause_aspects(plan: QueryPlan) -> list[str]:
    """Split the question into aspects (clauses) for coverage checking.

    Heuristic: splits on 'and'/'also'/'vs'/';' ONLY when the question is long
    enough that the conjunction joins real clauses, not list items.
    """
    q = plan.normalized_query
    if len(q) < 20:
        return [q]
    parts = [p.strip() for p in _SPLIT_RE.split(q) if p.strip()]
    return parts if len(parts) > 1 else [q]


class HeuristicEvidenceGate(GroundingGate):
    """Rule-based grounding gate. Deterministic and fully explained."""

    name = "heuristic-rules"
    version = "v20.0"

    def __init__(
        self,
        stats_provider: TermStatsProvider | None = None,
        min_relevance: float = MIN_RELEVANCE_TO_ANSWER,
    ) -> None:
        #: Corpus term statistics. `None` means "no provider configured", which
        #: makes the vocabulary-fit check explicitly NOT PERFORMED.
        self._stats = stats_provider or create_term_stats_provider(None)
        #: Raw-similarity floor. Configurable per deployment because a cosine
        #: scale is embedding-model dependent (see MIN_RELEVANCE_TO_ANSWER).
        self._min_relevance = float(min_relevance)

    def assess(
        self, plan: QueryPlan, evidence: EvidenceSet, *, kb_id: str = ""
    ) -> EvidenceAssessment:
        items = evidence.items
        signals: list[GateSignal] = []
        notes: list[str] = list(evidence.notes)

        q_terms = set(plan.extracted_terms) or set(
            t for t in plan.normalized_query.lower().split() if len(t) > 2
        )

        # -- signal: counts ---------------------------------------------------
        docs = {e.document_id for e in items}
        signals.append(
            GateSignal(
                name="evidence_count",
                value=float(len(items)),
                threshold=1.0,
                passed=bool(items),
                interpretation=(
                    f"{len(items)} evidence item(s) from {len(docs)} document(s)"
                    if items
                    else "no evidence retrieved"
                ),
            )
        )

        # -- signal: score distribution (only meaningful vs itself) -----------
        if items:
            scores = [e.retrieval_score for e in items]
            top = max(scores)
            mean = sum(scores) / len(scores)
            spread = top - min(scores)
            signals.append(
                GateSignal(
                    name="score_distribution",
                    value=round(top, 4),
                    threshold=TOP_SCORE_MODERATE,
                    passed=top >= TOP_SCORE_MODERATE,
                    interpretation=(
                        f"top={top:.3f} mean={mean:.3f} spread={spread:.3f} on the "
                        f"'{evidence.strategy}' scale (pool-relative; comparable only "
                        f"within this run — NOT a probability)"
                    ),
                )
            )
        else:
            signals.append(
                GateSignal(
                    name="score_distribution",
                    measured=False,
                    not_performed_reason="no evidence to score",
                )
            )

        # -- signal: lexical alignment ----------------------------------------
        if items and q_terms:
            evidence_text = " ".join(e.content.lower() for e in items)
            found = {t for t in q_terms if t in evidence_text}
            alignment = len(found) / len(q_terms)
            missing = sorted(q_terms - found)
            signals.append(
                GateSignal(
                    name="lexical_alignment",
                    value=round(alignment, 4),
                    threshold=MIN_ALIGNMENT_TO_ANSWER,
                    passed=alignment >= MIN_ALIGNMENT_TO_ANSWER,
                    interpretation=(
                        f"{len(found)}/{len(q_terms)} question terms appear in the "
                        f"evidence (HEURISTIC substring match)"
                        + (f"; missing: {', '.join(missing[:8])}" if missing else "")
                    ),
                )
            )
        else:
            signals.append(
                GateSignal(
                    name="lexical_alignment",
                    measured=False,
                    not_performed_reason=(
                        "no evidence" if not items else "question has no content terms to align"
                    ),
                )
            )

        # -- signal: relevance floor (V20) --------------------------------------
        #
        # `score_distribution` records the top score but its `passed` flag was
        # never consulted, which is how "How many moons does Jupiter have?" was
        # answered with a top cosine of 0.2317. This signal states whether an
        # ABSOLUTE relevance floor is even meaningful for the strategy in use.
        if items:
            top_score = max(e.retrieval_score for e in items)
            strategy_ok = evidence.strategy in _RAW_SIMILARITY_STRATEGIES
            semantic = not evidence.embedding_is_fallback
            applicable = strategy_ok and semantic
            if not strategy_ok:
                interpretation = (
                    f"strategy '{evidence.strategy}' reports a score that is "
                    "normalized within the candidate pool, so its top hit is near "
                    "1.0 by construction and an absolute similarity floor is not "
                    "comparable"
                )
                reason = (
                    f"strategy '{evidence.strategy}' does not report an "
                    "unnormalized similarity; only "
                    f"{sorted(_RAW_SIMILARITY_STRATEGIES)} do, so the absolute "
                    "relevance floor was NOT applied to this run"
                )
            elif not semantic:
                interpretation = (
                    "these similarities come from a DEVELOPMENT embedding fallback "
                    "(hashing), which measures character overlap rather than meaning; "
                    f"its top score was {top_score:.4f} but a semantic floor cannot be "
                    "read off that scale"
                )
                reason = (
                    "the retrieval run used a development embedding fallback, whose "
                    "similarities are not a semantic scale, so the absolute relevance "
                    "floor was NOT applied to this run"
                )
            else:
                interpretation = (
                    f"top reported similarity {top_score:.4f} vs floor "
                    f"{self._min_relevance} on the raw similarity scale of "
                    f"strategy '{evidence.strategy}'"
                )
                reason = ""
            signals.append(
                GateSignal(
                    name="evidence_relevance_floor",
                    measured=applicable,
                    value=round(top_score, 4) if applicable else None,
                    threshold=self._min_relevance,
                    passed=(top_score >= self._min_relevance) if applicable else None,
                    interpretation=interpretation,
                    not_performed_reason=reason,
                )
            )
        else:
            signals.append(
                GateSignal(
                    name="evidence_relevance_floor",
                    measured=False,
                    not_performed_reason="no evidence to measure relevance on",
                )
            )

        # -- signal: corpus vocabulary fit (V20) -------------------------------
        #
        # `lexical_alignment` above asks only whether SOME question term occurs in
        # the evidence, so a corpus-ubiquitous word ("many") satisfies it. This
        # signal instead measures how much of the question the CORPUS itself can
        # represent: how many of its content terms occur in none of the indexed
        # chunks, and whether the corpus holds any rare (IDF >= threshold) term
        # from the question at all. Both are corpus measurements, not guesses.
        fit = self._vocabulary_fit(plan, items, q_terms, kb_id)
        if not fit.measured:
            signals.append(
                GateSignal(
                    name="corpus_vocabulary_fit",
                    measured=False,
                    not_performed_reason=fit.not_performed_reason,
                )
            )
        else:
            signals.append(
                GateSignal(
                    name="corpus_vocabulary_fit",
                    value=round(fit.absent_share, 4),
                    threshold=MIN_ABSENT_TERM_SHARE,
                    passed=fit.absent_share < MIN_ABSENT_TERM_SHARE,
                    interpretation=(
                        f"{len(fit.absent_from_corpus)}/{len(fit.terms)} question "
                        f"term(s) occur nowhere in the {fit.n_docs} indexed chunk(s) "
                        f"({', '.join(fit.absent_from_corpus) if fit.absent_from_corpus else 'none'})"
                        + (
                            "; corpus holds NO rare term from this question "
                            f"(IDF >= {MIN_DISTINCTIVE_IDF}), so the corpus has no "
                            "distinctive handle on it"
                            if not fit.corpus_present_distinctive
                            else "; corpus DOES hold rare term(s) "
                            f"{', '.join(fit.corpus_present_distinctive)} "
                            f"(IDF >= {MIN_DISTINCTIVE_IDF})"
                        )
                        + f". Measured from source={fit.stats_source}; no term "
                        "assumed important."
                    ),
                )
            )

        # -- signal: independent-source agreement ------------------------------
        if items and q_terms:
            key_terms = [t for t in plan.domain_terms] or sorted(
                (t for t in q_terms if len(t) > 4), key=len, reverse=True
            )[:3]
            agree = 0
            per_term: list[str] = []
            for term in key_terms:
                docs_with = {e.document_id for e in items if term in e.content.lower()}
                if len(docs_with) >= INDEPENDENT_SOURCES_STRONG:
                    agree += 1
                    per_term.append(f"'{term}' in {len(docs_with)} docs")
            signals.append(
                GateSignal(
                    name="independent_source_agreement",
                    value=float(agree),
                    threshold=1.0,
                    passed=agree >= 1,
                    interpretation=(
                        "; ".join(per_term)
                        if per_term
                        else (
                            "no key term appears in more than one document "
                            "(agreement cannot be established from a single source)"
                        )
                    ),
                )
            )
        else:
            signals.append(
                GateSignal(
                    name="independent_source_agreement",
                    measured=False,
                    not_performed_reason="no evidence or no question terms",
                )
            )

        # -- signal: provenance completeness ------------------------------------
        if items:
            complete = 0
            for e in items:
                has_chain = bool(e.document_id and e.title and e.content_hash)
                if has_chain:
                    complete += 1
            frac = complete / len(items)
            signals.append(
                GateSignal(
                    name="provenance_completeness",
                    value=round(frac, 4),
                    threshold=1.0,
                    passed=frac >= 1.0,
                    interpretation=(
                        f"{complete}/{len(items)} items carry document id + title + content hash"
                    ),
                )
            )
        else:
            signals.append(
                GateSignal(
                    name="provenance_completeness",
                    measured=True,
                    value=0.0,
                    passed=False,
                    interpretation="no evidence, so no provenance chain exists",
                )
            )

        # -- signal: trust --------------------------------------------------------
        trusts = [e.trust_score for e in items if e.trust_score is not None]
        if trusts:
            mean_trust = sum(trusts) / len(trusts)
            signals.append(
                GateSignal(
                    name="source_trust",
                    value=round(mean_trust, 4),
                    interpretation=(
                        f"mean recorded trust {mean_trust:.2f} over {len(trusts)}/"
                        f"{len(items)} items (HEURISTIC source-quality score; "
                        f"unrated items excluded)"
                    ),
                )
            )
        else:
            signals.append(
                GateSignal(
                    name="source_trust",
                    measured=False,
                    not_performed_reason="no trust scores recorded for these sources",
                )
            )

        # -- signal: retrieval agreement ------------------------------------------
        if evidence.strategy.startswith("hybrid"):
            # Per-pool contribution is visible on the retrieval result's
            # score_breakdown, which evidence assembly copies into provenance.
            both = 0
            for e in items:
                sb = e.provenance.get("score_breakdown") or {}
                if sb.get("dense_score") is not None and sb.get("lexical_score") is not None:
                    both += 1
            signals.append(
                GateSignal(
                    name="retrieval_agreement",
                    value=float(both),
                    threshold=1.0,
                    passed=both >= 1,
                    interpretation=(
                        f"{both}/{len(items)} items were produced by BOTH dense and "
                        f"lexical pools (fusion ran)"
                        if both
                        else "hybrid strategy ran but per-pool contribution is not "
                        "visible on these items"
                    ),
                )
            )
        else:
            signals.append(
                GateSignal(
                    name="retrieval_agreement",
                    measured=False,
                    not_performed_reason=(
                        f"strategy '{evidence.strategy}' uses a single retrieval pool; "
                        "cross-strategy agreement would require a second retrieval run "
                        "(NOT IMPLEMENTED in V7)"
                    ),
                )
            )

        # -- signal: numerical conflicts across sources -------------------------
        conflicts = detect_numerical_conflicts(items)
        signals.append(
            GateSignal(
                name="conflicting_values",
                value=float(len(conflicts)),
                threshold=0.0,
                passed=not conflicts,
                interpretation=(
                    "; ".join(conflicts[:5])
                    if conflicts
                    else (
                        "no conflicting 'term = value' facts found across documents "
                        "(HEURISTIC: prose-form disagreement is not detected)"
                    )
                ),
            )
        )
        for c in conflicts:
            notes.append(f"conflicting evidence: {c}")

        # -- decision ------------------------------------------------------------
        return self._decide(plan, evidence, items, signals, notes, fit)

    def _vocabulary_fit(
        self,
        plan: QueryPlan,
        items: list,
        q_terms: set[str],
        kb_id: str,
    ) -> _VocabularyFit:
        """Measure how much of the question the corpus can represent at all.

        Two measurements, both straight from the persisted BM25 statistics:

        * `absent_share` — the fraction of the question's content terms that
          occur in NONE of the indexed chunks (df == 0). This is a provable
          statement about the corpus, not an inference.
        * `corpus_present_distinctive` — the terms that DO occur in the corpus
          with IDF >= `MIN_DISTINCTIVE_IDF`. Any such term is a rare, on-corpus
          handle for the question.

        Returns `measured=False` with a reason whenever the measurement is not
        possible, so the caller records a skipped check instead of a result.
        """
        if not items:
            return _VocabularyFit(
                measured=False, not_performed_reason="no evidence to compare against"
            )
        if not q_terms:
            return _VocabularyFit(
                measured=False,
                not_performed_reason="the question has no content terms to classify",
            )
        if not kb_id:
            return _VocabularyFit(
                measured=False,
                not_performed_reason=(
                    "the knowledge base id was not supplied, so corpus term "
                    "frequencies could not be read"
                ),
            )
        stats: CorpusTermStats = self._stats.stats(kb_id)
        if not stats.available:
            return _VocabularyFit(
                measured=False,
                not_performed_reason=f"corpus term statistics unavailable: {stats.reason}",
            )

        absent: list[str] = []
        present_distinctive: list[str] = []
        for term in sorted(q_terms):
            if stats.is_absent_from_corpus(term):
                absent.append(term)
                continue
            idf_value = stats.inverse_document_frequency(term)
            if idf_value is not None and idf_value >= MIN_DISTINCTIVE_IDF:
                present_distinctive.append(term)

        return _VocabularyFit(
            measured=True,
            terms=tuple(sorted(q_terms)),
            absent_from_corpus=tuple(absent),
            corpus_present_distinctive=tuple(present_distinctive),
            absent_share=len(absent) / len(q_terms),
            n_docs=stats.n_docs,
            stats_source=stats.source,
        )

    # -- decision rules ------------------------------------------------------

    def _decide(
        self,
        plan: QueryPlan,
        evidence: EvidenceSet,
        items: list,
        signals: list[GateSignal],
        notes: list[str],
        fit: _VocabularyFit,
    ) -> EvidenceAssessment:
        by_name = {s.name: s for s in signals}
        count = len(items)
        docs = len({e.document_id for e in items})
        align = by_name["lexical_alignment"].value
        agree = by_name["independent_source_agreement"].value or 0.0
        prov_ok = bool(by_name["provenance_completeness"].passed)
        conflicts = [
            s for s in signals
            if s.name == "conflicting_values" and s.value and s.value > 0
        ]
        trust = by_name.get("source_trust")
        trust_low = bool(
            trust and trust.measured and trust.value is not None and trust.value < TRUST_LOW
        )

        def assessment(
            *,
            sufficient: bool,
            decision: GateDecision,
            confidence: ConfidenceCategory,
            code: str,
            reason: str,
            supporting: list[str],
            unsupported: list[str],
            missing: list[str],
            action: str,
        ) -> EvidenceAssessment:
            return EvidenceAssessment(
                sufficient=sufficient,
                decision=decision,
                confidence=confidence,
                reason_code=code,
                reason=reason,
                evidence_count=count,
                document_count=docs,
                supporting_evidence_ids=supporting,
                unsupported_aspects=unsupported,
                missing_information=missing,
                recommended_action=action,
                signals=signals,
                notes=notes,
            )

        supporting = [e.evidence_id for e in items]

        # 1. Nothing retrieved -> abstain.
        if count == 0:
            return assessment(
                sufficient=False,
                decision=GateDecision.ABSTAIN,
                confidence=ConfidenceCategory.NONE,
                code="NO_EVIDENCE",
                reason=(
                    "The knowledge base returned no evidence for this question, so no "
                    "answer can be grounded. Nothing was retrieved from the corpus."
                ),
                supporting=[],
                unsupported=self._aspects(plan),
                missing=self._aspects(plan),
                action="abstain",
            )

        # 2. Question has no content terms -> clarification instead of guessing.
        if not plan.extracted_terms:
            return assessment(
                sufficient=False,
                decision=GateDecision.ASK_CLARIFICATION,
                confidence=ConfidenceCategory.NONE,
                code="QUESTION_HAS_NO_CONTENT_TERMS",
                reason=(
                    "The question contains no content terms to search for. Please "
                    "rephrase with the specific concept, part or term you are asking about."
                ),
                supporting=supporting,
                unsupported=self._aspects(plan),
                missing=["a searchable term in the question itself"],
                action="ask_clarification",
            )

        # 3. Alignment below the floor -> the corpus does not discuss this.
        if align is None or align < MIN_ALIGNMENT_TO_PARTIAL:
            missing = self._aspects(plan)
            terms_list = ", ".join(sorted(set(plan.extracted_terms))[:6])
            return assessment(
                sufficient=False,
                decision=GateDecision.ABSTAIN,
                confidence=ConfidenceCategory.LOW,
                code="LOW_LEXICAL_ALIGNMENT",
                reason=(
                    "Evidence insufficient because the retrieved documents do not "
                    "contain the requested information: none of the question terms "
                    f"({terms_list}) appear in the "
                    "retrieved text. The knowledge base does not appear to cover this topic."
                ),
                supporting=supporting,
                unsupported=missing,
                missing=missing,
                action="abstain",
            )

        # 4. V20 — no retrieved passage is topically related to the question.
        #
        # Rule 3 above only asks whether the QUESTION's terms occur in the text
        # (fraction >= 0.15), which one corpus-ubiquitous word satisfies: "How many
        # moons does Jupiter have?" scored alignment = 1/3 = 0.333 purely because
        # "many" occurs in motor-industry prose. The retrieval score is the signal
        # that actually separates the two situations, and the gate has recorded it
        # in `score_distribution` since V7 without ever consulting it. Measured on
        # the protected Automobile corpus, the top cosine for 12 supported in-domain
        # questions was 0.4365..0.6946 and for 6 off-domain questions 0.2317..0.3411
        # — no overlap, with MIN_RELEVANCE_TO_ANSWER (0.35) inside the gap.
        relevance = by_name.get("evidence_relevance_floor")
        if relevance is None or not relevance.measured:
            notes.append(
                "relevance floor (V20): NOT APPLIED to this run — "
                + (
                    relevance.not_performed_reason
                    if relevance is not None and relevance.not_performed_reason
                    else "the retrieval score scale for this strategy is not an "
                    "absolute similarity"
                )
            )
        elif relevance.value is not None and relevance.value < self._min_relevance:
            top = relevance.value
            # Only terms that ACTUALLY occur in the retrieved text, so the reason
            # never claims a match that did not happen.
            evidence_text = " ".join(e.content.lower() for e in items)
            generic_hits = sorted(t for t in plan.extracted_terms if t in evidence_text)
            missing = self._aspects(plan)
            return assessment(
                sufficient=False,
                decision=GateDecision.ABSTAIN,
                confidence=ConfidenceCategory.LOW,
                code="LOW_EVIDENCE_RELEVANCE",
                reason=(
                    f"Evidence insufficient because none of the retrieved passages is "
                    f"topically related to the question: the best similarity of any of "
                    f"the {count} retrieved chunk(s) is {top:.4f} on the "
                    f"'{evidence.strategy}' scale, below the {self._min_relevance} "
                    f"relevance floor. "
                    + (
                        f"The word(s) {', '.join(generic_hits)} do appear in the "
                        "retrieved text, but a lexical match is not relevance: "
                        "matching them is not evidence that the corpus covers the "
                        "question. "
                        if generic_hits
                        else ""
                    )
                    + "The knowledge base does not appear to cover this topic."
                ),
                # These chunks were returned, but they do NOT ground the question;
                # listing them as supporting ids would repeat the conflation being
                # fixed here.
                supporting=[],
                unsupported=missing,
                missing=missing,
                action="abstain",
            )

        # 5. V20 — the corpus cannot represent the question at all.
        #
        # Scale-free, so it also protects strategies where rule 4 does not apply.
        # BOTH conditions are required, and that pairing is deliberate: a rule that
        # abstained on IDF alone was measured to abstain on "How do brakes work?"
        # and "Explain the steering system.", because question-framing words
        # ("work" 3.316, "explain" never in corpus) look informative while saying
        # nothing about the topic. Requiring that MOST of the question is absent
        # from the corpus AND that the corpus holds no rare term from it keeps those
        # false abstentions out while still proving Jupiter/lyrebird/buoyancy are
        # outside this corpus.
        if (
            fit.measured
            and fit.terms
            and fit.absent_share >= MIN_ABSENT_TERM_SHARE
            and not fit.corpus_present_distinctive
        ):
            missing = self._aspects(plan)
            notes.append(
                "corpus vocabulary fit (V20): "
                f"{len(fit.absent_from_corpus)}/{len(fit.terms)} question term(s) occur "
                f"in none of the {fit.n_docs} indexed chunk(s) and the corpus holds no "
                f"term from this question with IDF >= {MIN_DISTINCTIVE_IDF}; measured "
                f"from {fit.stats_source}."
            )
            return assessment(
                sufficient=False,
                decision=GateDecision.ABSTAIN,
                confidence=ConfidenceCategory.LOW,
                code="TOPIC_OUTSIDE_CORPUS",
                reason=(
                    f"Evidence insufficient because the requested topic is outside this "
                    f"corpus: {len(fit.absent_from_corpus)} of the question's "
                    f"{len(fit.terms)} content term(s) "
                    f"({', '.join(fit.absent_from_corpus)}) occur in none of the "
                    f"{fit.n_docs} indexed chunk(s), and no term of this question that "
                    f"the corpus does contain is distinctive enough to be a handle on "
                    f"it (all are common in this corpus, IDF < {MIN_DISTINCTIVE_IDF}). "
                    "The knowledge base does not cover this topic."
                ),
                supporting=[],
                unsupported=missing,
                missing=missing,
                action="abstain",
            )

        # 6. Partial coverage: some clause of the question has no support.
        unsupported, partial_ids = self._coverage(plan, items, align)
        if unsupported:
            return assessment(
                sufficient=False,
                decision=GateDecision.PARTIAL_ANSWER,
                confidence=ConfidenceCategory.MODERATE,
                code="PARTIAL_COVERAGE",
                reason=(
                    f"Partial evidence: {count} item(s) from {docs} document(s) support "
                    f"part of the question, but not: {('; '.join(unsupported))}. "
                    "The answer will state what is missing rather than fill it from "
                    "general knowledge."
                ),
                supporting=partial_ids or supporting,
                unsupported=unsupported,
                missing=unsupported,
                action="answer_partial",
            )

        # 7. Sufficient (confidence capped when sources conflict or trust is low).
        confidence = (
            ConfidenceCategory.HIGH
            if (align >= MIN_ALIGNMENT_TO_ANSWER and (agree >= 1 or docs >= 2) and prov_ok)
            else ConfidenceCategory.MODERATE
        )
        caps: list[str] = []
        if conflicts and confidence is ConfidenceCategory.HIGH:
            confidence = ConfidenceCategory.MODERATE
            caps.append("sources report conflicting values for the same quantity")
        if trust_low:
            confidence = (
                ConfidenceCategory.LOW
                if confidence is ConfidenceCategory.MODERATE
                else ConfidenceCategory.MODERATE
            )
            caps.append(f"mean source trust is below {TRUST_LOW} (low-trust evidence)")
        agreement_note = (
            f"supported by {docs} independent document(s)"
            if docs >= 2 or agree >= 1
            else "supported by a single document (no independent corroboration)"
        )
        prov_note = "complete provenance chain" if prov_ok else "partial provenance chain"
        cap_note = ("; WARNING: " + "; ".join(caps)) if caps else ""
        return assessment(
            sufficient=True,
            decision=GateDecision.ANSWER,
            confidence=confidence,
            code="SUFFICIENT",
            reason=(
                f"Evidence status: SUFFICIENT. Basis: {count} retrieved chunk(s) from "
                f"{docs} document(s); {len(plan.extracted_terms)}"
                f" question term(s) matched in the retrieved text "
                f"({round((align or 0) * 100)}% lexical coverage, HEURISTIC); "
                f"{agreement_note}; {prov_note}{cap_note}."
            ),
            supporting=supporting,
            unsupported=[],
            missing=[],
            action="answer",
        )

    # -- helpers ---------------------------------------------------------------

    def _aspects(self, plan: QueryPlan) -> list[str]:
        return _clause_aspects(plan)

    def _coverage(self, plan: QueryPlan, items: list, overall_align: float) -> tuple[list[str], list[str]]:
        """Per-aspect term coverage. Returns (unsupported aspects, supporting ids).

        Only splits for multi-clause questions; a single-clause question is
        covered whenever overall alignment passed the floor (handled by caller).
        """
        aspects = _clause_aspects(plan)
        if len(aspects) <= 1:
            return ([], [])
        evidence_text = " ".join(e.content.lower() for e in items)
        unsupported: list[str] = []
        supporting_ids: list[str] = []
        for aspect in aspects:
            terms = [t for t in extract_terms(aspect) if t not in {"compare", "versus", "vs"}]
            if not terms:
                continue
            hits = [e for e in items if any(t in e.content.lower() for t in terms)]
            coverage = sum(1 for t in terms if t in evidence_text) / len(terms)
            if coverage <= ASPECT_MIN_COVERAGE:
                unsupported.append(aspect)
            else:
                supporting_ids.extend(e.evidence_id for e in hits)
        return (unsupported, sorted(set(supporting_ids)))


def create_evidence_gate(
    stats_provider: TermStatsProvider | None = None,
    min_relevance: float = MIN_RELEVANCE_TO_ANSWER,
) -> GroundingGate:
    """Factory. Pass a corpus term-statistics provider to enable the V20 checks."""
    return HeuristicEvidenceGate(stats_provider, min_relevance)


def create_grounding_gate(
    stats_provider: TermStatsProvider | None = None,
    min_relevance: float = MIN_RELEVANCE_TO_ANSWER,
) -> GroundingGate:
    """Explicit alias for the grounding-gate role (same implementation)."""
    return HeuristicEvidenceGate(stats_provider, min_relevance)


__all__ = [
    "EvidenceGate",
    "GroundingGate",
    "HeuristicEvidenceGate",
    "MIN_ABSENT_TERM_SHARE",
    "MIN_DISTINCTIVE_IDF",
    "MIN_RELEVANCE_TO_ANSWER",
    "create_evidence_gate",
    "create_grounding_gate",
    "MIN_ALIGNMENT_TO_ANSWER",
    "MIN_ALIGNMENT_TO_PARTIAL",
]
