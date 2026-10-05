"""Phase 11 — evidence sufficiency / grounding gate.

The gate answers ONE question: may this question be answered from this evidence,
and if not, what should happen instead? It is deterministic (no LLM) and every
decision is explained by recorded `GateSignal`s.

Signals used (all measured, all disclosed):
* evidence_count / document_count   — how much came back, from how many docs
* score distribution                — top/mean retrieval scores (strategy scale,
                                      recorded with the strategy name)
* lexical alignment                 — fraction of question content terms found
                                      in the evidence text (HEURISTIC)
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
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

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

# Thresholds: explicit constants so tests and docs can reference them.
MIN_ALIGNMENT_TO_ANSWER = 0.34      # >= of question terms found in evidence
MIN_ALIGNMENT_TO_PARTIAL = 0.15     # below this we do not answer at all
ASPECT_MIN_COVERAGE = 0.5           # a clause counts as covered only when a
                                    # STRICT MAJORITY of its terms appear in
                                    # evidence (prefer understating)
TOP_SCORE_MODERATE = 0.35           # dense/cosine-ish scale; see score notes
INDEPENDENT_SOURCES_STRONG = 2      # documents agreeing on a key term
TRUST_LOW = 0.3                     # mean recorded trust below this = low-trust evidence

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


class EvidenceGate(ABC):
    """Decides whether evidence may support an answer, and why.

    Kept as the pipeline-facing interface (one method, `assess`) so existing
    callers and tests are unaffected. `GroundingGate` below is the same
    capability expressed with the V7 five-state vocabulary.
    """

    name: str = "base"

    @abstractmethod
    def assess(self, plan: QueryPlan, evidence: EvidenceSet) -> EvidenceAssessment: ...


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

        assessment = self.assess(plan, evidence)
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

    def assess(self, plan: QueryPlan, evidence: EvidenceSet) -> EvidenceAssessment:
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
        return self._decide(plan, evidence, items, signals, notes)

    # -- decision rules ------------------------------------------------------

    def _decide(
        self,
        plan: QueryPlan,
        evidence: EvidenceSet,
        items: list,
        signals: list[GateSignal],
        notes: list[str],
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

        # 4. Partial coverage: some clause of the question has no support.
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

        # 5. Sufficient (confidence capped when sources conflict or trust is low).
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


def create_evidence_gate() -> GroundingGate:
    return HeuristicEvidenceGate()


def create_grounding_gate() -> GroundingGate:
    """Explicit alias for the grounding-gate role (same implementation)."""
    return HeuristicEvidenceGate()


__all__ = [
    "EvidenceGate",
    "GroundingGate",
    "HeuristicEvidenceGate",
    "create_evidence_gate",
    "create_grounding_gate",
    "MIN_ALIGNMENT_TO_ANSWER",
    "MIN_ALIGNMENT_TO_PARTIAL",
]
