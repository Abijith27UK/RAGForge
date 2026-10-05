"""Phase 13 — claim extraction, citation validation, grounding validation.

Pipeline: GeneratedAnswer -> ClaimExtractor -> CitationValidator ->
GroundingValidator -> AnswerPolicy -> FinalAnswer (schemas.answer.Answer).

What is VALIDATED deterministically (IMPLEMENTED):
* every citation id resolves to an evidence item that was actually retrieved in
  THIS answer's evidence set (catches fabricated, deleted and wrong-KB ids);
* every evidence id on every claim resolves the same way;
* provenance chain completeness (document id + title + content hash);
* lexical support: the cited evidence text shares terms with the claim
  (HEURISTIC — labelled `support_check = "lexical_overlap"`).

What is NOT validated (NOT IMPLEMENTED, and never implied):
* semantic entailment / whether the source truly proves the claim. No LLM or
  NLI model is involved, so nothing here may say "verified" or "AI-checked".
  The claim's `support_check` stays `not_performed` when no heuristic applies.

Failure actions are RECORDED, never silent: the pipeline reports which action
it took per problem (remove | downgrade | abstain | regenerate).
"""
from __future__ import annotations

import logging
import re

from app.schemas.answer import (
    Answer,
    AnswerMode,
    AnswerPolicy,
    AnswerStatus,
    Citation,
    Claim,
    ClaimDraft,
    ClaimType,
    ConfidenceCategory,
    EvidenceSet,
    GeneratedAnswer,
    SupportStatus,
)
from app.utils.ids import new_id

logger = logging.getLogger(__name__)

#: Minimum fraction of claim content terms that must appear in the cited
#: evidence for the claim to count as lexically supported (HEURISTIC).
MIN_LEXICAL_SUPPORT = 0.34


class ValidationAction:
    """Recorded outcome of one validation problem."""

    OK = "ok"
    REMOVE_CLAIM = "remove_claim"
    DOWNGRADE = "downgrade"
    ABSTAIN = "abstain"
    REGENERATE = "regenerate"


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]


class ClaimExtractor:
    """GeneratedAnswer -> list[Claim] drafts with ids (deterministic)."""

    def extract(self, generated: GeneratedAnswer) -> list[Claim]:
        claims: list[Claim] = []
        for i, draft in enumerate(generated.claims, start=1):
            claims.append(
                Claim(
                    claim_id=f"cl_{i:03d}",
                    text=draft.text.strip(),
                    claim_type=draft.claim_type,
                    evidence_ids=[e for e in draft.citation_evidence_ids],
                    citation_ids=[],
                )
            )
        # A generated text with no structured claims still becomes ONE claim so
        # nothing the model said escapes claim-level validation.
        if generated.text and not claims:
            claims.append(
                Claim(
                    claim_id="cl_001",
                    text=generated.text.strip(),
                    claim_type=ClaimType.FACT,
                    evidence_ids=[],
                    citation_ids=[],
                )
            )
        return claims


class CitationValidator:
    """Deterministic citation checks against THIS answer's evidence set."""

    def validate(
        self, claims: list[Claim], evidence: EvidenceSet, *, kb_id: str = ""
    ) -> tuple[list[Citation], list[dict]]:
        """Returns (citations, problems). Problems carry {claim_id, action, reason}.

        `kb_id` — when given, an evidence item whose recorded kb_id differs is
        rejected (wrong-KB citation), which matters whenever evidence can enter
        a pipeline from anywhere other than this KB's own retrieval run.
        """
        by_id = {e.evidence_id: e for e in evidence.items}
        citations: list[Citation] = []
        problems: list[dict] = []
        seen: dict[str, Citation] = {}

        for claim in claims:
            valid_evidence: list[str] = []
            for eid in claim.evidence_ids:
                item = by_id.get(eid)
                if item is None:
                    problems.append(
                        {
                            "claim_id": claim.claim_id,
                            "action": ValidationAction.REMOVE_CLAIM,
                            "reason": (
                                f"citation refers to evidence id {eid!r} that was not "
                                "retrieved for this answer (fabricated, deleted, or "
                                "from another knowledge base)"
                            ),
                        }
                    )
                    continue
                if kb_id and item.kb_id and item.kb_id != kb_id:
                    problems.append(
                        {
                            "claim_id": claim.claim_id,
                            "action": ValidationAction.REMOVE_CLAIM,
                            "reason": (
                                f"citation refers to evidence from knowledge base "
                                f"{item.kb_id!r}, not the selected KB {kb_id!r} "
                                f"(wrong-KB evidence id {eid})"
                            ),
                        }
                    )
                    continue
                if not item.content_hash or not item.document_id:
                    problems.append(
                        {
                            "claim_id": claim.claim_id,
                            "action": ValidationAction.DOWNGRADE,
                            "reason": (
                                f"evidence {eid} has an incomplete provenance chain "
                                "(missing document id or content hash)"
                            ),
                        }
                    )
                cid = f"cite_{eid}"
                if cid not in seen:
                    section_path = item.section_path or item.section
                    citation = Citation(
                        citation_id=cid,
                        evidence_id=eid,
                        chunk_id=item.chunk_id,
                        document_id=item.document_id,
                        # source_title falls back to the document id so a source
                        # card always has SOMETHING identifying it; it is never
                        # invented from the content.
                        source_title=item.title or item.document_id,
                        source_type=item.source_type,
                        title=item.title,
                        page=item.page,
                        # page_number/slide_number duplicate `page`/`slide` under
                        # the names the chat API exposes. Both are copied from the
                        # same measurement; neither is ever inferred.
                        page_number=item.page,
                        slide=item.slide,
                        slide_number=item.slide,
                        section=section_path,
                        section_path=section_path,
                        content_hash=item.content_hash,
                        url=item.url,
                        publisher=item.publisher,
                        snippet=item.content[:240],
                        validation="provenance_valid",
                        validation_detail=(
                            "id resolves to evidence retrieved for this answer and the "
                            "provenance chain exists (KB→document→chunk→content hash). "
                            "Semantic support check: NOT PERFORMED."
                        ),
                    )
                    seen[cid] = citation
                    citations.append(citation)
                valid_evidence.append(eid)

            invalid = [eid for eid in claim.evidence_ids if eid not in valid_evidence]
            if claim.evidence_ids and not valid_evidence:
                # Every citation it tried was invalid: the claim has no ground.
                claim.support_status = SupportStatus.UNSUPPORTED
                claim.support_note = "all citations failed validation"
            elif valid_evidence:
                claim.citation_ids = [f"cite_{e}" for e in valid_evidence]
                self._lexical_support(claim, by_id, valid_evidence)
                if invalid:
                    # The claim cited at least one id that does not resolve. It
                    # may still be grounded in the ids that DO resolve, but it
                    # must not be presented as fully supported: the generator
                    # demonstrably invented part of its provenance.
                    if claim.support_status is SupportStatus.SUPPORTED:
                        claim.support_status = SupportStatus.PARTIALLY_SUPPORTED
                    claim.support_note = (
                        f"{claim.support_note}; WARNING: {len(invalid)} citation(s) "
                        f"did not resolve to retrieved evidence "
                        f"({', '.join(invalid)}) and were dropped"
                    ).strip("; ")
            else:
                claim.support_status = SupportStatus.UNSUPPORTED
                claim.support_note = "claim carries no evidence id"
                problems.append(
                    {
                        "claim_id": claim.claim_id,
                        "action": ValidationAction.REMOVE_CLAIM,
                        "reason": "claim cites no evidence",
                    }
                )
        return citations, problems

    def _lexical_support(self, claim: Claim, by_id: dict, valid: list[str]) -> None:
        """HEURISTIC term-overlap support check — explicitly not entailment."""
        claim_terms = set(_terms(claim.text))
        if not claim_terms:
            claim.support_status = SupportStatus.PARTIALLY_SUPPORTED
            claim.support_check = "lexical_overlap"
            claim.support_note = "claim has no content terms to check"
            return
        evidence_text = " ".join(by_id[e].content.lower() for e in valid)
        hits = sum(1 for t in claim_terms if t in evidence_text)
        coverage = hits / len(claim_terms)
        if coverage >= MIN_LEXICAL_SUPPORT:
            claim.support_status = SupportStatus.SUPPORTED
            claim.support_check = "lexical_overlap"
            claim.support_note = (
                f"HEURISTIC lexical overlap {coverage:.0%} with cited evidence; "
                "semantic entailment NOT PERFORMED"
            )
        else:
            claim.support_status = SupportStatus.PARTIALLY_SUPPORTED
            claim.support_check = "lexical_overlap"
            claim.support_note = (
                f"weak lexical overlap ({coverage:.0%}) between claim and cited "
                "evidence; semantic entailment NOT PERFORMED — human review advised"
            )


class GroundingValidator:
    """Applies the AnswerPolicy to claim/citation results and produces the
    final status, recording the action taken."""

    def apply(
        self,
        *,
        generated: GeneratedAnswer,
        claims: list[Claim],
        citations: list[Citation],
        problems: list[dict],
        evidence: EvidenceSet,
        policy: AnswerPolicy,
        assessment,
        generator_name: str,
        generator_model: str,
        is_mock: bool,
        question: str,
        answer_id: str,
        trace_id: str,
        retrieval_run_id: str | None,
    ) -> tuple[Answer, list[str]]:
        actions: list[str] = []
        warnings: list[str] = list(generated.warnings)
        # Surface gate-detected conflicts as answer-level warnings (Phase 12
        # rule 7: if sources disagree, report the disagreement).
        for note in getattr(assessment, "notes", []) or []:
            if note.startswith("conflicting evidence:"):
                warnings.append(
                    f"sources disagree — {note.split(':', 1)[1].strip()} "
                    "(HEURISTIC numerical detection; reported, not resolved)"
                )

        # --- abstention paths ------------------------------------------------
        # NOTE: only ABSTAIN abstains here. PARTIAL_ANSWER falls through, so the
        # pipeline answers what the evidence supports and states what is missing
        # rather than refusing the whole question (spec case 6).
        if generated.abstain_requested or assessment.decision.value == "ABSTAIN" or not evidence.items:
            reason = (
                generated.abstention_reason
                or assessment.reason
                or "No grounded answer could be produced."
            )
            answer = self._build(
                status=AnswerStatus.ABSTAINED,
                text=f"I don't have enough evidence to answer this question. {reason}".strip(),
                claims=[],
                citations=[],
                answer_id=answer_id,
                question=question,
                trace_id=trace_id,
                retrieval_run_id=retrieval_run_id,
                assessment=assessment,
                policy=policy,
                generator_name=generator_name,
                generator_model=generator_model,
                is_mock=is_mock,
                confidence=ConfidenceCategory.NONE,
                confidence_basis="abstained: the evidence gate did not permit an answer",
                notes=generated.notes,
                warnings=warnings,
            )
            actions.append(f"{ValidationAction.ABSTAIN}: {reason}")
            return answer, actions

        if assessment.decision.value == "ASK_CLARIFICATION":
            answer = self._build(
                status=AnswerStatus.CLARIFICATION_REQUIRED,
                text=assessment.reason,
                claims=[],
                citations=[],
                answer_id=answer_id,
                question=question,
                trace_id=trace_id,
                retrieval_run_id=retrieval_run_id,
                assessment=assessment,
                policy=policy,
                generator_name=generator_name,
                generator_model=generator_model,
                is_mock=is_mock,
                confidence=ConfidenceCategory.NONE,
                confidence_basis="clarification requested before answering",
                notes=generated.notes,
                warnings=warnings,
            )
            actions.append(f"{ValidationAction.ABSTAIN}: clarification required")
            return answer, actions

        # --- generation failure ---------------------------------------------
        if not generated.text.strip() and not generated.claims and not generated.abstain_requested:
            answer = self._build(
                status=AnswerStatus.GENERATION_FAILED,
                text=(
                    "Answer generation produced no output. The retrieved evidence is "
                    "listed below for manual inspection."
                ),
                claims=[],
                citations=[],
                answer_id=answer_id,
                question=question,
                trace_id=trace_id,
                retrieval_run_id=retrieval_run_id,
                assessment=assessment,
                policy=policy,
                generator_name=generator_name,
                generator_model=generator_model,
                is_mock=is_mock,
                confidence=ConfidenceCategory.NONE,
                confidence_basis="generation failed: no output produced",
                notes=generated.notes,
                warnings=warnings,
            )
            actions.append(f"{ValidationAction.REGENERATE}: empty generator output (attempts exhausted)")
            return answer, actions

        # --- claim-level repair ---------------------------------------------
        problem_by_claim: dict[str, list[dict]] = {}
        for p in problems:
            problem_by_claim.setdefault(p["claim_id"], []).append(p)

        surviving: list[Claim] = []
        removed = 0
        for claim in claims:
            probs = problem_by_claim.get(claim.claim_id, [])
            fatal = [p for p in probs if p["action"] == ValidationAction.REMOVE_CLAIM]
            if fatal and policy.unsupported_claim_action == "remove":
                removed += 1
                actions.append(f"{ValidationAction.REMOVE_CLAIM} {claim.claim_id}: {fatal[0]['reason']}")
                continue
            if probs and policy.unsupported_claim_action == "abstain":
                answer = self._build(
                    status=AnswerStatus.ABSTAINED,
                    text="I don't have enough evidence to answer this question reliably.",
                    claims=[],
                    citations=[],
                    answer_id=answer_id,
                    question=question,
                    trace_id=trace_id,
                    retrieval_run_id=retrieval_run_id,
                    assessment=assessment,
                    policy=policy,
                    generator_name=generator_name,
                    generator_model=generator_model,
                    is_mock=is_mock,
                    confidence=ConfidenceCategory.NONE,
                    confidence_basis="abstained: unsupported claim and policy=abstain",
                    notes=generated.notes,
                    warnings=warnings,
                )
                actions.append(f"{ValidationAction.ABSTAIN} {claim.claim_id}: {probs[0]['reason']}")
                return answer, actions
            # Record what ACTUALLY happened to this claim, not what the problem
            # would have caused under a different policy. Under `downgrade` the
            # claim is KEPT, so logging `remove_claim` here would be a lie.
            downgraded = False
            if fatal and policy.unsupported_claim_action == "downgrade":
                claim.support_status = SupportStatus.PARTIALLY_SUPPORTED
                downgraded = True
                actions.append(
                    f"{ValidationAction.DOWNGRADE} {claim.claim_id}: "
                    f"{fatal[0]['reason']} (kept as partially supported, policy=downgrade)"
                )
            for p in probs:
                if downgraded and p["action"] == ValidationAction.REMOVE_CLAIM:
                    continue  # already reported above as a downgrade.
                actions.append(f"{p['action']} {claim.claim_id}: {p['reason']}")
            if (
                not downgraded
                and claim.support_status == SupportStatus.UNSUPPORTED
                and policy.unsupported_claim_action == "downgrade"
            ):
                claim.support_status = SupportStatus.PARTIALLY_SUPPORTED
                actions.append(
                    f"{ValidationAction.DOWNGRADE} {claim.claim_id}: marked partially supported"
                )
            surviving.append(claim)

        if removed:
            warnings.append(
                f"{removed} claim(s) were removed because their citations failed validation"
            )

        if not surviving:
            answer = self._build(
                status=AnswerStatus.ABSTAINED,
                text=(
                    "I don't have enough evidence to answer this question: every "
                    "generated claim failed citation validation and was removed."
                ),
                claims=[],
                citations=[],
                answer_id=answer_id,
                question=question,
                trace_id=trace_id,
                retrieval_run_id=retrieval_run_id,
                assessment=assessment,
                policy=policy,
                generator_name=generator_name,
                generator_model=generator_model,
                is_mock=is_mock,
                confidence=ConfidenceCategory.NONE,
                confidence_basis="all claims removed by citation validation",
                notes=generated.notes,
                warnings=warnings,
            )
            actions.append(f"{ValidationAction.ABSTAIN}: no claim survived validation")
            return answer, actions

        # --- status ------------------------------------------------------------
        has_missing = any(c.claim_type is ClaimType.MISSING_INFORMATION for c in surviving)
        all_grounded = all(c.evidence_ids for c in surviving)
        partial_gate = assessment.decision.value == "PARTIAL_ANSWER"
        weak = any(c.support_status is not SupportStatus.SUPPORTED for c in surviving)

        if partial_gate or has_missing or not all_grounded or weak:
            status = AnswerStatus.PARTIAL
        else:
            status = AnswerStatus.GROUNDED

        # Confidence: categorical and derived ONLY from recorded facts.
        if status is AnswerStatus.GROUNDED and assessment.confidence is ConfidenceCategory.HIGH:
            confidence = ConfidenceCategory.HIGH
            basis = assessment.reason
        elif status is AnswerStatus.PARTIAL:
            confidence = ConfidenceCategory.MODERATE
            basis = (
                "PARTIAL: some claims are only weakly supported lexically or parts "
                "of the question are uncovered; semantic check not performed"
            )
        else:
            confidence = assessment.confidence
            basis = assessment.reason

        text = generated.text.strip() or " ".join(c.text for c in surviving)
        answer = self._build(
            status=status,
            text=text,
            claims=surviving,
            citations=citations,
            answer_id=answer_id,
            question=question,
            trace_id=trace_id,
            retrieval_run_id=retrieval_run_id,
            assessment=assessment,
            policy=policy,
            generator_name=generator_name,
            generator_model=generator_model,
            is_mock=is_mock,
            confidence=confidence,
            confidence_basis=basis,
            notes=generated.notes,
            warnings=warnings,
        )
        # Drop citations no surviving claim references.
        referenced = {cid for c in surviving for cid in c.citation_ids}
        answer.citations = [c for c in citations if c.citation_id in referenced]
        answer.evidence_ids = sorted({e for c in surviving for e in c.evidence_ids} & set(
            e.evidence_id for e in (evidence.items)
        ))
        actions.append(f"status={status.value}; {len(surviving)} claim(s), {len(answer.citations)} citation(s)")
        return answer, actions

    def _build(self, **kw) -> Answer:
        # Translate the pipeline's internal kwarg names to the public Answer
        # fields (Pydantic would otherwise drop the unknown extras silently).
        if "generator_name" in kw:
            kw["generated_by"] = kw.pop("generator_name")
        if "generator_model" in kw:
            kw["model"] = kw.pop("generator_model")
        return Answer(**kw)


def validate_answer(
    generated: GeneratedAnswer,
    evidence: EvidenceSet,
    policy: AnswerPolicy,
    assessment,
    *,
    question: str,
    answer_id: str,
    trace_id: str,
    retrieval_run_id: str | None,
    generator_name: str,
    generator_model: str,
    is_mock: bool,
    kb_id: str = "",
) -> tuple[Answer, list[str], list[dict]]:
    """Run the full validation pipeline. Returns (answer, actions, problems)."""
    claims = ClaimExtractor().extract(generated)
    citations, problems = CitationValidator().validate(claims, evidence, kb_id=kb_id)
    answer, actions = GroundingValidator().apply(
        generated=generated,
        claims=claims,
        citations=citations,
        problems=problems,
        evidence=evidence,
        policy=policy,
        assessment=assessment,
        generator_name=generator_name,
        generator_model=generator_model,
        is_mock=is_mock,
        question=question,
        answer_id=answer_id,
        trace_id=trace_id,
        retrieval_run_id=retrieval_run_id,
    )
    return answer, actions, problems


__all__ = [
    "CitationValidator",
    "ClaimExtractor",
    "GroundingValidator",
    "MIN_LEXICAL_SUPPORT",
    "ValidationAction",
    "validate_answer",
]
