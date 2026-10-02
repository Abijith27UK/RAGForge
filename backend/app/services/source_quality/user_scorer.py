"""User-provided source integrity model (V4 Phase 4).

The external `SourceQualityScorer` measures PUBLICATION signals: domain
authority, HTTP accessibility, publication date, evidence markers. None of
those exist for a lecture PDF a student uploaded at 2am, so applying it to user
files would systematically reject the material the user cares most about.

This scorer therefore asks a different question:

    "Is this file real, intact, and did we actually get text out of it?"

It deliberately contains NO authority, recency, accessibility or evidence
weights. A user-uploaded document is never rejected for lacking public
authority signals; the uploader asserting relevance IS the relevance signal.
Integrity checks that still apply: file validity, parser success, duplicate
detection, corruption detection, content-extraction validation.
"""
from __future__ import annotations

import logging

from app.schemas.models import QualityAssessment, QualitySignals, SourceDecision

logger = logging.getLogger(__name__)

#: Weights sum to exactly 1.0 and cover ONLY integrity signals.
USER_WEIGHTS: dict[str, float] = {
    "file_validity": 0.30,
    "content_extraction": 0.25,
    "user_relevance": 0.20,
    "duplication": 0.15,
    "structure": 0.10,
}

#: Below this many extracted characters we stop trusting an "uploaded" file.
MIN_MEANINGFUL_CHARS = 200

#: Extraction quality saturates at this many characters.
COMFORTABLE_CHARS = 20_000


class UserProvidedIntegrityScorer:
    """Explainable integrity assessment for user-uploaded documents."""

    assessed_by = "user-integrity-v1"

    def assess(
        self,
        *,
        file_valid: bool,
        parse_ok: bool,
        extracted_chars: int,
        duplicate_of: str | None = None,
        unit_count: int | None = None,
        parse_error: str = "",
        warnings: list[str] | None = None,
    ) -> QualityAssessment:
        reasons: list[str] = []
        notes: list[str] = list(warnings or [])

        # -- file validity / corruption ------------------------------------
        if not file_valid:
            file_validity = 0.0
            reasons.append("File failed validation (unsupported type, corrupt container or empty)")
            if parse_error:
                notes.append(parse_error)
        elif parse_ok:
            file_validity = 1.0
            reasons.append("File validated and the parser opened it successfully")
        else:
            file_validity = 0.4
            reasons.append("File validated but the parser could not read its contents")
            if parse_error:
                notes.append(parse_error)

        # -- content extraction --------------------------------------------
        if parse_ok and extracted_chars <= 0:
            content_extraction = 0.0
            reasons.append("Parser succeeded but produced no text (likely a scanned/image document)")
            notes.append("No OCR: scanned documents are not supported")
        elif parse_ok and extracted_chars < MIN_MEANINGFUL_CHARS:
            content_extraction = round(extracted_chars / MIN_MEANINGFUL_CHARS, 3)
            reasons.append(
                f"Only {extracted_chars} characters extracted — below the "
                f"{MIN_MEANINGFUL_CHARS}-character minimum for a useful document"
            )
            notes.append("Very little text: this file may add little to the knowledge base")
        else:
            content_extraction = 1.0
            reasons.append(f"{extracted_chars} characters of text extracted")

        # -- structure (pages / slides / headings) -------------------------
        if unit_count:
            structure = 1.0
            reasons.append(f"Recovered {unit_count} structural unit(s) (page/slide/section)")
        elif parse_ok:
            structure = 0.5
            reasons.append("No pages/slides/headings were recovered (flat text)")
        else:
            structure = 0.0

        # -- duplication ----------------------------------------------------
        if duplicate_of:
            duplication = 0.0
            reasons.append(f"Identical content already exists in this knowledge base ({duplicate_of})")
            notes.append("Duplicate: the existing document was kept, nothing was replaced")
        else:
            duplication = 1.0
            reasons.append("No document in this knowledge base has the same content hash")

        # -- relevance asserted by the uploader ----------------------------
        user_relevance = 1.0
        reasons.append("Relevance asserted by the uploader (user-provided material)")

        signals = QualitySignals(
            # External-web signals are set to neutral, NOT zero: they are not
            # applicable here and must never drag the score down.
            authority=0.5,
            relevance=1.0,
            recency=0.5,
            source_type=0.5,
            accessibility=1.0,
            duplication=duplication,
            evidence_quality=0.5,
            file_validity=file_validity,
            content_extraction=content_extraction,
            structure=structure,
            user_relevance=user_relevance,
        )

        score = sum(getattr(signals, name) * weight for name, weight in USER_WEIGHTS.items())
        decision, decision_reason = self._decide(signals)
        reasons.append(decision_reason)

        return QualityAssessment(
            signals=signals,
            weights=dict(USER_WEIGHTS),
            score=round(score, 4),
            decision=decision,
            reasons=reasons,
            warnings=notes,
            assessed_by=self.assessed_by,
        )

    @staticmethod
    def _decide(signals: QualitySignals) -> tuple[SourceDecision, str]:
        """Integrity gate. Only hard technical failures block a user file.

        Thin content is NOT a rejection reason: a one-paragraph class note is a
        perfectly legitimate document. It is reported as a warning so the user
        can see why the file contributes little, without downgrading material
        they deliberately supplied.
        """
        if signals.file_validity <= 0.0:
            return SourceDecision.REJECT, "Decision REJECT: file failed validation"
        if signals.content_extraction <= 0.0:
            return SourceDecision.REJECT, "Decision REJECT: no text could be extracted"
        if signals.file_validity < 0.5:
            return (
                SourceDecision.REVIEW,
                "Decision REVIEW: the parser could not read the file contents",
            )
        if signals.duplication == 0.0:
            return SourceDecision.ACCEPT, "Decision ACCEPT: valid file, but identical content already exists"
        if signals.content_extraction < 1.0:
            return SourceDecision.ACCEPT, (
                "Decision ACCEPT: valid file with little extractable text "
                "(review the warning before relying on it)"
            )
        return SourceDecision.ACCEPT, "Decision ACCEPT: valid, intact, and text was extracted"


def assess_user_upload(
    *,
    file_valid: bool,
    parse_ok: bool,
    extracted_chars: int,
    duplicate_of: str | None = None,
    unit_count: int | None = None,
    parse_error: str = "",
    warnings: list[str] | None = None,
) -> QualityAssessment:
    return UserProvidedIntegrityScorer().assess(
        file_valid=file_valid,
        parse_ok=parse_ok,
        extracted_chars=extracted_chars,
        duplicate_of=duplicate_of,
        unit_count=unit_count,
        parse_error=parse_error,
        warnings=warnings,
    )