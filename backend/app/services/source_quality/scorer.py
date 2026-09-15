"""Source Quality Engine.

Produces EXPLAINABLE quality assessments. Every score comes with the signals,
weights, reasons, and warnings that produced it. The score is a transparent
heuristic — the UI must present it as an automated assessment, not truth.

Signals (all 0..1):
    authority, relevance, recency, source_type, accessibility, duplication,
    evidence_quality
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from app.schemas.models import (
    DomainSpec,
    QualityAssessment,
    QualitySignals,
    Source,
    SourceDecision,
    SourceType,
)

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "authority": 0.25,
    "relevance": 0.25,
    "recency": 0.10,
    "source_type": 0.15,
    "accessibility": 0.10,
    "duplication": 0.05,
    "evidence_quality": 0.10,
}

# Authority tiers by domain suffix / well-known hosts.
GOV_TLDS = (".gov", ".gov.uk", ".eu", ".int")
EDU_TLDS = (".edu", ".ac.uk", ".edu.au", ".ac.in")
ORG_TLDS = (".org",)
STANDARDS_BODIES = ("iso.org", "sae.org", "ieee.org", "astm.org", "nist.gov", "who.int", "unece.org")

# Marker phrases suggesting evidence/citations quality.
EVIDENCE_MARKERS = [
    "reference", "citation", "standard", "doi", "proceedings", "journal",
    "guideline", "specification", "dataset", "benchmark", "methodology",
]


class SourceQualityScorer:
    """Heuristic, transparent, modular scorer. Swap or subclass for experiments."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = DEFAULT_WEIGHTS if weights is None else weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Quality weights must sum to 1.0 (got {total})")

    def assess(
        self,
        source: Source,
        spec: DomainSpec | None = None,
        url_accessible: bool | None = None,
        seen_urls: set[str] | None = None,
        last_modified: datetime | None = None,
        http_status: int | None = None,
    ) -> QualityAssessment:
        reasons: list[str] = []
        warnings: list[str] = []

        authority = self._signal_authority(source, reasons)
        relevance = self._signal_relevance(source, spec, reasons)
        recency = self._signal_recency(source, reasons, warnings, last_modified=last_modified)
        stype = self._signal_source_type(source, reasons)
        accessibility = self._signal_accessibility(source, url_accessible, reasons, warnings, http_status=http_status)
        duplication = self._signal_duplication(source, seen_urls, reasons, warnings)
        evidence = self._signal_evidence(source, reasons)

        signals = QualitySignals(
            authority=authority,
            relevance=relevance,
            recency=recency,
            source_type=stype,
            accessibility=accessibility,
            duplication=duplication,
            evidence_quality=evidence,
        )

        score = sum(
            getattr(signals, name) * weight for name, weight in self.weights.items()
        )
        decision, threshold_reason = self._decide(score, signals, source)
        if threshold_reason:
            reasons.append(threshold_reason)

        return QualityAssessment(
            signals=signals,
            weights=dict(self.weights),
            score=round(score, 4),
            decision=decision,
            reasons=reasons,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Individual signals
    # ------------------------------------------------------------------

    def _signal_authority(self, source: Source, reasons: list[str]) -> float:
        host = (urlparse(source.url).hostname or "").lower()
        publisher = (source.publisher or "").lower()
        if any(sb in host or sb in publisher for sb in STANDARDS_BODIES):
            reasons.append(f"Host/publisher '{host or publisher}' is a recognized standards body")
            return 0.95
        if any(host.endswith(t) for t in GOV_TLDS):
            reasons.append(f"Government domain '{host}' (high institutional authority)")
            return 0.9
        if any(host.endswith(t) for t in EDU_TLDS):
            reasons.append(f"Academic domain '{host}' (peer/educational authority)")
            return 0.85
        if source.source_type == SourceType.ARXIV or "arxiv.org" in host:
            reasons.append("arXiv preprint server (research-grade, though not peer-reviewed)")
            return 0.75
        if any(host.endswith(t) for t in ORG_TLDS):
            reasons.append(f"Non-profit/organization domain '{host}'")
            return 0.6
        if host and re.search(r"\.(edu|gov)\.[a-z]{2}$", host):
            reasons.append(f"Academic/government domain '{host}'")
            return 0.85
        reasons.append(f"General web domain '{host}' (authority unknown)")
        return 0.35

    def _signal_relevance(self, source: Source, spec: DomainSpec | None, reasons: list[str]) -> float:
        if spec is None:
            reasons.append("No domain spec available; relevance estimated from title/URL only")
            text = f"{source.title or ''} {source.notes or ''} {source.url}".lower()
            return 0.5 if any(w in text for w in ("engineer", "research", "journal", "standard")) else 0.3

        domain_terms = set()
        for term in (
            spec.subdomains + spec.key_concepts + spec.entities + spec.terminology + [spec.domain]
        ):
            domain_terms.update(t.lower() for t in term.lower().split() if len(t) > 3)
        text = f"{source.title or ''} {source.notes or ''} {source.url}".lower()
        if not domain_terms:
            reasons.append("Domain spec had no extractable terms; relevance defaulted to 0.5")
            return 0.5
        hits = sum(1 for t in domain_terms if t in text)
        ratio = min(1.0, hits / max(3, len(domain_terms) * 0.05))
        reasons.append(
            f"Relevance: {hits} of {len(domain_terms)} domain-spec terms appear in title/notes/URL"
        )
        return round(ratio, 3)

    def _signal_recency(
        self,
        source: Source,
        reasons: list[str],
        warnings: list[str],
        last_modified: datetime | None = None,
    ) -> float:
        # Verified date (HTTP Last-Modified) takes precedence over text heuristics.
        if last_modified is not None:
            age_days = max(0, (datetime.now(timezone.utc) - last_modified).days)
            age_years = age_days / 365.25
            score = 1.0 if age_years <= 1 else max(0.1, 1.0 - age_years * 0.08)
            reasons.append(f"Server Last-Modified {last_modified.date().isoformat()} (age {age_years:.1f}y)")
            if age_years > 10:
                warnings.append(f"Content last modified {last_modified.year} may be outdated for fast-moving topics")
            return round(score, 3)
        m = re.search(r"(20\d{2})", f"{source.notes or ''} {source.title or ''}")
        if m:
            year = int(m.group(1))
            age = datetime.now(timezone.utc).year - year
            score = 1.0 if age <= 1 else max(0.1, 1.0 - age * 0.08)
            reasons.append(f"Content dated {year} (age {age}y, from title/notes)")
            if age > 10:
                warnings.append(f"Content from {year} may be outdated for fast-moving topics")
            return round(score, 3)
        if source.source_type == SourceType.ARXIV:
            reasons.append("arXiv source; publication year parsed from metadata when available")
            return 0.7
        warnings.append("No publication date found; recency unknown (default 0.5)")
        return 0.5

    def _signal_source_type(self, source: Source, reasons: list[str]) -> float:
        mapping = {
            SourceType.ARXIV: 0.8,
            SourceType.PDF: 0.65,
            SourceType.TEXT: 0.5,
            SourceType.WEB_PAGE: 0.45,
            SourceType.USER_UPLOAD: 0.5,
        }
        score = mapping.get(source.source_type, 0.4)
        reasons.append(f"Source type '{source.source_type.value}' baseline trust {score}")
        return score

    def _signal_accessibility(
        self,
        source: Source,
        url_accessible: bool | None,
        reasons: list[str],
        warnings: list[str],
        http_status: int | None = None,
    ) -> float:
        if source.notes and source.notes.startswith("INVALID"):
            warnings.append("URL is invalid and cannot be ingested")
            return 0.0
        if url_accessible is None:
            # Honest "unknown": neutral score AND a warning, so UIs can disclose
            # that accessibility was never verified.
            warnings.append("URL accessibility not verified (no probe performed)")
            return 0.5
        if url_accessible:
            if http_status is not None:
                reasons.append(f"URL reachable (HTTP {http_status})")
            else:
                reasons.append("URL reachable")
            return 1.0
        detail = f" (HTTP {http_status})" if http_status is not None else ""
        warnings.append(f"URL was unreachable during assessment{detail}")
        return 0.1

    def _signal_duplication(
        self, source: Source, seen_urls: set[str] | None, reasons: list[str], warnings: list[str]
    ) -> float:
        if seen_urls is None:
            return 1.0
        host = (urlparse(source.url).hostname or "").lower()
        path = urlparse(source.url).path.rstrip("/")
        similar = sum(1 for u in seen_urls if (urlparse(u).hostname or "").lower() == host)
        if similar > 0:
            warnings.append(f"{similar} other candidate(s) already from same host {host}")
            return round(max(0.3, 1.0 - 0.15 * similar), 3)
        reasons.append("First candidate from this host (no duplication detected)")
        return 1.0

    def _signal_evidence(self, source: Source, reasons: list[str]) -> float:
        text = f"{source.title or ''} {source.notes or ''} {source.url}".lower()
        hits = [m for m in EVIDENCE_MARKERS if m in text]
        score = min(1.0, 0.2 + 0.2 * len(hits))
        if hits:
            reasons.append(f"Evidence markers found: {', '.join(hits[:5])}")
        else:
            reasons.append("No explicit evidence markers in title/notes (may still be fine)")
        return round(score, 2)

    def _decide(
        self, score: float, signals: QualitySignals, source: Source
    ) -> tuple[SourceDecision, str]:
        if signals.accessibility == 0.0:
            return SourceDecision.REJECT, "Decision REJECT: URL invalid/unreachable"
        if score >= 0.70:
            return SourceDecision.ACCEPT, f"Decision ACCEPT: score {score:.2f} >= 0.70"
        if score >= 0.45:
            return SourceDecision.REVIEW, f"Decision REVIEW: score {score:.2f} in [0.45, 0.70)"
        return SourceDecision.REJECT, f"Decision REJECT: score {score:.2f} < 0.45"
