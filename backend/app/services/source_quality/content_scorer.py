"""Content-aware source quality scorer (v2).

Scores candidate sources by evaluating their ACTUAL CONTENT against explicit
domain knowledge requirements (Domain Knowledge Map), using the project's
existing embedding provider. Deterministic, explainable, evidence-backed.

Strict separation of concerns:
- SOURCE AUTHORITY (v1 signal, unchanged)
- CONTENT RELEVANCE (new: weighted per-requirement similarity of page content)
- DOMAIN COVERAGE (new: fraction of requirements with similarity >= threshold)
- ACCESSIBILITY (real probe, v1 behaviour)
- RECENCY (v1 signal, unchanged)

The composite weights content signals (0.60) above metadata signals (0.25).
The v1 scorer remains the default; this scorer is opt-in (SOURCE_SCORER_VERSION=v2).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from app.schemas.models import Source
from app.services.source_quality.scorer import SourceQualityScorer
from app.utils.url_validation import validate_public_http_url

logger = logging.getLogger(__name__)

# --- Stage B: content acquisition -------------------------------------------
MAX_CONTENT_BYTES = 256 * 1024
WINDOW_CHARS = 400
MIN_WINDOW_WORDS = 45  # minimum total words for a window to count as prose
MIN_WINDOW_WORDS_PER_LINE = 4.0  # link/nav lists average ~1 word per line; prose far more
WINDOW_OVERLAP = 80
TOP_K_WINDOWS = 3
MIN_WINDOWS_FOR_CONFIDENCE = 5
COVER_THRESHOLD = 0.50  # calibrated on the pool report (see docs/design-source-quality-v2.md)
SIM_NORM_OFFSET = 0.20  # raw MiniLM sim 0.2 maps to 0.0, so scores are comparable across sources

# --- Stage D: v2 composite weights -------------------------------------------
V2_WEIGHTS = {
    "content_relevance": 0.40,
    "domain_coverage": 0.20,
    "authority": 0.15,
    "accessibility": 0.10,
    "source_type": 0.05,
    "recency": 0.05,
    "duplication": 0.05,
}


def load_domain_map(domain: str) -> dict:
    """Load the Domain Knowledge Map for a domain (exact slug, else error).

    Maps are curated JSON files in domain_maps/ with explicit provenance.
    """
    slug = domain.lower().replace(" ", "_")
    path = Path(__file__).parent / "domain_maps" / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No Domain Knowledge Map for domain '{domain}' (expected {path.name}). "
            "v2 content scoring requires an explicit, reviewable map."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("requirement_areas"):
        raise ValueError(f"Domain map {path.name} has no requirement_areas")
    total = sum(a.get("weight", 0.0) for a in data["requirement_areas"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Domain map {path.name} area weights sum to {total}, expected 1.0")
    return data


@dataclass
class ContentAnalysis:
    content_relevance: float
    domain_coverage: float
    confidence: float
    analyzed_chars: int
    n_windows: int
    per_area: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


class ContentRelevanceScorer:
    """Stage C: embedding-based content relevance + coverage against a Domain Map.

    Each requirement area is embedded as MULTIPLE queries (its description plus
    every seed term separately). Short, sharp queries match far better with
    sentence embeddings than one long blended query; the area similarity is the
    mean of the top-3 (query, window) pairs across all of the area's queries.
    """

    def __init__(self, embedder, domain_map: dict) -> None:
        self._embedder = embedder
        self._map = domain_map
        self._areas = domain_map["requirement_areas"]
        # Embed each requirement area as several queries.
        self._area_queries: dict[str, list[tuple[str, list[float]]]] = {}
        for a in self._areas:
            query_texts = [a["description"]] + list(a["seed_terms"])
            vectors = self._embedder.embed_texts(query_texts)
            self._area_queries[a["id"]] = list(zip(query_texts, vectors))

    @staticmethod
    def _window(text: str, size: int = WINDOW_CHARS, overlap: int = WINDOW_OVERLAP) -> list[str]:
        step = max(1, size - overlap)
        windows = [text[i : i + size] for i in range(0, len(text), step)]
        # Keep prose-like windows only. Two discriminators: absolute word count
        # (filters tiny fragments) and words-per-line (filters link lists and
        # leftover navigation debris, which run ~1 word per line, while real
        # paragraphs run far denser).
        kept: list[str] = []
        for w in windows:
            words = len(w.split())
            lines = [ln for ln in w.split("\n") if ln.strip()]
            wpl = words / max(1, len(lines))
            if words >= MIN_WINDOW_WORDS and wpl >= MIN_WINDOW_WORDS_PER_LINE:
                kept.append(w)
        return kept

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def analyze_content(self, text: str) -> ContentAnalysis:
        limitations: list[str] = []
        text = (text or "").strip()
        if len(text) < 200:
            limitations.append(f"Very little content available ({len(text)} chars); relevance estimates unreliable")
            return ContentAnalysis(0.0, 0.0, 0.0, len(text), 0, [], limitations)

        windows = self._window(text)
        if not windows:
            limitations.append(
                f"No prose-like content windows survived filtering ({len(text)} chars of text, "
                "likely navigation/link debris); content signals unavailable"
            )
            return ContentAnalysis(0.0, 0.0, 0.0, len(text), 0, [], limitations)
        vectors = self._embedder.embed_texts(windows)
        norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in vectors]

        per_area: list[dict] = []
        covered = 0
        weighted_sum = 0.0
        total_weight = 0.0
        for a in self._areas:
            # Collect every (query, window) similarity pair for this area.
            pairs: list[tuple[float, int, str]] = []  # (sim, window_idx, query_kind)
            for query_text, q in self._area_queries[a["id"]]:
                kind = "description" if query_text == a["description"] else "term"
                qn = math.sqrt(sum(x * x for x in q)) or 1.0
                for wi, (v, n) in enumerate(zip(vectors, norms)):
                    raw = self._cos(q, v) / (n * qn)
                    pairs.append((max(0.0, raw - SIM_NORM_OFFSET) / (1 - SIM_NORM_OFFSET), wi, kind))
            pairs.sort(key=lambda p: -p[0])
            top_pairs = pairs[:TOP_K_WINDOWS]
            mean_top = sum(p[0] for p in top_pairs) / len(top_pairs)
            is_covered = mean_top >= COVER_THRESHOLD
            covered += 1 if is_covered else 0
            weighted_sum += a.get("weight", 0.0) * mean_top
            total_weight += a.get("weight", 0.0)
            evidence_queries = [p[2] for p in top_pairs]
            per_area.append(
                {
                    "area_id": a["id"],
                    "area_name": a["name"],
                    "similarity": round(mean_top, 4),
                    "covered": is_covered,
                    "threshold": COVER_THRESHOLD,
                    "evidence": windows[top_pairs[0][1]][:240],
                    "matched_by": evidence_queries,
                }
            )
        relevance = weighted_sum / total_weight if total_weight else 0.0
        coverage = covered / len(self._areas)

        # Confidence: content volume + margin of the areas around the threshold.
        margins = [abs(p["similarity"] - COVER_THRESHOLD) for p in per_area]
        margin_factor = min(1.0, (sum(margins) / len(margins)) / 0.15)
        volume_factor = min(1.0, len(text) / 5000)
        confidence = round(0.5 * margin_factor + 0.5 * volume_factor, 3)
        if len(windows) < MIN_WINDOWS_FOR_CONFIDENCE:
            limitations.append(
                f"Only {len(windows)} content windows analyzed; confidence reduced"
            )
        return ContentAnalysis(
            content_relevance=round(relevance, 4),
            domain_coverage=round(coverage, 4),
            confidence=confidence,
            analyzed_chars=len(text),
            n_windows=len(windows),
            per_area=per_area,
            limitations=limitations,
        )


def fetch_source_content(url: str, cache_dir: Path) -> tuple[str | None, int | None, str | None]:
    """Fetch (or load cached) page text for a candidate source.

    Returns (text, http_status, error). Never raises. SSRF-validated, size-capped.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # "v2" key namespace: the extraction pipeline changed (navbox stripping), so
    # cached v1 text must never be silently reused.
    key = hashlib.sha256(f"v2:{url}".encode("utf-8")).hexdigest()[:24]
    cache_file = cache_dir / f"{key}.txt"
    meta_file = cache_dir / f"{key}.meta.json"
    if cache_file.exists() and meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            return cache_file.read_text(encoding="utf-8"), meta.get("status"), None
        except Exception:
            pass  # fall through to refetch
    try:
        validated = validate_public_http_url(url)
        import httpx

        with httpx.Client(
            timeout=20.0, follow_redirects=True, headers={"User-Agent": "RAGForge/0.1 (+local research tool)"}
        ) as client:
            resp = client.get(validated)
            status = resp.status_code
            resp.raise_for_status()
            data = resp.content[:MAX_CONTENT_BYTES]
    except Exception as exc:  # noqa: BLE001 - fetch failure is data, not a crash
        return None, None, str(exc)[:200]

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    # Wikipedia nav infoboxes live in <table class="navbox"> etc. and are pure link
    # lists; without this they masquerade as domain coverage (e.g. "Transmission",
    # "Chassis" menu items on a brake page scoring as engine+transmission coverage).
    for sel in (
        "table.navbox",
        "table.vertical-navbox",
        "table.sidebar",
        "table.infobox",
        "table.metadata",
        "table.toc",
        "div.navbox",
        "div#toc",
        "div.thumb",
        "span.mw-editsection",
        "figure",
    ):
        for tag in soup.select(sel):
            tag.decompose()
    lines = [ln.strip() for ln in (soup.body or soup).get_text("\n").split("\n")]
    text = "\n".join(ln for ln in lines if ln)
    cache_file.write_text(text, encoding="utf-8")
    meta_file.write_text(json.dumps({"status": status, "url": url}), encoding="utf-8")
    return text, status, None


class ContentAwareSourceScorer:
    """v2 scorer: v1 metadata signals + new content signals, re-separated weights."""

    version = "heuristic-v2-content-aware"

    def __init__(self, embedder, domain_map: dict, weights: dict[str, float] | None = None) -> None:
        self._v1 = SourceQualityScorer()
        self._content = ContentRelevanceScorer(embedder, domain_map)
        self._weights = weights or V2_WEIGHTS
        total = sum(self._weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"v2 weights must sum to 1.0 (got {total})")

    def assess_with_content(
        self,
        source: Source,
        content: str | None,
        http_status: int | None = None,
        seen_urls: set[str] | None = None,
    ) -> dict:
        """Score one source. Returns a plain dict (not the v1 QualityAssessment model)
        because the structure differs (per-area detail, coverage, confidence)."""
        # --- v1 metadata signals (authority, recency, source_type, duplication) ---
        meta_reasons: list[str] = []
        authority = self._v1._signal_authority(source, meta_reasons)
        recency = self._v1._signal_recency(source, meta_reasons, [])
        stype = self._v1._signal_source_type(source, meta_reasons)
        duplication = self._v1._signal_duplication(source, seen_urls, meta_reasons, [])

        # --- accessibility (real probe result supplied by caller) ---
        if http_status is None:
            accessibility, acc_reason = 0.5, "URL accessibility not verified (no probe performed)"
        elif http_status < 400:
            accessibility, acc_reason = 1.0, f"URL reachable (HTTP {http_status})"
        else:
            accessibility, acc_reason = 0.1, f"URL unreachable (HTTP {http_status})"

        # --- content signals ---
        limitations: list[str] = []
        if content is None:
            limitations.append("Content could not be fetched; content signals are 0 and confidence is 0")
            content_rel, coverage, confidence, per_area = 0.0, 0.0, 0.0, []
            content_reason = "content unavailable"
        else:
            analysis = self._content.analyze_content(content)
            content_rel = analysis.content_relevance
            coverage = analysis.domain_coverage
            confidence = analysis.confidence
            per_area = analysis.per_area
            limitations = analysis.limitations
            covered_names = [p["area_name"] for p in per_area if p["covered"]]
            content_reason = (
                f"content relevance {content_rel:.3f}; covers {len(covered_names)}/{len(per_area)} "
                f"requirement areas: {', '.join(covered_names) if covered_names else 'none'}"
            )

        signals = {
            "content_relevance": content_rel,
            "domain_coverage": coverage,
            "authority": authority,
            "accessibility": accessibility,
            "source_type": stype,
            "recency": recency,
            "duplication": duplication,
        }
        score = sum(signals[k] * w for k, w in self._weights.items())

        reasons = [content_reason, acc_reason] + meta_reasons
        if limitations:
            reasons.append("limitations: " + "; ".join(limitations))

        return {
            "scorer_version": self.version,
            "signals": {k: round(v, 4) for k, v in signals.items()},
            "weights": dict(self._weights),
            "score": round(score, 4),
            "confidence": confidence,
            "per_requirement": per_area,
            "reasons": reasons,
            "limitations": limitations,
            "domain_map": {
                "domain": self._content._map["domain"],
                "version": self._content._map["version"],
                "provenance": self._content._map["provenance"],
            },
        }
