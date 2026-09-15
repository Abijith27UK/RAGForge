"""Lightweight URL accessibility probe for source quality assessment.

One small HEAD (GET fallback) request per candidate source, used to replace the
previous "accessibility unknown" default with an honest, verified signal and to
pick up Last-Modified for recency. Never raises: failures are data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from app.utils.url_validation import InvalidURLError, validate_public_http_url

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 6.0
USER_AGENT = "RAGForge/0.1 (+local research tool)"


@dataclass(frozen=True)
class ProbeResult:
    accessible: bool
    status_code: int | None = None
    last_modified: datetime | None = None
    error: str | None = None


def probe_url(url: str) -> ProbeResult:
    """HEAD with GET fallback. SSRF-validated. Returns a result, never raises."""
    try:
        validated = validate_public_http_url(url)
    except InvalidURLError as exc:
        return ProbeResult(accessible=False, error=f"invalid URL: {exc}")

    try:
        with httpx.Client(
            timeout=PROBE_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.head(validated)
            if resp.status_code in (405, 501) or resp.status_code >= 400:
                # Some servers reject HEAD; try a ranged GET (cheap body).
                resp = client.get(validated, headers={"Range": "bytes=0-1023"})
        last_modified: datetime | None = None
        lm_raw = resp.headers.get("last-modified")
        if lm_raw:
            try:
                last_modified = parsedate_to_datetime(lm_raw)
                if last_modified.tzinfo is None:
                    last_modified = last_modified.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                last_modified = None
        return ProbeResult(
            accessible=resp.status_code < 400,
            status_code=resp.status_code,
            last_modified=last_modified,
        )
    except httpx.HTTPError as exc:
        logger.info("Accessibility probe failed for %s: %s", url, exc)
        return ProbeResult(accessible=False, error=str(exc)[:200])
    except Exception as exc:  # noqa: BLE001 - probe must never break scoring
        logger.warning("Unexpected probe error for %s: %s", url, exc)
        return ProbeResult(accessible=False, error=str(exc)[:200])
