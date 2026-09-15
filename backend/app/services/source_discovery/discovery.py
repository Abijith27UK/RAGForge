"""Source discovery providers.

Each provider normalizes candidates to Source objects. Providers must be
surgical (no crawling) and handle network failures gracefully.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

import httpx

from app.schemas.models import Source, SourceType
from app.utils.ids import new_id
from app.utils.url_validation import InvalidURLError, validate_public_http_url

logger = logging.getLogger(__name__)

ARXIV_CATEGORY_HINTS = {
    "mechanical engineering": "cs.RO OR eess.SY OR cs.SY",
    "automotive": "cs.RO OR eess.SY OR cs.SY",
    "medicine": "q-bio OR stat.AP",
    "machine learning": "cs.LG OR stat.ML",
}


class DiscoveryError(RuntimeError):
    pass


class SourceDiscoveryProvider(ABC):
    name: str = "base"

    @abstractmethod
    def discover(self, kb_id: str, query: str, limit: int = 5) -> list[Source]:
        """Return normalized candidate sources for a query/domain."""


class UserURLProvider(SourceDiscoveryProvider):
    """User-provided URLs. Normalizes and validates them (SSRF-safe)."""

    name = "user-url"

    def discover(self, kb_id: str, query: str, limit: int = 5) -> list[Source]:
        """`query` here is a whitespace/comma separated list of URLs."""
        urls = [u.strip() for u in re.split(r"[\s,]+", query) if u.strip()]
        sources: list[Source] = []
        for url in urls[:limit]:
            try:
                validated = validate_public_http_url(url)
            except InvalidURLError as exc:
                logger.warning("Rejected URL %s: %s", url, exc)
                sources.append(
                    Source(
                        id=new_id("src"),
                        kb_id=kb_id,
                        url=url,
                        title=None,
                        source_type=SourceType.WEB_PAGE,
                        notes=f"INVALID: {exc}",
                    )
                )
                continue
            ext = validated.lower().split("?")[0].rsplit(".", 1)[-1]
            stype = SourceType.PDF if ext == "pdf" else SourceType.WEB_PAGE
            sources.append(
                Source(
                    id=new_id("src"),
                    kb_id=kb_id,
                    url=validated,
                    title=None,
                    source_type=stype,
                    discovered_via=self.name,
                )
            )
        return sources


class ArxivDiscoveryProvider(SourceDiscoveryProvider):
    """Discover research papers via the arXiv API (no API key needed)."""

    name = "arxiv"
    BASE = "http://export.arxiv.org/api/query"

    def discover(self, kb_id: str, query: str, limit: int = 5) -> list[Source]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, min(limit, 20)),
            "sortBy": "relevance",
        }
        try:
            resp = httpx.get(self.BASE, params=params, timeout=20.0, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("arXiv discovery failed: %s", exc)
            raise DiscoveryError(f"arXiv request failed: {exc}") from exc

        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            raise DiscoveryError(f"Could not parse arXiv response: {exc}") from exc

        sources: list[Source] = []
        for entry in root.findall("a:entry", ns):
            url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
            authors = [
                (a.findtext("a:name", default="", namespaces=ns) or "").strip()
                for a in entry.findall("a:author", ns)
            ]
            if not url:
                continue
            sources.append(
                Source(
                    id=new_id("src"),
                    kb_id=kb_id,
                    url=url,
                    title=re.sub(r"\s+", " ", title) or None,
                    source_type=SourceType.ARXIV,
                    publisher="arXiv",
                    discovered_via=self.name,
                    notes=(f"Published: {published[:10]} | Authors: {', '.join(authors[:3])} | {summary[:200]}"),
                )
            )
        return sources


def get_provider(name: str) -> SourceDiscoveryProvider:
    providers = {
        UserURLProvider.name: UserURLProvider(),
        ArxivDiscoveryProvider.name: ArxivDiscoveryProvider(),
    }
    provider = providers.get(name)
    if provider is None:
        raise DiscoveryError(f"Unknown discovery provider '{name}'. Available: {list(providers)}")
    return provider
