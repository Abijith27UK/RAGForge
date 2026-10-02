"""Source routes: add/discover, list, quality score, decision."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_repo
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.models import (
    DomainSpec,
    KBStatus,
    Source,
    SourceCreate,
    SourceDecision,
    SourceDecisionUpdate,
    SourceType,
)
from app.services.source_discovery.discovery import (
    DiscoveryError,
    UserURLProvider,
    get_provider,
)
from app.services.source_quality.scorer import SourceQualityScorer
from app.services.source_quality.user_scorer import assess_user_upload
from app.utils.ids import new_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["sources"])


def _get_kb_or_404(repo: Repository, kb_id: str):
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


def _score_sources(repo: Repository, kb_id: str, sources: list[Source]) -> None:
    """Score sources against the domain spec (if available) and persist.

    EXTERNAL candidates go through the web SourceQualityScorer: each URL gets
    one lightweight accessibility probe so the accessibility signal is verified
    rather than assumed, and Last-Modified (when the server reports it)
    improves the recency signal.

    USER-PROVIDED candidates (URLs the user explicitly supplied) are NOT run
    through the external scorer — they are asserted relevant by the user and
    assessed for integrity instead (see source_quality/user_scorer.py). This is
    the rule that stops RAGForge from downgrading a professor's lecture PDF for
    lacking a website, a publication date or an HTTP status.
    """
    spec = repo.get_domain_spec(kb_id)
    scorer = SourceQualityScorer()
    seen_urls = {s.url for s in sources}
    for source in sources:
        if source.user_provided:
            assessment = assess_user_upload(
                file_valid=True,
                parse_ok=True,
                extracted_chars=0,
                unit_count=None,
                warnings=[],
            )
            source.decision = SourceDecision.ACCEPT
            source.trust_score = assessment.score
            source.quality = assessment.model_dump(mode="json")
            if existing := repo.find_source_by_url(kb_id, source.url):
                continue
            repo.create_source(source)
            continue

        probe_data = {"accessible": None, "status": None, "last_modified": None}
        if not (source.notes and source.notes.startswith("INVALID")):
            from app.utils.http_probe import probe_url

            probe = probe_url(source.url)
            probe_data = {
                "accessible": probe.accessible,
                "status": probe.status_code,
                "last_modified": probe.last_modified,
            }
        assessment = scorer.assess(
            source,
            spec=spec,
            url_accessible=probe_data["accessible"],
            seen_urls=seen_urls,
            last_modified=probe_data["last_modified"],
            http_status=probe_data["status"],
        )
        source.trust_score = assessment.score
        source.quality = assessment.model_dump(mode="json")
        source.decision = assessment.decision  # automated decision; user can override
        # Probe evidence is stored in the quality dict (never in notes — notes
        # carry user/arXiv metadata that other signals score against).
        if source.quality is not None:
            source.quality["probe"] = {
                "performed": probe_data["status"] is not None or probe_data["accessible"] is False,
                "accessible": probe_data["accessible"],
                "http_status": probe_data["status"],
                "last_modified": (
                    probe_data["last_modified"].isoformat() if probe_data["last_modified"] else None
                ),
            }
        existing = repo.find_source_by_url(kb_id, source.url)
        if existing:
            continue  # don't duplicate on re-discovery
        repo.create_source(source)
    kb = repo.get_kb(kb_id)
    if kb and kb.status in (KBStatus.DRAFT, KBStatus.ANALYZED):
        kb.status = KBStatus.NEEDS_REVIEW if any(
            s.decision == SourceDecision.REVIEW for s in sources
        ) else kb.status
        repo.update_kb(kb)


class DiscoverRequest(BaseModel):
    provider: str = "user-url"
    query: str = Field(default="", description="URLs (user-url) or a search query (arxiv)")
    limit: int = 5


@router.post("/{kb_id}/discover-sources", response_model=list[Source])
def discover_sources(kb_id: str, payload: DiscoverRequest, repo: Repository = Depends(get_repo)):
    _get_kb_or_404(repo, kb_id)
    provider_name = payload.provider
    if provider_name == "user-url":
        provider = UserURLProvider()
    else:
        try:
            provider = get_provider(provider_name)
        except DiscoveryError as exc:
            raise HTTPException(400, str(exc)) from exc

    try:
        sources = provider.discover(kb_id, payload.query, limit=payload.limit)
    except DiscoveryError as exc:
        raise HTTPException(502, str(exc)) from exc

    if not sources:
        return []
    _score_sources(repo, kb_id, sources)
    return repo.list_sources(kb_id)


@router.get("/{kb_id}/sources", response_model=list[Source])
def list_sources(kb_id: str, repo: Repository = Depends(get_repo)):
    _get_kb_or_404(repo, kb_id)
    return repo.list_sources(kb_id)


@router.post("/{kb_id}/sources/{source_id}/decision", response_model=Source)
def set_decision(
    kb_id: str, source_id: str, payload: SourceDecisionUpdate, repo: Repository = Depends(get_repo)
):
    _get_kb_or_404(repo, kb_id)
    source = repo.get_source(kb_id, source_id)
    if not source:
        raise HTTPException(404, "Source not found")
    source.decision = payload.decision
    repo.update_source(source)
    return source


class UserURLRequest(BaseModel):
    """Add URLs the user explicitly supplied (USER_PROVIDED, not discovered)."""

    urls: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)


@router.post("/{kb_id}/sources/user-urls", response_model=list[Source])
def add_user_urls(kb_id: str, payload: UserURLRequest, repo: Repository = Depends(get_repo)):
    """Register user-supplied URLs for a knowledge base.

    These are marked `user_provided=True` so the UI can distinguish them from
    externally discovered sources and so the external authority scorer is not
    applied to them. They are still SSRF-validated and accessibility-probed.
    """
    _get_kb_or_404(repo, kb_id)
    if not payload.urls:
        raise HTTPException(400, "No URLs supplied")

    from app.utils.http_probe import probe_url
    from app.utils.url_validation import InvalidURLError, validate_public_http_url

    added: list[Source] = []
    for i, raw in enumerate(payload.urls):
        url = raw.strip()
        if not url:
            continue
        title = payload.titles[i] if i < len(payload.titles) else None
        if repo.find_source_by_url(kb_id, url):
            continue
        invalid_reason: str | None = None
        try:
            validate_public_http_url(url)
        except InvalidURLError as exc:
            invalid_reason = f"INVALID: {exc}"
        source = Source(
            id=new_id("src"),
            kb_id=kb_id,
            url=url,
            title=title,
            source_type=SourceType.WEB_PAGE,
            discovered_via="user-url",
            user_provided=True,
            provenance="user_url",
            notes=invalid_reason,
        )
        repo.create_source(source)
        probe = probe_url(source.url)
        assessment = assess_user_upload(
            file_valid=not (source.notes or "").startswith("INVALID"),
            parse_ok=bool(probe.accessible),
            extracted_chars=0,
            warnings=[] if probe.accessible else ["URL was not reachable when it was added"],
        )
        source.trust_score = assessment.score
        source.quality = assessment.model_dump(mode="json")
        source.decision = SourceDecision.ACCEPT if probe.accessible else SourceDecision.REVIEW
        repo.update_source(source)
        added.append(source)
    return added
