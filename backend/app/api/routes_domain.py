"""Domain analysis routes."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_repo
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.models import DomainSpec, KBStatus, KnowledgeBase

router = APIRouter(prefix="/api/knowledge-bases", tags=["domain-analysis"])


def _get_kb_or_404(repo: Repository, kb_id: str) -> KnowledgeBase:
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


@router.post("/{kb_id}/analyze-domain", response_model=DomainSpec)
def analyze_domain(kb_id: str, repo: Repository = Depends(get_repo)):
    kb = _get_kb_or_404(repo, kb_id)
    kb.status = KBStatus.ANALYZING
    repo.update_kb(kb)
    try:
        from app.services.domain_analyzer.analyzer import analyze_domain

        settings = get_settings()
        spec, used_mock = analyze_domain(kb, settings)
    except Exception as exc:
        kb.status = KBStatus.ERROR
        repo.update_kb(kb)
        raise HTTPException(502, f"Domain analysis failed: {exc}") from exc
    repo.save_domain_spec(spec)
    kb.status = KBStatus.ANALYZED
    kb.updated_at = datetime.now(timezone.utc)
    repo.update_kb(kb)
    return spec


@router.get("/{kb_id}/domain-spec", response_model=DomainSpec)
def get_domain_spec(kb_id: str, repo: Repository = Depends(get_repo)):
    _get_kb_or_404(repo, kb_id)
    spec = repo.get_domain_spec(kb_id)
    if not spec:
        raise HTTPException(404, "No domain spec yet. Run domain analysis first.")
    return spec
