"""Knowledge Base routes: create, list, get, delete, build status."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_repo
from app.schemas.models import BuildRun, KBStatus, KnowledgeBase, KnowledgeBaseCreate
from app.repositories.sqlite_repo import Repository
from app.utils.ids import new_id

router = APIRouter(prefix="/api/knowledge-bases", tags=["knowledge-bases"])


@router.post("", response_model=KnowledgeBase, status_code=201)
def create_kb(payload: KnowledgeBaseCreate, repo: Repository = Depends(get_repo)):
    kb = KnowledgeBase(id=new_id("kb"), **payload.model_dump())
    repo.create_kb(kb)
    return kb


@router.get("", response_model=list[KnowledgeBase])
def list_kbs(repo: Repository = Depends(get_repo)):
    return repo.list_kbs()


@router.get("/{kb_id}", response_model=KnowledgeBase)
def get_kb(kb_id: str, repo: Repository = Depends(get_repo)):
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


@router.delete("/{kb_id}", status_code=204)
def delete_kb(kb_id: str, repo: Repository = Depends(get_repo)):
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    from app.services.vector_store.factory import create_vector_store
    from app.services.vector_store.qdrant_store import VectorStoreError
    from app.config import get_settings

    settings = get_settings()
    try:
        store = create_vector_store(settings, backend=getattr(kb, "vector_backend", "qdrant"))
        store.delete_collection(kb_id)
    except Exception:
        pass  # Qdrant may be offline; DB cleanup still proceeds
    repo.delete_kb(kb_id)


@router.get("/{kb_id}/build-status", response_model=BuildRun | None)
def build_status(kb_id: str, repo: Repository = Depends(get_repo)):
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return repo.latest_build_run(kb_id)
