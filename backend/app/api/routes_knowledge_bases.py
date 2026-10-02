"""Knowledge Base routes: create, list, get, delete, build status, overview."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_repo
from app.config import get_settings
from app.schemas.models import (
    GROUND_TRUTH_UNAVAILABLE_EXPLANATION,
    BenchmarkStatus,
    BuildRun,
    DocumentStatus,
    GroundTruthStatus,
    KBOverview,
    KBStatus,
    KnowledgeBase,
    KnowledgeBaseCreate,
    QuestionStatus,
)
from app.repositories.sqlite_repo import Repository
from app.utils.ids import new_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["knowledge-bases"])


@router.post("", response_model=KnowledgeBase, status_code=201)
def create_kb(payload: KnowledgeBaseCreate, repo: Repository = Depends(get_repo)):
    kb = KnowledgeBase(id=new_id("kb"), **payload.model_dump())
    repo.create_kb(kb)
    logger.info(
        "Created knowledge base %s (%s, source_mode=%s)", kb.id, kb.domain, kb.source_mode.value
    )
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


@router.get("/{kb_id}/overview", response_model=KBOverview)
def kb_overview(kb_id: str, repo: Repository = Depends(get_repo)):
    """Everything the KB overview needs, honestly reported.

    A knowledge base is READY once it is indexed. Evaluation is OPTIONAL:
    `evaluation_status` is "not_configured" until the user creates questions
    and a benchmark, and that is a completely valid state.
    """
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    documents = repo.list_documents(kb_id)
    sources = repo.list_sources(kb_id)
    chunks = repo.count_chunks(kb_id)
    questions = repo.list_evaluation_questions(kb_id)
    versions = repo.list_benchmark_versions(kb_id)
    runs = repo.list_evaluation_runs(kb_id)
    last_run = repo.latest_build_run(kb_id)

    # Vector store state — reported as unreachable rather than fabricated.
    vectors: int | None = None
    store_status = "not_configured"
    try:
        from app.services.vector_store.factory import create_vector_store

        store = create_vector_store(get_settings(), backend=kb.vector_backend)
        info = store.collection_info(kb_id)
        vectors = int(info.get("points_count") or 0)
        store_status = str(info.get("status", "unknown")).lower()
    except Exception as exc:
        store_status = "unreachable"
        logger.info("Vector store unavailable for %s: %s", kb_id, exc)

    frozen_versions = [v for v in versions if v.status is BenchmarkStatus.FROZEN]
    if runs:
        evaluation_status = "evaluated"
    elif frozen_versions:
        evaluation_status = "benchmark_frozen"
    elif versions:
        evaluation_status = "benchmark_draft"
    elif questions:
        evaluation_status = "questions_only"
    else:
        evaluation_status = "not_configured"

    last_evaluation = None
    if runs:
        last_evaluation = {
            "id": runs[0].id,
            "started_at": runs[0].started_at.isoformat(),
            "top_k": runs[0].config.top_k,
            "questions_evaluated": runs[0].aggregate.questions_evaluated,
            "recall_at_k": runs[0].aggregate.recall_at_k,
            "mrr": runs[0].aggregate.mrr,
            "ndcg": runs[0].aggregate.ndcg,
            "benchmark_version": runs[0].benchmark_version,
        }

    # Ground-truth availability is a SEPARATE question from evaluation status.
    # A domain never gets ground truth by existing; only reviewed, frozen
    # questions count. Absence is reported explicitly, never as a missing field.
    if frozen_versions:
        ground_truth_status = GroundTruthStatus.FROZEN
        ground_truth_explanation = (
            f"{len(frozen_versions)} frozen benchmark version(s) exist, containing "
            f"{len(questions)} reviewed questions. Retrieval metrics measured against "
            "these questions are valid."
        )
    elif any(q.status is QuestionStatus.APPROVED for q in questions):
        ground_truth_status = GroundTruthStatus.APPROVED
        ground_truth_explanation = (
            "Questions exist and have been approved but no benchmark version has been "
            "frozen yet. Freezing is required before results are reproducible."
        )
    elif any(q.status is QuestionStatus.IN_REVIEW for q in questions):
        ground_truth_status = GroundTruthStatus.IN_REVIEW
        ground_truth_explanation = (
            f"{len(questions)} question(s) are drafted and awaiting human review. "
            "No retrieval metric may be reported yet."
        )
    elif questions:
        ground_truth_status = GroundTruthStatus.DRAFT
        ground_truth_explanation = (
            f"{len(questions)} draft question(s) exist, none reviewed. Ground truth is "
            "not yet available and no retrieval metric may be reported."
        )
    else:
        ground_truth_status = GroundTruthStatus.NOT_AVAILABLE
        ground_truth_explanation = GROUND_TRUTH_UNAVAILABLE_EXPLANATION

    build_status_value = "not_built"
    if kb.status is KBStatus.READY:
        build_status_value = "ready"
    elif kb.status is KBStatus.BUILDING:
        build_status_value = "building"
    elif kb.status is KBStatus.ERROR:
        build_status_value = "error"
    elif chunks:
        build_status_value = "indexed"

    return KBOverview(
        kb=kb,
        documents=len(documents),
        documents_ready=sum(1 for d in documents if d.status is DocumentStatus.READY),
        documents_failed=sum(1 for d in documents if d.status is DocumentStatus.FAILED),
        user_provided_documents=sum(1 for d in documents if d.user_provided),
        external_documents=sum(1 for d in documents if not d.user_provided),
        chunks=chunks,
        vectors=vectors,
        sources=len(sources),
        sources_user_provided=sum(1 for s in sources if s.user_provided),
        sources_discovered=sum(1 for s in sources if not s.user_provided),
        embedding_model=(kb.embedding_identity or {}).get("model", "") or get_settings().embedding_model,
        vector_backend=kb.vector_backend,
        vector_store_status=store_status,
        chunking_strategy=kb.chunking_strategy,
        chunking_config=kb.chunking_config,
        last_build_at=kb.last_build_at,
        last_build_status=last_run.status if last_run else "",
        version=kb.version,
        build_status=build_status_value,
        evaluation_status=evaluation_status,
        benchmark_versions=len(versions),
        frozen_benchmark_versions=len(frozen_versions),
        evaluation_questions=len(questions),
        evaluation_runs=len(runs),
        last_evaluation=last_evaluation,
        evaluation_required=False,
        ground_truth_status=ground_truth_status,
        ground_truth_explanation=ground_truth_explanation,
        ground_truth_question_count=len(questions),
    )
