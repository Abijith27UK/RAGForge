"""Retrieval and evaluation routes."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_repo
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.models import (
    EvaluationQuestion,
    EvaluationQuestionCreate,
    EvaluationRun,
    EvaluationRunConfig,
    RetrievalRequest,
    RetrievalResponse,
)
from app.services.evaluation.evaluator import EvaluationError, Evaluator
from app.services.retrieval.retriever import RetrievalError
from app.utils.ids import new_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["retrieval-evaluation"])

_NOT_READY = (
    "Knowledge base is not indexed yet. Run ingestion + indexing first "
    "(and make sure Qdrant and the embedding model are available)."
)


def _retriever(repo: Repository, kb_id: str | None = None):
    """Build a DenseRetriever; refuses to run when the KB was indexed with a
    different embedding model (prevents querying an incompatible vector space)."""
    from app.services.embeddings.provider import (
        EmbeddingError,
        EmbeddingIdentity,
        create_embedding_provider,
    )
    from app.services.vector_store.factory import create_vector_store

    settings = get_settings()
    expected = None
    if kb_id:
        kb = repo.get_kb(kb_id)
        if kb and kb.embedding_identity:
            expected = EmbeddingIdentity(**kb.embedding_identity)
    try:
        embedder = create_embedding_provider(settings, expected=expected)
        identity = embedder.identity()
    except EmbeddingError as exc:
        raise HTTPException(503, str(exc)) from exc
    store = create_vector_store(settings, backend=(kb.vector_backend if kb else "qdrant"))
    from app.services.retrieval.retriever import DenseRetriever

    return DenseRetriever(embedder, store, identity)


@router.post("/{kb_id}/retrieve", response_model=RetrievalResponse)
def retrieve(kb_id: str, payload: RetrievalRequest, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    retriever = _retriever(repo, kb_id)
    try:
        return retriever.retrieve(
            kb_id,
            payload.query,
            top_k=payload.top_k,
            filters=payload.filters or None,
        )
    except RetrievalError as exc:
        raise HTTPException(503, str(exc)) from exc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@router.post("/{kb_id}/evaluation-questions", response_model=EvaluationQuestion, status_code=201)
def add_question(
    kb_id: str, payload: EvaluationQuestionCreate, repo: Repository = Depends(get_repo)
):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    q = EvaluationQuestion(
        id=new_id("eq"),
        kb_id=kb_id,
        question=payload.question,
        expected_chunk_ids=payload.expected_chunk_ids,
        expected_document_ids=payload.expected_document_ids,
        expected_keywords=payload.expected_keywords,
        generated_by="manual",
        notes=payload.notes,
        provenance=payload.provenance,
        author=payload.author,
    )
    repo.create_evaluation_question(q)
    return q


@router.get("/{kb_id}/evaluation-questions", response_model=list[EvaluationQuestion])
def list_questions(kb_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return repo.list_evaluation_questions(kb_id)


@router.delete("/{kb_id}/evaluation-questions/{question_id}", status_code=204)
def delete_question(kb_id: str, question_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    # Simple implementation: list + recreate without the deleted one
    questions = repo.list_evaluation_questions(kb_id)
    keep = [q for q in questions if q.id != question_id]
    if len(keep) == len(questions):
        raise HTTPException(404, "Question not found")
    repo._execute(
        "DELETE FROM evaluation_questions WHERE kb_id = ? AND id = ?", (kb_id, question_id)
    )


@router.post("/{kb_id}/evaluate", response_model=EvaluationRun)
def evaluate(kb_id: str, config: EvaluationRunConfig, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    retriever = _retriever(repo, kb_id)
    evaluator = Evaluator(retriever, repo)
    settings = get_settings()
    try:
        return evaluator.run_evaluation(
            kb_id, config, embedding_model=settings.embedding_model
        )
    except EvaluationError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/{kb_id}/evaluation-runs", response_model=list[EvaluationRun])
def list_evaluation_runs(kb_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return repo.list_evaluation_runs(kb_id)
