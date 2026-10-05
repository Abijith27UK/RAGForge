"""Retrieval, retrieval-configuration and evaluation routes (V6-aware).

Endpoints added in V6 (all under /api/knowledge-bases/{kb_id}):
* GET  /retrieval-strategies   — what is registered and what each strategy needs
* GET  /retrieval-config       — the persisted configuration (or defaults)
* PUT  /retrieval-config       — persist a configuration for this KB
* GET  /retrieval-runs         — compact list of past runs (observability)
* GET  /retrieval-runs/{id}    — the full run record (params, weights, timings)
* GET  /bm25-index             — lexical index status incl. explicit staleness

`POST /{kb_id}/retrieve` keeps its V1-V5 request shape and response model; it now
accepts optional `strategy` and `config` overrides, so the existing frontend keeps
working while the Retrieval Lab can select a strategy.
"""
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
    RetrievalResponse,
)
from app.schemas.retrieval import (
    RetrievalConfigRecord,
    RetrievalParams,
    RetrievalRun,
    RetrievalRunSummary,
)
from app.services.evaluation.evaluator import EvaluationError, Evaluator
from app.services.retrieval.retriever import RetrievalError
from app.services.retrieval.service import (
    KnowledgeBaseNotFound,
    RetrievalConfigError,
    RetrievalService,
)
from app.utils.ids import new_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["retrieval-evaluation"])

_NOT_READY = (
    "Knowledge base is not indexed yet. Run ingestion + indexing first "
    "(and make sure Qdrant and the embedding model are available)."
)


# ---------------------------------------------------------------------------
# Request models (V6)
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    """`POST /retrieve` body.

    V1-V5 clients send {query, top_k, filters} and are unaffected: `strategy` and
    `config` are optional and default to the KB's persisted configuration.
    """

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    filters: dict = Field(default_factory=dict)
    strategy: str | None = Field(
        default=None, description="dense | bm25 | hybrid | hybrid_reranked"
    )
    config: RetrievalParams | None = Field(
        default=None, description="Explicit configuration overriding the stored one"
    )


class RetrievalConfigUpdate(BaseModel):
    params: RetrievalParams
    note: str = Field(default="", max_length=500)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _service(repo: Repository) -> RetrievalService:
    return RetrievalService(repo, get_settings())


def _params_overrides(payload: RetrieveRequest) -> dict:
    """Merge top-level shortcuts and an explicit config object into overrides."""
    overrides: dict = payload.config.model_dump(mode="json") if payload.config else {}
    if payload.top_k is not None:
        overrides["top_k"] = payload.top_k
    if payload.filters:
        overrides["filters"] = payload.filters
    return overrides


@router.post("/{kb_id}/retrieve", response_model=RetrievalResponse)
def retrieve(kb_id: str, payload: RetrieveRequest, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    service = _service(repo)
    try:
        return service.run(
            kb_id,
            payload.query,
            strategy=payload.strategy,
            overrides=_params_overrides(payload),
        )
    except KnowledgeBaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except RetrievalConfigError as exc:  # unknown strategy / invalid configuration
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:  # defensive: strategy resolution
        raise HTTPException(400, str(exc)) from exc
    except RetrievalError as exc:  # backend/embedding failure
        raise HTTPException(503, str(exc)) from exc


@router.get("/{kb_id}/retrieval-strategies")
def retrieval_strategies(kb_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return _service(repo).available_strategies()


@router.get("/{kb_id}/retrieval-config", response_model=RetrievalConfigRecord)
def get_retrieval_config(kb_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    service = _service(repo)
    stored = service.get_config(kb_id)
    if stored is not None:
        return stored
    # Nothing was ever saved: report the effective defaults as a NEW record whose
    # `updated_at` is the response time, and say so in the note. The defaults are
    # real values, not a fabrication of a user decision.
    record = RetrievalConfigRecord(
        kb_id=kb_id,
        params=RetrievalParams(),
        note="defaults (no configuration has been saved for this knowledge base yet)",
    )
    return record


@router.put("/{kb_id}/retrieval-config", response_model=RetrievalConfigRecord)
def put_retrieval_config(
    kb_id: str, payload: RetrievalConfigUpdate, repo: Repository = Depends(get_repo)
):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return _service(repo).save_config(kb_id, payload.params, note=payload.note)


@router.get("/{kb_id}/retrieval-runs", response_model=list[RetrievalRunSummary])
def list_retrieval_runs(
    kb_id: str, limit: int = 50, repo: Repository = Depends(get_repo)
):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return _service(repo).recent_runs(kb_id, limit=max(1, min(limit, 200)))


@router.get("/{kb_id}/retrieval-runs/{run_id}", response_model=RetrievalRun)
def get_retrieval_run(kb_id: str, run_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    run = _service(repo).get_run(kb_id, run_id)
    if run is None:
        raise HTTPException(404, "Retrieval run not found")
    return run


@router.get("/{kb_id}/bm25-index")
def bm25_index(kb_id: str, repo: Repository = Depends(get_repo)):
    """Lexical index diagnostics; `stale` is reported explicitly, never inferred."""
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return _service(repo).bm25_index_status(kb_id)


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
    kb = repo.get_kb(kb_id)
    service = _service(repo)
    params = service.resolve_params(kb_id, strategy=config.strategy)
    params = params.model_copy(update={"top_k": config.top_k})
    try:
        retriever, _ = service.build_retriever(kb, params)  # type: ignore[arg-type]
    except RetrievalError as exc:
        raise HTTPException(503, str(exc)) from exc
    evaluator = Evaluator(retriever, repo)
    settings = get_settings()
    try:
        return evaluator.run_evaluation(
            kb_id,
            config,
            embedding_model=settings.embedding_model,
            strategy=params.strategy.value,
            retrieval_params=params,
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
