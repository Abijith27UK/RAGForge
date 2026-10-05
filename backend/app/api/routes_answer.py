"""Grounded answer API (V7).

Endpoints (all under /api/knowledge-bases):
* POST /{kb_id}/answer                 — ask a question, get a validated answer
* GET  /{kb_id}/answers/{answer_id}    — fetch a stored answer
* GET  /{kb_id}/answer-traces/{id}     — fetch the full audit trace

Error mapping mirrors the retrieval routes: 404 unknown KB, 400 invalid request
(unknown mode/strategy), 503 backend/generation failure. Retrieval strategy
selection goes through the existing registry — unknown strategies surface as
400 from `RetrievalConfigError`.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_repo
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.answer import Answer, AnswerMode, AnswerPolicy, AnswerTrace
from app.services.answering.generator import AnswerGenerationError
from app.services.answering.service import AnswerConfigError, AnsweringService
from app.services.retrieval.retriever import RetrievalError
from app.services.retrieval.service import KnowledgeBaseNotFound, RetrievalConfigError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["answering"])


class AnswerRequest(BaseModel):
    """`POST /{kb_id}/answer` body.

    V1-V6 clients are unaffected (this endpoint is new). `retrieval_strategy`
    and `retrieval_params` are optional and default to the KB's persisted
    retrieval configuration.
    """

    question: str = Field(min_length=1, max_length=2000)
    answer_mode: str = Field(
        default=AnswerMode.ABSTAIN_IF_UNSUPPORTED.value,
        description="grounded | abstain_if_unsupported",
    )
    retrieval_strategy: str | None = Field(
        default=None, description="dense | bm25 | hybrid | hybrid_reranked"
    )
    retrieval_params: dict | None = Field(
        default=None, description="RetrievalParams overrides (top_k, min_score, ...)"
    )
    unsupported_claim_action: str | None = Field(
        default=None, description="remove | downgrade | abstain"
    )


class AnswerResponse(BaseModel):
    """Everything the UI needs for one answer, including the grounding
    assessment and generation metadata. No secrets, ever."""

    answer: Answer
    status: str
    citations: list
    claims: list
    evidence: list
    retrieval_run_id: str | None
    answer_trace_id: str
    grounding_assessment: dict
    generation_metadata: dict
    warnings: list[str] = Field(default_factory=list)


def _service(repo: Repository) -> AnsweringService:
    return AnsweringService(repo, get_settings())


def _overrides(payload: AnswerRequest) -> dict:
    return dict(payload.retrieval_params or {})


@router.post("/{kb_id}/answer", response_model=AnswerResponse)
def create_answer(kb_id: str, payload: AnswerRequest, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")

    policy_kwargs: dict = {}
    if payload.unsupported_claim_action:
        if payload.unsupported_claim_action not in ("remove", "downgrade", "abstain"):
            raise HTTPException(
                400,
                "unsupported_claim_action must be one of: remove, downgrade, abstain",
            )
        policy_kwargs["unsupported_claim_action"] = payload.unsupported_claim_action

    service = _service(repo)
    try:
        answer, trace, assessment = service.answer(
            kb_id,
            payload.question,
            mode=payload.answer_mode,
            strategy=payload.retrieval_strategy,
            overrides=_overrides(payload),
            policy=AnswerPolicy(**policy_kwargs) if policy_kwargs else None,
        )
    except KnowledgeBaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except AnswerConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RetrievalConfigError as exc:  # unknown strategy / bad config
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (RetrievalError, AnswerGenerationError) as exc:
        raise HTTPException(503, str(exc)) from exc

    return AnswerResponse(
        answer=answer,
        status=answer.status.value,
        citations=[c.model_dump(mode="json") for c in answer.citations],
        claims=[c.model_dump(mode="json") for c in answer.claims],
        evidence=[e.model_dump(mode="json") for e in trace.evidence_items],
        retrieval_run_id=answer.retrieval_run_id,
        answer_trace_id=trace.id,
        grounding_assessment=assessment.model_dump(mode="json"),
        generation_metadata={
            "generated_by": answer.generated_by,
            "model": answer.model,
            "is_mock": answer.is_mock,
            "prompt_version": answer.prompt_version,
            "answer_mode": answer.answer_mode.value,
        },
        warnings=list(answer.warnings),
    )


@router.get("/{kb_id}/answers/{answer_id}", response_model=Answer)
def get_answer(kb_id: str, answer_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    answer = _service(repo).get_answer(kb_id, answer_id)
    if answer is None:
        raise HTTPException(404, "Answer not found")
    return answer


@router.get("/{kb_id}/answer-traces/{trace_id}", response_model=AnswerTrace)
def get_answer_trace(kb_id: str, trace_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    trace = _service(repo).get_trace(kb_id, trace_id)
    if trace is None:
        raise HTTPException(404, "Answer trace not found")
    return trace
