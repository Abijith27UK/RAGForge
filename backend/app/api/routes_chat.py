"""Grounded chat API (V7 Phase 13).

Endpoints (all under /api/knowledge-bases):
* POST   /{kb_id}/chat                              — one grounded chat turn
* GET    /{kb_id}/conversations                     — list conversation threads
* POST   /{kb_id}/conversations                     — create a thread explicitly
* GET    /{kb_id}/conversations/{conversation_id}   — thread + its messages
* DELETE /{kb_id}/conversations/{conversation_id}   — delete a thread
* GET    /{kb_id}/answer-runs                       — answer-run history (observability)
* GET    /{kb_id}/answer-runs/{run_id}              — one answer run

Error mapping mirrors `/answer`: 404 unknown KB or conversation, 400 invalid
mode/strategy, 503 retrieval or generation failure. The response never hides the
retrieval or grounding process — that is the point of the endpoint.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_repo
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.answer import (
    AnswerRun,
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
)
from app.services.answering.chat import ConversationNotFound, GroundedChatService
from app.services.answering.generator import AnswerGenerationError
from app.services.answering.service import AnswerConfigError
from app.services.retrieval.retriever import RetrievalError
from app.services.retrieval.service import KnowledgeBaseNotFound, RetrievalConfigError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["chat"])


class CreateConversationRequest(BaseModel):
    """Optional explicit thread creation. `title` is free text, never required."""

    title: str = Field(default="", max_length=200)


def _service(repo: Repository) -> GroundedChatService:
    return GroundedChatService(repo, get_settings())


@router.post("/{kb_id}/chat", response_model=ChatResponse)
def chat(kb_id: str, payload: ChatRequest, repo: Repository = Depends(get_repo)):
    """One grounded chat turn.

    The answer is produced by the same pipeline as `/answer`; this endpoint adds
    conversation memory and the AnswerRun observability record. Conversation
    history is used to resolve references and NEVER as knowledge.
    """
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")

    service = _service(repo)
    try:
        return service.chat(kb_id, payload)
    except KnowledgeBaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ConversationNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except AnswerConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RetrievalConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (RetrievalError, AnswerGenerationError) as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/{kb_id}/conversations", response_model=list[ConversationSummary])
def list_conversations(kb_id: str, limit: int = 50, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return _service(repo).list_conversations(kb_id, limit=limit)


@router.post("/{kb_id}/conversations", response_model=ConversationDetail, status_code=201)
def create_conversation(
    kb_id: str,
    payload: CreateConversationRequest | None = None,
    repo: Repository = Depends(get_repo),
):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    conversation = _service(repo).create_conversation(
        kb_id, title=(payload.title if payload else "")
    )
    return ConversationDetail(conversation=conversation, messages=[])


@router.get("/{kb_id}/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(kb_id: str, conversation_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    detail = _service(repo).get_conversation(kb_id, conversation_id)
    if detail is None:
        raise HTTPException(404, "Conversation not found")
    return detail


@router.delete("/{kb_id}/conversations/{conversation_id}", status_code=204)
def delete_conversation(kb_id: str, conversation_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    if not _service(repo).delete_conversation(kb_id, conversation_id):
        raise HTTPException(404, "Conversation not found")


@router.get("/{kb_id}/answer-runs", response_model=list[AnswerRun])
def list_answer_runs(kb_id: str, limit: int = 50, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return _service(repo).list_runs(kb_id, limit=limit)


@router.get("/{kb_id}/answer-runs/{run_id}", response_model=AnswerRun)
def get_answer_run(kb_id: str, run_id: str, repo: Repository = Depends(get_repo)):
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    run = _service(repo).get_run(kb_id, run_id)
    if run is None:
        raise HTTPException(404, "Answer run not found")
    return run
