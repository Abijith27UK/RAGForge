"""V3 Phase A — benchmark lifecycle & versioning API.

Question lifecycle: DRAFT -> REVIEW -> APPROVED -> FROZEN.
- DRAFT/REVIEW questions are freely editable in place.
- Editing an APPROVED question creates a NEW DRAFT revision (supersedes=old id);
  the approved content is preserved untouched.
- FROZEN questions are immutable (freeze protection).

Benchmark versions are immutable snapshots of question sets:
- Creating a snapshot stores a deep copy of every selected question.
- A snapshot may only be FROZEN when every question in it is APPROVED/FROZEN.
- FROZEN versions can never be edited or deleted; only they are usable for
  official evaluation runs (enforced by the evaluator).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_repo
from app.repositories.sqlite_repo import Repository
from app.schemas.models import (
    BenchmarkStatus,
    BenchmarkVersion,
    BenchmarkVersionCreate,
    EvaluationQuestion,
    QuestionRevision,
    QuestionStatus,
    QuestionStatusUpdate,
)
from app.utils.ids import new_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["benchmark"])


def _get_kb_or_404(repo: Repository, kb_id: str) -> None:
    if not repo.get_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")


def _get_question_or_404(repo: Repository, kb_id: str, question_id: str) -> EvaluationQuestion:
    q = repo.get_evaluation_question(kb_id, question_id)
    if q is None:
        raise HTTPException(404, "Evaluation question not found")
    return q


# ---------------------------------------------------------------------------
# Question lifecycle
# ---------------------------------------------------------------------------

_LEGAL_TRANSITIONS: dict[QuestionStatus, set[QuestionStatus]] = {
    QuestionStatus.DRAFT: {QuestionStatus.REVIEW, QuestionStatus.APPROVED, QuestionStatus.FROZEN},
    QuestionStatus.REVIEW: {QuestionStatus.APPROVED, QuestionStatus.DRAFT, QuestionStatus.FROZEN},
    QuestionStatus.APPROVED: {QuestionStatus.FROZEN},
    QuestionStatus.FROZEN: set(),
}


@router.post("/{kb_id}/evaluation-questions/{question_id}/status", response_model=EvaluationQuestion)
def set_question_status(
    kb_id: str, question_id: str, payload: QuestionStatusUpdate, repo: Repository = Depends(get_repo)
):
    _get_kb_or_404(repo, kb_id)
    q = _get_question_or_404(repo, kb_id, question_id)
    try:
        new_status = QuestionStatus(payload.status)
    except ValueError as exc:
        raise HTTPException(422, f"Unknown status {payload.status!r}") from exc
    if new_status not in _LEGAL_TRANSITIONS[q.status]:
        raise HTTPException(
            409,
            f"Illegal transition {q.status.value} -> {new_status.value}. "
            "Approved questions can only be frozen; frozen questions are immutable.",
        )
    q.status = new_status
    if new_status in (QuestionStatus.APPROVED, QuestionStatus.FROZEN):
        if not payload.reviewer:
            raise HTTPException(422, "A reviewer name is required to APPROVE or FREEZE a question")
        q.reviewer = payload.reviewer
        q.reviewed_at = datetime.now(timezone.utc)
    repo_update_question(repo, q)
    logger.info("Question %s status -> %s (by %s)", q.id, new_status.value, payload.reviewer or "user")
    return q


@router.patch("/{kb_id}/evaluation-questions/{question_id}", response_model=EvaluationQuestion)
def edit_question(
    kb_id: str, question_id: str, payload: QuestionRevision, repo: Repository = Depends(get_repo)
):
    """Edit a question. DRAFT/REVIEW are edited in place; APPROVED/FROZEN create
    a new DRAFT revision superseding this one (approved content preserved)."""
    _get_kb_or_404(repo, kb_id)
    q = _get_question_or_404(repo, kb_id, question_id)

    if q.status in (QuestionStatus.APPROVED, QuestionStatus.FROZEN):
        if q.status is QuestionStatus.FROZEN:
            raise HTTPException(
                409,
                "This question is FROZEN and cannot be revised. "
                "Create a new benchmark version for revised ground truth.",
            )
        revision = q.model_copy(deep=True)
        revision.id = new_id("eq")
        revision.status = QuestionStatus.DRAFT
        revision.revision = q.revision + 1
        revision.supersedes = q.id
        revision.reviewer = None
        revision.reviewed_at = None
        revision.author = q.author or "revision"
        _apply_revision_fields(revision, payload)
        repo.create_evaluation_question(revision)
        logger.info("Question %s revised -> new DRAFT %s (rev %d)", q.id, revision.id, revision.revision)
        return revision

    _apply_revision_fields(q, payload)
    repo_update_question(repo, q)
    return q


def _apply_revision_fields(q: EvaluationQuestion, payload: QuestionRevision) -> None:
    if payload.question is not None:
        q.question = payload.question
    if payload.expected_chunk_ids is not None:
        q.expected_chunk_ids = payload.expected_chunk_ids
    if payload.expected_document_ids is not None:
        q.expected_document_ids = payload.expected_document_ids
    if payload.expected_keywords is not None:
        q.expected_keywords = payload.expected_keywords
    if payload.notes is not None:
        q.notes = payload.notes
    if payload.provenance is not None:
        q.provenance = payload.provenance


def repo_update_question(repo: Repository, q: EvaluationQuestion) -> None:
    """Persist an edited question (JSON-blob storage: rewrite the row)."""
    repo._execute(
        "UPDATE evaluation_questions SET data = ? WHERE id = ?",
        (repo._to_row(q), q.id),
    )


# ---------------------------------------------------------------------------
# Benchmark versions (frozen snapshots)
# ---------------------------------------------------------------------------


@router.post("/{kb_id}/benchmark-versions", response_model=BenchmarkVersion, status_code=201)
def create_benchmark_version(
    kb_id: str, payload: BenchmarkVersionCreate, repo: Repository = Depends(get_repo)
):
    _get_kb_or_404(repo, kb_id)
    questions = repo.list_evaluation_questions(kb_id)
    if payload.question_ids:
        wanted = set(payload.question_ids)
        selected = [q for q in questions if q.id in wanted]
        missing = wanted - {q.id for q in selected}
        if missing:
            raise HTTPException(404, f"Unknown question ids: {sorted(missing)}")
    else:
        selected = list(questions)
    if not selected:
        raise HTTPException(400, "No questions selected for the benchmark version")

    statuses = {q.status for q in selected}
    draftish = statuses & {QuestionStatus.DRAFT, QuestionStatus.REVIEW}
    if draftish:
        status = BenchmarkStatus.DRAFT
    elif QuestionStatus.APPROVED in statuses:
        status = BenchmarkStatus.APPROVED
    else:
        status = BenchmarkStatus.FROZEN

    now = datetime.now(timezone.utc)
    bv = BenchmarkVersion(
        id=new_id("bv"),
        kb_id=kb_id,
        version=payload.version,
        label=payload.label,
        status=status,
        question_ids=[q.id for q in selected],
        questions_snapshot=[q.model_dump(mode="json") for q in selected],
        created_by=payload.created_by,
        created_at=now,
        frozen_at=now if status is BenchmarkStatus.FROZEN else None,
        notes=payload.notes,
    )
    repo.create_benchmark_version(bv)
    logger.info(
        "Benchmark version %s created for %s: %d questions, status=%s",
        bv.version, kb_id, len(selected), status.value,
    )
    return bv


@router.get("/{kb_id}/benchmark-versions", response_model=list[BenchmarkVersion])
def list_benchmark_versions(kb_id: str, repo: Repository = Depends(get_repo)):
    _get_kb_or_404(repo, kb_id)
    return repo.list_benchmark_versions(kb_id)


@router.get("/{kb_id}/benchmark-versions/{bv_id}", response_model=BenchmarkVersion)
def get_benchmark_version(kb_id: str, bv_id: str, repo: Repository = Depends(get_repo)):
    _get_kb_or_404(repo, kb_id)
    bv = repo.get_benchmark_version(kb_id, bv_id)
    if bv is None:
        raise HTTPException(404, "Benchmark version not found")
    return bv


@router.post("/{kb_id}/benchmark-versions/{bv_id}/freeze", response_model=BenchmarkVersion)
def freeze_benchmark_version(
    kb_id: str, bv_id: str, repo: Repository = Depends(get_repo), reviewer: str = ""
):
    _get_kb_or_404(repo, kb_id)
    bv = repo.get_benchmark_version(kb_id, bv_id)
    if bv is None:
        raise HTTPException(404, "Benchmark version not found")
    if bv.status is BenchmarkStatus.FROZEN:
        return bv  # idempotent
    if bv.status is BenchmarkStatus.DRAFT:
        # Re-evaluate from the *snapshot* statuses (not live rows): a snapshot
        # freezes exactly what was captured, never later live edits.
        statuses = {qd.get("status", "DRAFT") for qd in bv.questions_snapshot}
        if {"DRAFT", "REVIEW"} & statuses:
            raise HTTPException(
                409,
                "Cannot freeze: the snapshot contains DRAFT/REVIEW questions. "
                "All questions must be APPROVED (or already FROZEN) first.",
            )
    bv.status = BenchmarkStatus.FROZEN
    bv.frozen_at = datetime.now(timezone.utc)
    if reviewer:
        bv.notes = (bv.notes + f" | frozen by {reviewer}").strip(" |")
    repo.update_benchmark_version(bv)
    logger.info("Benchmark version %s FROZEN (%d questions)", bv.version, len(bv.question_ids))
    return bv


@router.delete("/{kb_id}/benchmark-versions/{bv_id}", status_code=204)
def delete_benchmark_version(kb_id: str, bv_id: str, repo: Repository = Depends(get_repo)):
    _get_kb_or_404(repo, kb_id)
    bv = repo.get_benchmark_version(kb_id, bv_id)
    if bv is None:
        raise HTTPException(404, "Benchmark version not found")
    if bv.status is BenchmarkStatus.FROZEN:
        raise HTTPException(
            409, "FROZEN benchmark versions are immutable and cannot be deleted (freeze protection)"
        )
    repo._execute("DELETE FROM benchmark_versions WHERE id = ?", (bv_id,))
