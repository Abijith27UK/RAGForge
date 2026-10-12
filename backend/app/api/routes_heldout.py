"""Held-out validation API (V14).

Endpoints (under /api/knowledge-bases):

* GET  /{kb_id}/held-out/benchmark — identity, fingerprint, freeze tier and claim
  strength of the held-out instrument, plus the full validation report and the
  review completeness. One request answers "what is this instrument, may it be
  run, and what may a result from it be called?".
* GET  /{kb_id}/held-out/review-view — every candidate with the FULL text of the
  chunk its evidence label rests on and the availability verdict for that chunk.
  This is the review surface: the reviewer reads the evidence, not a summary of it.
* POST /{kb_id}/held-out/review — record a human decision (approve / edit /
  reject) per question, naming the reviewer.
* POST /{kb_id}/held-out/freeze — seal the artifact, computing its tier from the
  review records that exist at that moment. Refuses while validation errors exist
  and refuses once frozen.
* GET  /{kb_id}/held-out/experiments — the V14 held-out validation experiments,
  their development-reference re-measurements, and the cross-instrument
  generalization summaries. Each row carries its own claim strength so an
  exploratory number can never be read as a validated one.

Safety: nothing here writes to the knowledge base or the vector index, and no
endpoint can modify the development benchmark. The only writes are to the two
held-out artifacts under ``benchmarks/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_repo
from app.config import get_settings
from app.repositories.sqlite_repo import Repository
from app.services.answer_eval.heldout_benchmark import (
    DEFAULT_HELDOUT_BENCHMARK_PATH,
    DEFAULT_HELDOUT_CANDIDATE_PATH,
    HeldOutReviewRequest,
    HeldOutValidationError,
)
from app.services.answer_eval.heldout_service import HeldOutService
from app.services.answer_eval.retrieval_experiment_runner import (
    RetrievalExperimentRunner,
)
from app.services.retrieval.service import KnowledgeBaseNotFound

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["held-out-validation"])

CLAIM_NOTICE = (
    "An EXPLORATORY result is a provisional test of a mechanism on questions that "
    "were not human-reviewed when the instrument was sealed. A CONFIRMATORY result "
    "requires a VALIDATED instrument, frozen after every question was reviewed. No "
    "result here is promoted into the production configuration."
)


def _service(repo: Repository) -> HeldOutService:
    """The one wiring point for every endpoint here.

    Paths come from settings when set, so a test (or a run against a copy of the
    corpus) can use its own artifacts. Without that, a test could only exercise
    this API against the real research instrument, which it must never write to.
    """
    settings = get_settings()
    return HeldOutService(
        repo,
        settings,
        candidate_path=settings.heldout_candidate_path or DEFAULT_HELDOUT_CANDIDATE_PATH,
        benchmark_path=settings.heldout_benchmark_path or DEFAULT_HELDOUT_BENCHMARK_PATH,
    )


def _require_kb(repo: Repository, kb_id: str) -> None:
    if repo.get_kb(kb_id) is None:
        raise HTTPException(404, f"Knowledge base {kb_id!r} not found")


class ReviewDecisionRequest(BaseModel):
    """One human decision about one candidate question."""

    question_id: str
    decision: str = Field(description="approve | edit | reject")
    reviewer: str = Field(
        description="Who made the decision. Required: an anonymous decision is not "
        "treated as human review."
    )
    notes: str = ""
    question: str | None = Field(
        default=None, description="Replacement question text (edit only)"
    )
    expected_chunk_ids: list[str] | None = Field(
        default=None, description="Replacement evidence chunk ids (edit only)"
    )
    subdomain: str | None = Field(default=None, description="Replacement subdomain (edit only)")
    results_seen: bool = Field(
        default=False,
        description="True if the reviewer has already seen held-out results. Recorded "
        "because a label chosen with knowledge of the outcome is weaker evidence.",
    )


class ReviewRequest(BaseModel):
    decisions: list[ReviewDecisionRequest]


class FreezeRequest(BaseModel):
    frozen_by: str = Field(
        description="Who is sealing the instrument. Required and stored on the artifact."
    )
    notes: list[str] = Field(default_factory=list)
    require_clean_validation: bool = True


@router.get("/{kb_id}/held-out/benchmark")
def get_heldout_benchmark(
    kb_id: str, repo: Repository = Depends(get_repo)
) -> dict[str, Any]:
    """The held-out instrument's identity, validation state and freeze tier."""
    _require_kb(repo, kb_id)
    try:
        return _service(repo).status(kb_id)
    except HeldOutValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{kb_id}/held-out/review-view")
def get_review_view(kb_id: str, repo: Repository = Depends(get_repo)) -> dict[str, Any]:
    """Every candidate with the full text of its evidence chunk, for review."""
    _require_kb(repo, kb_id)
    try:
        return _service(repo).review_view(kb_id)
    except HeldOutValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KnowledgeBaseNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/{kb_id}/held-out/review")
def post_review(
    kb_id: str, payload: ReviewRequest, repo: Repository = Depends(get_repo)
) -> dict[str, Any]:
    """Record human decisions about the candidate questions."""
    _require_kb(repo, kb_id)
    try:
        requests = [
            HeldOutReviewRequest(
                question_id=d.question_id,
                decision=d.decision,
                reviewer=d.reviewer,
                notes=d.notes,
                question=d.question,
                expected_chunk_ids=d.expected_chunk_ids,
                subdomain=d.subdomain,
                results_seen=d.results_seen,
            )
            for d in payload.decisions
        ]
    except Exception as exc:  # pydantic validation messages are the useful part
        raise HTTPException(422, f"invalid review decision: {exc}") from exc
    try:
        return _service(repo).review(kb_id, requests)
    except HeldOutValidationError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{kb_id}/held-out/freeze", status_code=201)
def post_freeze(
    kb_id: str, payload: FreezeRequest, repo: Repository = Depends(get_repo)
) -> dict[str, Any]:
    """Seal the instrument. The tier is computed from the review records NOW."""
    _require_kb(repo, kb_id)
    try:
        return _service(repo).freeze(
            kb_id,
            frozen_by=payload.frozen_by,
            notes=payload.notes or None,
            require_clean_validation=payload.require_clean_validation,
        )
    except HeldOutValidationError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{kb_id}/held-out/experiments")
def get_heldout_experiments(
    kb_id: str, repo: Repository = Depends(get_repo)
) -> dict[str, Any]:
    """The V14 validation experiments, their dev references and the summaries."""
    _require_kb(repo, kb_id)
    service = _service(repo)
    runner = RetrievalExperimentRunner(repo, get_settings())

    frozen = service.frozen()
    experiments = []
    for experiment in runner.list_experiments(kb_id):
        experiments.append(
            {
                "id": experiment.id,
                "name": experiment.name,
                "instrument": experiment.instrument,
                "benchmark_name": experiment.benchmark_name,
                "benchmark_fingerprint": experiment.benchmark_fingerprint,
                "benchmark_lifecycle": experiment.benchmark_lifecycle,
                "claim_strength": experiment.claim_strength,
                "retests": experiment.retests,
                "hypothesis": experiment.hypothesis,
                "primary_metric": experiment.primary_metric,
                "guardrail_metrics": experiment.guardrail_metrics,
                "rejection_conditions": experiment.rejection_conditions,
                "outcome": experiment.outcome.value,
                "outcome_explanation": experiment.outcome_explanation,
                "accepted_candidate": experiment.accepted_candidate,
                "promoted_to_default": experiment.promoted_to_default,
                "question_count": len(experiment.matched_question_ids),
                "question_set_fingerprint": experiment.question_set_fingerprint,
                "status": experiment.status,
                "created_at": experiment.created_at,
                "integrity": experiment.integrity,
                "limitations": experiment.limitations,
                "arms": [
                    {
                        "label": arm.label,
                        "strategy": arm.strategy,
                        "top_k": arm.top_k,
                        "candidate_depth": arm.candidate_depth,
                        "params": arm.config.params,
                        "status": arm.status,
                        "failure_reason": arm.failure_reason,
                        "reranker_status": arm.reranker_status,
                        "metrics": {
                            name: {
                                "mean": stat.mean,
                                "median": stat.median,
                                "measured": stat.measured,
                                "sample_size": stat.sample_size,
                                "direction": stat.direction,
                                "reason": stat.reason,
                            }
                            for name, stat in (arm.metric_stats or {}).items()
                        },
                    }
                    for arm in experiment.arms
                ],
                "candidate_outcomes": [
                    {
                        "label": o.label,
                        "outcome": o.outcome.value,
                        "acceptance": o.acceptance.value,
                        "acceptance_reason": o.acceptance_reason,
                        "outcome_explanation": o.outcome_explanation,
                        "pairing": o.pairing,
                        "primary_mean_delta": o.primary_mean_delta,
                        "primary_required_delta": o.primary_required_delta,
                        "primary_sign_test_p": o.primary_sign_test_p,
                        "primary_sign_test_n": o.primary_sign_test_n,
                        "paired_question_count": o.paired_question_count,
                        "paired_wins": o.paired_wins,
                        "paired_losses": o.paired_losses,
                        "guardrails_regressed": o.guardrails_regressed,
                        "guardrail_assessment": o.guardrail_assessment,
                    }
                    for o in experiment.candidate_outcomes
                ],
                "per_question": [
                    {
                        "label": arm.label,
                        "question_id": p.question_id,
                        "question": p.question,
                        "baseline": p.baseline,
                        "candidate": p.candidate,
                        "delta": p.delta,
                        "outcome": p.outcome,
                        "direction": p.direction,
                    }
                    for arm in experiment.arms
                    for p in (arm.per_question or [])
                ],
            }
        )

    # The cross-instrument summaries written by the validation script. Read-only,
    # and reported verbatim so nothing is recomputed differently for the UI.
    summaries: list[dict[str, Any]] = []
    for path in sorted(runner.experiments_dir.glob("v14-generalization-*.json")):
        try:
            summaries.append(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read generalization summary %s: %s", path, exc)

    return {
        "kb_id": kb_id,
        "heldout": (
            None
            if frozen is None
            else {
                "benchmark": frozen.benchmark,
                "fingerprint": frozen.fingerprint(),
                "lifecycle": frozen.lifecycle.value,
                "claim_strength": frozen.claim_strength.value,
                "question_count": frozen.question_count,
                "frozen_at": frozen.frozen_at,
                "frozen_by": frozen.frozen_by,
            }
        ),
        "experiments": [e for e in experiments if e["instrument"] == "held-out"],
        "development_references": [
            e for e in experiments if e["instrument"] == "development"
        ],
        "generalization_summaries": summaries,
        "notice": CLAIM_NOTICE,
    }


@router.get("/{kb_id}/held-out/config")
def get_heldout_config(
    kb_id: str, repo: Repository = Depends(get_repo)
) -> dict[str, Any]:
    """What CAN be done right now, so the UI never offers a dead action."""
    _require_kb(repo, kb_id)
    service = _service(repo)
    frozen = service.frozen()
    candidates = service.candidates()
    return {
        "kb_id": kb_id,
        "can_review": frozen is None,
        "can_freeze": frozen is None,
        "can_run": frozen is not None and frozen.lifecycle.value != "draft",
        "supports_confirmatory_claim": (
            frozen is not None and frozen.claim_strength.value == "confirmatory"
        ),
        "review_complete": candidates.review_completeness()["complete"],
        "lifecycle": frozen.lifecycle.value if frozen else "draft",
        "question_count": len(candidates.surviving),
        "notice": (
            "The tier is sealed at freeze time from the review records that exist then. "
            "Reviewing AFTER a run cannot upgrade an instrument, because a reviewer who "
            "has seen the results is no longer blind to them."
        ),
    }
