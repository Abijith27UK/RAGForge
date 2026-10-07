"""V9 Phase 1 regression tests for benchmark review workflow.

Append-only question reviews, review status, and the official-gate contract.
No network, no Qdrant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.benchmark import (  # noqa: E402
    AnswerBenchmark,
    AnswerBenchmarkLifecycle,
    AnswerBenchmarkQuestion,
    Answerability,
    BenchmarkValidationError,
    CitationRequirements,
    HumanReviewStatus,
    RequiredEvidence,
    load_answer_benchmark,
    require_official_benchmark,
    require_reviewed_benchmark,
)

#: The SHIPPED answer benchmark. Note this is deliberately NOT
#: `automobile-engineering-baseline-v1.json`: that file is the RETRIEVAL
#: benchmark (its questions carry `expected_chunk_ids`, not `question_id`) and
#: cannot be loaded as an `AnswerBenchmark` at all.
SHIPPED_ANSWER_BENCHMARK = (
    Path(__file__).resolve().parent.parent.parent
    / "benchmarks"
    / "answer-quality-automobile-v1.json"
)


def _question(**overrides):
    base = dict(
        question_id="q-001",
        question="What must the metacentric height GM be for stable equilibrium?",
        answerability=Answerability.ANSWERABLE,
        required_evidence=[RequiredEvidence(chunk_id="chk_good", document_id="doc_chk", content_hash="h")],
        expected_grounding_state="ANY_ACCEPTABLE",
    )
    base.update(overrides)
    return AnswerBenchmarkQuestion(**base)


def _benchmark(**overrides):
    base = dict(
        benchmark="unit-review",
        kb_id="kb_unit",
        questions=[_question()],
    )
    base.update(overrides)
    return AnswerBenchmark(**base)


class TestReviewStatusClosedVocabulary:
    def test_pending_parses(self):
        q = _question(review_status="pending")
        assert q.review_status == "pending"

    def test_approved_parses(self):
        q = _question(review_status="approved")
        assert q.review_status == "approved"

    def test_out_of_vocabulary_rejected(self):
        with pytest.raises(ValueError, match="review_status"):
            _question(review_status="probably-fine")


class TestOfficialGate:
    def test_official_requires_frozen(self):
        with pytest.raises(BenchmarkValidationError, match="FROZEN"):
            require_official_benchmark(_benchmark())

    def test_frozen_passes(self):
        require_official_benchmark(_benchmark(lifecycle=AnswerBenchmarkLifecycle.FROZEN))

    def test_approved_still_not_official(self):
        with pytest.raises(BenchmarkValidationError, match="FROZEN"):
            require_official_benchmark(_benchmark(lifecycle=AnswerBenchmarkLifecycle.APPROVED))


class TestReviewedGate:
    def test_reviewed_passes(self):
        bench = _benchmark(questions=[_question(review_status="approved")])
        require_reviewed_benchmark(bench)

    def test_not_yet_reviewed_fails(self):
        with pytest.raises(BenchmarkValidationError, match="approved"):
            require_reviewed_benchmark(_benchmark())


class TestShippedArtifactState:
    """The honest state of the shipped benchmark — the V8 limitation V9 measures."""

    def test_shipped_artifact_loads_as_draft_and_pending(self):
        bench = load_answer_benchmark(SHIPPED_ANSWER_BENCHMARK)
        assert bench.lifecycle is AnswerBenchmarkLifecycle.DRAFT, (
            "a benchmark that does not say it is frozen must load as DRAFT"
        )
        assert bench.human_review is HumanReviewStatus.PENDING
        assert len(bench.questions) == 28

    def test_no_shipped_question_claims_human_approval(self):
        """No question may be treated as reviewed without a recorded review."""
        bench = load_answer_benchmark(SHIPPED_ANSWER_BENCHMARK)
        approved = [q.question_id for q in bench.questions if q.review_status == "approved"]
        assert approved == [], (
            "the shipped artifact was authored by an agent and must not claim "
            f"human approval for any question; found {approved}"
        )

    def test_reviewed_gate_refuses_the_shipped_artifact(self):
        bench = load_answer_benchmark(SHIPPED_ANSWER_BENCHMARK)
        with pytest.raises(BenchmarkValidationError, match="not approved"):
            require_reviewed_benchmark(bench)

    def test_official_gate_refuses_the_shipped_artifact(self):
        bench = load_answer_benchmark(SHIPPED_ANSWER_BENCHMARK)
        with pytest.raises(BenchmarkValidationError, match="FROZEN"):
            require_official_benchmark(bench)
