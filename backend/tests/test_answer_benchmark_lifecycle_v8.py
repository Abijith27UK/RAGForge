"""V8 STEP 3 regression tests — answer-benchmark lifecycle and official gating.

Key properties:
* lifecycle defaults to DRAFT — an artifact that does not say it is frozen is
  not frozen (a new domain never receives ground truth implicitly);
* only FROZEN benchmarks pass `require_official_benchmark`;
* optional human-review metadata (difficulty/reviewer/review_status) parses
  when present, rejects out-of-vocabulary values, and is never backfilled;
* the fingerprint covers question CONTENT, so re-labelling lifecycle does not
  silently change the identity of an artifact's question set.

Hermetic: reads the shipped artifacts read-only; no network, no Qdrant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.benchmark import (  # noqa: E402
    AnswerBenchmark,
    AnswerBenchmarkLifecycle,
    AnswerBenchmarkQuestion,
    Answerability,
    BenchmarkValidationError,
    ExpectedGroundingState,
    HumanReviewStatus,
    RequiredEvidence,
    load_answer_benchmark,
    require_official_benchmark,
)
from app.services.answer_eval.evaluator import DeterministicAnswerEvaluator  # noqa: E402
from app.services.answer_eval.run import AnswerEvaluationConfig, build_run  # noqa: E402

ANSWER_BENCHMARK = ROOT / "benchmarks" / "answer-quality-automobile-v1.json"


def _question(**overrides) -> AnswerBenchmarkQuestion:
    base = dict(
        question_id="q-001",
        question="What must the metacentric height GM be for stable equilibrium?",
        answerability=Answerability.ANSWERABLE,
        required_evidence=[
            RequiredEvidence(chunk_id="chk_good", document_id="doc_chk_good",
                             content_hash="hash_chk_good")
        ],
        expected_grounding_state=ExpectedGroundingState.ANY_ACCEPTABLE,
    )
    base.update(overrides)
    return AnswerBenchmarkQuestion(**base)


def _benchmark(**overrides) -> AnswerBenchmark:
    base = dict(
        benchmark="unit-lifecycle",
        kb_id="kb_unit",
        questions=[_question()],
    )
    base.update(overrides)
    return AnswerBenchmark(**base)


class TestLifecycleDefault:
    def test_default_lifecycle_is_draft(self):
        assert _benchmark().lifecycle is AnswerBenchmarkLifecycle.DRAFT

    def test_shipped_artifact_without_lifecycle_field_loads_as_draft(self):
        """A frozen-looking file that never declares `lifecycle` is DRAFT."""
        bench = load_answer_benchmark(ANSWER_BENCHMARK)
        assert bench.lifecycle is AnswerBenchmarkLifecycle.DRAFT
        assert bench.human_review is HumanReviewStatus.PENDING

    def test_frozen_lifecycle_parses(self):
        bench = _benchmark(lifecycle=AnswerBenchmarkLifecycle.FROZEN)
        assert bench.lifecycle is AnswerBenchmarkLifecycle.FROZEN
        payload = json.loads(bench.model_dump_json())
        assert payload["lifecycle"] == "frozen"


class TestOfficialGate:
    def test_official_run_requires_frozen(self):
        with pytest.raises(BenchmarkValidationError, match="FROZEN"):
            require_official_benchmark(_benchmark())  # draft

    def test_frozen_passes_the_gate(self):
        require_official_benchmark(
            _benchmark(lifecycle=AnswerBenchmarkLifecycle.FROZEN))

    def test_approved_still_is_not_official(self):
        with pytest.raises(BenchmarkValidationError, match="FROZEN"):
            require_official_benchmark(
                _benchmark(lifecycle=AnswerBenchmarkLifecycle.APPROVED))

    def test_gate_message_points_at_the_official_flag(self):
        with pytest.raises(BenchmarkValidationError, match="official=false"):
            require_official_benchmark(_benchmark())


class TestReviewMetadata:
    def test_metadata_round_trips(self):
        q = _question(difficulty="Medium", reviewer="ada@example.edu",
                      review_status="REVIEWED")
        assert q.difficulty == "medium"      # normalised
        assert q.reviewer == "ada@example.edu"
        assert q.review_status == "reviewed"

    def test_absent_metadata_stays_null(self):
        q = _question()
        assert q.difficulty is None
        assert q.reviewer is None
        assert q.review_status is None

    def test_empty_strings_normalise_to_null(self):
        q = _question(difficulty="", review_status="  ")
        assert q.difficulty is None
        assert q.review_status is None

    def test_out_of_vocabulary_values_are_rejected(self):
        with pytest.raises(ValueError, match="difficulty"):
            _question(difficulty="impossible")
        with pytest.raises(ValueError, match="review_status"):
            _question(review_status="probably-fine")


class TestRunRecordCarriesLifecycle:
    def test_build_run_records_official_and_lifecycle(self):
        bench = _benchmark(lifecycle=AnswerBenchmarkLifecycle.FROZEN)
        run = build_run(
            kb_id="kb_unit",
            benchmark=bench,
            benchmark_path="benchmarks/unit.json",
            questions=bench.questions,
            results=[],
            config=AnswerEvaluationConfig(benchmark_path="benchmarks/unit.json",
                                          official=True),
            retrieval_run_ids=[],
            generator="gen",
            model="model",
            is_mock=False,
            prompt_version="p1",
            answerer_version="a1",
            evaluator=DeterministicAnswerEvaluator(),
            run_id="aerun_test",
        )
        assert run.official is True
        assert run.benchmark_lifecycle == "frozen"
        assert run.benchmark_fingerprint  # still recorded

    def test_default_run_is_unofficial_and_draft(self):
        bench = _benchmark()
        run = build_run(
            kb_id="kb_unit",
            benchmark=bench,
            benchmark_path="benchmarks/unit.json",
            questions=bench.questions,
            results=[],
            config=AnswerEvaluationConfig(benchmark_path="benchmarks/unit.json"),
            retrieval_run_ids=[],
            generator="gen",
            model="model",
            is_mock=False,
            prompt_version="p1",
            answerer_version="a1",
            evaluator=DeterministicAnswerEvaluator(),
            run_id="aerun_test2",
        )
        assert run.official is False
        assert run.benchmark_lifecycle == "draft"


class TestFingerprintStability:
    def test_lifecycle_does_not_change_the_fingerprint(self):
        """Comparability depends on question CONTENT, not metadata labels."""
        a = _benchmark()
        b = _benchmark(lifecycle=AnswerBenchmarkLifecycle.FROZEN)
        assert a.fingerprint() == b.fingerprint()

    def test_question_metadata_does_not_change_the_fingerprint(self):
        a = _benchmark()
        b = _benchmark(questions=[_question(difficulty="hard",
                                            reviewer="ada",
                                            review_status="reviewed")])
        assert a.fingerprint() == b.fingerprint()

    def test_changed_question_text_does_change_it(self):
        a = _benchmark()
        b = _benchmark(questions=[_question(question="Something else?")])
        assert a.fingerprint() != b.fingerprint()
