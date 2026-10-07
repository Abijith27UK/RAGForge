"""V10 ground-truth workflow tests: states, validation, completeness, gate, freeze.

Pure unit tests. No network, no Qdrant, no filesystem writes.
"""

from __future__ import annotations

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
    HumanReviewStatus,
    RequiredEvidence,
    load_answer_benchmark,
)
from app.services.answer_eval.benchmark_review import (  # noqa: E402
    DimensionVerdict,
    QuestionReview,
    ReviewDimension,
)
from app.services.answer_eval.ground_truth import (  # noqa: E402
    ApprovalPolicy,
    GroundTruthAnnotation,
    GroundTruthError,
    GroundTruthLog,
    QuestionState,
    apply_question_states,
    build_answer_benchmark_version,
    build_author_provenance,
    compute_completeness,
    derive_question_state,
    evaluate_approval_gate,
    ground_truth_fingerprint,
    validate_annotation,
    validate_annotation_against_corpus,
    verify_answer_benchmark_version,
)

SHIPPED_ANSWER_BENCHMARK = (
    Path(__file__).resolve().parent.parent.parent
    / "benchmarks"
    / "answer-quality-automobile-v1.json"
)

ALL_DIMENSIONS = [
    ReviewDimension.REFERENCE_ANSWER,
    ReviewDimension.KEY_POINTS,
    ReviewDimension.AMBIGUITY,
    ReviewDimension.ANSWERABILITY,
    ReviewDimension.EVIDENCE_SUFFICIENCY,
]


def _benchmark(question_ids=("q1", "q2"), **overrides) -> AnswerBenchmark:
    base = dict(
        benchmark="unit-v10-benchmark",
        kb_id="kb_unit",
        questions=[
            AnswerBenchmarkQuestion(
                question_id=qid,
                question=f"Question {qid}?",
                answerability=Answerability.ANSWERABLE,
                required_evidence=[
                    RequiredEvidence(
                        chunk_id=f"chk_{qid}",
                        document_id=f"doc_{qid}",
                        content_hash=f"hash_{qid}",
                    )
                ],
            )
            for qid in question_ids
        ],
    )
    base.update(overrides)
    return AnswerBenchmark(**base)


def _chunk_index(question_ids=("q1", "q2")) -> dict[str, dict]:
    return {
        f"chk_{qid}": {
            "document_id": f"doc_{qid}",
            "content_hash": f"hash_{qid}",
            "kb_id": "kb_unit",
        }
        for qid in question_ids
    }


def _annotation(question_id: str = "q1", **overrides) -> GroundTruthAnnotation:
    base = dict(
        annotation_id=f"bgt_{question_id}_1",
        kb_id="kb_unit",
        benchmark_name="unit-v10-benchmark",
        question_id=question_id,
        author="ada-reviewer",
        expected_answer=f"Answer for {question_id}.",
        key_points=[f"point one for {question_id}"],
        acceptable_answer_elements=[],
        evidence=[
            RequiredEvidence(
                chunk_id=f"chk_{question_id}",
                document_id=f"doc_{question_id}",
                content_hash=f"hash_{question_id}",
            )
        ],
        provenance=[
            build_author_provenance(
                author="ada-reviewer",
                method="read the cited chunk",
                source=f"chk_{question_id}",
            )
        ],
    )
    base.update(overrides)
    return GroundTruthAnnotation(**base)


def _review(question_id: str = "q1", **overrides) -> QuestionReview:
    base = dict(
        review_id=f"bqrev_{question_id}_1",
        kb_id="kb_unit",
        benchmark_name="unit-v10-benchmark",
        question_id=question_id,
        reviewer="ada-reviewer",
        outcome=QuestionState.APPROVED.value,
        verdicts={d.value: DimensionVerdict.OK for d in ALL_DIMENSIONS},
    )
    base.update(overrides)
    return QuestionReview(**base)


def _apply(benchmark, annotations, reviews):
    return apply_question_states(benchmark, annotations, reviews, _chunk_index())


class TestShippedBenchmarkIdentity:
    """The artifact V10 operates on, and its honest starting state."""

    def test_identity_is_the_answer_benchmark_not_the_retrieval_one(self):
        bench = load_answer_benchmark(SHIPPED_ANSWER_BENCHMARK)
        assert bench.benchmark == "answer-quality-automobile-v1"
        assert bench.kb_id == "kb_f278c283c748"
        assert bench.question_count == 28
        assert bench.lifecycle is AnswerBenchmarkLifecycle.DRAFT
        assert bench.human_review is HumanReviewStatus.PENDING

    def test_no_question_carries_human_labels_yet(self):
        bench = load_answer_benchmark(SHIPPED_ANSWER_BENCHMARK)
        for q in bench.questions:
            assert q.expected_answer is None
            assert q.key_points == []
            assert q.acceptable_answer_elements == []

    def test_every_question_starts_pending(self):
        bench = load_answer_benchmark(SHIPPED_ANSWER_BENCHMARK)
        application = _apply(bench, [], [])
        assert set(application.states.values()) == {QuestionState.PENDING.value}
        assert application.counts[QuestionState.PENDING.value] == 28


class TestAnnotationValidation:
    def test_author_is_required(self):
        with pytest.raises(GroundTruthError, match="author"):
            _annotation(author="   ")

    def test_annotation_with_nothing_to_approve_is_a_problem(self):
        annotation = _annotation(expected_answer="", key_points=[])
        assert any("nothing a reviewer could approve" in p for p in validate_annotation(annotation))

    def test_blank_key_point_is_a_problem(self):
        annotation = _annotation(key_points=["ok", "   "])
        assert any("key_points[1]" in p for p in validate_annotation(annotation))

    def test_evidence_is_required(self):
        annotation = _annotation(evidence=[])
        assert any("cites no corpus evidence" in p for p in validate_annotation(annotation))

    def test_provenance_is_required(self):
        annotation = _annotation(provenance=[])
        assert any("provenance" in p for p in validate_annotation(annotation))

    def test_provenance_must_record_human_authorship(self):
        from app.services.answer_eval.provenance import agent_annotation_provenance, ProvenanceKind

        annotation = _annotation(
            provenance=[
                agent_annotation_provenance(
                    kind=ProvenanceKind.REFERENCE_ANSWER, method="agent draft"
                )
            ]
        )
        problems = validate_annotation(annotation)
        assert any("HUMAN_REVIEW-tagged" in p for p in problems)

    def test_valid_annotation_has_no_problems(self):
        assert validate_annotation(_annotation()) == []


class TestGroundTruthLog:
    def test_duplicate_annotation_id_is_refused(self):
        log = GroundTruthLog()
        log.append(_annotation("q1"))
        with pytest.raises(GroundTruthError, match="append-only"):
            log.append(_annotation("q1"))

    def test_correction_supersedes_while_history_survives(self):
        log = GroundTruthLog()
        log.append(_annotation("q1"))
        log.append(
            _annotation(
                "q1",
                annotation_id="bgt_q1_2",
                expected_answer="Corrected.",
                supersedes="bgt_q1_1",
            )
        )
        assert len(log.all()) == 2
        effective = log.effective_for_question("q1")
        assert effective is not None and effective.annotation_id == "bgt_q1_2"

    def test_log_exposes_no_update_or_delete(self):
        log = GroundTruthLog()
        for forbidden in ("update", "delete", "replace"):
            assert not hasattr(log, forbidden)


class TestProvenanceAgainstCorpus:
    def test_valid_evidence_passes(self):
        assert validate_annotation_against_corpus(_annotation(), _chunk_index()) == []

    def test_missing_chunk_is_reported(self):
        problems = validate_annotation_against_corpus(_annotation(), {})
        assert any("not in the corpus" in p for p in problems)

    def test_content_hash_mismatch_is_reported(self):
        index = _chunk_index()
        index["chk_q1"]["content_hash"] = "different"
        problems = validate_annotation_against_corpus(_annotation(), index)
        assert any("content hash differs" in p for p in problems)

    def test_document_mismatch_is_reported(self):
        index = _chunk_index()
        index["chk_q1"]["document_id"] = "doc_other"
        problems = validate_annotation_against_corpus(_annotation(), index)
        assert any("is now in document" in p for p in problems)

    def test_wrong_knowledge_base_is_reported(self):
        index = _chunk_index()
        index["chk_q1"]["kb_id"] = "kb_other"
        problems = validate_annotation_against_corpus(_annotation(), index)
        assert any("belongs to knowledge base" in p for p in problems)


class TestQuestionStates:
    def test_no_input_is_pending(self):
        state, reasons = derive_question_state(annotation=None, reviews=[])
        assert state is QuestionState.PENDING
        assert any("no reviewer" in r for r in reasons)

    def test_draft_without_review_is_in_review(self):
        state, _ = derive_question_state(annotation=_annotation(), reviews=[])
        assert state is QuestionState.IN_REVIEW

    def test_affirmative_review_approves_a_valid_annotation(self):
        state, _ = derive_question_state(
            annotation=_annotation(), annotation_problems=[], reviews=[_review()]
        )
        assert state is QuestionState.APPROVED

    def test_annotation_problems_block_approval(self):
        state, reasons = derive_question_state(
            annotation=_annotation(),
            annotation_problems=["cited chunk is not in the corpus"],
            reviews=[_review()],
        )
        assert state is QuestionState.IN_REVIEW
        assert any("not valid" in r for r in reasons)

    def test_unassessed_required_dimension_blocks_approval(self):
        partial = _review(
            outcome=None,
            verdicts={
                ReviewDimension.REFERENCE_ANSWER.value: DimensionVerdict.OK,
                ReviewDimension.KEY_POINTS.value: DimensionVerdict.OK,
                ReviewDimension.ANSWERABILITY.value: DimensionVerdict.OK,
                ReviewDimension.EVIDENCE_SUFFICIENCY.value: DimensionVerdict.OK,
                # ambiguity deliberately NOT assessed
            },
        )
        state, reasons = derive_question_state(
            annotation=_annotation(), reviews=[partial]
        )
        assert state is QuestionState.IN_REVIEW
        assert any("ambiguity" in r for r in reasons)

    @pytest.mark.parametrize(
        "outcome,expected",
        [
            (QuestionState.REJECTED, QuestionState.REJECTED),
            (QuestionState.AMBIGUOUS, QuestionState.AMBIGUOUS),
            (
                QuestionState.INSUFFICIENT_EVIDENCE,
                QuestionState.INSUFFICIENT_EVIDENCE,
            ),
        ],
    )
    def test_explicit_terminal_outcomes_win(self, outcome, expected):
        state, _ = derive_question_state(
            annotation=_annotation(),
            reviews=[_review(outcome=outcome.value, verdicts={})],
        )
        assert state is expected

    def test_approval_claimed_without_annotation_stays_in_review(self):
        state, reasons = derive_question_state(
            annotation=None, reviews=[_review()]
        )
        assert state is QuestionState.IN_REVIEW
        assert any("no reference answer" in r for r in reasons)

    def test_stale_annotation_fingerprint_blocks_approval(self):
        bench = _benchmark()
        stale = _annotation(benchmark_fingerprint="deadbeefdeadbeef")
        application = _apply(bench, [stale], [_review()])
        assert application.states["q1"] == QuestionState.IN_REVIEW.value
        assert any("STALE" in p for p in application.problems["q1"])

    def test_labels_are_copied_only_when_approved(self):
        bench = _benchmark()
        # Draft only: no labels are promoted.
        _apply(bench, [_annotation("q1")], [])
        q1 = bench.questions[0]
        assert q1.expected_answer is None and q1.key_points == []

        # Approved: labels ARE promoted so the evaluator can score them.
        bench2 = _benchmark()
        _apply(bench2, [_annotation("q1")], [_review("q1")])
        q1b = bench2.questions[0]
        assert q1b.expected_answer == "Answer for q1."
        assert q1b.key_points == ["point one for q1"]
        assert q1b.review_status == "approved"

    def test_second_question_untouched_by_first_decision(self):
        bench = _benchmark()
        application = _apply(bench, [_annotation("q1")], [_review("q1")])
        assert application.states == {"q1": "approved", "q2": "pending"}


class TestCompleteness:
    def test_buckets_sum_to_the_total_and_rates_are_unknown(self):
        bench = _benchmark(question_ids=tuple(f"q{i}" for i in range(1, 5)))
        application = _apply(bench, [], [])
        completeness = compute_completeness(bench, [], [], application)
        assert completeness.total == 4
        assert (
            completeness.pending
            + completeness.in_review
            + completeness.approved
            + completeness.rejected
            + completeness.ambiguous
            + completeness.insufficient_evidence
            == 4
        )
        # Missing information must NOT become zero.
        assert completeness.reference_answer_coverage.measured is False
        assert completeness.reference_answer_coverage.value is None
        assert completeness.provenance_coverage.measured is False
        assert set(completeness.unknown_metrics) == {
            "reference_answer_coverage",
            "key_point_coverage",
            "acceptable_element_coverage",
            "provenance_coverage",
        }

    def test_mixed_review_state_is_counted_per_bucket(self):
        bench = _benchmark(question_ids=("q1", "q2", "q3", "q4"))
        index = {
            f"chk_{qid}": {
                "document_id": f"doc_{qid}",
                "content_hash": f"hash_{qid}",
                "kb_id": "kb_unit",
            }
            for qid in ("q1", "q2", "q3", "q4")
        }
        application = apply_question_states(
            bench,
            [_annotation("q1"), _annotation("q2"), _annotation("q3")],
            [
                _review("q1"),
                _review("q2", outcome=QuestionState.AMBIGUOUS.value, verdicts={}),
                _review(
                    "q3",
                    outcome=QuestionState.INSUFFICIENT_EVIDENCE.value,
                    verdicts={},
                ),
            ],
            index,
        )
        completeness = compute_completeness(bench, [], [], application)
        assert completeness.approved == 1
        assert completeness.ambiguous == 1
        assert completeness.insufficient_evidence == 1
        assert completeness.pending == 1

    def test_authored_coverage_is_measured_when_denominator_exists(self):
        bench = _benchmark()
        annotations = [_annotation("q1")]
        application = _apply(bench, annotations, [_review("q1")])
        completeness = compute_completeness(bench, annotations, [_review("q1")], application)
        assert completeness.authored_reference_answer_count == 1
        assert completeness.key_point_coverage.measured is True
        assert completeness.key_point_coverage.value == 1.0
        assert completeness.provenance_coverage.measured is True


class TestApprovalGate:
    def _gate(self, bench, annotations, reviews, policy=None):
        application = apply_question_states(
            bench,
            annotations,
            reviews,
            {
                f"chk_{q.question_id}": {
                    "document_id": f"doc_{q.question_id}",
                    "content_hash": f"hash_{q.question_id}",
                    "kb_id": bench.kb_id,
                }
                for q in bench.questions
            },
        )
        completeness = compute_completeness(bench, annotations, reviews, application)
        return evaluate_approval_gate(bench, application, completeness, policy=policy)

    def test_pending_questions_block_and_are_named(self):
        gate = self._gate(_benchmark(), [], [])
        assert gate.approved is False
        assert gate.blocking["pending"] == ["q1", "q2"]
        assert any("BLOCKED" in r and "pending" in r for r in gate.reasons)

    def test_rejected_question_blocks(self):
        bench = _benchmark(question_ids=("q1",))
        gate = self._gate(
            bench,
            [_annotation("q1")],
            [_review("q1", outcome=QuestionState.REJECTED.value, verdicts={})],
        )
        assert gate.approved is False
        assert gate.blocking["rejected"] == ["q1"]

    def test_ambiguous_and_insufficient_are_permitted_but_explicit(self):
        bench = _benchmark(question_ids=("q1", "q2", "q3"))
        gate = self._gate(
            bench,
            [_annotation("q1"), _annotation("q2"), _annotation("q3")],
            [
                _review("q1"),
                _review("q2", outcome=QuestionState.AMBIGUOUS.value, verdicts={}),
                _review(
                    "q3",
                    outcome=QuestionState.INSUFFICIENT_EVIDENCE.value,
                    verdicts={},
                ),
            ],
        )
        assert gate.approved is True
        assert gate.approved_question_ids == ["q1"]
        assert gate.excluded_question_ids == ["q2", "q3"]
        assert any("APPROVED" in r for r in gate.reasons)
        assert any("non-scoring" in r for r in gate.reasons)

    def test_approved_question_without_key_points_is_blocked_by_policy(self):
        bench = _benchmark(question_ids=("q1",))
        gate = self._gate(
            bench,
            [_annotation("q1", key_points=[])],
            [_review("q1")],
        )
        assert gate.approved is False
        assert any("key points" in p for p in gate.problems["q1"])

    def test_policy_can_relax_key_point_requirement_explicitly(self):
        bench = _benchmark(question_ids=("q1",))
        gate = self._gate(
            bench,
            [_annotation("q1", key_points=[])],
            [_review("q1")],
            policy=ApprovalPolicy(require_key_points=False),
        )
        assert gate.approved is True
        assert gate.policy.require_key_points is False

    def test_broken_provenance_blocks_approval(self):
        bench = _benchmark(question_ids=("q1",))
        application = apply_question_states(
            bench, [_annotation("q1")], [_review("q1")], {}
        )  # empty corpus index -> the cited chunk does not exist
        assert application.states["q1"] == QuestionState.IN_REVIEW.value
        completeness = compute_completeness(
            bench, [_annotation("q1")], [_review("q1")], application
        )
        gate = evaluate_approval_gate(bench, application, completeness)
        assert gate.approved is False

    def test_zero_approved_questions_cannot_pass(self):
        bench = _benchmark(question_ids=("q1",))
        gate = self._gate(
            bench,
            [_annotation("q1")],
            [_review("q1", outcome=QuestionState.AMBIGUOUS.value, verdicts={})],
        )
        assert gate.approved is False
        assert any("below the policy minimum" in r for r in gate.reasons)


class TestFreeze:
    def _frozen(self):
        bench = _benchmark(question_ids=("q1", "q2"))
        annotations = [_annotation("q1"), _annotation("q2")]
        reviews = [
            _review("q1"),
            _review("q2", outcome=QuestionState.AMBIGUOUS.value, verdicts={}),
        ]
        index = {
            f"chk_{qid}": {
                "document_id": f"doc_{qid}",
                "content_hash": f"hash_{qid}",
                "kb_id": "kb_unit",
            }
            for qid in ("q1", "q2")
        }
        application = apply_question_states(bench, annotations, reviews, index)
        completeness = compute_completeness(bench, annotations, reviews, application)
        gate = evaluate_approval_gate(bench, application, completeness)
        version = build_answer_benchmark_version(
            benchmark=bench,
            annotations=annotations,
            reviews=reviews,
            application=application,
            gate=gate,
            frozen_by="ada-reviewer",
            version=1,
            version_id="abv_unit_1",
        )
        return version, bench, annotations, reviews, gate

    def test_freeze_refuses_an_unapproved_benchmark(self):
        bench = _benchmark(question_ids=("q1",))
        application = apply_question_states(bench, [], [], _chunk_index(("q1",)))
        completeness = compute_completeness(bench, [], [], application)
        gate = evaluate_approval_gate(bench, application, completeness)
        with pytest.raises(BenchmarkValidationError, match="approval gate has not passed"):
            build_answer_benchmark_version(
                benchmark=bench,
                annotations=[],
                reviews=[],
                application=application,
                gate=gate,
                frozen_by="ada-reviewer",
                version=1,
                version_id="abv_unit_1",
            )

    def test_frozen_version_is_frozen_and_intact(self):
        version, *_ = self._frozen()
        assert version.benchmark.lifecycle is AnswerBenchmarkLifecycle.FROZEN
        assert version.benchmark.human_review is HumanReviewStatus.HUMAN_REVIEWED
        assert verify_answer_benchmark_version(version) == []
        assert version.scoring_question_ids() == ["q1"]
        assert version.excluded_question_ids() == ["q2"]

    def test_tampering_with_a_frozen_version_is_detected(self):
        version, *_ = self._frozen()
        version.benchmark.questions[0].expected_answer = "a silently edited label"
        problems = verify_answer_benchmark_version(version)
        assert any("artifact fingerprint mismatch" in p for p in problems)
        assert any("ground-truth fingerprint mismatch" in p for p in problems)

    def test_labels_change_the_ground_truth_fingerprint_not_the_content_one(self):
        bench = _benchmark()
        before = bench.fingerprint()
        annotations = [_annotation("q1")]
        _apply(bench, annotations, [_review("q1")])
        assert bench.fingerprint() == before, (
            "review metadata/labels must not change the V8 content fingerprint"
        )
        fp_one = ground_truth_fingerprint(bench, annotations)

        # Same question content, different human labels: the GT fingerprint
        # must move (otherwise two frozen versions would collide).
        bench2 = _benchmark()
        different = _annotation(
            "q1",
            annotation_id="bgt_q1_2",
            expected_answer="A different reference answer.",
            supersedes="bgt_q1_1",
        )
        _apply(bench2, [different], [_review("q1")])
        fp_two = ground_truth_fingerprint(bench2, [different])
        assert fp_one != fp_two

    def test_mutating_the_working_benchmark_after_freeze_does_not_reach_the_record(self):
        version, bench, *_ = self._frozen()
        bench.questions[0].question = "Mutated after freezing?"
        assert version.benchmark.questions[0].question != "Mutated after freezing?"
        assert verify_answer_benchmark_version(version) == []

    def test_gate_policy_is_recorded_in_the_frozen_version(self):
        version, *_ = self._frozen()
        assert version.gate.policy.policy_version == "v10-answer-benchmark-approval-v1"
        assert version.gate.approved is True


class TestReviewModelAdditions:
    def test_v9_review_without_outcome_still_loads(self):
        review = QuestionReview(
            review_id="v9-style",
            kb_id="kb_unit",
            question_id="q1",
            reviewer="ada-reviewer",
        )
        assert review.outcome is None
        assert review.annotation_id == ""

    def test_outcome_is_part_of_the_model_fingerprint(self):
        a = _review("q1")
        b = _review("q1", outcome=QuestionState.REJECTED.value)
        assert a.model_fingerprint() != b.model_fingerprint()
