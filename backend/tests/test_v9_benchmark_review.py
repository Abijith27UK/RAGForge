"""V9 Phase 1 regression tests: append-only benchmark review + provenance.

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
    AnswerBenchmarkQuestion,
    Answerability,
    BenchmarkValidationError,
    RequiredEvidence,
)
from app.services.answer_eval.benchmark_review import (  # noqa: E402
    BenchmarkReviewError,
    DimensionVerdict,
    QuestionReview,
    QuestionReviewLog,
    REQUIRED_DIMENSIONS,
    ReviewDimension,
    ReviewStatus,
    apply_review_status,
    assert_not_frozen_artifact,
    derive_review_status,
    review_coverage,
    unreviewed_questions,
)
from app.services.answer_eval.provenance import (  # noqa: E402
    ProvenanceError,
    ProvenanceKind,
    ProvenanceTag,
    agent_annotation_provenance,
    assert_provenance_complete,
    create_provenance,
    hash_provenance,
    human_review_provenance,
    validate_provenance,
)


def _benchmark(question_ids=("q1", "q2")) -> AnswerBenchmark:
    return AnswerBenchmark(
        benchmark="unit-review",
        kb_id="kb_unit",
        questions=[
            AnswerBenchmarkQuestion(
                question_id=qid,
                question=f"Question {qid}?",
                answerability=Answerability.ANSWERABLE,
                required_evidence=[
                    RequiredEvidence(chunk_id=f"chk_{qid}", document_id=f"doc_{qid}")
                ],
            )
            for qid in question_ids
        ],
    )


def _approving_review(question_id: str = "q1", **overrides) -> QuestionReview:
    base = dict(
        review_id=f"rev_{question_id}_1",
        kb_id="kb_unit",
        benchmark_name="unit-review",
        question_id=question_id,
        reviewer="human-reviewer",
        verdicts={
            ReviewDimension.REFERENCE_ANSWER.value: DimensionVerdict.OK,
            ReviewDimension.KEY_POINTS.value: DimensionVerdict.OK,
            ReviewDimension.AMBIGUITY.value: DimensionVerdict.OK,
            ReviewDimension.ANSWERABILITY.value: DimensionVerdict.OK,
            ReviewDimension.EVIDENCE_SUFFICIENCY.value: DimensionVerdict.OK,
        },
    )
    base.update(overrides)
    return QuestionReview(**base)


class TestAppendOnly:
    def test_duplicate_review_id_is_refused(self):
        log = QuestionReviewLog()
        log.append(_approving_review("q1"))
        with pytest.raises(BenchmarkReviewError, match="append-only"):
            log.append(_approving_review("q1"))

    def test_log_exposes_no_update_or_delete(self):
        log = QuestionReviewLog()
        assert not hasattr(log, "update")
        assert not hasattr(log, "delete")
        assert not hasattr(log, "replace")

    def test_correction_is_a_new_review_that_supersedes(self):
        log = QuestionReviewLog()
        log.append(_approving_review("q1"))
        log.append(
            _approving_review(
                "q1",
                review_id="rev_q1_2",
                reviewer="second-reviewer",
                verdicts={ReviewDimension.REFERENCE_ANSWER.value: DimensionVerdict.WRONG},
                supersedes="rev_q1_1",
            )
        )
        # History is preserved...
        assert len(log.all()) == 2
        # ...while the effective review is the correction.
        effective = log.effective_for_question("q1")
        assert effective is not None and effective.review_id == "rev_q1_2"

    def test_reviewer_is_required(self):
        with pytest.raises(BenchmarkReviewError, match="reviewer"):
            _approving_review("q1", reviewer="  ")


class TestApprovalRequiresEveryDimension:
    def test_all_dimensions_assessed_and_affirmative_is_an_approval(self):
        review = _approving_review("q1")
        assert review.is_approval() is True
        assert review.blocks_approval() == []

    @pytest.mark.parametrize("dimension", REQUIRED_DIMENSIONS)
    def test_unassessed_required_dimension_blocks_approval(self, dimension):
        verdicts = {
            d.value: DimensionVerdict.OK
            for d in REQUIRED_DIMENSIONS
            if d is not dimension
        }
        review = _approving_review("q1", verdicts=verdicts)
        assert review.is_approval() is False
        assert any(dimension.value in b and "not assessed" in b for b in review.blocks_approval())

    def test_ambiguity_flag_blocks_approval(self):
        """An ambiguous question must never be laundered into an approval."""
        review = _approving_review(
            "q1",
            verdicts={
                ReviewDimension.REFERENCE_ANSWER.value: DimensionVerdict.OK,
                ReviewDimension.KEY_POINTS.value: DimensionVerdict.OK,
                ReviewDimension.AMBIGUITY.value: DimensionVerdict.AMBIGUOUS,
                ReviewDimension.ANSWERABILITY.value: DimensionVerdict.OK,
                ReviewDimension.EVIDENCE_SUFFICIENCY.value: DimensionVerdict.OK,
            },
        )
        assert review.is_approval() is False
        assert any("ambiguous" in b for b in review.blocks_approval())

    def test_unknown_verdict_blocks_approval(self):
        """A reviewer who could not determine a dimension has not approved."""
        review = _approving_review(
            "q1",
            verdicts={
                ReviewDimension.REFERENCE_ANSWER.value: DimensionVerdict.UNKNOWN,
                ReviewDimension.KEY_POINTS.value: DimensionVerdict.OK,
                ReviewDimension.ANSWERABILITY.value: DimensionVerdict.OK,
                ReviewDimension.EVIDENCE_SUFFICIENCY.value: DimensionVerdict.OK,
            },
        )
        assert review.is_approval() is False

    def test_ok_with_note_still_approves(self):
        review = _approving_review(
            "q1",
            verdicts={
                ReviewDimension.REFERENCE_ANSWER.value: DimensionVerdict.OK_WITH_NOTE,
                ReviewDimension.KEY_POINTS.value: DimensionVerdict.OK,
                ReviewDimension.ANSWERABILITY.value: DimensionVerdict.OK,
                ReviewDimension.EVIDENCE_SUFFICIENCY.value: DimensionVerdict.OK,
                ReviewDimension.AMBIGUITY.value: DimensionVerdict.OK,
            },
        )
        assert review.is_approval() is True


class TestDeriveReviewStatus:
    def test_no_review_is_pending(self):
        assert derive_review_status([]) is ReviewStatus.PENDING

    def test_blocking_review_is_reviewed_not_approved(self):
        review = _approving_review(
            "q1", verdicts={ReviewDimension.AMBIGUITY.value: DimensionVerdict.AMBIGUOUS}
        )
        assert derive_review_status([review]) is ReviewStatus.REVIEWED

    def test_approval_review_is_approved(self):
        assert derive_review_status([_approving_review("q1")]) is ReviewStatus.APPROVED

    def test_stale_fingerprint_review_is_ignored(self):
        review = _approving_review("q1", benchmark_fingerprint="old_fingerprint")
        assert (
            derive_review_status([review], benchmark_fingerprint="new_fingerprint")
            is ReviewStatus.PENDING
        )


class TestApplyReviewStatus:
    def test_apply_sets_statuses_and_counts(self):
        bench = _benchmark(("q1", "q2"))
        result = apply_review_status(bench, [_approving_review("q1")])
        assert result.statuses == {"q1": "approved", "q2": "pending"}
        assert result.approved_count == 1
        assert result.pending_count == 1
        assert [q.review_status for q in bench.questions] == ["approved", "pending"]
        assert result.all_approved is False

    def test_unbacked_approval_in_the_artifact_is_downgraded(self):
        """The core safety property: an approval with no review does not survive."""
        bench = _benchmark(("q1",))
        bench.questions[0].review_status = "approved"
        result = apply_review_status(bench, [])
        assert bench.questions[0].review_status == "pending"
        assert result.pending_count == 1
        assert result.approved_count == 0

    def test_stale_review_is_reported_and_does_not_approve(self):
        bench = _benchmark(("q1",))
        stale = _approving_review("q1", benchmark_fingerprint="different")
        result = apply_review_status(bench, [stale])
        assert result.stale_review_question_ids == ["q1"]
        assert bench.questions[0].review_status == "pending"

    def test_blockers_are_reported_for_reviewed_but_unapproved(self):
        bench = _benchmark(("q1",))
        review = _approving_review(
            "q1",
            verdicts={
                ReviewDimension.EVIDENCE_SUFFICIENCY.value: DimensionVerdict.NEEDS_REVISION
            },
        )
        result = apply_review_status(bench, [review])
        assert result.reviewed_count == 1
        assert "evidence_sufficiency" in " ".join(result.blockers["q1"])


class TestReviewCoverage:
    def test_coverage_counts_unreviewed_questions(self):
        bench = _benchmark(("q1", "q2", "q3"))
        coverage = review_coverage(bench, [_approving_review("q1")])
        assert coverage["question_count"] == 3
        assert coverage["reviewed_question_count"] == 1
        assert coverage["unreviewed_question_count"] == 2
        assert coverage["unreviewed_question_ids"] == ["q2", "q3"]
        assert coverage["approval_review_count"] == 1

    def test_unassessed_dimensions_are_reported(self):
        bench = _benchmark(("q1",))
        coverage = review_coverage(bench, [_approving_review("q1")])
        assert "acceptable_elements" in coverage["unassessed_dimensions"]

    def test_empty_reviews_report_no_reviewers(self):
        coverage = review_coverage(_benchmark(("q1",)), [])
        assert coverage["reviewers"] == []
        assert coverage["approval_review_count"] == 0


def test_unreviewed_questions_lists_everything_not_approved():
    bench = _benchmark(("q1", "q2"))
    apply_review_status(bench, [_approving_review("q1")])
    assert unreviewed_questions(bench) == ["q2"]


class TestFrozenArtifactGuard:
    def test_frozen_benchmark_file_is_refused(self):
        with pytest.raises(BenchmarkValidationError, match="FROZEN"):
            assert_not_frozen_artifact("benchmarks/answer-quality-automobile-v1.json")

    def test_results_suffix_is_refused(self):
        with pytest.raises(BenchmarkValidationError, match="frozen result artifact"):
            assert_not_frozen_artifact("benchmarks/whatever-results.json")

    def test_new_file_name_is_allowed(self):
        assert_not_frozen_artifact("benchmarks/v9-candidate-experiment.json")


class TestProvenance:
    def test_agent_annotation_is_not_tagged_human(self):
        block = agent_annotation_provenance(
            kind=ProvenanceKind.REFERENCE_ANSWER, method="read the indexed chunk"
        )
        assert ProvenanceTag.AGENT in block.tags
        assert ProvenanceTag.HUMAN_REVIEW not in block.tags
        assert block.is_human_reviewed() is False

    def test_human_review_requires_a_reviewer(self):
        with pytest.raises(ProvenanceError, match="reviewer"):
            human_review_provenance(
                kind=ProvenanceKind.REVIEW_HISTORY, reviewer="  ", method="checked"
            )

    def test_human_review_records_the_reviewer(self):
        block = human_review_provenance(
            kind=ProvenanceKind.REVIEW_HISTORY, reviewer="ada", method="verified claim"
        )
        assert block.is_human_reviewed() is True
        assert block.data["reviewer"] == "ada"

    def test_empty_method_is_reported(self):
        block = create_provenance(kind=ProvenanceKind.AMBIGUITY, method="")
        assert any("method is empty" in p for p in validate_provenance([block]))

    def test_duplicate_kind_is_ambiguous(self):
        blocks = [
            create_provenance(kind=ProvenanceKind.KEY_POINTS, method="a"),
            create_provenance(kind=ProvenanceKind.KEY_POINTS, method="b"),
        ]
        assert any("declared 2 times" in p for p in validate_provenance(blocks))

    def test_shared_source_across_kinds_is_not_a_problem(self):
        blocks = [
            create_provenance(
                kind=ProvenanceKind.KEY_POINTS, method="a", source="https://example.org"
            ),
            create_provenance(
                kind=ProvenanceKind.REFERENCE_ANSWER, method="b", source="https://example.org"
            ),
        ]
        assert validate_provenance(blocks) == []

    def test_empty_blocks_are_refused(self):
        with pytest.raises(ProvenanceError, match="must not be empty"):
            assert_provenance_complete(blocks=[])

    def test_hash_is_stable_and_order_sensitive(self):
        a = create_provenance(kind=ProvenanceKind.KEY_POINTS, method="m", timestamp="2026-01-01T00:00:00+00:00")
        b = create_provenance(kind=ProvenanceKind.AMBIGUITY, method="m", timestamp="2026-01-01T00:00:00+00:00")
        assert hash_provenance([a, b]) == hash_provenance([a, b])
        assert hash_provenance([a, b]) != hash_provenance([b, a])
