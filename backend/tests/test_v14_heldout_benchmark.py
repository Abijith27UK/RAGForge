"""V14 held-out benchmark: identity, review records and the freeze gate.

These tests pin down the RULES, not the plumbing:

* a held-out artifact cannot be created without a documented creation protocol;
* a "reviewed" status cannot exist without the review record that set it, and a
  rejection cannot exist without a named human behind it;
* the freeze tier is computed from the review records present at freeze time, and
  an unreviewed set can only seal as PROVISIONAL;
* a PROVISIONAL instrument cannot support a confirmatory claim, and there is no
  entry point anywhere that raises a tier after the fact;
* a frozen artifact is immutable: different content under the same path is
  refused, identical content is an idempotent no-op, and the label fingerprint
  ignores rejected candidates.

Everything here is pure: no database, no vector store, no corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.heldout_benchmark import (  # noqa: E402
    CandidateStatus,
    ClaimStrength,
    HeldOutBenchmark,
    HeldOutCandidate,
    HeldOutCandidateSet,
    HeldOutLifecycle,
    HeldOutReviewRequest,
    HeldOutValidationError,
    QueryType,
    ReviewDecision,
    apply_review,
    claim_strength_for,
    freeze,
    load_candidate_set,
    load_heldout_benchmark,
    require_confirmatory,
    require_runnable,
    save_candidate_set,
)

PROTOCOL = {
    "method": "agent-drafted from the full text of the named chunks",
    "question_selection": "read the corpus text; no retriever was run",
    "evidence_selection": "one necessary chunk per question",
    "review_policy": "a human decides on every question before a validated freeze",
}


def _candidate(qid: str = "hout-001", **overrides) -> HeldOutCandidate:
    payload = {
        "id": qid,
        "subdomain": "engine",
        "question": "What does the corpus say about the first engine design?",
        "expected_chunk_ids": ["chk_a"],
        "expected_document_ids": ["doc_a"],
        "evidence_quote": "the first engine was built in 1897",
        "coverage_rationale": "adds engine history, which the development set omits",
        "provenance": "Chunk chk_a (doc_a, section Introduction), read 2026-10-11",
        "draft_origin": "agent-drafted from the chunk text; no retrieval",
    }
    payload.update(overrides)
    return HeldOutCandidate(**payload)


def _candidateset(candidates: list[HeldOutCandidate] | None = None, **overrides) -> HeldOutCandidateSet:
    payload = {
        "benchmark": "automobile-engineering-heldout-v1",
        "version": 1,
        "kb_id": "kb_test",
        "kb_name": "Test KB",
        "derived_from": "benchmarks/automobile-engineering-baseline-v1.json",
        "protocol": dict(PROTOCOL),
        "authorship": {"author": "test", "agent_authored": True},
        "candidates": candidates if candidates is not None else [_candidate()],
    }
    payload.update(overrides)
    return HeldOutCandidateSet(**payload)


class TestCandidateSetShape:
    def test_a_set_without_a_creation_protocol_is_refused(self):
        """Independence cannot be claimed by an artifact that cannot say how it
        was built."""
        with pytest.raises(Exception) as exc:
            _candidateset(protocol={"method": "some questions"})
        assert "protocol must document" in str(exc.value)

    def test_a_rejected_candidate_must_carry_its_review_record(self):
        with pytest.raises(Exception) as exc:
            _candidateset([_candidate(status=CandidateStatus.REJECTED)])
        assert "no review record" in str(exc.value)

    def test_an_approved_candidate_must_carry_its_review_record(self):
        with pytest.raises(Exception) as exc:
            _candidateset([_candidate(status=CandidateStatus.APPROVED)])
        assert "no review record" in str(exc.value)

    def test_review_counts_separate_surviving_from_reviewed(self):
        reviewed = apply_review(
            _candidateset([_candidate("hout-001"), _candidate("hout-002")]),
            [
                HeldOutReviewRequest(
                    question_id="hout-001",
                    decision=ReviewDecision.APPROVE,
                    reviewer="A. Reviewer",
                ),
                HeldOutReviewRequest(
                    question_id="hout-002",
                    decision=ReviewDecision.REJECT,
                    reviewer="A. Reviewer",
                    notes="not answerable from that chunk alone",
                ),
            ],
        )
        completeness = reviewed.review_completeness()
        assert completeness["candidates_total"] == 2
        assert completeness["candidates_surviving"] == 1
        assert completeness["candidates_rejected"] == 1
        assert completeness["complete"] is True
        assert completeness["reviewers"] == ["A. Reviewer"]


class TestSeparationFingerprint:
    def test_fingerprint_ignores_whitespace_and_case_only_changes(self):
        a = _candidateset([_candidate(question="What is the first engine?")])
        b = _candidateset([_candidate(question="What  is the first engine? ")])
        assert a.fingerprint() != b.fingerprint(), (
            "the fingerprint must cover the question text; only re-FORMATTING that "
            "does not change the text should be irrelevant"
        )

    def test_fingerprint_moves_when_the_evidence_moves(self):
        a = _candidateset([_candidate(expected_chunk_ids=["chk_a"])])
        b = _candidateset([_candidate(expected_chunk_ids=["chk_b"])])
        assert a.fingerprint() != b.fingerprint()

    def test_fingerprint_moves_when_a_question_is_edited(self):
        a = _candidateset([_candidate(question="What is a brake?")])
        b = _candidateset([_candidate(question="What is a clutch?")])
        assert a.fingerprint() != b.fingerprint()

    def test_recording_an_approval_does_not_move_the_fingerprint(self):
        """A review record is provenance, not content: approving a question
        changes who vouches for it, not what it measures, so the identity a
        result is attributed to must not move."""
        before = _candidateset([_candidate("hout-001"), _candidate("hout-002")])
        after = apply_review(
            before,
            [
                HeldOutReviewRequest(
                    question_id="hout-001", decision=ReviewDecision.APPROVE, reviewer="R"
                )
            ],
        )
        assert before.fingerprint() == after.fingerprint()

    def test_rejecting_a_question_does_move_the_fingerprint(self):
        """Rejecting REMOVES a question from the instrument, so the resulting
        instrument measures something different and must not share an identity
        with the one that still contained the rejected question."""
        before = _candidateset([_candidate("hout-001"), _candidate("hout-002")])
        after = apply_review(
            before,
            [
                HeldOutReviewRequest(
                    question_id="hout-002", decision=ReviewDecision.REJECT, reviewer="R"
                )
            ],
        )
        assert before.fingerprint() != after.fingerprint()

    def test_fingerprint_is_stable_across_a_save_load_round_trip(self, tmp_path):
        original = _candidateset([_candidate("hout-001"), _candidate("hout-002")])
        path = save_candidate_set(original, tmp_path / "candidates.json")
        assert load_candidate_set(path).fingerprint() == original.fingerprint()


class TestReviewRules:
    def test_an_unknown_question_id_is_an_error(self):
        """A decision that silently matched nothing would let a reviewer believe
        they had reviewed more than they had."""
        with pytest.raises(HeldOutValidationError) as exc:
            apply_review(
                _candidateset(),
                [
                    HeldOutReviewRequest(
                        question_id="hout-nope",
                        decision=ReviewDecision.APPROVE,
                        reviewer="R",
                    )
                ],
            )
        assert "unknown candidate id" in str(exc.value)

    def test_an_anonymous_decision_is_refused(self):
        with pytest.raises(Exception) as exc:
            HeldOutReviewRequest(
                question_id="hout-001", decision=ReviewDecision.APPROVE, reviewer="  "
            )
        assert "name the reviewer" in str(exc.value)

    def test_an_edit_with_nothing_replaced_is_refused(self):
        with pytest.raises(Exception) as exc:
            HeldOutReviewRequest(
                question_id="hout-001", decision=ReviewDecision.EDIT, reviewer="R"
            )
        assert "replacement value" in str(exc.value)

    def test_an_edit_records_the_previous_values(self):
        updated = apply_review(
            _candidateset(),
            [
                HeldOutReviewRequest(
                    question_id="hout-001",
                    decision=ReviewDecision.EDIT,
                    reviewer="A. Reviewer",
                    question="Which patent covered the first horizontally opposed engine?",
                    notes="narrowed to one fact",
                )
            ],
        )
        cand = updated.candidates[0]
        assert cand.status is CandidateStatus.EDITED
        assert cand.question.startswith("Which patent")
        assert cand.review is not None
        assert cand.review.previous_question == "What does the corpus say about the first engine design?"
        assert cand.review.notes == "narrowed to one fact"

    def test_results_seen_is_recorded_on_the_decision(self):
        updated = apply_review(
            _candidateset(),
            [
                HeldOutReviewRequest(
                    question_id="hout-001",
                    decision=ReviewDecision.APPROVE,
                    reviewer="R",
                    results_seen=True,
                )
            ],
        )
        assert updated.candidates[0].review is not None
        assert updated.candidates[0].review.results_seen is True

    def test_a_rejected_question_cannot_be_un_rejected_in_the_same_version(self):
        rejected = apply_review(
            _candidateset(),
            [
                HeldOutReviewRequest(
                    question_id="hout-001", decision=ReviewDecision.REJECT, reviewer="R"
                )
            ],
        )
        with pytest.raises(HeldOutValidationError) as exc:
            apply_review(
                rejected,
                [
                    HeldOutReviewRequest(
                        question_id="hout-001", decision=ReviewDecision.APPROVE, reviewer="R"
                    )
                ],
            )
        assert "cannot be re-reviewed" in str(exc.value)

    def test_the_last_decision_for_a_question_wins(self):
        updated = apply_review(
            _candidateset(),
            [
                HeldOutReviewRequest(
                    question_id="hout-001", decision=ReviewDecision.APPROVE, reviewer="R"
                ),
                HeldOutReviewRequest(
                    question_id="hout-001",
                    decision=ReviewDecision.REJECT,
                    reviewer="R",
                    notes="changed my mind before freezing",
                ),
            ],
        )
        assert updated.candidates[0].status is CandidateStatus.REJECTED


class TestClaimStrength:
    def test_only_a_validated_instrument_supports_a_confirmatory_claim(self):
        assert (
            claim_strength_for(HeldOutLifecycle.VALIDATED) is ClaimStrength.CONFIRMATORY
        )
        assert (
            claim_strength_for(HeldOutLifecycle.PROVISIONAL) is ClaimStrength.EXPLORATORY
        )
        assert claim_strength_for(HeldOutLifecycle.DRAFT) is ClaimStrength.EXPLORATORY

    def test_a_draft_instrument_is_not_runnable(self):
        with pytest.raises(HeldOutValidationError) as exc:
            require_runnable(HeldOutBenchmark(benchmark="x", lifecycle=HeldOutLifecycle.DRAFT))
        assert "DRAFT" in str(exc.value)

    def test_a_provisional_instrument_is_runnable_but_not_confirmatory(self):
        bench = HeldOutBenchmark(
            benchmark="x",
            lifecycle=HeldOutLifecycle.PROVISIONAL,
            claim_strength=ClaimStrength.EXPLORATORY,
        )
        require_runnable(bench)
        with pytest.raises(HeldOutValidationError) as exc:
            require_confirmatory(bench)
        message = str(exc.value)
        assert "exploratory results only" in message
        assert "no longer blind" in message, (
            "the refusal must explain WHY a later review cannot upgrade it"
        )


class TestFreeze:
    def test_a_fully_unreviewed_set_freezes_only_as_provisional(self, tmp_path):
        bench = freeze(
            _candidateset(),
            path=tmp_path / "frozen.json",
            frozen_by="A. Reviewer",
            validation={"errors": []},
        )
        assert bench.lifecycle is HeldOutLifecycle.PROVISIONAL
        assert bench.claim_strength is ClaimStrength.EXPLORATORY
        assert "PENDING" in str(bench.authorship["human_review"])
        assert bench.authorship["agent_authored"] is True

    def test_a_fully_reviewed_set_freezes_as_validated(self, tmp_path):
        reviewed = apply_review(
            _candidateset([_candidate("hout-001"), _candidate("hout-002")]),
            [
                HeldOutReviewRequest(
                    question_id=q, decision=ReviewDecision.APPROVE, reviewer="A. Reviewer"
                )
                for q in ("hout-001", "hout-002")
            ],
        )
        bench = freeze(
            reviewed, path=tmp_path / "frozen.json", frozen_by="A. Reviewer", validation={"errors": []}
        )
        assert bench.lifecycle is HeldOutLifecycle.VALIDATED
        assert bench.claim_strength is ClaimStrength.CONFIRMATORY
        assert bench.authorship["human_review"].startswith("COMPLETE")
        assert bench.review["complete"] is True

    def test_one_unreviewed_question_holds_the_whole_set_at_provisional(self, tmp_path):
        """The tier is a property of the instrument, not an average of its
        questions: a result from a set with one unverified label is still a
        result from a set with one unverified label."""
        reviewed = apply_review(
            _candidateset([_candidate("hout-001"), _candidate("hout-002")]),
            [
                HeldOutReviewRequest(
                    question_id="hout-001", decision=ReviewDecision.APPROVE, reviewer="R"
                )
            ],
        )
        bench = freeze(
            reviewed, path=tmp_path / "frozen.json", frozen_by="R", validation={"errors": []}
        )
        assert bench.lifecycle is HeldOutLifecycle.PROVISIONAL

    def test_rejected_questions_are_kept_and_excluded_from_the_instrument(self, tmp_path):
        reviewed = apply_review(
            _candidateset([_candidate("hout-001"), _candidate("hout-002")]),
            [
                HeldOutReviewRequest(
                    question_id="hout-001", decision=ReviewDecision.APPROVE, reviewer="R"
                ),
                HeldOutReviewRequest(
                    question_id="hout-002",
                    decision=ReviewDecision.REJECT,
                    reviewer="R",
                    notes="the chunk does not answer this",
                ),
            ],
        )
        bench = freeze(
            reviewed, path=tmp_path / "frozen.json", frozen_by="R", validation={"errors": []}
        )
        assert bench.question_count == 1
        assert bench.rejected_question_ids == ["hout-002"]
        assert bench.lifecycle is HeldOutLifecycle.VALIDATED

    def test_freezing_with_validation_errors_is_refused(self, tmp_path):
        with pytest.raises(HeldOutValidationError) as exc:
            freeze(
                _candidateset(),
                path=tmp_path / "frozen.json",
                frozen_by="R",
                validation={"errors": ["hout-001: evidence does not resolve"]},
            )
        assert "refusing to freeze" in str(exc.value)

    def test_freezing_with_no_surviving_question_is_refused(self, tmp_path):
        all_rejected = apply_review(
            _candidateset(),
            [
                HeldOutReviewRequest(
                    question_id="hout-001", decision=ReviewDecision.REJECT, reviewer="R"
                )
            ],
        )
        with pytest.raises(HeldOutValidationError) as exc:
            freeze(
                all_rejected,
                path=tmp_path / "frozen.json",
                frozen_by="R",
                validation={"errors": []},
            )
        assert "no surviving questions" in str(exc.value)

    def test_freezing_must_name_who_sealed_it(self, tmp_path):
        with pytest.raises(HeldOutValidationError) as exc:
            freeze(_candidateset(), path=tmp_path / "frozen.json", frozen_by="   ")
        assert "must name who sealed" in str(exc.value)

    def test_a_frozen_artifact_is_immutable(self, tmp_path):
        target = tmp_path / "frozen.json"
        freeze(_candidateset(), path=target, frozen_by="R", validation={"errors": []})
        with pytest.raises(HeldOutValidationError) as exc:
            freeze(
                _candidateset([_candidate(question="A different question entirely?")]),
                path=target,
                frozen_by="R",
                validation={"errors": []},
            )
        assert "immutable" in str(exc.value)

    def test_re_freezing_identical_content_is_an_idempotent_no_op(self, tmp_path):
        target = tmp_path / "frozen.json"
        first = freeze(
            _candidateset(), path=target, frozen_by="R", validation={"errors": []}
        )
        second = freeze(
            _candidateset(), path=target, frozen_by="R", validation={"errors": []}
        )
        assert first.fingerprint() == second.fingerprint()
        assert load_heldout_benchmark(target).fingerprint() == first.fingerprint()

    def test_the_frozen_artifact_uses_the_retrieval_schema_fingerprint(self, tmp_path):
        """The engine reads the frozen file through the RETRIEVAL loader, so the
        fingerprint must come out identical whichever loader is used. A
        fingerprint that changed shape between loaders would make two artifacts of
        the same instrument look different."""
        from app.services.answer_eval.retrieval_diagnostic_runner import (
            load_retrieval_benchmark,
        )

        target = tmp_path / "frozen.json"
        bench = freeze(_candidateset(), path=target, frozen_by="R", validation={"errors": []})
        via_heldout = load_heldout_benchmark(target)
        via_retrieval = load_retrieval_benchmark(target)
        assert via_heldout.fingerprint() == via_retrieval.fingerprint() == bench.fingerprint()
        # And the retrieval schema sees the held-out questions with their labels.
        assert via_retrieval.questions[0].expected_chunk_ids == ["chk_a"]
        assert via_retrieval.relevance_model == "binary"

    def test_the_frozen_artifact_records_the_candidate_fingerprint_it_came_from(self, tmp_path):
        source = _candidateset()
        bench = freeze(
            source, path=tmp_path / "frozen.json", frozen_by="R", validation={"errors": []}
        )
        assert bench.source_fingerprint == source.fingerprint()

    def test_the_written_file_is_valid_json_with_a_trailing_newline(self, tmp_path):
        target = tmp_path / "frozen.json"
        freeze(_candidateset(), path=target, frozen_by="R", validation={"errors": []})
        raw = target.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        assert json.loads(raw)["lifecycle"] == "provisional"


class TestShippedCandidateArtifact:
    """The real artifact written by the authoring script, read through the gate.

    Read-only: this asserts the shipped drafts are structurally sound and that
    they are still unreviewed, which is the honest state for agent-drafted
    questions and the reason a freeze of them can only be PROVISIONAL.
    """

    PATH = BACKEND_DIR.parent / "benchmarks" / "automobile-engineering-heldout-v1-candidates.json"

    def test_it_exists_and_loads(self):
        candidates = load_candidate_set(self.PATH)
        assert candidates.benchmark == "automobile-engineering-heldout-v1"
        assert candidates.derived_from.endswith("automobile-engineering-baseline-v1.json")
        assert len(candidates.candidates) >= 20

    def test_every_shipped_question_is_still_a_draft(self):
        candidates = load_candidate_set(self.PATH)
        assert candidates.review_completeness()["candidates_unreviewed"] == len(
            candidates.candidates
        ), (
            "the authoring script cannot mark its own work reviewed; if this fails, "
            "something approved questions without a human decision"
        )
        assert all(c.review is None for c in candidates.candidates)

    def test_it_carries_a_complete_creation_protocol(self):
        candidates = load_candidate_set(self.PATH)
        for key in ("method", "question_selection", "evidence_selection", "review_policy"):
            assert candidates.protocol.get(key), f"protocol.{key} is empty"

    def test_every_question_declares_that_it_was_agent_drafted(self):
        candidates = load_candidate_set(self.PATH)
        for cand in candidates.candidates:
            assert "agent" in cand.draft_origin.lower()
            assert cand.evidence_quote.strip(), "no quote means sufficiency is unverifiable"
            assert cand.coverage_rationale.strip()
            assert cand.provenance.strip()
