"""V8 answer-evaluation unit tests.

Hermetic: no network, no API key, no Qdrant, no real LLM. Every test builds its
own evidence/answer objects so the assertions are about the EVALUATOR, not about
retrieval quality.

Covers the required case list:
  1. correct answer + correct citation -> pass
  2. correct answer + wrong citation   -> citation failure
  3. correct answer + missing citation -> citation completeness failure
  4. unsupported claim                -> failure
  5. mixed valid/fabricated citations -> failure
  6. correct abstention               -> pass
  7. incorrect confident answer       -> fail
  8. contradictory evidence           -> contradiction detection
  9. partial answer                   -> partial score
 10. strategy comparison uses identical questions
 11. frozen benchmark cannot be modified
 12. evaluation run is immutable
 13. evaluator version is persisted
 14. provider identity is persisted
 15. no network calls in deterministic tests
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.answer import (  # noqa: E402
    Answer, AnswerStatus, Claim, ClaimType, Citation, ConfidenceCategory,
    Evidence, EvidenceAssessment, GateDecision, GroundingState, SupportStatus,
)
from app.services.answer_eval.benchmark import (  # noqa: E402
    AnswerBenchmarkQuestion, Answerability, BenchmarkValidationError,
    CitationRequirements, ExpectedGroundingState, HumanReviewStatus,
    RequiredEvidence, load_answer_benchmark,
)
from app.services.answer_eval.entailment import (  # noqa: E402
    HeuristicCitationEntailment, SupportJudgement, create_entailment_evaluator,
)
from app.services.answer_eval.evaluator import (  # noqa: E402
    DeterministicAnswerEvaluator, create_answer_evaluator,
)
from app.services.answer_eval.metrics import (  # noqa: E402
    Measured, average_measured, compute_final_score, grounding_states_compatible,
)
from app.services.answer_eval.run import (  # noqa: E402
    AnswerEvaluationConfig, AnswerEvaluationRun, aggregate_results, select_questions,
)

FROZEN = ROOT / "benchmarks" / "automobile-engineering-baseline-v1.json"
ANSWER_BENCHMARK = ROOT / "benchmarks" / "answer-quality-automobile-v1.json"

EVAL = DeterministicAnswerEvaluator()

GOOD_EVIDENCE_TEXT = (
    "The metacentric height GM must be positive for stable equilibrium of a "
    "floating body. The transverse metacentric height is derived from the "
    "righting arm GZ divided by the heel angle phi."
)
UNRELATED_TEXT = (
    "Propeller cavitation occurs when the local pressure falls below the "
    "vapour pressure of water, producing noise and pitting of the blade surface."
)


def _evidence(chunk_id: str, text: str, *, evidence_id: str, score: float = 0.9) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        kb_id="kb_test",
        title=f"Source {chunk_id}",
        content=text,
        retrieval_score=score,
        retrieval_strategy="dense",
        rank=1,
        original_rank=1,
        content_hash=f"hash_{chunk_id}",
    )


def _question(**overrides) -> AnswerBenchmarkQuestion:
    base = dict(
        question_id="q-001",
        question="What must the metacentric height GM be for stable equilibrium?",
        answerability=Answerability.ANSWERABLE,
        required_evidence=[
            RequiredEvidence(chunk_id="chk_good", document_id="doc_chk_good",
                             content_hash="hash_chk_good")
        ],
        citation_requirements=CitationRequirements(),
        expected_grounding_state=ExpectedGroundingState.ANY_ACCEPTABLE,
    )
    base.update(overrides)
    return AnswerBenchmarkQuestion(**base)


def _claim(text: str, evidence_ids: list[str], claim_id: str = "cl_001") -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        claim_type=ClaimType.FACT,
        evidence_ids=evidence_ids,
        citation_ids=[f"cite_{e}" for e in evidence_ids],
        support_status=SupportStatus.SUPPORTED,
    )


def _answer(
    *,
    text: str = "The metacentric height GM must be positive for stable equilibrium.",
    claims: list[Claim] | None = None,
    citations: list[Citation] | None = None,
    status: AnswerStatus = AnswerStatus.GROUNDED,
    state: GroundingState = GroundingState.ANSWERED,
    decision: GateDecision = GateDecision.ANSWER,
    sufficient: bool = True,
    citations_for: list[Evidence] | None = None,
) -> Answer:
    citations = citations if citations is not None else [
        Citation(
            citation_id=f"cite_{e.evidence_id}",
            evidence_id=e.evidence_id,
            chunk_id=e.chunk_id,
            document_id=e.document_id,
            source_title=e.title,
            source_type="pdf",
            page_number=23,
            section_path="Stability",
            content_hash=e.content_hash,
            snippet=e.content[:200],
            validation="provenance_valid",
        )
        for e in (citations_for or [])
    ]
    return Answer(
        answer_id="ans_1",
        kb_id="kb_test",
        question="What must the metacentric height GM be for stable equilibrium?",
        status=status,
        text=text,
        claims=claims if claims is not None else [],
        citations=citations,
        assessment=EvidenceAssessment(
            sufficient=sufficient,
            decision=decision,
            grounding_state=state,
            confidence=ConfidenceCategory.HIGH,
            reason_code="SUFFICIENT" if sufficient else "NO_EVIDENCE",
            reason="test",
        ),
    )


# ---------------------------------------------------------------------------
# 1. correct answer + correct citation -> pass
# ---------------------------------------------------------------------------


class TestCase1CorrectCitationPasses:
    def test_ideal_answer_passes(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive for stable equilibrium.", ["ev_0001"])],
            citations_for=[ev],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert r.problems == [], r.problems
        assert r.passed
        assert r.citation_precision.measured and r.citation_precision.value == 1.0
        assert r.citation_recall.measured and r.citation_recall.value == 1.0
        assert r.false_supported is False
        assert r.claim_verdicts[0].entailment is SupportJudgement.SUPPORTED


# ---------------------------------------------------------------------------
# 2. correct answer + WRONG citation -> citation failure
# ---------------------------------------------------------------------------


class TestCase2WrongCitation:
    def test_citing_irrelevant_chunk_is_a_precision_failure(self):
        good = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        bad = _evidence("chk_other", UNRELATED_TEXT, evidence_id="ev_0002")
        answer = _answer(
            claims=[_claim(
                "The metacentric height GM must be positive for stable equilibrium.",
                ["ev_0002"],
            )],
            citations_for=[bad],
        )
        r = EVAL.evaluate(_question(), answer, [good, bad])
        assert not r.passed
        assert r.citation_precision.value == 0.0, "irrelevant citation must score 0 precision"
        assert r.citation_recall.value == 0.0, "required evidence was never cited"
        assert any("does not address" in p or "not cite the required" in p for p in r.problems)
        assert r.false_supported is True

    def test_wrong_citation_is_flagged_not_supported(self):
        good = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        bad = _evidence("chk_other", UNRELATED_TEXT, evidence_id="ev_0002")
        answer = _answer(
            claims=[_claim(
                "The metacentric height GM must be positive for stable equilibrium.",
                ["ev_0002"],
            )],
            citations_for=[bad],
        )
        r = EVAL.evaluate(_question(), answer, [good, bad])
        assert r.claim_verdicts[0].irrelevant_chunk_ids == ["chk_other"]
        assert r.claim_verdicts[0].entailment is SupportJudgement.NOT_SUPPORTED


# ---------------------------------------------------------------------------
# 3. correct answer + MISSING citation -> citation completeness failure
# ---------------------------------------------------------------------------


class TestCase3MissingCitation:
    def test_uncited_factual_claim_fails(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", [])],
            citations=[],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert not r.passed
        assert r.claim_verdicts[0].uncited is True
        assert any("no citation" in p for p in r.problems)
        assert r.unsupported_claim_rate.value == 1.0

    def test_citing_only_one_of_two_required_chunks_is_incomplete(self):
        ev_a = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        q = _question(required_evidence=[
            RequiredEvidence(chunk_id="chk_good", content_hash="hash_chk_good"),
            RequiredEvidence(chunk_id="chk_second", content_hash="hash_chk_second"),
        ])
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[ev_a],
        )
        r = EVAL.evaluate(q, answer, [ev_a])
        assert r.citation_recall.value == 0.5
        assert r.citation_completeness.value == 0.5
        assert r.claim_verdicts[0].missing_required_chunk_ids == ["chk_second"]
        assert not r.passed


# ---------------------------------------------------------------------------
# 4. unsupported claim + 5. mixed valid/fabricated citations
# ---------------------------------------------------------------------------


class TestCase4UnsupportedClaim:
    def test_claim_contradicted_by_its_own_evidence_is_flagged(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim(
                "The metacentric height GM must be negative for stable equilibrium.",
                ["ev_0001"],
            )],
            citations_for=[ev],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert not r.passed
        assert any("not supported" in p for p in r.problems)
        assert r.contradiction_rate.value is not None and r.contradiction_rate.value > 0


class TestCase5MixedCitations:
    def test_one_valid_one_fabricated_citation_fails(self):
        good = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        # ev_9999 is NOT in the retrieved evidence — a fabricated reference.
        answer = _answer(
            claims=[_claim(
                "The metacentric height GM must be positive for stable equilibrium.",
                ["ev_0001", "ev_9999"],
            )],
            citations_for=[good],
        )
        r = EVAL.evaluate(_question(), answer, [good])
        assert not r.passed
        assert r.claim_verdicts[0].has_invalid_citation is True
        assert any("not in the retrieved evidence set" in p for p in r.problems)

    def test_citation_pointing_to_evidence_outside_the_run_is_rejected(self):
        good = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        outside = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_5000")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_5000"])],
            citations_for=[outside],
        )
        r = EVAL.evaluate(_question(), answer, [good])
        assert r.claim_verdicts[0].has_invalid_citation is True
        assert not r.passed


# ---------------------------------------------------------------------------
# 6. correct abstention -> pass
# ---------------------------------------------------------------------------


class TestCase6CorrectAbstention:
    def _unanswerable(self) -> AnswerBenchmarkQuestion:
        return AnswerBenchmarkQuestion(
            question_id="q-unans",
            question="What is the manufacturer-specific torque tolerance for part X?",
            answerability=Answerability.UNANSWERABLE,
            required_evidence=[],
            abstention_required=True,
            expected_grounding_state=ExpectedGroundingState.INSUFFICIENT_EVIDENCE,
        )

    def test_abstaining_on_an_unanswerable_question_passes(self):
        answer = _answer(
            text="I don't have enough evidence to answer this question.",
            claims=[],
            citations=[],
            status=AnswerStatus.ABSTAINED,
            state=GroundingState.NO_RELEVANT_EVIDENCE,
            decision=GateDecision.ABSTAIN,
            sufficient=False,
        )
        r = EVAL.evaluate(self._unanswerable(), answer, [])
        assert r.passed, r.problems
        assert r.abstention_correct is True
        assert r.false_unsupported is False
        assert not r.problems

    def test_abstention_is_never_penalised_when_expected(self):
        answer = _answer(
            text="I don't have enough evidence.", claims=[], citations=[],
            status=AnswerStatus.ABSTAINED, state=GroundingState.INSUFFICIENT_EVIDENCE,
            decision=GateDecision.ABSTAIN, sufficient=False,
        )
        r = EVAL.evaluate(self._unanswerable(), answer, [])
        # citation metrics must be UNKNOWN, not 0 — abstention is not a failure.
        assert not r.citation_precision.measured
        assert not r.citation_recall.measured
        assert r.citation_precision.value is None


# ---------------------------------------------------------------------------
# 7. incorrect confident answer -> fail
# ---------------------------------------------------------------------------


class TestCase7IncorrectConfidentAnswer:
    def test_answering_an_unanswerable_question_fails(self):
        q = AnswerBenchmarkQuestion(
            question_id="q-unans",
            question="What is the manufacturer-specific torque tolerance?",
            answerability=Answerability.UNANSWERABLE,
            abstention_required=True,
            expected_grounding_state=ExpectedGroundingState.INSUFFICIENT_EVIDENCE,
        )
        ev = _evidence("chk_x", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(claims=[_claim("It is 25 Nm.", ["ev_0001"])], citations_for=[ev])
        r = EVAL.evaluate(q, answer, [ev])
        assert not r.passed
        assert r.false_unsupported is True
        assert r.abstention_correct is False
        assert any("answered anyway" in p for p in r.problems)

    def test_answering_answerable_question_without_required_evidence_is_false_supported(self):
        ev = _evidence("chk_other", UNRELATED_TEXT, evidence_id="ev_0002")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0002"])],
            citations_for=[ev],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert r.false_supported is True
        assert not r.passed


# ---------------------------------------------------------------------------
# 8. contradictory evidence
# ---------------------------------------------------------------------------


class TestCase8Contradiction:
    def test_flipped_polarity_is_detected_despite_high_term_overlap(self):
        """A claim that reverses the evidence keeps ~all the terms, so pure
        coverage would call it SUPPORTED. It must be flagged instead."""
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim(
                "The metacentric height GM must be negative for stable equilibrium.",
                ["ev_0001"],
            )],
            citations_for=[ev],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert r.claim_verdicts[0].entailment is SupportJudgement.NOT_SUPPORTED
        assert not r.passed
        assert r.contradiction_rate.value > 0

    def test_polarity_conflict_requires_topical_overlap(self):
        """Opposite words in unrelated sentences must NOT be a contradiction."""
        from app.services.answer_eval.entailment import polarity_conflict

        conflict, _ = polarity_conflict(
            "The lower limit applies here.",
            "The upper surface of the opposite gear was cold.",
        )
        assert conflict is False

    def test_agreeing_polarity_is_not_a_conflict(self):
        from app.services.answer_eval.entailment import polarity_conflict

        conflict, _ = polarity_conflict(
            "The metacentric height GM must be positive for stable equilibrium.",
            "The metacentric height GM must be positive for stable equilibrium of a body.",
        )
        assert conflict is False

    def test_polarity_conflict_is_deterministic(self):
        from app.services.answer_eval.entailment import polarity_conflict

        claim = "The GM must be negative for stability."
        ev = "The GM must be positive for stability of a floating body."
        results = {polarity_conflict(claim, ev)[0] for _ in range(5)}
        assert results == {True}

    def test_claim_contradicting_two_sources_is_flagged(self):
        a = _evidence("chk_a", "The metacentric height GM must be positive for stability.", evidence_id="ev_0001")
        b = _evidence("chk_b", "The metacentric height GM must be negative for stability.", evidence_id="ev_0002")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive for stability.", ["ev_0001"])],
            citations_for=[a],
        )
        r = EVAL.evaluate(_question(), answer, [a, b])
        assert r.contradiction_rate.measured

    def test_gate_conflict_state_is_preserved_in_the_result(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[ev],
            state=GroundingState.CONFLICTING_EVIDENCE,
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert r.actual_grounding_state == "CONFLICTING_EVIDENCE"


# ---------------------------------------------------------------------------
# 9. partial answer -> partial score
# ---------------------------------------------------------------------------


class TestCase9Partial:
    def test_partial_answer_gets_partial_citation_recall(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[ev],
            status=AnswerStatus.PARTIAL,
            state=GroundingState.PARTIALLY_SUPPORTED,
            decision=GateDecision.PARTIAL_ANSWER,
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert r.actual_grounding_state == "PARTIALLY_SUPPORTED"
        assert r.citation_recall.value == 1.0

    def test_partially_wrong_answer_scores_between_zero_and_one(self):
        good = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        other = _evidence("chk_other", UNRELATED_TEXT, evidence_id="ev_0002")
        answer = _answer(
            claims=[
                _claim("The metacentric height GM must be positive.", ["ev_0001"], "cl_001"),
                _claim("Cavitation happens below vapour pressure.", ["ev_0002"], "cl_002"),
            ],
            citations_for=[good, other],
        )
        r = EVAL.evaluate(_question(), answer, [good, other])
        assert 0.0 < r.citation_precision.value < 1.0
        assert 0.0 < r.final_score < 1.0


# ---------------------------------------------------------------------------
# 10. strategy comparison uses identical questions
# ---------------------------------------------------------------------------


class TestCase10Comparability:
    def _run(self, run_id: str, qids: list[str], fingerprint: str = "fp1") -> AnswerEvaluationRun:
        return AnswerEvaluationRun(
            id=run_id, kb_id="kb_test",
            benchmark_fingerprint=fingerprint, question_ids=qids, question_count=len(qids),
        )

    def test_identical_question_sets_are_comparable(self):
        a = self._run("r1", ["q1", "q2"])
        b = self._run("r2", ["q1", "q2"])
        ok, reason = a.is_comparable_with(b)
        assert ok, reason

    def test_different_question_sets_are_refused(self):
        a = self._run("r1", ["q1", "q2"])
        b = self._run("r2", ["q1", "q3"])
        ok, reason = a.is_comparable_with(b)
        assert not ok
        assert "different question subsets" in reason

    def test_different_benchmark_content_is_refused(self):
        a = self._run("r1", ["q1"], "fp1")
        b = self._run("r2", ["q1"], "fp2")
        ok, reason = a.is_comparable_with(b)
        assert not ok
        assert "fingerprint" in reason


class TestCase11IntersectionComparison:
    """V8 STEP 11: intersection-based plan with an INCONCLUSIVE verdict."""

    @staticmethod
    def _run(run_id: str, qids: list[str], fingerprint: str = "fp1") -> AnswerEvaluationRun:
        return AnswerEvaluationRun(
            id=run_id, kb_id="kb_test",
            benchmark_fingerprint=fingerprint, question_ids=qids,
            question_count=len(qids),
        )

    def test_identical_sets_plan_is_comparable(self):
        plan = self._run("r1", ["q1", "q2"]).compare_with(
            self._run("r2", ["q1", "q2"]))
        assert plan.verdict == "COMPARABLE"
        assert plan.mode == "identical"
        assert plan.comparable is True
        assert plan.question_ids == ["q1", "q2"]

    def test_partial_overlap_is_inconclusive_over_shared_questions_only(self):
        a = self._run("r1", ["q1", "q2", "q3"])
        b = self._run("r2", ["q2", "q3", "q4"])
        plan = a.compare_with(b)
        assert plan.verdict == "INCONCLUSIVE"
        assert plan.mode == "intersection"
        assert plan.comparable is False
        assert plan.question_ids == ["q2", "q3"]
        assert "shared question(s) ONLY" in plan.reason
        assert "INCONCLUSIVE" in plan.reason

    def test_zero_overlap_is_not_comparable_at_all(self):
        a = self._run("r1", ["q1", "q2"])
        b = self._run("r2", ["q3", "q4"])
        plan = a.compare_with(b)
        assert plan.verdict == "NOT_COMPARABLE"
        assert plan.mode == "no_overlap"
        assert plan.question_ids == []
        assert "zero common questions" in plan.reason

    def test_changed_benchmark_content_is_never_intersection_compared(self):
        """Same ids, different content: intersecting would compare different
        questions that happen to share an id — refused outright."""
        a = self._run("r1", ["q1", "q2"], "fp1")
        b = self._run("r2", ["q2", "q3"], "fp2")
        plan = a.compare_with(b)
        assert plan.verdict == "NOT_COMPARABLE"
        assert "benchmark content differs" in plan.reason

    def test_the_strict_contract_is_unchanged(self):
        """is_comparable_with must still refuse different subsets outright —
        the intersection plan is an ADDITION, not a relaxation."""
        a = self._run("r1", ["q1", "q2"])
        b = self._run("r2", ["q2", "q3"])
        ok, reason = a.is_comparable_with(b)
        assert not ok
        assert "different question subsets" in reason

    def test_selection_respects_explicit_subset(self):
        from app.services.answer_eval.benchmark import AnswerBenchmark

        bench = AnswerBenchmark(
            benchmark="b", kb_id="kb_test",
            questions=[_question(question_id=f"q{i}") for i in range(5)],
        )
        chosen = select_questions(bench, AnswerEvaluationConfig(question_ids=["q1", "q3"]))
        assert [q.question_id for q in chosen] == ["q1", "q3"]

    def test_unknown_question_id_is_rejected(self):
        from app.services.answer_eval.benchmark import AnswerBenchmark

        bench = AnswerBenchmark(
            benchmark="b", kb_id="kb_test", questions=[_question(question_id="q1")]
        )
        with pytest.raises(ValueError, match="not present in benchmark"):
            select_questions(bench, AnswerEvaluationConfig(question_ids=["q_nope"]))


# ---------------------------------------------------------------------------
# 11. frozen benchmark cannot be modified
# ---------------------------------------------------------------------------


class TestCase11FrozenBenchmark:
    def test_frozen_benchmark_file_is_untouched(self):
        """The generator writes a NEW file; the frozen one must be unchanged."""
        import subprocess

        result = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--",
             "benchmarks/automobile-engineering-baseline-v1.json"],
            cwd=str(ROOT), capture_output=True,
        )
        assert result.returncode == 0, "the frozen retrieval benchmark was modified"

    def test_benchmark_generator_refuses_to_write_the_frozen_file(self):
        source = (BACKEND_DIR / "scripts" / "build_answer_benchmark_v1.py").read_text(
            encoding="utf-8"
        )
        # The frozen path is only ever READ.
        assert 'FROZEN.read_text' in source
        assert "FROZEN.write_text" not in source

    def test_answer_benchmark_declares_pending_human_review(self):
        b = load_answer_benchmark(ANSWER_BENCHMARK)
        assert b.human_review is HumanReviewStatus.PENDING

    def test_answer_benchmark_inherits_the_source_review_status(self):
        raw = json.loads(FROZEN.read_text(encoding="utf-8"))
        b = load_answer_benchmark(ANSWER_BENCHMARK)
        assert raw["authorship"]["human_review"].startswith("PENDING")
        assert b.human_review.value == "pending"


# ---------------------------------------------------------------------------
# 12/13/14. immutability, versions, provider identity
# ---------------------------------------------------------------------------


class TestCase12to14RunRecords:
    def _run(self, **kw) -> AnswerEvaluationRun:
        base = dict(id="aerun_1", kb_id="kb_test", benchmark_name="b", benchmark_version=1)
        base.update(kw)
        return AnswerEvaluationRun(**base)

    def test_run_record_is_not_mutated_by_comparison(self):
        a = self._run(question_ids=["q1"], benchmark_fingerprint="fp")
        before = a.model_dump(mode="json")
        b = self._run(id="aerun_2", question_ids=["q1", "q2"], benchmark_fingerprint="fp")
        a.is_comparable_with(b)
        assert a.model_dump(mode="json") == before

    def test_evaluator_version_is_recorded_on_every_result(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[ev],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert r.evaluator_version
        assert r.evaluator_name

    def test_entailment_provider_identity_is_recorded(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[ev],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert r.entailment_provider == "heuristic-lexical-coverage"
        assert r.entailment_is_model_based is False

    def test_run_persists_every_producer(self):
        run = self._run(
            generator="llm", model="mock/x", is_mock=True,
            prompt_version="v7.1", answerer_version="v7.2",
            evaluator_name="deterministic-evidence", evaluator_version="v8.1",
            entailment_provider="heuristic-lexical-coverage",
        )
        assert run.generator and run.model
        assert run.prompt_version and run.answerer_version
        assert run.evaluator_name and run.evaluator_version
        assert run.entailment_provider

    def test_run_records_the_question_subset_explicitly(self):
        run = self._run(question_ids=["q1", "q2"], question_count=2, subset_note="explicit subset (2 of 28)")
        assert run.question_ids == ["q1", "q2"]
        assert run.question_count == 2
        assert "explicit subset" in run.subset_note


# ---------------------------------------------------------------------------
# 15. no network in deterministic tests + honesty of unknown metrics
# ---------------------------------------------------------------------------


class TestCase15NoNetworkAndHonesty:
    def test_deterministic_evaluator_needs_no_provider(self):
        e = create_answer_evaluator("deterministic")
        assert isinstance(e, DeterministicAnswerEvaluator)

    def test_llm_entailment_refuses_to_build_without_a_provider(self):
        with pytest.raises(ValueError, match="requires a provider"):
            create_entailment_evaluator("llm")

    def test_unknown_evaluator_kind_is_rejected_not_defaulted(self):
        with pytest.raises(ValueError, match="Unknown answer evaluator"):
            create_answer_evaluator("magic")

    def test_correctness_is_unknown_not_zero(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[ev],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert r.correctness.measured is False
        assert r.correctness.value is None
        assert "human-authored reference answer" in r.correctness.reason

    def test_key_point_recall_is_unknown(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[ev],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        assert r.key_point_recall.measured is False

    def test_final_score_is_only_computable_when_inputs_are_measured(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[ev],
        )
        r = EVAL.evaluate(_question(), answer, [ev])
        # correctness/key_point_recall are unknown but are NOT score inputs,
        # so the citation-based score IS computable. That is deliberate and is
        # exactly why the aggregate must surface unknown_metrics.
        assert r.final_score_computable is True
        assert r.correctness.measured is False

    def test_score_cannot_accompany_an_unmeasured_input(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        r = EVAL.evaluate(_question(), _answer(claims=[], citations=[]), [ev])
        payload = r.model_dump()
        payload["final_score"] = 0.99
        payload["final_score_computable"] = False
        with pytest.raises(ValueError, match="final_score was supplied"):
            type(r).model_validate(payload)

    def test_unmeasured_values_are_excluded_from_averages_not_treated_as_zero(self):
        out = average_measured(
            [Measured.of(1.0), Measured.unknown("n/a")], reason_if_empty="none"
        )
        assert out.measured and out.value == 1.0
        assert "1 question(s) excluded" in out.reason

    def test_average_of_only_unknown_is_unknown(self):
        out = average_measured([Measured.unknown("a"), Measured.unknown("b")],
                               reason_if_empty="none measured")
        assert out.measured is False and out.value is None

    def test_aggregate_lists_unknown_metrics_explicitly(self):
        ev = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        r = EVAL.evaluate(_question(), _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"])],
            citations_for=[ev],
        ), [ev])
        agg = aggregate_results([r])
        assert "correctness" in agg.unknown_metrics
        assert any("UNKNOWN" in w for w in agg.warnings)
        assert "CORRECTNESS" in " ".join(agg.warnings)


# ---------------------------------------------------------------------------
# benchmark schema integrity
# ---------------------------------------------------------------------------


class TestBenchmarkSchema:
    def test_unanswerable_question_must_require_abstention(self):
        with pytest.raises(ValueError, match="abstention_required"):
            AnswerBenchmarkQuestion(
                question_id="q", question="q?", answerability=Answerability.UNANSWERABLE
            )

    def test_unanswerable_question_cannot_also_require_evidence(self):
        with pytest.raises(ValueError, match="cannot require evidence"):
            AnswerBenchmarkQuestion(
                question_id="q", question="q?", answerability=Answerability.UNANSWERABLE,
                abstention_required=True,
                required_evidence=[RequiredEvidence(chunk_id="c")],
            )

    def test_answerable_question_must_name_its_evidence(self):
        with pytest.raises(ValueError, match="must name the required evidence"):
            AnswerBenchmarkQuestion(
                question_id="q", question="q?", answerability=Answerability.ANSWERABLE
            )

    def test_corpus_mismatch_is_detected(self):
        from app.services.answer_eval.benchmark import validate_against_corpus
        from app.services.answer_eval.benchmark import AnswerBenchmark

        bench = AnswerBenchmark(
            benchmark="b", kb_id="kb_test",
            questions=[_question(required_evidence=[
                RequiredEvidence(chunk_id="chk_missing", content_hash="abc")
            ])],
        )
        problems = validate_against_corpus(bench, {})
        assert any("not in the corpus" in p for p in problems)

    def test_rechunked_corpus_is_detected_by_content_hash(self):
        from app.services.answer_eval.benchmark import (
            AnswerBenchmark, validate_against_corpus,
        )

        bench = AnswerBenchmark(
            benchmark="b", kb_id="kb_test",
            questions=[_question(required_evidence=[
                RequiredEvidence(chunk_id="chk_good", content_hash="hash_old")
            ])],
        )
        problems = validate_against_corpus(
            bench, {"chk_good": {"content_hash": "hash_new", "document_id": "d"}}
        )
        assert any("content hash differs" in p for p in problems)

    def test_missing_benchmark_file_is_rejected(self):
        with pytest.raises(BenchmarkValidationError, match="not found"):
            load_answer_benchmark(ROOT / "benchmarks" / "does-not-exist.json")

    def test_real_benchmark_validates_against_its_declared_evidence(self):
        b = load_answer_benchmark(ANSWER_BENCHMARK)
        assert b.question_count == 28
        assert b.answerable_count == 28
        for q in b.questions:
            assert q.required_evidence
            assert q.expected_answer is None, "no reference answer may be invented"
            assert q.key_points == []
            assert q.reviewer_metadata["answer_label_status"] == "REQUIRES_HUMAN_AUTHORSHIP"


# ---------------------------------------------------------------------------
# grounding state comparison
# ---------------------------------------------------------------------------


class TestGroundingStateComparison:
    def test_insufficient_and_no_relevant_are_interchangeable(self):
        assert grounding_states_compatible("INSUFFICIENT_EVIDENCE", "NO_RELEVANT_EVIDENCE")
        assert grounding_states_compatible("NO_RELEVANT_EVIDENCE", "INSUFFICIENT_EVIDENCE")

    def test_answered_and_partial_are_interchangeable(self):
        assert grounding_states_compatible("ANSWERED", "PARTIALLY_SUPPORTED")
        assert grounding_states_compatible("PARTIALLY_SUPPORTED", "ANSWERED")

    def test_answered_is_not_acceptable_when_abstention_expected(self):
        assert not grounding_states_compatible("INSUFFICIENT_EVIDENCE", "ANSWERED")

    def test_any_acceptable_permits_everything(self):
        for state in ("ANSWERED", "PARTIALLY_SUPPORTED", "NO_RELEVANT_EVIDENCE"):
            assert grounding_states_compatible("ANY_ACCEPTABLE", state)

# ---------------------------------------------------------------------------
# necessity-vs-exhaustiveness semantics
#
# `required_evidence` records which chunks are NECESSARY, not which chunks are the
# only ones a correct answer may cite. Enforcing that per claim made every claim
# of every multi-claim answer a failure and drove the real-corpus pass_rate to
# 0.000. These tests lock the corrected level of judgement: strict where the
# benchmark actually has ground truth, permissive where it does not.
# ---------------------------------------------------------------------------


class TestRequiredEvidenceIsNecessityNotExhaustiveness:
    def _multi_claim(self):
        """A realistic answer: the canonical chunk, plus extra on-topic context."""
        good = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        extra = _evidence(
            "chk_extra",
            "Floating bodies are also classified by metacentric height during "
            "heel angle analysis of the righting arm GZ curve.",
            evidence_id="ev_0002",
        )
        answer = _answer(
            claims=[
                _claim("The metacentric height GM must be positive for stable "
                       "equilibrium.", ["ev_0001"], "cl_001"),
                _claim("GM is derived from the righting arm GZ divided by the "
                       "heel angle.", ["ev_0002"], "cl_002"),
            ],
            citations_for=[good, extra],
        )
        return answer, [good, extra]

    def test_extra_on_topic_context_does_not_fail_the_answer(self):
        """The regression: this exact shape drove real-corpus pass_rate to 0."""
        answer, evidence = self._multi_claim()
        r = EVAL.evaluate(_question(), answer, evidence)
        assert r.citation_recall.value == 1.0
        assert r.false_supported is False
        assert r.passed, f"well-sourced answer wrongly failed: {r.problems}"

    def test_per_claim_signals_are_still_recorded_for_failure_analysis(self):
        """Relaxing the FAILURE must not delete the diagnostic."""
        answer, evidence = self._multi_claim()
        r = EVAL.evaluate(_question(), answer, evidence)
        assert r.claim_verdicts[0].missing_required_chunk_ids == []
        assert r.claim_verdicts[1].missing_required_chunk_ids == ["chk_good"]
        assert r.claim_verdicts[1].irrelevant_chunk_ids == ["chk_extra"]

    def test_extra_citations_are_a_warning_not_a_problem(self):
        answer, evidence = self._multi_claim()
        r = EVAL.evaluate(_question(), answer, evidence)
        assert any("not labelled as required" in w for w in r.warnings)
        assert not any("not labelled" in p for p in r.problems)

    def test_citation_precision_still_counts_extra_citations(self):
        """Precision is a lower bound, not a fabrication: it still moves."""
        answer, evidence = self._multi_claim()
        r = EVAL.evaluate(_question(), answer, evidence)
        assert r.citation_precision.value == 0.5

    def test_answer_must_still_cite_the_required_chunk(self):
        """The necessary-evidence check is NOT weakened by the above."""
        good = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        extra = _evidence("chk_extra", UNRELATED_TEXT, evidence_id="ev_0002")
        answer = _answer(
            claims=[_claim("Cavitation happens below vapour pressure.", ["ev_0002"])],
            citations_for=[extra],
        )
        r = EVAL.evaluate(_question(), answer, [good, extra])
        assert r.citation_recall.value == 0.0
        assert not r.passed
        assert any("answer does not cite the required evidence" in p for p in r.problems)
        assert r.false_supported is True

    def test_the_evaluator_cannot_assert_a_chunk_is_irrelevant(self):
        """The benchmark has no ground truth on relevance; the wording must not claim it."""
        answer, evidence = self._multi_claim()
        r = EVAL.evaluate(_question(), answer, evidence)
        blob = " ".join(r.problems) + " ".join(r.warnings)
        assert "does not address this question" not in blob

    def test_observation_can_be_disabled_by_the_benchmark(self):
        answer, evidence = self._multi_claim()
        q = _question(citation_requirements=CitationRequirements(
            must_not_cite_irrelevant_chunks=False,
        ))
        r = EVAL.evaluate(q, answer, evidence)
        assert not any("not labelled as required" in w for w in r.warnings)
        assert r.citation_precision.value == 0.5, "the metric itself is unaffected"

    def test_each_claim_still_needs_a_citation_when_required(self):
        good = _evidence("chk_good", GOOD_EVIDENCE_TEXT, evidence_id="ev_0001")
        answer = _answer(
            claims=[_claim("The metacentric height GM must be positive.", ["ev_0001"],
                           "cl_001"),
                    _claim("Vessel plating thickness follows Lloyd rules.", [],
                           "cl_002")],
            citations_for=[good],
        )
        r = EVAL.evaluate(_question(), answer, [good])
        assert r.claim_verdicts[1].uncited is True
        assert any("no citation" in p for p in r.problems)
        assert not r.passed
