"""V12 retrieval-diagnostics: availability, taxonomy, comparison, recommendations.

Pure unit tests. No network, no Qdrant, no LLM.

The tests that matter most here are the ones that pin the HONESTY rules:

* a question with no reference evidence is UNKNOWN, not a failure;
* evidence missing from the corpus and evidence present-but-unretrieved are
  DIFFERENT codes (this is the whole point of V12);
* an unprobeable index never becomes \"present\" and never becomes \"missing\";
* LOW_PRECISION is never emitted, with the reason recorded;
* a recommendation that fires without a measured selection step is refused;
* a comparison only pairs questions both sides actually measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.comparison import PairingVerdict  # noqa: E402
from app.services.answer_eval.diagnostics import build_question_diagnostic  # noqa: E402
from app.services.answer_eval.recommendations import (  # noqa: E402
    RecommendationAction,
    analyse_retrieval,
)
from app.services.answer_eval.retrieval_diagnostics import (  # noqa: E402
    ALL_RETRIEVAL_CODES,
    AvailabilityRollup,
    EvidenceAvailability,
    EvidenceAvailabilityStatus,
    RetrievalFailureCode,
    RetrievalIntegritySnapshot,
    StrategyDiagnostics,
    assess_evidence_availability,
    build_probe,
    classify_retrieval_question,
    compare_strategies,
    critical_questions_by_code,
    recommendations_from_strategy,
    rollup_availability,
    summarise_strategy,
)

REQUIRED = "chk_req"
OTHER = "chk_other"


class FakeChunkRow:
    """Minimal stand-in for a `Chunk` row (only the fields availability reads)."""

    def __init__(self, chunk_id: str, document_id: str, **kw) -> None:
        self.id = chunk_id
        self.document_id = document_id
        self.section = kw.get("section", "Section")
        self.source_title = kw.get("source_title", "Title")
        self.content_hash = kw.get("content_hash", "hash")


def _row(chunk_id: str = REQUIRED, document_id: str = "doc_req") -> FakeChunkRow:
    return FakeChunkRow(chunk_id, document_id)


def _availability(
    *,
    chunk_id: str = REQUIRED,
    in_corpus: bool = True,
    in_index: bool | None = True,
    expected_docs: list[str] | None = None,
    actual_doc: str | None = "doc_req",
) -> EvidenceAvailability:
    if in_corpus and in_index is True:
        status = EvidenceAvailabilityStatus.INDEXED
    elif in_corpus and in_index is None:
        status = EvidenceAvailabilityStatus.CORPUS_ONLY
    elif in_corpus and in_index is False:
        status = EvidenceAvailabilityStatus.MISSING_FROM_INDEX
    elif not in_corpus and in_index is None:
        status = EvidenceAvailabilityStatus.MISSING_FROM_CORPUS
    elif not in_corpus:
        status = EvidenceAvailabilityStatus.MISSING_FROM_CORPUS
    else:  # pragma: no cover - defensive
        status = EvidenceAvailabilityStatus.UNKNOWN
    return EvidenceAvailability(
        chunk_id=chunk_id,
        expected_document_ids=expected_docs or [],
        in_corpus=in_corpus,
        in_index=in_index,
        status=status,
        actual_document_id=actual_doc if in_corpus else None,
    )


def _diag(
    *,
    ranked: list[str],
    required: list[str] | None = None,
    top_k: int = 10,
    question_id: str = "q1",
):
    return build_question_diagnostic(
        question_id=question_id,
        question="Q?",
        strategy="dense",
        top_k=top_k,
        ranked_chunk_ids=ranked,
        chunk_index={cid: {"document_id": "doc_req"} for cid in ranked},
        required_chunk_ids=list(required or []),
        required_document_ids=["doc_req"] if required else [],
    )


def _classify(*, ranked: list[str], required: list[str] | None = None, top_k: int = 10, availability=None, rollup=None):
    items = availability if availability is not None else [_availability()]
    return classify_retrieval_question(
        question_id="q1",
        diagnostic=_diag(ranked=ranked, required=required, top_k=top_k),
        availability=items,
        rollup=rollup or rollup_availability(items),
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestEvidenceAvailability:
    def test_indexed_chunk_is_reported_available(self):
        availability, rollup, warnings = assess_evidence_availability(
            {REQUIRED: ["doc_req"]},
            chunk_rows={REQUIRED: _row()},
            index_chunk_ids={REQUIRED},
            document_ids={"doc_req"},
        )
        assert availability[0].status is EvidenceAvailabilityStatus.INDEXED
        assert availability[0].retrievable is True
        assert rollup is AvailabilityRollup.ALL_INDEXED
        assert warnings == []

    def test_corpus_row_without_a_vector_is_missing_from_index(self):
        availability, rollup, _ = assess_evidence_availability(
            {REQUIRED: ["doc_req"]},
            chunk_rows={REQUIRED: _row()},
            index_chunk_ids=set(),
            document_ids={"doc_req"},
        )
        assert availability[0].status is EvidenceAvailabilityStatus.MISSING_FROM_INDEX
        assert availability[0].retrievable is False
        assert rollup is AvailabilityRollup.MISSING_FROM_INDEX
        assert "no retriever" in availability[0].notes[0]

    def test_absent_chunk_row_is_missing_from_corpus(self):
        availability, rollup, _ = assess_evidence_availability(
            {REQUIRED: ["doc_req"]},
            chunk_rows={},
            index_chunk_ids=set(),
            document_ids={"doc_req"},
        )
        assert availability[0].status is EvidenceAvailabilityStatus.MISSING_FROM_CORPUS
        assert rollup is AvailabilityRollup.MISSING_FROM_CORPUS

    def test_unprobeable_index_is_corpus_only_never_present_never_missing(self):
        availability, rollup, _ = assess_evidence_availability(
            {REQUIRED: ["doc_req"]},
            chunk_rows={REQUIRED: _row()},
            index_chunk_ids=None,
            document_ids={"doc_req"},
        )
        assert availability[0].status is EvidenceAvailabilityStatus.CORPUS_ONLY
        assert availability[0].in_index is None
        assert availability[0].retrievable is None, (
            "an unprobeable index must not claim the chunk is retrievable"
        )
        assert rollup is AvailabilityRollup.UNVERIFIED_INDEX

    def test_document_mismatch_is_recorded_not_silently_accepted(self):
        availability, _, _ = assess_evidence_availability(
            {REQUIRED: ["doc_benchmark_says"]},
            chunk_rows={REQUIRED: _row(document_id="doc_elsewhere")},
            index_chunk_ids={REQUIRED},
            document_ids={"doc_elsewhere"},
        )
        assert availability[0].document_matches_benchmark is False
        assert any(
            "do not describe where the answering text lives" in n
            for n in availability[0].notes
        )

    def test_shared_chunk_across_questions_is_warned_about(self):
        _, _, warnings = assess_evidence_availability(
            {REQUIRED: ["doc_req"]},
            chunk_rows={REQUIRED: _row()},
            index_chunk_ids={REQUIRED},
            document_ids={"doc_req"},
            multi_question_chunk_ids={REQUIRED},
        )
        assert any("shared with other questions" in w for w in warnings)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class TestClassification:
    def test_no_reference_evidence_is_unknown_not_a_failure(self):
        result = _classify(ranked=[OTHER], required=[], availability=[], rollup=AvailabilityRollup.UNKNOWN)
        assert result.code is RetrievalFailureCode.UNKNOWN
        assert result.is_critical is False
        assert "no required evidence" in result.evidence[0]

    def test_corpus_gap_outranks_every_retrieval_verdict(self):
        items = [_availability(in_corpus=False, in_index=None)]
        result = _classify(ranked=[OTHER], required=[REQUIRED], availability=items)
        assert result.code is RetrievalFailureCode.CORPUS_OR_SOURCE_DEFICIENCY
        assert result.rule == "EVIDENCE_ABSENT_FROM_CORPUS"
        assert any("no chunk row" in e for e in result.evidence)

    def test_corpus_but_unindexed_is_expected_evidence_unavailable(self):
        items = [_availability(in_index=False)]
        result = _classify(ranked=[OTHER], required=[REQUIRED], availability=items)
        assert result.code is RetrievalFailureCode.EXPECTED_EVIDENCE_UNAVAILABLE
        assert result.rule == "EVIDENCE_IN_CORPUS_BUT_NOT_INDEXED"

    def test_available_and_indexed_but_unretrieved_is_not_a_corpus_gap(self):
        """The distinction V12 exists for: same observations, opposite remedies."""
        result = _classify(ranked=[OTHER], required=[REQUIRED])
        assert result.code is RetrievalFailureCode.NO_RELEVANT_EVIDENCE
        assert "not a corpus gap" in result.reason
        assert any("appear anywhere in the" in e for e in result.evidence)
        assert any("required_recall_in_probe=0.0" in e for e in result.evidence)
        assert any("whole probed candidate set" in e for e in result.evidence)

    def test_empty_candidate_set_is_retrieval_failure(self):
        result = _classify(ranked=[], required=[REQUIRED])
        assert result.code is RetrievalFailureCode.RETRIEVAL_FAILURE
        assert result.rule == "RETRIEVAL_RETURNED_NO_CANDIDATES"

    def test_present_at_depth_but_outside_k_is_low_recall(self):
        ranked = [OTHER] * 4 + [REQUIRED] + [OTHER] * 5
        result = _classify(ranked=ranked, required=[REQUIRED], top_k=3)
        assert result.code is RetrievalFailureCode.LOW_RECALL
        assert result.rule == "REQUIRED_EVIDENCE_BEYOND_EVALUATED_CUTOFF"
        assert "rank of the missed required chunk(s)" in result.evidence[1].lower()
        assert "depth problem, not a coverage problem" in result.reason

    def test_absent_from_the_whole_probe_depth_is_no_relevant_evidence(self):
        """A deep candidate probe makes "absent" a strong claim, not a truncated one."""
        ranked = [OTHER] * 20
        result = _classify(ranked=ranked, required=[REQUIRED], top_k=10)
        assert result.code is RetrievalFailureCode.NO_RELEVANT_EVIDENCE
        assert any("whole probed candidate set, not merely from the window" in e for e in result.evidence)

    def test_inside_k_but_outranked_is_ranking_failure(self):
        result = _classify(ranked=[OTHER, REQUIRED], required=[REQUIRED], top_k=10)
        assert result.code is RetrievalFailureCode.RANKING_FAILURE
        assert any("non-required chunk(s) ranked above it" in e for e in result.evidence)

    def test_rank_one_is_retrieval_success(self):
        result = _classify(ranked=[REQUIRED, OTHER], required=[REQUIRED])
        assert result.code is RetrievalFailureCode.RETRIEVAL_SUCCESS
        assert result.is_critical is False

    def test_low_precision_is_never_emitted_and_says_why(self):
        """Necessity-only binary labels make precision a lower bound."""
        emitted = set()
        for ranked in ([REQUIRED], [REQUIRED] + [OTHER] * 9, [OTHER] * 9 + [REQUIRED]):
            for top_k in (1, 3, 10):
                emitted.add(_classify(ranked=ranked, required=[REQUIRED], top_k=top_k).code)
        assert RetrievalFailureCode.LOW_PRECISION not in emitted

    def test_unprobeable_index_limits_confidence_but_still_classifies(self):
        items = [_availability(in_index=None)]
        result = _classify(ranked=[OTHER], required=[REQUIRED], availability=items)
        assert result.code is RetrievalFailureCode.NO_RELEVANT_EVIDENCE
        assert result.confidence.value == "partial"
        assert any("not probed" in lim for lim in result.limitations)

    def test_all_published_codes_are_in_the_closed_set(self):
        assert len(ALL_RETRIEVAL_CODES) == len({c.value for c in ALL_RETRIEVAL_CODES})
        assert RetrievalFailureCode.LOW_PRECISION in ALL_RETRIEVAL_CODES


class TestDeepProbe:
    """The probe is what separates "below the cut-off" from "never retrieved"."""

    def test_probe_records_ranks_beyond_the_cutoff_and_absence(self):
        from app.services.answer_eval.retrieval_diagnostics import build_probe

        probe = build_probe(
            [OTHER] * 13 + [REQUIRED],
            required_chunk_ids=[REQUIRED, "chk_never"],
            top_k=3,
            candidate_depth=20,
        )
        assert probe.required_ranks_in_probe == {REQUIRED: 14}
        assert probe.required_beyond_cutoff == {REQUIRED: 14}
        assert probe.absent_from_probe == ["chk_never"]
        assert probe.required_recall_in_probe == 0.5
        assert probe.truncated is True, "14 candidates < candidate_depth 20"

    def test_required_chunk_ranked_beyond_k_is_low_recall_with_its_rank(self):
        from app.services.answer_eval.retrieval_diagnostics import build_probe

        ranked = [OTHER] * 13 + [REQUIRED]
        probe = build_probe(ranked, required_chunk_ids=[REQUIRED], top_k=3, candidate_depth=20)
        # Without a probe, a window-only list cannot tell depth from absence: the
        # same observations produce the wrong verdict. WITH the probe it is precise.
        windowed = _classify(ranked=ranked[:3], required=[REQUIRED], top_k=3)
        assert windowed.code is RetrievalFailureCode.NO_RELEVANT_EVIDENCE
        assert any("not widened beyond" in lim for lim in windowed.limitations)
        with_probe = classify_retrieval_question(
            question_id="q1",
            diagnostic=_diag(ranked=ranked[:3], required=[REQUIRED], top_k=3),
            availability=[_availability()],
            rollup=AvailabilityRollup.ALL_INDEXED,
            top_k=3,
            probe=probe,
        )
        assert with_probe.code is RetrievalFailureCode.LOW_RECALL
        assert "'chk_req': 14" in with_probe.evidence[1]
        assert all("not widened beyond" not in lim for lim in with_probe.limitations)

    def test_a_deep_probe_makes_absence_a_strong_claim(self):
        from app.services.answer_eval.retrieval_diagnostics import build_probe

        ranked = [OTHER] * 20
        probe = build_probe(ranked, required_chunk_ids=[REQUIRED], top_k=3, candidate_depth=20)
        result = classify_retrieval_question(
            question_id="q1",
            diagnostic=_diag(ranked=ranked[:3], required=[REQUIRED], top_k=3),
            availability=[_availability()],
            rollup=AvailabilityRollup.ALL_INDEXED,
            top_k=3,
            probe=probe,
        )
        assert result.code is RetrievalFailureCode.NO_RELEVANT_EVIDENCE
        assert any("probed to depth 20" in e for e in result.evidence)

    def test_a_window_only_run_says_so_in_its_limitations(self):
        result = _classify(ranked=[OTHER], required=[REQUIRED], top_k=10)
        assert any("not widened beyond" in lim for lim in result.limitations)


# ---------------------------------------------------------------------------
# Summary + comparison
# ---------------------------------------------------------------------------


def _question_diagnostic(*, qid: str, ranked: list[str], required: list[str], top_k: int = 10, strategy: str = "dense"):
    from app.services.answer_eval.retrieval_diagnostics import (
        QuestionRetrievalDiagnostic,
    )

    availability = [_availability()]
    diag = build_question_diagnostic(
        question_id=qid,
        question="Q?",
        strategy=strategy,
        top_k=top_k,
        ranked_chunk_ids=ranked,
        chunk_index={cid: {"document_id": "doc_req"} for cid in ranked},
        required_chunk_ids=required,
        required_document_ids=["doc_req"],
    )
    return QuestionRetrievalDiagnostic(
        question_id=qid,
        question="Q?",
        strategy=strategy,
        top_k=top_k,
        expected_chunk_ids=required,
        expected_document_ids=["doc_req"],
        availability=availability,
        availability_rollup=AvailabilityRollup.ALL_INDEXED,
        evidence=diag,
        classification=classify_retrieval_question(
            question_id=qid,
            diagnostic=diag,
            availability=availability,
            rollup=AvailabilityRollup.ALL_INDEXED,
            top_k=top_k,
        ),
    )


class TestSummary:
    def test_code_and_availability_counts_are_real_counts(self):
        questions = [
            _question_diagnostic(qid="q1", ranked=[REQUIRED], required=[REQUIRED]),
            _question_diagnostic(qid="q2", ranked=[OTHER, REQUIRED], required=[REQUIRED]),
            _question_diagnostic(qid="q3", ranked=[OTHER], required=[REQUIRED]),
        ]
        summary = summarise_strategy(questions)
        assert summary.question_count == 3
        assert summary.code_counts == {
            "RANKING_FAILURE": 1,
            "RETRIEVAL_SUCCESS": 1,
            "NO_RELEVANT_EVIDENCE": 1,
        }
        assert summary.availability_counts == {"all_indexed": 3}
        assert summary.critical_question_ids == ["q2", "q3"]
        assert summary.recall_at_k.value == pytest.approx(2 / 3)
        assert any("LOW_PRECISION" in w for w in summary.warnings)

    def test_no_questions_produces_unknown_not_zero(self):
        summary = summarise_strategy([])
        assert summary.recall_at_k.measured is False
        assert summary.recall_at_k.value is None
        assert summary.warnings


def _strategy(name: str, questions: list) -> StrategyDiagnostics:
    return StrategyDiagnostics(
        strategy=name,
        top_k=10,
        corpus_fingerprint="corpus1",
        questions=questions,
        summary=summarise_strategy(questions),
    )


class TestComparison:
    def test_only_questions_both_strategies_measured_are_paired(self):
        left = _strategy(
            "dense",
            [
                _question_diagnostic(qid="q1", ranked=[REQUIRED], required=[REQUIRED], strategy="dense"),
                _question_diagnostic(qid="q2", ranked=[OTHER], required=[REQUIRED], strategy="dense"),
                _question_diagnostic(qid="q3", ranked=[REQUIRED], required=[REQUIRED], strategy="dense"),
            ],
        )
        right = _strategy(
            "bm25",
            [
                _question_diagnostic(qid="q1", ranked=[OTHER, REQUIRED], required=[REQUIRED], strategy="bm25"),
                _question_diagnostic(qid="q3", ranked=[OTHER], required=[REQUIRED], strategy="bm25"),
            ],
        )
        comparison = compare_strategies(left, right, benchmark_name="bench", benchmark_fingerprint="fp")
        assert comparison.pairing is PairingVerdict.INCONCLUSIVE, (
            "one side is missing a question, so the pairing cannot be the full set"
        )
        mrr = comparison.comparison_for("mrr")
        assert mrr is not None
        assert {d.question_id for d in mrr.deltas} == {"q1", "q3"}
        assert all(d.question_id != "q2" for d in mrr.deltas)
        assert any("excluded and listed as unpaired" in n for n in comparison.notes)

    def test_identical_suites_are_comparable_with_no_movement(self):
        qs = [
            _question_diagnostic(qid=f"q{i}", ranked=[REQUIRED], required=[REQUIRED])
            for i in range(10)
        ]
        left = _strategy("dense", qs)
        right = _strategy("dense_copy", [
            _question_diagnostic(qid=f"q{i}", ranked=[REQUIRED], required=[REQUIRED], strategy="dense_copy")
            for i in range(10)
        ])
        comparison = compare_strategies(left, right, benchmark_name="bench", benchmark_fingerprint="fp")
        assert comparison.pairing is PairingVerdict.COMPARABLE
        assert comparison.improved_metrics() == []
        assert comparison.regressed_metrics() == []

    def test_a_failed_side_is_never_a_strategy_result(self):
        left = _strategy("dense", [_question_diagnostic(qid="q1", ranked=[REQUIRED], required=[REQUIRED])])
        right = StrategyDiagnostics(strategy="hybrid_reranked", status="failed", failure_reason="reranker missing")
        comparison = compare_strategies(left, right, benchmark_name="bench", benchmark_fingerprint="fp")
        assert any("NOT a strategy comparison" in n for n in comparison.notes)
        assert comparison.pairing is PairingVerdict.NOT_COMPARABLE


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_ranking_failures_recommend_a_reranking_experiment(self):
        questions = [
            _question_diagnostic(qid=f"q{i}", ranked=[OTHER, REQUIRED], required=[REQUIRED])
            for i in range(5)
        ]
        strategy = _strategy("dense", questions)
        report = recommendations_from_strategy(
            strategy,
            base_report=analyse_retrieval([q.evidence for q in questions], current_top_k=10),
            benchmark_name="bench",
        )
        rerank = report.for_action(RecommendationAction.ENABLE_RERANKING)
        assert len(rerank) == 1
        assert rerank[0].experiment_required is True
        assert rerank[0].proposed_config_change == {}, "a recommendation must not apply a change"
        assert "validation_experiment" in rerank[0].thresholds_used
        assert rerank[0].supporting_measurements["affected_question_ids"] == [
            "q0", "q1", "q2", "q3", "q4",
        ]
        assert rerank[0].strength.value in ("moderate", "strong")

    def test_no_selection_step_means_no_selection_recommendation(self):
        """The V12 live bug: an unrun selection step must not advise about selection."""
        questions = [
            _question_diagnostic(qid=f"q{i}", ranked=[REQUIRED], required=[REQUIRED])
            for i in range(5)
        ]
        strategy = _strategy("dense", questions)
        report = recommendations_from_strategy(
            strategy,
            base_report=analyse_retrieval([q.evidence for q in questions], current_top_k=10),
        )
        assert report.for_action(RecommendationAction.INVESTIGATE_EVIDENCE_SELECTION) == []
        assert report.for_action(RecommendationAction.NO_ACTION_JUSTIFIED), (
            "a clean run must say no action is justified rather than recommend something"
        )

    def test_depth_failures_refute_the_corpus_deficiency_advice(self):
        """The probe proves depth, so "missing content" advice must be replaced.

        This is the live k=3 case: V9 saw required evidence absent from its window
        and blamed the corpus; V12's probe found every one of them at rank 4-6.
        """
        questions = []
        for i in range(5):
            ranked = [OTHER] * 3 + [REQUIRED] + [OTHER] * 6
            q = _question_diagnostic(qid=f"q{i}", ranked=ranked, required=[REQUIRED], top_k=3)
            # The metrics of record are windowed at k=3; the probe saw the deeper list.
            q.evidence = _diag(ranked=ranked[:3], required=[REQUIRED], top_k=3)
            from app.services.answer_eval.retrieval_diagnostics import build_probe

            q.probe = build_probe(
                ranked, required_chunk_ids=[REQUIRED], top_k=3, candidate_depth=20
            )
            q.classification = classify_retrieval_question(
                question_id=f"q{i}",
                diagnostic=q.evidence,
                availability=q.availability,
                rollup=q.availability_rollup,
                top_k=3,
                probe=q.probe,
            )
            questions.append(q)
        assert all(q.classification.code is RetrievalFailureCode.LOW_RECALL for q in questions)
        strategy = _strategy("dense", questions)
        base = analyse_retrieval(
            [q.evidence for q in questions], current_top_k=3, current_strategy="dense"
        )
        # V9, seeing only the window, blames the corpus and the strategy.
        assert base.for_action(RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY)
        assert base.for_action(RecommendationAction.ENABLE_HYBRID_FUSION)
        report = recommendations_from_strategy(strategy, base_report=base)
        assert report.for_action(RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY) == []
        assert any("REPLACED" in n for n in report.notes)
        assert any("Probe ranks" in n for n in report.notes)
        increase = report.for_action(RecommendationAction.INCREASE_TOP_K)
        assert len(increase) == 1, "one action, one recommendation, carrying this run's evidence"
        assert increase[0].supporting_measurements["observed_required_ranks_beyond_cutoff"] == [4]
        # V9 also blames the strategy ("missed their required evidence entirely");
        # the probe disproves that sentence, so it is dropped too.
        assert report.for_action(RecommendationAction.ENABLE_HYBRID_FUSION) == []
        assert any("hybrid-fusion recommendation was REPLACED" in n for n in report.notes)

    def test_a_true_recall_miss_keeps_the_single_strategy_advice(self):
        """When the probe finds nothing at any depth, fusion IS a fair experiment."""
        from app.services.answer_eval.retrieval_diagnostics import QuestionRetrievalDiagnostic

        questions = []
        for i in range(4):
            ranked = [OTHER] * 20
            evidence = _diag(ranked=ranked[:10], required=[REQUIRED], top_k=10)
            probe = build_probe(
                ranked, required_chunk_ids=[REQUIRED], top_k=10, candidate_depth=20
            )
            availability = [_availability()]
            questions.append(
                QuestionRetrievalDiagnostic(
                    question_id=f"q{i}",
                    strategy="dense",
                    top_k=10,
                    expected_chunk_ids=[REQUIRED],
                    expected_document_ids=["doc_req"],
                    availability=availability,
                    availability_rollup=AvailabilityRollup.ALL_INDEXED,
                    evidence=evidence,
                    probe=probe,
                    classification=classify_retrieval_question(
                        question_id=f"q{i}",
                        diagnostic=evidence,
                        availability=availability,
                        rollup=AvailabilityRollup.ALL_INDEXED,
                        top_k=10,
                        probe=probe,
                    ),
                )
            )
        strategy = _strategy("dense", questions)
        base = analyse_retrieval(
            [q.evidence for q in questions], current_top_k=10, current_strategy="dense"
        )
        report = recommendations_from_strategy(strategy, base_report=base)
        assert all(q.classification.code is RetrievalFailureCode.NO_RELEVANT_EVIDENCE for q in questions)
        assert report.for_action(RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY) == []
        assert report.for_action(RecommendationAction.INVESTIGATE_RETRIEVAL)
        assert report.for_action(RecommendationAction.ENABLE_HYBRID_FUSION), (
            "a genuine single-strategy recall miss should still suggest trying fusion"
        )

    def test_available_but_unretrieved_refutes_the_corpus_deficiency_advice(self):
        questions = [
            _question_diagnostic(qid=f"q{i}", ranked=[OTHER], required=[REQUIRED])
            for i in range(4)
        ]
        strategy = _strategy("dense", questions)
        base = analyse_retrieval([q.evidence for q in questions], current_top_k=10)
        # V9's coarse rule sees "absent from candidates" and blames the corpus.
        assert base.for_action(RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY)
        report = recommendations_from_strategy(strategy, base_report=base)
        assert report.for_action(RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY) == []
        refined = report.for_action(RecommendationAction.INVESTIGATE_RETRIEVAL)
        assert len(refined) == 1
        assert any("REPLACED" in n for n in report.notes)
        assert refined[0].supporting_measurements["missing_from_candidates"]

    def test_corpus_gap_with_missing_evidence_keeps_the_corpus_advice(self):
        from app.services.answer_eval.retrieval_diagnostics import QuestionRetrievalDiagnostic

        availability = [_availability(in_corpus=False, in_index=None)]
        diag = build_question_diagnostic(
            question_id="q1",
            question="Q?",
            strategy="dense",
            top_k=10,
            ranked_chunk_ids=[OTHER],
            chunk_index={},
            required_chunk_ids=[REQUIRED],
        )
        question = QuestionRetrievalDiagnostic(
            question_id="q1",
            strategy="dense",
            top_k=10,
            expected_chunk_ids=[REQUIRED],
            availability=availability,
            availability_rollup=AvailabilityRollup.MISSING_FROM_CORPUS,
            evidence=diag,
            classification=classify_retrieval_question(
                question_id="q1",
                diagnostic=diag,
                availability=availability,
                rollup=AvailabilityRollup.MISSING_FROM_CORPUS,
                top_k=10,
            ),
        )
        strategy = _strategy("dense", [question])
        report = recommendations_from_strategy(strategy)
        assert report.for_action(RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY)
        assert report.for_action(RecommendationAction.RESTORE_INDEX_INTEGRITY) == []

    def test_critical_questions_are_grouped_by_code(self):
        questions = [
            _question_diagnostic(qid="q1", ranked=[REQUIRED], required=[REQUIRED]),
            _question_diagnostic(qid="q2", ranked=[OTHER, REQUIRED], required=[REQUIRED]),
            _question_diagnostic(qid="q3", ranked=[OTHER], required=[REQUIRED]),
        ]
        grouped = critical_questions_by_code(_strategy("dense", questions))
        assert grouped == {"NO_RELEVANT_EVIDENCE": ["q3"], "RANKING_FAILURE": ["q2"]}


# ---------------------------------------------------------------------------
# Integrity snapshot
# ---------------------------------------------------------------------------


class TestIntegritySnapshot:
    def test_unchanged_requires_readable_counts_on_both_sides(self):
        snap = RetrievalIntegritySnapshot(
            kb_id="kb1",
            index_points_before=812,
            index_points_after=812,
            chunk_rows_before=812,
            chunk_rows_after=812,
            corpus_revision_before=0,
            corpus_revision_after=0,
            benchmark_sha256_before="abc",
            benchmark_sha256_after="abc",
        )
        assert snap.protected_kb_unchanged is True
        assert snap.benchmark_unchanged is True

    def test_unreadable_counts_are_unknown_not_unchanged(self):
        snap = RetrievalIntegritySnapshot(index_points_before=812)
        assert snap.protected_kb_unchanged is None
        assert any("UNVERIFIED" in o for o in snap.observations())

    def test_a_changed_point_count_is_a_failure(self):
        snap = RetrievalIntegritySnapshot(
            index_points_before=812, index_points_after=800, chunk_rows_before=812, chunk_rows_after=812
        )
        assert snap.protected_kb_unchanged is False

    def test_a_changed_benchmark_is_flagged_loudly(self):
        snap = RetrievalIntegritySnapshot(benchmark_sha256_before="a", benchmark_sha256_after="b")
        assert snap.benchmark_unchanged is False
        assert any("VOID" in o for o in snap.observations())
