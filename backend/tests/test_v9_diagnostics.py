"""V9 Phase 3 regression tests: retrieval diagnostics with UNKNOWN semantics.

Pure unit tests. No network, no Qdrant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.diagnostics import (  # noqa: E402
    DiagnosticMetric,
    build_question_diagnostic,
    summarise_diagnostics,
)

CHUNK_INDEX = {
    "chk_engine": {
        "document_id": "doc_auto",
        "source_title": "Automobile Engineering Handbook",
        "source_url": "https://example.org/engine",
        "source_type": "reference",
        "publisher": "Example Press",
        "section": "Engines",
        "page": 12,
        "trust_score": 0.8,
        "content_hash": "hash_engine",
    },
    "chk_brake": {
        "document_id": "doc_auto",
        "source_title": "Automobile Engineering Handbook",
        "source_url": "https://example.org/brake",
        "source_type": "reference",
        "section": "Brakes",
        "page": 40,
        "trust_score": 0.8,
        "content_hash": "hash_brake",
    },
    "chk_tyre": {
        "document_id": "doc_tyres",
        "source_title": "Tyre Catalogue",
        "source_type": "vendor",
        "section": "Compounds",
        "trust_score": 0.5,
        "content_hash": "hash_tyre",
    },
}


class TestGroundTruthUnavailable:
    def test_metrics_are_unknown_not_zero_without_a_required_label(self):
        d = build_question_diagnostic(
            question_id="q1",
            question="What is an engine?",
            strategy="dense",
            top_k=5,
            ranked_chunk_ids=["chk_engine", "chk_brake"],
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=[],
        )
        assert d.ground_truth_available is False
        for metric in (d.recall_at_k, d.precision_at_k, d.mrr, d.ndcg_at_k):
            assert metric.measured is False
            assert metric.value is None
            assert "unknown, not zero" in metric.reason
        assert d.first_relevant_rank is None
        assert d.required_retrieved is None

    def test_unknown_is_distinguishable_from_a_real_zero(self):
        """A real miss is measured 0.0; a missing label is unmeasured."""
        no_gt = build_question_diagnostic(
            question_id="q1", ranked_chunk_ids=["chk_tyre"], chunk_index=CHUNK_INDEX
        )
        real_miss = build_question_diagnostic(
            question_id="q2",
            ranked_chunk_ids=["chk_tyre"],
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=["chk_engine"],
            top_k=5,
        )
        assert no_gt.recall_at_k.measured is False
        assert no_gt.recall_at_k.value is None
        assert real_miss.recall_at_k.measured is True
        assert real_miss.recall_at_k.value == 0.0
        assert real_miss.required_retrieved is False
        assert real_miss.ground_truth_available is True


class TestGroundTruthAvailable:
    def test_required_evidence_at_rank_one(self):
        d = build_question_diagnostic(
            question_id="q1",
            ranked_chunk_ids=["chk_engine", "chk_brake", "chk_tyre"],
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=["chk_engine"],
            top_k=5,
        )
        assert d.ground_truth_available is True
        assert d.first_relevant_rank == 1
        assert d.required_retrieved is True
        assert d.recall_at_k.value == 1.0
        assert d.mrr.value == 1.0
        assert d.required_recall_at_depth.value == 1.0

    def test_required_evidence_deep_in_the_list(self):
        ranked = ["chk_brake", "chk_tyre", "chk_brake", "chk_engine"]
        d = build_question_diagnostic(
            question_id="q1",
            ranked_chunk_ids=ranked,
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=["chk_engine"],
            top_k=4,
        )
        assert d.first_relevant_rank == 4
        assert d.mrr.value == pytest.approx(0.25)
        assert d.recall_at_k.value == 1.0
        assert d.recall_at_k.measured is True

    def test_required_evidence_beyond_top_k(self):
        ranked = ["chk_brake"] * 6 + ["chk_engine"]
        d = build_question_diagnostic(
            question_id="q1",
            ranked_chunk_ids=ranked,
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=["chk_engine"],
            top_k=5,
        )
        assert d.first_relevant_rank == 7
        # Recall@5 misses it, but depth-independent recall finds it: the two
        # numbers intentionally disagree and that disagreement is the signal.
        assert d.recall_at_k.value == 0.0
        assert d.required_recall_at_depth.value == 1.0
        # The evidence IS in the candidate set, so this is a DEPTH problem, not
        # a corpus gap: the gap warning must NOT fire, or every deep rank would
        # be misreported as missing evidence.
        assert d.required_retrieved is True
        assert not any("retrieval or corpus gap" in w for w in d.warnings)

    def test_multiple_required_chunks_partial_recall(self):
        d = build_question_diagnostic(
            question_id="q1",
            ranked_chunk_ids=["chk_engine", "chk_tyre"],
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=["chk_engine", "chk_brake"],
            top_k=2,
        )
        assert d.recall_at_k.value == pytest.approx(0.5)
        assert d.required_recall_at_depth.value == pytest.approx(0.5)
        # required_recall_at_depth sees the whole candidate set, so it reports
        # the 0.5 that recall@2 also reports; the two agree here and disagree
        # only when the evidence sits beyond top_k.
        assert d.required_recall_at_depth.measured is True
        assert d.required_retrieved is True


class TestProvenanceIsCarried:
    def test_retrieved_items_expose_source_provenance(self):
        d = build_question_diagnostic(
            question_id="q1",
            ranked_chunk_ids=["chk_engine"],
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=["chk_engine"],
        )
        item = d.retrieved[0]
        assert item.rank == 1
        assert item.document_id == "doc_auto"
        assert item.source_title == "Automobile Engineering Handbook"
        assert item.source_url == "https://example.org/engine"
        assert item.section == "Engines"
        assert item.page == 12
        assert item.content_hash == "hash_engine"
        assert item.is_required is True

    def test_unindexed_chunk_reports_unknown_provenance_not_fabricated(self):
        d = build_question_diagnostic(
            question_id="q1",
            ranked_chunk_ids=["chk_not_in_index"],
            chunk_index=CHUNK_INDEX,
        )
        item = d.retrieved[0]
        assert item.chunk_id == "chk_not_in_index"
        assert item.document_id is None
        assert item.source_title is None
        assert item.score is None


class TestEvidenceSelectionVsRetrieval:
    def test_retrieved_but_unselected_required_evidence_is_flagged(self):
        d = build_question_diagnostic(
            question_id="q1",
            ranked_chunk_ids=["chk_engine", "chk_brake"],
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=["chk_engine"],
            evidence_selected_ids=["chk_brake"],
        )
        assert d.required_missing_from_generation == ["chk_engine"]
        assert any("evidence-SELECTION fault" in w for w in d.warnings)

    def test_selected_required_evidence_produces_no_warning(self):
        d = build_question_diagnostic(
            question_id="q1",
            ranked_chunk_ids=["chk_engine"],
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=["chk_engine"],
            evidence_selected_ids=["chk_engine"],
        )
        assert d.required_missing_from_generation == []
        assert d.required_selected_ids == ["chk_engine"]


class TestEmptyRetrieval:
    def test_no_chunks_is_unknown_and_warned(self):
        d = build_question_diagnostic(
            question_id="q1",
            ranked_chunk_ids=[],
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=["chk_engine"],
        )
        assert d.retrieval_depth == 0
        assert d.recall_at_k.measured is True
        assert d.recall_at_k.value == 0.0
        assert any("nothing to rank" in w for w in d.warnings)


class TestSummary:
    def _diag(self, qid: str, ranked: list[str], required: list[str], top_k: int = 5):
        return build_question_diagnostic(
            question_id=qid,
            ranked_chunk_ids=ranked,
            chunk_index=CHUNK_INDEX,
            required_chunk_ids=required,
            top_k=top_k,
            latency_ms=10.0,
        )

    def test_unknown_questions_are_excluded_from_means_not_zeroed(self):
        with_gt = self._diag("q1", ["chk_engine"], ["chk_engine"])
        without_gt = self._diag("q2", ["chk_brake"], [])
        summary = summarise_diagnostics([with_gt, without_gt])

        assert summary.question_count == 2
        assert summary.ground_truth_question_count == 1
        assert summary.no_ground_truth_question_count == 1
        # Mean is over the ONE measured question, not over 2 with a zero.
        assert summary.recall_at_k.value == 1.0
        assert summary.recall_at_k.sample_size == 1
        assert "excluded as unmeasured" in summary.recall_at_k.reason
        assert any("excluded from the means" in w for w in summary.warnings)

    def test_no_ground_truth_at_all_yields_unknown_metrics(self):
        summary = summarise_diagnostics([self._diag("q1", ["chk_brake"], [])])
        assert summary.recall_at_k.measured is False
        assert summary.recall_at_k.value is None
        assert "recall_at_k" in summary.unknown_metrics

    def test_empty_summary_is_unknown_not_zero(self):
        summary = summarise_diagnostics([])
        assert summary.question_count == 0
        assert summary.recall_at_k.measured is False
        assert summary.warnings == ["no question diagnostics supplied"]

    def test_rank_histogram_uses_buckets(self):
        diags = [
            self._diag("q1", ["chk_engine"], ["chk_engine"]),
            self._diag("q2", ["chk_brake", "chk_engine"], ["chk_engine"]),
            self._diag("q3", ["chk_brake"], ["chk_engine"]),
        ]
        summary = summarise_diagnostics(diags)
        assert summary.first_relevant_rank_histogram["rank_1"] == 1
        assert summary.first_relevant_rank_histogram["rank_2_3"] == 1
        assert summary.first_relevant_rank_histogram["not_retrieved"] == 1

    def test_questions_with_no_results_are_listed(self):
        summary = summarise_diagnostics([self._diag("q1", [], ["chk_engine"])])
        assert summary.no_result_question_ids == ["q1"]
