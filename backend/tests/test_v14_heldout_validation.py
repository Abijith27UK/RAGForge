"""V14 held-out validation rules.

The validator is the only thing standing between an agent's plausible-looking
draft and a "held-out" claim, so these tests are written against the failure
modes it exists to catch:

* a question whose evidence has been deleted, or that exists in the corpus but has
  no vector — the second must NOT be reported as the first, because "the evidence
  is missing" and "retrieval was never given a chance" are different findings;
* a question that reuses a development chunk, which would silently make a
  "held-out" number a re-measurement of the data a configuration was chosen on;
* a question that is a paraphrase of a development question;
* an evidence label whose claimed answering sentence is not in the chunk, i.e. a
  sufficiency claim that cannot be shown.

Pure: the corpus is a dictionary of stand-in rows, the index is a set, and the
development questions are passed in. No database, no vector store.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.heldout_benchmark import (  # noqa: E402
    CandidateStatus,
    HeldOutCandidate,
    HeldOutCandidateSet,
    HeldOutReviewRequest,
    QueryType,
    ReviewDecision,
    apply_review,
)
from app.services.answer_eval.heldout_validation import (  # noqa: E402
    EvidenceProbe,
    char_ngram_jaccard,
    measure_separation,
    normalize_text,
    normalize_tokens,
    token_jaccard,
    validate_candidates,
)

PROTOCOL = {
    "method": "test protocol",
    "question_selection": "test selection",
    "evidence_selection": "test evidence",
    "review_policy": "test review",
}

CHUNK_TEXT = (
    "Brake fade is the reduction in braking effectiveness as the brake heats. "
    "The continuous power dissipation depends on the ambient cooling air."
)


@dataclass
class Row:
    """A stand-in for a Chunk row: only what the validator reads."""

    id: str
    document_id: str
    text: str
    section: str = "Introduction"


#: Sentinel so a test can distinguish "use the default index" from "the index is
#: unverifiable" (None), which is the case that must NOT be read as absence.
_DEFAULT = object()


def _probe(
    *,
    chunks: dict[str, Row] | None = None,
    index: object = _DEFAULT,
    documents: set[str] | None = None,
    index_note: str = "",
) -> EvidenceProbe:
    rows = chunks if chunks is not None else {"chk_a": Row("chk_a", "doc_a", CHUNK_TEXT)}
    return EvidenceProbe(
        chunk_rows=rows,
        document_ids=documents if documents is not None else {"doc_a"},
        index_chunk_ids={"chk_a"} if index is _DEFAULT else index,  # type: ignore[arg-type]
        index_note=index_note,
    )


def _cand(qid: str = "hout-001", **overrides) -> HeldOutCandidate:
    payload = {
        "id": qid,
        "subdomain": "braking",
        "question": "What is brake fade and what does dissipation depend on?",
        "expected_chunk_ids": ["chk_a"],
        "expected_document_ids": ["doc_a"],
        "evidence_quote": "Brake fade is the reduction in braking effectiveness as the brake heats.",
        "coverage_rationale": "adds the brake failure mode; the development set has none",
        "query_type": QueryType.TERMINOLOGY,
        "provenance": "Chunk chk_a read and selected 2026-10-11",
        "draft_origin": "agent-drafted from the chunk text; no retrieval",
    }
    payload.update(overrides)
    return HeldOutCandidate(**payload)


def _set(candidates: list[HeldOutCandidate], **overrides) -> HeldOutCandidateSet:
    payload = {
        "benchmark": "heldout-test",
        "version": 1,
        "kb_id": "kb_test",
        "protocol": dict(PROTOCOL),
        "candidates": candidates,
    }
    payload.update(overrides)
    return HeldOutCandidateSet(**payload)


DEV_QUESTIONS = [
    {
        "id": "auto-eng-001",
        "subdomain": "engine",
        "question": "What is an engine and what does it convert energy into?",
        "expected_chunk_ids": ["chk_dev_1"],
        "expected_document_ids": ["doc_dev"],
    }
]


def _codes(report, qid: str = "") -> set[str]:
    """Finding codes, from BOTH the structured list and the per-question views.

    Set-level findings (a duplicate id, identical question text) belong to no
    single question, so reading only the per-question findings would miss them
    and quietly pass a broken artifact.
    """
    codes = {f.code for f in report.findings if not qid or f.question_id == qid}
    codes |= {
        f.code for q in report.questions if q.question_id == qid for f in q.findings
    } if qid else {f.code for q in report.questions for f in q.findings}
    return codes


class TestSimilarityMeasures:
    def test_tokens_drop_stopwords_and_normalise_case(self):
        assert normalize_tokens("What is THE Engine?") == ["engine"]

    def test_whitespace_is_collapsed_and_lowercased(self):
        assert normalize_text("  Brake   FADE\n\nis") == "brake fade is"

    def test_identical_text_scores_one(self):
        assert token_jaccard("brake fade", "brake fade") == 1.0
        assert char_ngram_jaccard("brake fade", "brake fade") == 1.0

    def test_unrelated_text_scores_low(self):
        assert token_jaccard("brake fade", "propeller cavitation") < 0.2

    def test_empty_text_scores_zero_rather_than_raising(self):
        assert token_jaccard("", "brake fade") == 0.0
        assert char_ngram_jaccard("", "") == 0.0


class TestEvidenceResolution:
    def test_resolvable_evidence_produces_no_error(self):
        report = validate_candidates(
            _set([_cand()]), _probe(), dev_questions=DEV_QUESTIONS
        )
        assert report.errors == []
        assert report.freezeable is True
        assert "evidence.present_in_corpus" in report.checks_run
        assert "evidence.present_in_index" in report.checks_run

    def test_evidence_absent_from_the_corpus_is_reported_as_corpus_absence(self):
        report = validate_candidates(
            _set([_cand(expected_chunk_ids=["chk_gone"])]),
            _probe(),
            dev_questions=DEV_QUESTIONS,
        )
        codes = _codes(report, "hout-001")
        assert "EVIDENCE_MISSING_FROM_CORPUS" in codes
        assert "EVIDENCE_MISSING_FROM_INDEX" not in codes, (
            "a chunk that is not in the corpus cannot also be 'in the corpus but "
            "unindexed'; the two findings must not be conflated"
        )
        assert report.freezeable is False

    def test_evidence_in_the_corpus_but_not_the_index_is_reported_separately(self):
        """The distinction that stops a corpus gap being read as a retrieval
        failure: the chunk exists, so a query COULD have found it if it had a
        vector. It does not, so every retriever must fail on this question."""
        report = validate_candidates(
            _set([_cand()]), _probe(index=set()), dev_questions=DEV_QUESTIONS
        )
        codes = _codes(report, "hout-001")
        assert "EVIDENCE_MISSING_FROM_INDEX" in codes
        assert "EVIDENCE_MISSING_FROM_CORPUS" not in codes
        detail = next(
            f.detail
            for q in report.questions
            for f in q.findings
            if f.code == "EVIDENCE_MISSING_FROM_INDEX"
        )
        assert "no vector" in detail

    def test_an_unreadable_index_is_not_treated_as_evidence_absence(self):
        report = validate_candidates(
            _set([_cand()]),
            _probe(index=None, index_note="the vector index could not be read (boom)"),
            dev_questions=DEV_QUESTIONS,
        )
        codes = _codes(report, "hout-001")
        assert "EVIDENCE_MISSING_FROM_INDEX" not in codes
        assert report.errors == [], "an unprobeable index is not the candidate's fault"
        assert [c["check"] for c in report.checks_not_run] == ["evidence.present_in_index"]
        assert any("index presence" in l for l in report.limitations)
        assert report.questions[0].evidence_in_index is None


class TestEvidenceSufficiency:
    def test_a_quote_that_is_not_in_the_chunk_is_an_error(self):
        report = validate_candidates(
            _set([_cand(evidence_quote="Brake fade is a type of shoe.")]),
            _probe(),
            dev_questions=DEV_QUESTIONS,
        )
        assert "EVIDENCE_QUOTE_NOT_FOUND_IN_CHUNK" in _codes(report, "hout-001")
        assert report.freezeable is False

    def test_a_quote_is_matched_despite_formatting_differences(self):
        """Chunk text comes out of HTML with runs of whitespace, so a byte-exact
        test would fail on formatting rather than on content."""
        probe = _probe(
            chunks={"chk_a": Row("chk_a", "doc_a", "Introduction\n\n  Brake FADE  is the\nreduction in braking effectiveness.")}
        )
        report = validate_candidates(
            _set([_cand(evidence_quote="brake fade is the reduction in braking effectiveness")]),
            probe,
            dev_questions=DEV_QUESTIONS,
        )
        assert "EVIDENCE_QUOTE_NOT_FOUND_IN_CHUNK" not in _codes(report, "hout-001")

    def test_an_absent_quote_is_a_warning_not_a_silent_pass(self):
        report = validate_candidates(
            _set([_cand(evidence_quote="")]), _probe(), dev_questions=DEV_QUESTIONS
        )
        assert "EVIDENCE_QUOTE_MISSING" in _codes(report, "hout-001")
        assert any("sufficiency of the label cannot be verified" in w for w in report.warnings)


class TestLabelsAndProvenance:
    def test_a_question_with_no_expected_chunk_is_an_error(self):
        report = validate_candidates(
            _set([_cand(expected_chunk_ids=[])]), _probe(), dev_questions=DEV_QUESTIONS
        )
        assert "MISSING_EXPECTED_CHUNK" in _codes(report, "hout-001")

    def test_a_chunk_from_another_document_than_claimed_is_an_error(self):
        report = validate_candidates(
            _set([_cand(expected_document_ids=["doc_other"])]),
            _probe(documents={"doc_a", "doc_other"}),
            dev_questions=DEV_QUESTIONS,
        )
        assert "CHUNK_DOCUMENT_MISMATCH" in _codes(report, "hout-001")

    def test_a_missing_expected_document_is_an_error(self):
        report = validate_candidates(
            _set([_cand(expected_document_ids=["doc_ghost"])]),
            _probe(),
            dev_questions=DEV_QUESTIONS,
        )
        assert "EXPECTED_DOCUMENT_MISSING" in _codes(report, "hout-001")

    def test_missing_provenance_is_an_error_but_missing_metadata_is_a_warning(self):
        report = validate_candidates(
            _set([_cand(provenance="", draft_origin="", coverage_rationale="", subdomain="")]),
            _probe(),
            dev_questions=DEV_QUESTIONS,
        )
        codes = _codes(report, "hout-001")
        assert "PROVENANCE_MISSING" in codes
        assert report.freezeable is False
        assert {
            "DRAFT_ORIGIN_MISSING",
            "COVERAGE_RATIONALE_MISSING",
            "SUBDOMAIN_UNSPECIFIED",
        } <= codes


class TestUniquenessAndSeparation:
    def test_a_duplicate_question_id_is_an_error(self):
        report = validate_candidates(
            _set([_cand("hout-001"), _cand("hout-001")]),
            _probe(),
            dev_questions=DEV_QUESTIONS,
        )
        assert "DUPLICATE_QUESTION_ID" in _codes(report, "hout-001")

    def test_a_word_for_word_duplicate_question_is_an_error(self):
        report = validate_candidates(
            _set([_cand("hout-001"), _cand("hout-002")]),
            _probe(),
            dev_questions=DEV_QUESTIONS,
        )
        assert "DUPLICATE_QUESTION_TEXT" in _codes(report, "hout-002"), (
            "two ids with identical text are one question, not two"
        )

    def test_reusing_a_development_chunk_is_an_error(self):
        report = validate_candidates(
            _set([_cand(expected_chunk_ids=["chk_dev_1"])]),
            _probe(chunks={"chk_dev_1": Row("chk_dev_1", "doc_dev", CHUNK_TEXT), "chk_a": Row("chk_a", "doc_a", CHUNK_TEXT)}),
            dev_questions=DEV_QUESTIONS,
        )
        codes = _codes(report, "hout-001")
        assert "SHARED_EVIDENCE_WITH_DEVELOPMENT" in codes
        assert report.freezeable is False
        detail = next(
            f.detail for f in report.questions[0].findings if f.code == "SHARED_EVIDENCE_WITH_DEVELOPMENT"
        )
        assert "re-measurement" in detail

    def test_a_near_paraphrase_of_a_development_question_is_flagged(self):
        paraphrase = "What is an engine and what does it convert energy into?"  # identical text
        report = validate_candidates(
            _set([_cand(question=paraphrase)]),
            _probe(),
            dev_questions=DEV_QUESTIONS,
        )
        codes = _codes(report, "hout-001")
        assert "PARAPHRASE_SUSPECTED" in codes
        assert "SHARED_EVIDENCE_WITH_DEVELOPMENT" not in codes, (
            "no chunk is shared here, so only the paraphrase check may fire"
        )

    def test_separation_reports_the_nearest_development_question(self):
        report = validate_candidates(
            _set([_cand()]), _probe(), dev_questions=DEV_QUESTIONS
        )
        per_question = report.separation["per_question"]["hout-001"]
        assert per_question["nearest_development_question_id"] == "auto-eng-001"
        assert per_question["token_jaccard"] < 0.6
        assert per_question["paraphrase_suspected"] is False

    def test_separation_states_that_it_is_lexical_only(self):
        report = validate_candidates(
            _set([_cand()]), _probe(), dev_questions=DEV_QUESTIONS
        )
        assert "does not prove" in report.separation["note"]
        assert report.separation["development_benchmark_questions"] == 1

    def test_two_questions_sharing_a_chunk_between_themselves_is_not_dev_overlap(self):
        """Inside the held-out set, evidence reuse is a design choice; the ERROR
        is only about the DEVELOPMENT set."""
        report = validate_candidates(
            _set([_cand("hout-001"), _cand("hout-002", question="A different question about dissipation?")]),
            _probe(),
            dev_questions=DEV_QUESTIONS,
        )
        assert "SHARED_EVIDENCE_WITH_DEVELOPMENT" not in _codes(report, "hout-002")


class TestReviewStateAndSeverity:
    def test_an_unreviewed_question_is_recorded_as_pending_not_as_clean(self):
        report = validate_candidates(
            _set([_cand()]), _probe(), dev_questions=DEV_QUESTIONS
        )
        assert "REVIEW_PENDING" in _codes(report, "hout-001")
        assert report.questions[0].needs_review is True
        assert report.review["complete"] is False
        assert report.verdict.startswith("FREEZEABLE AS PROVISIONAL")

    def test_a_reviewed_set_reports_that_it_can_be_validated(self):
        reviewed = apply_review(
            _set([_cand()]),
            [
                HeldOutReviewRequest(
                    question_id="hout-001", decision=ReviewDecision.APPROVE, reviewer="R"
                )
            ],
        )
        report = validate_candidates(reviewed, _probe(), dev_questions=DEV_QUESTIONS)
        assert report.verdict.startswith("FREEZEABLE AS VALIDATED")
        assert report.questions[0].needs_review is False

    def test_errors_on_rejected_candidates_do_not_block_the_freeze(self):
        rejected = apply_review(
            _set([_cand("hout-001"), _cand("hout-002", expected_chunk_ids=["chk_gone"])]),
            [
                HeldOutReviewRequest(
                    question_id="hout-002",
                    decision=ReviewDecision.REJECT,
                    reviewer="R",
                    notes="evidence does not resolve",
                )
            ],
        )
        report = validate_candidates(rejected, _probe(), dev_questions=DEV_QUESTIONS)
        assert report.errors == [], "a rejected question must not hold the instrument hostage"
        assert report.freezeable is True
        findings = {
            f.code: f
            for q in report.questions
            if q.question_id == "hout-002"
            for f in q.findings
        }
        assert findings["EVIDENCE_MISSING_FROM_CORPUS"].applies_to_frozen is False
        assert any("rejection rate" in l for l in report.limitations)

    def test_the_verdict_names_the_blocking_error_count(self):
        report = validate_candidates(
            _set([_cand(expected_chunk_ids=[])]), _probe(), dev_questions=DEV_QUESTIONS
        )
        assert report.verdict.startswith("NOT FREEZEABLE")
        assert f"{len(report.errors)} error" in report.verdict
        assert len(report.errors) >= 1

    def test_set_level_findings_are_exposed_structurally(self):
        """A duplicate-id finding belongs to no single question, so it must be
        readable without parsing the human-readable error strings."""
        report = validate_candidates(
            _set([_cand("hout-001"), _cand("hout-001"), _cand("hout-002")]),
            _probe(),
            dev_questions=DEV_QUESTIONS,
        )
        structured = {f.code for f in report.findings}
        assert "DUPLICATE_QUESTION_ID" in structured
        assert "DUPLICATE_QUESTION_TEXT" in structured
