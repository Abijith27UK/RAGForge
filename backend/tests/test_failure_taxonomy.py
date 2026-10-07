"""V9 Phase 2 regression tests for the deterministic failure taxonomy.

All tests are pure unit tests; no network, no Qdrant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.failure import (  # noqa: E402
    ALL_CODES,
    FailureClassification,
    FailureCode,
    classify_failure,
)

ANSWERABLE_TEXT = (
    "The engine converts chemical energy into mechanical energy. "
    "The combustion chamber compresses the air-fuel mixture before ignition."
)

NO_EVIDENCE = [
    {"chunk_id": "chk_01", "document_id": "doc_a", "relevant": True, "required": False},
]


class TestFailureClassificationConstants:
    def test_all_codes_are_known(self):
        for code in ALL_CODES:
            assert code in {
                FailureCode.NO_RELEVANT_RETRIEVAL,
                FailureCode.LOW_RETRIEVAL_RECALL,
                FailureCode.WRONG_DOCUMENT,
                FailureCode.WRONG_CHUNK,
                FailureCode.INSUFFICIENT_EVIDENCE,
                FailureCode.CONFLICTING_EVIDENCE,
                FailureCode.CITATION_ERROR,
                FailureCode.UNSUPPORTED_CLAIM,
                FailureCode.ANSWER_RELEVANCE_FAILURE,
                FailureCode.GENERATION_FAILURE,
                FailureCode.UNKNOWN,
            }

    def test_unknown_classification_is_allowed(self):
        f = FailureClassification(classification="", confidence="unknown")
        assert f.classification == ""


class TestFailureNoRelevantRetrieval:
    def test_no_relevant_retrieval_classifies_deterministically(self):
        result = classify_failure(
            question="What is an engine?",
            answer_text="",
            evidence=[],
            retrieved_chunk_ids=["chk_01"],
            retrieved_rank_of_required=None,
            required_chunk_ids=["chk_02"],
            retrieval_hit_rate=0.0,
            question_answer_relevance=None,
            relevance_passed=None,
            claim_verdicts=[],
            fabrication_rate=None,
            unsupported_citation_rate=None,
            answerability="answerable",
        )
        assert result.classification == FailureCode.NO_RELEVANT_RETRIEVAL
        assert result.confidence == "deterministic"


class TestFailureLowRetrievalRecall:
    def test_low_recall_on_rank_country(self):
        result = classify_failure(
            question="What is an engine?",
            answer_text=ANSWERABLE_TEXT,
            evidence=[
                {"chunk_id": "chk_req", "document_id": "doc_a", "relevant": True,
                 "required": True},
            ],
            retrieved_chunk_ids=["chk_req"],
            retrieved_rank_of_required=5,
            required_chunk_ids=["chk_req"],
            retrieval_hit_rate=1.0,
            question_answer_relevance=0.45,
            relevance_passed=True,
            claim_verdicts=[],
            fabrication_rate=None,
            unsupported_citation_rate=None,
            answerability="answerable",
        )
        assert result.classification == FailureCode.LOW_RETRIEVAL_RECALL
        assert result.confidence == "deterministic"


class TestFailureWrongChunk:
    def test_wrong_chunk_classifies(self):
        result = classify_failure(
            question="What is an engine?",
            answer_text=ANSWERABLE_TEXT,
            evidence=[],
            retrieved_chunk_ids=["chk_01"],
            retrieved_rank_of_required=2,
            required_chunk_ids=["chk_req"],
            retrieval_hit_rate=1.0,
            question_answer_relevance=0.45,
            relevance_passed=True,
            claim_verdicts=[],
            fabrication_rate=None,
            unsupported_citation_rate=None,
            answerability="answerable",
        )
        assert result.classification == FailureCode.WRONG_CHUNK


class TestFailureAnswerRelevanceFailure:
    def test_relevance_failure_classifies(self):
        result = classify_failure(
            question="What is an engine?",
            answer_text="The atmosphere is composed primarily of nitrogen and oxygen.",
            evidence=[],
            retrieved_chunk_ids=[],
            retrieved_rank_of_required=None,
            required_chunk_ids=["chk_req"],
            retrieval_hit_rate=None,
            question_answer_relevance=0.15,
            relevance_passed=False,
            claim_verdicts=[],
            fabrication_rate=None,
            unsupported_citation_rate=None,
            answerability="answerable",
        )
        assert result.classification == FailureCode.ANSWER_RELEVANCE_FAILURE


class TestFailureGenerationFailure:
    def test_empty_generation_classifies(self):
        result = classify_failure(
            question="What is an engine?",
            answer_text="",
            evidence=[],
            retrieved_chunk_ids=[],
            retrieved_rank_of_required=None,
            required_chunk_ids=["chk_req"],
            retrieval_hit_rate=None,
            question_answer_relevance=None,
            relevance_passed=None,
            claim_verdicts=[],
            fabrication_rate=None,
            unsupported_citation_rate=None,
            answerability="answerable",
        )
        assert result.classification == FailureCode.GENERATION_FAILURE
        assert result.confidence == "partial"


class TestFailureUnknown:
    def test_unknown_when_evidence_to_short(self):
        result = classify_failure(
            question="What is an engine?",
            answer_text="The answer is unknown.",
            evidence=[],
            retrieved_chunk_ids=[],
            retrieved_rank_of_required=None,
            required_chunk_ids=["chk_req"],
            retrieval_hit_rate=None,
            question_answer_relevance=0.5,
            relevance_passed=True,
            claim_verdicts=[],
            fabrication_rate=None,
            unsupported_citation_rate=None,
            answerability="answerable",
        )
        assert result.classification == FailureCode.UNKNOWN
        assert result.confidence == "unknown"
