"""V11 benchmark quality audit tests: findings are reported, never applied.

Each test drives ONE check and asserts the code it must produce, so a check that
silently stops firing fails here rather than disappearing from the report. The
"could not run" paths are tested too: reporting "0 problems" for a check that
never executed is the exact failure this module exists to avoid.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.benchmark import (  # noqa: E402
    AnswerBenchmark,
    AnswerBenchmarkQuestion,
    CitationRequirements,
    RequiredEvidence,
)
from app.services.answer_eval.drafting import MIN_QUESTION_COVERAGE  # noqa: E402
from app.services.answer_eval.ground_truth import (  # noqa: E402
    GroundTruthAnnotation,
    build_author_provenance,
)
from app.services.answer_eval.quality_audit import (  # noqa: E402
    RecommendedAction,
    Severity,
    run_quality_audit,
)
from app.services.answer_eval.relevance import TermWeights  # noqa: E402

TEXT_A = (
    "A vehicle frame, also historically known as its chassis, is the main supporting "
    "structure of a motor vehicle to which all other components are attached."
)
TEXT_B = "Nearly all trucks and buses, and most pickups continue to use a separate frame as their chassis."
WEIGHTS = TermWeights(
    n_docs=812,
    df={"vehicle": 90, "frame": 60, "chassis": 55, "trucks": 20, "buses": 18},
    source="test-fixture",
)


def _question(
    text: str,
    chunk_ids: list[str],
    *,
    question_id: str = "q-001",
    hash_: str = "hash-1",
    document_id: str = "doc_1",
    title: str = "Vehicle frame - Wikipedia",
    answerability: str = "answerable",
    citation: CitationRequirements | None = None,
) -> AnswerBenchmarkQuestion:
    return AnswerBenchmarkQuestion(
        question_id=question_id,
        question=text,
        subdomain="chassis",
        answerability=answerability,
        required_evidence=[
            RequiredEvidence(
                chunk_id=cid,
                document_id=document_id,
                content_hash=hash_,
                source_title=title,
                required=True,
            )
            for cid in chunk_ids
        ],
        citation_requirements=citation or CitationRequirements(),
    )


def _benchmark(*questions: AnswerBenchmarkQuestion) -> AnswerBenchmark:
    return AnswerBenchmark(
        benchmark="audit-fixture",
        version=1,
        created_at="2026-01-01",
        kb_id="kb_test",
        questions=list(questions),
    )


def _meta(**overrides) -> dict:
    base = {
        "chk_a": {
            "document_id": "doc_1",
            "content_hash": "hash-1",
            "source_title": "Vehicle frame - Wikipedia",
        }
    }
    base.update(overrides)
    return base


def _codes(audit) -> set[str]:
    return {f.code for f in audit.findings}


def _codes_for(audit, question_id: str) -> set[str]:
    return {f.code for f in audit.findings if f.question_id == question_id}


def _run(benchmark, texts, *, annotations=None, weights=WEIGHTS, meta=None):
    return run_quality_audit(
        benchmark=benchmark,
        evidence_texts=texts,
        evidence_meta=meta if meta is not None else _meta(),
        annotations=annotations or [],
        weights=weights,
        benchmark_path="benchmarks/audit-fixture.json",
    )


# -- evidence integrity ------------------------------------------------------


def test_missing_chunk_is_an_error_and_makes_answerability_unsupported():
    benchmark = _benchmark(
        _question("What is a vehicle frame?", ["chk_gone"])
    )
    audit = _run(benchmark, {})
    assert audit.error_count >= 2
    assert "EVIDENCE_MISSING_FROM_CORPUS" in _codes(audit)
    assert "ANSWERABILITY_UNSUPPORTED" in _codes(audit)
    assert audit.ready_to_freeze is False
    assert audit.questions[0].recommended_action is RecommendedAction.CLASSIFY_INSUFFICIENT_EVIDENCE


def test_content_hash_mismatch_is_an_error():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    audit = _run(
        benchmark,
        {"chk_a": TEXT_A},
        meta=_meta(chk_a={"document_id": "doc_1", "content_hash": "hash-CHANGED"}),
    )
    assert "EVIDENCE_HASH_MISMATCH" in _codes(audit)
    assert audit.error_count >= 1


def test_document_mismatch_is_an_error():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    audit = _run(
        benchmark,
        {"chk_a": TEXT_A},
        meta=_meta(chk_a={"document_id": "doc_OTHER", "content_hash": "hash-1"}),
    )
    assert "EVIDENCE_DOCUMENT_MISMATCH" in _codes(audit)


def test_trivial_evidence_is_a_warning():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    audit = _run(benchmark, {"chk_a": "Frame."})
    assert "EVIDENCE_TEXT_TRIVIAL" in _codes(audit)
    assert audit.questions[0].recommended_action is RecommendedAction.CLASSIFY_INSUFFICIENT_EVIDENCE


def test_question_with_no_pinned_evidence_is_an_error():
    """The answerable-without-evidence case is refused by the model itself
    (V8 guard), so the audit covers the case that CAN reach it: a question whose
    answerability was never established and which pins no evidence."""
    import pytest as _pytest

    with _pytest.raises(Exception, match="must name the required evidence"):
        _question("What is a vehicle frame?", [])

    benchmark = _benchmark(
        _question("What is a vehicle frame?", [], answerability="unknown")
    )
    audit = _run(benchmark, {})
    assert "NO_REQUIRED_EVIDENCE" in _codes(audit)
    assert audit.error_count >= 1


# -- question/evidence match -------------------------------------------------


def test_evidence_that_barely_covers_the_question_is_flagged_for_a_human():
    benchmark = _benchmark(
        _question(
            "How does a wastegate turbocharger differ from a supercharger pulley?",
            ["chk_a"],
        )
    )
    audit = _run(benchmark, {"chk_a": TEXT_A})
    assert "EVIDENCE_QUESTION_COVERAGE_LOW" in _codes(audit)
    finding = next(f for f in audit.findings if f.code == "EVIDENCE_QUESTION_COVERAGE_LOW")
    assert finding.severity is Severity.WARNING
    assert finding.action is RecommendedAction.REVIEW_QUESTION_EVIDENCE_MATCH
    assert finding.detail["threshold"] == MIN_QUESTION_COVERAGE
    assert audit.questions[0].coverage is not None
    assert audit.questions[0].coverage < MIN_QUESTION_COVERAGE


def test_term_absence_alone_is_only_an_observation():
    """A light, paraphrase-shaped gap must not be escalated to a defect.

    'support' is a common corpus word (high document frequency, so low IDF) and
    the evidence states the same idea as 'supporting'. The observation is
    recorded; the question is not escalated for a human.
    """
    weights = TermWeights(
        n_docs=812,
        df={"vehicle": 90, "frame": 60, "chassis": 55, "support": 300},
        source="test-fixture",
    )
    benchmark = _benchmark(
        _question("What is a vehicle frame chassis and what does it support?", ["chk_a"])
    )
    audit = _run(benchmark, {"chk_a": TEXT_A}, weights=weights)
    codes = _codes_for(audit, "q-001")
    assert "QUESTION_TERMS_ABSENT_FROM_EVIDENCE" in codes
    assert "EVIDENCE_QUESTION_COVERAGE_LOW" not in codes
    finding = next(f for f in audit.findings if f.code == "QUESTION_TERMS_ABSENT_FROM_EVIDENCE")
    assert finding.severity is Severity.INFO
    assert audit.questions[0].recommended_action is not RecommendedAction.REVIEW_QUESTION_EVIDENCE_MATCH


def test_absent_terms_are_recorded_even_when_the_question_is_escalated():
    """Both findings appear together: the observation plus the escalation."""
    benchmark = _benchmark(
        _question("How does a wastegate turbocharger differ from a supercharger?", ["chk_a"])
    )
    audit = _run(benchmark, {"chk_a": TEXT_A})
    codes = _codes_for(audit, "q-001")
    assert "QUESTION_TERMS_ABSENT_FROM_EVIDENCE" in codes
    assert "EVIDENCE_QUESTION_COVERAGE_LOW" in codes


def test_without_corpus_statistics_coverage_is_reported_as_not_run():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    audit = _run(benchmark, {"chk_a": TEXT_A}, weights=None)
    assert "question_terms_covered_by_evidence" not in audit.checks_run
    assert any(
        c.startswith("question_terms_covered_by_evidence") for c in audit.checks_not_run
    )
    assert audit.questions[0].coverage is None, "coverage must be None, never a substituted 0"
    assert "EVIDENCE_QUESTION_COVERAGE_LOW" not in _codes(audit)


# -- wording heuristics ------------------------------------------------------


def test_under_specified_question_is_flagged():
    benchmark = _benchmark(_question("What is a frame?", ["chk_a"]))
    audit = _run(benchmark, {"chk_a": TEXT_A})
    assert "QUESTION_UNDER_SPECIFIED" in _codes(audit)


def test_dangling_reference_is_flagged():
    benchmark = _benchmark(
        _question("What role does the frame play in the structure described above?", ["chk_a"])
    )
    audit = _run(benchmark, {"chk_a": TEXT_A})
    assert "QUESTION_DANGLING_REFERENCE" in _codes(audit)

    forward = _benchmark(
        _question("Which of the following frame types does a truck use?", ["chk_a"])
    )
    assert "QUESTION_DANGLING_REFERENCE" in _codes(_run(forward, {"chk_a": TEXT_A}))


def test_multiple_interrogatives_and_compound_are_reported():
    benchmark = _benchmark(
        _question(
            "What is a vehicle frame and what role does it play in a ladder frame pickup?",
            ["chk_a"],
        )
    )
    audit = _run(benchmark, {"chk_a": TEXT_A})
    assert "QUESTION_COMPOUND" in _codes(audit)

    two_marks = _benchmark(
        _question("What is a vehicle frame? Which trucks still use one?", ["chk_a"])
    )
    audit_two = _run(two_marks, {"chk_a": TEXT_A})
    assert "QUESTION_MULTIPLE_INTERROGATIVES" in _codes(audit_two)


# -- pairwise and structural -------------------------------------------------


def test_near_duplicate_questions_are_reported():
    benchmark = _benchmark(
        _question("What is a vehicle frame chassis?", ["chk_a"]),
        _question(
            "What is a vehicle frame chassis in trucks?",
            ["chk_b"],
            question_id="q-002",
        ),
    )
    audit = _run(
        benchmark,
        {"chk_a": TEXT_A, "chk_b": TEXT_B},
        meta={
            **_meta(),
            "chk_b": {
                "document_id": "doc_1",
                "content_hash": "hash-1",
                "source_title": "Vehicle frame - Wikipedia",
            },
        },
    )
    dupes = [f for f in audit.findings if f.code == "NEAR_DUPLICATE_QUESTIONS"]
    assert dupes, "expected the near-duplicate pair to be reported"
    assert dupes[0].also_question_id
    assert audit.questions[0].recommended_action is RecommendedAction.RESOLVE_DUPLICATE


def test_contradictory_citation_requirements_are_an_error():
    benchmark = _benchmark(
        _question(
            "What is a vehicle frame?",
            ["chk_a"],
            citation=CitationRequirements(
                require_at_least_one_citation=False,
                must_cite_all_required_evidence=True,
            ),
        )
    )
    audit = _run(benchmark, {"chk_a": TEXT_A})
    assert "CITATION_REQUIREMENTS_CONTRADICTORY" in _codes(audit)


def test_shared_evidence_is_reported_as_an_observation():
    benchmark = _benchmark(
        _question("What is a vehicle frame?", ["chk_a"]),
        _question("Which trucks still use a separate frame?", ["chk_a"], question_id="q-002"),
    )
    audit = _run(benchmark, {"chk_a": TEXT_A})
    shared = [f for f in audit.findings if f.code == "SHARED_REQUIRED_EVIDENCE"]
    assert shared and shared[0].severity is Severity.INFO


def test_systematic_source_title_gap_is_raised_once():
    benchmark = _benchmark(
        _question("What is a vehicle frame?", ["chk_a"]),
        _question("Which trucks still use a separate frame?", ["chk_b"], question_id="q-002"),
    )
    audit = _run(
        benchmark,
        {"chk_a": TEXT_A, "chk_b": TEXT_B},
        meta={
            "chk_a": {"document_id": "doc_1", "content_hash": "hash-1", "source_title": None},
            "chk_b": {"document_id": "doc_1", "content_hash": "hash-1", "source_title": None},
        },
    )
    gaps = [f for f in audit.findings if f.code == "EVIDENCE_SOURCE_TITLE_NOT_IN_CORPUS"]
    assert len(gaps) == 1, "the same corpus-wide gap must not be reported per question"
    assert gaps[0].severity is Severity.WARNING
    assert gaps[0].detail["affected_question_ids"] == ["q-001", "q-002"]


# -- authored labels ---------------------------------------------------------


def _annotation(question_id: str, answer: str, key_points: list[str]) -> GroundTruthAnnotation:
    return GroundTruthAnnotation(
        annotation_id=f"ann_{question_id}",
        kb_id="kb_test",
        benchmark_name="audit-fixture",
        question_id=question_id,
        author="A Human Reviewer",
        expected_answer=answer,
        key_points=key_points,
        evidence=[RequiredEvidence(chunk_id="chk_a", document_id="doc_1", required=True)],
        provenance=[build_author_provenance(author="A Human Reviewer", method="read the corpus")],
    )


def test_unauthored_labels_are_reported_and_the_checks_say_they_did_not_run():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    audit = _run(benchmark, {"chk_a": TEXT_A})
    assert "ANSWER_LABEL_NOT_AUTHORED" in _codes(audit)
    assert audit.questions[0].has_annotation is False
    not_run = " ".join(audit.checks_not_run)
    assert "authored_key_points_supported_by_evidence" in not_run
    assert "annotation_reference_answer_grounded" in not_run
    assert "no annotations exist yet" in not_run


def test_supported_annotation_produces_no_label_findings():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    annotation = _annotation(
        "q-001",
        answer=(
            "A vehicle frame, also known as its chassis, is the main supporting structure "
            "of a motor vehicle to which all other components are attached."
        ),
        key_points=[
            "A vehicle frame is the main supporting structure of a motor vehicle.",
            "A vehicle frame is also known as its chassis.",
        ],
    )
    audit = _run(benchmark, {"chk_a": TEXT_A}, annotations=[annotation])
    assert "ANSWER_LABEL_NOT_AUTHORED" not in _codes(audit)
    assert audit.questions[0].key_points_checked == 2
    assert audit.questions[0].key_points_supported == 2
    assert "authored_key_points_supported_by_evidence" in audit.checks_run


def test_key_point_with_no_support_in_the_cited_evidence_is_an_error():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    annotation = _annotation(
        "q-001",
        answer=TEXT_A,
        key_points=["The frame is welded from carbon fibre in a monocoque tub."],
    )
    audit = _run(benchmark, {"chk_a": TEXT_A}, annotations=[annotation])
    assert "KEY_POINT_UNSUPPORTED_BY_CITED_EVIDENCE" in _codes(audit)
    assert audit.error_count >= 1
    assert audit.ready_to_freeze is False


# -- verdict and read-only behaviour ----------------------------------------


def test_a_complete_clean_benchmark_is_ready_to_freeze():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    annotation = _annotation(
        "q-001",
        answer=(
            "A vehicle frame, also known as its chassis, is the main supporting structure "
            "of a motor vehicle to which all other components are attached."
        ),
        key_points=["A vehicle frame is the main supporting structure of a motor vehicle."],
    )
    audit = _run(benchmark, {"chk_a": TEXT_A}, annotations=[annotation])
    assert audit.error_count == 0
    assert audit.ready_to_freeze is True
    assert "READY" in audit.verdict or "WARNING" in audit.verdict


def test_audit_does_not_modify_the_benchmark():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    before = benchmark.model_dump(mode="json")
    fingerprint_before = benchmark.fingerprint()
    _run(benchmark, {"chk_a": TEXT_A})
    assert benchmark.model_dump(mode="json") == before
    assert benchmark.fingerprint() == fingerprint_before


def test_audit_states_what_it_cannot_decide():
    benchmark = _benchmark(_question("What is a vehicle frame?", ["chk_a"]))
    audit = _run(benchmark, {"chk_a": TEXT_A})
    limitations = " ".join(audit.limitations)
    assert "cannot establish MEANING" in limitations
    assert "triage line" in limitations
    assert "OBSERVATIONS" in limitations
    assert set(audit.checks_run) <= set(
        __import__(
            "app.services.answer_eval.quality_audit", fromlist=["CHECKS"]
        ).CHECKS
    ), "checks_run must only name declared checks"
