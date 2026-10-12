"""V11 draft-assist tests: the drafts must be verbatim, honest, and inert.

The load-bearing property is that a draft cannot contain a fact that is not in
the cited corpus text, and that a draft is structurally incapable of becoming
ground truth. Both are tested here rather than assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.benchmark import (  # noqa: E402
    AnswerBenchmarkQuestion,
    RequiredEvidence,
)
from app.services.answer_eval.drafting import (  # noqa: E402
    DRAFT_LABEL,
    MIN_QUESTION_COVERAGE,
    DraftStatus,
    DraftingError,
    _assert_answer_composition,
    _assert_verbatim,
    _norm,
    build_draft_suggestion,
    summarise_drafts,
)
from app.services.answer_eval.relevance import TermWeights  # noqa: E402

# Real shape of the corpus text: parsed Wikipedia content carries layout breaks
# and punctuation on its own line. Kept in the fixture because that formatting
# is what broke the verbatim comparison when it was first written.
ENGINE_CHUNK = (
    "An\nengine\nor\nmotor\nis a\nmachine\ndesigned to convert one or more forms of\n"
    "energy\ninto\nmechanical energy\n."
)
BRAKE_CHUNK = (
    "A brake is a mechanical device that inhibits motion by absorbing energy from a "
    "moving system. It is used for stopping or slowing a vehicle."
)
WEIGHTS = TermWeights(
    n_docs=812,
    df={"engine": 40, "convert": 12, "energy": 90, "mechanical": 30, "brake": 55},
    source="test-fixture",
)


def _question(
    text: str,
    chunk_ids: list[str],
    *,
    question_id: str = "q-001",
    provenance: str = "",
) -> AnswerBenchmarkQuestion:
    return AnswerBenchmarkQuestion(
        question_id=question_id,
        question=text,
        subdomain="engine",
        answerability="answerable",
        required_evidence=[
            RequiredEvidence(chunk_id=cid, document_id="doc_1", required=True)
            for cid in chunk_ids
        ],
        provenance=provenance,
    )


def _draft(question, texts, *, weights=WEIGHTS, **kwargs):
    return build_draft_suggestion(
        kb_id="kb_test",
        benchmark_name="demo-benchmark",
        question=question,
        evidence_texts=texts,
        weights=weights,
        **kwargs,
    )


# -- the central guarantee ---------------------------------------------------


def test_every_draft_key_point_is_verbatim_in_the_chunk_it_cites():
    q = _question("What is an engine and what does it convert energy into?", ["chk_a"])
    draft = _draft(q, {"chk_a": ENGINE_CHUNK})

    assert draft.key_points_draft, "expected at least one key point"
    for point in draft.key_points_draft:
        assert _norm(point.text) in _norm(ENGINE_CHUNK)
    assert draft.expected_answer_draft == " ".join(
        p.text for p in draft.key_points_draft
    )


def test_verbatim_guard_refuses_invented_text():
    """A sentence that is not in the cited chunk must be refused outright."""
    with pytest.raises(DraftingError, match="not verbatim"):
        _assert_verbatim(
            "An engine converts fuel into money.", ENGINE_CHUNK, "key point"
        )


def test_verbatim_guard_normalises_layout_but_not_wording():
    """Layout artefacts compare equal; a real word difference never does."""
    # Same text as the corpus up to whitespace and the mid-line full stop.
    _assert_verbatim(
        "An engine or motor is a machine designed to convert one or more forms of "
        "energy into mechanical energy.",
        ENGINE_CHUNK,
        "key point",
    )
    with pytest.raises(DraftingError):
        _assert_verbatim(
            "An engine or motor is a machine designed to convert one or more forms of "
            "energy into electrical energy.",
            ENGINE_CHUNK,
            "key point",
        )


def test_answer_composition_guard_rejects_added_text():
    """The assembled answer may be the cited sentences and nothing else."""
    q = _question("What is an engine?", ["chk_a"])
    draft = _draft(q, {"chk_a": ENGINE_CHUNK})
    selected = [
        type(
            "C",
            (),
            {"chunk_id": p.source_chunk_id, "text": p.text},
        )()
        for p in draft.key_points_draft
    ]
    _assert_answer_composition(draft.expected_answer_draft, selected, {"chk_a": ENGINE_CHUNK})
    with pytest.raises(DraftingError, match="concatenation"):
        _assert_answer_composition(
            draft.expected_answer_draft + " Engines are also cheap.",
            selected,
            {"chk_a": ENGINE_CHUNK},
        )


# -- honesty about what the evidence supports --------------------------------


def test_unrelated_evidence_yields_no_draft_and_says_so():
    q = _question(
        "What is a brake and by what means does it usually slow a vehicle?",
        ["chk_b"],
    )
    draft = _draft(
        q,
        {"chk_b": "Marine chronometers were rated by longitude trials in the 19th century."},
    )
    assert draft.status is DraftStatus.INSUFFICIENT_EVIDENCE
    assert draft.expected_answer_draft == ""
    assert draft.key_points_draft == []
    assert any("Do not infer an answer" in c for c in draft.caveats)


def test_missing_chunk_is_never_drafted_from_nothing():
    q = _question("What is an engine?", ["chk_gone"])
    draft = _draft(q, {})
    assert draft.status is DraftStatus.INSUFFICIENT_EVIDENCE
    assert draft.expected_answer_draft == ""
    assert draft.evidence[0].found is False


def test_low_weighted_coverage_marks_the_draft_partial():
    """Evidence that only partly covers the question must not be called complete.

    `turbocharger` and `wastegate` really are in the corpus (they have document
    frequencies) but not in the chunk pinned to this question, which is the
    realistic PARTIAL case: the question is answerable in the KB, just not from
    this evidence.
    """
    partial_weights = TermWeights(
        n_docs=812,
        df={"engine": 40, "convert": 12, "turbocharger": 6, "wastegate": 5},
        source="test-fixture",
    )
    q = _question(
        "What is an engine and how does a turbocharger wastegate convert torque?",
        ["chk_a"],
    )
    draft = _draft(q, {"chk_a": ENGINE_CHUNK}, weights=partial_weights)
    assert draft.coverage_measured is True
    assert draft.status is DraftStatus.PARTIAL
    assert draft.coverage < MIN_QUESTION_COVERAGE
    assert draft.expected_answer_draft, "a partial draft must still offer what it can"
    assert {"turbocharger", "wastegate"} <= set(draft.uncovered_terms)
    assert any("phrasing" in c for c in draft.caveats)


def test_uncovered_terms_are_ordered_by_corpus_significance():
    q = _question("Which engine convert mechanical torque?", ["chk_a"])
    draft = _draft(q, {"chk_a": ENGINE_CHUNK})
    if len(draft.uncovered_terms) > 1:
        weights = [WEIGHTS.idf(t) for t in draft.uncovered_terms]
        assert weights == sorted(weights, reverse=True)


def test_without_corpus_statistics_coverage_is_reported_as_unavailable():
    q = _question("What is an engine and what does it convert energy into?", ["chk_a"])
    draft = _draft(q, {"chk_a": ENGINE_CHUNK}, weights=None)
    assert draft.coverage_measured is False
    assert "unavailable" in draft.coverage_method
    assert any("no corpus term statistics" in c for c in draft.caveats)


# -- the draft is not ground truth -------------------------------------------


def test_draft_is_labelled_and_flagged_everywhere():
    q = _question("What is an engine?", ["chk_a"])
    draft = _draft(q, {"chk_a": ENGINE_CHUNK})
    assert draft.is_draft is True
    assert draft.requires_human_review is True
    assert draft.label == DRAFT_LABEL
    assert "NOT HUMAN GROUND TRUTH" in draft.label
    assert draft.generator and "extractive" in draft.generator
    assert any("must never be approved as-is" in c for c in draft.caveats)


def test_draft_model_carries_no_human_authorship_fields():
    """Structural proof a draft cannot be mistaken for an authored label."""
    from app.services.answer_eval.ground_truth import GroundTruthAnnotation

    draft_fields = set(__import__("app.services.answer_eval.drafting", fromlist=["x"]).DraftSuggestion.model_fields)
    assert "author" not in draft_fields
    assert "annotation_id" not in draft_fields
    # The real annotation type has exactly the fields a draft lacks.
    annotation_fields = set(GroundTruthAnnotation.model_fields)
    assert {"author", "annotation_id"} <= annotation_fields


# -- determinism and the benchmark-quote sanity signal -----------------------


def test_generation_is_deterministic():
    q = _question("What is an engine?", ["chk_a"])
    first = _draft(q, {"chk_a": ENGINE_CHUNK})
    second = _draft(q, {"chk_a": ENGINE_CHUNK})
    assert first.suggestion_id != second.suggestion_id  # ids are unique…
    assert first.expected_answer_draft == second.expected_answer_draft  # …content is not
    assert [p.text for p in first.key_points_draft] == [
        p.text for p in second.key_points_draft
    ]
    assert [p.source_chunk_id for p in first.key_points_draft] == [
        p.source_chunk_id for p in second.key_points_draft
    ]
    assert first.coverage == second.coverage


def test_agreement_with_the_benchmarks_own_recorded_sentence():
    quoted = (
        "Chunk chk_a (Engine - Wikipedia, section Introduction): 'An engine or motor is "
        "a machine designed to convert one or more forms of energy into mechanical "
        "energy.' Manually read and selected 2026-09-15."
    )
    q = _question(
        "What is an engine and what does it convert energy into?",
        ["chk_a"],
        provenance=quoted,
    )
    draft = _draft(q, {"chk_a": ENGINE_CHUNK})
    assert draft.benchmark_quote.startswith("An engine or motor is a machine")
    assert draft.agrees_with_benchmark_quote is True


def test_disagreement_with_the_recorded_sentence_is_flagged():
    quoted = (
        "Chunk chk_a (Engine - Wikipedia): 'A piston is a component of reciprocating "
        "engines.' Manually read and selected 2026-09-15."
    )
    q = _question("What is an engine?", ["chk_a"], provenance=quoted)
    draft = _draft(q, {"chk_a": ENGINE_CHUNK})
    assert draft.agrees_with_benchmark_quote is False
    assert any("differs from the sentence recorded" in c for c in draft.caveats)


def test_no_recorded_sentence_means_no_verdict():
    q = _question("What is an engine?", ["chk_a"])
    draft = _draft(q, {"chk_a": ENGINE_CHUNK})
    assert draft.agrees_with_benchmark_quote is None


def test_every_required_chunk_that_can_contribute_does():
    q = _question(
        "What is an engine and how does a brake work?", ["chk_a", "chk_b"]
    )
    draft = _draft(q, {"chk_a": ENGINE_CHUNK, "chk_b": BRAKE_CHUNK})
    cited = {p.source_chunk_id for p in draft.key_points_draft}
    assert cited == {"chk_a", "chk_b"}


def test_summary_counts_only_what_was_produced():
    ok = _draft(_question("What is an engine?", ["chk_a"]), {"chk_a": ENGINE_CHUNK})
    bad = _draft(
        _question("What is a brake?", ["chk_b"], question_id="q-002"),
        {"chk_b": "Unrelated text about chronometers and longitude trials."},
    )
    summary = summarise_drafts([ok, bad])
    assert summary["total"] == 2
    assert summary["by_status"][DraftStatus.DRAFT_READY.value] == 1
    assert summary["by_status"][DraftStatus.INSUFFICIENT_EVIDENCE.value] == 1
    assert summary["with_draft"] == 1
    assert summary["key_points_offered"] == len(ok.key_points_draft)
    assert "not ground truth" in summary["note"]
