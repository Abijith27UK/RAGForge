"""V16 regression tests — a review workflow that cannot approve junk.

The V16 objective is a DEFENSIBLE answer benchmark. The failure modes these
tests pin are the ones that would make it indefensible:

* a placeholder (``TODO``) or a fat-fingered value becoming scored ground truth;
* duplicate key points inflating ``key_point_recall``;
* an approval that silently survives a correction to the reference answer it
  judged;
* the benchmark freezing, or being officially evaluated, while questions are
  still unreviewed or unfrozen;
* the reviewer never seeing what the artifact pins (intended scope, citation
  requirements, abstention expectation).

Hermetic, like the V10 review tests: temp DATA_DIR, in-memory vector store,
hashing embeddings. The benchmark is written to the temp DATA_DIR — the shipped
``benchmarks/answer-quality-automobile-v1.json`` is never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Reuse the V10 fixtures/helpers: the point of V16 is to strengthen that
# workflow, not to build a second one.
from tests.test_v10_review_api import (  # noqa: E402
    ALL_VERDICTS,
    _author,
    _chunk_ids,
    _review,
    _write_benchmark,
    client,  # noqa: F401 - pytest fixture
    env,  # noqa: F401 - pytest fixture
    indexed_kb,  # noqa: F401 - pytest fixture
)


def _packet(client, kb_id, path):
    r = client.get(
        f"/api/knowledge-bases/{kb_id}/answer-benchmark/review-packet",
        params={"benchmark_path": path},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _state(packet, question_id="nav-001"):
    return next(q for q in packet["questions"] if q["question_id"] == question_id)


class TestMalformedAnnotationsAreStoredButNeverApprovable:
    """`validate_annotation` problems are returned, stored, and blocking."""

    def test_placeholder_reference_answer_is_flagged(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(client, indexed_kb["id"], path, expected_answer="TODO")
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["stored"] is True
        assert any("placeholder" in p for p in body["problems"]), body["problems"]
        # Stored, but not approvable: the state and the gate must both say so.
        packet = _packet(client, indexed_kb["id"], path)
        assert _state(packet)["state"] == "in_review"

    def test_placeholder_annotation_blocks_freeze(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path, expected_answer="TBD")
        _review(client, indexed_kb["id"], path)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/freeze",
            json={"benchmark_path": path, "frozen_by": "ada-reviewer"},
        )
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail["reasons"], detail
        assert "BLOCKED" in " ".join(detail["reasons"])

    def test_duplicate_key_points_are_flagged(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(
            client,
            indexed_kb["id"],
            path,
            key_points=[
                "metacentric height must be positive for stability",
                "  Metacentric   height must be positive for stability. ",
            ],
        )
        assert r.status_code == 201, r.text
        problems = r.json()["problems"]
        assert any("duplicates" in p for p in problems), problems

    def test_placeholder_key_point_is_flagged(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(client, indexed_kb["id"], path, key_points=["N/A"])
        assert r.status_code == 201, r.text
        assert any("placeholder" in p for p in r.json()["problems"])

    def test_placeholder_acceptable_element_is_flagged(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(client, indexed_kb["id"], path, acceptable_answer_elements=["..."])
        assert r.status_code == 201, r.text
        assert any("placeholder" in p for p in r.json()["problems"])

    def test_a_stray_word_reference_answer_is_flagged(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(client, indexed_kb["id"], path, expected_answer="yes")
        assert r.status_code == 201, r.text
        assert any("shorter than" in p for p in r.json()["problems"])

    def test_a_terse_but_real_reference_answer_is_NOT_flagged(self, client, indexed_kb, env):  # noqa: F811
        """The guard must not police legitimate short answers."""
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = _author(client, indexed_kb["id"], path, expected_answer="GM must be positive")
        assert r.status_code == 201, r.text
        problems = r.json()["problems"]
        assert not any("shorter than" in p or "placeholder" in p for p in problems), problems

    def test_a_valid_annotation_approves_and_passes_the_gate(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        _review(client, indexed_kb["id"], path)
        packet = _packet(client, indexed_kb["id"], path)
        assert _state(packet)["state"] == "approved"
        assert packet["gate"]["approved"] is True


class TestApprovalIsBoundToTheAnswerItJudged:
    def test_correcting_the_annotation_invalidates_an_approval_that_named_it(
        self, client, indexed_kb, env  # noqa: F811
    ):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        first = _author(client, indexed_kb["id"], path)
        annotation_id = first.json()["annotation"]["annotation_id"]
        _review(client, indexed_kb["id"], path, annotation_id=annotation_id)
        assert _state(_packet(client, indexed_kb["id"], path))["state"] == "approved"

        # The author corrects the reference answer: new annotation, supersedes.
        second = _author(
            client,
            indexed_kb["id"],
            path,
            expected_answer=(
                "The metacentric height GM must be strictly positive for stable "
                "equilibrium."
            ),
            supersedes=annotation_id,
        )
        assert second.status_code == 201, second.text
        new_id = second.json()["annotation"]["annotation_id"]
        assert new_id != annotation_id

        packet = _packet(client, indexed_kb["id"], path)
        state = _state(packet)
        assert state["state"] == "in_review", (
            "a corrected reference answer must not stay approved on the strength of a "
            "review that judged the previous revision"
        )
        assert any("answer changed after the review" in r for r in state["reasons"])
        assert packet["gate"]["approved"] is False

    def test_approval_without_an_annotation_id_is_flagged_in_the_packet(
        self, client, indexed_kb, env  # noqa: F811
    ):
        """The residual hole is DISCLOSED, not hidden."""
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        _review(client, indexed_kb["id"], path)  # no annotation_id
        packet = _packet(client, indexed_kb["id"], path)
        assert _state(packet)["state"] == "approved"
        assert any(
            "does not name the annotation it judged" in w for w in packet["warnings"]
        ), packet["warnings"]

    def test_a_superseding_review_replaces_the_verdict(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        approve = _review(client, indexed_kb["id"], path)
        assert approve.status_code == 201, approve.text
        review_id = approve.json()["review_id"]

        # A later review that supersedes the approval and refuses the question.
        r = _review(
            client,
            indexed_kb["id"],
            path,
            outcome="rejected",
            supersedes=review_id,
            verdicts={"reference_answer": "wrong"},
        )
        assert r.status_code == 201, r.text
        packet = _packet(client, indexed_kb["id"], path)
        assert _state(packet)["state"] == "rejected"

    def test_unknown_dimension_or_verdict_is_refused(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        bad_dim = _review(
            client, indexed_kb["id"], path, verdicts={"vibes": "ok"}
        )
        assert bad_dim.status_code == 400, bad_dim.text
        bad_verdict = _review(
            client, indexed_kb["id"], path, verdicts={"reference_answer": "probably"}
        )
        assert bad_verdict.status_code == 400, bad_verdict.text
        # Nothing was appended by a refused review.
        packet = _packet(client, indexed_kb["id"], path)
        assert _state(packet)["state"] == "in_review"
        assert _state(packet)["review_count"] == 0


class TestScopeIsVisibleToTheReviewer:
    def test_packet_carries_intended_scope_and_pins(self, client, indexed_kb, env):  # noqa: F811
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        # Add pinned provenance + an abstention expectation to the artifact.
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["questions"][0]["provenance"] = (
            "Chunk chk_x (naval.md, section Hydrostatics and Stability): quoted passage."
        )
        raw["questions"][0]["abstention_required"] = True
        raw["questions"][0]["expected_grounding_state"] = "ANSWERED"
        raw["questions"][0]["numerical_tolerance"] = 0.05
        Path(path).write_text(json.dumps(raw, indent=2), encoding="utf-8")

        packet = _packet(client, indexed_kb["id"], path)
        q = _state(packet)
        assert "Chunk chk_x" in q["artifact_provenance"]
        assert q["abstention_required"] is True
        assert q["expected_grounding_state"] == "ANSWERED"
        assert q["numerical_tolerance"] == 0.05
        assert q["citation_requirements"]["must_cite_all_required_evidence"] is True


class TestLifecycleGates:
    def test_official_evaluation_is_refused_without_a_frozen_version(
        self, client, indexed_kb, env  # noqa: F811
    ):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        _review(client, indexed_kb["id"], path)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-evaluation/runs",
            json={"benchmark_path": path, "evaluator": "reference", "limit": 1},
        )
        # An unreviewed (draft) benchmark may not be scored by the reference
        # evaluator: refused with the reason, never scored as if it were ground
        # truth.
        assert r.status_code == 400, r.text
        assert "human-reviewed" in r.text.lower() and "draft" in r.text.lower(), r.text

    def test_freeze_produces_an_immutable_version_and_refuses_edits(
        self, client, indexed_kb, env  # noqa: F811
    ):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        _author(client, indexed_kb["id"], path)
        _review(client, indexed_kb["id"], path)
        r = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/freeze",
            json={"benchmark_path": path, "frozen_by": "ada-reviewer"},
        )
        assert r.status_code == 200, r.text
        version = r.json()
        assert version["version"] == 1

        verify = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/versions/"
            f"{version['version_id']}/verify"
        )
        assert verify.status_code == 200, verify.text
        assert verify.json()["intact"] is True

        # Unreviewed work cannot be frozen: a second question with no review blocks.
        other = _write_benchmark(
            env[1],
            indexed_kb["id"],
            _chunk_ids(indexed_kb["id"])[:1],
            name="scratch-v16-unreviewed",
        )
        refused = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/freeze",
            json={"benchmark_path": other, "frozen_by": "ada-reviewer"},
        )
        assert refused.status_code == 409
        assert refused.json()["detail"]["reasons"]


class TestDraftAssistanceStaysASuggestion:
    def test_a_generated_draft_is_not_ground_truth_and_cannot_approve(
        self, client, indexed_kb, env  # noqa: F811
    ):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        gen = client.post(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/draft-suggestions",
            json={"benchmark_path": path, "question_ids": ["nav-001"]},
        )
        assert gen.status_code == 200, gen.text
        drafts = gen.json()["suggestions"]
        assert len(drafts) == 1
        draft = drafts[0]
        assert draft["is_draft"] is True
        assert draft["requires_human_review"] is True
        assert "NOT HUMAN GROUND TRUTH" in draft["label"]

        # The draft is in its OWN store: the annotation history stays empty and
        # an approving review still has nothing to approve.
        history = client.get(
            f"/api/knowledge-bases/{indexed_kb['id']}/answer-benchmark/questions/nav-001/ground-truth",
            params={"benchmark_path": path},
        )
        assert history.status_code == 200
        assert history.json() == []

        _review(client, indexed_kb["id"], path)
        packet = _packet(client, indexed_kb["id"], path)
        assert _state(packet)["state"] == "in_review"
        assert _state(packet)["has_annotation"] is False
        assert packet["gate"]["approved"] is False
