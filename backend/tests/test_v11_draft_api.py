"""V11 draft-assist / quality-audit API tests.

The integrity tests are the point of this file: generating drafts for a whole
benchmark must leave ground truth EMPTY, must leave every question PENDING, and
must leave the approval gate closed. If a future change ever let a machine draft
count as a human label, these fail.

Hermetic: temporary DATA_DIR, in-memory vector store, hashing embeddings, mock
LLM, and a real document indexed through the real upload → chunk → embed → index
path. The benchmark is written to the temp DATA_DIR — never to the shipped
`benchmarks/answer-quality-automobile-v1.json`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# The hermetic environment and the scratch-benchmark writer are identical to the
# V10 review API tests, so they are reused rather than re-implemented: two copies
# of a fixture this fiddly would drift.
from tests.test_v10_review_api import (  # noqa: E402,F401
    _chunk_ids,
    _write_benchmark,
    client,
    env,
    indexed_kb,
)


DRAFT_URL = "/api/knowledge-bases/{kb}/answer-benchmark/questions/nav-001/draft-suggestions"
BATCH_URL = "/api/knowledge-bases/{kb}/answer-benchmark/draft-suggestions"
AUDIT_URL = "/api/knowledge-bases/{kb}/answer-benchmark/quality-audit"


def _generate_one(client, kb_id, path, question_id="nav-001", **overrides):
    body = {"benchmark_path": path}
    body.update(overrides)
    return client.post(
        DRAFT_URL.format(kb=kb_id).replace("nav-001", question_id), json=body
    )


class TestDraftEndpoint:
    def test_returns_a_labelled_draft_not_ground_truth(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        r = _generate_one(client, kb_id, path)
        assert r.status_code == 201, r.text
        draft = r.json()
        assert draft["is_draft"] is True
        assert draft["requires_human_review"] is True
        assert "NOT HUMAN GROUND TRUTH" in draft["label"]
        assert draft["method"].startswith("extractive-")
        assert "no LLM" in draft["generator"]
        # never an annotation's identity
        assert "author" not in draft and "annotation_id" not in draft
        assert draft["expected_answer_draft"]

    def test_draft_key_points_are_verbatim_in_the_cited_chunk(
        self, client, indexed_kb, env
    ):
        import app.api.deps as deps

        kb_id = indexed_kb["id"]
        chunks = _chunk_ids(kb_id)[:1]
        path = _write_benchmark(env[1], kb_id, chunks)
        draft = _generate_one(client, kb_id, path).json()
        chunk = deps.get_repo().get_chunks_by_ids(kb_id, chunks)[0]
        for point in draft["key_points_draft"]:
            assert point["text"] in chunk.text, (
                f"key point is not verbatim corpus text: {point['text']!r}"
            )
            assert point["source_chunk_id"] == chunk.id

    def test_history_is_append_only(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        first = _generate_one(client, kb_id, path).json()
        second = _generate_one(client, kb_id, path).json()
        assert first["suggestion_id"] != second["suggestion_id"]
        history = client.get(
            DRAFT_URL.format(kb=kb_id), params={"benchmark_path": path}
        )
        assert history.status_code == 200, history.text
        ids = [d["suggestion_id"] for d in history.json()]
        assert ids == [first["suggestion_id"], second["suggestion_id"]]

    def test_a_draft_can_be_generated_without_storing_it(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        assert _generate_one(client, kb_id, path, persist=False).status_code == 201
        history = client.get(DRAFT_URL.format(kb=kb_id), params={"benchmark_path": path})
        assert history.json() == []

    def test_unknown_question_is_refused(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        r = _generate_one(client, kb_id, path, question_id="nav-999")
        assert r.status_code in (400, 404), r.text

    def test_unknown_knowledge_base_is_404(self, client, indexed_kb, env):
        path = _write_benchmark(env[1], indexed_kb["id"], _chunk_ids(indexed_kb["id"])[:1])
        r = client.post(f"/api/knowledge-bases/kb_missing/answer-benchmark/draft-suggestions", json={"benchmark_path": path})
        assert r.status_code == 404


class TestDraftBatch:
    def test_batch_reports_what_it_produced(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        r = client.post(BATCH_URL.format(kb=kb_id), json={"benchmark_path": path})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["suggestions"]) == 1
        summary = body["summary"]
        assert summary["total"] == 1
        assert sum(summary["by_status"].values()) == 1
        assert "not ground truth" in summary["note"]
        assert "NOT human ground truth" in body["notice"]

    def test_batch_can_target_specific_questions(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        r = client.post(
            BATCH_URL.format(kb=kb_id),
            json={"benchmark_path": path, "question_ids": ["nav-999"]},
        )
        assert r.status_code == 400
        assert "nav-999" in r.text


class TestDraftsNeverBecomeGroundTruth:
    """The core integrity guarantee, checked through the public API."""

    def test_generating_drafts_leaves_ground_truth_empty(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        assert client.post(BATCH_URL.format(kb=kb_id), json={"benchmark_path": path}).status_code == 200

        history = client.get(
            f"/api/knowledge-bases/{kb_id}/answer-benchmark/questions/nav-001/ground-truth",
            params={"benchmark_path": path},
        )
        assert history.status_code == 200
        assert history.json() == [], "a draft must never be stored as an annotation"

        packet = client.get(
            f"/api/knowledge-bases/{kb_id}/answer-benchmark/review-packet",
            params={"benchmark_path": path},
        ).json()
        question = packet["questions"][0]
        assert question["has_annotation"] is False
        assert question["state"] == "pending"
        assert question["expected_answer"] == ""
        assert packet["completeness"]["approved"] == 0
        assert packet["completeness"]["review_count"] == 0
        assert packet["completeness"]["authored_reference_answer_count"] == 0
        assert packet["versions"] == []

    def test_drafts_are_stored_in_their_own_table(self, client, indexed_kb, env):
        import app.api.deps as deps

        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        client.post(BATCH_URL.format(kb=kb_id), json={"benchmark_path": path})
        repo = deps.get_repo()
        assert len(repo.list_draft_suggestions(kb_id)) == 1
        assert repo.list_ground_truth_annotations(kb_id) == []

    def test_the_freeze_gate_stays_closed_after_drafting(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        client.post(BATCH_URL.format(kb=kb_id), json={"benchmark_path": path})

        r = client.post(
            f"/api/knowledge-bases/{kb_id}/answer-benchmark/freeze",
            json={"benchmark_path": path, "frozen_by": "ada-reviewer"},
        )
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail["reasons"], "the refusal must say why"
        assert "approved" in detail["message"]
        assert detail["completeness"]["approved"] == 0
        assert client.get(
            f"/api/knowledge-bases/{kb_id}/answer-benchmark/versions"
        ).json() == []

    def test_a_human_can_still_author_after_a_draft_exists(self, client, indexed_kb, env):
        """The draft must not block or pre-empt human authorship."""
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        draft = _generate_one(client, kb_id, path).json()

        r = client.post(
            f"/api/knowledge-bases/{kb_id}/answer-benchmark/questions/nav-001/ground-truth",
            json={
                "benchmark_path": path,
                "author": "ada-reviewer",
                "expected_answer": "The metacentric height GM must be positive for stable equilibrium.",
                "key_points": ["metacentric height GM must be positive"],
                "evidence_chunk_ids": [draft["key_points_draft"][0]["source_chunk_id"]],
                "note": "started from the draft, corrected",
            },
        )
        assert r.status_code == 201, r.text
        annotation = r.json()["annotation"]
        assert r.json()["problems"] == []
        assert annotation["author"] == "ada-reviewer"
        assert annotation["annotation_id"] != draft["suggestion_id"]


class TestQualityAuditEndpoint:
    def test_audit_reports_findings_for_an_unauthored_benchmark(
        self, client, indexed_kb, env
    ):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        r = client.get(AUDIT_URL.format(kb=kb_id), params={"benchmark_path": path})
        assert r.status_code == 200, r.text
        audit = r.json()
        assert audit["kb_id"] == kb_id
        assert audit["ready_to_freeze"] is False
        assert audit["questions"][0]["question_id"] == "nav-001"
        codes = {f["code"] for f in audit["findings"]}
        assert "ANSWER_LABEL_NOT_AUTHORED" in codes
        # the annotation-dependent checks must declare that they did not run
        assert any(
            c.startswith("authored_key_points_supported_by_evidence")
            for c in audit["checks_not_run"]
        )
        assert audit["checks_run"], "the audit must name the checks it did run"
        assert audit["limitations"]

    def test_audit_catches_evidence_that_left_the_corpus(self, client, indexed_kb, env):
        import json

        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        # Point the benchmark at a chunk id that does not exist in the corpus.
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["questions"][0]["required_evidence"][0]["chunk_id"] = "chk_does_not_exist"
        broken = Path(path).with_name("broken.json")
        broken.write_text(json.dumps(payload), encoding="utf-8")

        audit = client.get(
            AUDIT_URL.format(kb=kb_id), params={"benchmark_path": str(broken)}
        ).json()
        codes = {f["code"] for f in audit["findings"]}
        assert "EVIDENCE_MISSING_FROM_CORPUS" in codes
        assert "ANSWERABILITY_UNSUPPORTED" in codes
        assert audit["error_count"] >= 2
        assert audit["ready_to_freeze"] is False

    def test_audit_does_not_change_anything(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _chunk_ids(kb_id)[:1])
        before = Path(path).read_bytes()
        client.get(AUDIT_URL.format(kb=kb_id), params={"benchmark_path": path})
        assert Path(path).read_bytes() == before

        packet = client.get(
            f"/api/knowledge-bases/{kb_id}/answer-benchmark/review-packet",
            params={"benchmark_path": path},
        ).json()
        assert packet["questions"][0]["state"] == "pending"
