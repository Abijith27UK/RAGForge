"""V3 Phase A tests: benchmark lifecycle, versioning, and freeze protection.

API-level (TestClient, temp SQLite) + hermetic evaluator tests with the
in-memory cosine vector store (same pattern as test_retrieval_integration.py).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.schemas.models import (  # noqa: E402
    BenchmarkStatus,
    BenchmarkVersion,
    EvaluationQuestion,
    EvaluationRunConfig,
    KnowledgeBase,
    QuestionStatus,
)
from app.services.evaluation.evaluator import EvaluationError, Evaluator  # noqa: E402
from app.utils.ids import new_id  # noqa: E402

from tests.test_retrieval_integration import (  # noqa: E402
    InMemoryVectorStore,
    TEXTS,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.api.deps as deps
    import app.config as config

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    config.get_settings.cache_clear()
    deps.get_repo.cache_clear()


def _create_kb(client) -> str:
    r = client.post(
        "/api/knowledge-bases",
        json={"name": "Bench KB", "domain": "Automobile Engineering",
              "purpose": "testing", "target_audience": "students"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _add_question(client, kb_id: str, text: str = "What drives the wheels?") -> dict:
    r = client.post(
        f"/api/knowledge-bases/{kb_id}/evaluation-questions",
        json={"question": text, "expected_keywords": ["engine"], "notes": "authored by test"},
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Question lifecycle
# ---------------------------------------------------------------------------

def test_question_defaults_to_draft_with_lifecycle_metadata(client):
    kb = _create_kb(client)
    q = _add_question(client, kb)
    assert q["status"] == "DRAFT"
    assert q["revision"] == 1
    assert q["supersedes"] is None
    assert q["author"] == ""
    assert q["created_at"]


def test_draft_to_review_to_approved_requires_reviewer(client):
    kb = _create_kb(client)
    q = _add_question(client, kb)
    qid = q["id"]
    r = client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}/status",
                    json={"status": "REVIEW"})
    assert r.status_code == 200 and r.json()["status"] == "REVIEW"
    # APPROVED without reviewer must be rejected
    r = client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}/status",
                    json={"status": "APPROVED", "reviewer": ""})
    assert r.status_code == 422
    r = client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}/status",
                    json={"status": "APPROVED", "reviewer": "alice"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "APPROVED" and body["reviewer"] == "alice" and body["reviewed_at"]


def test_approved_question_edit_creates_new_draft_revision(client):
    kb = _create_kb(client)
    q = _add_question(client, kb)
    qid = q["id"]
    client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}/status",
                json={"status": "APPROVED", "reviewer": "alice"})
    r = client.patch(
        f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}",
        json={"question": "Revised question text", "notes": "sharper wording"},
    )
    assert r.status_code == 200
    rev = r.json()
    assert rev["id"] != qid, "editing APPROVED must create a new question"
    assert rev["status"] == "DRAFT" and rev["supersedes"] == qid and rev["revision"] == 2
    # The approved ancestor is untouched:
    r = client.get(f"/api/knowledge-bases/{kb}/evaluation-questions")
    ancestor = next(x for x in r.json() if x["id"] == qid)
    assert ancestor["question"] == "What drives the wheels?" and ancestor["status"] == "APPROVED"


def test_approved_cannot_go_back_and_frozen_is_immutable(client):
    kb = _create_kb(client)
    q = _add_question(client, kb)
    qid = q["id"]
    client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}/status",
                json={"status": "APPROVED", "reviewer": "alice"})
    r = client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}/status",
                    json={"status": "DRAFT", "reviewer": "alice"})
    assert r.status_code == 409, "APPROVED -> DRAFT must be illegal"
    r = client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}/status",
                    json={"status": "FROZEN", "reviewer": "bob"})
    assert r.status_code == 200 and r.json()["status"] == "FROZEN"
    r = client.patch(f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}",
                     json={"question": "try to change frozen"})
    assert r.status_code == 409, "FROZEN questions must be immutable"


def test_illegal_transition_rejected(client):
    kb = _create_kb(client)
    q = _add_question(client, kb)
    r = client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{q['id']}/status",
                    json={"status": "FROZEN", "reviewer": "x"})
    assert r.status_code == 200  # DRAFT -> FROZEN is legal (explicit freeze)
    r = client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{q['id']}/status",
                    json={"status": "REVIEW", "reviewer": "x"})
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# Benchmark versions + freeze protection
# ---------------------------------------------------------------------------

def _freeze_question(client, kb: str, qid: str, reviewer: str = "alice") -> None:
    r = client.post(f"/api/knowledge-bases/{kb}/evaluation-questions/{qid}/status",
                    json={"status": "FROZEN", "reviewer": reviewer})
    assert r.status_code == 200


def test_version_snapshot_and_freeze_flow(client):
    kb = _create_kb(client)
    q1 = _add_question(client, kb, "Q1 about engines?")
    q2 = _add_question(client, kb, "Q2 about braking?")
    _freeze_question(client, kb, q1["id"])
    _freeze_question(client, kb, q2["id"])

    r = client.post(f"/api/knowledge-bases/{kb}/benchmark-versions",
                    json={"version": "auto-v1", "label": "first", "created_by": "alice"})
    assert r.status_code == 201, r.text
    bv = r.json()
    # All questions FROZEN -> version can be born FROZEN
    assert bv["status"] == "FROZEN" and bv["frozen_at"] is not None
    assert len(bv["questions_snapshot"]) == 2
    assert bv["questions_snapshot"][0]["question"] == "Q1 about engines?"

    # FROZEN version is immutable: cannot be deleted or re-frozen weirdly
    r = client.delete(f"/api/knowledge-bases/{kb}/benchmark-versions/{bv['id']}")
    assert r.status_code == 409, "frozen versions must refuse deletion"
    r = client.post(f"/api/knowledge-bases/{kb}/benchmark-versions/{bv['id']}/freeze")
    assert r.status_code == 200  # idempotent


def test_draft_snapshot_cannot_be_frozen(client):
    kb = _create_kb(client)
    q = _add_question(client, kb)  # stays DRAFT
    r = client.post(f"/api/knowledge-bases/{kb}/benchmark-versions",
                    json={"version": "drafty", "created_by": "alice"})
    assert r.status_code == 201
    bv = r.json()
    assert bv["status"] == "DRAFT"
    r = client.post(f"/api/knowledge-bases/{kb}/benchmark-versions/{bv['id']}/freeze")
    assert r.status_code == 409, "snapshot containing DRAFT questions must refuse to freeze"
    # Unfrozen (DRAFT) versions CAN be deleted
    r = client.delete(f"/api/knowledge-bases/{kb}/benchmark-versions/{bv['id']}")
    assert r.status_code == 204


def test_unknown_question_id_in_version_creation_rejected(client):
    kb = _create_kb(client)
    r = client.post(f"/api/knowledge-bases/{kb}/benchmark-versions",
                    json={"version": "v", "question_ids": ["eq_nonexistent"]})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Frozen-benchmark evaluation (hermetic evaluator)
# ---------------------------------------------------------------------------

from app.utils.ids import new_id  # noqa: E402


class _FakeRetriever:
    """Minimal dense retriever stand-in: one deterministic result per query."""

    backend = "fake-dense"

    def retrieve(self, kb_id: str, query: str, top_k: int = 5):
        from app.schemas.models import RetrievalResponse, RetrievalResult
        ql = query.lower()
        text = TEXTS["docA"][2] if ("braking" in ql or "energy" in ql) else TEXTS["docA"][0]
        return RetrievalResponse(
            query=query, top_k=top_k,
            results=[RetrievalResult(chunk_id="chk_x", document_id="doc1", text=text,
                                     score=0.9, provenance={})],
            embedding_model="fake",
        )


def _seed_frozen_version(repo: Repository, kb_id: str, status=BenchmarkStatus.FROZEN) -> BenchmarkVersion:
    question = EvaluationQuestion(
        id=new_id("eq"), kb_id=kb_id,
        question="How does the vehicle recover braking energy?",
        expected_chunk_ids=["chk_x"], generated_by="manual",
        status=QuestionStatus.FROZEN, author="tester",
    )
    repo.create_evaluation_question(question)
    bv = BenchmarkVersion(
        id=new_id("bv"), kb_id=kb_id, version="unit-frozen-v1", label="test",
        status=status,
        question_ids=[question.id],
        questions_snapshot=[question.model_dump(mode="json")],
        created_by="tester",
        frozen_at=datetime.now(timezone.utc) if status == BenchmarkStatus.FROZEN else None,
    )
    repo.create_benchmark_version(bv)
    return bv


def test_frozen_benchmark_evaluation_scores_snapshot(repo):
    kb_id = "kb_bench"
    repo.create_kb(
        KnowledgeBase(id=kb_id, name="t", domain="d", purpose="p", target_audience="a")
    )
    bv = _seed_frozen_version(repo, kb_id)
    run = Evaluator(_FakeRetriever(), repo).run_evaluation(
        kb_id, EvaluationRunConfig(top_k=3, benchmark_version=bv.id)
    )
    assert run.benchmark_version == bv.id
    assert run.aggregate.recall_at_k == 1.0, "snapshot GT chk_x is retrieved by the fake"
    assert run.question_statuses == {run.per_question[0].question_id: "FROZEN"}


def test_evaluation_refuses_unfrozen_benchmark_version(repo):
    kb_id = "kb_bench2"
    repo.create_kb(
        KnowledgeBase(id=kb_id, name="t", domain="d", purpose="p", target_audience="a")
    )
    bv = _seed_frozen_version(repo, kb_id, status=BenchmarkStatus.DRAFT)
    with pytest.raises(EvaluationError) as exc:
        Evaluator(_FakeRetriever(), repo).run_evaluation(
            kb_id, EvaluationRunConfig(top_k=3, benchmark_version=bv.id)
        )
    assert "not FROZEN" in str(exc.value)


def test_evaluation_refuses_unknown_benchmark_version(repo):
    kb_id = "kb_bench3"
    repo.create_kb(
        KnowledgeBase(id=kb_id, name="t", domain="d", purpose="p", target_audience="a")
    )
    with pytest.raises(EvaluationError) as exc:
        Evaluator(_FakeRetriever(), repo).run_evaluation(
            kb_id, EvaluationRunConfig(top_k=3, benchmark_version="bv_missing")
        )
    assert "not found" in str(exc.value)
