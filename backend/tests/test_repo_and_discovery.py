"""Repository persistence + discovery provider tests (no live network)."""
from __future__ import annotations

from app.repositories.sqlite_repo import Repository
from app.schemas.models import (
    Chunk,
    Document,
    EvaluationQuestion,
    KnowledgeBase,
    Source,
    SourceDecision,
    SourceType,
)
from app.services.source_discovery.discovery import UserURLProvider
from app.utils.ids import new_id


def make_kb(repo: Repository) -> KnowledgeBase:
    kb = KnowledgeBase(
        id=new_id("kb"), name="T", domain="Automobile Engineering",
        purpose="edu", target_audience="students",
    )
    repo.create_kb(kb)
    return kb


def test_kb_roundtrip(repo: Repository):
    kb = make_kb(repo)
    got = repo.get_kb(kb.id)
    assert got is not None
    assert got.domain == "Automobile Engineering"
    assert repo.list_kbs()[0].id == kb.id
    repo.delete_kb(kb.id)
    assert repo.get_kb(kb.id) is None


def test_source_roundtrip_and_url_lookup(repo: Repository):
    kb = make_kb(repo)
    src = Source(id=new_id("src"), kb_id=kb.id, url="https://a.example.com/x")
    repo.create_source(src)
    assert repo.find_source_by_url(kb.id, "https://a.example.com/x") is not None
    src.decision = SourceDecision.ACCEPT
    repo.update_source(src)
    assert repo.list_sources(kb.id)[0].decision == SourceDecision.ACCEPT


def test_document_dedup_by_hash(repo: Repository):
    kb = make_kb(repo)
    src = Source(id=new_id("src"), kb_id=kb.id, url="https://a.example.com/x")
    repo.create_source(src)
    doc = Document(
        id=new_id("doc"), kb_id=kb.id, source_id=src.id, url=src.url,
        source_type=SourceType.TEXT, content_hash="deadbeef",
    )
    repo.create_document(doc)
    assert repo.find_document_by_hash(kb.id, "deadbeef") is not None
    assert repo.find_document_by_hash(kb.id, "other") is None


def test_chunks_roundtrip(repo: Repository):
    kb = make_kb(repo)
    chunk = Chunk(
        id=new_id("chk"), document_id=new_id("doc"), kb_id=kb.id,
        chunk_index=0, text="hello world", char_count=11,
        source_url="https://a.example.com/x",
    )
    repo.create_chunks([chunk])
    assert repo.count_chunks(kb.id) == 1
    got = repo.list_chunks(kb.id)[0]
    assert got.text == "hello world"
    assert repo.get_chunks_by_ids(kb.id, [chunk.id])[0].id == chunk.id


def test_user_url_provider_validates_and_classifies():
    provider = UserURLProvider()
    sources = provider.discover("kb_x", "https://example.com/report.pdf https://example.com/page")
    assert len(sources) == 2
    assert sources[0].source_type == SourceType.PDF
    assert sources[1].source_type == SourceType.WEB_PAGE


def test_user_url_provider_records_invalid_urls():
    provider = UserURLProvider()
    sources = provider.discover("kb_x", "http://localhost:9000/secret")
    assert len(sources) == 1
    assert sources[0].notes and sources[0].notes.startswith("INVALID")


def test_evaluation_question_persistence(repo: Repository):
    kb = make_kb(repo)
    q = EvaluationQuestion(
        id=new_id("eq"), kb_id=kb.id, question="What is a BMS?",
        expected_keywords=["battery", "management"],
    )
    repo.create_evaluation_question(q)
    assert repo.list_evaluation_questions(kb.id)[0].question == "What is a BMS?"
