"""V19 — second-domain validation: deterministic logic under test.

Hermetic like the V17/V18 suites: no network, no Qdrant, no real LLM. What is
tested here is the honest-reporting machinery V19 adds or depends on:

* the LLM call budget (`LLM_TIMEOUT_S` / `LLM_MAX_RETRIES`) actually reaches
  the openai client, and mock detection is unchanged;
* the provider-identity record carries no secret material;
* the draft benchmark can only be written when every supported label is
  grounded in a real corpus chunk (exact-phrase + keyword self-check), and the
  artifact always says tier=draft / human_review=pending;
* the V19 driver's approval helpers behave (an operator approves exact URLs;
  anything not discovered cannot be approved).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings  # noqa: E402
from app.llm.provider import create_llm_provider  # noqa: E402
from scripts.v19_benchmark_tool import (  # noqa: E402
    AUTHORSHIP,
    benchmark_from_spec,
    find_grounding,
    grounding_problems,
)
from scripts.v19_second_domain_validation import (  # noqa: E402
    grounding_probe_questions,
    normalize_urls,
    provider_identity_report,
    uncovered,
)


# ---------------------------------------------------------------------------
# LLM call budget and provider identity
# ---------------------------------------------------------------------------


class _RecordingOpenAI:
    """Stands in for openai.OpenAI and remembers its construction kwargs."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):  # noqa: ANN003 - mirrors the SDK signature
        _RecordingOpenAI.last_kwargs = kwargs


@pytest.fixture
def openai_stub(monkeypatch):
    import openai

    monkeypatch.setattr(openai, "OpenAI", _RecordingOpenAI)
    _RecordingOpenAI.last_kwargs = {}
    return _RecordingOpenAI


def test_configured_timeout_and_retries_reach_the_client(openai_stub):
    settings = Settings(
        llm_provider="openai",
        llm_model="test-model",
        llm_api_key="unit-test-key",
        llm_timeout_s=7.5,
        llm_max_retries=0,
    )
    provider = create_llm_provider(settings)
    assert provider.is_mock is False
    assert provider.name == "openai"
    assert provider.model == "test-model"
    assert openai_stub.last_kwargs["timeout"] == 7.5
    assert openai_stub.last_kwargs["max_retries"] == 0
    # The budget is also recorded on the provider for the identity report.
    assert provider.timeout_s == 7.5
    assert provider.max_retries == 0


def test_default_budget_is_not_the_sdk_default(openai_stub):
    """V19's point: an interactive tool call must not inherit a 600 s budget."""
    settings = Settings(llm_provider="openai", llm_api_key="k", llm_model="m")
    provider = create_llm_provider(settings)
    assert provider.timeout_s == 120.0
    assert openai_stub.last_kwargs["timeout"] == 120.0


def test_mock_detection_is_unchanged(openai_stub):
    mock = create_llm_provider(Settings(llm_provider="", allow_mock_llm=True))
    assert mock.is_mock is True
    real = create_llm_provider(Settings(llm_provider="openai", llm_api_key="k", llm_model="m"))
    assert real.is_mock is False


def test_provider_identity_report_contains_no_secrets():
    settings = Settings(
        llm_provider="openai",
        llm_model="test-model",
        llm_api_key="super-secret-key",
        llm_base_url="https://api.example.com/v1",
    )
    provider = create_llm_provider(Settings(**settings.model_dump()))
    report = provider_identity_report(provider, base_url=settings.llm_base_url)
    assert report["is_mock"] is False
    assert report["base_url_host"] == "api.example.com"
    assert set(report) == {
        "name",
        "model",
        "is_mock",
        "base_url_host",
        "timeout_s",
        "max_retries",
    }
    serialized = repr(report)
    assert "super-secret-key" not in serialized
    assert "/v1" not in report["base_url_host"]


def test_probe_sets_are_declared():
    probes = grounding_probe_questions()
    assert len(probes["retrieval"]) >= 6, "retrieval probes must span the subdomains"
    assert len(probes["in_corpus"]) >= 2
    assert len(probes["unsupported"]) >= 3


# ---------------------------------------------------------------------------
# approval helpers
# ---------------------------------------------------------------------------


def test_normalize_urls_dedupes_and_strips():
    got = normalize_urls(
        [" https://a.example/x ", "https://a.example/x", "", "https://b.example/y"]
    )
    assert got == ["https://a.example/x", "https://b.example/y"]


def test_uncovered_reports_urls_outside_the_allowed_set():
    discovered = ["https://a", "https://b", "https://c"]
    assert uncovered(["https://b"], discovered) == []
    assert uncovered(["https://z", "https://a"], discovered) == ["https://z"]


# ---------------------------------------------------------------------------
# draft benchmark grounding
# ---------------------------------------------------------------------------


def _chunks():
    return [
        {
            "chunk_id": "chk_1",
            "document_id": "doc_1",
            "document_url": "https://rfc.example/rfc1.txt",
            "text": "The Time to Live field is decremented by each router; if it reaches zero the datagram is discarded.",
        },
        {
            "chunk_id": "chk_2",
            "document_id": "doc_1",
            "document_url": "https://rfc.example/rfc1.txt",
            "text": "The header also carries the Internet Header Length in 32-bit words.",
        },
        {
            "chunk_id": "chk_3",
            "document_id": "doc_2",
            "document_url": "https://rfc.example/rfc2.txt",
            "text": "Nothing relevant to the anchors above lives here.",
        },
    ]


def _spec():
    return [
        {
            "id": "q1",
            "subdomain": "ipv4",
            "question": "What happens to the Time to Live field?",
            "document_url": "https://rfc.example/rfc1.txt",
            "anchor": "Time to Live",
            "expected_keywords": ["router", "discarded"],
            "provenance": "rfc1 section 3",
        },
        {
            "id": "q2",
            "subdomain": "out-of-scope",
            "question": "Which RFC defines MQTT?",
            "unsupported": True,
            "provenance": "not in corpus",
        },
    ]


def test_find_grounding_respects_anchor_and_document():
    matches = find_grounding("time to live", "https://rfc.example/rfc1.txt", _chunks())
    assert [m["chunk_id"] for m in matches] == ["chk_1"]
    # The same anchor in a different document finds nothing.
    assert find_grounding("Time to Live", "https://rfc.example/rfc2.txt", _chunks()) == []


def test_benchmark_from_spec_is_a_draft_with_grounding():
    grounding = {"q1": find_grounding("Time to Live", "https://rfc.example/rfc1.txt", _chunks())}
    artifact = benchmark_from_spec(
        _spec(), grounding, kb_id="kb_x", kb_name="X", generated_at="2026-10-14T00:00:00+00:00"
    )
    assert artifact["authorship"]["tier"] == "draft"
    assert artifact["authorship"]["human_review"] == "pending"
    assert artifact["authorship"]["labels_are_machine_generated"] is True
    assert artifact["authorship"] == {**AUTHORSHIP, "drafted_at": "2026-10-14T00:00:00+00:00"}
    q1 = next(q for q in artifact["questions"] if q["id"] == "q1")
    q2 = next(q for q in artifact["questions"] if q["id"] == "q2")
    assert q1["expected_chunk_ids"] == ["chk_1"]
    assert q1["expected_document_ids"] == ["doc_1"]
    assert q2["expected_chunk_ids"] == [] and q2["expected_document_ids"] == []
    assert grounding_problems(artifact, _chunks()) == []


def test_grounding_rejects_labels_that_do_not_carry_the_evidence():
    grounding = {"q1": find_grounding("Time to Live", "https://rfc.example/rfc1.txt", _chunks())}
    artifact = benchmark_from_spec(
        _spec(), grounding, kb_id="kb_x", kb_name="X", generated_at="now"
    )
    # A keyword that is not in the chunk cannot be part of a verified label.
    broken = dict(artifact)
    broken["questions"] = [dict(q) for q in artifact["questions"]]
    broken["questions"][0]["expected_keywords"] = ["nonexistent-term"]
    assert grounding_problems(broken, _chunks())


def test_grounding_rejects_unknown_chunks_and_all_unsupported():
    grounding = {"q1": [{"chunk_id": "chk_missing", "document_id": "doc_z"}]}
    artifact = benchmark_from_spec(
        _spec(), grounding, kb_id="kb_x", kb_name="X", generated_at="now"
    )
    problems = grounding_problems(artifact, _chunks())
    assert any("not in the corpus" in p for p in problems)

    unsupported_only = benchmark_from_spec(
        [_spec()[1]], {}, kb_id="kb_x", kb_name="X", generated_at="now"
    )
    assert any("nothing is measurable" in p for p in grounding_problems(unsupported_only, _chunks()))


def test_grounding_rejects_duplicate_ids():
    grounding = {"q1": find_grounding("Time to Live", "https://rfc.example/rfc1.txt", _chunks())}
    artifact = benchmark_from_spec(
        _spec(), grounding, kb_id="kb_x", kb_name="X", generated_at="now"
    )
    artifact["questions"].append(dict(artifact["questions"][0]))
    assert any("duplicate question id" in p for p in grounding_problems(artifact, _chunks()))


# ---------------------------------------------------------------------------
# the shipped V19 question spec itself
# ---------------------------------------------------------------------------


def test_shipped_question_spec_is_well_formed():
    from scripts.v19_benchmark_tool import QUESTIONS

    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids)), "question ids must be unique"
    supported = [q for q in QUESTIONS if not q.get("unsupported")]
    unsupported = [q for q in QUESTIONS if q.get("unsupported")]
    assert len(supported) >= 20
    assert len(unsupported) >= 5
    subdomains = {q["subdomain"] for q in supported}
    assert len(subdomains) >= 6, "the draft must span multiple subdomains"
    for q in supported:
        assert q.get("anchor"), f"{q['id']} needs an anchor phrase"
        assert q.get("document_url", "").startswith("https://")
        assert q.get("expected_keywords"), f"{q['id']} needs expected keywords"
        assert q.get("provenance"), f"{q['id']} needs a provenance note"
    for q in unsupported:
        assert "expected" not in q, "unsupported questions must not carry expected evidence"
        assert q.get("provenance"), f"{q['id']} needs a provenance note"
