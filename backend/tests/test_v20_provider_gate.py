"""V20 — LLM provider gate: does a REAL provider actually resolve, and does a
failure stay a failure?

Why this file exists
--------------------
The workspace was configured `LLM_PROVIDER=gemini` while the factory rejected
that value ("declared but not yet implemented"). Two bad consequences:

1. the domain-analysis path failed outright (no analysis at all), and
2. the ANSWERING path silently fell back to a disclosed deterministic mock, so
   every answer in the UI said `is_mock=true` with a provider error attached.

V20 routes `gemini` through the existing OpenAI-compatible provider, because
Google exposes an OpenAI-compatible surface
(https://ai.google.dev/gemini-api/docs/openai). These tests pin the factory
contract **without a network call**: transport is faked, so what is verified is
resolution, configuration, and failure transparency — NOT that the live endpoint
works. No test here contacts Google, and none will pass a real key.

Claim boundary: "the factory resolves a real provider" and "a real API call
succeeded" are different claims. The second needs credentials and is verified by
`scripts/v20_verify_provider.py --live` with a rotated key; it is NOT verified
here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings  # noqa: E402
from app.llm.provider import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    GEMINI_OPENAI_BASE_URL,
    LLMError,
    OpenAICompatibleProvider,
    create_llm_provider,
)


class Tiny(BaseModel):
    """A structured-output schema small enough to fake deterministically."""

    answer: str
    confidence: str = "low"


def settings_for(monkeypatch, provider: str, **env) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", provider)
    for key, value in env.items():
        monkeypatch.setenv(key.upper(), value)
    return Settings()


def test_gemini_resolves_through_the_existing_openai_compatible_provider(monkeypatch):
    settings = settings_for(
        monkeypatch,
        "gemini",
        llm_model="gemini-2.5-flash",
        llm_api_key="test-key-not-real",
    )
    provider = create_llm_provider(settings)
    assert isinstance(provider, OpenAICompatibleProvider)
    # Identity is reported honestly, not as "openai" merely because the wire
    # format matches.
    assert provider.name == "gemini"
    assert provider.model == "gemini-2.5-flash"
    assert provider.is_mock is False
    assert str(provider._client.base_url) == GEMINI_OPENAI_BASE_URL
    # The call budget is bounded config, not SDK defaults.
    assert provider.timeout_s == settings.llm_timeout_s
    assert provider.max_retries == settings.llm_max_retries


def test_explicit_base_url_wins_over_the_documented_default(monkeypatch):
    settings = settings_for(
        monkeypatch,
        "gemini",
        llm_api_key="test-key-not-real",
        llm_base_url="https://gateway.internal/v1/",
    )
    provider = create_llm_provider(settings)
    assert str(provider._client.base_url).rstrip("/") == "https://gateway.internal/v1"


def test_model_defaults_only_when_unset(monkeypatch):
    settings = settings_for(monkeypatch, "gemini", llm_api_key="test-key-not-real")
    assert create_llm_provider(settings).model == DEFAULT_GEMINI_MODEL


def test_gemini_without_a_key_is_refused_with_an_actionable_message(monkeypatch):
    settings = settings_for(monkeypatch, "gemini", llm_api_key="")
    with pytest.raises(LLMError) as excinfo:
        create_llm_provider(settings)
    assert "LLM_API_KEY" in str(excinfo.value)
    assert "gemini" in str(excinfo.value)


def test_anthropic_stays_explicitly_unimplemented(monkeypatch):
    settings = settings_for(monkeypatch, "anthropic", llm_api_key="test-key-not-real")
    with pytest.raises(LLMError) as excinfo:
        create_llm_provider(settings)
    message = str(excinfo.value)
    assert "not implemented" in message
    assert "gemini" in message  # tells the operator what DOES work


def test_unknown_provider_is_rejected(monkeypatch):
    settings = settings_for(monkeypatch, "not-a-provider", llm_api_key="x")
    with pytest.raises(LLMError):
        create_llm_provider(settings)


def test_mock_provider_is_still_available_and_labelled(monkeypatch):
    settings = settings_for(monkeypatch, "mock")
    provider = create_llm_provider(settings)
    assert provider.is_mock is True


# ---------------------------------------------------------------------------
# Failure transparency (faked transport: no network)
# ---------------------------------------------------------------------------


def _gemini_provider(monkeypatch) -> OpenAICompatibleProvider:
    settings = settings_for(
        monkeypatch, "gemini", llm_model="gemini-2.5-flash", llm_api_key="test-key-not-real"
    )
    provider = create_llm_provider(settings)
    assert isinstance(provider, OpenAICompatibleProvider)
    return provider


def test_a_failed_real_call_is_an_error_never_a_mock_success(monkeypatch):
    provider = _gemini_provider(monkeypatch)

    def boom(**kwargs):
        raise RuntimeError("simulated 401 from the endpoint")

    monkeypatch.setattr(provider._client.chat.completions, "create", boom)
    with pytest.raises(LLMError) as excinfo:
        provider.generate_structured("prompt", Tiny)
    message = str(excinfo.value)
    assert "LLM call failed" in message
    assert "401" in message
    # The failure is NOT converted into a mock answer: the caller sees an error.
    assert provider.is_mock is False


def test_a_successful_call_returns_validated_structured_output(monkeypatch):
    provider = _gemini_provider(monkeypatch)
    seen: dict = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"answer": "yes", "confidence": "high"}')
                )
            ]
        )

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    result = provider.generate_structured("THE PROMPT", Tiny, system="be terse")
    assert isinstance(result, Tiny)
    assert result.answer == "yes"
    assert result.confidence == "high"
    # The model id actually sent is the configured one, and no secret is included.
    assert seen["model"] == "gemini-2.5-flash"
    assert seen["messages"][0] == {"role": "system", "content": "be terse"}
    assert "THE PROMPT" in seen["messages"][1]["content"]
    assert "test-key-not-real" not in str(seen)


def test_unparseable_output_is_reported_not_guessed(monkeypatch):
    provider = _gemini_provider(monkeypatch)
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **kw: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="I refuse to emit JSON."))]
        ),
    )
    with pytest.raises(LLMError) as excinfo:
        provider.generate_structured("prompt", Tiny)
    assert "JSON" in str(excinfo.value)
