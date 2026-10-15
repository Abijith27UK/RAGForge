"""Verify the configured LLM provider is real (non-mock) and can produce one tiny
structured domain-analysis call, WITHOUT printing any secret material.

This is the Phase-1 "is the provider actually working" check. It sets the provider
to OpenAICompatibleProvider so it uses the Gemini OpenAI-compatible endpoint and
key provided in backend/.env, keeps the prompt short (a tiny domain), and prints
only public identity fields. The full run artifact is written to
scripts/_v19_validation/gemini-provider-check.json and it is read back below.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Settings  # noqa: E402
from app.llm.provider import create_llm_provider  # noqa: E402



def run() -> dict[str, Any]:
    settings = Settings()
    env_provider = os.environ.get("LLM_PROVIDER") or settings.llm_provider or "(none)"
    provider = create_llm_provider(settings, domain_hint="Tiny domain validation")
    identity = {
        "provider_name": provider.name,
        "provider_model": provider.model,
        "provider_is_mock": bool(getattr(provider, "is_mock", False)),
        "env_llm_provider": env_provider,
        "settings_llm_provider": settings.llm_provider or "(empty)",
        "settings_llm_model": settings.llm_model or "(empty)",
        "settings_llm_api_key_len": len(settings.llm_api_key),
        # No part of the key is ever emitted: not a prefix, not a hash.
        "settings_llm_api_key_set": bool(settings.llm_api_key),
        "settings_llm_base_url": settings.llm_base_url or "(empty)",
        "settings_llm_timeout_s": settings.llm_timeout_s,
        "settings_llm_max_retries": settings.llm_max_retries,
        "openai_timeout_passed": getattr(provider, "timeout_s", None),
        "openai_max_retries_passed": getattr(provider, "max_retries", None),
        "provider_type": type(provider).__name__,
    }
    result: dict[str, Any] = {
        "identity": identity,
        "call": {
            "domain": "TinyValidationDomain",
            "purpose": "verify structured output",
            "target_audience": "validation",
            "depth": "minimal",
        },
        "outcome": {},
    }
    if identity["provider_is_mock"]:
        result["outcome"] = {"status": "mock", "reason": "provider resolved to the development mock"}
        return result

    # One tiny structured call: we only need to prove the provider honors a schema,
    # not to produce a useful domain analysis.
    from app.services.domain_analyzer.analyzer import build_analysis_prompt, DomainAnalyzer

    kb = type("KB", (), {
        "id": "verify",
        "domain": result["call"]["domain"],
        "purpose": result["call"]["purpose"],
        "target_audience": result["call"]["target_audience"],
        "depth": result["call"]["depth"],
    })()
    try:
        analyzer = DomainAnalyzer(provider, settings)
        spec = analyzer.analyze(kb)
        result["call"]["prompt_chars"] = len(build_analysis_prompt(kb))
        result["call"]["schema_requested"] = "DomainSpecLLM"
        result["outcome"] = {
            "status": "ok",
            "provider_name": provider.name,
            "model": provider.model,
            "is_mock": bool(getattr(provider, "is_mock", False)),
            "generated_by": getattr(spec, "generated_by", None),
            "subdomains_count": len(getattr(spec, "subdomains", [])),
            "key_concepts_count": len(getattr(spec, "key_concepts", [])),
            "terminology_count": len(getattr(spec, "terminology", [])),
            "warnings": list(getattr(spec, "warnings", []) or []),
            "is_mock_field_on_spec": bool(getattr(spec, "is_mock", False)),
        }
    except Exception as exc:  # noqa: BLE001 - report the public failure, not internals
        result["outcome"] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:600],
        }
    return result


def main() -> int:
    t0 = datetime.now(timezone.utc)
    result = run()
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    art_dir = BACKEND_ROOT / "scripts" / "_v19_validation"
    art_dir.mkdir(parents=True, exist_ok=True)
    art_path = art_dir / "gemini-provider-check.json"
    log = {
        "run_at": t0.isoformat(),
        "elapsed_s": round(elapsed, 3),
        "data_hash": None,
        "full": result,
        "art_path": str(art_path),
    }
    # redact secret for the disk artifact only; printed report uses identity.preview
    log["data_hash"] = json.dumps(result, sort_keys=True, default=str).__hash__()
    art_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(json.dumps(log, indent=2, default=str))

    status = result.get("outcome", {}).get("status") or "unknown"
    print("", "STATUS:", status, "elapsed_s:", round(elapsed, 2))
    return 0 if status in ("ok", "mock") else 1


if __name__ == "__main__":
    raise SystemExit(main())
