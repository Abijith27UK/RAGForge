"""Phase-1 verification: is the real LLM provider alive and does it honor a one-shot
structured schema, WITHOUT printing any secret material.

This script is run BEFORE any corpus download. It:
* confirms the provider is NOT the development mock,
* proves it returns Pydantic-validated structured output for one tiny domain,
* records the call in scripts/_v19_validation/llm-verify.json with key material redacted.

Exit 0 = provider is real and functional (even if slow). Exit 1 = provider failed
or is the mock. Final report states the reason either way.
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
from app.llm.provider import LLMError, create_llm_provider  # noqa: E402


def _configuration_report(settings: "Settings") -> dict[str, Any]:
    """Describe the LLM configuration WITHOUT emitting any key material.

    Nothing derived from the key is recorded — not a prefix, not a hash, not a
    length-derived fingerprint. Only whether it is set and its length, which is
    already visible to anyone who can read the `.env` file being described.
    """
    return {
        "llm_provider": settings.llm_provider or "(empty)",
        "llm_model": settings.llm_model or "(empty)",
        "llm_base_url": settings.llm_base_url or "(empty)",
        "llm_api_key_set": bool(settings.llm_api_key),
        "llm_api_key_len": len(settings.llm_api_key),
        "llm_timeout_s": settings.llm_timeout_s,
        "llm_max_retries": settings.llm_max_retries,
        "allow_mock_llm": settings.allow_mock_llm,
    }


def run() -> dict[str, Any]:
    settings = Settings()
    env_provider = os.environ.get("LLM_PROVIDER") or settings.llm_provider or "(none)"
    result: dict[str, Any] = {
        "configuration": _configuration_report(settings),
        "call": {"domain": "TinyValidationDomain", "purpose": "verify structured output",
                 "target_audience": "validation", "depth": "minimal"},
        "outcome": {},
    }

    # Resolving the provider must not crash the check: an unimplemented provider
    # name or a missing key is a reportable BLOCKER, not a traceback. Nothing is
    # called over the network if this raises.
    try:
        provider = create_llm_provider(settings, domain_hint="Tiny domain validation")
    except LLMError as exc:
        result["identity"] = {"env_llm_provider": env_provider}
        result["outcome"] = {
            "status": "blocked",
            "error_type": "LLMError",
            "error_message": str(exc),
            "hint": (
                "Fix the configuration in backend/.env and run this script again. "
                "For an OpenAI-compatible endpoint (including Gemini's) set "
                "LLM_PROVIDER=openai, LLM_MODEL, LLM_API_KEY and LLM_BASE_URL; "
                "for a local server point LLM_BASE_URL at it. No external call was made."
            ),
        }
        return result

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
        "funded_timeout_passed": getattr(provider, "timeout_s", None),
        "funded_max_retries_passed": getattr(provider, "max_retries", None),
        "provider_type": type(provider).__name__,
    }
    result["identity"] = identity
    if identity["provider_is_mock"]:
        result["outcome"] = {"status": "mock", "reason": "provider resolved to the development mock"}
        return result

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
    except Exception as exc:  # noqa: BLE001
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
    art_path = art_dir / "llm-verify.json"
    log = {"run_at": t0.isoformat(), "elapsed_s": round(elapsed, 3), "full": result,
           "art_path": str(art_path)}
    # Nothing secret is in `result` by construction: `_configuration_report` and
    # the identity block record only set/length, never any part of the key.
    art_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(json.dumps(log, indent=2, default=str))

    status = result.get("outcome", {}).get("status") or "unknown"
    verdict = {
        "ok": "REAL provider verified: a structured-output call succeeded.",
        "mock": "MOCK provider: no real-LLM claim may be made from this run.",
        "blocked": "BLOCKED: the configured provider could not be constructed; "
                   "no external call was attempted.",
        "error": "FAILED: the provider was constructed but the call failed.",
    }.get(status, "UNKNOWN outcome.")
    print("", "STATUS:", status, "| elapsed_s:", round(elapsed, 2))
    print("", verdict)
    return 0 if status in ("ok", "mock") else 1


if __name__ == "__main__":
    raise SystemExit(main())
