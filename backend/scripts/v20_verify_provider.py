"""V20 — verify the REAL LLM provider gate without ever leaking a secret.

Two distinct claims, kept distinct:

* **resolution** (default, no network) — the configured provider resolves through
  the real factory, reports a non-mock identity, the model id and the endpoint it
  would call. Proves configuration, not connectivity.
* **one real call** (`--live`) — exactly ONE minimal structured-output request, to
  prove authentication, endpoint compatibility, model availability and schema
  handling. Records provider, model, endpoint identity, duration and a SANITIZED
  failure only.

Secret handling
---------------
No API key, key prefix, key length or Authorization header is ever printed.

**The key in `backend/.env` was exposed in a transcript before 2026-10-15T16:27
and must be treated as compromised.** Rotate it first, then run `--live`. This
script never falls back to the mock: a failed real call is reported as a failure
(exit 1), and `--live` against a mock provider is refused (exit 3).

Usage
-----
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\v20_verify_provider.py
    ... --live --json out.json

Exit codes: 0 ok (or resolved, without --live), 1 the real call failed,
2 no real provider is configured, 3 refused (--live with a mock provider).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

from pydantic import BaseModel  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llm.provider import LLMError, create_llm_provider  # noqa: E402


class ProviderSmokeAnswer(BaseModel):
    """Minimal schema: structured output is what the pipeline actually needs."""

    provider_ok: bool
    echo: str


PROMPT = (
    "This is a connectivity check for a RAG knowledge-base builder. "
    "Set provider_ok to true and echo the word 'ragforge'."
)


def env_mtime() -> str:
    """When the local .env was last written — a rotation hint, never a value."""
    path = BACKEND_ROOT / ".env"
    if not path.exists():
        return "(no backend/.env)"
    try:
        from datetime import datetime, timezone

        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return stamp.isoformat()
    except OSError:
        return "(unreadable)"


def describe(provider) -> dict:
    """Identity + configuration only. No credential material, at any length."""
    base_url = ""
    client = getattr(provider, "_client", None)
    if client is not None:
        base_url = str(getattr(client, "base_url", "") or "")
    return {
        "provider_name": getattr(provider, "name", None),
        "provider_type": type(provider).__name__,
        "provider_model": getattr(provider, "model", None),
        "provider_is_mock": bool(getattr(provider, "is_mock", False)),
        "endpoint": base_url,
        "timeout_s": getattr(provider, "timeout_s", None),
        "max_retries": getattr(provider, "max_retries", None),
    }


def sanitize(exc: BaseException) -> str:
    """A failure description that can be pasted into a report verbatim.

    Keeps the exception TYPE and the HTTP status when the SDK provides one, and
    drops the message body, which can echo request headers.
    """
    status = getattr(exc, "status_code", None)
    kind = type(exc).__name__
    detail = str(exc)
    for marker in ("sk-", "AIza", "Bearer ", "api_key", "api-key"):
        if marker in detail:
            detail = "(message withheld: it may contain credential material)"
            break
    else:
        detail = detail[:300]
    return f"{kind}" + (f" (HTTP {status})" if status else "") + f": {detail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="make ONE real API call")
    parser.add_argument("--json", default="")
    ns = parser.parse_args(argv)

    settings = Settings()
    report: dict = {
        "env_llm_provider": settings.llm_provider or "(empty)",
        "env_llm_model": settings.llm_model or "(empty)",
        "env_llm_base_url": settings.llm_base_url or "(empty, provider default applies)",
        "api_key_present": bool(settings.llm_api_key),
        "backend_env_mtime_utc": env_mtime(),
        "live_call_requested": bool(ns.live),
    }
    print("=" * 74)
    print("RAGForge V20 provider gate")
    print("=" * 74)
    print(f"  LLM_PROVIDER      : {report['env_llm_provider']}")
    print(f"  LLM_MODEL         : {report['env_llm_model']}")
    print(f"  LLM_BASE_URL      : {report['env_llm_base_url']}")
    print(f"  API key present   : {report['api_key_present']}")
    print(f"  backend/.env mtime: {report['backend_env_mtime_utc']}")
    print("  (no key material is printed, at any length, ever)")

    try:
        provider = create_llm_provider(settings, domain_hint="provider gate")
    except LLMError as exc:
        report["resolution"] = {"status": "failed", "error": sanitize(exc)}
        print(f"\nRESOLUTION FAILED: {report['resolution']['error']}")
        _write(ns.json, report)
        return 2

    identity = describe(provider)
    report["resolution"] = {"status": "ok", **identity}
    print("\nresolved provider:")
    for key, value in identity.items():
        print(f"  {key:18s}: {value}")

    if identity["provider_is_mock"]:
        print(
            "\nRESOLUTION -> the MOCK provider. No real model is configured; a live "
            "call would prove nothing, so none is attempted."
        )
        if ns.live:
            print("REFUSED: --live with a mock provider. Rotate the key and configure a "
                  "real provider first.")
            _write(ns.json, report)
            return 3
        _write(ns.json, report)
        return 2

    if not ns.live:
        print(
            "\nRESOLUTION OK: a real provider is configured (is_mock=False). No call was "
            "made. Pass --live to make ONE real structured-output request."
        )
        print(
            "REMINDER: the key present before 2026-10-15T16:27 was exposed in a "
            "transcript and must be rotated before --live is used."
        )
        _write(ns.json, report)
        return 0

    print("\nmaking ONE real structured-output call...")
    started = time.perf_counter()
    try:
        parsed = provider.generate_structured(PROMPT, ProviderSmokeAnswer)
    except Exception as exc:  # noqa: BLE001 - the point is to report the failure
        elapsed = time.perf_counter() - started
        report["live_call"] = {
            "status": "failed",
            "duration_s": round(elapsed, 3),
            "error": sanitize(exc),
        }
        print(f"  FAILED after {elapsed:.2f}s: {report['live_call']['error']}")
        print(
            "  The experiment stops here: no real-model run is claimed, and the mock "
            "was NOT substituted."
        )
        _write(ns.json, report)
        return 1

    elapsed = time.perf_counter() - started
    report["live_call"] = {
        "status": "ok",
        "duration_s": round(elapsed, 3),
        "parsed_provider_ok": bool(getattr(parsed, "provider_ok", False)),
        "parsed_echo": str(getattr(parsed, "echo", ""))[:60],
    }
    print(f"  SUCCESS in {elapsed:.2f}s: schema validated, provider_ok="
          f"{report['live_call']['parsed_provider_ok']}")
    print("  A REAL API call succeeded; is_mock=False; structured output parsed.")
    _write(ns.json, report)
    return 0


def _write(path: str, report: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written: {out}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
