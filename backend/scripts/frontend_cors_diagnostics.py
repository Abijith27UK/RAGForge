"""
Backend CORS / reachability diagnostics — never prints secrets.
Intended to run from the frontend developer machine to explain a
"Backend unreachable: Failed to fetch" banner before opening an issue.
"""
from __future__ import annotations

import os
import socket
import urllib.request
import urllib.error
from contextlib import contextmanager
from typing import Iterator

BACKEND = os.environ.get("NEXT_PUBLIC_BACKEND_URL", "http://localhost:8000").rstrip("/")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
HEALTH_PATH = "/api/system/health"
STATUS_PATH = "/api/system/status"

RElevant_Headers = frozenset(
    {
        "access-control-allow-origin",
        "access-control-allow-methods",
        "access-control-allow-headers",
        "access-control-allow-credentials",
        "vary",
        "content-type",
        "referrer-policy",
    }
)


@contextmanager
def _request(method: str, path: str, headers: dict[str, str]) -> Iterator[urllib.request.Request]:
    url = f"{BACKEND}{path}"
    req = urllib.request.Request(url, method=method, headers=headers or {})
    yield req


def fmt_headers(headers: urllib.request.HTTPMessage) -> list[str]:
    out: list[str] = []
    for key in headers:
        if key.lower() in Relev ant_Headers:
            out.append(f"{key}: {headers[key]}")
    return out


def main() -> None:
    print(f"Backend: {BACKEND}")
    print(f"Frontend origin used for CORS check: {FRONTEND_ORIGIN}\n")

    # 1. Parse/connect check
    print("1) Network reachability")
    host = BACKEND.split("://", 1)[-1].split("/", 1)[0]
    print(f"   -> target host parsed: {host}")
    try:
        addr_info = socket.getaddrinfo(host.split(":")[0], 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
        print(f"   -> host resolved: {len(addr_info)} address(es)")
    except socket.gaierror as e:
        print(f"   -> RESOLVE FAILED: {e}")
        return

    # 2. Health (no Origin header — CORS does not apply to same-origin-ish checks here,
    #    but we still report what comes back so we can spot proxy rewrites)
    print("\n2) Backend health (no Origin header)")
    try:
        with urllib.request.urlopen(f"{BACKEND}{HEALTH_PATH}", timeout=8) as r:
            body = r.read().decode("utf-8", "replace")
            print(f"   -> HTTP {r.status}")
            for line in fmt_headers(r.headers):
                print(f"   {line}")
            print(f"   -> body: {body[:400]}")
    except Exception as e:
        print(f"   -> reachable check failed: {type(e).__name__} {e}")

    # 3. CORS-relevant response headers with the frontend origin present
    print("\n3) CORS-relevant headers with frontend origin")
    try:
        with _request("GET", STATUS_PATH, {"Origin": FRONTEND_ORIGIN, "Accept": "application/json"}) as req:
            try:
                with urllib.request.urlopen(req, timeout=8) as r:
                    print(f"   -> HTTP {r.status}")
                    for line in fmt_headers(r.headers):
                        print(f"   {line}")
                    if "access-control-allow-origin" in (h.lower() for h in r.headers):
                        print("   -> CORS ALLOW present for origin above")
                    else:
                        print("   -> NO access-control-allow-origin header returned")
            except urllib.error.HTTPError as e:
                print(f"   -> HTTP {e.code} {e.reason}")
                for line in fmt_headers(e.headers):
                    print(f"   {line}")
                try:
                    print(f"   -> body: {e.read().decode('utf-8', 'replace')[:400]}")
                except Exception:
                    pass
    except Exception as e:
        print(f"   -> CORS probe failed: {type(e).__name__} {e}")


if __name__ == "__main__":
    main()
