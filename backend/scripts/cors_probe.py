"""Backend CORS probe for frontend diagnostics — never prints secrets."""
from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

URL = "http://localhost:8000/api/system/status"
ORIGIN = "http://localhost:3000"
HEADERS = {"Origin": ORIGIN, "Accept": "application/json"}
HEADER_NAMES = {
    "access-control-allow-origin",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-allow-credentials",
    "vary",
    "content-type",
}


def main() -> None:
    req = Request(URL, headers=HEADERS)
    try:
        with urlopen(req, timeout=5) as r:
            body = r.read().decode()
            print(f"STATUS {r.status}")
            for k, v in r.headers.items():
                if k.lower() in HEADER_NAMES:
                    print(f"{k}: {v}")
            print(f"BODY {body}")
    except HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code} {e.reason}")
        for k, v in e.headers.items():
            if k.lower() in HEADER_NAMES:
                print(f"{k}: {v}")
        print(f"BODY {body}")
    except (URLError, OSError) as e:
        print(f"REACH_ERROR {type(e).__name__}: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
