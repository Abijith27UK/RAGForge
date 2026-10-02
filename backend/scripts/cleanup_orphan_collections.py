"""Delete Qdrant collections that no knowledge base or metadata row references.

Safety rule: a collection is only deleted when ALL of the following hold:

  * its ``kb_<kb_id>`` suffix does not match any row in ``knowledge_bases``,
  * no ``documents`` row, no ``chunks`` row and no ``sources`` row exists for
    that ``kb_id``.

Anything referenced by the database is kept, always. Real experiment knowledge
bases (Automobile Engineering and friends) therefore cannot be touched.

    cd backend && .venv/Scripts/python.exe scripts/cleanup_orphan_collections.py
    cd backend && .venv/Scripts/python.exe scripts/cleanup_orphan_collections.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402


def _get(url: str, timeout: int = 15):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = get_settings().qdrant_url.rstrip("/")
    con = sqlite3.connect(str(get_settings().db_path))
    real_kbs = {r[0] for r in con.execute("SELECT id FROM knowledge_bases")}
    names = [c["name"] for c in _get(f"{base}/collections")["result"]["collections"]]

    orphans, kept = [], []
    for name in names:
        kb_id = name[3:] if name.startswith("kb_") else name
        if kb_id in real_kbs:
            kept.append((name, "knowledge base exists"))
            continue
        counts = {
            table: con.execute(f"SELECT COUNT(*) FROM {table} WHERE kb_id = ?", (kb_id,)).fetchone()[0]
            for table in ("documents", "chunks", "sources")
        }
        if any(counts.values()):
            kept.append((name, f"referenced by rows {counts}"))
            continue
        orphans.append(name)

    for name, why in kept:
        print(f"  KEEP    {name}  ({why})")

    if args.dry_run:
        print(f"\nDry run: {len(orphans)} unreferenced collection(s) would be deleted.")
        for name in orphans:
            print(f"  DELETE  {name}")
        return 0

    failed = []
    for name in orphans:
        try:
            req = urllib.request.Request(f"{base}/collections/{name}", method="DELETE")
            urllib.request.urlopen(req, timeout=15).read()
        except Exception as exc:  # pragma: no cover - network dependent
            failed.append((name, str(exc)))

    print(f"\nDeleted {len(orphans) - len(failed)} unreferenced collection(s).")
    for name, err in failed:
        print(f"  FAILED  {name}: {err}")
    remaining = len(_get(f"{base}/collections")["result"]["collections"])
    print(f"Collections remaining: {remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())