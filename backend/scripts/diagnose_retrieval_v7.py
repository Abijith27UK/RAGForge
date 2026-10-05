"""Diagnose the V7 smoke-test relevance bug: a stability question returned a
propulsion chunk.

Read-only: creates and deletes its own scratch KB, and never touches any
pre-existing knowledge base.
"""
from __future__ import annotations

import sys

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8011").rstrip("/")

STABILITY = """# Hydrostatics and Stability

The metacentric height GM must be positive for stable equilibrium of a floating body.
Free surface effect reduces the effective metacentric height in a flooded compartment.
A tender ship has a small metacentric height and a long natural roll period.
The transverse metacentric height GM is derived from the righting arm GZ divided by
the heel angle phi.

# Propulsion

Propeller cavitation occurs when the local pressure falls below the vapour pressure of water.
The advance ratio relates the advance per revolution to the propeller diameter.
"""


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=180.0)
    kb = client.post(
        "/api/knowledge-bases",
        json={"name": "SMOKE — diagnosis", "domain": "Naval Architecture",
              "purpose": "diag", "target_audience": "students",
              "source_mode": "user_provided"},
    ).json()
    kb_id = kb["id"]
    try:
        client.post(
            f"/api/knowledge-bases/{kb_id}/documents/upload",
            files={"files": ("naval.md", STABILITY.encode("utf-8"), "text/markdown")},
            data={"index": "true", "chunker": "section-aware",
                  "target_size": "400", "overlap": "40"},
        )
        chunks = client.get(f"/api/knowledge-bases/{kb_id}/chunks").json()
        print(f"chunks: {len(chunks.get('chunks', chunks) if isinstance(chunks, dict) else chunks)}")
        for c in (chunks.get("chunks", chunks) if isinstance(chunks, dict) else chunks):
            print(f"  {c['id']}: score-free | section={c.get('section')} | {c['text'][:70]!r}")

        q = "What happens to GM when the center of gravity rises?"
        print(f"\nQUERY: {q}")
        for strategy in ("dense", "bm25", "hybrid"):
            r = client.post(
                f"/api/knowledge-bases/{kb_id}/retrieve",
                json={"query": q, "top_k": 5, "strategy": strategy},
            )
            body = r.json()
            print(f"\n--- {strategy} (backend={body.get('retrieval_backend')}) ---")
            for item in body["results"]:
                print(f"  score={item['score']:.4f} {item['chunk_id']} "
                      f"§{item.get('provenance', {}).get('section')} "
                      f"{item['text'][:80]!r}")

        body = client.post(
            f"/api/knowledge-bases/{kb_id}/chat",
            json={"message": q, "retrieval_strategy": "dense"},
        ).json()
        print(f"\nquery_trace: {body['query_trace']['normalized_query']!r}")
        print(f"extracted terms would be checked in plan; evidence returned:")
        for e in body["evidence"]:
            print(f"  {e['evidence_id']} score={e['retrieval_score']:.4f} "
                  f"§{e.get('section')} {e['content'][:70]!r}")
    finally:
        client.delete(f"/api/knowledge-bases/{kb_id}")
        print("\nscratch KB deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())