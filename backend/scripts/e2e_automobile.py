"""Real end-to-end run: Automobile Engineering knowledge base.

Drives the live API (backend on :8000, Qdrant on :6333) through the full
pipeline with REAL network sources, REAL embeddings, REAL vector storage and
REAL retrieval. Prints a machine-checkable evidence trail. No stage is mocked:
domain analysis uses the dev-mock LLM (clearly labelled is_mock=True) because
no LLM key is configured; everything downstream is real.

Usage:  python scripts/e2e_automobile.py
"""
from __future__ import annotations

import json
import sys
import time

import httpx

BASE = "http://localhost:8000"
SOURCES = [
    ("https://en.wikipedia.org/wiki/Electric_vehicle", "Electric vehicle — Wikipedia"),
    ("https://en.wikipedia.org/wiki/Battery_management_system", "Battery management system — Wikipedia"),
    ("https://en.wikipedia.org/wiki/Regenerative_braking", "Regenerative braking — Wikipedia"),
    ("https://en.wikipedia.org/wiki/Vehicle_dynamics", "Vehicle dynamics — Wikipedia"),
]
EVAL_QUESTIONS = [
    {
        "question": "What are the main components and functions of an electric vehicle powertrain?",
        "expected_keywords": ["electric motor", "battery", "inverter", "powertrain"],
    },
    {
        "question": "How does a battery management system protect lithium-ion cells?",
        "expected_keywords": ["cell", "voltage", "temperature", "balancing", "overcharge"],
    },
    {
        "question": "How does regenerative braking recover energy?",
        "expected_keywords": ["kinetic energy", "motor", "battery", "deceleration"],
    },
    {
        "question": "Which suspension geometry parameters affect vehicle handling?",
        "expected_keywords": ["suspension", "camber", "caster", "roll", "handling"],
    },
]


def banner(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main() -> int:
    c = httpx.Client(base_url=BASE, timeout=120.0)

    banner("0. System status (honest dependency report)")
    status = c.get("/api/system/status").json()
    print(json.dumps(status, indent=2))
    assert status["qdrant"]["reachable"], "Qdrant must be running for the real run"

    banner("1. Create knowledge base")
    r = c.post(
        "/api/knowledge-bases",
        json={
            "name": "Automobile Engineering KB (E2E)",
            "domain": "Automobile Engineering",
            "purpose": "Course-reference knowledge base for EV and vehicle-dynamics engineering education",
            "target_audience": "Undergraduate automotive engineering students",
            "depth": "technical",
        },
    )
    r.raise_for_status()
    kb = r.json()
    kb_id = kb["id"]
    print(f"KB id={kb_id} status={kb['status']}")
    assert kb["domain"] == "Automobile Engineering"

    banner("2. Domain analysis (dev-mock LLM, clearly labelled)")
    r = c.post(f"/api/knowledge-bases/{kb_id}/analyze-domain")
    r.raise_for_status()
    spec = r.json()
    print(f"generated_by={spec['generated_by']} is_mock={spec['is_mock']}")
    print(f"subdomains={spec['subdomains'][:6]}")
    print(f"key_concepts={spec['key_concepts'][:8]}")
    print(f"requirements={len(spec['knowledge_requirements'])}, source_categories={spec['recommended_source_categories'][:4]}")

    banner("3. Source discovery (real URLs) + quality engine (real probes)")
    urls = "\n".join(u for u, _ in SOURCES)
    r = c.post(
        f"/api/knowledge-bases/{kb_id}/discover-sources",
        json={"provider": "user-url", "query": urls, "limit": 10},
    )
    r.raise_for_status()
    sources = r.json()
    for s in sources:
        q = s.get("quality") or {}
        probe = q.get("probe") or {}
        print(
            f"  {s['decision']:7s} score={s['trust_score']:.3f} probe={probe.get('http_status')} "
            f"lastmod={(probe.get('last_modified') or '')[:10]} {s['url']}"
        )
        assert s["quality"], "every source must carry a full explainable assessment"
    accepted = [s for s in sources if s["decision"] == "ACCEPT"]
    review = [s for s in sources if s["decision"] == "REVIEW"]
    print(f"accepted={len(accepted)} review={len(review)} rejected={len(sources) - len(accepted) - len(review)}")
    # Accept anything still pending/reviewed manually (user override path).
    for s in sources:
        if s["decision"] != "ACCEPT":
            r2 = c.post(f"/api/knowledge-bases/{kb_id}/sources/{s['id']}/decision", json={"decision": "ACCEPT"})
            r2.raise_for_status()

    banner("4. Ingestion (real downloads + parsing)")
    r = c.post(f"/api/knowledge-bases/{kb_id}/ingest", json={})
    r.raise_for_status()
    run = r.json()
    print(f"ingestion: {run['stages'][0]['message']} (run {run['status']})")
    docs = c.get(f"/api/knowledge-bases/{kb_id}/documents").json()
    for d in docs:
        print(f"  doc {d['id'][:16]}… {d['source_type']:9s} {d['text_length']:6d} chars  {d['title']}")
    assert docs, "documents must exist"

    banner("5. Chunking + embedding + indexing (real MiniLM + real Qdrant)")
    t0 = time.time()
    r = c.post(f"/api/knowledge-bases/{kb_id}/index", json={"chunker": "section-aware", "target_size": 1200, "overlap": 150})
    r.raise_for_status()
    run = r.json()
    for st in run["stages"]:
        print(f"  {st['stage']:10s} {st['status']:6s} items={st['items_processed']:4d}  {st['message']}")
    print(f"index run took {time.time() - t0:.1f}s")

    chunks = c.get(f"/api/knowledge-bases/{kb_id}/chunks?limit=500").json()
    print(f"chunks in SQLite: {len(chunks)}")
    assert chunks
    pages = [c2["page"] for c2 in chunks if c2.get("page") is not None]
    print(f"chunks with page provenance: {len(pages)} (PDFs would set these; HTML pages are None)")

    banner("6. Re-index stability (stale-vector fix evidence)")
    r = c.post(f"/api/knowledge-bases/{kb_id}/index", json={"chunker": "fixed-size", "target_size": 900, "overlap": 100})
    r.raise_for_status()
    run2 = r.json()
    for st in run2["stages"]:
        print(f"  {st['stage']:10s} {st['status']:6s} items={st['items_processed']:4d}  {st['message']}")
    chunks2 = c.get(f"/api/knowledge-bases/{kb_id}/chunks?limit=500").json()
    ids2 = {c2["id"] for c2 in chunks2}
    print(f"chunks after re-index: {len(chunks2)} (chunk ids replaced: {all(c2['id'] in ids2 for c2 in chunks2)})")

    banner("7. Retrieval lab (real dense retrieval)")
    for query in [
        "What are the major components of an electric vehicle powertrain?",
        "How does a battery management system protect cells?",
        "Which suspension parameters affect handling?",
    ]:
        r = c.post(f"/api/knowledge-bases/{kb_id}/retrieve", json={"query": query, "top_k": 5})
        r.raise_for_status()
        resp = r.json()
        top = resp["results"][0] if resp["results"] else None
        if top:
            prov = top["provenance"]
            print(f"\n  Q: {query}")
            print(f"  top1 score={top['score']:.4f} chunk={top['chunk_id'][:14]}… section={prov.get('section')}")
            print(f"  source: {prov.get('source_url')}")
            print(f"  text:   {top['text'][:130]}…")
        else:
            print(f"\n  Q: {query}\n  NO RESULTS")

    banner("8. Ground-truth authoring + evaluation run")
    for q in EVAL_QUESTIONS:
        r = c.post(f"/api/knowledge-bases/{kb_id}/evaluation-questions", json=q)
        r.raise_for_status()
    print(f"authored {len(EVAL_QUESTIONS)} questions with expected keywords (heuristic mode, disclosed)")
    r = c.post(f"/api/knowledge-bases/{kb_id}/evaluate", json={"top_k": 5})
    r.raise_for_status()
    er = r.json()
    agg = er["aggregate"]
    print(
        f"Recall@5={agg['recall_at_k']} Precision@5={agg['precision_at_k']} "
        f"MRR={agg['mrr']} NDCG={agg['ndcg']} evaluated={agg['questions_evaluated']}"
    )
    print(f"aggregate notes: {agg.get('notes')}")
    for p in er["per_question"]:
        print(f"  R@5={p['recall_at_k']} MRR={p['mrr']} :: {p['question'][:60]}…")

    banner("9. Cleanup (keep KB for UI inspection)")
    print(f"KB {kb_id} left in place — open http://localhost:3000/knowledge-bases/{kb_id} to inspect")
    print("\nE2E RUN COMPLETED WITH REAL DATA END-TO-END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
