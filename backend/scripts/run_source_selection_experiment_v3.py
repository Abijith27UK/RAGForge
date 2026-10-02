"""Source-selection experiment v3 runner (V3 Phase B) — INFRASTRUCTURE ONLY.

Designed, deterministic, and gated: the runner REFUSES to start unless the
benchmark it evaluates against is a FROZEN BenchmarkVersion (human-reviewed
ground truth). This implements the Phase A -> Phase B contract: official
experiments only on frozen benchmarks.

Differences vs the v2 runner (reused protocol otherwise):
- multiple seeds (design default: 20260915..19); each seed is an independent
  RANDOM_BASELINE arm, deterministic via random.Random(seed).sample(...)
- evaluation targets a FROZEN benchmark version via the Phase A API
  (config.benchmark_version) instead of ad-hoc live questions; GT translation
  (content_hash) is still performed per corpus because chunk ids are regenerated
- records the full per-run provenance block required by the v3 design JSON
- run with --dry-run to validate the configuration and selection determinism
  WITHOUT building anything

NOT RUN yet by design: awaiting a frozen benchmark version.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import httpx

BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_v2_pool_scores import build_domain_scorer, load_pool, score_pool  # noqa: E402


def banner(msg: str) -> None:
    print(f"\n=== {msg} " + "=" * max(0, 60 - len(msg)))


def resolve_frozen_benchmark(c: httpx.Client, kb_id: str, bv_id: str) -> dict:
    r = c.get(f"/api/knowledge-bases/{kb_id}/benchmark-versions/{bv_id}")
    if r.status_code != 200:
        raise SystemExit(f"frozen benchmark version {bv_id!r} not found on {kb_id}: HTTP {r.status_code}")
    bv = r.json()
    if bv["status"] != "FROZEN":
        raise SystemExit(
            f"REFUSING to run: benchmark version {bv['version']!r} is {bv['status']}, not FROZEN. "
            "Official experiments require a human-reviewed, frozen benchmark (V3 Phase A rule)."
        )
    return bv


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path,
                    default=BENCH_DIR / "source-selection-experiment-v3.json")
    ap.add_argument("--baseline-kb", required=True, help="KB holding the frozen benchmark version")
    ap.add_argument("--benchmark-version", required=True, help="FROZEN BenchmarkVersion id")
    ap.add_argument("--seeds", default="20260915,20260916,20260917,20260918,20260919")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--ns", default="5,8,11")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate config + print deterministic selections; build nothing")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    exp = json.loads(args.experiment.read_text(encoding="utf-8"))
    frozen_pool: list[str] = load_pool(exp, args.experiment)
    c = httpx.Client(base_url=args.base, timeout=600.0)

    banner("0. Frozen benchmark gate")
    bv = resolve_frozen_benchmark(c, args.baseline_kb, args.benchmark_version)
    print(f"benchmark: {bv['version']} ({bv['id']}) status=FROZEN questions={len(bv['question_ids'])}")
    snap = bv["questions_snapshot"]
    chunk_gt = {q["id"]: q.get("expected_chunk_ids", []) for q in snap}
    doc_gt = {q["id"]: q.get("expected_document_ids", []) for q in snap}

    banner("1. v2 content-aware scoring of candidate pool (deterministic)")
    embedder, scorer, map_info = build_domain_scorer()
    records = score_pool(frozen_pool, embedder, scorer)
    ranked = sorted(records, key=lambda rec: (-rec["score"], rec["url"]))
    uniq = len({rec["signals"]["content_relevance"] for rec in records})
    print(f"scored {len(records)} URLs; unique content_relevance: {uniq}/{len(records)}")
    if uniq < len(records) // 2:
        print("STOP: scorer degenerate on this pool (v1 failure mode); not proceeding.")
        return 2

    def quality_selection(n: int) -> list[str]:
        return [rec["url"] for rec in ranked[:n]]

    def random_selection(n: int, seed: int) -> list[str]:
        return random.Random(seed).sample(frozen_pool, n)

    banner("2. Deterministic selection check")
    for seed in seeds:
        sel = random_selection(5, seed)
        print(f"  seed {seed} N=5 -> {sorted(u.rsplit('/', 1)[-1] for u in sel)}")
    print(f"  QUALITY N=5 -> {sorted(u.rsplit('/', 1)[-1] for u in quality_selection(5))}")

    if args.dry_run:
        print("\nDRY RUN OK — configuration, gate, scoring and selection determinism validated. "
              "Nothing was built. Run without --dry-run once a frozen benchmark version exists.")
        return 0

    # ------------------------------------------------------------------
    # 3. Corpus construction per (strategy, N) — QUALITY once + one arm/seed.
    #    Full GT translation + eval protocol mirrors v2 (per-corpus content_hash
    #    mapping; doc GT via source URL); evaluation targets the frozen
    #    benchmark version via config.benchmark_version.
    # ------------------------------------------------------------------
    runs: list[dict] = []
    chunk_cfg = {"chunker": "section-aware", "target_size": 1200, "overlap": 150}
    arms = [("QUALITY_SELECTED", None, n) for n in ns]
    for seed in seeds:
        arms += [("RANDOM_BASELINE", seed, n) for n in ns]

    baseline_chunks = c.get(f"/api/knowledge-bases/{args.baseline_kb}/chunks", params={"limit": 500}).json()
    chunk_to_hash = {ch["id"]: ch["content_hash"] for ch in baseline_chunks}

    for strategy, seed, n in arms:
        urls = (quality_selection(n) if strategy == "QUALITY_SELECTED" else random_selection(n, seed))
        banner(f"3. {strategy} N={n}" + (f" (seed {seed})" if seed else ""))
        r = c.post(
            "/api/knowledge-bases",
            json={
                "name": f"SourceSelV3 {strategy} N={n}" + (f" (seed {seed})" if seed else ""),
                "domain": "Automobile Engineering",
                "purpose": "v3 experiment corpus (frozen-benchmark evaluation)",
                "target_audience": "Undergraduate automotive engineering students",
                "depth": "technical",
            },
        )
        r.raise_for_status()
        kb_id = r.json()["id"]
        c.post(f"/api/knowledge-bases/{kb_id}/analyze-domain").raise_for_status()
        r = c.post(
            f"/api/knowledge-bases/{kb_id}/discover-sources",
            json={"provider": "user-url", "query": "\n".join(urls), "limit": len(urls) + 5},
        )
        r.raise_for_status()
        sources = [s for s in r.json() if s["url"] in set(urls)]
        selected, selected_ids = [], []
        for s in sorted(sources, key=lambda x: x["url"]):
            if (s.get("notes") or "").startswith("INVALID"):
                print(f"  INVALID URL in selection: {s['url']} (excluded honestly)")
                continue
            c.post(f"/api/knowledge-bases/{kb_id}/sources/{s['id']}/decision",
                   json={"decision": "ACCEPT"}).raise_for_status()
            rec = next(rec for rec in ranked if rec["url"] == s["url"])
            selected.append({
                "url": s["url"], "source_id": s["id"], "v2_composite": rec["score"],
                "v2_signals": rec["signals"], "v2_confidence": rec["confidence"],
            })
            selected_ids.append(s["id"])

        t0 = time.perf_counter()
        c.post(f"/api/knowledge-bases/{kb_id}/ingest", json={"source_ids": selected_ids}).raise_for_status()
        ingest_s = round(time.perf_counter() - t0, 2)
        docs = c.get(f"/api/knowledge-bases/{kb_id}/documents").json()
        t0 = time.perf_counter()
        index_run = c.post(f"/api/knowledge-bases/{kb_id}/index", json=chunk_cfg).raise_for_status().json()
        index_s = round(time.perf_counter() - t0, 2)
        chunks = []
        for d in docs:
            chunks += c.get(f"/api/knowledge-bases/{kb_id}/chunks",
                            params={"document_id": d["id"], "limit": 500}).json()
        url_to_doc = {d["url"]: d["id"] for d in docs}
        hash_to_chunk: dict[str, str] = {}
        for ch in chunks:
            hash_to_chunk.setdefault(ch["content_hash"], ch["id"])

        answerable: list[str] = []
        loaded_ids: list[str] = []
        for q in snap:
            translated = [hash_to_chunk[h] for h in
                          (chunk_to_hash.get(cid) for cid in chunk_gt.get(q["id"], [])) if h in hash_to_chunk]
            doc_translated = [url_to_doc[c.get(f"/api/knowledge-bases/{args.baseline_kb}/documents/{did}").json()["url"]]
                              for did in doc_gt.get(q["id"], []) if did]
            if not translated:
                continue
            answerable.append(q["id"])
            note = (f"[v3 exp] GT-translated from frozen benchmark {bv['version']} via content_hash. "
                    f"Original provenance: {q.get('provenance')}")
            r = c.post(
                f"/api/knowledge-bases/{kb_id}/evaluation-questions",
                json={"question": q["question"], "expected_chunk_ids": translated,
                      "expected_document_ids": [d for d in doc_translated if d], "notes": note,
                      "author": f"v3-runner (frozen:{bv['version']})"},
            )
            r.raise_for_status()
            loaded_ids.append(r.json()["id"])

        evals = {}
        for k in (3, 5, 10):
            if not loaded_ids:
                evals[str(k)] = None
                continue
            run = c.post(
                f"/api/knowledge-bases/{kb_id}/evaluate",
                json={"top_k": k, "question_ids": loaded_ids, "allow_keyword_fallback": False,
                      "label": f"{exp['experiment']} {strategy} N={n}"
                      + (f" seed={seed}" if seed else "") + f" k={k} strict",
                      "benchmark_version": bv["id"]},
            ).raise_for_status().json()
            evals[str(k)] = {"run_id": run["id"], "aggregate": run["aggregate"],
                             "per_question": run["per_question"]}
            a = run["aggregate"]
            print(f"  k={k}: R={a['recall_at_k']} P={a['precision_at_k']} MRR={a['mrr']} NDCG={a['ndcg']}")

        runs.append({
            "strategy": strategy, "n_sources_requested": n,
            "n_sources_selected": len(selected_ids), "random_seed": seed, "kb_id": kb_id,
            "selected_sources": selected, "documents": len(docs), "chunks": len(chunks),
            "corpus_chars": sum(d["text_length"] for d in docs),
            "ingest_seconds": ingest_s, "index_seconds": index_s,
            "chunking_config": chunk_cfg,
            "embedding_model": index_run["stages"][1]["message"],
            "vector_backend": "qdrant",
            "benchmark_version_id": bv["id"],
            "answerable_question_ids": answerable,
            "coverage": {"answerable": len(answerable), "total": len(snap)},
            "evaluation": evals,
        })

    out = BENCH_DIR / "source-selection-experiment-v3-results.json"
    out.write_text(json.dumps({
        "experiment": exp["experiment"], "status": "COMPLETE",
        "frozen_benchmark": {"id": bv["id"], "version": bv["version"]},
        "seeds": seeds, "scorer": {"version": scorer.version, "domain_map": map_info},
        "pool_ranking": [{"url": rec["url"], "composite": rec["score"]} for rec in ranked],
        "runs": runs,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
