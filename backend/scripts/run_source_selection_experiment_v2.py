"""Source-selection experiment v2 runner.

Same research question and protocol as v1 (run_source_selection_experiment.py),
with one independent-variable change: the QUALITY_SELECTED ranking now uses the
v2 content-aware scorer (heuristic-v2-content-aware) instead of the v1 metadata
scorer that scored 34/36 pool sources identically.

Protocol is deliberately identical to v1: real discovery+probes, real ingest,
same chunking config, same embedding model, Qdrant, strict evaluation of the
FROZEN automobile-engineering-baseline-v1 questions translated via chunk
content_hash. Coverage is reported separately; cross-run comparisons are only
valid on common answerable-question subsets (reported explicitly).

Usage:
  python scripts/run_source_selection_experiment_v2.py \
      --experiment ../benchmarks/source-selection-experiment-v2.json \
      --baseline-kb kb_f278c283c748
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_v2_pool_scores import (  # noqa: E402
    build_domain_scorer,
    load_pool,
    score_pool,
)

BENCH_PATH = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "automobile-engineering-baseline-v1.json"


def banner(msg: str) -> None:
    print(f"\n=== {msg} " + "=" * max(0, 62 - len(msg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, type=Path)
    ap.add_argument("--baseline-kb", required=True)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--strategies", default="QUALITY_SELECTED,RANDOM_BASELINE")
    ap.add_argument("--ns", default="5,8,11")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--only",
        default=None,
        help="resume mode: 'STRATEGY|N' to run a single corpus and merge into the results file",
    )
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    strategies = args.strategies.split(",")

    exp = json.loads(args.experiment.read_text(encoding="utf-8"))
    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    pool: list[str] = load_pool(exp, args.experiment)
    seed: int = exp["random_seed"]
    chunk_cfg = {"chunker": "section-aware", "target_size": 1200, "overlap": 150}
    c = httpx.Client(base_url=args.base, timeout=600.0)

    # ------------------------------------------------------------------
    # 0. Frozen GT hashes from the baseline KB (identical to v1 runner).
    # ------------------------------------------------------------------
    banner("0. Frozen ground-truth hashes from baseline KB")
    baseline_chunks = c.get(f"/api/knowledge-bases/{args.baseline_kb}/chunks", params={"limit": 500}).json()
    if len(baseline_chunks) >= 500:
        baseline_chunks = []
        for d in c.get(f"/api/knowledge-bases/{args.baseline_kb}/documents").json():
            baseline_chunks += c.get(
                f"/api/knowledge-bases/{args.baseline_kb}/chunks",
                params={"document_id": d["id"], "limit": 500},
            ).json()
    chunk_to_hash = {ch["id"]: ch["content_hash"] for ch in baseline_chunks}
    gt_chunk_hash = {}
    for q in bench["questions"]:
        gt_chunk_hash[q["id"]] = {cid: chunk_to_hash[cid] for cid in q["expected_chunk_ids"] if cid in chunk_to_hash}
    baseline_docs = c.get(f"/api/knowledge-bases/{args.baseline_kb}/documents").json()
    doc_url_by_id = {d["id"]: d["url"] for d in baseline_docs}
    gt_doc_url = {q["id"]: {did: doc_url_by_id[did] for did in q["expected_document_ids"] if did in doc_url_by_id}
                  for q in bench["questions"]}
    print(f"frozen GT: {len(bench['questions'])} questions; chunk hashes resolved for "
          f"{sum(1 for v in gt_chunk_hash.values() if v)} questions")

    # ------------------------------------------------------------------
    # 1. QUALITY ranking with the v2 content-aware scorer (cached content).
    # ------------------------------------------------------------------
    banner("1. v2 content-aware scoring of candidate pool")
    embedder, scorer, map_info = build_domain_scorer()
    records = score_pool(pool, embedder, scorer)
    ranked = sorted(
        records,
        key=lambda rec: (-rec["score"], rec["url"]),
    )
    score_by_url = {rec["url"]: rec["score"] for rec in ranked}
    signals_by_url = {rec["url"]: rec["signals"] for rec in ranked}
    conf_by_url = {rec["url"]: rec["confidence"] for rec in ranked}
    areas_by_url = {rec["url"]: [p for p in rec["per_requirement"] if p["covered"]] for rec in ranked}
    uniq_rel = len({rec["signals"]["content_relevance"] for rec in records})
    print(f"scored {len(records)} URLs; unique content_relevance: {uniq_rel}/{len(records)}")
    if uniq_rel < len(records) // 2:
        print("  STOP: scorer still degenerate on this pool; not proceeding (see stop condition).")
        return 2
    for rec in ranked[:12]:
        cov = ",".join(p["area_id"] for p in rec["per_requirement"] if p["covered"]) or "-"
        print(f"  {rec['score']:.4f} rel={rec['signals']['content_relevance']:.3f} "
              f"cov={rec['signals']['domain_coverage']:.2f} [{cov}] {rec['url'].rsplit('/', 1)[-1]}")

    def quality_selection(n: int) -> list[str]:
        return [rec["url"] for rec in ranked[:n]]

    def random_selection(n: int) -> list[str]:
        return random.Random(seed).sample(pool, n)

    selections = {s: {n: (quality_selection(n) if s == "QUALITY_SELECTED" else random_selection(n)) for n in ns}
                  for s in strategies}

    if args.only:
        s_filter, n_filter = args.only.split("|")
        n_filter = int(n_filter)
        if s_filter not in selections or n_filter not in selections[s_filter]:
            print(f"no such run: {args.only}"); return 1
        strategies, ns = [s_filter], [n_filter]

    # ------------------------------------------------------------------
    # 2. Build one corpus per (strategy, N) and evaluate (v1 protocol).
    # ------------------------------------------------------------------
    results: list[dict] = []
    for strategy in strategies:
        for n in ns:
            urls = selections[strategy][n]
            banner(f"2. Corpus {strategy} N={n} (seed {seed})")
            r = c.post(
                "/api/knowledge-bases",
                json={
                    "name": f"SourceSelV2 {strategy} N={n} (seed {seed})",
                    "domain": "Automobile Engineering",
                    "purpose": "Course-reference knowledge base for EV and vehicle-dynamics engineering education",
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
            selected_records = []
            selected_ids = []
            for s in sorted(sources, key=lambda x: x["url"]):
                if (s.get("notes") or "").startswith("INVALID"):
                    print(f"  INVALID URL in selection: {s['url']} (excluded honestly)")
                    continue
                c.post(f"/api/knowledge-bases/{kb_id}/sources/{s['id']}/decision",
                       json={"decision": "ACCEPT"}).raise_for_status()
                selected_records.append({
                    "url": s["url"],
                    "source_id": s["id"],
                    "v2_composite": score_by_url.get(s["url"]),
                    "v2_signals": signals_by_url.get(s["url"]),
                    "v2_confidence": conf_by_url.get(s["url"]),
                    "v2_covered_areas": sorted(p["area_id"] for p in areas_by_url.get(s["url"], [])),
                    "pipeline_decision": s["decision"],
                })
                selected_ids.append(s["id"])
            print(f"selected {len(selected_ids)}/{n} sources "
                  f"(composites: {[round(x['v2_composite'], 3) for x in selected_records]})")

            t0 = time.perf_counter()
            r = c.post(f"/api/knowledge-bases/{kb_id}/ingest", json={"source_ids": selected_ids})
            r.raise_for_status()
            ingest_s = round(time.perf_counter() - t0, 2)
            ingest_msg = r.json()["stages"][0]["message"]
            docs = c.get(f"/api/knowledge-bases/{kb_id}/documents").json()

            t0 = time.perf_counter()
            r = c.post(f"/api/knowledge-bases/{kb_id}/index", json=chunk_cfg)
            r.raise_for_status()
            index_s = round(time.perf_counter() - t0, 2)
            index_run = r.json()
            chunks = []
            for d in docs:
                chunks += c.get(f"/api/knowledge-bases/{kb_id}/chunks",
                                params={"document_id": d["id"], "limit": 500}).json()
            corpus_chars = sum(d["text_length"] for d in docs)
            print(f"ingest {ingest_s}s ({ingest_msg}); index {index_s}s; "
                  f"docs={len(docs)} chunks={len(chunks)} chars={corpus_chars}")

            # ---- GT translation ----------------------------------------
            hash_to_chunk: dict[str, str] = {}
            for ch in chunks:
                hash_to_chunk.setdefault(ch["content_hash"], ch["id"])
            url_to_doc = {d["url"]: d["id"] for d in docs}
            loaded_ids = []
            coverage = {"answerable": 0, "total": len(bench["questions"]), "excluded_unanswerable": []}
            for q in bench["questions"]:
                translated = [hash_to_chunk[h] for h in gt_chunk_hash.get(q["id"], {}).values() if h in hash_to_chunk]
                doc_translated = [url_to_doc[u] for u in gt_doc_url.get(q["id"], {}).values() if u in url_to_doc]
                if not translated:
                    coverage["excluded_unanswerable"].append(q["id"])
                    continue
                coverage["answerable"] += 1
                note = (
                    f"[v2 exp] GT-translated from frozen benchmark via content_hash "
                    f"({len(translated)}/{len(q['expected_chunk_ids'])} chunks matched). "
                    f"Original provenance: {q['provenance']}"
                )
                r = c.post(
                    f"/api/knowledge-bases/{kb_id}/evaluation-questions",
                    json={
                        "question": q["question"],
                        "expected_chunk_ids": translated,
                        "expected_document_ids": doc_translated,
                        "expected_keywords": [],
                        "notes": note,
                    },
                )
                r.raise_for_status()
                loaded_ids.append(r.json()["id"])
            print(f"coverage: {coverage['answerable']}/{coverage['total']} answerable")

            # ---- Strict evaluation -------------------------------------
            evals = {}
            for k in (3, 5, 10):
                if not loaded_ids:
                    evals[str(k)] = {"run_id": None, "recall_at_k": None, "precision_at_k": None,
                                     "mrr": None, "ndcg": None, "doc_recall_at_k": None,
                                     "questions_evaluated": 0,
                                     "note": "no answerable questions; evaluation not run"}
                    continue
                r = c.post(
                    f"/api/knowledge-bases/{kb_id}/evaluate",
                    json={
                        "top_k": k,
                        "question_ids": loaded_ids,
                        "allow_keyword_fallback": False,
                        "label": f"{exp['experiment']} {strategy} N={n} k={k} strict",
                    },
                )
                r.raise_for_status()
                run = r.json()
                a = run["aggregate"]
                evals[str(k)] = {
                    "run_id": run["id"],
                    "recall_at_k": a["recall_at_k"],
                    "precision_at_k": a["precision_at_k"],
                    "mrr": a["mrr"],
                    "ndcg": a["ndcg"],
                    "doc_recall_at_k": a["doc_recall_at_k"],
                    "questions_evaluated": a["questions_evaluated"],
                }
                print(f"  k={k}: R={a['recall_at_k']} P={a['precision_at_k']} "
                      f"MRR={a['mrr']} NDCG={a['ndcg']}")

            results.append({
                "strategy": strategy,
                "n_sources_requested": n,
                "n_sources_selected": len(selected_ids),
                "kb_id": kb_id,
                "random_seed": seed if strategy == "RANDOM_BASELINE" else None,
                "selected_sources": selected_records,
                "documents": len(docs),
                "chunks": len(chunks),
                "corpus_chars": corpus_chars,
                "ingest_seconds": ingest_s,
                "index_seconds": index_s,
                "ingest_message": ingest_msg,
                "chunking_config": chunk_cfg,
                "embedding_model": index_run["stages"][1]["message"],
                "ground_truth_translation": {
                    "method": "content_hash chunk translation from frozen baseline; doc GT via source URL",
                    "questions_loaded": len(loaded_ids),
                },
                "coverage": coverage,
                "evaluation": evals,
            })

    # ------------------------------------------------------------------
    # 3. Persist + answerable-question intersections.
    # ------------------------------------------------------------------
    out = args.out or args.experiment.parent / (args.experiment.stem + "-results.json")
    # Merge with any previous partial runs (supports the --only resume mode).
    existing_runs: list[dict] = []
    if out.exists():
        try:
            existing_runs = json.loads(out.read_text(encoding="utf-8")).get("runs", [])
        except Exception:
            existing_runs = []
    by_key = {f"{r_['strategy']}|{r_['n_sources_requested']}": r_ for r_ in existing_runs}
    for r_ in results:
        by_key[f"{r_['strategy']}|{r_['n_sources_requested']}"] = r_
    merged = sorted(by_key.values(), key=lambda x: (x["strategy"], x["n_sources_requested"]))

    per_run_qs = {f"{r_['strategy']}|{r_['n_sources_requested']}": set(r_["coverage"]["excluded_unanswerable"]) for r_ in merged}
    total_q = len(bench["questions"])
    answerable_sets = {k: set(q["id"] for q in bench["questions"]) - v for k, v in per_run_qs.items()}
    inter_all = set.intersection(*answerable_sets.values()) if answerable_sets else set()
    pair_inter = {}
    keys = sorted(answerable_sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pair_inter[f"{keys[i]} & {keys[j]}"] = len(answerable_sets[keys[i]] & answerable_sets[keys[j]])
    payload = {
        "experiment": exp["experiment"],
        "research_question": exp["research_question"],
        "scorer": {"version": scorer.version, "domain_map": map_info},
        "random_seed": seed,
        "frozen_baseline_caveat": exp["frozen_baseline"]["ground_truth_caveat"],
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pool_ranking": [
            {"url": rec["url"], "v2_composite": rec["score"], "v2_signals": rec["signals"],
             "confidence": rec["confidence"]}
            for rec in ranked
        ],
        "answerable_question_intersection": {
            "note": "Cross-run metric comparisons are only valid on common answerable subsets.",
            "global_intersection_size": len(inter_all),
            "global_intersection_question_ids": sorted(inter_all),
            "pairwise_sizes": pair_inter,
            "per_run_answerable_sizes": {k: len(v) for k, v in answerable_sets.items()},
        },
        "runs": merged,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved: {out}")
    print(f"global answerable intersection: {len(inter_all)}/{total_q} questions "
          f"({', '.join(sorted(inter_all)) or 'EMPTY'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
