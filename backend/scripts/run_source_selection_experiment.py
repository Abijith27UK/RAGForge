"""Source-selection experiment runner (source-selection-experiment-v1).

Research question: does explainable source-quality-based selection produce a
better and/or more efficient domain-specific RAG knowledge base than an
equivalent random-source baseline?

Everything runs through the LIVE API (same pipeline as the frozen baseline):
real discovery + quality scoring, real ingestion, real chunking/embedding/
Qdrant indexing, real retrieval, real evaluation in strict mode against the
FROZEN automobile-engineering-baseline-v1 questions.

Ground-truth translation: re-chunking regenerates chunk IDs, so the frozen
benchmark's expected_chunk_ids are translated per corpus via deterministic
chunk content_hash (same chunker config + same document text => same chunk
hash). Document-level GT is translated via source URL. The benchmark file is
never modified. Coverage (which questions are answerable in a corpus) is
reported separately and never mixed into retrieval metrics.

Usage:
  python scripts/run_source_selection_experiment.py \
      --experiment ../benchmarks/source-selection-experiment-v1.json \
      --baseline-kb kb_f278c283c748 \
      [--strategies QUALITY_SELECTED,RANDOM_BASELINE] [--ns 5,8,11]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx

BENCH_PATH = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "automobile-engineering-baseline-v1.json"


def banner(msg: str) -> None:
    print(f"\n=== {msg} " + "=" * max(0, 62 - len(msg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, type=Path)
    ap.add_argument("--baseline-kb", required=True, help="KB holding the frozen baseline corpus (for GT hash translation)")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--strategies", default="QUALITY_SELECTED,RANDOM_BASELINE")
    ap.add_argument("--ns", default="5,8,11")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args_ns = [int(x) for x in args.ns.split(",")]
    args_strategies = args.strategies.split(",")

    exp = json.loads(args.experiment.read_text(encoding="utf-8"))
    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    pool: list[str] = exp["candidate_pool"]["urls"]
    seed: int = exp["random_seed"]
    chunk_cfg = {"chunker": "section-aware", "target_size": 1200, "overlap": 150}
    c = httpx.Client(base_url=args.base, timeout=600.0)

    # ------------------------------------------------------------------
    # 0. Frozen GT: chunk_id -> content_hash (from the baseline KB), and
    #    baseline doc_id -> URL (for doc-level translation).
    # ------------------------------------------------------------------
    banner("0. Frozen ground-truth hashes from baseline KB")
    baseline_chunks = c.get(f"/api/knowledge-bases/{args.baseline_kb}/chunks", params={"limit": 500}).json()
    if len(baseline_chunks) >= 500:
        per_doc = []
        for d in c.get(f"/api/knowledge-bases/{args.baseline_kb}/documents").json():
            per_doc += c.get(
                f"/api/knowledge-bases/{args.baseline_kb}/chunks",
                params={"document_id": d["id"], "limit": 500},
            ).json()
        baseline_chunks = per_doc
    gt_chunk_hash: dict[str, dict[str, str]] = {}  # bench q id -> {chunk_id: content_hash}
    chunk_to_hash: dict[str, str] = {ch["id"]: ch["content_hash"] for ch in baseline_chunks}
    for q in bench["questions"]:
        gt_chunk_hash[q["id"]] = {
            cid: chunk_to_hash[cid] for cid in q["expected_chunk_ids"] if cid in chunk_to_hash
        }
        missing = set(q["expected_chunk_ids"]) - set(chunk_to_hash)
        if missing:
            print(f"  WARNING {q['id']}: chunk ids missing in baseline KB: {missing}")
    baseline_docs = c.get(f"/api/knowledge-bases/{args.baseline_kb}/documents").json()
    gt_doc_url: dict[str, dict[str, str]] = {}  # bench q id -> {baseline_doc_id: url}
    doc_url_by_id = {d["id"]: d["url"] for d in baseline_docs}
    for q in bench["questions"]:
        gt_doc_url[q["id"]] = {did: doc_url_by_id[did] for did in q["expected_document_ids"] if did in doc_url_by_id}
    print(f"frozen GT: {len(bench['questions'])} questions; chunk hashes resolved for "
          f"{sum(1 for v in gt_chunk_hash.values() if v) } questions")

    # ------------------------------------------------------------------
    # 1. QUALITY ranking: score the whole pool in a staging KB via the real
    #    discovery + quality pipeline (probes included). Deterministic mock
    #    DomainSpec => same relevance terms as every other KB.
    # ------------------------------------------------------------------
    banner("1. Quality scoring of candidate pool (staging KB, real pipeline)")
    r = c.post(
        "/api/knowledge-bases",
        json={
            "name": "SourceSel staging (candidate pool scoring)",
            "domain": "Automobile Engineering",
            "purpose": "Course-reference knowledge base for EV and vehicle-dynamics engineering education",
            "target_audience": "Undergraduate automotive engineering students",
            "depth": "technical",
        },
    )
    r.raise_for_status()
    staging_id = r.json()["id"]
    c.post(f"/api/knowledge-bases/{staging_id}/analyze-domain").raise_for_status()
    r = c.post(
        f"/api/knowledge-bases/{staging_id}/discover-sources",
        json={"provider": "user-url", "query": "\n".join(pool), "limit": len(pool) + 5},
    )
    r.raise_for_status()
    pool_sources = r.json()
    score_by_url = {s["url"]: s["trust_score"] for s in pool_sources if not (s.get("notes") or "").startswith("INVALID")}
    decision_by_url = {s["url"]: s["decision"] for s in pool_sources}
    print(f"scored {len(score_by_url)}/{len(pool)} pool URLs")
    ranked = sorted(pool, key=lambda u: (-(score_by_url.get(u, 0.0)), u))

    def quality_selection(n: int) -> list[str]:
        return ranked[:n]

    def random_selection(n: int) -> list[str]:
        return random.Random(seed).sample(pool, n)

    selections: dict[str, dict[int, list[str]]] = {
        "QUALITY_SELECTED": {n: quality_selection(n) for n in args_ns},
        "RANDOM_BASELINE": {n: random_selection(n) for n in args_ns},
    }

    # ------------------------------------------------------------------
    # 2. Build one corpus per (strategy, N) and evaluate.
    # ------------------------------------------------------------------
    results: list[dict] = []
    for strategy in args_strategies:
        for n in args_ns:
            urls = selections[strategy][n]
            banner(f"2. Corpus {strategy} N={n} (seed {seed})")
            r = c.post(
                "/api/knowledge-bases",
                json={
                    "name": f"SourceSel {strategy} N={n} (seed {seed})",
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
                c.post(f"/api/knowledge-bases/{kb_id}/sources/{s['id']}/decision", json={"decision": "ACCEPT"}).raise_for_status()
                selected_records.append({"url": s["url"], "source_id": s["id"], "quality_score": s["trust_score"], "automated_decision": s["decision"]})
                selected_ids.append(s["id"])
            print(f"selected {len(selected_ids)}/{n} sources (scores: "
                  f"{[round(x['quality_score'], 3) for x in selected_records]})")

            t0 = time.perf_counter()
            r = c.post(f"/api/knowledge-bases/{kb_id}/ingest", json={"source_ids": selected_ids})
            r.raise_for_status()
            ingest_s = round(time.perf_counter() - t0, 2)
            ingest_msg = r.json()["stages"][0]["message"]
            docs = c.get(f"/api/knowledge-bases/{kb_id}/documents").json()
            doc_urls = {d["url"] for d in docs}

            t0 = time.perf_counter()
            r = c.post(f"/api/knowledge-bases/{kb_id}/index", json=chunk_cfg)
            r.raise_for_status()
            index_s = round(time.perf_counter() - t0, 2)
            index_run = r.json()
            chunks = []
            for d in docs:
                chunks += c.get(f"/api/knowledge-bases/{kb_id}/chunks", params={"document_id": d["id"], "limit": 500}).json()
            corpus_chars = sum(d["text_length"] for d in docs)
            print(f"ingest {ingest_s}s ({ingest_msg}); index {index_s}s; docs={len(docs)} chunks={len(chunks)} chars={corpus_chars}")

            # ---- GT translation ----------------------------------------
            hash_to_chunk: dict[str, str] = {}
            for ch in chunks:
                hash_to_chunk.setdefault(ch["content_hash"], ch["id"])
            url_to_doc: dict[str, str] = {d["url"]: d["id"] for d in docs}
            loaded_ids: list[str] = []
            coverage = {"answerable": 0, "total": len(bench["questions"]), "excluded_unanswerable": []}
            for q in bench["questions"]:
                hashes = gt_chunk_hash.get(q["id"], {})
                translated = [hash_to_chunk[h] for h in hashes.values() if h in hash_to_chunk]
                doc_translated = [url_to_doc[u] for u in gt_doc_url.get(q["id"], {}).values() if u in url_to_doc]
                if not translated:
                    coverage["excluded_unanswerable"].append(q["id"])
                    continue
                coverage["answerable"] += 1
                note = (
                    f"[{q['id']}] GT-translated from frozen benchmark via content_hash "
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
            print(f"coverage: {coverage['answerable']}/{coverage['total']} questions answerable "
                  f"(excluded: {coverage['excluded_unanswerable'] or 'none'})")

            # ---- Strict evaluation -------------------------------------
            evals = {}
            if not loaded_ids:
                print("  no answerable questions -> evaluation skipped (recorded as null, not zero)")
            for k in (3, 5, 10):
                if not loaded_ids:
                    evals[str(k)] = {
                        "run_id": None,
                        "recall_at_k": None,
                        "precision_at_k": None,
                        "mrr": None,
                        "ndcg": None,
                        "doc_recall_at_k": None,
                        "questions_evaluated": 0,
                        "note": "no answerable questions in this corpus; evaluation not run",
                    }
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
                evals[str(k)] = {
                    "run_id": run["id"],
                    "recall_at_k": run["aggregate"]["recall_at_k"],
                    "precision_at_k": run["aggregate"]["precision_at_k"],
                    "mrr": run["aggregate"]["mrr"],
                    "ndcg": run["aggregate"]["ndcg"],
                    "doc_recall_at_k": run["aggregate"]["doc_recall_at_k"],
                    "questions_evaluated": run["aggregate"]["questions_evaluated"],
                }
                a = run["aggregate"]
                print(f"  k={k}: R={a['recall_at_k']} P={a['precision_at_k']} MRR={a['mrr']} NDCG={a['ndcg']}")

            results.append(
                {
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
                }
            )

    # ------------------------------------------------------------------
    # 3. Persist results (merge with any previous partial runs).
    # ------------------------------------------------------------------
    out = args.out or args.experiment.parent / (args.experiment.stem + "-results.json")
    existing = {"experiment": exp["experiment"], "runs": []}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            pass
    by_key = {f"{r_['strategy']}|{r_['n_sources_requested']}": r_ for r_ in existing.get("runs", [])}
    for r_ in results:
        by_key[f"{r_['strategy']}|{r_['n_sources_requested']}"] = r_
    payload = {
        "experiment": exp["experiment"],
        "research_question": exp["research_question"],
        "random_seed": seed,
        "frozen_baseline_caveat": exp["frozen_baseline"]["ground_truth_caveat"],
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "staging_kb_id": staging_id,
        "pool_ranking": [
            {"url": u, "quality_score": score_by_url.get(u), "automated_decision": decision_by_url.get(u)}
            for u in ranked
        ],
        "runs": sorted(by_key.values(), key=lambda x: (x["strategy"], x["n_sources_requested"])),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
