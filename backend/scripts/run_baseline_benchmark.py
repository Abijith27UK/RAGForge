"""Load and run the Automobile Engineering baseline benchmark (strict ground truth only).

Workflow:
  1. Load benchmark JSON; VALIDATE every expected_chunk_id / expected_document_id
     against the live KB (refuses to run if any ground truth is stale — the KB
     must be re-indexed with the same config first).
  2. Load the questions into the KB through the normal API (provenance in notes).
  3. Run STRICT evaluations (allow_keyword_fallback=False) at k=3,5,10.
  4. Optionally run a separate DIAGNOSTIC keyword run (never merged into headline).
  5. Save results JSON next to the benchmark for strategy comparison.

Usage:
  python scripts/run_baseline_benchmark.py --benchmark ../benchmarks/automobile-engineering-baseline-v1.json --kb kb_f278c283c748
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--kb", required=True)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--ks", default="3,5,10", help="comma-separated top_k values")
    ap.add_argument("--skip-load", action="store_true", help="questions already loaded")
    ap.add_argument("--diagnostic-too", action="store_true", help="also run a keyword diagnostic run")
    ap.add_argument("--out", type=Path, default=None, help="results JSON path")
    args = ap.parse_args()

    bench = json.loads(args.benchmark.read_text(encoding="utf-8"))
    c = httpx.Client(base_url=args.base, timeout=300.0)
    kb = args.kb

    # ---- 1. Validate ground truth against the live KB ------------------------
    chunks = c.get(f"/api/knowledge-bases/{kb}/chunks", params={"limit": 1000}).json()
    # list_chunks caps at 500 per page; fetch remaining pages by document.
    if len(chunks) >= 500:
        by_doc: dict[str, list[dict]] = {}
        for d in c.get(f"/api/knowledge-bases/{kb}/documents").json():
            by_doc[d["id"]] = c.get(
                f"/api/knowledge-bases/{kb}/chunks", params={"document_id": d["id"], "limit": 500}
            ).json()
        chunks = [ch for chs in by_doc.values() for ch in chs]
    chunk_ids = {ch["id"] for ch in chunks}
    doc_ids = {d["id"] for d in c.get(f"/api/knowledge-bases/{kb}/documents").json()}

    errors: list[str] = []
    for q in bench["questions"]:
        for cid in q["expected_chunk_ids"]:
            if cid not in chunk_ids:
                errors.append(f"{q['id']}: chunk {cid} not in KB (stale ground truth)")
        for did in q["expected_document_ids"]:
            if did not in doc_ids:
                errors.append(f"{q['id']}: document {did} not in KB (stale ground truth)")
        if not q["expected_chunk_ids"] and not q["expected_document_ids"]:
            errors.append(f"{q['id']}: no explicit ground truth (headline benchmark requires IDs)")
    if errors:
        print("GROUND-TRUTH VALIDATION FAILED — refusing to run:")
        for e in errors:
            print("  -", e)
        return 2
    print(f"Ground truth validated: {len(bench['questions'])} questions, all IDs exist in KB.")

    # ---- 2. Load questions through the API -----------------------------------
    if not args.skip_load:
        existing = c.get(f"/api/knowledge-bases/{kb}/evaluation-questions").json()
        existing_by_notes = {e["question"]: e for e in existing}
        loaded = 0
        for q in bench["questions"]:
            if q["question"] in existing_by_notes:
                continue
            r = c.post(
                f"/api/knowledge-bases/{kb}/evaluation-questions",
                json={
                    "question": q["question"],
                    "expected_chunk_ids": q["expected_chunk_ids"],
                    "expected_document_ids": q["expected_document_ids"],
                    "expected_keywords": q.get("expected_keywords", []),
                    "notes": f"[{q['id']}] {q['provenance']}",
                },
            )
            r.raise_for_status()
            loaded += 1
        print(f"Loaded {loaded} new questions (skipped {len(bench['questions']) - loaded} already present).")

    # ---- 3. Strict evaluation runs -------------------------------------------
    all_q = c.get(f"/api/knowledge-bases/{kb}/evaluation-questions").json()
    by_question_text = {q["question"]: q["id"] for q in all_q}
    bench_q_ids = [by_question_text[q["question"]] for q in bench["questions"] if q["question"] in by_question_text]

    tag = bench["benchmark"]
    runs = []
    for k in [int(x) for x in args.ks.split(",")]:
        label = f"{tag} strict k={k}"
        r = c.post(
            f"/api/knowledge-bases/{kb}/evaluate",
            json={
                "top_k": k,
                "question_ids": bench_q_ids,
                "allow_keyword_fallback": False,
                "label": label,
            },
        )
        r.raise_for_status()
        run = r.json()
        agg = run["aggregate"]
        runs.append(run)
        print(
            f"{label}: Recall@{k}={agg['recall_at_k']} P@{k}={agg['precision_at_k']} "
            f"MRR={agg['mrr']} NDCG={agg['ndcg']} "
            f"(explicit={agg['questions_with_explicit_gt']}, skipped={agg['questions_skipped_no_gt']})"
        )

    # ---- 4. Optional diagnostic keyword run ----------------------------------
    diag = None
    if args.diagnostic_too:
        # Keyword diagnostic needs keyword-bearing variants; the benchmark itself
        # carries none (explicit IDs only), so this only makes sense if the KB
        # also contains separate keyword-authored questions. Reported separately.
        r = c.post(
            f"/api/knowledge-bases/{kb}/evaluate",
            json={
                "top_k": 5,
                "allow_keyword_fallback": True,
                "label": f"{tag} keyword-diagnostic k=5 (NOT headline)",
            },
        )
        r.raise_for_status()
        diag = r.json()
        print(f"diagnostic run saved separately (run id {diag['id']}); NOT part of headline results.")

    # ---- 5. Save results ------------------------------------------------------
    out = args.out or (args.benchmark.parent / f"{args.benchmark.stem}-results.json")
    payload = {
        "benchmark": tag,
        "kb_id": kb,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": runs[0]["embedding_model"] if runs else "",
        "retrieval_backend": runs[0]["retrieval_backend"] if runs else "",
        "strict_runs": [
            {
                "run_id": r["id"],
                "label": r["aggregate"]["run_label"],
                "top_k": r["config"]["top_k"],
                "aggregate": r["aggregate"],
                "per_question": [
                    {
                        "benchmark_id": next(
                            (q["id"] for q in bench["questions"] if q["question"] == p["question"]), None
                        ),
                        "question": p["question"],
                        "recall_at_k": p["recall_at_k"],
                        "precision_at_k": p["precision_at_k"],
                        "mrr": p["mrr"],
                        "ndcg": p["ndcg"],
                        "doc_recall_at_k": p["doc_recall_at_k"],
                        "note": p["note"],
                    }
                    for p in r["per_question"]
                ],
            }
            for r in runs
        ],
        "diagnostic_run_id": diag["id"] if diag else None,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
