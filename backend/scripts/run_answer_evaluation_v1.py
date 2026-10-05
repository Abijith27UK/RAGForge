"""Run the answer-evaluation-v1 experiment and write its results artifact.

Reads `benchmarks/answer-evaluation-v1.json` (experiment SPEC — no ground
truth of its own), executes the shipping answering pipeline under each
strategy against the source benchmark, and writes
`benchmarks/answer-evaluation-v1-results.json`.

Safety contract (V8 STEP 18):
  * READ-ONLY on documents, chunks, Qdrant vectors and all frozen artifacts;
    it only asks questions and stores evaluation records.
  * the source benchmark fingerprint is verified BEFORE running and the
    artifact is never written to.
  * Qdrant collection count is captured before and after and asserted equal.
  * official=false: the source benchmark is lifecycle 'draft' / human_review
    'pending', so official results are refused by the service anyway — these
    numbers are an experiment record, stated as such in the output.
  * pairwise strategy comparisons must come back verdict COMPARABLE (all runs
    cover the identical 28-question subset); any other verdict fails the
    script rather than being reported as a comparison.

Requires a running backend (default http://localhost:8013) and Qdrant.

Run: python scripts/run_answer_evaluation_v1.py [base_url]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

SPEC_PATH = ROOT / "benchmarks" / "answer-evaluation-v1.json"
RESULTS_PATH = ROOT / "benchmarks" / "answer-evaluation-v1-results.json"
QDRANT = "http://localhost:6333/collections"

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8013").rstrip("/")

#: Headline metrics copied into the results artifact (aggregates only; the
#: full per-question records live in the immutable run rows).
REPORTED = (
    "citation_precision",
    "citation_recall",
    "evidence_support_rate",
    "unsupported_claim_ratio",
    "supported_claim_ratio",
    "fabricated_citation_rate",
    "unsupported_citation_rate",
    "contradiction_rate",
    "retrieval_hit_rate",
    "question_answer_relevance",
    "relevance_failure_rate",
    "abstention_accuracy",
    "grounding_state_accuracy",
    "hallucination_rate",
    "pass_rate",
)


def qdrant_count() -> int | None:
    try:
        return len(httpx.get(QDRANT, timeout=10).json()["result"]["collections"])
    except Exception:
        return None


def compact(metric: dict) -> dict:
    return {
        "value": metric.get("value"),
        "measured": metric.get("measured"),
        "sample_size": metric.get("sample_size"),
        "reason": metric.get("reason", ""),
    }


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    print(f"spec: {spec['experiment']} v{spec['version']} "
          f"(official={spec['official']})")

    # -- preflight: source benchmark identity ---------------------------------
    from app.services.answer_eval.benchmark import load_answer_benchmark

    source = load_answer_benchmark(ROOT / spec["source_benchmark"])
    fp = source.fingerprint()
    if fp != spec["source_benchmark_fingerprint"]:
        print(f"ABORT: source benchmark fingerprint changed "
              f"({fp} != {spec['source_benchmark_fingerprint']}); the spec "
              f"describes different content than what would be evaluated")
        return 1
    if source.kb_id != spec["kb_id"]:
        print(f"ABORT: source benchmark targets {source.kb_id}, spec says "
              f"{spec['kb_id']}")
        return 1
    print(f"source: {source.benchmark} v{source.version} fp={fp} "
          f"lifecycle={source.lifecycle.value} human_review={source.human_review.value}")
    print(f"questions: {source.question_count} "
          f"(answerable={source.answerable_count})")

    before_runs = len(httpx.get(
        f"{BASE}/api/answer-evaluation-runs",
        params={"kb_id": spec["kb_id"], "limit": 500}, timeout=30).json())
    q_before = qdrant_count()
    print(f"Qdrant collections before: {q_before}; "
          f"stored answer-eval runs before: {before_runs}")

    client = httpx.Client(base_url=BASE, timeout=3600.0)
    runs: list[dict] = []
    for strategy in spec["strategies"]:
        print(f"\n=== strategy: {strategy} ===")
        r = client.post(
            f"/api/knowledge-bases/{spec['kb_id']}/answer-evaluation/runs",
            json={
                "benchmark_path": spec["source_benchmark"],
                "strategy": strategy,
                "answer_mode": spec["answer_mode"],
                "official": False,
                "evaluator": "deterministic",
                "persist": True,
            },
        )
        if r.status_code != 200:
            print(f"FAILED {r.status_code}: {r.text[:500]}")
            return 1
        run = r.json()
        agg = run["aggregate"]
        print(f"  run {run['id']} questions={run['question_count']} "
              f"pass_rate={agg['pass_rate']['value']} "
              f"relevance={agg['question_answer_relevance']['value']} "
              f"relevance_failures={agg['relevance_failure_rate']['value']}")
        for name in REPORTED:
            m = agg[name]
            shown = f"{m['value']:.3f}" if m["measured"] and m["value"] is not None \
                else "UNKNOWN"
            print(f"    {name:28} {shown:>8}  n={m.get('sample_size')}")
        runs.append({
            "strategy": strategy,
            "run_id": run["id"],
            "created_at": run["created_at"],
            "question_count": run["question_count"],
            "failed_question_ids": run["failed_question_ids"],
            "aggregate": {name: compact(agg[name]) for name in REPORTED},
            "unknown_metrics": agg["unknown_metrics"],
            "relevance_method": run["relevance_method"],
            "relevance_weight_source": run["relevance_weight_source"],
            "generator": run["generator"],
            "model": run["model"],
            "is_mock": run["is_mock"],
            # Full run records carry name+version separately; the joined
            # "evaluator" string only exists on list summaries.
            "evaluator": f"{run['evaluator_name']} {run['evaluator_version']}",
            "evaluator_is_model_based": run["evaluator_is_model_based"],
            "benchmark_fingerprint": run["benchmark_fingerprint"],
            "benchmark_lifecycle": run["benchmark_lifecycle"],
            "official": run["official"],
            "warnings": run["warnings"],
        })

    # -- pairwise comparability (identical subsets required) -------------------
    comparisons: list[dict] = []
    valid = True
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            c = client.get(
                f"/api/knowledge-bases/{spec['kb_id']}/answer-evaluation/compare",
                params={"left": runs[i]["run_id"], "right": runs[j]["run_id"]},
            ).json()
            print(f"\ncompare {runs[i]['strategy']} vs {runs[j]['strategy']}: "
                  f"{c['verdict']} ({c['mode']}) — {c['reason'][:120]}")
            deltas = {
                name: d["delta"]
                for name, d in c["differences"].items()
                if d.get("delta") is not None
            }
            comparisons.append({
                "left_strategy": runs[i]["strategy"],
                "right_strategy": runs[j]["strategy"],
                "left_run_id": runs[i]["run_id"],
                "right_run_id": runs[j]["run_id"],
                "verdict": c["verdict"],
                "mode": c["mode"],
                "reason": c["reason"],
                "deltas": deltas,
            })
            if c["verdict"] != "COMPARABLE":
                valid = False

    q_after = qdrant_count()
    after_runs = len(httpx.get(
        f"{BASE}/api/answer-evaluation-runs",
        params={"kb_id": spec["kb_id"], "limit": 500}, timeout=30).json())

    results = {
        "artifact_type": "answer-evaluation-results",
        "experiment": spec["experiment"],
        "version": spec["version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_path": "benchmarks/answer-evaluation-v1.json",
        "kb_id": spec["kb_id"],
        "source_benchmark": spec["source_benchmark"],
        "source_benchmark_fingerprint": fp,
        "official": False,
        "official_reason": spec["official_reason"],
        "evaluator": spec["evaluator"],
        "generator_observed": {
            "generator": runs[0]["generator"],
            "model": runs[0]["model"],
            "is_mock": runs[0]["is_mock"],
        } if runs else {},
        "runs": runs,
        "comparisons": comparisons,
        "all_comparisons_valid": valid,
        "expected_unknown": spec["expected_unknown"],
        "data_impact": {
            "qdrant_collections_before": q_before,
            "qdrant_collections_after": q_after,
            "qdrant_unchanged": q_before == q_after,
            "answer_eval_runs_before": before_runs,
            "answer_eval_runs_after": after_runs,
            "frozen_artifacts_written": 0,
            "note": spec["data_impact"],
        },
        "reproduced_by": spec["reproduced_by"],
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Qdrant collections after: {q_after} "
          f"(unchanged={q_before == q_after})")
    print(f"answer-eval runs after: {after_runs} (+{after_runs - before_runs} "
          f"immutable rows on kb {spec['kb_id']})")

    if not valid:
        print("ABORT: a strategy comparison came back non-COMPARABLE; "
              "results written but experiment integrity check FAILED")
        return 1
    if q_before != q_after:
        print("ABORT: Qdrant collection count changed during the experiment")
        return 1
    print("experiment integrity: OK (all comparisons COMPARABLE, Qdrant unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
