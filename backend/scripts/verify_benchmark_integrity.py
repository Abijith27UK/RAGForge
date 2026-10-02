"""Frozen-benchmark integrity check + V4 source-selection v3 readiness (Phase 11).

READ-ONLY. This script never writes to the database and never modifies a
benchmark, a question, or a historical experiment artifact. Its only job is to
answer four questions honestly before an official experiment is allowed to run:

  1. Does the named BenchmarkVersion exist, and is it FROZEN?
  2. Is the frozen version recorded (id + version + frozen_at) so the
     experiment can reference the exact snapshot?
  3. Are all of its questions APPROVED/FROZEN, with a reviewer recorded?
  4. Would the v3 runner accept it (i.e. is `resolve_frozen_benchmark` happy)?

Usage:
    python backend/scripts/verify_benchmark_integrity.py \
        --kb-id kb_xxx --benchmark-version bv_xxx

    # also prepare (but NOT execute) the v3 experiment plan
    python backend/scripts/verify_benchmark_integrity.py \
        --kb-id kb_xxx --benchmark-version bv_xxx --prepare-v3

Exit codes:
    0 = frozen benchmark verified (and, with --prepare-v3, v3 is ready to run)
    2 = gate FAILED: the benchmark is not frozen / not reviewable
    3 = verification could not complete (API unreachable)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"

#: Seeds fixed by benchmarks/source-selection-experiment-v3.json.
V3_SEEDS = [20260915, 20260916, 20260917, 20260918, 20260919]
V3_CORPUS_SIZES = [5, 8, 11]
ALLOWED_STATUSES = {"APPROVED", "FROZEN"}


def banner(msg: str) -> None:
    print(f"\n=== {msg} " + "=" * max(0, 58 - len(msg)))


def verify(c: httpx.Client, kb_id: str, bv_id: str) -> tuple[bool, dict]:
    """Return (gate_passed, report). Purely read-only."""
    report: dict = {"kb_id": kb_id, "benchmark_version_id": bv_id, "checks": {}, "problems": []}

    r = c.get(f"/api/knowledge-bases/{kb_id}/benchmark-versions/{bv_id}")
    if r.status_code != 200:
        report["problems"].append(f"benchmark version {bv_id!r} not found on {kb_id} (HTTP {r.status_code})")
        return False, report
    bv = r.json()

    report["checks"]["exists"] = True
    report["version"] = bv.get("version")
    report["status"] = bv.get("status")
    report["frozen_at"] = bv.get("frozen_at")
    report["created_by"] = bv.get("created_by")
    report["question_count"] = len(bv.get("question_ids") or [])
    report["question_ids"] = list(bv.get("question_ids") or [])
    report["has_snapshot"] = bool(bv.get("questions_snapshot"))

    # 1. FROZEN status.
    if bv.get("status") != "FROZEN":
        report["problems"].append(
            f"benchmark is {bv.get('status')}, not FROZEN — official experiments require "
            "a human-reviewed, frozen benchmark"
        )
    report["checks"]["frozen"] = bv.get("status") == "FROZEN"

    # 2. The snapshot is what an experiment must score.
    if not report["has_snapshot"]:
        report["problems"].append("frozen version carries no questions_snapshot")
    report["checks"]["snapshot_present"] = report["has_snapshot"]

    # 3. Every snapshotted question is reviewed, and the reviewer is recorded.
    unreviewed: list[str] = []
    missing_reviewer: list[str] = []
    for q in bv.get("questions_snapshot") or []:
        if q.get("status") not in ALLOWED_STATUSES:
            unreviewed.append(f"{q.get('id')}({q.get('status')})")
        if not q.get("reviewer"):
            missing_reviewer.append(str(q.get("id")))
    report["checks"]["all_questions_reviewed"] = not unreviewed
    report["checks"]["all_reviewers_recorded"] = not missing_reviewer
    if unreviewed:
        report["problems"].append(f"questions not APPROVED/FROZEN: {', '.join(unreviewed[:10])}")
    if missing_reviewer:
        report["problems"].append(f"questions without a recorded reviewer: {', '.join(missing_reviewer[:10])}")

    # 4. The v3 runner's own gate accepts it (same code path, no execution).
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_source_selection_experiment_v3 import resolve_frozen_benchmark

        resolved = resolve_frozen_benchmark(c, kb_id, bv_id)
        report["checks"]["v3_runner_gate"] = True
        report["runner_snapshot_questions"] = len(resolved.get("questions_snapshot") or [])
    except SystemExit as exc:
        report["checks"]["v3_runner_gate"] = False
        report["problems"].append(f"v3 runner refuses this benchmark: {exc}")
    except Exception as exc:  # pragma: no cover - import/environment failure
        report["checks"]["v3_runner_gate"] = False
        report["problems"].append(f"could not evaluate the v3 runner gate: {exc}")

    return not report["problems"], report


def prepare_v3_plan(report: dict) -> dict:
    """Describe the v3 experiment that WOULD run. Builds nothing."""
    design_path = BENCH_DIR / "source-selection-experiment-v3.json"
    design = json.loads(design_path.read_text(encoding="utf-8")) if design_path.exists() else {}
    return {
        "experiment": design.get("experiment", "source-selection-experiment-v3"),
        "status": "PREPARED — NOT RUN",
        "frozen_benchmark": {
            "id": report["benchmark_version_id"],
            "version": report.get("version"),
            "status": report.get("status"),
            "frozen_at": report.get("frozen_at"),
        },
        "seeds": design.get("seeds", V3_SEEDS),
        "corpus_sizes": design.get("corpus_sizes", V3_CORPUS_SIZES),
        "strategies": ["QUALITY_SELECTED", "RANDOM_BASELINE"],
        "question_count": report.get("question_count"),
        "run_command": (
            "python backend/scripts/run_source_selection_experiment_v3.py "
            f"--baseline-kb {report['kb_id']} "
            f"--benchmark-version {report['benchmark_version_id']} --dry-run"
        ),
        "execute_command": (
            "python backend/scripts/run_source_selection_experiment_v3.py "
            f"--baseline-kb {report['kb_id']} "
            f"--benchmark-version {report['benchmark_version_id']}"
        ),
        "note": (
            "The full run builds 18 corpora (1 QUALITY arm + 5 RANDOM seeds, x N in "
            "{5,8,11}) and evaluates each at k in {3,5,10}. That is expensive: it "
            "downloads and embeds real documents. Run --dry-run first, then execute "
            "deliberately. Historical artifacts are never modified."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a frozen benchmark (read-only).")
    ap.add_argument("--kb-id", required=True, help="Knowledge base holding the benchmark")
    ap.add_argument("--benchmark-version", required=True, help="BenchmarkVersion id (bv_...)")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--prepare-v3", action="store_true", help="Also print the v3 experiment plan")
    ap.add_argument("--json", action="store_true", help="Emit the report as JSON only")
    args = ap.parse_args()

    if not args.json:
        banner("0. Frozen benchmark integrity gate (read-only)")

    try:
        with httpx.Client(base_url=args.base, timeout=60.0) as c:
            c.get("/api/system/health")
            ok, report = verify(c, args.kb_id, args.benchmark_version)
    except Exception as exc:
        print(f"Could not reach the API at {args.base}: {exc}", file=sys.stderr)
        print("Start the backend first (uvicorn app.main:app --reload).", file=sys.stderr)
        return 3

    if args.prepare_v3 and ok:
        report["v3_plan"] = prepare_v3_plan(report)

    if args.json:
        print(json.dumps({"gate_passed": ok, "report": report}, indent=2, ensure_ascii=False))
        return 0 if ok else 2

    banner("1. Verification report")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if ok:
        banner("2. Verdict")
        print(f"PASS — benchmark {report.get('version')!r} ({args.benchmark_version}) is frozen,")
        print("reviewed and snapshotted. An experiment may reference this exact version id.")
        if args.prepare_v3:
            banner("3. Source-selection v3 plan (PREPARED, NOT RUN)")
            print(json.dumps(report["v3_plan"], indent=2, ensure_ascii=False))
            print("\nNothing was built. Execute deliberately when ready:")
            print("  " + report["v3_plan"]["execute_command"])
        return 0

    banner("Verdict")
    print("FAIL — the frozen-benchmark gate did not pass:")
    for problem in report["problems"]:
        print(f"  - {problem}")
    print("\nNo v3 experiment may run against this benchmark.")
    return 2


if __name__ == "__main__":
    sys.exit(main())