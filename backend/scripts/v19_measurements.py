"""V19 — retrieval measurements over the DRAFT second-domain benchmark.

Runs the project's EXISTING instruments (no new measurement code) against the
second-domain KB and prints what they actually measured:

* `RetrievalDiagnosticRunner` — dense, bm25 and hybrid over the same draft
  question set at k=10 with a depth-20 candidate probe: per-strategy recall@k,
  precision@k (lower-bound caveat), MRR, nDCG@k, required-evidence recall at
  depth, latency, the failure-code counts and the per-question evidence
  availability rollup (the corpus/index coverage measurement);
* two PRE-DECLARED experiments (definitions below, in the repository, before
  any arm runs — the same discipline as scripts/run_retrieval_experiment.py):
    E1 "window"    baseline dense k=5  vs dense k=10   (primary recall@k)
    E2 "strategy"  baseline dense k=10 vs bm25, hybrid (primary MRR)

Everything printed is measured or reported UNKNOWN. Answer-level properties
(answer correctness, citation support, abstention quality) are UNKNOWN by
construction here: no generator ran and no human reference answers exist.

Usage (from backend/):
    py -m scripts.v19_measurements --kb <kb_id>
    py -m scripts.v19_measurements --kb <kb_id> --skip-experiments
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

BENCHMARK = "benchmarks/internet-protocols-draft-v1.json"
PROBE_DEPTH = 20

#: Pre-declared experiment definitions. Hypotheses, primary metrics, guardrails
#: and rejection conditions are declared HERE, in the repository, before any
#: arm is measured. Nothing in this script searches for a favourable result.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "e1-window": {
        "name": "v19-e1-retrieval-window-k5-vs-k10",
        "hypothesis": (
            "The second-domain corpus is roughly 17x larger than the automobile "
            "corpus the k=5 default was exercised on, so a material share of "
            "required evidence will sit just beyond rank 5. Raising the evaluated "
            "and returned k from 5 to 10 should raise recall@k without changing "
            "which candidates the dense retriever can see."
        ),
        "baseline": {"label": "dense_k5", "strategy": "dense", "top_k": 5},
        "candidates": [{"label": "dense_k10", "strategy": "dense", "top_k": 10}],
        "primary_metric": "recall_at_k",
        "guardrail_metrics": [],
        "proposed_change": "raise the evaluated and returned k from 5 to 10",
        "expected_effect": (
            "recall@k rises toward the probe-depth ceiling; latency rises with k "
            "and is REPORTED, not guardrailed, because a wider window does more "
            "work by definition"
        ),
        "targeted_failure_code": "LOW_RECALL",
        "rejection_conditions": [
            "recall@k does not rise by at least the declared meaningful bar (0.02), "
            "in which case k is not the binding constraint on this corpus",
        ],
    },
    "e2-strategy": {
        "name": "v19-e2-dense-vs-bm25-vs-hybrid",
        "hypothesis": (
            "RFC text is terminology-dense (exact protocol names, field names, "
            "numbers), so lexical and fused retrieval may rank required chunks "
            "higher than dense alone. The comparison is falsifiable in both "
            "directions: MRR may rise or fall, and recall@10 must not regress, "
            "because fusion that loses evidence is worse than no fusion."
        ),
        "baseline": {"label": "dense_k10", "strategy": "dense", "top_k": 10},
        "candidates": [
            {"label": "bm25_k10", "strategy": "bm25", "top_k": 10},
            {"label": "hybrid_k10", "strategy": "hybrid", "top_k": 10},
        ],
        "primary_metric": "mrr",
        "guardrail_metrics": ["recall_at_k"],
        "proposed_change": "retrieve with BM25 and with dense+BM25 fusion instead of dense alone",
        "expected_effect": (
            "recall@10 stays equal for all arms; MRR changes either way depending "
            "on whether exact terminology or semantics ranks the evidence first"
        ),
        "targeted_failure_code": "RANKING_FAILURE",
        "rejection_conditions": [
            "MRR does not rise by at least the declared meaningful bar (0.02) for any candidate",
            "recall@10 regresses for a candidate, which would mean fusion lost evidence",
        ],
    },
}


def _metric(obj: Any) -> str:
    """Render a DiagnosticMetric honestly: a value, or UNKNOWN with the reason."""
    if obj is None:
        return "UNKNOWN (not present)"
    if getattr(obj, "measured", False) and obj.value is not None:
        extra = f" (n={obj.sample_size})" if getattr(obj, "sample_size", None) else ""
        return f"{obj.value:.4f}{extra}"
    return f"UNKNOWN ({getattr(obj, 'reason', 'unmeasured') or 'unmeasured'})"


def _arm(definition: dict[str, Any]):
    from app.services.answer_eval.retrieval_experiment_models import RetrievalStrategyConfig

    return RetrievalStrategyConfig(
        label=definition["label"],
        strategy=definition["strategy"],
        top_k=definition["top_k"],
        candidate_depth=PROBE_DEPTH,
        params={},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kb", default="", help="knowledge base id (default: V19 state file)")
    parser.add_argument("--benchmark", default=BENCHMARK)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--strategies", default="dense,bm25,hybrid")
    parser.add_argument("--skip-experiments", action="store_true")
    parser.add_argument("--json-out", default="", help="write the measurement summary JSON here")
    args = parser.parse_args(argv)

    from app.api.deps import get_repo
    from app.config import get_settings
    from app.services.answer_eval.retrieval_diagnostic_runner import RetrievalDiagnosticRunner
    from app.services.answer_eval.retrieval_experiment_runner import RetrievalExperimentRunner

    repo = get_repo()
    settings = get_settings()
    kb_id = args.kb
    if not kb_id:
        state_path = BACKEND_ROOT / "data" / "v19-validation-state.json"
        if state_path.is_file():
            kb_id = json.loads(state_path.read_text(encoding="utf-8")).get("kb_id") or ""
    if not kb_id:
        print("pass --kb <kb_id>")
        return 2
    kb = repo.get_kb(kb_id)
    if kb is None:
        print(f"knowledge base {kb_id!r} not found")
        return 2

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    print(f"kb: {kb.id} ({kb.name}) — {repo.count_chunks(kb.id)} chunk rows")
    print(f"benchmark: {args.benchmark}  strategies: {strategies}  top_k={args.top_k}\n")

    runner = RetrievalDiagnosticRunner(repo, settings)
    report = runner.run(
        kb.id,
        strategies=strategies,
        top_k=args.top_k,
        benchmark_path=args.benchmark,
        persist=True,
        persist_retrieval_runs=False,
        candidate_depth=PROBE_DEPTH,
    )

    print("=" * 78)
    print(f"diagnostics run {report.id}  ({report.question_count} questions)")
    print(f"benchmark fingerprint: {report.benchmark_fingerprint}")
    authorship = report.benchmark_authorship or {}
    print(
        f"instrument authorship: tier={authorship.get('tier')!r} "
        f"human_review={authorship.get('human_review')!r} — EXPLORATORY numbers only"
    )
    print("-" * 78)
    for strategy in report.strategies:
        summary = strategy.summary
        print(f"\nstrategy {strategy.strategy}: status={strategy.status}")
        if strategy.status != "completed":
            print(f"  failure: {strategy.failure_reason}")
            continue
        print(f"  evaluated questions     {summary.evaluated_question_count}")
        print(f"  recall@{strategy.top_k:<17} {_metric(summary.recall_at_k)}")
        print(f"  precision@{strategy.top_k:<14} {_metric(summary.precision_at_k)} (lower bound)")
        print(f"  MRR                     {_metric(summary.mrr)}")
        print(f"  nDCG@{strategy.top_k:<18} {_metric(summary.ndcg_at_k)}")
        print(f"  required recall @depth {_metric(summary.required_recall_at_depth)}")
        print(f"  required recall in probe{_metric(summary.required_recall_in_probe)}")
        print(f"  first relevant rank     {_metric(summary.first_relevant_rank)}")
        print(f"  latency                 {_metric(summary.latency_ms)}")
        print(f"  failure codes           {summary.code_counts}")
        print(f"  evidence availability   {summary.availability_counts}")

    # -- coverage: the per-question availability rollup of the FIRST strategy --
    primary = next((s for s in report.strategies if s.status == "completed"), None)
    if primary is not None:
        print("\n" + "-" * 78)
        print("per-question evidence availability (first completed strategy)")
        for q in primary.questions:
            if str(q.availability_rollup.value).startswith("all"):
                continue
            print(
                f"  {q.question_id:<12} {q.availability_rollup.value:<20} "
                f"expected={q.expected_chunk_ids}"
            )

    print("\n" + "-" * 78)
    for line in report.integrity.observations():
        print(f"integrity: {line}")
    for limitation in report.limitations[:6]:
        print(f"limitation: {limitation}")

    # -- pre-declared experiments ------------------------------------------
    experiment_results: list[dict[str, Any]] = []
    if not args.skip_experiments:
        erunner = RetrievalExperimentRunner(repo, settings)
        for key, definition in EXPERIMENTS.items():
            print("\n" + "=" * 78)
            print(f">>> {key}: {definition['name']}")
            result = erunner.run(
                kb.id,
                name=definition["name"],
                hypothesis=definition["hypothesis"],
                baseline=_arm(definition["baseline"]),
                candidates=[_arm(c) for c in definition["candidates"]],
                primary_metric=definition["primary_metric"],
                guardrail_metrics=definition["guardrail_metrics"],
                proposed_change=definition["proposed_change"],
                expected_effect=definition["expected_effect"],
                targeted_failure_code=definition["targeted_failure_code"],
                targeted_question_ids=[],
                rejection_conditions=definition["rejection_conditions"],
                benchmark_path=args.benchmark,
                persist=True,
                created_by="scripts/v19_measurements.py",
                instrument="development",
                claim_strength="exploratory",
            )
            print(f"outcome: {result.outcome.value}")
            print(f"explanation: {result.outcome_explanation}")
            for arm in result.arms:
                stats = arm.metric_stats
                row = "  ".join(
                    f"{m}={stats[m].mean if stats.get(m) and stats[m].measured else 'n/a'}"
                    for m in ("recall_at_k", "mrr", "ndcg_at_k", "latency_ms")
                )
                print(f"  arm {arm.label:<12} status={arm.status:<10} {row}")
            for outcome in result.candidate_outcomes:
                print(
                    f"  -> {outcome.label}: {outcome.outcome.value} "
                    f"(primary delta={outcome.primary_mean_delta}, paired={outcome.paired_question_count})"
                )
            experiment_results.append(
                {
                    "key": key,
                    "id": result.id,
                    "outcome": result.outcome.value,
                    "explanation": result.outcome_explanation,
                    "arms": [
                        {
                            "label": arm.label,
                            "status": arm.status,
                            "metrics": {
                                m: (
                                    arm.metric_stats[m].mean
                                    if arm.metric_stats.get(m) and arm.metric_stats[m].measured
                                    else None
                                )
                                for m in ("recall_at_k", "mrr", "ndcg_at_k", "precision_at_k", "latency_ms")
                            },
                        }
                        for arm in result.arms
                    ],
                    "candidates": [
                        {
                            "label": o.label,
                            "outcome": o.outcome.value,
                            "primary_mean_delta": o.primary_mean_delta,
                            "paired": o.paired_question_count,
                        }
                        for o in result.candidate_outcomes
                    ],
                }
            )

    # -- answer-level properties are UNKNOWN by construction ----------------
    print("\n" + "=" * 78)
    print("answer-level properties (no generator ran in this script; no reference answers exist)")
    print("  answer correctness   UNKNOWN")
    print("  citation support     UNKNOWN")
    print("  abstention quality   UNKNOWN (measured separately via MCP answer probes)")

    summary_json = {
        "kb_id": kb.id,
        "benchmark": args.benchmark,
        "diagnostics_run_id": report.id,
        "benchmark_fingerprint": report.benchmark_fingerprint,
        "instrument_authorship": authorship,
        "strategies": [
            {
                "strategy": s.strategy,
                "status": s.status,
                "top_k": s.top_k,
                "recall_at_k": _metric(s.summary.recall_at_k) if s.status == "completed" else None,
                "precision_at_k": _metric(s.summary.precision_at_k) if s.status == "completed" else None,
                "mrr": _metric(s.summary.mrr) if s.status == "completed" else None,
                "ndcg_at_k": _metric(s.summary.ndcg_at_k) if s.status == "completed" else None,
                "latency_ms": _metric(s.summary.latency_ms) if s.status == "completed" else None,
                "code_counts": s.summary.code_counts if s.status == "completed" else {},
                "availability_counts": s.summary.availability_counts if s.status == "completed" else {},
            }
            for s in report.strategies
        ],
        "experiments": experiment_results,
        "answer_level": {
            "answer_correctness": "UNKNOWN",
            "citation_support": "UNKNOWN",
            "abstention_quality": "UNKNOWN",
        },
        "integrity": report.integrity.observations(),
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary_json, indent=2, default=str), encoding="utf-8")
        print(f"\nmeasurement summary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
