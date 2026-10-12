"""Run the PRE-REGISTERED V14 held-out validation experiments.

Usage (from ``backend/``):

    py -m scripts.run_heldout_validation --list
    py -m scripts.run_heldout_validation --experiment all
    py -m scripts.run_heldout_validation --experiment b-depth --no-dev-reference

## Pre-registration, and what makes it real

The three experiments below were declared — hypothesis, arms, primary metric,
guardrails and rejection conditions — BEFORE any arm was run against the held-out
benchmark, and they are versioned with the code on purpose. Two rules make that
more than a comment:

1. **The arms are the V13 arms.** Not "similar" arms: the same strategies and the
   same cut-offs that produced the V13 findings. A retest that quietly changed the
   configuration would answer a different question.
2. **Latency is a guardrail in C and A.** V13's hybrid experiment reported its
   latency overshoot as a caveat *after* the fact instead of holding it as a gate.
   Closing that hole is a deliberate, declared difference from the V13
   declaration, and it is stated in each hypothesis below rather than discovered
   afterwards.

## Where the numbers land

Each experiment runs TWICE on purpose:

* on the frozen held-out instrument — this is the generalization test, and it is
  the artifact whose ``instrument`` field says ``held-out``;
* on the development benchmark — a fresh re-measurement under the identical code
  path, stored as a separate artifact named ``...-dev-reference``.

The second run exists so the comparison between instruments is computed from the
SAME engine in the SAME session, rather than by diffing against numbers in a V13
document that a reader cannot re-verify. It is explicitly NOT a new finding: those
28 questions were the ones used to choose the arms, so the dev-reference run is a
baseline, not evidence.

## What this script will not do

It will not run against the development benchmark alone and call it validation, it
will not tune a fusion weight, a cut-off or a guardrail on held-out results, and
it will not write a confirmatory claim unless the frozen instrument's tier is
``validated``. The gate lives in ``HeldOutService.runnable()`` and in the runner's
``instrument``/``benchmark_lifecycle`` check, so this script cannot bypass it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402
from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.services.answer_eval.heldout_benchmark import (  # noqa: E402
    DEFAULT_DEV_BENCHMARK_PATH,
    HeldOutValidationError,
)
from app.services.answer_eval.heldout_service import HeldOutService  # noqa: E402
from app.services.answer_eval.retrieval_experiment_models import (  # noqa: E402
    RetrievalStrategyConfig,
)
from app.services.answer_eval.retrieval_experiment_runner import (  # noqa: E402
    RetrievalExperimentRunner,
)

#: The protected knowledge base from the V11/V12/V13 checkpoints.
PROTECTED_KB = "kb_f278c283c748"

#: Candidate probe depth, held CONSTANT across every arm of every experiment here
#: and equal to V13's, so no arm can disagree about whether the evidence was in the
#: candidate set and so the retest is like-for-like.
PROBE_DEPTH = 20

#: The declared latency budget for one interactive retrieval, in ms. The same bar
#: V13 declared, re-declared here so it cannot have been relaxed between the two
#: sessions.
LATENCY_BUDGET_MS = 50.0

#: Metrics reported for every arm on both instruments.
REPORT_METRICS: tuple[str, ...] = (
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "latency_ms",
)


def _arm(label: str, strategy: str, top_k: int, **params: Any) -> RetrievalStrategyConfig:
    return RetrievalStrategyConfig(
        label=label,
        strategy=strategy,
        top_k=top_k,
        candidate_depth=PROBE_DEPTH,
        params=params,
    )


EXPERIMENTS: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------------
    # B — retrieval depth. Retests the V13 finding that the LOW_RECALL
    # cases were a WINDOW effect rather than a retrieval failure.
    # ------------------------------------------------------------------
    "b-depth": {
        "name": "v14-b-retrieval-depth-heldout",
        "retests": "v13-b-retrieval-depth-k3-k5-k10",
        "hypothesis": (
            "V13 reported that raising the evaluated and returned k from 3 to 10 lifted "
            "recall@k from 0.8214 to 1.0 on the development benchmark, with all five "
            "failing questions improving and none regressing, and concluded that those "
            "LOW_RECALL cases were a WINDOW effect rather than a retrieval failure. That "
            "conclusion was drawn from the questions used to choose the arms. "
            "PREDICTION ON UNSEEN QUESTIONS: recall@k rises with k in the same way, "
            "because the mechanism (required evidence already present in the candidate "
            "pool probed to depth 20, merely outside a 3-item window) is a property of "
            "the retriever and the corpus, not of those particular questions. "
            "FALSIFIED IF: recall@k does not rise by at least the declared meaningful "
            "bar from k=3 to k=10, or latency exceeds the declared budget. precision@k "
            "is REPORTED rather than guardrailed for the same reason V13 gave: under "
            "necessity-only binary labels it falls definitionally as k grows."
        ),
        "baseline": _arm("dense_k3", "dense", 3),
        "candidates": [
            _arm("dense_k5", "dense", 5),
            _arm("dense_k10", "dense", 10),
        ],
        "primary_metric": "recall_at_k",
        "guardrail_metrics": ["latency_ms"],
        "proposed_change": "raise the evaluated and returned k from 3 to 5 and 10",
        "expected_effect": (
            "recall@k rises toward 1.0 with k; precision@k falls because its denominator "
            "grows while the labels do not; latency must stay inside the declared budget"
        ),
        "targeted_failure_code": "LOW_RECALL",
        "rejection_conditions": [
            "recall@k does not rise by at least the declared meaningful bar (0.02) from "
            "k=3 to k=10: the window hypothesis does not generalize",
            "latency regresses by at least its declared bar (50 ms/query)",
        ],
    },
    # ------------------------------------------------------------------
    # C — hybrid versus dense ranking. Retests V13's hybrid improvement,
    # and this time latency IS a gate.
    # ------------------------------------------------------------------
    "c-hybrid": {
        "name": "v14-c-dense-bm25-hybrid-heldout",
        "retests": "v13-c-dense-bm25-hybrid",
        "hypothesis": (
            "V13 reported that hybrid retrieval raised MRR over dense alone at k=10, but "
            "reported its latency overshoot as a post-hoc caveat rather than holding it "
            "as a guardrail. This retest closes that hole: latency_ms is a DECLARED "
            "GUARDRAIL here, so a hybrid arm that is slower than the budget fails even if "
            "its ranking improves. "
            "PREDICTION ON UNSEEN QUESTIONS: if dense already saturates recall@k at the "
            "evaluated k, no lexical or fused arm can recover missing evidence and the "
            "only movement available is in RANKING (MRR, nDCG). "
            "FALSIFIED IF: MRR does not rise by the declared bar for any candidate, or "
            "recall@k regresses for a candidate, or latency exceeds the budget. "
            "The fusion weights are the shipped defaults and are NOT tuned here."
        ),
        "baseline": _arm("dense_k10", "dense", 10),
        "candidates": [
            _arm("bm25_k10", "bm25", 10),
            _arm("hybrid_k10", "hybrid", 10),
        ],
        "primary_metric": "mrr",
        "guardrail_metrics": ["recall_at_k", "latency_ms"],
        "proposed_change": (
            "retrieve with BM25 and with dense+BM25 fusion at the shipped defaults, "
            "instead of dense alone"
        ),
        "expected_effect": (
            "recall@k unchanged for every arm; MRR may rise or fall, depending on whether "
            "exact terminology or semantics ranks the required chunk higher"
        ),
        "targeted_failure_code": "RANKING_FAILURE",
        "rejection_conditions": [
            "MRR does not rise by at least the declared meaningful bar (0.02) for any "
            "candidate",
            "recall@k regresses for a candidate: fusion lost evidence the dense arm found",
            "latency regresses by at least its declared bar (50 ms/query)",
        ],
    },
    # ------------------------------------------------------------------
    # A — the reranking follow-up. V13 rejected cross-encoder reranking
    # because it LOST recall. This arm confines the reranker so it cannot.
    # ------------------------------------------------------------------
    "a-rerank-windowed": {
        "name": "v14-a-windowed-reranking-heldout",
        "retests": "v13-a-cross-encoder-reranking",
        "hypothesis": (
            "V13 rejected the cross-encoder reranker: MRR rose by less than a third of the "
            "declared bar while recall@k FELL from 1.0 to 0.9643, because the reranker was "
            "allowed to reorder the whole 20-candidate pool and could therefore push "
            "required evidence out of the 10-item window. The mechanism is visible in the "
            "code: candidates outside the reranked slice keep their relative order and "
            "sort AFTER the reranked ones, so with reranker_top_n == top_k the returned set "
            "is the SAME set of candidates and recall@k is invariant BY CONSTRUCTION. "
            "PREDICTION ON UNSEEN QUESTIONS: with reranker_top_n == top_k == 10, recall@k "
            "is identical to the un-reranked arm, so the recall guardrail holds trivially; "
            "the only open question is whether MRR or nDCG improves enough to justify a "
            "reranker. "
            "FALSIFIED IF: MRR does not rise by the declared bar, or latency exceeds the "
            "budget, or recall@k DIFFERS from the un-reranked arm (which would mean the "
            "invariance argument is wrong and is itself a finding to investigate)."
        ),
        "baseline": _arm("hybrid_k10", "hybrid", 10),
        "candidates": [
            _arm(
                "hybrid_reranked_windowed",
                "hybrid_reranked",
                10,
                reranker="cross-encoder",
                reranker_top_n=10,
            )
        ],
        "primary_metric": "mrr",
        "guardrail_metrics": ["recall_at_k", "latency_ms"],
        "proposed_change": (
            "apply the cross-encoder reranker to ONLY the candidates that will be returned "
            "(reranker_top_n = top_k = 10) instead of to the full 20-candidate probe pool"
        ),
        "expected_effect": (
            "recall@k identical to the un-reranked hybrid arm by construction; MRR may rise "
            "or fall within the window; latency rises by the reranker's cost on 10 rather "
            "than 20 candidates"
        ),
        "targeted_failure_code": "RANKING_FAILURE",
        "rejection_conditions": [
            "MRR does not rise by at least the declared meaningful bar (0.02)",
            "recall@k differs from the un-reranked arm: the invariance argument would be "
            "wrong",
            "latency regresses by at least its declared bar (50 ms/query)",
            "the reranker is unavailable or falls back, in which case the arm is recorded "
            "as failed and no substitute implementation is used",
        ],
    },
}


def _means(experiment: Any) -> dict[str, dict[str, Any]]:
    """Per-arm metric means as recorded on the artifact."""
    out: dict[str, dict[str, Any]] = {}
    for arm in experiment.arms:
        stats = arm.metric_stats or {}
        out[arm.label] = {
            metric: (
                stats[metric].mean
                if stats.get(metric) and stats[metric].measured
                else None
            )
            for metric in REPORT_METRICS
        }
        out[arm.label]["status"] = arm.status
    return out


def summarize_generalization(
    key: str,
    definition: dict[str, Any],
    heldout: Any,
    dev: Any | None,
) -> dict[str, Any]:
    """Compare the same arms across the two instruments.

    The comparison is UNPAIRED by necessity — the two benchmarks ask different
    questions, so there is no per-question pairing to exploit. What this reports
    is therefore whether the DIRECTION and rough size of each arm's effect persist
    across instruments, which is the entire content of a generalization question.
    Unpaired means it cannot be a significance test, and it is not reported as one.
    """
    base_label = definition["baseline"].label
    summary: dict[str, Any] = {
        "experiment": key,
        "heldout_artifact": heldout.id,
        "dev_artifact": dev.id if dev else "",
        "retests": definition.get("retests", ""),
        "primary_metric": definition["primary_metric"],
        "heldout_outcome": heldout.outcome.value,
        "dev_outcome": dev.outcome.value if dev else "",
        "heldout_claim_strength": heldout.claim_strength,
        "heldout_benchmark_lifecycle": heldout.benchmark_lifecycle,
        "heldout_question_count": len(heldout.matched_question_ids),
        "dev_question_count": len(dev.matched_question_ids) if dev else 0,
        "arms": {},
        "agreement": {},
        "note": (
            "Unpaired comparison across instruments: the two benchmarks ask different "
            "questions, so no per-question pairing or significance test is available here. "
            "Only the direction and magnitude of each arm's effect on the declared metric "
            "are compared."
        ),
    }
    ho_means = _means(heldout)
    dev_means = _means(dev) if dev else {}

    all_labels = list(dict.fromkeys([*ho_means, *dev_means]))
    for label in all_labels:
        summary["arms"][label] = {
            "heldout": ho_means.get(label),
            "development": dev_means.get(label),
        }

    metric = definition["primary_metric"]
    for label in all_labels:
        if label == base_label:
            continue
        ho = ho_means.get(label) or {}
        ho_base = ho_means.get(base_label) or {}
        if ho.get(metric) is None or ho_base.get(metric) is None:
            summary["agreement"][label] = {
                "comparable": False,
                "reason": "the held-out arm has no measured value for the primary metric",
            }
            continue
        ho_delta = ho[metric] - ho_base[metric]
        entry: dict[str, Any] = {
            "comparable": True,
            "heldout_delta": ho_delta,
            "heldout_outcome": next(
                (
                    o.outcome.value
                    for o in heldout.candidate_outcomes
                    if o.label == label
                ),
                "",
            ),
        }
        if dev:
            dv = dev_means.get(label) or {}
            dv_base = dev_means.get(base_label) or {}
            if dv.get(metric) is not None and dv_base.get(metric) is not None:
                dev_delta = dv[metric] - dv_base[metric]
                entry["development_delta"] = dev_delta
                entry["direction_agrees"] = (ho_delta > 0) == (dev_delta > 0) and (
                    ho_delta != 0 or dev_delta != 0
                )
                entry["dev_outcome"] = next(
                    (o.outcome.value for o in dev.candidate_outcomes if o.label == label),
                    "",
                )
        summary["agreement"][label] = entry
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\n--- generalization: {summary['experiment']} ---")
    print(
        f"    dev {summary['dev_question_count']} q -> held-out "
        f"{summary['heldout_question_count']} q | dev {summary['dev_outcome']} -> "
        f"held-out {summary['heldout_outcome']} ({summary['heldout_claim_strength']})"
    )
    metric = summary["primary_metric"]
    for label, entry in summary["agreement"].items():
        if not entry.get("comparable"):
            print(f"    {label}: not comparable ({entry['reason']})")
            continue
        dev_delta = entry.get("development_delta")
        agrees = entry.get("direction_agrees")
        print(
            f"    {label}: held-out {metric} delta {entry['heldout_delta']:+.4f} "
            f"({entry['heldout_outcome']})"
            + (
                f" | dev delta {dev_delta:+.4f} ({entry.get('dev_outcome','')})"
                f" | direction {'AGREES' if agrees else 'DIFFERS'}"
                if dev_delta is not None
                else " | dev reference not measured"
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default=PROTECTED_KB)
    parser.add_argument("--experiment", default="all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--no-dev-reference",
        action="store_true",
        help="skip the development-benchmark re-measurement (no cross-instrument compare)",
    )
    parser.add_argument(
        "--require-confirmatory",
        action="store_true",
        help=(
            "refuse to run unless the frozen instrument is VALIDATED. Use this to make a "
            "confirmatory claim; without it, a PROVISIONAL instrument still runs but every "
            "artifact is stamped exploratory."
        ),
    )
    args = parser.parse_args()

    if args.list:
        for key, definition in EXPERIMENTS.items():
            print(f"{key}: {definition['name']}")
            print(f"    retests: {definition['retests']}")
            print(
                f"    primary={definition['primary_metric']} "
                f"guardrails={definition['guardrail_metrics']}"
            )
            print(f"    hypothesis: {definition['hypothesis'][:140]}...")
        return 0

    keys = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for key in keys:
        if key not in EXPERIMENTS:
            print(f"unknown experiment {key!r}; available: {', '.join(EXPERIMENTS)}")
            return 2

    settings = get_settings()
    repo = Repository(str(settings.db_path))
    service = HeldOutService(repo, settings)
    runner = RetrievalExperimentRunner(repo, settings)

    try:
        heldout, sha = service.runnable()
    except HeldOutValidationError as exc:
        print(f"REFUSING TO RUN: {exc}")
        return 3

    print("held-out instrument")
    print(f"    benchmark        {heldout.benchmark} v{heldout.version}")
    print(f"    fingerprint      {heldout.fingerprint()}   sha256 {sha[:16]}...")
    print(f"    questions        {heldout.question_count}")
    print(f"    lifecycle        {heldout.lifecycle.value}")
    print(f"    claim strength   {heldout.claim_strength.value}")
    print(f"    frozen at        {heldout.frozen_at}  by {heldout.frozen_by}")
    print(f"    review           {json.dumps(heldout.review)}")

    if args.require_confirmatory:
        try:
            service.assert_confirmatory_allowed()
        except HeldOutValidationError as exc:
            print(f"\nREFUSING TO RUN: {exc}")
            return 3

    summaries: list[dict[str, Any]] = []
    for key in keys:
        definition = EXPERIMENTS[key]
        print(f"\n>>> {key}: {definition['name']}")

        heldout_result = runner.run(
            args.kb,
            name=definition["name"],
            hypothesis=definition["hypothesis"],
            baseline=definition["baseline"],
            candidates=definition["candidates"],
            primary_metric=definition["primary_metric"],
            guardrail_metrics=definition["guardrail_metrics"],
            proposed_change=definition["proposed_change"],
            expected_effect=definition["expected_effect"],
            targeted_failure_code=definition["targeted_failure_code"],
            targeted_question_ids=[],
            rejection_conditions=definition["rejection_conditions"],
            benchmark_path=service.benchmark_path,
            persist=True,
            created_by="scripts/run_heldout_validation.py",
            instrument="held-out",
            benchmark_lifecycle=heldout.lifecycle.value,
            claim_strength=heldout.claim_strength.value,
            retests=definition["retests"],
        )
        print(
            f"    HELD-OUT  {heldout_result.outcome.value}  "
            f"({heldout_result.claim_strength})  {heldout_result.id}"
        )
        for arm in heldout_result.arms:
            stats = arm.metric_stats or {}
            row = "  ".join(
                f"{m}={(stats[m].mean if stats.get(m) and stats[m].measured else 'n/a')}"
                for m in REPORT_METRICS
            )
            print(f"      {arm.label:<26} {arm.status:<10} {row}")

        dev_result = None
        if not args.no_dev_reference:
            from app.services.answer_eval.heldout_service import resolve_artifact

            dev_result = runner.run(
                args.kb,
                name=f"{definition['name']}-dev-reference",
                hypothesis=(
                    "Reference re-measurement of the same arms on the DEVELOPMENT "
                    "benchmark, under identical code, to compare instruments. Not a "
                    "finding: these 28 questions are the ones the arms were chosen on."
                ),
                baseline=definition["baseline"],
                candidates=definition["candidates"],
                primary_metric=definition["primary_metric"],
                guardrail_metrics=definition["guardrail_metrics"],
                proposed_change=definition["proposed_change"],
                expected_effect=definition["expected_effect"],
                targeted_failure_code=definition["targeted_failure_code"],
                targeted_question_ids=[],
                rejection_conditions=definition["rejection_conditions"],
                benchmark_path=resolve_artifact(DEFAULT_DEV_BENCHMARK_PATH),
                persist=True,
                created_by="scripts/run_heldout_validation.py (dev reference)",
                instrument="development",
                retests="",
            )
            print(f"    DEV-REF   {dev_result.outcome.value}  {dev_result.id}")
            for arm in dev_result.arms:
                stats = arm.metric_stats or {}
                row = "  ".join(
                    f"{m}={(stats[m].mean if stats.get(m) and stats[m].measured else 'n/a')}"
                    for m in REPORT_METRICS
                )
                print(f"      {arm.label:<26} {arm.status:<10} {row}")

        summary = summarize_generalization(key, definition, heldout_result, dev_result)
        _print_summary(summary)
        summaries.append(summary)

        summary_path = runner.experiments_dir / f"v14-generalization-{key}.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        print(f"    summary written to {summary_path}")

    print(f"\n{len(summaries)} generalization summary(ies) under {runner.experiments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
