"""Run a predeclared V13 controlled retrieval experiment.

Usage (from ``backend/``):

    py -m scripts.run_retrieval_experiment --experiment b-depth
    py -m scripts.run_retrieval_experiment --experiment all
    py -m scripts.run_retrieval_experiment --list

The experiment DEFINITIONS live in ``EXPERIMENTS`` below and are versioned with
the code on purpose: the hypothesis, the primary metric, the guardrail metrics
and the rejection conditions are declared here, in the repository, BEFORE any
arm runs. Nothing in this script searches for a good configuration.

Safety: every shipped strategy is read-only with respect to the index, so these
runs mutate nothing. Each artifact records the before/after index point count,
chunk-row count, corpus revision and the benchmark's sha256, so that claim is
checkable rather than asserted. The script refuses to run an experiment whose
declared benchmark does not belong to the target knowledge base.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402
from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.services.answer_eval.retrieval_experiment_models import (  # noqa: E402
    RetrievalStrategyConfig,
)
from app.services.answer_eval.retrieval_experiment_runner import (  # noqa: E402
    RetrievalExperimentRunner,
)

#: The frozen retrieval benchmark. Never written by anything in this script.
BENCHMARK = "benchmarks/automobile-engineering-baseline-v1.json"

#: The protected knowledge base from the V11/V12 checkpoints.
PROTECTED_KB = "kb_f278c283c748"

#: Candidate probe depth, held CONSTANT across the arms of an experiment so the
#: arms cannot disagree about whether the evidence was in the candidate set.
PROBE_DEPTH = 20

#: The declared latency budget for one interactive retrieval, in ms. Declared
#: here so it is the same bar in every experiment and cannot be relaxed after a
#: slow arm is measured. It is a POLICY choice about this project's interactive
#: use, not a measured quantity.
LATENCY_BUDGET_MS = 50.0


def _arm(label: str, strategy: str, top_k: int, **params) -> RetrievalStrategyConfig:
    return RetrievalStrategyConfig(
        label=label,
        strategy=strategy,
        top_k=top_k,
        candidate_depth=PROBE_DEPTH,
        params=params,
    )


EXPERIMENTS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # B-strict: retrieval depth under the STRICT guardrail declaration.
    #
    # Declared first, before any measurement, and kept because it is the more
    # conservative reading: precision@k IS a real cost of returning more
    # candidates. Its verdict is reported as measured.
    # ------------------------------------------------------------------
    "b-depth-strict": {
        "name": "v13-b-retrieval-depth-strict-guardrails",
        "hypothesis": (
            "The LOW_RECALL cases the V12 k=3 diagnostic reported are a WINDOW "
            "effect, not a retrieval failure: the required evidence is in the "
            "candidate set probed to depth 20 and merely falls outside a 3-item "
            "window. Under the STRICT declaration both predicted costs are held as "
            "hard guardrails: precision@k must not fall and latency must not rise "
            "by more than the declared bar."
        ),
        "baseline": _arm("dense_k3", "dense", 3),
        "candidates": [
            _arm("dense_k5", "dense", 5),
            _arm("dense_k10", "dense", 10),
        ],
        "primary_metric": "recall_at_k",
        "guardrail_metrics": ["precision_at_k", "latency_ms"],
        "proposed_change": "raise the evaluated and returned k from 3 to 5 and 10",
        "expected_effect": (
            "recall@k rises; precision@k falls because the denominator grows while "
            "the necessity-only labels do not; latency rises with k"
        ),
        "targeted_failure_code": "LOW_RECALL",
        "targeted_question_ids": [],
        "rejection_conditions": [
            "recall@k does not rise by at least the declared meaningful bar (0.02)",
            "precision@k regresses by at least its declared bar (0.02)",
            "latency regresses by at least its declared bar (50 ms/query)",
        ],
    },
    # ------------------------------------------------------------------
    # B: retrieval depth under the CORRECTED guardrail declaration.
    #
    # Same arms and same primary metric; the ONLY difference is that
    # precision@k is reported rather than guardrailed, because under
    # necessity-only binary labels it is a LOWER BOUND that falls
    # definitionally as k grows — the V12 diagnostics say so themselves and
    # refuse to emit LOW_PRECISION for exactly that reason. Holding a
    # definitional quantity as a hard guardrail would reject every k increase
    # by construction, which is not a safety property. Latency stays a hard
    # guardrail because it is a real, measured cost. Both artifacts are kept
    # so the reframing is visible instead of silent.
    # ------------------------------------------------------------------
    "b-depth": {
        "name": "v13-b-retrieval-depth-k3-k5-k10",
        "hypothesis": (
            "The LOW_RECALL cases the V12 k=3 diagnostic reported are a WINDOW "
            "effect, not a retrieval failure: the required evidence is present in "
            "the candidate set probed to depth 20 but falls outside a 3-item "
            "window, so widening the evaluated k recovers recall without changing "
            "which candidates the retriever can see. The predicted cost is that "
            "returning more candidates must lower precision@k and raise latency, "
            "because both denominators grow with k."
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
            "recall@k rises toward 1.0; latency must stay inside the declared "
            "50 ms/query budget. precision@k necessarily falls and is therefore "
            "REPORTED with its lower-bound caveat rather than guardrailed"
        ),
        "targeted_failure_code": "LOW_RECALL",
        "targeted_question_ids": [
            "auto-eng-003",
            "auto-eng-007",
            "auto-eng-011",
            "auto-eng-012",
            "auto-eng-015",
            "auto-eng-019",
        ],
        "rejection_conditions": [
            "recall@k does not rise by at least the declared meaningful bar (0.02) "
            "from k=3 to k=10, in which case k is not the binding constraint and "
            "the window hypothesis is not supported",
            "latency regresses by at least its declared bar (50 ms/query), which "
            "would make the wider window unaffordable for interactive use",
        ],
    },
    # ------------------------------------------------------------------
    # A: reranking over an IDENTICAL candidate pool.
    # ------------------------------------------------------------------
    "a-rerank": {
        "name": "v13-a-cross-encoder-reranking",
        "hypothesis": (
            "The 11 RANKING_FAILURE questions the V12 k=10 diagnostic reported are "
            "an ordering problem, not a coverage problem: the required evidence is "
            "already in the top-10 that the fused retriever returns, but sits below "
            "rank 1. Reordering the SAME candidate pool with a cross-encoder should "
            "raise MRR and nDCG without changing recall, because a reranker can "
            "only permute the candidates it is given."
        ),
        "baseline": _arm("hybrid", "hybrid", 10),
        "candidates": [
            _arm(
                "hybrid_reranked",
                "hybrid_reranked",
                10,
                reranker="cross-encoder",
                reranker_top_n=PROBE_DEPTH,
            )
        ],
        "primary_metric": "mrr",
        "guardrail_metrics": ["recall_at_k", "latency_ms"],
        "proposed_change": (
            "apply the cross-encoder reranker to the fused candidate pool; the "
            "candidate retriever and its fusion parameters are IDENTICAL on both "
            "sides, so reranking is the only difference"
        ),
        "expected_effect": (
            "MRR and nDCG rise; recall@k is unchanged because reranking permutes "
            "rather than adds candidates; latency rises substantially"
        ),
        "targeted_failure_code": "RANKING_FAILURE",
        "targeted_question_ids": [],
        "rejection_conditions": [
            "MRR does not rise by at least the declared meaningful bar (0.02)",
            "recall@k REGRESSES: a reranker that loses evidence is worse than no "
            "reranker regardless of its ordering",
            "the reranker is unavailable or falls back, in which case the arm is "
            "recorded as failed and no substitute implementation is used",
        ],
    },
    # ------------------------------------------------------------------
    # C: hybrid and lexical retrieval vs dense.
    # ------------------------------------------------------------------
    "c-hybrid": {
        "name": "v13-c-dense-bm25-hybrid",
        "hypothesis": (
            "Dense retrieval already saturates recall@10 at 1.0 on this benchmark, "
            "so no lexical or fused strategy can RECOVER missing evidence — there "
            "is none missing. The only place hybrid retrieval can win is RANKING: "
            "where dense places the required chunk below rank 1 but BM25 ranks it "
            "higher on exact terminology, fusion should raise MRR. The hypothesis "
            "is therefore falsifiable in BOTH directions: recall should be "
            "unchanged for every arm, and MRR may rise or fall."
        ),
        "baseline": _arm("dense_k10", "dense", 10),
        "candidates": [
            _arm("bm25_k10", "bm25", 10),
            _arm("hybrid_k10", "hybrid", 10),
        ],
        "primary_metric": "mrr",
        "guardrail_metrics": ["recall_at_k"],
        "proposed_change": "retrieve with BM25 and with dense+BM25 fusion instead of dense alone",
        "expected_effect": (
            "recall@10 stays at 1.0 for all three arms; MRR changes either way, "
            "depending on whether exact terminology or semantics ranks better"
        ),
        "targeted_failure_code": "RANKING_FAILURE",
        "targeted_question_ids": [],
        "rejection_conditions": [
            "MRR does not rise by at least the declared meaningful bar (0.02) for "
            "any candidate",
            "recall@10 regresses below 1.0 for a candidate, which would mean fusion "
            "lost evidence the dense arm found",
        ],
    },
}


def _print(result) -> None:
    print(f"\n{'=' * 78}")
    print(f"{result.name}  ->  {result.outcome.value}")
    print("=" * 78)
    print(f"id                      {result.id}")
    print(f"benchmark fingerprint   {result.benchmark_fingerprint}")
    print(f"question-set fingerprint{result.question_set_fingerprint}")
    print(f"matched questions       {len(result.matched_question_ids)}")
    print(f"primary metric          {result.primary_metric}")
    print(f"guardrails              {result.guardrail_metrics}")
    for arm in result.arms:
        stats = arm.metric_stats
        row = "  ".join(
            f"{m}={stats[m].mean if stats.get(m) and stats[m].measured else 'n/a'}"
            for m in ("recall_at_k", "mrr", "ndcg_at_k", "precision_at_k", "latency_ms")
        )
        print(f"  arm {arm.label:<18} status={arm.status:<10} {row}")
    for outcome in result.candidate_outcomes:
        print(
            f"  -> {outcome.label}: {outcome.outcome.value} "
            f"(acceptance={outcome.acceptance.value}, pairing={outcome.pairing}, "
            f"primary delta={outcome.primary_mean_delta}, "
            f"paired={outcome.paired_question_count})"
        )
    print(f"\nOUTCOME: {result.outcome_explanation}")
    if result.errors:
        print(f"ERRORS: {result.errors}")
    print(f"integrity: {json.dumps({k: v for k, v in result.integrity.items() if k != 'note'})}")
    print(f"persisted to: {result.notes[-1] if result.notes else 'not persisted'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default=PROTECTED_KB)
    parser.add_argument("--experiment", default="all")
    parser.add_argument("--benchmark", default=BENCHMARK)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--question-limit", type=int, default=0)
    args = parser.parse_args()

    if args.list:
        for key, definition in EXPERIMENTS.items():
            print(f"{key}: {definition['name']}")
            print(f"    primary={definition['primary_metric']} guardrails={definition['guardrail_metrics']}")
            print(f"    hypothesis: {definition['hypothesis'][:100]}...")
        return 0

    keys = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for key in keys:
        if key not in EXPERIMENTS:
            print(f"unknown experiment {key!r}; available: {', '.join(EXPERIMENTS)}")
            return 2

    settings = get_settings()
    repo = Repository(str(settings.db_path))
    runner = RetrievalExperimentRunner(repo, settings)

    question_ids: list[str] = []
    if args.question_limit:
        from app.services.answer_eval.retrieval_diagnostic_runner import (
            load_retrieval_benchmark,
            resolve_benchmark_path,
        )

        bench = load_retrieval_benchmark(resolve_benchmark_path(args.benchmark))
        question_ids = [q.id for q in bench.questions][: args.question_limit]

    results = []
    for key in keys:
        definition = EXPERIMENTS[key]
        print(f"\n>>> running {key}: {definition['name']}")
        result = runner.run(
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
            targeted_question_ids=definition["targeted_question_ids"],
            rejection_conditions=definition["rejection_conditions"],
            benchmark_path=args.benchmark,
            question_ids=question_ids,
            persist=True,
            created_by="scripts/run_retrieval_experiment.py",
        )
        _print(result)
        results.append(result)

    print(f"\n{len(results)} experiment artifact(s) written under {runner.experiments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
