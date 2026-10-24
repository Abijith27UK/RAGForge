"""V22 retrieval experiments: fusion, reranking and a second-domain replication.

Run from ``backend/``:

    py -m scripts.v22_retrieval_experiments --suite a
    py -m scripts.v22_retrieval_experiments --suite b
    py -m scripts.v22_retrieval_experiments --suite d
    py -m scripts.v22_retrieval_experiments --list

Design rules (same as V13, kept deliberately):

* Every arm is declared HERE, in the repository, before it runs. Nothing in this
  script searches for a good configuration; each suite tests one named factor.
* The measurement engine is the existing ``RetrievalExperimentRunner``, so an arm
  here is scored by exactly the same code, metric definitions and availability
  probe as a V12 diagnostic or a V13 experiment.
* Nothing is written to the index, the corpus or the benchmark. Each artifact
  records the before/after point count, chunk count and benchmark sha256.
* The Automobile KB ``kb_f278c283c748`` is PROTECTED: it is read only, and its
  held-out instrument is never used here. Suite A/B use the development
  benchmark; suite D uses the machine-drafted Network Engineering draft with
  ``instrument='exploratory'`` so its numbers can never be read as a validation.
"""

from __future__ import annotations

import argparse
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

#: The Automobile development benchmark (chunk ids verified present in the
#: protected index). NOT the held-out instrument.
DEV_BENCHMARK = "benchmarks/automobile-engineering-baseline-v1.json"
#: Machine-drafted, human review PENDING: an exploratory probe only.
DRAFT_BENCHMARK = "benchmarks/internet-protocols-draft-v1.json"

AUTOMOBILE_KB = "kb_f278c283c748"
NETWORK_KB = "kb_84ebff3c5baa"  # Internet Protocols KB (V19): 2 431 indexed chunks

#: Candidate probe depth, held CONSTANT across arms of one suite so the arms
#: cannot disagree about whether the evidence was in the candidate set.
DEPTH = 20


def _arm(label: str, strategy: str, top_k: int = 10, **params) -> RetrievalStrategyConfig:
    return RetrievalStrategyConfig(
        label=label, strategy=strategy, top_k=top_k, candidate_depth=DEPTH, params=params
    )


#: Suite A: which fusion strategy ranks the required evidence highest?
SUITE_A = {
    "name": "v22-a-fusion-weighted-vs-rrf",
    "hypothesis": (
        "On the Automobile development benchmark dense retrieval already finds the "
        "required chunk (recall@10 = 1.0), so no fusion can recover missing evidence; "
        "the only possible gain is RANKING. Combining dense and BM25 should lift the "
        "required evidence toward rank 1 where exact terminology or its synonyms "
        "matter, and RRF (rank-based) should be more robust than a fixed 0.65/0.35 "
        "weighted sum when the two scores live on incomparable scales."
    ),
    "baseline": _arm("dense_k10", "dense"),
    "candidates": [
        _arm("bm25_k10", "bm25"),
        _arm("hybrid_weighted_k10", "hybrid", fusion="weighted"),
        _arm("hybrid_rrf_k10", "hybrid", fusion="rrf"),
    ],
    "primary_metric": "mrr",
    "guardrail_metrics": ["recall_at_k", "latency_ms"],
    "proposed_change": "replace the dense strategy with BM25 or dense+BM25 fusion",
    "expected_effect": (
        "recall@10 stays at 1.0 for the fusion arms; MRR rises if exact terminology "
        "carries signal the dense embedding misses; BM25 alone may lose recall"
    ),
    "targeted_failure_code": "RANKING_FAILURE",
    "rejection_conditions": [
        "MRR does not rise by the declared meaningful bar (0.02)",
        "recall@10 falls below the baseline for a fusion arm",
        "latency rises above the interactive budget (see docs/v22-retrieval-intelligence.md)",
    ],
}

#: Suite B: is the residual ordering error a reranking problem?
SUITE_B = {
    "name": "v22-b-cross-encoder-reranking",
    "hypothesis": (
        "The failures left after hybrid fusion are pure ordering: the required chunk "
        "is inside the fused top-10 but below rank 1. A cross-encoder scores the "
        "query against each candidate jointly instead of comparing two independently "
        "embedded vectors, so it should reorder that SAME pool toward the evidence "
        "that actually answers the question. Recall must not move: a reranker can "
        "only permute the candidates it is given."
    ),
    "baseline": _arm("hybrid_weighted_k10", "hybrid", fusion="weighted"),
    "candidates": [
        _arm(
            "hybrid_cross_encoder_k10",
            "hybrid_reranked",
            fusion="weighted",
            reranker="cross-encoder",
            reranker_top_n=DEPTH,
        ),
    ],
    "primary_metric": "mrr",
    "guardrail_metrics": ["recall_at_k", "latency_ms"],
    "proposed_change": "rerank the fused candidate pool with a cross-encoder",
    "expected_effect": (
        "MRR and nDCG rise, recall@10 is unchanged, latency rises substantially "
        "because every candidate is scored by a second model"
    ),
    "targeted_failure_code": "RANKING_FAILURE",
    "rejection_conditions": [
        "MRR does not rise by the declared meaningful bar (0.02)",
        "recall@10 regresses: a reranker that loses evidence is worse than none",
        "the reranker is unavailable or falls back, in which case the arm is recorded "
        "as failed and no substitute implementation is used",
    ],
}

#: Suite D: does the Automobile result replicate on a different domain/corpus?
SUITE_D = {
    "name": "v22-d-second-domain-fusion-replication",
    "hypothesis": (
        "The Automobile corpus is Wikipedia prose; the Network Engineering corpus is "
        "IETF RFC text, whose questions use exact protocol terminology and numeric "
        "parameter values. If BM25/hybrid helps on the Automobile benchmark because "
        "of terminology, it should help at least as much here, and the dense arm "
        "should be relatively weaker. Directional replication only: the labels are "
        "agent-drafted and unreviewed."
    ),
    "baseline": _arm("dense_k10", "dense"),
    "candidates": [
        _arm("bm25_k10", "bm25"),
        _arm("hybrid_weighted_k10", "hybrid", fusion="weighted"),
        _arm("hybrid_rrf_k10", "hybrid", fusion="rrf"),
    ],
    "primary_metric": "mrr",
    "guardrail_metrics": ["recall_at_k", "latency_ms"],
    "proposed_change": "same fusion change as suite A, on a second domain and corpus",
    "expected_effect": (
        "the direction of the fusion effect matches suite A; absolute values are "
        "NOT comparable across domains because the corpora and labels differ"
    ),
    "targeted_failure_code": "RANKING_FAILURE",
    "rejection_conditions": [
        "MRR does not rise by the declared meaningful bar (0.02) for any arm",
        "recall@10 falls below the baseline for a fusion arm",
    ],
}

SUITES: dict[str, dict] = {
    "a": {**SUITE_A, "kb": AUTOMOBILE_KB, "benchmark": DEV_BENCHMARK,
          "instrument": "development", "lifecycle": "", "claim": ""},
    "b": {**SUITE_B, "kb": AUTOMOBILE_KB, "benchmark": DEV_BENCHMARK,
          "instrument": "development", "lifecycle": "", "claim": ""},
    "d": {**SUITE_D, "kb": NETWORK_KB, "benchmark": DRAFT_BENCHMARK,
          "instrument": "exploratory", "lifecycle": "", "claim": "exploratory"},
}


def _print(result) -> None:
    print(f"\n{'=' * 78}")
    print(f"{result.name}  ->  {result.outcome.value}")
    print("=" * 78)
    print(f"kb={result.kb_id}  instrument={result.instrument}  "
          f"benchmark={result.benchmark_name}")
    print(f"questions={len(result.matched_question_ids)}  "
          f"benchmark_sha256={result.benchmark_sha256_before}")
    for arm in result.arms:
        stats = arm.metric_stats

        def fmt(metric: str) -> str:
            stat = stats.get(metric)
            if stat is None or not stat.measured or stat.mean is None:
                return "n/a"
            return f"{stat.mean:.4f}"

        print(
            f"  arm {arm.label:<26} status={arm.status:<10} "
            f"R@k={fmt('recall_at_k')} P@k={fmt('precision_at_k')} "
            f"MRR={fmt('mrr')} nDCG={fmt('ndcg_at_k')} "
            f"latency={fmt('latency_ms')}ms codes={arm.code_counts}"
        )
    for outcome in result.candidate_outcomes:
        print(
            f"  -> {outcome.label}: {outcome.outcome.value} "
            f"(acceptance={outcome.acceptance.value}, delta={outcome.primary_mean_delta}, "
            f"paired={outcome.paired_question_count})"
        )
    print(f"\nOUTCOME: {result.outcome_explanation}")
    if result.errors:
        print(f"ERRORS: {result.errors}")
    for limitation in result.limitations:
        print(f"  limitation: {limitation}")
    print(f"integrity: index_unchanged={result.integrity.get('index_unchanged')} "
          f"corpus_unchanged={result.integrity.get('corpus_unchanged')} "
          f"benchmark_unchanged={result.integrity.get('benchmark_unchanged')}")
    print(f"persisted to: {result.notes[-1] if result.notes else 'not persisted'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--question-limit", type=int, default=0)
    args = parser.parse_args()

    if args.list:
        for key, suite in SUITES.items():
            print(f"{key}: {suite['name']}  kb={suite['kb']} instrument={suite['instrument']}")
            print(f"    primary={suite['primary_metric']} "
                  f"guardrails={suite['guardrail_metrics']}")
            print(f"    hypothesis: {suite['hypothesis'][:120]}...")
        return 0

    keys = list(SUITES) if args.suite == "all" else [s.strip() for s in args.suite.split(",")]
    for key in keys:
        if key not in SUITES:
            print(f"unknown suite {key!r}; available: {', '.join(SUITES)}")
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

        suite = SUITES[keys[0]]
        bench = load_retrieval_benchmark(resolve_benchmark_path(suite["benchmark"]))
        question_ids = [q.id for q in bench.questions][: args.question_limit]

    for key in keys:
        suite = SUITES[key]
        print(f"\n>>> running suite {key}: {suite['name']} on {suite['kb']}")
        result = runner.run(
            suite["kb"],
            name=suite["name"],
            hypothesis=suite["hypothesis"],
            baseline=suite["baseline"],
            candidates=suite["candidates"],
            primary_metric=suite["primary_metric"],
            guardrail_metrics=suite["guardrail_metrics"],
            proposed_change=suite["proposed_change"],
            expected_effect=suite["expected_effect"],
            targeted_failure_code=suite["targeted_failure_code"],
            rejection_conditions=suite["rejection_conditions"],
            benchmark_path=suite["benchmark"],
            question_ids=question_ids,
            persist=True,
            created_by="scripts/v22_retrieval_experiments.py",
            instrument=suite["instrument"],
            benchmark_lifecycle=suite["lifecycle"],
            claim_strength=suite["claim"],
        )
        _print(result)

    print(f"\nartifacts written under {runner.experiments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
