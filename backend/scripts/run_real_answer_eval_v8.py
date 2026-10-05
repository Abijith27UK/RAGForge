"""Run the REAL 28-question answer benchmark against the real Automobile KB.

READ-ONLY with respect to the corpus: this asks questions and scores answers. It
does NOT modify documents, chunks, sources, the benchmark artifact, or Qdrant
vectors.

SIDE EFFECT (measured, not assumed): `persist=False` suppresses only the
answer-EVALUATION-RUN row. Each question still goes through the normal answering
path, which writes an `answers` row (plus its retrieval run/trace) against the
real knowledge base. Verified on 2026-10-03: three runs left 84 answer rows in
kb_f278c283c748 while `answer_evaluation_runs` stayed at 0. Use a scratch KB if
you need a genuinely clean database, and report the row count either way.

This is the honest headline measurement: what fraction of answers are correctly
CITED and correctly ABSTAINING on a corpus a human selected evidence for.

Run: python scripts/run_real_answer_eval_v8.py [base_url] [strategy]
"""
from __future__ import annotations

import json
import sys

import httpx

# Windows consoles default to cp1252 and raise UnicodeEncodeError on any
# non-ASCII character that leaked in from real corpus text (e.g. the '↑' used in
# some automobile documents). Reconfigure instead of mangling the output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - non-tty streams
        pass

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8013").rstrip("/")
STRATEGY = sys.argv[2] if len(sys.argv) > 2 else "dense"

# The API resolves benchmark_path relative to the backend working directory.
BENCH = "benchmarks/answer-quality-automobile-v1.json"


def _safe(text: str | None) -> str:
    """Collapse text to printable ASCII so console encoding cannot break a run."""
    if not text:
        return ""
    return text.encode("ascii", "replace").decode("ascii")


def main() -> int:
    c = httpx.Client(base_url=BASE, timeout=3600.0)
    r = c.post("/api/knowledge-bases/kb_f278c283c748/answer-evaluation/runs",
               json={"benchmark_path": BENCH, "strategy": STRATEGY, "persist": False})
    if r.status_code != 200:
        print(f"FAILED {r.status_code}: {r.text[:400]}")
        return 1
    run = r.json()
    agg = run["aggregate"]

    print(f"benchmark : {run['benchmark_name']} v{run['benchmark_version']}")
    print(f"fingerprint: {run['benchmark_fingerprint']}")
    print(f"strategy  : {run['strategy']}")
    print(f"generator : {run['generator']} / {run['model']} (is_mock={run['is_mock']})")
    print(f"evaluator : {run['evaluator_name']} {run['evaluator_version']}")
    print(f"entailment: {run['entailment_provider']} (model_based={run['entailment_is_model_based']})")
    print(f"questions : {run['question_count']} "
          f"(answerable={run['answerable_count']}, unanswerable={run['unanswerable_count']})")

    print("\n=== MEASURED ===")
    for name in ("citation_precision", "citation_recall", "citation_completeness",
                 "evidence_support_rate", "unsupported_claim_rate",
                 "contradiction_rate", "retrieval_hit_rate",
                 "abstention_accuracy", "grounding_state_accuracy",
                 "false_supported_rate", "false_unsupported_rate",
                 "hallucination_rate", "pass_rate"):
        m = agg[name]
        val = f"{m['value']:.3f}" if m["measured"] and m["value"] is not None else "UNKNOWN"
        print(f"  {name:26} {val:>8}   n={m.get('sample_size')}")

    print("\n=== UNKNOWN (not zero) ===")
    print("  " + (", ".join(agg["unknown_metrics"]) or "none"))
    print(f"  {agg['unknown_metrics'] and 'reason: no human-authored reference answers exist' or ''}")

    print(f"\ngrounding distribution: {json.dumps(agg['grounding_state_distribution'])}")

    print("\n=== FAILURES ===")
    for q in run["per_question"]:
        passed = (q.get("abstention_correct") is not False
                  and q.get("grounding_state_correct") is not False
                  and not q.get("problems"))
        if passed:
            continue
        print(f"\n  {q['question_id']}  state={q['actual_grounding_state']} "
              f"status={q['answer_status']}")
        print(f"    Q: {q['question'][:80]}")
        print(f"    A: {_safe(q['answer_text'])[:110]}")
        print(f"    required={q['required_chunk_ids']} cited={q['cited_chunk_ids']}")
        print(f"    retrieved_rank_of_required={q['retrieved_rank_of_required']}")
        print(f"    retrieval_hit_rate={q['retrieval_hit_rate']['value']} "
              f"citation_precision={q['citation_precision']['value']}")
        for p in q["problems"][:4]:
            print(f"    ! {_safe(p)}")

    print("\n=== RETRIEVAL MISSES (answerer not to blame) ===")
    misses = [q for q in run["per_question"] if q["retrieval_hit_rate"]["measured"]
              and q["retrieval_hit_rate"]["value"] == 0]
    print(f"  {len(misses)} question(s) where the required chunk was never retrieved")
    for q in misses[:8]:
        print(f"    {q['question_id']}: {q['question'][:70]}")

    print(f"\nfinal_score computable: "
          f"{sum(1 for q in run['per_question'] if q['final_score'] is not None)}/"
          f"{len(run['per_question'])}")
    print("\nwarnings:")
    for w in run["warnings"]:
        print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())