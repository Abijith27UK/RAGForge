"""Live end-to-end answer-quality evaluation on a SCRATCH knowledge base.

Safety contract (the critical part):
  * it creates its OWN scratch KB and records the id;
  * it never enumerates, modifies or deletes any pre-existing KB;
  * it asserts the Qdrant collection count returns to its starting value;
  * it deletes the scratch KB in a `finally` block.

It runs a real evaluation against real indexed chunks through the shipping
answering pipeline, then reports the ACTUAL metrics — including which metrics
are UNKNOWN, which is the honest headline here.

Run: python scripts/smoke_answer_eval_v8.py [base_url]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8013").rstrip("/")

DOC = """# Hydrostatics and Stability

The metacentric height GM must be positive for stable equilibrium of a floating body.
The transverse metacentric height is derived from the righting arm GZ divided by the heel angle phi.
Free surface effect reduces the effective metacentric height in a flooded compartment.

# Propulsion

Propeller cavitation occurs when the local pressure falls below the vapour pressure of water.
The advance ratio relates the advance per revolution to the propeller diameter.
"""

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def qdrant_count() -> int | None:
    try:
        return len(httpx.get("http://localhost:6333/collections", timeout=10).json()["result"]["collections"])
    except Exception:
        return None


def main() -> int:
    c = httpx.Client(base_url=BASE, timeout=600.0)
    before = qdrant_count()
    print(f"Qdrant collections before: {before}")

    kb = c.post("/api/knowledge-bases", json={
        "name": "SMOKE — answer evaluation scratch",
        "domain": "Naval Architecture",
        "purpose": "answer evaluation smoke",
        "target_audience": "students",
        "source_mode": "user_provided",
    })
    check("scratch KB created", kb.status_code == 201, str(kb.status_code))
    kb_id = kb.json()["id"]
    print(f"  scratch kb_id = {kb_id}")

    try:
        r = c.post(f"/api/knowledge-bases/{kb_id}/documents/upload",
                   files={"files": ("naval.md", DOC.encode(), "text/markdown")},
                   data={"index": "true", "chunker": "section-aware",
                         "target_size": "350", "overlap": "40"})
        check("document indexed", r.status_code == 201 and r.json()["indexed"] is True)

        chunks_raw = c.get(f"/api/knowledge-bases/{kb_id}/chunks").json()
        chunk_list = chunks_raw if isinstance(chunks_raw, list) else chunks_raw.get("chunks", [])
        check("chunks created", len(chunk_list) >= 2, f"{len(chunk_list)} chunks")

        # Build a scratch benchmark from the REAL chunk ids the indexer produced.
        # Q1 targets the stability chunk; Q2 is deliberately off-topic and must abstain.
        stability = next((c_ for c_ in chunk_list
                          if "metacentric" in (c_.get("text") or "").lower()), chunk_list[0])
        unrelated = next((c_ for c_ in chunk_list if c_["id"] != stability["id"]), None)

        questions = [{
            "question_id": "smoke-001",
            "question": "How is the transverse metacentric height derived?",
            "answerability": "answerable",
            "required_evidence": [{
                "chunk_id": stability["id"],
                "document_id": stability["document_id"],
                "content_hash": stability.get("content_hash", ""),
                "required": True,
            }],
            "citation_requirements": {
                "require_at_least_one_citation": True,
                "must_cite_all_required_evidence": True,
                "must_not_cite_irrelevant_chunks": True,
            },
            "expected_answer": None,
            "key_points": [],
            "expected_grounding_state": "ANY_ACCEPTABLE",
            "abstention_required": False,
        }]
        if unrelated:
            questions.append({
                "question_id": "smoke-002",
                "question": "What is the recommended hull plating thickness for a 40 metre yacht?",
                "answerability": "unanswerable",
                "required_evidence": [],
                "citation_requirements": {
                    "require_at_least_one_citation": False,
                    "must_cite_all_required_evidence": False,
                    "must_not_cite_irrelevant_chunks": True,
                },
                "expected_answer": None,
                "key_points": [],
                "expected_grounding_state": "INSUFFICIENT_EVIDENCE",
                "abstention_required": True,
            })

        bench = {
            "benchmark": "smoke-answer-eval",
            "version": 1, "created_at": "2026-01-01", "kb_id": kb_id,
            "human_review": "pending", "questions": questions,
        }
        # Absolute path: the API resolves benchmark_path relative to ITS own
        # working directory, which is not this script's.
        path = str((BACKEND_DIR / "data" / f"_smoke_answer_eval_{kb_id}.json").resolve())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bench, f, indent=2)

        r = c.post(f"/api/knowledge-bases/{kb_id}/answer-evaluation/runs",
                   json={"benchmark_path": path, "strategy": "dense"})
        check("evaluation run 200", r.status_code == 200, r.text[:200])
        run = r.json()

        agg = run["aggregate"]
        print("\n--- MEASURED METRICS ---")
        for name in ("citation_precision", "citation_recall", "citation_completeness",
                     "evidence_support_rate", "unsupported_claim_rate",
                     "contradiction_rate", "retrieval_hit_rate",
                     "abstention_accuracy", "grounding_state_accuracy",
                     "false_supported_rate", "false_unsupported_rate",
                     "hallucination_rate", "pass_rate"):
            m = agg[name]
            shown = f"{m['value']:.3f}" if m["measured"] and m["value"] is not None else "UNKNOWN"
            print(f"  {name:26} {shown:>8}  (n={m.get('sample_size')})")
        print("\n--- UNKNOWN METRICS ---")
        for name in agg["unknown_metrics"]:
            print(f"  {name}")
        print("\ngrounding distribution:", json.dumps(agg["grounding_state_distribution"]))
        print("confusion:", json.dumps(agg["confusion_matrix"]))

        print("\n--- PER QUESTION ---")
        for q in run["per_question"]:
            # `passed` is a derived property, not a serialized field.
            passed = (q.get("abstention_correct") is not False
                      and q.get("grounding_state_correct") is not False)
            print(f"  {q['question_id']} [{q['answerability']}] "
                  f"state={q['actual_grounding_state']} status={q['answer_status']} "
                  f"passed={passed}")
            print(f"     question : {q['question'][:70]}")
            print(f"     answer   : {q['answer_text'][:100]}")
            print(f"     required ={q['required_chunk_ids']}")
            print(f"     cited    ={q['cited_chunk_ids']}")
            print(f"     abstention_correct={q['abstention_correct']} "
                  f"false_supported={q['false_supported']} "
                  f"false_unsupported={q['false_unsupported']}")
            for v in q["claim_verdicts"]:
                print(f"     claim {v['claim_id']}: entailment={v['entailment']} "
                      f"cited={v['resolved_chunk_ids']} problems={v['problems']}")
            if q["warnings"]:
                print(f"     warnings : {q['warnings']}")

        print("\n--- RUN PROVENANCE ---")
        print(f"  evaluator={run['evaluator_name']} {run['evaluator_version']}")
        print(f"  entailment={run['entailment_provider']} "
              f"model_based={run['entailment_is_model_based']}")
        print(f"  generator={run['generator']} model={run['model']} is_mock={run['is_mock']}")
        print(f"  strategy={run['strategy']} subset={run['subset_note']}")
        print(f"  warnings: {run['warnings']}")

        # ---- assertions -------------------------------------------------
        check("run records evaluator identity", bool(run["evaluator_name"] and run["evaluator_version"]))
        check("entailment provider recorded",
              run["entailment_provider"] == "heuristic-lexical-coverage"
              and run["entailment_is_model_based"] is False)
        check("correctness is UNKNOWN (no human answers)", "correctness" in agg["unknown_metrics"])
        check("correctness is not reported as zero",
              agg["correctness"]["measured"] is False and agg["correctness"]["value"] is None)
        check("run warns that correctness is unmeasured",
              any("CORRECTNESS" in w for w in run["warnings"]))
        check("grounding distribution populated", bool(agg["grounding_state_distribution"]))

        # The unanswerable question, if present, must be handled correctly.
        unans = next((q for q in run["per_question"] if q["answerability"] == "unanswerable"), None)
        if unans:
            check("unanswerable question was not hallucinated",
                  unans["false_unsupported"] is False,
                  f"state={unans['actual_grounding_state']} status={unans['answer_status']}")
        else:
            check("unanswerable case present", False, "no unanswerable question was generated")

        listed = c.get(f"/api/knowledge-bases/{kb_id}/answer-evaluation/runs").json()
        check("run persisted and listed", any(x["id"] == run["id"] for x in listed))

    finally:
        r = c.delete(f"/api/knowledge-bases/{kb_id}")
        check("scratch KB deleted", r.status_code in (200, 204), str(r.status_code))
        for leftover in (BACKEND_DIR / "data").glob(f"_smoke_answer_eval_*.json"):
            leftover.unlink(missing_ok=True)

    after = qdrant_count()
    print(f"\nQdrant collections after: {after}")
    if before is not None and after is not None:
        check("no collection leaked", after == before, f"{before} -> {after}")

    print("\n" + "=" * 66)
    if FAILURES:
        print(f"SMOKE FAILED — {len(FAILURES)}: {FAILURES}")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())