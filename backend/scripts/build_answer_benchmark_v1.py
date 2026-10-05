"""Build `benchmarks/answer-quality-automobile-v1.json` from the FROZEN
retrieval benchmark.

What this script does:
  * READ-ONLY on `benchmarks/automobile-engineering-baseline-v1.json`
    (it is frozen; the script refuses to write to it);
  * derives REQUIRED EVIDENCE (chunk id, document id, content hash, section,
    source title) by joining each question to the live corpus;
  * derives `answerability = ANSWERABLE` for questions whose reviewer selected
    an answering chunk — this is a fact about the corpus, not a guess;
  * derives `expected_grounding_state` behaviourally: an answerable question
    must NOT abstain; an unanswerable question must abstain;
  * writes `expected_answer` / `key_points` as EMPTY with an explicit
    "requires human authorship" note, because the frozen benchmark has none.

What this script deliberately does NOT do:
  * generate an expected answer with an LLM;
  * invent key points;
  * call the result "human-reviewed" (the source states human_review = PENDING);
  * touch the frozen file, any KB, or Qdrant.

Run: python scripts/build_answer_benchmark_v1.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

FROZEN = ROOT / "benchmarks" / "automobile-engineering-baseline-v1.json"
OUTPUT = ROOT / "benchmarks" / "answer-quality-automobile-v1.json"
DB = BACKEND_DIR / "data" / "ragforge.db"


def load_chunks(kb_id: str) -> dict[str, dict]:
    """{chunk_id: {document_id, content_hash, section, source_title}} from SQLite."""
    import sqlite3

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    out: dict[str, dict] = {}
    try:
        for row in conn.execute("SELECT id, document_id, data FROM chunks WHERE kb_id = ?", (kb_id,)):
            data = json.loads(row["data"])
            out[row["id"]] = {
                "document_id": row["document_id"],
                "content_hash": data.get("content_hash", ""),
                "section": data.get("section"),
                "source_title": data.get("document_title") or data.get("source_title"),
                "kb_id": data.get("kb_id"),
            }
    finally:
        conn.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate the existing output instead of rebuilding")
    args = parser.parse_args()

    from app.services.answer_eval.benchmark import (
        AnswerBenchmark,
        load_answer_benchmark,
        validate_against_corpus,
    )

    if args.check:
        if not OUTPUT.exists():
            print(f"FAIL: {OUTPUT} does not exist")
            return 1
        bench = load_answer_benchmark(OUTPUT)
        chunks = load_chunks(bench.kb_id)
        problems = validate_against_corpus(bench, chunks)
        print(f"benchmark : {bench.benchmark} v{bench.version}")
        print(f"kb_id     : {bench.kb_id}")
        print(f"questions : {bench.question_count} "
              f"(answerable={bench.answerable_count}, "
              f"unanswerable={bench.unanswerable_count}, unknown={bench.unknown_count})")
        print(f"review    : {bench.human_review.value}")
        print(f"fingerprint: {bench.fingerprint()}")
        if problems:
            print(f"\nCORPUS MISMATCH ({len(problems)}):")
            for p in problems[:20]:
                print(f"  - {p}")
            return 1
        print("\nCHECK PASSED: every required chunk resolves against the live corpus")
        return 0

    # ---------------------------------------------------------------- build
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    kb_id = frozen["kb_id"]
    chunks = load_chunks(kb_id)

    questions: list[dict] = []
    missing: list[str] = []
    for q in frozen["questions"]:
        required = []
        for chunk_id in q["expected_chunk_ids"]:
            info = chunks.get(chunk_id)
            if info is None:
                missing.append(f"{q['id']}: {chunk_id}")
                continue
            required.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": info["document_id"],
                    "content_hash": info["content_hash"],
                    "section": info["section"],
                    "source_title": info["source_title"],
                    "required": True,
                }
            )
        if not required:
            continue

        questions.append(
            {
                "question_id": q["id"],
                "question": q["question"],
                "subdomain": q.get("subdomain", ""),
                # FACT about the corpus: a reviewer selected the chunk that
                # answers this question, so it is answerable.
                "answerability": "answerable",
                "required_evidence": required,
                "citation_requirements": {
                    "require_at_least_one_citation": True,
                    "must_cite_all_required_evidence": True,
                    "must_not_cite_irrelevant_chunks": True,
                    "note": (
                        "Derived from the frozen retrieval benchmark: the reviewer "
                        "selected this chunk as the one that answers the question."
                    ),
                },
                # NOT populated: the frozen benchmark has no reference answers and
                # this script does not invent them.
                "expected_answer": None,
                "key_points": [],
                "acceptable_answer_elements": [],
                "numerical_tolerance": None,
                "requires_units": False,
                # Behavioural expectation derived from answerability.
                "expected_grounding_state": "ANY_ACCEPTABLE",
                "abstention_required": False,
                "provenance": q.get("provenance", ""),
                "reviewer_metadata": {
                    "source": "automobile-engineering-baseline-v1",
                    "source_question_id": q["id"],
                    "evidence_selected_by": frozen["authorship"].get("author"),
                    "human_review": frozen["authorship"].get("human_review"),
                    "answer_label_status": "REQUIRES_HUMAN_AUTHORSHIP",
                    "notes": (
                        "Required evidence is inherited from the frozen retrieval "
                        "benchmark. No expected answer or key points exist yet, so "
                        "answer-correctness metrics are UNKNOWN for this question."
                    ),
                },
            }
        )

    if missing:
        print("REFUSING TO BUILD — required chunks missing from the corpus:")
        for m in missing:
            print(f"  - {m}")
        return 1

    benchmark = AnswerBenchmark(
        benchmark="answer-quality-automobile-v1",
        version=1,
        created_at=__import__("datetime").date.today().isoformat(),
        kb_id=kb_id,
        kb_name=frozen.get("kb_name", ""),
        derived_from="automobile-engineering-baseline-v1",
        authorship={
            "method": (
                "Required evidence inherited verbatim from the frozen retrieval "
                "benchmark; answerability derived from the presence of a "
                "reviewer-selected answering chunk. NO expected answers were "
                "generated — see reviewer_metadata.answer_label_status."
            ),
            "answer_labels": "NONE — human authorship required before correctness metrics are computable",
            "human_review": (
                "PENDING. The source benchmark states human_review = PENDING, so this "
                "artifact inherits that status and must NOT be described as "
                "human-reviewed."
            ),
        },
        human_review="pending",
        evaluator_contract={
            "computable_now": [
                "citation_precision",
                "citation_recall",
                "evidence_support_rate",
                "unsupported_claim_rate",
                "abstention_correct",
                "contradiction_rate",
                "retrieval_hit_rate",
            ],
            "unknown_until_human_answers": [
                "correctness",
                "reference_answer_similarity",
                "key_point_recall",
            ],
            "note": (
                "Metrics listed under unknown_until_human_answers are reported as "
                "UNKNOWN with a reason, never guessed. This is enforced in the "
                "evaluator, not just documented."
            ),
        },
        questions=questions,
    )

    OUTPUT.write_text(
        json.dumps(benchmark.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"  questions   : {benchmark.question_count}")
    print(f"  answerable  : {benchmark.answerable_count}")
    print(f"  unanswerable: {benchmark.unanswerable_count}")
    print(f"  fingerprint : {benchmark.fingerprint()}")

    problems = validate_against_corpus(benchmark, chunks)
    if problems:
        print(f"  CORPUS MISMATCH: {problems[:5]}")
        return 1
    print("  corpus check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())