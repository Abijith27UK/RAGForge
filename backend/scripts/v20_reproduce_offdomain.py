"""V20 — reproduce the off-domain evidence-gate behaviour against the real KB.

Read-only probe. It runs the SAME pipeline the MCP tool `answer_from_knowledge_base`
runs (AnsweringService -> RetrievalService -> EvidenceGate), against the real
workspace SQLite + Qdrant, and prints the gate signals that drove each decision so
the root cause is visible rather than inferred.

It writes nothing: no ingest, no build, no benchmark scoring, no review. Retrieval
runs are recorded exactly as a UI/API query would record them.

Usage
-----
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\v20_reproduce_offdomain.py --kb kb_f278c283c748
    ... --question "How many moons does Jupiter have?" --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Settings  # noqa: E402
from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.services.answering.query_processor import QueryNature, extract_terms  # noqa: E402
from app.services.answering.service import AnsweringService  # noqa: E402

DEFAULT_QUESTIONS = [
    "What is an engine?",
    "How many moons does Jupiter have?",
    "Describe the courtship display of the Australian lyrebird.",
]

#: Measured separation on the protected Automobile corpus (see MIN_RELEVANCE_TO_ANSWER):
#: 12 in-domain questions top out at cosine 0.4365..0.6946, 6 off-domain questions at
#: 0.2317..0.3411. Kept in the script so the calibration can be re-run, not just cited.


def probe(kb_id: str, question: str) -> dict:
    settings = Settings()
    repo = Repository(str(settings.db_path))
    service = AnsweringService(repo, settings)
    answer, trace, assessment = service.answer(kb_id, question)

    plan = trace.query_plan
    signals = {s.name: {"value": s.value, "threshold": s.threshold,
                        "passed": s.passed, "measured": s.measured,
                        "interpretation": s.interpretation} for s in assessment.signals}
    items = list(trace.evidence_items or [])
    evidence_text = " ".join((e.content or "").lower() for e in items)
    q_terms = set(plan.extracted_terms) if plan else set()
    matched = sorted(t for t in q_terms if t and t in evidence_text)

    return {
        "question": question,
        "extracted_terms": list(q_terms),
        "terms_matched_in_evidence": matched,
        "retrieved_chunks": len(items),
        "status": answer.status.value if hasattr(answer.status, "value") else str(answer.status),
        "grounding_state": assessment.grounding_state.value,
        "decision": assessment.decision.value,
        "reason_code": assessment.reason_code,
        "confidence": assessment.confidence.value if hasattr(assessment.confidence, "value") else str(assessment.confidence),
        "sufficient": assessment.sufficient,
        "citations": len(answer.citations or []),
        "claims": len(answer.claims or []),
        "nature": trace.query_trace.nature.value if trace.query_trace else None,
        "signals": signals,
        "reason": assessment.reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--question", action="append", default=None)
    parser.add_argument("--json", default="")
    ns = parser.parse_args(argv)

    questions = ns.question or DEFAULT_QUESTIONS
    out = []
    for q in questions:
        try:
            row = probe(ns.kb, q)
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not crash
            row = {"question": q, "error": f"{type(exc).__name__}: {exc}"}
        out.append(row)
        print("=" * 78)
        print(f"Q: {q}")
        if "error" in row:
            print(f"  ERROR: {row['error']}")
            continue
        print(f"  terms={row['extracted_terms']}")
        print(f"  matched_in_evidence={row['terms_matched_in_evidence']}")
        print(f"  chunks={row['retrieved_chunks']} nature={row['nature']}")
        print(f"  -> status={row['status']} state={row['grounding_state']} "
              f"decision={row['decision']} code={row['reason_code']}")
        print(f"     sufficient={row['sufficient']} citations={row['citations']} claims={row['claims']}")
        la = row["signals"].get("lexical_alignment", {})
        print(f"     lexical_alignment={la.get('value')} (threshold {la.get('threshold')}, "
              f"passed={la.get('passed')})")
        rf = row["signals"].get("evidence_relevance_floor", {})
        print(f"     relevance_floor={rf.get('value')} (threshold {rf.get('threshold')}, "
              f"passed={rf.get('passed')}, measured={rf.get('measured')})")
        if not rf.get("measured"):
            print(f"       NOT APPLICABLE: {rf.get('not_performed_reason')}")
        cf = row["signals"].get("corpus_vocabulary_fit", {})
        print(f"     corpus_vocabulary_fit={cf.get('value')} "
              f"(threshold {cf.get('threshold')}, passed={cf.get('passed')}, "
              f"measured={cf.get('measured')})")
        print(f"       {cf.get('interpretation') or cf.get('not_performed_reason')}")
        print(f"     reason[{row['reason_code']}]: {row['reason'][:400]}")

    if ns.json:
        p = Path(ns.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwritten: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
