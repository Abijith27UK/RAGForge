"""V20 — measure a retrieval RELEVANCE signal for in-domain vs off-domain questions.

Read-only. The gate already records retrieval scores but never uses them in a
decision. This script measures whether the raw similarity actually separates
answerable from unanswerable questions on the real corpus, before any threshold
is chosen from it.

For each question it prints, for the top-k retrieved chunks:
  * the reported retrieval_score (pool-relative)
  * every entry of score_breakdown (raw dense cosine / raw BM25, unnormalized)

Nothing is written except the optional --json report.
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
from app.services.retrieval.service import RetrievalService  # noqa: E402

IN_DOMAIN = [
    "What is an engine?",
    "How do brakes work?",
    "Explain the steering system.",
    "What are the main components of a suspension system?",
    "What is a turbocharger and how does it work?",
    "What is the firing order of a 4-cylinder engine?",
    "What is the difference between a diesel and a petrol engine?",
    "What is the purpose of a catalytic converter?",
    "What is the torque specification for the cylinder head bolts?",
    "What is the purpose of a clutch?",
    "How does a transmission work?",
    "What is the function of a piston?",
]

OFF_DOMAIN = [
    "How many moons does Jupiter have?",
    "Describe the courtship display of the Australian lyrebird.",
    "What are the EASA regulations for commercial drone certification?",
    "What is buoyancy?",
    "Who won the 2018 FIFA World Cup?",
    "What is the capital of Peru?",
]


def run(kb_id: str, question: str, top_k: int) -> dict:
    settings = Settings()
    repo = Repository(str(settings.db_path))
    service = RetrievalService(repo, settings)
    response = service.run(kb_id, question, persist_run=False)
    rows = []
    for r in response.results[:top_k]:
        rows.append({
            "rank": getattr(r, "rank", None),
            "retrieval_score": getattr(r, "retrieval_score", None),
            "score_breakdown": dict(getattr(r, "score_breakdown", None) or {}),
        })
    return {
        "question": question,
        "strategy": response.strategy,
        "top_k": top_k,
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", default="")
    ns = parser.parse_args(argv)

    out = []
    for group, questions in (("IN-DOMAIN", IN_DOMAIN), ("OFF-DOMAIN", OFF_DOMAIN)):
        print("#" * 78)
        print(f"# {group}")
        for q in questions:
            row = run(ns.kb, q, ns.top_k)
            row["group"] = group
            out.append(row)
            breakdown = row["results"][0]["score_breakdown"] if row["results"] else {}
            raw = {k: v for k, v in breakdown.items() if k != "strategy"}
            print(f"  {row['strategy']:>14s}  {str(round(row['results'][0]['retrieval_score'], 4)) if row['results'] else 'n/a':>7s}"
                  f"  raw={json.dumps(raw, default=str)[:110]:<112s}  {q}")

    if ns.json:
        p = Path(ns.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwritten: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
