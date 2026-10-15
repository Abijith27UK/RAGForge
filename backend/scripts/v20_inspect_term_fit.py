"""V20 — inspect question terms against the KB domain spec and corpus statistics.

Read-only analysis used to calibrate the V20 distinctive-term rule. Prints, per
question:
  * extracted content terms
  * which of them the knowledge base's DomainSpec recognises (plan.domain_terms)
  * corpus df / BM25 IDF for each term (from the PERSISTED lexical index)
  * whether the term occurs in the retrieved evidence

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
from app.services.answering.query_processor import create_query_processor  # noqa: E402
from app.services.answering.service import AnsweringService  # noqa: E402
from app.services.answering.term_stats import (  # noqa: E402
    create_term_stats_provider,
)

QUESTIONS = [
    "What is an engine?",
    "How do brakes work?",
    "Explain the steering system.",
    "What are the main components of a suspension system?",
    "What is a turbocharger and how does it work?",
    "What is the firing order of a 4-cylinder engine?",
    "What is the difference between a diesel and a petrol engine?",
    "What is the purpose of a catalytic converter?",
    "What is the torque specification for the cylinder head bolts?",
    "What is buoyancy?",
    "How many moons does Jupiter have?",
    "Describe the courtship display of the Australian lyrebird.",
    "What are the EASA regulations for commercial drone certification?",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--json", default="")
    ns = parser.parse_args(argv)

    settings = Settings()
    repo = Repository(str(settings.db_path))
    kb = repo.get_kb(ns.kb)
    spec = repo.get_domain_spec(ns.kb)
    stats = create_term_stats_provider(repo).stats(ns.kb)
    proc = create_query_processor()
    service = AnsweringService(repo, settings)

    print(f"kb={ns.kb} domain={kb.domain!r}")
    print(f"stats: available={stats.available} n_docs={stats.n_docs} "
          f"terms={len(stats.df)} source={stats.source} reason={stats.reason or '-'}")
    if spec is not None:
        print(f"domain spec terminology({len(spec.terminology)}): {spec.terminology[:25]}")
        print(f"domain spec key_concepts({len(spec.key_concepts)}): {spec.key_concepts[:25]}")
        print(f"domain spec entities({len(spec.entities)}): {spec.entities[:15]}")
        print(f"domain spec subdomains({len(spec.subdomains)}): {spec.subdomains[:15]}")

    rows = []
    for q in QUESTIONS:
        plan = proc.process(q, kb, spec)
        answer, trace, assessment = service.answer(ns.kb, q)
        items = list(trace.evidence_items or [])
        text = " ".join((e.content or "").lower() for e in items)
        terms = list(plan.extracted_terms)
        detail = []
        for t in terms:
            idf_v = stats.inverse_document_frequency(t)
            detail.append({
                "term": t,
                "df": stats.document_frequency(t),
                "idf": None if idf_v is None else round(idf_v, 3),
                "absent_from_corpus": stats.is_absent_from_corpus(t),
                "in_evidence": t in text,
            })
        row = {
            "question": q,
            "terms": terms,
            "domain_terms": list(plan.domain_terms),
            "term_detail": detail,
            "status": answer.status.value,
            "reason_code": assessment.reason_code,
        }
        rows.append(row)
        print("=" * 76)
        print(f"Q: {q}")
        print(f"  domain_terms={row['domain_terms']}")
        print(f"  -> {row['status']} / {row['reason_code']}")
        for d in detail:
            print(f"     {d['term']:16s} df={str(d['df']):>6} idf={str(d['idf']):>7} "
                  f"absent={str(d['absent_from_corpus']):5s} in_evidence={d['in_evidence']}")

    if ns.json:
        p = Path(ns.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwritten: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
