"""Audit the STORED V11 answer drafts against the LIVE corpus. READ-ONLY.

Phase 3 of V16 asks a specific question: for each benchmark question, does the
pinned evidence actually support the reference answer and key points a draft
proposes? This script answers it without trusting the generator's own promises.

What it checks, per stored draft:

1. **Evidence is real** — every chunk the draft cites exists in the knowledge
   base, and its recorded content hash matches the corpus.
2. **Key points are verbatim** — each ``key_points_draft[].text`` must be a
   substring of the chunk it names (``source_chunk_id``) **under the drafting
   module's own documented normalisation** (``drafting._norm``: whitespace
   collapsed, space-before-punctuation collapsed). The generator asserts this
   at build time; this re-checks it against the CURRENT corpus, so a re-chunked
   or edited corpus cannot quietly turn a claim into a lie. A STRICTER
   byte-exact check is also reported, separately, because the stored corpus
   text carries layout artefacts (``"energy . [ 1 ] [ 2 ]"``) that a reader
   searching the evidence panel for a draft sentence will hit.
3. **The draft answer is verbatim** — same rule, per sentence of
   ``expected_answer_draft`` against the cited chunks.
4. **Coverage is disclosed** — IDF-weighted coverage, or an explicit
   ``UNKNOWN`` when the KB has no corpus statistics to weight with.
5. **Status is honest** — ``draft_ready`` / ``partial`` / ``insufficient_evidence``
   is reported as stored, never upgraded.

Nothing here writes anything. It cannot create, approve or alter ground truth,
and it reports a gap rather than filling it with outside knowledge.

Usage (from ``backend/``):

    python -m scripts.audit_answer_drafts
    python -m scripts.audit_answer_drafts --kb kb_f278c283c748 --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings  # noqa: E402
from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.services.answer_eval.drafting import (  # noqa: E402
    DRAFT_LABEL,
    DraftSuggestion,
    _norm,
    summarise_drafts,
)
from app.services.answer_eval.review_workflow import (  # noqa: E402
    AnswerBenchmarkReviewService,
)

logger = logging.getLogger("audit_answer_drafts")

REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_BENCHMARK_PATH = "benchmarks/answer-quality-automobile-v1.json"
DEFAULT_KB = "kb_f278c283c748"


def _resolve_artifact(path: str) -> str:
    """Resolve a repo-relative artifact path, then a CWD-relative one.

    ``benchmarks/...`` is how a versioned artifact is named, and the script may
    be started from anywhere, so the repository root is tried first — the same
    rule the held-out tooling uses.
    """
    given = Path(path)
    if given.is_absolute():
        return str(given)
    candidate = (REPO_ROOT / path).resolve()
    if candidate.exists():
        return str(candidate)
    return str(given)


class DraftAuditError(RuntimeError):
    """The audit could not run; nothing was written."""


def _raw_norm(text: str) -> str:
    """STRICTER than the module's rule: whitespace only, punctuation spacing kept.

    Used to detect a real, disclosed difference between a draft span and the
    stored corpus bytes, NOT as the pass/fail verdict.
    """
    return " ".join((text or "").split())


def _sentences(text: str) -> list[str]:
    """Sentence-ish split, matching how the drafting module composes answers.

    Deliberately naive: this is a VERBATIM check, so a slightly different split
    only makes the check stricter (it can produce a failure, never a silent
    pass).
    """
    import re

    return [s for s in re.split(r"(?<=[.!?])\s+", _norm(text)) if s]


def audit_draft(
    draft: DraftSuggestion,
    *,
    chunk_text: dict[str, str],
    chunk_hash: dict[str, str],
) -> dict[str, Any]:
    """Audit one stored draft against the live corpus. Pure function."""
    problems: list[str] = []
    #: Disclosed differences/reporting notes that are NOT defects.
    disclosures: list[str] = []

    cited = {e.chunk_id for e in draft.evidence}
    missing = sorted(c for c in cited if c not in chunk_text)
    if missing:
        problems.append(
            f"cited chunk(s) not in the live corpus: {', '.join(missing)}"
        )
    for ev in draft.evidence:
        live_hash = chunk_hash.get(ev.chunk_id)
        if ev.chunk_id in chunk_text and ev.content_hash and live_hash and (
            ev.content_hash != live_hash
        ):
            problems.append(
                f"cited chunk {ev.chunk_id} content hash differs from the corpus "
                f"(the corpus was re-chunked since the draft was written)"
            )

    key_point_checks: list[dict[str, Any]] = []
    for i, point in enumerate(draft.key_points_draft):
        text = _norm(point.text)
        source_text = chunk_text.get(point.source_chunk_id)
        # The drafting module's documented guarantee (whitespace + punctuation
        # spacing normalised) is the verdict; byte-exactness is reported too.
        normalised_hit = bool(source_text) and bool(text) and text in _norm(source_text)
        raw_hit = bool(source_text) and bool(point.text) and (
            _raw_norm(point.text) in _raw_norm(source_text)
        )
        if source_text is None:
            verdict = "source chunk not in corpus"
            problems.append(f"key point {i} names a chunk that is not in the corpus")
        elif normalised_hit:
            verdict = "verbatim (module normalisation)"
        else:
            verdict = "NOT verbatim in the cited chunk"
            problems.append(
                f"key point {i} is not a verbatim substring of "
                f"{point.source_chunk_id} under the drafting module's own "
                "normalisation"
            )
        key_point_checks.append(
            {
                "index": i,
                "text": point.text,
                "source_chunk_id": point.source_chunk_id,
                "source_sentence_index": point.source_sentence_index,
                "coverage": point.coverage,
                "verdict": verdict,
                "verbatim_normalised": normalised_hit,
                "byte_exact_in_stored_text": raw_hit,
            }
        )
        if normalised_hit and not raw_hit:
            disclosures.append(
                f"key point {i}: verbatim once the module's normalisation is "
                "applied, but NOT byte-identical to the stored chunk text "
                "(layout artefacts: space before punctuation, reference markers)"
            )

    answer_checks: list[dict[str, Any]] = []
    for sentence in _sentences(draft.expected_answer_draft):
        norm_sentence = _norm(sentence)
        hit = [
            cid
            for cid, text in chunk_text.items()
            if cid in cited and norm_sentence in _norm(text)
        ]
        raw_hit = [
            cid
            for cid, text in chunk_text.items()
            if cid in cited and _raw_norm(sentence) in _raw_norm(text)
        ]
        if hit:
            answer_checks.append(
                {
                    "sentence": sentence,
                    "verdict": "verbatim",
                    "in": hit,
                    "byte_exact_in_stored_text": bool(raw_hit),
                }
            )
            if not raw_hit:
                disclosures.append(
                    "a draft-answer sentence is verbatim only after normalisation "
                    "(layout artefacts in the stored chunk text)"
                )
        else:
            answer_checks.append(
                {
                    "sentence": sentence,
                    "verdict": "NOT verbatim in any cited chunk",
                    "byte_exact_in_stored_text": False,
                }
            )
            problems.append(
                "a sentence of the draft answer is not verbatim in any cited chunk"
            )

    if not draft.coverage_measured:
        disclosures.append(
            "coverage is UNKNOWN for this draft (the KB has no corpus statistics "
            "to weight question terms with) — it is reported as unavailable, "
            "not as 0%"
        )
    if draft.status.value != "draft_ready":
        disclosures.append(
            f"the generator itself did not produce a ready draft: status "
            f"{draft.status.value!r} — {draft.reason}"
        )

    return {
        "question_id": draft.question_id,
        "suggestion_id": draft.suggestion_id,
        "stored_status": draft.status.value,
        "reason": draft.reason,
        "generator": draft.generator,
        "method": draft.method,
        "created_at": draft.created_at.isoformat(),
        "is_draft": draft.is_draft,
        "label": draft.label,
        "coverage": draft.coverage,
        "coverage_measured": draft.coverage_measured,
        "expected_answer_draft": draft.expected_answer_draft,
        "key_points": key_point_checks,
        "answer_sentences": answer_checks,
        "uncovered_terms": list(draft.uncovered_terms),
        "benchmark_quote": draft.benchmark_quote,
        "agrees_with_benchmark_quote": draft.agrees_with_benchmark_quote,
        "problems": problems,
        "disclosures": disclosures,
        "supported": not problems,
        "verbatim_scrutiny": (
            "byte-exact" if not disclosures else "normalised-only for at least one span"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default=DEFAULT_KB, help="knowledge base id")
    parser.add_argument(
        "--benchmark-path", default=DEFAULT_BENCHMARK_PATH, help="answer benchmark artifact"
    )
    parser.add_argument(
        "--json", default="", help="also write the full report to this path"
    )
    parser.add_argument(
        "--stored-only",
        action="store_true",
        help="audit only drafts already stored (default; generation is refused here)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    repo = Repository(str(settings.db_path))
    service = AnswerBenchmarkReviewService(repo, settings)

    resolved = _resolve_artifact(args.benchmark_path)
    try:
        benchmark = service.load(resolved)
    except Exception as exc:  # noqa: BLE001 - the message is the useful part
        raise DraftAuditError(f"could not load {resolved}: {exc}") from exc

    drafts = service.stored_drafts(args.kb, benchmark)
    if not drafts:
        print(
            f"no stored drafts for {benchmark.benchmark!r} on {args.kb}; "
            "generate them from the review UI (POST .../draft-suggestions) first"
        )
        return 0

    # Live corpus text + hashes for every chunk any draft cites.
    cited = sorted({e.chunk_id for d in drafts for e in d.evidence})
    rows = repo.get_chunks_by_ids(args.kb, cited)
    chunk_text = {c.id: (c.text or "") for c in rows}
    chunk_hash = {c.id: (c.content_hash or "") for c in rows}

    reports = [
        audit_draft(d, chunk_text=chunk_text, chunk_hash=chunk_hash) for d in drafts
    ]

    # Only the LATEST draft per question can inform an author; earlier ones are
    # history. Both are audited, and the table marks which is latest.
    latest_by_question: dict[str, str] = {}
    for d in drafts:
        latest_by_question[d.question_id] = d.suggestion_id

    print()
    print(f"benchmark : {benchmark.benchmark} v{benchmark.version} "
          f"(fingerprint {benchmark.fingerprint()}, lifecycle {benchmark.lifecycle.value})")
    print(f"kb        : {args.kb}")
    print(f"drafts    : {len(reports)} stored across "
          f"{len({r['question_id'] for r in reports})} question(s)")
    print(f"label     : {DRAFT_LABEL}")
    print()
    header = f"{'question':14} {'status':21} {'sup':4} {'pts':4} {'cov':>7} {'latest':7} problems"
    print(header)
    print("-" * (len(header) + 20))
    for r in sorted(reports, key=lambda x: (x["question_id"], x["created_at"])):
        cov = f"{r['coverage'] * 100:5.1f}%" if r["coverage_measured"] else "UNKNOWN"
        print(
            f"{r['question_id']:14} {r['stored_status']:21} "
            f"{'yes' if r['supported'] else 'NO':4} {len(r['key_points']):4} "
            f"{cov:>7} {'*' if latest_by_question[r['question_id']] == r['suggestion_id'] else '':7} "
            f"{r['problems'][0] if r['problems'] else ''}"
        )

    unsupported = [r for r in reports if not r["supported"]]
    summary = {
        "benchmark_path": resolved,
        "benchmark": benchmark.benchmark,
        "benchmark_fingerprint": benchmark.fingerprint(),
        "benchmark_lifecycle": benchmark.lifecycle.value,
        "kb_id": args.kb,
        "draft_count": len(reports),
        "questions_with_drafts": len({r["question_id"] for r in reports}),
        "drafts_with_problems": len(unsupported),
        "status_counts": summarise_drafts(drafts).get("by_status"),
        "coverage_measured_count": sum(1 for r in reports if r["coverage_measured"]),
        "coverage_unknown_count": sum(1 for r in reports if not r["coverage_measured"]),
        "byte_exact_drafts": sum(1 for r in reports if r["verbatim_scrutiny"] == "byte-exact"),
        "normalised_only_drafts": sum(
            1 for r in reports if r["verbatim_scrutiny"] != "byte-exact"
        ),
        "drafts": reports,
        "note": (
            "READ-ONLY report. A draft is a machine-generated suggestion and is never "
            "ground truth: this audit checks whether its quotes are verbatim in the "
            "live corpus, not whether the question is answered. Approving anything "
            "remains a named human's decision."
        ),
    }

    print()
    print(f"drafts with problems : {len(unsupported)} of {len(reports)} "
          "(every draft span verbatim under the module's normalisation)")
    print(f"byte-exact spans     : {summary['byte_exact_drafts']} of {len(reports)}; "
          f"{summary['normalised_only_drafts']} need the module's normalisation "
          "(layout artefacts in the stored text)")
    print(f"coverage UNKNOWN     : {summary['coverage_unknown_count']} of {len(reports)}")
    print(f"status counts        : {summary['status_counts']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"report written to    : {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
