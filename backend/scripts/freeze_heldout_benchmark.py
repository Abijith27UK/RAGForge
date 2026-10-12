"""Freeze the V14 held-out benchmark, sealing its review tier.

Usage (from ``backend/``):

    py -m scripts.freeze_heldout_benchmark --frozen-by "Your Name"
    py -m scripts.freeze_heldout_benchmark --frozen-by "Your Name" --dry-run

## What freezing does, and why it is a one-way door

Freezing computes the instrument's TIER from the review records that exist at that
moment and writes it into the file:

* every surviving question has a human decision -> ``validated`` / ``confirmatory``
* at least one does not                     -> ``provisional`` / ``exploratory``

The tier cannot be raised afterwards. This is deliberate rather than an omission:
if the tier could be raised once results were known, a reviewer could read the
outcome and then bless the labels that produced it, which is exactly the taint the
tier exists to prevent. A stronger instrument means a new version and a fresh
review — and, because the questions would then be known, ideally fresh questions
too.

This script refuses to freeze while the candidate set has validation ERRORS (a
question citing evidence that does not resolve, or a quote that is not in its
chunk). Those must be fixed or the affected questions rejected first: a frozen
artifact with unverifiable labels would make every result from it unverifiable.
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
from app.services.answer_eval.heldout_benchmark import HeldOutValidationError  # noqa: E402
from app.services.answer_eval.heldout_service import HeldOutService  # noqa: E402

PROTECTED_KB = "kb_f278c283c748"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default=PROTECTED_KB)
    parser.add_argument(
        "--frozen-by",
        required=True,
        help="Who is sealing this instrument. Required: an unattributed freeze "
        "cannot be audited, and the tier it sets is what any later claim rests on.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    repo = Repository(str(settings.db_path))
    service = HeldOutService(repo, settings)

    existing = service.frozen()
    candidates = service.candidates()
    report = service.validate(args.kb)

    print(f"candidate set            {service.candidate_path}")
    print(f"  candidates             {len(candidates.candidates)}")
    print(f"  surviving              {len(candidates.surviving)}")
    print(f"  fingerprint            {candidates.fingerprint()}")
    print(f"  review                 {json.dumps(candidates.review_completeness())}")
    print(f"validation               {report.verdict}")
    for problem in report.errors:
        print(f"  ERROR   {problem}")
    for warning in report.warnings:
        print(f"  WARNING {warning}")
    for missing in report.checks_not_run:
        print(f"  NOT RUN {missing['check']}: {missing['reason']}")

    if existing is not None:
        print(
            f"\nalready frozen: {existing.benchmark} {existing.fingerprint()} "
            f"({existing.lifecycle.value} / {existing.claim_strength.value})"
        )
        print("a frozen instrument is immutable; a changed one needs a new version file")
        if not args.dry_run:
            print("nothing written")
        return 0

    if report.errors:
        print(f"\nREFUSING to freeze: {len(report.errors)} validation error(s)")
        return 1

    if args.dry_run:
        tier = (
            "VALIDATED"
            if candidates.review_completeness()["complete"]
            else "PROVISIONAL"
        )
        print(f"\ndry run: would freeze as {tier}; nothing written")
        return 0

    try:
        result = service.freeze(args.kb, frozen_by=args.frozen_by)
    except HeldOutValidationError as exc:
        print(f"\nREFUSING to freeze: {exc}")
        return 1

    print(f"\nfroze {result['path']}")
    print(f"  fingerprint            {result['fingerprint']}")
    print(f"  sha256                 {result['sha256']}")
    print(f"  lifecycle              {result['lifecycle']}")
    print(f"  claim strength         {result['claim_strength']}")
    if result["lifecycle"] != "validated":
        print(
            "\n  PROVISIONAL: every result measured on this instrument is EXPLORATORY.\n"
            "  That is not a failure — it is the honest description of a set whose\n"
            "  questions were drafted by an agent and not yet checked by a human."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
