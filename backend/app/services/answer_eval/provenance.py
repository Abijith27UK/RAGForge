"""Annotation provenance blocks (V9 Phase 1 support).

Deterministic helpers that record WHERE an annotation came from and HOW it was
produced, so an agent-authored label is never indistinguishable from a
human-reviewed one after the fact.

## What changed from the first draft of this module

The first draft declared its own ``ReviewVerdict``, ``HumanReviewStatus``,
``EvidenceClassification``, ``EvidenceVerdict`` and ``EvidenceAnchor`` enums.
That was a mistake and it is deliberately removed: the project already has
canonical vocabularies for exactly those concepts —

* ``benchmark.HumanReviewStatus``    — whether human review happened at all
* ``benchmark.AnswerBenchmarkLifecycle`` — the official-use gate
* ``review.ReviewVerdict``           — the answer-review verdict set
* ``failure.FailureCode``            — the closed failure-label set

Re-declaring them here created a THIRD failure vocabulary and a SECOND review
verdict vocabulary with different members, which is precisely the silent drift
AGENTS.md forbids: the same concept meaning two things in two modules. This
module now only owns the provenance block itself and imports everything else
from the module that already defines it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProvenanceError(ValueError):
    """A provenance block is invalid as written."""


class ProvenanceKind(str, Enum):
    """What an annotation is about. Closed set, one entry per annotation type."""

    REFERENCE_ANSWER = "reference_answer"
    KEY_POINTS = "key_points"
    ACCEPTABLE_ELEMENTS = "acceptable_elements"
    ANSWERABILITY = "answerability"
    CITATION_REQUIREMENTS = "citation_requirements"
    GROUNDING_STATE = "grounding_state"
    AMBIGUITY = "ambiguity"
    ANSWER_SOURCE = "answer_source"
    REVIEW_HISTORY = "review_history"


class ProvenanceLevel(str, Enum):
    """Whether this block describes the original annotation or a revision."""

    ORIGINAL = "original"
    REVISION = "revision"


class ProvenanceTag(str, Enum):
    """Machine-readable character of the annotation's origin.

    ``AGENT`` and ``HUMAN_REVIEW`` are the pair that matters most: the shipped
    Automobile benchmark is ``AGENT``, and nothing may relabel it as
    ``HUMAN_REVIEW`` without a recorded review.
    """

    METHOD = "method"
    SOURCE = "source"
    JUSTIFICATION = "justification"
    HUMAN_REVIEW = "human_review"
    AGENT = "agent"
    AUTO = "auto"
    PROXY = "proxy"


class Provenance(BaseModel):
    """One provenance block attached to an annotation."""

    kind: ProvenanceKind
    level: ProvenanceLevel = ProvenanceLevel.ORIGINAL
    method: str = Field(
        default="",
        description="How the annotation was produced, e.g. 'read the indexed chunk "
        "text and selected the passage that states the procedure'.",
    )
    source: str = Field(default="", description="Where it came from: URL, chunk id, document id")
    justification: str = ""
    timestamp: str = ""
    tags: list[ProvenanceTag] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)

    def is_human_reviewed(self) -> bool:
        """True only when this block claims human review.

        Used to keep an agent annotation from being counted as human-reviewed.
        """
        return ProvenanceTag.HUMAN_REVIEW in self.tags


def create_provenance(
    *,
    kind: ProvenanceKind,
    method: str,
    source: str = "",
    justification: str = "",
    tags: list[ProvenanceTag] | None = None,
    data: dict[str, Any] | None = None,
    level: ProvenanceLevel = ProvenanceLevel.ORIGINAL,
    timestamp: str | None = None,
) -> Provenance:
    """Build a provenance block with an explicit method and timestamp.

    ``method`` is required in practice: ``validate_provenance`` reports an empty
    method as a problem, because an annotation with no stated method cannot be
    audited.
    """
    return Provenance(
        kind=kind,
        method=method,
        source=source,
        justification=justification,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        tags=list(tags or []),
        data=dict(data or {}),
        level=level,
    )


def assert_provenance_complete(*, blocks: list[Provenance]) -> None:
    """Reject an empty provenance list: every annotation carries at least one."""
    if not blocks:
        raise ProvenanceError(
            "provenance must not be empty: an annotation with no provenance is "
            "indistinguishable from a guess"
        )


def validate_provenance(blocks: list[Provenance]) -> list[str]:
    """Return human-readable problems; empty means valid.

    Checks what actually matters and nothing more:

    * a block with no ``method`` cannot be audited;
    * a block with no ``timestamp`` has no recorded origin in time;
    * the SAME ``kind`` appearing twice in one annotation is ambiguous — which
      block is the annotation's provenance? Duplicate ``source`` values are NOT
      a problem: one document legitimately backs several annotation kinds.
    """
    problems: list[str] = []
    seen_kinds: dict[ProvenanceKind, int] = {}
    for block in blocks:
        if not block.method.strip():
            problems.append(f"{block.kind.value}: provenance method is empty")
        if not block.timestamp:
            problems.append(f"{block.kind.value}: missing timestamp")
        seen_kinds[block.kind] = seen_kinds.get(block.kind, 0) + 1
    for kind, count in sorted(seen_kinds.items(), key=lambda kv: kv[0].value):
        if count > 1:
            problems.append(
                f"{kind.value}: provenance declared {count} times; which block is "
                f"authoritative is ambiguous"
            )
    return problems


def hash_provenance(blocks: list[Provenance]) -> str:
    """Stable short hash of a provenance list, for audit records."""
    payload = json.dumps(
        [b.model_dump(mode="json") for b in blocks],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def agent_annotation_provenance(
    *,
    kind: ProvenanceKind,
    method: str,
    source: str = "",
    justification: str = "",
    data: dict[str, Any] | None = None,
) -> Provenance:
    """Provenance for an AGENT-produced annotation.

    Tagged ``AGENT`` and explicitly NOT ``HUMAN_REVIEW``, so an agent's real work
    is recorded as exactly that. This is the helper the shipped benchmark's
    labels should carry.
    """
    return create_provenance(
        kind=kind,
        method=method,
        source=source,
        justification=justification,
        tags=[ProvenanceTag.AGENT],
        data=data,
    )


def human_review_provenance(
    *,
    kind: ProvenanceKind,
    reviewer: str,
    method: str,
    source: str = "",
    justification: str = "",
    level: ProvenanceLevel = ProvenanceLevel.REVISION,
    data: dict[str, Any] | None = None,
) -> Provenance:
    """Provenance for a HUMAN review. Requires a named reviewer."""
    if not reviewer.strip():
        raise ProvenanceError(
            "a human-review provenance block must name the reviewer; an "
            "unattributed review cannot be distinguished from an agent claim"
        )
    return create_provenance(
        kind=kind,
        method=method,
        source=source,
        justification=justification,
        level=level,
        tags=[ProvenanceTag.HUMAN_REVIEW],
        data={**dict(data or {}), "reviewer": reviewer},
    )


__all__ = [
    "Provenance",
    "ProvenanceError",
    "ProvenanceKind",
    "ProvenanceLevel",
    "ProvenanceTag",
    "agent_annotation_provenance",
    "assert_provenance_complete",
    "create_provenance",
    "hash_provenance",
    "human_review_provenance",
    "validate_provenance",
]
