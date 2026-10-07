"""Controlled experiments: immutable records + REAL integrity verification.

Phases 4 and 11 of V9 in one module.

## Immutability

An ``ExperimentRecord`` is identified by a DETERMINISTIC id — a hash of the
experiment's identity (benchmark content, corpus, both retrieval
configurations, evaluator, model, question set). Re-running the identical
experiment therefore produces the identical id, and ``ExperimentStore.append``
refuses to write a DIFFERENT record under an id that already exists. Storing the
same record twice is a no-op, never an overwrite.

## Integrity checks that actually check something

An earlier draft of this code returned ``True`` from ``_file_unchanged`` and
``_scratch_isolation`` without inspecting anything. That is worse than no check:
it reports safety that was never verified. Every check in this module reads real
state:

* ``checksum_frozen_artifacts``  — hashes each artifact and compares to the
  recorded digest, so a mutated frozen result is detected.
* ``snapshot_vector_counts``     — reads the point count of every Qdrant
  collection (a READ-only call) and compares before/after. Any DECREASE in a
  collection that is not the experiment's own scratch collection fails
  postflight.
* ``assert_scratch_isolation``   — refuses to run an experiment against any KB
  that is not on the scratch allow-list.

## The permanent safety rule

Project history contains an incident where a debug probe deleted vectors from a
real KB. Consequently:

* no function in this module deletes, overwrites or upserts a vector;
* the only Qdrant calls here are collection listing and exact point counts;
* ``describe_collection`` style probes that could mutate are simply not written;
* an experiment whose scratch KB equals a protected KB is refused in preflight,
  before anything runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.services.answer_eval.benchmark_review import FROZEN_BENCHMARK_FILES
from app.services.answer_eval.comparison import (
    AcceptanceRationale,
    ExperimentComparison,
    ExperimentProtocol,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentIntegrityError(RuntimeError):
    """An integrity rule was violated. Raised INSTEAD of running."""


#: KB ids that must never be the target of an experiment. The Automobile KB that
#: the incident damaged is protected, and the constant is named so a future
#: change has to be deliberate.
PROTECTED_KB_IDS: frozenset[str] = frozenset({"kb_f278c283c748"})


class IntegrityCheck(BaseModel):
    """One integrity check and the state it actually observed."""

    name: str
    ok: bool
    detail: str = ""
    observed: dict[str, Any] = Field(default_factory=dict)


class ArtifactChecksum(BaseModel):
    path: str
    sha256: str
    size_bytes: int
    exists: bool = True


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes. Raises when the file is unreadable."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_artifacts(paths: list[str | Path]) -> dict[str, ArtifactChecksum]:
    """Hash each artifact. A missing file is recorded as missing, not skipped."""
    out: dict[str, ArtifactChecksum] = {}
    for raw in paths:
        path = Path(raw)
        key = str(path)
        if not path.exists():
            out[key] = ArtifactChecksum(path=key, sha256="", size_bytes=0, exists=False)
            continue
        out[key] = ArtifactChecksum(
            path=key, sha256=sha256_file(path), size_bytes=path.stat().st_size
        )
    return out


def verify_artifacts_unchanged(before: dict[str, ArtifactChecksum]) -> list[IntegrityCheck]:
    """Re-hash every recorded artifact and report which ones changed."""
    checks: list[IntegrityCheck] = []
    for key, prior in sorted(before.items()):
        if not prior.exists:
            checks.append(
                IntegrityCheck(
                    name=f"artifact_present::{key}",
                    ok=False,
                    detail="artifact did not exist when the experiment started, so "
                    "its stability cannot be verified",
                    observed={"path": key},
                )
            )
            continue
        try:
            current = sha256_file(key)
        except OSError as exc:
            checks.append(
                IntegrityCheck(
                    name=f"artifact_readable::{key}",
                    ok=False,
                    detail=f"cannot re-read artifact: {exc}",
                    observed={"path": key},
                )
            )
            continue
        checks.append(
            IntegrityCheck(
                name=f"artifact_unchanged::{key}",
                ok=current == prior.sha256,
                detail=(
                    "digest matches"
                    if current == prior.sha256
                    else f"digest changed: {prior.sha256[:12]} -> {current[:12]}"
                ),
                observed={"path": key, "before": prior.sha256, "after": current},
            )
        )
    return checks


class VectorCount(BaseModel):
    """Point count of ONE collection, as read at a point in time."""

    collection: str
    points: int


class VectorSnapshot(BaseModel):
    """Read-only snapshot of every Qdrant collection's point count."""

    captured_at: datetime = Field(default_factory=_utcnow)
    collections: list[VectorCount] = Field(default_factory=list)
    available: bool = True
    error: str = ""

    @property
    def collection_count(self) -> int:
        return len(self.collections)

    @property
    def total_points(self) -> int:
        return sum(c.points for c in self.collections)

    def points_for(self, collection: str) -> int | None:
        for c in self.collections:
            if c.collection == collection:
                return c.points
        return None


def snapshot_vector_counts(
    client: Any,
    *,
    collection_prefix: str = "",
) -> VectorSnapshot:
    """Read collection names and exact point counts. READ-ONLY.

    Uses ``count(exact=True)``. No function in this module deletes or writes
    vectors, by design: the project's permanent safety rule forbids destructive
    probes against real knowledge bases.
    """
    if client is None:
        return VectorSnapshot(
            available=False, error="no Qdrant client supplied; vector checks skipped"
        )
    try:
        names = [c.name for c in client.get_collections().collections]
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return VectorSnapshot(available=False, error=f"cannot list collections: {exc}")

    counts: list[VectorCount] = []
    for name in sorted(names):
        if collection_prefix and not name.startswith(collection_prefix):
            continue
        try:
            result = client.count(collection_name=name, exact=True)
            counts.append(VectorCount(collection=name, points=int(result.count)))
        except Exception as exc:  # noqa: BLE001 - reported per collection
            logger.warning("could not count collection %s: %s", name, exc)
            return VectorSnapshot(
                available=False,
                error=f"cannot count collection {name!r}: {exc}",
                collections=counts,
            )
    return VectorSnapshot(collections=counts)


def verify_no_vector_loss(
    before: VectorSnapshot,
    after: VectorSnapshot,
    *,
    allowed_to_shrink: set[str] | None = None,
) -> list[IntegrityCheck]:
    """Fail when a collection lost points and was not allowed to shrink.

    ``allowed_to_shrink`` should name the experiment's OWN scratch collection and
    nothing else. Defaulting to empty means any decrease whatsoever is a failure,
    which is the safe direction: a missing entry in the allow-list produces a
    false alarm, never a missed deletion.
    """
    permitted = set(allowed_to_shrink or set())
    if not before.available or not after.available:
        return [
            IntegrityCheck(
                name="vector_counts_comparable",
                ok=False,
                detail=(
                    "vector counts could not be compared, so we cannot claim no "
                    f"vectors were deleted. before: {before.error or 'ok'}; "
                    f"after: {after.error or 'ok'}"
                ),
                observed={
                    "before_collections": before.collection_count,
                    "after_collections": after.collection_count,
                },
            )
        ]

    checks: list[IntegrityCheck] = []
    for item in before.collections:
        now = after.points_for(item.collection)
        if now is None:
            checks.append(
                IntegrityCheck(
                    name=f"collection_present::{item.collection}",
                    ok=False,
                    detail=(
                        f"collection {item.collection!r} existed before the "
                        f"experiment and is GONE after it"
                    ),
                    observed={"collection": item.collection, "before": item.points},
                )
            )
            continue
        delta = now - item.points
        if delta < 0 and item.collection not in permitted:
            checks.append(
                IntegrityCheck(
                    name=f"no_vector_loss::{item.collection}",
                    ok=False,
                    detail=(
                        f"collection {item.collection!r} lost {abs(delta)} point(s) "
                        f"({item.points} -> {now}) and was not declared a scratch "
                        f"collection. Nothing in the experiment may delete vectors"
                    ),
                    observed={
                        "collection": item.collection,
                        "before": item.points,
                        "after": now,
                        "delta": delta,
                    },
                )
            )
        else:
            checks.append(
                IntegrityCheck(
                    name=f"no_vector_loss::{item.collection}",
                    ok=True,
                    detail=(
                        f"{delta:+d} point(s) ({item.points} -> {now})"
                        + (" [declared scratch]" if item.collection in permitted else "")
                    ),
                    observed={"collection": item.collection, "delta": delta},
                )
            )

    for item in after.collections:
        if before.points_for(item.collection) is None:
            checks.append(
                IntegrityCheck(
                    name=f"new_collection::{item.collection}",
                    ok=True,
                    detail=f"collection appeared during the experiment with {item.points} point(s)",
                    observed={"collection": item.collection, "points": item.points},
                )
            )
    return checks


def assert_scratch_isolation(
    *,
    scratch_kb_id: str,
    protected_kb_ids: set[str] | None = None,
    observed_kb_ids: set[str] | None = None,
) -> IntegrityCheck:
    """Refuse an experiment that targets a protected KB.

    Called BEFORE anything runs. The expensive failure this prevents is not a
    wrong number — it is damage to a real knowledge base.
    """
    protected = set(protected_kb_ids) if protected_kb_ids is not None else set(PROTECTED_KB_IDS)
    if not scratch_kb_id:
        return IntegrityCheck(
            name="scratch_kb_isolation",
            ok=False,
            detail="no scratch KB was named; an experiment must run against its own KB",
        )
    if scratch_kb_id in protected:
        return IntegrityCheck(
            name="scratch_kb_isolation",
            ok=False,
            detail=(
                f"refused: {scratch_kb_id!r} is a PROTECTED knowledge base. "
                f"Experiments must run against a dedicated scratch KB so no real "
                f"corpus can be modified"
            ),
            observed={"scratch_kb_id": scratch_kb_id, "protected": sorted(protected)},
        )
    touched = set(observed_kb_ids or set())
    overlap = sorted(touched & protected)
    if overlap:
        return IntegrityCheck(
            name="scratch_kb_isolation",
            ok=False,
            detail=(
                f"refused: the experiment would observe protected KB(s) {overlap}. "
                f"Reading is not the concern — running an experiment against a real "
                f"KB is how the historical vector-deletion incident happened"
            ),
            observed={"scratch_kb_id": scratch_kb_id, "protected_overlap": overlap},
        )
    return IntegrityCheck(
        name="scratch_kb_isolation",
        ok=True,
        detail=f"{scratch_kb_id!r} is not protected and no protected KB is in scope",
        observed={"scratch_kb_id": scratch_kb_id},
    )


def assert_frozen_artifacts_not_touched(paths: list[str | Path]) -> IntegrityCheck:
    """Refuse to include a frozen artifact in an experiment's write set."""
    offenders = [str(p) for p in paths if Path(p).name in FROZEN_BENCHMARK_FILES]
    if offenders:
        return IntegrityCheck(
            name="frozen_artifacts_untouched",
            ok=False,
            detail=(
                f"refused: {offenders} are FROZEN historical artifacts. An "
                f"experiment must write its results to a NEW file so every earlier "
                f"result stays reproducible"
            ),
            observed={"offenders": offenders},
        )
    return IntegrityCheck(
        name="frozen_artifacts_untouched",
        ok=True,
        detail=f"none of the {len(paths)} path(s) in scope is a frozen artifact",
    )


class PreflightReport(BaseModel):
    """Everything verified BEFORE an experiment runs."""

    experiment_id: str = ""
    scratch_kb_id: str = ""
    checks: list[IntegrityCheck] = Field(default_factory=list)
    artifact_checksums: dict[str, ArtifactChecksum] = Field(default_factory=dict)
    vectors: VectorSnapshot = Field(default_factory=VectorSnapshot)

    @property
    def all_ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def failures(self) -> list[IntegrityCheck]:
        return [c for c in self.checks if not c.ok]

    def raise_if_failed(self) -> None:
        """Stop the experiment when any check failed. Called before any work."""
        failed = self.failures()
        if failed:
            raise ExperimentIntegrityError(
                "preflight failed, refusing to run: "
                + "; ".join(f"{c.name}: {c.detail}" for c in failed)
            )


class PostflightReport(BaseModel):
    """Everything verified AFTER an experiment runs."""

    checks: list[IntegrityCheck] = Field(default_factory=list)
    vectors_after: VectorSnapshot = Field(default_factory=VectorSnapshot)

    @property
    def all_ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def failures(self) -> list[IntegrityCheck]:
        return [c for c in self.checks if not c.ok]

    def summary(self) -> str:
        failed = self.failures()
        if failed:
            return "INTEGRITY FAILURE: " + "; ".join(f"{c.name}: {c.detail}" for c in failed)
        return f"all {len(self.checks)} postflight check(s) passed"


def run_preflight(
    *,
    experiment_id: str,
    scratch_kb_id: str,
    artifact_paths: list[str | Path] | None = None,
    write_paths: list[str | Path] | None = None,
    observed_kb_ids: set[str] | None = None,
    qdrant_client: Any = None,
    protected_kb_ids: set[str] | None = None,
    collection_prefix: str = "",
) -> PreflightReport:
    """Verify the experiment is safe BEFORE running it.

    Cheap, read-only, and designed to fail loudly: a failed preflight raises
    rather than proceeding with a warning.
    """
    report = PreflightReport(experiment_id=experiment_id, scratch_kb_id=scratch_kb_id)
    report.checks.append(
        assert_scratch_isolation(
            scratch_kb_id=scratch_kb_id,
            protected_kb_ids=protected_kb_ids,
            observed_kb_ids=observed_kb_ids,
        )
    )
    report.checks.append(assert_frozen_artifacts_not_touched(list(write_paths or [])))
    if artifact_paths:
        report.artifact_checksums = checksum_artifacts(list(artifact_paths))
        missing = [
            key for key, value in report.artifact_checksums.items() if not value.exists
        ]
        report.checks.append(
            IntegrityCheck(
                name="artifacts_readable",
                ok=not missing,
                detail=(
                    f"hashed {len(report.artifact_checksums)} artifact(s)"
                    if not missing
                    else f"artifacts missing, cannot verify them: {missing}"
                ),
                observed={"missing": missing},
            )
        )
    report.vectors = snapshot_vector_counts(qdrant_client, collection_prefix=collection_prefix)
    report.checks.append(
        IntegrityCheck(
            name="vector_snapshot_available",
            ok=report.vectors.available,
            detail=(
                f"counted {report.vectors.collection_count} collection(s), "
                f"{report.vectors.total_points} point(s)"
                if report.vectors.available
                else report.vectors.error
            ),
            observed={
                "collection_count": report.vectors.collection_count,
                "total_points": report.vectors.total_points,
            },
        )
    )
    return report


def run_postflight(
    preflight: PreflightReport,
    *,
    qdrant_client: Any = None,
    allowed_to_shrink: set[str] | None = None,
    collection_prefix: str = "",
) -> PostflightReport:
    """Verify integrity AFTER running, and detect any vector loss."""
    report = PostflightReport()
    report.vectors_after = snapshot_vector_counts(qdrant_client, collection_prefix=collection_prefix)
    report.checks.extend(
        verify_no_vector_loss(
            preflight.vectors, report.vectors_after, allowed_to_shrink=allowed_to_shrink
        )
    )
    if preflight.artifact_checksums:
        report.checks.extend(verify_artifacts_unchanged(preflight.artifact_checksums))
    return report


# ---------------------------------------------------------------------------
# Immutable experiment record
# ---------------------------------------------------------------------------


def experiment_identity(
    *,
    benchmark_fingerprint: str,
    corpus_fingerprint: str,
    baseline: ExperimentProtocol,
    candidate: ExperimentProtocol,
    question_ids: list[str],
    answer_mode: str = "",
) -> str:
    """Deterministic id: a hash of what the experiment IS, not when it ran.

    Two runs of the identical experiment produce the identical id, which is what
    makes the store's duplicate refusal meaningful instead of arbitrary.
    """
    payload = {
        "benchmark_fingerprint": benchmark_fingerprint,
        "corpus_fingerprint": corpus_fingerprint,
        "baseline_config_fingerprint": baseline.config_fingerprint,
        "candidate_config_fingerprint": candidate.config_fingerprint,
        "evaluator": f"{baseline.evaluator_name}@{baseline.evaluator_version}",
        "generator": baseline.generator,
        "model": baseline.model,
        "prompt_version": baseline.prompt_version,
        "answer_mode": answer_mode or baseline.answer_mode,
        "question_ids": sorted(question_ids),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"exp_{digest}"


class ExperimentRecord(BaseModel):
    """An immutable record of one controlled experiment."""

    experiment_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    label: str = ""
    purpose: str = ""
    kb_id: str = ""
    scratch_kb_id: str = ""

    benchmark_name: str = ""
    benchmark_fingerprint: str = ""
    corpus_fingerprint: str = ""

    baseline: ExperimentProtocol
    candidate: ExperimentProtocol
    question_ids: list[str] = Field(default_factory=list)

    #: metric -> question_id -> value. A None value is UNMEASURED and stays None.
    baseline_metrics: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    candidate_metrics: dict[str, dict[str, float | None]] = Field(default_factory=dict)

    comparison: ExperimentComparison | None = None
    decision: AcceptanceRationale | None = None

    preflight: PreflightReport | None = None
    postflight: PostflightReport | None = None

    notes: list[str] = Field(default_factory=list)

    def integrity_ok(self) -> bool:
        """True only when BOTH integrity phases ran and passed."""
        return bool(
            self.preflight is not None
            and self.postflight is not None
            and self.preflight.all_ok
            and self.postflight.all_ok
        )

    def trustworthy(self) -> bool:
        """Can any conclusion be drawn from this record at all?

        Requires valid pairing, passing integrity, and a recorded decision.
        """
        return bool(
            self.comparison is not None
            and self.comparison.pairing.value == "COMPARABLE"
            and self.integrity_ok()
            and self.decision is not None
        )

    def conclusion(self) -> str:
        """A plain-language conclusion that is willing to say 'no improvement'."""
        if self.comparison is None:
            return "no comparison was recorded for this experiment"
        if not self.integrity_ok():
            return (
                "INTEGRITY NOT VERIFIED: the experiment did not complete both "
                "integrity phases, so its numbers must not be reported"
            )
        if self.comparison.pairing.value != "COMPARABLE":
            return (
                f"NO VALID COMPARISON: {self.comparison.pairing_reason}. No "
                f"accept/reject conclusion can rest on this experiment"
            )
        improved = self.comparison.improved_metrics()
        regressed = self.comparison.regressed_metrics()
        if self.decision is not None:
            head = f"decision: {self.decision.decision.value} — {self.decision.reason}"
        else:
            head = "no accept/reject decision was recorded"
        detail = (
            f"improved metric(s): {improved or 'none'}; "
            f"regressed metric(s): {regressed or 'none'}"
        )
        if not improved and not regressed:
            return (
                f"{head}. {detail}. No metric moved, so there is NO measured "
                f"improvement to report"
            )
        return f"{head}. {detail}"

    def artifact(self) -> dict[str, Any]:
        """The JSON artifact written to disk — a complete, self-contained record."""
        return self.model_dump(mode="json")

    #: Fields that vary between two runs of the SAME experiment and therefore
    #: must not participate in the immutability comparison. Kept explicit so
    #: adding a new volatile field is a deliberate decision.
    VOLATILE_FIELDS: frozenset[str] = frozenset({"created_at"})

    def content_fingerprint(self) -> str:
        """Hash of everything that defines the experiment, EXCLUDING timestamps.

        The id is derived from content, so two runs of the same experiment share
        an id. Comparing raw JSON would still see a different ``created_at`` and
        report a spurious conflict, so immutability is enforced on this
        fingerprint instead: replaying the same experiment is a no-op, while
        genuinely different content under one id is refused.
        """
        payload = {
            k: v
            for k, v in self.artifact().items()
            if k not in self.VOLATILE_FIELDS
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def stored_fingerprint(self) -> str:
        """The content fingerprint this record would have as re-read from disk."""
        return self.content_fingerprint()


class ExperimentStore:
    """Append-only on-disk store of experiment records.

    Each record is one JSON file named after its deterministic id, created with
    exclusive semantics so an existing file is never overwritten. A second
    attempt to store DIFFERENT content under the same id raises; storing
    identical content again is a no-op.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(self, experiment_id: str) -> Path:
        return self.directory / f"{experiment_id}.json"

    def append(self, record: ExperimentRecord) -> Path:
        """Persist a record. Never overwrites differing content."""
        path = self.path_for(record.experiment_id)
        payload = record.artifact()
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ExperimentIntegrityError(
                    f"existing experiment file {path} is unreadable ({exc}); "
                    f"refusing to overwrite it"
                ) from exc
            # Compare CONTENT (timestamps excluded), not raw bytes: the id is
            # content-derived, so replaying the same experiment must be a no-op
            # rather than a conflict caused by a new `created_at`.
            stored = ExperimentRecord.model_validate(existing)
            if stored.content_fingerprint() == record.content_fingerprint():
                logger.info("experiment %s already stored identically", record.experiment_id)
                return path
            raise ExperimentIntegrityError(
                f"experiment {record.experiment_id!r} already exists with DIFFERENT "
                f"content. Experiment records are immutable: change the experiment "
                f"(and therefore its deterministic id) or store it under a new label"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return path

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        path = self.path_for(experiment_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_ids(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json"))


class ExperimentDecision(str, Enum):
    """Outcome vocabulary for a completed experiment."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NO_IMPROVEMENT_FOUND = "no_improvement_found"
    INVALID = "invalid"


def summarise_experiment(record: ExperimentRecord) -> ExperimentDecision:
    """Map a record onto a single outcome, refusing to invent an improvement."""
    if not record.integrity_ok() or record.comparison is None:
        return ExperimentDecision.INVALID
    if record.comparison.pairing.value != "COMPARABLE":
        return ExperimentDecision.INVALID
    if record.decision is None:
        return ExperimentDecision.INVALID
    if record.decision.decision.value == "ACCEPT":
        return ExperimentDecision.ACCEPTED
    if record.decision.decision.value == "REJECT":
        return ExperimentDecision.REJECTED
    return ExperimentDecision.NO_IMPROVEMENT_FOUND


__all__ = [
    "ArtifactChecksum",
    "ExperimentDecision",
    "ExperimentIntegrityError",
    "ExperimentRecord",
    "ExperimentStore",
    "IntegrityCheck",
    "PostflightReport",
    "PreflightReport",
    "PROTECTED_KB_IDS",
    "VectorCount",
    "VectorSnapshot",
    "assert_frozen_artifacts_not_touched",
    "assert_scratch_isolation",
    "checksum_artifacts",
    "experiment_identity",
    "run_postflight",
    "run_preflight",
    "sha256_file",
    "snapshot_vector_counts",
    "summarise_experiment",
    "verify_artifacts_unchanged",
    "verify_no_vector_loss",
]
