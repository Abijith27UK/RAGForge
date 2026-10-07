"""V9 Phase 4/11 regression tests: experiment record + integrity verification.

These tests exist mainly to prove the integrity checks actually CHECK: the first
draft of this code returned True from stubs, which would have reported safety
that was never verified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.comparison import (  # noqa: E402
    AcceptanceDecision,
    AcceptanceRationale,
    ComparisonOutcome,
    ExperimentProtocol,
    PairingVerdict,
)
from app.services.answer_eval.experiment import (  # noqa: E402
    ExperimentIntegrityError,
    ExperimentRecord,
    ExperimentStore,
    PROTECTED_KB_IDS,
    VectorCount,
    VectorSnapshot,
    assert_frozen_artifacts_not_touched,
    assert_scratch_isolation,
    checksum_artifacts,
    experiment_identity,
    run_postflight,
    run_preflight,
    sha256_file,
    summarise_experiment,
    verify_artifacts_unchanged,
    verify_no_vector_loss,
    ExperimentDecision,
)


class _FakeCount:
    def __init__(self, count: int) -> None:
        self.count = count


class _FakeCollections:
    def __init__(self, names: list[str]) -> None:
        self.collections = [type("C", (), {"name": n})() for n in names]


class FakeQdrantClient:
    """A READ-ONLY fake. It records every call so a destructive probe would fail."""

    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = dict(counts)
        self.calls: list[str] = []

    def get_collections(self):
        self.calls.append("get_collections")
        return _FakeCollections(list(self.counts))

    def count(self, collection_name: str, exact: bool = False):
        self.calls.append(f"count:{collection_name}")
        return _FakeCount(self.counts[collection_name])


def _protocol(label: str = "base", **overrides) -> ExperimentProtocol:
    base = dict(
        label=label,
        kb_id="kb_scratch_v9",
        benchmark_name="answer-quality-automobile-v1",
        benchmark_fingerprint="bench_fp",
        corpus_fingerprint="corpus_fp",
        evaluator_name="deterministic-evidence",
        evaluator_version="v8.3",
        generator="extractive-mock",
        model="",
        prompt_version="aej-v1",
        answer_mode="abstain_if_unsupported",
        is_mock=True,
        config_fingerprint="cfg_a",
    )
    base.update(overrides)
    return ExperimentProtocol(**base)


def _snapshot(counts: dict[str, int]) -> VectorSnapshot:
    return VectorSnapshot(
        collections=[VectorCount(collection=k, points=v) for k, v in sorted(counts.items())]
    )


class TestReadOnlyQdrantAccess:
    def test_snapshot_uses_read_only_calls(self):
        client = FakeQdrantClient({"kb_real": 100, "kb_scratch": 10})
        report = run_preflight(
            experiment_id="exp_test",
            scratch_kb_id="kb_scratch_v9",
            qdrant_client=client,
        )
        assert report.vectors.total_points == 110
        assert report.vectors.collection_count == 2
        # Only listing and counting: no delete/upsert method is ever called.
        assert all(call.startswith(("get_collections", "count:")) for call in client.calls)
        assert not any(
            hasattr(client, name) for name in ("delete", "delete_collection", "upsert")
        )

    def test_unavailable_client_is_reported_not_assumed_ok(self):
        report = run_preflight(
            experiment_id="exp_test", scratch_kb_id="kb_scratch_v9", qdrant_client=None
        )
        check = [c for c in report.checks if c.name == "vector_snapshot_available"][0]
        assert check.ok is False
        assert report.all_ok is False


class TestScratchIsolation:
    def test_protected_kb_is_refused(self):
        check = assert_scratch_isolation(scratch_kb_id="kb_f278c283c748")
        assert check.ok is False
        assert "PROTECTED" in check.detail

    def test_protected_kb_ids_include_the_incident_kb(self):
        assert "kb_f278c283c748" in PROTECTED_KB_IDS

    def test_real_kb_in_scope_is_refused_even_when_scratch_is_fine(self):
        check = assert_scratch_isolation(
            scratch_kb_id="kb_scratch_v9", observed_kb_ids={"kb_f278c283c748"}
        )
        assert check.ok is False
        assert "protected_overlap" in check.observed

    def test_empty_scratch_id_is_refused(self):
        assert assert_scratch_isolation(scratch_kb_id="").ok is False

    def test_dedicated_scratch_kb_passes(self):
        check = assert_scratch_isolation(scratch_kb_id="kb_scratch_v9")
        assert check.ok is True

    def test_preflight_refuses_to_run_against_a_real_kb(self):
        report = run_preflight(
            experiment_id="exp_test",
            scratch_kb_id="kb_f278c283c748",
            qdrant_client=FakeQdrantClient({"kb_f278c283c748": 100}),
        )
        with pytest.raises(ExperimentIntegrityError, match="scratch_kb_isolation"):
            report.raise_if_failed()


class TestFrozenArtifactGuard:
    def test_frozen_benchmark_in_the_write_set_is_refused(self):
        check = assert_frozen_artifacts_not_touched(
            ["benchmarks/answer-quality-automobile-v1.json"]
        )
        assert check.ok is False
        assert "FROZEN" in check.detail

    def test_new_experiment_file_is_allowed(self):
        assert assert_frozen_artifacts_not_touched(
            ["benchmarks/v9-candidate-experiment.json"]
        ).ok is True


class TestVectorLossDetection:
    def test_decrease_in_a_real_collection_is_a_failure(self):
        before = _snapshot({"kb_real": 100, "kb_scratch": 10})
        after = _snapshot({"kb_real": 90, "kb_scratch": 12})
        checks = verify_no_vector_loss(before, after)
        failures = [c for c in checks if not c.ok]
        assert len(failures) == 1
        assert failures[0].observed["collection"] == "kb_real"
        assert failures[0].observed["delta"] == -10

    def test_decrease_in_a_declared_scratch_collection_is_allowed(self):
        before = _snapshot({"kb_scratch": 10})
        after = _snapshot({"kb_scratch": 5})
        checks = verify_no_vector_loss(before, after, allowed_to_shrink={"kb_scratch"})
        assert all(c.ok for c in checks)

    def test_scratch_collection_must_be_named_explicitly(self):
        """No implicit allow-list: an undeclared shrink always fails."""
        before = _snapshot({"kb_scratch": 10})
        after = _snapshot({"kb_scratch": 5})
        checks = verify_no_vector_loss(before, after)
        assert not all(c.ok for c in checks)

    def test_vanished_collection_is_a_failure(self):
        before = _snapshot({"kb_real": 100, "kb_gone": 5})
        after = _snapshot({"kb_real": 100})
        failures = [c for c in verify_no_vector_loss(before, after) if not c.ok]
        assert len(failures) == 1
        assert "GONE" in failures[0].detail

    def test_growth_is_fine_and_reported(self):
        before = _snapshot({"kb_scratch": 10})
        after = _snapshot({"kb_scratch": 25})
        checks = verify_no_vector_loss(before, after)
        assert all(c.ok for c in checks)
        assert checks[0].observed["delta"] == 15

    def test_unavailable_snapshot_cannot_claim_no_deletion(self):
        checks = verify_no_vector_loss(
            VectorSnapshot(available=False, error="qdrant down"),
            VectorSnapshot(available=False, error="qdrant down"),
        )
        assert len(checks) == 1
        assert checks[0].ok is False
        assert "cannot claim no vectors were deleted" in checks[0].detail


class TestArtifactChecksums:
    def test_unchanged_artifact_verifies(self, tmp_path: Path):
        target = tmp_path / "artifact.json"
        target.write_text('{"a": 1}', encoding="utf-8")
        before = checksum_artifacts([target])
        checks = verify_artifacts_unchanged(before)
        assert all(c.ok for c in checks)

    def test_mutated_artifact_is_detected(self, tmp_path: Path):
        target = tmp_path / "artifact.json"
        target.write_text('{"a": 1}', encoding="utf-8")
        before = checksum_artifacts([target])
        target.write_text('{"a": 2}', encoding="utf-8")
        checks = verify_artifacts_unchanged(before)
        assert len(checks) == 1
        assert checks[0].ok is False
        assert "digest changed" in checks[0].detail

    def test_missing_artifact_is_recorded_as_missing(self, tmp_path: Path):
        before = checksum_artifacts([tmp_path / "nope.json"])
        assert list(before.values())[0].exists is False

    def test_sha256_matches_a_known_digest(self, tmp_path: Path):
        target = tmp_path / "x.txt"
        target.write_text("abc", encoding="utf-8")
        assert sha256_file(target) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )


class TestPostflight:
    def test_full_postflight_passes_when_nothing_changes(self, tmp_path: Path):
        target = tmp_path / "bench.json"
        target.write_text('{"benchmark": "x"}', encoding="utf-8")
        client = FakeQdrantClient({"kb_scratch": 10})
        pre = run_preflight(
            experiment_id="exp_test",
            scratch_kb_id="kb_scratch_v9",
            artifact_paths=[target],
            qdrant_client=client,
        )
        assert pre.all_ok is True
        post = run_postflight(pre, qdrant_client=client)
        assert post.all_ok is True
        assert "all" in post.summary()

    def test_postflight_fails_when_vectors_disappear(self, tmp_path: Path):
        target = tmp_path / "bench.json"
        target.write_text('{"benchmark": "x"}', encoding="utf-8")
        client = FakeQdrantClient({"kb_scratch": 10, "kb_real": 100})
        pre = run_preflight(
            experiment_id="exp_test",
            scratch_kb_id="kb_scratch_v9",
            artifact_paths=[target],
            qdrant_client=client,
        )
        client.counts["kb_real"] = 80
        post = run_postflight(pre, qdrant_client=client)
        assert post.all_ok is False
        assert "INTEGRITY FAILURE" in post.summary()


class TestDeterministicIdentity:
    def test_identical_inputs_produce_the_same_id(self):
        kwargs = dict(
            benchmark_fingerprint="b",
            corpus_fingerprint="c",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
        )
        # Question ORDER must not change the id: same experiment, same questions.
        assert experiment_identity(question_ids=["q2", "q1"], **kwargs) == experiment_identity(
            question_ids=["q1", "q2"], **kwargs
        )

    def test_changed_config_produces_a_different_id(self):
        base = dict(
            benchmark_fingerprint="b",
            corpus_fingerprint="c",
            baseline=_protocol("dense"),
            question_ids=["q1"],
        )
        first = experiment_identity(candidate=_protocol("hybrid", config_fingerprint="cfg_a"), **base)
        second = experiment_identity(candidate=_protocol("hybrid", config_fingerprint="cfg_b"), **base)
        assert first != second

    def test_changed_corpus_produces_a_different_id(self):
        base = dict(
            benchmark_fingerprint="b",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            question_ids=["q1"],
        )
        assert experiment_identity(corpus_fingerprint="c1", **base) != experiment_identity(
            corpus_fingerprint="c2", **base
        )


class TestImmutableStore:
    def _record(self) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id="exp_abc",
            label="dense vs hybrid",
            kb_id="kb_scratch_v9",
            benchmark_fingerprint="bench_fp",
            corpus_fingerprint="corpus_fp",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            question_ids=["q1"],
        )

    def test_append_writes_a_json_artifact(self, tmp_path: Path):
        store = ExperimentStore(tmp_path)
        path = store.append(self._record())
        assert path.exists()
        assert store.list_ids() == ["exp_abc"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["experiment_id"] == "exp_abc"
        assert payload["candidate"]["label"] == "hybrid"

    def test_appending_identical_content_is_a_no_op(self, tmp_path: Path):
        """Replaying the same experiment must not look like a conflict."""
        store = ExperimentStore(tmp_path)
        first = store.append(self._record())
        second = store.append(self._record())
        assert first == second
        assert store.list_ids() == ["exp_abc"]

    def test_appending_different_content_under_the_same_id_raises(self, tmp_path: Path):
        store = ExperimentStore(tmp_path)
        store.append(self._record())
        conflicting = self._record()
        conflicting.label = "a different experiment"
        with pytest.raises(ExperimentIntegrityError, match="immutable"):
            store.append(conflicting)

    def test_get_returns_none_for_unknown_id(self, tmp_path: Path):
        assert ExperimentStore(tmp_path).get("exp_missing") is None


class TestRecordConclusionIsWillingToSayNoImprovement:
    def _record(self, **overrides) -> ExperimentRecord:
        base = dict(
            experiment_id="exp_abc",
            kb_id="kb_scratch_v9",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
        )
        base.update(overrides)
        return ExperimentRecord(**base)

    def test_missing_integrity_phases_blocks_any_conclusion(self):
        record = self._record()
        assert record.integrity_ok() is False
        assert record.trustworthy() is False

    def test_no_comparison_yields_no_conclusion(self):
        assert "no comparison" in self._record().conclusion()

    def test_comparison_without_integrity_is_refused(self):
        """A comparison whose integrity was never verified must not be reported."""
        from app.services.answer_eval.comparison import compare_suite_explicit

        comparison = compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            baseline_metrics={"recall_at_k": {"q1": 0.2}},
            candidate_metrics={"recall_at_k": {"q1": 0.5}},
        )
        record = self._record(comparison=comparison)
        assert record.integrity_ok() is False
        assert record.conclusion().startswith("INTEGRITY NOT VERIFIED")
        assert record.trustworthy() is False

    def test_no_metric_movement_is_reported_as_no_improvement(self):
        from app.services.answer_eval.comparison import compare_suite_explicit

        comparison = compare_suite_explicit(
            baseline_label="dense",
            candidate_label="hybrid",
            baseline=_protocol("dense"),
            candidate=_protocol("hybrid"),
            baseline_metrics={"recall_at_k": {"q1": 0.2}},
            candidate_metrics={"recall_at_k": {"q1": 0.2}},
        )
        record = self._record(comparison=comparison)
        assert comparison.pairing is PairingVerdict.COMPARABLE
        assert record.conclusion().startswith("INTEGRITY NOT VERIFIED")

    def test_integrity_ok_requires_both_phases(self, tmp_path: Path):
        client = FakeQdrantClient({"kb_scratch": 10})
        pre = run_preflight(
            experiment_id="exp_abc", scratch_kb_id="kb_scratch_v9", qdrant_client=client
        )
        record = self._record(preflight=pre)
        assert record.integrity_ok() is False
        record.postflight = run_postflight(pre, qdrant_client=client)
        assert record.integrity_ok() is True

    def test_summarise_experiment_refuses_to_invent_an_outcome(self):
        assert summarise_experiment(self._record()) is ExperimentDecision.INVALID
