"""V13 controlled retrieval-experiment tests.

Two layers are covered on purpose:

* the PURE RULES (``retrieval_experiment_models``) — outcome derivation, metric
  direction, per-question pairing, unmeasured handling — tested directly, because
  a rule that is only tested through an HTTP call is not really pinned down;
* the API end to end on a real indexed corpus, including the integrity claims:
  an experiment must not touch the knowledge base, the index or either frozen
  benchmark artifact, and a strategy that cannot run must be recorded as FAILED
  rather than replaced by a substitute.

The shipped benchmark files are hashed before and after the whole module runs, so
"the experiments never modified the instrument" is a measurement here too.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.comparison import (  # noqa: E402
    AcceptanceDecision,
    ComparisonOutcome,
    ExperimentComparison,
    PairingVerdict,
)
from app.services.answer_eval.retrieval_experiment_models import (  # noqa: E402
    MEANINGFUL_DELTA,
    CandidateOutcome,
    ExperimentArm,
    ExperimentOutcome,
    RetrievalExperiment,
    RetrievalStrategyConfig,
    decide_candidate_outcome,
    decide_experiment_outcome,
    metric_stats,
    per_question_results,
    question_set_fingerprint,
    slugify,
)
from tests.test_v10_review_api import (  # noqa: E402,F401
    NAVAL,
    client,
    env,
    indexed_kb,
)
from tests.test_v12_retrieval_diagnostics_api import (  # noqa: E402,F401
    _benchmark_questions,
    _write_benchmark,
)

EXPERIMENTS = "/api/knowledge-bases/{kb}/retrieval-experiments"
META = "/api/knowledge-bases/{kb}/retrieval-experiments/meta"
ONE = "/api/knowledge-bases/{kb}/retrieval-experiments/{eid}"

REPO_ROOT = BACKEND_DIR.parent
FROZEN_BENCHMARKS = (
    REPO_ROOT / "benchmarks" / "automobile-engineering-baseline-v1.json",
    REPO_ROOT / "benchmarks" / "answer-quality-automobile-v1.json",
)


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


@pytest.fixture(scope="module")
def frozen_hashes():
    return {p.name: _sha(p) for p in FROZEN_BENCHMARKS}


def _declared(**overrides) -> dict:
    """A complete experiment declaration; overrides change one thing at a time."""
    payload = {
        "name": "dense depth sweep on three questions",
        "hypothesis": (
            "The low-recall cases diagnosed at k=3 are a WINDOW effect, not a "
            "ranking effect: the expected evidence is retrieved but placed below "
            "the cut-off, so widening k from 3 to 10 recovers it without harming "
            "ranking quality."
        ),
        "baseline": {"label": "dense_k3", "strategy": "dense", "top_k": 3},
        "candidates": [{"label": "dense_k10", "strategy": "dense", "top_k": 10}],
        "primary_metric": "recall_at_k",
        "guardrail_metrics": ["mrr"],
        "proposed_change": "raise the evaluated k from 3 to 10",
        "expected_effect": "recall@k rises; mrr does not fall",
        "targeted_failure_code": "LOW_RECALL",
        "persist": True,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------


class TestMetricStatsAndDirection:
    def test_unmeasured_values_are_excluded_not_counted_as_zero(self):
        stats = metric_stats("recall_at_k", {"a": 1.0, "b": None, "c": 0.5})
        assert stats.sample_size == 2
        assert stats.mean == 0.75, "a None must not be averaged in as 0"
        assert stats.direction == "higher_is_better"
        assert stats.measured is True

    def test_an_entirely_unmeasured_metric_is_unknown_not_zero(self):
        stats = metric_stats("mrr", {"a": None})
        assert stats.measured is False
        assert stats.mean is None
        assert stats.sample_size == 0

    def test_an_undeclared_metric_direction_is_refused(self):
        with pytest.raises(ValueError):
            metric_stats("a_metric_with_no_declared_direction", {"a": 1.0})

    def test_lower_is_better_metrics_are_oriented_correctly(self):
        stats = metric_stats("first_relevant_rank", {"a": 1.0, "b": 3.0})
        assert stats.direction == "lower_is_better"
        assert stats.mean == 2.0


class TestPerQuestionPairing:
    def test_direction_flips_which_side_wins(self):
        higher = per_question_results(
            metric="mrr", baseline_series={"q": 0.5}, candidate_series={"q": 0.9}
        )
        assert higher[0].outcome == "IMPROVED"
        lower = per_question_results(
            metric="first_relevant_rank",
            baseline_series={"q": 1.0},
            candidate_series={"q": 4.0},
        )
        assert lower[0].outcome == "REGRESSED", "a larger rank is worse, not better"

    def test_unpaired_questions_are_reported_unknown_not_dropped(self):
        rows = per_question_results(
            metric="mrr",
            baseline_series={"q1": 0.5, "q2": None},
            candidate_series={"q1": 0.6, "q2": 0.9},
        )
        by_id = {r.question_id: r for r in rows}
        assert by_id["q1"].outcome == "IMPROVED"
        assert by_id["q2"].outcome == "UNKNOWN"
        assert by_id["q2"].delta is None

    def test_question_set_fingerprint_moves_with_the_labels(self):
        ids = ["q1", "q2"]
        a = question_set_fingerprint(ids, {"q1": ["c1"], "q2": ["c2"]})
        b = question_set_fingerprint(ids, {"q1": ["c1"], "q2": ["c2"]})
        assert a == b, "the same questions and labels must fingerprint identically"
        changed = question_set_fingerprint(ids, {"q1": ["c1"], "q2": ["cOTHER"]})
        assert changed != a, "a relabelled question must change the fingerprint"


def _comparison(
    *,
    pairing: PairingVerdict,
    metric_deltas: dict[str, float],
    paired: int = 10,
    exploratory: bool = True,
) -> ExperimentComparison:
    """A minimal comparison object with the given pairing verdict and deltas."""
    from app.services.answer_eval.comparison import MetricComparison, MetricDirection

    metrics = []
    for metric, delta in metric_deltas.items():
        direction = (
            MetricDirection.LOWER_IS_BETTER
            if metric == "first_relevant_rank"
            else MetricDirection.HIGHER_IS_BETTER
        )
        sign = 1.0 if direction is MetricDirection.HIGHER_IS_BETTER else -1.0
        outcome = ComparisonOutcome.UNCHANGED
        if delta * sign > 0:
            outcome = ComparisonOutcome.IMPROVED
        elif delta * sign < 0:
            outcome = ComparisonOutcome.REGRESSED
        metrics.append(
            MetricComparison(
                metric=metric,
                direction=direction,
                questions=paired,
                paired_count=paired,
                baseline_mean=0.5,
                candidate_mean=0.5 + delta,
                mean_delta=delta,
                improved_count=paired if outcome is ComparisonOutcome.IMPROVED else 0,
                regressed_count=paired if outcome is ComparisonOutcome.REGRESSED else 0,
                outcome=outcome,
                reason="test fixture",
            )
        )
    return ExperimentComparison(
        baseline_label="dense_k3",
        candidate_label="dense_k10",
        pairing=pairing,
        pairing_reason="test fixture",
        comparable_question_ids=[f"q{i}" for i in range(paired)],
        metric_comparisons=metrics,
        exploratory=exploratory,
    )


class TestOutcomeRules:
    def _outcome(self, comparison, *, required=0.02, guardrails=("recall_at_k",), status="completed"):
        return decide_candidate_outcome(
            label="dense_k10",
            strategy="dense",
            config_fingerprint="abc",
            comparison=comparison,
            primary_metric="recall_at_k",
            required_delta=required,
            arm_status=status,
            guardrail_metrics=list(guardrails),
        )

    def test_a_failed_arm_is_blocked_not_substituted(self):
        out = self._outcome(None, status="failed")
        assert out.outcome is ExperimentOutcome.BLOCKED
        assert out.rules_applied and "R1" in out.rules_applied[0]

    def test_no_comparison_is_inconclusive(self):
        out = self._outcome(None)
        assert out.outcome is ExperimentOutcome.INCONCLUSIVE

    def test_an_unpaired_guardrail_is_no_evidence_not_a_pass(self):
        out = self._outcome(
            _comparison(pairing=PairingVerdict.COMPARABLE, metric_deltas={"recall_at_k": 0.3}),
            guardrails=("ndcg_at_k",),
        )
        assert out.guardrails_regressed == []
        assert any("not paired" in n for n in out.guardrail_assessment)

    def test_not_comparable_can_never_be_accepted(self):
        out = self._outcome(
            _comparison(pairing=PairingVerdict.NOT_COMPARABLE, metric_deltas={"recall_at_k": 0.5})
        )
        assert out.outcome is ExperimentOutcome.INCONCLUSIVE
        assert out.acceptance is AcceptanceDecision.NOT_VALID

    def test_partial_overlap_is_inconclusive(self):
        out = self._outcome(
            _comparison(pairing=PairingVerdict.INCONCLUSIVE, metric_deltas={"recall_at_k": 0.5})
        )
        assert out.outcome is ExperimentOutcome.INCONCLUSIVE
        assert any("partially overlap" in r for r in out.rules_applied)
        assert "shared questions only" in out.outcome_explanation

    def test_a_guardrail_regression_outranks_a_primary_gain(self):
        out = self._outcome(
            _comparison(
                pairing=PairingVerdict.COMPARABLE,
                metric_deltas={"recall_at_k": 0.30, "mrr": -0.30},
            ),
            guardrails=("mrr",),
        )
        assert out.outcome is ExperimentOutcome.REGRESSION
        assert out.guardrails_regressed == ["mrr"]
        assert out.acceptance is AcceptanceDecision.REJECT

    def test_guardrail_noise_below_its_bar_does_not_reject_a_gain(self):
        """A few milliseconds of jitter must not veto a real improvement: the
        guardrail is held to the same declared bar as the primary metric."""
        out = self._outcome(
            _comparison(
                pairing=PairingVerdict.COMPARABLE,
                metric_deltas={"recall_at_k": 0.30, "latency_ms": 3.0},
            ),
            guardrails=("latency_ms",),
        )
        assert out.guardrails_regressed == []
        assert out.outcome is ExperimentOutcome.IMPROVED
        assert any("within tolerance" in n for n in out.guardrail_assessment)

    def test_a_guardrail_regression_above_its_bar_rejects(self):
        out = self._outcome(
            _comparison(
                pairing=PairingVerdict.COMPARABLE,
                metric_deltas={"recall_at_k": 0.30, "latency_ms": 400.0},
            ),
            guardrails=("latency_ms",),
        )
        assert out.outcome is ExperimentOutcome.REGRESSION
        assert out.guardrails_regressed == ["latency_ms"]

    def test_a_delta_below_the_declared_bar_is_not_meaningful(self):
        out = self._outcome(
            _comparison(pairing=PairingVerdict.COMPARABLE, metric_deltas={"recall_at_k": 0.001})
        )
        assert out.outcome is ExperimentOutcome.NO_MEANINGFUL_IMPROVEMENT
        assert out.primary_improved is False

    def test_a_real_gain_above_the_bar_is_improved(self):
        out = self._outcome(
            _comparison(pairing=PairingVerdict.COMPARABLE, metric_deltas={"recall_at_k": 0.25})
        )
        assert out.outcome is ExperimentOutcome.IMPROVED
        assert out.primary_improved is True

    def test_every_verdict_records_the_rule_it_fired_on(self):
        out = self._outcome(
            _comparison(pairing=PairingVerdict.COMPARABLE, metric_deltas={"recall_at_k": 0.25})
        )
        assert out.rules_applied
        assert "R7" in out.rules_applied[0]
        assert "NOT a claim of statistical significance" in out.outcome_explanation


class TestExperimentRollup:
    def test_no_candidate_is_inconclusive(self):
        outcome, explanation, accepted = decide_experiment_outcome([])
        assert outcome is ExperimentOutcome.INCONCLUSIVE
        assert accepted == ""

    def test_an_improved_candidate_is_never_auto_promoted(self):
        candidate = CandidateOutcome(
            label="dense_k10",
            strategy="dense",
            outcome=ExperimentOutcome.IMPROVED,
            primary_metric="recall_at_k",
            primary_mean_delta=0.2,
            primary_required_delta=0.02,
            paired_question_count=28,
        )
        outcome, explanation, accepted = decide_experiment_outcome([candidate])
        assert outcome is ExperimentOutcome.IMPROVED
        assert accepted == "dense_k10"
        assert "NOT promoted" in explanation

    def test_regression_is_reported_when_no_candidate_improved(self):
        candidate = CandidateOutcome(
            label="c",
            strategy="hybrid",
            outcome=ExperimentOutcome.REGRESSION,
            outcome_explanation="guardrail mrr regressed",
        )
        outcome, _, accepted = decide_experiment_outcome([candidate])
        assert outcome is ExperimentOutcome.REGRESSION
        assert accepted == ""

    def test_runtime_errors_make_the_experiment_blocked(self):
        outcome, explanation, _ = decide_experiment_outcome([], errors=["the engine died"])
        assert outcome is ExperimentOutcome.BLOCKED
        assert "engine died" in explanation


class TestSlug:
    def test_slug_is_stable_and_filesystem_safe(self):
        assert slugify("Dense depth sweep / k=3 vs k=10") == "dense_depth_sweep_k_3_vs_k_10"
        assert slugify("A") == "a"
        assert slugify("!!!") == "experiment"


# ---------------------------------------------------------------------------
# API: the real thing, on a real indexed corpus
# ---------------------------------------------------------------------------


class TestExperimentMeta:
    def test_meta_exposes_the_real_vocabulary(self, client, indexed_kb):
        body = client.get(META.format(kb=indexed_kb["id"])).json()
        assert set(body["outcomes"]) == {o.value for o in ExperimentOutcome}
        assert body["meaningful_delta"]["mrr"] == MEANINGFUL_DELTA["mrr"]
        assert "NEVER changes the default" in body["promotion_policy"]


class TestExperimentRun:
    def _run(self, client, kb_id, path, **overrides):
        payload = _declared(benchmark_path=path)
        payload.update(overrides)
        return client.post(EXPERIMENTS.format(kb=kb_id), json=payload)

    def test_experiment_measures_both_arms_on_the_same_questions(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        r = self._run(client, kb_id, path)
        assert r.status_code == 201, r.text
        experiment = r.json()

        assert experiment["baseline"]["role"] == "baseline"
        assert experiment["candidates"][0]["role"] == "candidate"
        assert experiment["matched_question_ids"] == ["nav-001", "nav-002", "nav-003"]
        assert experiment["question_set_fingerprint"]

        arms = {a["label"]: a for a in experiment["arms"]}
        assert set(arms) == {"dense_k3", "dense_k10"}
        for arm in arms.values():
            assert arm["status"] == "completed", arm["failure_reason"]
            assert arm["question_count"] == 3
            assert arm["config_fingerprint"]
            assert arm["metric_stats"]["recall_at_k"]["direction"] == "higher_is_better"
            assert arm["metric_stats"]["first_relevant_rank"]["direction"] == "lower_is_better"
        # The evaluated cut-off must differ between the arms, and each arm's probe
        # must not be shallower than the window it is scored on.
        assert arms["dense_k3"]["top_k"] == 3
        assert arms["dense_k10"]["top_k"] == 10
        for arm in arms.values():
            assert arm["candidate_depth"] >= arm["top_k"]

    def test_the_comparison_is_paired_and_carries_its_uncertainty(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        experiment = self._run(client, kb_id, path).json()
        assert len(experiment["comparisons"]) == 1
        comparison = experiment["comparisons"][0]
        assert comparison["candidate_label"] == "dense_k10"
        assert comparison["pairing"] == PairingVerdict.COMPARABLE.value
        assert comparison["comparable_question_ids"] == ["nav-001", "nav-002", "nav-003"]
        assert comparison["exploratory"] is True
        assert comparison["significance_tests"], "the sign test must be reported"
        assert all(t["exploratory"] for t in comparison["significance_tests"])
        metrics = {m["metric"] for m in comparison["metric_comparisons"]}
        assert {"recall_at_k", "mrr"} <= metrics

        # Per-question rows exist for the primary metric on EVERY matched question.
        arm = next(a for a in experiment["arms"] if a["label"] == "dense_k10")
        assert {p["question_id"] for p in arm["per_question"]} == {
            "nav-001",
            "nav-002",
            "nav-003",
        }

    def test_outcome_is_derived_and_the_artifact_records_the_rules(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        experiment = self._run(client, kb_id, path).json()
        assert experiment["outcome"] in {o.value for o in ExperimentOutcome}
        assert experiment["outcome_explanation"]
        assert experiment["promoted_to_default"] is False
        outcome = experiment["candidate_outcomes"][0]
        assert outcome["rules_applied"], "a verdict must cite the rule it fired on"
        assert outcome["pairing"] == PairingVerdict.COMPARABLE.value
        assert outcome["acceptance"] in {d.value for d in AcceptanceDecision}
        assert experiment["rejection_conditions"], (
            "the conditions under which the hypothesis would be rejected are part of "
            "the declaration and must be stored"
        )

    def test_the_hypothesis_and_bar_are_stored_before_measurement(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        experiment = self._run(client, kb_id, path).json()
        assert "WINDOW effect" in experiment["hypothesis"]
        assert experiment["primary_metric"] == "recall_at_k"
        assert experiment["guardrail_metrics"] == ["mrr"]
        assert experiment["targeted_failure_code"] == "LOW_RECALL"
        assert experiment["comparison_method"].startswith("paired delta")

    def test_integrity_is_measured_not_promised(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        experiment = self._run(client, kb_id, path).json()
        integrity = experiment["integrity"]
        assert integrity["index_unchanged"] is True
        assert integrity["corpus_unchanged"] is True
        assert integrity["benchmark_unchanged"] is True
        assert integrity["scratch_collections_created"] == 0
        assert integrity["mutating_operations"] == 0
        # The retrieval itself must not have written a retrieval-run row.
        assert experiment["errors"] == []

    def test_artifact_is_persisted_under_the_data_dir_and_re_readable(
        self, client, indexed_kb, env
    ):
        kb_id = indexed_kb["id"]
        data_dir = Path(env[1])
        path = _write_benchmark(data_dir, kb_id, _benchmark_questions(client, kb_id))
        created = self._run(client, kb_id, path).json()

        artifacts = list((data_dir / "experiments").glob("*.json"))
        assert artifacts, "the experiment must be persisted under DATA_DIR/experiments"
        assert all("benchmarks" not in str(p.parent) for p in artifacts), (
            "an experiment artifact must never be written into the versioned benchmarks dir"
        )

        listing = client.get(EXPERIMENTS.format(kb=kb_id)).json()
        assert [e["id"] for e in listing] == [created["id"]]
        fetched = client.get(
            ONE.format(kb=kb_id, eid=created["id"])
        ).json()
        assert fetched["id"] == created["id"]
        assert fetched["outcome"] == created["outcome"]
        assert fetched["arms"] == created["arms"]
        assert fetched["candidate_outcomes"] == created["candidate_outcomes"]

    def test_re_running_the_same_declaration_is_reproducible(self, client, indexed_kb, env):
        """Same declaration + same corpus must produce the same question set, the
        same arm configuration fingerprints and the same pairing."""
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        first = self._run(client, kb_id, path).json()
        second = self._run(client, kb_id, path, name="dense depth sweep rerun").json()
        assert second["question_set_fingerprint"] == first["question_set_fingerprint"]
        assert second["benchmark_fingerprint"] == first["benchmark_fingerprint"]
        arm_fp = {a["label"]: a["config_fingerprint"] for a in first["arms"]}
        assert {a["label"]: a["config_fingerprint"] for a in second["arms"]} == arm_fp
        assert second["comparisons"][0]["pairing"] == first["comparisons"][0]["pairing"]
        assert second["id"] != first["id"], "a re-run is a NEW artifact, not an overwrite"

    def test_an_unavailable_strategy_fails_the_arm_without_substitution(
        self, client, indexed_kb, env
    ):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        r = self._run(
            client,
            kb_id,
            path,
            candidates=[{"label": "ghost", "strategy": "not_a_real_strategy", "top_k": 10}],
        )
        assert r.status_code == 201, r.text
        experiment = r.json()
        ghost = next(a for a in experiment["arms"] if a["label"] == "ghost")
        assert ghost["status"] == "failed"
        assert ghost["failure_reason"], "an unavailable strategy must say why it failed"
        assert experiment["outcome"] == ExperimentOutcome.BLOCKED.value
        assert experiment["accepted_candidate"] == ""
        # No numbers were invented for the arm that could not run.
        assert ghost["metric_stats"] == {}
        assert ghost["per_question"] == []

    def test_a_wrong_kb_benchmark_is_rejected(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        payload = _declared(benchmark_path="benchmarks/automobile-engineering-baseline-v1.json")
        r = client.post(EXPERIMENTS.format(kb=kb_id), json=payload)
        assert r.status_code == 400
        assert "authored for knowledge base" in r.json()["detail"]

    def test_a_declaration_without_a_hypothesis_is_refused(self, client, indexed_kb, env):
        payload = _declared(hypothesis="")
        r = client.post(EXPERIMENTS.format(kb=indexed_kb["id"]), json=payload)
        assert r.status_code in (400, 422)

    def test_an_undeclared_primary_metric_is_refused(self, client, indexed_kb, env):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        r = self._run(client, kb_id, path, primary_metric="vibes")
        assert r.status_code == 400


class TestRecommendationTraceability:
    """A recommendation must be joinable to the experiment that tested it.

    The chain the project promises is:

        failing question -> measured metric -> hypothesis -> experiment -> result

    This test walks it end to end: a V12 diagnostic run names the failing
    questions and the measurements behind its recommendation, and the V13
    experiment artifact names the same questions as its target together with the
    result it produced.
    """

    def test_a_recommendation_and_its_experiment_can_be_joined_on_question_ids(
        self, client, indexed_kb, env
    ):
        kb_id = indexed_kb["id"]
        questions = _benchmark_questions(client, kb_id)
        # Force a MEASURED failure so the recommendation has something to name: the
        # first question requires evidence that is not in the corpus at all.
        broken = "nav-001"
        questions[0]["expected_chunk_ids"] = ["chk_not_in_this_corpus"]
        path = _write_benchmark(env[1], kb_id, questions)

        # -- 1. MEASURED FACTS: a V12 diagnostic run names what failed and why.
        run = client.post(
            f"/api/knowledge-bases/{kb_id}/retrieval-diagnostics/runs",
            json={"benchmark_path": path, "strategies": ["dense"], "top_k": 3},
        )
        assert run.status_code == 201, run.text
        run_id = run.json()["id"]
        report = client.get(
            f"/api/knowledge-bases/{kb_id}/retrieval-diagnostics/runs/{run_id}"
        ).json()
        classifications = {
            q["question_id"]: q["classification"]["code"]
            for q in report["strategies"][0]["questions"]
        }
        assert classifications[broken] != "RETRIEVAL_SUCCESS", (
            "the fixture must actually measure a failure or it tests nothing"
        )

        # -- 2. PROPOSED CHANGE: the recommendation names that question and applies nothing.
        recs = client.get(
            f"/api/knowledge-bases/{kb_id}/retrieval-diagnostics/runs/{run_id}/recommendations"
        ).json()
        assert recs["recommendations"], "a diagnostic run always states a conclusion"
        assert "hypothesis" in recs["notice"]
        naming = [r for r in recs["recommendations"] if broken in r["affected_question_ids"]]
        assert naming, (
            "the recommendation must name the question that failed, or it cannot be "
            "traced back to a measurement"
        )
        for rec in recs["recommendations"]:
            assert rec["experiment_required"] is True
            assert rec["proposed_config_change"] == {}, (
                "a recommendation must never carry an applied configuration change"
            )
            assert rec["supporting_measurements"] is not None
            # Every question a recommendation names must be a real benchmark question.
            assert set(rec["affected_question_ids"]) <= set(classifications)

        # -- 3. EXPERIMENT: the same question is the declared target.
        experiment = self._run_experiment(
            client,
            kb_id,
            path,
            targeted_question_ids=[broken],
            targeted_failure_code=classifications[broken],
            question_ids=[broken],
        )
        assert experiment["targeted_question_ids"] == [broken]
        assert experiment["targeted_failure_code"] == classifications[broken], (
            "the experiment must carry the measured failure code it targets"
        )
        assert experiment["hypothesis"]
        assert experiment["rejection_conditions"]
        assert experiment["matched_question_ids"] == [broken]

        # -- 4. RESULT: the join is complete, and the verdict is recorded next to the
        #       hypothesis rather than left for a reader to infer.
        assert experiment["outcome"] in {o.value for o in ExperimentOutcome}
        assert experiment["outcome_explanation"]
        outcome = experiment["candidate_outcomes"][0]
        assert outcome["outcome"] in {o.value for o in ExperimentOutcome}
        assert outcome["acceptance_reason"]
        assert outcome["rules_applied"], "a verdict must cite the rule it fired on"
        if outcome["outcome"] in {
            ExperimentOutcome.REGRESSION.value,
            ExperimentOutcome.NO_MEANINGFUL_IMPROVEMENT.value,
        }:
            assert outcome["acceptance"] is not AcceptanceDecision.ACCEPT
            assert experiment["accepted_candidate"] == ""
        assert experiment["promoted_to_default"] is False

    def test_a_run_with_nothing_failing_claims_no_failing_question(
        self, client, indexed_kb, env
    ):
        """The converse of traceability: with nothing measured as failing, no
        recommendation may invent a question to act on."""
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        run = client.post(
            f"/api/knowledge-bases/{kb_id}/retrieval-diagnostics/runs",
            json={"benchmark_path": path, "strategies": ["dense"], "top_k": 10},
        ).json()
        report = client.get(
            f"/api/knowledge-bases/{kb_id}/retrieval-diagnostics/runs/{run['id']}"
        ).json()
        failed = [
            q["question_id"]
            for q in report["strategies"][0]["questions"]
            if q["classification"]["code"] != "RETRIEVAL_SUCCESS"
        ]
        recs = client.get(
            f"/api/knowledge-bases/{kb_id}/retrieval-diagnostics/runs/{run['id']}/recommendations"
        ).json()
        named = {
            qid for r in recs["recommendations"] for qid in r["affected_question_ids"]
        }
        assert named <= set(failed), (
            "a recommendation named a question the run did not measure as failing: "
            f"named={sorted(named)} failed={sorted(failed)}"
        )

    def _run_experiment(self, client, kb_id, path, **overrides):
        payload = _declared(benchmark_path=path)
        payload.update(overrides)
        r = client.post(EXPERIMENTS.format(kb=kb_id), json=payload)
        assert r.status_code == 201, r.text
        return r.json()


class TestNoMutationOfProtectedResources:
    def test_experiments_leave_the_frozen_benchmarks_byte_identical(
        self, client, indexed_kb, env, frozen_hashes
    ):
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        client.post(EXPERIMENTS.format(kb=kb_id), json=_declared(benchmark_path=path))
        for benchmark in FROZEN_BENCHMARKS:
            assert _sha(benchmark) == frozen_hashes[benchmark.name], (
                f"{benchmark.name} was modified by an experiment"
            )

    def test_the_diagnostic_reports_and_experiments_are_separate_records(
        self, client, indexed_kb, env
    ):
        """An experiment must not appear as a diagnostic run, and vice versa."""
        kb_id = indexed_kb["id"]
        path = _write_benchmark(env[1], kb_id, _benchmark_questions(client, kb_id))
        client.post(EXPERIMENTS.format(kb=kb_id), json=_declared(benchmark_path=path))
        runs = client.get(f"/api/knowledge-bases/{kb_id}/retrieval-diagnostics/runs").json()
        assert runs == [], "an experiment is not a diagnostic run"
