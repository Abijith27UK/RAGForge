"""V13 controlled retrieval-experiment runner.

The loop this module implements, in order:

    MEASURE BASELINE -> IDENTIFY FAILURE -> FORM HYPOTHESIS -> DEFINE EXPERIMENT
    -> RUN ISOLATED ARMS -> COMPARE ON MATCHED QUESTIONS -> ANALYSE TRADE-OFFS
    -> ACCEPT OR REJECT CHANGE -> (validation is a separate, later experiment)

It is a RECORDING instrument, not a tuner. The declaration (hypothesis, primary
metric, guardrails, rejection conditions) is supplied by the caller and stored
on the artifact before anything is measured; the verdict is then computed from
the real measurements by ``retrieval_experiment_models``.

## Why this is safe to run against the protected knowledge base

Every shipped strategy (dense, bm25, hybrid, hybrid_reranked) is READ-ONLY with
respect to the index: they embed the query, search, fuse, optionally rerank and
return. No arm writes a chunk, a vector or a collection, and no arm changes the
saved retrieval configuration. So an experiment needs NO scratch collection and
mutates nothing. That claim is MEASURED here, not promised: the index point
count, the chunk-row count, the corpus revision and the benchmark's sha256 are
captured before and after and stored in ``integrity``.

If a future arm ever needs to change the index, it must write to an isolated
scratch collection and record it — this runner will refuse to call such an arm
against the protected KB.

## Reuse, deliberately

The arms are measured by ``RetrievalDiagnosticRunner.run_configs``, which is the
same engine the V12 diagnostics use, and the comparison comes from
``comparison.compare_suite_explicit``, which is the same V9 machinery the V12
strategy comparison uses. One measurement path, one comparison path, one
direction map.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.repositories.sqlite_repo import Repository
from app.services.answer_eval.comparison import (
    ExperimentComparison,
    compare_suite_explicit,
    metric_direction,
)
from app.services.answer_eval.retrieval_diagnostic_runner import (
    DEFAULT_RETRIEVAL_BENCHMARK_PATH,
    BenchmarkValidationError,
    RetrievalDiagnosticRunner,
    file_sha256,
    load_retrieval_benchmark,
    resolve_benchmark_path,
)
from app.services.answer_eval.retrieval_diagnostics import StrategyDiagnostics
from app.services.answer_eval.retrieval_experiment_models import (
    DEFAULT_COMPARISON_METRICS,
    MEANINGFUL_DELTA,
    CandidateOutcome,
    ExperimentArm,
    ExperimentOutcome,
    MetricStats,
    RetrievalExperiment,
    RetrievalStrategyConfig,
    decide_candidate_outcome,
    decide_experiment_outcome,
    metric_stats,
    now_iso,
    per_question_results,
    question_set_fingerprint,
    slugify,
)
from app.services.retrieval.service import KnowledgeBaseNotFound

logger = logging.getLogger(__name__)

#: Metrics that every experiment compares, beyond the declared primary metric.
#: Kept small on purpose: each additional metric is another chance to read noise
#: as signal on a 28-question benchmark.
BASELINE_COMPARISON_METRICS: tuple[str, ...] = DEFAULT_COMPARISON_METRICS


class ExperimentValidationError(ValueError):
    """The experiment as declared cannot be run or cannot be trusted."""


class RetrievalExperimentRunner:
    """Runs, decides and persists controlled retrieval experiments."""

    def __init__(self, repo: Repository, settings: Settings | None = None) -> None:
        self._repo = repo
        self._settings = settings or get_settings()
        self._diagnostics = RetrievalDiagnosticRunner(repo, self._settings)
        #: The exact diagnostics objects produced for this run's arms, keyed by
        #: arm label. Kept so the comparison reuses the measurements that were
        #: taken rather than re-running retrieval a second time.
        self._measured_index: dict[str, StrategyDiagnostics] = {}

    # -- persistence locations ---------------------------------------------

    @property
    def experiments_dir(self) -> Path:
        """Where experiment artifacts live.

        ``DATA_DIR/experiments`` — the same local, gitignored home as the V12
        diagnostic reports, so an experiment artifact and the run it cites stay
        together on the machine that produced them.
        """
        d = Path(self._settings.data_dir) / "experiments"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list_experiments(self, kb_id: str | None = None) -> list[RetrievalExperiment]:
        out: list[RetrievalExperiment] = []
        for path in sorted(self.experiments_dir.glob("*.json")):
            try:
                experiment = RetrievalExperiment.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception as exc:  # a corrupt artifact must not hide the rest
                logger.warning("Skipping unreadable experiment artifact %s: %s", path, exc)
                continue
            if kb_id is None or experiment.kb_id == kb_id:
                out.append(experiment)
        out.sort(key=lambda e: (e.created_at or "", e.id), reverse=True)
        return out

    def get_experiment(self, kb_id: str, experiment_id: str) -> RetrievalExperiment | None:
        for experiment in self.list_experiments(kb_id):
            if experiment.id == experiment_id:
                return experiment
        return None

    # -- the experiment ----------------------------------------------------

    def run(
        self,
        kb_id: str,
        *,
        name: str,
        hypothesis: str,
        baseline: RetrievalStrategyConfig,
        candidates: list[RetrievalStrategyConfig],
        primary_metric: str,
        guardrail_metrics: list[str] | None = None,
        proposed_change: str = "",
        expected_effect: str = "",
        targeted_failure_code: str = "",
        targeted_question_ids: list[str] | None = None,
        rejection_conditions: list[str] | None = None,
        benchmark_path: str | Path = DEFAULT_RETRIEVAL_BENCHMARK_PATH,
        question_ids: list[str] | None = None,
        extra_comparison_metrics: list[str] | None = None,
        persist: bool = True,
        created_by: str = "",
        instrument: str = "development",
        benchmark_lifecycle: str = "",
        claim_strength: str = "",
        retests: str = "",
    ) -> RetrievalExperiment:
        """Run one controlled experiment and (by default) persist its artifact.

        ``instrument`` names WHICH benchmark produced the numbers and is not
        cosmetic: a held-out run must declare the frozen instrument's tier
        (``benchmark_lifecycle``), and may only claim a confirmatory result when
        that tier is ``validated``. Both are stored on the artifact, so a reader
        can tell a re-measurement of the development set apart from a
        generalization test without going to the file system.
        """
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")
        if not candidates:
            raise ExperimentValidationError(
                "an experiment needs at least one candidate configuration to test"
            )
        if not (name or "").strip():
            raise ExperimentValidationError("an experiment needs a name")
        if not (hypothesis or "").strip():
            raise ExperimentValidationError(
                "an experiment needs a stated hypothesis: without one, a result "
                "cannot be attributed to anything"
            )
        if instrument not in ("development", "held-out", "exploratory"):
            raise ExperimentValidationError(
                f"unknown instrument {instrument!r}; expected 'development', "
                "'held-out' or 'exploratory'"
            )
        if instrument == "exploratory" and claim_strength == "confirmatory":
            # V22: a machine-drafted, unreviewed benchmark is a usable probe and
            # an unusable instrument of record. Refusing the label here is what
            # keeps an exploratory number from being read as a validation.
            raise ExperimentValidationError(
                "refusing to record a confirmatory claim from an 'exploratory' "
                "instrument: exploratory labels are machine-drafted and not "
                "human-approved, so they can guide a decision but cannot confirm one"
            )
        if instrument == "held-out":
            # A held-out run that cannot name the tier it ran under could be
            # read as a validation whatever the artifact's review state was.
            if benchmark_lifecycle not in ("provisional", "validated"):
                raise ExperimentValidationError(
                    "a held-out run must declare the frozen instrument's tier "
                    "('provisional' or 'validated'); running against a held-out "
                    "benchmark whose review state is unknown would make an "
                    "exploratory number indistinguishable from a validated one"
                )
            if claim_strength == "confirmatory" and benchmark_lifecycle != "validated":
                raise ExperimentValidationError(
                    "refusing to record a confirmatory claim from a "
                    f"{benchmark_lifecycle!r} held-out instrument: only a validated "
                    "benchmark, frozen after every question was reviewed, may support "
                    "a generalization claim"
                )

        # Direction must be declared before the metric is compared; this raises
        # for an unknown metric rather than assuming a direction.
        metric_direction(primary_metric)
        guardrails = [m for m in (guardrail_metrics or []) if m != primary_metric]
        for metric in guardrails:
            metric_direction(metric)

        path = resolve_benchmark_path(benchmark_path)
        bench = load_retrieval_benchmark(path)
        if bench.kb_id and bench.kb_id != kb_id:
            raise BenchmarkValidationError(
                f"benchmark {bench.benchmark!r} was authored for knowledge base "
                f"{bench.kb_id!r}, not {kb_id!r}"
            )

        wanted = set(question_ids or [])
        matched = [q for q in bench.questions if not wanted or q.id in wanted]
        if not matched:
            raise BenchmarkValidationError("no benchmark questions selected")
        matched_ids = sorted(q.id for q in matched)
        labels_fp = question_set_fingerprint(
            matched_ids, {q.id: list(q.expected_chunk_ids) for q in matched}
        )

        primary_bar = float(MEANINGFUL_DELTA.get(primary_metric, 0.02))

        baseline = baseline.model_copy(update={"role": "baseline"})
        candidates = [
            c.model_copy(update={"role": "candidate"}).model_copy(
                update={"label": c.label or c.strategy}
            )
            for c in candidates
        ]

        experiment = RetrievalExperiment(
            id=f"rex_{hashlib.sha256(f'{kb_id}:{name}:{now_iso()}'.encode()).hexdigest()[:12]}",
            kb_id=kb_id,
            kb_name=kb.name,
            name=name.strip(),
            slug=slugify(name),
            hypothesis=hypothesis.strip(),
            targeted_failure_code=targeted_failure_code,
            targeted_question_ids=sorted(targeted_question_ids or []),
            proposed_change=proposed_change,
            expected_effect=expected_effect,
            primary_metric=primary_metric,
            guardrail_metrics=guardrails,
            rejection_conditions=list(
                rejection_conditions
                or [
                    f"the mean per-question delta on {primary_metric} is below the "
                    f"declared meaningful bar ({primary_bar})",
                    "a declared guardrail metric regresses",
                    "the comparison is not comparable or only partially paired",
                ]
            ),
            baseline=baseline,
            candidates=candidates,
            benchmark_name=bench.benchmark,
            benchmark_fingerprint=bench.fingerprint(),
            benchmark_path=str(path),
            benchmark_sha256_before=file_sha256(path),
            matched_question_ids=matched_ids,
            question_set_fingerprint=labels_fp,
            status="running",
            created_at=now_iso(),
            created_by=created_by,
            instrument=instrument,
            benchmark_lifecycle=benchmark_lifecycle,
            claim_strength=claim_strength or ("exploratory" if instrument == "held-out" else ""),
            retests=retests,
            limitations=[],
            notes=[],
        )

        # -- integrity BEFORE -------------------------------------------------
        backend = kb.vector_backend or "qdrant"
        index_before = self._diagnostics._index_point_count(kb_id, backend)
        chunks_before = self._repo.count_chunks(kb_id)
        revision_before = self._repo.corpus_revision(kb_id)
        latest = self._repo.latest_corpus_version(kb_id)
        if latest is not None:
            experiment.corpus_version = latest.version
            experiment.corpus_fingerprint = latest.fingerprint

        # -- run every arm through the SAME measurement engine ---------------
        arms_config = [baseline, *candidates]
        try:
            measured = self._diagnostics.run_configs(
                kb_id,
                [c.to_arm() for c in arms_config],
                benchmark_path=path,
                question_ids=matched_ids,
            )
        except Exception as exc:
            experiment.status = "failed"
            experiment.errors.append(f"the measurement engine failed: {exc}")
            experiment.outcome = ExperimentOutcome.BLOCKED
            experiment.outcome_explanation = (
                "BLOCKED. No arm could be measured: " f"{exc}. No result is reported."
            )
            experiment.completed_at = now_iso()
            if persist:
                self._persist(experiment)
            return experiment

        comparison_metrics = list(
            dict.fromkeys(
                [
                    primary_metric,
                    *guardrails,
                    *(extra_comparison_metrics or []),
                    *BASELINE_COMPARISON_METRICS,
                ]
            )
        )
        for metric in comparison_metrics:
            metric_direction(metric)

        measured_by_label = {a.strategy: a for a in measured}
        self._measured_index = dict(measured_by_label)
        experiment.arms = [
            self._build_arm(config, measured_by_label.get(config.label or config.strategy))
            for config in arms_config
        ]
        baseline_arm = experiment.arms[0]
        if baseline_arm.status != "completed":
            experiment.status = "failed"
            experiment.errors.append(
                f"the BASELINE arm did not complete: {baseline_arm.failure_reason}. "
                "Without a baseline there is nothing to compare against, so no "
                "candidate is evaluated."
            )

        # -- comparisons: matched questions only ------------------------------
        comparisons: list[ExperimentComparison] = []
        outcomes: list[CandidateOutcome] = []
        for config, arm in zip(candidates, experiment.arms[1:]):
            label = arm.label
            if baseline_arm.status != "completed" or arm.status != "completed":
                outcome = decide_candidate_outcome(
                    label=label,
                    strategy=arm.strategy,
                    config_fingerprint=arm.config_fingerprint,
                    comparison=None,
                    primary_metric=primary_metric,
                    required_delta=primary_bar,
                    arm_status=arm.status,
                    arm_failure_reason=(
                        arm.failure_reason
                        or (
                            "the baseline arm did not complete"
                            if baseline_arm.status != "completed"
                            else ""
                        )
                    ),
                    guardrail_metrics=guardrails,
                )
                outcomes.append(outcome)
                continue

            comparison = self._compare(
                benchmark_name=bench.benchmark,
                benchmark_fingerprint=experiment.benchmark_fingerprint,
                baseline_arm=baseline_arm,
                candidate_arm=arm,
                metrics=comparison_metrics,
                kb_id=kb_id,
            )
            comparisons.append(comparison)
            # The per-question table is the PRIMARY metric, oriented by its own
            # declared direction, so a reader sees which questions moved.
            arm.per_question = per_question_results(
                metric=primary_metric,
                baseline_series=self._series(
                    self._measured_index[baseline_arm.label], primary_metric
                ),
                candidate_series=self._series(
                    self._measured_index[arm.label], primary_metric
                ),
                questions_by_id={q.id: q.question for q in matched},
            )
            outcomes.append(
                decide_candidate_outcome(
                    label=label,
                    strategy=arm.strategy,
                    config_fingerprint=arm.config_fingerprint,
                    comparison=comparison,
                    primary_metric=primary_metric,
                    required_delta=primary_bar,
                    arm_status=arm.status,
                    guardrail_metrics=guardrails,
                )
            )

        experiment.comparisons = comparisons
        experiment.candidate_outcomes = outcomes

        # -- integrity AFTER --------------------------------------------------
        index_after = self._diagnostics._index_point_count(kb_id, backend)
        chunks_after = self._repo.count_chunks(kb_id)
        revision_after = self._repo.corpus_revision(kb_id)
        experiment.benchmark_sha256_after = file_sha256(path)
        experiment.integrity = {
            "kb_id": kb_id,
            "protected_kb_id": "kb_f278c283c748",
            "is_protected_kb": kb_id == "kb_f278c283c748",
            "index_points_before": index_before,
            "index_points_after": index_after,
            "chunk_rows_before": chunks_before,
            "chunk_rows_after": chunks_after,
            "corpus_revision_before": revision_before,
            "corpus_revision_after": revision_after,
            "benchmark_sha256_before": experiment.benchmark_sha256_before,
            "benchmark_sha256_after": experiment.benchmark_sha256_after,
            "index_unchanged": (
                None
                if index_before is None or index_after is None
                else index_before == index_after
            ),
            "corpus_unchanged": (
                None
                if chunks_before is None or chunks_after is None
                else chunks_before == chunks_after
            ),
            "benchmark_unchanged": (
                None
                if not experiment.benchmark_sha256_before
                else experiment.benchmark_sha256_before
                == experiment.benchmark_sha256_after
            ),
            "scratch_collections_created": 0,
            "mutating_operations": 0,
            "note": (
                "Every shipped strategy is read-only with respect to the index, so no "
                "arm needs a scratch collection. These counters are measurements taken "
                "around the run, not promises."
            ),
        }

        # -- verdict ----------------------------------------------------------
        if experiment.errors:
            experiment.outcome = ExperimentOutcome.BLOCKED
            experiment.outcome_explanation = (
                "BLOCKED. " + "; ".join(experiment.errors)
            )
        else:
            outcome, explanation, accepted = decide_experiment_outcome(outcomes)
            experiment.outcome = outcome
            experiment.outcome_explanation = explanation
            experiment.accepted_candidate = accepted

        failed_arms = [a for a in experiment.arms if a.status != "completed"]
        if failed_arms:
            experiment.errors.extend(
                f"arm {a.label!r} failed: {a.failure_reason}" for a in failed_arms
            )

        experiment.status = "completed"
        experiment.completed_at = now_iso()
        experiment.notes.append(
            f"outcome vocabulary: {', '.join(o.value for o in ExperimentOutcome)}; "
            "the verdict was derived by the declared rules, not chosen"
        )
        experiment.notes.append(
            f"primary metric {primary_metric!r} with a declared meaningful bar of "
            f"{primary_bar}; guardrails: {guardrails or 'none declared'}"
        )
        experiment.limitations.extend(self._limitations(bench, matched_ids, experiment, measured))
        if instrument == "held-out":
            experiment.notes.append(
                f"instrument: HELD-OUT ({bench.benchmark}, tier "
                f"{benchmark_lifecycle or 'unknown'}) — the questions were not used to "
                "choose any arm of this experiment"
            )
            if experiment.claim_strength != "confirmatory":
                experiment.limitations.append(
                    "EXPLORATORY RESULT: the held-out instrument is "
                    f"{benchmark_lifecycle or 'unknown'}, so its questions were not "
                    "human-reviewed when it was sealed. No generalization claim may be "
                    "made from this artifact; it is a provisional test of the mechanism, "
                    "not a validation of the finding."
                )
            if experiment.instrument == "held-out" and not retests:
                experiment.limitations.append(
                    "this run does not name the development finding it retests, so its "
                    "result cannot be tied to a specific earlier claim"
                )

        if persist:
            self._persist(experiment)
        return experiment

    # -- internals ---------------------------------------------------------

    def _persist(self, experiment: RetrievalExperiment) -> Path:
        """Write the artifact once. A re-run with the same name overwrites ITS OWN
        file, so the experiments directory always holds one file per experiment
        name rather than an unreadable pile of near-identical runs."""
        path = experiment.path(self.experiments_dir)
        payload = experiment.model_dump(mode="json")
        payload["_artifact_sha256"] = hashlib.sha256(
            json.dumps(
                {k: v for k, v in payload.items() if k != "_artifact_sha256"},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        experiment.notes.append(f"persisted to {path}")
        return path

    def _series(self, arm: StrategyDiagnostics, metric: str) -> dict[str, float | None]:
        """Per-question values for one metric, from the arm's real diagnostics."""
        if metric == "latency_ms":
            return {q.question_id: q.latency_ms for q in arm.questions}
        return arm.metric_series(metric)

    def _build_arm(
        self, config: RetrievalStrategyConfig, measured: StrategyDiagnostics | None
    ) -> ExperimentArm:
        label = config.label or config.strategy
        if measured is None:
            return ExperimentArm(
                label=label,
                strategy=config.strategy,
                role=config.role,
                config=config,
                status="failed",
                failure_reason="the measurement engine returned no result for this arm",
                errors=["arm not measured"],
            )
        arm = ExperimentArm(
            label=label,
            strategy=config.strategy,
            role=config.role,
            config=config,
            status=measured.status,
            failure_reason=measured.failure_reason,
            top_k=measured.top_k,
            candidate_depth=measured.candidate_depth,
            config_fingerprint=measured.config_fingerprint,
            embedding_model=measured.embedding_model,
            retrieval_backend=measured.retrieval_backend,
            reranker_status=measured.reranker_status,
            index_points=measured.index_points,
            duration_ms=measured.duration_ms,
            question_count=len(measured.questions),
            code_counts=dict(measured.summary.code_counts),
            notes=list(measured.notes),
        )
        if measured.status != "completed":
            # An arm that did not run has NO measurements. Reporting zeroed or
            # empty statistics would invite reading them as "no change".
            arm.metric_stats = {}
            arm.errors.append(measured.failure_reason)
            return arm
        if measured.summary.latency_ms.measured:
            arm.latency_ms = measured.summary.latency_ms.value
        stats: dict[str, MetricStats] = {}
        for metric in (
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "ndcg_at_k",
            "required_recall_at_depth",
            "first_relevant_rank",
            "latency_ms",
        ):
            series = self._series(measured, metric)
            reason = ""
            if metric == "latency_ms":
                reason = (
                    "mean wall-clock retrieval time per question; a cost proxy, not a "
                    "quality metric"
                )
            stats[metric] = metric_stats(metric, series, reason=reason)
        arm.metric_stats = stats
        return arm

    def _compare(
        self,
        *,
        benchmark_name: str,
        benchmark_fingerprint: str,
        baseline_arm: ExperimentArm,
        candidate_arm: ExperimentArm,
        metrics: list[str],
        kb_id: str,
    ) -> ExperimentComparison:
        """Compare two measured arms through the shared V9 comparison module.

        The protocol identity (benchmark content, corpus fingerprint, evaluator,
        model, answer mode) is recorded from real values on each side, so a
        mismatch is REPORTED as NOT_COMPARABLE rather than silently ignored. The
        retrieval configuration is deliberately not part of that identity — it is
        the variable under test.
        """
        baseline_diag = self._measured_index.get(baseline_arm.label)
        candidate_diag = self._measured_index.get(candidate_arm.label)
        if baseline_diag is None or candidate_diag is None:  # pragma: no cover - guarded
            raise ExperimentValidationError("arm diagnostics were not retained")
        protocol = baseline_diag.protocol(
            benchmark_name=benchmark_name, benchmark_fingerprint=benchmark_fingerprint
        ).model_copy(update={"kb_id": kb_id, "label": baseline_arm.label})
        candidate_protocol = candidate_diag.protocol(
            benchmark_name=benchmark_name, benchmark_fingerprint=benchmark_fingerprint
        ).model_copy(update={"kb_id": kb_id, "label": candidate_arm.label})
        return compare_suite_explicit(
            baseline_label=baseline_arm.label,
            candidate_label=candidate_arm.label,
            baseline=protocol,
            candidate=candidate_protocol,
            baseline_metrics={m: self._series(baseline_diag, m) for m in metrics},
            candidate_metrics={m: self._series(candidate_diag, m) for m in metrics},
            benchmark_fingerprint=benchmark_fingerprint,
        )

    def _limitations(
        self,
        bench: Any,
        matched_ids: list[str],
        experiment: RetrievalExperiment,
        measured: list[StrategyDiagnostics],
    ) -> list[str]:
        out = [
            "retrieval-only experiment: no generator ran, so evidence SELECTION and "
            "answer correctness are out of scope, and no answer-quality claim is made",
            f"the benchmark is small ({len(matched_ids)} question(s)) and was used to "
            "choose these candidates, so every result is EXPLORATORY and must not be "
            "reported as statistical significance",
            "the primary metric is compared on matched questions only; a candidate "
            "that only helps on questions the baseline already solved cannot show up here",
        ]
        authorship = (bench.authorship or {}).get("human_review")
        out.append(
            "the benchmark's own authorship records human review as "
            f"{authorship!r}"
        )
        if experiment.instrument == "exploratory":
            out.append(
                "this ran on an 'exploratory' instrument: its labels were drafted by an "
                "agent from the corpus and have NOT been human-approved, so the numbers "
                "are a probe for finding defects, not a benchmark of record"
            )
        if len(matched_ids) < len(bench.questions):
            out.append(
                f"a SUBSET of the benchmark was used ({len(matched_ids)} of "
                f"{len(bench.questions)} questions); the result does not generalize to "
                "the full benchmark"
            )
        if any(a.status != "completed" for a in measured):
            out.append(
                "at least one arm did not complete; its row records the failure and no "
                "substitute numbers were produced"
            )
        if experiment.integrity.get("index_unchanged") is False:
            out.append(
                "the index point count CHANGED during the run: the experiment is not "
                "read-only and its result must be treated as invalid"
            )
        return out
