"""V12 retrieval-diagnostics runner — runs a retrieval benchmark and measures it.

Orchestration only: the rules live in `retrieval_diagnostics.py`. This module
loads the FROZEN retrieval benchmark read-only, probes evidence availability in
the live corpus and vector index, runs each requested strategy over the same
question set, classifies every question, compares strategies on matched
questions, and records the whole thing as ONE append-only diagnostic run.

## Safety rules this module enforces

* The benchmark file is NEVER written. Its sha256 is captured before and after
  the run and stored on the report, so "the instrument was not touched" is a
  measurement, not a promise.
* The knowledge base is NEVER mutated. No chunk is written, no vector is
  upserted or deleted, no collection is created or dropped, and no retrieval
  configuration is saved. Only read-only vector-store calls are made
  (`collection_info`, `all_point_payloads`, `fetch_vectors`).
* A diagnostic probe does NOT persist retrieval runs by default: a probe is not
  a user query, and polluting the retrieval history with 3 x 28 synthetic runs
  would make the KB's own history unreadable. The report records the count that
  WAS persisted (0 by default) so the claim is checkable.
* A strategy that cannot run is recorded as failed WITH its error. No other
  strategy's numbers are ever substituted for it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.repositories.sqlite_repo import Repository
from app.services.answer_eval.comparison import ExperimentComparison
from app.services.answer_eval.diagnostics import (
    build_question_diagnostic,
    summarise_diagnostics,
)
from app.services.answer_eval.retrieval_diagnostics import (
    ALL_RETRIEVAL_CODES,
    MIN_CANDIDATE_DEPTH,
    AvailabilityRollup,
    EvidenceAvailability,
    build_probe,
    QuestionRetrievalDiagnostic,
    RetrievalDiagnosticsReport,
    RetrievalIntegritySnapshot,
    StrategyDiagnostics,
    assess_evidence_availability,
    classify_retrieval_question,
    compare_strategies,
    rollup_availability,
    summarise_strategy,
)
from app.services.answer_eval.recommendations import (
    RecommendationReport,
    analyse_retrieval,
)
from app.services.answer_eval.retrieval_diagnostics import (
    recommendations_from_strategy,
)
from app.services.retrieval.service import (
    KnowledgeBaseNotFound,
    RetrievalConfigError,
    RetrievalService,
)
from app.services.vector_store import factory as vector_factory

logger = logging.getLogger(__name__)

#: The frozen retrieval benchmark shipped with the project.
DEFAULT_RETRIEVAL_BENCHMARK_PATH = "benchmarks/automobile-engineering-baseline-v1.json"

#: Strategies run when the caller does not choose. Dense is the documented
#: baseline; bm25 and hybrid are the two the registry declares as available.
DEFAULT_STRATEGIES: tuple[str, ...] = ("dense",)


class BenchmarkValidationError(ValueError):
    """The retrieval benchmark cannot be trusted as written."""


class RetrievalBenchmarkQuestion(BaseModel):
    """One question of the retrieval benchmark (its real schema)."""

    id: str
    subdomain: str = ""
    question: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    provenance: str = ""


class RetrievalBenchmark(BaseModel):
    """The retrieval benchmark artifact, validated on load."""

    benchmark: str
    version: int = 1
    created_at: str = ""
    kb_id: str = ""
    kb_name: str = ""
    authorship: dict[str, Any] = Field(default_factory=dict)
    relevance_model: str = "binary"
    questions: list[RetrievalBenchmarkQuestion] = Field(default_factory=list)
    #: Questions that carry NO expected evidence at all. They cannot be scored, so
    #: they are diagnosed as UNKNOWN — the benchmark still loads, because refusing
    #: the whole artifact would hide the questions that CAN be measured.
    unlabelled_question_ids: list[str] = Field(default_factory=list)

    def fingerprint(self) -> str:
        """Stable content hash of the benchmark's identity and labels.

        Whitespace or comment changes do not move it; a changed question, id,
        expected chunk or expected document does. Recorded on every run so a
        result can never be attributed to a benchmark it did not use.
        """
        payload = json.dumps(
            {
                "benchmark": self.benchmark,
                "version": self.version,
                "kb_id": self.kb_id,
                "questions": [
                    {
                        "id": q.id,
                        "question": q.question,
                        "expected_chunk_ids": sorted(q.expected_chunk_ids),
                        "expected_document_ids": sorted(q.expected_document_ids),
                    }
                    for q in self.questions
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


#: Repository root = backend/app/services/answer_eval/<this file>
REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_benchmark_path(path: str | Path) -> Path:
    """Resolve a benchmark path the way callers name it: repo-relative.

    ``benchmarks/foo.json`` is how a versioned artifact is referenced, and the
    server may have been started from anywhere, so a bare relative path is tried
    against the process CWD first and then against the repository root. The
    HTTP layer additionally enforces an allowed-root guard (see
    ``routes_answer_eval._resolve_benchmark_path``) so a caller cannot use this
    to read arbitrary files; this helper only makes the developer-facing default
    work from any directory.
    """
    given = Path(path)
    if given.is_file():
        return given
    candidate = (REPO_ROOT / str(path)).resolve()
    if candidate.is_file():
        return candidate
    return given


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes; "" when it cannot be read."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def load_retrieval_benchmark(path: str | Path) -> RetrievalBenchmark:
    """Load and validate the retrieval benchmark. Read-only."""
    p = resolve_benchmark_path(path)
    if not p.exists():
        raise BenchmarkValidationError(f"retrieval benchmark not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkValidationError(f"retrieval benchmark is not valid JSON: {exc}") from exc
    try:
        bench = RetrievalBenchmark.model_validate(raw)
    except Exception as exc:
        raise BenchmarkValidationError(f"retrieval benchmark failed validation: {exc}") from exc

    problems: list[str] = []
    if not bench.questions:
        problems.append("benchmark contains no questions")
    seen: set[str] = set()
    unlabelled: list[str] = []
    for q in bench.questions:
        if q.id in seen:
            problems.append(f"duplicate question id {q.id!r}")
        seen.add(q.id)
        if not q.expected_chunk_ids and not q.expected_document_ids:
            unlabelled.append(q.id)
    # A question with no reference evidence is NOT a reason to reject the artifact:
    # it is diagnosed as UNKNOWN with a reason, which is the useful outcome. A
    # benchmark where NOTHING is labelled can measure nothing, so that IS refused.
    if unlabelled and len(unlabelled) == len(bench.questions):
        problems.append(
            "no question carries expected chunk or document ids, so no retrieval "
            "metric could be measured at all"
        )
    if problems:
        raise BenchmarkValidationError(
            "retrieval benchmark is not scorable as written: " + "; ".join(problems)
        )
    return bench.model_copy(update={"unlabelled_question_ids": unlabelled})


@dataclass
class _AvailabilityContext:
    """KB-scoped evidence-availability facts shared by every arm of a run.

    Built ONCE per run because the index enumeration and the document list are
    properties of the knowledge base, not of the retrieval configuration under
    test. Sharing them is what makes 'the evidence was there and this arm still
    failed to rank it' a fair statement about every arm.
    """

    chunk_rows: dict[str, Any]
    availability_by_question: dict[str, list[EvidenceAvailability]]
    rollup_by_question: dict[str, AvailabilityRollup]
    probe_notes: list[str]
    availability_warnings: list[str]
    index_probe: str = "unavailable"
    expected_chunk_count: int = 0


class RetrievalDiagnosticRunner:
    """Runs and records retrieval diagnostics for one knowledge base."""

    def __init__(self, repo: Repository, settings: Settings | None = None) -> None:
        self._repo = repo
        self._settings = settings or get_settings()
        self._retrieval = RetrievalService(repo, self._settings)

    # -- read-only probes ---------------------------------------------------

    def _index_chunk_ids(
        self, kb_id: str, backend: str | None
    ) -> tuple[set[str] | None, list[str]]:
        """Chunk ids with a vector in the index. (None, notes) when unprobeable."""
        notes: list[str] = []
        try:
            store = vector_factory.create_vector_store(self._settings, backend=backend)
            payloads = store.all_point_payloads(kb_id)
            return set(payloads.keys()), notes
        except NotImplementedError:
            notes.append(
                "the configured vector backend cannot enumerate points, so index "
                "presence of the expected evidence is UNVERIFIED for this run"
            )
            return None, notes
        except Exception as exc:  # network/collection errors must not be treated as absence
            notes.append(
                f"the vector index could not be read ({exc}), so index presence of the "
                "expected evidence is UNVERIFIED for this run"
            )
            return None, notes

    def _index_point_count(self, kb_id: str, backend: str | None) -> int | None:
        try:
            store = vector_factory.create_vector_store(self._settings, backend=backend)
            info = store.collection_info(kb_id)
            count = info.get("points_count")
            return int(count) if count is not None else None
        except Exception as exc:
            logger.warning("Could not read index point count for %s: %s", kb_id, exc)
            return None

    def evidence_audit(
        self,
        kb_id: str,
        bench: RetrievalBenchmark,
        *,
        benchmark_path: str | Path = DEFAULT_RETRIEVAL_BENCHMARK_PATH,
    ) -> dict[str, Any]:
        """Read-only: can the benchmark's expected evidence be retrieved at all?

        Nothing is run and nothing is written. This is the question that has to be
        answered BEFORE any strategy is compared, because 'the evidence is not in
        the corpus' and 'the evidence was not retrieved' are different findings
        that a bare metric cannot tell apart.
        """
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")
        benchmark_path = resolve_benchmark_path(benchmark_path)

        expected_ids = sorted({c for q in bench.questions for c in q.expected_chunk_ids})
        expected_docs = sorted({d for q in bench.questions for d in q.expected_document_ids})
        chunk_rows = {c.id: c for c in self._repo.get_chunks_by_ids(kb_id, expected_ids)}
        document_ids = {d.id for d in self._repo.list_documents(kb_id)}
        index_chunk_ids, probe_notes = self._index_chunk_ids(
            kb_id, kb.vector_backend or "qdrant"
        )

        usage: dict[str, list[str]] = {}
        docs_by_chunk: dict[str, list[str]] = {}
        for q in bench.questions:
            for cid in q.expected_chunk_ids:
                usage.setdefault(cid, []).append(q.id)
                docs = docs_by_chunk.setdefault(cid, [])
                for doc_id in q.expected_document_ids:
                    if doc_id not in docs:
                        docs.append(doc_id)
        shared = {c for c, qs in usage.items() if len(qs) > 1}

        availability, _rollup, warnings = assess_evidence_availability(
            docs_by_chunk,
            chunk_rows=chunk_rows,
            index_chunk_ids=index_chunk_ids,
            document_ids=document_ids,
            multi_question_chunk_ids=shared,
        )
        by_chunk = {a.chunk_id: a for a in availability}
        by_question = {
            q.id: [by_chunk[c] for c in q.expected_chunk_ids if c in by_chunk]
            for q in bench.questions
        }
        rollups = {
            qid: rollup_availability(items) for qid, items in by_question.items()
        }

        counts: dict[str, int] = {}
        for rollup in rollups.values():
            counts[rollup.value] = counts.get(rollup.value, 0) + 1

        index_points = self._index_point_count(kb_id, kb.vector_backend)
        return {
            "kb_id": kb_id,
            "benchmark": bench.benchmark,
            "benchmark_fingerprint": bench.fingerprint(),
            "benchmark_path": str(benchmark_path),
            "benchmark_sha256": file_sha256(benchmark_path),
            "relevance_model": bench.relevance_model,
            "authorship": bench.authorship,
            "question_count": len(bench.questions),
            "expected_chunk_count": len(expected_ids),
            "expected_document_count": len(expected_docs),
            "index_probe": (
                "verified" if index_chunk_ids is not None else "unavailable"
            ),
            "index_points": index_points,
            "index_probe_notes": probe_notes,
            "probe_notes": warnings,
            "availability_counts": dict(sorted(counts.items())),
            "questions": [
                {
                    "question_id": q.id,
                    "availability_rollup": rollups[q.id].value,
                    "expected_chunk_ids": q.expected_chunk_ids,
                    "availability": [
                        a.model_dump(mode="json") for a in by_question[q.id]
                    ],
                }
                for q in bench.questions
            ],
            "verdict": (
                "expected evidence fully resolvable in the corpus and index"
                if counts.get(AvailabilityRollup.ALL_INDEXED.value, 0) == len(bench.questions)
                else "EVIDENCE GAPS FOUND — read availability_counts before blaming retrieval"
            ),
            "note": (
                "Read-only audit. It runs no strategy, writes no row and never modifies "
                "the corpus, the index or the benchmark."
            ),
        }

    # -- shared availability preparation -------------------------------------

    def _prepare_availability(
        self,
        kb_id: str,
        kb: Any,
        questions: list[RetrievalBenchmarkQuestion],
    ) -> _AvailabilityContext:
        """Probe the corpus and the index for every question's required evidence.

        KB-scoped work (the index enumeration and the document list) is done ONCE
        here and shared by every strategy or experiment arm, so: (a) it is
        cheaper, and (b) two arms of the same experiment cannot disagree about
        whether the evidence was present in the first place.
        """
        all_expected_ids = sorted({c for q in questions for c in q.expected_chunk_ids})
        chunk_rows = {c.id: c for c in self._repo.get_chunks_by_ids(kb_id, all_expected_ids)}
        document_ids = {d.id for d in self._repo.list_documents(kb_id)}

        index_chunk_ids, probe_notes = self._index_chunk_ids(
            kb_id, (kb.vector_backend or "qdrant") if kb is not None else None
        )

        # Chunk-level availability: the same chunk can be required by several
        # questions, so the check is keyed by chunk and reused.
        expected_docs_by_chunk: dict[str, list[str]] = {}
        usage: dict[str, list[str]] = {}
        for q in questions:
            for cid in q.expected_chunk_ids:
                docs = expected_docs_by_chunk.setdefault(cid, [])
                for doc_id in q.expected_document_ids:
                    if doc_id not in docs:
                        docs.append(doc_id)
                usage.setdefault(cid, []).append(q.id)
        shared_chunks = {c for c, qs in usage.items() if len(qs) > 1}

        availability, _benchmark_rollup, availability_warnings = assess_evidence_availability(
            expected_docs_by_chunk,
            chunk_rows=chunk_rows,
            index_chunk_ids=index_chunk_ids,
            document_ids=document_ids,
            availability_is_verified=True,
            multi_question_chunk_ids=shared_chunks,
        )
        availability_by_chunk: dict[str, EvidenceAvailability] = {
            a.chunk_id: a for a in availability
        }
        return _AvailabilityContext(
            chunk_rows=chunk_rows,
            availability_by_question={
                q.id: [
                    availability_by_chunk[c]
                    for c in q.expected_chunk_ids
                    if c in availability_by_chunk
                ]
                for q in questions
            },
            rollup_by_question={
                q.id: rollup_availability(
                    [
                        availability_by_chunk[c]
                        for c in q.expected_chunk_ids
                        if c in availability_by_chunk
                    ]
                )
                for q in questions
            },
            probe_notes=probe_notes,
            availability_warnings=list(availability_warnings),
            index_probe="verified" if index_chunk_ids is not None else "unavailable",
            expected_chunk_count=len(all_expected_ids),
        )

    # -- the run ------------------------------------------------------------

    def run(
        self,
        kb_id: str,
        *,
        strategies: list[str] | None = None,
        top_k: int = 10,
        benchmark_path: str | Path = DEFAULT_RETRIEVAL_BENCHMARK_PATH,
        question_ids: list[str] | None = None,
        persist: bool = True,
        persist_retrieval_runs: bool = False,
        candidate_depth: int | None = None,
    ) -> RetrievalDiagnosticsReport:
        started = time.perf_counter()
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")

        path = resolve_benchmark_path(benchmark_path)
        bench = load_retrieval_benchmark(path)
        if bench.kb_id and bench.kb_id != kb_id:
            raise BenchmarkValidationError(
                f"benchmark {bench.benchmark!r} was authored for knowledge base "
                f"{bench.kb_id!r}, not {kb_id!r}: scoring it here would compare a corpus "
                "it was not written against"
            )
        wanted = set(question_ids or [])
        questions = [q for q in bench.questions if not wanted or q.id in wanted]
        if not questions:
            raise BenchmarkValidationError("no benchmark questions selected for this run")

        # -- integrity: BEFORE -------------------------------------------------
        snapshot = RetrievalIntegritySnapshot(
            kb_id=kb_id,
            index_points_before=self._index_point_count(kb_id, kb.vector_backend),
            chunk_rows_before=self._repo.count_chunks(kb_id),
            corpus_revision_before=self._repo.corpus_revision(kb_id),
            benchmark_sha256_before=file_sha256(path),
        )

        ctx = self._prepare_availability(kb_id, kb, questions)
        chunk_rows = ctx.chunk_rows
        probe_notes = ctx.probe_notes

        # Candidates are probed deeper than the evaluated cut-off, so "retrieved
        # below k" and "not retrieved at all" stay distinguishable. A caller may
        # widen it further, never narrow it below the cut-off it is measuring.
        depth = candidate_depth if candidate_depth is not None else max(top_k, MIN_CANDIDATE_DEPTH)
        if depth < top_k:
            raise BenchmarkValidationError(
                f"candidate_depth ({depth}) must be at least top_k ({top_k}): a candidate "
                "set shallower than the evaluated window would silently turn ranking "
                "failures into apparent recall failures"
            )

        strategy_names = [s.strip().lower() for s in (strategies or list(DEFAULT_STRATEGIES)) if s.strip()]
        results: list[StrategyDiagnostics] = []
        for name in strategy_names:
            results.append(
                self._run_strategy(
                    kb=kb,
                    kb_id=kb_id,
                    name=name,
                    top_k=top_k,
                    candidate_depth=depth,
                    questions=questions,
                    chunk_rows=ctx.chunk_rows,
                    availability_by_question=ctx.availability_by_question,
                    rollup_by_question=ctx.rollup_by_question,
                    probe_notes=ctx.probe_notes,
                    availability_warnings=ctx.availability_warnings,
                )
            )

        # -- comparisons: matched questions only ------------------------------
        comparisons: list[ExperimentComparison] = []
        completed = [s for s in results if s.status == "completed"]
        for other in completed[1:]:
            comparisons.append(
                compare_strategies(
                    completed[0],
                    other,
                    benchmark_name=bench.benchmark,
                    benchmark_fingerprint=bench.fingerprint(),
                )
            )

        # -- recommendations ---------------------------------------------------
        recommendations = RecommendationReport()
        if completed:
            primary = completed[0]
            base = analyse_retrieval(
                [q.evidence for q in primary.questions],
                current_top_k=primary.top_k,
                current_strategy=primary.strategy,
                summary=summarise_diagnostics([q.evidence for q in primary.questions]),
            )
            recommendations = recommendations_from_strategy(
                primary, base_report=base, benchmark_name=bench.benchmark
            )
        else:
            recommendations.data_gaps.append(
                "no strategy completed, so no recommendation can be derived from measurement"
            )
            recommendations.data_gaps.append(probe_notes and probe_notes[0] or "")

        # -- integrity: AFTER --------------------------------------------------
        snapshot.index_points_after = self._index_point_count(kb_id, kb.vector_backend)
        snapshot.chunk_rows_after = self._repo.count_chunks(kb_id)
        snapshot.corpus_revision_after = self._repo.corpus_revision(kb_id)
        snapshot.benchmark_sha256_after = file_sha256(path)
        snapshot.retrieval_runs_persisted = 0 if not persist_retrieval_runs else len(strategy_names) * len(questions)

        limitations: list[str] = [
            "retrieval-only diagnostics: no generator ran, so evidence SELECTION was not "
            "exercised and no correctness claim of any kind is made here",
            f"metrics are computed at a single evaluated cut-off (top_k={top_k}) while "
            f"candidates were probed to depth {depth}: a question classified LOW_RECALL "
            "was retrieved outside the window and could become RETRIEVAL_SUCCESS at a "
            "larger k, and NO_RELEVANT_EVIDENCE means absent from the FULL probe depth, "
            "not merely from the window",
            "the benchmark's own authorship records human review as "
            f"{bench.authorship.get('human_review', 'NOT RECORDED')!r}",
        ]
        if probe_notes:
            limitations.extend(probe_notes)
        if bench.relevance_model == "binary":
            limitations.append(
                "relevance is binary and the labels are necessity-only, so precision@k is "
                "a lower bound and LOW_PRECISION is never emitted"
            )
        if bench.unlabelled_question_ids:
            limitations.append(
                f"{len(bench.unlabelled_question_ids)} question(s) carry no expected "
                f"evidence ({', '.join(bench.unlabelled_question_ids[:5])}) and are "
                "classified UNKNOWN rather than scored"
            )
        if any(s.status != "completed" for s in results):
            limitations.append(
                "at least one requested strategy did not run; its row records the failure "
                "and no substitute numbers were produced"
            )

        notes: list[str] = [
            f"{len(questions)} benchmark question(s) x {len(strategy_names)} strategy(ies)",
            f"strategies requested: {', '.join(strategy_names)}",
            f"evaluated cut-off k={top_k}; candidate probe depth={depth}",
            f"outcome vocabulary: {', '.join(c.value for c in ALL_RETRIEVAL_CODES)}",
            f"probe duration: {round((time.perf_counter() - started) * 1000, 1)} ms",
        ]

        report = RetrievalDiagnosticsReport(
            id=f"rdx_{hashlib.sha256(f'{kb_id}:{bench.fingerprint()}:{time.time()}'.encode()).hexdigest()[:12]}",
            kb_id=kb_id,
            kb_name=kb.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            benchmark_name=bench.benchmark,
            benchmark_version=bench.version,
            benchmark_path=str(path),
            benchmark_fingerprint=bench.fingerprint(),
            relevance_model=bench.relevance_model,
            benchmark_authorship=dict(bench.authorship),
            top_k=top_k,
            question_count=len(questions),
            matched_question_ids=sorted(q.id for q in questions),
            strategies=results,
            comparisons=comparisons,
            recommendations=recommendations,
            integrity=snapshot,
            limitations=limitations,
            notes=notes,
        )
        if persist:
            self._repo.create_retrieval_diagnostic_run(report)
        return report

    # -- experiment arms ----------------------------------------------------

    def run_configs(
        self,
        kb_id: str,
        arms: list[dict[str, Any]],
        *,
        benchmark_path: str | Path = DEFAULT_RETRIEVAL_BENCHMARK_PATH,
        question_ids: list[str] | None = None,
    ) -> list["StrategyDiagnostics"]:
        """Measure several retrieval CONFIGURATIONS over the same questions.

        Each arm is ``{"label", "strategy", "top_k", "candidate_depth", "params"}``.
        This is the measurement engine the controlled-experiment runner uses: it
        deliberately reuses :meth:`_run_strategy`, so an experiment arm and a
        diagnostic strategy are scored by exactly the same code with exactly the
        same metric definitions and the same availability probe.

        It is READ-ONLY: no chunk, vector, collection or retrieval configuration
        is written. An arm that cannot be built is returned as ``status=failed``
        with its error — never dropped, never substituted.
        """
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")
        path = resolve_benchmark_path(benchmark_path)
        bench = load_retrieval_benchmark(path)
        if bench.kb_id and bench.kb_id != kb_id:
            raise BenchmarkValidationError(
                f"benchmark {bench.benchmark!r} was authored for knowledge base "
                f"{bench.kb_id!r}, not {kb_id!r}"
            )
        wanted = set(question_ids or [])
        questions = [q for q in bench.questions if not wanted or q.id in wanted]
        if not questions:
            raise BenchmarkValidationError("no benchmark questions selected")
        if not arms:
            raise BenchmarkValidationError("no retrieval configuration was supplied")

        ctx = self._prepare_availability(kb_id, kb, questions)
        out: list[StrategyDiagnostics] = []
        for arm in arms:
            name = (arm.get("strategy") or "").strip().lower()
            top_k = int(arm.get("top_k") or 10)
            depth = arm.get("candidate_depth")
            depth = int(depth) if depth is not None else max(top_k, MIN_CANDIDATE_DEPTH)
            if depth < top_k:
                raise BenchmarkValidationError(
                    f"arm {arm.get('label')!r}: candidate_depth ({depth}) must be at "
                    f"least top_k ({top_k})"
                )
            out.append(
                self._run_strategy(
                    kb=kb,
                    kb_id=kb_id,
                    name=name,
                    top_k=top_k,
                    candidate_depth=depth,
                    questions=questions,
                    chunk_rows=ctx.chunk_rows,
                    availability_by_question=ctx.availability_by_question,
                    rollup_by_question=ctx.rollup_by_question,
                    probe_notes=ctx.probe_notes,
                    availability_warnings=ctx.availability_warnings,
                    label=(arm.get("label") or name),
                    param_overrides=dict(arm.get("params") or {}),
                )
            )
        return out

    # -- one strategy -------------------------------------------------------

    def _run_strategy(
        self,
        *,
        kb,
        kb_id: str,
        name: str,
        top_k: int,
        candidate_depth: int,
        questions,
        chunk_rows,
        availability_by_question: dict[str, list[EvidenceAvailability]],
        rollup_by_question: dict[str, AvailabilityRollup],
        probe_notes: list[str],
        availability_warnings: list[str],
        label: str | None = None,
        param_overrides: dict[str, Any] | None = None,
    ) -> StrategyDiagnostics:
        """Measure ONE arm of a retrieval configuration over the given questions.

        ``name`` selects the registered strategy; ``label`` names the ARM in the
        report (so two arms of the same strategy — e.g. dense at k=3 and dense at
        k=10 — stay distinguishable). ``param_overrides`` is the configuration
        under test. The evaluated cut-off stays ``top_k`` and the probe depth
        stays ``candidate_depth``: an experiment may widen the probe, never the
        window it is scored on.
        """
        base = StrategyDiagnostics(
            strategy=label or name,
            top_k=top_k,
            candidate_depth=candidate_depth,
            notes=list(probe_notes),
        )
        if availability_warnings:
            base.notes.extend(availability_warnings)
        try:
            overrides: dict[str, Any] = {"top_k": candidate_depth}
            # An experiment arm varies the retrieval configuration; the metric
            # cut-off (`top_k`) stays a separate, explicitly recorded knob so a
            # deeper probe can never be mistaken for a wider evaluation window.
            for key, value in (param_overrides or {}).items():
                if key == "top_k":
                    continue
                overrides[key] = value
            params = self._retrieval.resolve_params(
                kb_id, strategy=name, overrides=overrides
            )
        except RetrievalConfigError as exc:
            base.status = "failed"
            base.failure_reason = str(exc)
            base.notes.append(
                "the strategy is not available in this build; it was NOT substituted"
            )
            return base

        # The parameters recorded are the ones ACTUALLY EXECUTED (a wider window
        # than the metric cut-off), so the run is reproducible as it ran rather
        # than as it was described.
        base.retrieval_params = params.model_dump(mode="json")
        base.config_fingerprint = params.config_fingerprint()
        if candidate_depth != top_k:
            base.notes.append(
                f"candidates probed to depth {candidate_depth}; metrics and verdicts "
                f"evaluated at k={top_k}"
            )
        latest = self._repo.latest_corpus_version(kb_id)
        if latest is not None:
            base.corpus_version = latest.version
            base.corpus_fingerprint = latest.fingerprint
        base.index_points = self._index_point_count(kb_id, kb.vector_backend)

        try:
            retriever, notes = self._retrieval.build_retriever(kb, params)
        except RetrievalConfigError as exc:
            base.status = "failed"
            base.failure_reason = str(exc)
            return base
        except Exception as exc:
            base.status = "failed"
            base.failure_reason = f"could not build the retriever: {exc}"
            base.notes.append("no other strategy was run in its place")
            return base
        base.notes.extend(notes)

        started = time.perf_counter()
        diagnostics: list[QuestionRetrievalDiagnostic] = []
        for q in questions:
            t0 = time.perf_counter()
            try:
                response = retriever.retrieve_with_params(kb_id, q.question, params)
            except Exception as exc:
                base.status = "failed"
                base.failure_reason = f"retrieval failed on {q.id}: {exc}"
                base.notes.append(
                    "questions measured before the failure are kept; the rest are absent "
                    "rather than reported as misses"
                )
                break
            latency_ms = round((time.perf_counter() - t0) * 1000, 3)

            chunk_index: dict[str, dict[str, Any]] = {}
            for item in response.results:
                row = chunk_rows.get(item.chunk_id)
                prov = dict(getattr(item, "provenance", {}) or {})
                breakdown = getattr(item, "score_breakdown", None)
                chunk_index[item.chunk_id] = {
                    "score": item.score,
                    "score_breakdown": breakdown.model_dump(mode="json")
                    if hasattr(breakdown, "model_dump")
                    else dict(breakdown or {}),
                    "document_id": getattr(row, "document_id", None) or item.document_id,
                    "source_id": getattr(row, "source_id", None) or prov.get("source_id"),
                    "source_title": getattr(row, "source_title", None) or prov.get("source_title"),
                    "source_url": getattr(row, "source_url", None) or prov.get("source_url"),
                    "source_type": getattr(row, "source_type", None) or prov.get("source_type"),
                    "publisher": getattr(row, "publisher", None) or prov.get("publisher"),
                    "section": getattr(row, "section", None) or prov.get("section"),
                    "page": getattr(row, "page", None) or prov.get("page"),
                    "trust_score": getattr(row, "trust_score", None),
                    "content_hash": getattr(row, "content_hash", None),
                }

            # Metrics of record are computed over the WINDOW the evaluated k
            # produced, so they keep the published meaning at k and stay
            # comparable with the frozen artifact. The deeper candidate list is
            # kept separately, because it is what distinguishes "below the
            # cut-off" from "not retrieved at all".
            all_ids = [r.chunk_id for r in response.results]
            window_ids = all_ids[:top_k]
            diag = build_question_diagnostic(
                question_id=q.id,
                question=q.question,
                subdomain=q.subdomain,
                answerability="unknown",
                strategy=label or name,
                top_k=top_k,
                retrieval_params=params.model_dump(mode="json"),
                ranked_chunk_ids=window_ids,
                chunk_index=chunk_index,
                required_chunk_ids=list(q.expected_chunk_ids),
                required_document_ids=list(q.expected_document_ids),
                evidence_selected_ids=[],
                latency_ms=latency_ms,
                trace_id=getattr(response, "retrieval_run_id", "") or "",
            )
            probe = build_probe(
                all_ids,
                required_chunk_ids=list(q.expected_chunk_ids),
                top_k=top_k,
                candidate_depth=candidate_depth,
            )
            availability = availability_by_question.get(q.id, [])
            rollup = rollup_by_question.get(q.id, AvailabilityRollup.UNKNOWN)
            classification = classify_retrieval_question(
                question_id=q.id,
                diagnostic=diag,
                availability=availability,
                rollup=rollup,
                top_k=top_k,
                probe=probe,
            )
            diagnostics.append(
                QuestionRetrievalDiagnostic(
                    question_id=q.id,
                    question=q.question,
                    subdomain=q.subdomain,
                    strategy=label or name,
                    top_k=top_k,
                    retrieval_params=params.model_dump(mode="json"),
                    expected_chunk_ids=list(q.expected_chunk_ids),
                    expected_document_ids=list(q.expected_document_ids),
                    availability=availability,
                    availability_rollup=rollup,
                    evidence=diag,
                    probe=probe,
                    classification=classification,
                    latency_ms=latency_ms,
                )
            )
            if response.reranker is not None:
                status = getattr(response.reranker, "status", None)
                base.reranker_status = getattr(status, "value", None) or str(status or "")
            if not base.embedding_model and response.embedding_model:
                base.embedding_model = response.embedding_model
            if not base.retrieval_backend and response.retrieval_backend:
                base.retrieval_backend = response.retrieval_backend

        base.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        base.questions = diagnostics
        base.summary = summarise_strategy(diagnostics)
        if base.summary.no_ground_truth_question_count:
            base.notes.append(
                f"{base.summary.no_ground_truth_question_count} question(s) have no "
                "reference evidence and are classified UNKNOWN"
            )
        if base.status == "completed" and len(diagnostics) != len(questions):
            base.status = "failed"
        return base


__all__ = [
    "BenchmarkValidationError",
    "DEFAULT_RETRIEVAL_BENCHMARK_PATH",
    "DEFAULT_STRATEGIES",
    "RetrievalBenchmark",
    "RetrievalBenchmarkQuestion",
    "RetrievalDiagnosticRunner",
    "REPO_ROOT",
    "file_sha256",
    "load_retrieval_benchmark",
    "resolve_benchmark_path",
]
