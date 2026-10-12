"""Retrieval-side failure analysis (V12) — measured, classified, explainable.

This module answers ONE question per benchmark question, using only measured data:

    "did the retrieval pipeline surface the evidence the benchmark says is
     required to answer this question, and if not, WHERE did it fail?"

It is deliberately scoped to RETRIEVAL. It never reads answer correctness, never
calls an LLM, and never claims that retrieval succeeded merely because a chunk
came back: every verdict is formed by comparing the returned evidence against the
benchmark's expected evidence, and against the live corpus/index.

## What this module adds to what already existed

V9 already computed per-question retrieval metrics (`answer_eval.diagnostics`),
classified ANSWER failures (`answer_eval.failure`) and produced advice with
evidence attached (`answer_eval.recommendations`). V12 adds the missing half:

1. **Evidence availability.** Before blaming the retriever, V12 asks whether the
   expected evidence is still *in the corpus* and still *in the vector index*.
   V9 could only observe "the required chunk never appeared in the candidate
   set" — which is equally consistent with a corpus gap, a missing vector, or a
   genuine ranking miss. Those require completely different fixes, so V12
   measures the difference and classifies accordingly.

2. **A retrieval-specific taxonomy** (`RetrievalFailureCode`) whose every verdict
   names the RULE that produced it and the exact observations behind it.

3. **Matched-question strategy comparison** reusing V9's protocol gate and paired
   statistics (`comparison.py`) rather than inventing new ones.

Nothing here reimplements metric arithmetic: recall/precision/MRR/nDCG come from
`services.evaluation.metrics` via `answer_eval.diagnostics`, and every metric
carries the `Measured` tri-state (value / measured / reason) so UNKNOWN never
renders as 0.0.

## Why LOW_PRECISION is in the vocabulary but never fired

The taxonomy published by V12 includes `LOW_PRECISION`, because a retrieval
diagnostic system must be able to name a precision problem. In THIS benchmark it
cannot fire honestly, and the reason is a property of the labels rather than of
the retriever:

* relevance is binary, and the benchmark labels the chunk(s) that are NECESSARY
  to answer the question (`RequiredEvidence.required` semantics, see
  `answer_eval/bundled... benchmark.py`);
* the labels are explicitly NOT exhaustive — a human selecting one canonical
  passage is not asserting that the rest of the retrieved context is irrelevant;
* therefore precision@k over these labels is a LOWER BOUND that falls as k grows
  purely because the window is large, not because retrieval got worse.

Classifying that as failure is the exact mistake `benchmark.py` warns about
("how a benchmark starts punishing answers for being well-sourced"). So V12
keeps `LOW_PRECISION` in the closed set, documents it here, and a test asserts
the classifier never emits it. Precision is still reported as a measured metric
with the lower-bound caveat attached, so a reviewer can see it.

## Precedence (why the rules are in this order)

1. no reference evidence in the benchmark            -> UNKNOWN
2. required evidence missing from the CORPUS         -> CORPUS_OR_SOURCE_DEFICIENCY
3. required chunk in the corpus but NOT in the index -> EXPECTED_EVIDENCE_UNAVAILABLE
4. retrieval returned no candidates at all           -> RETRIEVAL_FAILURE
5. candidates returned, none of them required        -> NO_RELEVANT_EVIDENCE
6. required evidence present only beyond top-k       -> LOW_RECALL
7. required evidence inside top-k but outranked      -> RANKING_FAILURE
8. required evidence at rank 1                       -> RETRIEVAL_SUCCESS

Availability (2, 3) outranks every retrieval verdict because no ranking change
can surface a passage the index does not contain. Within the retrieval rules,
"nothing came back" (4) outranks "nothing relevant came back" (5), and a
depth problem (6) outranks a ranking problem (7) because a smaller top_k window
would erase (6) entirely while (7) survives.

## Honesty rules enforced here

* No rule fires without a measured observation; `evidence` lists them verbatim.
* `UNKNOWN` means UNKNOWN. A missing label, an unprobeable index or an empty
  candidate set never becomes a failure claim.
* A verdict that could be wrong because a probe was unavailable carries
  `limitations` and PARTIAL confidence instead of a bare assertion.
* Nothing in this module writes to the corpus, the index, the benchmark or a
  retrieval configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field

from app.services.answer_eval.comparison import (
    ExperimentComparison,
    ExperimentProtocol,
    compare_suite_explicit,
)
from app.services.answer_eval.diagnostics import (
    DiagnosticMetric,
    QuestionDiagnostic,
    summarise_diagnostics,
)
from app.services.answer_eval.failure import FailureConfidence
#: `_strength_for` is V9's single definition of "how much evidence supports this
#: pattern". V12 reuses it rather than re-deriving a second strength rule that
#: could disagree with the V9 report shown beside it in the same UI.
from app.services.answer_eval.recommendations import (
    Recommendation,
    RecommendationAction,
    RecommendationReport,
    RecommendationStrength,
    _strength_for,
)


# ---------------------------------------------------------------------------
# Named constants — every threshold a rule reads, with its rationale
# ---------------------------------------------------------------------------

#: A required chunk must be at rank 1 for retrieval to be called a success.
#: Rank 1 is not an arbitrary "good" cut-off: the metric that matters downstream
#: (MRR) and the ordering the generator reads both key on the FIRST result, and
#: a required chunk at rank 2 was demonstrably outranked by a chunk the
#: benchmark does not consider necessary.
SUCCESS_RANK = 1

#: Metrics compared between two retrieval strategies. All four are declared in
#: `comparison.METRIC_DIRECTIONS`, so no direction is ever assumed.
COMPARABLE_METRICS: tuple[str, ...] = (
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "required_recall_at_depth",
    "first_relevant_rank",
)

#: Below this many paired, non-tied questions a paired test is reported but
#: marked exploratory and no claim is drawn from it. 8 is the smallest n for
#: which an exact two-sided sign test can reach p < 0.05 at all.
MIN_PAIRS_FOR_INFERENCE = 8

#: How deep the CANDIDATE SET is probed, independently of the evaluated cut-off.
#:
#: This constant exists because two of the distinctions this module must make are
#: invisible when the candidate list is truncated at k:
#:
#: * "the required passage is not in the corpus" (a coverage problem), and
#: * "the required passage was retrieved BELOW the evaluated cut-off" (a depth
#:   problem, recoverable by raising k),
#:
#: both look identical if the retriever only ever returns k results. So retrieval is
#: asked for a deeper window than the metric cut-off, and the classification reads
#: the full candidate set. 20 is chosen because the deepest evaluated cut-off the
#: project publishes for this benchmark is k=10, so 20 leaves headroom to observe
#: "deeper than the window" without turning every run into a full-corpus scan.
MIN_CANDIDATE_DEPTH = 20


class RetrievalFailureCode(str, Enum):
    """Closed set of retrieval-side outcomes.

    `RETRIEVAL_SUCCESS` is a code rather than an absence of one: "we measured
    this question and retrieval did what was asked" is itself a finding, and
    conflating it with "unclassified" is how a clean system looks broken.
    """

    #: Required evidence was retrieved and ranked first.
    RETRIEVAL_SUCCESS = "RETRIEVAL_SUCCESS"
    #: Required evidence is inside the top-k window but was outranked.
    RANKING_FAILURE = "RANKING_FAILURE"
    #: Required evidence exists at some depth but not within the evaluated k.
    LOW_RECALL = "LOW_RECALL"
    #: Reserved. Documented in the module docstring; the necessity-only binary
    #: labels make precision a lower bound, so this is never emitted.
    LOW_PRECISION = "LOW_PRECISION"
    #: Candidates were returned but none of them is required evidence.
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
    #: Retrieval returned NOTHING — a pipeline/supply failure, not a ranking one.
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    #: The required chunk is in the corpus but has no vector in the index, so no
    #: retriever could return it. An index-integrity problem.
    EXPECTED_EVIDENCE_UNAVAILABLE = "EXPECTED_EVIDENCE_UNAVAILABLE"
    #: The required chunk (or its document) is not in the corpus at all.
    CORPUS_OR_SOURCE_DEFICIENCY = "CORPUS_OR_SOURCE_DEFICIENCY"
    #: Not classifiable from the available measurements.
    UNKNOWN = "UNKNOWN"


#: Every published code, in precedence order.
ALL_RETRIEVAL_CODES: tuple[RetrievalFailureCode, ...] = (
    RetrievalFailureCode.CORPUS_OR_SOURCE_DEFICIENCY,
    RetrievalFailureCode.EXPECTED_EVIDENCE_UNAVAILABLE,
    RetrievalFailureCode.RETRIEVAL_FAILURE,
    RetrievalFailureCode.NO_RELEVANT_EVIDENCE,
    RetrievalFailureCode.LOW_RECALL,
    RetrievalFailureCode.RANKING_FAILURE,
    RetrievalFailureCode.RETRIEVAL_SUCCESS,
    RetrievalFailureCode.LOW_PRECISION,
    RetrievalFailureCode.UNKNOWN,
)

#: Codes that mean "a human should look at this question".
CRITICAL_CODES: frozenset[RetrievalFailureCode] = frozenset(
    {
        RetrievalFailureCode.CORPUS_OR_SOURCE_DEFICIENCY,
        RetrievalFailureCode.EXPECTED_EVIDENCE_UNAVAILABLE,
        RetrievalFailureCode.RETRIEVAL_FAILURE,
        RetrievalFailureCode.NO_RELEVANT_EVIDENCE,
        RetrievalFailureCode.LOW_RECALL,
        RetrievalFailureCode.RANKING_FAILURE,
    }
)

#: The name of each rule, published so a verdict can be traced to its logic.
RULE_NO_GROUND_TRUTH = "NO_GROUND_TRUTH_REFERENCE"
RULE_EVIDENCE_ABSENT_FROM_KB = "EVIDENCE_ABSENT_FROM_CORPUS"
RULE_EVIDENCE_NOT_INDEXED = "EVIDENCE_IN_CORPUS_BUT_NOT_INDEXED"
RULE_NO_CANDIDATES = "RETRIEVAL_RETURNED_NO_CANDIDATES"
RULE_NO_REQUIRED_IN_CANDIDATES = "NO_REQUIRED_CHUNK_IN_CANDIDATES"
RULE_BEYOND_CUTOFF = "REQUIRED_EVIDENCE_BEYOND_EVALUATED_CUTOFF"
RULE_OUTRANKED = "REQUIRED_EVIDENCE_OUTRANKED_WITHIN_WINDOW"
RULE_TOP_RANKED = "REQUIRED_EVIDENCE_AT_RANK_1"


class EvidenceAvailabilityStatus(str, Enum):
    """Whether a required chunk can be retrieved at all, and how we know.

    The distinction between CORPUS_ONLY and MISSING_FROM_INDEX is the point:
    'we did not verify the vector' and 'there is no vector' are different
    findings and only the second is an integrity defect.
    """

    #: Chunk row exists AND a vector for it exists in the index.
    INDEXED = "indexed"
    #: Chunk row exists; index presence could NOT be verified (probe unavailable).
    CORPUS_ONLY = "corpus_only"
    #: Chunk row exists but the index has no vector for it.
    MISSING_FROM_INDEX = "missing_from_index"
    #: No chunk row for this id.
    MISSING_FROM_CORPUS = "missing_from_corpus"
    #: Neither check could run.
    UNKNOWN = "unknown"


class AvailabilityRollup(str, Enum):
    """Per-question rollup over its required chunks."""

    ALL_INDEXED = "all_indexed"
    UNVERIFIED_INDEX = "unverified_index"
    MISSING_FROM_INDEX = "missing_from_index"
    MISSING_FROM_CORPUS = "missing_from_corpus"
    UNKNOWN = "unknown"


class EvidenceAvailability(BaseModel):
    """One required chunk checked against the live corpus AND the live index."""

    chunk_id: str
    expected_document_ids: list[str] = Field(default_factory=list)
    in_corpus: bool | None = Field(
        default=None, description="True/False when the chunk table was readable; None otherwise"
    )
    in_index: bool | None = Field(
        default=None,
        description="True/False when the vector index was probed; None when the probe was "
        "unavailable — never assume a vector exists",
    )
    status: EvidenceAvailabilityStatus = EvidenceAvailabilityStatus.UNKNOWN
    actual_document_id: str | None = None
    document_matches_benchmark: bool | None = Field(
        default=None,
        description="Whether the chunk's real document_id is one the benchmark expects. "
        "None when the benchmark ran no document check for this chunk.",
    )
    section: str | None = None
    source_title: str | None = None
    content_hash: str | None = None
    notes: list[str] = Field(default_factory=list)

    #: A computed field (not a private property) so the API and the UI receive it:
    #: "can this chunk be retrieved at all?" is the field a reviewer sorts on, and a
    #: value that only exists inside Python would silently vanish from every response.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def retrievable(self) -> bool | None:
        """True only when proven present in corpus AND index; False when proven absent."""
        if self.status is EvidenceAvailabilityStatus.INDEXED:
            return True
        if self.status in (
            EvidenceAvailabilityStatus.MISSING_FROM_CORPUS,
            EvidenceAvailabilityStatus.MISSING_FROM_INDEX,
        ):
            return False
        return None


class RetrievalProbe(BaseModel):
    """What a DEEPER candidate probe reveals beyond the evaluated cut-off.

    This model exists because the project publishes its retrieval metrics at a
    cut-off k, and `services.evaluation.metrics` defines MRR/recall/precision over
    the ranked list it is given. Truncating the list at k makes three different
    situations indistinguishable:

    * the required passage is not in the indexed corpus at all,
    * it is indexed but never entered even a deep candidate set (a recall gap),
    * it WAS retrieved, just below the evaluated window (a depth problem).

    So retrieval is probed to `candidate_depth` and this model records the extra
    information the probe adds, while the metrics of record stay windowed at k so a
    result keeps the same meaning as the frozen benchmark artifact it is compared
    with.
    """

    candidate_depth: int = 0
    candidate_chunk_ids: list[str] = Field(default_factory=list)
    #: Rank (1-indexed, in the full probe) of each required chunk that WAS found.
    required_ranks_in_probe: dict[str, int] = Field(default_factory=dict)
    #: Required chunks found in the probe but ranked beyond the evaluated k.
    required_beyond_cutoff: dict[str, int] = Field(default_factory=dict)
    #: Required chunks that appear nowhere in the probe.
    absent_from_probe: list[str] = Field(default_factory=list)
    required_recall_in_probe: float | None = Field(
        default=None,
        description="|required ∩ probe| / |required| over the FULL probe depth. None when "
        "the benchmark names no required evidence.",
    )
    truncated: bool = Field(
        default=False,
        description="True when the probe returned fewer candidates than candidate_depth, "
        "so 'absent from the probe' may only mean the retriever stopped early.",
    )


def build_probe(
    candidate_chunk_ids: list[str],
    *,
    required_chunk_ids: list[str],
    top_k: int,
    candidate_depth: int,
) -> RetrievalProbe:
    """Summarise a deep candidate list relative to the evaluated cut-off."""
    required = list(required_chunk_ids or [])
    ranks = {
        cid: rank
        for rank, cid in enumerate(candidate_chunk_ids, start=1)
        if cid in set(required)
    }
    beyond = {cid: rank for cid, rank in ranks.items() if rank > top_k}
    absent = [cid for cid in required if cid not in ranks]
    recall = (
        len(set(candidate_chunk_ids) & set(required)) / len(required) if required else None
    )
    return RetrievalProbe(
        candidate_depth=candidate_depth,
        candidate_chunk_ids=list(candidate_chunk_ids),
        required_ranks_in_probe=dict(sorted(ranks.items())),
        required_beyond_cutoff=dict(sorted(beyond.items())),
        absent_from_probe=sorted(absent),
        required_recall_in_probe=recall,
        truncated=len(candidate_chunk_ids) < candidate_depth,
    )


class RetrievalClassification(BaseModel):
    """One question's retrieval verdict, with the rule and observations behind it."""

    code: RetrievalFailureCode = RetrievalFailureCode.UNKNOWN
    rule: str = ""
    confidence: FailureConfidence = FailureConfidence.UNKNOWN
    evidence: list[str] = Field(default_factory=list)
    reason: str = ""
    limitations: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_critical(self) -> bool:
        """Whether a human should look at this question."""
        return self.code in CRITICAL_CODES


class QuestionRetrievalDiagnostic(BaseModel):
    """Everything measured for one benchmark question under ONE strategy.

    ``evidence`` is the V9 `QuestionDiagnostic` verbatim: the ranked items, the
    tri-state metrics and the ground-truth flags. V12 adds availability and a
    classification on top rather than recomputing anything V9 already measures.
    """

    question_id: str
    question: str = ""
    subdomain: str = ""
    strategy: str = ""
    top_k: int | None = None
    retrieval_params: dict[str, Any] = Field(default_factory=dict)

    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    availability: list[EvidenceAvailability] = Field(default_factory=list)
    availability_rollup: AvailabilityRollup = AvailabilityRollup.UNKNOWN

    #: The metrics of record, computed over the window the evaluated k produced.
    #: These have exactly the published meaning at k, so they are comparable with
    #: the frozen benchmark artifact and with `services.evaluation.evaluator`.
    evidence: QuestionDiagnostic
    #: The deeper probe, which the CLASSIFICATION reads but the metrics do not.
    probe: RetrievalProbe = Field(default_factory=RetrievalProbe)
    classification: RetrievalClassification = Field(default_factory=RetrievalClassification)

    latency_ms: float | None = Field(
        default=None, description="Wall-clock retrieval time for this question, when measured"
    )

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [item.chunk_id for item in self.evidence.retrieved]

    def expected_vs_retrieved(self) -> dict[str, Any]:
        """The comparison a reviewer reads: expected, retrieved, and where each landed."""
        rank_of: dict[str, int] = {
            item.chunk_id: item.rank for item in self.evidence.retrieved
        }
        for item in self.evidence.retrieved:
            rank_of[item.chunk_id] = item.rank
        return {
            "question_id": self.question_id,
            "strategy": self.strategy,
            "top_k": self.top_k,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "rank_of_expected": {
                cid: rank_of.get(cid) for cid in self.expected_chunk_ids
            },
            "expected_retrieved": [
                cid for cid in self.expected_chunk_ids if cid in rank_of
            ],
            "expected_missing": [
                cid for cid in self.expected_chunk_ids if cid not in rank_of
            ],
            "first_relevant_rank": self.evidence.first_relevant_rank,
            "candidate_depth": self.probe.candidate_depth,
            "required_ranks_in_probe": dict(self.probe.required_ranks_in_probe),
            "required_beyond_cutoff": dict(self.probe.required_beyond_cutoff),
            "absent_from_probe": list(self.probe.absent_from_probe),
            "required_recall_in_probe": self.probe.required_recall_in_probe,
            # Named distinctly from the full availability records so an API layer
            # can expose both without one silently overwriting the other.
            "availability_by_chunk": {
                a.chunk_id: a.status.value for a in self.availability
            },
        }


class RetrievalDiagnosticSummary(BaseModel):
    """Aggregate over one strategy's question diagnostics."""

    question_count: int = 0
    ground_truth_question_count: int = 0
    no_ground_truth_question_count: int = 0
    evaluated_question_count: int = 0

    code_counts: dict[str, int] = Field(default_factory=dict)
    availability_counts: dict[str, int] = Field(default_factory=dict)

    recall_at_k: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("not computed")
    )
    precision_at_k: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("not computed")
    )
    mrr: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("not computed")
    )
    ndcg_at_k: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("not computed")
    )
    required_recall_at_depth: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("not computed")
    )
    first_relevant_rank: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("not computed")
    )
    latency_ms: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("not computed")
    )
    #: How much of the required evidence the DEEPER probe found, at any probed
    #: depth. Distinct from `recall_at_k` on purpose: the gap between the two is
    #: exactly the "recoverable by a larger window" set.
    required_recall_in_probe: DiagnosticMetric = Field(
        default_factory=lambda: DiagnosticMetric.unknown("not computed")
    )

    first_relevant_rank_histogram: dict[str, int] = Field(default_factory=dict)
    critical_question_ids: list[str] = Field(default_factory=list)
    not_retrieved_question_ids: list[str] = Field(default_factory=list)
    evidence_gap_question_ids: list[str] = Field(default_factory=list)
    no_candidate_question_ids: list[str] = Field(default_factory=list)
    unknown_metrics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StrategyDiagnostics(BaseModel):
    """One retrieval strategy's full diagnostic result.

    ``status`` is explicit: a strategy that could not run is recorded as FAILED
    with its error, and NEVER replaced by another strategy's numbers.
    """

    strategy: str
    status: str = Field(
        default="completed",
        description="completed | failed | skipped — never silently substituted",
    )
    failure_reason: str = ""
    top_k: int = Field(
        default=0, description="the EVALUATED cut-off: every metric and every verdict uses k"
    )
    candidate_depth: int = Field(
        default=0,
        description="how deep the candidate list was probed. When this exceeds top_k, a "
        "required chunk found beyond k is LOW_RECALL (recoverable by raising k) rather "
        "than NO_RELEVANT_EVIDENCE (absent from the candidate set).",
    )
    retrieval_params: dict[str, Any] = Field(default_factory=dict)
    config_fingerprint: str = ""
    embedding_model: str = ""
    retrieval_backend: str = ""
    reranker_status: str = ""
    corpus_version: str | None = None
    corpus_fingerprint: str | None = None
    index_points: int | None = Field(
        default=None, description="Point count read from the vector index before the run"
    )
    notes: list[str] = Field(default_factory=list)
    questions: list[QuestionRetrievalDiagnostic] = Field(default_factory=list)
    summary: RetrievalDiagnosticSummary = Field(default_factory=RetrievalDiagnosticSummary)
    duration_ms: float | None = None

    def by_code(self, code: RetrievalFailureCode) -> list[QuestionRetrievalDiagnostic]:
        return [q for q in self.questions if q.classification.code is code]

    def question(self, question_id: str) -> QuestionRetrievalDiagnostic | None:
        return next((q for q in self.questions if q.question_id == question_id), None)

    def protocol(self, *, benchmark_name: str, benchmark_fingerprint: str) -> ExperimentProtocol:
        """The comparison protocol for this strategy.

        Retrieval configuration is deliberately NOT part of the identity: it is
        the variable under test (see `comparison.check_protocol`). Everything
        else the two sides must share is recorded from real values.

        The EMBEDDING model is part of that configurable variable, not of the
        protocol: a BM25 strategy has no embedder at all, and putting it in the
        `model` identity field made every dense-vs-bm25 comparison report
        NOT_COMPARABLE — a comparison of two retrieval strategies on the same
        corpus and benchmark is precisely what this is for. `model` therefore
        stays empty (no generator ran) and the embedder rides with the strategy's
        recorded retrieval parameters.
        """
        params = dict(self.retrieval_params)
        params.setdefault("embedding_model", self.embedding_model)
        params.setdefault("reranker_status", self.reranker_status)
        return ExperimentProtocol(
            label=self.strategy,
            benchmark_name=benchmark_name,
            benchmark_fingerprint=benchmark_fingerprint,
            corpus_fingerprint=self.corpus_fingerprint or "",
            evaluator_name="retrieval-diagnostics",
            evaluator_version="v12.0",
            generator="",
            model="",
            answer_mode="",
            kb_id="",
            retrieval_params=params,
            config_fingerprint=self.config_fingerprint,
        )

    def metric_series(self, metric: str) -> dict[str, float | None]:
        """{question_id: measured value or None} for one metric, across questions."""
        out: dict[str, float | None] = {}
        for q in self.questions:
            metric_obj = getattr(q.evidence, metric, None)
            if isinstance(metric_obj, DiagnosticMetric):
                out[q.question_id] = metric_obj.value if metric_obj.measured else None
            elif metric == "first_relevant_rank":
                out[q.question_id] = (
                    float(q.evidence.first_relevant_rank)
                    if q.evidence.first_relevant_rank is not None
                    else None
                )
        return out


class RetrievalIntegritySnapshot(BaseModel):
    """Read-only evidence that the run did not touch the corpus or the benchmark.

    Captured BEFORE and AFTER the run so the claim "nothing was mutated" is a
    measurement rather than a promise.
    """

    kb_id: str = ""
    index_points_before: int | None = None
    index_points_after: int | None = None
    chunk_rows_before: int | None = None
    chunk_rows_after: int | None = None
    corpus_revision_before: int | None = None
    corpus_revision_after: int | None = None
    benchmark_sha256_before: str = ""
    benchmark_sha256_after: str = ""
    retrieval_runs_persisted: int = Field(
        default=0,
        description="Retrieval-run rows the diagnostic wrote. 0 = the run left no "
        "retrieval history behind (a diagnostic probe is not a user query).",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def protected_kb_unchanged(self) -> bool | None:
        """None when a count could not be read — UNKNOWN, not 'unchanged'."""
        pairs = (
            (self.index_points_before, self.index_points_after),
            (self.chunk_rows_before, self.chunk_rows_after),
        )
        for before, after in pairs:
            if before is None or after is None:
                return None
            if before != after:
                return False
        return True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def benchmark_unchanged(self) -> bool | None:
        if not (self.benchmark_sha256_before and self.benchmark_sha256_after):
            return None
        return self.benchmark_sha256_before == self.benchmark_sha256_after

    def observations(self) -> list[str]:
        out: list[str] = []
        unchanged = self.protected_kb_unchanged
        if unchanged is None:
            out.append(
                "corpus/index counts could not be read before and after the run, so "
                "'no mutation' is UNVERIFIED rather than confirmed"
            )
        else:
            out.append(
                "corpus/index counts identical before and after "
                f"(index points {self.index_points_before} -> {self.index_points_after}, "
                f"chunk rows {self.chunk_rows_before} -> {self.chunk_rows_after})"
            )
        benchmark = self.benchmark_unchanged
        if benchmark is None:
            out.append("the benchmark file hash could not be read on both sides")
        else:
            out.append(
                "benchmark artifact byte-identical before and after "
                f"(sha256 {self.benchmark_sha256_before[:16]}...)"
                if benchmark
                else "BENCHMARK ARTIFACT CHANGED during the run — RESULTS ARE VOID"
            )
        out.append(
            f"{self.retrieval_runs_persisted} retrieval-run row(s) were written by this "
            "diagnostic (0 expected: probing a KB is not a user query)"
        )
        return out


class RetrievalDiagnosticsReport(BaseModel):
    """A complete, persisted retrieval-diagnostics run."""

    id: str
    kb_id: str
    kb_name: str = ""
    created_at: str = ""
    benchmark_name: str = ""
    benchmark_version: int = 0
    benchmark_path: str = ""
    benchmark_fingerprint: str = ""
    relevance_model: str = ""
    benchmark_authorship: dict[str, Any] = Field(default_factory=dict)

    top_k: int = 0
    question_count: int = 0
    matched_question_ids: list[str] = Field(default_factory=list)

    strategies: list[StrategyDiagnostics] = Field(default_factory=list)
    comparisons: list[ExperimentComparison] = Field(default_factory=list)
    recommendations: RecommendationReport = Field(default_factory=RecommendationReport)
    integrity: RetrievalIntegritySnapshot = Field(default_factory=RetrievalIntegritySnapshot)
    limitations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def strategy(self, name: str) -> StrategyDiagnostics | None:
        return next((s for s in self.strategies if s.strategy == name), None)

    def critical_question_ids(self) -> list[str]:
        ids: set[str] = set()
        for s in self.strategies:
            ids.update(q.question_id for q in s.questions if q.classification.is_critical)
        return sorted(ids)


# ---------------------------------------------------------------------------
# Evidence availability
# ---------------------------------------------------------------------------


def rollup_availability(items: list[EvidenceAvailability]) -> AvailabilityRollup:
    """Collapse per-chunk availability to one state for a question or a run.

    Order matters: a corpus gap outranks a missing vector (report the more
    fundamental problem), and both outrank 'unverified' so a real defect is
    never hidden behind a check that could not run.
    """
    statuses = {a.status for a in items}
    if not items:
        return AvailabilityRollup.UNKNOWN
    if EvidenceAvailabilityStatus.MISSING_FROM_CORPUS in statuses:
        return AvailabilityRollup.MISSING_FROM_CORPUS
    if EvidenceAvailabilityStatus.MISSING_FROM_INDEX in statuses:
        return AvailabilityRollup.MISSING_FROM_INDEX
    if statuses == {EvidenceAvailabilityStatus.INDEXED}:
        return AvailabilityRollup.ALL_INDEXED
    if EvidenceAvailabilityStatus.CORPUS_ONLY in statuses:
        return AvailabilityRollup.UNVERIFIED_INDEX
    return AvailabilityRollup.UNKNOWN


def assess_evidence_availability(
    expected: dict[str, list[str]],
    *,
    chunk_rows: dict[str, Any],
    index_chunk_ids: set[str] | None,
    document_ids: set[str] | None = None,
    availability_is_verified: bool = True,
    multi_question_chunk_ids: set[str] | None = None,
) -> tuple[list[EvidenceAvailability], AvailabilityRollup, list[str]]:
    """Check each expected chunk against the corpus and the index.

    ``index_chunk_ids`` is None when the index could not be probed; that is
    recorded as CORPUS_ONLY / UNKNOWN, never as "present" and never as "missing".

    Returns (per-chunk availability, rollup, question-level warnings).
    """
    out: list[EvidenceAvailability] = []
    warnings: list[str] = []
    for chunk_id, expected_docs in sorted(expected.items()):
        row = chunk_rows.get(chunk_id)
        in_corpus = row is not None
        notes: list[str] = []

        if index_chunk_ids is None:
            in_index: bool | None = None
            if not availability_is_verified:
                notes.append("the vector index was not probed for this run")
        else:
            in_index = chunk_id in index_chunk_ids

        if in_corpus and in_index:
            status = EvidenceAvailabilityStatus.INDEXED
        elif in_corpus and in_index is None:
            status = EvidenceAvailabilityStatus.CORPUS_ONLY
        elif in_corpus and in_index is False:
            status = EvidenceAvailabilityStatus.MISSING_FROM_INDEX
            notes.append(
                "the chunk exists in the corpus but has no vector, so no retriever "
                "can return it: an index-integrity problem, not a ranking problem"
            )
        elif not in_corpus and in_index is True:
            # A vector with no chunk row cannot be returned with valid provenance.
            status = EvidenceAvailabilityStatus.MISSING_FROM_CORPUS
            notes.append(
                "a vector exists for this id but the chunk metadata row does not, so "
                "the evidence is not usable with provenance"
            )
        elif not in_corpus and in_index is None:
            status = (
                EvidenceAvailabilityStatus.MISSING_FROM_CORPUS
                if availability_is_verified
                else EvidenceAvailabilityStatus.UNKNOWN
            )
        elif not in_corpus:
            status = EvidenceAvailabilityStatus.MISSING_FROM_CORPUS
        else:  # pragma: no cover - defensive
            status = EvidenceAvailabilityStatus.UNKNOWN

        actual_doc = getattr(row, "document_id", None) if in_corpus else None
        doc_check: bool | None = None
        if expected_docs:
            if actual_doc is None:
                doc_check = False if availability_is_verified else None
            else:
                doc_check = actual_doc in expected_docs
                if not doc_check:
                    notes.append(
                        f"the chunk's document is {actual_doc}, which is not one of the "
                        f"benchmark's expected documents ({', '.join(sorted(expected_docs))}) "
                        "— document-level reference labels for this question do not "
                        "describe where the answering text lives"
                    )
        if in_corpus and document_ids is not None and actual_doc and actual_doc not in document_ids:
            notes.append(
                f"the chunk's document {actual_doc} is not in this knowledge base"
            )

        out.append(
            EvidenceAvailability(
                chunk_id=chunk_id,
                expected_document_ids=sorted(expected_docs),
                in_corpus=in_corpus,
                in_index=in_index,
                status=status,
                actual_document_id=actual_doc,
                document_matches_benchmark=doc_check,
                section=getattr(row, "section", None) if in_corpus else None,
                source_title=getattr(row, "source_title", None) if in_corpus else None,
                content_hash=getattr(row, "content_hash", None) if in_corpus else None,
                notes=notes,
            )
        )

    rollup = rollup_availability(out)

    if multi_question_chunk_ids:
        shared = sorted({a.chunk_id for a in out} & multi_question_chunk_ids)
        if shared:
            warnings.append(
                f"{len(shared)} required chunk(s) are shared with other questions "
                f"({', '.join(shared)}), so a single ranking error can appear as several "
                "question failures at once"
            )
    return out, rollup, warnings


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_retrieval_question(
    *,
    question_id: str,
    diagnostic: QuestionDiagnostic,
    availability: list[EvidenceAvailability],
    rollup: AvailabilityRollup,
    top_k: int,
    probe: RetrievalProbe | None = None,
) -> RetrievalClassification:
    """Deterministically classify ONE question's retrieval outcome.

    Reads the windowed metrics of record from `diagnostic` and the deeper
    `probe` for the depth questions. Every branch states the observation it
    fired on. The probe is what separates "retrieved below your cut-off" from
    "never retrieved at all"; without it those two collapse into one verdict.
    """
    if probe is None:
        # No probe supplied: fall back to the window itself, and SAY so rather than
        # pretending the candidate set was searched deeper than it was.
        probe = build_probe(
            [item.chunk_id for item in diagnostic.retrieved],
            required_chunk_ids=list(diagnostic.required_chunk_ids),
            top_k=top_k,
            candidate_depth=top_k,
        )
    limitations: list[str] = []
    if probe.candidate_depth <= top_k:
        limitations.append(
            "the candidate window was not widened beyond the evaluated k, so a "
            "passage ranked below k cannot be distinguished from one that was never "
            "retrieved: NO_RELEVANT_EVIDENCE here means absent from the k results the "
            "retriever returned"
        )
    index_verified = rollup in (
        AvailabilityRollup.ALL_INDEXED,
        AvailabilityRollup.MISSING_FROM_INDEX,
        AvailabilityRollup.MISSING_FROM_CORPUS,
    )
    if rollup is AvailabilityRollup.UNVERIFIED_INDEX:
        limitations.append(
            "the vector index was not probed, so a missing vector cannot be ruled "
            "out as the reason a required chunk never came back"
        )
    if rollup is AvailabilityRollup.UNKNOWN and availability:
        limitations.append("neither the corpus nor the index could be checked for these chunks")

    # 1. No reference evidence -> UNKNOWN. Not a failure.
    if not diagnostic.ground_truth_available:
        return RetrievalClassification(
            code=RetrievalFailureCode.UNKNOWN,
            rule=RULE_NO_GROUND_TRUTH,
            confidence=FailureConfidence.UNKNOWN,
            evidence=["the benchmark names no required evidence for this question"],
            reason=(
                "with no reference evidence there is nothing to compare retrieval "
                "against: this is unknown, not a pass and not a failure"
            ),
            limitations=limitations,
        )

    required = list(diagnostic.required_chunk_ids)
    retrieved = [item.chunk_id for item in diagnostic.retrieved]
    recall = diagnostic.recall_at_k.value if diagnostic.recall_at_k.measured else None
    first_rank = diagnostic.first_relevant_rank

    # 2. Evidence absent from the knowledge base entirely.
    absent = [
        a for a in availability if a.status is EvidenceAvailabilityStatus.MISSING_FROM_CORPUS
    ]
    docs_absent = [
        a
        for a in availability
        if a.document_matches_benchmark is False and a.status
        is EvidenceAvailabilityStatus.MISSING_FROM_CORPUS
    ]
    if absent or docs_absent:
        return RetrievalClassification(
            code=RetrievalFailureCode.CORPUS_OR_SOURCE_DEFICIENCY,
            rule=RULE_EVIDENCE_ABSENT_FROM_KB,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                (
                    f"expected chunk {a.chunk_id} has no chunk row in this knowledge "
                    f"base (benchmark expects document(s) "
                    f"{', '.join(a.expected_document_ids) or 'n/a'})"
                )
                for a in absent
            ]
            or [
                (
                    f"expected document(s) "
                    f"{', '.join(sorted({d for a in docs_absent for d in a.expected_document_ids}))} "
                    "are not in this knowledge base"
                )
            ],
            reason=(
                "the benchmark cites evidence this knowledge base does not contain, so no "
                "retriever and no configuration change can surface it. The remedy is corpus "
                "work: ingest a source that covers it, or re-label the question"
            ),
            limitations=limitations,
        )

    # 3. In the corpus but not in the index — unretrievable by construction.
    unindexed = [
        a for a in availability if a.status is EvidenceAvailabilityStatus.MISSING_FROM_INDEX
    ]
    if unindexed:
        return RetrievalClassification(
            code=RetrievalFailureCode.EXPECTED_EVIDENCE_UNAVAILABLE,
            rule=RULE_EVIDENCE_NOT_INDEXED,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"required chunk {a.chunk_id} exists in the corpus (document "
                f"{a.actual_document_id}) but has no vector in the index"
                for a in unindexed
            ],
            reason=(
                "the required passage is indexed metadata without a vector, so it is "
                "invisible to every retriever. This is index integrity, not retrieval quality"
            ),
            limitations=limitations,
        )

    # 4. Retrieval returned nothing at all. Read from the PROBE, because an empty
    #    window with a non-empty candidate set would merely be a tiny k.
    probe_ids = probe.candidate_chunk_ids or retrieved
    if not probe_ids:
        return RetrievalClassification(
            code=RetrievalFailureCode.RETRIEVAL_FAILURE,
            rule=RULE_NO_CANDIDATES,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                "retrieval returned 0 candidate chunks",
                f"{len(required)} required chunk(s) were never candidates",
            ],
            reason=(
                "the retriever produced no candidates at all, which is a pipeline or "
                "supply failure rather than a ranking problem"
            ),
            limitations=limitations,
        )

    # 5. Not present anywhere in the probed candidate set.
    absent = probe.absent_from_probe
    if absent and len(absent) == len(required):
        probe_note = (
            f"the probe was truncated at {len(probe_ids)} < {probe.candidate_depth} "
            "candidates, so absence is reported for the depth actually searched"
            if probe.truncated
            else f"the candidate set was probed to depth {probe.candidate_depth}"
        )
        return RetrievalClassification(
            code=RetrievalFailureCode.NO_RELEVANT_EVIDENCE,
            rule=RULE_NO_REQUIRED_IN_CANDIDATES,
            confidence=(
                FailureConfidence.DETERMINISTIC if index_verified else FailureConfidence.PARTIAL
            ),
            evidence=[
                f"none of the {len(required)} required chunk(s) appear anywhere in the "
                f"{len(probe_ids)} candidate(s) returned",
                f"required_recall_in_probe={probe.required_recall_in_probe} — absent from "
                "the whole probed candidate set, not merely from the window",
                probe_note,
            ],
            reason=(
                "the required evidence is available in the corpus and index but never "
                "entered the candidate set at any probed depth, so this is a retrieval "
                "recall or chunk representation problem for this query — not a corpus gap"
            ),
            limitations=limitations,
        )

    # 6. Present in the candidate set, but outside the evaluated window.
    beyond = dict(probe.required_beyond_cutoff)
    if beyond or (recall is not None and recall < 1.0):
        probe_note = (
            f"the candidate set was probed to depth {probe.candidate_depth} "
            f"(collection returned {len(probe_ids)})"
            if probe.candidate_depth
            else "candidate depth was not widened beyond the window"
        )
        return RetrievalClassification(
            code=RetrievalFailureCode.LOW_RECALL,
            rule=RULE_BEYOND_CUTOFF,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"recall@{top_k}={recall} — {len(beyond)} required chunk(s) were "
                f"retrieved beyond rank {top_k}",
                f"rank of the missed required chunk(s): {beyond or 'outside the window'}",
                probe_note,
            ],
            reason=(
                "the required evidence WAS retrieved, just deeper than the evaluated "
                f"cut-off of {top_k}, so a larger window would recover it. This is a "
                "depth problem, not a coverage problem"
            ),
            limitations=limitations,
        )

    # 7. Inside the window but outranked.
    if first_rank is not None and first_rank > SUCCESS_RANK:
        above = [item.chunk_id for item in diagnostic.retrieved if item.rank < first_rank]
        return RetrievalClassification(
            code=RetrievalFailureCode.RANKING_FAILURE,
            rule=RULE_OUTRANKED,
            confidence=FailureConfidence.DETERMINISTIC,
            evidence=[
                f"the first required chunk is at rank {first_rank} (success requires "
                f"rank {SUCCESS_RANK})",
                f"{len(above)} non-required chunk(s) ranked above it: {', '.join(above)}",
                f"recall@{top_k}={recall} — the evidence is in the window, only its "
                "position is wrong",
            ],
            reason=(
                "the required evidence is inside the evaluated window but a chunk the "
                "benchmark does not consider necessary outranked it, so the generator "
                "reads the wrong passage first"
            ),
            limitations=limitations,
        )

    # 8. Success: required evidence retrieved, at rank 1.
    return RetrievalClassification(
        code=RetrievalFailureCode.RETRIEVAL_SUCCESS,
        rule=RULE_TOP_RANKED,
        confidence=FailureConfidence.DETERMINISTIC,
        evidence=[
            f"required chunk(s) {', '.join(required)} retrieved",
            f"first required chunk at rank {first_rank} of {len(retrieved)} returned "
            f"within top-{top_k}",
            f"recall@{top_k}={recall}",
        ],
        reason="retrieval surfaced the required evidence at the top of the ranked list",
        limitations=limitations
        + [
            "precision@k over these labels is a LOWER BOUND: the benchmark labels "
            "NECESSARY evidence, not all relevant evidence, so additional retrieved "
            "context is not a defect"
        ],
    )


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarise_strategy(diagnostics: list[QuestionRetrievalDiagnostic]) -> RetrievalDiagnosticSummary:
    """Aggregate one strategy: V9's measured-only means + V12's code counts."""
    summary = RetrievalDiagnosticSummary(question_count=len(diagnostics))
    if not diagnostics:
        summary.warnings.append("no question diagnostics were produced for this strategy")
        return summary

    base = summarise_diagnostics([d.evidence for d in diagnostics])
    summary.ground_truth_question_count = base.ground_truth_question_count
    summary.no_ground_truth_question_count = base.no_ground_truth_question_count
    summary.recall_at_k = base.recall_at_k
    summary.precision_at_k = base.precision_at_k
    summary.mrr = base.mrr
    summary.ndcg_at_k = base.ndcg_at_k
    summary.required_recall_at_depth = base.required_recall_at_depth
    summary.first_relevant_rank = base.first_relevant_rank
    summary.latency_ms = base.latency_ms
    summary.first_relevant_rank_histogram = base.first_relevant_rank_histogram
    summary.unknown_metrics = list(base.unknown_metrics)
    summary.warnings.extend(base.warnings)

    probe_values = [
        d.probe.required_recall_in_probe
        for d in diagnostics
        if d.probe.required_recall_in_probe is not None
    ]
    depths = {d.probe.candidate_depth for d in diagnostics if d.probe.candidate_depth}
    if probe_values:
        summary.required_recall_in_probe = DiagnosticMetric.of(
            sum(probe_values) / len(probe_values),
            reason=(
                "required evidence found anywhere in the candidate probe "
                f"(depth {max(depths) if depths else 'unknown'}), as opposed to "
                "inside the evaluated window"
            ),
            sample_size=len(probe_values),
        )
    else:
        summary.required_recall_in_probe = DiagnosticMetric.unknown(
            "no question had a probe with reference evidence"
        )
    if not summary.required_recall_in_probe.measured:
        summary.unknown_metrics = [
            *summary.unknown_metrics, "required_recall_in_probe"
        ]

    code_counts: dict[str, int] = {}
    availability_counts: dict[str, int] = {}
    for d in diagnostics:
        code = d.classification.code.value
        code_counts[code] = code_counts.get(code, 0) + 1
        availability_counts[d.availability_rollup.value] = (
            availability_counts.get(d.availability_rollup.value, 0) + 1
        )
    summary.code_counts = dict(sorted(code_counts.items()))
    summary.availability_counts = dict(sorted(availability_counts.items()))

    summary.evaluated_question_count = sum(
        1 for d in diagnostics if d.evidence.ground_truth_available
    )
    summary.critical_question_ids = sorted(
        d.question_id for d in diagnostics if d.classification.is_critical
    )
    summary.not_retrieved_question_ids = sorted(
        d.question_id
        for d in diagnostics
        if d.evidence.ground_truth_available
        and d.probe.required_recall_in_probe == 0.0
    )
    summary.evidence_gap_question_ids = sorted(
        d.question_id
        for d in diagnostics
        if d.availability_rollup
        in (AvailabilityRollup.MISSING_FROM_CORPUS, AvailabilityRollup.MISSING_FROM_INDEX)
    )
    summary.no_candidate_question_ids = sorted(
        d.question_id for d in diagnostics if not d.evidence.retrieved
    )
    # Reported once per summary so the reader is never left wondering why a
    # published code never appears in the counts above.
    summary.warnings.append(
        "LOW_PRECISION is part of the published vocabulary but is never emitted: binary "
        "relevance over necessity-only labels makes precision@k a lower bound, so a low "
        "value is not evidence of a retrieval fault. It is still reported as a metric."
    )
    return summary


# ---------------------------------------------------------------------------
# Strategy comparison (matched questions, paired, protocol-gated)
# ---------------------------------------------------------------------------


def compare_strategies(
    baseline: StrategyDiagnostics,
    candidate: StrategyDiagnostics,
    *,
    benchmark_name: str = "",
    benchmark_fingerprint: str = "",
    tie_epsilon: float = 0.0,
) -> ExperimentComparison:
    """Compare two strategies on the questions BOTH actually measured.

    Reuses V9's `compare_suite_explicit`, so the pairing rules, metric
    directions, effect sizes and significance tests are the ones the project
    already publishes. Questions only one side measured are reported as unpaired
    and never silently dropped.
    """
    comparison = compare_suite_explicit(
        baseline_label=baseline.strategy,
        candidate_label=candidate.strategy,
        baseline=baseline.protocol(
            benchmark_name=benchmark_name, benchmark_fingerprint=benchmark_fingerprint
        ),
        candidate=candidate.protocol(
            benchmark_name=benchmark_name, benchmark_fingerprint=benchmark_fingerprint
        ),
        baseline_metrics={m: baseline.metric_series(m) for m in COMPARABLE_METRICS},
        candidate_metrics={m: candidate.metric_series(m) for m in COMPARABLE_METRICS},
        tie_epsilon=tie_epsilon,
        benchmark_fingerprint=benchmark_fingerprint,
    )
    if baseline.status != "completed" or candidate.status != "completed":
        comparison.notes.append(
            "at least one side did not complete "
            f"({baseline.strategy}={baseline.status}, {candidate.strategy}={candidate.status}): "
            "any delta is reported for completeness but is NOT a strategy comparison"
        )
        comparison.exploratory = True
    n_pairs = max((c.paired_count for c in comparison.metric_comparisons), default=0)
    if n_pairs < MIN_PAIRS_FOR_INFERENCE:
        comparison.notes.append(
            f"largest paired sample is {n_pairs} question(s), below the "
            f"{MIN_PAIRS_FOR_INFERENCE}-pair floor for inference: deltas are reported as "
            "measured directions only"
        )
    if benchmark_name:
        comparison.notes.append(
            f"paired on the SAME {n_pairs} question(s) of benchmark {benchmark_name!r}: "
            "only questions BOTH strategies measured are paired, and the rest are "
            "excluded and listed as unpaired"
        )
    return comparison


def comparison_summary(comparison: ExperimentComparison) -> str:
    """One-line, non-misleading summary of a comparison."""
    if comparison.pairing.value != "COMPARABLE" and comparison.pairing.value != "INCONCLUSIVE":
        return f"NOT COMPARABLE: {comparison.pairing_reason}"
    improved = comparison.improved_metrics()
    regressed = comparison.regressed_metrics()
    if not improved and not regressed:
        return "no metric moved measurably under this comparison"
    parts = []
    if improved:
        parts.append(f"improved on {', '.join(improved)}")
    if regressed:
        parts.append(f"regressed on {', '.join(regressed)}")
    return f"{comparison.candidate_label} vs {comparison.baseline_label}: " + "; ".join(parts) + (
        " (exploratory: paired sample below the inference floor)" if comparison.exploratory else ""
    )


def critical_questions_by_code(strategy: StrategyDiagnostics) -> dict[str, list[str]]:
    """Question ids grouped by failure code — the UI's primary breakdown."""
    out: dict[str, list[str]] = {}
    for q in strategy.questions:
        if not q.classification.is_critical:
            continue
        out.setdefault(q.classification.code.value, []).append(q.question_id)
    return {code: sorted(ids) for code, ids in sorted(out.items())}


# ---------------------------------------------------------------------------
# Recommendations (V12 refinement of the V9 rules)
# ---------------------------------------------------------------------------

#: Rules the V12 refinement owns, mapped to the action they justify.
_CODE_ACTIONS: dict[RetrievalFailureCode, RecommendationAction] = {
    RetrievalFailureCode.CORPUS_OR_SOURCE_DEFICIENCY: RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY,
    RetrievalFailureCode.EXPECTED_EVIDENCE_UNAVAILABLE: RecommendationAction.RESTORE_INDEX_INTEGRITY,
    RetrievalFailureCode.RETRIEVAL_FAILURE: RecommendationAction.INVESTIGATE_RETRIEVAL_PIPELINE,
    RetrievalFailureCode.NO_RELEVANT_EVIDENCE: RecommendationAction.INVESTIGATE_RETRIEVAL,
    RetrievalFailureCode.LOW_RECALL: RecommendationAction.INCREASE_TOP_K,
    RetrievalFailureCode.RANKING_FAILURE: RecommendationAction.ENABLE_RERANKING,
}

#: What a validation experiment for each action must hold fixed.
_VALIDATION_EXPERIMENT: dict[RecommendationAction, str] = {
    RecommendationAction.INCREASE_TOP_K: (
        "re-run the identical benchmark on the identical corpus with only top_k raised "
        "to the deepest observed required rank, then re-diagnose: LOW_RECALL should "
        "convert to RANKING_FAILURE or RETRIEVAL_SUCCESS, with precision reported"
    ),
    RecommendationAction.ENABLE_RERANKING: (
        "re-run with the same candidate set and a reranker, changing nothing else; the "
        "hypothesis is that the required chunk moves to rank 1 for these questions, and "
        "the paired MRR/NDCG delta decides"
    ),
    RecommendationAction.INVESTIGATE_RETRIEVAL: (
        "compare the failing query against the queries that succeeded under the same "
        "strategy (query length, rare terms, chunk length), then re-run that question "
        "set under a second strategy to see whether the failure is vocabulary-dependent"
    ),
    RecommendationAction.RESTORE_INDEX_INTEGRITY: (
        "re-index the affected document and confirm the chunk gains a vector, then "
        "re-diagnose: EXPECTED_EVIDENCE_UNAVAILABLE must disappear without any "
        "retrieval-configuration change"
    ),
    RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY: (
        "this is not a retriever hypothesis: add a source that covers the missing "
        "topic (or record the question as unanswerable) and re-diagnose"
    ),
    RecommendationAction.INVESTIGATE_RETRIEVAL_PIPELINE: (
        "inspect the run's stages and candidate counts: a zero-candidate result points at "
        "the embedding/query path or an empty index, not at ranking"
    ),
}


def recommendations_from_strategy(
    strategy: StrategyDiagnostics,
    *,
    base_report: RecommendationReport | None = None,
    benchmark_name: str = "",
) -> RecommendationReport:
    """V12 recommendations from classification + availability, plus the V9 baseline.

    `base_report` is the output of V9's `analyse_retrieval` for the same questions.
    When V12 has MEASURED that V9's coarse "corpus deficiency" advice is wrong —
    the evidence is present in the corpus and index, it simply was never retrieved
    — that recommendation is replaced by the more precise one, and the replacement
    says why. Nothing is removed from the report without a measured reason.
    """
    report = RecommendationReport()
    report.total_questions = len(strategy.questions)
    report.questions_with_ground_truth = strategy.summary.ground_truth_question_count
    report.questions_analysed = strategy.summary.evaluated_question_count

    if base_report is not None:
        report.recommendations.extend(base_report.recommendations)
        report.data_gaps.extend(base_report.data_gaps)
        report.notes.extend(base_report.notes)
        report.unknown_failure_count = base_report.unknown_failure_count

    critical = [q for q in strategy.questions if q.classification.is_critical]
    total_grounded = max(1, strategy.summary.ground_truth_question_count)

    # V9's corpus-deficiency rule fires whenever required evidence is missing from
    # the WINDOW it was shown. V12 can now MEASURE whether that evidence exists at
    # all: either the deep probe found it (so it is a depth/ranking problem) or the
    # corpus and index both carry it (so absence from the candidate set is a
    # retrieval problem, not missing content). When that is the case, V9's advice is
    # not merely coarse — its text asserts a cause the measurement contradicts, so
    # it is REPLACED, and the replacement carries the measurement.
    evidence_found = [
        q
        for q in strategy.questions
        if q.classification.is_critical
        and (
            q.probe.required_ranks_in_probe
            or q.availability_rollup
            in (AvailabilityRollup.ALL_INDEXED, AvailabilityRollup.UNVERIFIED_INDEX)
        )
    ]
    dropped_v9_corpus = [
        r
        for r in report.recommendations
        if r.action is RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY
    ]
    if evidence_found and dropped_v9_corpus:
        report.recommendations = [
            r
            for r in report.recommendations
            if r.action is not RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY
        ]
        probe_examples = {
            q.question_id: q.probe.required_ranks_in_probe for q in evidence_found[:5]
        }
        report.notes.append(
            "the V9 corpus-deficiency recommendation was REPLACED: V12 measured that the "
            f"required evidence is present for {len(evidence_found)} of the failing "
            "question(s) — found by the deep probe at the ranks recorded below, and/or "
            "present in the corpus AND the index — so the failure is a retrieval, depth "
            "or ranking problem rather than missing content. V9 could only observe that "
            f"the passage was absent from its window. Probe ranks: {probe_examples}"
        )

    # V9's vocabulary-dependent rule fires on "required evidence not retrieved" and
    # its text says the questions "missed their required evidence entirely". When
    # the probe proves the evidence WAS retrieved, that sentence is false for these
    # questions, so the recommendation goes and V12's depth/ranking advice (which
    # carries the ranks) is what remains.
    # Only the PROBE refutes "missed entirely": a passage that is indexed but never
    # retrieved by this strategy really was missed by it, which is what V9's hybrid
    # advice addresses.
    probe_found = [q for q in strategy.questions if q.probe.required_ranks_in_probe]
    if probe_found:
        v9_hybrid = [
            r
            for r in report.recommendations
            if r.action is RecommendationAction.ENABLE_HYBRID_FUSION
        ]
        if v9_hybrid:
            report.recommendations = [
                r
                for r in report.recommendations
                if r.action is not RecommendationAction.ENABLE_HYBRID_FUSION
            ]
            report.notes.append(
                "a V9 hybrid-fusion recommendation was REPLACED as well: it fired on "
                "questions whose required evidence V12's probe actually retrieved, so its "
                "'missed entirely' premise does not hold for them. Changing the fusion "
                "strategy remains a legitimate experiment for SINGLE-STRATEGY recall misses "
                "(which V12 reports as NO_RELEVANT_EVIDENCE with required_recall_in_probe=0), "
                "and was not proposed here because V12 measured none"
            )

    # Availability findings always come first: they invalidate retrieval tuning.
    availability_actions: dict[RecommendationAction, list[QuestionRetrievalDiagnostic]] = {}
    for q in critical:
        action = _CODE_ACTIONS.get(q.classification.code)
        if action is None:
            continue
        availability_actions.setdefault(action, []).append(q)

    ordered_actions = [
        RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY,
        RecommendationAction.RESTORE_INDEX_INTEGRITY,
        RecommendationAction.INVESTIGATE_RETRIEVAL_PIPELINE,
        RecommendationAction.INVESTIGATE_RETRIEVAL,
        RecommendationAction.INCREASE_TOP_K,
        RecommendationAction.ENABLE_RERANKING,
    ]
    emitted_actions: set[RecommendationAction] = set()
    for action in ordered_actions:
        questions = availability_actions.get(action)
        if not questions:
            continue
        emitted_actions.add(action)
        codes = sorted({q.classification.code.value for q in questions})
        supporting: dict[str, object] = {
            "strategy": strategy.strategy,
            "top_k": strategy.top_k,
            "questions_affected": len(questions),
            "questions_with_ground_truth": strategy.summary.ground_truth_question_count,
            "codes": codes,
            "affected_question_ids": sorted(q.question_id for q in questions),
        }
        if action is RecommendationAction.INCREASE_TOP_K:
            # The rank that justifies a bigger window is the rank the PROBE found,
            # not a rank inside the window that already contained it.
            beyond = sorted(
                {rank for q in questions for rank in q.probe.required_beyond_cutoff.values()}
            )
            in_window = sorted(
                {
                    item.rank
                    for q in questions
                    for item in q.evidence.retrieved
                    if item.chunk_id in set(q.expected_chunk_ids)
                }
            )
            supporting["observed_required_ranks"] = beyond or in_window
            supporting["observed_required_ranks_beyond_cutoff"] = beyond
            supporting["observed_required_ranks_in_window"] = in_window
            supporting["candidate_depth"] = strategy.candidate_depth
            supporting["current_top_k"] = strategy.top_k
        if action in (
            RecommendationAction.RESTORE_INDEX_INTEGRITY,
            RecommendationAction.CORPUS_OR_SOURCE_DEFICIENCY,
        ):
            supporting["availability"] = {
                q.question_id: {
                    a.chunk_id: a.status.value for a in q.availability
                }
                for q in questions
            }
        if action is RecommendationAction.INVESTIGATE_RETRIEVAL:
            supporting["missing_from_candidates"] = {
                q.question_id: sorted(
                    set(q.expected_chunk_ids) - set(q.retrieved_chunk_ids)
                )
                for q in questions
            }
        if action is RecommendationAction.ENABLE_RERANKING:
            supporting["first_relevant_ranks"] = {
                q.question_id: q.evidence.first_relevant_rank for q in questions
            }

        report.recommendations.append(
            Recommendation(
                action=action,
                reason=(
                    f"{len(questions)} of {strategy.summary.ground_truth_question_count} "
                    f"question(s) with reference evidence were classified {', '.join(codes)} "
                    f"under strategy {strategy.strategy!r} at top_k={strategy.top_k}"
                ),
                supporting_measurements=supporting,
                expected_effect=(
                    "unknown until the validation experiment runs; this is a hypothesis "
                    "derived from the measurements above"
                ),
                risk=(
                    "the recommendation is only as good as the benchmark labels: the "
                    "shipped benchmark's authorship records human review as PENDING, so a "
                    "conclusion drawn from it is not publication-grade"
                ),
                strength=_strength_for(len(questions), total_grounded),
                affected_question_ids=sorted(q.question_id for q in questions),
                proposed_config_change={},
                experiment_required=True,
                thresholds_used={
                    "source": "v12-retrieval-diagnostics",
                    "SUCCESS_RANK": SUCCESS_RANK,
                    "validation_experiment": _VALIDATION_EXPERIMENT.get(action, ""),
                },
            )
        )

    # A V9 recommendation whose ACTION V12 also reached is dropped in favour of the
    # V12 one: V12's version carries the availability and probe measurements dated
    # to this run, and two recommendations for the same action with different
    # evidence invite a reader to treat them as two independent findings.
    if emitted_actions:
        superseded = [
            r
            for r in report.recommendations
            if r.action in emitted_actions
            and (r.thresholds_used or {}).get("source") != "v12-retrieval-diagnostics"
        ]
        if superseded:
            report.recommendations = [
                r
                for r in report.recommendations
                if r not in superseded
            ]
            report.notes.append(
                "V9 recommendation(s) for "
                f"{sorted(a.value for a in emitted_actions)} were superseded by the V12 "
                "version of the same action, which carries this run's availability and "
                "probe measurements"
            )

    if not report.recommendations:
        report.recommendations.append(
            Recommendation(
                action=RecommendationAction.NO_ACTION_JUSTIFIED,
                reason=(
                    "every question with reference evidence was classified RETRIEVAL_SUCCESS "
                    "or UNKNOWN: no measured retrieval failure justifies a change"
                ),
                supporting_measurements={
                    "strategy": strategy.strategy,
                    "questions_with_ground_truth": strategy.summary.ground_truth_question_count,
                    "code_counts": strategy.summary.code_counts,
                },
                expected_effect="none; the current configuration met the measured bar",
                risk="changing configuration anyway would be tuning without evidence",
                strength=RecommendationStrength.STRONG,
            )
        )
    if strategy.summary.no_ground_truth_question_count:
        report.data_gaps.append(
            f"{strategy.summary.no_ground_truth_question_count} question(s) carry no "
            "reference evidence and were classified UNKNOWN rather than counted as failures"
        )
    if strategy.status != "completed":
        report.data_gaps.append(
            f"strategy {strategy.strategy!r} did not complete ({strategy.status}): "
            f"{strategy.failure_reason}"
        )
    if benchmark_name:
        report.notes.append(f"benchmark: {benchmark_name}")
    return report


__all__ = [
    "ALL_RETRIEVAL_CODES",
    "AvailabilityRollup",
    "COMPARABLE_METRICS",
    "CRITICAL_CODES",
    "EvidenceAvailability",
    "EvidenceAvailabilityStatus",
    "MIN_PAIRS_FOR_INFERENCE",
    "QuestionRetrievalDiagnostic",
    "RetrievalClassification",
    "RetrievalDiagnosticSummary",
    "RetrievalDiagnosticsReport",
    "RetrievalFailureCode",
    "RetrievalIntegritySnapshot",
    "SUCCESS_RANK",
    "StrategyDiagnostics",
    "assess_evidence_availability",
    "classify_retrieval_question",
    "rollup_availability",
    "compare_strategies",
    "comparison_summary",
    "critical_questions_by_code",
    "recommendations_from_strategy",
    "summarise_strategy",
]
