"""Answer-evaluation service: orchestrates a full evaluation run.

Flow per question: answer through the REAL `AnsweringService` (so the pipeline
under test is the shipping pipeline, not a simplified one), then score the
resulting answer against the benchmark's required evidence.

Two properties this service guarantees:

* **It never mutates the corpus.** Answering is read-only; retrieval and
  generation write nothing but their own observability rows.
* **It records producers.** The generator, model, evaluator and entailment
  provider travel with the run, so a metric can always be attributed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings, get_settings
from app.repositories.sqlite_repo import Repository
from app.schemas.answer import Answer, Evidence
from app.services.answer_eval.benchmark import (
    AnswerBenchmark,
    AnswerBenchmarkQuestion,
    BenchmarkValidationError,
    require_official_benchmark,
    validate_against_corpus,
)
from app.services.answer_eval.evaluator import AnswerEvaluator, create_answer_evaluator
from app.services.answer_eval.metrics import AnswerQualityResult
from app.services.answer_eval.relevance import (
    TermWeights,
    term_weights_from_lexical_index,
)
from app.services.answer_eval.run import (
    AnswerEvaluationConfig,
    AnswerEvaluationRun,
    aggregate_results,
    build_run,
    select_questions,
)
from app.services.answering.chat import ANSWERER_VERSION
from app.services.answering.service import AnsweringService
from app.utils.ids import new_id

logger = logging.getLogger(__name__)


class AnswerEvaluationError(RuntimeError):
    """Evaluation could not run (HTTP 400/503 by cause)."""


class BenchmarkCorpusMismatch(AnswerEvaluationError):
    """The benchmark references chunks that no longer match the corpus."""


class AnswerEvaluationService:
    def __init__(
        self,
        repo: Repository,
        settings: Settings | None = None,
        *,
        answering: AnsweringService | None = None,
        evaluator: AnswerEvaluator | None = None,
    ) -> None:
        self._repo = repo
        self._settings = settings or get_settings()
        self._answering = answering or AnsweringService(repo, self._settings)
        self._evaluator = evaluator or create_answer_evaluator()

    # -- benchmark loading ---------------------------------------------------

    def load_benchmark(self, path: str) -> AnswerBenchmark:
        from app.services.answer_eval.benchmark import (
            BenchmarkValidationError,
            load_answer_benchmark,
        )

        try:
            return load_answer_benchmark(path)
        except BenchmarkValidationError as exc:
            raise AnswerEvaluationError(str(exc)) from exc

    def _chunk_index(self, kb_id: str) -> dict[str, dict]:
        """{chunk_id: {document_id, content_hash}} from the repository.

        Paginates deliberately: `list_chunks` defaults to 500 and the benchmark
        corpus has >800 chunks, so a single call would silently hide chunks the
        benchmark may reference.
        """
        out: dict[str, dict] = {}
        offset = 0
        page = 500
        while True:
            batch = self._repo.list_chunks(kb_id, limit=page, offset=offset)
            if not batch:
                break
            for chunk in batch:
                out[chunk.id] = {
                    "document_id": chunk.document_id,
                    "content_hash": chunk.content_hash,
                    "kb_id": chunk.kb_id,
                }
            if len(batch) < page:
                break
            offset += page
        return out

    def chunk_index(self, kb_id: str) -> dict[str, dict]:
        """Public access to the paginated chunk index (id -> document/hash/kb).

        Used by the V10 review workflow to validate that a reference answer's
        cited evidence really exists in the corpus; identical to the index the
        evaluator uses, so validation and evaluation cannot disagree.
        """
        return self._chunk_index(kb_id)

    def validate_benchmark(self, benchmark: AnswerBenchmark) -> list[str]:
        """Corpus-consistency problems; empty means safe to run."""
        return validate_against_corpus(benchmark, self._chunk_index(benchmark.kb_id))

    def _relevance_weights(self, kb_id: str) -> TermWeights | None:
        """Corpus IDF statistics for relevance scoring, or None when unavailable.

        Read from the KB's persisted BM25 lexical index — the same document
        frequencies the retriever uses — so relevance weighting is reproducible
        from stored state rather than recomputed. A missing or unreadable index
        is NOT an error: relevance then uses the plain fallback, which is
        explicitly labelled as weaker on every result it produces.
        """
        try:
            row = self._repo.get_lexical_index(kb_id)
            if row is None:
                return None
            _revision, payload_json = row
            payload = json.loads(payload_json)
        except Exception as exc:  # a scoring aid must never break a run
            logger.warning(
                "could not load lexical index for relevance weighting (%s): %s",
                kb_id,
                exc,
            )
            return None
        return term_weights_from_lexical_index(payload)

    # -- the run -------------------------------------------------------------

    def run(
        self,
        *,
        benchmark_path: str,
        config: AnswerEvaluationConfig | None = None,
        benchmark: AnswerBenchmark | None = None,
    ) -> AnswerEvaluationRun:
        """Run an answer-quality evaluation and return an immutable record.

        ``benchmark`` is an optional in-memory override (V10): it lets an
        evaluation score an immutable FROZEN benchmark version loaded from the
        version store without materialising it as a file. The path is still
        recorded for provenance (``db:answer-benchmark-versions/<id>``).
        """
        config = config or AnswerEvaluationConfig(benchmark_path=benchmark_path)
        benchmark = benchmark or self.load_benchmark(benchmark_path)

        if config.official:
            try:
                require_official_benchmark(benchmark)
            except BenchmarkValidationError as exc:
                raise AnswerEvaluationError(str(exc)) from exc

        if self._repo.get_kb(benchmark.kb_id) is None:
            raise AnswerEvaluationError(
                f"benchmark knowledge base {benchmark.kb_id!r} does not exist; "
                f"the benchmark cannot be evaluated against a missing corpus"
            )

        problems = self.validate_benchmark(benchmark)
        if problems:
            raise BenchmarkCorpusMismatch(
                "benchmark does not match the live corpus, so citation scoring "
                "would be invalid: " + "; ".join(problems[:5])
            )

        questions = select_questions(benchmark, config)

        # Relevance needs corpus term statistics; inject them when the
        # evaluator supports it (a custom evaluator may not, which simply
        # leaves the labelled plain fallback in place).
        weights = self._relevance_weights(benchmark.kb_id)
        setter = getattr(self._evaluator, "set_weights", None)
        if callable(setter):
            setter(weights)
        logger.info(
            "answer evaluation relevance weights for %s: %s",
            benchmark.kb_id,
            weights.source if weights and weights.available else "plain fallback (no IDF stats)",
        )

        results: list[AnswerQualityResult] = []
        retrieval_run_ids: list[str] = []
        generator = model = prompt_version = answerer_version = ""
        is_mock = False
        failures: list[str] = []

        for question in questions:
            try:
                answer, trace, _assessment = self._answering.answer(
                    benchmark.kb_id,
                    question.question,
                    mode=config.answer_mode,
                    strategy=config.strategy or None,
                    overrides=(
                        config.retrieval_params.model_dump(mode="json")
                        if config.retrieval_params
                        else None
                    ),
                )
            except Exception as exc:
                # One question failing must not destroy the run; it is recorded.
                failures.append(f"{question.question_id}: {exc}")
                logger.warning("answer evaluation failed for %s: %s", question.question_id, exc)
                continue

            if answer.retrieval_run_id:
                retrieval_run_ids.append(answer.retrieval_run_id)
            evidence: list[Evidence] = list(trace.evidence_items)

            results.append(self._evaluator.evaluate(question, answer, evidence))

            generator = generator or answer.generated_by
            model = model or answer.model
            prompt_version = prompt_version or answer.prompt_version
            is_mock = is_mock or answer.is_mock

        if not results:
            raise AnswerEvaluationError(
                "no question produced an answer; evaluation cannot report metrics"
            )

        run = build_run(
            kb_id=benchmark.kb_id,
            benchmark=benchmark,
            benchmark_path=benchmark_path,
            questions=questions,
            results=results,
            config=config,
            retrieval_run_ids=retrieval_run_ids,
            generator=generator,
            model=model,
            is_mock=is_mock,
            prompt_version=prompt_version,
            answerer_version=answerer_version or ANSWERER_VERSION,
            evaluator=self._evaluator,
            run_id=new_id("aerun"),
        )
        if failures:
            run.warnings.extend(
                [
                    f"{len(failures)} question(s) could not be answered and were "
                    f"EXCLUDED from the metrics — the run does not cover them"
                ]
            )
            run.notes.extend(failures[:20])
        # V10: a frozen, reviewed benchmark may deliberately exclude some
        # questions (ambiguous / insufficient evidence). They are NAMED here so
        # their absence from the aggregate is never mistaken for a zero.
        excluded_by_policy = [
            q.question_id
            for q in questions
            if (q.review_status or "")
            in {"ambiguous", "insufficient_evidence"}
        ]
        if excluded_by_policy:
            shown = ", ".join(excluded_by_policy[:8]) + (
                " …" if len(excluded_by_policy) > 8 else ""
            )
            run.warnings.append(
                f"{len(excluded_by_policy)} question(s) are classified as "
                f"NON-SCORING by the benchmark's approval policy ({shown}); their "
                f"correctness/key-point metrics are UNKNOWN and are excluded "
                f"from the aggregates rather than counted as zero."
            )

        if not any(r.correctness.measured for r in results):
            if any(r.reference_answer_similarity.measured for r in results):
                run.warnings.append(
                    "answer CORRECTNESS is UNKNOWN for this run: human reference "
                    "answers exist and lexical similarity to them is reported as "
                    "reference_answer_similarity; correctness requires human "
                    "review or a model-based judge. Citation, relevance and "
                    "abstention metrics ARE measured."
                )
            else:
                run.warnings.append(
                    "answer CORRECTNESS is UNKNOWN for this run: the benchmark has no "
                    "human-authored reference answers. Citation and abstention "
                    "metrics ARE measured."
                )

        return run

    # -- persistence ---------------------------------------------------------

    def save_run(self, run: AnswerEvaluationRun) -> None:
        self._repo.create_answer_evaluation_run(run)

    def derive_human_run(self, kb_id: str, run_id: str) -> AnswerEvaluationRun:
        """Produce a NEW run whose correctness comes from stored human reviews.

        The review->metric pass is deliberately DERIVATION, not mutation:

        * the source run is untouched (runs are immutable),
        * the reviews are untouched (append-only),
        * the derived run records the source id in its notes, keeps the same
          question set and benchmark fingerprint, so the two remain directly
          comparable (same-subset rule holds by construction),
        * questions WITHOUT a review keep correctness UNKNOWN — a missing
          review is never replaced by a default verdict.
        """
        from app.services.answer_eval.evaluator import (
            EVALUATOR_VERSION,
            apply_reviews,
        )

        source = self.get_run(kb_id, run_id)
        if source is None:
            raise AnswerEvaluationError(
                f"answer evaluation run {run_id!r} not found in knowledge base {kb_id!r}"
            )
        reviews = self._repo.list_answer_reviews(kb_id, run_id=run_id)
        if not reviews:
            raise AnswerEvaluationError(
                f"run {run_id!r} has no human reviews; a human-evaluation pass "
                f"without reviews would attribute nothing to humans — review "
                f"answers first"
            )

        by_question: dict[str, list] = {}
        for review in reviews:
            by_question.setdefault(review.question_id, []).append(review)

        new_results = []
        for result in source.per_question:
            clone = result.model_copy(deep=True)
            apply_reviews(clone, by_question.get(result.question_id, []))
            new_results.append(clone)

        reviewers = sorted({r.reviewer for r in reviews})
        derived = source.model_copy(deep=True)
        derived.id = new_id("aerun")
        derived.created_at = datetime.now(timezone.utc)
        derived.evaluator_name = "human-reviews"
        derived.evaluator_version = EVALUATOR_VERSION
        derived.evaluator_is_model_based = False
        derived.evaluator_detail = (
            f"reviewers: {', '.join(reviewers)} ({len(reviews)} review(s))"
        )
        derived.per_question = new_results
        derived.aggregate = aggregate_results(new_results)
        derived.failed_question_ids = [
            r.question_id for r in new_results if not r.passed
        ]
        derived.notes = list(source.notes) + [
            f"human-evaluation pass derived from run {source.id}"
        ]
        covered = sum(1 for r in new_results if r.correctness.measured)
        derived.warnings = list(source.warnings) + [
            f"human-evaluation pass: correctness measured from reviews for "
            f"{covered}/{len(new_results)} question(s); questions without a "
            f"review remain UNKNOWN"
        ]
        return derived

    def list_runs(self, kb_id: str, limit: int = 50) -> list[AnswerEvaluationRun]:
        return self._repo.list_answer_evaluation_runs(kb_id, limit=limit)

    def get_run(self, kb_id: str, run_id: str) -> AnswerEvaluationRun | None:
        return self._repo.get_answer_evaluation_run(kb_id, run_id)


__all__ = [
    "AnswerEvaluationError",
    "AnswerEvaluationService",
    "BenchmarkCorpusMismatch",
]