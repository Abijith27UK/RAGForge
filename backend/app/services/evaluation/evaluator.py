"""Evaluation orchestrator: runs questions through retrieval and computes REAL metrics.

Ground truth comes from EvaluationQuestion records:
- expected_chunk_ids    -> chunk-level metrics (headline basis)
- expected_document_ids -> document-level metrics (headline basis, doc grain)
- expected_keywords     -> keyword-overlap heuristic, DIAGNOSTIC ONLY. Scored
  solely in runs explicitly marked allow_keyword_fallback=True and never mixed
  into headline aggregates without disclosure.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.repositories.sqlite_repo import Repository
from app.schemas.models import (
    AggregateMetrics,
    EvaluationQuestion,
    EvaluationRun,
    EvaluationRunConfig,
    PerQuestionMetrics,
)
from app.services.evaluation.metrics import (
    average_metrics,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.services.retrieval.retriever import DenseRetriever

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    pass


class Evaluator:
    def __init__(self, retriever: DenseRetriever, repo: Repository) -> None:
        self._retriever = retriever
        self._repo = repo

    def run_evaluation(
        self,
        kb_id: str,
        config: EvaluationRunConfig,
        embedding_model: str = "",
    ) -> EvaluationRun:
        # --- V3 Phase A: frozen-benchmark runs ---
        # When config.benchmark_version is set, the run scores the immutable
        # snapshot content (never live rows) and refuses non-FROZEN versions.
        benchmark_version_id: str | None = None
        question_statuses: dict[str, str] | None = None
        if config.benchmark_version:
            bv = self._repo.get_benchmark_version(kb_id, config.benchmark_version)
            if bv is None:
                raise EvaluationError(f"Benchmark version {config.benchmark_version!r} not found")
            if bv.status != "FROZEN":
                raise EvaluationError(
                    f"Benchmark version {bv.version!r} is {bv.status}, not FROZEN — "
                    "only frozen versions are usable for official evaluation runs"
                )
            snapshot = [EvaluationQuestion.model_validate(qd) for qd in bv.questions_snapshot]
            questions = snapshot
            if config.question_ids:
                wanted = set(config.question_ids)
                questions = [q for q in questions if q.id in wanted]
                if not questions:
                    raise EvaluationError("None of the requested question ids are in the frozen benchmark")
            benchmark_version_id = bv.id
            question_statuses = {q.id: q.status.value for q in snapshot}
        else:
            questions = self._repo.list_evaluation_questions(kb_id)
            if config.question_ids:
                wanted = set(config.question_ids)
                questions = [q for q in questions if q.id in wanted]
            question_statuses = {q.id: q.status.value for q in questions}
        if not questions:
            raise EvaluationError(
                "No evaluation questions defined for this knowledge base. "
                "Add questions with expected evidence first."
            )
        k = config.top_k

        per_question: list[PerQuestionMetrics] = []
        n_explicit = 0
        n_keyword = 0
        n_skipped = 0

        for q in questions:
            has_explicit = bool(q.expected_chunk_ids or q.expected_document_ids)
            response = self._retriever.retrieve(kb_id, q.question, top_k=k)
            ranked_chunk_ids = [r.chunk_id for r in response.results]
            ranked_doc_ids = [r.document_id for r in response.results]

            relevant_chunks = set(q.expected_chunk_ids)
            relevant_docs = set(q.expected_document_ids)
            note = ""

            # Headline metrics come ONLY from explicitly authored ground truth
            # (chunk/document IDs). Keyword-overlap questions are scored only in
            # runs explicitly marked diagnostic (allow_keyword_fallback=True).
            if not has_explicit and q.expected_keywords and not config.allow_keyword_fallback:
                n_skipped += 1
                per_question.append(
                    PerQuestionMetrics(
                        question_id=q.id,
                        question=q.question,
                        note=(
                            "DIAGNOSTIC-ONLY (keywords, no explicit ground truth): "
                            "excluded from headline metrics in strict mode."
                        ),
                    )
                )
                continue

            if not has_explicit:
                if not q.expected_keywords:
                    n_skipped += 1
                    per_question.append(
                        PerQuestionMetrics(
                            question_id=q.id,
                            question=q.question,
                            note="No ground truth (no expected chunks/documents/keywords); skipped.",
                        )
                    )
                    continue
                n_keyword += 1
                relevant_chunks = self._keyword_relevant_chunks(q, response.results)
                note = (
                    "Relevance via transparent keyword-overlap heuristic on retrieved chunks "
                    "(no explicit ground-truth IDs provided) — diagnostic only, never headline."
                )
                if not relevant_chunks:
                    note += " No retrieved chunk matched any expected keyword."
            else:
                n_explicit += 1

            recall = recall_at_k(ranked_chunk_ids, relevant_chunks, k) if relevant_chunks else None
            precision = precision_at_k(ranked_chunk_ids, relevant_chunks, k) if relevant_chunks else None
            recip_rank = mrr(ranked_chunk_ids, relevant_chunks) if relevant_chunks else None
            ndcg = ndcg_at_k(ranked_chunk_ids, relevant_chunks, k) if relevant_chunks else None

            # Document-level metrics from real document-basis ground truth.
            doc_recall = doc_precision = doc_mrr_val = doc_ndcg = None
            doc_found = 0
            if relevant_docs:
                doc_recall = recall_at_k(ranked_doc_ids, relevant_docs, k)
                doc_precision = precision_at_k(ranked_doc_ids, relevant_docs, k)
                doc_mrr_val = mrr(ranked_doc_ids, relevant_docs)
                doc_ndcg = ndcg_at_k(ranked_doc_ids, relevant_docs, k)
                doc_found = len(set(ranked_doc_ids[:k]) & relevant_docs)

            per_question.append(
                PerQuestionMetrics(
                    question_id=q.id,
                    question=q.question,
                    recall_at_k=recall,
                    precision_at_k=precision,
                    mrr=recip_rank,
                    ndcg=ndcg,
                    num_relevant_found=len(set(ranked_chunk_ids[:k]) & relevant_chunks) if relevant_chunks else 0,
                    num_relevant_total=len(relevant_chunks),
                    doc_recall_at_k=doc_recall,
                    doc_precision_at_k=doc_precision,
                    doc_mrr=doc_mrr_val,
                    doc_ndcg=doc_ndcg,
                    doc_num_relevant_found=doc_found,
                    doc_num_relevant_total=len(relevant_docs),
                    note=note,
                )
            )

        # Aggregate counts describe exactly how the numbers were formed.
        aggregate = AggregateMetrics(
            recall_at_k=average_metrics([m.recall_at_k for m in per_question]),
            precision_at_k=average_metrics([m.precision_at_k for m in per_question]),
            mrr=average_metrics([m.mrr for m in per_question]),
            ndcg=average_metrics([m.ndcg for m in per_question]),
            doc_recall_at_k=average_metrics([m.doc_recall_at_k for m in per_question]),
            doc_precision_at_k=average_metrics([m.doc_precision_at_k for m in per_question]),
            doc_mrr=average_metrics([m.doc_mrr for m in per_question]),
            doc_ndcg=average_metrics([m.doc_ndcg for m in per_question]),
            questions_evaluated=len(
                [m for m in per_question if m.recall_at_k is not None or m.mrr is not None]
            ),
            questions_with_explicit_gt=n_explicit,
            questions_with_keyword_fallback=n_keyword,
            questions_skipped_no_gt=n_skipped,
            strict_mode=not config.allow_keyword_fallback,
            run_label=config.label,
        )

        notes: list[str] = []
        if n_keyword:
            notes.append(
                f"{n_keyword} question(s) scored with the keyword-overlap heuristic "
                "(diagnostic run): treat as indicative, not rigorous."
            )
        if n_skipped:
            notes.append(
                f"{n_skipped} question(s) skipped (no explicit ground truth"
                + (" and strict mode excludes keyword-only questions)." if not config.allow_keyword_fallback else ").")
            )
        if config.label:
            notes.append(f"Run label: {config.label}")
        aggregate.notes = " ".join(notes)

        run = EvaluationRun(
            id=f"eval_{uuid.uuid4().hex[:12]}",
            kb_id=kb_id,
            config=config,
            aggregate=aggregate,
            per_question=per_question,
            retrieval_backend=self._retriever.backend,
            embedding_model=embedding_model,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            benchmark_version=benchmark_version_id,
            question_statuses=question_statuses,
        )
        self._repo.create_evaluation_run(run)
        logger.info("Evaluation run %s for KB %s: %s", run.id, kb_id, aggregate)
        return run

    def _keyword_relevant_chunks(self, q: EvaluationQuestion, results) -> set[str]:
        """A retrieved chunk counts as relevant if it contains >=1 expected keyword
        (case-insensitive). Transparent, deterministic, reported as heuristic.

        Because relevance is judged ONLY on retrieved items, the fallback can
        never count non-retrieved documents as false negatives.
        """
        kws = [kw.lower() for kw in q.expected_keywords if kw]
        relevant: set[str] = set()
        for r in results:
            text = r.text.lower()
            if any(kw in text for kw in kws):
                relevant.add(r.chunk_id)
        return relevant
