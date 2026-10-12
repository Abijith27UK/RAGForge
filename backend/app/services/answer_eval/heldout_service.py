"""Held-out benchmark service — one path for the API, the scripts and the UI (V14).

Every operation on the held-out instrument goes through this class so the gates
cannot be side-stepped by choosing a different entry point:

* the API calls it, so the review view cannot write a state the freeze gate would
  refuse;
* the scripts call it, so "freeze" from the command line runs the same validation
  the UI shows;
* the experiment runner reads the frozen artifact through ``frozen()``, so a run
  always knows the tier and the claim strength it is entitled to.

Nothing here writes to the knowledge base. The only writes are to the two
benchmark artifacts under ``benchmarks/``: the mutable candidate file and the
frozen instrument.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.repositories.sqlite_repo import Repository
from app.services.answer_eval.heldout_benchmark import (
    DEFAULT_DEV_BENCHMARK_PATH,
    DEFAULT_HELDOUT_BENCHMARK_PATH,
    DEFAULT_HELDOUT_CANDIDATE_PATH,
    HeldOutBenchmark,
    HeldOutCandidateSet,
    HeldOutLifecycle,
    HeldOutReviewRequest,
    HeldOutValidationError,
    apply_review,
    claim_strength_for,
    file_sha256,
    freeze as freeze_artifact,
    load_candidate_set,
    load_heldout_benchmark,
    require_confirmatory,
    require_runnable,
    save_candidate_set,
)
from app.services.answer_eval.heldout_validation import (
    HeldOutValidationReport,
    probe_evidence,
    validate_candidates,
)
from app.services.answer_eval.retrieval_diagnostic_runner import (
    REPO_ROOT,
    KnowledgeBaseNotFound,
)

logger = logging.getLogger(__name__)


def resolve_artifact(path: str | Path) -> Path:
    """Resolve a benchmark artifact path repo-relative, then CWD-relative.

    ``benchmarks/foo.json`` is how a versioned artifact is named, and the server
    may have been started from anywhere, so the repository root is tried first.
    """
    given = Path(path)
    if given.is_absolute():
        return given
    candidate = (REPO_ROOT / str(path)).resolve()
    return candidate if candidate.exists() else given


class HeldOutService:
    """Read, review and freeze the held-out retrieval benchmark."""

    def __init__(
        self,
        repo: Repository,
        settings: Settings | None = None,
        *,
        candidate_path: str | Path = DEFAULT_HELDOUT_CANDIDATE_PATH,
        benchmark_path: str | Path = DEFAULT_HELDOUT_BENCHMARK_PATH,
        dev_benchmark_path: str | Path = DEFAULT_DEV_BENCHMARK_PATH,
    ) -> None:
        self._repo = repo
        self._settings = settings or get_settings()
        self._candidate_path = resolve_artifact(candidate_path)
        self._benchmark_path = resolve_artifact(benchmark_path)
        self._dev_benchmark_path = resolve_artifact(dev_benchmark_path)

    # -- paths --------------------------------------------------------------

    @property
    def candidate_path(self) -> Path:
        return self._candidate_path

    @property
    def benchmark_path(self) -> Path:
        return self._benchmark_path

    @property
    def dev_benchmark_path(self) -> Path:
        return self._dev_benchmark_path

    # -- read ---------------------------------------------------------------

    def candidates(self) -> HeldOutCandidateSet:
        """Load the mutable candidate artifact."""
        return load_candidate_set(self._candidate_path)

    def frozen(self) -> HeldOutBenchmark | None:
        """The frozen instrument, or None when it has not been sealed yet."""
        if not self._benchmark_path.exists():
            return None
        return load_heldout_benchmark(self._benchmark_path)

    def validate(self, kb_id: str) -> HeldOutValidationReport:
        """Validate the candidate set against the live corpus, index and dev set."""
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")
        candidates = self.candidates()
        if candidates.kb_id and candidates.kb_id != kb_id:
            raise HeldOutValidationError(
                f"the candidate set was authored for knowledge base {candidates.kb_id!r}, "
                f"not {kb_id!r}: validating it here would compare a corpus it was not "
                "written against"
            )
        probe = probe_evidence(
            self._repo, kb_id, candidates, backend=kb.vector_backend or "qdrant"
        )
        return validate_candidates(
            candidates,
            probe,
            dev_benchmark_path=str(self._dev_benchmark_path),
        )

    def status(self, kb_id: str) -> dict[str, Any]:
        """Everything a reader needs to judge the instrument in one response."""
        candidates = self.candidates()
        report = self.validate(kb_id)
        frozen = self.frozen()
        return {
            "kb_id": kb_id,
            "candidate_path": str(self._candidate_path),
            "benchmark_path": str(self._benchmark_path),
            "dev_benchmark_path": str(self._dev_benchmark_path),
            "candidate_fingerprint": candidates.fingerprint(),
            # A content hash of the candidate FILE as well as the label
            # fingerprint: the fingerprint covers the surviving labels, this
            # covers the whole artifact including review records.
            "candidate_sha256": file_sha256(self._candidate_path),
            "candidate_count": len(candidates.candidates),
            "surviving_count": len(candidates.surviving),
            "rejected_count": len(candidates.rejected),
            "review": candidates.review_completeness(),
            "composition": {
                "subdomains": candidates.subdomain_counts(),
                "query_types": candidates.query_type_counts(),
                "difficulty": candidates.difficulty_counts(),
            },
            "validation": report.model_dump(mode="json"),
            # What the artifacts allow RIGHT NOW. Reported so no caller has to
            # infer the gate's verdict from prose.
            "lifecycle": (frozen.lifecycle.value if frozen else HeldOutLifecycle.DRAFT.value),
            "claim_strength": (
                frozen.claim_strength.value
                if frozen
                else claim_strength_for(HeldOutLifecycle.DRAFT).value
            ),
            "frozen": (
                None
                if frozen is None
                else {
                    "benchmark": frozen.benchmark,
                    "version": frozen.version,
                    "fingerprint": frozen.fingerprint(),
                    "sha256": file_sha256(self._benchmark_path),
                    "frozen_at": frozen.frozen_at,
                    "frozen_by": frozen.frozen_by,
                    "question_count": frozen.question_count,
                    "lifecycle": frozen.lifecycle.value,
                    "claim_strength": frozen.claim_strength.value,
                    "review": frozen.review,
                    "composition": frozen.composition,
                    "authorship": frozen.authorship,
                    "protocol": frozen.protocol,
                    "can_be_run": frozen.lifecycle is not HeldOutLifecycle.DRAFT,
                    "supports_confirmatory_claim": (
                        frozen.claim_strength.value == "confirmatory"
                    ),
                }
            ),
            "notice": (
                "The held-out benchmark is a SEPARATE instrument from the development "
                "benchmark. It has not been used to choose any configuration, and its "
                "questions are drafts until a human records a decision for each."
            ),
        }

    def review_view(self, kb_id: str) -> dict[str, Any]:
        """Everything a human needs to hold one decision per question in view.

        Each entry carries the question, its claim about the evidence, the FULL
        text of the chunk that claim rests on (truncated only for transport, with
        the truncation disclosed), and the availability verdict for that chunk.

        This exists so a reviewer never has to take the draft's word for what the
        chunk says: the evidence is in front of them, and the same text the
        validator checked the quote against is the text they read.
        """
        kb = self._repo.get_kb(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFound(f"Knowledge base {kb_id!r} not found")
        candidates = self.candidates()
        report = self.validate(kb_id)

        expected_ids = sorted(
            {c for cand in candidates.candidates for c in cand.expected_chunk_ids}
        )
        chunk_rows = {c.id: c for c in self._repo.get_chunks_by_ids(kb_id, expected_ids)}
        documents = {d.id: d.title for d in self._repo.list_documents(kb_id)}
        index_chunk_ids: set[str] | None = None
        try:
            from app.services.vector_store import factory as vector_factory

            store = vector_factory.create_vector_store(
                self._settings, backend=kb.vector_backend or "qdrant"
            )
            index_chunk_ids = set(store.all_point_payloads(kb_id).keys())
        except Exception as exc:  # availability is reported, not required, here
            logger.warning("Review view could not enumerate the index: %s", exc)

        validation_by_id = {q.question_id: q for q in report.questions}
        questions: list[dict[str, Any]] = []
        for cand in candidates.candidates:
            evidence: list[dict[str, Any]] = []
            for chunk_id in cand.expected_chunk_ids:
                row = chunk_rows.get(chunk_id)
                text = getattr(row, "text", "") or ""
                evidence.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": getattr(row, "document_id", ""),
                        "document_title": documents.get(
                            getattr(row, "document_id", ""), ""
                        ),
                        "section": getattr(row, "section", "") or "",
                        "char_count": len(text),
                        "text": text,
                        "resolves_in_corpus": row is not None,
                        "present_in_index": (
                            None if index_chunk_ids is None else chunk_id in index_chunk_ids
                        ),
                    }
                )
            verdict = validation_by_id.get(cand.id)
            questions.append(
                {
                    "id": cand.id,
                    "question": cand.question,
                    "subdomain": cand.subdomain,
                    "query_type": cand.query_type.value,
                    "difficulty": cand.difficulty.value,
                    "status": cand.status.value,
                    "review": cand.review.model_dump(mode="json") if cand.review else None,
                    "expected_chunk_ids": list(cand.expected_chunk_ids),
                    "expected_document_ids": list(cand.expected_document_ids),
                    "evidence_quote": cand.evidence_quote,
                    "coverage_rationale": cand.coverage_rationale,
                    "provenance": cand.provenance,
                    "draft_origin": cand.draft_origin,
                    "needs_review": cand.status.survives and not cand.status.is_reviewed,
                    "evidence": evidence,
                    "findings": (
                        [f.model_dump(mode="json") for f in verdict.findings]
                        if verdict
                        else []
                    ),
                }
            )

        return {
            "kb_id": kb_id,
            "benchmark": candidates.benchmark,
            "candidate_fingerprint": candidates.fingerprint(),
            "question_count": len(questions),
            "review": candidates.review_completeness(),
            "questions": questions,
            "notice": (
                "These are DRAFT questions. Approving one records that a named human "
                "read the evidence and agreed the question is answerable from it; it "
                "does not make the question human-AUTHORED. Both facts are stored "
                "separately."
            ),
        }

    # -- review -------------------------------------------------------------

    def review(
        self, kb_id: str, requests: list[HeldOutReviewRequest]
    ) -> dict[str, Any]:
        """Apply human decisions. Refuses to run once the artifact is frozen."""
        if self.frozen() is not None:
            raise HeldOutValidationError(
                "the held-out benchmark is already frozen and is immutable. A frozen "
                "instrument cannot be re-reviewed, because a reviewer who has seen the "
                "results is no longer blind to them. Create a new version instead."
            )
        candidates = self.candidates()
        if candidates.kb_id and candidates.kb_id != kb_id:
            raise HeldOutValidationError(
                f"the candidate set belongs to knowledge base {candidates.kb_id!r}"
            )
        updated = apply_review(candidates, requests)
        save_candidate_set(updated, self._candidate_path)
        report = self.validate(kb_id)
        return {
            "candidates": updated.model_dump(mode="json"),
            "review": updated.review_completeness(),
            "validation": report.model_dump(mode="json"),
            "notice": (
                "Decisions are recorded per question with the reviewer's name. They are "
                "not yet sealed: the frozen instrument takes its tier from the records "
                "that exist when it is frozen."
            ),
        }

    # -- freeze -------------------------------------------------------------

    def freeze(
        self,
        kb_id: str,
        *,
        frozen_by: str,
        notes: list[str] | None = None,
        require_clean_validation: bool = True,
    ) -> dict[str, Any]:
        """Seal the candidate set as the frozen instrument, computing its tier NOW."""
        candidates = self.candidates()
        if candidates.kb_id and candidates.kb_id != kb_id:
            raise HeldOutValidationError(
                f"the candidate set belongs to knowledge base {candidates.kb_id!r}"
            )
        report = self.validate(kb_id)
        benchmark = freeze_artifact(
            candidates,
            path=self._benchmark_path,
            frozen_by=frozen_by,
            validation=report.model_dump(mode="json"),
            notes=notes
            or [
                "Sealed from the V14 candidate set. Review decisions taken before this "
                "freeze are the only ones that count for the tier.",
                report.verdict,
            ],
            require_clean_validation=require_clean_validation,
        )
        final_notes = list(benchmark.notes)
        if benchmark.lifecycle is HeldOutLifecycle.PROVISIONAL:
            final_notes.append(
                "PROVISIONAL: results measured on this instrument are EXPLORATORY. "
                "No human review can be added afterwards to strengthen them, because a "
                "reviewer who has seen the results is no longer blind to them."
            )
        return {
            "benchmark": benchmark.model_dump(mode="json"),
            "fingerprint": benchmark.fingerprint(),
            "sha256": file_sha256(self._benchmark_path),
            "lifecycle": benchmark.lifecycle.value,
            "claim_strength": benchmark.claim_strength.value,
            "path": str(self._benchmark_path),
            "validation": report.model_dump(mode="json"),
        }

    # -- gate helpers used by the experiment runner -------------------------

    def runnable(self) -> tuple[HeldOutBenchmark, str]:
        """The frozen instrument plus its sha256, refusing a DRAFT one."""
        benchmark = self.frozen()
        if benchmark is None:
            raise HeldOutValidationError(
                f"no frozen held-out benchmark at {self._benchmark_path}. Review the "
                "candidates and freeze the artifact before running a validation "
                "experiment: an unfrozen instrument can change, so its numbers are not "
                "reproducible."
            )
        require_runnable(benchmark)
        return benchmark, file_sha256(self._benchmark_path)

    def assert_confirmatory_allowed(self) -> HeldOutBenchmark:
        """Refuse to let an unreviewed instrument support a validation claim."""
        benchmark, _ = self.runnable()
        require_confirmatory(benchmark)
        return benchmark


__all__ = ["HeldOutService", "resolve_artifact"]
