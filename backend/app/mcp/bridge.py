"""The bridge between MCP tool calls and RAGForge's existing services.

This module is the ONLY place the MCP server touches RAGForge, and it touches it
through the same services the HTTP API uses:

    retrieval   -> app.services.retrieval.service.RetrievalService (the registry,
                   the KB's persisted configuration, the recorded run)
    answering   -> app.services.answering.service.AnsweringService (query plan,
                   evidence gate, generator, citation validation)
    evaluation  -> app.services.evaluation.evaluator.Evaluator (the frozen
                   benchmark gate and the real metrics)
    storage     -> app.repositories.sqlite_repo.Repository (chunks, documents,
                   build runs, benchmark versions)

There is deliberately NO parallel retrieval path, no second scoring scheme and no
strategy fallback here: an MCP client gets the same ranking, the same scores, the
same provenance and the same grounding states the UI gets. What this module adds
is only what an external client needs: validation of untrusted arguments, size
limits, and an error vocabulary that says what went wrong.

Deliberately free of any `mcp` import, so the logic is testable (and reusable)
without the SDK installed.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.mcp import limits
from app.mcp.errors import (
    CapabilityUnavailableError,
    InvalidArgumentError,
    KnowledgeBaseNotFoundError,
    LifecycleBlockedError,
    NotFoundError,
)
from app.mcp.schemas import (
    UNTRUSTED_CONTENT_WARNING,
    BenchmarkVersionSummary,
    BuildRunSummary,
    CitationOut,
    ClaimOut,
    DependencyStatus,
    DocumentMetadata,
    EvaluationReport,
    EvidenceBundle,
    EvidenceChunk,
    EvidenceHit,
    GroundedAnswerResult,
    IndexStatus,
    KnowledgeBaseCapabilities,
    KnowledgeBaseList,
    KnowledgeBaseStatusReport,
    KnowledgeBaseSummary,
    LifecycleCheck,
    MetricUnavailable,
    MetricValue,
    QuestionMetric,
    RetrievalTrace,
    SearchResponse,
    StrategyCapability,
)
from app.repositories.sqlite_repo import Repository
from app.schemas.answer import AnswerMode, AnswerPolicy, AnswerStatus, SupportStatus
from app.schemas.models import BenchmarkVersion, Chunk, Document, KnowledgeBase
from app.services.answering.generator import AnswerGenerationError, create_answer_generator
from app.services.answering.service import AnswerConfigError, AnsweringService
from app.services.evaluation.evaluator import EvaluationError, Evaluator
from app.services.retrieval.retriever import (
    RetrievalError,
    available_retrievers,
    describe_retrievers,
)
from app.services.retrieval.service import (
    KnowledgeBaseNotFound,
    RetrievalConfigError,
    RetrievalService,
)
# Imported as a MODULE (not the function) so anything that registers or replaces a
# vector backend — including a test double — takes effect here too. A direct
# `from ... import create_vector_store` would bind the default backend at import
# time and silently ignore later registrations.
from app.services.vector_store import factory as vector_factory
from app.services.vector_store.qdrant_store import VectorStoreError

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent

#: The shipped answer-quality benchmark, used ONLY to report honestly why
#: answer-level metrics are unavailable. This tool never edits or runs it.
ANSWER_BENCHMARK_ARTIFACT = "benchmarks/answer-quality-automobile-v1.json"

#: Statuses an official retrieval evaluation may score. Freeze is the gate.
SCORABLE_BENCHMARK_STATUS = "FROZEN"

#: Answer statuses that count as a grounded answer text (a partial answer states
#: what is missing, so it is grounded in part and says so in `grounding_state`).
_GROUNDED_STATUSES = {AnswerStatus.GROUNDED.value, AnswerStatus.PARTIAL.value}

ACCESS_MODEL_NOTE = (
    "Local single-user install: this lists every knowledge base in the RAGForge "
    "workspace, because RAGForge has no accounts and no per-user isolation. Do "
    "not describe these results as scoped to a particular user."
)


def _iso(value: Any) -> str | None:
    """ISO-8601 string for a datetime-ish value; None stays None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class RAGForgeBridge:
    """Implements the MCP surface over RAGForge's own services."""

    def __init__(
        self,
        repo: Repository | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._repo = repo

    # -- wiring -------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def repo(self) -> Repository:
        """The same SQLite database the API uses (same DATA_DIR resolution).

        Opened lazily so an MCP client can complete the handshake and discover
        tools even when the database is not there yet; the failure then appears
        on the call that actually needs it.
        """
        if self._repo is None:
            self._settings.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._repo = Repository(str(self._settings.db_path))
        return self._repo

    # -- shared helpers -----------------------------------------------------

    def _kb(self, kb_id: str) -> KnowledgeBase:
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise InvalidArgumentError("kb_id must be a non-empty string.")
        kb = self.repo.get_kb(kb_id.strip())
        if kb is None:
            raise KnowledgeBaseNotFoundError(
                f"Knowledge base {kb_id!r} does not exist in this workspace."
            )
        return kb

    def _retrieval(self) -> RetrievalService:
        return RetrievalService(self.repo, self._settings)

    def _store(self, kb: KnowledgeBase):
        try:
            return vector_factory.create_vector_store(
                self._settings, backend=(kb.vector_backend or "qdrant")
            )
        except ValueError as exc:  # unknown backend
            raise CapabilityUnavailableError(
                f"This knowledge base records vector backend {kb.vector_backend!r}, "
                f"which is not available in this installation: {exc}"
            ) from exc

    def _embedding_config(self) -> dict[str, Any]:
        return {
            "provider": self._settings.embedding_provider,
            "model": self._settings.embedding_model,
        }

    def _generator_status(self) -> DependencyStatus:
        """What answer generation CAN do right now, without calling a model."""
        generator = create_answer_generator(self._settings)
        unavailable = bool(getattr(generator, "unavailable", False))
        reason = str(getattr(generator, "unavailable_reason", "") or "")
        return DependencyStatus(
            name="answer_generator",
            available=not unavailable,
            detail=(
                f"generator={generator.name} model={getattr(generator, 'model', '')} "
                f"is_mock={getattr(generator, 'is_mock', False)}"
                + (f"; DEGRADED: {reason}" if unavailable else "")
            ),
        )

    def _benchmark_versions(self, kb_id: str) -> list[BenchmarkVersion]:
        try:
            return self.repo.list_benchmark_versions(kb_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not list benchmark versions for %s: %s", kb_id, exc)
            return []

    def _answer_benchmark_lifecycle(self) -> tuple[str | None, str]:
        """(lifecycle state, why) for the shipped answer-quality benchmark.

        Read-only. Used to explain why answer-level metrics are UNKNOWN rather
        than to run anything: the benchmark is NOT scored here.
        """
        path = REPO_ROOT / ANSWER_BENCHMARK_ARTIFACT
        if not path.exists():
            return None, f"the artifact {ANSWER_BENCHMARK_ARTIFACT} was not found"
        try:
            from app.services.answer_eval.review_workflow import (
                AnswerBenchmarkReviewService,
            )

            benchmark = AnswerBenchmarkReviewService(self.repo, self._settings).load(
                str(path)
            )
        except Exception as exc:  # pragma: no cover - artifact/parse problem
            return None, f"the artifact could not be read: {str(exc)[:180]}"
        return benchmark.lifecycle.value, (
            f"the shipped answer benchmark is in lifecycle state "
            f"{benchmark.lifecycle.value!r}"
        )

    # ------------------------------------------------------------------
    # Tool 1 — list_knowledge_bases
    # ------------------------------------------------------------------

    def list_knowledge_bases(self) -> KnowledgeBaseList:
        generator = self._generator_status()
        summaries: list[KnowledgeBaseSummary] = []
        notes: list[str] = []
        if not generator.available:
            notes.append(
                "Answer generation is degraded in this installation: "
                f"{generator.detail}. Grounded-answer calls will still run and will "
                "disclose the degradation on the answer."
            )
        for kb in self.repo.list_kbs():
            summaries.append(self._summarise_kb(kb, generator_available=generator.available))
        return KnowledgeBaseList(
            knowledge_bases=summaries,
            total=len(summaries),
            access_model=ACCESS_MODEL_NOTE,
            notes=notes,
        )

    def _summarise_kb(
        self, kb: KnowledgeBase, *, generator_available: bool
    ) -> KnowledgeBaseSummary:
        chunk_count = self.repo.count_chunks(kb.id)
        document_count = self.repo.count_documents(kb.id)
        versions = self._benchmark_versions(kb.id)
        frozen = [v for v in versions if v.status == SCORABLE_BENCHMARK_STATUS]
        identity = kb.embedding_identity or {}
        basis = [
            f"{chunk_count} chunk row(s) and {document_count} document(s) in the corpus",
            (
                "answer generation: " + ("available" if generator_available else "degraded (see notes)")
            ),
            f"{len(frozen)} FROZEN retrieval benchmark version(s) of {len(versions)} total",
        ]
        return KnowledgeBaseSummary(
            id=kb.id,
            name=kb.name,
            domain=kb.domain,
            purpose=kb.purpose,
            target_audience=kb.target_audience,
            depth=kb.depth,
            status=getattr(kb.status, "value", str(kb.status)),
            version=kb.version,
            document_count=document_count,
            chunk_count=chunk_count,
            has_domain_spec=self.repo.get_domain_spec(kb.id) is not None,
            vector_backend=kb.vector_backend,
            embedding_model=(identity.get("model") if isinstance(identity, dict) else None),
            benchmark_versions=len(versions),
            frozen_benchmark_versions=len(frozen),
            last_build_at=_iso(kb.last_build_at),
            capabilities=KnowledgeBaseCapabilities(
                search=chunk_count > 0,
                answer=chunk_count > 0 and generator_available,
                evaluate_retrieval=bool(frozen),
                basis=basis,
            ),
        )

    # ------------------------------------------------------------------
    # Tool 2 — get_knowledge_base_status
    # ------------------------------------------------------------------

    def get_knowledge_base_status(self, kb_id: str) -> KnowledgeBaseStatusReport:
        kb = self._kb(kb_id)
        chunk_count = self.repo.count_chunks(kb.id)
        document_count = self.repo.count_documents(kb.id)
        corpus_revision = self.repo.corpus_revision(kb.id)
        latest = self.repo.latest_corpus_version(kb.id)
        build = self.repo.latest_build_run(kb.id)

        unavailable: list[str] = []
        warnings: list[str] = []

        # -- index (a REAL probe of the configured vector backend) ----------
        points: int | None = None
        points_measured = False
        points_reason = ""
        collection = ""
        try:
            store = self._store(kb)
            info = store.collection_info(kb.id)
            collection = str(info.get("name") or "")
            raw_points = info.get("points_count")
            if isinstance(raw_points, int):
                points = raw_points
                points_measured = True
            else:
                points_reason = "the vector store reported no point count for this collection"
        except CapabilityUnavailableError as exc:
            points_reason = exc.message
        except VectorStoreError as exc:
            points_reason = f"vector store not reachable: {str(exc)[:200]}"
        except Exception as exc:  # pragma: no cover - defensive
            points_reason = f"vector store could not be queried: {str(exc)[:200]}"
        if not points_measured:
            unavailable.append(f"vector point count: {points_reason}")

        matches: bool | None = None
        if points_measured and chunk_count >= 0:
            matches = points == chunk_count
        recorded = kb.embedding_identity if isinstance(kb.embedding_identity, dict) else None
        configured = self._embedding_config()
        embedding_matches: bool | None = None
        if recorded:
            embedding_matches = (
                str(recorded.get("provider", "")) == str(configured["provider"])
                and str(recorded.get("model", "")) == str(configured["model"])
            )
        else:
            unavailable.append(
                "embedding identity recorded at index time: this knowledge base has "
                "no recorded identity, so the configured embedding model cannot be "
                "checked against what built the vectors"
            )
        if matches is False:
            warnings.append(
                f"the vector store holds {points} point(s) while the local corpus has "
                f"{chunk_count} chunk row(s); the index and the corpus disagree"
            )
        if embedding_matches is False:
            warnings.append(
                "the configured embedding provider/model differs from the identity "
                "recorded at index time; vector search will be refused rather than "
                "compared across incompatible embeddings"
            )

        generator = self._generator_status()
        dependencies = [
            DependencyStatus(
                name="vector_store",
                available=points_measured,
                detail=(
                    f"backend={kb.vector_backend} collection={collection or 'unknown'} "
                    f"points={points}"
                    if points_measured
                    else f"backend={kb.vector_backend}; {points_reason}"
                ),
            ),
            DependencyStatus(
                name="embedding_provider",
                available=bool(configured["provider"]),
                detail=(
                    f"provider={configured['provider']} model={configured['model']}; "
                    "the model is loaded lazily on the first query and was NOT loaded here"
                ),
            ),
            generator,
        ]

        strategies = self._strategy_capabilities(
            chunk_count=chunk_count, store_reachable=points_measured
        )

        can_search = chunk_count > 0 and points_measured and embedding_matches is not False
        ready_basis = (
            f"{chunk_count} chunk row(s); vector store "
            + ("reachable" if points_measured else f"NOT reachable ({points_reason})")
            + f"; embedding configuration "
            + (
                "matches the recorded identity"
                if embedding_matches is True
                else "NOT verified (no recorded identity)"
                if embedding_matches is None
                else "does NOT match the recorded identity"
            )
        )
        if not chunk_count:
            unavailable.append(
                "retrieval: the corpus has no chunks, so nothing can be retrieved"
            )

        return KnowledgeBaseStatusReport(
            kb_id=kb.id,
            name=kb.name,
            domain=kb.domain,
            status=getattr(kb.status, "value", str(kb.status)),
            version=kb.version,
            created_at=_iso(getattr(kb, "created_at", None)),
            updated_at=_iso(getattr(kb, "updated_at", None)),
            last_build_at=_iso(kb.last_build_at),
            document_count=document_count,
            chunk_count=chunk_count,
            corpus_revision=corpus_revision,
            latest_corpus_version=(latest.version if latest else None),
            latest_corpus_fingerprint=(latest.fingerprint if latest else None),
            last_build_run=(
                BuildRunSummary(
                    id=build.id,
                    status=getattr(build, "status", ""),
                    started_at=_iso(build.started_at),
                    finished_at=_iso(build.finished_at),
                    stage_count=len(getattr(build, "stages", []) or []),
                )
                if build
                else None
            ),
            index=IndexStatus(
                vector_backend=kb.vector_backend,
                collection=collection,
                points=points,
                points_measured=points_measured,
                points_reason=points_reason,
                chunk_rows=chunk_count,
                points_match_chunk_rows=matches,
                embedding_recorded=recorded,
                embedding_configured=configured,
                embedding_config_matches=embedding_matches,
                note=(
                    "The point count is a live probe of the vector backend. It is "
                    "reported as unknown — never 0 — when the backend cannot be reached."
                ),
            ),
            retrieval_strategies=strategies,
            dependencies=dependencies,
            ready_for_retrieval=can_search,
            ready_basis=ready_basis,
            unavailable=unavailable,
            warnings=warnings,
        )

    def _strategy_capabilities(
        self, *, chunk_count: int, store_reachable: bool
    ) -> list[StrategyCapability]:
        by_name: dict[str, dict] = {s["name"]: s for s in describe_retrievers()}
        out: list[StrategyCapability] = []
        for name, spec in sorted(by_name.items()):
            if chunk_count == 0:
                usable, basis = False, "the corpus has no chunk rows"
            elif spec.get("needs_embedding"):
                if name.endswith("reranked"):
                    usable, basis = (
                        None,
                        "needs the vector store AND a cross-encoder reranker model; "
                        "the reranker is not probed here and the strategy reports "
                        "unavailability rather than silently skipping the rerank",
                    )
                else:
                    usable, basis = (
                        store_reachable,
                        "needs the vector store and the embedding model"
                        + ("" if store_reachable else " (vector store not reachable)"),
                    )
            else:
                usable, basis = (
                    True,
                    "lexical strategy: reads chunk rows and the persisted index; "
                    "no vector store and no embedding model involved",
                )
            out.append(
                StrategyCapability(
                    name=name,
                    needs_embedding=bool(spec.get("needs_embedding")),
                    requires_repo=bool(spec.get("requires_repo")),
                    is_baseline=bool(spec.get("is_baseline")),
                    notes=str(spec.get("notes") or ""),
                    usable_now=usable,
                    usable_basis=basis,
                )
            )
        if not out:  # pragma: no cover - registry is never empty in practice
            raise CapabilityUnavailableError("No retrieval strategy is registered.")
        return out

    # ------------------------------------------------------------------
    # Tool 3 — search_knowledge_base
    # ------------------------------------------------------------------

    def search_knowledge_base(
        self,
        kb_id: str,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
        params: dict | None = None,
        include_text: bool = True,
    ) -> SearchResponse:
        # Validate the untrusted arguments FIRST: a malformed call is rejected
        # with the argument error it deserves, before any workspace lookup.
        clean_query = limits.check_query(query)
        k = limits.check_top_k(top_k)
        if params is not None and not isinstance(params, dict):
            raise InvalidArgumentError(
                "params must be an object of retrieval-parameter overrides, or omitted."
            )
        requested = (strategy or "").strip().lower() or None
        if requested is not None and requested not in available_retrievers():
            # Fail loudly with the real list: never quietly choose another strategy.
            raise InvalidArgumentError(
                f"Unknown retrieval strategy {strategy!r}. Registered strategies: "
                f"{available_retrievers()}. Omit `strategy` to use this knowledge "
                "base's configured strategy.",
            )
        kb = self._kb(kb_id)
        overrides: dict[str, Any] = {"top_k": k}
        if params:
            overrides.update(params)

        service = self._retrieval()
        try:
            response = service.run(kb.id, clean_query, strategy=requested, overrides=overrides)
        except KnowledgeBaseNotFound as exc:
            raise KnowledgeBaseNotFoundError(str(exc)) from exc
        except RetrievalConfigError as exc:
            raise InvalidArgumentError(str(exc)) from exc
        except ValueError as exc:  # e.g. an unknown vector backend on the KB record
            raise InvalidArgumentError(str(exc)) from exc
        except RetrievalError as exc:
            raise CapabilityUnavailableError(
                f"Retrieval could not run: {exc}",
                hint=(
                    "Call get_knowledge_base_status to see whether the vector store "
                    "and the embedding configuration are usable. No other strategy "
                    "was tried on your behalf."
                ),
            ) from exc

        limits_applied = {
            "top_k": k,
            "max_top_k": limits.MAX_TOP_K,
            "text_chars_per_chunk": limits.MAX_TEXT_CHARS if include_text else 0,
            "include_text": include_text,
        }
        hits = [
            self._hit(result, rank=index, include_text=include_text)
            for index, result in enumerate(response.results, start=1)
        ]
        warnings: list[str] = []
        reranker = None
        if response.reranker is not None:
            reranker = response.reranker.model_dump(mode="json")
            status = str(response.reranker.status)
            if status not in ("applied", "not_requested"):
                warnings.append(
                    f"reranker status is {status!r}: {response.reranker.detail or 'no detail recorded'}"
                )
        params_out = response.params.model_dump(mode="json") if response.params else {}
        return SearchResponse(
            kb_id=kb.id,
            query=clean_query,
            top_k=k,
            result_count=len(hits),
            results=hits,
            requested_strategy=requested,
            applied_strategy=response.strategy,
            limits_applied=limits_applied,
            warnings=warnings,
            trace=RetrievalTrace(
                strategy=response.strategy,
                params=params_out,
                config_fingerprint=(
                    response.params.config_fingerprint() if response.params else None
                ),
                corpus_version=response.corpus_version,
                corpus_fingerprint=response.corpus_fingerprint,
                retrieval_run_id=response.retrieval_run_id,
                embedding_model=response.embedding_model,
                retrieval_backend=response.retrieval_backend,
                reranker=reranker,
                stages=[s.model_dump(mode="json") for s in response.stages],
                timings=response.timings.model_dump(mode="json") if response.timings else {},
                notes=list(response.notes),
            ),
        )

    def _hit(self, result: Any, *, rank: int, include_text: bool) -> EvidenceHit:
        prov = dict(result.provenance or {})
        text = result.text or ""
        returned, truncated = (
            limits.truncate_text(text) if include_text else ("", False)
        )
        breakdown = (
            result.score_breakdown.model_dump(mode="json") if result.score_breakdown else {}
        )
        return EvidenceHit(
            rank=rank,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            source_id=prov.get("source_id"),
            document_title=prov.get("document_title"),
            source_title=prov.get("source_title"),
            source_url=prov.get("source_url"),
            source_type=prov.get("source_type"),
            publisher=prov.get("publisher"),
            section=prov.get("section"),
            section_path=prov.get("section_path"),
            page=prov.get("page"),
            slide=prov.get("slide"),
            slide_title=prov.get("slide_title"),
            domain=prov.get("domain"),
            subdomain=prov.get("subdomain"),
            trust_score=prov.get("trust_score"),
            content_hash=str(prov.get("content_hash") or ""),
            document_version=prov.get("document_version"),
            char_count=len(text),
            retrieval_method=result.retrieval_method,
            retrieval_score=float(result.score),
            rerank_score=result.rerank_score,
            score_breakdown=breakdown,
            why=result.why,
            stages=list(result.stages or []),
            text=returned,
            text_truncated=truncated,
            provenance=prov,
        )

    # ------------------------------------------------------------------
    # Tool 4 — get_evidence
    # ------------------------------------------------------------------

    def get_evidence(
        self,
        kb_id: str,
        chunk_ids: list[str] | None = None,
        document_id: str | None = None,
        max_chunks: int | None = None,
        include_text: bool = True,
    ) -> EvidenceBundle:
        has_ids = bool(chunk_ids)
        has_doc = bool(document_id and str(document_id).strip())
        if has_ids == has_doc:
            raise InvalidArgumentError(
                "Provide exactly one of `chunk_ids` or `document_id`.",
                hint=(
                    "Use chunk_ids for specific evidence returned by "
                    "search_knowledge_base, or document_id to inspect one document's "
                    "chunks."
                ),
            )
        cap = limits.check_max_chunks(max_chunks)
        kb = self._kb(kb_id)
        limits_applied = {
            "max_chunks": cap,
            "max_chunks_hard_limit": limits.MAX_EVIDENCE_CHUNKS,
            "text_chars_per_chunk": limits.MAX_TEXT_CHARS if include_text else 0,
        }
        notes: list[str] = []

        if has_ids:
            requested = [str(c).strip() for c in (chunk_ids or []) if str(c).strip()]
            if not requested:
                raise InvalidArgumentError("chunk_ids must contain at least one id.")
            if len(requested) > limits.MAX_EVIDENCE_CHUNKS:
                raise InvalidArgumentError(
                    f"chunk_ids contains {len(requested)} ids; the maximum per call is "
                    f"{limits.MAX_EVIDENCE_CHUNKS}.",
                    hint="Split the request into several calls.",
                )
            rows = {c.id: c for c in self.repo.get_chunks_by_ids(kb.id, requested)}
            chunks = [
                self._chunk(rows[cid], include_text=include_text)
                for cid in requested
                if cid in rows
            ]
            missing = [cid for cid in requested if cid not in rows]
            if missing:
                notes.append(
                    f"{len(missing)} requested chunk id(s) are not in this knowledge base"
                )
            return EvidenceBundle(
                kb_id=kb.id,
                requested_chunk_ids=requested,
                chunks=chunks,
                found=len(chunks),
                missing_chunk_ids=missing,
                truncated=False,
                limits_applied=limits_applied,
                notes=notes,
            )

        doc = self.repo.get_document(kb.id, str(document_id).strip())
        if doc is None:
            raise NotFoundError(
                f"Document {document_id!r} is not in knowledge base {kb.id!r}."
            )
        rows = self.repo.list_chunks(kb.id, document_id=doc.id, limit=cap)
        total = doc.chunk_count if doc.chunk_count else None
        truncated = total is not None and total > len(rows)
        if truncated:
            notes.append(
                f"Returned the first {len(rows)} of {total} chunk(s) in this document "
                "(chunk_index order). Raise max_chunks or page through the document."
            )
        return EvidenceBundle(
            kb_id=kb.id,
            chunks=[self._chunk(c, include_text=include_text) for c in rows],
            found=len(rows),
            document=self._document(doc),
            document_chunk_total=total,
            truncated=truncated,
            limits_applied=limits_applied,
            notes=notes,
        )

    def _chunk(self, chunk: Chunk, *, include_text: bool) -> EvidenceChunk:
        text = chunk.text or ""
        returned, truncated = limits.truncate_text(text) if include_text else ("", False)
        return EvidenceChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            kb_id=chunk.kb_id,
            chunk_index=chunk.chunk_index,
            text=returned,
            text_truncated=truncated,
            char_count=len(text),
            content_hash=chunk.content_hash or "",
            source_id=chunk.source_id,
            source_title=chunk.source_title,
            source_url=chunk.source_url,
            source_type=chunk.source_type,
            publisher=chunk.publisher,
            document_title=chunk.document_title,
            document_version=chunk.document_version,
            section=chunk.section,
            section_path=chunk.section_path,
            page=chunk.page,
            slide=chunk.slide,
            slide_title=chunk.slide_title,
            domain=chunk.domain,
            subdomain=chunk.subdomain,
            trust_score=chunk.trust_score,
            chunking_strategy=chunk.chunking_strategy,
            user_provided=chunk.user_provided,
        )

    def _document(self, doc: Document) -> DocumentMetadata:
        """Document record with filesystem locations deliberately dropped."""
        return DocumentMetadata(
            document_id=doc.id,
            kb_id=doc.kb_id,
            source_id=doc.source_id,
            title=doc.title,
            url=doc.url,
            source_type=getattr(doc.source_type, "value", str(doc.source_type)),
            publisher=doc.publisher,
            status=getattr(doc.status, "value", str(doc.status)),
            content_hash=doc.content_hash or "",
            text_length=doc.text_length,
            page_count=doc.page_count,
            slide_count=doc.slide_count,
            section_count=doc.section_count,
            chunk_count=doc.chunk_count,
            parser=doc.parser,
            mime_type=doc.mime_type,
            file_size=doc.file_size,
            user_provided=doc.user_provided,
            document_version=doc.document_version,
            replaces_document_id=doc.replaces_document_id,
            ingestion_timestamp=_iso(doc.ingestion_timestamp),
            indexed_at=_iso(doc.indexed_at),
            parse_error=doc.parse_error,
        )

    # ------------------------------------------------------------------
    # Tool 5 — answer_from_knowledge_base
    # ------------------------------------------------------------------

    def answer_from_knowledge_base(
        self,
        kb_id: str,
        question: str,
        answer_mode: str = "abstain_if_unsupported",
        strategy: str | None = None,
        top_k: int | None = None,
        unsupported_claim_action: str = "remove",
        include_evidence: bool = True,
    ) -> GroundedAnswerResult:
        clean_question = limits.check_query(question)
        mode = (answer_mode or "").strip().lower()
        if mode not in ("grounded", "abstain_if_unsupported"):
            raise InvalidArgumentError(
                f"Unknown answer_mode {answer_mode!r}. Supported: 'grounded', "
                "'abstain_if_unsupported'.",
            )
        if (unsupported_claim_action or "") not in ("remove", "downgrade", "abstain"):
            raise InvalidArgumentError(
                "unsupported_claim_action must be one of: remove, downgrade, abstain."
            )
        requested = (strategy or "").strip().lower() or None
        if requested is not None and requested not in available_retrievers():
            raise InvalidArgumentError(
                f"Unknown retrieval strategy {strategy!r}. Registered strategies: "
                f"{available_retrievers()}.",
            )
        k = limits.check_top_k(top_k) if top_k is not None else None
        overrides = {"top_k": k} if k is not None else {}
        kb = self._kb(kb_id)

        service = AnsweringService(self.repo, self._settings)
        try:
            answer, trace, assessment = service.answer(
                kb.id,
                clean_question,
                mode=mode,
                strategy=requested,
                overrides=overrides,
                policy=AnswerPolicy(
                    mode=AnswerMode(mode),
                    unsupported_claim_action=unsupported_claim_action,
                ),
            )
        except KnowledgeBaseNotFound as exc:
            raise KnowledgeBaseNotFoundError(str(exc)) from exc
        except (AnswerConfigError, RetrievalConfigError, ValueError) as exc:
            raise InvalidArgumentError(str(exc)) from exc
        except (RetrievalError, AnswerGenerationError) as exc:
            raise CapabilityUnavailableError(
                f"No grounded answer could be produced: {exc}",
                hint=(
                    "Retrieval or generation failed before an answer existed. "
                    "Nothing was substituted for it — use get_knowledge_base_status "
                    "to check the dependencies, or search_knowledge_base to inspect "
                    "the evidence directly."
                ),
            ) from exc

        status = answer.status.value
        claims = [self._claim(c) for c in answer.claims][: limits.MAX_ANSWER_CLAIMS]
        citations = [self._citation(c) for c in answer.citations]
        evidence = (
            [
                self._evidence_from_answer(e)
                for e in (trace.evidence_items or [])[: limits.MAX_ANSWER_EVIDENCE]
            ]
            if include_evidence
            else []
        )
        unsupported = sum(
            1 for c in answer.claims if c.support_status is SupportStatus.UNSUPPORTED
        )
        abstention_reason = (
            assessment.reason
            if status in (AnswerStatus.ABSTAINED.value, AnswerStatus.CLARIFICATION_REQUIRED.value)
            else (assessment.reason if not assessment.sufficient else "")
        )
        warnings = list(answer.warnings)
        if len(answer.claims) > len(claims):
            warnings.append(
                f"claims were truncated for transport: {len(claims)} of {len(answer.claims)} returned"
            )
        return GroundedAnswerResult(
            kb_id=kb.id,
            question=clean_question,
            answer_id=answer.answer_id,
            status=status,
            grounded=status in _GROUNDED_STATUSES,
            grounded_basis=(
                f"pipeline status {status!r}; citations={len(answer.citations)}; "
                f"grounding_state={assessment.grounding_state.value}"
            ),
            grounding_state=assessment.grounding_state.value,
            sufficient=assessment.sufficient,
            confidence=getattr(answer.confidence, "value", str(answer.confidence)),
            confidence_basis=answer.confidence_basis,
            reason_code=assessment.reason_code,
            reason=assessment.reason,
            answer_text=answer.text or "",
            abstention_reason=abstention_reason,
            unsupported_aspects=list(assessment.unsupported_aspects),
            missing_information=list(assessment.missing_information),
            claims=claims,
            claim_count=len(answer.claims),
            unsupported_claim_count=unsupported,
            citations=citations,
            citation_count=len(answer.citations),
            evidence=evidence,
            evidence_count=len(trace.evidence_items or []),
            evidence_document_count=len({e.document_id for e in (trace.evidence_items or [])}),
            generation={
                "generated_by": answer.generated_by,
                "model": answer.model,
                "is_mock": answer.is_mock,
                "prompt_version": answer.prompt_version,
                "answer_mode": getattr(answer.answer_mode, "value", str(answer.answer_mode)),
            },
            retrieval_run_id=answer.retrieval_run_id,
            answer_trace_id=answer.answer_trace_id or trace.id,
            limits_applied={
                "max_answer_evidence": limits.MAX_ANSWER_EVIDENCE,
                "max_answer_claims": limits.MAX_ANSWER_CLAIMS,
                "top_k": k,
            },
            warnings=warnings,
        )

    def _claim(self, claim: Any) -> ClaimOut:
        return ClaimOut(
            claim_id=claim.claim_id,
            text=claim.text,
            claim_type=getattr(claim.claim_type, "value", str(claim.claim_type)),
            support_status=getattr(claim.support_status, "value", str(claim.support_status)),
            support_check=claim.support_check,
            support_note=claim.support_note,
            citation_ids=list(claim.citation_ids),
            evidence_ids=list(claim.evidence_ids),
        )

    def _citation(self, citation: Any) -> CitationOut:
        return CitationOut(
            citation_id=citation.citation_id,
            evidence_id=citation.evidence_id,
            chunk_id=citation.chunk_id,
            document_id=citation.document_id,
            validation=citation.validation,
            validation_detail=citation.validation_detail,
            source_title=citation.source_title,
            title=citation.title,
            url=citation.url,
            publisher=citation.publisher,
            section=citation.section,
            section_path=citation.section_path,
            page=citation.page,
            slide=citation.slide,
            content_hash=citation.content_hash,
            snippet=citation.snippet,
        )

    def _evidence_from_answer(self, item: Any) -> EvidenceChunk:
        text = item.content or ""
        returned, truncated = limits.truncate_text(text)
        return EvidenceChunk(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            kb_id=item.kb_id,
            chunk_index=int((item.provenance or {}).get("chunk_index") or 0),
            text=returned,
            text_truncated=truncated,
            char_count=len(text),
            content_hash=item.content_hash or "",
            source_id=item.source_id,
            source_title=item.title,
            source_url=item.url,
            source_type=item.source_type,
            publisher=item.publisher,
            document_title=item.title,
            document_version=item.document_version,
            section=item.section,
            section_path=item.section_path,
            page=item.page,
            slide=item.slide,
            slide_title=(item.provenance or {}).get("slide_title"),
            trust_score=item.trust_score,
            user_provided=bool((item.provenance or {}).get("user_provided")),
        )

    # ------------------------------------------------------------------
    # Tool 6 — evaluate_retrieval
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        kb_id: str,
        benchmark_version: str | None = None,
        strategy: str | None = None,
        top_k: int | None = None,
        question_ids: list[str] | None = None,
        label: str = "",
        dry_run: bool = False,
        allow_keyword_fallback: bool = False,
    ) -> EvaluationReport:
        k = limits.check_top_k(top_k)
        requested = (strategy or "").strip().lower() or None
        if requested is not None and requested not in available_retrievers():
            raise InvalidArgumentError(
                f"Unknown retrieval strategy {strategy!r}. Registered strategies: "
                f"{available_retrievers()}.",
            )
        if question_ids and len(question_ids) > limits.MAX_EVALUATION_QUESTIONS:
            raise InvalidArgumentError(
                f"question_ids contains {len(question_ids)} ids; the maximum per "
                f"run is {limits.MAX_EVALUATION_QUESTIONS}."
            )
        kb = self._kb(kb_id)

        lifecycle = self._lifecycle_check(kb.id, benchmark_version)
        if dry_run:
            return EvaluationReport(
                kb_id=kb.id,
                ran=False,
                lifecycle=lifecycle,
                top_k=k,
                strategy=requested or "",
                unavailable_metrics=self._unavailable_metrics(),
                limitations=self._evaluation_limitations(keyword_fallback=allow_keyword_fallback),
                warnings=[
                    "dry_run: the benchmark lifecycle was checked and NO retrieval, "
                    "scoring or persistence was performed"
                ],
            )
        if not lifecycle.allowed or lifecycle.selected_version_id is None:
            raise LifecycleBlockedError(
                lifecycle.reason,
                hint=(
                    "Create and freeze a benchmark version through the existing "
                    "benchmark workflow, or call this tool with dry_run=true to see "
                    "the current version states without running anything."
                ),
                versions=[v.model_dump(mode="json") for v in lifecycle.versions],
            )

        from app.schemas.models import EvaluationRunConfig

        config = EvaluationRunConfig(
            top_k=k,
            question_ids=list(question_ids or []),
            strategy=requested,
            label=label,
            benchmark_version=lifecycle.selected_version_id,
            allow_keyword_fallback=bool(allow_keyword_fallback),
        )
        service = self._retrieval()
        try:
            params = service.resolve_params(kb.id, strategy=requested)
            params = params.model_copy(update={"top_k": k})
            retriever, notes = service.build_retriever(kb, params)
        except RetrievalConfigError as exc:
            raise InvalidArgumentError(str(exc)) from exc
        except RetrievalError as exc:
            raise CapabilityUnavailableError(
                f"Retrieval could not be built for this run: {exc}",
                hint="Check get_knowledge_base_status; the evaluation was not started.",
            ) from exc

        try:
            run = Evaluator(retriever, self.repo).run_evaluation(
                kb.id,
                config,
                embedding_model=self._settings.embedding_model,
                strategy=params.strategy.value,
                retrieval_params=params,
            )
        except EvaluationError as exc:
            message = str(exc)
            if "FROZEN" in message:
                raise LifecycleBlockedError(message) from exc
            if "No evaluation questions" in message:
                raise CapabilityUnavailableError(message) from exc
            raise InvalidArgumentError(message) from exc

        aggregate = run.aggregate
        selected = next(
            (v for v in lifecycle.versions if v.version_id == run.benchmark_version),
            None,
        )
        warnings = list(notes)
        if run.benchmark_version is None:  # pragma: no cover - the gate forbids this
            warnings.append(
                "this run is not attached to a benchmark version, so it is not an "
                "official result"
            )
        if aggregate.questions_with_keyword_fallback:
            warnings.append(
                f"{aggregate.questions_with_keyword_fallback} question(s) were scored by "
                "the keyword-overlap fallback (diagnostic only; excluded from the "
                "headline chunk/document metrics)"
            )
        if aggregate.questions_skipped_no_gt:
            warnings.append(
                f"{aggregate.questions_skipped_no_gt} question(s) had no ground truth and "
                "were skipped rather than scored as zero"
            )

        return EvaluationReport(
            kb_id=kb.id,
            ran=True,
            lifecycle=lifecycle,
            run_id=run.id,
            benchmark_version_id=run.benchmark_version,
            benchmark_version_label=(selected.label if selected else ""),
            benchmark_lifecycle_status=(
                selected.status if selected else SCORABLE_BENCHMARK_STATUS
            ),
            strategy=run.retrieval_strategy,
            top_k=config.top_k,
            retrieval_params=(
                run.retrieval_params.model_dump(mode="json") if run.retrieval_params else {}
            ),
            config_fingerprint=(
                run.retrieval_params.config_fingerprint() if run.retrieval_params else None
            ),
            embedding_model=run.embedding_model,
            retrieval_backend=run.retrieval_backend,
            evaluator={
                "implementation": "app.services.evaluation.evaluator.Evaluator",
                "chunk_grain_ground_truth": "expected_chunk_ids",
                "document_grain_ground_truth": "expected_document_ids",
                "strict_mode": aggregate.strict_mode,
                "run_label": aggregate.run_label,
                "k_equals_retrieval_cutoff": True,
            },
            question_count=aggregate.questions_evaluated,
            questions_with_explicit_gt=aggregate.questions_with_explicit_gt,
            questions_with_keyword_fallback=aggregate.questions_with_keyword_fallback,
            questions_skipped_no_gt=aggregate.questions_skipped_no_gt,
            question_statuses=dict(run.question_statuses or {}),
            metrics=self._metrics(run),
            per_question=[self._question_metric(q) for q in run.per_question],
            unavailable_metrics=self._unavailable_metrics(),
            started_at=_iso(run.started_at),
            finished_at=_iso(run.finished_at),
            limitations=self._evaluation_limitations(keyword_fallback=allow_keyword_fallback),
            warnings=warnings,
        )

    def _lifecycle_check(self, kb_id: str, requested: str | None) -> LifecycleCheck:
        versions = [
            BenchmarkVersionSummary(
                version_id=v.id,
                label=v.version,
                status=v.status,
                question_count=len(v.question_ids or []),
                created_at=_iso(getattr(v, "created_at", None)),
                frozen_at=_iso(getattr(v, "frozen_at", None)),
                created_by=getattr(v, "created_by", "") or "",
            )
            for v in self._benchmark_versions(kb_id)
        ]
        frozen = [v for v in versions if v.status == SCORABLE_BENCHMARK_STATUS]
        requirement = (
            "Only FROZEN benchmark versions may be used for official evaluation runs "
            "(enforced by app.services.evaluation.evaluator.Evaluator); question "
            "lifecycle is DRAFT -> REVIEW -> APPROVED -> FROZEN."
        )
        wanted = (requested or "").strip() or None
        if wanted:
            match = next((v for v in versions if v.version_id == wanted), None)
            if match is None:
                return LifecycleCheck(
                    kb_id=kb_id,
                    versions=versions,
                    frozen_version_ids=[v.version_id for v in frozen],
                    requested_version_id=wanted,
                    allowed=False,
                    reason=f"Benchmark version {wanted!r} does not exist for this knowledge base.",
                    required_lifecycle=requirement,
                )
            if match.status != SCORABLE_BENCHMARK_STATUS:
                return LifecycleCheck(
                    kb_id=kb_id,
                    versions=versions,
                    frozen_version_ids=[v.version_id for v in frozen],
                    requested_version_id=wanted,
                    selected_version_id=match.version_id,
                    allowed=False,
                    reason=(
                        f"Benchmark version {match.label!r} is {match.status}, not "
                        f"{SCORABLE_BENCHMARK_STATUS}, so it may not be scored."
                    ),
                    required_lifecycle=requirement,
                )
            return LifecycleCheck(
                kb_id=kb_id,
                versions=versions,
                frozen_version_ids=[v.version_id for v in frozen],
                requested_version_id=wanted,
                selected_version_id=wanted,
                allowed=True,
                reason=f"Benchmark version {match.label!r} is FROZEN and may be scored.",
                required_lifecycle=requirement,
            )
        if not frozen:
            detail = (
                ", ".join(f"{v.label}={v.status}" for v in versions) or "no versions exist"
            )
            return LifecycleCheck(
                kb_id=kb_id,
                versions=versions,
                frozen_version_ids=[],
                requested_version_id=None,
                allowed=False,
                reason=(
                    "No FROZEN benchmark version exists for this knowledge base "
                    f"(versions: {detail})."
                ),
                required_lifecycle=requirement,
            )
        chosen = frozen[-1]
        return LifecycleCheck(
            kb_id=kb_id,
            versions=versions,
            frozen_version_ids=[v.version_id for v in frozen],
            requested_version_id=None,
            selected_version_id=chosen.version_id,
            allowed=True,
            reason=(
                f"No version was requested; the most recent FROZEN version "
                f"{chosen.label!r} was selected. Non-frozen versions were ignored."
            ),
            required_lifecycle=requirement,
        )

    def _metrics(self, run: Any) -> list[MetricValue]:
        agg = run.aggregate
        explicit = agg.questions_with_explicit_gt
        no_gt_reason = (
            "no question in this run carried explicit chunk/document ground truth, "
            "so this metric has no denominator — it is UNKNOWN, not zero"
        )
        pairs = [
            ("recall_at_k", agg.recall_at_k, "chunks", explicit),
            ("precision_at_k", agg.precision_at_k, "chunks", explicit),
            ("mrr", agg.mrr, "chunks", explicit),
            ("ndcg", agg.ndcg, "chunks", explicit),
            ("doc_recall_at_k", agg.doc_recall_at_k, "documents", explicit),
            ("doc_precision_at_k", agg.doc_precision_at_k, "documents", explicit),
            ("doc_mrr", agg.doc_mrr, "documents", explicit),
            ("doc_ndcg", agg.doc_ndcg, "documents", explicit),
        ]
        out: list[MetricValue] = []
        for name, value, grain, sample in pairs:
            measured = value is not None
            out.append(
                MetricValue(
                    name=name,
                    value=(round(float(value), 6) if measured else None),
                    measured=measured,
                    reason=(
                        f"mean over {sample} question(s) with explicit {grain} ground truth"
                        if measured
                        else no_gt_reason
                    ),
                    sample_size=(sample if measured else None),
                )
            )
        return out

    def _question_metric(self, item: Any) -> QuestionMetric:
        return QuestionMetric(
            question_id=item.question_id,
            question=item.question,
            recall_at_k=item.recall_at_k,
            precision_at_k=item.precision_at_k,
            mrr=item.mrr,
            ndcg=item.ndcg,
            num_relevant_found=item.num_relevant_found,
            num_relevant_total=item.num_relevant_total,
            doc_recall_at_k=item.doc_recall_at_k,
            doc_precision_at_k=item.doc_precision_at_k,
            doc_mrr=item.doc_mrr,
            doc_ndcg=item.doc_ndcg,
            doc_num_relevant_found=item.doc_num_relevant_found,
            doc_num_relevant_total=item.doc_num_relevant_total,
            note=item.note,
        )

    def _unavailable_metrics(self) -> list[MetricUnavailable]:
        state, why = self._answer_benchmark_lifecycle()
        answer_reason = (
            "answer-level metrics (correctness, key-point recall, citation support, "
            "abstention quality) require a human-reviewed answer benchmark; "
            + why
            + (
                " and is therefore not usable for official answer scoring"
                if state not in ("approved", "frozen")
                else ""
            )
        )
        return [
            MetricUnavailable(name="answer_correctness", reason=answer_reason),
            MetricUnavailable(name="key_point_recall", reason=answer_reason),
            MetricUnavailable(name="citation_support", reason=answer_reason),
            MetricUnavailable(name="abstention_quality", reason=answer_reason),
        ]

    def _evaluation_limitations(self, *, keyword_fallback: bool) -> list[str]:
        limitations = [
            "These are RETRIEVAL metrics: they measure whether the expected "
            "chunks/documents were ranked into the top k. They say nothing about "
            "answer quality.",
            "The metric cut-off k and the retrieval cut-off are the same number, so "
            "recall@k is computed over exactly the ranking that was produced.",
            "A metric with no matching ground truth is reported as UNKNOWN with a "
            "reason, never as 0.",
        ]
        if keyword_fallback:
            limitations.append(
                "allow_keyword_fallback=True was set: questions without explicit "
                "ground truth were scored by a transparent keyword-overlap heuristic "
                "for diagnosis only. Those scores are NOT headline metrics and are "
                "counted separately."
            )
        else:
            limitations.append(
                "Strict mode: questions whose only ground truth is keywords were "
                "skipped rather than scored."
            )
        return limitations

    # ------------------------------------------------------------------
    # Resources (read-only views)
    # ------------------------------------------------------------------

    def document_metadata(self, kb_id: str, document_id: str) -> DocumentMetadata:
        kb = self._kb(kb_id)
        doc = self.repo.get_document(kb.id, str(document_id).strip())
        if doc is None:
            raise NotFoundError(f"Document {document_id!r} is not in knowledge base {kb.id!r}.")
        return self._document(doc)

    def evidence_chunk(self, kb_id: str, chunk_id: str) -> EvidenceChunk:
        kb = self._kb(kb_id)
        rows = self.repo.get_chunks_by_ids(kb.id, [str(chunk_id).strip()])
        if not rows:
            raise NotFoundError(f"Chunk {chunk_id!r} is not in knowledge base {kb.id!r}.")
        return self._chunk(rows[0], include_text=True)

    #: Exposed so the server can name the warning it adds to text-bearing payloads.
    UNTRUSTED_CONTENT_WARNING = UNTRUSTED_CONTENT_WARNING


__all__ = ["RAGForgeBridge", "ANSWER_BENCHMARK_ARTIFACT", "ACCESS_MODEL_NOTE"]
