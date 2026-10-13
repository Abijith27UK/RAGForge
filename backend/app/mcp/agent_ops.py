"""V18 agent operations: controlled, auditable knowledge-base construction.

This mixin adds the agentic tools to the V17 bridge. Two rules shape all of it:

1. **It is not a second implementation of anything.** Reads go through the
   services the UI uses (`DomainAnalyzer`, the discovery providers, the quality
   engine, `CorpusIntegrityService`, `RetrievalDiagnosticRunner`), and the
   writes *are* the API's own route handlers (`create_kb`, `ingest`, `index_kb`,
   `_build_source` / `ingest_uploaded_bytes` / `_index_documents`). If the HTTP
   API changes behaviour, MCP changes with it — there is no parallel path to
   drift.
2. **A plan is not a confirmation.** Planning is read-only and records a plan
   row; executing anything consequential requires a confirmation recorded by
   :class:`app.mcp.operations.OperationService` that is bound to the exact
   parameters and can be used once. See that module for what this transport can
   and cannot verify.

Host class contract: `repo`, `settings`, `_kb(kb_id)`, `list_knowledge_bases()`.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from app.mcp import limits, planning
from app.mcp.errors import (
    CapabilityUnavailableError,
    ConfirmationError,
    DuplicateOperationError,
    InvalidArgumentError,
    KnowledgeBaseNotFoundError,
    LifecycleBlockedError,
    MCPToolError,
    NotFoundError,
)
from app.mcp.operations import OperationService
from app.mcp.schemas import (
    AgentActionResult,
    BuildRunSummary,
    ChunkingPlan,
    ConfirmationResult,
    DiscoveryResult,
    DocumentIntakeItem,
    DocumentIntakeResult,
    DomainAnalysisResult,
    DomainProposal,
    KnowledgeRequirementOut,
    KnowledgeBasePlan,
    OperationStage,
    OperationStatusReport,
    PlanAction,
    ProposedDocument,
    ProposedSourcePreview,
    ReadinessCheck,
    RetrievalPlan,
    SourceAssessment,
    SourceAssessmentList,
    StrategyComparison,
    StrategyComparisonReport,
    ValidationReport,
)
from app.schemas.agent_ops import AgentOperation, OperationStatus
from app.schemas.models import (
    KBStatus,
    Document,
    DocumentStatus,
    KnowledgeBase,
    KnowledgeBaseCreate,
    Source,
    SourceDecision,
    SourceMode,
)

logger = logging.getLogger(__name__)

#: The retrieval benchmark an operator gets when they do not name one. It is the
#: same default the diagnostics runner uses.
DEFAULT_BENCHMARK_PATH = "benchmarks/automobile-engineering-baseline-v1.json"

#: Strategies compared when the caller does not choose. Kept to the two the
#: project already reports on; nothing here promotes one over another.
DEFAULT_COMPARE_STRATEGIES = ("dense", "bm25")

EXPLORATORY_LIMITATION = (
    "Exploratory instrument: these numbers may not be used for confirmatory claims and "
    "nothing is promoted to a default by this tool."
)


def _http_status(exc: HTTPException) -> int:
    return int(getattr(exc, "status_code", 500) or 500)


class AgentOperationsMixin:
    """The V18 tools. Mixed into :class:`app.mcp.bridge.RAGForgeBridge`."""

    # -- helpers -------------------------------------------------------------

    def _ops(self) -> OperationService:
        return OperationService(self.repo)

    def _route_error(self, exc: HTTPException, *, action: str) -> MCPToolError:
        """Translate the API's own error into an MCP error with the same reason.

        The API is the source of truth for what went wrong (e.g. "All selected
        sources are user-provided..."), so the message is passed through and only
        the code is chosen here.
        """
        detail = str(getattr(exc, "detail", exc))
        status = _http_status(exc)
        if status == 404:
            return KnowledgeBaseNotFoundError(detail)
        if status in (400, 422):
            return InvalidArgumentError(detail, hint=f"No state was changed by {action}.")
        return CapabilityUnavailableError(
            detail, hint="The dependency reported above is required before this can run."
        )

    def _build_run_summary(self, run) -> dict[str, Any]:
        return {
            "id": run.id,
            "status": run.status,
            "started_at": _iso(run.started_at),
            "finished_at": _iso(run.finished_at),
            "stage_count": len(run.stages),
        }

    def _build_run_stages(self, run) -> list[OperationStage]:
        return [
            OperationStage(
                stage=str(getattr(s.stage, "value", s.stage)),
                status=str(getattr(s.status, "value", s.status)),
                message=s.message or "",
                items_processed=s.items_processed,
            )
            for s in run.stages
        ]

    def _kb_id_from_summary(self, kb_id: str) -> dict[str, Any]:
        """Reuse the V17 summary shape so a client parses one KB object."""
        listing = self.list_knowledge_bases()
        for summary in listing.knowledge_bases:
            if summary.id == kb_id:
                return summary.model_dump(mode="json")
        return {}

    def _source_assessment(self, source: Source, ingested: bool = False) -> SourceAssessment:
        quality = dict(source.quality or {})
        return SourceAssessment(
            source_id=source.id,
            url=source.url,
            title=source.title,
            source_type=str(getattr(source.source_type, "value", source.source_type)),
            publisher=source.publisher,
            user_provided=bool(source.user_provided),
            decision=str(getattr(source.decision, "value", source.decision)),
            trust_score=source.trust_score,
            assessed_by=str(quality.get("assessed_by") or ""),
            signals={k: float(v) for k, v in (quality.get("signals") or {}).items()},
            weights={k: float(v) for k, v in (quality.get("weights") or {}).items()},
            reasons=list(quality.get("reasons") or []),
            warnings=list(quality.get("warnings") or []),
            probe=quality.get("probe"),
            ingested=ingested,
        )

    def _point_count(self, kb: KnowledgeBase) -> tuple[int | None, bool, str]:
        """Vector points for a KB, or (None, False, reason). Never a guess."""
        try:
            store = self._store(kb)
            info = store.collection_info(kb.id)
        except Exception as exc:  # noqa: BLE001 - the reason is the honest answer
            return None, False, f"the vector store could not be read: {str(exc)[:200]}"
        raw = info.get("points_count")
        if raw is None:
            return None, False, "the vector store reported no points_count for this collection"
        return int(raw), True, ""

    # ------------------------------------------------------------------
    # analyze_domain
    # ------------------------------------------------------------------

    def analyze_domain(
        self,
        *,
        domain: str | None = None,
        purpose: str = "",
        target_audience: str = "",
        depth: str = "intermediate",
        kb_id: str | None = None,
    ) -> DomainAnalysisResult:
        """Propose a domain specification and knowledge map. Never persisted."""
        from app.llm.provider import LLMError, create_llm_provider
        from app.services.domain_analyzer.analyzer import DomainAnalyzer
        from app.utils.ids import new_id

        notes: list[str] = []
        warnings: list[str] = []
        kb_exists = False

        if kb_id:
            kb = self._kb(kb_id)
            kb_exists = True
            notes.append(
                f"Analysis used the stored metadata of {kb.id} ({kb.domain}). "
                "No DomainSpec row was written: persisting is a separate, explicit action."
            )
        else:
            domain = (domain or "").strip()
            if not domain:
                raise InvalidArgumentError(
                    "Provide `domain` (and ideally purpose/target_audience), or an existing kb_id."
                )
            kb = KnowledgeBase(
                id=new_id("kbdraft"),
                name=f"{domain} (proposal)",
                domain=domain,
                purpose=(purpose or "General reference use").strip(),
                target_audience=(target_audience or "General audience").strip(),
                depth=(depth or "intermediate").strip(),
                source_mode=SourceMode.EXTERNAL,
            )
            notes.append(
                "No knowledge base exists yet: this is a PROPOSAL computed from the parameters you "
                "supplied, and it was not stored anywhere."
            )

        try:
            provider = create_llm_provider(self._settings, domain_hint=kb.domain)
        except LLMError as exc:
            raise CapabilityUnavailableError(
                f"Domain analysis needs an LLM provider: {exc}",
                hint=(
                    "Configure LLM_PROVIDER/LLM_API_KEY, or allow the labelled development mock "
                    "(ALLOW_MOCK_LLM=true). A mock analysis is reported as is_mock=true."
                ),
            ) from exc

        try:
            spec = DomainAnalyzer(provider, self._settings).analyze(kb)
        except LLMError as exc:
            raise CapabilityUnavailableError(
                f"The configured LLM provider could not produce a domain analysis: {exc}"
            ) from exc

        if spec.is_mock:
            warnings.append(
                "is_mock=true: this domain specification came from RAGForge's development mock "
                "provider, not a real model. Treat it as a placeholder."
            )
        if not spec.knowledge_requirements:
            warnings.append("The proposal contains no explicit knowledge requirements.")

        return DomainAnalysisResult(
            kb_id=kb.id if kb_exists else None,
            kb_exists=kb_exists,
            persisted=False,
            generated_by=spec.generated_by,
            is_mock=bool(spec.is_mock),
            provider_available=True,
            proposal=DomainProposal(
                domain=spec.domain,
                description=spec.description,
                subdomains=list(spec.subdomains),
                key_concepts=list(spec.key_concepts),
                entities=list(spec.entities),
                terminology=list(spec.terminology),
                knowledge_requirements=[
                    KnowledgeRequirementOut(
                        area=r.area,
                        description=r.description,
                        priority=str(getattr(r.priority, "value", r.priority)),
                    )
                    for r in spec.knowledge_requirements
                ],
                recommended_source_categories=list(spec.recommended_source_categories),
            ),
            notes=notes,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # discover_sources / assess_sources
    # ------------------------------------------------------------------

    def discover_sources(
        self,
        *,
        kb_id: str,
        provider: str = "user-url",
        query: str = "",
        limit: int = 5,
    ) -> DiscoveryResult:
        """Register candidate sources and score them with the real quality engine.

        This writes Source rows (candidates + their assessment). It never
        downloads a document, never creates a vector and never ingests.
        """
        from app.api.routes_sources import _score_sources
        from app.services.source_discovery.discovery import (
            DiscoveryError,
            UserURLProvider,
            get_provider,
        )

        kb = self._kb(kb_id)
        text = (query or "").strip()
        if not text:
            raise InvalidArgumentError(
                "query is required: URLs for provider='user-url', a search phrase for provider='arxiv'."
            )
        if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= limits.MAX_DISCOVERY_LIMIT):
            raise InvalidArgumentError(
                f"limit must be an integer between 1 and {limits.MAX_DISCOVERY_LIMIT}; got {limit!r}."
            )

        provider_name = (provider or "user-url").strip().lower()
        try:
            discoverer = UserURLProvider() if provider_name == "user-url" else get_provider(provider_name)
        except DiscoveryError as exc:
            raise InvalidArgumentError(str(exc)) from exc

        before_urls = {s.url for s in self.repo.list_sources(kb.id)}
        try:
            candidates = discoverer.discover(kb.id, text, limit=limit)
        except DiscoveryError as exc:
            raise CapabilityUnavailableError(
                f"Discovery provider {provider_name!r} could not run: {exc}",
                hint="This is an external service; retry later or register the sources by URL instead.",
            ) from exc

        if candidates:
            # The API's own scoring/persistence orchestration: identical scoring,
            # identical persistence, identical status side effects.
            _score_sources(self.repo, kb.id, candidates)

        stored = {s.id: s for s in self.repo.list_sources(kb.id)}
        candidate_ids = {c.id for c in candidates}
        new_sources = [stored[cid] for cid in candidate_ids if cid in stored]
        duplicates = [c for c in candidates if c.url in before_urls]
        invalid = [c for c in candidates if (c.notes or "").startswith("INVALID")]
        ingested_source_ids = {d.source_id for d in self.repo.list_documents(kb.id)}

        result = DiscoveryResult(
            kb_id=kb.id,
            provider=provider_name,
            query=text,
            requested_limit=limit,
            discovered=len(candidates),
            registered_new=len(new_sources),
            duplicates_skipped=len(duplicates),
            rejected_invalid=len(invalid),
            sources=[
                self._source_assessment(s, ingested=s.id in ingested_source_ids) for s in new_sources
            ],
            notes=[
                "Candidate sources are registered with their quality assessment; nothing is "
                "downloaded until ingest_sources runs with a confirmation.",
                "URLs are validated against SSRF rules before any request, and a candidate the "
                "provider rejected is kept visible as INVALID instead of being silently dropped.",
            ],
        )
        if not candidates:
            result.warnings.append(
                "The provider returned no candidates. For provider='arxiv' this can mean the "
                "external service is unreachable or the query matched nothing."
            )
        if invalid:
            result.warnings.append(
                f"{len(invalid)} candidate URL(s) failed validation and cannot be ingested."
            )
        return result

    def assess_sources(
        self,
        *,
        kb_id: str,
        source_ids: list[str] | None = None,
        decision: str | None = None,
        user_provided_only: bool = False,
    ) -> SourceAssessmentList:
        """Explain the stored quality decision for each source. Read-only."""
        kb = self._kb(kb_id)
        wanted = {s.strip() for s in (source_ids or []) if s and s.strip()}
        decision_filter = (decision or "").strip().upper() or None
        if decision_filter is not None:
            allowed = {d.value.upper() for d in SourceDecision}
            if decision_filter not in allowed:
                raise InvalidArgumentError(
                    f"decision must be one of {sorted(allowed)}; got {decision!r}."
                )

        ingested_source_ids = {d.source_id for d in self.repo.list_documents(kb.id)}
        sources = self.repo.list_sources(kb.id)
        if wanted:
            missing = sorted(wanted - {s.id for s in sources})
            if missing:
                raise NotFoundError(
                    f"No source with id {missing[0]!r} exists in {kb.id}.",
                    hint="Source ids come from discover_sources or assess_sources.",
                )
            sources = [s for s in sources if s.id in wanted]
        if decision_filter:
            sources = [
                s for s in sources if str(getattr(s.decision, "value", s.decision)).upper() == decision_filter
            ]
        if user_provided_only:
            sources = [s for s in sources if s.user_provided]

        counts: dict[str, int] = {}
        for source in self.repo.list_sources(kb.id):
            key = str(getattr(source.decision, "value", source.decision))
            counts[key] = counts.get(key, 0) + 1

        unassessed = [s.id for s in sources if s.trust_score is None]
        result = SourceAssessmentList(
            kb_id=kb.id,
            total=len(sources),
            counts_by_decision=counts,
            sources=[self._source_assessment(s, ingested=s.id in ingested_source_ids) for s in sources],
            unassessed=unassessed,
            notes=[
                "Every score, signal, reason and warning is copied from the assessment that was "
                "stored when the source was scored; nothing is recomputed here.",
                "REVIEW means a human must decide. Only ACCEPTED sources can be ingested through MCP.",
            ],
        )
        if unassessed:
            result.notes.append(
                f"{len(unassessed)} source(s) have no stored assessment yet, so they cannot be "
                "shown as scored: " + ", ".join(unassessed[:5])
            )
        return result

    # ------------------------------------------------------------------
    # plan_knowledge_base
    # ------------------------------------------------------------------

    def plan_knowledge_base(
        self,
        *,
        kb_id: str | None = None,
        name: str | None = None,
        domain: str | None = None,
        purpose: str = "",
        target_audience: str = "",
        depth: str = "intermediate",
        source_mode: str = "external",
        urls: list[str] | None = None,
        user_documents: list[dict] | None = None,
        chunking: dict | None = None,
        retrieval: dict | None = None,
        index_documents: bool = True,
    ) -> KnowledgeBasePlan:
        """Propose the next construction steps. Writes plans only — no KB changes."""
        request: dict[str, Any] = {
            "name": (name or "").strip(),
            "domain": (domain or "").strip(),
            "purpose": (purpose or "").strip(),
            "target_audience": (target_audience or "").strip(),
            "depth": (depth or "intermediate").strip(),
            "source_mode": (source_mode or "external").strip(),
            "chunking": dict(chunking or {}),
            "retrieval": dict(retrieval or {}),
            "urls": limits.check_urls(urls),
            "document_count": len(user_documents or []),
            "index_documents": bool(index_documents),
        }
        ops = self._ops()

        if kb_id:
            kb = self._kb(kb_id)
            documents = self.repo.list_documents(kb.id)
            described: list[dict[str, Any]] = []
            for item in limits.check_documents(user_documents):
                file_name, data = planning.decode_document_payload(item)
                described.append(self._describe_user_document(kb.id, file_name, data))
            spec = self.repo.get_domain_spec(kb.id)
            draft = planning.build_extension_plan(
                kb=kb,
                sources=self.repo.list_sources(kb.id),
                documents=documents,
                request={**request, "_domain_spec": spec},
                retrieval_config=self.repo.get_retrieval_config(kb.id),
                described_documents=described,
            )
            draft.current_state["chunk_rows"] = self.repo.count_chunks(kb.id)
            plan_parameters = {"mode": "extend", "kb_id": kb.id, "request": draft.request}
            plan_kb_id = kb.id
        else:
            draft = planning.build_creation_plan(
                request=request,
                existing_kbs=self.repo.list_kbs(),
                planned_urls=request["urls"],
            )
            plan_parameters = {"mode": "create", "request": draft.request}
            plan_kb_id = None

        plan_row = ops.create_plan(
            kb_id=plan_kb_id,
            parameters=plan_parameters,
            summary=f"{draft.mode} plan with {len(draft.actions)} proposed action(s)",
            warnings=draft.warnings,
        )
        actions: list[PlanAction] = []
        for action in draft.actions:
            row = ops.propose_action(
                plan_id=plan_row.id,
                action=action.action,
                kb_id=plan_kb_id,
                parameters=action.parameters,
                summary=action.summary,
                warnings=action.warnings,
            )
            actions.append(
                PlanAction(
                    operation_id=row.id,
                    action=row.action,
                    parameters=row.parameters,
                    parameters_fingerprint=row.fingerprint,
                    summary=action.summary,
                    already_satisfied=action.already_satisfied,
                    warnings=action.warnings,
                )
            )

        plan = KnowledgeBasePlan(
            plan_id=plan_row.id,
            plan_fingerprint=plan_row.fingerprint,
            mode=draft.mode,
            kb_id=draft.kb_id,
            request=draft.request,
            mutates=False,
            domain_spec_present=draft.domain_spec_present,
            knowledge_requirements=[
                KnowledgeRequirementOut(**req) for req in draft.knowledge_requirements
            ],
            source_categories=draft.source_categories,
            proposed_corpus=[
                ProposedSourcePreview.model_validate(preview) for preview in draft.proposed_corpus
            ],
            proposed_documents=[
                ProposedDocument.model_validate(d) for d in draft.proposed_documents
            ],
            current_state=draft.current_state,
            selection_rules=draft.selection_rules,
            ingestion_plan=draft.ingestion_plan,
            chunking=ChunkingPlan.model_validate(draft.chunking) if draft.chunking else None,
            retrieval=RetrievalPlan.model_validate(draft.retrieval) if draft.retrieval else None,
            validation_plan=draft.validation_plan,
            actions=actions,
            confirmation_instructions=planning.confirmation_instructions(plan_row.id),
            warnings=draft.warnings,
        )
        if not actions:
            plan.warnings.append(
                "This plan proposes no action: the workspace already satisfies it, or a prerequisite "
                "(discovery, assessment or ingestion) has not happened yet."
            )
        return plan

    def _describe_user_document(self, kb_id: str, file_name: str, data: bytes) -> dict[str, Any]:
        from app.services.ingestion.parsers import supported_upload_extensions
        from app.utils.text import sha256_bytes

        described = planning.describe_document(
            file_name=file_name,
            data=data,
            allowed_extensions=supported_upload_extensions(),
        )
        described["sha256"] = sha256_bytes(data)
        existing = self.repo.find_document_by_hash(kb_id, described["sha256"])
        described["duplicate_of"] = existing.id if existing else None
        if existing is not None:
            described["warnings"] = list(described["warnings"]) + [
                f"Identical content already exists as {existing.id} "
                f"({existing.file_name or existing.title}); it will be skipped."
            ]
        return described

    # ------------------------------------------------------------------
    # confirm_plan
    # ------------------------------------------------------------------

    def confirm_plan(
        self,
        *,
        plan_id: str,
        action: str,
        parameters_fingerprint: str,
        note: str = "",
    ) -> ConfirmationResult:
        """Record a user confirmation for ONE plan action. Single-use, expiring."""
        plan, op = self._ops().confirm(
            plan_id=plan_id,
            action=action,
            parameters_fingerprint=parameters_fingerprint,
            note=note,
        )
        return ConfirmationResult(
            confirmation_id=op.confirmation_id or "",
            plan_id=plan.id,
            action=op.action,
            parameters_fingerprint=op.fingerprint,
            parameters=dict(op.parameters),
            kb_id=op.kb_id,
            confirmed_at=_iso(op.confirmed_at) or "",
            expires_at=_iso(op.expires_at),
            single_use=True,
            note=note,
            next_step=(
                f"Call {op.action} with confirmation_id={op.confirmation_id} and exactly these "
                "parameters."
            ),
            limitations=[
                "This records that the user approved these exact parameters. MCP over stdio cannot "
                "verify who approved them: the host client is responsible for asking the user "
                "before calling confirm_plan.",
                "The confirmation expires and cannot be reused; a changed parameter set is a "
                "different operation and needs a new plan.",
            ],
        )

    # ------------------------------------------------------------------
    # create_knowledge_base (consequential)
    # ------------------------------------------------------------------

    def create_knowledge_base(
        self,
        *,
        confirmation_id: str,
        name: str,
        domain: str,
        purpose: str,
        target_audience: str,
        depth: str = "intermediate",
        source_mode: str = "external",
    ) -> AgentActionResult:
        """Create the knowledge base container. Requires a confirmation."""
        from app.api.routes_knowledge_bases import create_kb

        parameters = {
            "name": (name or "").strip(),
            "domain": (domain or "").strip(),
            "purpose": (purpose or "").strip(),
            "target_audience": (target_audience or "").strip(),
            "depth": (depth or "intermediate").strip(),
            "source_mode": (source_mode or "external").strip(),
        }
        # The gate is checked BEFORE anything else, so a forged, reused, expired or
        # mismatched confirmation is always reported as such. `check` does not
        # consume it: a refused call (e.g. a duplicate name) must not burn the
        # user's approval.
        ops = self._ops()
        ops.check(
            confirmation_id=confirmation_id,
            action="create_knowledge_base",
            parameters=parameters,
        )

        same_name = [
            kb
            for kb in self.repo.list_kbs()
            if kb.name.strip().lower() == parameters["name"].lower()
            and kb.domain.strip().lower() == parameters["domain"].lower()
        ]
        if same_name:
            existing = same_name[0]
            raise DuplicateOperationError(
                f"A knowledge base with this name and domain already exists: {existing.id}.",
                hint="Reuse that id (it is not modified by this refused call) or plan a different name.",
                existing_kb_id=existing.id,
            )

        op = ops.begin_execution(
            confirmation_id=confirmation_id,
            action="create_knowledge_base",
            parameters=parameters,
        )

        try:
            payload = KnowledgeBaseCreate(**parameters)
            kb = create_kb(payload, self.repo)
        except ValidationError as exc:
            ops.fail(op, code="INVALID_ARGUMENT", message=str(exc))
            raise InvalidArgumentError(f"The knowledge base parameters were rejected: {exc}") from exc
        except HTTPException as exc:
            error = self._route_error(exc, action="create_knowledge_base")
            ops.fail(op, code=error.code, message=error.message)
            raise error from exc

        ops.complete(
            op,
            result_summary={
                "kb_id": kb.id,
                "name": kb.name,
                "domain": kb.domain,
                "status": str(getattr(kb.status, "value", kb.status)),
            },
            warnings=[
                "The container exists and is empty: no source, document or vector was created. "
                "Readiness is never granted by this call.",
                "Next: discover_sources, assess_sources, then plan_knowledge_base on the new kb_id "
                "for the ingest and build actions.",
            ],
        )
        return AgentActionResult(
            operation_id=op.id,
            action="create_knowledge_base",
            status=op.status.value,
            kb_id=kb.id,
            result_summary=dict(op.result_summary),
            warnings=list(op.warnings),
            knowledge_base=self._kb_id_from_summary(kb.id) or None,
            details={"created": True},
        )

    # ------------------------------------------------------------------
    # ingest_sources (consequential)
    # ------------------------------------------------------------------

    def ingest_sources(
        self, *, confirmation_id: str, kb_id: str, source_ids: list[str]
    ) -> AgentActionResult:
        """Download and parse APPROVED sources. Requires a confirmation."""
        from app.api.routes_build import IngestRequest, ingest

        kb = self._kb(kb_id)
        ordered = sorted({s.strip() for s in (source_ids or []) if s and s.strip()})
        if not ordered:
            raise InvalidArgumentError(
                "source_ids must list at least one source; only ACCEPTED sources can be ingested.",
                hint="Discover and assess sources first, then plan the ingest.",
            )

        parameters = {"kb_id": kb.id, "source_ids": ordered}

        # Gate first (without consuming), then cheap refusals: a refused call
        # leaves the approval intact and downloads nothing.
        ops = self._ops()
        ops.check(confirmation_id=confirmation_id, action="ingest_sources", parameters=parameters)

        sources = {s.id: s for s in self.repo.list_sources(kb.id)}
        missing = [sid for sid in ordered if sid not in sources]
        if missing:
            raise NotFoundError(
                f"Source {missing[0]!r} does not exist in {kb.id}.",
                hint="Source ids come from discover_sources or assess_sources.",
            )
        not_accepted = [
            f"{sid} ({str(getattr(sources[sid].decision, 'value', sources[sid].decision))})"
            for sid in ordered
            if sources[sid].decision != SourceDecision.ACCEPT
        ]
        if not_accepted:
            raise InvalidArgumentError(
                "Only ACCEPTED sources may be ingested; these are not accepted: "
                + ", ".join(not_accepted),
                hint=(
                    "REVIEW requires a human decision recorded in the API/UI; REJECT is never "
                    "overridden here. Nothing was downloaded."
                ),
            )
        user_provided = [sid for sid in ordered if sources[sid].user_provided]
        if user_provided:
            raise InvalidArgumentError(
                "User-provided sources are parsed at upload time and are not downloaded: "
                + ", ".join(user_provided),
                hint="Use add_user_documents for material the user hands over.",
            )

        op = ops.begin_execution(
            confirmation_id=confirmation_id, action="ingest_sources", parameters=parameters
        )

        try:
            run = ingest(kb.id, IngestRequest(source_ids=ordered, document_ids=[]), self.repo)
        except HTTPException as exc:
            error = self._route_error(exc, action="ingest_sources")
            ops.fail(op, code=error.code, message=error.message)
            raise error from exc

        stage = run.stages[0] if run.stages else None
        documents = self.repo.list_documents(kb.id)
        document_ids = [d.id for d in documents if d.source_id in set(ordered)]
        warnings = list(op.warnings)
        if run.status != "done":
            warnings.append(
                f"The ingestion run finished with status {run.status!r}"
                + (f": {stage.message}" if stage and stage.message else "")
            )
        ops.complete(
            op,
            result_summary={
                "run_id": run.id,
                "run_status": run.status,
                "message": stage.message if stage else "",
                "documents_from_these_sources": len(document_ids),
            },
            warnings=warnings,
        )
        summary = self._build_run_summary(run)
        return AgentActionResult(
            operation_id=op.id,
            action="ingest_sources",
            status=op.status.value,
            kb_id=kb.id,
            run_id=run.id,
            run_status=run.status,
            result_summary=dict(op.result_summary),
            warnings=list(op.warnings),
            details={"document_ids": sorted(document_ids), "source_ids": ordered},
            build=_as_build_run_summary(summary),
            notes=[
                "Documents are parsed locally and stored with page/slide/section provenance; "
                "nothing is embedded yet.",
                "Duplicate content is detected by hash and skipped, so a repeat ingest is safe.",
            ],
        )

    # ------------------------------------------------------------------
    # build_knowledge_base (consequential)
    # ------------------------------------------------------------------

    def build_knowledge_base(
        self,
        *,
        confirmation_id: str,
        kb_id: str,
        chunker: str = planning.DEFAULT_CHUNKER,
        target_size: int = planning.DEFAULT_TARGET_SIZE,
        overlap: int = planning.DEFAULT_OVERLAP,
    ) -> AgentActionResult:
        """Chunk, embed and index the corpus. Requires a confirmation."""
        from app.api.routes_build import IndexRequest, index_kb

        kb = self._kb(kb_id)
        parameters = {
            "kb_id": kb.id,
            "chunker": (chunker or planning.DEFAULT_CHUNKER).strip(),
            "target_size": int(target_size),
            "overlap": int(overlap),
        }
        ops = self._ops()
        ops.check(
            confirmation_id=confirmation_id, action="build_knowledge_base", parameters=parameters
        )
        op = ops.begin_execution(
            confirmation_id=confirmation_id, action="build_knowledge_base", parameters=parameters
        )

        try:
            run = index_kb(
                kb.id,
                IndexRequest(
                    chunker=parameters["chunker"],
                    target_size=parameters["target_size"],
                    overlap=parameters["overlap"],
                ),
                self.repo,
            )
        except HTTPException as exc:
            error = self._route_error(exc, action="build_knowledge_base")
            ops.fail(op, code=error.code, message=error.message)
            raise error from exc

        refreshed = self.repo.get_kb(kb.id)
        chunk_rows = self.repo.count_chunks(kb.id)
        points, measured, reason = self._point_count(refreshed or kb)
        warnings = list(op.warnings)
        if run.status != "done":
            warnings.append(f"The build run finished with status {run.status!r}; see the stages.")
        if measured and points is not None and points != chunk_rows:
            warnings.append(
                f"Vector points ({points}) and chunk rows ({chunk_rows}) do not match; run "
                "validate_knowledge_base before trusting this index."
            )
        if not measured:
            warnings.append(f"Vector count could not be read after the build: {reason}")

        ops.complete(
            op,
            result_summary={
                "run_id": run.id,
                "run_status": run.status,
                "chunk_rows": chunk_rows,
                "vector_points": points if measured else None,
                "vectors_match_chunks": (points == chunk_rows) if measured else None,
            },
            warnings=warnings,
        )
        summary = self._build_run_summary(run)
        return AgentActionResult(
            operation_id=op.id,
            action="build_knowledge_base",
            status=op.status.value,
            kb_id=kb.id,
            run_id=run.id,
            run_status=run.status,
            result_summary=dict(op.result_summary),
            warnings=list(op.warnings),
            details={
                "chunk_rows": chunk_rows,
                "vector_points": points,
                "vector_points_measured": measured,
                "embedding_identity": dict((refreshed.embedding_identity or {}) if refreshed else {}),
                "stale_vectors_removed_label": _stage_message(run, 2),
            },
            build=_as_build_run_summary(summary),
            notes=[
                "A succeeded build is NOT readiness: validate_knowledge_base measures the index "
                "before anything is called usable.",
            ],
        )

    # ------------------------------------------------------------------
    # add_user_documents (consequential)
    # ------------------------------------------------------------------

    def add_user_documents(
        self,
        *,
        confirmation_id: str,
        kb_id: str,
        documents: list[dict],
        index: bool = True,
        chunker: str = planning.DEFAULT_CHUNKER,
        target_size: int = planning.DEFAULT_TARGET_SIZE,
        overlap: int = planning.DEFAULT_OVERLAP,
    ) -> DocumentIntakeResult:
        """Add user-provided documents (content, never paths). Requires a confirmation."""
        from app.api.routes_documents import _build_source, _index_documents
        from app.services.ingestion.ingestion import IngestionError, ingest_uploaded_bytes
        from app.services.ingestion.parsers import supported_upload_extensions
        from app.services.source_quality.user_scorer import assess_user_upload
        from app.utils.ids import new_id
        from app.utils.text import sha256_bytes

        kb = self._kb(kb_id)
        items = limits.check_documents(documents)
        if not items:
            raise InvalidArgumentError("documents must contain at least one file.")

        decoded: list[tuple[str, bytes, dict[str, Any]]] = []
        for item in items:
            file_name, data = planning.decode_document_payload(item)
            described = planning.describe_document(
                file_name=file_name,
                data=data,
                allowed_extensions=supported_upload_extensions(),
            )
            decoded.append((described["file_name"], data, described))

        parameters = planning.document_action_parameters(
            kb.id,
            [d for _, _, d in decoded],
            chunking={"chunker": chunker, "target_size": target_size, "overlap": overlap},
            index=bool(index),
        )
        ops = self._ops()
        ops.check(
            confirmation_id=confirmation_id, action="add_user_documents", parameters=parameters
        )
        op = ops.begin_execution(
            confirmation_id=confirmation_id, action="add_user_documents", parameters=parameters
        )

        settings = self._settings
        upload_dir = settings.kb_upload_dir(kb.id)
        results: list[DocumentIntakeItem] = []
        new_documents: list[Document] = []
        duplicates = 0
        rejected = 0

        for file_name, data, described in decoded:
            if not described["valid"]:
                rejected += 1
                results.append(
                    DocumentIntakeItem(
                        file_name=file_name,
                        size_bytes=described["size_bytes"],
                        content_hash=described["sha256"],
                        status="rejected",
                        accepted=False,
                        error=described["error"],
                        warnings=list(described["warnings"]),
                    )
                )
                continue

            existing = self.repo.find_document_by_hash(kb.id, described["sha256"])
            if existing is not None:
                duplicates += 1
                results.append(
                    DocumentIntakeItem(
                        file_name=file_name,
                        document_id=existing.id,
                        size_bytes=described["size_bytes"],
                        content_hash=described["sha256"],
                        status="duplicate",
                        accepted=False,
                        warnings=list(described["warnings"])
                        + [f"Identical content already exists as {existing.id}; nothing was replaced."],
                    )
                )
                continue

            source = _build_source(kb.id, file_name, len(data), _extension(file_name))
            self.repo.create_source(source)
            try:
                doc = ingest_uploaded_bytes(
                    kb_id=kb.id,
                    source=source,
                    upload_dir=upload_dir,
                    file_name=file_name,
                    data=data,
                    mime_type=None,
                )
            except IngestionError as exc:
                doc = Document(
                    id=new_id("doc"),
                    kb_id=kb.id,
                    source_id=source.id,
                    url=source.url,
                    title=file_name,
                    source_type=source.source_type,
                    file_path=None,
                    content_hash=described["sha256"],
                    text_length=0,
                    parse_error=str(exc),
                    status=DocumentStatus.FAILED,
                    user_provided=True,
                    file_name=file_name,
                    file_size=len(data),
                    parser=described["detected_format"],
                )
                assessment = assess_user_upload(
                    file_valid=True, parse_ok=False, extracted_chars=0, parse_error=str(exc)
                )
                source.quality = assessment.model_dump(mode="json")
                source.integrity = assessment.model_dump(mode="json")
                source.trust_score = assessment.score
                source.decision = assessment.decision
                self.repo.update_source(source)
                self.repo.create_document(doc)
                rejected += 1
                results.append(
                    DocumentIntakeItem(
                        file_name=file_name,
                        document_id=doc.id,
                        source_id=source.id,
                        size_bytes=len(data),
                        content_hash=described["sha256"],
                        status="failed",
                        decision=str(getattr(assessment.decision, "value", assessment.decision)),
                        score=assessment.score,
                        accepted=False,
                        error=str(exc),
                        warnings=list(assessment.warnings),
                    )
                )
                continue

            unit_count = doc.page_count or doc.slide_count or doc.section_count
            assessment = assess_user_upload(
                file_valid=True,
                parse_ok=True,
                extracted_chars=doc.text_length,
                unit_count=unit_count,
                warnings=list(described["warnings"]),
            )
            source.trust_score = assessment.score
            source.decision = assessment.decision
            source.quality = assessment.model_dump(mode="json")
            source.integrity = assessment.model_dump(mode="json")
            self.repo.update_source(source)
            doc.status = (
                DocumentStatus.PARSED
                if assessment.decision.value != "REJECT"
                else DocumentStatus.FAILED
            )
            self.repo.create_document(doc)
            new_documents.append(doc)
            results.append(
                DocumentIntakeItem(
                    file_name=file_name,
                    document_id=doc.id,
                    source_id=source.id,
                    status="parsed",
                    content_hash=described["sha256"],
                    size_bytes=len(data),
                    text_length=doc.text_length,
                    page_count=doc.page_count,
                    slide_count=doc.slide_count,
                    section_count=doc.section_count,
                    decision=str(getattr(assessment.decision, "value", assessment.decision)),
                    score=assessment.score,
                    accepted=assessment.decision != SourceDecision.REJECT,
                    warnings=list(assessment.warnings),
                )
            )

        indexed = 0
        chunks_created = 0
        vectors_indexed = 0
        stale_removed: int | None = None
        indexing_error = ""
        if index and new_documents:
            # Same side effect as the upload API: the KB is marked as building while
            # its documents are being indexed, and marked READY by the index path.
            kb.status = KBStatus.BUILDING
            self.repo.update_kb(kb)
            try:
                outcome = _index_documents(
                    repo=self.repo,
                    kb=kb,
                    documents=new_documents,
                    chunker=chunker,
                    target_size=target_size,
                    overlap=overlap,
                )
                indexed = int(outcome["documents_indexed"])
                chunks_created = int(outcome["chunk_count"])
                vectors_indexed = int(outcome["vectors_indexed"])
                stale_removed = int(outcome["stale_vectors_removed"])
                for item in results:
                    if item.document_id and item.document_id in outcome["document_ids"]:
                        item.status = "indexed"
            except Exception as exc:  # HTTPException/IndexingError/EmbeddingError
                indexing_error = str(getattr(exc, "detail", exc))
                logger.warning("MCP document intake could not index: %s", indexing_error)

        warnings = list(op.warnings)
        if rejected:
            warnings.append(f"{rejected} document(s) were rejected or failed to parse; see items.")
        if duplicates:
            warnings.append(f"{duplicates} duplicate document(s) were skipped; the existing rows were kept.")
        if indexing_error:
            warnings.append(f"Documents were stored but NOT indexed: {indexing_error}")

        ops.complete(
            op,
            result_summary={
                "accepted": sum(1 for i in results if i.accepted),
                "rejected": rejected,
                "duplicates_skipped": duplicates,
                "indexed": indexed,
                "chunks_created": chunks_created,
                "vectors_indexed": vectors_indexed,
            },
            warnings=warnings,
        )
        return DocumentIntakeResult(
            kb_id=kb.id,
            submitted=len(decoded),
            accepted=sum(1 for i in results if i.accepted),
            rejected=rejected,
            duplicates_skipped=duplicates,
            indexed=indexed,
            chunks_created=chunks_created,
            vectors_indexed=vectors_indexed,
            stale_vectors_removed=stale_removed,
            stale_vectors_removed_label=_stale_label(stale_removed),
            items=results,
            operation_id=op.id,
            notes=[
                "Content travelled in the call, not as a filesystem path: this tool cannot read "
                "files from disk.",
                "Format validation, size caps and the parser are the same ones the upload API uses; "
                "provenance (page/slide/section, content hash) is preserved on every chunk.",
                "User-provided material is assessed for integrity, never for publication authority.",
            ],
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # get_operation_status
    # ------------------------------------------------------------------

    def get_operation_status(self, *, operation_id: str, kb_id: str | None = None) -> OperationStatusReport:
        """Report one operation, joined with the real record it produced."""
        oid = (operation_id or "").strip()
        if not oid:
            raise InvalidArgumentError("operation_id is required.")

        op = self.repo.get_mcp_operation(oid)
        if op is not None:
            return self._operation_report(op)

        # Not an MCP operation: resolve it as one of the records the existing
        # services write, so a client can poll a run it started through the API.
        report = self._resolve_foreign_run(oid, kb_id)
        if report is not None:
            return report
        raise NotFoundError(
            f"No operation, build run, diagnostic run or evaluation run has id {oid!r}.",
            hint=(
                "Operation ids start with plan_/op_ (MCP), build_ (ingest/index), rdx_ "
                "(diagnostics) or eval_ (evaluation). Pass kb_id when polling a run id that is not "
                "an MCP operation."
            ),
        )

    def _operation_report(self, op: AgentOperation) -> OperationStatusReport:
        stage = "request"
        if op.kind == "plan":
            stage = "confirmation" if any(
                child.status == OperationStatus.CONFIRMED
                for child in self.repo.list_mcp_operations(plan_id=op.id, limit=100)
            ) else "plan"
        elif op.status in (OperationStatus.PROPOSED, OperationStatus.CONFIRMED):
            stage = "confirmation"
        elif op.status == OperationStatus.RUNNING:
            stage = "execution"
        else:
            stage = "report"

        report = OperationStatusReport(
            operation_id=op.id,
            kind=op.kind,
            action=op.action,
            lifecycle_stage=stage,
            status=op.status.value,
            kb_id=op.kb_id,
            plan_id=op.plan_id,
            parameters=dict(op.parameters),
            parameters_fingerprint=op.fingerprint,
            confirmation_id=op.confirmation_id,
            confirmation_used=op.consumed_at is not None,
            created_at=_iso(op.created_at) or "",
            updated_at=_iso(op.updated_at) or "",
            started_at=_iso(op.starts_at),
            finished_at=_iso(op.finished_at),
            result_summary=dict(op.result_summary),
            warnings=list(op.warnings),
            error_code=op.error_code,
            error_message=op.error_message,
        )
        run_id = str(op.result_summary.get("run_id") or "")
        if run_id:
            self._attach_run(report, run_id, op.kb_id)
        if op.status == OperationStatus.CONFIRMED:
            report.notes.append(
                "Confirmed (single-use). Execution is the next call; nothing has changed yet "
                f"({_iso(op.expires_at) or 'no'} expiry)."
            )
        elif op.status == OperationStatus.PROPOSED and op.kind != "plan":
            report.notes.append("Proposed only: this row authorizes nothing until it is confirmed.")
        if op.status == OperationStatus.DONE and op.kb_id:
            report.notes.append(
                "A finished operation is not a readiness claim: call validate_knowledge_base for "
                "that."
            )
        return report

    def _attach_run(self, report: OperationStatusReport, run_id: str, kb_id: str | None) -> None:
        run = self.repo.get_build_run(run_id)
        if run is not None:
            report.run_id = run.id
            report.run_status = run.status
            report.run_stages = self._build_run_stages(run)
            report.run_message = next(
                (s.message for s in run.stages if s.message), ""
            )
            report.resolves_to = "build_run"
            return
        if kb_id:
            for rdx in self.repo.list_retrieval_diagnostic_runs(kb_id, limit=100):
                if rdx.id == run_id:
                    report.run_id = rdx.id
                    report.run_status = "completed"
                    report.resolves_to = "retrieval_diagnostics_run"
                    report.run_message = (
                        f"{len(rdx.strategies)} strategy result(s) over "
                        f"{rdx.question_count} question(s)"
                    )
                    return
            for evaluation in self.repo.list_evaluation_runs(kb_id):
                if evaluation.id == run_id:
                    report.run_id = evaluation.id
                    report.run_status = str(getattr(evaluation, "status", "recorded"))
                    report.resolves_to = "evaluation_run"
                    return

    def _resolve_foreign_run(
        self, run_id: str, kb_id: str | None
    ) -> OperationStatusReport | None:
        run = self.repo.get_build_run(run_id)
        if run is not None:
            report = OperationStatusReport(
                operation_id=run.id,
                kind="build_run",
                action="",
                lifecycle_stage="report",
                status=run.status,
                kb_id=run.kb_id,
                created_at=_iso(run.started_at) or "",
                updated_at=_iso(run.finished_at or run.started_at) or "",
                started_at=_iso(run.started_at),
                finished_at=_iso(run.finished_at),
                run_id=run.id,
                run_status=run.status,
                run_stages=self._build_run_stages(run),
                run_message=next((s.message for s in run.stages if s.message), ""),
                resolves_to="build_run",
                notes=[
                    "This id belongs to a build run written by the existing services; it was not "
                    "started through an MCP operation."
                ],
            )
            return report
        if not kb_id:
            return None
        for rdx in self.repo.list_retrieval_diagnostic_runs(kb_id, limit=200):
            if rdx.id == run_id:
                return OperationStatusReport(
                    operation_id=rdx.id,
                    kind="retrieval_diagnostics_run",
                    lifecycle_stage="report",
                    status="completed",
                    kb_id=rdx.kb_id,
                    created_at=str(rdx.created_at or ""),
                    updated_at=str(rdx.created_at or ""),
                    run_id=rdx.id,
                    run_status="completed",
                    run_message=f"{len(rdx.strategies)} strategy result(s)",
                    resolves_to="retrieval_diagnostics_run",
                )
        for evaluation in self.repo.list_evaluation_runs(kb_id):
            if evaluation.id == run_id:
                return OperationStatusReport(
                    operation_id=evaluation.id,
                    kind="evaluation_run",
                    lifecycle_stage="report",
                    status=str(getattr(evaluation, "status", "recorded")),
                    kb_id=kb_id,
                    created_at=_iso(getattr(evaluation, "started_at", None)) or "",
                    updated_at=_iso(getattr(evaluation, "finished_at", None)) or "",
                    run_id=evaluation.id,
                    run_status=str(getattr(evaluation, "status", "recorded")),
                    resolves_to="evaluation_run",
                )
        return None

    # ------------------------------------------------------------------
    # validate_knowledge_base
    # ------------------------------------------------------------------

    def validate_knowledge_base(
        self,
        *,
        kb_id: str,
        include_vectors: bool = True,
        include_diagnostics: bool = True,
    ) -> ValidationReport:
        """Measure the corpus/index and report readiness from EVIDENCE. Read-only."""
        from app.services.corpus.integrity import CorpusIntegrityService

        kb = self._kb(kb_id)
        documents = self.repo.list_documents(kb.id)
        chunk_rows = self.repo.count_chunks(kb.id)
        counts: dict[str, int] = {}
        for doc in documents:
            key = str(getattr(doc.status, "value", doc.status))
            counts[key] = counts.get(key, 0) + 1

        points, measured, reason = self._point_count(kb) if include_vectors else (None, False, "not requested")
        integrity = CorpusIntegrityService(
            self.repo, store_factory=lambda _kb_id: self._store(kb)
        ).scan(kb, include_vectors=include_vectors)

        checks: list[ReadinessCheck] = []

        def add(check: str, passed: bool | None, detail: str) -> None:
            checks.append(ReadinessCheck(check=check, passed=passed, detail=detail))

        parsed_documents = sum(
            1
            for d in documents
            if str(getattr(d.status, "value", d.status)) in ("parsed", "ready", "indexed")
        )
        add(
            "documents_parsed",
            parsed_documents > 0 if documents else False,
            f"{parsed_documents} of {len(documents)} document(s) are parsed or indexed",
        )
        add(
            "chunk_rows_present",
            chunk_rows > 0,
            f"{chunk_rows} chunk row(s) in the local database",
        )
        add(
            "vector_points_measured",
            measured if include_vectors else None,
            f"{points} point(s) in the collection" if measured else f"not measured: {reason}",
        )
        add(
            "vector_points_match_chunks",
            (points == chunk_rows) if measured else None,
            (
                f"{points} vector point(s) vs {chunk_rows} chunk row(s)"
                if measured
                else "cannot be compared: the vector store did not answer"
            ),
        )
        add(
            "integrity_audit",
            None if integrity.overall_status == "UNKNOWN" else integrity.overall_status == "HEALTHY",
            f"corpus integrity status {integrity.overall_status}"
            + (
                f"; counts {integrity.counts()}"
                if integrity.overall_status != "HEALTHY"
                else ""
            ),
        )
        encoding_mismatch = bool(integrity.embedding_mismatches)
        add(
            "embedding_identity_consistent",
            (not encoding_mismatch) if integrity.vectors_checked != "UNKNOWN" else None,
            (
                "the recorded embedding identity matches the vectors"
                if not encoding_mismatch and integrity.vectors_checked != "UNKNOWN"
                else f"{len(integrity.embedding_mismatches)} embedding mismatch(es)"
                if encoding_mismatch
                else "not checked (vector store unreachable)"
            ),
        )

        diagnostics: dict[str, Any] = {"requested": bool(include_diagnostics)}
        if include_diagnostics:
            versions = self.repo.list_benchmark_versions(kb.id)
            frozen = [v for v in versions if str(v.status).upper() == "FROZEN"]
            diagnostics.update(
                {
                    "benchmark_versions": len(versions),
                    "frozen_versions": len(frozen),
                    "permitted": bool(frozen),
                    "reason": (
                        f"{len(frozen)} FROZEN benchmark version(s) available"
                        if frozen
                        else "no FROZEN benchmark version exists for this knowledge base, so an "
                        "official diagnostic/experiment may not be run against it"
                    ),
                }
            )

        blocking = [
            f"{c.check}: {c.detail}" for c in checks if c.passed is False
        ]
        unmeasured = [c.check for c in checks if c.passed is None]
        ready = not blocking and chunk_rows > 0 and parsed_documents > 0 and (points == chunk_rows if measured else False)
        verdict = "READY" if ready else ("NOT_READY" if blocking else "UNKNOWN")

        basis = [
            f"{len(documents)} document(s); statuses {counts}",
            f"{chunk_rows} chunk row(s)",
            (
                f"{points} vector point(s) measured"
                if measured
                else f"vector points NOT measured ({reason})"
            ),
            f"corpus integrity {integrity.overall_status}",
        ]
        report = ValidationReport(
            kb_id=kb.id,
            ready_for_retrieval=ready,
            verdict=verdict,
            basis=basis,
            blocking=blocking,
            checks=checks,
            document_counts=counts,
            chunk_rows=chunk_rows,
            vector_points=points,
            vector_points_measured=measured,
            integrity_status=integrity.overall_status,
            integrity_counts=integrity.counts(),
            integrity_unknown=list(integrity.unknown_counts),
            integrity_findings=[f.model_dump(mode="json") for f in integrity.findings[:20]],
            embedding_identity=dict(kb.embedding_identity or {}),
            diagnostics=diagnostics,
            notes=[
                "Read-only: no document, chunk, vector or benchmark was modified.",
                "A capability that could not be measured is reported as UNKNOWN - it is never "
                "counted as passing, and a metadata-only knowledge base is never called ready.",
            ],
        )
        if unmeasured:
            report.notes.append(
                "Not measurable in this environment: " + ", ".join(unmeasured)
            )
        if integrity.unknown_counts:
            report.notes.append(
                "Corpus checks that could not be confirmed: " + ", ".join(integrity.unknown_counts)
            )
        return report

    # ------------------------------------------------------------------
    # compare_retrieval_strategies
    # ------------------------------------------------------------------

    def compare_retrieval_strategies(
        self,
        *,
        kb_id: str,
        benchmark_path: str | None = None,
        strategies: list[str] | None = None,
        top_k: int = 10,
        label: str = "",
        acknowledge_exploratory: bool = False,
        candidate_depth: int | None = None,
    ) -> StrategyComparisonReport:
        """Compare strategies with the EXISTING diagnostics runner, lifecycle-gated.

        Nothing here promotes a strategy: a comparison is a measurement over the
        project's own instrument, and an exploratory instrument is refused unless
        the caller acknowledges that its numbers cannot support a claim.
        """
        from app.api.routes_answer_eval import _resolve_benchmark_path
        from app.services.answer_eval.retrieval_diagnostic_runner import (
            BenchmarkValidationError,
            RetrievalDiagnosticRunner,
            load_retrieval_benchmark,
        )
        from app.services.retrieval.service import KnowledgeBaseNotFound

        kb = self._kb(kb_id)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not (1 <= top_k <= limits.MAX_TOP_K):
            raise InvalidArgumentError(
                f"top_k must be an integer between 1 and {limits.MAX_TOP_K}; got {top_k!r}."
            )

        path = (benchmark_path or DEFAULT_BENCHMARK_PATH).strip()
        try:
            resolved = _resolve_benchmark_path(path)
        except HTTPException as exc:
            raise InvalidArgumentError(
                str(getattr(exc, "detail", exc)),
                hint=(
                    "Benchmarks are addressed relative to the repository root or the configured "
                    "DATA_DIR; nothing outside those roots can be read."
                ),
            ) from exc

        try:
            benchmark = load_retrieval_benchmark(resolved)
        except BenchmarkValidationError as exc:
            return StrategyComparisonReport(
                kb_id=kb.id,
                permitted=False,
                reason=f"the benchmark could not be used: {exc}",
                benchmark={"path": path},
                label=label,
                limitations=[EXPLORATORY_LIMITATION],
            )

        authorship = dict(benchmark.authorship or {})
        tier = str(authorship.get("tier") or authorship.get("lifecycle") or "").strip().lower()
        review = str(authorship.get("human_review") or "").strip().lower()
        exploratory = (
            tier in ("provisional", "exploratory", "draft")
            or "pending" in review
            or tier not in ("frozen", "confirmatory", "confirmed", "reviewed")
        )
        benchmark_summary = {
            "benchmark": benchmark.benchmark,
            "version": benchmark.version,
            "questions": len(benchmark.questions),
            "fingerprint": benchmark.fingerprint(),
            "kb_id": benchmark.kb_id,
            "tier": tier or "unspecified",
            "human_review": review or "unspecified",
            "path": path,
        }

        if benchmark.kb_id and benchmark.kb_id != kb.id:
            return StrategyComparisonReport(
                kb_id=kb.id,
                permitted=False,
                reason=(
                    f"benchmark {benchmark.benchmark!r} was authored for knowledge base "
                    f"{benchmark.kb_id!r}; scoring it here would measure a corpus it was not "
                    "written against"
                ),
                benchmark=benchmark_summary,
                label=label,
                limitations=[EXPLORATORY_LIMITATION],
            )
        if exploratory and not acknowledge_exploratory:
            return StrategyComparisonReport(
                kb_id=kb.id,
                permitted=False,
                reason=(
                    "this instrument is exploratory "
                    f"(tier={benchmark_summary['tier']!r}, human_review={benchmark_summary['human_review']!r}), "
                    "so it may only be used when you explicitly acknowledge that no confirmatory "
                    "claim follows from it"
                ),
                benchmark=benchmark_summary,
                label=label,
                warnings=[
                    "Pass acknowledge_exploratory=true to run it as an exploratory measurement, or "
                    "complete the human review and freeze a benchmark version first."
                ],
                limitations=[EXPLORATORY_LIMITATION],
            )

        names = [s.strip().lower() for s in (strategies or DEFAULT_COMPARE_STRATEGIES) if s.strip()]
        try:
            report = RetrievalDiagnosticRunner(self.repo, self._settings).run(
                kb.id,
                strategies=names,
                top_k=top_k,
                benchmark_path=resolved,
                persist=True,
                persist_retrieval_runs=False,
                candidate_depth=candidate_depth if candidate_depth is not None else max(top_k, 20),
            )
        except KnowledgeBaseNotFound as exc:
            raise KnowledgeBaseNotFoundError(str(exc)) from exc
        except BenchmarkValidationError as exc:
            raise InvalidArgumentError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - dependency or retrieval failure
            raise CapabilityUnavailableError(
                f"the retrieval diagnostics run could not complete: {str(exc)[:300]}"
            ) from exc

        comparisons: list[StrategyComparison] = []
        for strategy in report.strategies:
            summary = getattr(strategy, "summary", None)
            metrics: dict[str, float | None] = {}
            measured = False
            if summary is not None:
                for name in (
                    "recall_at_k",
                    "precision_at_k",
                    "mrr",
                    "ndcg_at_k",
                    "required_recall_at_depth",
                    "required_recall_in_probe",
                ):
                    metric = getattr(summary, name, None)
                    if metric is None:
                        continue
                    metrics[name] = float(metric.value) if metric.measured and metric.value is not None else None
                    measured = measured or bool(metric.measured)
            comparisons.append(
                StrategyComparison(
                    strategy=strategy.strategy,
                    metrics=metrics,
                    measured=measured and strategy.status == "completed",
                    note=(
                        f"status={strategy.status} evaluated={getattr(summary, 'evaluated_question_count', 0)}"
                        + (f"; {strategy.failure_reason}" if strategy.failure_reason else "")
                    ),
                )
            )

        return StrategyComparisonReport(
            kb_id=kb.id,
            permitted=True,
            reason="run completed against the named instrument",
            benchmark=benchmark_summary,
            strategies=comparisons,
            top_k=top_k,
            label=label,
            claim_strength="exploratory" if exploratory else "instrument-reviewed",
            promoted=False,
            run_id=report.id,
            warnings=list(report.limitations[:3]),
            limitations=[
                EXPLORATORY_LIMITATION if exploratory else "Nothing is promoted by this tool.",
                "Only questions both strategies measured are comparable; per-question pairing is "
                "recorded in the persisted diagnostics report.",
                "Metrics that could not be computed are null (UNKNOWN), never zero.",
            ],
        )


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _stage_message(run, index: int) -> str:
    try:
        stage = run.stages[index]
    except IndexError:
        return ""
    return stage.message or ""


def _extension(file_name: str) -> str:
    from pathlib import Path

    return Path(file_name).suffix.lower()


def _stale_label(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value < 0:
        return "unknown"
    return str(value)


def _as_build_run_summary(summary: dict[str, Any]) -> BuildRunSummary | None:
    """Shape a build run for the client; a mismatch is reported, never faked."""
    try:
        return BuildRunSummary.model_validate(summary)
    except ValidationError:  # pragma: no cover - defensive
        logger.warning("Could not shape build run summary: %s", summary)
        return None
