"""Planning for the agentic workflow (V18): pure functions, no side effects.

Everything here answers a read-only question — "what WOULD happen?":

* which sources a plan would register or ingest,
* which documents it would add (validated, hashed, duplicate-checked),
* which chunking and retrieval configuration it would use,
* which evidence would be required afterwards to call the KB usable.

No function in this module writes a row, downloads a document, calls a model or
touches a vector store. The one thing a plan is allowed to record (a plan row and
its proposed actions) is written by :mod:`app.mcp.operations`, not here, so this
module stays unit-testable without a database.

Two rules are encoded and reused everywhere:

1. **Plan parameters must be reproducible.** The parameters a plan proposes are
   the exact dicts that a confirmation binds and an execution must re-derive
   byte-for-byte, so list-valued parameters are sorted and scalars normalised.
2. **A plan never asserts a source is good.** It shows the real decision the
   scorer already made and states the rule it came from.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass, field
from typing import Any

from app.mcp.errors import InvalidArgumentError
from app.schemas.models import (
    Document,
    KnowledgeBase,
    Source,
    SourceDecision,
    SourceMode,
)

#: The decision rule of the external scorer, quoted from
#: `app.services.source_quality.scorer.SourceQualityScorer._decide`. Kept as
#: prose so a plan reports the rule a user is relying on, and a test asserts the
#: numbers still match the implementation.
SELECTION_RULES: tuple[str, ...] = (
    "External sources are scored by the existing quality engine (authority, relevance, "
    "recency, source type, accessibility, duplication, evidence quality).",
    "score >= 0.70 -> ACCEPT; 0.45 <= score < 0.70 -> REVIEW; score < 0.45 -> REJECT; "
    "an unreachable URL is REJECT regardless of score.",
    "Only ACCEPTED sources may be ingested through MCP. REVIEW means a human decides, "
    "and REJECT is not overridden by this server.",
    "User-provided material is never scored by the external authority model: it is "
    "assessed for integrity (valid file, parser success, extracted text) instead, and "
    "the uploader's assertion is the relevance signal.",
    "A source already registered for this knowledge base is not duplicated (matched by URL).",
)

#: Chunk-size bounds, identical to the API's IndexRequest.
CHUNK_TARGET_RANGE = (200, 8000)
CHUNK_OVERLAP_RANGE = (0, 1000)
DEFAULT_CHUNKER = "section-aware"
DEFAULT_TARGET_SIZE = 1200
DEFAULT_OVERLAP = 150


@dataclass
class ActionDraft:
    """One consequential action a plan proposes."""

    action: str
    parameters: dict[str, Any]
    summary: str
    warnings: list[str] = field(default_factory=list)
    already_satisfied: bool = False


@dataclass
class PlanDraft:
    mode: str
    request: dict[str, Any]
    kb_parameters: dict[str, Any] | None = None
    kb_id: str | None = None
    domain_spec_present: bool = False
    knowledge_requirements: list[dict[str, Any]] = field(default_factory=list)
    source_categories: list[str] = field(default_factory=list)
    proposed_corpus: list[dict[str, Any]] = field(default_factory=list)
    proposed_documents: list[dict[str, Any]] = field(default_factory=list)
    current_state: dict[str, Any] = field(default_factory=dict)
    selection_rules: list[str] = field(default_factory=lambda: list(SELECTION_RULES))
    ingestion_plan: list[str] = field(default_factory=list)
    chunking: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    validation_plan: list[str] = field(default_factory=list)
    actions: list[ActionDraft] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Known-good user material (Phase 4): content in, never paths
# ---------------------------------------------------------------------------


def decode_document_payload(item: dict[str, Any]) -> tuple[str, bytes]:
    """Return (file_name, bytes) for one caller-supplied document.

    The transport carries CONTENT, never a path: an agent cannot ask this server
    to read a file. Exactly one of `content_text` / `content_base64` must be set.
    """
    file_name = str(item.get("file_name") or "").strip()
    if not file_name:
        raise InvalidArgumentError("Each document needs a file_name (used for display and format detection).")

    text = item.get("content_text")
    encoded = item.get("content_base64")
    if (text is None) == (encoded is None):
        raise InvalidArgumentError(
            f"Document {file_name!r}: provide exactly one of content_text or content_base64.",
            hint="Text documents use content_text; binary formats (pdf/docx/pptx) use content_base64.",
        )
    if text is not None:
        if not isinstance(text, str):
            raise InvalidArgumentError(f"Document {file_name!r}: content_text must be a string.")
        return file_name, text.encode("utf-8")
    if not isinstance(encoded, str):
        raise InvalidArgumentError(f"Document {file_name!r}: content_base64 must be a string.")
    try:
        return file_name, base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidArgumentError(
            f"Document {file_name!r}: content_base64 is not valid base64 ({exc})."
        ) from exc


def describe_document(
    *,
    file_name: str,
    data: bytes,
    allowed_extensions: list[str],
    duplicate_of: str | None = None,
) -> dict[str, Any]:
    """Validate one document and describe it EXACTLY (size + sha256).

    Validation is the real upload validator (`validate_upload`), so a format the
    HTTP API would reject is rejected here too, with the same message.
    """
    from app.services.ingestion.upload import validate_upload

    result = validate_upload(file_name, data, allowed_extensions=allowed_extensions)
    return {
        "file_name": result.file_name,
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
        "detected_format": result.detected_format,
        "valid": bool(result.ok),
        "error": result.error,
        "warnings": list(result.warnings or []),
        "duplicate_of": duplicate_of,
    }


def document_action_parameters(
    kb_id: str,
    described: list[dict[str, Any]],
    *,
    chunking: dict[str, Any] | None = None,
    index: bool = True,
) -> dict[str, Any]:
    """The exact parameter set a document intake would be authorized with.

    Documents are sorted by (file name, sha256) so the fingerprint depends on the
    SET of documents, not on the order the caller happened to list them in. The
    chunking configuration is part of the binding too: re-chunking the same bytes
    differently is a different corpus.
    """
    items = sorted(
        (
            {
                "file_name": d["file_name"],
                "size_bytes": int(d["size_bytes"]),
                "sha256": d["sha256"],
            }
            for d in described
        ),
        key=lambda d: (d["file_name"], d["sha256"]),
    )
    config = dict(chunking or {})
    return {
        "kb_id": kb_id,
        "documents": items,
        "index": bool(index),
        "chunker": str(config.get("chunker") or DEFAULT_CHUNKER),
        "target_size": int(config.get("target_size") or DEFAULT_TARGET_SIZE),
        "overlap": int(config.get("overlap", DEFAULT_OVERLAP)),
    }


# ---------------------------------------------------------------------------
# Source previews (validated, not fetched)
# ---------------------------------------------------------------------------


def preview_user_urls(urls: list[str]) -> list[dict[str, Any]]:
    """SSRF-validate each URL. No request is made and nothing is registered."""
    from app.utils.url_validation import InvalidURLError, validate_public_http_url

    previews: list[dict[str, Any]] = []
    for raw in urls:
        url = (raw or "").strip()
        if not url:
            continue
        try:
            validated = validate_public_http_url(url)
        except InvalidURLError as exc:
            previews.append(
                {
                    "url": url,
                    "source_type": "web_page",
                    "accepted_for_planning": False,
                    "reason": f"Rejected by URL validation: {exc}",
                    "provider": "user-url",
                }
            )
            continue
        suffix = validated.lower().split("?")[0].rsplit(".", 1)[-1]
        previews.append(
            {
                "url": validated,
                "source_type": "pdf" if suffix == "pdf" else "web_page",
                "accepted_for_planning": True,
                "reason": "Public http(s) URL; a probe and the quality engine run at discovery time.",
                "provider": "user-url",
            }
        )
    return previews


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------


def resolve_chunking(request: dict[str, Any], kb: KnowledgeBase | None) -> dict[str, Any]:
    """Chunking configuration for a build, defaulting to the KB's own history."""
    from app.services.indexing.document_indexer import chunker_names

    available = chunker_names()
    proposed = dict(request.get("chunking") or {})
    name = str(proposed.get("chunker") or (kb.chunking_strategy if kb else "") or DEFAULT_CHUNKER)
    if name not in available:
        raise InvalidArgumentError(
            f"Unknown chunking strategy {name!r}.",
            hint=f"Registered chunkers: {', '.join(available)}.",
        )
    config = dict(kb.chunking_config or {}) if kb else {}
    target = int(proposed.get("target_size") or config.get("target_size") or DEFAULT_TARGET_SIZE)
    overlap = int(proposed.get("overlap") if proposed.get("overlap") is not None else config.get("overlap", DEFAULT_OVERLAP))
    if not (CHUNK_TARGET_RANGE[0] <= target <= CHUNK_TARGET_RANGE[1]):
        raise InvalidArgumentError(
            f"target_size must be between {CHUNK_TARGET_RANGE[0]} and {CHUNK_TARGET_RANGE[1]}; got {target}."
        )
    if not (CHUNK_OVERLAP_RANGE[0] <= overlap <= CHUNK_OVERLAP_RANGE[1]):
        raise InvalidArgumentError(
            f"overlap must be between {CHUNK_OVERLAP_RANGE[0]} and {CHUNK_OVERLAP_RANGE[1]}; got {overlap}."
        )
    if overlap >= target:
        raise InvalidArgumentError(
            f"overlap ({overlap}) must be smaller than target_size ({target}); "
            "an overlap this large would repeat whole sections."
        )
    note = (
        "Inherited from the knowledge base's recorded chunking configuration."
        if kb is not None and kb.chunking_strategy == name and not proposed
        else "Proposed for this build; recorded on the knowledge base when the build runs."
    )
    return {
        "chunker": name,
        "target_size": target,
        "overlap": overlap,
        "available_chunkers": available,
        "note": note,
    }


def resolve_retrieval(request: dict[str, Any], kb: KnowledgeBase | None, configured: dict | None) -> dict[str, Any]:
    """Retrieval strategy + persisted config. Validates against the registry."""
    from app.services.retrieval.retriever import available_retrievers

    available = available_retrievers()
    proposed = dict(request.get("retrieval") or {})
    strategy = str(proposed.get("strategy") or "dense").strip().lower()
    if strategy not in available:
        raise InvalidArgumentError(
            f"Unknown retrieval strategy {strategy!r}.",
            hint=f"Registered strategies: {', '.join(available)}.",
        )
    return {
        "strategy": strategy,
        "available_strategies": available,
        "configured": dict(configured or {}),
        "note": (
            "The plan records the strategy to test after the build; retrieval behaviour itself is "
            "unchanged and comes from the retrieval registry."
        ),
    }


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

VALIDATION_PLAN: tuple[str, ...] = (
    "validate_knowledge_base runs the read-only corpus integrity audit (missing/orphan/stale "
    "vectors, duplicate content, provenance gaps, failed or partial documents).",
    "Readiness requires measured evidence: parsed documents, chunk rows, and vector points that "
    "match the chunk rows. A succeeded build alone is not readiness.",
    "A retrieval diagnostic is only permitted when a benchmark instrument exists for the knowledge "
    "base, and exploratory instruments are never used for confirmatory claims.",
)


def build_creation_plan(
    *,
    request: dict[str, Any],
    existing_kbs: list[KnowledgeBase],
    planned_urls: list[str],
) -> PlanDraft:
    """Plan the creation of a new knowledge base. Writes nothing."""
    name = str(request.get("name") or "").strip()
    domain = str(request.get("domain") or "").strip()
    purpose = str(request.get("purpose") or "").strip()
    audience = str(request.get("target_audience") or "").strip()
    # The creation API requires all four of these (min_length=1). Checking here
    # means a plan can never propose parameters its own execution would reject.
    missing = [
        label
        for label, value in (("name", name), ("domain", domain), ("purpose", purpose), ("target_audience", audience))
        if not value
    ]
    if missing:
        raise InvalidArgumentError(
            f"Creating a knowledge base needs {', '.join(missing)}; "
            f"{len(missing)} required field(s) are empty.",
            hint=(
                "Provide name, domain, purpose (the intended use case) and target_audience. "
                "They are stored on the knowledge base and used by every later step."
            ),
        )

    parameters = {
        "name": name,
        "domain": domain,
        "purpose": purpose,
        "target_audience": audience,
        "depth": str(request.get("depth") or "intermediate").strip() or "intermediate",
        "source_mode": str(request.get("source_mode") or SourceMode.EXTERNAL.value),
    }
    allowed_modes = {m.value for m in SourceMode}
    if parameters["source_mode"] not in allowed_modes:
        raise InvalidArgumentError(
            f"source_mode must be one of {sorted(allowed_modes)}; got {parameters['source_mode']!r}."
        )

    warnings: list[str] = []
    same_name = [kb for kb in existing_kbs if kb.name.strip().lower() == name.lower()]
    duplicate = next((kb for kb in same_name if kb.domain.strip().lower() == domain.lower()), None)
    if duplicate is not None:
        warnings.append(
            f"Knowledge base {duplicate.id} already has this name and domain "
            f"(status {duplicate.status.value}, {duplicate.version} version(s)). Creating another is "
            "usually an accident; reuse that id or change the name."
        )

    draft = PlanDraft(
        mode="create",
        request=dict(request),
        kb_parameters=parameters,
        kb_id=None,
        proposed_corpus=preview_user_urls(planned_urls),
        ingestion_plan=[
            "1. create_knowledge_base (this plan) creates the container only: no sources, no documents.",
            "2. analyze_domain proposes a DomainSpec; persist it through the API if the user wants it stored.",
            "3. discover_sources registers candidate sources and scores them with the real quality engine.",
            "4. assess_sources shows every score, reason and warning so the user can accept or reject.",
            "5. plan_knowledge_base again on the new kb_id: that plan proposes the ingest and build actions.",
        ],
        chunking=resolve_chunking(request, None),
        retrieval=resolve_retrieval(request, None, None),
        validation_plan=list(VALIDATION_PLAN),
        warnings=warnings,
        current_state={
            "existing_knowledge_bases": len(existing_kbs),
            "same_name_knowledge_bases": [kb.id for kb in same_name],
        },
    )
    draft.actions.append(
        ActionDraft(
            action="create_knowledge_base",
            parameters=parameters,
            summary=f"Create knowledge base {name!r} ({domain}) with source_mode={parameters['source_mode']}",
            warnings=list(warnings),
        )
    )
    return draft


def build_extension_plan(
    *,
    kb: KnowledgeBase,
    sources: list[Source],
    documents: list[Document],
    request: dict[str, Any],
    retrieval_config: dict | None,
    described_documents: list[dict[str, Any]] | None = None,
) -> PlanDraft:
    """Plan the next actions for an existing knowledge base, from measured state."""
    accepted_external = [
        s
        for s in sources
        if s.decision == SourceDecision.ACCEPT and not s.user_provided
    ]
    review = [s for s in sources if s.decision == SourceDecision.REVIEW]
    rejected = [s for s in sources if s.decision == SourceDecision.REJECT]
    user_sources = [s for s in sources if s.user_provided]

    ingested_source_ids = {d.source_id for d in documents}
    pending_ingest = sorted(
        (s for s in accepted_external if s.id not in ingested_source_ids), key=lambda s: s.id
    )
    ready_documents = [d for d in documents if not d.user_provided]
    user_documents = [d for d in documents if d.user_provided]

    # The caller injects the KB's DomainSpec (when one exists) so the plan can
    # echo the real knowledge requirements instead of inventing a knowledge map.
    spec = request.get("_domain_spec")

    chunking = resolve_chunking(request, kb)
    retrieval = resolve_retrieval(request, kb, retrieval_config)

    draft = PlanDraft(
        mode="extend",
        request={k: v for k, v in request.items() if not k.startswith("_")},
        kb_id=kb.id,
        domain_spec_present=bool(spec),
        knowledge_requirements=[
            {"area": r.area, "description": r.description, "priority": str(getattr(r.priority, "value", r.priority))}
            for r in (getattr(spec, "knowledge_requirements", []) or [])
        ]
        if spec
        else [],
        source_categories=list(getattr(spec, "recommended_source_categories", []) or []) if spec else [],
        proposed_documents=described_documents or [],
        current_state={
            "status": kb.status.value,
            "version": kb.version,
            "sources": {
                "total": len(sources),
                "accepted_external": len(accepted_external),
                "review": len(review),
                "rejected": len(rejected),
                "user_provided": len(user_sources),
                "accepted_not_yet_ingested": len(pending_ingest),
            },
            "documents": {
                "total": len(documents),
                "from_external_sources": len(ready_documents),
                "user_provided": len(user_documents),
                "statuses": _status_counts(documents),
            },
            "chunk_rows": None,
        },
        chunking=chunking,
        retrieval=retrieval,
        validation_plan=list(VALIDATION_PLAN),
    )

    # -- source discovery / assessment guidance -------------------------------
    if not sources:
        draft.ingestion_plan.append(
            "No sources are registered yet: call discover_sources (user-url for URLs the user "
            "supplied, arxiv for research queries), then assess_sources before ingesting anything."
        )
    if review:
        draft.warnings.append(
            f"{len(review)} source(s) are REVIEW and will NOT be ingested through MCP: "
            + ", ".join(s.id for s in review[:5])
            + ". A human decision is required (the API/UI records it)."
        )
    if rejected:
        draft.ingestion_plan.append(
            f"{len(rejected)} source(s) were REJECTED by the quality engine and are excluded from every plan."
        )

    # -- ingest action ---------------------------------------------------------
    if pending_ingest:
        parameters = {"kb_id": kb.id, "source_ids": sorted(s.id for s in pending_ingest)}
        draft.ingestion_plan.append(
            f"{len(pending_ingest)} ACCEPTED external source(s) would be downloaded and parsed: "
            + ", ".join(s.id for s in pending_ingest[:5])
            + ("…" if len(pending_ingest) > 5 else "")
        )
        draft.actions.append(
            ActionDraft(
                action="ingest_sources",
                parameters=parameters,
                summary=f"Download and parse {len(pending_ingest)} accepted source(s) into {kb.id}",
                warnings=[
                    "This performs outbound HTTP requests to the registered source URLs "
                    "(SSRF-validated, size-capped, one failure never aborts the batch)."
                ],
            )
        )
    elif accepted_external:
        draft.ingestion_plan.append(
            "Every ACCEPTED source is already ingested; ingestion would be a no-op."
        )
    else:
        draft.ingestion_plan.append(
            "No ACCEPTED external source is available to ingest."
        )

    # -- document intake action -----------------------------------------------
    if described_documents:
        valid = [d for d in described_documents if d["valid"]]
        invalid = [d for d in described_documents if not d["valid"]]
        index_documents = bool(request.get("index_documents", True))
        if invalid:
            draft.warnings.append(
                f"{len(invalid)} document(s) fail upload validation and will be rejected when the "
                "intake runs: "
                + "; ".join(f"{d['file_name']}: {d['error']}" for d in invalid[:3])
            )
        if described_documents:
            # The parameters cover the WHOLE submission, including documents that
            # validation will reject: the confirmation must bind exactly what the
            # caller is about to send, not a subset this server pre-filtered.
            parameters = document_action_parameters(
                kb.id, described_documents, chunking=chunking, index=index_documents
            )
            draft.actions.append(
                ActionDraft(
                    action="add_user_documents",
                    parameters=parameters,
                    summary=(
                        f"Add {len(valid)} valid user-provided document(s) "
                        f"({sum(d['size_bytes'] for d in valid)} bytes) to {kb.id}"
                        + (f"; {len(invalid)} would be rejected" if invalid else "")
                        + ("" if index_documents else "; without indexing")
                    ),
                    warnings=[
                        "Document bytes are parsed and embedded locally; no external LLM call is made "
                        "for user-provided content.",
                        "Content is bound by sha256: the confirmed fingerprint covers exactly these bytes.",
                    ],
                )
            )

    # -- build action ----------------------------------------------------------
    indexable = [d for d in documents if d.user_provided or d.source_id in ingested_source_ids]
    already_indexed = [d for d in indexable if int(d.chunk_count or 0) > 0]
    build_parameters = {
        "kb_id": kb.id,
        "chunker": chunking["chunker"],
        "target_size": chunking["target_size"],
        "overlap": chunking["overlap"],
    }
    if indexable:
        draft.actions.append(
            ActionDraft(
                action="build_knowledge_base",
                parameters=build_parameters,
                summary=(
                    f"Chunk, embed and index {len(indexable)} document(s) with {chunking['chunker']} "
                    f"({chunking['target_size']}/{chunking['overlap']})"
                ),
                already_satisfied=False,
                warnings=(
                    [
                        "This is a full rebuild: vectors of the re-indexed documents are deleted before "
                        "the new ones are upserted, then orphans are swept. Retrieval during the build "
                        "can see a partially rebuilt index."
                    ]
                    if already_indexed
                    else []
                ),
            )
        )
    else:
        draft.ingestion_plan.append(
            "No documents are ingested yet, so no build action is proposed. Ingest first."
        )

    return draft


def _status_counts(documents: list[Document]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in documents:
        key = str(getattr(doc.status, "value", doc.status))
        counts[key] = counts.get(key, 0) + 1
    return counts


def confirmation_instructions(plan_id: str) -> list[str]:
    """How to turn this plan into an executable operation. Shown to the user."""
    return [
        "1. Show the plan and its action fingerprints to the user; nothing has changed yet.",
        "2. The user approves the exact action and parameters they want.",
        "3. Call confirm_plan with this plan_id, the action, and the action's parameters_fingerprint. "
        "This records a single-use confirmation bound to those parameters.",
        "4. Call the action tool with the confirmation id and the confirmed parameters, byte for byte.",
        "Changing any parameter invalidates the confirmation: plan and confirm again.",
        "Confirmations expire and cannot be reused; a rejected or failed execution must be re-planned.",
    ]


__all__ = [
    "ActionDraft",
    "PlanDraft",
    "CHUNK_OVERLAP_RANGE",
    "CHUNK_TARGET_RANGE",
    "SELECTION_RULES",
    "VALIDATION_PLAN",
    "build_creation_plan",
    "build_extension_plan",
    "confirmation_instructions",
    "decode_document_payload",
    "describe_document",
    "document_action_parameters",
    "preview_user_urls",
    "resolve_chunking",
    "resolve_retrieval",
    "sha256_bytes",
]
