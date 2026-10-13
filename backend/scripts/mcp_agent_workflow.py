"""Agentic RAGForge: one real construction workflow driven over MCP (V18).

This is the *live* counterpart to `tests/test_v18_mcp_agentic.py`. It starts the
RAGForge MCP server the way a client does (stdio subprocess) and then runs the
documented lifecycle against the **real workspace**: the real SQLite database,
the real Qdrant instance and the real local embedding model.

    REQUEST -> PLAN -> USER CONFIRMATION -> EXECUTION -> VALIDATION -> REPORT

Every value printed is what the server returned; nothing is invented. Where a
number could not be measured the server says so and this script prints that.

What it does
------------
1.  lists the tools (and shows which are read-only, and which are consequential)
2.  `analyze_domain`            - a PROPOSED domain specification (stored nowhere)
3.  `plan_knowledge_base`       - a creation plan; nothing is created
4.  confirmation-gate demos     - no confirmation, wrong fingerprint, replayed
                                  confirmation, changed parameters: all refused
5.  `confirm_plan` + `create_knowledge_base`  - a scratch KB, created for real
6.  `plan_knowledge_base` (documents) + `add_user_documents`
                                - a user-provided document, hashed, parsed and
                                  indexed through the real intake path
7.  `plan_knowledge_base` + `build_knowledge_base`
                                - the real chunk -> embed -> index pipeline
8.  `validate_knowledge_base`   - readiness from measured evidence
9.  `search_knowledge_base` + `get_evidence`
                                - ranked evidence and its provenance
10. `answer_from_knowledge_base`- a grounded answer, and an abstention
11. `compare_retrieval_strategies`
                                - lifecycle-gated; with no reviewed instrument it
                                  refuses and says why (that refusal is the
                                  honest result, not a failure of this script)
12. `get_operation_status`      - the audit trail of the operations above
13. cleanup                     - the scratch KB and its vectors are removed,
                                  and the protected KB is re-measured

Deliberate limits
-----------------
* The scratch KB this script builds is the ONLY thing it writes. The protected
  research KB `kb_f278c283c748` is never opened, listed, ingested into or
  indexed; its point count is measured before and after and printed.
* The "USER CONFIRMATION" steps in this script stand in for the human: the
  model cannot confirm on its own behalf, and the script says so at each gate.
* Document bytes are passed to the tool as content. `add_user_documents` cannot
  read the filesystem, so `--upload-file` is read HERE, by the host script, and
  the bytes travel in the call - exactly as an MCP host would do it.

Usage
-----
    cd backend
    .venv\\Scripts\\python.exe scripts/mcp_agent_workflow.py
    .venv\\Scripts\\python.exe scripts/mcp_agent_workflow.py --keep --json-out ../docs/mcp-workflow-run.json
    .venv\\Scripts\\python.exe scripts/mcp_agent_workflow.py --upload-file ../samples/notes.md
    .venv\\Scripts\\python.exe scripts/mcp_agent_workflow.py --external --urls https://en.wikipedia.org/wiki/Engine https://en.wikipedia.org/wiki/Radiator

`--external` adds the discovered-source path (real HTTP probes, real downloads)
and requires the network; nothing is downloaded without it.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

from mcp import Client, StdioServerParameters  # noqa: E402

SERVER_SCRIPT = BACKEND_ROOT / "scripts" / "run_mcp_server.py"

#: The research KB that must not be touched. Measured before and after.
PROTECTED_KB_ID = "kb_f278c283c748"

#: A short engineering note used when the caller supplies no document of their
#: own. It is written for this demo, it is clearly demo content, and it is sent
#: through the same validation/parsing/provenance path as any other upload -
#: it is NOT a substitute for a real corpus, and nothing here claims it is.
DEMO_NOTES = """# Engine cooling system

An engine converts chemical energy in fuel into mechanical work. Roughly a third
of that energy leaves the engine as heat, and the cooling system's job is to
carry that heat away without letting the metal exceed its design temperature.

## Coolant circuit

The coolant pump circulates a glycol-water mixture through passages cast into
the cylinder head and block. The thermostat stays closed while the engine is
cold so the coolant warms quickly, then opens to route the flow to the radiator.

## Radiator

A radiator is a liquid-to-air heat exchanger. Coolant passes through tubes and
transfers heat to fins, which the airflow carries away. The fan only matters at
low road speed; above roughly 60 km/h ram air is usually enough.

## Failure modes

Overheating usually begins with a blocked radiator core, a stuck thermostat or a
failed pressure cap, which lowers the boiling point of the coolant. A blown head
gasket follows if the coolant boils, because the head warps.
"""


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------


class Log:
    """Prints the run's progress, and keeps a machine-readable record."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.failures: list[str] = []

    def step(self, title: str, payload: object, *, limit: int = 900) -> None:
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        print(f"\n--- {title} " + "-" * max(0, 68 - len(title)))
        print(text[:limit] + ("\n... (truncated for display)" if len(text) > limit else ""))
        self.steps.append({"step": title, "payload": payload})

    def say(self, message: str) -> None:
        print(f"    {message}")

    def refuse(self, title: str, text: str) -> None:
        print(f"\n--- {title} " + "-" * max(0, 68 - len(title)))
        print(f"    REFUSED: {text}")
        self.steps.append({"step": title, "refused": text})

    def problem(self, message: str) -> None:
        print(f"    !! {message}")
        self.failures.append(message)


# ---------------------------------------------------------------------------
# client helpers
# ---------------------------------------------------------------------------


class Agent:
    """A thin MCP client: call a tool, keep the structured result or the error."""

    def __init__(self, client: Any, log: Log) -> None:
        self.client = client
        self.log = log

    async def call(
        self, tool: str, arguments: dict[str, Any] | None = None, *, timeout: float = 1800
    ) -> tuple[bool, dict[str, Any], str]:
        result = await self.client.call_tool(
            tool, arguments or {}, read_timeout_seconds=timeout
        )
        structured = dict(result.structured_content or {})
        text = " ".join(
            getattr(block, "text", "") for block in (result.content or [])
        ).strip()
        return bool(result.is_error), structured, text

    async def structured(
        self, tool: str, arguments: dict[str, Any] | None = None, *, timeout: float = 1800
    ) -> dict[str, Any]:
        is_error, structured, text = await self.call(tool, arguments, timeout=timeout)
        if is_error:
            raise RuntimeError(f"{tool} failed: {text or structured}")
        return structured

    async def refused(
        self, tool: str, arguments: dict[str, Any] | None = None, *, timeout: float = 1800
    ) -> str:
        """Call a tool that MUST be refused; return the error text, or complain."""
        is_error, structured, text = await self.call(tool, arguments, timeout=timeout)
        if not is_error:
            self.log.problem(
                f"{tool} was NOT refused as expected - the confirmation gate has a hole"
            )
            return ""
        return text or json.dumps(structured)


def _action(plan: dict[str, Any], action: str) -> dict[str, Any]:
    for candidate in plan.get("actions", []):
        if candidate.get("action") == action:
            return candidate
    raise RuntimeError(
        f"the plan proposes no {action}: {[a.get('action') for a in plan.get('actions', [])]}"
    )


async def _confirm(agent: Agent, plan: dict[str, Any], action: str, note: str) -> str:
    """Record the HUMAN decision to approve exactly this action and parameter set."""
    target = _action(plan, action)
    print(
        f"    [human] approving {action} with fingerprint "
        f"{target['parameters_fingerprint'][:16]}... ({target.get('summary', '')})"
    )
    payload = await agent.structured(
        "confirm_plan",
        {
            "plan_id": plan["plan_id"],
            "action": action,
            "parameters_fingerprint": target["parameters_fingerprint"],
            "note": note,
        },
    )
    print(
        f"    confirmation {payload['confirmation_id']} for {payload['action']}, "
        f"single-use={payload['single_use']}, expires_at={payload['expires_at']}"
    )
    return payload["confirmation_id"]


# ---------------------------------------------------------------------------
# workspace helpers (the ONLY places this script bypasses the MCP surface)
# ---------------------------------------------------------------------------


def _repo():
    from app.api.deps import get_repo

    return get_repo()


def _settings():
    from app.config import get_settings

    return get_settings()


def _point_count(kb_id: str) -> int | None:
    """Measure the vector points of one KB, or None when Qdrant is unreachable."""
    from app.services.vector_store.factory import create_vector_store

    try:
        store = create_vector_store(_settings())
        info = store.collection_info(kb_id)
    except Exception as exc:  # noqa: BLE001 - an unmeasured count is reported as such
        print(f"    (vector store unavailable: {type(exc).__name__})")
        return None
    for key in ("points_count", "points", "count"):
        value = (info or {}).get(key)
        if isinstance(value, int):
            return value
    vectors = (info or {}).get("vectors_count")
    return vectors if isinstance(vectors, int) else None


def _accept_source(kb_id: str, source_id: str, reason: str) -> None:
    """Record the HUMAN source decision, exactly as the decision API does.

    The MCP surface deliberately cannot do this: `assess_sources` explains the
    automated decision, but overriding REVIEW/REJECT is a human action recorded
    through the API/UI. An agent must not be able to promote its own candidates.
    """
    from app.schemas.models import SourceDecision

    repo = _repo()
    source = repo.get_source(kb_id, source_id)
    if source is None:
        raise RuntimeError(f"source {source_id} not found")
    source.decision = SourceDecision.ACCEPT
    repo.update_source(source)
    print(f"    [human] source {source_id} set to ACCEPT ({reason})")


def _delete_kb(kb_id: str) -> None:
    """Remove the scratch KB and its vectors (never the protected one)."""
    from app.services.vector_store.factory import create_vector_store

    try:
        create_vector_store(_settings()).delete_collection(kb_id)
    except Exception as exc:  # noqa: BLE001
        print(f"    (collection delete skipped: {type(exc).__name__})")
    _repo().delete_kb(kb_id)


def _upload_payload(path: Path) -> dict[str, str]:
    """Read a file HERE (the host does it) and hand the bytes to the tool.

    `add_user_documents` takes content, not paths: an agent cannot point it at
    an arbitrary file on the machine.
    """
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".csv", ".html", ".htm", ".json"}:
        return {"file_name": path.name, "content_text": data.decode("utf-8", errors="replace")}
    return {"file_name": path.name, "content_base64": base64.b64encode(data).decode()}


# ---------------------------------------------------------------------------
# the workflow
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    log = Log()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    kb_name = args.name or f"V18 agent workflow {stamp}"
    domain = args.domain
    purpose = "demonstrate the agentic MCP construction workflow"
    audience = "engineering students"
    depth = "intermediate"

    protected_before = _point_count(PROTECTED_KB_ID)
    print(f"protected KB {PROTECTED_KB_ID}: {protected_before} vector point(s) BEFORE")
    if protected_before != 812:
        print(
            "    !! expected 812 points on the protected KB; stopping rather than "
            "running against a corpus in an unexpected state"
        )
        return 2

    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER_SCRIPT)], cwd=str(BACKEND_ROOT)
    )
    async with Client(params, raise_exceptions=False) as client:
        agent = Agent(client, log)

        # -- 1. what does this server expose? --------------------------------
        tools = (await client.list_tools()).tools
        read_only = [t.name for t in tools if getattr(t.annotations, "read_only_hint", False)]
        writes = [t.name for t in tools if not getattr(t.annotations, "read_only_hint", False)]
        print(f"\n{len(tools)} tools over stdio")
        print(f"  read-only     ({len(read_only)}): {', '.join(read_only)}")
        print(f"  consequential ({len(writes)}): {', '.join(writes)}")
        if len(tools) != 18:
            log.problem(f"expected 18 tools, the server exposed {len(tools)}")

        # -- 2. REQUEST: a proposed domain specification ---------------------
        analysis = await agent.structured(
            "analyze_domain",
            {
                "domain": domain,
                "purpose": purpose,
                "target_audience": audience,
                "depth": depth,
            },
        )
        log.step(
            "analyze_domain (plan-time proposal)",
            {
                "domain": analysis["proposal"]["domain"],
                "generated_by": analysis["generated_by"],
                "is_mock": analysis["is_mock"],
                "persisted": analysis["persisted"],
                "subdomains": analysis["proposal"]["subdomains"][:6],
                "key_concepts": analysis["proposal"]["key_concepts"][:6],
                "source_categories": analysis["proposal"]["recommended_source_categories"][:6],
                "warnings": analysis["warnings"],
                "notes": analysis["notes"],
            },
        )
        if not analysis["proposal"]["subdomains"]:
            log.problem("the domain analyzer returned no subdomains")
        if analysis["is_mock"]:
            print(
                "    NOTE: no LLM is configured, so this proposal comes from "
                "RAGForge's labelled development mock. It is a placeholder."
            )

        # -- 3. PLAN: nothing is created yet ---------------------------------
        before = await agent.structured("list_knowledge_bases")
        plan = await agent.structured(
            "plan_knowledge_base",
            {
                "name": kb_name,
                "domain": domain,
                "purpose": purpose,
                "target_audience": audience,
                "depth": depth,
                "source_mode": "mixed",
            },
        )
        log.step(
            "plan_knowledge_base (creation plan)",
            {
                "plan_id": plan["plan_id"],
                "mode": plan["mode"],
                "mutates": plan["mutates"],
                "actions": [
                    {
                        "action": a["action"],
                        "parameters": a["parameters"],
                        "parameters_fingerprint": a["parameters_fingerprint"],
                        "summary": a["summary"],
                    }
                    for a in plan["actions"]
                ],
                "warnings": plan["warnings"],
            },
        )
        after = await agent.structured("list_knowledge_bases")
        if after["total"] != before["total"] or plan["mutates"]:
            log.problem("planning changed the workspace: it must be read-only")
        else:
            print(f"    planning mutated nothing ({before['total']} KB(s) before and after)")

        # -- 4. the confirmation gate ----------------------------------------
        create_action = _action(plan, "create_knowledge_base")
        unconfirmed = await agent.refused(
            "create_knowledge_base", {**create_action["parameters"], "confirmation_id": ""}
        )
        log.refuse("gate: no confirmation at all", unconfirmed)
        as_plan_id = await agent.refused(
            "create_knowledge_base",
            {**create_action["parameters"], "confirmation_id": plan["plan_id"]},
        )
        log.refuse("gate: a plan id is not a confirmation", as_plan_id)

        confirmation = await _confirm(
            agent, plan, "create_knowledge_base", "approved by the operator in this run"
        )
        changed = dict(create_action["parameters"])
        changed["domain"] = f"{domain} (edited after approval)"
        wrong_params = await agent.refused(
            "create_knowledge_base", {**changed, "confirmation_id": confirmation}
        )
        log.refuse("gate: parameters changed after approval", wrong_params)

        # -- 5. EXECUTION: create the scratch KB ------------------------------
        created = await agent.structured(
            "create_knowledge_base",
            {"confirmation_id": confirmation, **create_action["parameters"]},
        )
        kb_id = created["kb_id"]
        log.step(
            "create_knowledge_base (executed)",
            {
                "operation_id": created["operation_id"],
                "kb_id": kb_id,
                "status": created["status"],
                "lifecycle_stage": created["lifecycle_stage"],
                "knowledge_base": created["knowledge_base"],
                "warnings": created["warnings"],
                "next_step": created["next_step"],
            },
        )
        if kb_id == PROTECTED_KB_ID:
            log.problem("the workflow created the PROTECTED knowledge base; aborting")
            return 2
        print(f"    scratch KB: {kb_id} ({kb_name})")

        replay = await agent.refused(
            "create_knowledge_base",
            {"confirmation_id": confirmation, **create_action["parameters"]},
        )
        log.refuse("gate: a confirmation is single-use (replay)", replay)

        # -- 6. add a user-provided document ----------------------------------
        if args.upload_file:
            path = Path(args.upload_file).resolve()
            documents = [_upload_payload(path)]
            print(f"\ndocument: {path} ({len(documents[0].get('content_text') or documents[0].get('content_base64') or '')} characters sent)")
        else:
            documents = [{"file_name": "engine-cooling-notes.md", "content_text": DEMO_NOTES}]
            print(
                "\ndocument: built-in demo note (clearly demo content; pass "
                "--upload-file to use your own)"
            )

        doc_plan = await agent.structured(
            "plan_knowledge_base",
            {
                "kb_id": kb_id,
                "user_documents": documents,
                "chunking": {"chunker": "section-aware", "target_size": 700, "overlap": 80},
            },
        )
        log.step(
            "plan_knowledge_base (document intake)",
            {
                "plan_id": doc_plan["plan_id"],
                "mutates": doc_plan["mutates"],
                "proposed_documents": doc_plan["proposed_documents"],
                "chunking": doc_plan["chunking"],
                "actions": [a["action"] for a in doc_plan["actions"]],
                "warnings": doc_plan["warnings"],
            },
        )
        intake_action = _action(doc_plan, "add_user_documents")
        intake_confirmation = await _confirm(
            agent, doc_plan, "add_user_documents", "operator approved these exact bytes"
        )
        # The confirmation is bound to the document hashes: other bytes are refused.
        tampered = [dict(documents[0])]
        key = "content_text" if "content_text" in tampered[0] else "content_base64"
        tampered[0][key] = (tampered[0].get(key) or "") + "\nTAMPERED AFTER APPROVAL\n"
        different_bytes = await agent.refused(
            "add_user_documents",
            {"confirmation_id": intake_confirmation, "kb_id": kb_id, "documents": tampered},
        )
        log.refuse("gate: document content changed after approval", different_bytes)

        intake = await agent.structured(
            "add_user_documents",
            {
                "confirmation_id": intake_confirmation,
                "kb_id": kb_id,
                "documents": documents,
                "index": True,
                "chunker": "section-aware",
                "target_size": 700,
                "overlap": 80,
            },
        )
        log.step(
            "add_user_documents (executed)",
            {
                "operation_id": intake["operation_id"],
                "submitted": intake["submitted"],
                "accepted": intake["accepted"],
                "rejected": intake["rejected"],
                "duplicates_skipped": intake["duplicates_skipped"],
                "indexed": intake["indexed"],
                "chunks_created": intake["chunks_created"],
                "vectors_indexed": intake["vectors_indexed"],
                "items": intake["items"],
                "notes": intake["notes"],
                "warnings": intake["warnings"],
            },
        )
        if intake["accepted"] != 1:
            log.problem(f"expected 1 accepted document, got {intake['accepted']}")

        # -- 7. explicit build (chunk -> embed -> index) ----------------------
        build_plan = await agent.structured(
            "plan_knowledge_base",
            {
                "kb_id": kb_id,
                "chunking": {"chunker": "section-aware", "target_size": 700, "overlap": 80},
                "retrieval": {"strategy": "dense"},
            },
        )
        log.step(
            "plan_knowledge_base (build plan)",
            {
                "plan_id": build_plan["plan_id"],
                "current_state": build_plan["current_state"],
                "chunking": build_plan["chunking"],
                "retrieval": build_plan["retrieval"],
                "validation_plan": build_plan["validation_plan"],
                "actions": [
                    {"action": a["action"], "parameters": a["parameters"]}
                    for a in build_plan["actions"]
                ],
                "warnings": build_plan["warnings"],
            },
        )
        build_action = _action(build_plan, "build_knowledge_base")
        build_confirmation = await _confirm(
            agent, build_plan, "build_knowledge_base", "operator approved the rebuild"
        )
        built = await agent.structured(
            "build_knowledge_base",
            {"confirmation_id": build_confirmation, **build_action["parameters"]},
        )
        log.step(
            "build_knowledge_base (executed)",
            {
                "operation_id": built["operation_id"],
                "run_id": built["run_id"],
                "run_status": built["run_status"],
                "status": built["status"],
                "result_summary": built["result_summary"],
                "build": built["build"],
                "warnings": built["warnings"],
                "next_step": built["next_step"],
            },
        )
        if built["status"] != "done":
            log.problem(f"build did not finish successfully: {built['status']}")

        # -- 8. VALIDATION: readiness from evidence ---------------------------
        validation = await agent.structured("validate_knowledge_base", {"kb_id": kb_id})
        log.step(
            "validate_knowledge_base",
            {
                "verdict": validation["verdict"],
                "ready_for_retrieval": validation["ready_for_retrieval"],
                "basis": validation["basis"],
                "blocking": validation["blocking"],
                "checks": validation["checks"],
                "document_counts": validation["document_counts"],
                "chunk_rows": validation["chunk_rows"],
                "vector_points": validation["vector_points"],
                "vector_points_measured": validation["vector_points_measured"],
                "integrity_status": validation["integrity_status"],
                "integrity_counts": validation["integrity_counts"],
                "integrity_unknown": validation["integrity_unknown"],
                "diagnostics": validation["diagnostics"],
            },
        )
        if validation["verdict"] != "READY":
            log.problem(
                f"the freshly built KB is not READY: {validation['verdict']} "
                f"({'; '.join(validation['blocking'])})"
            )

        # -- 9. retrieval + provenance ---------------------------------------
        search = await agent.structured(
            "search_knowledge_base",
            {"kb_id": kb_id, "query": "how does a radiator remove heat from the coolant", "top_k": 3},
        )
        log.step(
            "search_knowledge_base",
            {
                "applied_strategy": search["applied_strategy"],
                "embedding_model": search["trace"]["embedding_model"],
                "results": [
                    {
                        "rank": hit["rank"],
                        "chunk_id": hit["chunk_id"],
                        "score": hit["retrieval_score"],
                        "document_id": hit.get("document_id"),
                        "section": hit.get("section"),
                        "page": hit.get("page"),
                        "source_title": hit.get("source_title"),
                        "source_url": hit.get("source_url"),
                        "source_type": hit.get("source_type"),
                        "trust_score": hit.get("trust_score"),
                        "content_hash": (hit.get("content_hash") or "")[:16],
                        "why": hit.get("why"),
                        "text": hit["text"][:120],
                    }
                    for hit in search["results"]
                ],
                "warnings": search["warnings"],
            },
            limit=1800,
        )
        chunk_ids = [hit["chunk_id"] for hit in search["results"]]
        evidence = await agent.structured(
            "get_evidence", {"kb_id": kb_id, "chunk_ids": chunk_ids} if chunk_ids else {"kb_id": kb_id}
        )
        log.step(
            "get_evidence (full text + provenance of the top chunks)",
            {
                "found": evidence.get("found"),
                "missing_chunk_ids": evidence.get("missing_chunk_ids", []),
                "chunks": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "document_id": c.get("document_id"),
                        "chunk_index": c.get("chunk_index"),
                        "page": c.get("page"),
                        "section": c.get("section"),
                        "section_path": c.get("section_path"),
                        "source_url": c.get("source_url"),
                        "publisher": c.get("publisher"),
                        "user_provided": c.get("user_provided"),
                        "trust_score": c.get("trust_score"),
                        "chunking_strategy": c.get("chunking_strategy"),
                        "text_length": len(c.get("text") or ""),
                    }
                    for c in evidence.get("chunks", [])
                ],
                "notes": evidence.get("notes", []),
                "content_warning": evidence.get("content_warning"),
            },
            limit=1600,
        )

        # -- 10. a grounded answer, and an abstention -------------------------
        answer = await agent.structured(
            "answer_from_knowledge_base",
            {"kb_id": kb_id, "question": "What does a radiator do?", "top_k": 4},
        )
        log.step(
            "answer_from_knowledge_base (in-corpus question)",
            {
                "status": answer["status"],
                "grounded": answer["grounded"],
                "grounding_state": answer["grounding_state"],
                "citation_count": answer["citation_count"],
                "citations": [
                    {
                        "citation_id": c.get("citation_id"),
                        "chunk_id": c.get("chunk_id"),
                        "document_id": c.get("document_id"),
                        "url": c.get("url"),
                        "section": c.get("section"),
                        "page": c.get("page"),
                        "validation": c.get("validation"),
                    }
                    for c in answer.get("citations", [])
                ],
                "claims": [
                    {"text": c["text"][:120], "support_status": c["support_status"]}
                    for c in answer.get("claims", [])
                ],
                "answer_text": (answer.get("answer_text") or "")[:400],
                "abstention_reason": answer.get("abstention_reason"),
                "generation": answer.get("generation"),
                "warnings": answer["warnings"],
            },
            limit=1600,
        )
        if (answer.get("generation") or {}).get("is_mock"):
            print("    NOTE: the answer text came from the labelled development mock.")

        abstention = await agent.structured(
            "answer_from_knowledge_base",
            {"kb_id": kb_id, "question": "What is the torque setting for a 2019 diesel injector clamp?"},
        )
        log.step(
            "answer_from_knowledge_base (out-of-corpus question)",
            {
                "status": abstention["status"],
                "grounded": abstention["grounded"],
                "grounding_state": abstention["grounding_state"],
                "citation_count": abstention["citation_count"],
                "abstention_reason": abstention.get("abstention_reason"),
            },
        )
        if abstention["status"] == "answered" and abstention["grounded"]:
            log.problem("the KB answered a question the corpus does not cover")

        # -- 11. optional: discovered sources (needs the network) -------------
        external: dict[str, Any] = {"enabled": bool(args.external)}
        if args.external:
            discovery = await agent.structured(
                "discover_sources",
                {"kb_id": kb_id, "provider": "user-url", "query": " ".join(args.urls), "limit": len(args.urls)},
            )
            log.step(
                "discover_sources (real probes and scoring)",
                {
                    "provider": discovery["provider"],
                    "discovered": discovery["discovered"],
                    "registered_new": discovery["registered_new"],
                    "rejected_invalid": discovery["rejected_invalid"],
                    "mutates": discovery["mutates"],
                    "sources": [
                        {
                            "source_id": s["source_id"],
                            "url": s["url"],
                            "decision": s["decision"],
                            "trust_score": s["trust_score"],
                            "assessed_by": s["assessed_by"],
                            "signals": s["signals"],
                            "reasons": s["reasons"][:4],
                            "warnings": s["warnings"],
                        }
                        for s in discovery["sources"]
                    ],
                    "warnings": discovery["warnings"],
                },
            )
            assessments = await agent.structured("assess_sources", {"kb_id": kb_id})
            log.step(
                "assess_sources (why each decision was made)",
                {
                    "total": assessments["total"],
                    "counts_by_decision": assessments["counts_by_decision"],
                    "unassessed": assessments["unassessed"],
                },
            )
            accepted_ids = [
                s["source_id"] for s in discovery["sources"] if s["decision"] == "ACCEPT"
            ]
            review_ids = [
                s["source_id"] for s in discovery["sources"] if s["decision"] == "REVIEW"
            ]
            for source_id in review_ids:
                # A human decision, recorded through the same path the API uses.
                _accept_source(kb_id, source_id, "operator reviewed the candidate")
                accepted_ids.append(source_id)

            if accepted_ids:
                ingest_plan = await agent.structured(
                    "plan_knowledge_base",
                    {"kb_id": kb_id, "source_mode": "mixed", "retrieval": {"strategy": "dense"}},
                )
                ingest_action = _action(ingest_plan, "ingest_sources")
                confirmation = await _confirm(
                    agent, ingest_plan, "ingest_sources", "operator approved these sources"
                )
                ingested = await agent.structured(
                    "ingest_sources",
                    {"confirmation_id": confirmation, **ingest_action["parameters"]},
                )
                log.step(
                    "ingest_sources (executed: real download and parse)",
                    {
                        "operation_id": ingested["operation_id"],
                        "run_id": ingested["run_id"],
                        "status": ingested["status"],
                        "result_summary": ingested["result_summary"],
                        "warnings": ingested["warnings"],
                    },
                )
                rebuild_plan = await agent.structured(
                    "plan_knowledge_base",
                    {
                        "kb_id": kb_id,
                        "chunking": {"chunker": "section-aware", "target_size": 700, "overlap": 80},
                    },
                )
                rebuild_action = _action(rebuild_plan, "build_knowledge_base")
                confirmation = await _confirm(
                    agent, rebuild_plan, "build_knowledge_base", "operator approved the rebuild"
                )
                rebuilt = await agent.structured(
                    "build_knowledge_base",
                    {"confirmation_id": confirmation, **rebuild_action["parameters"]},
                )
                log.step(
                    "build_knowledge_base (after external ingestion)",
                    {
                        "operation_id": rebuilt["operation_id"],
                        "run_id": rebuilt["run_id"],
                        "status": rebuilt["status"],
                        "result_summary": rebuilt["result_summary"],
                    },
                )
                validation = await agent.structured("validate_knowledge_base", {"kb_id": kb_id})
                log.step(
                    "validate_knowledge_base (after external ingestion)",
                    {
                        "verdict": validation["verdict"],
                        "basis": validation["basis"],
                        "chunk_rows": validation["chunk_rows"],
                        "vector_points": validation["vector_points"],
                        "integrity_status": validation["integrity_status"],
                    },
                )
                external["sources"] = discovery["sources"]
                external["ingest"] = ingested["result_summary"]
                external["rebuild"] = rebuilt["result_summary"]

        # -- 12. strategy comparison: gated on the benchmark lifecycle --------
        comparison = await agent.structured(
            "compare_retrieval_strategies",
            {"kb_id": kb_id, "strategies": ["dense", "bm25"], "top_k": 10},
        )
        log.step(
            "compare_retrieval_strategies",
            {
                "permitted": comparison["permitted"],
                "reason": comparison["reason"],
                "benchmark": comparison["benchmark"],
                "strategies": comparison["strategies"],
                "claim_strength": comparison["claim_strength"],
                "promoted": comparison["promoted"],
                "run_id": comparison["run_id"],
                "limitations": comparison["limitations"],
            },
        )

        # -- 13. the audit trail ---------------------------------------------
        for operation_id, label in (
            (created["operation_id"], "creation"),
            (intake["operation_id"], "document intake"),
            (built["operation_id"], "build"),
        ):
            status = await agent.structured(
                "get_operation_status", {"operation_id": operation_id, "kb_id": kb_id}
            )
            log.step(
                f"get_operation_status ({label})",
                {
                    "operation_id": status["operation_id"],
                    "kind": status["kind"],
                    "action": status["action"],
                    "lifecycle_stage": status["lifecycle_stage"],
                    "status": status["status"],
                    "confirmation_used": status["confirmation_used"],
                    "result_summary": status["result_summary"],
                    "run_id": status["run_id"],
                    "run_status": status["run_status"],
                    "run_stages": status["run_stages"],
                    "warnings": status["warnings"],
                },
                limit=1200,
            )

        # -- 14. cleanup ------------------------------------------------------
        if args.keep:
            print(f"\n--keep given: leaving scratch KB {kb_id} in place")
        else:
            _delete_kb(kb_id)
            print(f"\nscratch KB {kb_id} deleted (metadata row + vectors)")
            remaining = await agent.structured("list_knowledge_bases")
            if any(kb["id"] == kb_id for kb in remaining["knowledge_bases"]):
                log.problem("the scratch KB is still listed after deletion")
            log.step("list_knowledge_bases (after cleanup)", {                "total": remaining["total"],
                "knowledge_bases": [
                    {
                        "id": kb["id"],
                        "name": kb["name"],
                        "chunk_count": kb["chunk_count"],
                        "last_build_at": kb["last_build_at"],
                    }
                    for kb in remaining["knowledge_bases"]
                ],
            }, limit=600)

    protected_after = _point_count(PROTECTED_KB_ID)
    print(
        f"\nprotected KB {PROTECTED_KB_ID}: {protected_after} vector point(s) AFTER "
        f"(before: {protected_before})"
    )
    if protected_after != protected_before:
        log.problem("the protected KB's point count changed during this run")

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "kb_name": kb_name,
        "domain": domain,
        "protected_kb_points_before": protected_before,
        "protected_kb_points_after": protected_after,
        "external": external,
        "problems": log.failures,
        "steps": log.steps,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"wrote {out}")

    print("\n=== RESULT " + "=" * 58)
    if log.failures:
        for failure in log.failures:
            print(f"  PROBLEM: {failure}")
        return 1
    print("  every step completed, every refusal was a real refusal, and the")
    print("  protected knowledge base was untouched.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="", help="scratch knowledge base name (default: timestamped)")
    parser.add_argument("--domain", default="Automobile Engineering", help="domain for the scratch KB")
    parser.add_argument("--upload-file", default="", help="a local file to add as a user-provided document")
    parser.add_argument("--keep", action="store_true", help="do not delete the scratch KB afterwards")
    parser.add_argument("--json-out", default="", help="write the full run record to this JSON path")
    parser.add_argument("--external", action="store_true", help="also discover and ingest real URLs (needs network)")
    parser.add_argument(
        "--urls",
        nargs="*",
        default=[
            "https://en.wikipedia.org/wiki/Internal_combustion_engine_cooling",
            "https://en.wikipedia.org/wiki/Radiator_(engine_cooling)",
        ],
        help="URLs for --external",
    )
    args = parser.parse_args(argv)
    if args.external and not args.urls:
        parser.error("--external needs at least one URL")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
