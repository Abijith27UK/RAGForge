"""V19 — second-domain validation driven over the real MCP server (stdio).

The live counterpart to `docs/v19-second-domain-validation.md`. It runs the
documented agentic lifecycle against a SECOND domain (Internet Protocols /
Network Engineering, corpus = primary IETF RFCs) using the real workspace:
the real SQLite database, the real Qdrant instance, the real local embedding
model and the configured LLM provider (or the labelled dev mock, disclosed).

    discover            -> analyze_domain, create KB, discover+score sources
                           (read-only towards the corpus: nothing downloaded)
    ingest --approve-url -> the HUMAN source decision, then download, parse,
                           chunk, embed, index, validate, probe
    audit               -> protection checks only (read-only)

Why split at approval
---------------------
`discover_sources` scores candidates with the real quality engine, and a
REVIEW decision is a human gate by design. This script therefore STOPS after
discovery: it prints the scored candidates and exits. The operator approves the
exact URL list (the stable identifier) on the command line, and only then does
`ingest` record those ACCEPT decisions and download anything. An agent cannot
promote its own candidates.

Honesty rules inherited from V18
--------------------------------
* Every printed value comes from a server result or a measurement; nothing is
  invented.
* LLM-backed steps record the resolved provider identity (`name`, `model`,
  `is_mock`, base-URL host only). A mock run is reported as a mock run.
* The protected research KB `kb_f278c283c748` is never opened, listed,
  ingested into or indexed; its point count is measured before and after.
* Confirmation gates are demonstrated, not bypassed: each consequential call
  carries a confirmation bound to its exact parameters.

Usage (from backend/, with Qdrant reachable):
    py -m scripts.v19_second_domain_validation discover
    py -m scripts.v19_second_domain_validation ingest --approve-url https://www.rfc-editor.org/rfc/rfc791.txt ...
    py -m scripts.v19_second_domain_validation audit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

SERVER_SCRIPT = BACKEND_ROOT / "scripts" / "run_mcp_server.py"
STATE_PATH = BACKEND_ROOT / "data" / "v19-validation-state.json"

#: The research KB that must not be touched. Measured before and after.
PROTECTED_KB_ID = "kb_f278c283c748"
PROTECTED_EXPECTED_POINTS = 812

KB_NAME = "Internet Protocols KB (V19)"
KB_DOMAIN = "Internet Protocols / Network Engineering"
KB_PURPOSE = (
    "Course-reference knowledge base for network engineering students: protocol "
    "behaviour, parameter values and terminology from the primary IETF standards "
    "(RFCs), answered with citations."
)
KB_AUDIENCE = "Network engineering students and junior engineers"
KB_DEPTH = "technical"

#: The planned corpus: RFC Editor plain-text renderings of published RFCs.
#: Subdomain labels are this experiment's knowledge map (Phase 2 of the plan).
RFC_CORPUS: list[dict[str, str]] = [
    {"url": "https://www.rfc-editor.org/rfc/rfc791.txt", "subdomain": "ipv4", "title": "RFC 791 — Internet Protocol"},
    {"url": "https://www.rfc-editor.org/rfc/rfc792.txt", "subdomain": "icmp", "title": "RFC 792 — Internet Control Message Protocol"},
    {"url": "https://www.rfc-editor.org/rfc/rfc826.txt", "subdomain": "link-layer", "title": "RFC 826 — ARP"},
    {"url": "https://www.rfc-editor.org/rfc/rfc1918.txt", "subdomain": "addressing", "title": "RFC 1918 — Private Internets"},
    {"url": "https://www.rfc-editor.org/rfc/rfc5737.txt", "subdomain": "addressing", "title": "RFC 5737 — Documentation address blocks"},
    {"url": "https://www.rfc-editor.org/rfc/rfc8200.txt", "subdomain": "ipv6", "title": "RFC 8200 — IPv6"},
    {"url": "https://www.rfc-editor.org/rfc/rfc9293.txt", "subdomain": "transport", "title": "RFC 9293 — Transmission Control Protocol"},
    {"url": "https://www.rfc-editor.org/rfc/rfc1034.txt", "subdomain": "dns", "title": "RFC 1034 — DNS Concepts and Facilities"},
    {"url": "https://www.rfc-editor.org/rfc/rfc1035.txt", "subdomain": "dns", "title": "RFC 1035 — DNS Implementation and Specification"},
    {"url": "https://www.rfc-editor.org/rfc/rfc2131.txt", "subdomain": "dhcp", "title": "RFC 2131 — Dynamic Host Configuration Protocol"},
    {"url": "https://www.rfc-editor.org/rfc/rfc4271.txt", "subdomain": "routing", "title": "RFC 4271 — BGP-4"},
    {"url": "https://www.rfc-editor.org/rfc/rfc2328.txt", "subdomain": "routing", "title": "RFC 2328 — OSPF Version 2"},
    {"url": "https://www.rfc-editor.org/rfc/rfc8446.txt", "subdomain": "security", "title": "RFC 8446 — TLS 1.3"},
    {"url": "https://www.rfc-editor.org/rfc/rfc9110.txt", "subdomain": "http", "title": "RFC 9110 — HTTP Semantics"},
]

#: Chunking configuration, declared in the plan BEFORE any measurement:
#: the pipeline defaults, deliberately not tuned for this domain.
CHUNKING = {"chunker": "section-aware", "target_size": 1200, "overlap": 150}

#: Retrieval probes — one per subdomain (not benchmark questions; the formal
#: draft benchmark is authored separately against the built corpus).
RETRIEVAL_PROBES: list[dict[str, str]] = [
    {"subdomain": "ipv4", "query": "How does a router use the Time to Live field in an IPv4 datagram?"},
    {"subdomain": "addressing", "query": "Which IPv4 address ranges are reserved for private networks?"},
    {"subdomain": "transport", "query": "Describe the three-way handshake used to establish a TCP connection."},
    {"subdomain": "dns", "query": "What is a recursive resolver and how does DNS query forwarding work?"},
    {"subdomain": "dhcp", "query": "Which messages does a DHCP client exchange to obtain an IP address?"},
    {"subdomain": "routing", "query": "What are the BGP finite state machine states and how does peering start?"},
    {"subdomain": "security", "query": "Which messages are exchanged during the TLS 1.3 handshake?"},
    {"subdomain": "http", "query": "What does the HTTP 405 status code indicate?"},
]

#: In-corpus answer probes (answerable from the corpus by construction).
ANSWER_PROBES_IN_CORPUS = [
    "What are the private IPv4 address blocks reserved by RFC 1918?",
    "What does an IPv4 router do to the Time to Live field of a datagram it forwards?",
]

#: Unsupported probes: well-formed, in-domain-looking, but provably absent from
#: this corpus. Each must end in abstention or explicit insufficient evidence.
ANSWER_PROBES_UNSUPPORTED = [
    "Which RFC specifies the MQTT protocol?",
    "How do I configure a Cisco IOS interface with an access control list?",
    "What is CAN bus bit timing and how is it configured?",
]


# ---------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_v19_second_domain.py)
# ---------------------------------------------------------------------------


def normalize_urls(values: list[str] | None) -> list[str]:
    """Normalize a CLI approval list to unique, order-preserving URLs."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        url = str(raw).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def uncovered(items: list[str], allowed: list[str]) -> list[str]:
    """Items that are not present in `allowed` (order-preserving)."""
    allowed_set = set(allowed)
    return [u for u in items if u not in allowed_set]


def provider_identity_report(provider: Any, base_url: str = "") -> dict[str, Any]:
    """The provider identity, with NO secret material.

    Only the model/provider names, the mock flag, the configured timeout/retry
    budget and the base-URL HOST (never a key, never a path) are recorded.
    """
    host = ""
    if base_url:
        from urllib.parse import urlparse

        host = urlparse(base_url).hostname or ""
    return {
        "name": getattr(provider, "name", "unknown"),
        "model": getattr(provider, "model", "unknown"),
        "is_mock": bool(getattr(provider, "is_mock", False)),
        "base_url_host": host,
        "timeout_s": getattr(provider, "timeout_s", None),
        "max_retries": getattr(provider, "max_retries", None),
    }


def grounding_probe_questions() -> dict[str, list[str]]:
    """The probe sets, exposed for tests and documentation."""
    return {
        "retrieval": [p["query"] for p in RETRIEVAL_PROBES],
        "in_corpus": list(ANSWER_PROBES_IN_CORPUS),
        "unsupported": list(ANSWER_PROBES_UNSUPPORTED),
    }


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


def _accept_source(kb_id: str, source_id: str, note: str) -> None:
    """Record the HUMAN source decision exactly as the decision API does.

    The MCP surface deliberately cannot do this: overriding REVIEW/REJECT is a
    human action recorded through the API/UI. This host script performs it on
    the operator's explicit instruction (the approved URL list on the CLI).
    """
    from app.schemas.models import SourceDecision

    repo = _repo()
    source = repo.get_source(kb_id, source_id)
    if source is None:
        raise RuntimeError(f"source {source_id} not found")
    source.decision = SourceDecision.ACCEPT
    repo.update_source(source)
    print(f"    [human] source {source_id} set to ACCEPT ({note})")


def _protected_check(label: str) -> int | None:
    points = _point_count(PROTECTED_KB_ID)
    print(f"protected KB {PROTECTED_KB_ID}: {points} vector point(s) ({label})")
    if points is not None and points != PROTECTED_EXPECTED_POINTS:
        print(
            f"    !! expected {PROTECTED_EXPECTED_POINTS} points on the protected KB; "
            "stopping rather than running against a corpus in an unexpected state"
        )
    return points


# ---------------------------------------------------------------------------
# MCP client harness (same shape as the V18 workflow script)
# ---------------------------------------------------------------------------


class Log:
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


class Agent:
    def __init__(self, client: Any, log: Log) -> None:
        self.client = client
        self.log = log

    async def call(
        self, tool: str, arguments: dict[str, Any] | None = None, *, timeout: float = 3600
    ) -> tuple[bool, dict[str, Any], str]:
        result = await self.client.call_tool(
            tool, arguments or {}, read_timeout_seconds=timeout
        )
        structured = dict(result.structured_content or {})
        text = " ".join(getattr(block, "text", "") for block in (result.content or [])).strip()
        return bool(result.is_error), structured, text

    async def structured(
        self, tool: str, arguments: dict[str, Any] | None = None, *, timeout: float = 3600
    ) -> dict[str, Any]:
        is_error, structured, text = await self.call(tool, arguments, timeout=timeout)
        if is_error:
            raise RuntimeError(f"{tool} failed: {text or structured}")
        return structured

    async def refused(
        self, tool: str, arguments: dict[str, Any] | None = None, *, timeout: float = 3600
    ) -> str:
        is_error, structured, text = await self.call(tool, arguments, timeout=timeout)
        if not is_error:
            self.log.problem(f"{tool} was NOT refused as expected - the gate has a hole")
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
    return payload["confirmation_id"]


# ---------------------------------------------------------------------------
# state (survives between the discover and ingest subcommands)
# ---------------------------------------------------------------------------


def _load_state() -> dict[str, Any]:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    print(f"state written to {STATE_PATH}")


# ---------------------------------------------------------------------------
# subcommand: discover
# ---------------------------------------------------------------------------


async def cmd_discover(args: argparse.Namespace) -> int:
    log = Log()
    from mcp import Client, StdioServerParameters

    from app.llm.provider import create_llm_provider

    protected_before = _protected_check("before discovery")
    if protected_before != PROTECTED_EXPECTED_POINTS:
        return 2

    provider = create_llm_provider(_settings(), domain_hint=KB_DOMAIN)
    identity = provider_identity_report(provider, base_url=_settings().llm_base_url)
    print(f"\nLLM provider identity: {json.dumps(identity)}")
    if identity["is_mock"]:
        print(
            "    NOTE: no real LLM is configured, so domain analysis and answers "
            "come from RAGForge's labelled development mock (is_mock=true)."
        )

    state = _load_state()
    kb_id = state.get("kb_id") or ""

    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)], cwd=str(BACKEND_ROOT))
    async with Client(params, raise_exceptions=False) as client:
        agent = Agent(client, log)

        tools = (await client.list_tools()).tools
        print(f"\n{len(tools)} tools over stdio")
        if len(tools) < 18:
            log.problem(f"expected at least 18 tools, the server exposed {len(tools)}")

        analysis = await agent.structured(
            "analyze_domain",
            {
                "domain": KB_DOMAIN,
                "purpose": KB_PURPOSE,
                "target_audience": KB_AUDIENCE,
                "depth": KB_DEPTH,
            },
        )
        log.step(
            "analyze_domain (plan-time proposal)",
            {
                "generated_by": analysis["generated_by"],
                "is_mock": analysis["is_mock"],
                "persisted": analysis["persisted"],
                "subdomains": analysis["proposal"]["subdomains"][:8],
                "recommended_source_categories": analysis["proposal"][
                    "recommended_source_categories"
                ][:8],
                "warnings": analysis["warnings"],
            },
        )
        if analysis["is_mock"] != identity["is_mock"]:
            log.problem("the analyzer's is_mock flag disagrees with the resolved provider")

        if kb_id:
            log.say(f"reusing existing KB {kb_id} from {STATE_PATH}")
        else:
            plan = await agent.structured(
                "plan_knowledge_base",
                {
                    "name": KB_NAME,
                    "domain": KB_DOMAIN,
                    "purpose": KB_PURPOSE,
                    "target_audience": KB_AUDIENCE,
                    "depth": KB_DEPTH,
                    "source_mode": "external",
                },
            )
            log.step(
                "plan_knowledge_base (creation plan)",
                {
                    "plan_id": plan["plan_id"],
                    "mode": plan["mode"],
                    "mutates": plan["mutates"],
                    "actions": [
                        {"action": a["action"], "parameters_fingerprint": a["parameters_fingerprint"]}
                        for a in plan["actions"]
                    ],
                    "warnings": plan["warnings"],
                },
            )
            action = _action(plan, "create_knowledge_base")

            # Gate demonstrations: a missing confirmation and a plan id used as
            # a confirmation must both be refused.
            missing = await agent.refused(
                "create_knowledge_base", {**action["parameters"], "confirmation_id": ""}
            )
            log.refuse("gate: no confirmation at all", missing)
            as_plan_id = await agent.refused(
                "create_knowledge_base", {**action["parameters"], "confirmation_id": plan["plan_id"]}
            )
            log.refuse("gate: a plan id is not a confirmation", as_plan_id)

            confirmation = await _confirm(
                agent, plan, "create_knowledge_base", "operator approved the V19 second-domain KB"
            )
            created = await agent.structured(
                "create_knowledge_base", {"confirmation_id": confirmation, **action["parameters"]}
            )
            kb_id = created["kb_id"]
            log.step(
                "create_knowledge_base (executed)",
                {"operation_id": created["operation_id"], "kb_id": kb_id, "status": created["status"]},
            )
            if kb_id == PROTECTED_KB_ID:
                log.problem("the workflow created the PROTECTED knowledge base; aborting")
                return 2

            replay = await agent.refused(
                "create_knowledge_base", {"confirmation_id": confirmation, **action["parameters"]}
            )
            log.refuse("gate: a confirmation is single-use (replay)", replay)

        urls = [item["url"] for item in RFC_CORPUS]
        discovery = await agent.structured(
            "discover_sources",
            {"kb_id": kb_id, "provider": "user-url", "query": " ".join(urls), "limit": len(urls)},
        )
        log.step(
            "discover_sources (real probes and scoring; nothing downloaded)",
            {
                "provider": discovery["provider"],
                "discovered": discovery["discovered"],
                "registered_new": discovery["registered_new"],
                "rejected_invalid": discovery["rejected_invalid"],
                "sources": [
                    {
                        "source_id": s["source_id"],
                        "url": s["url"],
                        "decision": s["decision"],
                        "trust_score": s["trust_score"],
                        "assessed_by": s["assessed_by"],
                        "reasons": s["reasons"][:3],
                        "warnings": s["warnings"],
                    }
                    for s in discovery["sources"]
                ],
                "warnings": discovery["warnings"],
            },
            limit=4000,
        )
        assessments = await agent.structured("assess_sources", {"kb_id": kb_id})
        log.step(
            "assess_sources",
            {
                "total": assessments["total"],
                "counts_by_decision": assessments["counts_by_decision"],
                "unassessed": assessments["unassessed"],
            },
        )

        state.update(
            {
                "kb_id": kb_id,
                "provider_identity": identity,
                "discovered": [
                    {"source_id": s["source_id"], "url": s["url"], "decision": s["decision"], "trust_score": s["trust_score"]}
                    for s in discovery["sources"]
                ],
                "discover_failed": log.failures,
                "discover_ran_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save_state(state)

        print("\n=== DISCOVER COMPLETE " + "=" * 56)
        if log.failures:
            for failure in log.failures:
                print(f"  PROBLEM: {failure}")
            return 1
        review = [s for s in state["discovered"] if s["decision"] != "ACCEPT"]
        print(f"  {kb_id} created; {len(state['discovered'])} candidate(s) scored.")
        if review:
            print(f"  {len(review)} source(s) need a human decision (REVIEW/REJECT).")
        print(
            "  Nothing was downloaded. Approve the exact URL list you accept, then run:\n"
            "    py -m scripts.v19_second_domain_validation ingest --approve-url <url> [...]"
        )
        return 0


# ---------------------------------------------------------------------------
# subcommand: ingest (requires explicit URL approvals)
# ---------------------------------------------------------------------------


async def cmd_ingest(args: argparse.Namespace) -> int:
    log = Log()
    from mcp import Client, StdioServerParameters

    from app.llm.provider import create_llm_provider

    state = _load_state()
    kb_id = state.get("kb_id") or ""
    if not kb_id:
        print("no state file: run the `discover` subcommand first")
        return 2

    approved = normalize_urls(args.approve_url)
    if not approved:
        print(
            "refusing to ingest without explicit approval: pass --approve-url for every "
            "URL you accept (repeat the flag)."
        )
        return 2
    discovered_urls = [s["url"] for s in state.get("discovered", [])]
    unknown = uncovered(approved, discovered_urls)
    if unknown:
        print(f"refusing: these approved URLs were never discovered/scored: {unknown}")
        return 2

    protected_before = _protected_check("before ingestion")
    if protected_before != PROTECTED_EXPECTED_POINTS:
        return 2
    provider = create_llm_provider(_settings(), domain_hint=KB_DOMAIN)
    identity = provider_identity_report(provider, base_url=_settings().llm_base_url)

    by_url = {s["url"]: s for s in state.get("discovered", [])}
    approved_ids = [by_url[u]["source_id"] for u in approved]

    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)], cwd=str(BACKEND_ROOT))
    async with Client(params, raise_exceptions=False) as client:
        agent = Agent(client, log)

        # -- the human source decision, recorded before any download ---------
        for url in approved:
            entry = by_url[url]
            if entry["decision"] != "ACCEPT":
                _accept_source(
                    kb_id,
                    entry["source_id"],
                    f"operator approved this exact URL on the V19 command line "
                    f"(engine decision was {entry['decision']}, score {entry['trust_score']})",
                )
        print(f"\noperator approved {len(approved)} URL(s); ingesting {len(approved_ids)} source(s)")

        # -- ingest ----------------------------------------------------------
        ingest_plan = await agent.structured(
            "plan_knowledge_base",
            {"kb_id": kb_id, "source_mode": "external", "retrieval": {"strategy": "dense"}},
        )
        log.step(
            "plan_knowledge_base (ingest plan)",
            {
                "plan_id": ingest_plan["plan_id"],
                "actions": [a["action"] for a in ingest_plan["actions"]],
                "proposed_corpus": [
                    {"url": p.get("url"), "decision": p.get("decision")} for p in ingest_plan.get("proposed_corpus", [])
                ],
                "warnings": ingest_plan["warnings"],
            },
            limit=1800,
        )
        ingest_action = _action(ingest_plan, "ingest_sources")
        # The plan's source set may be wider than the approval: only the
        # operator-approved ids travel into the execution call.
        planned_ids = list((ingest_action["parameters"] or {}).get("source_ids") or [])
        extra = [sid for sid in planned_ids if sid not in approved_ids]
        if extra:
            log.say(
                f"plan proposed {len(planned_ids)} source(s); the operator approved "
                f"{len(approved_ids)}. Executing with the APPROVED subset only."
            )
        confirmation = await _confirm(
            agent, ingest_plan, "ingest_sources", "operator approved these exact URLs on the CLI"
        )
        started = time.perf_counter()
        ingested = await agent.structured(
            "ingest_sources",
            {"confirmation_id": confirmation, "kb_id": kb_id, "source_ids": approved_ids},
        )
        ingest_s = time.perf_counter() - started
        log.step(
            "ingest_sources (executed: real download and parse)",
            {
                "operation_id": ingested["operation_id"],
                "run_status": ingested["run_status"],
                "result_summary": ingested["result_summary"],
                "wall_clock_s": round(ingest_s, 2),
                "warnings": ingested["warnings"],
            },
        )
        summary = ingested.get("result_summary") or {}
        docs_from_sources = int(summary.get("documents_from_these_sources") or 0)
        if docs_from_sources < len(approved_ids):
            log.problem(
                f"only {docs_from_sources} document(s) came back from {len(approved_ids)} "
                "approved source(s); failed downloads are reported, not hidden"
            )

        # -- build ------------------------------------------------------------
        build_plan = await agent.structured(
            "plan_knowledge_base",
            {"kb_id": kb_id, "chunking": CHUNKING, "retrieval": {"strategy": "dense"}},
        )
        log.step(
            "plan_knowledge_base (build plan)",
            {
                "plan_id": build_plan["plan_id"],
                "chunking": build_plan["chunking"],
                "retrieval": build_plan["retrieval"],
                "current_state": build_plan["current_state"],
                "warnings": build_plan["warnings"],
            },
        )
        build_action = _action(build_plan, "build_knowledge_base")
        build_confirmation = await _confirm(
            agent, build_plan, "build_knowledge_base", "operator approved the V19 build"
        )
        started = time.perf_counter()
        built = await agent.structured(
            "build_knowledge_base",
            {"confirmation_id": build_confirmation, **build_action["parameters"]},
        )
        build_s = time.perf_counter() - started
        log.step(
            "build_knowledge_base (executed)",
            {
                "operation_id": built["operation_id"],
                "run_status": built["run_status"],
                "result_summary": built["result_summary"],
                "wall_clock_s": round(build_s, 2),
                "embedding_identity": (built.get("details") or {}).get("embedding_identity"),
                "warnings": built["warnings"],
            },
        )
        if built["status"] != "done":
            log.problem(f"build did not finish successfully: {built['status']}")

        # -- validation -------------------------------------------------------
        validation = await agent.structured(
            "validate_knowledge_base", {"kb_id": kb_id, "include_diagnostics": False}
        )
        log.step(
            "validate_knowledge_base",
            {
                "verdict": validation["verdict"],
                "ready_for_retrieval": validation["ready_for_retrieval"],
                "basis": validation["basis"],
                "blocking": validation["blocking"],
                "document_counts": validation["document_counts"],
                "chunk_rows": validation["chunk_rows"],
                "vector_points": validation["vector_points"],
                "vector_points_measured": validation["vector_points_measured"],
                "integrity_status": validation["integrity_status"],
                "integrity_counts": validation["integrity_counts"],
                "embedding_identity": validation["embedding_identity"],
            },
        )
        if validation["verdict"] != "READY":
            log.problem(
                f"the freshly built KB is not READY: {validation['verdict']} "
                f"({'; '.join(validation['blocking'])})"
            )        # -- retrieval probes across subdomains --------------------------------
        # Each post-build section is guarded: a failed probe is RECORDED as a
        # failure (exit 1, PROBLEM printed, stored in the run record) instead of
        # crashing the script before the record — and the protected-KB check —
        # are written. Failures are never turned into successes.
        probe_rows: list[dict[str, Any]] = []
        for probe in RETRIEVAL_PROBES:
            try:
                search = await agent.structured(
                    "search_knowledge_base", {"kb_id": kb_id, "query": probe["query"], "top_k": 5}
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                log.problem(f"search_knowledge_base [{probe['subdomain']}] failed: {exc}")
                continue
            timings = (search.get("trace") or {}).get("timings") or {}
            row = {
                "subdomain": probe["subdomain"],
                "query": probe["query"],
                "applied_strategy": search["applied_strategy"],
                "total_ms": timings.get("total_ms"),
                "top": [
                    {
                        "rank": h["rank"],
                        "chunk_id": h["chunk_id"],
                        "score": round(float(h["retrieval_score"]), 4),
                        "source_url": h.get("source_url"),
                        "section": h.get("section"),
                        "text_head": (h.get("text") or "")[:90],
                    }
                    for h in search["results"][:3]
                ],
            }
            probe_rows.append(row)
            log.step(f"search_knowledge_base [{probe['subdomain']}]", row, limit=1200)

        # -- evidence + provenance ---------------------------------------------
        sample_ids = [row["top"][0]["chunk_id"] for row in probe_rows if row["top"]][:5]
        try:
            evidence = await agent.structured(
                "get_evidence", {"kb_id": kb_id, "chunk_ids": sample_ids}
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            evidence = {}
            log.problem(f"get_evidence failed: {exc}")
        log.step("get_evidence (provenance of one chunk per subdomain)",
            {
                "found": evidence.get("found"),
                "missing_chunk_ids": evidence.get("missing_chunk_ids", []),
                "chunks": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "source_url": c.get("source_url"),
                        "chunk_index": c.get("chunk_index"),
                        "section": c.get("section"),
                        "page": c.get("page"),
                        "trust_score": c.get("trust_score"),
                        "chunking_strategy": c.get("chunking_strategy"),
                        "text_length": len(c.get("text") or ""),
                    }
                    for c in evidence.get("chunks", [])
                ],
            },
            limit=1600,
        )

        # -- grounded answers + abstention ---------------------------------------
        answer_rows: list[dict[str, Any]] = []
        for question in ANSWER_PROBES_IN_CORPUS:
            try:
                answer = await agent.structured(
                    "answer_from_knowledge_base", {"kb_id": kb_id, "question": question, "top_k": 5}
                )
            except Exception as exc:  # noqa: BLE001 - a failed real-LLM call stays a failure
                log.problem(f"answer probe failed: {question!r}: {exc}")
                continue
            row = {
                "kind": "in_corpus",
                "question": question,
                "status": answer["status"],
                "grounded": answer["grounded"],
                "grounding_state": answer["grounding_state"],
                "citation_count": answer["citation_count"],
                "claim_count": answer["claim_count"],
                "unsupported_claim_count": answer.get("unsupported_claim_count"),
                "answer_text_head": (answer.get("answer_text") or "")[:300],
                "abstention_reason": answer.get("abstention_reason"),
                "generation": answer.get("generation"),
                "warnings": answer["warnings"],
            }
            answer_rows.append(row)
            log.step("answer_from_knowledge_base (in-corpus probe)", row, limit=1400)
            if (answer.get("generation") or {}).get("is_mock"):
                log.say("NOTE: this answer text came from the labelled development mock.")

        abstain_rows: list[dict[str, Any]] = []
        for question in ANSWER_PROBES_UNSUPPORTED:
            try:
                answer = await agent.structured(
                    "answer_from_knowledge_base", {"kb_id": kb_id, "question": question, "top_k": 5}
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                log.problem(f"unsupported probe failed: {question!r}: {exc}")
                continue
            row = {
                "kind": "unsupported",
                "question": question,
                "status": answer["status"],
                "grounded": answer["grounded"],
                "grounding_state": answer["grounding_state"],
                "citation_count": answer["citation_count"],
                "abstention_reason": answer.get("abstention_reason"),
                "generation": answer.get("generation"),
            }
            abstain_rows.append(row)
            log.step("answer_from_knowledge_base (unsupported probe)", row, limit=1200)
            if answer["status"] == "answered" and answer["grounded"]:
                log.problem(f"the KB confidently answered an unsupported question: {question!r}")

        # -- the gated comparison tool, against whatever instrument exists -----
        # The draft benchmark is AUTHORED AFTER this build (its grounding must
        # reference real chunk ids from THIS build), so on a first run the
        # instrument does not exist yet: record that as the outcome and defer
        # the comparison to Phase 3 instead of crashing before the record and
        # the protected-KB check are written.
        benchmark_rel = "benchmarks/internet-protocols-draft-v1.json"
        benchmark_abs = BACKEND_ROOT.parent / benchmark_rel
        if not benchmark_abs.is_file():
            log.step(
                "compare_retrieval_strategies (deferred)",
                {
                    "permitted": False,
                    "deferred": True,
                    "reason": (
                        f"{benchmark_rel} does not exist yet: the draft instrument is "
                        "grounded in chunk ids from THIS build, so it is authored after "
                        "the build. Deferred to Phase 3 (after v19_benchmark_tool finalize)."
                    ),
                },
            )
        else:
            try:
                comparison = await agent.structured(
                    "compare_retrieval_strategies",
                    {
                        "kb_id": kb_id,
                        "benchmark_path": benchmark_rel,
                        "strategies": ["dense", "bm25"],
                        "top_k": 10,
                        "acknowledge_exploratory": True,
                        "label": "v19-draft-exploratory",
                    },
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                comparison = {}
                log.problem(f"compare_retrieval_strategies failed: {exc}")
            log.step(
                "compare_retrieval_strategies (draft instrument)",
                {
                    "permitted": comparison.get("permitted"),
                    "reason": comparison.get("reason"),
                    "claim_strength": comparison.get("claim_strength"),
                    "promoted": comparison.get("promoted"),
                    "run_id": comparison.get("run_id"),
                    "strategies": comparison.get("strategies"),
                    "warnings": comparison.get("warnings"),
                },
                limit=2000,
            )

        # -- audit trail ---------------------------------------------------------
        try:
            operations = await agent.structured(
                "get_operation_status", {"operation_id": ingested["operation_id"], "kb_id": kb_id}
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            operations = {}
            log.problem(f"get_operation_status failed: {exc}")
        log.step(
            "get_operation_status (ingest operation)",
            {
                "operation_id": operations.get("operation_id"),
                "action": operations.get("action"),
                "lifecycle_stage": operations.get("lifecycle_stage"),
                "status": operations.get("status"),
                "confirmation_used": operations.get("confirmation_used"),
                "run_status": operations.get("run_status"),
            },
        )

    protected_after = _protected_check("after the V19 run")
    if protected_after != protected_before:
        log.problem("the protected KB's point count changed during this run")

    record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "kb_id": kb_id,
        "domain": KB_DOMAIN,
        "provider_identity": identity,
        "approved_urls": approved,
        "chunking": CHUNKING,
        "ingest_wall_clock_s": round(ingest_s, 2),
        "build_wall_clock_s": round(build_s, 2),
        "validation": {
            "verdict": validation["verdict"],
            "chunk_rows": validation["chunk_rows"],
            "vector_points": validation["vector_points"],
            "integrity_status": validation["integrity_status"],
        },
        "probes": {"retrieval": probe_rows, "answers": answer_rows, "abstentions": abstain_rows},
        "protected_kb_points_before": protected_before,
        "protected_kb_points_after": protected_after,
        "problems": log.failures,
        "steps": log.steps,
    }
    out = Path(args.json_out) if args.json_out else BACKEND_ROOT.parent / "docs" / "v19-run-record.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"\nrun record written to {out}")

    print("\n=== INGEST/BUILD COMPLETE " + "=" * 53)
    if log.failures:
        for failure in log.failures:
            print(f"  PROBLEM: {failure}")
        return 1
    print("  every step completed, every refusal was a real refusal, and the")
    print("  protected knowledge base was untouched.")
    return 0


# ---------------------------------------------------------------------------
# subcommand: audit
# ---------------------------------------------------------------------------


def cmd_audit(_args: argparse.Namespace) -> int:
    import sqlite3
    import subprocess

    problems: list[str] = []
    points = _point_count(PROTECTED_KB_ID)
    print(f"protected collection points: {points} (expected {PROTECTED_EXPECTED_POINTS})")
    if points != PROTECTED_EXPECTED_POINTS:
        problems.append(f"protected KB has {points} points, expected {PROTECTED_EXPECTED_POINTS}")

    repo = _repo()
    settings = _settings()
    con = sqlite3.connect(str(settings.db_path))
    try:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        print(f"sqlite integrity_check: {integrity}; foreign_key_check rows: {len(fk)}")
        if integrity != "ok" or fk:
            problems.append("sqlite integrity/foreign-key check failed")
        ops_on_protected = [
            row[0]
            for row in con.execute(
                # `data` is the JSON payload column; there is no `parameters`
                # column in this schema (verified against PRAGMA table_info).
                "SELECT id FROM mcp_operations WHERE kb_id = ? OR data LIKE ?",
                (PROTECTED_KB_ID, f"%{PROTECTED_KB_ID}%"),
            )
        ]
        print(f"mcp_operations rows referencing the protected KB: {len(ops_on_protected)}")
        if ops_on_protected:
            problems.append(f"{len(ops_on_protected)} mcp_operations row(s) reference the protected KB")
    finally:
        con.close()

    # Frozen benchmark artifacts must be byte-identical to what git has.
    repo_root = BACKEND_ROOT.parent
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--stat", "--", "benchmarks/"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if diff.stdout.strip():
        problems.append(f"frozen benchmark artifacts changed: {diff.stdout.strip()}")
        print(f"git diff benchmarks/: {diff.stdout.strip()}")
    else:
        print("git diff HEAD -- benchmarks/: clean (frozen artifacts unchanged)")

    # Orphan collections: Qdrant collections with no KB row.
    known = {f"kb_{kb.id}" for kb in repo.list_kbs()}
    try:
        import httpx

        resp = httpx.get(f"{settings.qdrant_url}/collections", timeout=10)
        resp.raise_for_status()
        names = [c["name"] for c in resp.json().get("result", {}).get("collections", [])]
        orphans = [n for n in names if n.startswith("kb_") and n not in known]
        print(f"qdrant collections: {len(names)}; orphan kb_* collections: {orphans or 'none'}")
        if orphans:
            problems.append(f"orphan collections left behind: {orphans}")
    except Exception as exc:  # noqa: BLE001
        print(f"(collection listing unavailable: {type(exc).__name__})")

    print("\n=== AUDIT " + "=" * 70)
    if problems:
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        return 1
    print("  protected data intact; no orphans; frozen artifacts unchanged.")
    return 0


# ---------------------------------------------------------------------------


async def _amain(args: argparse.Namespace) -> int:
    if args.command == "discover":
        return await cmd_discover(args)
    if args.command == "ingest":
        return await cmd_ingest(args)
    if args.command == "audit":
        return cmd_audit(args)
    print(f"unknown command {args.command!r}")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("discover", help="analyze, create the KB, discover and score sources")
    ingest = sub.add_parser("ingest", help="human-approved ingest, build, validate and probe")
    ingest.add_argument(
        "--approve-url",
        action="append",
        default=[],
        help="a URL the operator approves for ingestion (repeat for each; required)",
    )
    ingest.add_argument("--json-out", default="", help="where to write the run record JSON")
    sub.add_parser("audit", help="read-only protection checks")
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
