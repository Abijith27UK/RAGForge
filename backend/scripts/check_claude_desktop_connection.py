"""Check the RAGForge MCP server the way Claude Desktop launches it (V19).

This script exists because "the server works" and "the host can reach the server"
are different claims. It reads the client's own configuration file, takes the
`ragforge` entry **verbatim** (command, args, env), launches that exact process
over stdio, and runs the acceptance sequence a user would exercise by hand:

    handshake -> tools discovered -> resources discovered
              -> find the Automobile knowledge base
              -> search (real provenance) -> evidence by chunk id
              -> grounded answer (with its generation identity disclosed)
              -> an unsupported question (must abstain, not invent)
              -> an OFF-DOMAIN question whose generic terms collide with corpus
                 vocabulary (the V20 gate must refuse it; this was an open defect
                 until V20 and is now a PASS/FAIL criterion)
              -> read one resource

It never writes: no ingest, no build, no benchmark scoring, no review. The only
side effect is the retrieval run a search records, exactly like a UI query.

Usage
-----
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\check_claude_desktop_connection.py
    ... --config "%APPDATA%\\Claude\\claude_desktop_config.json" --server ragforge
    ... --kb kb_f278c283c748 --supported "What is an engine?" --unsupported "..."
    ... --json scripts/_claude_desktop_check/report.json

Statuses are reported honestly:

* ``PASS`` / ``FAIL`` — the connection-level acceptance criteria. A FAIL sets
  the exit code to 1.
* ``WARN`` — something the operator should know (for example: the answer text
  came from the *disclosed* development mock generator because no real LLM is
  configured). Does not fail the run.
* ``KNOWN-DEFECT`` — a check that reproduces a defect this project has already
  documented and left open. It is **not** a pass: it is counted, printed
  separately and cross-referenced, but it does not fail a *connection* check.
  V20 fixed the only case this script knew about (the off-domain evidence-gate
  weakness), so that check is now PASS/FAIL; the status stays available for the
  next documented defect.

Extra environment
-----------------
``--env KEY=VALUE`` (repeatable) adds variables to the launched server's
environment, MERGED OVER the client config's own ``env``. ``command`` and
``args`` always come from the client config **verbatim**, so the transport under
test is the configured one. The override exists so an acceptance run can pin the
answer generator (e.g. ``--env LLM_PROVIDER=mock``) without editing
``backend/.env`` or the client config, and every override is printed and written
into the report so a run is never silently different from the configured one.

Exit code 0 = no FAIL. 1 = at least one check FAILED (the individual results say
which).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

from mcp import Client, StdioServerParameters  # noqa: E402

DEFAULT_SERVER_NAME = "ragforge"
DEFAULT_SUPPORTED = "What is an engine?"
#: Terms in this question do not occur anywhere in an Automobile corpus, so the
#: gate has something to reject and abstention is a real connection-level check.
DEFAULT_UNSUPPORTED = "Describe the courtship display of the Australian lyrebird."
#: Every content term here is generic English that also occurs in motor-industry
#: text, which is exactly the case the documented off-domain gate defect misses.
DEFAULT_OFFDOMAIN = "How many moons does Jupiter have?"

OFFDOMAIN_FIX_REF = (
    "V20: the gate now abstains when no retrieved passage is topically related to "
    "the question (measured raw-similarity floor) or when the corpus cannot "
    "represent the question at all; see docs/v20-offdomain-gate.md. Failing here "
    "means the defect is BACK, not that the check is misconfigured."
)

CHECK_RESULTS: list[dict[str, Any]] = []


def record(name: str, status: str, detail: str) -> None:
    """status: PASS | FAIL | WARN | UNKNOWN | KNOWN-DEFECT."""
    CHECK_RESULTS.append({"check": name, "status": status, "detail": detail})
    marker = {
        "PASS": "  ok  ",
        "FAIL": " FAIL ",
        "WARN": " warn ",
        "UNKNOWN": "  ??  ",
        "KNOWN-DEFECT": " KNOWN",
    }[status]
    print(f"[{marker}] {name}: {detail}")


def default_config_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def load_server_entry(config_path: Path, server_name: str) -> dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(
            f"FAIL configuration not found: {config_path}\n"
            "Hint: the file is created by the MCP client. On Windows it is\n"
            "%APPDATA%\\Claude\\claude_desktop_config.json"
        )
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FAIL configuration is not valid JSON: {exc}") from exc
    servers = cfg.get("mcpServers") or {}
    if server_name not in servers:
        raise SystemExit(
            f"FAIL no '{server_name}' entry in {config_path}. "
            f"Servers present: {sorted(servers) or 'none'}"
        )
    return servers[server_name]


def parse_env_overrides(pairs: list[str] | None) -> dict[str, str]:
    """`--env KEY=VALUE` -> dict. Rejects malformed pairs loudly."""
    out: dict[str, str] = {}
    for item in pairs or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"FAIL --env expects KEY=VALUE, got {item!r}")
        out[key.strip()] = value
    return out


def preflight(entry: dict[str, Any]) -> tuple[bool, str]:
    """Check the configured paths exist BEFORE starting anything."""
    command = entry.get("command")
    args = entry.get("args") or []
    if not command:
        return False, "the entry has no 'command'"
    if not Path(command).exists():
        return False, f"command does not exist: {command}"
    if not args:
        return False, "the entry has no 'args'; the launcher script should be arg[0]"
    if not Path(str(args[0])).exists():
        return False, f"launcher script does not exist: {args[0]}"
    data_dir = (entry.get("env") or {}).get("DATA_DIR")
    if data_dir and not Path(data_dir).is_dir():
        return False, f"DATA_DIR does not exist: {data_dir}"
    return True, "command, launcher and DATA_DIR all exist"


async def run(
    args: argparse.Namespace, entry: dict[str, Any], overrides: dict[str, str]
) -> int:
    params = StdioServerParameters(
        command=entry["command"],
        args=[str(a) for a in entry.get("args") or []],
        env={
            str(k): str(v) for k, v in (entry.get("env") or {}).items()
        }
        | {str(k): str(v) for k, v in overrides.items()},
        # Deliberately NO cwd: the host does not set one, so a launcher that
        # depends on the working directory must fail this check.
    )

    async with Client(params, raise_exceptions=False) as client:
        # --- handshake -----------------------------------------------------
        info = client.server_info
        instructions = client.instructions or ""
        capabilities = client.server_capabilities
        print(f"server: {info.name if info else '(anonymous)'} {info.version if info else ''}")
        print(f"protocol: {client.protocol_version}")
        record(
            "mcp-handshake",
            "PASS" if info and info.name else "FAIL",
            f"serverInfo={info.name if info else None!r} "
            f"version={info.version if info else None!r} "
            f"protocolVersion={client.protocol_version} "
            f"capabilities={sorted((capabilities.model_dump(exclude_none=True) if hasattr(capabilities, 'model_dump') else {}).keys())}",
        )
        record(
            "server-instructions",
            "PASS" if instructions.strip() else "WARN",
            f"{len(instructions)} chars of guidance sent to the host model",
        )

        # --- discovery -----------------------------------------------------
        tools = (await client.list_tools()).tools
        names = [t.name for t in tools]
        print(f"tools ({len(names)}): {', '.join(names)}")
        required = {
            "list_knowledge_bases",
            "get_knowledge_base_status",
            "search_knowledge_base",
            "get_evidence",
            "answer_from_knowledge_base",
            "evaluate_retrieval",
        }
        missing = sorted(required - set(names))
        record(
            "tools-discovered",
            "PASS" if not missing else "FAIL",
            f"{len(names)} tools; missing {missing}" if missing else f"{len(names)} tools, all required read tools present",
        )
        reader = next((t for t in tools if t.name == "search_knowledge_base"), None)
        record(
            "tool-annotations",
            "PASS" if reader is not None and reader.annotations is not None else "WARN",
            (
                f"search_knowledge_base read_only={reader.annotations.read_only_hint} "
                f"destructive={reader.annotations.destructive_hint}"
                if reader is not None and reader.annotations is not None
                else "no annotations published"
            ),
        )

        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        resource_uris = [str(r.uri) for r in resources.resources]
        template_uris = [t.uri_template for t in templates.resource_templates]
        print(f"resources: {resource_uris}")
        print(f"resource templates: {template_uris}")
        record(
            "resources-discovered",
            "PASS" if resource_uris else "FAIL",
            f"{len(resource_uris)} resource(s), {len(template_uris)} template(s)",
        )

        # --- find the Automobile knowledge base ----------------------------
        listing = await client.call_tool("list_knowledge_bases", {})
        if listing.is_error:
            record("list_knowledge_bases", "FAIL", _error_text(listing))
            return 1
        body = listing.structured_content or {}
        kbs = body.get("knowledge_bases", [])
        record("list_knowledge_bases", "PASS", f"{len(kbs)} knowledge base(s) listed")

        target = args.kb or _pick_kb(kbs)
        if not target:
            record("automobile-kb-found", "FAIL", "no knowledge base with a non-empty corpus")
            return 1
        chosen = next((k for k in kbs if k["id"] == target), {})
        record(
            "automobile-kb-found",
            "PASS" if chosen else "FAIL",
            f"{target} domain={chosen.get('domain')!r} status={chosen.get('status')!r} "
            f"documents={chosen.get('document_count')} chunks={chosen.get('chunk_count')}",
        )

        status = await client.call_tool("get_knowledge_base_status", {"kb_id": target})
        sbody = status.structured_content or {}
        index = sbody.get("index") or {}
        points = index.get("points")
        record(
            "kb-status-live-probe",
            "PASS" if not status.is_error else "FAIL",
            f"ready_for_retrieval={sbody.get('ready_for_retrieval')} "
            f"chunk_rows={index.get('chunk_rows')} vector_points={points} "
            f"points_measured={index.get('points_measured')} "
            f"match={index.get('points_match_chunk_rows')} "
            f"collection={index.get('collection')}",
        )

        # --- search: the retrieval path the host must use -------------------
        search = await client.call_tool(
            "search_knowledge_base", {"kb_id": target, "query": args.supported, "top_k": 3}
        )
        if search.is_error:
            record("search_knowledge_base", "FAIL", _error_text(search))
            return 1
        results = (search.structured_content or {}).get("results", [])
        print("\nsearch results:")
        for hit in results:
            print(
                f"  #{hit.get('rank')} {hit.get('chunk_id')} score={hit.get('retrieval_score')} "
                f"title={(hit.get('source_title') or '')[:60]!r} url={(hit.get('source_url') or '')[:70]!r}"
            )
        record(
            "search-returns-evidence",
            "PASS" if results else "FAIL",
            f"{len(results)} ranked chunk(s) for {args.supported!r}",
        )
        if results:
            top = results[0]
            prov_fields = ["chunk_id", "document_id", "source_title", "source_url", "content_hash"]
            have = [f for f in prov_fields if top.get(f)]
            record(
                "search-provenance",
                "PASS" if {"chunk_id", "document_id", "source_url"} <= set(have) else "WARN",
                f"fields present: {', '.join(have) or 'none'}; trust_score={top.get('trust_score')}",
            )
            leak = _path_leak(top)
            record(
                "search-no-filesystem-paths",
                "FAIL" if leak else "PASS",
                f"filesystem path exposed: {leak}" if leak else "no raw filesystem path in the payload",
            )

        # --- evidence by id -------------------------------------------------
        if results:
            chunk_id = results[0]["chunk_id"]
            evidence = await client.call_tool(
                "get_evidence", {"kb_id": target, "chunk_ids": [chunk_id]}
            )
            ebody = evidence.structured_content or {}
            chunks = ebody.get("chunks", [])
            first = chunks[0] if chunks else {}
            print(
                f"\nevidence {chunk_id}: document={first.get('document_id')} "
                f"index={first.get('chunk_index')} chars={first.get('char_count')}"
            )
            record(
                "get_evidence-by-chunk-id",
                "PASS" if chunks else "FAIL",
                f"resolved {chunk_id} -> document {first.get('document_id')} "
                f"chunk_index {first.get('chunk_index')}",
            )
            record(
                "evidence-no-filesystem-paths",
                "FAIL" if _path_leak(first) else "PASS",
                _path_leak(first) or "no raw filesystem path in the payload",
            )
            source_url = first.get("source_url") or ""
            record(
                "evidence-provenance-url",
                "PASS" if source_url.startswith(("http://", "https://")) else "WARN",
                f"source_url={source_url[:90]!r}",
            )

        # --- grounded answer: supported -------------------------------------
        answer = await client.call_tool(
            "answer_from_knowledge_base", {"kb_id": target, "question": args.supported}
        )
        abody = answer.structured_content or {}
        gen = abody.get("generation") or {}
        print(
            f"\nanswer({args.supported!r}): status={abody.get('status')} "
            f"grounding_state={abody.get('grounding_state')} grounded={abody.get('grounded')} "
            f"citations={abody.get('citation_count')}"
        )
        print(f"  generation: {json.dumps(gen, default=str)[:300]}")
        if answer.is_error:
            record("answer-supported", "FAIL", _error_text(answer))
        else:
            grounded = bool(abody.get("grounded")) or abody.get("status") in {
                "grounded",
                "partial",
            }
            record(
                "answer-supported",
                "PASS" if grounded else "FAIL",
                f"status={abody.get('status')} grounding_state={abody.get('grounding_state')} "
                f"citations={abody.get('citation_count')}",
            )
            if gen.get("is_mock"):
                record(
                    "answer-generation-identity",
                    "WARN",
                    f"generated_by={gen.get('generated_by')!r} model={gen.get('model')!r} "
                    f"is_mock=True - the answer text came from the DISCLOSED development "
                    "mock generator, not a real model. Retrieval and citations are real.",
                )
            else:
                record(
                    "answer-generation-identity",
                    "PASS",
                    f"generated_by={gen.get('generated_by')!r} model={gen.get('model')!r} is_mock=False",
                )
            cites = abody.get("citations") or []
            ok_cites = [
                c
                for c in cites
                if isinstance(c, dict) and (c.get("chunk_id") or c.get("evidence_id"))
            ]
            record(
                "answer-citations-resolve",
                "PASS" if cites and len(ok_cites) == len(cites) else ("WARN" if not cites else "FAIL"),
                f"{len(ok_cites)}/{len(cites)} citation(s) carry a chunk/evidence id",
            )

        # --- grounded answer: unsupported -> abstention ----------------------
        un = await client.call_tool(
            "answer_from_knowledge_base", {"kb_id": target, "question": args.unsupported}
        )
        ubody = un.structured_content or {}
        ugen = ubody.get("generation") or {}
        print(
            f"\nanswer({args.unsupported!r}): status={ubody.get('status')} "
            f"grounding_state={ubody.get('grounding_state')} grounded={ubody.get('grounded')} "
            f"abstention_reason={(ubody.get('abstention_reason') or '')[:140]!r}"
        )
        print(f"  generation: {json.dumps(ugen, default=str)[:200]}")
        if un.is_error:
            record("answer-unsupported", "FAIL", _error_text(un))
        else:
            abstained = (not ubody.get("grounded")) and str(ubody.get("status")) in {
                "abstained",
                "clarification_required",
                "generation_failed",
            }
            record(
                "answer-unsupported-abstains",
                "PASS" if abstained else "FAIL",
                f"status={ubody.get('status')} grounding_state={ubody.get('grounding_state')} "
                f"reason={(ubody.get('abstention_reason') or '(none)')[:120]!r}",
            )
            record(
                "answer-unsupported-no-invention",
                "PASS"
                if not ubody.get("grounded") and not (ubody.get("claims") or [])
                else "FAIL",
                f"{len(ubody.get('claims') or [])} claim(s) returned for an unsupported question",
            )

        # --- off-domain question: the documented gate defect ------------------
        off = await client.call_tool(
            "answer_from_knowledge_base", {"kb_id": target, "question": args.offdomain}
        )
        obody = off.structured_content or {}
        print(
            f"\nanswer({args.offdomain!r}): status={obody.get('status')} "
            f"grounding_state={obody.get('grounding_state')} sufficient={obody.get('sufficient')} "
            f"reason_code={obody.get('reason_code')} citations={obody.get('citation_count')}"
        )
        off_abstained = (not obody.get("grounded")) and str(obody.get("status")) in {
            "abstained",
            "clarification_required",
            "generation_failed",
        }
        record(
            "answer-offdomain-abstains",
            "PASS" if off_abstained else "FAIL",
            f"status={obody.get('status')} reason_code={obody.get('reason_code')} "
            f"citations={obody.get('citation_count')} - "
            + ("abstained as required" if off_abstained else OFFDOMAIN_FIX_REF),
        )

        # --- resource read ---------------------------------------------------
        for uri in (f"ragforge://knowledge-bases/{target}", "ragforge://knowledge-bases"):
            try:
                res = await client.read_resource(uri)
                contents = getattr(res, "contents", None) or []
                text = ""
                for c in contents:
                    text = getattr(c, "text", "") or text
                if text:
                    json.loads(text)
                record(
                    "resource-read",
                    "PASS" if text else "WARN",
                    f"{uri} returned {len(text)} chars of parseable JSON",
                )
            except Exception as exc:  # noqa: BLE001 - report, do not crash the check
                record("resource-read", "FAIL", f"{uri}: {type(exc).__name__}: {exc}")

        # --- lifecycle gate (must refuse to score anything unfrozen) --------
        evaluation = await client.call_tool("evaluate_retrieval", {"kb_id": target, "dry_run": True})
        lifecycle = (evaluation.structured_content or {}).get("lifecycle", {})
        print(
            f"\nevaluate_retrieval (dry run): allowed={lifecycle.get('allowed')} "
            f"reason={lifecycle.get('reason')}"
        )
        record(
            "evaluation-lifecycle-without-side-effects",
            "PASS" if not lifecycle.get("allowed") else "WARN",
            f"allowed={lifecycle.get('allowed')} reason={lifecycle.get('reason')}",
        )

    return 0


def _pick_kb(kbs: list[dict[str, Any]]) -> str | None:
    """Prefer an Automobile knowledge base with a real corpus."""
    with_corpus = [k for k in kbs if (k.get("chunk_count") or 0) > 0]
    auto = [k for k in with_corpus if "automobile" in str(k.get("domain", "")).lower()]
    pool = auto or with_corpus
    return pool[0]["id"] if pool else None


def _path_leak(payload: dict[str, Any]) -> str:
    for key in ("file_path", "raw_file_path", "path", "local_path", "source_path"):
        if payload.get(key):
            return f"{key}={payload[key]!r}"
    return ""


def _describe_failure(exc: BaseException) -> str:
    """Unwrap TaskGroup/ExceptionGroup nesting so the real cause is visible.

    Without this a single bug inside the client session surfaces only as
    "unhandled errors in a TaskGroup", which tells a user nothing.
    """
    seen: list[str] = []
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop(0)
        seen.append(f"{type(current).__name__}: {current}")
        stack.extend(getattr(current, "exceptions", []) or [])
        cause = current.__cause__ or current.__context__
        if cause is not None and cause is not current:
            stack.append(cause)
    return " | ".join(seen[:6])


def _error_text(result: Any) -> str:
    try:
        return result.content[0].text
    except Exception:  # noqa: BLE001
        return "tool error (no text)"


def summarize() -> int:
    failed = [r for r in CHECK_RESULTS if r["status"] == "FAIL"]
    warned = [r for r in CHECK_RESULTS if r["status"] == "WARN"]
    unknown = [r for r in CHECK_RESULTS if r["status"] == "UNKNOWN"]
    defects = [r for r in CHECK_RESULTS if r["status"] == "KNOWN-DEFECT"]
    passed = len(CHECK_RESULTS) - len(failed) - len(warned) - len(unknown) - len(defects)
    print("\n" + "=" * 72)
    print(
        f"SUMMARY: {len(CHECK_RESULTS)} checks - {passed} passed, {len(failed)} failed, "
        f"{len(warned)} warning(s), {len(defects)} known defect(s), {len(unknown)} unknown"
    )
    for r in failed:
        print(f"  FAILED: {r['check']}: {r['detail']}")
    for r in warned:
        print(f"  WARNING: {r['check']}: {r['detail']}")
    for r in defects:
        print(f"  KNOWN DEFECT (not a pass): {r['check']}: {r['detail']}")
    print(
        "\nNote: this proves the CONFIGURED COMMAND speaks MCP correctly and returns\n"
        "real RAGForge evidence. It cannot prove a specific host application shows a\n"
        "connector icon - start the host after editing its configuration."
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="client config file (default: %%APPDATA%%\\Claude\\claude_desktop_config.json)")
    parser.add_argument("--server", default=DEFAULT_SERVER_NAME, help="server key in mcpServers")
    parser.add_argument("--kb", default="", help="knowledge base id (default: first Automobile KB with chunks)")
    parser.add_argument("--supported", default=DEFAULT_SUPPORTED, help="a question the corpus answers")
    parser.add_argument(
        "--unsupported",
        default=DEFAULT_UNSUPPORTED,
        help="a question whose terms are absent, so the gate must abstain",
    )
    parser.add_argument(
        "--offdomain",
        default=DEFAULT_OFFDOMAIN,
        help="an off-domain question with generic terms (documented gate defect)",
    )
    parser.add_argument("--json", default="", help="also write the check results to this JSON file")
    parser.add_argument(
        "--env",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="extra server environment (repeatable); merged OVER the config's env",
    )
    ns = parser.parse_args(argv)
    overrides = parse_env_overrides(ns.env)

    config_path = Path(ns.config) if ns.config else default_config_path()
    print(f"config: {config_path}")
    entry = load_server_entry(config_path, ns.server)
    print(f"command: {entry.get('command')}")
    print(f"args: {entry.get('args')}")
    print(f"env: {entry.get('env') or {}}")
    if overrides:
        print(f"env overrides for this run (merged over the above): {overrides}")
        record(
            "env-overrides",
            "WARN",
            "this run added {" + ", ".join(sorted(overrides)) + "} to the "
            "configured server environment; command and args are still the "
            "client config's own",
        )
    ok, detail = preflight(entry)
    record("client-config-preflight", "PASS" if ok else "FAIL", detail)
    if not ok:
        return summarize()

    try:
        asyncio.run(run(ns, entry, overrides))
    except BaseException as exc:  # noqa: BLE001 - the check must report, not traceback
        record("server-launch", "FAIL", _describe_failure(exc))

    if ns.json:
        out = Path(ns.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "config": str(config_path),
                    "server": ns.server,
                    "env_overrides": overrides,
                    "checks": CHECK_RESULTS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nreport written: {out}")
    return summarize()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
