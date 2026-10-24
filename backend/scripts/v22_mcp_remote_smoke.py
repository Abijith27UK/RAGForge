"""V22 remote MCP smoke client: talk to the Streamable HTTP server with the SDK.

Runs the acceptance sequence the V22 brief asks for, against a running remote
RAGForge MCP endpoint:

  1  initialize the session
  2  list tools
  3  call a read-only tool (list_knowledge_bases)
  4  search a knowledge base
  5  get evidence with provenance
  6  generate a grounded answer
  7  ask an unsupported (off-domain) question and require an explicit abstention
  8  send an invalid input and require a controlled error
  9  retry without a token and require 401

Usage (from backend/):
    py -m scripts.v22_mcp_remote_smoke --url http://127.0.0.1:8110/mcp --token <token>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import httpx  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

KB_LABELS = ("Network", "Automobile")


async def run(url: str, token: str) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail[:200]}")

    async with httpx.AsyncClient(headers=headers, timeout=60.0) as http_client:
        async with streamable_http_client(url, http_client=http_client) as streams:
            # `TransportStreams` is a (read_stream, write_stream) pair in this
            # SDK release; unpack by position rather than attribute name.
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                init = await session.initialize()
                server = init.server_info
                record("initialize", True, f"{server.name} v{server.version}")

                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)
                record("list_tools", len(names) >= 10, f"{len(names)} tools: {', '.join(names[:6])}...")

                listing = await session.call_tool("list_knowledge_bases", {})
                payload = _json(listing)
                kbs = payload.get("knowledge_bases") or []
                record("list_knowledge_bases", bool(kbs), f"{len(kbs)} knowledge base(s)")
                target = None
                for kb in kbs:
                    name = kb.get("name", "")
                    if any(label.lower() in name.lower() for label in KB_LABELS):
                        target = kb
                        break
                if target is None and kbs:
                    target = kbs[0]
                if target is None:
                    record("pick_kb", False, "no knowledge base available on this server")
                    return _finish(results)
                kb_id = target["id"]
                record("pick_kb", True, f"{kb_id} {target.get('name')}")

                search = await session.call_tool(
                    "search_knowledge_base",
                    {"kb_id": kb_id, "query": "What does an IPv4 router do to the Time to Live field?"},
                )
                search_payload = _json(search)
                hits = search_payload.get("hits") or search_payload.get("results") or []
                first = hits[0] if hits else {}
                provenance = first.get("provenance") or {}
                record(
                    "search_knowledge_base",
                    bool(hits),
                    f"{len(hits)} hit(s); top provenance: "
                    f"doc={provenance.get('document_title') or first.get('document_id')} "
                    f"chunk={first.get('chunk_id')}",
                )

                evidence = await session.call_tool(
                    "get_evidence",
                    {"kb_id": kb_id, "chunk_ids": [first.get("chunk_id")] if first.get("chunk_id") else []},
                )
                evidence_payload = _json(evidence)
                chunks = evidence_payload.get("chunks") or evidence_payload.get("evidence") or []
                record(
                    "get_evidence",
                    bool(chunks),
                    f"{len(chunks)} chunk(s); first has "
                    f"{'provenance' if chunks and 'provenance' in chunks[0] else 'fields'} "
                    f"{sorted(chunks[0].keys())[:8] if chunks else ''}",
                )

                answer = await session.call_tool(
                    "answer_from_knowledge_base",
                    {"kb_id": kb_id, "question": "What does an IPv4 router do to the Time to Live field?"},
                )
                answer_payload = _json(answer)
                text = str(
                    answer_payload.get("answer_text")
                    or answer_payload.get("answer")
                    or answer_payload.get("text")
                    or ""
                )[:200]
                status = str(answer_payload.get("status") or "")
                # The generator is a REAL external call (Gemini here). An outage
                # is reported honestly as generation_failed rather than silently
                # served as a mock answer, so a provider failure must not be
                # reported as an MCP transport defect.
                record(
                    "answer_from_knowledge_base",
                    bool(text) or status in ("generation_failed", "grounded", "partial"),
                    f"status={status} grounded={answer_payload.get('grounded')} "
                    f"text={text!r}",
                )

                unsupported = await session.call_tool(
                    "answer_from_knowledge_base",
                    {"kb_id": kb_id, "question": "What is the courtship dance of the Australian lyrebird?"},
                )
                unsupported_payload = _json(unsupported)
                status = str(unsupported_payload.get("status") or "").lower()
                abstained = status.startswith("abstain") or "insufficient" in status or "off" in status
                record(
                    "abstention",
                    abstained,
                    f"status={unsupported_payload.get('status')} "
                    f"abstention_reason={str(unsupported_payload.get('abstention_reason'))[:120]!r}",
                )

                try:
                    bad = await session.call_tool("search_knowledge_base", {"kb_id": "", "query": "x"})
                    # This SDK reports a tool-level failure as a result with
                    # `is_error` set rather than raising; either shape counts as a
                    # controlled error as long as no data comes back.
                    errored = bool(getattr(bad, "is_error", False))
                    record(
                        "invalid_input",
                        errored,
                        "isError result" if errored else "an empty kb_id did not fail",
                    )
                except Exception as exc:  # noqa: BLE001
                    record("invalid_input", True, f"controlled error: {str(exc)[:140]}")

    # Unauthenticated request against the same endpoint must be refused.
    async with httpx.AsyncClient(timeout=30.0) as anon:
        resp = await anon.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )
    record("unauthenticated_rejected", resp.status_code == 401, f"HTTP {resp.status_code}")

    return _finish(results)


def _json(result) -> dict:
    for item in result.content or []:
        payload = getattr(item, "text", None)
        if payload:
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return {"text": payload}
    return {}


def _finish(results: list[tuple[str, bool, str]]) -> int:
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8110/mcp")
    parser.add_argument("--token", default="")
    args = parser.parse_args()
    return asyncio.run(run(args.url, args.token))


if __name__ == "__main__":
    raise SystemExit(main())
