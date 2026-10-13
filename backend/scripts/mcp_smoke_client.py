"""A real MCP client for manual smoke checks (V17).

Starts the RAGForge MCP server exactly the way an MCP client does (stdio
subprocess, absolute paths) and runs the documented flow end to end:

    handshake -> list tools -> list knowledge bases -> status -> search
              -> evidence by chunk id -> grounded answer -> evaluation lifecycle

It prints what the server returned, with nothing added, so it is also a
reproducible record of what an external client actually sees.

Usage (from anywhere):

    backend\\.venv\\Scripts\\python.exe backend\\scripts\\mcp_smoke_client.py
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\mcp_smoke_client.py --kb kb_f278c283c748 --query "What is an engine?"

Notes:
* it reads the real RAGForge workspace (DATA_DIR from .env / the environment),
  so a search records a retrieval run just as a UI query would;
* it never writes knowledge, never ingests and never scores a benchmark: the
  evaluation step is a dry run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

from mcp import Client, StdioServerParameters  # noqa: E402

SERVER_SCRIPT = BACKEND_ROOT / "scripts" / "run_mcp_server.py"


def _show(title: str, payload: object, limit: int = 700) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))
    print(text[:limit] + ("\n… (truncated)" if len(text) > limit else ""))


async def run(kb_id: str | None, query: str, question: str | None) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(BACKEND_ROOT),
    )
    async with Client(params, raise_exceptions=False) as client:
        tools = (await client.list_tools()).tools
        print("tools:", ", ".join(t.name for t in tools))

        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        print("resources:", [str(r.uri) for r in resources.resources])
        print("resource templates:", [t.uri_template for t in templates.resource_templates])

        listing = await client.call_tool("list_knowledge_bases", {})
        print("is_error:", listing.is_error)
        body = listing.structured_content or {}
        _show("list_knowledge_bases", body, limit=900)
        if listing.is_error:
            return 1

        target = kb_id or (body["knowledge_bases"][0]["id"] if body["knowledge_bases"] else None)
        if not target:
            print("\nno knowledge base exists in this workspace; nothing to search")
            return 0

        status = await client.call_tool("get_knowledge_base_status", {"kb_id": target})
        _show("get_knowledge_base_status", status.structured_content)

        search = await client.call_tool(
            "search_knowledge_base", {"kb_id": target, "query": query, "top_k": 3}
        )
        print("\nsearch is_error:", search.is_error)
        results = (search.structured_content or {}).get("results", [])
        for hit in results:
            print(
                f"  #{hit['rank']} {hit['chunk_id']} score={hit['retrieval_score']} "
                f"{(hit.get('source_title') or '')} :: {hit['text'][:90]!r}"
            )
        if results:
            chunk_id = results[0]["chunk_id"]
            evidence = await client.call_tool(
                "get_evidence", {"kb_id": target, "chunk_ids": [chunk_id]}
            )
            _show("get_evidence", evidence.structured_content, limit=500)

        answer = await client.call_tool(
            "answer_from_knowledge_base", {"kb_id": target, "question": question or query}
        )
        payload = answer.structured_content or {}
        print(
            "\nanswer:",
            f"status={payload.get('status')} grounded={payload.get('grounded')} "
            f"grounding_state={payload.get('grounding_state')} citations={payload.get('citation_count')}",
        )
        print("answer text:", (payload.get("answer_text") or payload.get("abstention_reason") or "")[:300])

        evaluation = await client.call_tool(
            "evaluate_retrieval", {"kb_id": target, "dry_run": True}
        )
        lifecycle = (evaluation.structured_content or {}).get("lifecycle", {})
        print(
            "\nevaluate_retrieval (dry run):",
            f"allowed={lifecycle.get('allowed')} reason={lifecycle.get('reason')}",
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default="", help="knowledge base id (default: the first one listed)")
    parser.add_argument("--query", default="What is an engine?", help="search query")
    parser.add_argument("--question", default="", help="question for the grounded answer")
    args = parser.parse_args(argv)
    return asyncio.run(run(args.kb or None, args.query, args.question or None))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
