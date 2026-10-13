"""RAGForge MCP integration (V17).

An MCP server that exposes RAGForge knowledge bases to MCP-compatible clients.
It is a separate entry point from the FastAPI app: the API process never imports
this package, and the MCP process never starts a web server.

Layering (kept strict so the logic is testable without the SDK):

    app.mcp.schemas  — the structured payloads a client receives
    app.mcp.errors   — the error vocabulary (safe to show a client)
    app.mcp.limits   — input/result bounds for untrusted callers
    app.mcp.bridge   — calls RAGForge's EXISTING services (no parallel retrieval)
    app.mcp.server   — MCP wiring: tools, resources, stdio transport

Requires the optional dependency `mcp` (see requirements-mcp.txt). Importing the
bridge or the schemas does NOT require it; only `app.mcp.server` does.
"""

__all__ = ["bridge", "errors", "limits", "schemas"]
