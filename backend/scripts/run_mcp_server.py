"""Absolute-path launcher for the RAGForge MCP server (V17).

MCP clients such as Claude Desktop launch a server with

    "command": "<python>",
    "args": ["<absolute path to this file>"]

and no working directory, so this script puts `backend/` on `sys.path` itself and
then hands over to the real entry point. That makes the same file work from any
cwd and from any client config.

It is a thin wrapper on purpose: all behaviour lives in `app.mcp.server`, so the
launcher cannot drift from the module entry point (`python -m app.mcp`).

Usage (normally started BY the MCP client, not by hand):

    backend\\.venv\\Scripts\\python.exe backend\\scripts\\run_mcp_server.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(BACKEND_ROOT))

from app.mcp.server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
