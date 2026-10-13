"""`python -m app.mcp` — run the RAGForge MCP server over stdio."""
from __future__ import annotations

import sys

from app.mcp.server import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
