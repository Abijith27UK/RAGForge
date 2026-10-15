"""Shared pytest fixtures.

Includes one SESSION-WIDE GUARD (`_real_workspace_db_guard`) whose only job is to
fail the suite if it wrote to the developer's real workspace database. It exists
because that already happened once: the V18 investigation (2026-10-15) found that
`mcp`'s `StdioServerParameters.env` is merged over `get_default_environment()`,
which inherits a small whitelist (PATH, HOME, ...) rather than `os.environ`, so a
nested stdio MCP server launched WITHOUT `DATA_DIR` silently opened
`backend/data/ragforge.db` and wrote `mcp_operations` rows there. Tests must hand
a scratch `DATA_DIR` to every server they spawn; this guard turns a silent leak
into a failing test, and it never writes to the real database itself.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

#: Tables that record user activity. A test run must change NONE of their row
#: counts in the real workspace database.
_REAL_DB_WATCH_TABLES = (
    "knowledge_bases",
    "documents",
    "chunks",
    "mcp_operations",
    "retrieval_runs",
    "answers",
    "answer_runs",
    "answer_traces",
    "evaluation_questions",
    "corpus_revisions",
)


def real_workspace_db_path() -> Path | None:
    """The configured database path, resolved OUTSIDE any test's monkeypatching."""
    try:
        from app.config import Settings

        return Settings().db_path
    except Exception:  # noqa: BLE001 - the guard must never break collection
        return None


def _row_counts(db: Path) -> dict[str, int] | None:
    if not db.exists():
        return None
    counts: dict[str, int] = {}
    try:
        conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        for table in _REAL_DB_WATCH_TABLES:
            try:
                counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                continue
    finally:
        conn.close()
    return counts@pytest.fixture(scope="session", autouse=True)
def _real_workspace_db_guard():
    """Fail if the suite created or deleted rows in the real workspace DB."""
    db = real_workspace_db_path()
    before = _row_counts(db) if db else None
    yield
    after = _row_counts(db) if db else None
    if before is None or after is None:
        return  # no real database here (fresh checkout): nothing to protect
    changed = {
        table: (before.get(table), after.get(table))
        for table in set(before) | set(after)
        if before.get(table) != after.get(table)
    }
    assert not changed, (
        "the test suite wrote to the developer's REAL workspace database "
        f"{db}.\nTable(s) whose row counts changed (before -> after): {changed}\n"
        "Every nested MCP server must be launched with a scratch DATA_DIR: "
        "`mcp`'s StdioServerParameters.env only ADDS to a whitelist, so an "
        "omitted DATA_DIR silently opens this database."
    )

from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.schemas.models import KnowledgeBase, Source, SourceType  # noqa: E402
from app.utils.ids import new_id  # noqa: E402


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository(str(tmp_path / "test.db"))


@pytest.fixture
def kb() -> KnowledgeBase:
    return KnowledgeBase(
        id=new_id("kb"),
        name="Test KB",
        domain="Automobile Engineering",
        purpose="Engineering education",
        target_audience="Engineering students",
        depth="technical",
    )


@pytest.fixture
def source(kb) -> Source:
    return Source(
        id=new_id("src"),
        kb_id=kb.id,
        url="https://www.sae.org/standards/some-standard",
        title="SAE Standard Example",
        source_type=SourceType.WEB_PAGE,
        publisher="SAE International",
    )
