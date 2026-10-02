"""SQLite persistence via repositories.

Design decision (documented in docs/architecture.md): the MVP uses SQLite with
JSON-serialized Pydantic models. This keeps setup trivial while the repository
interfaces allow swapping to PostgreSQL later without touching services.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Generic, TypeVar

from app.schemas.models import (
    BenchmarkVersion,
    BuildRun,
    Chunk,
    Document,
    DomainSpec,
    EvaluationQuestion,
    EvaluationRun,
    KnowledgeBase,
    Source,
)

ModelT = TypeVar("ModelT", bound=Any)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS domain_specs (
    kb_id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    url TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks(kb_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE TABLE IF NOT EXISTS evaluation_questions (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS build_runs (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmark_versions (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    version TEXT NOT NULL,
    data TEXT NOT NULL
);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


class Repository:
    """SQLite-backed storage for RAGForge entities."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = _connect(db_path)
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- generic helpers ----------------------------------------------------

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    @staticmethod
    def _to_row(model: Any) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=False)

    @staticmethod
    def _from_row(row: sqlite3.Row | None, cls: type[ModelT]) -> ModelT | None:
        if row is None:
            return None
        return cls.model_validate(json.loads(row["data"]))

    # -- knowledge bases -----------------------------------------------------

    def create_kb(self, kb: KnowledgeBase) -> None:
        self._execute("INSERT INTO knowledge_bases (id, data) VALUES (?, ?)", (kb.id, self._to_row(kb)))

    def list_kbs(self) -> list[KnowledgeBase]:
        rows = self._conn.execute("SELECT data FROM knowledge_bases ORDER BY data->>'$.created_at' DESC").fetchall()
        return [self._from_row(r, KnowledgeBase) for r in rows]  # type: ignore[misc]

    def get_kb(self, kb_id: str) -> KnowledgeBase | None:
        row = self._conn.execute("SELECT data FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
        return self._from_row(row, KnowledgeBase)

    def update_kb(self, kb: KnowledgeBase) -> None:
        self._execute("UPDATE knowledge_bases SET data = ? WHERE id = ?", (self._to_row(kb), kb.id))

    def delete_kb(self, kb_id: str) -> None:
        for sql, params in [
            ("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,)),
            ("DELETE FROM domain_specs WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM sources WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM documents WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM chunks WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM evaluation_questions WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM evaluation_runs WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM build_runs WHERE kb_id = ?", (kb_id,)),
        ]:
            self._execute(sql, params)

    # -- domain spec ----------------------------------------------------------

    def save_domain_spec(self, spec: DomainSpec) -> None:
        self._execute(
            "INSERT INTO domain_specs (kb_id, data) VALUES (?, ?) "
            "ON CONFLICT(kb_id) DO UPDATE SET data = excluded.data",
            (spec.kb_id, self._to_row(spec)),
        )

    def get_domain_spec(self, kb_id: str) -> DomainSpec | None:
        row = self._conn.execute("SELECT data FROM domain_specs WHERE kb_id = ?", (kb_id,)).fetchone()
        return self._from_row(row, DomainSpec)

    # -- sources ---------------------------------------------------------------

    def create_source(self, source: Source) -> None:
        self._execute(
            "INSERT INTO sources (id, kb_id, url, data) VALUES (?, ?, ?, ?)",
            (source.id, source.kb_id, source.url, self._to_row(source)),
        )

    def list_sources(self, kb_id: str) -> list[Source]:
        rows = self._conn.execute(
            "SELECT data FROM sources WHERE kb_id = ? ORDER BY data->>'$.created_at'", (kb_id,)
        ).fetchall()
        return [self._from_row(r, Source) for r in rows]  # type: ignore[misc]

    def get_source(self, kb_id: str, source_id: str) -> Source | None:
        row = self._conn.execute(
            "SELECT data FROM sources WHERE kb_id = ? AND id = ?", (kb_id, source_id)
        ).fetchone()
        return self._from_row(row, Source)

    def update_source(self, source: Source) -> None:
        self._execute("UPDATE sources SET data = ? WHERE id = ?", (self._to_row(source), source.id))

    def find_source_by_url(self, kb_id: str, url: str) -> Source | None:
        row = self._conn.execute(
            "SELECT data FROM sources WHERE kb_id = ? AND url = ?", (kb_id, url)
        ).fetchone()
        return self._from_row(row, Source)

    # -- documents ---------------------------------------------------------------

    def create_document(self, doc: Document) -> None:
        self._execute(
            "INSERT INTO documents (id, kb_id, source_id, content_hash, data) VALUES (?, ?, ?, ?, ?)",
            (doc.id, doc.kb_id, doc.source_id, doc.content_hash, self._to_row(doc)),
        )

    def list_documents(self, kb_id: str) -> list[Document]:
        rows = self._conn.execute(
            "SELECT data FROM documents WHERE kb_id = ? ORDER BY data->>'$.ingestion_timestamp'", (kb_id,)
        ).fetchall()
        return [self._from_row(r, Document) for r in rows]  # type: ignore[misc]

    def get_document(self, kb_id: str, document_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT data FROM documents WHERE kb_id = ? AND id = ?", (kb_id, document_id)
        ).fetchone()
        return self._from_row(row, Document)

    def find_document_by_hash(self, kb_id: str, content_hash: str) -> Document | None:
        row = self._conn.execute(
            "SELECT data FROM documents WHERE kb_id = ? AND content_hash = ?", (kb_id, content_hash)
        ).fetchone()
        return self._from_row(row, Document)

    # -- chunks ---------------------------------------------------------------

    def create_chunks(self, chunks: list[Chunk]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO chunks (id, kb_id, document_id, chunk_index, data) VALUES (?, ?, ?, ?, ?)",
                [(c.id, c.kb_id, c.document_id, c.chunk_index, self._to_row(c)) for c in chunks],
            )
            self._conn.commit()

    def list_chunks(self, kb_id: str, document_id: str | None = None, limit: int = 500, offset: int = 0) -> list[Chunk]:
        if document_id:
            rows = self._conn.execute(
                "SELECT data FROM chunks WHERE kb_id = ? AND document_id = ? "
                "ORDER BY document_id, chunk_index LIMIT ? OFFSET ?",
                (kb_id, document_id, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data FROM chunks WHERE kb_id = ? ORDER BY document_id, chunk_index LIMIT ? OFFSET ?",
                (kb_id, limit, offset),
            ).fetchall()
        return [self._from_row(r, Chunk) for r in rows]  # type: ignore[misc]

    def count_chunks(self, kb_id: str) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE kb_id = ?", (kb_id,)).fetchone()
        return int(row["n"]) if row else 0

    def get_chunks_by_ids(self, kb_id: str, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT data FROM chunks WHERE kb_id = ? AND id IN ({placeholders})",
            (kb_id, *chunk_ids),
        ).fetchall()
        return [self._from_row(r, Chunk) for r in rows]  # type: ignore[misc]

    def delete_chunks_for_document(self, kb_id: str, document_id: str) -> None:
        self._execute("DELETE FROM chunks WHERE kb_id = ? AND document_id = ?", (kb_id, document_id))

    # -- evaluation ---------------------------------------------------------------

    def create_evaluation_question(self, q: EvaluationQuestion) -> None:
        self._execute(
            "INSERT INTO evaluation_questions (id, kb_id, data) VALUES (?, ?, ?)",
            (q.id, q.kb_id, self._to_row(q)),
        )

    def list_evaluation_questions(self, kb_id: str) -> list[EvaluationQuestion]:
        rows = self._conn.execute(
            "SELECT data FROM evaluation_questions WHERE kb_id = ?", (kb_id,)
        ).fetchall()
        return [self._from_row(r, EvaluationQuestion) for r in rows]  # type: ignore[misc]

    def get_evaluation_question(self, kb_id: str, question_id: str) -> EvaluationQuestion | None:
        row = self._conn.execute(
            "SELECT data FROM evaluation_questions WHERE kb_id = ? AND id = ?", (kb_id, question_id)
        ).fetchone()
        return self._from_row(row, EvaluationQuestion)

    # -- benchmark versions (V3 Phase A) -------------------------------------

    def create_benchmark_version(self, bv: BenchmarkVersion) -> None:
        self._execute(
            "INSERT INTO benchmark_versions (id, kb_id, version, data) VALUES (?, ?, ?, ?)",
            (bv.id, bv.kb_id, bv.version, self._to_row(bv)),
        )

    def get_benchmark_version(self, kb_id: str, bv_id: str) -> BenchmarkVersion | None:
        row = self._conn.execute(
            "SELECT data FROM benchmark_versions WHERE kb_id = ? AND id = ?", (kb_id, bv_id)
        ).fetchone()
        return self._from_row(row, BenchmarkVersion)

    def list_benchmark_versions(self, kb_id: str) -> list[BenchmarkVersion]:
        rows = self._conn.execute(
            "SELECT data FROM benchmark_versions WHERE kb_id = ? ORDER BY data->>'$.created_at' DESC",
            (kb_id,),
        ).fetchall()
        return [self._from_row(r, BenchmarkVersion) for r in rows]  # type: ignore[misc]

    def update_benchmark_version(self, bv: BenchmarkVersion) -> None:
        self._execute(
            "UPDATE benchmark_versions SET data = ? WHERE id = ?",
            (self._to_row(bv), bv.id),
        )

    def create_evaluation_run(self, run: EvaluationRun) -> None:
        self._execute(
            "INSERT INTO evaluation_runs (id, kb_id, data) VALUES (?, ?, ?)",
            (run.id, run.kb_id, self._to_row(run)),
        )

    def list_evaluation_runs(self, kb_id: str) -> list[EvaluationRun]:
        rows = self._conn.execute(
            "SELECT data FROM evaluation_runs WHERE kb_id = ? ORDER BY data->>'$.started_at' DESC", (kb_id,)
        ).fetchall()
        return [self._from_row(r, EvaluationRun) for r in rows]  # type: ignore[misc]

    # -- build runs ---------------------------------------------------------------

    def create_build_run(self, run: BuildRun) -> None:
        self._execute(
            "INSERT INTO build_runs (id, kb_id, data) VALUES (?, ?, ?)",
            (run.id, run.kb_id, self._to_row(run)),
        )

    def update_build_run(self, run: BuildRun) -> None:
        self._execute("UPDATE build_runs SET data = ? WHERE id = ?", (self._to_row(run), run.id))

    def latest_build_run(self, kb_id: str) -> BuildRun | None:
        rows = self._conn.execute(
            "SELECT data FROM build_runs WHERE kb_id = ? ORDER BY data->>'$.started_at' DESC LIMIT 1",
            (kb_id,),
        ).fetchall()
        return self._from_row(rows[0], BuildRun) if rows else None
