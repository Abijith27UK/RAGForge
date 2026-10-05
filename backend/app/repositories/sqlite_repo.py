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

from app.schemas.answer import AnswerRun, Conversation, Message
from app.schemas.corpus import CorpusVersion, IngestionBatch, IngestionItem
from app.schemas.retrieval import (
    RetrievalConfigRecord,
    RetrievalRun,
    RetrievalRunSummary,
)
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
CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(kb_id, content_hash);
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

-- ---------------------------------------------------------------------------
-- V5 corpus engineering: resumable ingestion + corpus snapshots.
-- `item_key` is UNIQUE per batch so a resume can never create a second row
-- for the same upload slot, which is what makes resume idempotent at the
-- database level rather than only in application code.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_batches (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_batches_kb ON ingestion_batches(kb_id, created_at);
CREATE TABLE IF NOT EXISTS ingestion_items (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    kb_id TEXT NOT NULL,
    item_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    stage TEXT NOT NULL DEFAULT 'queued',
    content_hash TEXT,
    document_id TEXT,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL,
    UNIQUE (batch_id, item_key)
);
CREATE INDEX IF NOT EXISTS idx_items_batch ON ingestion_items(batch_id);
CREATE INDEX IF NOT EXISTS idx_items_kb ON ingestion_items(kb_id, status);
CREATE TABLE IF NOT EXISTS corpus_versions (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    version TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL,
    UNIQUE (kb_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_corpus_versions_kb ON corpus_versions(kb_id, created_at);
CREATE TABLE IF NOT EXISTS corpus_revisions (
    kb_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS lexical_indexes (
    kb_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retrieval_configs (
    kb_id TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retrieval_runs (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieval_runs_kb ON retrieval_runs(kb_id, created_at);
CREATE TABLE IF NOT EXISTS answers (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answers_kb ON answers(kb_id, created_at);
CREATE TABLE IF NOT EXISTS answer_traces (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    answer_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answer_traces_kb ON answer_traces(kb_id, created_at);
CREATE TABLE IF NOT EXISTS answer_runs (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answer_runs_kb ON answer_runs(kb_id, created_at);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_kb ON conversations(kb_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    kb_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS answer_evaluation_runs (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    benchmark_fingerprint TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answer_eval_runs_kb ON answer_evaluation_runs(kb_id, created_at);
CREATE TABLE IF NOT EXISTS answer_reviews (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    answer_id TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL,
    verdict TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answer_reviews_run
    ON answer_reviews(kb_id, run_id, question_id, created_at);
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
            ("DELETE FROM benchmark_versions WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM ingestion_items WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM ingestion_batches WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM corpus_versions WHERE kb_id = ?", (kb_id,)),
            # V6: derived/debug stores must not outlive the KB they describe.
            ("DELETE FROM corpus_revisions WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM lexical_indexes WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM retrieval_configs WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM retrieval_runs WHERE kb_id = ?", (kb_id,)),
            # V7: answers and their audit traces must not outlive the KB.
            ("DELETE FROM answers WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM answer_traces WHERE kb_id = ?", (kb_id,)),
            # V7 Phase 13: chat history and answer runs go with the KB too.
            ("DELETE FROM answer_runs WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM messages WHERE kb_id = ?", (kb_id,)),
            ("DELETE FROM conversations WHERE kb_id = ?", (kb_id,)),
            # V8: answer-quality evaluation runs are derived and go with the KB.
            ("DELETE FROM answer_evaluation_runs WHERE kb_id = ?", (kb_id,)),
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

    def update_document(self, doc: Document) -> None:
        self._execute(
            "UPDATE documents SET data = ?, content_hash = ? WHERE id = ?",
            (self._to_row(doc), doc.content_hash, doc.id),
        )

    def delete_document(self, kb_id: str, document_id: str) -> bool:
        row = self._conn.execute(
            "SELECT id FROM documents WHERE kb_id = ? AND id = ?", (kb_id, document_id)
        ).fetchone()
        if row is None:
            return False
        self.delete_chunks_for_document(kb_id, document_id)
        self._execute("DELETE FROM documents WHERE kb_id = ? AND id = ?", (kb_id, document_id))
        return True

    def count_documents(self, kb_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        return int(row["n"]) if row else 0

    def delete_source(self, kb_id: str, source_id: str) -> bool:
        row = self._conn.execute(
            "SELECT id FROM sources WHERE kb_id = ? AND id = ?", (kb_id, source_id)
        ).fetchone()
        if row is None:
            return False
        self._execute("DELETE FROM sources WHERE kb_id = ? AND id = ?", (kb_id, source_id))
        return True

    def find_source_by_id(self, kb_id: str, source_id: str) -> Source | None:
        return self.get_source(kb_id, source_id)

    # -- chunks ---------------------------------------------------------------

    def create_chunks(self, chunks: list[Chunk]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO chunks (id, kb_id, document_id, chunk_index, data) VALUES (?, ?, ?, ?, ?)",
                [(c.id, c.kb_id, c.document_id, c.chunk_index, self._to_row(c)) for c in chunks],
            )
            self._conn.commit()
        # V6: every chunk write invalidates derived lexical indexes. One bump per
        # call keeps the staleness check O(1) instead of re-reading all chunk text.
        for kb_id in {c.kb_id for c in chunks}:
            self.bump_corpus_revision(kb_id)

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
        self.bump_corpus_revision(kb_id)

    def list_chunk_lexical_rows(self, kb_id: str) -> list[dict[str, Any]]:
        """Bare rows needed to build a lexical index: id + text (+ provenance keys).

        Ordered by (document_id, chunk_index) so index construction is
        deterministic for a given corpus state.
        """
        rows = self._conn.execute(
            "SELECT id, data FROM chunks WHERE kb_id = ? ORDER BY document_id, chunk_index",
            (kb_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            data = json.loads(row["data"])
            out.append(
                {
                    "chunk_id": row["id"],
                    "document_id": data.get("document_id") or "",
                    "source_id": data.get("source_id"),
                    "text": data.get("text") or "",
                }
            )
        return out

    # -- corpus revision (V6) -------------------------------------------------

    def corpus_revision(self, kb_id: str) -> int:
        """Monotonic counter bumped on every chunk write. 0 when never written."""
        row = self._conn.execute(
            "SELECT revision FROM corpus_revisions WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        return int(row["revision"]) if row else 0

    def bump_corpus_revision(self, kb_id: str) -> int:
        """Increment (and return) the corpus revision for a knowledge base."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO corpus_revisions (kb_id, revision) VALUES (?, 1) "
                "ON CONFLICT(kb_id) DO UPDATE SET revision = revision + 1",
                (kb_id,),
            )
            self._conn.commit()
        return self.corpus_revision(kb_id)

    # -- lexical indexes (V6 Phase 3) ----------------------------------------

    def save_lexical_index(
        self, kb_id: str, revision: int, created_at: str, payload_json: str
    ) -> None:
        """Persist the BM25 statistics built for `revision` (idempotent)."""
        self._execute(
            "INSERT INTO lexical_indexes (kb_id, revision, created_at, data) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(kb_id) DO UPDATE SET revision = excluded.revision, "
            "created_at = excluded.created_at, data = excluded.data",
            (kb_id, int(revision), created_at, payload_json),
        )

    def get_lexical_index(self, kb_id: str) -> tuple[int, str] | None:
        """(revision, payload_json) or None when no index was persisted."""
        row = self._conn.execute(
            "SELECT revision, data FROM lexical_indexes WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        if row is None:
            return None
        return (int(row["revision"]), str(row["data"]))

    def delete_lexical_index(self, kb_id: str) -> None:
        self._execute("DELETE FROM lexical_indexes WHERE kb_id = ?", (kb_id,))

    # -- retrieval configuration + runs (V6 Phase 8) -------------------------

    def save_retrieval_config(self, record: RetrievalConfigRecord) -> None:
        self._execute(
            "INSERT INTO retrieval_configs (kb_id, updated_at, data) VALUES (?, ?, ?) "
            "ON CONFLICT(kb_id) DO UPDATE SET updated_at = excluded.updated_at, "
            "data = excluded.data",
            (record.kb_id, record.updated_at.isoformat(), self._to_row(record)),
        )

    def get_retrieval_config(self, kb_id: str) -> RetrievalConfigRecord | None:
        row = self._conn.execute(
            "SELECT data FROM retrieval_configs WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        return self._from_row(row, RetrievalConfigRecord)

    def create_retrieval_run(self, run: RetrievalRun) -> None:
        self._execute(
            "INSERT INTO retrieval_runs (id, kb_id, created_at, data) VALUES (?, ?, ?, ?)",
            (run.id, run.kb_id, run.created_at.isoformat(), self._to_row(run)),
        )

    def get_retrieval_run(self, kb_id: str, run_id: str) -> RetrievalRun | None:
        row = self._conn.execute(
            "SELECT data FROM retrieval_runs WHERE kb_id = ? AND id = ?", (kb_id, run_id)
        ).fetchone()
        return self._from_row(row, RetrievalRun)

    def list_retrieval_runs(self, kb_id: str, limit: int = 50) -> list[RetrievalRun]:
        rows = self._conn.execute(
            "SELECT data FROM retrieval_runs WHERE kb_id = ? ORDER BY created_at DESC LIMIT ?",
            (kb_id, limit),
        ).fetchall()
        return [self._from_row(r, RetrievalRun) for r in rows]  # type: ignore[misc]

    def list_retrieval_run_summaries(self, kb_id: str, limit: int = 50) -> list[RetrievalRunSummary]:
        return [
            RetrievalRunSummary(
                id=run.id,
                kb_id=run.kb_id,
                created_at=run.created_at,
                query=run.query,
                strategy=run.strategy,
                result_count=run.result_count,
                total_ms=run.timings.total_ms,
                reranker_status=run.reranker_status,
            )
            for run in self.list_retrieval_runs(kb_id, limit=limit)
        ]

    # -- answers + answer traces (V7) ----------------------------------------

    def create_answer(self, answer: "Answer") -> None:
        self._execute(
            "INSERT INTO answers (id, kb_id, created_at, data) VALUES (?, ?, ?, ?)",
            (answer.answer_id, answer.kb_id, answer.created_at.isoformat(),
             self._to_row(answer)),
        )

    def get_answer(self, kb_id: str, answer_id: str):
        row = self._conn.execute(
            "SELECT data FROM answers WHERE kb_id = ? AND id = ?", (kb_id, answer_id)
        ).fetchone()
        from app.schemas.answer import Answer

        return self._from_row(row, Answer)

    def create_answer_trace(self, trace: "AnswerTrace") -> None:
        self._execute(
            "INSERT INTO answer_traces (id, kb_id, answer_id, created_at, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (trace.id, trace.kb_id, trace.answer_id, trace.created_at.isoformat(),
             self._to_row(trace)),
        )

    def get_answer_trace(self, kb_id: str, trace_id: str):
        row = self._conn.execute(
            "SELECT data FROM answer_traces WHERE kb_id = ? AND id = ?", (kb_id, trace_id)
        ).fetchone()
        from app.schemas.answer import AnswerTrace

        return self._from_row(row, AnswerTrace)

    # -- answer runs + conversations (V7 Phase 13) ----------------------------

    def create_answer_run(self, run: "AnswerRun") -> None:
        self._execute(
            "INSERT INTO answer_runs (id, kb_id, created_at, data) VALUES (?, ?, ?, ?)",
            (run.id, run.kb_id, run.created_at.isoformat(), self._to_row(run)),
        )

    def get_answer_run(self, kb_id: str, run_id: str):
        row = self._conn.execute(
            "SELECT data FROM answer_runs WHERE kb_id = ? AND id = ?", (kb_id, run_id)
        ).fetchone()
        from app.schemas.answer import AnswerRun

        return self._from_row(row, AnswerRun)

    def list_answer_runs(self, kb_id: str, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT data FROM answer_runs WHERE kb_id = ? ORDER BY created_at DESC LIMIT ?",
            (kb_id, limit),
        ).fetchall()
        from app.schemas.answer import AnswerRun

        return [self._from_row(r, AnswerRun) for r in rows]  # type: ignore[misc]

    def create_conversation(self, conversation: "Conversation") -> None:
        self._execute(
            "INSERT INTO conversations (id, kb_id, created_at, updated_at, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                conversation.id,
                conversation.kb_id,
                conversation.created_at.isoformat(),
                conversation.updated_at.isoformat(),
                self._to_row(conversation),
            ),
        )

    def update_conversation(self, conversation: "Conversation") -> None:
        self._execute(
            "UPDATE conversations SET updated_at = ?, data = ? WHERE id = ?",
            (
                conversation.updated_at.isoformat(),
                self._to_row(conversation),
                conversation.id,
            ),
        )

    def get_conversation(self, kb_id: str, conversation_id: str):
        row = self._conn.execute(
            "SELECT data FROM conversations WHERE kb_id = ? AND id = ?",
            (kb_id, conversation_id),
        ).fetchone()
        from app.schemas.answer import Conversation

        return self._from_row(row, Conversation)

    def list_conversations(self, kb_id: str, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT data FROM conversations WHERE kb_id = ? ORDER BY updated_at DESC LIMIT ?",
            (kb_id, limit),
        ).fetchall()
        from app.schemas.answer import Conversation

        return [self._from_row(r, Conversation) for r in rows]  # type: ignore[misc]

    def delete_conversation(self, kb_id: str, conversation_id: str) -> None:
        self._execute(
            "DELETE FROM messages WHERE kb_id = ? AND conversation_id = ?",
            (kb_id, conversation_id),
        )
        self._execute(
            "DELETE FROM conversations WHERE kb_id = ? AND id = ?",
            (kb_id, conversation_id),
        )

    def create_message(self, message: "Message") -> None:
        self._execute(
            "INSERT INTO messages (id, conversation_id, kb_id, created_at, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                message.id,
                message.conversation_id,
                message.kb_id,
                message.created_at.isoformat(),
                self._to_row(message),
            ),
        )

    def list_messages(self, conversation_id: str, limit: int = 200) -> list:
        rows = self._conn.execute(
            "SELECT data FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        from app.schemas.answer import Message

        return [self._from_row(r, Message) for r in rows]  # type: ignore[misc]

    # -- answer evaluation runs (V8) ------------------------------------------

    def create_answer_evaluation_run(self, run: "AnswerEvaluationRun") -> None:
        """Insert an answer-evaluation run. Immutable: a duplicate id raises
        rather than overwriting, so an earlier run can never be silently lost."""
        self._execute(
            "INSERT INTO answer_evaluation_runs "
            "(id, kb_id, created_at, benchmark_name, benchmark_fingerprint, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run.id, run.kb_id, run.created_at.isoformat(), run.benchmark_name,
             run.benchmark_fingerprint, self._to_row(run)),
        )

    def get_answer_evaluation_run(self, kb_id: str, run_id: str):
        row = self._conn.execute(
            "SELECT data FROM answer_evaluation_runs WHERE kb_id = ? AND id = ?",
            (kb_id, run_id),
        ).fetchone()
        from app.services.answer_eval.run import AnswerEvaluationRun

        return self._from_row(row, AnswerEvaluationRun)

    def list_answer_evaluation_runs(self, kb_id: str, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT data FROM answer_evaluation_runs WHERE kb_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (kb_id, limit),
        ).fetchall()
        from app.services.answer_eval.run import AnswerEvaluationRun

        return [self._from_row(r, AnswerEvaluationRun) for r in rows]  # type: ignore[misc]

    # -- human answer reviews (V8 STEP 4, append-only) ------------------------

    def create_answer_review(self, review) -> None:
        """Insert ONE human review. Append-only by design: there is no update
        or delete method for reviews, so a later judgement is always a new row
        and earlier ones remain exactly as recorded.

        A duplicate id raises (PRIMARY KEY) rather than overwriting — same
        immutability rule as evaluation runs.
        """
        self._execute(
            "INSERT INTO answer_reviews "
            "(id, kb_id, run_id, question_id, answer_id, reviewer, verdict, "
            "created_at, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                review.id,
                review.kb_id,
                review.run_id,
                review.question_id,
                review.answer_id,
                review.reviewer,
                review.verdict.value,
                review.created_at.isoformat(),
                self._to_row(review),
            ),
        )

    def list_answer_reviews(
        self,
        kb_id: str,
        run_id: str | None = None,
        question_id: str | None = None,
        limit: int = 500,
    ) -> list:
        """Full review history for a KB/run/question, oldest first.

        Ordered by created_at ASC so a reader sees the review sequence a human
        produced; nothing is collapsed to "latest wins".
        """
        sql = "SELECT data FROM answer_reviews WHERE kb_id = ?"
        params: list = [kb_id]
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        if question_id is not None:
            sql += " AND question_id = ?"
            params.append(question_id)
        sql += " ORDER BY created_at ASC, id ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        from app.services.answer_eval.review import AnswerReview

        return [self._from_row(r, AnswerReview) for r in rows]  # type: ignore[misc]

    # -- top-level answer-evaluation access (V8 STEP 10) ----------------------

    def list_answer_evaluation_runs_all(
        self, kb_id: str | None = None, limit: int = 100
    ) -> list:
        """Runs across all knowledge bases (optionally filtered), newest first."""
        if kb_id is not None:
            rows = self._conn.execute(
                "SELECT data FROM answer_evaluation_runs WHERE kb_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (kb_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data FROM answer_evaluation_runs "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        from app.services.answer_eval.run import AnswerEvaluationRun

        return [self._from_row(r, AnswerEvaluationRun) for r in rows]  # type: ignore[misc]

    def get_answer_evaluation_run_by_id(self, run_id: str):
        """Fetch one run by its global id, regardless of knowledge base."""
        row = self._conn.execute(
            "SELECT data FROM answer_evaluation_runs WHERE id = ?", (run_id,)
        ).fetchone()
        from app.services.answer_eval.run import AnswerEvaluationRun

        return self._from_row(row, AnswerEvaluationRun)

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

    # -- ingestion batches (V5 Phase 1) -------------------------------------
    # Idempotency is enforced by UNIQUE(batch_id, item_key) at the schema level,
    # so a resume can never double-insert even if the service retries.

    def create_ingestion_batch(self, batch: IngestionBatch) -> None:
        self._execute(
            "INSERT INTO ingestion_batches (id, kb_id, status, created_at, data) VALUES (?, ?, ?, ?, ?)",
            (batch.id, batch.kb_id, batch.status, batch.created_at.isoformat(), self._to_row(batch)),
        )

    def get_ingestion_batch(self, batch_id: str) -> IngestionBatch | None:
        row = self._conn.execute(
            "SELECT data FROM ingestion_batches WHERE id = ?", (batch_id,)
        ).fetchone()
        return self._from_row(row, IngestionBatch)

    def list_ingestion_batches(self, kb_id: str, limit: int = 50) -> list[IngestionBatch]:
        rows = self._conn.execute(
            "SELECT data FROM ingestion_batches WHERE kb_id = ? ORDER BY created_at DESC LIMIT ?",
            (kb_id, limit),
        ).fetchall()
        return [self._from_row(r, IngestionBatch) for r in rows]  # type: ignore[misc]

    def update_ingestion_batch(self, batch: IngestionBatch) -> None:
        self._execute(
            "UPDATE ingestion_batches SET data = ?, status = ? WHERE id = ?",
            (self._to_row(batch), batch.status, batch.id),
        )

    def create_ingestion_item(self, item: IngestionItem) -> None:
        self._execute(
            "INSERT INTO ingestion_items "
            "(id, batch_id, kb_id, item_key, status, stage, content_hash, document_id, created_at, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.id, item.batch_id, item.kb_id, item.item_key, item.status.value,
                item.stage.value, item.content_hash, item.document_id,
                item.created_at.isoformat(), self._to_row(item),
            ),
        )

    def get_ingestion_item(self, item_id: str) -> IngestionItem | None:
        row = self._conn.execute(
            "SELECT data FROM ingestion_items WHERE id = ?", (item_id,)
        ).fetchone()
        return self._from_row(row, IngestionItem)

    def find_ingestion_item_by_key(self, batch_id: str, item_key: str) -> IngestionItem | None:
        row = self._conn.execute(
            "SELECT data FROM ingestion_items WHERE batch_id = ? AND item_key = ?",
            (batch_id, item_key),
        ).fetchone()
        return self._from_row(row, IngestionItem)

    def list_ingestion_items(self, batch_id: str) -> list[IngestionItem]:
        rows = self._conn.execute(
            "SELECT data FROM ingestion_items WHERE batch_id = ? ORDER BY item_key", (batch_id,)
        ).fetchall()
        return [self._from_row(r, IngestionItem) for r in rows]  # type: ignore[misc]

    def update_ingestion_item(self, item: IngestionItem) -> None:
        self._execute(
            "UPDATE ingestion_items SET data = ?, status = ?, stage = ?, content_hash = ?, document_id = ? "
            "WHERE id = ?",
            (
                self._to_row(item), item.status.value, item.stage.value,
                item.content_hash, item.document_id, item.id,
            ),
        )

    def count_items_by_status(self, kb_id: str) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM ingestion_items WHERE kb_id = ? GROUP BY status",
            (kb_id,),
        ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    def count_batch_items_by_status(self, batch_id: str) -> dict[str, int]:
        """Per-status counts for ONE batch, aggregated in SQL.

        Batch progress is recomputed after every file during a 200-file upload.
        Loading and re-validating every item each time would be O(n^2) Pydantic
        work, so the counts come straight from SQLite instead.
        """
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM ingestion_items WHERE batch_id = ? GROUP BY status",
            (batch_id,),
        ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    # -- corpus versions (V5 Phase 5) ----------------------------------------

    def create_corpus_version(self, version: CorpusVersion) -> CorpusVersion:
        """Insert a corpus version, returning the existing one for a known fingerprint.

        Two snapshots with identical metadata must share one fingerprint; this
        keeps `versions/` append-only and makes determinism observable.
        """
        existing = self._conn.execute(
            "SELECT data FROM corpus_versions WHERE kb_id = ? AND fingerprint = ?",
            (version.kb_id, version.fingerprint),
        ).fetchone()
        if existing is not None:
            return self._from_row(existing, CorpusVersion)  # type: ignore[return-value]
        self._execute(
            "INSERT INTO corpus_versions (id, kb_id, version, fingerprint, created_at, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                version.id, version.kb_id, version.version, version.fingerprint,
                version.created_at.isoformat(), self._to_row(version),
            ),
        )
        return version

    def get_corpus_version(self, kb_id: str, version_id: str) -> CorpusVersion | None:
        row = self._conn.execute(
            "SELECT data FROM corpus_versions WHERE kb_id = ? AND id = ?", (kb_id, version_id)
        ).fetchone()
        return self._from_row(row, CorpusVersion)

    def list_corpus_versions(self, kb_id: str, limit: int = 100) -> list[CorpusVersion]:
        rows = self._conn.execute(
            "SELECT data FROM corpus_versions WHERE kb_id = ? ORDER BY created_at DESC LIMIT ?",
            (kb_id, limit),
        ).fetchall()
        return [self._from_row(r, CorpusVersion) for r in rows]  # type: ignore[misc]

    def latest_corpus_version(self, kb_id: str) -> CorpusVersion | None:
        versions = self.list_corpus_versions(kb_id, limit=1)
        return versions[0] if versions else None
