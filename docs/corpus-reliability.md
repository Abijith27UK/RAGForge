# Corpus Reliability (RAGForge V5)

RAGForge is not a chatbot with a document upload box. It is a **corpus-engineering
platform**: the product is the knowledge base, and the job is to make that corpus
trustworthy, inspectable, reproducible and repairable.

This document describes the machinery behind that claim: the ingestion batch model,
the corpus manifest, the integrity engine, the repair system, corpus fingerprints
and versions, failure recovery, performance instrumentation, and the end-to-end
Naval Architecture workflow.

---

## 0. The two questions this system keeps separate

RAGForge deliberately refuses to answer one question by way of another.

**CORPUS COVERAGE** answers:

> Does the corpus contain the information?

**RETRIEVAL QUALITY** answers:

> Can the retrieval system actually find that information?

A corpus can be perfectly complete and still retrieve poorly. A corpus can be
small and retrieve beautifully. The manifest, the corpus map and the health strip
answer coverage. Only a **frozen, human-reviewed benchmark** answers retrieval
quality. The UI never substitutes one for the other, and the integrity engine
never reports a quality number.

### A domain does not automatically have ground truth

This is the single most important honesty rule in the project.

Entering "Naval Architecture", analysing the domain, indexing 200 documents and
reaching `READY` produces **no ground truth**. Ground truth exists only once humans
have authored questions, reviewed them, approved them, and frozen them into a
benchmark version.

Until then `ground_truth_status` is **`NOT_AVAILABLE`**, and the API refuses to
report Recall@K, Precision@K, MRR or NDCG for that knowledge base. This is exposed
on `GET /api/knowledge-bases/{kb_id}/overview` as an explicit field with an
explanation, not as a missing value:

```json
{
  "ground_truth_status": "NOT_AVAILABLE",
  "ground_truth_question_count": 0,
  "ground_truth_explanation": "This knowledge base has no human-reviewed ground truth, so no retrieval-quality metric ... A domain name never produces ground truth — questions must be authored, reviewed and frozen."
}
```

The Automobile Engineering benchmark is a hand-built research artifact. Naval
Architecture can eventually have `naval-architecture-v1` the same way — by a human
doing the work. See `docs/user-knowledge-workflow.md` §1.1.

---

## 1. Ingestion batches (Phase 1)

### Model

A batch is a persistent record created **before any processing begins**. Each file
becomes an `IngestionItem` row, and every state transition is committed immediately.

```
IngestionBatch   1 ──── * IngestionItem   1 ──── 0..1 Document
```

Per-item lifecycle (`IngestionStage`):

```
QUEUED → VALIDATING → PARSING → CHUNKING → EMBEDDING → INDEXING → COMPLETE
                          │                                    │
                          └────────────── FAILED ◄──────────────┘
                                         CANCELLED
```

Persisted per item: `batch_id`, `document_id`, `file_name`, `content_hash`,
`size_bytes`, `mime_type`, `status`, `stage`, `progress`, `error_code`,
`error_message`, `attempts`, `parser_used`, `chunk_count`, `vector_count`,
`warnings`, `started_at`, `completed_at`, `duplicate_of`.

### Design constraints and how they are met

| Requirement | Mechanism |
|---|---|
| Multi-file upload in one operation | `POST /{kb_id}/ingestion-batches`, up to `MAX_UPLOAD_FILES_PER_REQUEST` (default 200) |
| Persistent record before processing | batch + item rows written before the first parse |
| Stable document identity | `item_key = index + name + size + sha256[:12]`, `UNIQUE(batch_id, item_key)` |
| One failure must not fail the batch | `process_item` never raises for a per-file problem; the item records the failure |
| Refresh must not lose state | batch state lives in SQLite, not in request memory |
| Detect incomplete documents after a crash | `IngestionItem.is_incomplete` — any non-terminal stage |
| Resume must not duplicate vectors | completed items are skipped; a resumed item reuses its existing `Document` |
| Resume must not duplicate documents | schema-level `UNIQUE(batch_id, item_key)` plus document reuse |
| Skip already-completed unless forced | `resume(include_completed=False)` by default |
| No Celery/Redis | plain SQLite + the existing shared indexer; deterministic |

### Batch outcome vocabulary

`complete` · `partial` · `incomplete` · `failed`

A batch where some files failed is **`partial`**, never `complete`. A batch still
running is `incomplete`. This distinction matters: reporting a partially-failed
corpus as a complete ingestion is the failure mode this whole phase exists to
prevent.

### Endpoints

```
POST   /api/knowledge-bases/{kb_id}/ingestion-batches
GET    /api/knowledge-bases/{kb_id}/ingestion-batches
GET    /api/knowledge-bases/{kb_id}/ingestion-batches/{batch_id}
POST   /api/knowledge-bases/{kb_id}/ingestion-batches/{batch_id}/resume
```

---

## 2. Corpus manifest (Phase 2)

`GET /{kb_id}/corpus-manifest` answers *"what exactly is inside this knowledge base?"*

Per document: `document_id`, original and normalized filename, `content_hash`,
`document_version`, `source_mode`, `source_url`, `file_type`, `file_size`,
`parser`, `parser_version`, `page_count`, `slide_count`, `section_count`,
`text_length`, `chunk_count`, `vector_count`, `embedding_provider`,
`embedding_model`, `embedding_dimension`, `chunking_strategy`,
`chunking_config_hash`, `indexed_at`, `status`, `error_message`, `provenance`.

Corpus summary: totals for documents (completed / failed / processing), pages,
slides, sections, chunks, vectors, corpus size, duplicates, stale documents,
orphan vectors, embedding identity, chunking identity, vector backend, last
successful indexing time.

### The `-1` convention

Every count that could not be *measured* is `-1` and renders as `"unknown"`.

- Vector store unreachable → `total_vectors: -1`, `vector_count_confirmed: false`.
- Orphan count not yet computed → `orphan_vectors: -1`.

An unreachable store is never reported as "0 vectors". A check that could not run
is not a check that passed.

---

## 3. Integrity engine (Phase 3)

`CorpusIntegrityService` (`app/services/corpus/integrity.py`) is **strictly
read-only**. It has no repair path, no delete path, and no write path at all. This
is enforced by test (`test_integrity_scan_is_read_only`) and by design.

| Code | Check | Detects |
|---|---|---|
| A | `missing_vectors` | chunk metadata exists, vector does not — the chunk is not retrievable |
| B | `orphan_vectors` | vector exists, chunk metadata does not — retrievable but untraceable |
| C | `stale_vectors` | vector belongs to a superseded or deleted document |
| D | `embedding_mismatches` | vector embedded with a different model than the KB uses now |
| E | `chunking_mismatches` | chunks produced by more than one chunking configuration |
| F | `duplicate_documents` | several documents share a content hash |
| G | `duplicate_chunks` | several chunks share a content hash |
| H | `failed_documents` | documents that never completed ingestion |
| I | `partial_documents` | parsed but not indexed, and not superseded |
| J | `provenance_gaps` | chunks missing `source_id` / `document_id` / `content_hash` |
| K | `broken_source_references` | a document points at a source row that no longer exists |

Response shape:

```json
{
  "overall_status": "HEALTHY | WARNING | ERROR",
  "documents_checked": 153,
  "chunks_checked": 48392,
  "vectors_checked": 48392,
  "missing_vectors": [...], "orphan_vectors": [...], "stale_vectors": [...],
  "embedding_mismatches": [...], "chunking_mismatches": [...],
  "duplicate_documents": [...], "duplicate_chunks": [...],
  "failed_documents": [...], "partial_documents": [...],
  "provenance_gaps": [...], "broken_source_references": [...],
  "unknown_counts": []
}
```

`overall_status` is `ERROR` when a check that breaks retrieval or provenance found
something (A, B, C, D, H, K), `WARNING` for advisory findings or when some checks
could not run, and `HEALTHY` only when every check ran and found nothing.

Every finding names the document, chunk or vector it concerns. Nothing is repaired.

---

## 4. Safe repair (Phase 4)

Integrity checking and repair are **separate operations**. Repair requires an
explicit API call.

```
POST /{kb_id}/corpus-repair/plan   →  what WOULD change
POST /{kb_id}/corpus-repair        →  perform it
```

Actions: `reindex_document`, `reindex_failed_documents`, `reindex_batch`,
`remove_orphan_vectors`, `rebuild_document_vectors`, `rebuild_kb`,
`recompute_embeddings`.

Destructive actions (`remove_orphan_vectors`, `rebuild_kb`, `recompute_embeddings`)
must be confirmed by echoing the action:

```json
{ "action": "rebuild_kb" }                       // 400: confirm_action required
{ "action": "rebuild_kb", "confirm_action": "rebuild_kb" }   // runs
```

The plan states the blast radius — affected documents, expected chunks, expected
vectors — and sets `counts_confirmed: false` when those counts could not be
measured. Repairs reuse the vector store's measured point-count delta, so a repair
never reports "0 removed" when it removed hundreds of points.

---

## 5. Corpus fingerprint and versions (Phase 5)

A **fingerprint** answers *"which exact corpus produced these vectors?"* without
building a Git-like system. It is a SHA-256 over canonicalized manifest metadata.

Deliberately **excluded** from the fingerprint: document IDs (random per ingest),
file names (cosmetic), ingestion timestamps, chunk/vector counts (derived from the
chunking identity already included), and the `kb_id` itself. Including any of these
would make the fingerprint change on every run and destroy its value as a
determinism check.

Consequences:

- the same documents + versions + chunking + embedding identity → the same
  fingerprint, regardless of insertion order;
- any change to a content hash, a document version, the chunking config or the
  embedding identity → a different fingerprint;
- two different knowledge bases holding identical corpora agree, because the
  fingerprint identifies the *corpus*, not where it lives.

`POST /{kb_id}/corpus-versions` records a version. Snapshotting an **unchanged**
corpus returns the existing version instead of growing the history — determinism
is observable, not asserted.

---

## 6. Document diff (Phase 6)

`diff_manifests(old, new)` classifies every document as **added**, **removed**,
**changed** or **unchanged**. For a changed document it reports the old and new
content hashes, old and new chunk counts, and the vectors invalidated and created.

This integrates with incremental indexing: replacing a PDF changes its content hash,
supersedes the previous document version, removes only the affected vectors, indexes
the new version, and shows up as a `changed` entry.

---

## 7. Incremental indexing

Adding one document must never rebuild the knowledge base. `index_documents()`
(`app/services/indexing/document_indexer.py`) is the single chunk → embed → index
path used by the build route, the batch service and every repair action, so they
cannot drift apart.

Stale-vector contract:

1. Delete the documents' vectors **before** upserting. Re-chunking always generates
   new chunk IDs, so upserting alone would orphan points forever.
2. Write the chunk rows **after** a successful upsert, so SQLite always holds
   exactly the chunks present in the vector store.
3. `full_build=True` additionally sweeps orphans. Incremental runs deliberately do
   not sweep: they must not scan the whole collection and must never touch vectors
   belonging to documents they did not process.

A backend that cannot delete vectors is **refused** rather than allowed to corrupt
the corpus.

### Deletion-count honesty

The installed `qdrant-client` returns no deleted-count for `delete()`, so reading a
count off the response always yields zero. RAGForge therefore measures the
collection's `points_count` delta itself. When the count cannot be confirmed it
returns `-1`, surfaced as `"unknown"` and `vectors_removal_confirmed: false` — never
as `0`.

---

## 8. Failure recovery

| Failure | Detection | Recovery |
|---|---|---|
| Process dies mid-batch | `IngestionItem.is_incomplete` | `POST .../resume` |
| File fails validation | item `FAILED`, `error_code: INVALID_FILE` | re-upload (bytes were never parsed) |
| Parser fails | item `FAILED`, `error_code: PARSE_FAILED`, document row `FAILED` **with `raw_file_path` retained** | `reindex_failed_documents` re-parses from disk |
| Indexing fails | item `FAILED`, `error_code: INDEX_FAILED` | resume |
| Qdrant down mid-index | `IndexingError`; nothing half-written | re-index when Qdrant returns |
| Orphan vectors | integrity check B | `remove_orphan_vectors` |
| Embedding model changed | integrity check D | `recompute_embeddings` |

A failed parse keeps its original bytes precisely so the user never has to re-upload
a 200-file corpus to fix one bad file.

---

## 9. Performance instrumentation (Phase 7)

`backend/scripts/corpus_benchmark.py` measures the real pipeline at 10 / 50 / 100 /
200 documents using **deterministic synthetic fixtures**, and writes a
machine-readable artifact to `backend/data/benchmarks/corpus-scale.json` — inside
`data/`, deliberately **not** the repository's frozen `benchmarks/` directory.

```
cd backend && .venv/Scripts/python.exe scripts/corpus_benchmark.py
```

It reports documents/minute, chunks/second, vectors/second and MB/second, from
`time.perf_counter()` measurements — never estimates.

**What it does not measure:** real transformer inference speed, real Qdrant network
throughput, or retrieval quality of any kind. It uses the hashing embedding provider
and the in-memory store, so it measures RAGForge's own orchestration. Optimisation
before measurement is explicitly discouraged.

---

## 10. The Naval Architecture workflow (Phase 10)

```
1. Create KB: "Naval Architecture — IITM", source mode USER_PROVIDED
2. Corpus Command Center → drop the corpus (50–200 files)
   PDF · PPTX · DOCX · TXT · MD · HTML
3. Validate → parse → provenance → chunk → embed → index
4. Read the batch report: 147 indexed · 3 failed · 4 duplicates
5. Integrity scan: no orphan vectors, no stale vectors, embedding consistent
6. Corpus map: pages / slides / sections / chunks by document type
7. Snapshot the corpus version, record the fingerprint
8. Retrieval Lab: query, trace every hit to document → page/slide → section
9. Evaluation: ground truth is NOT_AVAILABLE until questions are authored
```

The system knows only what the indexed corpus contains. It does **not** "know
Naval Architecture" because the domain name says so.

---

## 11. UI

`/knowledge-bases/{id}/corpus` — the **Corpus Command Center**. Progressive
disclosure: the health strip answers "is this corpus trustworthy?" in one glance;
the corpus map and the individual integrity findings sit behind toggles.

It follows the same honesty rules as the backend: unknown counts render as
`"unknown"`, repair is plan-then-confirm, and an integrity scan is visibly
read-only.

---

## 12. What is deliberately not built

Not in this phase, by design: TurboVec, BM25, hybrid retrieval, reranking,
autonomous optimisation, MCP, LLM answer generation, OCR, CSV/XLSX, auth, cloud
deployment. The priority is making USER_PROVIDED corpus ingestion and retrieval
reliable, not adding retrieval features that cannot yet be measured.