# RAGForge — Architecture (as implemented)

This document describes what actually exists in the code, not aspirations.

## High-level flow

```
Knowledge Base (SQLite)  ── source_mode: EXTERNAL | USER_PROVIDED | MIXED
  ├─ DomainSpec          (LLM provider, validated Pydantic output)
  ├─ Sources             (discovery providers → quality engine → decisions)
  │                       user-provided files carry integrity, not authority, signals
  ├─ Documents           (document library: upload/download → parse → clean → hash → status)
  ├─ Chunks              (chunking strategies; full provenance incl. page/slide/section)
  ├─ EvaluationQuestions → BenchmarkVersions → EvaluationRuns   ← OPTIONAL, never required
  └─ BuildRuns           (stage statuses for transparency)

Qdrant: one collection per KB (kb_<id>), points carry full provenance payloads.
Filesystem: backend/data/documents/<source_id>.<ext>      (downloaded external sources)
            backend/data/uploads/<kb_id>/<raw_id>.<ext>  (user-provided originals, local only)
Both keep the normalized text in a <file>.parsed.txt sidecar.
```

A knowledge base becomes READY once it is indexed. Evaluation is a separate,
optional instrument — `KBOverview.evaluation_required` is always `false`.

## Backend layout

```
backend/app/
  main.py                     FastAPI app, CORS, global error handler
  config.py                   Pydantic Settings from env / backend/.env
  schemas/models.py           ALL domain models (single source of truth)
  api/                        routers: KBs, domain, sources, documents, build,
                              retrieval/eval, benchmark, experiments, system
  llm/provider.py             LLMProvider ABC + OpenAI-compatible + Mock provider
  repositories/sqlite_repo.py SQLite persistence (JSON-serialized models)
  services/
    domain_analyzer/          prompt → structured DomainSpec
    source_discovery/         SourceDiscoveryProvider ABC; user-url, arxiv
    source_quality/           explainable heuristic scorer (signals→score→decision)
                              + user_scorer.py: integrity model for user files
    ingestion/
      parsers.py              DocumentParser ABC + ParserRegistry
                              (Pdf, Pptx, LegacyPpt, Docx, Html, Markdown, Text)
      upload.py               name safety, size, magic bytes, OOXML container checks
      ingestion.py            download (external) / parse (local) orchestration
    indexing/
      document_indexer.py     chunk → embed → index for a SET of documents.
                              Shared by the full build and per-document incremental
                              indexing so stale-vector handling cannot drift.
    chunking/                 Chunker ABC; section-aware + fixed-size; page/slide markers
    embeddings/               EmbeddingProvider ABC; ST, OpenAI, hashing fallback
    vector_store/             VectorStore ABC; Qdrant implementation
    retrieval/                Retriever ABC + registry; DenseRetriever (dense-only)
    evaluation/               pure metric functions + Evaluator orchestrator
  utils/                      ids, hashing/cleaning, URL validation (SSRF), logging
scripts/
  verify_benchmark_integrity.py   read-only frozen-benchmark gate + v3 plan
```

## Key decisions

1. **SQLite + JSON columns** instead of PostgreSQL/ORM: simplest robust persistence for a
   single-user local MVP. Repository interface isolates services from storage; a swap is
   localized.
2. **Every external integration behind an ABC** so experiments can swap implementations
   (chunkers, embedding providers, vector stores) without touching services or API.
3. **Quality scores are heuristics, stored with their derivation**: each assessment persists
   the signals, weights, reasons, warnings, and assessor version. The UI shows them as an
   automated assessment and lets the user override ACCEPT/REVIEW/REJECT.
4. **Mock/dev outputs are labelled at the data level** (`is_mock`, `generated_by`,
   `is_fallback`), not just in the UI, so fake data cannot masquerade as real downstream.
5. **Evaluation metrics are pure functions** over ranked IDs and known-relevant sets.
   Missing ground truth → `null` (unknown), never 0 or a fabricated number.
6. **SSRF-safe ingestion**: user URLs must resolve to public addresses; scheme/host checks;
   size cap; timeouts. Downloaded files are parsed as data, never executed.
7. **Point IDs in Qdrant** are deterministic 63-bit hashes of chunk IDs (Qdrant requires
   unsigned ints/UUIDs); the original ID travels in the payload.
8. **Benchmark ≠ knowledge base (V4).** A KB is a product artifact; ground truth is an
   optional evaluation instrument. `source_mode` describes where knowledge comes from and
   never implies a benchmark exists. Nothing in the build path reads evaluation questions.
   Note the consequence: a *new* domain has **no** benchmark and therefore **no measured
   retrieval quality** until a human authors, reviews and freezes one. Benchmarks are
   per-domain hand-made research artifacts (only the Automobile Engineering one exists here);
   RAGForge never auto-generates ground truth, and metrics stay `null` until it does.
9. **User files are scored on integrity, not publication authority (V4).** A lecture PDF has
   no website, HTTP status or publication date. `UserProvidedIntegrityScorer` weights only
   file validity, content extraction, uploader-asserted relevance, duplication and structure —
   no authority/recency/accessibility/evidence weight exists in it.
10. **One shared indexer (V4).** `index_documents()` is the only chunk→embed→index path.
    Vectors for the documents being (re)indexed are deleted *before* upsert (re-chunking
    makes new IDs, so upserting alone orphans points forever); chunk rows are written *after*
    a successful upsert. A backend that cannot delete vectors is refused rather than corrupted.
11. **Structural numbers are read, never inferred.** `[PAGE n]` / `[SLIDE n]` markers come from
    the parsers; a deck with no title placeholder yields `slide_title = None`, and a scanned PDF
    fails with an explicit "OCR is not implemented" error.
12. **Deletion counts are measured, and "unknown" is a valid answer (V4).** The installed
    `qdrant-client` returns no deleted-count for `delete()`, so RAGForge measures the collection's
    `points_count` delta itself instead of trusting a response shape that silently yields zero.
    If the collection cannot be read it returns `-1`, which the indexer and the API surface as
    `"unknown"` / `vectors_removal_confirmed: false`. A confirmed count is never replaced by an
    unconfirmed one, and an unconfirmed count is never rendered as `0`.

## Failure handling

- Ingestion errors are per-document (one bad URL doesn't abort the batch) and surfaced in
  the build run status.
- Upload validation rejects per file with an actionable message; a corrupt container, a
  mislabelled extension or an empty file never becomes a half-ingested document.
- A document whose parser fails is stored with `status=FAILED` and its real error, so the
  library shows the failure instead of hiding it.
- Indexing refuses to proceed when the vector store cannot delete stale vectors for the
  documents being re-indexed.
- Qdrant unavailability returns HTTP 503 with an actionable message ("start Qdrant, see README").
- LLM misconfiguration returns actionable errors; with `ALLOW_MOCK_LLM=true` (default) an
  unconfigured LLM degrades to the labelled mock instead of failing.
- Unhandled exceptions are logged server-side; the frontend receives a generic 500 (no stack traces).

## API surface (implemented)

```
POST/GET/DELETE /api/knowledge-bases[/{id}]
GET  /api/knowledge-bases/{id}/overview              # one honest payload for the overview page
GET  /api/knowledge-bases/{id}/build-status
POST /api/knowledge-bases/{id}/analyze-domain
GET  /api/knowledge-bases/{id}/domain-spec
POST /api/knowledge-bases/{id}/discover-sources      {provider, query, limit}
POST /api/knowledge-bases/{id}/sources/user-urls     {urls, titles}   # user-provided URLs
GET  /api/knowledge-bases/{id}/sources
POST /api/knowledge-bases/{id}/sources/{sid}/decision {decision}
POST /api/knowledge-bases/{id}/ingest                {source_ids?}   # external sources only
POST /api/knowledge-bases/{id}/index                 {chunker, target_size, overlap}
GET  /api/knowledge-bases/{id}/documents | chunks

# --- V4 document library ---
POST   /api/knowledge-bases/{id}/documents/upload    # multipart files[] + index?
GET    /api/knowledge-bases/{id}/document-library
GET    /api/knowledge-bases/{id}/documents/{did}
GET    /api/knowledge-bases/{id}/documents/{did}/text
GET    /api/knowledge-bases/{id}/documents/{did}/chunks
POST   /api/knowledge-bases/{id}/documents/{did}/rebuild  {reindex?}
POST   /api/knowledge-bases/{id}/documents/{did}/index    # incremental: this document only
POST   /api/knowledge-bases/{id}/documents/{did}/replace  # new document_version + history
DELETE /api/knowledge-bases/{id}/documents/{did}          # removes chunks AND vectors

POST /api/knowledge-bases/{id}/retrieve              {query, top_k, filters}
POST/GET/DELETE /api/knowledge-bases/{id}/evaluation-questions[/{qid}]
POST /api/knowledge-bases/{id}/evaluate              {top_k, question_ids?, benchmark_version?}
GET  /api/knowledge-bases/{id}/evaluation-runs
GET  /api/system/health | status
```

## Extension points for experiments (V3)

- **Benchmark lifecycle** — questions move DRAFT → REVIEW → APPROVED → FROZEN with reviewer
  attribution (`routes_benchmark.py`); APPROVED edits spawn new DRAFT revisions (`supersedes`);
  FROZEN questions/versions are immutable. Benchmark versions are snapshot objects
  (`BenchmarkVersion`) whose FROZEN instances are the only valid basis for official evaluation runs
  (enforced in `Evaluator.run_evaluation` via `config.benchmark_version`).
- **Chunking strategy registry** — `CHUNKING_REGISTRY` / `get_chunker()`; chunks stamp
  `chunking_strategy` + `chunking_config_version`; KB records the strategy used.
- **Vector backend factory** — `create_vector_store(settings, kb.vector_backend)` with
  `register_vector_backend()` for experimental backends (TurboVec, Phase E).
- **Retriever registry (V4 Phase 12)** — `Retriever` ABC + `register_retriever()` /
  `get_retriever()` in `services/retrieval/retriever.py`. Dense is the baseline and the only
  implemented strategy; BM25 / hybrid / reranked register here and must return the same
  `RetrievalResponse`, so the API, evaluator and UI never change shape.
- **Parser registry (V4)** — `PARSER_REGISTRY` in `services/ingestion/parsers.py`. CSV/XLSX/OCR
  register exactly like `PdfParser`; no call site changes.
- **Read-only experiment artifacts** — `routes_experiments.py` serves frozen results JSON verbatim;
  the UI renders them without hardcoding metrics.
- **Frozen-benchmark gate** — `scripts/verify_benchmark_integrity.py` is read-only and refuses to
  bless a benchmark that is not FROZEN, fully reviewed, and snapshotted.

## Extension points for experiments

- `Chunker` — add `SemanticChunker`; compare against `section-aware` / `fixed-size`.
- `SourceDiscoveryProvider` — add more connectors (pubmed, Semantic Scholar, ...).
- `SourceQualityScorer` — new signals/weights or an LLM-based assessor.
- `EmbeddingProvider` — any model; dimensionality is checked against the collection.
- `VectorStore` — second backend for comparison runs.
- Retrieval — add BM25/hybrid/rerank as new retriever classes; the response already carries
  `retrieval_backend` so runs are self-describing.

---

## Corpus engineering layer (V5)

The corpus services live in `app/services/corpus/` and are deliberately separate
from the original domain pipeline:

| Module | Responsibility |
|---|---|
| `batches.py` | persistent, resumable bulk ingestion; per-item failure isolation |
| `manifest.py` | "what exactly is inside this KB?" — per document + corpus summary |
| `integrity.py` | **read-only** audit: missing / orphan / stale vectors, mismatches, duplicates, provenance gaps |
| `repair.py` | explicit, confirmed, separately-invoked repairs |
| `fingerprint.py` | deterministic corpus fingerprint, versions and document diffs |
| `perf.py` | measured pipeline instrumentation + the scale benchmark harness |

Key decisions:

13. **A batch is persisted before any processing begins.** Every file becomes an
    `ingestion_item` row with a `UNIQUE(batch_id, item_key)`. Resume is therefore
    idempotent at the *schema* level, not merely in application code, and a page
    refresh or a process restart cannot lose batch state. No Celery, no Redis.
14. **Integrity checking and repair are separate operations.** `CorpusIntegrityService`
    has no write path at all; repair lives in `CorpusRepairService` and requires an
    explicit call, plus `confirm_action` for destructive actions. Provenance of a
    finding is never conflated with fixing it.
15. **Unmeasured counts are `-1`, never `0`.** `-1` surfaces as `"unknown"` /
    `vectors_removal_confirmed: false`. An unreachable vector store reports unknown
    totals and lists the skipped checks in `unknown_counts`; it never reports zero
    findings for checks that did not run.
16. **The corpus fingerprint excludes anything non-deterministic.** Document IDs,
    file names, timestamps and the `kb_id` are all excluded, so identical corpora
    produce identical fingerprints and a version history only grows on real change.
17. **Ground truth is an explicit, per-KB state.** `GroundTruthStatus.NOT_AVAILABLE`
    is the honest default for every new domain. Corpus coverage and retrieval quality
    are reported by different surfaces and must never be substituted for one another.

### Why the fingerprint excludes `kb_id`

A fingerprint answers *"which exact corpus produced these vectors?"*. If it included
the knowledge-base ID, two KBs holding byte-identical corpora under identical
chunking/embedding configuration would disagree, and the fingerprint could never be
compared across environments or machines. The corpus is the thing being identified.
