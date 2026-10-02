# RAGForge — Architecture (as implemented)

This document describes what actually exists in the code, not aspirations.

## High-level flow

```
Knowledge Base (SQLite)
  ├─ DomainSpec          (LLM provider, validated Pydantic output)
  ├─ Sources             (discovery providers → quality engine → decisions)
  ├─ Documents           (ingestion: download → parse → clean → hash)
  ├─ Chunks              (chunking strategies; full provenance)
  ├─ EvaluationQuestions → EvaluationRuns
  └─ BuildRuns           (stage statuses for transparency)

Qdrant: one collection per KB (kb_<id>), points carry full provenance payloads.
Filesystem: backend/data/documents/<source_id>.<ext> raw + .parsed.txt
```

## Backend layout

```
backend/app/
  main.py                     FastAPI app, CORS, global error handler
  config.py                   Pydantic Settings from env / backend/.env
  schemas/models.py           ALL domain models (single source of truth)
  api/                        routers: KBs, domain, sources, build, retrieval/eval, system
  llm/provider.py             LLMProvider ABC + OpenAI-compatible + Mock provider
  repositories/sqlite_repo.py SQLite persistence (JSON-serialized models)
  services/
    domain_analyzer/          prompt → structured DomainSpec
    source_discovery/         SourceDiscoveryProvider ABC; user-url, arxiv
    source_quality/           explainable heuristic scorer (signals→score→decision)
    ingestion/                downloaders + per-format parsers + hashing
    chunking/                 Chunker ABC; section-aware + fixed-size
    embeddings/               EmbeddingProvider ABC; ST, OpenAI, hashing fallback
    vector_store/             VectorStore ABC; Qdrant implementation
    retrieval/                DenseRetriever (dense-only, honestly named)
    evaluation/               pure metric functions + Evaluator orchestrator
  utils/                      ids, hashing/cleaning, URL validation (SSRF), logging
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

## Failure handling

- Ingestion errors are per-document (one bad URL doesn't abort the batch) and surfaced in
  the build run status.
- Qdrant unavailability returns HTTP 503 with an actionable message ("start Qdrant, see README").
- LLM misconfiguration returns actionable errors; with `ALLOW_MOCK_LLM=true` (default) an
  unconfigured LLM degrades to the labelled mock instead of failing.
- Unhandled exceptions are logged server-side; the frontend receives a generic 500 (no stack traces).

## API surface (implemented)

```
POST/GET/DELETE /api/knowledge-bases[/{id}]
GET  /api/knowledge-bases/{id}/build-status
POST /api/knowledge-bases/{id}/analyze-domain
GET  /api/knowledge-bases/{id}/domain-spec
POST /api/knowledge-bases/{id}/discover-sources      {provider, query, limit}
GET  /api/knowledge-bases/{id}/sources
POST /api/knowledge-bases/{id}/sources/{sid}/decision {decision}
POST /api/knowledge-bases/{id}/ingest                {source_ids?}
POST /api/knowledge-bases/{id}/index                 {chunker, target_size, overlap}
GET  /api/knowledge-bases/{id}/documents | chunks
POST /api/knowledge-bases/{id}/retrieve              {query, top_k, filters}
POST/GET/DELETE /api/knowledge-bases/{id}/evaluation-questions[/{qid}]
POST /api/knowledge-bases/{id}/evaluate              {top_k, question_ids?}
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
- **Read-only experiment artifacts** — `routes_experiments.py` serves frozen results JSON verbatim;
  the UI renders them without hardcoding metrics.

## Extension points for experiments

- `Chunker` — add `SemanticChunker`; compare against `section-aware` / `fixed-size`.
- `SourceDiscoveryProvider` — add more connectors (pubmed, Semantic Scholar, ...).
- `SourceQualityScorer` — new signals/weights or an LLM-based assessor.
- `EmbeddingProvider` — any model; dimensionality is checked against the collection.
- `VectorStore` — second backend for comparison runs.
- Retrieval — add BM25/hybrid/rerank as new retriever classes; the response already carries
  `retrieval_backend` so runs are self-describing.
