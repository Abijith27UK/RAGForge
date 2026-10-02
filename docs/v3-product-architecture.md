# RAGForge — V3 Product Architecture (Phase A–D foundation)

**Status:** implemented and tested. Superseded in product scope by **V4 — User Knowledge
Ingestion** (see `docs/user-knowledge-workflow.md`); the V3 foundation below is unchanged and
still verified. 177 backend tests pass, `npm run typecheck` and `npm run build` are clean.

This document records what V3 added to the product foundation, what it deliberately does *not*
implement, and where the extension points live.

> **V4 status note.** The 28 Automobile Engineering benchmark questions are now HUMAN-REVIEWED
> and the benchmark is APPROVED/FROZEN. V4 does not recreate, modify or invalidate it: frozen
> questions, frozen `BenchmarkVersion` snapshots and every historical experiment artifact are
> untouched, and `verify_benchmark_integrity.py` verifies that read-only before any official
> experiment is allowed to run.
>
> **This does not generalise.** The Automobile benchmark is a domain-specific research artifact
> produced by hand. A new knowledge base ships with **no** benchmark, so its retrieval quality
> is unmeasured until the same human authoring/review/freeze process is repeated for that
> domain. See §1.1 of `docs/user-knowledge-workflow.md`.

## What V3 adds on top of the verified baseline

The Automobile Engineering baseline (28 questions, 11 documents, 812 chunks, MiniLM-L6-v2, Qdrant)
and all frozen artifacts (baseline benchmark/results, source-selection v1+v2 experiments) are
**unchanged**. V3 builds the product foundation around them:

| Phase | What was implemented | Extension point |
|---|---|---|
| **A — Benchmark lifecycle** | Question statuses DRAFT → REVIEW → APPROVED → FROZEN with reviewer attribution; edit-of-approved ⇒ new DRAFT revision (`supersedes`); FROZEN immutable. Benchmark versions = immutable snapshots; only FROZEN versions usable for official evaluation (enforced in `Evaluator`). | Human review of the existing 28 agent-authored questions (explicitly pending; nothing auto-migrated). |
| **B — v3 experiment framework** | `benchmarks/source-selection-experiment-v3.json` design + `scripts/run_source_selection_experiment_v3.py`: multi-seed (5 official seeds), gated on a frozen benchmark version, pairwise-intersection analysis, mergeable results file. | Runner NOT executed — blocked by design until the benchmark is human-frozen. |
| **C — Chunking architecture** | `ChunkingStrategyRegistry` (+ `get_chunker` compat accessor); every chunk stamps `chunking_strategy` + `chunking_config_version`; KB records the strategy/config used. Reproducibility test: same doc + config ⇒ identical `content_hash` sequence. | `DomainAwareChunker` registers exactly like `FixedSizeChunker` — no call-site changes. |
| **D — Vector abstraction** | `create_vector_store()` factory + `register_vector_backend()`; KB carries `vector_backend`; all three call sites go through the factory. | `TurboVecVectorStore` (Phase E) registers via one `register_vector_backend("turbovec", ...)` call. |

## New API surface

```
POST /api/knowledge-bases/{kb}/evaluation-questions/{qid}/status   # lifecycle transition (+reviewer)
PATCH /api/knowledge-bases/{kb}/evaluation-questions/{qid}          # edit; approved ⇒ new draft revision
POST /api/knowledge-bases/{kb}/benchmark-versions                   # immutable snapshot
GET  /api/knowledge-bases/{kb}/benchmark-versions[/{bv_id}]
POST /api/knowledge-bases/{kb}/benchmark-versions/{bv_id}/freeze    # refuses DRAFT/REVIEW content
DELETE /api/knowledge-bases/{kb}/benchmark-versions/{bv_id}         # refuses FROZEN
POST /api/knowledge-bases/{kb}/evaluate                             # + optional benchmark_version
GET  /api/experiments                                               # read-only artifact index
GET  /api/experiments/{id}/results                                  # verbatim frozen JSON
```

Schema additions (backward compatible — JSON-blob storage deserializes with defaults):
`EvaluationQuestion.{status, author, reviewer, created_at, reviewed_at, revision, supersedes, provenance}`,
`BenchmarkVersion` (snapshot + freeze metadata), `KnowledgeBase.{vector_backend, chunking_strategy, chunking_config}`,
`EvaluationRun.{benchmark_version, question_statuses}`.

## UI (V3-relevant parts of the redesign)

- **Evaluation page**: lifecycle badges and per-question review actions (approve/freeze with reviewer
  attribution, revision-aware editing), benchmark-version panel (snapshot / freeze / delete with
  freeze protection errors surfaced), and run configuration against a FROZEN version — the run then
  scores the immutable snapshot, never live rows.
- **Experiments page** (`/experiments`): renders the frozen v1/v2 artifacts verbatim from
  `/api/experiments/*` — coverage bars, strict per-k metrics, paired-intersection analysis, and the
  "aggregate rows are computed on different question subsets" disclaimer. No metric is hardcoded in
  the client.

## Deliberately NOT implemented (per plan)

TurboVec (Phase E), BM25 / hybrid / reranking (F), autonomous optimization (G), MCP (H),
domain-aware chunking algorithm (C scaffold only). The interfaces above are the only preparation.

## V4 additions that touch this foundation

| V4 concern | Relationship to V3 |
|---|---|
| **User knowledge ingestion** | Additive. External discovery, the `SourceQualityScorer` and all benchmark machinery are unchanged. User files go through a *separate* integrity scorer. |
| **Benchmark stays optional** | Reinforces Phase A's separation: a benchmark version is an evaluation instrument, never a prerequisite for a KB to reach `READY`. |
| **Incremental indexing** | Refactors the build route to call the shared `index_documents()` helper. The stale-vector contract documented in V3 is preserved verbatim and is now exercised by both the full build and the per-document path. |
| **Retriever registry** | Formalises the V3 note that runs should be "self-describing": `Retriever` ABC + registry, with dense as the baseline and no new strategies implemented. |
| **Source-selection v3 runner** | Unblocked by the human review, **but still not executed**. `verify_benchmark_integrity.py --prepare-v3` prints the plan and the exact command; the run is deliberately a manual, conscious step because it builds 18 corpora. |

## Reproducibility contract

1. Frozen benchmark version id recorded on every official `EvaluationRun`.
2. Experiment results files are append-merge, never rewritten.
3. Random selection uses `random.Random(seed).sample` — verified deterministic across processes.
4. Chunk sets are hash-reproducible for identical document + strategy + config.
