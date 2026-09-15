# Experiment: Automobile Engineering Retrieval Baseline v1

**Status:** baseline established 2026-09-15 · **Benchmark:** `benchmarks/automobile-engineering-baseline-v1.json` · **Results:** `benchmarks/automobile-engineering-baseline-v1-results.json` · **KB:** `kb_f278c283c748`

This document pins the first rigorous, reproducible retrieval baseline for RAGForge so that
future changes to chunking, source selection, or retrieval can be compared against it.
Nothing in this experiment is fabricated: every metric below was produced by the live
retrieval pipeline (real Qdrant search, real embeddings) against explicitly authored
chunk-level ground truth.

## 1. Dataset / corpus

| Property | Value |
|---|---|
| Knowledge base | `kb_f278c283c748` — "Automobile Engineering KB (E2E)" |
| Documents | 11 (all English Wikipedia articles, ingested 2026-09-15) |
| Chunks | 812 (section-aware, target 1200 chars, overlap 150) |
| Vectors in Qdrant | 812 (verified `points_count` = SQLite chunk count after re-index) |

Documents (subdomain coverage in parentheses):

| # | Document | URL | Subdomain |
|---|---|---|---|
| 1 | Engine - Wikipedia | https://en.wikipedia.org/wiki/Engine | engine |
| 2 | Transmission (mechanical device) - Wikipedia | https://en.wikipedia.org/wiki/Transmission_(mechanical_device) | transmission |
| 3 | Brake - Wikipedia | https://en.wikipedia.org/wiki/Brake | braking |
| 4 | Steering - Wikipedia | https://en.wikipedia.org/wiki/Steering | steering |
| 5 | Car suspension - Wikipedia | https://en.wikipedia.org/wiki/Suspension_(vehicle) | suspension |
| 6 | Vehicle frame - Wikipedia | https://en.wikipedia.org/wiki/Vehicle_frame | chassis |
| 7 | Vehicle dynamics - Wikipedia | https://en.wikipedia.org/wiki/Vehicle_dynamics | vehicle dynamics |
| 8 | Electric vehicle - Wikipedia | https://en.wikipedia.org/wiki/Electric_vehicle | electric vehicles |
| 9 | Battery management system - Wikipedia | https://en.wikipedia.org/wiki/Battery_management_system | electric vehicles / BMS |
| 10 | Regenerative braking - Wikipedia | https://en.wikipedia.org/wiki/Regenerative_braking | braking / EV |
| 11 | Car - Wikipedia | https://en.wikipedia.org/wiki/Automobile | general automobile |

Corpus construction: user-URL discovery → explainable source-quality scoring (real
accessibility probes, all HTTP 200) → manual ACCEPT → ingestion (pypdf/BS4 parse, content-hash
dedup: 4 duplicates skipped) → section-aware chunking → embedding → Qdrant upsert.

## 2. Embedding model

| Property | Value |
|---|---|
| Provider | sentence-transformers (local) |
| Model | sentence-transformers/all-MiniLM-L6-v2 |
| Dimensions | 384 |
| Normalization | L2 (cosine distance) |
| Identity guard | `KB.embedding_identity` recorded at index time; retrieval refuses a mismatched model |

## 3. Vector database

| Property | Value |
|---|---|
| Engine | Qdrant 1.19.1 (Windows build, `qdrant-dl/qdrant.exe`) |
| Collection | `kb_kb_f278c283c748` |
| Distance | Cosine |
| Point ID mapping | SHA-1(chunk_id)[:15] → unsigned int (deterministic) |
| Payload | full provenance (source_url, title, type, publisher, section, section_path, page, domain, subdomain, trust_score, content_hash, char_count) |

## 4. Retrieval configuration

| Property | Value |
|---|---|
| Backend | `qdrant-dense` (dense vector retrieval only — no BM25/hybrid/rerank in this baseline) |
| Query embedding | same model as indexing (enforced) |
| Filters | none for the benchmark runs |
| top_k | 3, 5, 10 (three strict runs) |

## 5. Benchmark questions

28 questions (`benchmarks/automobile-engineering-baseline-v1.json`), covering the requested
subdomains:

| Subdomain | Questions |
|---|---|
| engine | 4 (auto-eng-001..004) |
| transmission | 4 (auto-eng-005..008) |
| braking | 4 (auto-eng-009..012) |
| steering | 3 (auto-eng-013..015) |
| suspension | 3 (auto-eng-016..018) |
| chassis | 3 (auto-eng-019..021) |
| vehicle dynamics | 3 (auto-eng-022..024) |
| electric vehicles | 3 (auto-eng-025..027) |
| general automobile | 1 (auto-eng-028) |

## 6. Ground-truth methodology

- **Basis:** explicit `expected_chunk_ids` (+ `expected_document_ids`) only. No keyword
  generation, no LLM generation. 27 of 28 questions are single-relevant-chunk; auto-eng-016
  has 2 relevant chunks.
- **Authoring procedure:** each chunk ID was selected by reading the actual indexed chunk text
  (dumped from the live KB) and quoting the supporting sentence in the `provenance` field of
  the benchmark JSON. The `notes` field of each stored EvaluationQuestion carries the same
  provenance plus the benchmark question ID.
- **Integrity:** the benchmark loader **validates every ID against the live KB before running**
  and refuses to run if any ground truth is stale (e.g. after re-chunking, which regenerates
  chunk IDs).
- **Authorship status:** selections were made by the coding agent under explicit human
  instruction ("agent-authored, human review pending"). The UI supports human review (each
  question can be inspected against its quoted passage and re-authored). This must be treated
  as a *machine-assisted* baseline until a human reviewer signs off.
- **Relevance model:** binary. Relevance grades are not supported in this version, so NDCG is
  binary-relevance NDCG; graded relevance (e.g. graded NDCG) is future work.

## 7. Evaluation metrics

Computed by `app/services/evaluation/metrics.py` from real retrieval results (strict mode —
`allow_keyword_fallback=False`; keyword heuristic questions are excluded and reported as
skipped, never mixed in):

| Metric | Definition |
|---|---|
| Recall@K | \|relevant ∩ top-K\| / \|relevant\| |
| Precision@K | \|relevant ∩ top-K\| / K |
| MRR | 1 / rank of first relevant hit |
| NDCG@K | binary-relevance NDCG |

Document-basis metrics (Recall/Precision/MRR/NDCG over `expected_document_ids`) are also
recorded per run in the results JSON.

## 8. Baseline results (strict, explicit chunk ground truth, 28 questions)

| Run | Recall@K | Precision@K | MRR | NDCG@K |
|---|---|---|---|---|
| strict k=3 | 0.821 | 0.286 | 0.708 | 0.738 |
| strict k=5 | 0.964 | 0.200 | 0.737 | 0.793 |
| strict k=10 | 1.000 | 0.104 | 0.743 | 0.806 |

(All 28 questions evaluated in every run; 0 skipped, 0 keyword-scored. Full per-question
results with benchmark IDs are in `benchmarks/automobile-engineering-baseline-v1-results.json`.)

Honest observations:
- Recall@10 = 1.0 means the correct chunk appears in the top 10 for every question; but with
  MRR ≈ 0.74, the correct chunk is often not rank 1 — dense retrieval with MiniLM finds the
  right *neighbourhood* but not always the exact chunk.
- Precision@K values are structurally low because most questions have exactly 1 relevant chunk
  in an 812-chunk corpus (P@10 ≈ 0.10 is the ceiling for a perfect run under this design).
- These are single-corpus, single-embedding numbers; they are a baseline, not a claim of
  retrieval quality in general.

## 9. Reproducibility

To re-run this baseline exactly:

1. Start Qdrant (`qdrant-dl/qdrant.exe`) and the backend (`python -m uvicorn app.main:app --port 8000`).
2. Ingest + index the 11 sources listed above with chunker `section-aware`, target 1200,
   overlap 150 (this regenerates chunk IDs; the loader validates the benchmark against the
   fresh IDs, so the benchmark file must be re-authored against the new corpus if chunks
   change).
3. `cd backend && python scripts/run_baseline_benchmark.py --benchmark ../benchmarks/automobile-engineering-baseline-v1.json --kb <kb_id>`

The results JSON records run IDs, labels, embedding model, backend, per-question metrics, and
aggregate integrity counts (`questions_with_explicit_gt=28`, `questions_with_keyword_fallback=0`,
`strict_mode=true`), so future strategy comparisons (different chunker settings, other
embeddings) can be diffed mechanically.

## 10. Threats to validity / limitations

- Corpus is Wikipedia-only; source diversity (standards, papers, lecture notes) may change
  difficulty and scores.
- Ground truth is single-chunk-per-question by design (22 of 28 exactly one chunk); this
  inflates Recall@K relative to multi-chunk benchmarks and makes Precision@K structurally small.
- Wikipedia articles were parsed with navigation/reference boilerplate partially retained in
  chunk text (visible in chunk dumps); a cleaner extraction would likely raise MRR.
- Agent-authored ground truth (human review pending) — selection bias is possible despite the
  quoted-passage provenance rule.
- Binary relevance only; graded NDCG requires graded labels (future work).

## 11. Next comparisons (planned, not implemented here)

Per project scope, none of these exist yet: BM25/hybrid retrieval, reranking, semantic
chunking, automatic optimization. The infrastructure above (pinned benchmark + validated
ground truth + labelled strict runs) is what makes those future comparisons meaningful.
