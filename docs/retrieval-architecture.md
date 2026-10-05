# Retrieval architecture (V6)

Status of each element is labelled explicitly: **IMPLEMENTED**, **EXPERIMENTAL**,
**PLANNED**, **NOT AVAILABLE**. Nothing in this document describes a feature that
does not exist in the code.

## Scope

V1–V5 shipped one retrieval strategy: dense vector search through Qdrant. V6 turns
retrieval into a pluggable, inspectable layer:

```
query
  -> query processing (PLANNED, Phase 9 of the V6 plan)
  -> candidate retrieval      dense (Qdrant)  +  lexical (BM25 over SQLite chunks)
  -> normalization            min_max | rank, per candidate pool
  -> fusion                   weighted (0.65/0.35 default) | RRF
  -> diversification          document cap | MMR            (optional, off by default)
  -> reranking                cross-encoder                 (optional)
  -> final top_k              with min_score floor
```

Strategies selectable by name (`retrieval_strategy`):

| name | what it does | embeddings needed | corpus (repo) needed |
|---|---|---|---|
| `dense` (alias `qdrant-dense`) | dense vector search | yes | no |
| `bm25` | lexical BM25 over indexed chunks | **no** | yes |
| `hybrid` | dense + BM25, normalized then fused | yes | yes |
| `hybrid_reranked` | hybrid + cross-encoder rerank | yes | yes |

## The registry is the only selection mechanism

`app/services/retrieval/retriever.py` holds the `Retriever` ABC and a registry of
`RetrieverSpec`s. Call sites resolve a strategy by NAME; nothing branches on a
strategy inside business logic.

* `register_retriever(name, factory, ...)` — a plain 3-argument factory (the V4
  contract) is still accepted; `requires_repo`, `needs_embedding`, `description`,
  `is_baseline` and `aliases` are opt-in.
* `get_retriever(name, embedder, store, identity, repo=..., settings=...)` —
  extra dependencies are forwarded only to factories that declare them, so a
  legacy factory cannot break.
* `available_retrievers()` / `describe_retrievers()` — the UI reads the registry
  instead of hard-coding a list.
* Built-ins register LAZILY on first registry query, so importing the module never
  triggers a heavy import and there is no import cycle.

`Retriever.retrieve(kb_id, query, top_k, filters, min_score)` remains the required
abstract surface (evaluator, scripts and V5 callers use it unchanged).
`Retriever.retrieve_with_params(kb_id, query, RetrievalParams)` is the V6 entry
point; the default implementation maps params onto the classic five arguments, so
a pre-V6 strategy never receives a keyword it does not understand.

## BM25 (IMPLEMENTED)

`app/services/retrieval/lexical.py` implements real BM25, not keyword overlap:

```
score(q,d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
IDF(t)     = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
```

* defaults `k1 = 1.2`, `b = 0.75`; both configurable per request and recorded.
* term-frequency saturation and document-length normalization are real: a long or
  repetitive document cannot win by length or repetition (asserted by tests).
* tokenizer is Unicode-aware (`\w`-based), lowercased after NFKC normalization, and
  keeps technical identifiers whole: `gm-0.5`, `iso_9001`, `s/n`, `3.2.1`.
* query term multiplicity is ignored (classic BM25 has no query-TF component), so
  repeating a word never changes the ranking.
* deterministic: ties break by chunk id, never by dict/insertion order.
* no external dependency, no network, no model download.

### Where the index lives

```
chunks (SQLite)      durable source of truth for text + provenance
corpus_revision      integer bumped on EVERY chunk write
lexical_indexes      persisted BM25 statistics (postings / df / doc lengths)
```

The persisted index stores **no chunk text and no provenance**; the top candidates
are hydrated from SQLite in one batched query, which keeps the stored index small
and guarantees the provenance shape is identical to the dense path.

### Staleness is exact, not a guess

`corpus_revision` is bumped by `Repository.create_chunks`,
`delete_chunks_for_document` (which also covers document deletion) — the only two
ways chunks change. A persisted index records the revision it was built from, so:

* matching revision -> the index is reused (memory, else persisted across restarts);
* mismatching revision -> the index is **rebuilt before use** and the rebuild is
  reported on the response (`bm25_index` stage + a note), never hidden;
* a tokenizer-version change also forces a rebuild;
* if a store cannot report a corpus revision, the response states that staleness
  **could not be verified** — a capability gap is surfaced, not asserted away.

Diagnostics: `GET /api/knowledge-bases/{kb_id}/bm25-index` reports `source`,
`revision`, `current_revision`, `stale` (`true`/`false`/`null`) and
`staleness_check` (`revision` | `unavailable`).

## Hybrid fusion (IMPLEMENTED)

Dense cosine similarity and BM25 live on different, corpus-dependent scales.
Fusion therefore **normalizes first** (`min_max`, default; `rank` alternative) and
records the method with every run.

* weighted: `fused = w_dense·norm_dense + w_bm25·norm_bm25` (defaults 0.65/0.35).
* RRF: `fused = Σ_i w_i / (rrf_k + rank_i(d))`, ranks 1-based, `rrf_k` default 60.
* contributions per source are reported per candidate (`dense_contribution`,
  `lexical_contribution`), so "found by both" is distinguishable from
  "found by one".
* if one side returns an empty pool, the weights are **rescaled** to the surviving
  side and the response says so; keeping 0.65/0.35 would cap every fused score at
  0.65 and look like poor relevance.
* `min_score` applies to the fused score for hybrid, to cosine for dense, and to
  the within-pool **normalized** BM25 score for `bm25` (raw BM25 is unbounded, so a
  raw threshold would not be comparable across corpora — raw scores stay visible in
  `score_breakdown`).

## Reranking (IMPLEMENTED, optional, honest fallback)

`app/services/retrieval/rerank.py` defines the `Reranker` ABC, a `NoReranker`, and
`CrossEncoderReranker` (sentence-transformers cross-encoder; default model
`cross-encoder/ms-marco-MiniLM-L-6-v2`). It is loaded lazily and only when
requested.

* `status = not_requested | applied | unavailable_fallback`.
* When the model cannot be loaded (offline, dependency missing, bad name) the
  UNRERANKED order is returned with `status = unavailable_fallback`, the reason, and
  a `UNAVAILABLE` note on the response. Reranking is never silently faked.
* `candidate_k`, `reranked` and `final_k` are reported for every request.
* A reranker exception can never break retrieval.
* Cross-encoder logits are unbounded, so they are reported as `rerank_score` and
  drive the ORDER only; the displayed `score` stays the comparable fused score.
* The reranker is NOT part of the default path: `hybrid_reranked` requests it,
  plain `hybrid` does not.

Verified locally: the cross-encoder model loads and the reranked strategy applies
(the model is cached under `~/.cache/huggingface/hub`). Tests use a fake reranker or
a forced-unavailable reranker so the suite never depends on a download.

## Diversity (IMPLEMENTED, off by default)

Fixed with evidence loss in mind — relevance must not be sacrificed silently:

* `max_per_document` — document cap; candidates over the cap are **demoted**, not
  dropped, so a small corpus can still fill `top_k`.
* `diversity = "mmr"` — Maximal Marginal Relevance with `diversity_lambda`
  (default 0.7) over stored candidate vectors. Cosine similarity is mapped from
  [−1, 1] to [0, 1] before combining so both MMR terms share a scale.
* MMR needs stored vectors (`VectorStore.fetch_vectors`). When a backend cannot
  provide them, the `diversity` stage is recorded as **unavailable** and the
  relevance/document-cap order is used instead.

## Configuration persistence (IMPLEMENTED)

* `retrieval_configs` (SQLite, one row per KB) stores a `RetrievalParams`:
  strategy, top_k/candidate_k/min_score, weights, fusion, rrf_k, normalization,
  bm25 k1/b, reranker + model, diversity + lambda + cap, filters.
* Resolution order: request override > stored KB configuration > defaults.
  `None` in an override means "not specified" and never clobbers a stored value.
* `GET/PUT /api/knowledge-bases/{kb_id}/retrieval-config`. When nothing is stored,
  GET returns the real defaults with an explicit "defaults" note rather than
  pretending a decision was made.
* `RetrievalParams.config_fingerprint()` is a stable hash of the configuration,
  used to tie experiment artifacts to the exact configuration that produced them.

## Observability (IMPLEMENTED, Phase 18 groundwork)

Every execution produces a `RetrievalRun` row (`retrieval_runs` table) with:
query, strategy, config fingerprint, full params, applied weights, normalization,
rrf_k, embedding model, vector backend, reranker + model + status, corpus version
and fingerprint, candidate counts per stage, result count, stage timings and notes.

* `POST /retrieve` returns `retrieval_run_id`; `GET /retrieval-runs` lists summaries
  and `GET /retrieval-runs/{id}` returns the full record.
* Stage timings are measured (`embed`, `dense`, `lexical`, `fusion`, `rerank`,
  `diversity`, `total`). An unmeasured stage stays `null`, never 0.
* Runs are never edited or deleted. (Retention/rotation is PLANNED.)

## Provenance-first results (IMPLEMENTED)

Every result carries the shared provenance key set (document, title, source,
publisher, section path, page, slide, slide title, trust score, content hash,
chunk index, chunking strategy/config, kb version, `user_provided`) plus score
provenance:

| field | meaning |
|---|---|
| `score` | the score this strategy ranks by (comparable, documented per strategy) |
| `retrieval_score` | the strategy's own score (fused when fusion ran) |
| `rerank_score` | cross-encoder score, only when reranking applied |
| `score_breakdown` | dense / lexical raw + normalized + fused + contributions |
| `rank`, `retrieval_method`, `stages` | position, strategy id, stages survived |
| `why` | deterministic sentence: "dense rank 1 (cosine 0.5603) + lexical rank 2 (BM25 3.41) -> fused 0.812" |

`why` is generated from measured values only — no LLM, no invented justification.

## API surface added in V6 (IMPLEMENTED)

All under `/api/knowledge-bases/{kb_id}`:

| method | path | purpose |
|---|---|---|
| POST | `/retrieve` | run retrieval; accepts `strategy` + `config` overrides (V1–V5 body still works) |
| GET | `/retrieval-strategies` | registry contents (name, description, requirements, baseline flag) |
| GET | `/retrieval-config` | persisted configuration, or the real defaults with a "defaults" note |
| PUT | `/retrieval-config` | persist a configuration for this KB |
| GET | `/retrieval-runs` | compact run list (strategy, result count, latency, reranker status) |
| GET | `/retrieval-runs/{run_id}` | full run record (params, applied weights, stage counts, timings, notes) |
| GET | `/bm25-index` | lexical index status incl. explicit `stale` + `staleness_check` |
| POST | `/evaluate` | accept `strategy`; the run records `retrieval_strategy` + `retrieval_params` |

Failure mapping: unknown strategy / invalid configuration -> **400** (client error);
missing KB -> **404**; embedding mismatch or vector-store failure -> **503** with an
explicit message. A configuration problem is never reported as a service outage.

## What is NOT implemented

* **Query processing / interpretation** — PLANNED (V6 Phase 9). Raw queries are used
  as given; there is no LLM or heuristic rewriting, and none is claimed.
* **Grounded answering, citations, abstention, chat API/UI** — NOT AVAILABLE in
  this repository state (V6 Phases 10–14).
* **Retrieval Lab V2 UI** — NOT AVAILABLE. The existing Lab is dense-only; the API
  already accepts `strategy`/`config` so the UI can be upgraded without backend work.
* **Evaluation comparison across strategies** — the evaluator now accepts any
  strategy and records `retrieval_strategy` + `retrieval_params`, but the
  frozen-benchmark comparison runner (experiment artifacts) is PLANNED.
* **TurboVec backend** — NOT AVAILABLE (optional experimental backend, Phase 17).
  The product works fully without it.
* **MCP** — deliberately NOT implemented; documented as a future boundary only.
* **Stemming, stop-word lists, CJK segmentation** — NOT AVAILABLE in the tokenizer
  (documented limitation; IDF already discounts common terms).

## Engineering rules this layer follows

* No LLM is used where deterministic code suffices (ranking, normalization, fusion,
  filtering, thresholds, configuration, provenance).
* A missing measurement is `null` / `"unknown"`, never a fabricated `0` or `%`.
* Nothing in the retrieval layer deletes, mutates or re-indexes corpus data: it is
  read-only apart from writing its own derived index/cache and run records.
* Tests stub the vector factory at the module attribute
  (`app.services.vector_store.factory.create_vector_store`) and never create or
  touch a real Qdrant collection.
