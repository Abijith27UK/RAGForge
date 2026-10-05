# V6 continuation checkpoint

Written at the end of the session that delivered V6 phases **1–8** (plus phase 18
groundwork). Read this before continuing V6 so the state is not re-derived from
scratch.

> **Update after V7:** the answering phases below (9–14, 19) were delivered by V7
> under a different architecture shape than sketched here — the API is
> `POST .../answer` (not `/chat`), and the implementation lives in
> `app/services/answering/` + `app/schemas/answer.py` + `app/api/routes_answer.py`.
> See [answering-architecture.md](answering-architecture.md) and
> [v7-continuation-checkpoint.md](v7-continuation-checkpoint.md) for the real
> state. Phases 15–18 and 20 remain open as listed below.

## Repository state at this point

* Branch: `Dev_1_midterm`. Working tree contains V6 changes **uncommitted**
  (see "Files" below); V1–V5 are committed.
* Backend tests: **318 passed, 0 failed** (244 V5 + 74 V6).
* `python -m compileall -q app` → exit 0.
* Frontend: `npx tsc --noEmit` → clean; `npm run build` → 14 routes, exit 0.
  **No frontend files were changed in this session.**
* Qdrant: 14 collections before and after the full test suite (13 real KB
  collections + the pre-existing `kb_live_itest`). No test collection was created.
* Frozen benchmarks: `git diff --stat HEAD -- benchmarks/` is empty.
* One unrelated working-tree change: `frontend/tsconfig.tsbuildinfo` (a tracked
  generated file; it changes whenever `tsc` runs). It should be `git rm --cached`
  in a separate housekeeping commit — do not bundle it into a feature commit.

## What is DONE (do not rebuild)

Phases 1–8 of the 20-phase V6 plan. Full detail in
[retrieval-architecture.md](retrieval-architecture.md).

| phase | deliverable |
|---|---|
| 1 | repository audit (read-only) |
| 2 | registry: `RetrieverSpec`, lazy built-ins, `describe_retrievers()`, legacy-safe `get_retriever`, no hard-coded strategy at any call site |
| 3 | real BM25 (`lexical.py`), `Bm25Retriever`, persisted index + exact corpus-revision staleness detection |
| 4 | hybrid retrieval with per-pool normalization and weighted fusion |
| 5 | Reciprocal Rank Fusion behind the same interface |
| 6 | `Reranker` ABC, `CrossEncoderReranker`, honest `unavailable_fallback`, status/model/candidate_k/final_k reported |
| 7 | diversity: per-document cap (demotes, never drops) + MMR with documented unavailability |
| 8 | `RetrievalParams` persisted per KB; `RetrievalRun` recorded per request; config/run/bm25-index endpoints |
| 18 (partial) | stage statuses, measured timings, `retrieval_run_id` on every response |

## What is NOT done (the remaining plan)

| phase | item | notes / constraints |
|---|---|---|
| 9 | Query processing | **done (V7)** — `services/answering/query_processor.py`, deterministic, heuristic-labelled |
| 10 | `AnsweringService` | **done (V7)** — `services/answering/service.py`; provider-agnostic via `AnswerGenerator`; retrieved text is UNTRUSTED DATA wrapped in `<EVIDENCE>` |
| 11 | Citation validation | **done (V7)** — `services/answering/validation.py`; deterministic; invalid citations removed/downgraded with recorded actions |
| 12 | Evidence gating + abstention | **done (V7)** — `services/answering/gate.py`; 8 signals, categorical confidence, thresholds in code + documented |
| 13 | Chat API (`POST /api/knowledge-bases/{kb_id}/chat`) | **done (V7) as `POST .../answer`** + `GET .../answers/{id}` + `GET .../answer-traces/{id}`; returns answer + claims + citations + evidence + grounding_assessment + retrieval_run_id + trace id |
| 14 | Chat UI + evidence panel | **done (V7) as the Answer page** (`/knowledge-bases/[id]/answer`) reusing the V5 design system |
| 15 | Retrieval Lab V2 | partial — strategy selection + retrieval-vs-answer comparison added in V7; fusion/rerank parameter controls still open |
| 16 | Evaluation extension | planned |
| 17 | Retrieval experiment runner | planned |
| 18 | TurboVec | not started |
| 19 | Prompt-injection + security tests | **done (V7)** — `test_answering_v7.py` (markers, HTML/PDF-style payloads, no-secrets) + `test_answer_api_v7.py` (injected document) |
| 20 | Documentation + final verification | ongoing |

## Hard constraints to carry forward

1. Do not rewrite V1–V5 functionality; all 244 original tests must keep passing.
2. Do not modify, regenerate or delete anything in `benchmarks/`.
3. Do not run destructive operations against real Qdrant collections. Classify any
   target as test / experiment / real user / frozen benchmark KB first.
4. Tests must stub `app.services.vector_store.factory.create_vector_store` (module
   attribute, resolved at call time) and must not create Qdrant collections.
5. No LLM where deterministic code suffices (ranking, fusion, filtering, citation
   validation, thresholds, configuration, provenance).
6. Never fabricate a score, count, metric or status: use `null`/`-1`/`NOT_AVAILABLE`.
7. Naval Architecture retrieval quality stays **unmeasured** until a
   human-reviewed benchmark exists for it. Do not report metrics for it.

## Known limitations introduced by phases 1–8

* BM25 normalization is pool-relative: `score` for the `bm25` strategy depends on
  `candidate_k`. Raw scores are always reported in `score_breakdown`.
* Min-max fusion is likewise pool-relative (inherent to min-max normalization);
  `rank` normalization is available as a scale-free alternative.
* BM25 has no stemming, no stop-word list and no CJK segmentation.
* MMR requires `VectorStore.fetch_vectors`; implemented for Qdrant and the test
  fake, absent (and reported as unavailable) elsewhere.
* `retrieval_runs` grows without bound (no retention policy yet).
* Only `dense` and `bm25` were exercised against live Qdrant in this session;
  hybrid/reranked were exercised hermetically (in-memory cosine store) plus the
  cross-encoder model path once against the real download.

## Suggested execution order for the next session

Phases 9 → 10 → 11 → 12 → 13 (backend answering stack), then 14 → 15 (UI), then
16 → 17, then 18 → 19 → 20. Run `pytest tests/test_retrieval_v6.py
tests/test_retrieval_api_v6.py` first (≈17 s) as a cheap smoke test before the
full suite.
