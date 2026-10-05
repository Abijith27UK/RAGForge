# Verification log

## Initial MVP (before stabilization)

Commands run and results:

| Check | Command | Result |
|---|---|---|
| Backend tests | `cd backend && python -m pytest` | **50 passed** |
| Backend boot | `python -m uvicorn app.main:app --port 8000` | started; `/api/system/health` → `{"status":"ok"}`; `/api/system/status` reports Qdrant unreachable (honest), KB creation returned 201 |
| Frontend build | `cd frontend && npm run build` | compiled + typechecked, all 9 routes built |
| Frontend deps | `npm install` | ok (Next 15.5.4, React 19.1.0, Tailwind 3) |
| Backend deps | `pip install -r backend/requirements.txt -r backend/requirements-openai.txt` | ok on Python 3.14 |

Environment notes:
- Windows, no Docker → Qdrant to be run as `qdrant.exe` (README instructions).
- Full end-to-end vector pipeline (ingest → embed → Qdrant → retrieve → evaluate) requires
  Qdrant running and sentence-transformers installed; these are documented in README and
  verified by unit tests at the component level (metric math, chunking, scoring, API contracts).

---

## Stabilization phase (Tasks 1–4 + real E2E run)

Date: 2026-09-15 · Machine: Windows, Git Bash · Python 3.14.7 (backend/.venv)

### Commands and results

| Check | Command | Result |
|---|---|---|
| Backend tests (all, incl. new integration tests) | `cd backend && .venv/Scripts/python.exe -m pytest -q` | **65 passed** (was 50; +8 retrieval/evaluation integration, +7 source-quality/page-provenance tests) |
| Live embedding smoke | python: `create_embedding_provider` + embed | sentence-transformers/all-MiniLM-L6-v2 → 384-dim, L2-normalized (model downloaded from HF Hub) |
| Qdrant smoke | ensure_collection / search / delete against `localhost:6333` | ok (qdrant 1.19.1, Windows build, running from `qdrant-dl/`) |
| Frontend typecheck | `cd frontend && npx tsc --noEmit` | clean |
| Frontend build | `cd frontend && npm run build` | all routes built |
| Real E2E run | `cd backend && .venv/Scripts/python.exe scripts/e2e_automobile.py` | **completed end-to-end with real data** (evidence below) |

New test files: `backend/tests/test_retrieval_integration.py`
(hermetic in-memory cosine store + live-Qdrant tests that **skip honestly** when Qdrant is down).
New E2E driver: `backend/scripts/e2e_automobile.py` (re-runnable).

### Real end-to-end run — Automobile Engineering (evidence)

Driver: `backend/scripts/e2e_automobile.py` against the live API (`:8000`) and Qdrant (`:6333`).
KB kept for inspection: `kb_f278c283c748` ("Automobile Engineering KB (E2E)", status `ready`).

| Stage | What actually happened (evidence from run output) |
|---|---|
| System status | Qdrant reachable=true; embedding=sentence-transformers/all-MiniLM-L6-v2; LLM "not configured (dev mock will be used)" — honest disclosure |
| KB created | `kb_f278c283c748`, domain "Automobile Engineering", 201 |
| Domain analysis | dev-mock analyzer ran, **is_mock=True, generated_by=mock:mock-1** (no LLM key configured — labelled, not faked as real) |
| Source discovery + quality | 4 real Wikipedia URLs scored with **real accessibility probes**: all HTTP 200, Last-Modified captured (2026-09-12/14), scores 0.458 → REVIEW (explained reasons); user-overridden to ACCEPT via the decision API |
| Ingestion | "4 ingested, 0 duplicate skipped, 0 failed"; real downloads: 118,213 + 15,838 + 50,232 + 8,734 chars parsed |
| Chunk+Embed+Index | section-aware: **351 chunks → 351 vectors (real MiniLM 384d) → Qdrant upsert**, run took 32.5 s |
| Stale-vector fix proof | re-indexed with fixed-size (900/100): 242 new chunks; **Qdrant `points_count` = 242 exactly = SQLite chunk count** (without the fix: 593 points = 351 stale + 242 new); message "242 indexed; 0 stale vectors removed"; chunk IDs fully replaced |
| Retrieval | 3 real queries; top-1 hits from the correct source each time (e.g. "battery management system protect cells" → Battery_management_system page, score 0.659) with full provenance payloads |
| Ground truth + evaluation | 4 questions authored with expected keywords; run: **Recall@5 = 1.0, Precision@5 = 0.9, MRR = 1.0, NDCG = 0.985** (real retrieval, k=5) |
| Integrity disclosure | aggregate note: "4/4 question(s) scored with the keyword-overlap heuristic (no explicit ground-truth IDs); treat as indicative, not rigorous." |

Raw driver output is reproduced in the session transcript; re-run the script to reproduce.

### Fixes applied in this phase

1. **Stale vectors (Task 1)** — `routes_build.index_kb` now deletes per-document vectors
   in Qdrant *before* upserting new ones, replaces SQLite chunks per document, and sweeps
   remaining orphans (`QdrantVectorStore.delete_orphaned_points`); re-ingest no longer
   creates duplicate Document rows for identical content (hash-dedup skips + counts them).
2. **Evaluation integrity (Task 2)** — real document-level metrics (doc Recall/Precision/
   MRR/NDCG) computed from `expected_document_ids` (previously dead code); aggregate run
   note discloses how many questions used the keyword-overlap fallback; the fallback's
   relevance universe is retrieved items only (it cannot count non-retrieved docs as
   false negatives); "no ground truth" stays `None` (never 0); retrieval/evaluation refuse
   to run when the KB was indexed with a different embedding model
   (`KB.embedding_identity` recorded at index time).
3. **Ground-truth authoring UI (Task 3)** — Evaluation page now authors all three ground-truth
   modes: keywords (heuristic, labelled), expected chunks (rigorous; searchable chunk picker),
   expected documents (doc-level); run cards show doc-basis metrics and the aggregate
   disclosure banner.
4. **Source quality + PDF provenance (Task 4)** — every candidate source gets one SSRF-safe
   accessibility probe (`utils/http_probe.py`): verified HTTP status and Last-Modified feed the
   accessibility and recency signals; unverified accessibility now warns instead of silently
   scoring neutral; probe evidence stored in `quality.probe`. PDF ingestion emits page markers
   that the section-aware chunker converts into per-chunk `page` provenance (verified by tests;
   markers stripped from chunk text).

### Known limitations after this phase

- Domain analysis used the labelled dev-mock (no LLM key configured); with `LLM_API_KEY` set
  the same pipeline produces a real LLM-generated spec.
- E2E evaluation used keyword ground truth (rigorous chunk IDs can be authored in the new UI);
  the disclosure note makes this visible on every run.
- Qdrant binary lives in `qdrant-dl/` (gitignored); `qdrant/` per README also gitignored.
- No git repository has been initialized yet; `.gitignore` already excludes `.env`,
  `*.env.local`, `backend/data/`, and the Qdrant binary directories.

---

## Research baseline (automobile-engineering-baseline-v1)

Date: 2026-09-15 · Full write-up: `docs/experiment-automobile-baseline-v1.md`

| Check | Command / evidence | Result |
|---|---|---|
| Backend tests after strict-mode change | `cd backend && .venv/Scripts/python.exe -m pytest -q` | **66 passed** |
| Frontend typecheck | `cd frontend && npx tsc --noEmit` | clean |
| Corpus extension | discover → probe (7× HTTP 200) → accept → ingest → index | 11 documents, **812 chunks**, Qdrant points_count = 812 (verified via API) |
| Ground-truth validation | `python scripts/run_baseline_benchmark.py ...` | 28/28 questions validated (all chunk/doc IDs exist in KB); loader refuses stale GT |
| Strict eval runs (explicit chunk GT only) | same script, `allow_keyword_fallback=False` | k=3: R=0.821 P=0.286 MRR=0.708 NDCG=0.738 · k=5: R=0.964 P=0.200 MRR=0.737 NDCG=0.793 · k=10: R=1.000 P=0.104 MRR=0.743 NDCG=0.806 |
| Integrity counts in every run | aggregate fields | explicit_gt=28, keyword_fallback=0, skipped=0, strict_mode=true, run_label recorded |
| Reproducible artifacts | `benchmarks/automobile-engineering-baseline-v1.json` + `-results.json` | benchmark (28 questions, quoted-passage provenance) + results (run ids, labels, per-question metrics) saved |

Ground truth for the benchmark was authored by manually reading indexed chunk texts (quoted
passage recorded per question); authorship is agent-assisted under explicit human instruction
and marked "human review pending" in the benchmark file. Keyword relevance was NOT used for
any headline metric; it remains a strictly separated diagnostic mode
(`allow_keyword_fallback=True`, surfaced as a checkbox in the UI).

---

## Source-selection experiment v1

Date: 2026-09-15 · Design: `benchmarks/source-selection-experiment-v1.json` ·
Results: `benchmarks/source-selection-experiment-v1-results.json` · Report:
`docs/experiment-source-selection-v1.md`

| Check | Evidence | Result |
|---|---|---|
| Candidate pool | 36 real Wikipedia URLs, all HTTP 200 before freezing | frozen in design JSON |
| Pool quality scoring | staging KB `kb_deaef26a2c0f`, real pipeline + probes | 36/36 scored; **34/36 tied at 0.4525** (recorded in results `pool_ranking`) |
| Six corpora built | QUALITY/RANDOM × N=5,8,11, seed 20260915, all through the live pipeline | 277–1191 chunks each; Qdrant/SQLite consistent |
| Frozen GT translation | content-hash translation in the runner script | 28/28 questions resolved per corpus; benchmark file untouched |
| Strict evaluations | strict mode, labelled runs | metrics recorded per run with coverage; **global answerable-question intersection = 0** → cross-run comparison invalid in v1 |
| Primary finding | score distribution | the current scorer cannot discriminate same-type sources; "top-N" degenerated to alphabetical selection |
| Tests after experiment | `cd backend && .venv/Scripts/python.exe -m pytest -q` | **66 passed** |

The experiment is **inconclusive by instrument failure** (documented in the report): neither
hypothesis H1 nor H2 can be accepted or rejected. All results inherit the
"agent-authored, human review pending" ground-truth limitation. The frozen baseline
(`automobile-engineering-baseline-v1`) was not modified.

## Source-quality scorer v2 (content-aware redesign) + source-selection experiment v2

**Date:** 2026-09-15 · **Design:** `docs/design-source-quality-v2.md` · **Experiment report:**
`docs/experiment-source-selection-v2.md` · **Artifacts:**
`benchmarks/source-selection-experiment-v2.json`, `-results.json`, `-paired-analysis.json`,
`benchmarks/v2-pool-report.json`

| Check | Method | Result |
|---|---|---|
| v2 scorer instrument check | `report_v2_pool_scores.py` on the frozen 36-URL pool | **36/36 unique** content_relevance and composites (v1: 2 unique); two pre-validation iterations rejected by evidence audit (blended-query mismatch; navbox boilerplate faking coverage) and fixed |
| Scorer unit tests | `backend/tests/test_content_scorer.py` | 12 passed (domain map, prose filtering, boilerplate stripping, weighting, honest degradation) |
| Full backend suite | `pytest -q` | **78 passed** |
| v2 experiment runs | `run_source_selection_experiment_v2.py`, 6 corpora, seed 20260915 | all built through the live pipeline; strict evals k=3/5/10 saved with run IDs |
| Coverage finding | answerable frozen-benchmark questions per corpus | QUALITY 11/13/17 vs RANDOM 4/4/10 of 28 — content-aware selection packs 1.7–3.25× more answerable knowledge per source (2.2–4.5× fewer chars per answerable q) |
| Paired metric comparison | per-question metrics on common answerable subsets (`-paired-analysis.json`) | QUALITY N=11 vs RANDOM N=11 (n=6): Recall identical on all 6; MRR 0.650 vs 0.667 — **no detectable retrieval-quality difference at this sample size**; global intersection 0 so raw rows are not comparable |
| Artifact verification | independent recomputation of every paired value from live evaluation runs | all intersections, means, win/tie counts, and stored aggregates match; no fabricated values; labels verified (left=QUALITY, right=RANDOM) |
| Selection mechanism | Spearman ρ of each signal with the composite ranking + unique-value counts | content_relevance ρ=+0.882, 36/36 unique values drives selection; authority/accessibility/type/recency/duplication each constant (1 unique value) in this pool — metadata cannot affect order |
| Frozen artifacts | baseline benchmark + v1 experiment/results | untouched |

**Verdict recorded in the experiment doc:** H2 (efficiency/coverage) supported; H1 neither accepted
nor rejected (floor effects, one seed, n≤17 answerable per corpus). Ground truth remains
"agent-authored, human review pending" — inherited by every number above.

Explicitly: **experimentally demonstrated** = the v2 scorer discriminates a homogeneous pool
(36/36 unique vs v1's 2 unique) and content-aware selection yields corpora that cover 1.75–3.25×
more benchmark questions per source with 2.2–4.5× fewer characters per answerable question.
**Remains inconclusive** = whether QUALITY selection improves retrieval *ranking* (the only valid
paired subset, n=6 at N=11, shows identical Recall and statistically indistinguishable MRR), and
anything beyond this single seed. **Known issue preserved** = the six runs have different
answerable-question subsets (global intersection 0); aggregate rows across runs are descriptive
only and were never used for conclusions — the paired analysis is the sole basis for
QUALITY-vs-RANDOM performance statements.

One integrity note: the first full-run attempt hit the 10-minute tool timeout after 5 of 6 corpora
and results were not persisted; the runner was extended with mergeable per-corpus runs
(`--only "STRATEGY\|N"`), the 5 orphan KBs deleted, and all six runs re-executed cleanly and saved.

## V3 foundation: benchmark lifecycle, experiment framework, pluggable infrastructure

**Commands and results**
- `python -m pytest tests/ -q` → **95 passed, 1 skipped** (skip = live-Qdrant integration test
  reporting honestly when the service is unreachable)
- New test files: `tests/test_benchmark_lifecycle.py` (11 tests: lifecycle transitions, reviewer
  requirement, approved-edit⇒new-revision, FROZEN immutability, snapshot freeze rules, freeze
  protection on delete, frozen-version evaluation in the hermetic evaluator, refusal of unfrozen /
  unknown versions), `tests/test_v3_infrastructure.py` (6 tests: seeded determinism + 5-seed
  distinctness, chunking registry + hash reproducibility, strategy/config round-trip, vector
  factory with unknown-backend rejection + experimental registration)
- `npx tsc --noEmit` → clean; `npm run build` → clean (12 routes incl. new `/experiments`)
- Live smoke: question → APPROVED (reviewer required, verified 422 without) → benchmark-version
  snapshot (status APPROVED) → scratch KB deleted; `/api/experiments/*` serves frozen artifacts

**What was implemented (all real, no mock data paths added)**
- Phase A: question lifecycle + benchmark version snapshots + freeze protection + frozen-version
  evaluation runs (`routes_benchmark.py`, `Evaluator`), full review UI on the Evaluation page
- Phase B (infra only, NOT run): `benchmarks/source-selection-experiment-v3.json` +
  `scripts/run_source_selection_experiment_v3.py` — multi-seed (20260915–20260919), gated on a
  FROZEN benchmark version id; refuses to start otherwise
- Phase C/D: chunking registry + strategy/config stamping; vector-backend factory +
  `vector_backend` on KB (qdrant default, unchanged)
- Read-only `/api/experiments` endpoints + `/experiments` page rendering frozen v1/v2 artifacts
  verbatim (coverage bars, per-k strict metrics, paired analysis, subset-difference disclaimer)

**Frozen artifacts re-verified untouched:** baseline benchmark/results, source-selection v1+v2
results and paired analysis. The existing 28-question Automobile benchmark remains
"agent-authored, human review pending" — lifecycle statuses were NOT auto-migrated; human review
is the next gating step before any v3 experiment executes.

---

## V4 — User knowledge ingestion, KB maturation, source-selection v3 readiness

**Date:** 2026-10-02 · Machine: Windows, Git Bash · Python 3.14.7 (`backend/.venv`)
**Product doc:** `docs/user-knowledge-workflow.md`

### Commands and results

| Check | Command | Result |
|---|---|---|
| Backend tests (all) | `cd backend && .venv/Scripts/python.exe -m pytest -q` | **177 passed** (was 96; +81 V4 tests) |
| Pre-existing V1–V3 tests | same command, filtered to the original 11 files | **96 passed — zero regressions** |
| Frontend typecheck | `cd frontend && npm run typecheck` | clean (script added; `tsc --noEmit`) |
| Frontend build | `cd frontend && npm run build` | clean, 13 routes incl. new `/knowledge-bases/[id]/documents` |
| Frozen-benchmark gate | `backend/scripts/verify_benchmark_integrity.py` | read-only; refuses any non-FROZEN benchmark; exercised by tests |

### New test files (81 tests)

| File | Tests | Covers |
|---|---|---|
| `tests/test_user_ingestion.py` | 28 | PPTX/DOCX/PDF/TXT/MD/HTML parsing on **real in-memory files**; slide numbers + slide titles; DOCX heading paths + table preservation; PDF page provenance; upload validation (traversal, size, magic bytes, mislabelled OOXML, corrupt ZIP, plain-ZIP-as-DOCX); legacy `.ppt` rejection; image-only PDF rejection; USER_PROVIDED integrity scoring; full provenance set on every supported format |
| `tests/test_incremental_documents.py` | 21 | incremental indexing touches only the new document; stale vectors deleted on re-index; statuses/counts recorded; indexing **refuses** when the store cannot delete vectors; skip reasons are reported; document delete removes vectors + rows; replacement keeps history and drops the old version's vectors; source-mode round-trips; retrieval provenance incl. page/slide/section/version; absent keys omitted not invented; retriever registry; KB with no benchmark; frozen benchmark untouched by document work; ingestion performs **no network I/O** |
| `tests/test_documents_api.py` | 32 | API contracts for all three source modes; multi-format upload; duplicate upload creates no second document; per-file rejection; traversal neutralisation; library/detail/text/chunks endpoints; incremental index endpoint; failed-document indexing refused; replacement versioning; identical-byte replacement refused; overview for a READY KB with **no benchmark**; evaluation never fabricates metrics; frozen benchmark immutable through V4 flows; the v3 runner's frozen gate; `ingest` explains user-provided sources; user URLs marked `user_provided` |

### Product bugs found and fixed by these tests

1. `upload_documents` treated the indexer outcome as an object when it returns a dict → 500 on
   every indexed upload.
2. `Source.quality` was left `None` for user-uploaded files, so the Sources page could not
   render the integrity assessment uniformly → now set alongside `Source.integrity`.
3. `UserProvidedIntegrityScorer` returned `REVIEW` for any file under 200 characters, which
   downgraded perfectly valid short class notes → thin content is now `ACCEPT` + warning;
   only hard technical failures reject.
4. `POST /documents/{id}/replace` superseded a document **without deleting its vectors**, so
   the replaced version stayed retrievable → vectors are now deleted on replacement.

### Verified existing behaviour (not regressed)

* All 96 pre-existing tests pass unchanged, including source-quality scoring, benchmark
  lifecycle/freeze protection, chunking, metrics, live-Qdrant integration and API contracts.
* `benchmarks/` frozen artifacts are byte-identical; no historical experiment result was read,
  rewritten or deleted.
* The frozen Automobile Engineering benchmark is never auto-migrated or mutated by any V4 code
  path — proven by `test_frozen_benchmark_stays_immutable_across_document_work` and
  `test_frozen_benchmark_cannot_be_edited_or_deleted_through_v4_flows`.

### Explicitly NOT done in this phase

TurboVec, BM25, hybrid retrieval, reranking, autonomous optimization, MCP, and LLM answering.
The only preparation is the `Retriever` registry, the parser registry and the vector-backend
factory. Source-selection v3 was **prepared but not executed** — it builds 18 corpora and must
be a deliberate, conscious run.

---

## V4 final pass — stale-vector count honesty + end-to-end re-verification

**Date:** 2026-10-02 · Same machine/venv as above. No new features; this pass fixed a
**reporting** defect and re-ran every gate.

### Defect found and fixed: "0 stale vectors removed" was always printed

The installed `qdrant-client` returns `UpdateResult(operation_id, status)` for `delete()` —
**no deleted-count field**. The old code read `len(result.operation.result)`, which always
raised, was swallowed, and fell back to `0`. The delete itself worked, but *every* caller
(build-run stage message, the incremental-index stage message, and the
`DELETE /documents/{id}` response) reported `vectors_removed: 0` even when hundreds of points
were deleted. That is exactly the kind of fabricated number this project must never show.

Fix ([qdrant_store.py](../backend/app/services/vector_store/qdrant_store.py)):

* `_delete_by_document_filter()` now measures the collection's `points_count` **before and
  after** the delete and returns the real delta.
* When the collection cannot be read, it returns **`-1` = unknown** — never a fabricated `0`.
* `delete_document_vectors()` propagates `-1`; `delete_orphaned_points()` prefers the measured
  delta and only falls back to the attempted count when the collection is unreadable.

Fix ([document_indexer.py](../backend/app/services/indexing/document_indexer.py)):

* `IndexOutcome.stale_label()` → `"unknown"` for `-1`; `as_dict()` also carries
  `stale_vectors_removed_label`.
* An orphan sweep returning `-1` **no longer cancels** a known count
  (`if outcome.stale_vectors_removed >= 0 and swept >= 0`).

Propagation: `routes_build.py`, `routes_documents.py` (stage messages, delete response now also
returns `vectors_removed_label` and `vectors_removal_confirmed`), and the frontend
(`api.ts`, Document Library page) render the label and warn honestly when a removal could not
be confirmed.

### Re-verification results

| Check | Command | Result |
|---|---|---|
| Backend tests (all) | `cd backend && .venv/Scripts/python.exe -m pytest -q` | **181 passed, 7 warnings** (177 + 4 new honesty tests) |
| Byte-compile | `.venv/Scripts/python.exe -m compileall -q app` | exit 0 |
| Frontend typecheck | `cd frontend && npm run typecheck` | clean |
| Frontend build | `cd frontend && npm run build` | clean, 13 routes |
| Frozen artifacts | `git diff --stat HEAD -- benchmarks/` + canonical-hash compare | **all 10 files semantically identical to HEAD**; no untracked files in `benchmarks/` |
| Secrets | `git grep -nIE "sk-…|AKIA…|PRIVATE KEY"` on tracked files | no matches; `backend/.env` untracked and ignored |
| Live product sanity | `scripts/sanity_check_user_kb.py --base http://localhost:8011` (real Qdrant on :6333, real MiniLM) | **15/15 PASS** |

### Live proof the count fix works

Sanity step 12 against real Qdrant now reports a **confirmed non-zero** count:

```
PASS  12. Delete a document and its vectors
        vectors_removed=1, store_error=None; Qdrant 9 -> 8, docs 7 -> 6
```

Before the fix this line printed `vectors_removed=0` for the same deletion.

### Note on raw-bytes vs content (benchmarks/)

Six benchmark files differ **raw-byte-wise** from HEAD while their canonical JSON hashes match
exactly. Cause: `core.autocrlf=true` on this Windows checkout rewrites LF → CRLF on checkout.
This is **not** a content change — verified per file by `json.dumps(..., sort_keys=True)`
equality plus SHA-256 of the canonical form.

### Known housekeeping (not changed unilaterally)

`frontend/tsconfig.tsbuildinfo` is a **tracked generated file**, so `.gitignore` cannot help
until it is untracked: `git rm --cached frontend/tsconfig.tsbuildinfo`. `.qdrant-initialized`
is likewise a tracked marker file. Both were pre-existing; neither was removed here.

### Not done in this phase (unchanged)

TurboVec, BM25, hybrid retrieval, reranking, autonomous optimization, MCP, LLM answering, OCR,
CSV/XLSX, auth, cloud deployment.

---

## V5 — Corpus reliability verification

**Date:** 2026-10-02 · Windows, Git Bash · Python 3.14.7 (`backend/.venv`)

### Commands and results

| Check | Command | Result |
|---|---|---|
| Backend tests (chunk 1 of 2) | `pytest tests/test_api … test_incremental_documents -q` | **157 passed** |
| Backend tests (chunk 2 of 2) | `pytest tests/test_llm_and_embeddings … test_v3_infrastructure -q` | **87 passed** |
| **Backend total** | — | **244 passed, 0 failed** (181 pre-existing + 63 new corpus tests) |
| Byte-compile | `python -m compileall -q app` | exit 0 |
| Frontend typecheck | `npm run typecheck` | clean, exit 0 |
| Frontend build | `npm run build` | exit 0, 14 routes incl. new `/knowledge-bases/[id]/corpus` |
| Frozen artifacts | canonical-hash compare vs `HEAD` | **all 10 files semantically identical**; `git diff HEAD -- benchmarks/` empty |
| Secrets | `git grep` for key patterns on tracked files | no matches; `backend/.env` untracked + ignored |
| **Live integration** | `scripts/sanity_check_corpus.py --base http://localhost:8012` (live Qdrant, real MiniLM, isolated temp KB) | **20/20 PASS** |
| Scale benchmark | `scripts/corpus_benchmark.py` | 10/50/100/200 docs — see below |

### Corpus benchmark results (measured)

| Documents | Chunks | docs/min | chunks/s | vectors/s | total s |
|---:|---:|---:|---:|---:|---:|
| 10 | 80 | 2700.3 | 360.0 | 360.0 | 0.22 |
| 50 | 400 | 2409.9 | 321.3 | 321.3 | 1.24 |
| 100 | 800 | 2209.9 | 294.7 | 294.7 | 2.71 |
| 200 | 1600 | 2425.5 | 323.4 | 323.4 | 4.95 |

Throughput is **flat from 10 to 200 documents** — no super-linear degradation at the
target corpus size. Artifact: `backend/data/benchmarks/corpus-scale.json` (inside
`data/`, never the repository's frozen `benchmarks/`).

**Stated limits:** this measures RAGForge's own orchestration using the hashing
embedding provider and the in-memory store. It does **not** measure transformer
inference speed, real Qdrant network throughput, or retrieval quality of any kind.

### Live integration evidence (20/20)

Real Qdrant + real `all-MiniLM-L6-v2` against an isolated temporary KB
(`SANITY-CORPUS-TEMP`), cleaned up afterwards:

```
PASS  Bulk batch ingestion                          7 files, HTTP 201
PASS  Batch accounted for every file                total=7 completed=6 failed=1 status=partial
PASS  One bad file does NOT fail the batch          6 ok / 1 failed / status=partial
PASS  Failure carries an actionable reason          INVALID_FILE: not a readable ZIP container
PASS  Manifest counts the corpus                    documents=6 chunks=8 vectors=8 confirmed=True
PASS  Vector count is CONFIRMED against live Qdrant total_vectors=8
PASS  PPTX slide provenance survives                slides=2 parser=pptx
PASS  DOCX section provenance survives              sections=2 parser=docx
PASS  Duplicate content detected, existing kept     duplicates=1 completed=0
PASS  Resume is idempotent                          documents 6 -> 6; resumable=1
PASS  Integrity scan ran against live vectors       WARNING docs=6 chunks=8 vectors=8
PASS  Integrity scan is clean on a fresh corpus     orphans=0 missing=0 stale=0
PASS  Destructive repair refused without confirm     HTTP 400: confirm_action required
PASS  Corpus fingerprint is deterministic           631695cd54f2… (6 docs)
PASS  Unchanged corpus reuses one version           v1
PASS  New domain reports ground truth NOT_AVAILABLE status=NOT_AVAILABLE, no metrics
PASS  Retrieval returns provenance-bearing results  3 results, score=0.5021
```

### Bugs found and fixed by this phase

1. `process_item` built a `Source` but never persisted it → every indexed upload
   failed with *"source … no longer exists"*.
2. A batch with failures reported status `complete`. Now `partial` — a partially
   failed corpus must never read as a complete ingestion.
3. A failed parse **discarded the user's bytes**, forcing a re-upload of a
   200-file corpus to fix one file. `IngestionError.raw_path` now carries the path
   and the failed document keeps `raw_file_path`, so
   `reindex_failed_documents` can retry from disk.
4. `REINDEX_FAILED_DOCUMENTS` filtered out failed documents and could therefore
   never retry one — a self-contradicting action. It now re-parses from stored
   bytes first.
5. A no-op repair reported `ok=False`. "Nothing to do" is now a plain message.
6. The corpus fingerprint included `kb_id`, so two knowledge bases holding
   identical corpora disagreed — defeating the point of a reproducible
   fingerprint. Excluded (with the other non-deterministic fields).
7. `routes_corpus` bound `create_vector_store` at import time, so the vector
   factory could not be stubbed and the corpus API tests created ~28 real Qdrant
   collections per run. Now resolved through the factory module at call time.
8. `test_documents_api.py` also hit real Qdrant; now stubbed at the factory
   boundary. A full suite run no longer pollutes the vector store.
9. `MAX_UPLOAD_FILES_PER_REQUEST` was 50, which made the required 100-document
   batch impossible. Raised to 200 to match the stated 50–200 corpus target.

### Explicitly NOT done

TurboVec, BM25, hybrid retrieval, reranking, autonomous optimisation, MCP, LLM
answer generation, OCR, CSV/XLSX, auth, cloud deployment. The corpus reliability
layer and its observability came first, because retrieval features cannot be
honestly evaluated for a domain with no ground truth.

---

## V6 — Retrieval intelligence (phases 1–8) verification

### Commands and results

| check | command | result |
|---|---|---|
| backend tests (chunked) | `.venv/Scripts/python.exe -m pytest tests/... -q` | **302 passed, 0 failed** (89 + 78 + 77 + 58) |
| new V6 retrieval tests | `pytest tests/test_retrieval_v6.py -q` | 58 passed in ~10 s (hermetic, no Qdrant, no model download) |
| new V6 API tests | `pytest tests/test_retrieval_api_v6.py -q` | 16 passed in ~7 s (upload → index → retrieve → evaluate) |
| bytecode compile | `python -m compileall -q app` | exit 0 |
| frontend typecheck | `npx tsc --noEmit` | exit 0, no output |
| frozen benchmarks | `git diff --stat HEAD -- benchmarks/` | empty (untouched) |
| Qdrant safety | collection list before/after the full suite | 14 → 14; **no test collection created** |

### Test isolation (the V5 lesson, applied again)

`RetrievalService` resolves the vector store through the factory MODULE attribute
at call time, so `test_retrieval_v6.py` and `test_retrieval_api_v6.py` monkeypatch
`app.services.vector_store.factory.create_vector_store` and route every vector
operation to the in-memory cosine store. The API fixture additionally stubs
`routes_build` / `routes_documents` / the embedding factory, so an entire
upload → chunk → embed → index → retrieve → evaluate cycle runs with no services.

### Real-model evidence (not part of the suite)

`CrossEncoderReranker` with `cross-encoder/ms-marco-MiniLM-L-6-v2` was exercised
once against the live download path and applied successfully (the model is now
cached under `~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2`).
The committed tests do NOT depend on that: they cover the applied path with a fake
reranker and the fallback path by forcing the reranker unavailable, so the suite is
deterministic and offline-safe.

### Bugs found and fixed by this phase

1. **`get_retriever` broke every legacy 3-argument factory** by unconditionally
   forwarding `repo=`/`settings=` keywords. Extra dependencies are now forwarded
   only to factories that declare them (signature inspection), so the documented
   V4 registration contract still works verbatim.
2. **The BM25 tokenizer split accented words** (`Métacentre` → `m` + `tacentre`)
   because the character class was ASCII-only. It is now Unicode-aware, so European
   technical text tokenizes correctly.
3. **An unknown retrieval strategy returned HTTP 503** (a service-outage status)
   because configuration errors were raised as `RetrievalError`. There is now a
   distinct `RetrievalConfigError` mapped to **400**: a bad configuration is the
   caller's problem, not an outage.
4. **`score` was ambiguous for BM25**: raw BM25 is unbounded, so a `[0,1]`
   `min_score` would not be comparable across corpora. BM25 now reports a normalized
   score (with the raw value kept in `score_breakdown`) and says so in a note.
5. **Weights that did not sum to 1 silently changed the fused scale** (0.65/0.35
   applied as-is capped every score). Weights now keep their ratio and the APPLIED
   values are recorded with the run.
6. **An empty lexical pool capped every fused score at the dense weight** (0.65),
   which reads as "poor relevance" when the truth is "one retriever found nothing".
   Weights are now rescaled to the surviving side and the response explains why.
7. **The test-double vector store carried a thinner payload than real Qdrant**, so
   dense-path provenance could regress unnoticed. The fake now mirrors every key the
   real store writes.

### Explicitly NOT done in this phase

Query processing, grounded answering, citation validation, evidence gating, chat API
and UI, Retrieval Lab V2, strategy-comparison experiment runner, TurboVec, MCP, OCR,
CSV/XLSX, auth, cloud deployment. Retrieval quality for Naval Architecture is still
**unmeasured** (`ground_truth_status = NOT_AVAILABLE`) — the four strategies can now
be scored, but only on a benchmark that has real human-authored ground truth.

## V7 — Grounded answer engine verification

### Commands and results

| check | command | result |
|---|---|---|
| backend tests (chunked) | `.venv/Scripts/python.exe -m pytest tests/... -q` (4 chunks) | **414 passed, 0 failed** = 318 pre-V7 + 96 new V7 |
| new V7 unit tests | `pytest tests/test_answering_v7.py -q` | 73 passed (query, evidence, gate, generators, citations, injection, trace) |
| new V7 API tests | `pytest tests/test_answer_api_v7.py -q` | 23 passed (success, abstention, errors, fetch, backwards-compat) |
| bytecode compile | `python -m compileall -q app` | exit 0 |
| frontend typecheck | `npx tsc --noEmit` | exit 0, no output |
| frontend build | `npm run build` | exit 0, **15 routes** (14 + `/knowledge-bases/[id]/answer`) |
| frozen benchmarks | `git diff --stat HEAD -- benchmarks/` | empty (untouched) |
| Qdrant safety | collection list before/after V7 + live smoke | 14 → 14; temp smoke KB created then auto-deleted |

### Chunked test commands (full suite exceeds one 600 s window)

```bash
cd backend
./.venv/Scripts/python.exe -m pytest tests/test_answering_v7.py tests/test_answer_api_v7.py tests/test_retrieval_v6.py tests/test_retrieval_api_v6.py -q   # 170
./.venv/Scripts/python.exe -m pytest tests/test_api.py tests/test_benchmark_lifecycle.py tests/test_chunking.py tests/test_content_scorer.py tests/test_corpus_api.py tests/test_corpus_engine.py tests/test_documents_api.py -q   # 132
./.venv/Scripts/python.exe -m pytest tests/test_incremental_documents.py tests/test_llm_and_embeddings.py tests/test_metrics.py tests/test_repo_and_discovery.py tests/test_retrieval_integration.py tests/test_schemas.py -q   # 58
./.venv/Scripts/python.exe -m pytest tests/test_source_quality.py tests/test_urls_and_hashing.py tests/test_user_ingestion.py tests/test_v3_infrastructure.py -q   # 54
```

### Live smoke test (dedicated temp KB, auto-deleted)

Against the running backend + real Qdrant, using a KB created only for the smoke
test (`V7 SMOKE TEMP KB`):

| step | result |
|---|---|
| create KB + upload + index markdown (2 chunks, 2 vectors) | HTTP 201, `indexed: true` |
| `POST /answer` "What is metacentric height?" (dense) | 200, `grounded`, 1 claim / 1 citation / 2 evidence, `ANSWER/SUFFICIENT`, `retrieval_run_id` + `answer_trace_id` present, claim `supported` via `lexical_overlap` |
| `POST /answer` off-domain EASA/turbofan question | 200, `abstained`, `LOW_LEXICAL_ALIGNMENT`, **0 citations** |
| `POST /answer` hybrid strategy | 200, `grounded` (registry path, no answer-side branching) |
| `GET /answer-traces/{id}` | 200, all 7 canonical stages `ok`, query plan + strategy + run + raw generator text present |
| `GET /answers/{id}` | 200, answer refetched |
| delete temp KB | 204; Qdrant back to **14** collections, temp collection removed |

### Bugs found and fixed by this phase

1. **`Answer.generated_by` / `Answer.model` were silently empty.** The validation
   pipeline passed `generator_name=` / `generator_model=` but the `Answer` fields
   are `generated_by` / `model`; Pydantic drops unknown extras, so metadata vanished
   without an error. `_build()` now translates the names explicitly.
2. **`AnswerTrace.query_plan` was never assigned** — the trace stored `None` for
   the query plan even though the stage ran. The service now records the plan on
   the recorder; caught by the API trace test.
3. **`is_mock` lied when wrapping the mock provider.** `LLMAnswerGenerator.is_mock`
   was a class-level `False` even when its provider was `MockLLMProvider`. It is now
   instance-level and mirrors the wrapped provider, so the UI pill is honest.
4. **Multi-part query classification misfired** on any "…and…?" sentence (a
   comparison question became `multi-part`, `Why does X…` became
   `troubleshooting`). The rules now require two interrogative clauses (or two
   question marks) for multi-part, and troubleshooting matches failure/repair
   vocabulary only — `why` alone maps to `explanation`.
5. **Aspect coverage used a 0.15 floor**, so a multi-part question whose clause
   shared one incidental word with the evidence counted as "covered" (a partial
   question returned `ANSWER`). Clause coverage now requires a strict majority
   (`ASPECT_MIN_COVERAGE = 0.5`).
6. **Wrong-KB citations could not be rejected** because `CitationValidator` had no
   KB to compare against; it now takes `kb_id` and rejects evidence recorded under
   another knowledge base.

### Explicitly NOT done in V7

Semantic/entailment validation (labelled `NOT PERFORMED` everywhere), answer-level
benchmark or any answer-quality metric, query decomposition / multi-query retrieval /
HyDE, evidence reranking for generation, calculation tools (`CalculationTrace`
is schema-only and always `performed=false`), answer modes beyond
`grounded`/`abstain_if_unsupported`, streaming, Retrieval Lab
fusion/rerank parameter controls, TurboVec, MCP, OCR, CSV/XLSX, auth, cloud
deployment. Naval Architecture retrieval **and answer quality remain unmeasured**
(`ground_truth_status = NOT_AVAILABLE`); no evaluation number for either was
produced or displayed.

---

## V7.2 — Grounded knowledge assistant

### Test counts

| Suite | Tests |
|---|---|
| V1–V5 (pre-existing) | 244 |
| V6 retrieval (pre-existing) | 74 |
| V7 answering (previous session) | 96 |
| **V7.2 units** `tests/test_answering_v7_2.py` | **65** |
| **V7.2 chat API** `tests/test_chat_api_v7.py` | **34** |
| **Total** | **511 passed, 0 failed** |

Run (two chunks; the full suite exceeds a single command timeout):

```bash
cd backend
./.venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/test_retrieval_integration.py -p no:randomly
./.venv/Scripts/python.exe -m pytest tests/test_retrieval_integration.py -q -p no:randomly
./.venv/Scripts/python.exe -m compileall -q app tests        # exit 0
```

`-p no:randomly` is required: with random ordering a pre-existing registry leak
(bug 7 below) made the failures non-deterministic.

### Frontend

```bash
cd frontend
npx tsc --noEmit     # clean
npm run build        # 16 routes, exit 0
```

### Live chat smoke

```bash
cd backend
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8012
./.venv/Scripts/python.exe scripts/smoke_chat_v7.py http://localhost:8012
```

Creates its own scratch KB, indexes a real document through the real upload →
parse → chunk → embed → Qdrant path, then checks:

* grounded answer with `ANSWERED` state, citations, full provenance;
* `query_trace` proves no rewrite (`rewritten_query: null`, no transformations);
* abstention on an off-domain question, with generation **skipped**;
* conversation memory: follow-up recorded its `resolved_from`, history never used
  as knowledge;
* prompt injection: marker surfaced as a warning, system prompt not echoed, no
  API key / env / path leaked;
* answer runs: measured latencies, every version string, unmeasured latency `None`;
* conversational turn flagged and not answered from evidence;
* scratch KB deleted; Qdrant returns to its starting collection count.

**Result: all checks green.** Qdrant 14 → 14.

`scripts/diagnose_retrieval_v7.py` is the read-only diagnostic used to isolate
bug 2 below; it also creates and deletes only its own scratch KB.

### Bugs found and fixed in V7.2

1. **Registry leak broke the suite depending on test order** (8 failures).
   `tests/test_incremental_documents.py` replaced the real `bm25` retriever in the
   process-global registry and never restored it, so `test_retrieval_v6.py` saw a
   stub. Added `snapshot_retrievers()` / `restore_retrievers()` to
   `app/services/retrieval/retriever.py`; both polluting tests now restore in a
   `finally` block.
2. **The mock provider answered a stability question with a propulsion sentence.**
   `_answer_draft` selected sentences with `len(term) > 3` substring matching,
   which dropped the acronym "GM" and kept the stop-word "when" — so the only
   matching sentence came from the unrelated Propulsion section. It now uses the
   stop-word-aware `extract_terms` and ranks by term-hit count. Regression tests:
   `test_mock_provider_quotes_the_relevant_chunk_not_a_stopword_match` and
   `test_mock_provider_abstains_when_no_sentence_is_relevant`.
   *(Retrieval itself was correct — dense ranked the stability chunk first at
   0.4965 vs 0.0677. This was a generation-side selection bug.)*
3. **A claim citing one valid and one fabricated evidence id was reported as fully
   supported.** It is now downgraded to `partially_supported` with the dropped ids
   named.
4. **Under `policy=downgrade` the action log recorded `remove_claim`** although the
   claim was kept. The log now records the action actually taken.
5. **`_nature_of` reported "who are you" as *underspecified*** rather than
   *non-knowledge*, because the content-term check ran first and that phrase is
   all stop-words.
6. **The underspecified check counted non-topic words**, so "What about the
   previous case?" looked specific. `_NON_TOPIC` exclusion added; two-letter
   acronyms (GM, KG) are no longer filtered by length, so "What is GM?" is
   correctly treated as a real question.
7. **The gate short-circuited underspecified turns**, masking the true
   `NO_EVIDENCE` reason on an empty KB. Only conversational and non-knowledge
   turns short-circuit now.

### Explicitly NOT done in V7.2

Semantic/entailment validation, answer-level benchmark or any answer-quality
metric, query decomposition, query expansion in the UI (programmatic
`expand_query` only), streaming answers, regeneration UI, multi-turn answer reuse,
conversation history as a retrievable source, prompt-injection *filtering*
(detection is a heuristic marker scan), TurboVec, MCP, OCR, CSV/XLSX, auth, cloud
deployment, multi-user permissions, agent tool execution.

Naval Architecture answer quality **remains unmeasured**
(`ground_truth_status = NOT_AVAILABLE`); the smoke test demonstrates grounded,
cited behaviour but is not a quality score.

---

# V8 — Answer-quality evaluation (verified 2026-10-03)

## Checks

| Check | Baseline (pre-V8) | After V8 |
|---|---|---|
| Backend tests | 513 passed | **596 passed**, 0 failed |
| New V8 tests | — | 63 unit + 20 API |
| `compileall` | exit 0 | exit 0 |
| `npx tsc --noEmit` | clean | clean |
| `npm run build` | 16 routes | **17 routes** |
| Frozen benchmark byte-identical | exit 0 | **exit 0** |
| Qdrant collections | 14 | **14** (before and after) |

The frozen `benchmarks/automobile-engineering-baseline-v1.json` was **not
modified**. `git diff --exit-code HEAD` returns 0.

## Measured run

28 questions, dense, evaluator `deterministic-evidence v8.2`, entailment
`heuristic-lexical-coverage`, generator **mock**:

| Metric | Value | Metric | Value |
|---|---|---|---|
| `pass_rate` | 0.821 | `abstention_accuracy` | 1.000 |
| `citation_recall` | 0.821 | `grounding_state_accuracy` | 1.000 |
| `citation_precision` | 0.286 *(lower bound)* | `unsupported_claim_rate` | 0.000 |
| `evidence_support_rate` | 1.000 | `contradiction_rate` | 0.000 |
| `retrieval_hit_rate` | 0.964 | `hallucination_rate` | 0.179 |
| `correctness` | **UNKNOWN** | `key_point_recall` | **UNKNOWN** |

Failures: 5 of 28 — 4 citation failures, 1 retrieval miss.

## What is NOT verified

- **Factual correctness of any answer.** UNKNOWN for all 28 questions.
- **Abstention on unanswerable questions.** The benchmark has 0 such questions.
- **Model-based entailment.** The heuristic is the default.

## Data impact

Four evaluation runs were executed against the real KB `kb_f278c283c748`
during verification, writing **112 `answers` rows**. Qdrant collection count
returned to 14 and no scratch knowledge bases remain. This is the documented
behaviour of `persist=false` (it gates the evaluation-run row only) and is now
stated in the OpenAPI description and in `run_real_answer_eval_v8.py`.

## Bugs found and fixed during V8

1. Polarity/negation conflicts undetected despite high lexical overlap.
2. `list_chunks` 500-default truncated an 812-chunk corpus.
3. `Field(...)` used on FastAPI query params.
4. Broken `BaseModel if False else object` placeholder.
5. Non-existent `chat_versions` import + dead call.
6. Windows cp1252 `UnicodeEncodeError` printing `↑` from real corpus text.
7. Benchmark path guard rejected `DATA_DIR`, failing 7 API tests.
8. **`passed` not serialized** — the UI showed 28/28 failures while `pass_rate`
   reported 0.821. Now a `@computed_field`, with tests asserting consistency.
9. CORS hardcoded to port 3000, ignoring `FRONTEND_URL`.
10. `persist=false` implied read-only but is not; now documented.

## Evaluator semantics correction

The evaluator initially enforced "cite all required evidence" **per claim**,
which drove the real-corpus `pass_rate` to `0.000` and asserted that
cited-but-unlabelled chunks "do not address this question" — ground truth the
benchmark does not have. Corrected to enforce completeness **per answer** and
report unlabelled citations as observations. `pass_rate` 0.000 → 0.821 with
**no benchmark label changed**. 8 tests lock the corrected behaviour.

## V8 continuation — answer evaluation & reliability (2026-10-06)

Audit-first extension against the 20-step V8 continuation spec. No frozen
artifact was modified; all experiments ran against real (read-only for
corpus/Qdrant) or scratch resources.

### Commands and results

```bash
# backend unit + API suite (after the extension)
cd backend && ./.venv/Scripts/python.exe -m pytest -q
# → 693 passed, 4 warnings in 333.77s

# targeted V8 continuation files during development
cd backend && python -m pytest tests/test_answer_eval_v8.py \
  tests/test_answer_eval_api_v8.py tests/test_answer_relevance_v8.py \
  tests/test_answer_claim_states_v8.py tests/test_answer_completeness_v8.py \
  tests/test_answer_benchmark_lifecycle_v8.py \
  tests/test_answer_evaluator_abstractions_v8.py -q
# → 180 passed

# frontend typecheck + build
cd frontend && npx tsc --noEmit   # → clean
cd frontend && npm run build      # → clean; reliability + answer-quality routes built

# live scratch-KB smoke (STEP 17/18)
cd backend && python scripts/smoke_answer_eval_v8.py http://127.0.0.1:8013
# → SMOKE PASSED; Qdrant 14 → 14; scratch KB deleted

# answer-evaluation-v1 experiment (STEP 12)
cd backend && python -u scripts/run_answer_evaluation_v1.py http://127.0.0.1:8013
# → all comparisons COMPARABLE; Qdrant unchanged; results artifact written
```

### New tests (97)

| file | tests | covers |
|---|---|---|
| `test_answer_relevance_v8.py` | 13 | irrelevant-but-grounded detection, abstention exemption, plain-fallback labelling, close calls |
| `test_answer_claim_states_v8.py` | 17 | five-state claims, ratios, fabricated/unsupported citation rates |
| `test_answer_completeness_v8.py` | 11 | key-point coverage, reference similarity, correctness stays UNKNOWN |
| `test_answer_benchmark_lifecycle_v8.py` | 16 | lifecycle defaults, official gate, review metadata, fingerprint stability |
| `test_answer_evaluator_abstractions_v8.py` | 19 | human/LLM evaluators, malformed output, unavailability, provider metadata, reproducibility, mock labelling |
| `test_answer_eval_api_v8.py` (grew 20 → 36) | +16 | official gating, reviews append-only, human-evaluation derivation, INCONCLUSIVE/zero-overlap compare, top-level endpoints |
| `test_answer_eval_v8.py` (grew 63 → 68) | +5 | intersection comparison plan; strict contract unchanged |

### Live experiment (answer-evaluation-v1)

See [`answer-evaluation-v1.md`](./answer-evaluation-v1.md) for the full
report. Headline: hybrid `pass_rate` 0.893 vs dense 0.821 vs bm25 0.786 on the
identical 28-question subset; all three pairwise comparisons COMPARABLE;
relevance measured with corpus IDF weights (`n_docs=812`); fabricated
citations 0.000 (measured); correctness UNKNOWN (no human reference answers).
Non-official by design — the source benchmark is lifecycle `draft`.

### Data impact

* Qdrant: **14 → 14**, no vectors written or deleted.
* `answer_evaluation_runs`: 6 → 15 (all on `kb_f278c283c748`): 3 from the
  first experiment execution, 3 from the re-run under the fixed backend, 1
  crashed-attempt dense row, 2 human-evaluation derivations (one via UI,
  one via API to re-verify the label fix), plus pre-V8 rows.
* `answers`: 280 → 482 (re-run experiment 84 + first execution 84 + crashed
  attempt 28 + scratch smoke ~6).
* `answer_reviews`: **1** — a browser round-trip verification review on
  `aerun_4d474005c78c` / `auto-eng-011` (reviewer `v8-ui-verification`), used
  to verify the review→derive workflow end-to-end in the UI. Append-only; no
  existing review was modified.
* Frozen artifacts: **0 bytes written**.

### Frozen artifact integrity

```bash
git diff --exit-code HEAD -- benchmarks/automobile-engineering-baseline-v1.json \
  benchmarks/automobile-engineering-baseline-v1-results.json \
  benchmarks/answer-quality-automobile-v1.json benchmarks/source-selection-experiment-v1
# → exit 0 (no changes)
```

### Live UI round-trip (browser-verified)

Exercised the review workflow through the real UI against the real API
(no hermetic stubs):

1. Opened *Human review* on the failing `auto-eng-011` card — form renders
   with reviewer (required), 6 verdicts, 9 structured labels, notes; the
   *Store review* button is disabled while reviewer is blank.
2. Stored a review (reviewer `v8-ui-verification`, verdict `correct`, note)
   → count went 0 → 1 and the record rendered back verbatim (append-only).
3. Clicked *Derive human-evaluation run* → new immutable run
   `aerun_9b6d8630396a` with evaluator `human-reviews`, lineage note
   `derived from run aerun_4d474005c78c`, and `correctness` moving from the
   UNKNOWN list (5 → 4 entries) — source run and review untouched.
4. Reliability dashboard renders all runs with UNKNOWN cells shown as
   UNKNOWN (never 0), flags, and per-dimension columns with no combined score.

### Bugs found and fixed in this final pass

1. **Per-question `correctness` label was hard-coded to UNKNOWN**
   (`answer-quality/page.tsx`): the card always printed
   "correctness: UNKNOWN (needs human labels/review)" even when
   `correctness.measured` was true, contradicting the aggregate that had
   already dropped `correctness` from `unknown_metrics`. Now renders
   `1.00 (from 1 human review)` with the policy-mapping reason as tooltip.
   Verified live on the derived human-evaluation run.
2. **Duplicated `key_point_recall` in `unknown_metrics`** (stale server
   process): the backend was started without `--reload` before the
   `run.py` dedupe fix, so every run it recorded carried
   `[..., 'key_point_recall', 'key_point_recall', ...]`. The source was
   already correct; fixed operationally by restarting the backend, and the
   experiment artifact was regenerated under the fixed code (headline
   numbers reproduce identically — see the table above). The UI also
   de-duplicates at render time so historical run rows display honestly.

### Final integrity state

* Frozen artifacts: `git diff --exit-code HEAD` → **exit 0** on all
  `automobile-engineering-baseline-v1*`, `answer-quality-automobile-v1.json`,
  `source-selection-experiment-v*`.
* Qdrant: **14 collections** (unchanged all session).
* DB: 15 answer-evaluation runs / 482 answers / 1 review, all on
  `kb_f278c283c748`; no corpus, chunk, or vector rows touched.
* Frontend: `npx tsc --noEmit` clean, `npm run build` clean, both pages
  browser-verified on :3000 against the restarted backend on :8013.
