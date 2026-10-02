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
