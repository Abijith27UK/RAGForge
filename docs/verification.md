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
