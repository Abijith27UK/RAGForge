# V9 Audit Report — Evaluation → Optimization Loop

**Date:** 2026-10-06 (re-derived from disk after a process restart; servers were down)
**Branch:** `Dev_1_midterm`
**Scope:** Phase 0 audit before any V9 change. No writes made to source during the audit.

---

## 0. Reference state already established in V8 (unchanged by V9)

| Check | Result |
|---|---|
| Backend tests | **692 passed + 1 skipped** (the skip is the live-Qdrant integration test, which runs and passes when Qdrant is up). Collected total: **693**. Docs saying "596" were stale — see §6. |
| Frontend `tsc --noEmit` | clean |
| Frontend `npm run build` | clean |
| `git diff HEAD -- backend/ frontend/ docs/` | **empty** — no uncommitted source changes |
| Qdrant collections | **14** (`kb_*` + `kb_live_itest` under `qdrant-dl/storage/collections/`) |
| Frozen artifacts | `benchmarks/automobile-engineering-baseline-v1.json` + `-results.json`, `benchmarks/answer-quality-automobile-v1.json`, `benchmarks/source-selection-experiment-*` |
| Existing V9 work | none — V9 source is pristine (branch log: 4 commits, all pre-V9) |

V8 deliverables that V9 stands on:
- evaluator abstraction (`deterministic-evidence` / `human-reviews` / `llm-judge`), append-only `answer_reviews`, benchmark lifecycle `draft→approved→frozen` with `require_official_benchmark` gate, run comparability, real dense/BM25/hybrid evaluation, reliability dashboard, 693-test suite.
- Frozen `benchmarks/automobile-engineering-baseline-v1.json` loads as **DRAFT** / `human_review: PENDING` — it must never be auto-migrated.

---

## 1. Benchmark schema / Automobile artifact (answered from disk)

> **CORRECTION (found while implementing Phase 1).** The first draft of this
> report named the wrong artifact. `benchmarks/automobile-engineering-baseline-v1.json`
> is the **RETRIEVAL** benchmark: its questions carry `expected_chunk_ids` / 
> `expected_keywords` and an `id`, **not** `question_id`, so it cannot be loaded
> as an `AnswerBenchmark` at all (Pydantic rejects all 28 rows with
> `question_id: Field required`). The answer benchmark is
> **`benchmarks/answer-quality-automobile-v1.json`**, which has `question_id`,
> `answerability`, `required_evidence`, `expected_grounding_state` and
> `human_review: pending`.

- `benchmarks/answer-quality-automobile-v1.json` is the **answer benchmark** (28 questions; each carries `question_id`, `answerability`, `required_evidence` with `content_hash`, and quoted `provenance`). It has no `lifecycle` key, so it loads as **DRAFT**.
- **Zero** human-authored `expected_answer` / `key_points` / `acceptable_answer_elements` across all 28; `human_review: PENDING`; no per-question `difficulty`/`reviewer`/`review_status`; no `lifecycle` field.
- Author block: `"method": manual chunk selection by reading indexed chunk texts; "author": "Buffy (Codebuff agent), operating under explicit human instruction"; "human_review": "PENDING - ... a human must review before treating results as publication-grade"`.
- The module docstrings in `backend/app/services/answer_eval/benchmark.py` state the design: ground truth is **evidence-based, not answer-based**; `correctness` and `key_point_recall` are engineered to be `UNKNOWN` without human labels.
- The loader (`load_answer_benchmark`) and corpus validator (`validate_against_corpus`) reject a benchmark whose required chunks no longer exist / hashes changed — the integrity gate V9's `require_official_benchmark` reuses.

---

## 2. Answer-evaluation result artifact

- One real run record exists in the database: `aerun_4d474005c78c` on KB `kb_f278c283c748`.
- The in-memory run carries per-question results as `Measured` triples with a `summary()` that surfaces `unknown_metrics` and `warnings`, per-question claim verdicts, relevance provenance, and the evaluator/entailment/relevance producer footprints.
- Persistence is JSON: `answer_evaluation_runs(id, kb_id, created_at, benchmark_name, benchmark_fingerprint, data)` and `answer_reviews(id, kb_id, run_id, question_id, answer_id, reviewer, verdict, created_at, data)`.
- V9 does **not** rely on the DB shape: all experimental result artifacts are written as their own JSON files under `benchmarks/` (Phase 4 approval gate blocks any write into `benchmarks/` except by the experiment runner).

---

## 3. Retrieval strategy configuration persistence

- `KnowledgeBase` persists `retrieval_config` (an embedded JSON blob; the `retrieval_configs` table keys on `kb_id`).
- `RetrievalService.resolve_params()` merges stored config → request overrides → defaults, so a **candidate configuration** is a first-class persisted object, and nothing in the optimizer is allowed to alter the stored production config (see the Phase 5 recommendation engine design).

---

## 4. Run comparison logic

- `AnswerEvaluationRun.is_comparable_with()` — strict: same question subset **and** same benchmark fingerprint.
- `compare_with()` returns `COMPARABLE` / `INCONCLUSIVE` (intersection over shared questions only) / `NOT_COMPARABLE` (no overlap or changed benchmark content).

---

## 5. Chat answer-generation pipeline

- `AnsweringService.answer()` → `QueryProcessor` → retrieval → evidence assembly → gate → generator → `AnswerValidation` → `Answer`. Writes `answer_runs`, `answer_traces`, and `answers`.
- Answer evaluation re-uses this path: each benchmark question is answered through the **real** ` AnsweringService` and then scored.

---

## 6. V8 test count (actual, not trustulated)

- **693 tests collected**; **692 passed + 1 skipped** with Qdrant down, **693 passed** with Qdrant up. This supersedes every "596" figure in earlier docs.
- The answer-eval V8 subset is 180 (API=36, core=68, claim states=17, completeness=11, relevance=13, lifecycle=16, evaluator abstractions=19).
- Note for `grep`-based counting: many V7-era files define tests inside classes, so `grep -c '^def test_'` returns 0 for them and a from-grep count is unreliable — use `pytest --collect-only`.
- Qdrant: **14 collections** confirmed by directory listing.
- Frozen artifacts: **byte-identical** to HEAD (`git diff` empty).
- Database: **15 answer-evaluation runs**, **1 review** (`areview_d486ba101f97`, reviewer `v8-ui-verification`, verdict `correct` on `auto-eng-011`, run `aerun_4d474005c78c`).

---

## 7. Discrepancy accepted (and deliberately left untouched)

- The V8 spec claimed the benchmark was "HUMAN REVIEWED"; the artifact itself says `human_review: PENDING` and the code treats a missing `lifecycle` as `DRAFT` and refuses `official:true` runs until `FROZEN`.
- V9 does **not** auto-migrate or rewrite the frozen artifact. The reviewer must author and review questions through the UI (Phase 1), then freeze, and only then produce official results.

---

## 8. Immediate conclusion for V9

The Phase-0 deliverables are satisfied by the on-disk state above. V9 proceeds Phase-by-Phase. Known constraints that shape the V9 design:

1. **The shared result tables are JSON blobs** (`answer_evaluation_runs`, `evaluation_questions`, `evaluation_runs`), so experimental result artifacts are written to their own `benchmarks/` files, never mixed into the shared run schema.
2. **No ground-truth answers exist** → `correctness`/`key_point_recall` remain `UNKNOWN` unless the reviewer authors reference answers through the Phase-1 review workflow.
3. **Only FROZEN question snapshots are official** → any V9 experiment that claims to be an official answer-quality experiment must score the frozen snapshot, not live rows.
4. **Producers are recorded per run** (evaluator + version, entailment provider + model, relevance method + weight source) → every V9 experiment result artifact records them identically so comparisons are trustworthy.
