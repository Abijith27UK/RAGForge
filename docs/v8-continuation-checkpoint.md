# V8 Continuation Checkpoint — Answer-Quality Evaluation

**Date:** 2026-10-03
**Branch:** `Dev_1_midterm`
**Scope:** Make RAGForge able to measure whether its grounded answers are
correct, supported, correctly cited, complete, and appropriately abstained.

---

## 1. Headline

RAGForge can now **measure** citation correctness, claim support, grounding
state and abstention behaviour on a real 28-question corpus, and can **prove
what it cannot measure**: factual correctness is UNKNOWN for all 28 questions
because no human-authored reference answer exists.

**Answer quality is not solved.** What exists is a measurement apparatus with
an honest boundary around it.

---

## 2. Baseline (before any V8 change)

| Check | Result |
|---|---|
| Backend tests | **513 passed**, 0 failed |
| `compileall` | exit 0 |
| `npx tsc --noEmit` | clean |
| `npm run build` | 16 routes |
| `git diff --exit-code HEAD -- benchmarks/automobile-engineering-baseline-v1.json` | exit 0 — **byte-identical** |
| Qdrant collections | **14** |

---

## 3. The finding that shaped everything

The frozen benchmark has `expected_chunk_ids` for all 28 questions but **zero**
expected answers, key points or keywords, and its own
`authorship.human_review` reads `PENDING`.

**Design consequence:** answer-benchmark ground truth is **evidence-based, not
answer-based**. `expected_answer` / `key_points` are left NULL with
`answer_label_status: REQUIRES_HUMAN_AUTHORSHIP`. Metrics that need human-written
answers (`correctness`, `key_point_recall`) are tri-state UNKNOWN, never guessed.

Fuzzy-matching an answer against retrieved text would "work" and would
re-measure retrieval while labelling it answer quality. Not implemented.

---

## 4. Delivered

### Backend — `backend/app/services/answer_eval/`
| Module | Role |
|---|---|
| `benchmark.py` | Schemas, loader, corpus validation (content-hash re-chunk detection), fingerprint |
| `entailment.py` | Entailment ABC, heuristic provider, `LLMCitationEntailment` (raises rather than silently degrading), `polarity_conflict` + 30 antonym pairs |
| `metrics.py` | `Measured` (value **or** reason), `AnswerQualityResult`, `AnswerQualityAggregate`, `compute_final_score`, `grounding_states_compatible` |
| `evaluator.py` | `DeterministicAnswerEvaluator` (v8.1 → **v8.2**), per-claim audit |
| `run.py` | Immutable run records, `is_comparable_with`, aggregation |
| `service.py` | Orchestration, paginated chunk index |

`AnswerQualityResult` **refuses** to carry a `final_score` whose inputs are
unmeasured.

### API — `backend/app/api/routes_answer_eval.py`
5 endpoints, benchmark path guarded to repo + `DATA_DIR`.

### UI — `/knowledge-bases/[id]/answer-quality`
Provenance panel, measured-vs-UNKNOWN metric grid, grounding-state distribution,
expected-vs-actual confusion, and per-question failure cards with a full claim
audit. Citation failures and retrieval misses are separated.

### Docs
`answer-evaluation.md`, `answer-benchmark-design.md`, this file.

---

## 5. Bugs found and fixed

1. **Polarity/negation undetected.** "must be negative" vs "positive" scored 87%
   lexical overlap → `polarity_conflict` + antonym pairs; 3 tests added.
2. **`list_chunks` 500-default truncated** an 812-chunk corpus → paginated
   `_chunk_index`.
3. **`Field(...)` on FastAPI query params** → `Query(...)`.
4. **Broken `BaseModel if False else object`** placeholder in `entailment.py`.
5. **Non-existent `chat_versions` import** + dead call in `service.py`.
6. **Windows cp1252 `UnicodeEncodeError`** printing `↑` from real corpus text →
   stdout reconfigured + ASCII-safe text in the runner.
7. **Benchmark path guard rejected `DATA_DIR`** → 7 API tests failed. Widened to
   repo **or** `DATA_DIR` (both legitimate benchmark locations).
8. **`persist` flag implied read-only; it is not.** It gates only the run row —
   every question still writes an `Answer` row. Documented in OpenAPI and in the
   runner rather than silently left as a trap.
9. **`passed` was not serialized.** A `@property`, so any client filtering on it
   classified **every** question as failed while `pass_rate` said 0.821. The UI
   contradicted itself. Now `@computed_field`, with 2 tests asserting
   `pass_rate == derived from per_question[].passed`.
10. **CORS ignored `FRONTEND_URL`** and hardcoded port 3000 → a dev UI on any
    other port got a bare "Failed to fetch".

---

## 6. The evaluator semantics fix (most important change)

**Symptom:** `pass_rate = 0.000` on the real corpus, with 28/28 questions failing
on messages like *"claim cl_002 does not cite the required evidence chk_…"*.

**Diagnosis — a level-of-judgement bug, not a benchmark problem.**

`required_evidence` is a **necessity** label: a human marked chunks a correct
answer *must* use. The evaluator enforced it **per claim**, which is wrong — a
claim about alternative fuels has no reason to cite the engine-definition chunk.
It also asserted cited-but-unlabelled chunks *"do not address this question"*,
which fabricates ground truth the benchmark does not have.

**Fix:** completeness enforced **per answer** (strict, legitimate); unlabelled
citations reported as an **observation** (warning), never a failure. Both
signals stay recorded on every claim verdict for failure analysis.

**Result: `pass_rate` 0.000 → 0.821.** No benchmark label was touched, and 8
tests lock the corrected semantics.

This was a **measurement-correctness** fix, not a tuning knob. The benchmark
labels are the same file, byte for byte.

---

## 7. Measured result (28 questions, dense, mock generator)

| Metric | Value | Metric | Value |
|---|---|---|---|
| `pass_rate` | 0.821 | `abstention_accuracy` | 1.000 |
| `citation_recall` | 0.821 | `grounding_state_accuracy` | 1.000 |
| `citation_precision` | 0.286 *(lower bound)* | `unsupported_claim_rate` | 0.000 |
| `evidence_support_rate` | 1.000 | `contradiction_rate` | 0.000 |
| `retrieval_hit_rate` | 0.964 | `hallucination_rate` | 0.179 |
| `correctness` | **UNKNOWN** | `key_point_recall` | **UNKNOWN** |

Failures: 5 of 28 — 4 citation failures, 1 retrieval miss (`auto-eng-015`).

---

## 8. Defect found and deliberately NOT fixed

`ExtractiveMockAnswerGenerator` walks evidence in rank order and stops at a
5-claim budget (~2 sentences/chunk). With `top_k=5`, **evidence ranked 4th or
lower is structurally uncitable**.

Evidence: in 4 of 5 failures the required chunk **was retrieved at rank 5** and
never cited, while the top 3 were.

**Not fixed here.** Changing the generator's selection strategy and then
re-reporting the metric it affects is exactly how a benchmark gets tuned to look
good. It is reported here, and should be fixed as its own change followed by a
re-measurement.

---

## 9. Tests

| File | Count |
|---|---|
| `backend/tests/test_answer_eval_v8.py` | 63 |
| `backend/tests/test_answer_eval_api_v8.py` | 20 |
| **Full suite** | **596 passed, 0 failed** |

Hermetic: no network, no API key, no Qdrant, no real LLM.

---

## 10. Limitations (unchanged or introduced)

1. **Correctness is unmeasurable** without human-authored reference answers.
2. **0 unanswerable questions** in the shipped benchmark, so
   `false_unsupported_rate` cannot be measured on the real corpus.
3. **Entailment is heuristic** lexical coverage — no model-based judge is active.
   `entailment_is_model_based` is recorded on every result so a heuristic
   verdict is never mistaken for a model's.
4. **The generator is a mock.** `temperature=0.2` with no recorded seed means LLM
   answers are **not** deterministically replayable; the mock provider is.
5. **Every V6–V8 change is uncommitted.**

---

## 11. Recommended next step

**Author human reference answers for a 10-question subset** (plus ~5 genuinely
unanswerable questions). That is the single change that converts `correctness`
and `key_point_recall` from UNKNOWN to measured, and it is the only thing
blocking a real answer-quality score. Fixing the generator's rank-truncation is
the natural parallel change.


---

## 12. Continuation session (2026-10-06) — 20-step V8 spec delivered

Audit-first extension of everything above. Full details:
[`answer-evaluation-architecture.md`](./answer-evaluation-architecture.md) and
[`answer-evaluation-v1.md`](./answer-evaluation-v1.md).

Delivered beyond this checkpoint's original scope:

* **STEP 8 relevance**: IDF-weighted question-term coverage, chosen from a
  measurement (plain coverage is *inverted* on known off-domain answers:
  0.400 vs on-domain min 0.333). Threshold 0.30 with a documented 0.04
  margin; abstentions exempt; plain fallback warns instead of failing.
* **STEP 6/7 claims & citations**: five-state `evaluated_state`, claim
  ratios over all claims, `fabricated_citation_rate`,
  `unsupported_citation_rate` (judged-only denominator).
* **STEP 2/F completeness**: `key_point_recall` /
  `expected_information_coverage` + `reference_answer_similarity` when human
  labels exist; `correctness` still needs a human or an explicitly
  model-based judge.
* **STEP 3 lifecycle**: `draft → approved → frozen`, `official` runs refused
  unless FROZEN; review metadata (`difficulty`/`reviewer`/`review_status`)
  validated, never backfilled; fingerprint unchanged by metadata.
* **STEP 4 reviews**: append-only, attributed `answer_reviews` table +
  POST/GET endpoints; derivation pass (`/human-evaluation`) produces a NEW
  run with review-based `correctness` — source run and reviews untouched.
* **STEP 5/14 evaluators**: deterministic / human / LLM behind one ABC;
  explicit opt-in only; LLM judge stores model+prompt+`judge_raw`,
  degrades to UNKNOWN on failure or malformed output, never ground truth.
* **STEP 10/11**: top-level `/api/answer-evaluation-runs`; compare verdicts
  `COMPARABLE` / `INCONCLUSIVE` (intersection only) / `NOT_COMPARABLE`
  (zero overlap or changed content) — the strict refusal is unchanged.
* **STEP 12 experiment**: `benchmarks/answer-evaluation-v1.json` (spec),
  `backend/scripts/run_answer_evaluation_v1.py` (runner),
  `benchmarks/answer-evaluation-v1-results.json` (real results: hybrid
  pass_rate 0.893 > dense 0.821 > bm25 0.786; all comparisons COMPARABLE;
  Qdrant 14→14; **official=false** because the source benchmark is `draft`).
* **STEP 13-16 UI**: answer-quality page extended (claim states, relevance +
  method labels, completeness, review form/history, derive-human-run,
  lifecycle/official badges) + new Reliability dashboard page with
  **no combined score**.
* **STEP 17/18/19**: 97 new tests (suite 596 → **693 passed**), frontend
  tsc + build clean, live scratch smoke passed (Qdrant 14→14, scratch
  cleaned), data impact and frozen-artifact integrity reported.

Discrepancies surfaced rather than accepted: the spec's "513 tests" was
already 596 before this session; the spec's "HUMAN REVIEWED" claim conflicts
with the answer benchmark artifact's own `authorship.human_review: PENDING`
(left untouched; the artifact therefore loads as `draft` and cannot be used
for official runs).

Recommended next step is unchanged from §11 — human reference answers — but
the **review workflow (STEP 4) now exists**, so authoring them has a home:
reviews can be recorded against real runs and converted into measured
`correctness` via the derivation pass.