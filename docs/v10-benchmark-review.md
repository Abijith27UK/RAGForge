# V10 — Human-Reviewed Benchmark Ground Truth

Date: 2026-10-08 · Branch `Dev_1_midterm`

V10 turns the 28-question Automobile **answer benchmark**
(`benchmarks/answer-quality-automobile-v1.json`) into a trustworthy,
human-reviewed ground-truth instrument — or, more precisely, it builds every
mechanism needed for a human to do that, and refuses to pretend the human work
has been done.

## The problem it solves

As of V9, the answer benchmark was `human_review: pending`, every
`expected_answer` was `null`, and every `key_points` list was empty. The V8
evaluator therefore kept `correctness` **UNKNOWN** for all 28 questions —
correctly. The V9 optimization loop (Phase 13) was blocked on exactly this: no
reviewed ground truth means no measured correctness, which means no candidate
configuration can ever be justified from data.

RAGForge's core rule applies here: *no fabricated labels, ever.* Generating
reference answers with an LLM and calling them ground truth would make every
downstream number meaningless. The only path to a measured `correctness` is a
human writing and reviewing the labels.

## The workflow

```text
QUESTION → CORPUS EVIDENCE → HUMAN REFERENCE ANSWER → KEY POINTS
→ ACCEPTABLE ELEMENTS → PROVENANCE → HUMAN REVIEW → APPROVED → FROZEN
```

1. **QUESTION** — a benchmark question (the shipped artifact's 28, or any
   answer benchmark targeting the same KB).
2. **CORPUS EVIDENCE** — the real chunks that answer it, served in the CENTER
   panel with full provenance (document, section, URL, content hash, text).
   An annotation's evidence is validated against the live corpus: chunk must
   exist, belong to this KB, and its content hash must still match (a
   re-chunked corpus invalidates the label instead of silently grounding it in
   a passage that no longer exists).
3. **HUMAN REFERENCE ANSWER** — authored in the LEFT panel. Required author
   identity; append-only (saving again adds a revision, never overwrites).
4. **KEY POINTS** — one per line; each is a scorable unit for correctness.
5. **ACCEPTABLE ELEMENTS** — alternative phrasings a human judged acceptable.
6. **PROVENANCE** — author, method, note, evidence chunk ids, timestamps, and
   a fingerprint binding the annotation to the exact benchmark bytes
   (`benchmark_fingerprint`).
7. **HUMAN REVIEW** — a *different-capacity step*: a named reviewer assesses
   the annotation across the V9 review dimensions (answerability,
   evidence_sufficiency, reference_answer, key_points, ambiguity — all
   required — plus acceptable_elements). Verdicts: `ok`, `ok_with_note`,
   `needs_revision`, `wrong`, `ambiguous`, `unknown`.
8. **APPROVED** — labels are copied onto the question **only when the review
   outcome is `approved`**. Draft annotations never influence scoring.
9. **FROZEN** — an immutable in-database `AnswerBenchmarkVersion`: the
   benchmark deep-copy marked `FROZEN` + `HUMAN_REVIEWED`, plus three
   fingerprints (benchmark content, ground truth, whole artifact). Corrections
   create a new version; nothing is edited.

### Question states

`PENDING · IN_REVIEW · APPROVED · REJECTED · AMBIGUOUS · INSUFFICIENT_EVIDENCE`

`AMBIGUOUS` and `INSUFFICIENT_EVIDENCE` are **permitted non-scoring states**:
a human has looked at the question and determined it should not score. They
are excluded from evaluation with an explicit warning — never scored as 0,
never silently dropped.

### Approval policy

`v10-answer-benchmark-approval-v1` (recorded in every gate result):

* blocking states: `pending`, `in_review`, `rejected`
* scoring state: `approved` (minimum 1)
* permitted non-scoring: `ambiguous`, `insufficient_evidence`
* key points required for approval

The gate returns **reasons**, naming the offending questions, so a refused
freeze always explains what to do next.

## Where everything lives

| Thing | Location | Why |
|---|---|---|
| Benchmark bytes | `benchmarks/*.json` (git) | **Never modified by V10.** All 13 files remain in `FROZEN_BENCHMARK_FILES` and byte-identical (`git diff HEAD -- benchmarks/` is empty). |
| Annotations (author input) | SQLite `benchmark_ground_truth` | append-only create/list, no update/delete |
| Reviews (reviewer verdicts) | SQLite `benchmark_question_reviews` | append-only; `supersedes` links corrections |
| Frozen versions | SQLite `answer_benchmark_versions` | immutable; tamper-checked |
| Question state labels | copied into the annotation/review views via `apply_question_states` | labels change only through new approved records |

The state lives in SQLite, not in the artifact, because the artifact is the
*instrument* — a reviewed instrument must keep proving its own bytes are the
bytes that were reviewed (`fingerprint f72c30eb5c0c363a` for the shipped one).

## Code map

### Domain layer — `backend/app/services/answer_eval/ground_truth.py` (new)

* `GroundTruthAnnotation` / `GroundTruthLog` — append-only author input with
  evidence + provenance; `validate_annotation` (labels present, non-empty key
  points when required) and `validate_annotation_against_corpus` (chunk
  exists / belongs to KB / hash unchanged).
* `derive_question_state`, `apply_question_states` — state transitions; a
  question's scoring labels appear only after approval.
* `compute_completeness` → `ReviewCompleteness` — every rate is a
  `Measured` tri-state: no denominator ⇒ `value: null, measured: false` with a
  reason, **never 0**.
* `ApprovalPolicy` / `evaluate_approval_gate` — the policy above, with
  per-question blocking reasons.
* `ground_truth_fingerprint` — hashes benchmark labels **and** annotations.
* `build_answer_benchmark_version` — refuses unapproved questions, deep-copies
  the benchmark marked `FROZEN`/`HUMAN_REVIEWED`, computes the artifact
  fingerprint. `verify_answer_benchmark_version` re-derives all fingerprints
  and reports tampering.

### Evaluator — `ReferenceAnswerEvaluator` (`evaluator.py`, factory key `reference`)

The V8 `DeterministicAnswerEvaluator` is **unchanged** — a test pins that it
keeps `correctness` UNKNOWN even with a reference answer. The new evaluator
(`reference-labels`, version `v10.1`, policy `reference-key-point-coverage-v1`):

* correctness = key-point coverage (fallback: reference-term coverage)
* UNKNOWN without reviewed labels; abstention-expected questions not scored;
  no answer → 0.0 **by published policy**, not by accident

The API refuses `evaluator: "reference"` with 400 unless the benchmark
lifecycle is `approved`/`frozen` — draft labels can never be scored as if a
human had approved them, and there is no silent fallback to another evaluator.

### API — `backend/app/api/routes_benchmark_review.py` (new, registered in `main.py`)

```text
GET  /api/knowledge-bases/{kb}/answer-benchmark/review-packet        states + completeness + gate + versions
GET  /api/knowledge-bases/{kb}/answer-benchmark/questions/{qid}      question + REAL corpus evidence + full histories
POST .../questions/{qid}/ground-truth                                append annotation (404 for unknown questions)
POST .../questions/{qid}/reviews                                     append review
GET  .../questions/{qid}/{ground-truth|reviews}                      histories, oldest first
POST .../answer-benchmark/freeze                                     409 + gate reasons when blocked
GET  .../answer-benchmark/versions[/{id}]                            frozen versions
POST .../answer-benchmark/versions/{id}/verify                       tamper check
GET  .../answer-benchmark/completeness                               measured coverage rates
```

`routes_answer_eval.py` gained `benchmark_version_id` on the run request:
the run scores the **immutable frozen version** (verified before the run
starts), records `db:answer-benchmark-versions/{id}` as its benchmark path,
and stores both the version id and artifact fingerprint on the run.
`is_comparable_with` now also compares version ids.

### UI — `/knowledge-bases/[id]/benchmark-review`

Three panels (responsive `lg:grid-cols-3`): **Authoring LEFT** (author,
reference answer, key points, acceptable elements, note) · **Evidence CENTER**
(the chunk exactly as the corpus holds it, with provenance and hash) ·
**Review RIGHT** (reviewer identity, outcome, six dimension verdict grids,
unresolved list). Plus: question navigator `X/28` with per-question state
dots, review completeness + approval policy + gate reasons, and the freeze /
frozen-version verify panel. Nav entry in `AppShell` (ShieldQuestion).

## Verification (2026-10-08)

| Check | Command / method | Result |
|---|---|---|
| Backend suite | `pytest -q -p no:randomly` | **1062 passed, 0 failed** (978 V9 + **84 V10**: 48 ground-truth, 12 correctness-unlock, 24 API) in 276s |
| Frontend typecheck | `npx tsc --noEmit` | clean |
| Frontend build | `npm run build` | clean incl. new route `/knowledge-bases/[id]/benchmark-review` |
| Live write-path smoke | scratch benchmark, real HTTP on :8013 | **24/24 PASS**: author → review → freeze → verify → reference eval; `correctness.measured=true` from frozen labels; draft-file reference run **refused 400**; official benchmark still 28/28 pending with **0** frozen versions |
| Smoke cleanup | scoped SQL deletes of the script's own rows | DB back to baseline: 15 answer-evaluation runs, 482 answers, 0 smoke rows |
| Live UI | browser on :3000 | page renders all panels with real data, navigation works, gate shows named blocking reasons, coverage shows **UNKNOWN (0)** not 0%, console clean |
| Qdrant | collections API before/after | **14 → 14**, **8822 → 8822** points, protected `kb_kb_f278c283c748` **812 → 812** |
| Frozen artifacts | `git diff HEAD -- benchmarks/` | **empty**; SHA-256 of the answer benchmark unchanged |

The live smoke intentionally used a **scratch benchmark**
(`backend/data/smoke-answer-benchmark-v1.json`, gitignored) with two questions
copied from the official artifact, labelled `SMOKE` everywhere. The official
benchmark's review state was verified untouched afterwards. The smoke run's
`correctness` came back **0.0 — measured**: the smoke labels ("[SMOKE] key
point A") are not facts any real answer contains, and the evaluator scored
them honestly rather than rubber-stamping. That is the instrument working.

## What remains — the human work

V10 delivers the workflow, not the labels. Still to do **by a human**:

1. Author reference answers + key points for the 28 questions (LEFT panel,
   question by question, reading the CENTER evidence).
2. Review each one (RIGHT panel) → `approved` (or classify
   `ambiguous` / `insufficient_evidence` with reasons).
3. Freeze the approved benchmark → immutable version.
4. Run `evaluator: "reference"` against the frozen version → **measured**
   `correctness` / `key_point_recall`.
5. Only then is the V9 Phase-13 optimization experiment admissible.

Until steps 1–3 are done, the API and UI will keep reporting
`GATE BLOCKED` with the exact list of pending questions — which is the correct
behaviour, not a bug.

## Known limitations

* The official 28 questions have **0** annotations and **0** reviews; all
  coverage rates are UNKNOWN by design.
* The off-domain gate (V9) is still unmeasurable at n_off = 1.
* V9's Phase-10 failure-analysis UI remains unbuilt; this session built the
  V10 review UI instead (the review UI is the blocker for Phase 13).
* Review dimensions are assessed per question, not per key point.
* `reference` evaluator's fallback to reference-term coverage is lexical
  (deterministic, offline); a model-based judge stays opt-in and refused
  without a provider.
