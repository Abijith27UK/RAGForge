# V11 — Ground-Truth Assist: Drafting and Pre-Freeze Benchmark Audit

**Date:** 2026-10-10 · **Benchmark:** `answer-quality-automobile-v1.json` · **Fingerprint:** `f72c30eb5c0c363a`

V10 built the complete authoring → provenance → review → approval → freeze →
reference-evaluator workflow, and left the official answer benchmark with
**28 pending questions and 0 authored labels**. That is not an implementation
gap: a human has to author the reference answers, and nobody may do it on the
human's behalf.

V11 attacks the *cost* of that human work, and nothing else. It adds two
read-only-to-ground-truth capabilities:

| Capability | File | What it can do | What it can never do |
|---|---|---|---|
| Draft assist | `backend/app/services/answer_eval/drafting.py` | assemble a starting point from **verbatim** corpus sentences and show how much of the question the pinned evidence covers | author, approve, or score ground truth |
| Pre-freeze audit | `backend/app/services/answer_eval/quality_audit.py` | report problems in the benchmark before it is frozen | repair a benchmark problem silently |

Both were verified live against the real 28-question benchmark on the protected
KB `kb_f278c283c748` (812 points). No human labels were invented. The benchmark
is still `28 pending / 0 authored / 0 approved` after this phase, and the freeze
gate still refuses with a 409 — which is the correct outcome.

## Why drafting instead of "AI answers"

The research-integrity rule forbids a machine draft becoming ground truth, and
also forbids outside knowledge silently filling corpus gaps. An LLM asked to
"write the reference answers" violates both: it is fluent precisely where the
corpus is thin, so the human reviewer cannot tell which sentences are grounded.

The drafting engine is therefore **extractive only**:

* it selects whole sentences from the chunks the question already pins,
* it never paraphrases, summarizes, or generates — every character of a draft
  occurs in the corpus,
* it never consults a model or a network,
* every key point carries `source_chunk_id` and the sentence index inside it,
* `is_draft = true`, `requires_human_review = true`, and a fixed
  `label = "DRAFT — MACHINE-GENERATED SUGGESTION, NOT HUMAN GROUND TRUTH"`.

A draft that no sentence supports is reported as `insufficient_evidence` rather
than padded with plausible text.

### How a draft is selected

1. The question's content terms are weighted by corpus IDF (the same lexical
   instrument already validated in V9 `relevance.py`), so rare domain terms
   count for more than `use`/`make`/`type`.
2. Each pinned evidence chunk is split into sentences; layout artifacts are
   normalized (the corpus text is markup-flattened, e.g. `energy\n.`) so
   punctuation-joined text does not defeat matching.
3. Sentences are scored by the IDF weight of the question terms they contain.
4. The best-supported sentence per evidence chunk forms the draft, capped at a
   small number of key points.
5. Coverage = share of the question's IDF-weighted content terms present in the
   pinned evidence. Below a 45% triage line the draft is `partial` (with the
   missing terms named); if no sentence clears a 20% floor it is
   `insufficient_evidence`.

The triage line is a **judgement calibrated on the shipped 28-question
benchmark, not a validated classifier**, and the audit says so in its own output.

### The signal the reviewer actually needs

Each draft carries `benchmark_quote` and `agrees_with_benchmark_quote`: the
sentence the frozen retrieval benchmark already pinned for that question, and
whether the machine's selection matches it. Divergence is not an error — it
marks exactly the places where a human must decide which sentence answers the
question.

## Pre-freeze benchmark audit

`quality_audit.py` is deterministic and read-only. It runs 8 checks and reports
findings; it never edits the benchmark, so a benchmark problem survives in the
audit trail and is classified by a human.

Checks that ran (2026-10-10):

```
answerability_supported_by_evidence      question_wording_heuristics
citation_requirement_consistency         shared_required_evidence
evidence_present_in_corpus               near_duplicate_questions
evidence_source_title_recorded           question_terms_covered_by_evidence
```

Checks that **could not run** are reported as such, never as "clean":

```
annotation_reference_answer_grounded      — no annotations exist yet
authored_key_points_supported_by_evidence — no annotations exist yet
```

The output separates `checks_run` / `checks_not_run`, `error_count` /
`warning_count` / `info_count`, a `verdict`, `ready_to_freeze`, per-finding
`action` codes, and a `limitations` block that states what a lexical audit
cannot decide (meaning, factual correctness, differently-worded duplicates).

## Measured live results (2026-10-10)

### Drafts, all 28 questions

| Status | Count | Meaning |
|---|---|---|
| `draft_ready` | 23 | a supporting sentence was found and coverage ≥ 45% |
| `partial` | 3 | a sentence was found but coverage is below the line (011, 016, 019) |
| `insufficient_evidence` | 2 | no sentence clears the 20% floor (024, 028) |

27 key points offered in total, each traceable to a chunk and sentence index.
**7 drafts agree** with the sentence the frozen retrieval benchmark pinned,
**21 differ** — the divergence set is the reviewer's shortlist.

Persisted append-only in `benchmark_draft_suggestions`: 28 rows for 28 distinct
questions, plus 1 more when the UI button was pressed again (29 total), which
demonstrates that regenerating never overwrites.

### Audit of the 28-question benchmark

```
0 error · 34 warning · 35 info   (69 findings)
8 checks ran · 2 could not run
ready_to_freeze: false
verdict: STRUCTURALLY SOUND BUT INCOMPLETE — 28 of 28 questions have no authored
         reference answer, so correctness is UNKNOWN. Authoring is the remaining work.
```

| Code | Severity | Count | Reading |
|---|---|---|---|
| `ANSWER_LABEL_NOT_AUTHORED` | warning | 28 | expected; the remaining human work |
| `QUESTION_TERMS_ABSENT_FROM_EVIDENCE` | info | 24 | observations, usually the question's own phrasing |
| `SHARED_REQUIRED_EVIDENCE` | info | 6 | questions pinning the same chunk |
| `QUESTION_COMPOUND` | info | 5 | questions asking more than one thing |
| `EVIDENCE_QUESTION_COVERAGE_LOW` | warning | 5 | pinned evidence may not answer the question |
| `EVIDENCE_SOURCE_TITLE_NOT_IN_CORPUS` | warning | 1 | benchmark-level provenance observation |

**Review these five first** (lowest IDF-weighted coverage of pinned evidence):

| Question | Coverage | Terms absent from the evidence |
|---|---|---|
| auto-eng-028 | 5% | `recovers, wasted, secondary, otherwise, would, friction, source, brakes, cars` |
| auto-eng-024 | 11% | `behaviours, listed, phenomena, pitch, yaw, key` |
| auto-eng-011 | 34% | `happens, differ, conventional, friction` |
| auto-eng-016 | 42% | `influence, axles, shift, between` |
| auto-eng-019 | 39% | `play, role` |

Low coverage is a **flag to read, not a verdict**: the audit itself notes a
question can use a different word for something the evidence states plainly.

The provenance observation is a genuine corpus finding: **29 of 29 required-evidence
slots carry a source title on the benchmark label that the live corpus chunk does
not record** (`chunk.source_title` is empty). Chunk id, document id and content
hash all still agree, so citations remain valid; but any view that reads the
title from the *chunk* rather than the label will show a blank.

### Invariants after the live run

* Benchmark completeness: `28 pending · 0 in_review · 0 approved · 0 rejected ·
  0 ambiguous · 0 insufficient_evidence · 0 authored reference answers · 0 reviews`.
* `POST /answer-benchmark/freeze` → **409** with the gate's own reasons.
* `git status benchmarks/` → empty; the frozen retrieval benchmark
  `automobile-engineering-baseline-v1.json` is untouched.
* Qdrant: 14 collections / 8822 points / protected KB 812 points — unchanged.
* Coverage metrics still report **UNKNOWN**, never 0 (`unknown_metrics` list).

## API and UI

```
POST /api/knowledge-bases/{kb}/answer-benchmark/draft-suggestions
GET  /api/knowledge-bases/{kb}/answer-benchmark/questions/{qid}/draft-suggestions
POST /api/knowledge-bases/{kb}/answer-benchmark/questions/{qid}/draft-suggestions
GET  /api/knowledge-bases/{kb}/answer-benchmark/quality-audit
```

UI, on `/knowledge-bases/kb_f278c283c748/benchmark-review`:

* **Draft assist (machine-generated)** in the authoring panel: generate /
  regenerate, draft status + coverage, "matches / differs from the sentence
  recorded in the benchmark", the draft's caveats and generator identity,
  "Use as starting point" (copies text only — nothing is saved), and the
  append-only draft history.
* **Quality audit findings for this question** inside the question card, so a
  reviewer sees the flags for the question in front of them.
* **Benchmark quality audit (read-only)** panel: counts, verdict, findings with
  a "show all N findings" expander, the checks that did not run, and the
  limitations block.

## Verification

| Check | Command | Result |
|---|---|---|
| V11 unit/API tests | `pytest tests/test_v11_*.py` | **54 passed** |
| Full backend suite | `pytest -q` | **1116 passed, 0 failed** (V10: 1062) |
| Frontend types | `npx tsc --noEmit` | clean |
| Frontend build | `npm run build` | clean |
| Live audit | `GET …/quality-audit` | 200, 69 findings, verdict as above |
| Live drafts | `POST …/draft-suggestions` (persist) | 200, 28 drafts, 28 rows persisted |
| Draft history | `GET …/questions/auto-eng-001/draft-suggestions` | 2 rows, append-only |
| Ground truth not created | `GET …/completeness` | 28 pending, 0 authored |
| Freeze blocked | `POST …/freeze` | 409 |
| Benchmark artifact | `git status benchmarks/` | empty |
| UI round-trip | browser on the review page | draft generated, audit run, findings expanded, question 024 flags rendered |

One UI defect was found during that round-trip and fixed: the audit panel's
"show the remaining warning-level findings" control was gated on
`errors + benchmark-level findings > 4`, which is false for this dataset, so all
33 warning-level findings were unreachable in the panel. The expander now also
dedupes a finding that appears in both the severity and benchmark lists.

## What the human must now decide

V11 changes the work from "read 22 chunks and write 28 answers" to "check 28
drafts and sign them". The decisions that remain are all human ones:

1. For the 23 `draft_ready` questions — accept, amend, or replace the draft
   sentence, and record the key points as reference key points.
2. For 011 / 016 / 019 — read the evidence and decide whether the pinned chunks
   really answer the question; if not, classify the question rather than
   patching the evidence.
3. For 024 / 028 — the selector could quote nothing. Read them: the evidence may
   hold the answer as a **list** (the extractive selector works on sentences) or
   the answer may genuinely be absent from the corpus. `AMBIGUOUS` or
   `INSUFFICIENT_EVIDENCE` are legitimate, recorded outcomes.
4. Decide what to do about the source-title provenance observation: label-only
   titles are valid for citation, but any consumer reading titles from chunks
   sees a blank.
5. When every question is authored and reviewed, the gate decides whether the
   benchmark may be frozen. Freezing is the point at which the first official,
   reference-based answer baseline can run.

Nothing in V11 produces a correctness number, changes `UNKNOWN` to a value, or
runs an optimization. Those remain blocked until this step is done, exactly as
the roadmap requires.

## Known limitations

* Coverage is **lexical and IDF-weighted**, calibrated on this benchmark; a
  question above the line is not thereby proven answerable.
* Drafts are single-sentence extracts: they do not synthesize a multi-sentence
  answer, so a reviewer who wants prose must compose it.
* `agrees_with_benchmark_quote` compares against the sentence the *frozen
  retrieval benchmark* pinned. Agreement is not correctness; disagreement is
  not error.
* The audit cannot detect two differently-worded questions testing the same
  fact, and it never claims to.
* Drafts are stored in the local SQLite database (`backend/data/`), which is a
  live artifact and not committed; the draft history is per-machine.
* Draft suggestions are append-only by construction and are never read by the
  evaluator, the gate, or the freeze path.
