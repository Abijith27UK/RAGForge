# Answer Benchmark Design

> Why `benchmarks/answer-quality-automobile-v1.json` looks the way it does, and
> exactly what a human must author to make more metrics computable.

---

## 1. The problem with the obvious design

The natural move is to take the frozen retrieval benchmark's 28 questions and
add `expected_answer` fields — generating reference answers with the LLM, then
scoring answers against them.

**That is circular and it is not done here.** An LLM-authored reference answer
grades an LLM-generated answer against itself, and the resulting percentage
looks rigorous while measuring agreement between two samples from one
distribution. It would be the most publishable-looking and least trustworthy
number in the project.

So the benchmark ships with `expected_answer: null` and `key_points: []` for all
28 questions, and `answer-evaluation.md` explains why `correctness` is UNKNOWN.

---

## 2. What ground truth actually exists

`benchmarks/automobile-engineering-baseline-v1.json` was audited before anything
was built. Findings:

| Signal | Present | Notes |
|---|---|---|
| `expected_chunk_ids` | 28/28 questions | 22 unique chunks, all verified in the live corpus |
| `expected_document_ids` | 28/28 | |
| `subdomain` | 28/28 | |
| `provenance` with quoted source text | 28/28 | Real indexed text, not paraphrased |
| `expected_answer` | **0** | |
| `key_points` | **0** | |
| `expected_keywords` | **0** | |
| `answerability` | **0** | |
| `authorship.human_review` | `"PENDING ..."` | |

All 22 unique required chunks were verified present in KB `kb_f278c283c748`
(812 chunks) with matching content hashes.

**Conclusion:** the ground truth is *evidence-based*, not *answer-based*. The
benchmark was designed around that fact instead of around the hope that someone
would later fill in the blanks.

---

## 3. Derivation rules

Implemented in `backend/app/services/answer_eval/benchmark.py` and applied by
`backend/scripts/build_answer_benchmark_v1.py`.

### `answerability`
- `answerable` **iff** the reviewer selected at least one answering chunk.
- `unanswerable` is **never inferred**. A human must assert it, and such a
  question must also set `abstention_required: true`.
- The model validator **rejects** an unanswerable question that also requires
  evidence, and an answerable question that requires none.

### `required_evidence`
Inherited verbatim from `expected_chunk_ids`, enriched with `document_id` and
`content_hash` read from the live corpus.

The content hash is what makes the artifact safe: if the corpus is re-chunked,
`validate_against_corpus()` fails and the run is **refused** rather than scored
against a silently-shifted chunk set.

### `citation_requirements`
Mechanically derived:
- `require_at_least_one_citation: true`
- `must_cite_all_required_evidence: true` — enforced **per answer**
- `must_not_cite_irrelevant_chunks: true` — reported as an **observation**,
  never a failure, because the required set is a necessity selection, not an
  exhaustive whitelist

### `expected_grounding_state`
`ANY_ACCEPTABLE` for every current question. This is a *behavioural
expectation*, not a prediction of what the system does. Inventing a specific
expected state would manufacture ground truth.

---

## 4. Safety properties

The generator and loader enforce, and tests assert:

1. **The frozen file is never written.** `build_answer_benchmark_v1.py` opens it
   read-only and the test suite asserts byte-identity via
   `git diff --exit-code`.
2. **A benchmark that references a missing chunk is rejected**, not scored.
3. **A re-chunked corpus is detected** via content-hash mismatch and the run is
   refused with a 409.
4. **A benchmark targeting another KB is rejected** — an answer benchmark is
   always evaluated against its own corpus.
5. **The fingerprint** (`f72c30eb5c0c363a`) covers benchmark name, version,
   KB id and the per-question evidence sets. It is recorded on every run.
6. **`human_review` is inherited honestly.** It stays `pending`; no agent
   upgrades it.

---

## 5. What a human must author to extend coverage

Each of these is currently empty *by design*, not by oversight.

### To make `correctness` measurable
Per question, a human writes:
- `expected_answer` — a reference answer, **or**
- `key_points` — the required factual points.

The evaluator already detects these fields and will refuse to report UNKNOWN
once present; scoring them is the next increment, not a silent change.

### To make `false_unsupported_rate` measurable on the real corpus
The shipped benchmark has **0 unanswerable questions**, so abstention on
unanswerable input cannot be measured here — only on scratch fixtures. A human
must author questions the corpus genuinely does not contain.

### To make `key_point_recall` and numeric checks measurable
- `key_points`
- `acceptable_answer_elements` — alternative phrasings judged acceptable
- `numerical_tolerance` / `requires_units` — for quantitative answers

---

## 6. Rules for extending the benchmark

1. **Never auto-generate ground truth.** Author it, or leave it empty and let
   the metric stay UNKNOWN.
2. **Never mutate an existing version.** Add a new `version` and bump the
   fingerprint; `is_comparable_with()` will then refuse cross-version comparison
   rather than silently mixing.
3. **Never compare across question subsets.** The run records `question_ids`
   and refuses mismatched comparisons.
4. **Re-validate against the corpus** before every run.
5. **Keep `human_review` honest.** `pending` is the normal state for
   agent-authored work.

---

## 7. Honest summary

| Question | Measurable today? |
|---|---|
| Does the answer cite the evidence a human marked necessary? | **Yes** |
| Is each claim supported by the evidence it cites? | **Yes** (heuristic entailment) |
| Did the system abstain exactly when it should? | **Yes** |
| Did it label its own confidence correctly? | **Yes** |
| Was the required evidence even retrieved? | **Yes**, reported separately |
| **Is the answer factually correct?** | **No — UNKNOWN** |
| **Does it contain the required key points?** | **No — UNKNOWN** |
| Does it abstain correctly on *unanswerable* questions? | **No — no such questions exist** |

Three of the eight questions an answer-quality benchmark should answer are
genuinely unanswerable with the ground truth that exists. That is stated here
rather than papered over.