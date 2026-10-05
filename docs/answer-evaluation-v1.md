# Answer-Evaluation v1 — experiment report

**Status:** executed 2026-10-06. **Official: NO** — deliberately. The source
benchmark `benchmarks/answer-quality-automobile-v1.json` has lifecycle
`draft` (it carries no `lifecycle` field) and `human_review: pending`
inherited from its own authorship block. Official answer-quality results
require a FROZEN benchmark, so every number here is an **experiment record**
and is labelled `official=false` in the artifact itself.

- Spec: [`benchmarks/answer-evaluation-v1.json`](../benchmarks/answer-evaluation-v1.json)
  (experiment spec — contains no questions and no ground truth of its own)
- Results: [`benchmarks/answer-evaluation-v1-results.json`](../benchmarks/answer-evaluation-v1-results.json)
- Runner: `backend/scripts/run_answer_evaluation_v1.py`
- Architecture & metric definitions: [`answer-evaluation-architecture.md`](./answer-evaluation-architecture.md)

## Setup

| | |
|---|---|
| Knowledge base | `kb_f278c283c748` (Automobile Engineering, 812 chunks) |
| Questions | 28, identical subset across strategies (answerable=28) |
| Generator | `llm / mock/mock-1` — **`is_mock=true`**: the extractive mock generator is the baseline, not a capable answerer |
| Evaluator | `deterministic-evidence` v8.3, `is_model_based=false` |
| Entailment | `heuristic-lexical-coverage` |
| Relevance | `idf-weighted-question-term-coverage`, weights from the corpus lexical index (`n_docs=812`) |
| Answer mode | `abstain_if_unsupported` |

## Results (aggregate over 28 questions)

| metric | dense | bm25 | hybrid |
|---|---|---|---|
| pass_rate | 0.821 | 0.786 | **0.893** |
| citation_recall | 0.821 | 0.786 | **0.893** |
| citation_precision (lower bound) | 0.286 | 0.274 | **0.310** |
| retrieval_hit_rate | 0.964 | 0.821 | 0.964 |
| question_answer_relevance | 0.616 | **0.699** | 0.635 |
| relevance_failure_rate (↓) | 0.036 (1/28) | 0.000 | 0.000 |
| fabricated_citation_rate (↓) | 0.000 | 0.000 | 0.000 |
| unsupported_citation_rate (↓) | 0.000 | 0.000 | 0.000 |
| supported_claim_ratio | 1.000 | 1.000 | 1.000 |
| abstention_accuracy | 1.000 | 1.000 | 1.000 |
| hallucination_rate (↓) | 0.179 | 0.214 | **0.107** |
| correctness | UNKNOWN | UNKNOWN | UNKNOWN |
| key_point_recall / expected_information_coverage | UNKNOWN | UNKNOWN | UNKNOWN |
| reference_answer_similarity | UNKNOWN | UNKNOWN | UNKNOWN |

UNKNOWN here is not zero: the source benchmark has **no human-authored
reference answers and no key points**, so correctness and completeness are
unmeasured by design (see the architecture doc §8).

Comparisons: all three pairwise comparisons returned **verdict COMPARABLE,
mode `identical`** (same 28 questions, same fingerprint
`f72c30eb5c0c363a`); any other verdict would have failed the script.

## Reading these numbers honestly

- **Hybrid leads on citations, retrieval and hallucination** (+0.071
  pass_rate over dense, +0.107 over bm25; 3/28 fewer confident-wrong answers
  than dense, 3/28 fewer than bm25). With a mock generator and 28 questions
  this is a directional observation, not a validated ranking — no
  significance testing has been done on n=28.
- **bm25 has the most relevant answers but the weakest retrieval hit rate**
  (0.699 relevance vs 0.821 hit). This is exactly why the dimensions stay
  separate: the "best relevance" strategy would look worst under any single
  combined score.
- **fabricated_citation_rate = 0.000 everywhere** is a *measured* zero: the
  citation-validation stage removes/repairs unresolvable references before
  answers are stored. It is reported as a measured value with sample sizes,
  not assumed.
- **The single dense relevance failure** (0.036) is an answer whose text did
  not reach the IDF relevance threshold — grounded but about the wrong
  subject. Under the older plain-coverage metric this failure class was
  *inverted* (off-domain scored higher than on-domain), so the zero-failure
  rows for bm25/hybrid are only meaningful because IDF weights were present.
- **citation_precision is a lower bound**, not a defect rate: extra on-topic
  context counts against it by construction.
- **The known open defects still shape these numbers** (architecture doc
  §9): the mock generator's 5-claim budget and the off-domain gate weakness
  are upstream of evaluation and were deliberately not fixed inside V8.

## Reproducing

```bash
# 1. Qdrant running (cwd = qdrant-dl), backend on :8013
cd backend && python -m uvicorn app.main:app --port 8013
# 2. run the experiment (writes the results artifact)
cd backend && python scripts/run_answer_evaluation_v1.py http://127.0.0.1:8013
```

The script verifies the source benchmark fingerprint before running,
records Qdrant collection count before/after, refuses to write any frozen
artifact, and fails (exit 1) if any comparison is not COMPARABLE or the
Qdrant count changes.

## Data impact of this run

- Qdrant collections: **14 → 14 (unchanged)**; no vectors written, none
  deleted (asserted by the script before/after each execution).
- `answer_evaluation_runs`: 11 → 14 on this (second) execution of the
  script, which re-ran the experiment under the fixed backend so the
  artifact's `unknown_metrics` no longer contains the duplicated
  `key_point_recall` entry produced by a stale server process. The first
  execution's 3 rows (plus 1 crashed-attempt dense row and 1 human-evaluation
  derivation) remain as immutable history: **14 rows at artifact write time
  (15 after a follow-up human-evaluation derivation), all on
  `kb_f278c283c748`**. Each execution's rows are embedded in the artifact it
  writes (`data_impact.answer_eval_runs_before/after`).
- `answers`: 398 → 482 (+84 = 3 × 28 questions from this execution). The
  answering path always persists its own rows; measured from the database,
  not assumed.
- `answer_reviews`: 1 row exists in the database — a UI round-trip
  verification review (reviewer `v8-ui-verification`, verdict `correct` on
  `auto-eng-011`) stored while verifying the review workflow in the browser,
  plus the human-evaluation run derived from it. **No review was part of, or
  used by, this experiment** — its correctness metrics are UNKNOWN exactly as
  reported above.
- Frozen artifacts: **0 written**. `automobile-engineering-baseline-v1*`,
  `answer-quality-automobile-v1.json`, `source-selection-experiment-*` are
  read-only inputs.
