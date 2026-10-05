# Answer-Evaluation Architecture (V8)

This document describes RAGForge's answer-quality evaluation layer: what is
measured, what is deliberately NOT measured, and the rules that keep every
number attributable. The companion report [`answer-evaluation-v1.md`](./answer-evaluation-v1.md)
documents the v1 experiment executed against this architecture.

## 1. Design rules

1. **Every metric is a `Measured` triple** (`value`, `measured`, `reason`,
   optional `sample_size`). Unmeasured is `measured=false` with a reason —
   never `0.0`. "We could not check this" and "this scored zero" are different
   findings.
2. **Dimensions stay separate.** Retrieval quality, evidence quality, citation
   correctness, groundedness, answer relevance, completeness, abstention and
   pass/fail are individual metrics. There is no combined "RAGForge score" in
   any API or UI; `final_score` is an explicitly-weighted, per-question
   convenience (0.4 citation precision / 0.3 citation recall / 0.3 evidence
   support) that refuses to compute when any input is unmeasured.
3. **Every metric carries its producer.** Evaluator name+version, evaluator
   model-based flag, entailment provider, relevance method and weight source
   are stored on results and runs.
4. **Runs are immutable and comparable only when they cover the same
   questions.** See §7.
5. **No ground truth is invented.** Human-authored labels are the only source
   for correctness/completeness; missing labels produce UNKNOWN, not proxies
   dressed up as verdicts.

## 2. Module map

```
backend/app/services/answer_eval/
  benchmark.py    AnswerBenchmark + lifecycle (draft→approved→frozen), fingerprint,
                  corpus validation, require_official_benchmark()
  evaluator.py    AnswerEvaluator ABC; DeterministicAnswerEvaluator (default),
                  HumanAnswerEvaluator, LLMAnswerEvaluator; claim auditing;
                  apply_reviews() overlay; create_answer_evaluator() factory
  entailment.py   SupportJudgement (supported/not_supported/unknown);
                  heuristic lexical provider (default), LLM provider (explicit)
  relevance.py    question↔answer relevance: IDF-weighted term coverage with a
                  measured evidence table; plain fallback labelled weaker
  metrics.py      Measured, ClaimCitationVerdict (+evaluated_state),
                  AnswerQualityResult, AnswerQualityAggregate, SCORE_INPUTS
  run.py          AnswerEvaluationRun (immutable), aggregate_results, build_run,
                  is_comparable_with (strict), compare_with (intersection plan)
  review.py       AnswerReview (append-only human review), ReviewVerdict,
                  ReviewLabel, VERDICT_SCORES policy mapping
  service.py      orchestration: corpus validation, IDF weight injection,
                  run persistence, derive_human_run() (reviews → new run)
backend/app/api/routes_answer_eval.py
  KB-scoped run endpoints, compare, reviews (POST/GET), human-evaluation,
  plus top-level /api/answer-evaluation-runs (list + get by id)
```

## 3. Metrics catalogue

### Retrieval (kept separate from answering)
- `retrieval_hit_rate` — was the *required* evidence retrieved at all. A
  retrieval miss is never blamed on the answerer, and never folded into
  citation metrics.

### Citation correctness
- `citation_precision` — fraction of CITED chunks the benchmark marks as
  required. Documented as a **lower bound**: the required set is a
  necessity selection, not an exhaustive whitelist, so citing extra on-topic
  context lowers it without being an error.
- `citation_recall` / `citation_completeness` — fraction of required chunks
  cited. An answer citing nothing scores 0, not "n/a" (except abstentions,
  where citations are not applicable → UNKNOWN).
- `fabricated_citation_rate` — distinct evidence references that do not
  resolve to retrieved evidence, or whose provenance validation says
  `invalid`, over all distinct references.
- `unsupported_citation_rate` — citations on claims the entailment provider
  judged NOT_SUPPORTED, over **judged** citations only; unjudged citations
  are excluded (never counted as supported).

### Groundedness / claims (STEP 6)
Each claim gets a five-state `evaluated_state`:
`SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | CONTRADICTED | UNVERIFIABLE`.
Entailment wins over the generator's own status; an uncited factual claim is
UNSUPPORTED; a claim whose support could not be determined is UNVERIFIABLE,
never silently supported. Aggregates:
- `supported_claim_ratio`, `partial_claim_ratio`, `unsupported_claim_ratio`
  (over ALL claims; the four buckets partition: supported + partial +
  unsupported + unverifiable = 1),
- `contradiction_rate`, `evidence_support_rate`, `unsupported_claim_rate`
  (legacy: uncited only).

### Answer relevance (STEP 8)
- `question_answer_relevance` — IDF-weighted coverage of the question's
  content terms in the answer text (`idf-weighted-question-term-coverage`).
  Measured evidence for this choice (286 stored answers on the real corpus):

  | method | on-domain min | known off-domain (gate defect) | verdict |
  |---|---|---|---|
  | plain term coverage | 0.333 | 0.400 | **inverted — unusable** |
  | IDF-weighted coverage | 0.300 | 0.260 | separates |

  The threshold (0.30) sits in a **thin margin (0.04 on a single negative
  sample)** — documented, not claimed as a validated separation;
  `relevance_close_call` flags marginal verdicts. When corpus IDF statistics
  are absent, the plain fallback is used, its shortfall is reported as a
  WARNING (never a failure), and `relevance_method`/`relevance_weight_source`
  record which method produced each number.
- `relevance_failure_rate` — answers measured below threshold, abstentions
  excluded (a refusal is not "irrelevant").
- `excess_information` — sentences sharing no question term; observation
  only, never a failure.
- This is a **lexical proxy, not semantic relevance**; `is_model_based` is
  always false and a paraphrase with none of the question's terms scores low.

### Completeness (STEP 2/F, human labels only)
- `key_point_recall` and its spec-named alias
  `expected_information_coverage` — the same measurement: fraction of
  human-authored key points whose content terms are covered in the answer
  (threshold 0.6, published; labelled LEXICAL PROXY). Requires human key
  points; otherwise UNKNOWN.
- `reference_answer_similarity` — lexical similarity to a human reference
  answer, reported **separately from correctness** because a proxy must not
  be promoted to a verdict.
- `correctness` — stays UNKNOWN in the deterministic evaluator even when a
  reference exists. A correctness verdict requires human review
  (§6) or an explicitly model-based judge (§5).

### Abstention
- `abstention_correct` per question (answerable → must answer;
  unanswerable → must abstain; unknown answerability → UNKNOWN),
  `abstention_accuracy`, `false_supported_rate`, `false_unsupported_rate`,
  `hallucination_rate` in the aggregate.

### Overall
- `pass_rate` = citations correct AND abstention correct AND no evaluator
  problems (a relevance shortfall under IDF counts as a problem — an answer
  about the wrong subject cannot pass on citations alone).

## 4. Evaluator abstraction (STEP 5/14)

```
AnswerEvaluator (ABC)
├── DeterministicAnswerEvaluator  — offline, reproducible, default
├── HumanAnswerEvaluator          — correctness from stored human reviews
└── LLMAnswerEvaluator            — LLM-as-judge, explicit opt-in
```

- The factory (`create_answer_evaluator`) **requires the kind to be
  requested explicitly**; an unknown kind raises, and `human`/`llm` raise
  without their provider. A judge is never silently substituted.
- Every result records `evaluator_name`, `evaluator_version`,
  `evaluator_is_model_based`, `evaluator_detail` (model+prompt version, or
  reviewer identities).

### LLM-as-judge limitations (explicit)
- Constructing without a provider raises.
- A provider failure or unparseable output at judge time degrades to
  correctness UNKNOWN with a warning — a failed judge is not a verdict, and
  out-of-range scores are rejected, not clamped.
- A recorded judgement stores `judge_raw` verbatim, carries
  `LLM-as-judge (model, prompt …)` in its reason, sets
  `evaluator_is_model_based=true`, and appends a warning that human review or
  a validated benchmark answer outranks it. It is **never ground truth**.
- Requires a better-than-mock LLM for meaningful scores; with the dev-mock
  provider the output is placeholders (the run is labelled `is_mock`).

## 5. Human review workflow (STEP 4)

- `answer_reviews` table, append-only: `create_answer_review` is an INSERT;
  there is no update or delete path. `GET .../reviews` returns the whole
  history oldest-first — disagreement between reviewers stays visible.
- Every review stores a non-blank `reviewer` identity and a timestamp;
  blank identities are rejected at both schema and API level.
- Closed vocabularies: 6 verdicts (correct, mostly correct, partially
  correct, incorrect, should have abstained, correctly abstained) and 9
  structured labels.
- **Derivation, not mutation:** `POST /runs/{id}/human-evaluation` produces a
  NEW run whose `correctness` is the mean of the reviews' verdict scores
  (published ordinal policy: 1.0 / 0.75 / 0.5 / 0.0), noting source lineage.
  The source run and the reviews are untouched; questions without reviews
  keep correctness UNKNOWN.

## 6. Benchmark lifecycle (STEP 3)

- `AnswerBenchmarkLifecycle`: `draft → approved → frozen`, separate from
  `human_review` provenance (lifecycle gates *use*; provenance claims
  *authorship*).
- Default is **draft**: an artifact that does not say it is frozen is not
  frozen. A new domain therefore never receives implicit ground truth.
- `official: true` runs are refused unless the benchmark is FROZEN
  (`require_official_benchmark`); development runs record
  `official=false` so draft numbers are never mistakable for official ones.
- Per-question optional review metadata (`difficulty`, `reviewer`,
  `review_status`) parses when present, rejects out-of-vocabulary values, and
  is never backfilled. The fingerprint covers question content only, so
  metadata relabelling does not change an artifact's identity.
- The shipped `benchmarks/answer-quality-automobile-v1.json` has no
  `lifecycle` field and inherits `human_review: pending` from its authorship
  block — so it loads as DRAFT and official runs against it are refused.

## 7. Comparability rules (STEP 11)

- `is_comparable_with` (strict): same question ids AND same benchmark
  fingerprint, else refused. Used to block casual misuse.
- `compare_with` (intersection plan) adds three verdicts:
  - `COMPARABLE` / `identical` — full aggregate comparison;
  - `INCONCLUSIVE` / `intersection` — subsets differ but overlap;
    differences are computed on the **shared questions only** and the verdict
    says INCONCLUSIVE so partial overlap never masquerades as a full-set
    comparison;
  - `NOT_COMPARABLE` / `no_overlap` — zero shared questions, or the benchmark
    content changed; no differences are produced at all.
- The compare response also warns when two runs measured relevance with
  different methods.

## 8. What is NOT measured (and why)

- **Factual correctness** against a reference — no human reference answers
  exist yet; lexical similarity would be a proxy (reported separately as
  `reference_answer_similarity`).
- **Semantic relevance** — the relevance metric is lexical by design and
  says so.
- **Semantic entailment** — the default judge is lexical coverage with a
  published threshold; `UNKNOWN` is returned when it cannot decide.
- **LLM-as-judge scores as ground truth** — model-based, labelled, raw
  output stored.
- Numerical/unit verification against `numerical_tolerance` — schema field
  exists; not yet implemented.

## 9. Known open defects (reported, deliberately not fixed in V8)

- **Off-domain gate weakness** (`answering/gate.py`): `MIN_ALIGNMENT_*`
  thresholds are unweighted, so a known off-domain (drone) answer passed the
  gate at 0.40. Relevance now *detects* such answers downstream but the gate
  itself is unchanged — fixing it before re-measuring would be benchmark
  tuning.
- **5-claim budget** of `ExtractiveMockAnswerGenerator` structurally
  uncites evidence ranked ≥4th (4 of 5 real citation-recall failures).
  A generator fix belongs to a separate change, not to evaluation code.

## 10. Test map

| concern | file |
|---|---|
| core evaluator + run records (68) | `tests/test_answer_eval_v8.py` |
| API + official gating + reviews + human pass + compare + top-level (36) | `tests/test_answer_eval_api_v8.py` |
| relevance (13) | `tests/test_answer_relevance_v8.py` |
| claim states + citation metrics (17) | `tests/test_answer_claim_states_v8.py` |
| completeness from human labels (11) | `tests/test_answer_completeness_v8.py` |
| lifecycle + official gate (16) | `tests/test_answer_benchmark_lifecycle_v8.py` |
| evaluator abstractions + reproducibility (19) | `tests/test_answer_evaluator_abstractions_v8.py` |
