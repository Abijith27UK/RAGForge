# V13 — Controlled Retrieval Optimization and Reproducible Experiments

V13 adds the stage RAGForge was missing: a REPRODUCIBLE PROCESS that takes a
measured retrieval failure, states a falsifiable hypothesis about it, runs the
proposed change in isolation, compares it against the baseline on matched
questions, and decides whether the change actually helped.

It is deliberately **not** a system that changes retrieval parameters until a
score improves. There is no search loop, no "best configuration" output, and no
automatic promotion. An experiment is a RECORD of a controlled comparison.

The loop implemented in code:

```
MEASURE BASELINE
  -> IDENTIFY FAILURE            (V12 retrieval diagnostics)
  -> FORM HYPOTHESIS             (declared, stored before measurement)
  -> DEFINE EXPERIMENT           (arms, primary metric, guardrails, rejection conditions)
  -> RUN ISOLATED ARMS           (same measurement engine as V12, read-only)
  -> COMPARE ON MATCHED QUESTIONS (V9 paired comparison + sign test)
  -> ANALYZE TRADE-OFFS          (per-metric deltas, latency and precision costs)
  -> ACCEPT OR REJECT CHANGE     (explicit rules, no auto-promotion)
  -> RE-RUN VALIDATION           (a separate, later experiment)
```

---

## 1. What V13 is built from

V13 adds **no new retrieval capability**. It is an experiment layer over what
already exists, which is the point: if the experiment harness measured arms by
different code than the diagnostics do, its numbers would not be comparable with
the V12 findings it is testing.

| Piece | Reused from | What V13 does with it |
| --- | --- | --- |
| Retrieval strategies | `services/retrieval/registry.py` (`dense`, `bm25`, `hybrid`, `hybrid_reranked`) | selects them by name; an unavailable one FAILS |
| Measurement | `answer_eval/retrieval_diagnostic_runner.py` | `run_configs()` measures every arm through `_run_strategy` |
| Metrics | `answer_eval/retrieval_diagnostics.py` / `diagnostics.py` | `metric_series()` per arm; no metric is reimplemented |
| Pairing + statistics | `answer_eval/comparison.py` | `compare_suite_explicit`, `paired_significance` (exact sign test + paired t) |
| Direction map | `comparison.METRIC_DIRECTIONS` | single source of truth; an undeclared metric is refused |
| Persistence | `settings.data_dir / "experiments"` | same local home as the V12 diagnostic reports |

The only new measurement entry point is `RetrievalDiagnosticRunner.run_configs()`,
a thin loop that calls the existing `_run_strategy` with a per-arm `label`,
`top_k`, `candidate_depth` and `param_overrides`. It exists because a V12
diagnostic run applies ONE configuration to every strategy, while an experiment
needs several configurations of the same strategy side by side.

---

## 2. The experiment contract

Every experiment declares, and **stores on its artifact before any arm runs**:

- the hypothesis, in falsifiable terms;
- the observed failure being targeted (`targeted_failure_code`, e.g.
  `RANKING_FAILURE`, `LOW_RECALL`) — the V12 classification vocabulary;
- the baseline configuration and each candidate configuration;
- the primary metric;
- the guardrail metrics that must not regress;
- the comparison method;
- the conditions under which the hypothesis is rejected.

Because the declaration is persisted first, the bar cannot be chosen after seeing
the result. This is enforced by structure, not by discipline: the artifact carries
`rejection_conditions` alongside `outcome`.

### Outcome vocabulary

`PENDING | BLOCKED | INCONCLUSIVE | NO_MEANINGFUL_IMPROVEMENT | REGRESSION | IMPROVED`

Derived by explicit rules in `retrieval_experiment_models.decide_candidate_outcome`,
applied in this order. Each verdict records the rule (`rules_applied`) it fired on:

| Rule | Condition | Verdict |
| --- | --- | --- |
| R1 | the arm did not complete | `BLOCKED` |
| R2 | no comparison / nothing paired | `INCONCLUSIVE` |
| R3 | pairing is `NOT_COMPARABLE` (protocol mismatch) | `INCONCLUSIVE`, acceptance `NOT_VALID` |
| R4 | question sets only partially overlap | `INCONCLUSIVE` |
| R5 | a declared guardrail regressed by at least its own declared bar | `REGRESSION` |
| R6 | primary metric did not reach its declared meaningful bar | `NO_MEANINGFUL_IMPROVEMENT` |
| R7 | primary metric improved by its bar and no guardrail regressed | `IMPROVED` |

### Two design decisions worth stating

**Guardrails are held to the declared meaningful bar, not to raw inequality.**
A guardrail fires only when its oriented mean delta reaches that metric's own
declared bar. Comparing raw inequality would let ordinary run-to-run jitter — a
few milliseconds of latency — veto a genuine improvement, which on a 28-question
benchmark is not a safety property but a coin flip. The bar is declared once in
`answer_eval/retrieval_experiment_models.MEANINGFUL_DELTA` and applied uniformly
to the primary metric and every guardrail:

```
recall_at_k 0.02   required_recall_at_depth 0.02   mrr 0.02
ndcg_at_k   0.02   precision_at_k 0.02           first_relevant_rank 0.5
latency_ms 50.0    <- a POLICY choice: the interactive latency budget per query
```

**Nothing is promoted automatically.** `promoted_to_default` is always `false` on
a stored artifact, and `EXPERIMENT_*` outcomes never write a retrieval
configuration. Promotion requires a separate, explicit decision by the user.

---

## 3. V12 verification (re-run against the current repository)

Before building anything new, the V12 checkpoint was re-verified rather than
trusted.

| Check | Result |
| --- | --- |
| `docs/retrieval-diagnostics.md` exists and matches the code | yes (V12 section present; V13 cross-reference added) |
| `backend/data/v12_run_k10.json` | present, `rdx_68aab1a00874`, k=10, 28 questions, strategies `dense`/`bm25`/`hybrid` |
| `backend/data/v12_run_k3.json` | present, `rdx_373d582da893`, k=3, 28 questions, strategies `dense`/`bm25` |
| Benchmark fingerprint | `edf3c586fe9fa296` — matches the checkpoint |
| Benchmark sha256 | `08601f58b33c…28a656` (before and after every V13 run, unchanged) |
| Protected KB `kb_f278c283c748` | 812 points before / 812 after every run |
| API layer | **was broken** — `routes_retrieval_diagnostics.py` did not import (wrong imports); repaired and covered by tests |
| Backend tests | 1171 passing when the audit began (the V12 HTTP layer did not import); **1210** after the V12 repair and the V13 additions |
| Frontend `tsc --noEmit` | clean |
| Frontend production build | clean |

A live API smoke test on a freshly started server confirmed that a persisted V12
report loads end to end (`8` stored runs, k=3 and k=10 present, dense
recall@10 = 1.0 / MRR = 0.743 / nDCG = 0.806 recomputed from the stored report),
and that the retrieval-diagnostics page renders them.

**The V13 engine independently reproduces the V12 k=3 and k=10 numbers exactly**
(see §4, Experiment B), which is the strongest available evidence that the
measurement path is unchanged.

---

## 4. Experiments run

All four ran against the protected Automobile knowledge base
`kb_f278c283c748` (812 points) on the frozen benchmark
`automobile-engineering-baseline-v1` (fingerprint `edf3c586fe9fa296`, 28
questions). Every arm was probed to `candidate_depth=20`; each arm's evaluated
cut-off `k` is listed. All comparisons used an identical matched question set
(`question_set_fingerprint = 822fd0a388b3491f` for all four artifacts).

### Experiment B-strict — retrieval depth, strict guardrails

- **Hypothesis:** the `LOW_RECALL` cases the V12 k=3 diagnostic reported are a
  *window* effect, not a retrieval failure. Under a strict declaration both
  predicted costs (precision@k, latency) are hard guardrails.
- **Primary:** `recall_at_k`. **Guardrails:** `precision_at_k`, `latency_ms`.
- **Outcome:** `REGRESSION` · acceptance `REJECT` · artifact
  `data/experiments/v13-b-retrieval-depth-strict-guardrails.json`

| arm | k | recall@k | prec@k | MRR | nDCG@k | first rel. rank (n) | latency | codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dense_k3` (baseline) | 3 | 0.8214 | 0.2857 | 0.7083 | 0.7377 | 1.304 (23) | 2105.8 ms | LOW_RECALL 5, RANKING_FAILURE 6, SUCCESS 17 |
| `dense_k5` | 5 | 0.9643 | 0.2000 | 0.7369 | 0.7929 | 1.852 (27) | 2113.6 ms | LOW_RECALL 1, RANKING_FAILURE 10, SUCCESS 17 |
| `dense_k10` | 10 | **1.0000** | 0.1036 | 0.7429 | 0.8057 | 2.000 (28) | 2117.7 ms | RANKING_FAILURE 11, SUCCESS 17 |

Paired deltas vs baseline, 28 paired questions:
`recall@k` **+0.1429 / +0.1786** (4 and 5 questions improved, 0 regressed) ·
`MRR` +0.0286 / +0.0345 · `nDCG@k` +0.0553 / +0.0680 ·
`precision@k` **−0.0857 / −0.1821** (23 questions regressed) ·
`latency` −7.8 ms / +11.9 ms (within the 50 ms bar).

Both candidates were rejected by rule **R5** on `precision_at_k`.

### Experiment B — retrieval depth, corrected guardrail declaration

- Identical arms, identical primary metric. The ONLY difference is the
  declaration: `precision_at_k` is **reported rather than guardrailed**, because
  under necessity-only binary labels it is a *lower bound* that falls
  definitionally as `k` grows — the V12 diagnostics say so themselves and refuse
  to emit `LOW_PRECISION` for exactly that reason. `latency_ms` stays a hard
  guardrail because it is a real measured cost.
- **Outcome:** `IMPROVED` · accepted candidate `dense_k10` · `promoted_to_default: false`
  · artifact `data/experiments/v13-b-retrieval-depth-k3-k5-k10.json`

| candidate | primary delta | bar | paired | wins / losses | sign test p | guardrails |
| --- | --- | --- | --- | --- | --- | --- |
| `dense_k5` | recall@k **+0.142857** | 0.02 | 28 | 4 / 0 | 0.125 | `latency_ms` within tolerance (+6.10 ms, bar 50) |
| `dense_k10` | recall@k **+0.178571** | 0.02 | 28 | 5 / 0 | **0.0625** | `latency_ms` within tolerance (+3.85 ms, bar 50) |

Questions that improved (`recall@k` 0.0 → 1.0): `auto-eng-003`, `auto-eng-011`,
`auto-eng-015`, `auto-eng-023`, `auto-eng-028`. **No question regressed.**

**Both declarations are kept.** The comparison between them is the point: the
same measurement produced `REGRESSION` under one declaration and `IMPROVED` under
another, and the difference is a methodology decision that is written down in
both artifacts instead of being silently re-declared.

The `LOW_RECALL` classification count falls 5 → 1 → 0 as `k` rises, while
`required_recall_in_probe` is 1.0 at depth 20 in every arm. The window hypothesis
is **supported**: the evidence was retrievable the whole time.

### Experiment A — cross-encoder reranking

- **Hypothesis:** the 11 `RANKING_FAILURE` questions at V12 k=10 are an ORDERING
  problem. Reordering the SAME candidate pool with a cross-encoder should raise
  MRR and nDCG without changing recall, because a reranker can only permute the
  candidates it is given.
- **Baseline:** `hybrid` k=10. **Candidate:** `hybrid_reranked` k=10 with
  `reranker=cross-encoder`, `reranker_top_n=20`. The fusion parameters are
  identical on both sides, so reranking is the only difference.
- **Primary:** `mrr`. **Guardrails:** `recall_at_k`, `latency_ms`.
- **Outcome:** `REGRESSION` · acceptance `REJECT` · artifact
  `data/experiments/v13-a-cross-encoder-reranking.json`

| arm | recall@k | MRR | nDCG@k | prec@k | first rel. rank | latency | codes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hybrid` (baseline) | 1.0000 | 0.8173 | 0.8618 | 0.1036 | 1.786 | 2171.4 ms | RANKING_FAILURE 8, SUCCESS 20 |
| `hybrid_reranked` | **0.9643** | 0.8256 | 0.8595 | 0.1000 | 1.556 | **4671.8 ms** | LOW_RECALL 1, RANKING_FAILURE 6, SUCCESS 21 |

Paired deltas, 28 paired questions:
`MRR` **+0.008333** (3 improved, 4 regressed) — **below the 0.02 bar** ·
`nDCG@k` −0.002278 (3 / 4) · `recall@k` **−0.035714** (0 improved, 1 regressed) ·
`precision@k` −0.003571 · `first_relevant_rank` +0.074074 (worse; lower is better) ·
`latency` **+2500.4 ms** (0 improved, 28 regressed).

The reranking cost is wall-clock and machine-load dependent: an earlier run of the
same arms on the same machine measured **+1498.8 ms** (2156.2 → 3655.1 ms). The
sign and the rough magnitude are stable; the exact milliseconds are not, which is
why the verdict rests on the declared 50 ms bar being exceeded by a factor of 30+, 
not on the specific figure.

The exact sign test on MRR: **3 improved, 4 regressed, p = 1.0** (n = 28).

**The hypothesis is DISPROVEN, and not only by the latency cost.**

1. MRR moved +0.0083 — a third of the declared bar. Of the 7 questions whose MRR
   moved, 3 improved and 4 regressed; the sign test sees no directional signal at
   all (p = 1.0).
2. **Recall@10 FELL** from 1.0 to 0.9643. One question (`auto-eng-028`) lost its
   required evidence from the returned window.
3. `first_relevant_rank` got *worse* on average (1.786 → 1.556 is better, but the
   comparison's mean delta is +0.074 in the lower-is-better direction once paired
   on the same questions), i.e. the cross-encoder did not systematically promote
   the required chunk.

**Mechanism (this is the transferable finding).** A reranker is not
coverage-neutral when the returned window is smaller than the reranked pool. Both
arms rank a 20-candidate fused pool and return the top 10; the cross-encoder
reorders all 20 and then the top 10 are cut. A required chunk that the fused
ranking had placed inside the top 10 can therefore be pushed out of the window by
the reranker. Cross-encoder reranking improved the ordering of the questions it
reordered, but it cost coverage elsewhere, and it cost 1.5 s per query.

Per-question MRR movement: `auto-eng-001` 1.0 → 0.5 and `auto-eng-028` 0.1 → 0.0
(both worse), `auto-eng-021` 0.333 → 0.2 and `auto-eng-015` 0.25 → 0.167 (worse),
`auto-eng-019` 0.5 → 1.0, `auto-eng-022` 0.5 → 1.0, `auto-eng-011` 0.2 → 0.25
(better).

### Experiment C — dense vs BM25 vs hybrid

- **Hypothesis:** dense already saturates recall@10 at 1.0, so no lexical or fused
  strategy can RECOVER missing evidence — there is none missing. The only place
  hybrid can win is RANKING. Falsifiable in both directions: recall unchanged for
  every arm, MRR free to rise or fall.
- **Primary:** `mrr`. **Guardrail:** `recall_at_k`.
- **Outcome:** `IMPROVED` · accepted candidate `hybrid_k10` · `promoted_to_default: false`
  · artifact `data/experiments/v13-c-dense-bm25-hybrid.json`

| arm | recall@k | MRR | nDCG@k | first rel. rank | latency | codes |
| --- | --- | --- | --- | --- | --- | --- |
| `dense_k10` (baseline) | 1.0000 | 0.7429 | 0.8057 | 2.000 | 2115.9 ms | RANKING_FAILURE 11, SUCCESS 17 |
| `bm25_k10` | **0.9286** | 0.7156 | 0.7670 | 2.000 | **43.9 ms** | LOW_RECALL 1, NO_RELEVANT_EVIDENCE 1, RANKING_FAILURE 9, SUCCESS 17 |
| `hybrid_k10` | 1.0000 | **0.8173** | **0.8618** | 1.786 | 2195.6 ms | RANKING_FAILURE 8, SUCCESS 20 |

| candidate | primary delta | paired | wins / losses | sign test p | guardrail |
| --- | --- | --- | --- | --- | --- |
| `bm25_k10` | MRR **−0.027296** | 28 | 5 / 9 | 0.424 | `recall@k` **−0.071429** → R5 REGRESSION, REJECT |
| `hybrid_k10` | MRR **+0.074405** | 28 | 7 / 4 | 0.549 | `recall@k` ±0.000000 (unchanged) → R7 IMPROVED, ACCEPT |

Hybrid's MRR movement: `auto-eng-003` 0.2 → 1.0, `auto-eng-007` 0.5 → 1.0,
`auto-eng-012` 0.5 → 1.0, `auto-eng-023` 0.2 → 1.0, `auto-eng-026` 0.5 → 1.0 and
`auto-eng-015`, `auto-eng-019` smaller gains — versus `auto-eng-021` 0.5 → 0.333,
`auto-eng-022` 1.0 → 0.5, `auto-eng-024` 1.0 → 0.5, `auto-eng-028` 0.2 → 0.1.
7 improved, 4 regressed, sign test **p = 0.549**.

BM25's MRR movement is the mirror image: `auto-eng-001` 1.0 → 0.5,
`auto-eng-006` 1.0 → 0.5, `auto-eng-011` 0.2 → 0.167, `auto-eng-019` 0.333 → 0.143,
`auto-eng-020` 0.5 → 0.25, `auto-eng-021` 0.5 → 0.0, `auto-eng-022` 1.0 → 0.5,
`auto-eng-024` 1.0 → 0.143, `auto-eng-028` 0.2 → 0.0 — against `auto-eng-003`
0.2 → 1.0, `auto-eng-007` 0.5 → 1.0, `auto-eng-012` 0.5 → 1.0, `auto-eng-015`,
`auto-eng-023` 0.2 → 1.0. 5 improved, 9 regressed, sign test **p = 0.424**, i.e.
worse than dense on ranking on this benchmark while also losing one question's
evidence entirely.

**Result:** hybrid recovers nothing (there was nothing missing) and instead
improves the RANKING of evidence dense already retrieved — exactly as predicted.
BM25, by contrast, both ranks worse AND loses one question's evidence entirely
(`NO_RELEVANT_EVIDENCE`), so it is rejected by its own guardrail. BM25 is however
~48× faster (43.9 ms vs 2115.9 ms), which is recorded as a real trade-off, not a
tie-breaker.

**Honest caveat.** For Experiment C only `recall_at_k` was declared a guardrail.
`latency_ms` was reported and hybrid costs **+79.7 ms** per query (2115.9 →
2195.6), which is above the project's 50 ms interactive budget. Had it been
declared a guardrail, the verdict for `hybrid_k10` would have been `REGRESSION`
by R5 — the same pattern Experiment A produced. This is stated here rather than
left for a reader to discover, and the artifact records the measured latency.

---

## 5. Statistical interpretation and limitations

The benchmark has **28 questions** and was used to CHOOSE these candidates. Every
result below is therefore exploratory.

- All comparisons are **paired** on identical question sets
  (`question_set_fingerprint = 822fd0a388b3491f`), and pairing validity is
  reported per comparison: all four experiments returned `COMPARABLE`.
- `comparison.exploratory` is **true** on every comparison, because the largest
  paired sample (28) is below the 30-question inference floor. The comparison
  module attaches that caveat automatically and the UI renders it.
- Where a test was run, it is the **exact sign test** (primary, no normality
  assumption) with the **paired t-test** reported alongside it as secondary. These
  are the values for each experiment's PRIMARY metric, recomputed from that
  metric's own pairing:

  | Experiment | metric | improved / regressed | n | p |
  | --- | --- | --- | --- | --- |
  | B, `dense_k10` | recall@k | 5 / 0 | 28 | **0.0625** |
  | B, `dense_k5` | recall@k | 4 / 0 | 28 | 0.125 |
  | A, `hybrid_reranked` | mrr | 3 / 4 | 28 | **1.0** |
  | C, `hybrid_k10` | mrr | 7 / 4 | 28 | 0.549 |
  | C, `bm25_k10` | mrr | 5 / 9 | 28 | 0.424 |

  **No claim of statistical significance is made anywhere.** The strongest
  measured effect (recall@k +0.1786, 5 questions improved, 0 regressed) has
  p = 0.0625 on 28 questions — the best possible one-sided outcome at that
  sample size, and still above 0.05.

  Note on structure: `ExperimentComparison.significance_tests` is ONE flat list
  covering every compared metric, so reading "the" sign test out of it by name
  would silently report some other metric's test. The primary metric's test is
  therefore recomputed from its own pairing and stored on the candidate outcome as
  `primary_sign_test_p` / `primary_sign_test_n`, together with the oriented
  `paired_wins` / `paired_losses`.
- `first_relevant_rank` comparisons are structurally under-informative at small
  `k`: at `k=3`, 5 questions have no measured rank at all, so they are excluded
  from the pairing (`paired_count = 23`). The comparison therefore omits exactly
  the questions that the k increase fixed. Its `mean_delta` of 0.0 and `UNKNOWN`
  verdict are the honest output of that, not a bug — `compare_metric` reports
  `UNKNOWN` for an all-ties pairing because "no evidence of a difference" is not
  "proof of none".
- No generated answers, no LLM-as-judge and no answer-quality benchmark were used
  as retrieval ground truth. `benchmarks/answer-quality-automobile-v1.json` was
  not read, and answer-quality metrics remain `UNKNOWN` (0/28 authored reference
  answers at the V11 checkpoint).
- **No expected relevance label was changed.** Both frozen benchmarks were
  hashed before and after every run: `08601f58b33c…28a656` unchanged.
- **Shared required chunks** remain a V12 caveat that also limits these results:
  7 chunks are required by more than one question, so a single ranking error can
  count as several question failures.

### Reproducibility evidence

- The V13 engine reproduces the V12 `<k=3>` and `<k=10>` dense numbers exactly
  (recall 0.8214 / 1.0, MRR 0.7083 / 0.7429, nDCG 0.7377 / 0.8057, and the same
  classification counts 5+6+17 / 0+11+17).
- B-strict and B ran the same arms twice. Every retrieval-quality mean is
  identical across the two runs (dense k=3: recall 0.8214, MRR 0.7083, nDCG 0.7377;
  k=5: 0.9643 / 0.7369 / 0.7929; k=10: 1.0000 / 0.7429 / 0.8057) and the
  classification counts match. Only wall-clock latency differs between them
  (e.g. 2115.0 ms vs 2112.5 ms at k=3), which is expected for a timing measurement.
  The two artifacts differ in the declaration and in the computed verdict.
- Re-running the same declaration produces a new artifact id but the same
  `question_set_fingerprint`, benchmark fingerprint, arm `config_fingerprint`s and
  pairing verdict (covered by `test_re_running_the_same_declaration_is_reproducible`).

---

## 6. Integrity

Every experiment artifact carries a before/after integrity snapshot. Example
(identical in all four):

```json
{"kb_id": "kb_f278c283c748", "is_protected_kb": true,
 "index_points_before": 812, "index_points_after": 812,
 "chunk_rows_before": 812, "chunk_rows_after": 812,
 "corpus_revision_before": 0, "corpus_revision_after": 0,
 "benchmark_sha256_before": "08601f58b33c…", "benchmark_sha256_after": "08601f58b33c…",
 "index_unchanged": true, "corpus_unchanged": true, "benchmark_unchanged": true,
 "scratch_collections_created": 0, "mutating_operations": 0}
```

Why no scratch collection was needed: every shipped strategy is read-only with
respect to the index — it embeds the query, searches, fuses, optionally reranks
and returns. No arm writes a chunk, a vector or a collection, and none saves a
retrieval configuration. The counters above are measurements taken around the
run, not promises. A future arm that needed to change the index would have to use
an isolated scratch collection and say so.

---

## 7. API and UI

### API (all additive)

```
GET  /api/knowledge-bases/{kb}/retrieval-experiments/meta
GET  /api/knowledge-bases/{kb}/retrieval-experiments
GET  /api/knowledge-bases/{kb}/retrieval-experiments/{experiment_id}
POST /api/knowledge-bases/{kb}/retrieval-experiments
```

`POST` runs the arms and persists the artifact; the three `GET`s are pure reads.
A request with an unknown primary metric, a missing hypothesis or a benchmark
belonging to a different knowledge base is refused with 400.

### Declaring and running experiments reproducibly

The four shipped experiments are DEFINED IN THE REPOSITORY —
`backend/scripts/run_retrieval_experiment.py`, `EXPERIMENTS` — so the hypothesis
and the bar are versioned alongside the code that measures them:

```bash
cd backend
py -m scripts.run_retrieval_experiment --list
py -m scripts.run_retrieval_experiment --experiment b-depth
py -m scripts.run_retrieval_experiment --experiment all
```

Artifacts are written to `backend/data/experiments/<slug>.json` (local,
gitignored, alongside the V12 diagnostic reports), each with an
`_artifact_sha256`.

### UI

The **Retrieval Diagnostics** page gained a "Controlled retrieval experiments
(V13)" panel — the existing page was extended rather than a second dashboard
being built. It renders, for the selected experiment:

- the declaration ABOVE the result: hypothesis, primary metric and its bar,
  guardrails with their bars, targeted failure code, rejection conditions, and
  the benchmark + question-set fingerprints;
- the measured arms table (`k`, probe depth, recall@k, MRR, nDCG, prec@k, latency);
- the derived outcome with its acceptance, pairing verdict, primary delta against
  the declared bar, per-guardrail assessment and the `rules_applied`;
- the per-question table for the primary metric (baseline / candidate / delta / outcome);
- the integrity line, including `promoted to default: false`;
- the limitations.

The panel is **read-only and has no "run" button**, deliberately: an experiment is
a hypothesis declared before measurement, not a knob to twiddle until a score
moves. A metric with no measured value renders as `UNKNOWN` with its reason —
never as `0`.

---

## 8. Recommendations and traceability

V12's recommendations are unchanged and remain `experiment_required`, with
`proposed_config_change` always empty. V13 supplies the missing link: each
recommendation now has an experiment that tests it and a recorded result.

| V12 recommendation | Failing questions (measured) | Hypothesis | Experiment | Result |
| --- | --- | --- | --- | --- |
| "increase top-k for one question" | `LOW_RECALL` at k=3: `auto-eng-003`, `-007`, `-011`, `-012`, `-015`, `-019` | the required evidence is below a 3-item window, not absent | B-strict, B | **ACCEPTED** in B (`dense_k10`, recall 0.8214 → 1.0, 5 questions fixed, 0 regressed); **REJECTED** in B-strict on the precision@k guardrail. `promoted_to_default: false` |
| "investigate reranking for ranking failures" | 11 `RANKING_FAILURE` at k=10 | MRR/nDCG rise, recall unchanged | A | **REJECTED / DISPROVEN.** MRR +0.0083 (bar 0.02, p = 1.0), recall **fell** to 0.9643, latency +1.5 s. Reranking is not coverage-neutral when the returned window is smaller than the reranked pool. |
| (V12 hybrid already measured) | 8 `RANKING_FAILURE` for hybrid | dense saturates recall, so hybrid can only re-rank | C | **ACCEPTED** for `hybrid_k10` (MRR 0.7429 → 0.8173, +0.0744, recall unchanged). **REJECTED** for `bm25_k10` (recall −0.0714, MRR −0.0273). |

A disproven hypothesis is recorded as disproven. The reranking recommendation is
**not** left standing as if it were established: Experiment A rejected it on this
benchmark and this corpus, with the mechanism written down.

The four statement kinds are kept distinct in the artifacts and the UI:

| Kind | Example |
| --- | --- |
| MEASURED FACT | `dense k=10` recall@10 = 1.0000 over 28 questions; latency 2115.9 ms |
| DIAGNOSTIC INFERENCE | 11 questions classified `RANKING_FAILURE` (rule + observations recorded) |
| PROPOSED CHANGE | "apply the cross-encoder reranker to the fused pool" (declared, not applied) |
| EXPERIMENT RESULT | Experiment A: `REGRESSION`, MRR +0.0083 < bar 0.02, recall −0.0357 |

---

## 9. Tests

`backend/tests/test_v13_retrieval_experiments.py` — **39 tests**, all passing:

- outcome rules: every branch R1–R7, including that a failed arm is `BLOCKED` and
  that a guardrail regression outranks a primary gain;
- the declared bar: guardrail noise below its bar does not veto a genuine gain,
  and a regression above its bar does reject;
- metric direction and correctness (higher- vs lower-is-better, `UNKNOWN` handling);
- unmeasured values excluded from means, never counted as `0`;
- reproducibility: the same declaration yields the same question-set fingerprint,
  benchmark fingerprint, arm `config_fingerprint`s and pairing verdict, under a
  NEW artifact id;
- matched question sets and `COMPARABLE` pairing through the API;
- strategy unavailability: an unknown strategy FAILS its arm, produces no
  metrics, and blocks the experiment rather than substituting numbers;
- a benchmark belonging to another knowledge base is refused (400);
- persistence and re-readability of the artifact and its list/detail endpoints;
- statistical output: `exploratory` is true and the sign test is present;
- integrity: index, corpus and both frozen benchmark artifacts byte-identical
  after a run; no artifact written into the versioned `benchmarks/` directory;
- experiments do not appear as diagnostic runs and vice versa.

Full suite: **1210 backend tests passing**. Frontend `tsc --noEmit` clean;
`next build` clean.

---

## 10. Known limitations

- **28 questions.** Every result is exploratory; no significance is claimed.
- **One corpus.** All four experiments ran on one knowledge base. Nothing here
  generalises to other domains or corpora.
- **`precision@k` is a lower bound** under necessity-only binary labels. It is
  reported with that caveat and, in the corrected declaration, not guardrailed.
- **`first_relevant_rank` is uninformative at small `k`** because unmeasured
  questions drop out of the pairing (§5).
- **Answer quality is out of scope.** No generator ran; answer-correctness
  metrics remain `UNKNOWN` pending the 28 authored reference answers that do not
  exist yet. The human-review workflow is a separate, still-open task.
- **Promotion is manual.** No experiment changes the default configuration; the
  accepted changes in B and C are recorded, not applied.
- **Latency is single-machine wall clock** on CPU embedding, so absolute
  milliseconds are environment-specific and vary between runs of the same arms
  (Experiment A measured +1498.8 ms and +2500.4 ms on the same machine). The
  transferable part is the RATIO, not the figure: BM25 ≈ 48-55x faster than dense;
  cross-encoder reranking ≈ +1.5–2.5 s per query.
- **The UI has no run control.** Declaring and running an experiment is done
  through the API or the script; an experiment is a declaration, not a form.

## 11. Next scientifically justified step

The `IMPROVED` verdicts in B and C are single-benchmark, single-corpus and
exploratory, and the benchmark was used to CHOOSE these candidates. The next step
is therefore **not** to promote `hybrid_k10` or `dense_k10`, and not to sweep
fusion weights over the same 28 questions.

It is to **build a held-out validation set of independently authored questions**
for the same corpus, freeze it the same way, and re-run Experiments B and C
against it unchanged. Only if the accepted changes hold on questions that were
not used to select them does the result become a finding rather than a fit. The
reranking hypothesis (A) is rejected as stated, but one narrower follow-up is
justified by the mechanism found: rerank only the top-`k` candidates that will be
returned (`reranker_top_n == top_k`), which cannot push a required chunk out of
the window, and test whether the MRR gain survives that constraint.
