# V14 — Held-out validation of the V13 retrieval findings

**Status: PROVISIONAL / EXPLORATORY. Nothing here is a generalization claim, and
nothing here is promoted.**

| | |
|---|---|
| Instrument | `automobile-engineering-heldout-v1` v1 |
| Lifecycle | **provisional** → claim strength **exploratory** (no confirmatory claim is permitted) |
| Fingerprint | `c738f73c9fd52c4a` |
| Frozen artifact | [`benchmarks/automobile-engineering-heldout-v1.json`](../benchmarks/automobile-engineering-heldout-v1.json) |
| Frozen sha256 | `536e6541718ae5010a2d6f7400fc1f91d874d6d9edcb1d8bc1ca424c4ee795e4` |
| Frozen at / by | 2026-10-12T17:29:19.306865+00:00 / Abijith |
| Questions | 26 (11 subdomains; 12 mechanism, 6 terminology, 4 quantitative, 2 comparison, 2 enumeration; 10 advanced, 14 intermediate, 2 basic) |
| Human review | **25 of 26 decided** (25 approve, 0 edit, 0 reject), reviewer `Abijith`. `auto-eng-hout-007` carries **no decision** |
| Separation from the development set | 0 shared expected chunks; max lexical similarity 0.30 token / 0.3136 char over the last measured validation view |
| Held-out runs | `rex_0515afbb8c37` (b-depth), `rex_82bc3ef0ff37` (c-hybrid), `rex_6e736f52f8a0` (a-rerank-windowed) |
| Development reference runs | `rex_9f7038f3f6a7`, `rex_bba1f6b6fc40`, `rex_79c3a6282dbf` (28 questions, `automobile-engineering-baseline-v1`) |

## 1. What was run, and what was NOT changed

The three arms were run exactly as pre-registered in
[`backend/scripts/run_heldout_validation.py`](../backend/scripts/run_heldout_validation.py),
with the shipped fusion weights, the shipped probe depth and the shipped
evaluator. Each arm was retested against the configuration V13 left it in:

| Arm | Retests | Baseline | Candidates |
|---|---|---|---|
| `b-depth` | `v13-b-retrieval-depth-k3-k5-k10` | `dense_k3` | `dense_k5`, `dense_k10` |
| `c-hybrid` | `v13-c-dense-bm25-hybrid` | `dense_k10` | `bm25_k10`, `hybrid_k10` |
| `a-rerank-windowed` | `v13-a-cross-encoder-reranking` | `hybrid_k10` | `hybrid_reranked_windowed` |

The development baseline was re-measured in the same configuration, on the same
machine, in the same run, so the cross-instrument comparison is between two
measurements taken under identical conditions.

**Not changed after seeing any held-out number:** no hypothesis, metric,
threshold, fusion weight, probe depth or question set. `promoted_to_default` is
`false` on all six runs; promotion requires an explicit, separate decision that
has not been taken. Every artifact is stamped `instrument=held-out`,
`benchmark_lifecycle=provisional`, `claim_strength=exploratory`.

Why the tier is provisional rather than validated: the instrument was sealed
while one question still had no human decision, and freezing is a one-way door —
the tier cannot be raised afterwards, because a reviewer who has already seen
results is no longer blind to them. `auto-eng-hout-007` therefore stays
unreviewed in v1, and its per-question values below are reported flagged, never
treated as human-approved evidence.

## 2. Held-out results (26 questions, all exploratory)

### b-depth — primary metric `recall_at_k`, guardrail `latency_ms`

| Arm | recall@k | precision@k | MRR | nDCG@k | latency |
|---|---|---|---|---|---|
| `dense_k3` (baseline) | 0.8846 | 0.2949 | 0.7628 | 0.7944 | 2112.7 ms |
| `dense_k5` | **0.9231** | 0.1846 | 0.7705 | 0.8093 | 2112.5 ms |
| `dense_k10` | **0.9231** | 0.0923 | 0.7705 | 0.8093 | 2111.7 ms |

**Verdict: IMPROVED, accepted candidate `dense_k5`.** Mean `recall_at_k` delta
**+0.0385** against a declared bar of 0.02, with **1 question improved, 0
regressed, 25 unchanged**. Latency guardrail held (delta −0.15 ms). Exact sign
test **p = 0.327** (n = 26) — not significant, and the paired sample is below the
30-question inference floor, so this is a measured result on this benchmark and
nothing more. On held-out questions the effect **saturates at k = 5**: `k10` is
identical to `k5` on every metric.

### c-hybrid — primary metric `mrr`, guardrails `recall_at_k`, `latency_ms`

| Arm | recall@k | precision@k | MRR | nDCG@k | latency |
|---|---|---|---|---|---|
| `dense_k10` (baseline) | 0.9231 | 0.0923 | 0.7705 | 0.8093 | 2115.5 ms |
| `bm25_k10` | **1.0000** | 0.1000 | **1.0000** | **1.0000** | **43.4 ms** |
| `hybrid_k10` | **1.0000** | 0.1000 | 0.9615 | 0.9716 | 2170.1 ms |

**Verdict: IMPROVED, accepted candidate `bm25_k10`; `hybrid_k10` REJECTED.**
BM25 raised MRR by **+0.2295** (9 questions improved, 0 regressed), raised recall
by +0.0769, and was **2072 ms/query faster** (−98%). Sign test on MRR:
**p = 0.0039** (n = 26); recall p = 0.5; latency p < 0.0001.

`hybrid_k10` moved MRR the right way (**+0.1910**) but was rejected on a
declared guardrail: latency **2170.1 ms vs 2115.5 ms = +54.6 ms**, above the 50
ms/query bar. A gain on the primary metric does not license a regression on a
guardrail.

### a-rerank-windowed — primary metric `mrr`, guardrails `recall_at_k`, `latency_ms`

| Arm | recall@k | precision@k | MRR | nDCG@k | latency |
|---|---|---|---|---|---|
| `hybrid_k10` (baseline) | 1.0000 | 0.1000 | 0.9615 | 0.9716 | 2187.8 ms |
| `hybrid_reranked_windowed` | 1.0000 | 0.1000 | **1.0000** | **1.0000** | 3411.1 ms |

**Verdict: REGRESSION; rejected.** The windowed reranker achieved **perfect
ranking** (MRR and nDCG both 1.0, 2 questions improved) and preserved recall
exactly as predicted — recall is invariant by construction when the reranked
slice equals the returned window. It fails on the declared latency guardrail:
**+1223.3 ms/query (+56%)**, 26 of 26 questions slower.

## 3. Development reference (28 questions), same configuration and machine

| Arm | recall@k | precision@k | MRR | nDCG@k | latency |
|---|---|---|---|---|---|
| `b-depth` `dense_k3` | 0.8214 | 0.2857 | 0.7083 | 0.7377 | 2116.1 ms |
| `b-depth` `dense_k5` | 0.9643 | 0.2000 | 0.7369 | 0.7929 | 2106.4 ms |
| `b-depth` `dense_k10` | 1.0000 | 0.1036 | 0.7429 | 0.8057 | 2109.0 ms |
| `c-hybrid` `dense_k10` | 1.0000 | 0.1036 | 0.7429 | 0.8057 | 2114.6 ms |
| `c-hybrid` `bm25_k10` | 0.9286 | 0.0964 | 0.7156 | 0.7670 | 40.4 ms |
| `c-hybrid` `hybrid_k10` | 1.0000 | 0.1036 | 0.8173 | 0.8618 | 2159.5 ms |
| `a-rerank` `hybrid_k10` | 1.0000 | 0.1036 | 0.8173 | 0.8618 | 2168.6 ms |
| `a-rerank` `hybrid_reranked_windowed` | 1.0000 | 0.1036 | 0.8313 | 0.8719 | 3249.3 ms |

Development verdicts: `b-depth` **IMPROVED** (+0.1786 recall for `dense_k10`, 5
questions improved), `c-hybrid` **IMPROVED** for `hybrid_k10` (+0.0744 MRR) and
**REGRESSION** for `bm25_k10` (−0.0273 MRR, recall guardrail −0.0714),
`a-rerank-windowed` **REGRESSION** (+0.0140 MRR, latency guardrail +1080.7 ms).
These reproduce the V13 verdicts, which is the control: the pipeline that
produced the V13 findings produces them again.

## 4. Held-out versus development, arm by arm

| Arm | Held-out delta | Held-out verdict | Dev delta | Dev verdict | Direction |
|---|---|---|---|---|---|
| `dense_k5` (recall) | **+0.0385** | IMPROVED | +0.1429 | IMPROVED | **agrees** |
| `dense_k10` (recall) | **+0.0385** | IMPROVED | +0.1786 | IMPROVED | **agrees** |
| `bm25_k10` (MRR) | **+0.2295** | IMPROVED | −0.0273 | REGRESSION | **DIFFERS** |
| `hybrid_k10` (MRR) | +0.1910 | REGRESSION (latency guardrail) | +0.0744 | IMPROVED | metric direction agrees, verdict differs |
| `hybrid_reranked_windowed` (MRR) | +0.0385 | REGRESSION (latency guardrail) | +0.0140 | REGRESSION | **agrees** |

What replicates and what does not, stated plainly:

* **The depth/window finding replicates in direction but not in size.** The
  claim that low recall at k = 3 is a window effect survives unseen questions,
  but on held-out questions the effect is one improved question out of 26
  (+0.0385) against five out of 28 (+0.1786) on the questions that chose it, and
  it saturates at k = 5 rather than k = 10.
* **The BM25 rejection does not replicate — it inverts.** On the development
  questions BM25 loses recall and MRR; on held-out questions BM25 retrieves every
  required chunk at rank 1 (MRR 1.0) while being ~48× faster. One instrument
  says "reject", the other says "this is the best ranker measured here". That
  disagreement is the most important result in this document, and it is a
  statement about the two instruments as much as about BM25.
* **The hybrid acceptance does not survive its own guardrail.** V13 accepted
  hybrid on ranking and reported its latency as a post-hoc caveat; with latency
  declared as a guardrail *before* the run, the same gain is a rejection. This is
  exactly the hole the V14 declaration was written to close, and closing it
  changed the verdict while the metric moved the same way.
* **The reranker rejection replicates, for a different reason.** V13 rejected it
  for losing recall; the windowed variant does not lose recall and instead
  reaches perfect ranking at 1.56× the latency, so it is rejected on the latency
  guardrail instead. Same verdict, different mechanism, both honest.

## 5. Mechanism: what the two dense misses were

`dense_k10` fails to return the required chunk for exactly two held-out
questions: **`auto-eng-hout-007`** (the unreviewed one) and
**`auto-eng-hout-018`**. Both are recovered by lexical matching at rank 1
(BM25 MRR 1.0) and both are lifted from rank 2 to rank 1 by the reranker.
Recall is identical for `dense_k3..k10` on these two questions, so widening the
window does not help them — the evidence is not merely outside the window.

Measured lexical anchoring (token Jaccard between each question and the text of
its expected chunk) puts those two questions among the *least* anchored in the
set:

| Benchmark | mean | median | min | max |
|---|---|---|---|---|
| Held-out (26) | 0.1293 | 0.1212 | 0.0645 | 0.2273 |
| Development (28) | 0.0696 | 0.0693 | 0.0333 | 0.1231 |

`auto-eng-hout-007` sits at 0.067 and `auto-eng-hout-018` at 0.114. Note the
direction of this measurement carefully: held-out questions are on average
**1.9× more lexically anchored to their evidence than development questions**,
which structurally favours exact-term matching and is a plausible partial
explanation for BM25's perfect score. But the two questions dense misses are
among the *least* anchored, so lexical anchoring does not explain the dense
failures — those look like single-rare-term questions (a name, a date, a
standard) that embeddings dilute and lexical matching hits exactly.

## 6. Uncertainty, and what may not be claimed

* Both instruments are **below the 30-question floor** the comparison module uses
  for inference (`MIN_RELIABLE_PAIRED_N = 30`): 26 held-out, 28 development.
  Every statistical result above is therefore stamped `exploratory` and
  **must not be reported as significance**, including the small p-values
  (p = 0.0039 for BM25 MRR; p < 0.0001 for both latency effects).
* The held-out instrument is **provisional**: one question has no human
  decision, so no confirmatory or generalization claim is permitted from it.
* **Retrieval only.** No generator ran, so answer correctness, citation quality
  and abstention are out of scope; no answer-quality claim is made.
* **Latency is single-machine** with CPU embedding (dense ≈ 2.1 s/query is
  dominated by embedding the query on CPU, BM25 ≈ 43 ms needs no embedding).
  Absolute milliseconds are environment-specific; the 48× ratio is a property of
  this deployment, not of the algorithms in general.
* **One corpus, one domain.** 11 documents, 812 chunks, automobile engineering.
  Nothing here generalises to another knowledge base.
* **Drafting provenance.** The questions were drafted by an agent from the full
  text of named chunks, and the same agent ran the V13 experiments; the artifact
  records this, and the qualitative direction of tooling decisions must not be
  attributed to an independent evaluator.

## 7. Highest-impact remaining weakness, and the smallest next experiment

The largest single weakness is not any arm's metric: it is that **the two
available instruments disagree about the best ranker, and the deciding
difference (BM25 vs dense) is concentrated in a handful of questions.** Two
questions out of 26 determine BM25's perfect MRR; one question out of 26 carries
the entire depth improvement; two questions carry the reranker's gain.

The smallest justified next experiment is therefore a **mechanism probe, not a
new configuration**, using tooling that already exists (the V12 retrieval
diagnostics, which probes to depth 20 and classifies per question):

* **Hypothesis.** For `auto-eng-hout-007` and `auto-eng-hout-018`, the required
  chunk IS present in the probed candidate pool (depth 20) but ranked below 10,
  i.e. the dense failure is a RANKING failure, not a representation failure.
* **Control.** Same frozen instrument, same questions, same `dense_k10`
  configuration, same probe depth and evaluator; no retriever, weight or
  threshold is touched.
* **Metrics.** Per-question rank of the required chunk inside the pool, presence
  in the pool, the diagnostic classification and its stated limitations.
* **Acceptance.** The mechanism is identified for both questions: if the chunk is
  in the pool but ranked low, the lever is ranking (rerank the *pool*, or fuse
  scores) and the reranker's +2 perfect ranks become the relevant evidence; if
  the chunk is absent from the pool, the lever is the embedding, and no reranker
  or window change can recover it.
* **Cost.** One diagnostic run, no configuration change, no tuning, and it can be
  repeated identically on the development instrument for comparison.

A second, larger decision that the evidence now supports but does not settle:
this corpus' held-out questions were answered perfectly by a 43 ms lexical
retriever, so the *default* retrieval configuration for a KB like this one is an
open engineering question. That is a promotion decision, and promotions are
deliberately not automatic — it should be taken explicitly, on a validated
instrument, after the mechanism above is known.

## 8. Integrity of the run

| Check | Before | After |
|---|---|---|
| `kb_f278c283c748` Qdrant points | 812 | **812** |
| Protected KB chunk rows | 812 | **812** (per-run integrity blocks) |
| Corpus revision | 0 | **0** |
| Qdrant collections | 14 | **14** (no scratch collection created) |
| `benchmarks/automobile-engineering-baseline-v1.json` | `08601f58…a656` | **unchanged** |
| `benchmarks/answer-quality-automobile-v1.json` | `16db9dbd…9f25` | **unchanged** |
| `benchmarks/automobile-engineering-heldout-v1.json` | `536e6541…95e4` | **unchanged** |
| `benchmarks/automobile-engineering-heldout-v1-candidates.json` | `e4a75253…be264` | **unchanged** |
| SQLite `PRAGMA integrity_check` | `ok` | **`ok`** |

Reproduce:

```bash
# from backend/
HF_HUB_OFFLINE=1 py -m scripts.run_heldout_validation --experiment all
```

The run refuses to start against a DRAFT instrument and stamps every artifact
with the tier it was run at (`--require-confirmatory` would refuse this
provisional instrument outright). Results are written to
`backend/data/experiments/` (`v14-*-heldout.json`, `*-dev-reference.json`,
`v14-generalization-*.json`) and exposed read-only through
`GET /api/knowledge-bases/{kb}/held-out/experiments`.
