# Retrieval Diagnostics — why a question failed to retrieve

**Status: IMPLEMENTED and TESTED. Metrics are UNKNOWN when ground truth is
absent, never zero.**

Module: `backend/app/services/answer_eval/diagnostics.py`
Tests: `backend/tests/test_v9_diagnostics.py`

---

## What a diagnostic row contains

For each benchmark question, `QuestionDiagnostic` exposes everything a reviewer
needs to reason about retrieval without re-running anything:

| Group | Fields |
|---|---|
| Query | `question_id`, `question`, `subdomain`, `answerability` |
| Configuration | `strategy`, `top_k`, `retrieval_params` (persisted verbatim) |
| Results | `retrieved[]` with `rank`, `chunk_id`, `score`, `score_breakdown` |
| Provenance | per item: `document_id`, `source_id`, `source_title`, `source_url`, `source_type`, `publisher`, `section`, `page`, `trust_score`, `content_hash` |
| Ground truth | `required_chunk_ids`, `required_document_ids`, `ground_truth_available`, `first_relevant_rank`, `required_retrieved` |
| Generation input | `evidence_selected_ids`, `required_selected_ids`, `required_missing_from_generation`, per item: `is_required`, `selected_for_generation` |
| Metrics | `recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k`, `required_recall_at_depth` |
| Cost | `latency_ms`, `stage_timings`, `trace_id` |

A chunk with no entry in the chunk index is reported with **`None` provenance and
a `None` score** rather than a fabricated one.

---

## The UNKNOWN contract

Every metric is a `DiagnosticMetric` with the same three-field contract as V8's
answer-side `Measured`:

```python
DiagnosticMetric(value=None, measured=False,
                 reason="the benchmark names no required evidence for this "
                        "question, so there is no reference list to score "
                        "against — this is unknown, not zero")
```

`ground_truth_available` is `True` only when the benchmark names required
evidence for the question. When it is `False`, **all five metrics are UNKNOWN**
and `first_relevant_rank` / `required_retrieved` are `None`.

A test pins the distinction directly:

```python
no_gt      = build_question_diagnostic(required_chunk_ids=[])          # UNKNOWN
real_miss  = build_question_diagnostic(required_chunk_ids=["chk_x"])   # measured 0.0
assert no_gt.recall_at_k.measured is False and no_gt.recall_at_k.value is None
assert real_miss.recall_at_k.measured is True and real_miss.recall_at_k.value == 0.0
```

---

## Metrics are NOT reimplemented

`recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k` and `average_metrics` are
imported from `services/evaluation/metrics.py` — the module the retrieval
evaluator already uses. Answer evaluation and retrieval diagnostics therefore
**cannot disagree about what MRR means**, which is what happens when two modules
each grow their own copy.

(The first draft of the V9 experiment module *did* grow its own copy, with an
nDCG discount of `1 / (2 ** (i + 1)) ** 0.5`. For rank 1 that yields 0.707, where
the correct binary NDCG discount is `1/log2(2) = 1.0`. It was deleted; see
`docs/v9-evaluation-to-optimization.md` §"Superseded modules".)

---

## Depth-independent recall, and why it matters

`required_recall_at_depth` answers a different question from `recall_at_k`:

- `recall_at_k` — did the required evidence make it into the **top-k window**
  that generation actually saw?
- `required_recall_at_depth` — did it appear **anywhere** in the retrieved
  candidate set? It is depth-independent on purpose.

When the two disagree you learn something neither reports alone:

| `recall_at_k` | `required_recall_at_depth` | Finding |
|---|---|---|
| 0.0 | 0.0 | The passage is **absent** — a corpus/source deficiency |
| 0.0 | 1.0 | The passage is present but **ranked deep** — a depth problem |
| 1.0 | 1.0 | Retrieval succeeded |

The vocabulary around this is deliberately careful: a deep-ranked passage
**must not** produce the "corpus gap" warning, because it *was* retrieved. A test
asserts exactly that, since mislabelling depth as absence would send a reader to
fix the corpus instead of the ranking.

---

## Evidence selection is separated from retrieval

`required_missing_from_generation` lists required evidence that **was retrieved
but was not passed to the generator**. When that set is non-empty and those
chunks are in the retrieved list, the diagnostic emits:

> *"N required chunk(s) were retrieved but not selected for generation — an
> evidence-SELECTION fault, distinct from a retrieval miss."*

This distinction is load-bearing: a retrieval miss and a selection miss call for
completely different fixes, and the first draft of the recommendation engine
would have recommended tuning retrieval for both.

---

## Aggregate summary

`summarise_diagnostics()` produces a `DiagnosticSummary` that:

- reports `ground_truth_question_count` and `no_ground_truth_question_count`;
- computes each mean over **measured values only**, and says so in `reason`
  (`"; N question(s) excluded as unmeasured (NOT counted as 0)"`);
- lists every metric it could not compute in `unknown_metrics`;
- buckets `first_relevant_rank` into `rank_1`, `rank_2_3`, `rank_4_5`,
  `rank_6_10`, `rank_11_20`, `rank_21_plus`, `not_retrieved`;
- lists `no_result_question_ids` for questions that returned nothing;
- warns explicitly when questions lack ground truth:

> *"N of M question(s) have no benchmark reference evidence; their retrieval
> metrics are UNKNOWN and are excluded from the means above rather than counted
> as zero."*

An empty input list returns an UNKNOWN summary with
`warnings == ["no question diagnostics supplied"]` — never a zeroed one.

---

## Live evidence captured for this build

`snapshot_vector_counts()` was run against the local Qdrant during the V9 build
(read-only: `get_collections()` + `count(exact=True)`):

```
available: True
collection_count: 14
total_points: 8823
  kb_kb_3217fd29bc60              0
  kb_kb_334143bb8cca           1191
  kb_kb_4247706c2612            953
  kb_kb_49147f23c7c1            511
  kb_kb_58aee3f9b703            277
  kb_kb_761a8ed86369            431
  kb_kb_8dfa42d87061            431
  kb_kb_9ddec3658c96            949
  kb_kb_b1a989957b54            953
  kb_kb_c1a41117c392            386
  kb_kb_ca4af9e662d3           1191
  kb_kb_d88f8163fb5e            737
  kb_kb_f278c283c748            812     <- the PROTECTED Automobile KB
  kb_live_itest                   1
```

`kb_kb_f278c283c748` is the real Automobile knowledge base and is on the
protected list (`PROTECTED_KB_IDS`). **812 points before. 812 points after.**
Nothing in V9 wrote to it.

Qdrant logged one startup error about a stale WAL shard for
`kb_kb_3217fd29bc60` (`Can't init WAL: Kind(WouldBlock)`); that collection reports
**0 points** and predates this work. It is recorded here rather than hidden, and
V9 did not attempt to repair it.

---

## What is NOT implemented

- **Reranker-stage diagnostics.** `RetrievalTrace` records stage timings, and
  `RerankerStatus` reports `applied` / `not_requested` / `unavailable_fallback`,
  but a pre/post-rerank rank comparison per question is not built.
- **Query-decomposition attribution.** The multi-part question path is still a
  recorded warning (V8 limitation), so a diagnostic cannot yet say which
  sub-query failed.
- **Graded nDCG.** Only binary relevance is available, so `ndcg_at_k` is
  binary-relevance NDCG and is labelled as such.
