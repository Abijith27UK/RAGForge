# Answer-Quality Evaluation (V8)

> What this system measures about a grounded answer, what it deliberately refuses
> to measure, and where the numbers come from.

RAGForge can already answer with citations and refuse to answer when the corpus
does not support an answer. **V8 asks the harder question: are those answers
actually any good?** The answer is a set of metrics — and, just as importantly, a
precise account of which ones do not exist yet.

---

## 1. The central finding

The task this work was scoped against assumed that answer correctness could be
measured from the existing benchmark. **It cannot.** The frozen retrieval
benchmark `benchmarks/automobile-engineering-baseline-v1.json` contains 28
human-reviewed questions with `expected_chunk_ids`, quoted source provenance and
subdomain labels — and **no expected answers, no key points, no expected
keywords**. Its own authorship field reads:

```
"human_review": "PENDING - selections were made by reading real indexed chunk
texts; a human must review before treating results as publication-grade"
```

So the only ground truth that genuinely exists is **which chunk answers the
question** — not **what the answer should say**.

Everything below follows from taking that seriously rather than working around
it.

---

## 2. Ground truth: evidence-based, not answer-based

`benchmarks/answer-quality-automobile-v1.json` is derived from the frozen
benchmark by a generator (`backend/scripts/build_answer_benchmark_v1.py`) that
**refuses to write if the frozen file is modified** and re-validates every
required chunk against the live corpus by content hash.

| Field | Value | Why |
|---|---|---|
| `questions` | 28 | Inherited from the frozen benchmark |
| `answerable` | 28 | Derived *only* from a human having selected an answering chunk |
| `unanswerable` | **0** | No human has authored any; must not be synthesised |
| `expected_answer` | **all null** | No human wrote one |
| `key_points` | **all empty** | No human wrote any |
| `human_review` | `pending` | Inherited honestly; never upgraded by an agent |
| `fingerprint` | `f72c30eb5c0c363a` | Content hash recorded on every run |

### Necessity, not exhaustiveness

`required_evidence[].required` is a **necessity** label: *a correct answer must
use this chunk*. It is **not** an exhaustive whitelist of the only chunks a
correct answer may cite. A human selecting one canonical passage out of a
Wikipedia article is not asserting the surrounding context is irrelevant.

This distinction decides two metrics:

- **`citation_recall` / `citation_completeness`** — strict and legitimate. These
  chunks really are necessary.
- **`citation_precision`** — a **lower bound**. Any extra retrieved context the
  answerer cites counts against it, even when that context is genuinely on-topic.

Getting this backwards is how a benchmark starts punishing answers for being
*well-sourced*. The UI labels it accordingly.

---

## 3. The metrics

All are `Measured`: a value **or** an explicit reason why there is none. UNKNOWN
is never rendered as `0`.

### Measured

| Metric | Definition | Per-question failure meaning |
|---|---|---|
| `citation_recall` | fraction of required chunks that were cited | necessary evidence went unused |
| `citation_completeness` | alias of recall | same |
| `citation_precision` | fraction of cited chunks that are required (**lower bound**) | — soft signal, never a hard failure |
| `evidence_support_rate` | judged claims judged SUPPORTED by the entailment provider | claims not backed by what they cite |
| `unsupported_claim_rate` | claims asserting fact with no resolving citation | ungrounded assertions |
| `contradiction_rate` | claims whose cited evidence was judged NOT_SUPPORTED | contradicted by its own source |
| `retrieval_hit_rate` | was the required evidence retrieved **at all** | separate defect — see §5 |
| `abstention_accuracy` | abstained exactly when it should | over/under-refusal |
| `grounding_state_accuracy` | produced state met its expectation | mislabelled confidence |
| `false_supported_rate` | answered without citing required evidence | confident answer, unsupported coverage |
| `false_unsupported_rate` | refused a question the corpus answers | over-abstention |
| `hallucination_rate` | either of the above | confident and wrong about coverage |
| `pass_rate` | citation + abstention + grounding checks all passed | — |

### Reported as UNKNOWN

| Metric | Why it cannot be computed |
|---|---|
| `correctness` | No human-authored reference answer exists. Scoring against retrieved text would re-measure **retrieval** and dress it up as answer quality. |
| `key_point_recall` | No human-authored key points. |

Approximating correctness with fuzzy string matching against the retrieved chunk
would be **the single most tempting and most dishonest move available here**.
It would produce a plausible-looking percentage that measures retrieval, not
answer quality. It is not implemented, and `AnswerQualityResult` refuses to
carry a `final_score` whose inputs are unmeasured.

---

## 4. What a "pass" actually requires

Strict where the benchmark has real ground truth; permissive where it does not.

**Answer level (hard failures)**
- required evidence not cited at all → `citation_recall < 1`
- `false_supported` / `false_unsupported`
- abstention behaviour wrong
- grounding state wrong
- no questions were excluded from the metrics

**Claim level (hard failures)**
- a factual claim with no citation
- a claim contradicted by the evidence it cites
- a citation pointing outside the retrieved evidence set

**Deliberately NOT a per-claim failure**
- *a claim does not cite every required chunk* — a claim about alternative fuels
  has no reason to cite the engine-definition chunk. Completeness is a property
  of the **answer**.
- *a claim cites a chunk the benchmark does not label as required* — the
  benchmark never established that chunk is irrelevant, only that a human picked
  a different canonical chunk. Asserting irrelevance would fabricate ground
  truth. Reported as an **observation**; its effect is already `citation_precision`.

Both signals are still **recorded on every claim verdict**
(`missing_required_chunk_ids`, `irrelevant_chunk_ids`) so failure analysis can
show them. Relaxing the *failure* never deleted the *diagnostic*.

---

## 5. Retrieval misses are not answerer failures

`retrieval_hit_rate` is computed and reported **separately** from citation
quality. If the required chunk was never retrieved, the answerer was never given
the chance to cite it, and folding that into the answer score would blame the
wrong component. The UI splits failures into *citation failure* and *retrieval
miss* for exactly this reason.

---

## 6. Reproducibility

Every run records, and the UI displays, **all** of it:

- benchmark name, version and **content fingerprint**
- exact `question_ids` covered + `subset_note`
- generator, model, prompt version, answerer version, `is_mock`
- evaluator name + version
- entailment provider + whether it is model-based
- retrieval strategy, params and run ids

`AnswerEvaluationRun.is_comparable_with()` **refuses** to compare runs over
different question subsets or different benchmark content, because attributing a
difference in question mix to a strategy is a way to invent a result.

`persist=false` suppresses only the **evaluation-run row**. Every question still
goes through the normal answering path and writes an `Answer` row — documented
in the OpenAPI description and in `run_real_answer_eval_v8.py`, because a flag
named `persist` that does not make a call read-only is a trap.

---

## 7. Reproducing the headline numbers

```bash
# End-to-end on a scratch KB (creates then deletes it, verifies Qdrant count)
cd backend
.venv/Scripts/python.exe scripts/smoke_answer_eval_v8.py

# The real 28-question benchmark against the real Automobile KB
.venv/Scripts/python.exe scripts/run_real_answer_eval_v8.py http://127.0.0.1:8015 dense

# Validate the benchmark artifact without writing anything
.venv/Scripts/python.exe scripts/build_answer_benchmark_v1.py --check
```

---

## 8. Known limitation found by this work

The measured run exposes a **real defect in the extractive mock generator**,
not in the evaluator or the benchmark.

`ExtractiveMockAnswerGenerator` (`backend/app/services/answering/generator.py`)
walks evidence in rank order and stops at a 5-claim budget, taking at most two
sentences per chunk. With `top_k=5` that makes **evidence ranked 4th or lower
structurally uncitable**.

Measured evidence: in 4 of the 5 failures the required chunk *was* retrieved at
**rank 5** and was never cited, while the top 3 chunks were.

This was **reported, not fixed**. Changing the generator's selection strategy
immediately before re-reporting the metric it affects is how a benchmark gets
tuned to look good. The correct sequence is: fix the generator as its own change,
then re-measure. The number below is the honest pre-fix figure.

---

## 9. Measured result

28 questions, dense retrieval, evaluator `deterministic-evidence v8.2`,
entailment `heuristic-lexical-coverage`, generator **mock**.

| Metric | Value |
|---|---|
| `pass_rate` | 0.821 |
| `citation_recall` | 0.821 |
| `citation_precision` | 0.286 *(lower bound)* |
| `evidence_support_rate` | 1.000 |
| `retrieval_hit_rate` | 0.964 |
| `abstention_accuracy` | 1.000 |
| `grounding_state_accuracy` | 1.000 |
| `unsupported_claim_rate` | 0.000 |
| `contradiction_rate` | 0.000 |
| `hallucination_rate` | 0.179 |
| `correctness` | **UNKNOWN** |
| `key_point_recall` | **UNKNOWN** |

Failures: 5 of 28 — 4 citation failures (`auto-eng-003`, `-011`, `-023`, `-028`),
1 retrieval miss (`auto-eng-015`).

**This is not evidence that answer quality is solved.** It is evidence that
citation completeness, grounding behaviour and abstention behaviour are now
*measurable*, over a corpus a human selected evidence for, with the parts that
are not measurable reported as such. The generator producing these answers is a
mock, and factual correctness is UNKNOWN for all 28 questions.