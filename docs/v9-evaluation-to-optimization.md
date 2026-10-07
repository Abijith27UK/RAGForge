# V9 — Closing the Evaluation → Optimization Loop

**Branch:** `Dev_1_midterm` · **Base:** `67c4a00 feat: completed V8 answer evaluation and reliability`

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Audit report | **DONE** — [`v9-audit-report.md`](./v9-audit-report.md) |
| 1 | Benchmark review workflow | **DONE** (gates + append-only review) — measurement pending human review |
| 2 | Failure taxonomy | **DONE** — [`failure-analysis.md`](./failure-analysis.md) |
| 3 | Retrieval diagnostics | **DONE** — [`retrieval-diagnostics.md`](./retrieval-diagnostics.md) |
| 4 | Controlled experiment record | **DONE** (record + integrity). Live run **NOT RUN** |
| 5 | Recommendation engine | **DONE** |
| 6 | Baseline vs candidate | **DONE** |
| 7 | Statistical validity | **DONE** (exact tests, labelled exploratory) |
| 8 | Off-domain gate | **IMPLEMENTED; NOT MEASURABLE AT n_off = 1** |
| 9 | Mock-generator limitation study | **DONE** |
| 10 | UI "Why did this answer fail?" | **NOT DONE** — backend only |
| 11 | Experiment integrity | **DONE** |
| 12 | Testing | **DONE** — 978 backend tests; frontend clean |
| 13 | Real Automobile experiment | **NOT RUN** — precondition unmet (§7) |
| 14 | Documentation | **THIS DOCUMENT + 2 others + updates** |

The four status words are used strictly:
**IMPLEMENTED** = code + tests exist · **MEASURED** = a real number was produced
· **UNKNOWN** = not determinable from available data · · **NOT RUN** = the code
exists but no live execution happened.

---

## 1. V8 was damaged during a prior attempt, and was restored first

Before any V9 work could be trusted, the V8 baseline had to be re-established.

Seven V8 modules had been **overwritten destructively** rather than extended:

```
backend/app/services/answer_eval/{__init__,benchmark,evaluator,metrics,review,run,service}.py
```

The rewrites deleted the V8 public API (`Answerability`, `DeterministicAnswerEvaluator`,
`build_run`'s original signature, `AnswerQualityResult`'s behavioural fields,
`CLAIM_EVALUATED_STATES`, `SCORE_INPUTS`, `average_measured`, …) while leaving
callers and ~180 tests depending on it. `import app.services.answer_eval` raised
`NameError: name 'EvaluationService' is not defined` — the package did not import
at all, so **no V9 verification was possible**.

**Repair:** `git checkout --` on those seven paths restored them byte-for-byte
from `67c4a00`. Net effect: `-2327 / +846` lines of damage reverted.

**Verified baseline:** `692 passed, 1 skipped` with Qdrant down; `693 passed`
with Qdrant up. (The skip is the live-Qdrant integration test.)

**One V8 file was then changed, minimally and additively** — see §4. Every V8
test was re-run afterwards and the whole suite is green.

---

## 2. Failure taxonomy, diagnostics, recommendations

Three modules form the analysis layer. Details and rationale live in their own
documents; the summary of what they make possible:

```
evaluated question
   -> QuestionDiagnostic        (what retrieval returned, with provenance and UNKNOWN metrics)
   -> FailureClassification     (one dominant code, its evidence, and a categorical confidence)
   -> Recommendation            (a PROPOSED config change, with expected effect and risk)
   -> ExperimentRecord          (a controlled measurement of that proposal)
   -> AcceptanceRationale       (accept / reject, from declared criteria)
```

Each layer is pure and deterministic, and each refuses to guess:

- **Diagnostics** are UNKNOWN when the benchmark carries no reference evidence —
  `retrieval_hit_rate is None` means "no ground truth", so no retrieval fault can
  be asserted from it.
- **The taxonomy** falls through to `UNKNOWN` rather than choosing a label.
- **Recommendations** fire only on measured values, always carry
  `experiment_required=True`, and name every threshold they read in
  `thresholds_used`. A recommendation **cannot** modify production retrieval
  configuration — nothing in the module has a write path.

### The one rule that outranks all tuning rules

If the required evidence is **nowhere in the candidate set**, the engine reports
`CORPUS_OR_SOURCE_DEFICIENCY` with an **empty** `proposed_config_change`. Tuning
`top_k`, fusion or rerank cannot surface a passage the index does not contain;
recommending a knob turn there sends a reader chasing a change that cannot help.
A test asserts the empty config change explicitly.

---

## 3. Comparison and statistics

`comparison.py` implements baseline-vs-candidate comparison under five rules:

1. **No single magic score.** `compare_suite_explicit` returns one row per
   metric. A large nDCG gain cannot hide a hallucination-rate regression. A test
   asserts the result has no `overall_score` attribute.
2. **Direction is declared, not assumed.** `METRIC_DIRECTIONS` maps every known
   metric to higher- or lower-is-better; an undeclared metric raises rather than
   defaulting, because assuming the direction wrong **inverts the verdict**.
   `unsupported_claim_rate` dropping is an improvement; `recall_at_k` dropping
   is a regression.
3. **Pairing is validated.** `check_protocol` compares benchmark fingerprint,
   corpus fingerprint, evaluator, model, prompt version and answer mode. A
   difference makes the comparison `NOT_COMPARABLE`, and a mismatched experiment
   **can never be accepted** (`AcceptanceDecision.NOT_VALID`).
4. **UNKNOWN ≠ UNCHANGED.** No measured pair, all ties, or a split with zero net
   movement all yield `UNKNOWN` with a reason. Only questions measured on **both**
   sides are paired; `None` is never treated as `0.0`.
5. **Acceptance needs declared criteria.** The caller supplies
   `required_improvements` and `guardrail_metrics`. Any guardrail regression is a
   `REJECT` regardless of gains elsewhere.

### Statistics are stdlib-only and verified against scipy

`scipy` and `numpy` are installed in the venv but are **not declared** in
`requirements.txt`, so the implementation depends on neither:

- **Exact two-sided sign test** — `math.comb`, assumption-light, reported as the
  **primary** test.
- **Student's t** — implemented via the regularised incomplete beta (Lentz's
  continued-fraction method for `betacf`), **not** a normal approximation.

Both were checked against scipy as an oracle across a grid:

| Test | Max absolute difference vs scipy |
|---|---|
| `student_t_two_sided_p` (8 df × 10 t values) | **2.27e-14** |
| `exact_sign_test_p` (15 × 15 grid) | **1.11e-16** |

The oracle tests `pytest.importorskip("scipy.stats")`, so the suite still passes
without scipy.

A regression test pins the reason a normal approximation was rejected: at
`df=1, t=3` it understates p by more than a factor of two, which would let a weak
result be described as significant.

### Statistical honesty at n = 28

`MIN_RELIABLE_PAIRED_N = 30`. The shipped benchmark has **28** questions, so
**every result is `exploratory = True`** and the comparison appends:

> *"largest paired sample is N question(s), below the 30-question floor for
> inference: every statistical result here is EXPLORATORY and must not be
> reported as significance"*

Both tests carry their assumptions as text. No significance is claimed anywhere.

---

## 4. Benchmark review (Phase 1)

The shipped answer benchmark is
`benchmarks/answer-quality-automobile-v1.json`: 28 questions, **zero**
human-authored reference answers, `human_review: pending`, no `lifecycle` key →
loads as **DRAFT**. This is unchanged and correct: the labels were selected by an
agent reading real indexed chunk text, which is real work and real provenance,
but it is **not human review**.

> **Correction to the phase-0 audit.** The first draft of
> `v9-audit-report.md` named `automobile-engineering-baseline-v1.json` as the
> answer benchmark. That file is the **retrieval** benchmark — its questions
> carry `expected_chunk_ids` and `id`, not `question_id`, and Pydantic rejects all
> 28 rows. The error was found by the Phase-1 test and the audit has been fixed.

### Two separate gates

| Gate | Claim it protects |
|---|---|
| `require_official_benchmark` | the **FILE** will not change, so results stay reproducible (requires FROZEN). V8. |
| `require_reviewed_benchmark` | each **QUESTION** was checked by a human: answerability, evidence sufficiency, reference answer, key points, ambiguity. **NEW in V9.** |

Splitting them is deliberate: a benchmark can be frozen while some questions were
never reviewed, and a frozen-but-unreviewed question is exactly how an
agent-authored label becomes mistaken for human ground truth. The shipped
artifact is refused by **both** gates, and tests pin that refusal.

### Append-only review

`QuestionReviewLog.append` is the **only** mutation. There is no update, no
delete, no replace; a correction is a new review carrying `supersedes`, so the
record that a label was once claimed differently survives. A duplicate
`review_id` raises rather than merging.

`is_approval()` is deliberately strict: **every** required dimension must be
assessed and affirmative. `AMBIGUITY` is a **required** dimension — a question
nobody assessed for ambiguity is precisely the one that gets silently approved
with two defensible readings. `UNKNOWN` on a dimension blocks approval: a
reviewer who could not determine something has not approved it.

`apply_review_status` **downgrades** a question that claims `approved` with no
approval review behind it. That is the single most important property here, and
a test asserts it.

### No human labels were invented

28 questions remain unreviewed. `review_coverage()` reports the gap rather than
closing it.

---

## 5. Phase 8 — the off-domain gate: implemented, and honestly unmeasurable

**The framework works. The measurement cannot be made from available data, and
that is the finding.**

The gate study (`gate.py`) computes on/off distributions, false positives and
false negatives, and proposes a threshold **only** when the data supports one.
Two real defects were fixed:

1. **The median was not a median.** `sorted(v)[len(v) // 2]` returns the *upper*
   of the two middle values for even-sized samples, biasing any recommended
   threshold. Replaced with `statistics.median`.
2. **The score polarity was assumed, and assumed backwards.** The first draft
   compared `off_median > on_median`, i.e. it assumed a *higher-means-more-off-domain*
   axis. The signal this project actually measures — `question_answer_relevance`
   — is a **higher-means-more-on-domain** axis. Wired to real data, the
   separability test would have been **inverted** and would have recommended
   *lowering* the threshold. Polarity is now an explicit parameter
   (`HIGHER_MEANS_ON_DOMAIN` / `HIGHER_MEANS_OFF_DOMAIN`) that **must** be
   declared; an unknown value raises. A new `no_threshold_possible` outcome
   reports an inverted signal, because no threshold can fix a ranking that puts
   the wrong examples first.

### The measurement floor

`MIN_SAMPLES_PER_SIDE_FOR_A_RECOMMENDATION = 5`. The documented V8 measurement
has **exactly one** known off-domain answer, so:

```
reason: "insufficient samples to recommend a threshold: N on-domain and 1
off-domain, against a floor of 5 per side. No threshold is proposed, because
tuning a gate to fewer observations than this fits the sample rather than the
problem"
```

### The real V8 numbers, pinned as a regression guard

| Method | on-domain min | known off-domain | Verdict |
|---|---|---|---|
| Plain term coverage | 0.333 | **0.400** | **INVERTED** |
| IDF-weighted coverage | 0.300 | 0.260 | separates — margin 0.04 on **n = 1** |

These are real measurements from `services/answer_eval/relevance.py`, taken over
286 stored answers in `kb_f278c283c748` plus one known off-domain answer. The
tests reproduce both rows, so a future change that "fixes" the gate by reverting
to plain coverage **fails the suite**.

**What would make it measurable:** generate ≥ 5 genuinely off-domain questions
(≥ 10 to be comfortable), run them through the real answering pipeline against a
scratch KB, and score the same signal. That is a live run, not a computation.

---

## 6. Phase 9 — the mock-generator limitation

`mock_generator_study.py` documents a **limitation, not a fix**:
`ExtractiveMockAnswerGenerator` walks evidence in rank order with a **5-claim
budget**, so evidence ranked 4th or lower can be structurally **uncitable**. V8
observed this in real failures: the required chunk was retrieved at rank 5 and
never cited in 4 of 5 failures.

The study reports that the mock generator is a **test double only**, that
production evaluations do not depend on it, and recommends that **tests expose
the limitation** rather than that the budget be raised. Raising the budget would
tune the instrument to improve the number it produces — the exact circularity
V8 avoided and V9 preserves.

---

## 7. Phase 13 — the real experiment was NOT run, and why

The experiment framework is complete and verified offline (immutable store,
deterministic ids, checksums, vector-loss detection, acceptance decisions). The
**live** baseline-vs-candidate run is **NOT RUN**, because its preconditions are
not met:

1. **No trustworthy candidate to test.** The recommendation engine's rules need
   ground-truthed diagnostics. There is no measured pattern that justifies a
   specific configuration change, and inventing one to have something to run
   would be exactly the fabrication the project forbids.
2. **The benchmark is DRAFT / unreviewed.** 28 questions have no human review, so
   `require_reviewed_benchmark` refuses them. An answer-quality comparison run
   against them would produce numbers that cannot support a claim.
3. **`correctness` and `key_point_recall` are UNKNOWN** for all 28 questions, so
   the answer-metric half of the comparison would be UNKNOWN by construction.

Running it anyway would produce a record whose `conclusion()` reads
`INTEGRITY NOT VERIFIED` or `NO VALID COMPARISON`. Reporting that as a result
would be worse than reporting no experiment.

---

## 8. Superseded modules (deleted, and why)

Five modules from the interrupted attempt were **deleted rather than patched**,
because each was unsafe in a specific way:

| Deleted | Defect |
|---|---|
| `reporting.py` | Fabricated output: `tie_count`/`improved`/`regressed` were **hardcoded to 0**; `ComparisonVerdict.COMPARABLE = "IMPROVED"`; compared **run ids** where question ids were needed; `statistics.mean` would crash on `None`. |
| `statistics.py` | Superseded by `comparison.py`. Its "t-test" used a broken normal approximation (the Abramowitz–Stegun `d` factor was wrong, going negative for larger z) and would have produced **anti-conservative** p-values — an overclaim risk. |
| `controlled_experiment.py` | Duplicated `RetrievalParams` (drift risk), and its nDCG discount `1/(2^(i+1))^0.5` gives 0.707 at rank 1 where the correct value is 1.0 — contradicting `services/evaluation/metrics.py`. |
| `experiment_integrity.py` | Named `class ExperimentDotNet`; `fingerprint_dict` crashed on a missing `json` import; and **`_file_unchanged` / `_scratch_isolation` were stubs returning `True`** — reporting safety that was never verified. |
| `experiment_runner.py` | Inline `__import__("datetime")`; empty registry; superseded. |

The `_file_unchanged()` stub is worth calling out: a check that always returns
`True` is **worse than no check**, because it converts "unverified" into
"verified". The replacement hashes real files and compares digests, and a test
mutates an artifact and asserts the change is **detected**.

---

## 9. Safety

### The permanent rule

Project history contains an incident where a debug probe **deleted vectors from a
real KB**. V9 treats this as a permanent constraint:

- No function in `experiment.py` deletes, overwrites or upserts a vector. The
  only Qdrant calls are `get_collections()` and `count(exact=True)`.
- `PROTECTED_KB_IDS = {"kb_f278c283c748"}` — the real Automobile KB.
- `assert_scratch_isolation` **refuses** an experiment whose scratch KB is
  protected, or which would observe a protected KB, **before anything runs**.
- `verify_no_vector_loss` fails on any decrease in a collection **not explicitly
  declared** a scratch collection. The default allow-list is empty, so a missing
  declaration produces a false alarm rather than a missed deletion — the safe
  direction.
- An unavailable snapshot **cannot** claim "no vectors were deleted"; it reports
  the failure instead.

### Frozen artifacts

`FROZEN_BENCHMARK_FILES` names all 13 frozen `benchmarks/` artifacts.
`assert_not_frozen_artifact` refuses to write to them, and
`assert_frozen_artifacts_not_touched` refuses to accept one as a write target.
They were **verified byte-identical** to `HEAD` during this work.

### Measured Qdrant state

14 collections, 8823 points total; `kb_kb_f278c283c748` = **812 points**.
See [`retrieval-diagnostics.md`](./retrieval-diagnostics.md) for the full table.
**812 before, 812 after.**

---

## 10. Verification

| Check | Result |
|---|---|
| Backend tests collected | **978** (693 V8 baseline + **285 V9**) |
| **Full backend suite (final)** | **978 passed, 0 failed, 0 skipped** in 286s |
| V9-only suite × 9 files | **285 passed** |
| Frontend `tsc --noEmit` | **clean** (exit 0) |
| Frontend `npm run build` | **clean** |
| Statistics vs scipy oracle | t: **2.27e-14**, sign: **1.11e-16** max abs diff |
| Frozen artifacts modified | **0** |
| Qdrant collections before/after | **14 → 14** |

V9 test files and counts:

| File | Tests |
|---|---|
| `test_v9_comparison.py` | comparison, pairing, acceptance, scipy oracle |
| `test_v9_diagnostics.py` | diagnostics, UNKNOWN semantics |
| `test_v9_benchmark_review.py` | review workflow, provenance |
| `test_v9_recommendations.py` | recommendation rules, floors |
| `test_v9_experiment.py` | immutable store, integrity, vector loss |
| `test_v9_gate.py` | gate study, polarity, real V8 numbers |
| `test_failure_taxonomy.py` | 8 |
| `test_benchmark_review_workflow.py` | gates + shipped-artifact state |
| `test_mock_generator_study.py` | 3 |
| **Total** | **285** |

---

## 11. Known limitations

1. **No human labels.** 28/28 questions unreviewed; `correctness` and
   `key_point_recall` remain **UNKNOWN**. This is the single largest gap and only
   a human can close it.
2. **n = 28.** Every statistical result is exploratory. No significance is
   claimed or implied anywhere.
3. **Phase 13 not run.** See §7.
4. **No UI.** Phase 10 is **NOT DONE**; the analysis layer is backend-only, so
   the "Why did this answer fail?" view does not exist yet.
5. **Off-domain gate not measurable** at n_off = 1.
6. **No API routes for the V9 modules.** They are importable services with tests,
   but no HTTP surface exposes them yet.
7. **One Qdrant collection is degraded.** `kb_kb_3217fd29bc60` has 0 points and a
   stale-WAL startup error that predates this work. Not repaired.
8. **No reranker-stage diagnostics** — a pre/post-rerank rank comparison is not
   built.
9. **Binary-relevance nDCG only** — no graded relevance exists in the benchmark.
10. **The mock generator's rank truncation remains unfixed**, by design.

---

## 12. Where V9 sits against the research question

> *Can the construction of a domain-specific RAG knowledge base be automated
> from a high-level domain specification while maintaining source authority,
> knowledge coverage, retrieval quality, and provenance?*

V9 answers a narrower question well: **the loop from measurement to a justified,
safe, recorded change now exists end to end** — diagnosis, recommendation,
controlled comparison, accept/reject — and every step **refuses to guess**. The
automation claim remains unproven, and it is unproven in a specific, reportable
way: the loop cannot produce a trustworthy improvement until the **benchmark**
can produce trustworthy ground truth.

That ordering is the honest result. Machinery that optimizes against a
fabricated benchmark is worse than machinery that reports it cannot yet.

---

## 13. What V10 should do next

**Recommended first:** author and review the 28 Automobile reference answers
through the Phase-1 workflow, then freeze, then run the Phase-13 experiment. This
is the only path that converts `correctness` and `key_point_recall` from UNKNOWN
into measured numbers, and it is the precondition every other improvement is
waiting on.

Then, in order:

1. **Phase 10 UI** — the "Why did this answer fail?" view over
   `QuestionDiagnostic` + `FailureClassification`, so the review in step 0 can
   actually be done efficiently.
2. **V9 API routes** — expose diagnostics, failure classifications,
   recommendations and experiment records over HTTP.
3. **Off-domain gate measurement** — author ≥ 10 off-domain questions, run them,
   measure FP/FN, and only then consider a threshold change.
4. **Reranker-stage diagnostics** — pre/post-rerank rank comparison.
5. **Naval Architecture support** — explicitly out of scope for V8 and V9.
