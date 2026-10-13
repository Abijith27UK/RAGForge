# V16 — Unlocking trustworthy answer-quality evaluation (2026-10-13)

**Objective.** Make it possible for a human to review the 28 answer-benchmark
questions safely, so RAGForge can later evaluate grounded answers, citation
correctness, unsupported claims and abstention — not just retrieval rankings.

**Outcome.** The review path is repaired and hardened, the draft assistance is
audited against the live corpus, and the lifecycle refuses to score unreviewed
work. **No reference answer, key point, review, approval or decision was created
by this work.** The benchmark is still `draft`: 28 pending, 0 approved, 0 frozen.

---

## 1. What the benchmark actually is (verified 2026-10-13, not carried over)

Read from the artifact, the live API (`GET .../answer-benchmark/review-packet`)
and the SQLite database, not from earlier reports.

| Fact | Value |
|---|---|
| Artifact | [`benchmarks/answer-quality-automobile-v1.json`](../benchmarks/answer-quality-automobile-v1.json) |
| Content fingerprint | `f72c30eb5c0c363a` |
| Lifecycle | `draft`; `human_review: pending`; `answer_labels: NONE` |
| Questions | **28** (`auto-eng-001` … `auto-eng-028`) |
| States | **28 pending**, 0 in review, 0 approved, 0 rejected, 0 ambiguous, 0 insufficient evidence |
| Authored reference answers / key points / acceptable elements | **0 / 0 / 0** |
| `benchmark_ground_truth` rows | **0** |
| `benchmark_question_reviews` rows | **0** |
| Frozen answer-benchmark versions | **0** |
| Stored draft suggestions (`benchmark_draft_suggestions`) | **29** across 28 questions (`auto-eng-001` has two) |
| Knowledge base | `kb_f278c283c748`, 812 Qdrant points, 812 chunk rows, 11 documents |

Every question already pins the evidence it rests on: a required chunk id, its
content hash, section, source title, plus the quoted passage the labels were
selected from (e.g. `auto-eng-001` → `chk_d15901e67e35`, *Engine – Wikipedia*,
section *Introduction*, pinned 2026-09-15). That provenance is what makes a
review possible at all.

### Measurable vs UNKNOWN today

| Metric | State | Why |
|---|---|---|
| Coverage (`answer_benchmark_coverage`) | **MEASURED** | 28 questions, labels, required evidence |
| Retrieval metrics (recall@k, MRR, nDCG) | **MEASURED** | frozen retrieval benchmark version |
| Answer correctness | **UNKNOWN** | requires a human-reviewed reference answer |
| Key-point recall | **UNKNOWN** | requires human-authored key points |
| Citation support / unsupported claims | **UNKNOWN** | requires reviewed reference answers |
| Abstention quality | **UNKNOWN** | requires questions marked as abstention cases |
| Reference-answer / key-point / acceptable-element / provenance coverage | **UNKNOWN** (`value: null`, `measured: false`) | no annotation exists, so there is no denominator |

The packet returns these four as UNKNOWN with a reason, never as 0.

### Defects found and fixed in this pass

1. **A review could approve text it never saw.** `derive_question_state` treated
   any approving review as approval of the *current* annotation. Correcting the
   reference answer after approval left the question APPROVED on the strength of
   a review of the previous text. Now: a review that names the annotation it
   judged is invalidated when the effective annotation changes (state drops to
   `in_review` with the reason). A review that names no annotation still cannot be
   invalidated automatically — that residual hole is now **disclosed in the packet
   warnings** instead of being silent.
2. **Placeholder and degenerate text could become scored ground truth.** `TODO`,
   `TBD`, `N/A`, `...`, `lorem ipsum`, a one-word answer, and duplicate key points
   (which would double-count in `key_point_recall`) are now recorded as
   annotation **problems**: stored, never approvable, and blocking at freeze time.
3. **The reviewer could not see the question's intended scope.** The packet and
   UI now carry the artifact's provenance text, citation requirements, expected
   grounding state, abstention requirement, units requirement and numeric
   tolerance.
4. **No queue, no progress, no keyboard flow.** The review page now has an
   "awaiting a decision only" filter, a per-state filter with counts, a position
   counter, keyboard navigation (`←`/`k`, `→`/`j`, `?`), and a documented
   guarantee that no shortcut records a decision.

---

## 2. Draft assistance, audited (Phase 3)

Read-only audit: [`backend/scripts/audit_answer_drafts.py`](../backend/scripts/audit_answer_drafts.py)
(`python -m scripts.audit_answer_drafts --kb kb_f278c283c748 --json <out>`).
It re-checks every stored draft against the **current** corpus — not against the
generator's own promises.

| Finding | Value |
|---|---|
| Drafts audited | 29 (28 questions) |
| Draft spans that are verbatim in the live corpus | **29 / 29** (0 problems) |
| Byte-exact without normalisation | 6 / 29; 23 match only under the drafting module's documented normalisation (stored text carries layout artefacts such as `"energy . [ 1 ] [ 2 ]"`) |
| Coverage measured (IDF-weighted) | 29 / 29 — min 0.048, p25 0.480, median 0.640, max 1.000 |
| Drafts below 0.60 coverage | **13** |
| Generator status | 24 `draft_ready`, 3 `partial` (`011`, `016`, `019`), 2 `insufficient_evidence` (`024`, `028`) |
| Drafts selecting the same sentence as the artifact's own unreviewed quote | 8 of 28 agree; **20 differ** |

**The gap this reveals, reported rather than filled:**

* `auto-eng-024` and `auto-eng-028` have **no quotable answer** in their pinned
  evidence (question terms present in the evidence: 2 of 9 and 1 of 10). A human
  must decide whether to cite more evidence or classify the question
  `insufficient_evidence` — the tool will not invent an answer.
* `auto-eng-011`, `016`, `019` are `partial`: the cited evidence covers 34 %, 42 %
  and 39 % of the question's weighted terms.
* 20 of 28 drafts disagree with the artifact's own unreviewed quote. Both are
  machine-produced; neither is ground truth, and the disagreement is a signal the
  reviewer must resolve, not a bug to be auto-resolved.
* Nothing in the draft path can approve, freeze or label anything as
  human-reviewed: `is_draft`, `requires_human_review` and the `DRAFT —
  MACHINE-GENERATED SUGGESTION, NOT HUMAN GROUND TRUTH` label travel with every
  suggestion, and drafts live in their own append-only store.

---

## 3. Lifecycle readiness (Phase 4)

`DRAFT → REVIEW → APPROVED → FROZEN`, on the existing V10 policy
(`v10-answer-benchmark-approval-v1`) — no parallel lifecycle was introduced.

Verified live against the real benchmark:

* `POST /answer-evaluation/runs` with `evaluator: "reference"` →
  **HTTP 400**: *"the reference evaluator requires a human-reviewed benchmark:
  lifecycle is 'draft'"*. Nothing was written (`answer_evaluation_runs` unchanged,
  0 ground-truth rows, 0 review rows after the call).
* Freeze is refused (`409`) while any question is pending/in review/unresolved;
  the gate names the 28 blocking questions and the 0-approved minimum.
* Metrics stay distinct: coverage and retrieval are measured; answer correctness,
  key-point recall, citation support and abstention remain UNKNOWN until reviewed
  data exists.

---

## 4. How to review the 28 questions (the human step)

Start the backend and frontend, open the review surface for the protected KB:

```
http://localhost:3000/knowledge-bases/kb_f278c283c748/benchmark-review
```

Backend (Windows, from `backend/`):

```
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

1. **Set your reviewer identity** in the right-hand panel. It is stored with
   every annotation and review, and an anonymous review is refused.
2. **Work one question at a time.** Left panel: the question, its intended scope
   (subdomain, answerability, grounding expectation, abstention, units, tolerance,
   citation requirements, and the pinned provenance). Centre: the actual indexed
   chunk text. Right: your review.
3. **Use the queue.** Tick *awaiting a decision only*, or filter by state. `→`/`j`
   and `←`/`k` move between questions; `?` shows the shortcuts. No shortcut
   records anything.
4. **Optionally press "Generate draft"** to see a machine suggestion assembled
   from verbatim evidence sentences. It is explicitly **not** ground truth; press
   *Use as starting point* only if you have checked it, then edit it.
5. **Author the reference answer and key points yourself**, then *Save reference
   answer* (append-only; a correction is a new revision).
6. **Record the review** — outcome plus the five dimensions (answerability,
   evidence sufficiency, reference answer, key points, ambiguity). `approved` is
   refused unless every required dimension is affirmative.
7. **For questions you cannot settle honestly**, choose `ambiguous` or
   `insufficient_evidence` rather than weakening the reference answer; they are
   excluded from scoring by policy and named explicitly.
8. **Freeze** once every question carries a decision; freezing creates the
   immutable version that official correctness evaluation requires.

Nothing here is automated on your behalf, and no count in this document moves
until you do it.

---

## 5. Recommended order after V16

1. **Review the 28 questions** (this document §4) — the only remaining blocker to
   answer-correctness measurement. Then freeze v1.
2. **Re-derive the answer metrics** on the frozen version: correctness,
   key-point recall, citation support, unsupported-claim rate, abstention quality
   — each reported with its own denominator, and UNKNOWN where data is missing.
3. **Audit the 20 disagreeing drafts against the human verdicts** — a cheap,
   pre-registered measurement of whether draft assistance helps or misleads.
4. **Decide `auto-eng-024` / `auto-eng-028` policy** — cite more evidence, or
   classify as `insufficient_evidence`; do not let them become scored questions
   by accident.
5. Unchanged from V13/V15: reranker-stage diagnostics, MMR/diversity, and the
   open default-retriever decision (not to be decided on one small instrument).
