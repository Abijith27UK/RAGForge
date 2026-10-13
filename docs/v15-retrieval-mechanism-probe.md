# V15 — Retrieval mechanism probe: what actually caused the two dense misses

**Status: PROBE RUN. No configuration changed, nothing promoted, no new retrieval
feature added. The instrument stays PROVISIONAL / EXPLORATORY, so every number
below is exploratory.**

| | |
|---|---|
| Question answered | Are `auto-eng-hout-007` and `auto-eng-hout-018` **representation/indexing** failures or **ranking/depth** failures? |
| Instrument | `automobile-engineering-heldout-v1` v1, fingerprint `c738f73c9fd52c4a` |
| Instrument sha256 at run time | `faca16de…432b` (before **and** after; unchanged) |
| Configuration | `dense`, evaluated `top_k = 10`, candidate probe depth `20`, config fingerprint `6ca51d50f98bc616` |
| Diagnostic run | `rdx_93a4f54720a9`, 2026-10-13T17:57:45Z, 201.2 s, 26 questions, 1 strategy |
| KB | `kb_f278c283c748` (protected, 812 points, 11 documents, 812 chunks) |
| Verdict | **Both misses are RANKING/DEPTH failures.** The required chunks are indexed, present in the depth-20 candidate pool, and ranked **12th** — two places beyond the k=10 cut-off. |

## 1. What was run, and what was NOT changed

One read-only V12 diagnostics run through the shipped API
(`POST /api/knowledge-bases/{kb}/retrieval-diagnostics/runs`):

```json
{"benchmark_path": "benchmarks/automobile-engineering-heldout-v1.json",
 "strategies": ["dense"], "top_k": 10, "candidate_depth": 20,
 "persist": true, "persist_retrieval_runs": false}
```

* The probe depth 20 is the **pre-existing** constant `MIN_CANDIDATE_DEPTH`; it
  was not chosen to make this result appear.
* No retrieval weight, top-k, chunking strategy, question, expected-evidence
  label, threshold or fusion parameter was touched. `dense_k10` was the declared
  configuration in V14; it is the configuration used here.
* Nothing was written to the knowledge base or the index: the run's own integrity
  block records `retrieval_runs_persisted: 0`, index points 812 → 812, chunk rows
  812 → 812, corpus revision 0 → 0, and the benchmark sha256 unchanged.
* `auto-eng-hout-007` is the one question with **no human decision** (25 approve /
  0 edit / 0 reject / 1 pending). Its per-question result is reported **flagged**,
  exactly as V14 reported it: it is not human-approved evidence, and no decision
  was created for it here.

## 2. Aggregate result (26 questions, `dense`, k=10 over a depth-20 pool)

| Classification | Count | Questions |
|---|---|---|
| `RETRIEVAL_SUCCESS` (required chunk at rank 1) | **17** | 002, 003, 004, 005, 006, 008, 009, 010, 013, 015, 019, 020, 021, 022, 023, 025, 026 |
| `RANKING_FAILURE` (in the window, outranked) | **7** | 001 (rank 2), 011 (5), 012 (3), 014 (2), 016 (2), 017 (2), 024 (2) |
| `LOW_RECALL` (beyond the evaluated cut-off) | **2** | **007 (rank 12)**, **018 (rank 12)** |
| `CORPUS_OR_SOURCE_DEFICIENCY` / `EXPECTED_EVIDENCE_UNAVAILABLE` / `RETRIEVAL_FAILURE` / `NO_RELEVANT_EVIDENCE` | **0** | — |

Availability rollup: **`all_indexed` for 26 of 26 questions** — every expected
chunk is present in the SQLite corpus **and** in the Qdrant index, and its actual
document matches the benchmark's expected document in every case.

| Metric | Value |
|---|---|
| `recall_at_k` | 0.923077 (24/26) |
| `precision_at_k` | 0.092308 (lower bound: necessity-only binary labels) |
| `mrr` | 0.770513 |
| `ndcg_at_k` | 0.809289 |
| `required_recall_in_probe` | **1.000000** — every required chunk was found inside the depth-20 pool |
| `first_relevant_rank` | 1.458333 over the 24 questions that placed required evidence in the window |
| `latency_ms` | 2115.3 (single machine, CPU embedding; this run 201.2 s for 26 questions) |

`required_recall_in_probe = 1.0` is the headline: **no question in this instrument
fails because the evidence is missing.** The dense failures are about *order* and
*window*, not about coverage or representation.

## 3. Question-level diagnostic table

`probe rank` = rank of the required chunk inside the depth-20 candidate pool;
`1st rel` = first required rank inside the k=10 window (`—` = none in the window).

| Question | Subdomain | Classification | Rule | probe rank | 1st rel | recall@10 | MRR |
|---|---|---|---|---|---|---|---|
| auto-eng-hout-001 | engine | RANKING_FAILURE | REQUIRED_EVIDENCE_OUTRANKED_WITHIN_WINDOW | 2 | 2 | 1.0 | 0.500 |
| auto-eng-hout-002 | engine | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-003 | engine | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-004 | transmission | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-005 | transmission | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-006 | braking | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| **auto-eng-hout-007** *(unreviewed)* | braking | **LOW_RECALL** | REQUIRED_EVIDENCE_BEYOND_EVALUATED_CUTOFF | **12** | — | **0.0** | **0.000** |
| auto-eng-hout-008 | steering | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-009 | steering | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-010 | suspension | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-011 | suspension | RANKING_FAILURE | REQUIRED_EVIDENCE_OUTRANKED_WITHIN_WINDOW | 5 | 5 | 1.0 | 0.200 |
| auto-eng-hout-012 | suspension | RANKING_FAILURE | REQUIRED_EVIDENCE_OUTRANKED_WITHIN_WINDOW | 3 | 3 | 1.0 | 0.333 |
| auto-eng-hout-013 | chassis | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-014 | chassis | RANKING_FAILURE | REQUIRED_EVIDENCE_OUTRANKED_WITHIN_WINDOW | 2 | 2 | 1.0 | 0.500 |
| auto-eng-hout-015 | vehicle-dynamics | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-016 | vehicle-dynamics | RANKING_FAILURE | REQUIRED_EVIDENCE_OUTRANKED_WITHIN_WINDOW | 2 | 2 | 1.0 | 0.500 |
| auto-eng-hout-017 | electric-vehicles | RANKING_FAILURE | REQUIRED_EVIDENCE_OUTRANKED_WITHIN_WINDOW | 2 | 2 | 1.0 | 0.500 |
| **auto-eng-hout-018** | electric-vehicles | **LOW_RECALL** | REQUIRED_EVIDENCE_BEYOND_EVALUATED_CUTOFF | **12** | — | **0.0** | **0.000** |
| auto-eng-hout-019 | electric-vehicles | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-020 | electric-vehicles | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-021 | battery-management | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-022 | battery-management | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-023 | regenerative-braking | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-024 | regenerative-braking | RANKING_FAILURE | REQUIRED_EVIDENCE_OUTRANKED_WITHIN_WINDOW | 2 | 2 | 1.0 | 0.500 |
| auto-eng-hout-025 | general-automobile | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |
| auto-eng-hout-026 | general-automobile | RETRIEVAL_SUCCESS | REQUIRED_EVIDENCE_AT_RANK_1 | 1 | 1 | 1.0 | 1.000 |

## 4. The two dense misses, question by question

Scores below were read directly from the **same retrieval configuration** the
probe executed (`config_fingerprint 6ca51d50f98bc616`, identical to the run's),
in a read-only call that persisted nothing. The V12 probe block records candidate
ids and ranks but **not** scores for candidates outside the window, so the scores
are reported from that identical configuration rather than invented from the run.

### `auto-eng-hout-007` (flagged: no human decision)

| Fact | Value |
|---|---|
| Question | "What did Bosch and Mercedes update in 1978, and how was that system described?" |
| Required chunk | `chk_502dcf67ec92` (Brake - Wikipedia, section Introduction, content hash `ba8596bb…f8c`) |
| Availability | `in_corpus: true`, `in_index: true`, `status: indexed`, `document_matches_benchmark: true` |
| Rank in the depth-20 pool | **12** (`required_beyond_cutoff: {chk_502dcf67ec92: 12}`; `absent_from_probe: []`) |
| Score at rank 12 | **0.3897** |
| Rank-10 boundary score | 0.3966 (margin **0.0069**) |
| Rank-1 result | `chk_9ac925776a5f`, 0.4899 — Regenerative braking |
| Window composition (k=10) | Automobile ×5, Regenerative braking ×2, Vehicle frame ×1, Engine ×1, Transmission ×1 |
| Required document in the window? | **No** — the Brake document never enters the top 10 |

### `auto-eng-hout-018`

| Fact | Value |
|---|---|
| Question | "What do current electric vehicle safety regulations focus on, and which materials are being developed to address it?" |
| Required chunk | `chk_60b255a941d5` (Electric vehicle - Wikipedia, section Maz-7907, content hash `8c2b4588…7246`) |
| Availability | `in_corpus: true`, `in_index: true`, `status: indexed`, `document_matches_benchmark: true` |
| Rank in the depth-20 pool | **12** (`required_beyond_cutoff: {chk_60b255a941d5: 12}`; `absent_from_probe: []`) |
| Score at rank 12 | **0.5346** |
| Rank-10 boundary score | 0.5460 (margin **0.0114**) |
| Rank-1 result | `chk_f16fd6ca644c`, 0.5958 — Electric vehicle (same document) |
| Window composition (k=10) | Electric vehicle ×9, Automobile ×1 |
| Required document in the window? | **Yes**, 9 of 10 slots — the failure is *inside* one document |

Both failures share one shape: the required passage is retrievable, retrieved,
and sits **two places below the cut-off**, separated from the boundary by a score
gap under 0.012 on a 0–1 cosine scale. Nothing is missing; the window is one
step too narrow and the ordering inside the window is not the right one.

## 5. What outranks the required evidence (the 7 `RANKING_FAILURE` questions)

Every chunk ranked above a required chunk, classified by whether it belongs to
the same source document:

| Question | 1st required rank | Chunks above it | Of those, same document as the required chunk |
|---|---|---|---|
| auto-eng-hout-001 | 2 | 1 | 1 |
| auto-eng-hout-011 | 5 | 4 | 4 |
| auto-eng-hout-012 | 3 | 2 | 2 |
| auto-eng-hout-014 | 2 | 1 | 1 |
| auto-eng-hout-016 | 2 | 1 | 1 |
| auto-eng-hout-017 | 2 | 1 | 1 |
| auto-eng-hout-024 | 2 | 1 | 1 |
| **Total** | — | **11** | **11 (100%)** |

The dense retriever is not confusing *documents*: it puts the right document at
the top of the window and then ranks the wrong *siblings* of that document above
the specific passage the question needs. Zero of the 11 outranking chunks comes
from another document.

## 6. Observed facts, separated from hypotheses

**Observed (deterministic, stored in `rdx_93a4f54720a9`):**

1. All 26 expected chunks are `indexed`: present in the corpus **and** in the
   vector index, with matching document ids. There is no evidence-availability
   failure in this instrument.
2. `required_recall_in_probe = 1.0` at depth 20: every required chunk was inside
   the candidate pool.
3. The two dense misses sit at **rank 12** with a depth-20 pool, i.e. two places
   beyond k=10 — a *window* failure, not a coverage failure.
4. The two misses are near-ties on score (0.0069 and 0.0114 below the rank-10
   boundary score), not outliers.
5. All 11 chunks that outrank a required chunk in the 7 `RANKING_FAILURE`
   questions come from the **same document** as the required chunk.
6. `auto-eng-hout-018`'s required document fills 9 of the 10 window slots while
   the required chunk is still only 12th; `auto-eng-hout-007`'s required document
   does not enter the window at all.

**Hypotheses (NOT established by this probe):**

* That a lexical term match is what separates these passages (V14 measured BM25
  at rank 1 for exactly these two questions, but this probe ran **dense only**,
  so it cannot attribute the failure to the absence of a lexical signal).
* That the same-document distractors are near-duplicate boilerplate; that needs a
  text-similarity measurement of its own.
* That raising k is cheap in practice: a wider window also means more chunks in a
  generation prompt, which this probe did not measure.
* That any of this generalises beyond this one corpus, these 26 questions and
  this machine.

## 7. What this confirms and corrects in V14

**Confirmed.** V14 §7 proposed the hypothesis that for these two questions "the
required chunk IS present in the probed candidate pool (depth 20) but ranked
below 10, i.e. the dense failure is a RANKING failure, not a representation
failure". That hypothesis was written before this probe ran and is now supported
by the run: present in the pool, rank 12, indexed, in corpus. The declared
acceptance criterion — "if the chunk is in the pool but ranked low, the lever is
ranking" — is met for both questions.

**Corrected.** V14 §5 stated that for these two questions "recall is identical for
`dense_k3..k10` ... so widening the window does not help them — the evidence is
not merely outside the window". The second half of that sentence is **wrong** and
is corrected here: the evidence *is* outside the k=10 window, at rank 12, and a
window of k ≥ 12 would recover it. The first half remains true as stated (these
two questions yield 0.0 recall at every k the V14 arms evaluated, because none of
those arms went past k=10). No V14 number is changed by this correction and the
sealed instrument is untouched — the V14 artifacts still record the instrument at
`536e6541…`, the revision they actually ran against (see §11).

## 8. A reporting hazard found while reading the run (not fixed here)

In this run the summary's `required_recall_at_depth` is **0.923077 — identical
to `recall_at_k`**, and it reads `0.0` for the two questions whose evidence the
probe *did* find at rank 12. Cause: `build_question_diagnostic` computes that
metric over the list it is handed, and the V12 runner hands it the **windowed**
(k=10) list; the depth-independent figure lives in
`probe.required_recall_in_probe` (1.0 here). The field's own documentation in
`docs/retrieval-diagnostics.md` describes it as depth-independent, so a reader
can conclude "the evidence is absent even at depth" from a value that only
reports the window.

Nothing was changed to fix this: the metric's name and value are part of
`COMPARABLE_METRICS` and of comparisons already recorded, so redefining it
silently would invalidate stored comparisons. Recorded as an observation for an
explicit decision. The depth-independent signal is `required_recall_in_probe`,
and that is the one this probe relies on.

## 9. The smallest justified follow-up experiment

Not a new retrieval feature — the mechanism is now identified, and the lever it
names is the window/ranking, so the next measurement is a **window experiment** on
the same frozen instrument.

| Element | Declaration |
|---|---|
| Hypothesis | Both `LOW_RECALL` questions are window failures: at a wider evaluated `k` they convert, because their required chunks are pooled at rank 12. |
| Arms | baseline `dense_k10` (unchanged) vs candidate `dense_k20` |
| Why k=20 and not k=12 | 12 is the observed rank — choosing it now would be tuning against this instrument's result. 20 is the **pre-existing** probe-depth constant (`MIN_CANDIDATE_DEPTH`), declared in the code before this probe existed. Stated plainly so the choice can be judged. |
| Controls | same frozen instrument (fingerprint `c738f73c9fd52c4a`), same corpus revision, same embedding model, same machine, same question set, no weight or fusion change; new artifact, previous V13/V14 artifacts untouched; `promoted_to_default` stays false |
| Primary metric | `recall_at_k` (the metric the failure was observed in) |
| Guardrails | `latency_ms` at the V14 bar (≤ +50 ms/query) and `mrr` reported. `precision_at_k` is **reported, not guarded**: a wider window mechanically lowers precision@k under necessity-only labels, so guarding it would pre-commit the experiment to failure (V13 kept both declarations, and that is the honest precedent). |
| Acceptance | both `auto-eng-hout-007` and `auto-eng-hout-018` convert out of `LOW_RECALL`, and no question regresses on `recall_at_k`. Prediction: **0.9231 → 1.0000**. |
| Falsifier | if recall stays 0.9231 at k=20, this probe's mechanism reading does not reproduce under the experiment harness, and the lever is not depth — the next candidate would be fusion/reranking, measured the same way. |
| Cost / limits | one experiment run, no configuration promoted. Still exploratory: 26 questions, below the 30-question inference floor, on a PROVISIONAL instrument. A passing result licenses *no* promotion. |

**Explicitly not proposed here:** promoting BM25 or hybrid to default. V14 measured
BM25 at MRR 1.0 and 43 ms on these 26 questions, and this probe shows the dense
failures sit in exactly the near-tie band lexical matching resolves — but one small
instrument is not a basis for changing a default, and the promotion decision is
deliberately separate and explicit.

## 10. What may not be claimed

* Not significance: 26 questions, below the 30-question floor the comparison
  module uses (`MIN_RELIABLE_PAIRED_N`). Everything here is exploratory.
* Not generalisation: one corpus (11 documents, 812 chunks), one domain, one
  machine, one embedding model. The two failures are two questions.
* Not answer quality: no generator ran; only retrieval was measured.
* Not a default change: nothing is promoted, and `promoted_to_default` is false on
  every artifact.
* Not human-approved for `auto-eng-hout-007`: it still carries no decision, so its
  row stays flagged and the tier stays PROVISIONAL.
* The separate **answer-quality benchmark is still unreviewed**: 28 questions, 0 of
  28 carry a human decision, 0 carry an `expected_answer`, 0 frozen versions
  exist. Answer-correctness optimization remains blocked, and no reference answers
  were fabricated here.

## 11. Integrity

| Check | Before | After |
|---|---|---|
| `kb_f278c283c748` Qdrant points (`kb_kb_f278c283c748`, status `green`, 384-d cosine) | 812 | **812** |
| Protected KB chunk rows | 812 | **812** (`retrieval_runs_persisted: 0`) |
| Corpus revision | 0 | **0** |
| Qdrant collections | 14 | **14** (no scratch collection) |
| `benchmarks/automobile-engineering-heldout-v1.json` | `faca16de…432b` | **unchanged** (`faca16de…432b`) |
| `benchmarks/automobile-engineering-heldout-v1-candidates.json` | `e4a75253…be264` | **unchanged** |
| `benchmarks/automobile-engineering-baseline-v1.json` | `08601f58…a656` | **unchanged** |
| `benchmarks/answer-quality-automobile-v1.json` | `16db9dbd…9f25` | **unchanged** |
| SQLite `PRAGMA integrity_check` / `foreign_key_check` | `ok` / clean | **`ok` / clean** |
| V13/V14 experiment artifacts under `backend/data/experiments/` | — | **not rewritten** |

Two facts about the instrument's revision history that this probe makes visible
and that belong on the record:

1. The V14 experiment artifacts ran against the instrument at `536e6541…` (they
   record that sha before and after each run). The file on disk is now
   `faca16de…` — a **re-seal of the same content**: `freeze()` refuses any write
   whose content fingerprint differs, and the fingerprint is `c738f73c9fd52c4a`
   in both revisions. Only seal metadata (`frozen_at`, `frozen_by`, notes)
   changed, at 2026-10-12T21:03:29Z. The tier did not change (still PROVISIONAL,
   25/26 reviewed), so no lifecycle transition occurred and no tier was raised
   after results were seen.
2. `benchmarks/automobile-engineering-heldout-v1.json` is **not versioned**: by
   design, a test asserts the shipped frozen held-out artifact does not exist in
   the repository, because a freeze is a research event performed on an
   installation, not a file to be committed. The frozen sha256 is recorded in
   this document, in the README and in the V14 analysis, and the tracked
   candidate artifact plus the freeze tooling are what make the seal
   reproducible; a fresh clone can re-freeze it and compare the fingerprint.

## 12. The review gate — explicit human decision (2026-10-13)

`auto-eng-hout-007` was left **pending** by explicit human decision, not by
oversight, and no v2 instrument was authorised:

* the sealed v1 instrument **stays PROVISIONAL / EXPLORATORY**, and its results
  (V14 and this probe) remain exploratory;
* no replacement question, no re-seal and no new version was created;
* the state is exactly what the artifact and the API already reported — 25
  approved / 0 edit / 0 reject / **1 pending**, reviewers `["Abijith"]`;
* the window experiment declared in §9 is likewise **recorded only** — it was not
  run, so the instrument still has no post-probe measurement attached to it.

Revisiting this requires a fresh, explicit decision: the only documented route to
closing the gate is a **new instrument version** (new candidate set → blind review
→ freeze), because a decision recorded after the results are known cannot raise
the claim strength of results already seen.

## Reproduce

```bash
# Qdrant on :6333, backend on :8013, HF_HUB_OFFLINE=1
curl -s -X POST http://localhost:8013/api/knowledge-bases/kb_f278c283c748/retrieval-diagnostics/runs \
  -H 'Content-Type: application/json' \
  -d '{"benchmark_path":"benchmarks/automobile-engineering-heldout-v1.json","strategies":["dense"],"top_k":10,"candidate_depth":20,"persist":true,"persist_retrieval_runs":false}'

# re-read the stored run
curl -s http://localhost:8013/api/knowledge-bases/kb_f278c283c748/retrieval-diagnostics/runs/rdx_93a4f54720a9
curl -s http://localhost:8013/api/knowledge-bases/kb_f278c283c748/retrieval-diagnostics/runs \
  | python -m json.tool    # summaries, newest first
```

The run takes ~200 s on CPU because each question embeds a query locally
(`dense` ≈ 2.1 s/question).
