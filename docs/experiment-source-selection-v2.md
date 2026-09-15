# Experiment: Source Selection v2 — Content-Aware Quality Scoring vs Random Baseline

**Status:** completed 2026-09-15 · **Design:** `benchmarks/source-selection-experiment-v2.json` ·
**Results:** `benchmarks/source-selection-experiment-v2-results.json` ·
**Paired analysis:** `benchmarks/source-selection-experiment-v2-paired-analysis.json` ·
**Scorer design:** `docs/design-source-quality-v2.md`

> **Integrity caveat (applies to every number below):** the frozen benchmark's ground truth is
> **agent-authored, human review pending**. All results inherit that limitation and must not be
> presented as publication-grade until a human has reviewed the benchmark selections. This **v2**
> experiment supersedes `source-selection-experiment-v1`, whose instrument failure (34/36
> identical v1 scores) made its QUALITY arm an alphabetical selection; v1 artifacts remain frozen
> and untouched.

## 1. Research question

Does explainable source-quality-based selection produce a better and/or more efficient
domain-specific RAG knowledge base than an equivalent random-source baseline?

## 2. Hypothesis

- **H1:** QUALITY_SELECTED corpora (top-N by the v2 content-aware composite) achieve higher
  retrieval metrics on the frozen benchmark than RANDOM_BASELINE corpora of equal N, on common
  answerable-question subsets.
- **H2:** QUALITY_SELECTED reaches equal-or-better retrieval quality with fewer
  documents/chunks/characters (efficiency), i.e. more benchmark-relevant knowledge per source.
- **H0:** selection strategy makes no difference.

## 3. Variables

- **Independent variable:** source-selection strategy — QUALITY_SELECTED vs RANDOM_BASELINE
  (definitions in §6).
- **Controlled variables:** the same 36-URL candidate pool as v1; N ∈ {5, 8, 11}; the same live
  ingestion pipeline; the same chunking (section-aware, target 1200, overlap 150); the same
  embedding model (sentence-transformers MiniLM, 384-d, normalized); Qdrant dense retrieval;
  strict evaluation (keyword fallback disabled) of the frozen
  `automobile-engineering-baseline-v1` questions; fixed seed 20260915.
- **Dependent variables:** Recall@3/5/10, Precision@3/5/10, MRR, NDCG (chunk basis; doc-recall
  recorded); coverage (answerable questions per corpus, reported strictly separately);
  documents, chunks, corpus_chars, ingest/index seconds; per-source v2 signals.

## 4. Candidate pool

The frozen v1 pool: 36 real Wikipedia automobile URLs (all verified HTTP 200 at design time;
listed in `benchmarks/source-selection-experiment-v1.json#/candidate_pool/urls`, referenced by
the v2 design — never duplicated). Both arms select from exactly this pool.

## 5. Domain knowledge map

`backend/app/services/source_quality/domain_maps/automobile_engineering.json` (v1.0.0): nine
explicit requirement areas — engine; transmission & drivetrain; braking; steering; suspension;
chassis & body structure; vehicle dynamics & handling; electric vehicles & electrification;
automobile fundamentals & general systems — each with a description, deterministic seed terms,
and a weight (sum validated at load). Provenance: *curated domain configuration (agent-drafted,
human review pending)*, aligned with the domain definition, NOT derived from benchmark questions.

## 6. Source-quality scoring methodology and selection protocols

**v2 scorer (`heuristic-v2-content-aware`):** each candidate's page content is fetched
(SSRF-validated, ≤256 KB, navbox/infobox-stripped extraction, cached), windowed into prose-only
400-char windows (≥45 words and ≥4 words/line — filters link lists), embedded with the same
MiniLM model used for the corpus, and matched against each requirement area embedded as multiple
short queries (description + seed terms). Per area: `sim = mean(top-3 query-window pairs)`,
normalized (`(raw − 0.20)/0.80`) so 0 = unrelated; evidence passage and matched queries are
recorded; area covered iff normalized sim ≥ 0.50. Outputs per source:
**CONTENT_RELEVANCE** (weighted mean of per-area sims), **DOMAIN_COVERAGE** (fraction of areas
covered), **CONFIDENCE**, per-area evidence, and limitations — each reported independently from
**AUTHORITY, ACCESSIBILITY, SOURCE_TYPE, RECENCY, DUPLICATION** (v1 signals, unchanged).

**Composite:** content_relevance·0.40 + domain_coverage·0.20 + authority·0.15 +
accessibility·0.10 + source_type·0.05 + recency·0.05 + duplication·0.05 — content signals 0.60
vs metadata 0.25, so an authoritative page that does not cover the domain cannot outrank a
relevant one.

**Instrument check on the frozen pool:** 36/36 unique content_relevance values and composites
(v1: 2 unique values; 34/36 identical at 0.4525). Evidence passages manually audited. Mechanism
verification from the saved ranking: metadata signals are constant across the pool (1 unique
value each) so they cannot affect order, while content_relevance correlates ρ = +0.882
(Spearman) with the composite ranking — selection is genuinely content-driven.

- **QUALITY_SELECTED protocol:** rank all 36 pool sources by v2 composite (tie-break: URL
  ascending, deterministic); select the top N.
- **RANDOM_BASELINE protocol:** `random.Random(20260915).sample(pool, N)` — fixed seed, uniform
  without replacement, same pool, same N.

## 7. Corpus sizes and protocol

Six corpora: {QUALITY_SELECTED, RANDOM_BASELINE} × N ∈ {5, 8, 11}. Per run: create KB →
analyze-domain (deterministic mock spec, identical across KBs) → discover sources (real probes,
HTTP 200 verified) → ACCEPT selected sources → ingest → index with the controlled chunking
config → translate ground truth → strict evaluation at k ∈ {3, 5, 10} → persist everything.

## 8. Ground-truth translation methodology

Re-chunking regenerates chunk IDs, so the frozen benchmark's `expected_chunk_ids` cannot be used
verbatim in new KBs. Translation is per corpus, in the runner: frozen chunk ID → chunk
`content_hash` (resolved from the frozen baseline KB `kb_f278c283c748`) → chunk ID in the new
corpus (same chunker config + same document text ⇒ same hash). Document-level GT translates via
source URL. **The benchmark file is never modified.** Questions whose translated GT is empty are
excluded as unanswerable and listed by ID. Chunk translation matched 100% of expected hashes in
every corpus — chunking behaved identically across runs.

## 9. Coverage vs retrieval-quality separation

- **A. Corpus coverage:** how many of the 28 benchmark questions are answerable (GT present)
  from the selected corpus. Reported separately; never mixed into retrieval metrics.
- **B. Retrieval performance:** metrics computed ONLY on questions whose ground-truth chunks
  exist in that corpus (the answerable set), in strict mode with keyword fallback disabled.
- **C. Paired comparison:** any comparison between two runs uses ONLY their common
  answerable-question intersection, recomputed from per-question metrics.

The six corpora do NOT share the same answerable-question set (**global intersection = 0**);
therefore raw aggregate rows across runs are not comparable and are not used for conclusions.

## 10. Exact experimental results

### 10.1 Corpus construction (all six corpora real)

| Run | Selected sources | Docs | Chunks | Chars | Answerable | Index s |
|---|---|---|---|---|---|---|
| QUALITY N=5 | ABS, Electric_vehicle, Regenerative_braking, Steering, Suspension | 5 | 511 | 316,770 | **11/28** | 34 |
| QUALITY N=8 | + Anti-roll_bar, Brake, Tire | 8 | 737 | 450,236 | **13/28** | 39 |
| QUALITY N=11 | + Disc_brake, Drum_brake, Engine | 11 | 949 | 598,051 | **17/28** | 68 |
| RANDOM N=5 | ABS, Automobile, Shock_absorber, Transmission, Wankel | 5 | 431 | 289,176 | **4/28** | 35 |
| RANDOM N=8 | + Diesel, Double_wishbone, Li-ion, Tire | 8 | 953 | 628,646 | **4/28** | 59 |
| RANDOM N=11 | + Automatic_transmission, Regenerative_braking, Suspension | 11 | 1,191 | 774,049 | **10/28** | 89 |

### 10.2 Retrieval metrics (strict, explicit GT only; per-corpus answerable sets)

| Run | R@3 | R@5 | R@10 | P@3 | MRR@5 | NDCG@5 | n |
|---|---|---|---|---|---|---|---|
| QUALITY N=5 | 0.727 | 0.909 | 1.000 | 0.273 | 0.682 | 0.738 | 11 |
| QUALITY N=8 | 0.769 | 0.923 | 1.000 | 0.282 | 0.723 | 0.772 | 13 |
| QUALITY N=11 | 0.765 | 0.941 | 1.000 | 0.275 | 0.744 | 0.792 | 17 |
| RANDOM N=5 | 1.000 | 1.000 | 1.000 | 0.333 | 0.875 | 0.908 | 4 |
| RANDOM N=8 | 1.000 | 1.000 | 1.000 | 0.333 | 0.875 | 0.908 | 4 |
| RANDOM N=11 | 0.800 | 1.000 | 1.000 | 0.300 | 0.733 | 0.799 | 10 |

**These rows are NOT directly comparable** (§9): the RANDOM N=5/N=8 "perfect" rows are computed
on only 4 questions — a small-sample artifact, not a superiority.

### 10.3 Paired comparisons — the primary basis for QUALITY vs RANDOM conclusions

Method: for each pair, take the intersection of the two runs' answerable sets, pull the saved
per-question metrics of both strict evaluation runs, and average only over the common questions.
Artifact: `benchmarks/source-selection-experiment-v2-paired-analysis.json` (independently
re-verified against the live evaluation runs; labels left=first-named run, right=second-named).

| Pair (left vs right) | Common q | k | Left R | Right R | Left MRR | Right MRR | Left NDCG | Right NDCG | Recall wins (L=R / L> / R>) |
|---|---|---|---|---|---|---|---|---|---|
| QUALITY11 vs RANDOM11 | 6 | 3 | 0.667 | 0.667 | 0.583 | 0.583 | 0.605 | 0.605 | 6 / 0 / 0 |
| QUALITY11 vs RANDOM11 | 6 | 5 | 1.000 | 1.000 | 0.650 | 0.667 | 0.734 | 0.749 | 6 / 0 / 0 |
| QUALITY11 vs RANDOM11 | 6 | 10 | 1.000 | 1.000 | 0.650 | 0.667 | 0.734 | 0.749 | 6 / 0 / 0 |
| QUALITY5 vs RANDOM5 | 0 | 3/5/10 | — | — | — | — | — | — | intersection empty: comparison impossible |
| QUALITY5 vs QUALITY8 | 11 | 5 | 0.909 | 0.909 | 0.682 | 0.673 | — | — | 11 / 0 / 0 |
| QUALITY8 vs QUALITY11 | 13 | 5 | 0.923 | 0.923 | 0.723 | 0.723 | — | — | 13 / 0 / 0 |

On the only non-empty cross-strategy subset (6 questions at N=11): **Recall identical on all 6
questions at every k; MRR differs by one rank on one question in favour of RANDOM (0.650 vs
0.667). No retrieval-quality difference is detectable at this sample size.**

### 10.4 Coverage and efficiency (valid across runs; the decisive difference)

| N | QUALITY answerable | RANDOM answerable | QUALITY chars / answerable q | RANDOM chars / answerable q |
|---|---|---|---|---|
| 5 | 11/28 | 4/28 | 28.8k | 72.3k |
| 8 | 13/28 | 4/28 | 34.6k | 157.2k |
| 11 | 17/28 | 10/28 | 35.2k | 77.4k |

1. **H2 supported on the coverage/efficiency dimension:** content-aware selection packs 1.75–3.25×
   more benchmark-answerable knowledge into the same number of sources, and needs 2.2–4.5× fewer
   characters per answerable question. This follows from selection behaviour, not small-sample
   metric noise.
2. **H1 not resolvable at this sample size** (§10.3): H0 is likewise not established.
3. Selection avoided the v1 failure mode (quality top-N covering 0/28); the v2 top-11 spans
   Engine, brakes, steering, suspension, and EV pages — the benchmark's subdomain spread.
4. Reproducibility note: with a fixed seed, CPython's `Random.sample` yields nested samples
   (RANDOM N=5 ⊂ N=8 — hence identical coverage); RANDOM N=11 happened to draw the two
   highest-v2-score sources, explaining its coverage jump to 10/28.

## 11. Pairwise-intersection methodology (reproducibility)

Answerable sets are derived from each run's recorded coverage (28 − excluded). For a pair
(A, B) and each k, the common set is A∩B; means are simple averages of the per-question
`recall_at_k` / `mrr` / `ndcg` stored on the strict evaluation runs; win/tie counts are computed
per question. The artifact was independently re-verified: every intersection, mean, and
win/tie count recomputed from the live API matched the saved values, and stored aggregates match
recomputation from per-question data (no fabricated values).

## 12. Limitations

1. **Ground truth is agent-authored, human review pending** — inherited by every number above.
2. Single seed; RANDOM arms are one draw each. Coverage conclusions are robust; metric
   conclusions are not.
3. Answerable-set sizes are small (4–17); the cross-strategy paired comparison caps at n=6, and
   the N=5 cross-strategy comparison is impossible (empty intersection).
4. The six runs' answerable sets differ (global intersection 0): any aggregate cross-run table
   is descriptive only.
5. Wikipedia-homogeneous pool: metadata signals were constant, which cleanly isolates content
   signals here but means metadata behaviour is untested within this experiment.
6. Construct overlap: the domain map's areas coincide with the benchmark's subdomain taxonomy
   (both derive from the domain definition), so the coverage result is partly definition-coupled.
7. Coverage answers "is the relevant document present", not "is it chunked/retrievable" — though
   100% hash-match translation shows chunking behaved identically.

## 13. Conclusion

**v1 was inconclusive because its scorer assigned 34/36 sources the identical score 0.4525, so
"quality selection" collapsed into alphabetical tie-breaking (its N=5 arm covered 0/28
benchmark questions). v2 differentiated the pool fully (36/36 unique scores) and made selection
genuinely content-aware.** With that instrument fixed, the experiment demonstrates a clear
**coverage/efficiency advantage** for content-aware selection (11/13/17 vs 4/4/10 answerable
questions at N=5/8/11; 2.2–4.5× fewer characters per answerable question) — H2 supported —
while **retrieval-quality superiority (H1) is unproven**: on the only valid paired subset
(n=6, N=11), Recall is identical and MRR is statistically indistinguishable. The honest summary:
content-aware source selection builds smaller, better-targeted corpora that can answer more of
the benchmark; whether it also improves retrieval *ranking* needs more seeds and a larger,
human-reviewed benchmark.

## 14. Reproduction

```
# instrument check (scorer validation, frozen pool)
python scripts/report_v2_pool_scores.py --experiment ../benchmarks/source-selection-experiment-v1.json
# full experiment (resumable per corpus with --only "STRATEGY|N")
python scripts/run_source_selection_experiment_v2.py \
  --experiment ../benchmarks/source-selection-experiment-v2.json --baseline-kb kb_f278c283c748
```

Requires backend on :8000 (current code) and Qdrant on :6333. The six experiment KBs
(`SourceSelV2 …`) are retained for UI inspection; the frozen baseline and v1 KBs are untouched.
