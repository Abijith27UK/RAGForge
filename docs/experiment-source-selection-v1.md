# Experiment: Source Selection v1 — quality-based vs random corpus construction

**Status:** completed 2026-09-15 · **Design:** `benchmarks/source-selection-experiment-v1.json` · **Raw results:** `benchmarks/source-selection-experiment-v1-results.json` · **Frozen baseline:** `automobile-engineering-baseline-v1` (untouched)

> ⚠️ **Ground-truth caveat (applies to every number below):** the frozen benchmark's ground
> truth is **agent-authored, human review pending**. All results in this experiment inherit
> that limitation and are **NOT publication-grade**. They are internally consistent
> instrument readings for engineering decisions, not scientific claims.

## 1. Research question

Does explainable source-quality-based selection produce a better and/or more efficient
domain-specific RAG knowledge base than an equivalent random-source baseline?

## 2. Hypotheses

- **H1 (quality):** quality-selected corpora answer more benchmark questions correctly at
  equal N (higher Recall@K / MRR).
- **H2 (efficiency):** quality-selected corpora reach equal quality with fewer chunks/chars.
- **H0:** no difference beyond selection noise.

**Verdict for v1: neither H1 nor H2 can be accepted or rejected.** The quality scorer could
not discriminate within the candidate pool (see §7), so the independent variable was not
manipulated as intended. The experiment is inconclusive *by instrument failure*, which is
itself a finding.

## 3. Variables

- **Independent variable:** selection strategy — `QUALITY_SELECTED` (top-N by the existing
  `SourceQualityScorer` score) vs `RANDOM_BASELINE` (uniform random N, seed **20260915**).
- **Controlled variables:** identical 36-URL candidate pool; identical N per pair
  (5/8/11); same ingestion pipeline (download/parse/hash-dedup); same chunking
  (section-aware, 1200/150); same embedding (all-MiniLM-L6-v2, 384d, L2); same vector DB
  (Qdrant 1.19.1, cosine); same retrieval (qdrant-dense, k=3/5/10, no filters); same frozen
  benchmark and strict evaluation; same deterministic (mock) DomainSpec for relevance scoring.
- **Dependent variables:** Recall/Precision@3/5/10, MRR, NDCG (binary, chunk basis);
  doc-basis Recall@5; **coverage** (share of the 28 questions whose ground-truth chunk exists
  in the corpus — reported separately, never mixed into retrieval metrics); documents,
  chunks, corpus chars, ingest/index wall time.

## 4. Dataset / candidate pool

36 real Wikipedia Automobile Engineering URLs (11 from the frozen baseline corpus + 25
subdomain articles), each verified HTTP 200 before freezing. Frozen in
`source-selection-experiment-v1.json`; no source was invented or substituted at run time.

## 5. Experimental procedure

For each (strategy, N): create KB → run the deterministic domain analysis → discover the
selected URLs through the real pipeline (quality scoring with live accessibility probes) →
accept exactly the selected sources → ingest → index (same config as baseline) → translate
the frozen benchmark's ground truth into the new corpus via deterministic chunk
`content_hash` (same chunker config ⇒ same chunk text ⇒ same hash; document GT via source
URL) → load only **answerable** questions → strict evaluation at k=3/5/10. One corpus per
pair, six corpora total, all left in the system for inspection (`kb_id`s in the results file).

The baseline benchmark file was not modified; translation happens in
`backend/scripts/run_source_selection_experiment.py` and per-run translation stats are
recorded (28/28 chunk hashes resolved; no manual mapping).

## 6. Results

### 6.1 Coverage (answerable questions out of 28)

| N | QUALITY_SELECTED | RANDOM_BASELINE |
|---|---|---|
| 5 | **0** | 4 |
| 8 | 3 | 4 |
| 11 | 3 | **10** |

### 6.2 Retrieval metrics — NOT directly comparable across runs

| Run | Answerable n | Recall@3 | Recall@5 | Recall@10 | MRR | NDCG@5 | chunks | chars | index s |
|---|---|---|---|---|---|---|---|---|---|
| QUALITY N=5 | 0 | — | — | — | — | — | 277 | 166k | 22.4 |
| QUALITY N=8 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 386 | 243k | 25.3 |
| QUALITY N=11 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 684 | 379k | 36.2 |
| RANDOM N=5 | 4 | 1.000 | 1.000 | 1.000 | 0.875 | 0.908 | 431 | 289k | 49.2 |
| RANDOM N=8 | 4 | 1.000 | 1.000 | 1.000 | 0.875 | 0.908 | 953 | 629k | 87.8 |
| RANDOM N=11 | 10 | 0.800 | 1.000 | 1.000 | 0.733 | 0.799 | 1191 | 774k | 63.9 |

**Why cross-run comparison is invalid in v1:** the answerable-question sets are disjoint
(e.g. QUALITY N=8 scored only {009, 010, 025}; RANDOM N=5 scored only {005, 006, 007, 008};
**global intersection = 0 questions**). A run scoring 3 easy questions cannot be compared to
a run scoring 10 harder ones. The metrics above are per-run instruments on different subsets,
and the results JSON preserves `coverage.excluded_unanswerable` per run to make this explicit.

### 6.3 What the numbers do support

- **The scorer cannot rank same-type sources (primary finding).** 34 of 36 pool URLs scored
  an identical **0.4525** (only `Automobile` and `Automobile_layout` scored 0.5358, via
  recency/evidence text markers). "Top-N by quality" therefore degenerated to
  *alphabetical* selection among 34 tied sources — the independent variable was not truly
  manipulated. This is a defect of the current scoring instrument for homogeneous web pools,
  not of the experiment.
- **Quality-top-N ignored benchmark-critical sources.** At N=5 the quality strategy selected
  zero documents carrying any ground truth (coverage 0/28) while the random draw covered 4/28.
  The heuristic's signals (domain suffix, source-type baseline, title-term relevance vs the
  mock spec) do not measure topical fit to *this* benchmark's knowledge requirements.
- **Efficiency (observed, confounded):** quality-selected corpora were consistently smaller
  (277–684 chunks / 166–379k chars) than random ones (431–1191 chunks / 289–774k chars) —
  but this reflects *which short alphabetical articles were picked*, not a quality mechanism.
  H2 has no support in v1.
- **Retrieval itself behaved correctly:** every answerable question's relevant chunk was
  retrieved in top-10 across all corpora (Recall@10 = 1.0 everywhere), consistent with the
  frozen baseline's Recall@10 = 1.0.

## 7. Root cause of the instrument failure

With one source type (Wikipedia), the weighted sum collapses: authority ≈ constant (0.35),
source_type ≈ constant (0.45), duplication ≈ constant, accessibility ≈ constant (all 200),
recency driven by superficial year strings, and relevance ≈ constant because the
deterministic mock DomainSpec's generic terms match nearly every automobile article equally.
The remaining variance comes from tiny title-text effects, producing the 34-way tie.

## 8. Limitations

1. **Frozen-benchmark ground truth is agent-authored, human review pending** — inherited by
   every number above; nothing here is publication-grade.
2. **Inconclusive comparison:** no metric difference between strategies is attributable to
   *quality* selection, because scores were degenerate (34-way tie; alphabetical tie-break).
3. Single seed, single repetition; sample sizes (3–10 answerable questions per run) are far
   too small for statistical inference.
4. Candidate pool is Wikipedia-only and homogeneous, the worst case for a type-based scorer.
5. Coverage vs quality trade-off: scoring only answerable questions avoids punishing corpora
   for missing sources, but means metrics ride on different question subsets per run
   (documented per run; global intersection = 0 in v1).
6. Wall-clock timings are single-machine measurements, not controlled benchmarks.

## 9. What would make v2 informative (design notes, not implemented)

- Give the scorer discriminating power for homogeneous pools: content-based relevance
  (probe/page text vs DomainSpec), corpus-coverage-aware signals, or LLM-based source
  summarization (would require a configured LLM and must be labelled).
- Re-run with ≥ 5 seeds per (strategy, N) and a pool with genuinely diverse source types
  (edu, gov, standards bodies) where the authority signal has variance.
- Define the primary comparison on the **intersection of answerable questions** plus a
  separately reported coverage delta.
- Optionally select N from a larger pool with quality *budget* constraints.
