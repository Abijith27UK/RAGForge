# Design: Content-Aware Source Quality Scorer (v2)

**Status:** design + mechanism validated on the frozen 36-source pool (2026-09-15) — awaiting
approval for the source-selection v2 experiment. The v1 experiment
(`docs/experiment-source-selection-v1.md`), its results, and the
`automobile-engineering-baseline-v1` benchmark remain FROZEN and untouched.

## 1. Problem statement (from v1)

The v1 instrument check showed 34/36 homogeneous Wikipedia sources scored an identical 0.4525,
so "top-N by quality" degenerated into alphabetical selection. Root causes, verified in code:

1. **No content access:** the v1 scorer ranks sources by URL/title/notes text only
   (`_signal_relevance`, `_signal_evidence`). It never reads the page.
2. **Content-free domain spec:** the deterministic mock `DomainSpec` contains generic terms
   ("Automobile Engineering: Fundamentals", "Core Automobile Engineering theory"), so
   spec-term matching cannot separate automobile articles from each other.
3. **Authority/type domination:** with all metadata signals nearly constant, the composite
   collapses; and in heterogeneous pools a `.gov` TLD could outrank topical fit regardless
   of content.

## 2. Design principles

- Evaluate **actual source content** against **explicit domain knowledge requirements**.
- Deterministic and embedding-based; no opaque LLM 0–1 score as the sole mechanism (the
  embedding model is a fixed, inspectable component; evidence passages are quoted).
- Strict separation of concerns: SOURCE AUTHORITY, CONTENT RELEVANCE, DOMAIN COVERAGE,
  ACCESSIBILITY, RECENCY are reported independently; the composite weights content highest.
- Explainability: every score carries per-requirement detail, evidence passages, matched
  queries, thresholds, confidence, and limitations.

## 3. Architecture (final, as implemented)

### Stage A — Domain Knowledge Map

Versioned JSON per domain in `backend/app/services/source_quality/domain_maps/`.
`automobile_engineering.json` (v1.0.0): 9 requirement areas — engine; transmission &
drivetrain; braking; steering; suspension; chassis & body structure; vehicle dynamics &
handling; electric vehicles & electrification; automobile fundamentals & general systems —
each with `name`, `description`, deterministic `seed_terms`, and `weight` (weights sum 1.0,
validated at load).

Provenance: `curated domain configuration (agent-drafted, human review pending)`, aligned with
the domain definition; NOT derived from benchmark questions. When a real LLM is configured, the
analyzer's `knowledge_requirements` may generate future maps; the provenance string records
which path was used. The map's areas coincide with the benchmark's subdomain taxonomy because
both derive from the same domain definition — this construct overlap is disclosed as a
limitation of the eventual v2 experiment (benchmark questions/ground truth are untouched).

### Stage B — Content acquisition & caching

SSRF-validated fetch (existing `validate_public_http_url`), ≤256 KB, HTML→text with the
project's BeautifulSoup parser, cached at `backend/data/content_cache/` under a **v2-namespaced
key** (the extraction pipeline changed, so older cached text can never be silently reused).
Failures degrade honestly (accessibility signal + limitation), never fabricated.

Boilerplate removal (essential for Wikipedia): besides `script/style/nav/footer/header/aside/
form/noscript`, the extractor strips `table.navbox`, `table.vertical-navbox`, `table.sidebar`,
`table.infobox`, `table.metadata`, `table.toc`, `div.navbox`, `div#toc`, `div.thumb`,
`span.mw-editsection`, and `figure`. Without this, Wikipedia's "Automobile series" navigation
box masquerades as domain coverage (observed: a brake page "covering" engine+transmission+EV
via its nav menu — caught by the evidence audit, then fixed).

### Stage C — Content relevance & coverage (the new core)

Each requirement area is embedded as **multiple queries** (its description plus every seed term
separately). Short, sharp queries match far better with sentence embeddings than one long
blended query. Source text is windowed (400 chars, 80 overlap) and embedded. Then:

- `sim(a, s) = mean(top-3 (query, window) pairs)` across all of the area's queries/windows.
- **Prose-window filter:** a window counts only if it has ≥45 words AND ≥4.0 words per line.
  Link/nav lists run ~1 word per line; real paragraphs are dense. (A words-only floor is
  insufficient: 100 words of link debris fits a 400-char window.)
- **Similarity normalization:** raw MiniLM cosines have a high floor (~0.2 for unrelated
  text); `sim = max(0, raw − 0.20) / (1 − 0.20)` makes 0 = no relatedness and scores
  comparable across sources. τ is applied to the normalized scale.
- **Evidence:** the top window's text (quoted, ≤240 chars) plus which queries matched.
- Area **covered** iff `sim(a, s) ≥ τ` (τ = 0.50 normalized; equivalent to raw ≈ 0.60).
- Outputs:
  - **CONTENT_RELEVANCE(s)** = Σ_a weight_a · sim(a, s) ∈ [0, 1]
  - **DOMAIN_COVERAGE(s)** = |{a : covered}| / |A| ∈ [0, 1]
  - per-area detail: similarity, covered, evidence, matched_by
  - **CONFIDENCE(s)** = 0.5·(mean margin from τ) + 0.5·(volume factor), with explicit
    limitations when content was short, over-filtered (0 prose windows), or unfetchable.

Zero-prose-window and tiny-content cases return honest zeros with limitation strings rather
than crashing or guessing.

### Stage D — Re-separated composite

| Signal | Weight (v2) | Source |
|---|---|---|
| content_relevance | **0.40** | Stage C |
| domain_coverage | **0.20** | Stage C |
| authority | 0.15 | v1 `_signal_authority` (unchanged) |
| accessibility | 0.10 | real probe status (unchanged) |
| source_type | 0.05 | v1 signal (unchanged) |
| recency | 0.05 | v1 signal (unchanged) |
| duplication | 0.05 | v1 signal (unchanged) |

Content signals (0.60) dominate metadata signals (0.25): an authoritative page that does not
cover the domain can no longer outrank a relevant one. `scorer_version:
heuristic-v2-content-aware`; the domain-map version and provenance are embedded in every
assessment. The v1 scorer and default behaviour remain unchanged (`SOURCE_SCORER_VERSION=v1`);
v2 is opt-in until approved.

## 4. Design iterations (honesty log)

The pool report triggered the user's stop-and-improve rule twice before validation passed:

1. **Single blended query per area** → scores varied but assignments were semantically wrong
   (`Tire`/`Battery_management_system` near zero coverage everywhere; `Anti-roll_bar` ranked
   #1). Cause: long blended queries embed poorly with MiniLM. Fix: multi-query areas.
2. **Navbox leakage** → 35/36 unique scores and a plausible-looking ranking, but the evidence
   audit showed coverage driven by Wikipedia navigation boxes (Drum_brake "covering"
   engine/transmission/EV via menu text; Vehicle_frame's chassis evidence was footer
   categories). Fix: navbox/infobox stripping in extraction + words-per-line prose filter +
   similarity offset normalization; v2-namespaced content cache.

## 5. Instrument check on the frozen 36-URL pool (validation results)

`backend/scripts/report_v2_pool_scores.py --experiment benchmarks/source-selection-experiment-v1.json`
(report: `benchmarks/v2-pool-report.json`; embedding: MiniLM 384-d; wall time 50 s, cached).

- **Score distribution (content_relevance):** min 0.141, max 0.374, spread across 18 distinct
  0.01-buckets; **36/36 unique** content_relevance values, **36/36 unique** composites.
  Verdict: DISCRIMINATING (stop condition not triggered).
- **Top 10:** Regenerative_braking (0.446), Suspension_(vehicle) (0.419), Electric_vehicle
  (0.411), Anti-lock_braking_system (0.406), Steering (0.405), Anti-roll_bar (0.402),
  Brake (0.401), Tire (0.396), Disc_brake (0.394), Drum_brake (0.391) — all core-automobile
  system pages, matching the domain definition.
- **Bottom 10:** Battery_management_system (0.317), Lithium-ion_battery (0.331),
  Dual-clutch_transmission (0.342), Charging_station (0.347), Fuel_injection (0.347),
  Transmission_(mechanical_device) (0.349), Ackermann_steering_geometry (0.351),
  Double_wishbone_suspension (0.353), Wankel_engine (0.354), Diesel_engine (0.358).
- **Interpretation of the bottom ranks:** the composite is breadth-aware (coverage = 0.20).
  Specialist pages (Wankel, Diesel, DCT, Ackermann) cover exactly one area, so they rank below
  pages spanning two (Regenerative_braking = braking+EV). Their *per-area* similarities are
  healthy (Four-stroke engine-area sim 0.606 on genuine combustion prose; Transmission
  transmission-area sim 0.455 on gearbox prose). This is the documented behaviour of the
  coverage term, not an error; a depth-first variant (max-area instead of weighted mean) is a
  possible v3 knob, deliberately not added now.
- **Coverage by area (sources covering at τ):** engine 5/36, transmission 1/36, braking 5/36,
  steering 1/36, suspension 5/36, chassis 1/36, vehicle-dynamics 0/36, electric-vehicles 6/36,
  fundamentals 0/36. τ = 0.50 normalized is strict: specialist pages pass only their own area,
  which is the intended honest behaviour (no more nav-menu phantoms).
- **Evidence audit performed:** top-area evidence passages were manually inspected for
  Regenerative_braking, Four-stroke_engine, Diesel_engine, Automobile, Transmission, BMS —
  all genuine topical prose, no boilerplate.

## 6. Tests

`backend/tests/test_content_scorer.py` (12 tests, deterministic — hashed bag-of-words embedder
stands in for the real model): domain-map validation, specialist-vs-unrelated separation,
zero-overlap ⇒ ~0, prose-window filtering, offset arithmetic, navbox stripping end-to-end,
composite weighting, authority-does-not-beat-relevance, honest unfetchable degradation,
evidence/matched_by presence. Full backend suite: **78 passed**.

## 7. Awaiting approval — next step on approval

Run source-selection experiment **v2**: QUALITY_SELECTED (v2 composite) vs RANDOM_BASELINE
over the same frozen pool and seeds protocol as v1, frozen benchmark via content-hash GT
translation, coverage separated from retrieval metrics. v1 artifacts remain frozen.

## 8. Out of scope (unchanged from project constraints)

BM25/hybrid/rerank, semantic chunking, LLM-as-judge, automatic optimization, MCP, chatbot,
multi-agent. The frozen baseline benchmark and v1 experiment artifacts are not modified.
