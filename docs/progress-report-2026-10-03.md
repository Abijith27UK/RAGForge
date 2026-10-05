# Project Progress Update — ID4100 / AM3100

**Project:** RAGForge — Automated Domain-Specific RAG Knowledge Base Builder
**Date:** 3 October 2026 · **Branch:** `Dev_1_midterm`
**Verification:** every metric below was re-measured against the running system on 3 Oct 2026, not copied from documentation.

---

## 1. Timeline Alignment

### 1.1 Proposed vs. Actual

> **Week number assumption:** the original proposal document is not in the repository. Anchored on available evidence — git history begins **15 Sep 2026**, last commit **2 Oct**, and **three weeks** remain to the final presentation. That places the project at approximately **week 10–11 of ~13**: the final stretch, integration-and-evidence phase. Amend this line if the proposal's week numbering differs.

| Plan phase | Planned | Actual | Status |
|---|---|---|---|
| 1–2 | Foundation, domain analysis, source management | FastAPI + Next.js + SQLite + Qdrant, structured `DomainSpec`, explainable source scoring | Complete |
| 3 | Ingestion, chunking, embedding, Qdrant | PDF/HTML/TXT/MD/PPTX/DOCX parsers, section-aware + fixed-size chunkers, MiniLM-L6-v2, provenance payloads | Complete |
| 4 | Retrieval Lab, retrieval evaluation | Dense → pluggable registry: `dense`, `bm25`, `hybrid`, `hybrid_reranked` | Complete |
| 5 | UI | 16 Next.js routes, building and passing | Complete |
| 6 | Evaluation as research instrument | Frozen 28-question benchmark, strict-mode metrics, answer-quality evaluation | Complete |
| 7 | Experiments (source selection) | v1 (inconclusive → diagnosed and discarded), v2 (real coverage/efficiency result) | Complete |
| 8 | Robustness and final evidence | Corpus reliability engine, grounded answer engine, chat, answer evaluation | Complete |
| 9 | Human-authored reference answers | Not started | **Critical gap — §4.2** |

### 1.2 Completed Milestones Since the Proposal

**Phase A — Foundation (V1–V3).** End-to-end vertical slice: KB creation → domain spec → source discovery → source quality decisions → ingestion → chunking → embedding → Qdrant → retrieval → evaluation. Plus the benchmark lifecycle (`DRAFT → IN_REVIEW → APPROVED → FROZEN`) with reviewer attribution and immutable versions — the research-integrity backbone.

**Phase B — User knowledge ingestion (V4).** Upload → parse → chunk → index with full slide/page/section provenance. Key decision: **stale-vector safety** — old vectors are deleted *before* upsert on re-index, and a backend that cannot delete vectors is *refused* rather than corrupted.

**Phase C — Corpus reliability (V5).** An 11-check read-only integrity service, safe repair (plan → confirm → execute → verify), deterministic corpus fingerprints, document diffing, resumable bulk batches, and a 10/50/100/200-document scale benchmark.

**Phase D — Retrieval intelligence (V6).** Real BM25 with a persistent index and exact staleness detection, hybrid fusion with score normalization, Reciprocal Rank Fusion, optional cross-encoder reranking, MMR + document-cap diversity, per-stage timings, and an immutable `retrieval_runs` record per query.

**Phase E — Grounded answering (V7 → V7.2).** Query processing → retrieval → evidence assembly → evidence gate → generation → citation validation → answer, with a full audit trace. Then a five-state grounding vocabulary, conversation memory used for reference resolution only, and a three-pane chat UI.

**Phase F — Answer-quality evaluation (V8).** The ability to *score* groundedness rather than assert it — 14 answer-quality metrics, per-claim citation audits, and an explicit tri-state UNKNOWN for anything that cannot be honestly measured.

### 1.3 Deviation — Where the Project Is Behind, and the Recovery Plan

The build is not behind. **Human verification is behind**, and that distinction matters.

**Bottleneck 1 — ground truth is agent-authored.** All 28 benchmark questions were selected by the coding agent under explicit instruction, with quoted-passage provenance recorded per question, but `authorship.human_review` still reads `PENDING`. Every metric in the project inherits this caveat. This is not fixable by code.

**Bottleneck 2 — `correctness` is unmeasurable.** The frozen benchmark defines which *chunk* answers each question, never what the answer should *say*. `correctness` and `key_point_recall` are therefore **UNKNOWN for all 28 questions**. Fuzzy-matching an answer against retrieved text was deliberately rejected: it re-measures retrieval while labelling the result answer quality.

**Bottleneck 3 — off-domain gate defect, found by live probe on 3 Oct 2026.** The live API answered *"What are the EASA regulations for commercial drone certification?"* from the **Automobile** knowledge base with `status=grounded`, `grounding_state=ANSWERED`, at 40% lexical alignment. Root cause: the evidence gate computes lexical alignment as the *raw fraction of question content terms present in the evidence*, with no IDF weighting, so generic terms (`regulations`, `commercial`, `certification`) match any technical corpus. The abstain threshold is `MIN_ALIGNMENT_TO_PARTIAL = 0.15` (`gate.py:48`), evaluated at `gate.py:560`. This is the mechanism behind the measured `hallucination_rate = 0.179`.

Three off-domain probes were run to confirm the failure mode:

| Probe question | Result | Lexical alignment |
|---|---|---|
| EASA commercial drone certification | **wrongly answered** | 0.40 |
| Philippine labour law / holiday pay | correctly abstained | 0.14 |
| Sourdough baking at 450F | correctly abstained | 0.00 |

Consequence for documentation: the claim in `answering-architecture.md` that off-domain EASA questions abstain with `LOW_LEXICAL_ALIGNMENT` holds for the Naval Architecture KB but is **not a general guarantee**. That is a documentation defect, recorded rather than omitted.

**Recovery plan before the final presentation:**
1. IDF-weighted lexical alignment, reusing the `df` statistics already persisted by the BM25 index.
2. Human review pass over all 28 ground-truth selections.
3. 10 human-authored reference answers plus ~5 genuinely unanswerable questions.
4. Re-measure and report before/after for every affected metric.

---

## 2. Development Status

### 2.1 The Solution — Current Prototype State

The system is running and was verified end-to-end during this review.

| Check | Command | Result |
|---|---|---|
| Backend tests | `pytest -q` | **596 passed, 0 failed** (4m19s) |
| Typecheck | `npx tsc --noEmit` | clean, exit 0 |
| Frontend build | `npm run build` | success, **16 routes** |
| Qdrant | `GET :6333/collections` | live, **14 collections** |
| Backend | `GET :8000/api/system/health` | `{"status":"ok"}` |
| Live retrieval | `POST /retrieve` | `rr_4733ffd13f65`, dense, top-3 = 0.4014 / 0.3255 / 0.2831 |
| Live grounded answer | `POST /answer` | `grounded`, 3 claims, 3 citations, `ANSWERED / SUFFICIENT`, trace `atr_4bff922a3869` |
| Live off-domain probe | 3 questions | 2 of 3 abstained correctly, **1 of 3 wrongly answered** |

**Real data in the system:** 16 knowledge bases · 107 documents · **9,506 chunks** · 179 sources · 280 stored answers · 280 answer traces · 41 retrieval-evaluation runs · 6 answer-evaluation runs.

**Pipeline, technically:**

```
domain spec → knowledge requirements → source discovery → source quality decisions
→ ingestion (PDF/PPTX/DOCX/HTML/MD/TXT) → structure-aware chunking
→ embedding (MiniLM-L6-v2, 384-d, L2) → Qdrant (cosine)
→ retrieval (dense | bm25 | hybrid | hybrid_reranked)
→ evidence assembly → EVIDENCE GATE → answer generation
→ citation validation → answer + audit trace → answer-quality evaluation
```

**Retrieval quality — frozen benchmark, 28 questions, strict mode:**

| Run | Recall@K | Precision@K | MRR | NDCG@K |
|---|---|---|---|---|
| strict k=3 | 0.821 | 0.286 | 0.708 | 0.738 |
| strict k=5 | **0.964** | 0.200 | 0.737 | 0.793 |
| strict k=10 | **1.000** | 0.104 | 0.743 | 0.806 |

Precision@K is structurally low by design: most questions have exactly one relevant chunk in an 812-chunk corpus, so P@10 ≈ 0.10 is the ceiling for a perfect run.

**Answer quality — measured from live run `aerun_3da855772001`:**

| Metric | Value | Metric | Value |
|---|---|---|---|
| `pass_rate` | 0.821 | `citation_recall` | 0.821 |
| `citation_precision` | 0.286 *(lower bound)* | `evidence_support_rate` | 1.000 |
| `retrieval_hit_rate` | 0.964 | `unsupported_claim_rate` | 0.000 |
| `grounding_state_accuracy` | 1.000 | `abstention_accuracy` | 1.000 |
| `hallucination_rate` | **0.179** | `contradiction_rate` | 0.000 |
| `correctness` | **UNKNOWN** | `key_point_recall` | **UNKNOWN** |

**Research experiment result (source-selection v2).** Content-aware selection covered **11 / 13 / 17** of 28 benchmark questions at N = 5 / 8 / 11, against a random baseline's **4 / 4 / 10**, using **2.2–4.5× fewer characters per answerable question**. Retrieval-*ranking* superiority is **unproven**: on the only valid paired subset (6 questions at N=11), Recall was identical and MRR differed by one rank in favour of the baseline.

The v1 experiment was discarded after its instrument was found to assign 34 of 36 sources the identical score 0.4525, collapsing "quality selection" into alphabetical ordering; its QUALITY arm covered 0/28 questions. The v2 instrument produced 36/36 unique scores.

### 2.2 Refinement — Changes From the Original Key Idea

**Refinement 1 — the product pivot from "constructor" to "construction *and* measurement."** The original idea assumed that a running loop yields a good knowledge base. Building it exposed the opposite risk: a loop that runs happily on a bad corpus is worse than no loop, because it produces confident wrong answers. The deliverable therefore became **a system that can prove what it does and does not know**. This is why abstention is a first-class outcome with its own status and reason code, not an error path.

**Refinement 2 — abstention happens *before* generation, not after.** The single most important design decision: in `abstain_if_unsupported` mode, an insufficient `EvidenceGate` verdict **never calls the model at all**. The trace records `generation: skipped — evidence gate returned ABSTAIN; no model call was made (the abstention is deterministic)` (`answering/service.py:215`). There is no model call available to be tempted into guessing. Confirmed live: the Philippine-labour and sourdough probes both returned `INSUFFICIENT_EVIDENCE`.

**Refinement 3 — a measurement-correctness fix in the evaluator.** V8 initially reported `pass_rate = 0.000`, with 28/28 questions failing. The cause was a level-of-judgement bug in the evaluator, not a bad benchmark: `required_evidence` is a *necessity* label (chunks a correct answer **must** use), but it was being enforced **per claim** — so a claim about alternative fuels was failed for not citing the engine-definition chunk. The evaluator also asserted that cited-but-unlabelled chunks "do not address this question", which fabricates ground truth the benchmark does not contain.

Fix: completeness enforced **per answer** (strict, legitimate); unlabelled citations recorded as an **observation**, never a failure. `pass_rate` moved 0.000 → 0.821 with the benchmark file **byte-identical**. This was a measurement-correctness fix, not a tuning knob, and 8 tests lock the corrected semantics.

### 2.3 Creative Edge — Maintaining the Distinctive Approach

The differentiator is **refusal to assert what has not been measured**, enforced structurally rather than by intention:

- **UNKNOWN is a type, not a number.** Every answer-quality metric is a `Measured` triple (`value | measured | reason`). `Measured.unknown("no question produced a measured correctness")` is the literal value on screen. An uncomputable metric is never silently `0.0`.
- **`final_score` physically cannot lie.** A Pydantic `model_validator` rejects any result carrying a `final_score` whose `final_score_computable` is `False` — a score can never accompany an unmeasured input.
- **Heuristics are labelled where they are used.** `classification_method: "heuristic_rules"`, `entailment_is_model_based: false`, `support_check: "lexical_overlap"`. Semantic entailment is NOT IMPLEMENTED and the citation detail ends literally with *"Semantic support check: NOT PERFORMED"*.
- **Degradation is exposed, never silent.** A cross-encoder that cannot load yields `status: unavailable_fallback` with a reason, not a faked ranking. A generator falling back to mock sets `is_mock: true` on the Answer itself, plus a degraded-generator warning.
- **Unmeasured ≠ zero.** A stage that did not run has latency `null`, never `0`. A skipped trace stage is `skipped`, never `ok`.
- **Novelty is claimed narrowly and deliberately.** RAGForge does **not** claim novelty for ingestion, chunking or vector search — those exist in Dify and LangChain. The claimed contribution is the construction-and-measurement loop and its evidence discipline.

---

## 3. Preparation for Final Deliverables

### 3.1 Output Preview

A **live, un-mocked end-to-end demonstration** plus the research evidence:

1. Build a knowledge base live — Automobile Engineering, real source discovery and ingestion into real Qdrant.
2. Retrieval Lab — switch between `dense`, `bm25`, `hybrid`, `hybrid_reranked` on one query; show the stage-by-stage pipeline trace and score breakdown.
3. Grounded answering — an in-domain question producing a cited answer with provenance, then an off-domain question producing an explicit abstention with its reason code.
4. Chat with evidence — three-pane UI; click a citation to open the exact supporting chunk with page/slide/section.
5. Evaluation — live retrieval metrics on the frozen benchmark, and the source-selection v2 result stated with its limitation.
6. Answer Quality page — the measured-vs-UNKNOWN metric grid and per-question claim audit, including hallucination cases.

The strongest single artefact is the **UNKNOWN column**: showing that the system reports `correctness = UNKNOWN` for all 28 questions, with the reason, is a stronger research contribution than a fabricated 92%.

### 3.2 Presentation Format — Recommendation

**Live Videoconference, with a recorded backup.**

Reasoning: the strongest claim this project makes is that it measures its own limits correctly, and that claim is only credible under live challenge. A recorded video invites the question "is that staged?".

The demo is fully deterministic — `LLM_PROVIDER=mock` plus local embeddings — so the recording and the live run are identical on screen.

**If Recorded Video is preferred:** the demo already runs headless via `backend/scripts/` (`smoke_chat_v7.py`, `smoke_answer_eval_v8.py`, `run_real_answer_eval_v8.py`), so a capture script produces a reproducible recording.

### 3.3 Final Tasks — The Three Critical Items Remaining

**1. Human ground-truth verification and reference answers (highest priority).**
`authorship.human_review` is `PENDING`. A human must review all 28 questions against their quoted passages and author **10 reference answers plus ~5 genuinely unanswerable questions**. This is the only change that converts `correctness` and `key_point_recall` from UNKNOWN to measured, and it cannot be automated without becoming circular (an LLM grading an LLM).

**2. Fix the off-domain gate defect (IDF-weighted lexical alignment), then re-measure.**
Replace the raw term-hit fraction at `gate.py:560` with IDF-weighted alignment using the `df` statistics already persisted in the BM25 index. Target: `hallucination_rate` 0.179 → below 0.05, with a regression test asserting that EASA/drone, labour-law and cooking questions all abstain on the Automobile KB. **This must be reported as its own change with before/after numbers** — editing the generator and re-reporting the metric it affects is how a benchmark gets tuned to look good.

**3. Second-domain validation, plus committing and documenting the uncommitted work.**
The roadmap's top open item is a Naval Architecture knowledge base from ~50 lecture PPTs and ~20 PDFs, proving the domain-general path. Also blocking: **the entire V6–V8 body of work is uncommitted.**

---

## 4. Resource & Learning Check

### 4.1 Remaining Needs

| Resource | Status | Action needed |
|---|---|---|
| Human reviewer (faculty/peer) | **Required** | 2–3 h to sign off 28 ground-truth selections. This is the binding constraint. |
| Qdrant | Working | Local Windows build, 14 collections. No Docker required. |
| GPU / compute | Sufficient | MiniLM-L6-v2 and cross-encoder run on CPU. No cloud cost. |
| LLM API key | Optional | Whole system runs on `LLM_PROVIDER=mock`; all 596 tests are hermetic. A real key would permit reporting real LLM answers, but current numbers are already reproducible. |
| Second-domain corpus | Partial | Naval Architecture lecture material — access to the PPT/PDF set needed. |
| Final deliverables | Pending | Slide deck + recorded demo, ~15 min. |

**No procurement, budget, or external service is required to finish.**

### 4.2 Learning Progress — Most Prominent Outcomes

1. **Evaluation design and measurement validity.** The dominant learning. A metric can be wrong even when every number it prints is arithmetically correct — enforcing a necessity label at the wrong granularity produced a valid-looking `pass_rate = 0.000`. Ground-truth semantics must be understood before they are automated.

2. **Software architecture for experimentation.** Registry and factory patterns (`Retriever`, `VectorStore`, `Chunker`, `AnswerGenerator`, `CitationEntailmentEvaluator`) so variants are comparable without editing business logic. Proven: four retrieval strategies and two generators exist with zero `if strategy == …` branching in the answering layer.

3. **AI/LLM systems engineering.** Provider abstraction (`LLMProvider`, `AnswerGenerator`), injected dependencies, strict Pydantic structured output, and prompt-injection defense via `<EVIDENCE>` wrapping plus marker scanning.

4. **Software testing.** 596 hermetic tests with no network, no API key and no live Qdrant; vector factories stubbed at the module attribute. Tests encode *semantics*, e.g. asserting `pass_rate == derived from per_question[].passed`.

5. **Research methodology.** Frozen benchmarks, benchmark fingerprints, refusal of cross-subset comparisons, and **instrument validation before experiments**. The most transferable lesson: validating the measuring instrument before trusting its output. The v1 experiment's scorer produced 34/36 identical scores, silently converting quality selection into alphabetical ordering.

6. **Research integrity as professional practice.** A defect found in the system's own favour was documented rather than shipped around: the extractive mock generator walks evidence in rank order with a 5-claim budget, so with `top_k=5`, evidence ranked 4th or lower is **structurally uncitable** — and in 4 of 5 real failures the required chunk was retrieved at rank 5 and never cited. It was deliberately **not** fixed inside V8.

---

## Faculty Feedback Loop

**Mentor Comments**

**Status: On Track (engineering) / Needs Intervention (human verification)**

On **Track**: the vertical slice is real, tested and running; the measured results are defensible; the research-integrity discipline is the strongest aspect of the work.

**Needs Intervention** on one item: **human ground-truth review has not occurred.** This is not a code problem and cannot be closed by the student. It gates `correctness`, `key_point_recall`, and abstention-on-unanswerable, and is the single item standing between the project and publication-grade claims.

---

## Student Acknowledgment

I confirm that the progress reported here is accurate and the work is being carried out as per the faculty mentor's guidance. All metrics quoted were produced by executed pipeline runs against real Qdrant collections and a real SQLite store. Limitations and known defects — unverified human ground truth, UNKNOWN `correctness`, the off-domain gate defect, and the uncommitted V6–V8 work — are disclosed rather than omitted.

**Digital Signature:** ____________________  **Date:** 3 October 2026

---

## Appendix A — Verified Technical Detail

### A.1 Code volume

| Scope | New files | New lines |
|---|---|---|
| Backend application code (V6–V8) | 30 | 9,584 |
| Backend insertions into modified files | — | 1,449 |
| Frontend pages (answer, chat, answer-quality) | 3 | 2,079 |
| Frontend insertions into modified files | — | 724 |
| Backend test modules | 8 | 5,379 |
| **Total** | **41 new files** | **19,215 lines** |

Plus **6 architecture documents** (`retrieval-architecture.md`, `answering-architecture.md`, `grounding-and-citations.md`, `chat-architecture.md`, `answer-evaluation.md`, `answer-benchmark-design.md`) and three continuation checkpoints.

**Test suite: 596 tests across 25 test modules.** Largest: `test_answering_v7.py` (66), `test_answer_eval_v8.py` (63), `test_answering_v7_2.py` (59), `test_retrieval_v6.py` (58).

### A.2 Grounding and retrieval internals

**Five grounding states:** `ANSWERED` · `PARTIALLY_SUPPORTED` · `INSUFFICIENT_EVIDENCE` · `CONFLICTING_EVIDENCE` · `NO_RELEVANT_EVIDENCE`.

**Evidence gate signals (8, all disclosed):** `evidence_count`, `score_distribution`, `lexical_alignment`, `independent_source_agreement`, `provenance_completeness`, `source_trust`, `retrieval_agreement`, `conflicting_values`. A high retrieval score **alone** never yields "sufficient" — this is enforced by test.

**Gate thresholds:** `MIN_ALIGNMENT_TO_ANSWER = 0.34`, `MIN_ALIGNMENT_TO_PARTIAL = 0.15` (`gate.py:47–48`).

**Retrieval registry (verified live):** `bm25` (no embedding required) · `dense` (baseline) · `hybrid` · `hybrid_reranked`.

### A.3 Design decisions to defend

- **The BM25 index stores no chunk text.** Chunks remain the source of truth in SQLite; the persisted index holds only postings, `df` and document lengths. This keeps the index small and guarantees the provenance shape is identical to the dense path. Staleness is exact via `corpus_revision`, bumped only by `create_chunks` and `delete_chunks_for_document`.
- **Weight rescaling on partial fusion.** If one retrieval pool is empty, weights rescale to the survivor and the response says so. Keeping 0.65/0.35 would cap every fused score at 0.65 and falsely resemble poor relevance.
- **Three-valued entailment.** `SUPPORTED` / `NOT_SUPPORTED` / `UNKNOWN`. `LLMCitationEntailment` **raises** when no provider is available rather than degrading silently, so a run can never attribute LLM-judged metrics to a heuristic. Thirty antonym pairs provide polarity detection: pure term overlap scored "must be negative" against "positive" at 87% lexical overlap and called it SUPPORTED.
- **Conversation memory is not knowledge.** `messages` are never chunked, embedded, indexed or retrieved. `Message.used_as_knowledge` is permanently `False` so the UI can *assert* this rather than trust it.
- **Prompt-injection defence is heuristic and labelled as such.** Evidence is wrapped in `<EVIDENCE>` within the user message only, the system prompt states evidence cannot override instructions, and a marker scan records detections as warnings. A match does not prove the corpus is clean; no match does not prove an instruction is absent.

### A.4 Current limitations

1. Ground truth is agent-authored, human review `PENDING`.
2. `correctness` and `key_point_recall` are UNKNOWN for all 28 questions.
3. The benchmark contains 0 unanswerable questions, so abstention-on-unanswerable cannot be measured on the real corpus.
4. Entailment is heuristic lexical coverage; no model-based judge is active.
5. The default generator is a mock. LLM answers are not deterministically replayable (`temperature=0.2`, no recorded seed); the mock provider is.
6. The off-domain gate defect (§1.3) is open.
7. Scanned PDFs are not supported — OCR is the largest remaining ingestion gap.
8. V6–V8 is uncommitted.

---

## Appendix B — Reproduction

```bash
# Backend
cd backend && .venv/Scripts/python.exe -m pytest -q          # 596 passed
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

# Frontend
cd frontend && npx tsc --noEmit && npm run build            # clean, 16 routes
npm run dev                                                  # :3000

# Headless demos (no UI, deterministic)
.venv/Scripts/python.exe scripts/smoke_chat_v7.py
.venv/Scripts/python.exe scripts/run_real_answer_eval_v8.py

# Experiments
.venv/Scripts/python.exe scripts/report_v2_pool_scores.py --experiment ../benchmarks/source-selection-experiment-v1.json
.venv/Scripts/python.exe scripts/run_source_selection_experiment_v2.py \
  --experiment ../benchmarks/source-selection-experiment-v2.json --baseline-kb kb_f278c283c748
```

Qdrant is required on `:6333`. No LLM API key is required; `LLM_PROVIDER=mock` runs the real prompt path offline.