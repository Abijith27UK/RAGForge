# RAGForge — Roadmap

## Done (initial MVP, Phases 1–8)

- ✅ Foundation: FastAPI backend, Next.js frontend, SQLite, config, health/status endpoints, tests, .env.example
- ✅ Knowledge base creation (schema, API, UI)
- ✅ Domain analysis (LLM provider abstraction, structured DomainSpec, labelled dev mock)
- ✅ Source management (user-URL + arXiv discovery, explainable quality engine, ACCEPT/REVIEW/REJECT decisions, UI)
- ✅ Ingestion + chunking (PDF/HTML/TXT/MD, content hashing, dedup, section-aware + fixed-size chunkers, chunk inspection UI)
- ✅ Embeddings + Qdrant (provider abstraction, local SentenceTransformers, provenance payloads)
- ✅ Retrieval Lab (dense retrieval, filters, provenance display)
- ✅ Evaluation (Recall@K, Precision@K, MRR, NDCG; per-question + aggregate; question management UI)

## V4 — User knowledge ingestion + KB maturation (done)

- ✅ **Source modes** — `EXTERNAL` / `USER_PROVIDED` / `MIXED` on the KB, defaulting to
  `EXTERNAL` for backwards compatibility
- ✅ **Upload workflow** — multipart multi-file upload with drag & drop, upload progress,
  file-name sanitisation, size limits, magic-byte sniffing and OOXML container integrity
- ✅ **Parsers** — `PdfParser`, `PptxParser` (slide + title + speaker notes), `DocxParser`
  (heading path + tables), `HtmlParser`, `MarkdownParser`, `TextParser`, plus an explicit
  `LegacyPptParser` rejection; all behind `PARSER_REGISTRY`
- ✅ **Document library** — statuses `UPLOADED → PARSING → PARSED → CHUNKING → INDEXING →
  READY / FAILED`, inspect, read extracted text, view chunks, rebuild, re-index, replace, remove
- ✅ **User source quality model** — `UserProvidedIntegrityScorer` (validity, extraction,
  duplication, structure, uploader-asserted relevance). No authority/recency/accessibility
  weight exists in it; user files are never downgraded for lacking public authority signals
- ✅ **Incremental ingestion** — one shared `index_documents()` for full and per-document
  builds; only the new document is chunked, embedded and indexed
- ✅ **Stale-vector safety** — old vectors deleted before upsert on re-index, replace and
  delete; a backend that cannot delete vectors is refused rather than corrupted
- ✅ **Provenance** — page, slide, slide title, section path, document version and KB version
  on every chunk and every retrieval result; absent values omitted, never invented
- ✅ **KB overview** — one honest endpoint reporting source mode, documents, chunks, vectors,
  embedding model, vector backend, chunking strategy, build version/status and *optional*
  evaluation status
- ✅ **UI** — guided Create KB wizard, KB overview, Document Library page, improved
  Retrieval Lab, Sources page split into "User provided" vs "Externally discovered"
- ✅ **Benchmark integrity** — read-only `verify_benchmark_integrity.py` gate; the frozen
  Automobile Engineering benchmark is verified untouched

## Next (recommended order)

1. **Real user-knowledge validation run** — build a Naval Architecture knowledge base from
   ~50 lecture PPTs, ~20 PDFs and notes; verify slide/page provenance against the originals,
   then run Retrieval Lab queries and check citation usefulness by hand.
2. **Ground-truth authoring process for a second domain** — the Automobile benchmark is
   hand-made and domain-specific. Any new domain needs the same human process
   (author → review → APPROVED → snapshot → FROZEN → evaluate) before its retrieval quality
   can be measured at all. Tooling to make that authoring faster is the real gap, not
   automatic ground-truth generation.
3. **Execute source-selection v3** (deliberately, after `--dry-run`). The benchmark is frozen
   and the gate passes; the run builds 18 corpora and takes a long time, so it must be a
   conscious decision rather than an automatic one.
4. **Background ingestion** — a job queue so a 500-document upload does not block the request.
5. **Ground-truth benchmark authoring UI** — attach expected chunk/document IDs by selecting
   from real chunks instead of relying on the keyword heuristic.
6. **OCR** — the honest failure today is "scanned PDFs are not supported"; Tesseract or a
   vision model would close the biggest remaining ingestion gap.
7. **CSV / XLSX ingestion** — the `PARSER_REGISTRY` extension point already accepts them.

## V3 foundation (done)

- ✅ **Phase A** — benchmark lifecycle (DRAFT→REVIEW→APPROVED→FROZEN, reviewer attribution,
  revision-on-approved-edit, freeze protection) + immutable benchmark versions; only FROZEN
  versions usable for official evaluation; full review UI on the Evaluation page
- ✅ **Phase B (infra)** — multi-seed source-selection v3 runner + design JSON, gated on a frozen
  benchmark version (NOT executed until human review completes)
- ✅ **Phase C (arch)** — chunking strategy registry + strategy/config provenance on chunks and KBs
- ✅ **Phase D (arch)** — vector-backend factory (`create_vector_store`) + `vector_backend` on KB;
  qdrant unchanged as default
- ✅ Experiments page rendering frozen v1/v2 artifacts verbatim via read-only endpoints

## Next phases (planned, not implemented)

- ~~**Human review of the 28 agent-authored benchmark questions**~~ — ✅ complete; the benchmark
  is APPROVED/FROZEN and V4 never touched it. Source-selection v3 is unblocked but still
  requires a deliberate `--dry-run` first, then a conscious execute.
- **Phase E — TurboVec** experimental vector backend behind the factory, with a controlled
  Qdrant-vs-TurboVec comparison (quality metrics separated from latency/size metrics)
- **Phase F** — BM25 → hybrid → optional reranking behind the `Retriever` registry
- **Phase G** — version/build architecture for build→evaluate→diagnose→rebuild optimization
- **Phase H** — MCP server exposing RAGForge operations as external tools
- **Optional LLM answering layer** — retrieve → context → answer → citations. Deliberately not
  built yet: retrieval quality and provenance come first.

## Later phases

- BM25 + hybrid retrieval, reranking (cross-encoder or LLM-based) behind the retrieval interface
- LLM-as-judge context relevance and groundedness metrics (kept strictly separate from retrieval metrics)
- Semantic/domain-aware chunking strategies
- Automated optimization loop: propose configuration changes from evaluation weaknesses
- CSV/XLSX/OCR ingestion; additional discovery providers (Semantic Scholar, PubMed)
- Qdrant docker-compose file and dev-container setup
- Experiment 1–5 comparisons from the research plan, documented with real results

## Explicitly out of scope (for now)

Autonomous agents, MCP, scheduling, auth/billing, multi-user, cloud deployment, chat UI,
voice, mobile, elaborate visualizations.

---

## V5 — Corpus reliability (delivered)

| # | Item | Status |
|---|---|---|
| 1 | Bulk ingestion: persistent batches, per-item lifecycle, resumable | **done** |
| 2 | Corpus manifest + summary | **done** |
| 3 | `CorpusIntegrityService` — 11 checks, read-only | **done** |
| 4 | Safe repair: plan → confirm → execute → verify | **done** |
| 5 | Corpus fingerprint + deterministic versions | **done** |
| 6 | Document diff (added / removed / changed / unchanged) | **done** |
| 7 | Stage instrumentation + 10/50/100/200-document scale benchmark | **done** |
| 8 | Corpus Command Center UI | **done** |
| 9 | Document inspector (manifest-backed, per-document provenance) | **done** |
| 10 | Naval Architecture workflow documentation | **done** |
| 11 | Domain-independent ground-truth status (`NOT_AVAILABLE`) | **done** |
| 12 | Test suite: 63 corpus tests, hermetic, no live Qdrant | **done** |
| 13 | Documentation incl. `docs/corpus-reliability.md` | **done** |

## V6 — Retrieval intelligence + grounded answering (in progress)

Planned in 20 phases; status below reflects what is actually in the repository.

| phase | item | status |
|---|---|---|
| 1 | Repository audit (read-only) | **done** |
| 2 | Registry cleanup (`RetrieverSpec`, lazy built-ins, no hard-coded strategy) | **done** |
| 3 | Real BM25 + persistent index + exact staleness detection | **done** |
| 4 | Hybrid retrieval with score normalization + weighted fusion | **done** |
| 5 | Reciprocal Rank Fusion behind the same interface | **done** |
| 6 | Optional reranking (cross-encoder + honest fallback + status) | **done** |
| 7 | Diversity: per-document cap + MMR | **done** |
| 8 | Retrieval configuration + retrieval run persistence | **done** |
| 9 | Query processing (lightweight, deterministic) | **done in V7** |
| 10 | Grounded `AnsweringService` (OpenAI-compatible, provider-agnostic) | **done in V7** |
| 11 | Citation validation (no invalid citation can appear trusted) | **done in V7** |
| 12 | Evidence gating + abstention | **done in V7** |
| 13 | Answer API (`POST /api/knowledge-bases/{kb_id}/answer`) | **done in V7** |
| 14 | Answer UI + evidence panel (reusing the V5 design system) | **done in V7** |
| 15 | Retrieval Lab V2 (strategy/fusion/param selection + pipeline trace) | partial — strategy selection + retrieval-vs-answer comparison done; fusion/rerank parameter controls not started |
| 16 | Evaluation extension comparing the four strategies on frozen benchmarks | planned |
| 17 | Retrieval experiment runner producing immutable artifacts | planned |
| 18 | TurboVec (optional, experimental, only if locally verifiable) | not started |
| 19 | Security / prompt-injection tests for the answering layer | **done in V7** |
| 20 | Documentation + final verification | ongoing |

Observability groundwork shipped with phase 8 (`retrieval_runs` + `retrieval_run_id`).

## V7 — Grounded answer engine (delivered)

Full detail: [answering-architecture.md](answering-architecture.md).

| # | Item | Status |
|---|---|---|
| 1 | `QueryPlan` + deterministic `QueryProcessor` (heuristic-labelled) | **done** |
| 2 | `Evidence` model + provenance-preserving assembly + dedup with reasons | **done** |
| 3 | `EvidenceGate` → `ANSWER / PARTIAL_ANSWER / ABSTAIN / ASK_CLARIFICATION` (8 signals) | **done** |
| 4 | `AnswerGenerator` ABC + LLM impl + deterministic mock (no key needed) | **done** |
| 5 | Structured `Answer` / `Claim` / `Citation` with Claim → Evidence → Source chain | **done** |
| 6 | CitationValidator + validation pipeline with recorded failure actions | **done** |
| 7 | Abstention + partial answers (first-class, no model call when abstaining) | **done** |
| 8 | `AnswerTrace` audit trail + persistence (`answers`, `answer_traces`) | **done** |
| 9 | `POST /answer`, `GET /answers/{id}`, `GET /answer-traces/{id}` | **done** |
| 10 | Prompt-injection defense (`<EVIDENCE>` wrapping + marker scan) | **done** (HEURISTIC) |
| 11 | Answer UI + "How was this answer produced?" trace panel | **done** |
| 12 | Retrieval Lab strategy selection + retrieval-vs-answer comparison | **done** |
| 13 | Tests: 96 V7 tests (query/evidence/gate/generator/citation/security/trace/API) | **done** |
| 14 | Docs: `answering-architecture.md` + checkpoint | **done** |

Not implemented in V7 (explicitly): semantic/entailment validation, answer-level
benchmark, query decomposition/multi-query retrieval, calculation tools, answer
modes beyond `grounded`/`abstain_if_unsupported`, streaming.

---

## V7.2 — Grounded knowledge assistant

Extends V7 into a grounded assistant: the chat layer, the five-state grounding
vocabulary, explicit query traces, full citation provenance and answer-run
observability.

| # | deliverable | status |
|---|---|---|
| 1 | `QueryTrace` (rewritten_query / subqueries / processor_version / timing_ms) + `QueryNature` detection | **done** |
| 2 | `EvidenceSelector` ABC + provenance-preserving implementation | **done** |
| 3 | `FallbackAnswerGenerator` — fallback exposed, never silent | **done** |
| 4 | `Citation` provenance fields (`source_title`, `source_type`, `page_number`, `slide_number`, `section_path`, `content_hash`) | **done** |
| 5 | `GroundingGate` + five explicit states (`ANSWERED`, `PARTIALLY_SUPPORTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, `NO_RELEVANT_EVIDENCE`) | **done** |
| 6 | `Conversation` / `Message` — memory for reference resolution only, never knowledge | **done** |
| 7 | `POST /chat` + conversation + answer-run endpoints | **done** |
| 8 | `AnswerRun` observability record (measured latencies, versions, reasons) | **done** |
| 9 | 3-pane grounded chat UI (corpus / conversation / evidence) | **done** |
| 10 | Tests: 99 new (65 unit + 34 API); full suite **511 passed** | **done** |
| 11 | Docs: `grounding-and-citations.md`, `chat-architecture.md`, checkpoint | **done** |

Not implemented in V7.2 (explicitly): semantic/entailment validation,
answer-level benchmark or any answer-quality metric, query decomposition,
streaming, regeneration UI, multi-turn answer reuse, prompt-injection filtering
(detection is a heuristic scan), conversation history as a retrievable source.


## V8 — Answer-quality evaluation (delivered)

Extends V7.2 with the measurement that was missing: groundedness can now be
**scored rather than asserted**.

| # | Item | Status |
|---|---|---|
| 1 | Audit of existing ground truth → discovered the frozen benchmark has **no expected answers** | **done** |
| 2 | Answer benchmark with **evidence-based** ground truth (`answer-quality-automobile-v1.json`) | **done** |
| 3 | `AnswerEvaluator` framework, deterministic offline evaluator (`deterministic-evidence`) | **done** |
| 4 | Citation entailment + claim-level audit, incl. polarity/negation conflicts | **done** |
| 5 | Grounding-gate, abstention and answerable-vs-unanswerable evaluation | **done** |
| 6 | Immutable evaluation runs with full producer identity + comparability rules | **done** |
| 7 | Comparison matrix refusing cross-subset comparison | **done** |
| 8 | Failure-analysis UI (Answer Quality page) with claim audits | **done** |
| 9 | Regression tests: 83 new (63 unit + 20 API); suite 596 passed | **done** |
| 10 | Docs: `answer-evaluation.md`, `answer-benchmark-design.md`, checkpoint | **done** |
| 11 | Human-authored reference answers | **not done — blocks `correctness`** |

### What V8 could NOT measure, and why

`correctness` and `key_point_recall` are **UNKNOWN for all 28 questions**. The
frozen benchmark defines which *chunk* answers each question, not what the
answer should *say*. Fuzzy-matching an answer against retrieved text would
re-measure retrieval and relabel it answer quality, so it is not implemented.
The shipped benchmark also contains **0 unanswerable questions**, so
abstention-on-unanswerable cannot be measured on the real corpus.

Not implemented in V8 (explicitly): model-based entailment as the default,
autonomous optimization loops, giant benchmark generation, LLM fine-tuning,
TurboVec, MCP, OCR, CSV/XLSX, auth, cloud deployment, multi-user.

### Next candidates (not started)

1. **Author human reference answers** for a ~10-question subset of the Automobile
   benchmark, plus ~5 genuinely unanswerable questions. This is the **single
   highest-value next step**: it is the only thing that converts `correctness`
   and `key_point_recall` from UNKNOWN into measured numbers, and it cannot be
   automated without becoming circular (an LLM grading an LLM).
2. **Fix the extractive mock generator's rank truncation.** It walks evidence in
   rank order with a fixed 5-claim budget, so evidence ranked 4th or lower is
   structurally uncitable. Measured: the required chunk was retrieved at rank 5
   and never cited in 4 of 5 real failures. Deliberately not fixed inside V8 —
   changing the generator and then re-reporting the metric it affects is how a
   benchmark gets tuned to look good.
3. **Benchmark authoring UI** — make the `DRAFT → IN_REVIEW → APPROVED → FROZEN`
   lifecycle usable for a non-Automobile domain, so Naval Architecture can acquire
   real ground truth. This is the only route to ever reporting retrieval quality for
   a new domain.
4. **Model-based entailment** — `LLMCitationEntailment` exists and refuses to run
   without a provider rather than degrading silently, but the heuristic remains
   the default so every verdict stays reproducible.
5. **Query decomposition** — the multi-part question path is a recorded warning
   today; implementing it needs multi-query retrieval plus fusion.
6. **Watch mode** — detect a changed source document and offer a targeted
   re-index, using the document diff machinery that now exists.


## V8 continuation — answer evaluation & reliability (2026-10-06, delivered)

Extends the V8 layer against the 20-step V8 continuation spec: audited first,
then extended (no duplicated stack, no UI-first rewrite, frozen artifacts
untouched). Architecture: [`answer-evaluation-architecture.md`](./answer-evaluation-architecture.md);
experiment: [`answer-evaluation-v1.md`](./answer-evaluation-v1.md).

| Step | Item | Status |
|---|---|---|
| 1 | Audit of repo vs 20-step spec | **done** |
| 2 | Schema: claim ratios, fabricated/unsupported citation rates, relevance, completeness metrics — all `Measured` | **done** |
| 3 | Benchmark lifecycle `draft → approved → frozen` + `official` gating (FROZEN required) + per-question review metadata | **done** |
| 4 | Human review workflow: append-only `answer_reviews`, attributed, closed vocabularies, history never overwritten | **done** |
| 5 | `AnswerEvaluator` ABC → deterministic / **human** / **LLM** implementations, provider-optional, graceful degradation | **done** |
| 6 | Claim-level five-state `evaluated_state` (SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED/CONTRADICTED/UNVERIFIABLE) + ratios | **done** |
| 7 | Citation evaluation: precision/recall + `fabricated_citation_rate` + `unsupported_citation_rate` | **done** |
| 8 | Answer relevance: IDF-weighted coverage (measured: plain coverage is inverted), threshold 0.30 from data, abstention exempt | **done** |
| 9 | Groundedness & abstention — extended (existing V8 layer) | done (prior) |
| 10 | Top-level `/api/answer-evaluation-runs` endpoints | **done** |
| 11 | Comparability: strict refusal + intersection plan with `INCONCLUSIVE` / `NOT_COMPARABLE` verdicts | **done** |
| 12 | Experiment artifacts: spec + runner + **real results** + report | **done** (non-official by design) |
| 13/15 | UI: claim states, relevance + method labels, completeness, review form + history, derive-human-run, lifecycle/official badges | **done** |
| 14 | `HumanAnswerEvaluator` + `LLMAnswerEvaluator` with provider/model/prompt metadata + `judge_raw` audit | **done** |
| 16 | Reliability dashboard (separate page, **no combined score**, verdict-first comparison) | **done** |
| 17 | 20/20 listed test cases + live scratch-KB smoke (Qdrant 14→14, scratch cleaned) | **done** |
| 18 | Data safety: Qdrant count asserted, frozen artifacts 0 written, impact reported | **done** |
| 19 | Docs: architecture + experiment report + README/architecture/roadmap/verification updates | **done** |
| 20 | Not implemented (as specified): TurboVec, MCP, OCR, CSV/XLSX, auth, cloud, multi-user, … | respected |

Tests: **693 backend passed** (596 prior + 97 new V8-continuation tests: 6 new
test files plus additions to the two existing V8 files), frontend
`tsc --noEmit` clean, `npm run build` clean.

### Discrepancies surfaced (not silently accepted)

1. The spec claimed **513 tests** — the pre-continuation suite was already
   **596**.
2. The spec claimed the Automobile 28-question benchmark is **HUMAN REVIEWED**
   — both frozen artifact JSONs still read `authorship.human_review = PENDING`
   (roadmap/APPROVED notes refer to the *retrieval* benchmark lifecycle; the
   answer artifact's own string was never updated). The artifacts were left
   untouched, so the answer benchmark loads as `lifecycle=draft` and official
   runs against it are refused by design.

### Still not measured

`correctness`, `key_point_recall` / `expected_information_coverage` and
`reference_answer_similarity` remain UNKNOWN on the real corpus — no human
reference answers exist. The new human-review workflow (STEP 4) + derived
human-evaluation runs are the path to measuring correctness; LLM-as-judge is
available as an explicitly model-based, non-ground-truth option.
7. **Streaming ingest** — process a 200-file batch progressively instead of
   synchronously, for genuinely large corpora.


## V9 continuation — evaluation → optimization loop (2026-10-07, partly delivered)

Closes the loop from measurement to a justified, safe, recorded configuration
change. Full detail: [`v9-evaluation-to-optimization.md`](./v9-evaluation-to-optimization.md);
subsystem docs: [`failure-analysis.md`](./failure-analysis.md),
[`retrieval-diagnostics.md`](./retrieval-diagnostics.md). Audit:
[`v9-audit-report.md`](./v9-audit-report.md).

| Phase | Item | Status |
|---|---|---|
| 0 | Audit of the V8 state before any change | **done** (and corrected: the audit's first draft named the retrieval benchmark as the answer benchmark) |
| 1 | Benchmark review workflow: append-only per-question reviews, `DRAFT → REVIEWED → APPROVED → FROZEN`, `require_reviewed_benchmark` separate from the FROZEN gate | **done** — but **28/28 questions remain unreviewed** |
| 2 | Failure taxonomy: 11 closed labels, evidence + reason + categorical confidence, `UNKNOWN` preferred over a guess | **done** |
| 3 | Retrieval diagnostics: ranked chunks with provenance, first relevant rank, depth-independent recall, selection-vs-retrieval split, **UNKNOWN never 0** | **done** |
| 4 | Controlled experiment record: deterministic ids, append-only store, immutable artifacts | **done** (no live run) |
| 5 | Recommendation engine: deterministic, evidence-backed, `experiment_required=True`, never writes production config | **done** |
| 6 | Baseline vs candidate: per-metric rows, declared directions, protocol identity check, guardrail-aware accept/reject | **done** |
| 7 | Statistical validity: exact sign test + Student's t (stdlib, scipy-verified), always **exploratory** at n=28 | **done** |
| 8 | Off-domain gate: explicit score polarity, sample floor | **framework done; NOT MEASURABLE at n_off = 1** |
| 9 | Mock-generator limitation study (5-claim rank truncation) | **done** — limitation documented, **not** fixed by design |
| 10 | UI "Why did this answer fail?" | **NOT DONE** — backend only |
| 11 | Experiment integrity: checksums, vector-loss detection, scratch-KB isolation before running | **done** |
| 12 | Testing | **done** — 978 backend tests, frontend clean |
| 13 | Real Automobile experiment | **NOT RUN** — precondition unmet |
| 14 | Documentation | **done** |

### Why Phase 13 did not run

Not for lack of machinery — the framework is complete and verified offline. The
**preconditions** fail: the benchmark is DRAFT with 28 unreviewed questions and
no human reference answers, so `correctness` and `key_point_recall` are UNKNOWN
by construction and no candidate configuration can be justified from measured
data. Running it anyway would produce a record whose own `conclusion()` reads
`INTEGRITY NOT VERIFIED` or `NO VALID COMPARISON`. Reporting that as a result
would be worse than reporting no experiment.

This also explains why roadmap item **#2** (mock-generator rank truncation)
stays unfixed: raising the claim budget would tune the instrument to improve the
number it produces.

### V10 recommended order

1. **Author and review the 28 Automobile reference answers** through the Phase-1
   workflow, then freeze, then run the Phase-13 experiment. This is the only path
   that turns UNKNOWN into measured, and everything else waits on it.
2. **Phase 10 UI** — the failure-analysis view, so step 1 can be done efficiently.
3. **V9 API routes** — diagnostics, classifications, recommendations and
   experiment records over HTTP.
4. **Off-domain gate measurement** — author ≥ 10 off-domain questions and measure.
5. **Reranker-stage diagnostics** — pre/post-rerank rank comparison.

## V10 — Human-reviewed benchmark ground truth (2026-10-08, delivered)

Builds the full authoring → review → freeze workflow that item 1 above needs,
so `correctness` can become measured instead of UNKNOWN **without ever
fabricating a label**. Full detail:
[`v10-benchmark-review.md`](./v10-benchmark-review.md).

| Phase | Item | Status |
|---|---|---|
| 0 | Audit of the V9 state (confirmed the answer benchmark identity, reusable review infra, what must change) | **done** |
| 1 | Ground-truth domain layer: annotations, question states, corpus-validated provenance, completeness (UNKNOWN ≠ 0), approval policy + gate, immutable frozen versions with tamper detection | **done** — 48 tests |
| 2 | Persistence: 3 append-only SQLite tables (`benchmark_ground_truth`, `benchmark_question_reviews`, `answer_benchmark_versions`), no update/delete paths | **done** |
| 3 | `ReferenceAnswerEvaluator` (`reference` / `reference-labels` v10.1): correctness from reviewed key-point coverage; refused 400 unless approved/frozen; V8 evaluator untouched | **done** — 12 tests |
| 4 | HTTP API: review packet, question detail with REAL evidence, append-only authoring/reviews, freeze with gate reasons, version list/verify, completeness; run accepts `benchmark_version_id` | **done** — 24 tests |
| 5 | Benchmark Review UI: authoring / evidence / review panels, state dots, gate + policy, freeze + verify | **done** — `tsc` + build clean |
| 6 | Live verification: full suite, write-path smoke over real HTTP, browser round-trip, Qdrant + frozen-artifact integrity | **done** — 1062 tests; smoke 24/24; Qdrant 14→14 / 8822→8822 / 812→812; `git diff benchmarks/` empty |
| 7 | Author + review the 28 official questions, freeze, run the reference evaluator | **NOT DONE — human work (0/28 reviewed)** |
| 8 | Phase-13 optimization experiment on measured correctness | **still blocked** until phase 7 completes |

V9's recommended order is now partly consumed: item 1's *workflow and UI* are
built (this phase); the human authoring itself remains. Next in V9's order:
item 2 (Phase-10 failure-analysis UI), item 3 (V9 API routes), item 4
(off-domain gate measurement).

## V11 — Ground-truth assist: drafting + pre-freeze audit (2026-10-10, delivered)

V11 does not run the evaluation loop; it removes the reason the loop is stuck.
The only missing input is 28 human-authored reference answers, so this phase
makes that authoring cheap and auditable while keeping every integrity rule
intact. Full detail: [`v11-ground-truth-assist.md`](./v11-ground-truth-assist.md).

| Phase | Item | Status |
|---|---|---|
| 0 | Audit of the V10 state (benchmark identity, freeze gate, what was actually blocking) | **done** |
| 1 | `drafting.py` — extractive, corpus-grounded DRAFT engine: IDF-weighted sentence selection, per-key-point chunk + sentence provenance, `draft_ready` / `partial` / `insufficient_evidence`, divergence vs the frozen benchmark's pinned sentence | **done** |
| 2 | `quality_audit.py` — deterministic, read-only, 8 pre-freeze checks; findings with severity + action; `checks_not_run` reported honestly; explicit limitations block; never repairs | **done** |
| 3 | Persistence: append-only `benchmark_draft_suggestions` (no update/delete path) | **done** |
| 4 | HTTP API: batch + per-question draft suggestions, draft history, quality audit | **done** — 54 tests |
| 5 | Review UI: draft-assist panel with history and "use as starting point", per-question audit findings, benchmark audit panel with full-findings expander | **done** — `tsc` + build clean |
| 6 | Live verification on the real benchmark: 28 drafts, 69 findings, 0 errors, UNKNOWN preserved, freeze still 409, benchmark untouched | **done** |
| 7 | Author + review the 28 official questions, freeze, run the first official reference baseline | **NOT DONE — human work (0/28 reviewed)** |
| 8 | Failure-mode analysis of the real baseline, then the "why did this answer fail?" UI | **blocked** until phase 7 completes |
| 9 | Controlled optimization experiments driven by the observed failure modes | **blocked** — no measured baseline exists to justify a change |

### Measured live state after V11

* Drafts: **23 `draft_ready`, 3 `partial` (011/016/019), 2
  `insufficient_evidence` (024/028)**; 27 key points; 7 drafts agree with the
  sentence the frozen retrieval benchmark pinned, 21 differ.
* Audit: **0 error / 34 warning / 35 info**, 8 checks ran, 2 could not (no
  annotations exist). Verdict: *STRUCTURALLY SOUND BUT INCOMPLETE*.
* Review-first shortlist: **auto-eng-028 (5%), 024 (11%), 011 (34%), 019 (39%),
  016 (42%)** — pinned evidence may not answer these questions.
* One real provenance finding: 29/29 required-evidence slots carry a source title
  on the benchmark label that the live corpus chunk does not record.
* Unchanged: 28 pending / 0 authored / 0 approved, freeze **409**, Qdrant
  **14 / 8822 / 812**, `git status benchmarks/` empty.

### V11 recommended order

1. **Author and review the 28 questions using the drafts and the audit** (the
   review-first shortlist above), then freeze and measure the official baseline.
   Nothing else in this roadmap can produce a trustworthy number until this is
   done, and V11 has made it the cheapest it can be without inventing labels.
2. **Phase-10 failure-analysis UI** (V9 item 2) — worth building only once real
   official runs exist to explain; its inputs (retrieval diagnostics, failure
   taxonomy, claim-level grounding) are already stored per run.
3. **V9 API routes** — diagnostics, classifications, recommendations and
   experiment records over HTTP.
4. **Off-domain gate measurement** — author ≥ 10 off-domain questions.
5. **Reranker-stage diagnostics** — pre/post-rerank rank comparison.
6. **Controlled optimization** — only against failure modes measured in the
   official baseline, with scratch-KB isolation and no destructive Qdrant work.

---

## V12 — Retrieval diagnostics (2026-10-10, delivered)

Per-question, evidence-backed diagnosis of why retrieval failed. Delivered:

- evidence availability against the live corpus AND the live index;
- the retrieval failure taxonomy (with `LOW_PRECISION` reserved and never
  emitted, because precision@k is a lower bound under necessity-only labels);
- a candidate probe deeper than the evaluated `k`, so "below your cut-off" and
  "never retrieved" stay distinguishable;
- matched-question strategy comparison reusing the V9 protocol gate and pairing;
- append-only diagnostic runs with an integrity snapshot, plus API and UI.

Measured at the time: dense recall@10 = 1.0, MRR@10 ≈ 0.743, nDCG@10 ≈ 0.806,
11 `RANKING_FAILURE` / 17 `RETRIEVAL_SUCCESS`; at k=3, 5 `LOW_RECALL`, 6
`RANKING_FAILURE`, 17 `RETRIEVAL_SUCCESS`. `required_recall_in_probe` = 1.0 in
both runs, i.e. the expected evidence was retrievable at depth 20 even when the
evaluated window was 3.

The two recommendations that came out of it were "increase top-k for one
question" and "investigate reranking for ranking failures". Both were tested in
V13.

---

## V13 — Controlled retrieval experiments (2026-10-11, delivered)

The measured-failure → hypothesis → experiment → verdict loop. It adds **no new
retrieval capability**; it is an experiment layer over the V12 engine and the V9
comparison utilities. Full detail:
[`docs/v13-controlled-retrieval-optimization.md`](v13-controlled-retrieval-optimization.md).

Delivered:

- a declaration contract (hypothesis, targeted failure code, arms, primary
  metric, guardrails, rejection conditions) stored BEFORE measurement;
- `RetrievalDiagnosticRunner.run_configs()` — several configurations of the same
  strategy measured by the same code path as a V12 diagnostic;
- derived outcomes (`IMPROVED` / `NO_MEANINGFUL_IMPROVEMENT` / `REGRESSION` /
  `INCONCLUSIVE` / `BLOCKED`) with the rule each verdict fired on recorded;
- guardrails held to a declared per-metric bar rather than raw inequality;
- immutable JSON experiment artifacts with before/after integrity snapshots;
- four shipped, versioned experiment definitions and a read-only UI panel.

Measured results (all exploratory, 28 questions, one corpus):

| Experiment | Outcome | Headline |
|---|---|---|
| B — depth, strict guardrails | `REGRESSION` | `precision@k` −0.1821 dominates a recall gain of +0.1786 |
| B — depth, corrected guardrails | `IMPROVED` | recall@k **0.8214 → 1.0000**, 5 questions improved, 0 regressed |
| A — cross-encoder reranking | `REGRESSION` (hypothesis **disproven**) | MRR +0.0083 (bar 0.02, p = 1.0), recall **0.9643**, latency **+1.5 s** |
| C — dense vs BM25 vs hybrid | `IMPROVED` (hybrid), `REGRESSION` (BM25) | hybrid MRR **0.7429 → 0.8173** with recall unchanged |

The exact sign test on each experiment's PRIMARY metric gives p = 0.0625 (B, depth),
1.0 (A, reranking), 0.549 (C, hybrid) and 0.424 (C, BM25). **None clears 0.05**, so
nothing here is a claim of statistical significance.

### V13 recommended order

1. **Build the held-out validation set** — independently authored questions for
   the same corpus, frozen the same way, then re-run B and C against it
   unchanged. Nothing accepted in V13 is a finding until it survives questions
   that were not used to select it.
2. **Re-test the reranking mechanism narrowly** — rerank only the candidates that
   will be returned (`reranker_top_n == top_k`), which cannot push required
   evidence out of the window, and see whether the MRR gain survives.
3. **Author and review the 28 answer-benchmark questions** (the V11 item still
   open) — answer-correctness optimization remains blocked without it.
4. **Reranker-stage diagnostics** — pre/post-rerank rank comparison per question,
   which Experiment A showed is where the coverage loss happens.
5. **Diversity/MMR experiment** — `apply_diversity` exists but has never been
   measured on the benchmark.
6. **Promote an accepted change** only after (1) — as an explicit, recorded user
   decision, never automatically.

### V14 status — held-out validation (provisional tier)

Items 1 and 2 of the order above are now done, and one promotion decision is open.

1. **DONE — held-out validation set.** 26 questions, frozen at
   **PROVISIONAL → EXPLORATORY** (25 of 26 carry a human decision; one could not
   be reviewed before the one-way freeze). B and C were re-run unchanged against
   it, with the development baseline re-measured in the same run:
   [`docs/v14-held-out-validation.md`](./v14-held-out-validation.md).
   * depth **replicates in direction, not in size** (+0.0385 vs +0.1786; one
     improved question out of 26; saturates at k=5);
   * the **reranker rejection replicates** (perfect ranking, +1223 ms);
   * the **BM25 verdict inverts** — perfect MRR 1.0 at 43 ms on held-out, while
     the development set rejected it. One question out of 26 carries the entire
     depth effect and two carry BM25's perfect score;
   * the V13-accepted **hybrid now fails its own declared latency guardrail**
     (+54.6 ms > 50 ms). Nothing was promoted.
2. **DONE — windowed reranking re-test.** Recall held exactly (it is invariant
   by construction when the reranked slice equals the window) and ranking reached
   1.0, but the latency guardrail rejects it on both instruments.
3. **DONE — mechanism probe (V15).** The V12 diagnostics were run once against the
   frozen held-out instrument with the unchanged `dense_k10` configuration. Both
   dense misses (`auto-eng-hout-007`, `auto-eng-hout-018`) have their required
   chunk **in the corpus, in the index and in the depth-20 candidate pool at rank
   12**, i.e. they are **ranking/depth** failures, not representation failures;
   `required_recall_in_probe = 1.0` for all 26 questions and every chunk that
   outranks a required one comes from that chunk's own document. Full report:
   [`docs/v15-retrieval-mechanism-probe.md`](./v15-retrieval-mechanism-probe.md).
4. **Open decision, not automatic — the default retriever for this corpus.** A
   43 ms lexical retriever answered every held-out question at rank 1. Whether it
   becomes the default is an explicit promotion decision, to be taken on a
   validated instrument and after (3), never automatically.
5. Still open from V13: the 28 answer-benchmark questions, reranker-stage
   diagnostics, and the MMR/diversity experiment.

---

## V15 — Retrieval mechanism probe (2026-10-13, delivered)

V14's open item 3 is closed. One read-only V12 diagnostic run (`rdx_93a4f54720a9`,
`dense`, k=10 evaluated over a depth-20 candidate probe) on the frozen held-out
instrument, with no weight, threshold, question or label changed:

| Question | Classification | Required chunk | Where it is |
|---|---|---|---|
| all 26 | availability `all_indexed`; `required_recall_in_probe` **1.0** | — | corpus **and** Qdrant index |
| 17 | `RETRIEVAL_SUCCESS` | rank 1 in window | — |
| 7 | `RANKING_FAILURE` | ranks 2–5 in window | every outranking chunk (**11 of 11**) is from the required chunk's own document |
| **007**, **018** | `LOW_RECALL` | **rank 12** of the depth-20 pool (scores 0.3897 and 0.5346 vs boundaries 0.3966 / 0.5460) | two places beyond the cut-off |

**Conclusion:** the two dense misses are window/ranking failures, not
representation failures. V14 §7's declared hypothesis is confirmed; V14 §5's
sentence claiming the misses are "not merely outside the window" is **corrected**
(they are, at rank 12).

### V15 recommended order

1. **Window experiment `dense_k10` vs `dense_k20`** on the frozen held-out
   instrument — hypothesis, controls, metrics, latency budget, acceptance and
   falsifier are declared in the V15 report §9. k=20 is the pre-existing
   probe-depth constant, not the observed rank 12, precisely to avoid tuning
   against the result. Expected: recall 0.9231 → 1.0000; still exploratory.
   **Decided 2026-10-13: declared only, not run** — the instrument has no
   post-probe measurement attached to it.
2. **Reconcile `required_recall_at_depth`** — in a V12 run it is computed over the
   windowed list and duplicates `recall_at_k`, while its documentation calls it
   depth-independent. Decide explicitly (rename, or compute over the probe list)
   before any further comparison uses it.
3. **Close the human review gate** — `auto-eng-hout-007` has no decision and the
   sealed v1 instrument refuses new ones, so this needs a **new instrument
   version** (v2 candidate set → review → freeze). Until then every held-out
   result stays PROVISIONAL / EXPLORATORY. Requires a human decision; not
   automatic. **Decided 2026-10-13: the gate stays open** — `auto-eng-hout-007`
   remains pending by explicit choice, no replacement question and no v2 seal.
4. **Author and review the 28 answer-benchmark questions** — unchanged V11/V13
   blocker: 0 of 28 reviewed, 0 reference answers, 0 frozen versions.
5. **The default retriever decision for this corpus** — still open, still not
   automatic. On 26 held-out questions BM25 scored MRR 1.0 in 43 ms, and V15 now
   shows the dense misses live in the near-tie band a lexical match resolves; one
   small instrument is not a basis for changing a default.
6. Still open from V13: reranker-stage diagnostics and the MMR/diversity
   experiment.

---

## V16 — Unlocking trustworthy answer-quality evaluation (2026-10-13, delivered)

V15's open item 4 ("author and review the 28 questions") moved from *blocked by the
workflow* to *blocked by a human decision*. The review path was audited, repaired
and hardened; **no reference answer, review or approval was created**, so the
benchmark is still `draft` (28 pending, 0 approved, 0 frozen). Full report:
[`docs/v16-answer-review-unlock.md`](./v16-answer-review-unlock.md).

### Verified state (live API + SQLite + artifact, 2026-10-13)

| Fact | Value |
|---|---|
| `answer-quality-automobile-v1` | fingerprint `f72c30eb5c0c363a`, lifecycle `draft`, `human_review: pending`, `answer_labels: NONE` |
| Question states | **28 pending**, 0 in review / approved / rejected / ambiguous / insufficient evidence |
| Ground truth | 0 authored reference answers, 0 key points, 0 `benchmark_ground_truth` rows, 0 `benchmark_question_reviews` rows, 0 frozen versions |
| Draft suggestions | **29** rows across 28 questions (`auto-eng-001` twice), append-only |
| Measurable metrics | coverage + retrieval metrics only; correctness, key-point recall, citation support and abstention stay **UNKNOWN** (never 0) |

### Defects fixed

1. **An approval could outlive the answer it judged.** A review that names its
   annotation now invalidates when the effective annotation changes (state drops to
   `in_review`, with the reason). Approvals that name no annotation cannot be
   invalidated automatically — that residual hole is now **disclosed in the packet
   warnings** rather than silent.
2. **Placeholder / degenerate text could be frozen as ground truth.** `TODO`,
   `TBD`, `N/A`, `...`, a one-word answer and duplicate key points (which would
   double-count in `key_point_recall`) are stored but flagged, never approvable,
   and blocking at freeze time.
3. **The reviewer could not see the intended scope.** The packet and UI now carry
   the artifact's provenance text, citation requirements, expected grounding
   state, abstention / units requirement and numeric tolerance.
4. **No queue or review ergonomics.** Added: "awaiting a decision only" filter,
   per-state filter with counts, position counter, keyboard navigation and a
   documented guarantee that no shortcut records a decision.

### Draft assistance, audited against the live corpus (read-only)

`backend/scripts/audit_answer_drafts.py` re-checks every stored draft against the
current corpus. **29 / 29 draft spans are verbatim** (0 problems; 6 byte-exact,
23 only under the drafting module's documented normalisation). Coverage is
measured for all 29: median 0.640, min 0.048; **13 drafts are below 0.60**.
Two questions (`auto-eng-024`, `auto-eng-028`) have **no quotable answer** in
their pinned evidence and 3 more (`011`, `016`, `019`) are `partial`. **20 of 28
drafts select a different sentence than the artifact's own unreviewed quote** —
two machine products disagreeing; neither is ground truth. The audit reports these
gaps and does not fill them.

### V16 recommended order

1. **Review the 28 questions** in the review surface (§4 of the V16 report), then
   freeze v1. This is now the only blocker to answer-correctness measurement.
2. **Re-derive answer metrics** on the frozen version — correctness, key-point
   recall, citation support, unsupported-claim rate, abstention quality — each with
   its own denominator, UNKNOWN where data is missing.
3. **Measure whether draft assistance helps** — audit the 20 disagreeing drafts
   against the human verdicts, pre-registered, before claiming any value from the
   generator.
4. **Decide the `auto-eng-024` / `auto-eng-028` policy** — cite more evidence or
   classify as `insufficient_evidence`; never let them become scored questions by
   accident.
5. Unchanged: reranker-stage diagnostics, MMR/diversity, and the open
   default-retriever decision (not to be taken on one small instrument).

---

## V17 — MCP integration (2026-10-14, delivered)

RAGForge knowledge bases are now reachable from MCP-compatible AI clients. Full
guide: [`docs/mcp-integration.md`](./mcp-integration.md).

### What was built

* **Official SDK, stdio transport.** `mcp` 2.3.0 (`MCPServer`), launched by the
  client as a subprocess. Streamable HTTP was **not** shipped: RAGForge has no
  auth, so an unauthenticated port would expose every knowledge base on the
  machine. stdio inherits the local process boundary instead.
* **Six read-only tools** — `list_knowledge_bases`, `get_knowledge_base_status`,
  `search_knowledge_base`, `get_evidence`, `answer_from_knowledge_base`,
  `evaluate_retrieval` — plus four read-only resources (knowledge-base index,
  one KB, one document, one evidence chunk). No ingest, delete, review, config or
  benchmark-mutation tool exists.
* **Service reuse, not a parallel stack.** `app/mcp/bridge.py` is the only
  RAGForge-facing code and it calls the existing `RetrievalService`,
  `AnsweringService`, `Evaluator` and `Repository`. Scores, provenance, grounding
  states, lifecycle gates and recorded runs are the same ones the UI shows.
* **Separation.** The FastAPI app never imports `app.mcp`, the MCP process starts
  no web server, and `mcp` stays an optional dependency
  (`backend/requirements-mcp.txt`).

### Honesty and safety carried into the MCP surface

* Real scores with per-pool `score_breakdown`, the retriever's own `why`, and a
  trace with strategy, resolved params, config fingerprint, stages, timings and
  corpus version; an unknown strategy is an error listing the registry — **no
  silent fallback**.
* Grounding preserved: `status` and the five `GroundingState` values, claims with
  support verdicts, validated citations, and the abstention reason. An
  `abstain_if_unsupported` turn with insufficient evidence makes no model call.
* `evaluate_retrieval` refuses anything but a **FROZEN** benchmark version, has a
  `dry_run` that executes nothing, and lists answer-level metrics as unavailable
  with the human-review reason; metrics with no ground truth are UNKNOWN, not 0.
* Limits and boundaries: validated ranges, truncated-text disclosure, no
  filesystem access, no paths/secrets in any payload, coded non-internal errors,
  and corpus text marked untrusted with a tested prompt-injection path.

### Defect found and fixed

While testing the answer tool: `AnsweringService._failure_answer(kb_id=…)` did
not match its `kb=` parameter, so **every** generation failure raised `TypeError`
instead of returning the documented `generation_failed` answer. Fixed, with the
MCP test now covering that path.

### V17 recommended order

1. **Review and freeze the 28 answer-benchmark questions** (V16 §4) — the only
   blocker left for answer-correctness metrics, which this MCP surface already
   reports as unavailable.
2. **Pick an MCP client and use the server for real work** — the transport,
   tools and resources are done; the useful next evidence is a real session
   transcript, not more surface area.
3. Still open from V13/V15/V16: reranker-stage diagnostics, MMR/diversity, the
   default-retriever decision, and the held-out instrument's open review gate
   (`auto-eng-hout-007` stays pending by explicit decision).
4. **Only if a real need appears:** a secured HTTP transport (with auth and
   per-client scoping) or curated MCP prompts. Neither is required today and
   neither should be added speculatively.
