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
