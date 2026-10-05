# RAGForge — Architecture (as implemented)

This document describes what actually exists in the code, not aspirations.

## High-level flow

```
Knowledge Base (SQLite)  ── source_mode: EXTERNAL | USER_PROVIDED | MIXED
  ├─ DomainSpec          (LLM provider, validated Pydantic output)
  ├─ Sources             (discovery providers → quality engine → decisions)
  │                       user-provided files carry integrity, not authority, signals
  ├─ Documents           (document library: upload/download → parse → clean → hash → status)
  ├─ Chunks              (chunking strategies; full provenance incl. page/slide/section)
  ├─ EvaluationQuestions → BenchmarkVersions → EvaluationRuns   ← OPTIONAL, never required
  └─ BuildRuns           (stage statuses for transparency)

Qdrant: one collection per KB (kb_<id>), points carry full provenance payloads.
Filesystem: backend/data/documents/<source_id>.<ext>      (downloaded external sources)
            backend/data/uploads/<kb_id>/<raw_id>.<ext>  (user-provided originals, local only)
Both keep the normalized text in a <file>.parsed.txt sidecar.
```

A knowledge base becomes READY once it is indexed. Evaluation is a separate,
optional instrument — `KBOverview.evaluation_required` is always `false`.

## Backend layout

```
backend/app/
  main.py                     FastAPI app, CORS, global error handler
  config.py                   Pydantic Settings from env / backend/.env
  schemas/models.py           ALL domain models (single source of truth)
  api/                        routers: KBs, domain, sources, documents, build,
                              retrieval/eval, benchmark, experiments, system
  llm/provider.py             LLMProvider ABC + OpenAI-compatible + Mock provider
  repositories/sqlite_repo.py SQLite persistence (JSON-serialized models)
  services/
    domain_analyzer/          prompt → structured DomainSpec
    source_discovery/         SourceDiscoveryProvider ABC; user-url, arxiv
    source_quality/           explainable heuristic scorer (signals→score→decision)
                              + user_scorer.py: integrity model for user files
    ingestion/
      parsers.py              DocumentParser ABC + ParserRegistry
                              (Pdf, Pptx, LegacyPpt, Docx, Html, Markdown, Text)
      upload.py               name safety, size, magic bytes, OOXML container checks
      ingestion.py            download (external) / parse (local) orchestration
    indexing/
      document_indexer.py     chunk → embed → index for a SET of documents.
                              Shared by the full build and per-document incremental
                              indexing so stale-vector handling cannot drift.
    chunking/                 Chunker ABC; section-aware + fixed-size; page/slide markers
    embeddings/               EmbeddingProvider ABC; ST, OpenAI, hashing fallback
    vector_store/             VectorStore ABC; Qdrant implementation
    retrieval/                Retriever ABC + registry; DenseRetriever (dense-only)
    evaluation/               pure metric functions + Evaluator orchestrator
  utils/                      ids, hashing/cleaning, URL validation (SSRF), logging
scripts/
  verify_benchmark_integrity.py   read-only frozen-benchmark gate + v3 plan
```

## Key decisions

1. **SQLite + JSON columns** instead of PostgreSQL/ORM: simplest robust persistence for a
   single-user local MVP. Repository interface isolates services from storage; a swap is
   localized.
2. **Every external integration behind an ABC** so experiments can swap implementations
   (chunkers, embedding providers, vector stores) without touching services or API.
3. **Quality scores are heuristics, stored with their derivation**: each assessment persists
   the signals, weights, reasons, warnings, and assessor version. The UI shows them as an
   automated assessment and lets the user override ACCEPT/REVIEW/REJECT.
4. **Mock/dev outputs are labelled at the data level** (`is_mock`, `generated_by`,
   `is_fallback`), not just in the UI, so fake data cannot masquerade as real downstream.
5. **Evaluation metrics are pure functions** over ranked IDs and known-relevant sets.
   Missing ground truth → `null` (unknown), never 0 or a fabricated number.
6. **SSRF-safe ingestion**: user URLs must resolve to public addresses; scheme/host checks;
   size cap; timeouts. Downloaded files are parsed as data, never executed.
7. **Point IDs in Qdrant** are deterministic 63-bit hashes of chunk IDs (Qdrant requires
   unsigned ints/UUIDs); the original ID travels in the payload.
8. **Benchmark ≠ knowledge base (V4).** A KB is a product artifact; ground truth is an
   optional evaluation instrument. `source_mode` describes where knowledge comes from and
   never implies a benchmark exists. Nothing in the build path reads evaluation questions.
   Note the consequence: a *new* domain has **no** benchmark and therefore **no measured
   retrieval quality** until a human authors, reviews and freezes one. Benchmarks are
   per-domain hand-made research artifacts (only the Automobile Engineering one exists here);
   RAGForge never auto-generates ground truth, and metrics stay `null` until it does.
9. **User files are scored on integrity, not publication authority (V4).** A lecture PDF has
   no website, HTTP status or publication date. `UserProvidedIntegrityScorer` weights only
   file validity, content extraction, uploader-asserted relevance, duplication and structure —
   no authority/recency/accessibility/evidence weight exists in it.
10. **One shared indexer (V4).** `index_documents()` is the only chunk→embed→index path.
    Vectors for the documents being (re)indexed are deleted *before* upsert (re-chunking
    makes new IDs, so upserting alone orphans points forever); chunk rows are written *after*
    a successful upsert. A backend that cannot delete vectors is refused rather than corrupted.
11. **Structural numbers are read, never inferred.** `[PAGE n]` / `[SLIDE n]` markers come from
    the parsers; a deck with no title placeholder yields `slide_title = None`, and a scanned PDF
    fails with an explicit "OCR is not implemented" error.
12. **Deletion counts are measured, and "unknown" is a valid answer (V4).** The installed
    `qdrant-client` returns no deleted-count for `delete()`, so RAGForge measures the collection's
    `points_count` delta itself instead of trusting a response shape that silently yields zero.
    If the collection cannot be read it returns `-1`, which the indexer and the API surface as
    `"unknown"` / `vectors_removal_confirmed: false`. A confirmed count is never replaced by an
    unconfirmed one, and an unconfirmed count is never rendered as `0`.

## Failure handling

- Ingestion errors are per-document (one bad URL doesn't abort the batch) and surfaced in
  the build run status.
- Upload validation rejects per file with an actionable message; a corrupt container, a
  mislabelled extension or an empty file never becomes a half-ingested document.
- A document whose parser fails is stored with `status=FAILED` and its real error, so the
  library shows the failure instead of hiding it.
- Indexing refuses to proceed when the vector store cannot delete stale vectors for the
  documents being re-indexed.
- Qdrant unavailability returns HTTP 503 with an actionable message ("start Qdrant, see README").
- LLM misconfiguration returns actionable errors; with `ALLOW_MOCK_LLM=true` (default) an
  unconfigured LLM degrades to the labelled mock instead of failing.
- Unhandled exceptions are logged server-side; the frontend receives a generic 500 (no stack traces).

## API surface (implemented)

```
POST/GET/DELETE /api/knowledge-bases[/{id}]
GET  /api/knowledge-bases/{id}/overview              # one honest payload for the overview page
GET  /api/knowledge-bases/{id}/build-status
POST /api/knowledge-bases/{id}/analyze-domain
GET  /api/knowledge-bases/{id}/domain-spec
POST /api/knowledge-bases/{id}/discover-sources      {provider, query, limit}
POST /api/knowledge-bases/{id}/sources/user-urls     {urls, titles}   # user-provided URLs
GET  /api/knowledge-bases/{id}/sources
POST /api/knowledge-bases/{id}/sources/{sid}/decision {decision}
POST /api/knowledge-bases/{id}/ingest                {source_ids?}   # external sources only
POST /api/knowledge-bases/{id}/index                 {chunker, target_size, overlap}
GET  /api/knowledge-bases/{id}/documents | chunks

# --- V4 document library ---
POST   /api/knowledge-bases/{id}/documents/upload    # multipart files[] + index?
GET    /api/knowledge-bases/{id}/document-library
GET    /api/knowledge-bases/{id}/documents/{did}
GET    /api/knowledge-bases/{id}/documents/{did}/text
GET    /api/knowledge-bases/{id}/documents/{did}/chunks
POST   /api/knowledge-bases/{id}/documents/{did}/rebuild  {reindex?}
POST   /api/knowledge-bases/{id}/documents/{did}/index    # incremental: this document only
POST   /api/knowledge-bases/{id}/documents/{did}/replace  # new document_version + history
DELETE /api/knowledge-bases/{id}/documents/{did}          # removes chunks AND vectors

POST /api/knowledge-bases/{id}/retrieve              {query, top_k, filters}

# --- V6 retrieval (see retrieval-architecture.md) ---
GET  /api/knowledge-bases/{id}/retrieval-strategies
GET/PUT /api/knowledge-bases/{id}/retrieval-config
GET  /api/knowledge-bases/{id}/retrieval-runs[/{run_id}]
GET  /api/knowledge-bases/{id}/bm25-index

# --- V7 grounded answers (see answering-architecture.md) ---
POST /api/knowledge-bases/{id}/answer                 {question, answer_mode?, retrieval_strategy?, retrieval_params?}
GET  /api/knowledge-bases/{id}/answers/{answer_id}
GET  /api/knowledge-bases/{id}/answer-traces/{trace_id}

# --- V7.2 grounded chat (see chat-architecture.md) ---
POST   /api/knowledge-bases/{id}/chat                 {message, retrieval_strategy?, retrieval_params?, conversation_id?, answer_mode?}
GET/POST /api/knowledge-bases/{id}/conversations
GET/DELETE /api/knowledge-bases/{id}/conversations/{cid}
GET    /api/knowledge-bases/{id}/answer-runs[/{run_id}]
POST/GET/DELETE /api/knowledge-bases/{id}/evaluation-questions[/{qid}]
POST /api/knowledge-bases/{id}/evaluate              {top_k, question_ids?, benchmark_version?}
GET  /api/knowledge-bases/{id}/evaluation-runs
GET  /api/system/health | status
```

## Extension points for experiments (V3)

- **Benchmark lifecycle** — questions move DRAFT → REVIEW → APPROVED → FROZEN with reviewer
  attribution (`routes_benchmark.py`); APPROVED edits spawn new DRAFT revisions (`supersedes`);
  FROZEN questions/versions are immutable. Benchmark versions are snapshot objects
  (`BenchmarkVersion`) whose FROZEN instances are the only valid basis for official evaluation runs
  (enforced in `Evaluator.run_evaluation` via `config.benchmark_version`).
- **Chunking strategy registry** — `CHUNKING_REGISTRY` / `get_chunker()`; chunks stamp
  `chunking_strategy` + `chunking_config_version`; KB records the strategy used.
- **Vector backend factory** — `create_vector_store(settings, kb.vector_backend)` with
  `register_vector_backend()` for experimental backends (TurboVec, Phase E).
- **Retriever registry (V4 Phase 12)** — `Retriever` ABC + `register_retriever()` /
  `get_retriever()` in `services/retrieval/retriever.py`. Dense is the baseline and the only
  implemented strategy; BM25 / hybrid / reranked register here and must return the same
  `RetrievalResponse`, so the API, evaluator and UI never change shape.
- **Parser registry (V4)** — `PARSER_REGISTRY` in `services/ingestion/parsers.py`. CSV/XLSX/OCR
  register exactly like `PdfParser`; no call site changes.
- **Read-only experiment artifacts** — `routes_experiments.py` serves frozen results JSON verbatim;
  the UI renders them without hardcoding metrics.
- **Frozen-benchmark gate** — `scripts/verify_benchmark_integrity.py` is read-only and refuses to
  bless a benchmark that is not FROZEN, fully reviewed, and snapshotted.

## Extension points for experiments

- `Chunker` — add `SemanticChunker`; compare against `section-aware` / `fixed-size`.
- `SourceDiscoveryProvider` — add more connectors (pubmed, Semantic Scholar, ...).
- `SourceQualityScorer` — new signals/weights or an LLM-based assessor.
- `EmbeddingProvider` — any model; dimensionality is checked against the collection.
- `VectorStore` — second backend for comparison runs.
- Retrieval — add BM25/hybrid/rerank as new retriever classes; the response already carries
  `retrieval_backend` so runs are self-describing.

---

## Corpus engineering layer (V5)

The corpus services live in `app/services/corpus/` and are deliberately separate
from the original domain pipeline:

| Module | Responsibility |
|---|---|
| `batches.py` | persistent, resumable bulk ingestion; per-item failure isolation |
| `manifest.py` | "what exactly is inside this KB?" — per document + corpus summary |
| `integrity.py` | **read-only** audit: missing / orphan / stale vectors, mismatches, duplicates, provenance gaps |
| `repair.py` | explicit, confirmed, separately-invoked repairs |
| `fingerprint.py` | deterministic corpus fingerprint, versions and document diffs |
| `perf.py` | measured pipeline instrumentation + the scale benchmark harness |

Key decisions:

13. **A batch is persisted before any processing begins.** Every file becomes an
    `ingestion_item` row with a `UNIQUE(batch_id, item_key)`. Resume is therefore
    idempotent at the *schema* level, not merely in application code, and a page
    refresh or a process restart cannot lose batch state. No Celery, no Redis.
14. **Integrity checking and repair are separate operations.** `CorpusIntegrityService`
    has no write path at all; repair lives in `CorpusRepairService` and requires an
    explicit call, plus `confirm_action` for destructive actions. Provenance of a
    finding is never conflated with fixing it.
15. **Unmeasured counts are `-1`, never `0`.** `-1` surfaces as `"unknown"` /
    `vectors_removal_confirmed: false`. An unreachable vector store reports unknown
    totals and lists the skipped checks in `unknown_counts`; it never reports zero
    findings for checks that did not run.
16. **The corpus fingerprint excludes anything non-deterministic.** Document IDs,
    file names, timestamps and the `kb_id` are all excluded, so identical corpora
    produce identical fingerprints and a version history only grows on real change.
17. **Ground truth is an explicit, per-KB state.** `GroundTruthStatus.NOT_AVAILABLE`
    is the honest default for every new domain. Corpus coverage and retrieval quality
    are reported by different surfaces and must never be substituted for one another.

## Retrieval intelligence layer (V6, in progress)

Full detail: [retrieval-architecture.md](retrieval-architecture.md).

| module | responsibility |
|---|---|
| `retrieval/retriever.py` | `Retriever` ABC, provenance helpers, the **registry** (`RetrieverSpec`, lazy built-ins), `DenseRetriever` |
| `retrieval/lexical.py` | deterministic tokenizer + a real BM25 scorer (pure functions) |
| `retrieval/bm25.py` | `Bm25Retriever` + `LexicalIndexStore` (persisted index, exact staleness check) |
| `retrieval/fusion.py` | `Candidate`, min-max/rank normalization, weighted fusion, RRF |
| `retrieval/diversity.py` | document cap + MMR |
| `retrieval/rerank.py` | `Reranker` ABC, `NoReranker`, `CrossEncoderReranker` |
| `retrieval/hybrid.py` | `HybridRetriever`, `HybridRerankedRetriever` |
| `retrieval/trace.py` | stage statuses, measured timings, notes |
| `retrieval/service.py` | configuration resolution, strategy construction, run recording |

Key decisions:

18. **Fusion normalizes before combining.** Dense cosine and BM25 are not
    comparable; every fusion path normalizes first and records the method. Weights
    that do not sum to 1 keep their RATIO, and the applied weights are recorded with
    the run, so a configuration is never ambiguous after the fact.
19. **A strategy that needs no embeddings loads no embedding model.** `bm25` runs
    without SentenceTransformers; the response states that no embedding model was
    involved instead of showing a misleading model name.
20. **Derived indexes are derived.** The BM25 index is built from `chunks` and
    persisted as statistics only (no copied text, no provenance). Staleness is
    decided by a corpus revision counter bumped on every chunk write — exact and
    O(1) — and a mismatch forces a rebuild that is *reported* on the response.
21. **Reranking is never silently faked.** A requested-but-unavailable reranker
    yields the unreranked order plus `status = unavailable_fallback`, the reason and
    a response note. Cross-encoder logits are reported separately from the
    comparable fused score and drive the ordering only.
22. **Diversification demotes, it does not delete.** A document cap moves extra
    chunks to the end instead of dropping them, and MMR reports when candidate
    vectors were unavailable for it.
23. **Configuration and observability are first-class.** `RetrievalParams` is
    persisted per KB (request override > stored config > defaults) and every run
    writes a `RetrievalRun` (params, applied weights, corpus version, stage counts,
    measured timings). Every retrieval response carries `retrieval_run_id`.

### Why retrieval configurations and runs are persisted separately

A configuration is a *decision* (what the KB should do by default); a run is an
*observation* (what actually happened, with which exact parameters). Mixing them
would let a later configuration edit rewrite the meaning of an earlier result. They
therefore live in separate tables (`retrieval_configs`, `retrieval_runs`) and a run
never changes after it is written.

### Why the fingerprint excludes `kb_id`

A fingerprint answers *"which exact corpus produced these vectors?"*. If it included
the knowledge-base ID, two KBs holding byte-identical corpora under identical
chunking/embedding configuration would disagree, and the fingerprint could never be
compared across environments or machines. The corpus is the thing being identified.

## Grounded answer layer (V7)

Full detail: [answering-architecture.md](answering-architecture.md).

| module | responsibility |
|---|---|
| `schemas/answer.py` | all V7 schemas: `QueryPlan`, `Evidence`, `EvidenceAssessment`, `Answer`/`Claim`/`Citation`, `AnswerPolicy`, `AnswerTrace` |
| `answering/query_processor.py` | deterministic, heuristic-labelled query processing (Phase 9) |
| `answering/evidence.py` | provenance-preserving assembly + dedup with recorded reasons (Phase 10); `EvidenceSelector` ABC (V7.2) |
| `answering/gate.py` | multi-signal sufficiency gate → `ANSWER / PARTIAL_ANSWER / ABSTAIN / ASK_CLARIFICATION` (Phase 11); `GroundingGate` + five states (V7.2) |
| `answering/generator.py` | `AnswerGenerator` ABC, `LLMAnswerGenerator` (wraps any `LLMProvider`), deterministic extractive mock, explicit `FallbackAnswerGenerator` (Phase 12) |
| `answering/prompting.py` | strict system prompt, `<EVIDENCE>` wrapping, prompt-injection marker scan |
| `answering/validation.py` | `ClaimExtractor → CitationValidator → GroundingValidator → AnswerPolicy` (Phase 13) |
| `answering/trace.py` | `AnswerTraceRecorder` — canonical stage list, raw generator output, no secrets |
| `answering/chat.py` | `GroundedChatService` — conversation memory + `AnswerRun` observability, delegates to `AnsweringService` (V7.2) |
| `answering/service.py` | orchestration: gate BEFORE generation, registry-based retrieval, persistence |
| `api/routes_answer.py` | `POST /answer`, `GET /answers/{id}`, `GET /answer-traces/{id}` |
| `api/routes_chat.py` | `POST /chat`, conversations, answer runs (V7.2) |

Key decisions:

24. **The gate runs before the generator.** In `abstain_if_unsupported` mode an
    insufficient assessment makes no model call at all — abstention is a
    deterministic code path, not a model's opinion. Original questions are
    preserved byte-for-byte; everything else in the `QueryPlan` is derived.
25. **Evidence is copied, never re-derived.** Assembly copies the chunk's provenance
    dict verbatim (absent stays absent), keeps BOTH the original retrieval rank and
    the final evidence rank, and drops only true duplicates — identical text in a
    *different* document is never deduplicated, because independent sources
    agreeing is evidence the gate wants to see.
26. **No numeric grounding confidence, ever.** Confidence is categorical
    (`high/moderate/low/none`) with a human-readable basis. Signal measurements
    that could not be made are reported as *not performed* with the reason, never
    as 0 or a made-up percentage.
27. **Validation is deterministic and honest about its ceiling.** Citations are
    checked for existence, KB ownership, retrieval membership, provenance and
    lexical overlap; semantic entailment is NOT IMPLEMENTED and every validation
    detail says "Semantic support check: NOT PERFORMED". Failure actions
    (remove/downgrade/abstain/regenerate) are recorded per problem.
28. **Answer generation is provider-agnostic and mockable.** The generator wraps
    the existing `LLMProvider` abstraction; `LLM_PROVIDER=mock` exercises the real
    prompt path with no key/network (required by tests), and total provider failure
    falls back to a clearly-labelled extractive mock instead of a 500. Retrieved
    document text is untrusted data wrapped in `<EVIDENCE>`; injection markers are
    recorded as warnings (HEURISTIC, never claimed as perfect protection).

---

## Grounded chat layer (V7.2)

Full detail: [chat-architecture.md](chat-architecture.md) and
[grounding-and-citations.md](grounding-and-citations.md).

| module | responsibility |
|---|---|
| `answering/query_processor.py` | adds `QueryTrace` + `QueryNature` (conversational / non-knowledge / underspecified / multi-hop); pass-through when disabled (V7.2) |
| `answering/chat.py` | `GroundedChatService`: reference resolution, conversation persistence, `AnswerRun` record |
| `api/routes_chat.py` | `POST /chat`, conversation CRUD, answer-run history |
| `repositories/sqlite_repo.py` | tables `conversations`, `messages`, `answer_runs` (all cleaned in `delete_kb`) |
| `frontend/.../chat/page.tsx` | three-pane investigation UI: corpus / conversation / evidence |

Key decisions:

29. **Chat is a thin front end, not a second pipeline.** Every turn delegates to
    `AnsweringService.answer`, so chat and `/answer` cannot drift apart: an answer
    reached through chat is grounded identically. Chat adds only conversation
    memory and an `AnswerRun` record.
30. **Conversation history is never knowledge.** It is stored separately from
    documents and chunks, never chunked, embedded, indexed or retrieved, and
    `Message.used_as_knowledge` is permanently `False` — the field exists so an
    audit can assert this rather than trust it. History is used for ONE purpose:
    resolving references like "What about the previous case?" into a standalone
    question, and the borrowed source is recorded in `Message.resolved_from`.
31. **A conversation is scoped to exactly one knowledge base.** Using a
    conversation id against a different KB returns 404 rather than answering from
    the wrong corpus.
32. **Five explicit grounding states, not a collapsed score.** `ANSWERED`,
    `PARTIALLY_SUPPORTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`,
    `NO_RELEVANT_EVIDENCE` map onto the coarser internal `GateDecision` through an
    exhaustive, unit-tested table, so the two vocabularies cannot drift apart.
33. **Never fall back silently.** When the configured generator cannot be
    constructed, `FallbackAnswerGenerator` sets `unavailable=True` with the reason
    and the answer carries a warning naming the fallback. `LLM_PROVIDER=mock`
    (the prompt path genuinely ran against a deterministic stand-in) is reported
    separately from a fallback (the configured provider could not be built at all).
34. **Unmeasured is `null`, never `0`.** A skipped generation stage records no
    latency; reporting `0 ms` would imply a measurement that never happened. The
    same rule governs page numbers: absent stays absent and renders as
    "not recorded".

---

## Answer-quality evaluation layer (V8)

Everything above produces answers. This layer **scores** them.

```
benchmarks/answer-quality-automobile-v1.json   (evidence-based ground truth)
        │
        │  validate_against_corpus()  ── content-hash re-chunk detection
        ▼
answer_eval/service.py    ── for each question: retrieve → answer → evaluate
        │                     (IDF relevance weights injected from the lexical index)
        ▼
answer_eval/evaluator.py  DeterministicAnswerEvaluator (v8.3)
        │  ├── claim-level citation audit        → ClaimCitationVerdict
        │  │     five-state evaluated_state (SUPPORTED / PARTIALLY_SUPPORTED /
        │  │     UNSUPPORTED / CONTRADICTED / UNVERIFIABLE)
        │  ├── question↔answer relevance         → relevance.py (IDF-weighted;
        │  │     plain fallback labelled, abstentions exempt)
        │  ├── completeness from human labels    → key_point_recall /
        │  │     expected_information_coverage (labels required, else UNKNOWN)
        │  └── answer-level completeness checks   → problems[]
        │     HumanAnswerEvaluator  → correctness from append-only reviews
        │     LLMAnswerEvaluator    → model-based judge, labelled, degrades
        ▼
answer_eval/metrics.py    Measured(value | reason, sample_size)
        ▼
answer_eval/run.py        immutable AnswerEvaluationRun + aggregate
        │                    compare_with(): COMPARABLE / INCONCLUSIVE /
        │                    NOT_COMPARABLE over shared questions
        ▼
sqlite `answer_evaluation_runs`  →  GET /answer-evaluation/runs
        + `answer_reviews` (append-only) → human-evaluation derivation pass
```

### Key decisions

35. **`Measured` carries a value OR a reason, never both — and never `0` by
    default.** `Measured.unknown("no human-authored reference answer")` and
    `Measured.of(0.0)` are different facts. The UI renders UNKNOWN as the word
    UNKNOWN, visually distinct from any number.
36. **Answer ground truth is evidence-based, not answer-based.** The frozen
    retrieval benchmark has 28 `expected_chunk_ids` and zero expected answers.
    Rather than generate reference answers (an LLM grading an LLM), the answer
    benchmark inherits the evidence and leaves the answer fields empty.
    Consequently `correctness` and `key_point_recall` are UNKNOWN by
    construction.
37. **`final_score` may not accompany an unmeasured input.** A model validator on
    `AnswerQualityResult` rejects a score whose `final_score_computable` is
    False, so a partial measurement can never be dressed as a complete one.
38. **`required_evidence` is a NECESSITY label, not an exhaustive whitelist.**
    Enforced per **answer** (a correct answer must use the necessary evidence),
    never per **claim** — a claim about alternative fuels has no reason to cite
    the engine-definition chunk. Extra retrieved context lowers
    `citation_precision` (documented as a lower bound) and is reported as an
    observation, never a failure, because the benchmark never established that
    unlabelled chunks are irrelevant.
39. **Retrieval misses are scored separately from citation quality.** Folding
    "the evidence was never retrieved" into the answerer's score blames the
    wrong component. Both are reported; they are never summed into one verdict.
40. **A benchmark that does not match the live corpus is refused, not scored.**
    `validate_against_corpus()` compares content hashes; a re-chunked corpus
    raises `BenchmarkCorpusMismatch` (HTTP 409) instead of silently scoring
    against shifted chunks.
41. **Runs are immutable and carry their producers.** Generator, model, mock
    flag, prompt/answerer/evaluator versions, entailment provider and
    model-based flag, benchmark fingerprint, and the exact `question_ids`
    covered. `is_comparable_with()` refuses to compare runs over different
    question subsets, because attributing a question-mix difference to a
    strategy is a way to invent a result.
42. **The benchmark path guard allows the repository OR `DATA_DIR`,** and nothing
    else — the endpoint must not become an arbitrary-file reader. Both are
    legitimate benchmark locations (versioned artifact vs. per-installation).
43. **`persist=false` is honestly documented as not meaning read-only.** It
    suppresses only the evaluation-run row; every question still goes through
    the normal answering path and writes an `Answer` row. A flag named `persist`
    that silently leaves side effects is a trap, so the OpenAPI description and
    the runner script both state it.
44. **`passed` is a serialized computed field, not a bare property.** It is the
    input to `pass_rate`; if it is absent from the response, a client filtering
    on it classifies every question as failed while the metric says otherwise.
    Two API tests assert `pass_rate == mean(per_question[].passed)`.
