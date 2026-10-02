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

### Next candidates (not started)

1. **Benchmark authoring UI** — make the `DRAFT → IN_REVIEW → APPROVED → FROZEN`
   lifecycle usable for a non-Automobile domain, so Naval Architecture can acquire
   real ground truth. This is the only route to ever reporting retrieval quality for
   a new domain.
2. **Watch mode** — detect a changed source document and offer a targeted
   re-index, using the document diff machinery that now exists.
3. **Retrieval-side corpora** — BM25/hybrid, measured against a frozen benchmark so
   the comparison is real rather than anecdotal. Blocked on (1).
4. **Streaming ingest** — process a 200-file batch progressively instead of
   synchronously, for genuinely large corpora.
