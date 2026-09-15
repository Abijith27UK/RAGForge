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

## Next (recommended order)

1. **Real-world validation run** — Automobile Engineering KB with a configured LLM, a
   real corpus (SAE/IEEE/edu PDFs, arXiv papers, Wikipedia), full pipeline, manual review
   of quality decisions and chunk quality.
2. **Ground-truth benchmark authoring** — UI to attach expected chunk/document IDs to
   evaluation questions (selecting from actual chunks), removing reliance on the keyword
   heuristic for rigorous numbers.
3. **Experiment framework** — persist per-run configuration (chunker, embedding model,
   source set, top_k); support A/B comparisons with metric deltas; results table UI.
4. **PDF page-level chunking** — thread page numbers through PDF parsing into chunk
   metadata (currently only text is page-joined).
5. **LLM-generated candidate evaluation questions** — labelled as generated, human-approved
   before use as ground truth.

## Later phases

- BM25 + hybrid retrieval, reranking (cross-encoder or LLM-based) behind the retrieval interface
- LLM-as-judge context relevance and groundedness metrics (kept strictly separate from retrieval metrics)
- Semantic/domain-aware chunking strategies
- Automated optimization loop: propose configuration changes from evaluation weaknesses
- DOCX ingestion; additional discovery providers (Semantic Scholar, PubMed)
- Qdrant docker-compose file and dev-container setup
- Experiment 1–5 comparisons from the research plan, documented with real results

## Explicitly out of scope (for now)

Autonomous agents, MCP, scheduling, auth/billing, multi-user, cloud deployment, chat UI,
voice, mobile, elaborate visualizations.
