# AGENTS.md — RAGForge Project Instructions

## Project
RAGForge is an ID4100 Creative Engineering project:
**Automated Domain-Specific RAG Knowledge Base Builder**

The goal is NOT to build another generic RAG chatbot. The product should help a user start with a domain and intended use case, then progressively construct a reusable, provenance-rich, evaluated RAG knowledge base.

Core research question:

> Can the construction of a domain-specific RAG knowledge base be automated from a high-level domain specification while maintaining source authority, knowledge coverage, retrieval quality, and provenance?

## Product principle
Do not implement a fake AI pipeline that claims to have discovered or evaluated data when it has not.

Every source, document, chunk, score, metric, and status shown by the UI must either:
1. come from a real executed pipeline step,
2. be explicitly marked as demo/mock data, or
3. be unavailable.

Prefer a smaller working vertical slice over a large collection of mocked screens.

## Initial target workflow
The first usable version should support:
1. Create a knowledge base with name, domain, intended purpose/use case, and target audience/depth.
2. Domain Analyzer produces a structured domain specification: subdomains, key concepts/entities, terminology, suggested source categories, and knowledge requirements.
3. Source Discovery produces normalized candidate sources.
4. Source Quality Engine assigns transparent scores/reasons and ACCEPT / REVIEW / REJECT decisions.
5. Ingestion downloads/parses supported documents.
6. Chunking creates inspectable chunks with metadata.
7. Embedding creates vectors.
8. Qdrant stores vectors and provenance metadata.
9. Retrieval API returns top-k chunks with source information.
10. Evaluation runs a small benchmark and reports actual retrieval metrics.
11. UI exposes build progress, sources, chunks, retrieval testing, and evaluation.

For the first implementation, support a limited number of source/document types and one primary vector database.

## Recommended architecture
Frontend:
- Next.js
- React
- TypeScript
- Tailwind CSS
- clean dashboard UI

Backend:
- Python
- FastAPI
- Pydantic
- modular services

Storage:
- Qdrant as the primary vector store
- local filesystem for raw/processed documents in the MVP
- SQLite/PostgreSQL only if genuinely needed for project metadata

Embeddings:
- provider abstraction
- local SentenceTransformers support where practical
- keep the interface open for API-based embeddings later

LLM:
- provider abstraction
- structured JSON/Pydantic output for domain analysis
- do not hard-code a single provider into core business logic

## Core data models
Design clean schemas for:
- KnowledgeBase
- DomainSpec
- Source
- Document
- Chunk
- EmbeddingConfig
- RetrievalConfig
- EvaluationQuestion
- EvaluationRun
- BuildRun

Chunks should retain provenance such as:
- source URL
- source title
- source type
- publisher/organization if known
- document ID
- section
- page if known
- domain/subdomain
- trust score
- content hash

## Source quality
Source quality must be explainable.

Store component signals such as:
- authority
- relevance
- recency
- source type
- accessibility
- duplication
- evidence quality

Return:
- score
- decision
- reasons
- warnings

Never silently treat a random website as authoritative.

## Chunking
Implement a clear baseline first:
- structure-aware/section-aware chunking where possible
- configurable target chunk size and overlap
- preserve headings/section metadata
- avoid destroying tables or structured content unnecessarily

Keep chunking behind an interface so later experiments can compare fixed-size, section-aware, and semantic/domain-aware strategies.

## Retrieval
Start with dense vector retrieval through Qdrant.

Keep the retrieval interface extensible for:
- keyword/BM25
- hybrid retrieval
- reranking

Do not pretend a feature exists until it is implemented.

## Evaluation
Evaluation is a core research feature, not a cosmetic score.

Start with:
- Recall@K
- Precision@K where applicable
- MRR
- NDCG where applicable

Clearly distinguish retrieval metrics from LLM-as-judge and answer-generation metrics.

Never display fabricated percentages.

## Research integrity
The project will be compared against existing systems such as Dify and open-source RAG knowledge-base builders. Do not claim novelty merely because a feature is implemented.

The potentially novel research angle is the automated construction loop:

Domain specification
→ knowledge requirements
→ source discovery
→ source quality/inclusion decisions
→ ingestion
→ domain-aware chunking
→ retrieval
→ evaluation
→ optimization

Keep the implementation modular enough for experimental comparison.

## Engineering rules
- Inspect the repository before making changes.
- Do not overwrite working files without understanding them.
- Use typed interfaces and Pydantic models.
- Keep configuration in environment variables/config files.
- No secrets committed to git.
- Add .env.example files.
- Add useful logging.
- Handle external network failures gracefully.
- Add tests for core deterministic logic.
- Prefer small, composable modules.
- Avoid unnecessary dependencies.
- Keep README setup instructions accurate.
- The application must be runnable locally on Windows.
- Use Docker Compose for Qdrant if useful, with clear setup instructions.

## UI rules
The UI should feel like a research/engineering tool rather than a generic chatbot.

Primary views:
1. Dashboard / Knowledge Bases
2. Create Knowledge Base
3. Domain Analysis
4. Sources
5. Processing / Build Run
6. Chunks
7. Retrieval Lab
8. Evaluation
9. Configuration / Experiments

A simple working UI is better than an elaborate mock dashboard.

## Definition of done for the first build
A developer should be able to:
1. install dependencies,
2. start backend,
3. start frontend,
4. start Qdrant,
5. create a knowledge base,
6. enter a domain such as Automobile Engineering,
7. obtain a structured domain specification,
8. ingest a small real corpus,
9. create real chunks and embeddings,
10. store them in Qdrant,
11. query the knowledge base,
12. inspect retrieved chunks and provenance,
13. run a small real retrieval evaluation,
14. see actual metrics in the UI.

If an external API key is required, make it optional where possible and document configuration.

## Do NOT do in the initial MVP
- full autonomous agent system
- MCP
- scheduling
- multi-agent orchestration
- authentication/billing/cloud deployment
- complex queues unless required
- every file format
- every vector database
- dozens of screens filled with fake data
- fabricated quality scores/evaluation results
- scraping that violates robots.txt, terms, authentication boundaries, or access restrictions

## Development workflow
Before coding:
1. inspect the empty repository;
2. create a concise implementation plan;
3. create the project structure;
4. implement a minimal end-to-end vertical slice;
5. run tests/build/lint;
6. fix errors;
7. document setup;
8. then expand functionality.

For major architectural decisions, record them in docs/architecture.md or equivalent.

At the end of a task, report:
- files created/changed
- commands run
- tests/build checks
- known limitations
- next recommended step

## Priority order
1. Correctness
2. Real end-to-end functionality
3. Reproducibility
4. Provenance and transparency
5. Evaluation
6. Clean architecture
7. UI polish
8. Advanced features
