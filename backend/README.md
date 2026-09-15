# RAGForge Backend

The RAGForge backend is the **FastAPI-based research and knowledge-base construction engine** behind RAGForge.

It is responsible for the core RAG knowledge-engineering pipeline:

```text
Domain
  ↓
Domain Analysis
  ↓
Source Discovery
  ↓
Source Quality Assessment
  ↓
Source Selection
  ↓
Document Ingestion
  ↓
Chunking
  ↓
Embeddings
  ↓
Qdrant
  ↓
Retrieval
  ↓
Evaluation
```

The backend exposes the functionality required by the RAGForge frontend and provides the foundation for future MCP integration.

---

# Responsibilities

The backend is responsible for:

* Domain analysis
* Knowledge requirement generation
* Source discovery
* Source validation
* Source quality assessment
* Content-aware source selection
* Document ingestion
* Document parsing
* Duplicate detection
* Content hashing
* Chunking
* Embedding generation
* Vector indexing
* Retrieval
* Ground-truth evaluation
* Retrieval metrics
* Provenance tracking
* Experiment artifacts

---

# Architecture

```text
                   FastAPI Application
                          │
                          ▼
                 Research Pipeline
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
   Domain Analysis   Source Discovery   Evaluation
        │                 │                  │
        ▼                 ▼                  ▼
 Domain Knowledge    Candidate Sources   Benchmark
        │                 │                  │
        └─────────────────┼──────────────────┘
                          ▼
                 Source Quality Engine
                          │
                          ▼
                  Selected Sources
                          │
                          ▼
                      Ingestion
                          │
                          ▼
                  Structured Documents
                          │
                          ▼
                      Chunking
                          │
                          ▼
                       Chunks
                          │
                          ▼
                     Embeddings
                          │
                          ▼
                       Qdrant
                          │
                          ▼
                      Retrieval
                          │
                          ▼
                     Evaluation
```

---

# Vector Database

The current backend uses **Qdrant** as the primary vector database.

The backend communicates with the locally running Qdrant instance to:

* Create/manage vector collections
* Store document embeddings
* Store chunk metadata
* Perform similarity search
* Retrieve provenance information

The architecture is intended to allow alternative vector-index implementations in the future.

A planned experimental backend is TurboVec, which can be evaluated against Qdrant using the same corpus, embeddings, benchmark, and retrieval metrics.

---

# Embedding Model

The current default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Output dimension:

```text
384
```

Pipeline:

```text
Chunk
  ↓
Sentence Transformer
  ↓
384-dimensional embedding
  ↓
Qdrant
```

---

# Source Selection

Source selection is one of the primary research components of the backend.

The content-aware selection pipeline evaluates candidate sources against domain requirements.

Factors can include:

* Content relevance
* Domain coverage
* Authority
* Accessibility
* Recency
* Source type
* Evidence quality
* Duplication

The source-selection system evolved from an initial metadata-oriented approach after experiments showed that the initial scoring mechanism did not sufficiently differentiate candidate sources.

---

# Ingestion

The ingestion pipeline currently supports:

```text
PDF
HTML
TXT
Markdown
```

The ingestion system handles:

* Parsing
* Content extraction
* Hashing
* Duplicate detection
* Size restrictions
* Timeouts
* Graceful failures
* Provenance

For PDFs, page-level information can be propagated into chunk metadata.

---

# Chunking

RAGForge currently uses section-aware chunking.

Example:

```text
Document
│
├── Introduction
│
├── 1. Vehicle Dynamics
│   ├── 1.1 Forces
│   └── 1.2 Stability
│
└── 2. Propulsion
```

The resulting chunks retain structural metadata where available.

Future research can compare:

```text
Fixed-size chunking
        vs
Section-aware chunking
        vs
Domain-aware chunking
```

---

# Retrieval

The current retrieval path is dense vector retrieval:

```text
Query
  ↓
Query Embedding
  ↓
Qdrant Similarity Search
  ↓
Top-K Results
  ↓
Chunk + Metadata + Provenance
```

Current retrieval infrastructure provides the foundation for future:

* BM25
* Hybrid retrieval
* Reranking
* MMR
* Retrieval optimization

experiments.

---

# Evaluation

The backend contains an evaluation framework for measuring retrieval quality.

Current metrics include:

```text
Recall@K
Precision@K
MRR
NDCG
```

Evaluation can use:

* Chunk-level ground truth
* Document-level ground truth
* Benchmark questions
* Strict evaluation
* Diagnostic keyword evaluation

The evaluation framework is designed to prevent circular evaluation and provide reproducible experimental results.

---

# Research Experiments

The backend supports controlled comparisons between different RAG construction strategies.

Conceptually:

```text
                 Experiment
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      Strategy A            Strategy B
          │                     │
          └──────────┬──────────┘
                     ▼
              Same Benchmark
                     │
                     ▼
                  Metrics
                     │
                     ▼
              Research Result
```

One completed source-selection experiment found that the initial scoring approach was insufficiently discriminative. A subsequent content-aware scorer produced differentiated scores and improved tested domain coverage, while retrieval-quality superiority still requires further validation.

---

# Backend Setup

## Requirements

Install:

* Python
* pip
* Qdrant

Qdrant is provided with the repository as a local executable.

---

## Create Virtual Environment

From the RAGForge root:

```bash
cd backend
```

Create the environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create the local environment file:

```bash
copy .env.example .env
```

Update `.env` with the configuration required by your local setup.

Do not commit `.env` if it contains secrets or API credentials.

---

# Start Qdrant

Qdrant should be running before starting the backend.

From the repository root:

```bash
qdrant\qdrant.exe
```

Keep the Qdrant process running.

---

# Start FastAPI

With the virtual environment activated:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

# Development Workflow

A typical development session uses two terminals for the backend infrastructure.

### Terminal 1

```bash
qdrant\qdrant.exe
```

### Terminal 2

```bash
cd backend
.venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000
```

---

# Backend Data Flow

```text
Domain Request
      ↓
Research Pipeline
      ↓
Domain Knowledge Map
      ↓
Source Discovery
      ↓
Source Assessment
      ↓
Source Selection
      ↓
Document Download
      ↓
Parsing
      ↓
Chunking
      ↓
Embedding
      ↓
Qdrant
      ↓
Retrieval
      ↓
Evaluation
```

---

# Future Backend Extensions

## Alternative Vector Index

The vector-store layer can eventually support:

```text
Qdrant
   │
   └── Primary vector database

TurboVec
   │
   └── Experimental efficient vector index
```

The comparison should use the same embeddings and evaluation benchmark and measure:

* Retrieval quality
* Query latency
* Index construction time
* Memory usage
* Index size

This makes the vector-store comparison a controlled research experiment rather than simply an additional dependency.

---

## Automated Optimization

A future backend version can implement:

```text
Evaluate
   ↓
Diagnose failure
   ↓
Identify possible cause
   ├── Source quality
   ├── Domain coverage
   ├── Chunking
   ├── Embeddings
   └── Retrieval configuration
   ↓
Modify strategy
   ↓
Rebuild
   ↓
Evaluate again
   ↓
Keep improved version
```

This is a major direction for RAGForge's research contribution.

---

## MCP Server

The backend is also the natural location for a future RAGForge MCP server.

Potential tools:

```text
analyze_domain
discover_sources
assess_sources
build_knowledge_base
retrieve
evaluate
compare_experiments
inspect_source
inspect_chunk
```

The MCP layer would expose the backend to external AI agents without changing the internal RAG pipeline.

---

# Backend Status

### Implemented

* FastAPI application
* Domain analysis
* Source discovery
* Source quality assessment
* Content-aware source selection
* Document ingestion
* Chunking
* Sentence Transformer embeddings
* Qdrant integration
* Dense retrieval
* Retrieval evaluation
* Ground truth
* Provenance
* Experiment infrastructure

### Planned

* Improved source-selection experiments
* Domain-aware chunking
* Hybrid retrieval
* Reranking
* TurboVec backend
* Automated optimization
* MCP server
