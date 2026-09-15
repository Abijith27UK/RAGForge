# RAGForge Frontend

The RAGForge frontend is the web interface for the **RAG knowledge-base engineering platform**.

It provides an interactive interface for creating and inspecting domain-specific knowledge bases and interacting with the backend research pipeline.

The frontend communicates with the RAGForge FastAPI backend, while the backend handles the actual domain analysis, source processing, embeddings, vector indexing, retrieval, and evaluation.

---

# Role of the Frontend

The frontend provides a visual interface for the RAGForge workflow:

```text
Domain
   ↓
Analysis
   ↓
Sources
   ↓
Selection
   ↓
Ingestion
   ↓
Chunks
   ↓
Vector Database
   ↓
Retrieval
   ↓
Evaluation
```

The frontend is intentionally separated from the research pipeline so that the backend can also be accessed through APIs and, in the future, MCP.

---

# Technology Stack

* Next.js
* React
* TypeScript
* Tailwind CSS

---

# Frontend Architecture

```text
                    RAGForge Frontend
                           │
                           ▼
                    Next.js / React
                           │
                           ▼
                    Frontend UI
                           │
                           │ HTTP/API
                           ▼
                    FastAPI Backend
                           │
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
          Sources       Retrieval     Evaluation
             │             │              │
             └─────────────┼──────────────┘
                           ▼
                         Qdrant
```

---

# Main Product Concept

The frontend represents RAGForge as a **knowledge-base construction workflow**, rather than simply as a chatbot.

A user should be able to start with a high-level request such as:

```text
Underwater Marine Robotics
```

and inspect the construction process:

```text
Domain Analysis
       ↓
Knowledge Requirements
       ↓
Candidate Sources
       ↓
Source Quality
       ↓
Selected Sources
       ↓
Documents
       ↓
Chunks
       ↓
Embeddings
       ↓
Vector Database
       ↓
Retrieval
       ↓
Evaluation
```

---

# Domain Input

The user provides the target domain that they want to construct a knowledge base for.

Example:

```text
Underwater Marine Robotics
```

The backend analyzes the domain and produces a structured representation of relevant knowledge areas.

The frontend can present these requirements so the user can understand what the generated knowledge base is intended to cover.

---

# Source Selection Interface

The source-selection stage is important because RAGForge is not intended to blindly ingest every available document.

The UI can expose:

* Candidate sources
* Source relevance
* Domain coverage
* Authority
* Accessibility
* Source type
* Selection status

This makes the source-selection process inspectable instead of treating the vector database as a black box.

---

# Knowledge Base View

Once processing is complete, the frontend can present the resulting knowledge base.

A knowledge base conceptually contains:

```text
Knowledge Base
│
├── Domain
├── Knowledge Requirements
├── Sources
├── Documents
├── Chunks
├── Embeddings
├── Vector Index
├── Retrieval Configuration
└── Evaluation Results
```

The vector database itself is only one component of this output.

---

# Retrieval Interface

The frontend can be used to test the generated knowledge base.

Example:

```text
Query:

What is the difference between an AUV and an ROV?
```

The retrieval system returns relevant chunks from the vector database.

A result can contain information such as:

```text
Document
Section
Page
Similarity Score
Retrieved Chunk
Source
```

This allows users to inspect not only the retrieved answer context but also its provenance.

---

# Evaluation Interface

RAGForge provides retrieval evaluation rather than relying only on visual inspection.

The frontend can expose metrics such as:

```text
Recall@K
Precision@K
MRR
NDCG
```

It can also display:

* Benchmark questions
* Retrieved chunks
* Ground-truth matches
* Source provenance
* Evaluation results
* Experiment comparisons

This is particularly important because RAGForge is intended as a research platform for studying knowledge-base construction.

---

# Research Experiment Visualization

The frontend can be used to compare alternative RAG construction strategies.

For example:

```text
Source Selection A
        VS
Source Selection B
```

or:

```text
Fixed-size Chunking
        VS
Section-aware Chunking
```

or eventually:

```text
Qdrant
   VS
TurboVec
```

Relevant metrics can then be compared using the same benchmark.

---

# Backend Connection

The frontend requires the RAGForge FastAPI backend to be running.

The backend is started using:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --reload --port 8000
```

The backend runs on:

```text
http://localhost:8000
```

The frontend should be configured to communicate with the appropriate backend URL according to the project's environment configuration.

---

# Frontend Setup

## 1. Navigate to the Frontend

From the RAGForge root:

```bash
cd frontend
```

---

## 2. Install Dependencies

```bash
npm install
```

---

## 3. Start Development Server

```bash
npm run dev
```

Next.js will display the local development URL in the terminal.

Open that URL in your browser.

---

# Complete Local Setup

RAGForge requires three running components:

```text
┌─────────────────────┐
│       Qdrant        │
│    Vector Store     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   FastAPI Backend   │
│      Port 8000      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Next.js Frontend  │
│    Development UI   │
└─────────────────────┘
```

### Terminal 1 — Qdrant

From the RAGForge root:

```bash
qdrant\qdrant.exe
```

### Terminal 2 — Backend

```bash
cd backend
.venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000
```

### Terminal 3 — Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# Development Workflow

When developing the frontend:

1. Start Qdrant.
2. Start the FastAPI backend.
3. Start the Next.js development server.
4. Open the frontend in the browser.
5. Use the UI to initiate or inspect RAGForge workflows.
6. Inspect the FastAPI logs for backend errors.
7. Inspect Qdrant state when debugging vector indexing or retrieval.

---

# Frontend Design Philosophy

The UI should make the RAG construction process transparent.

Instead of:

```text
Input → Magic → Vector DB
```

RAGForge should expose:

```text
Input
  ↓
Why these knowledge requirements?
  ↓
Why these sources?
  ↓
Why were these sources selected?
  ↓
How were the documents processed?
  ↓
How were they chunked?
  ↓
How were they embedded?
  ↓
What is stored in the vector index?
  ↓
What does retrieval return?
  ↓
How good is the resulting knowledge base?
```

This transparency is important for both engineering usability and the project's research objectives.

---

# Future Frontend Extensions

## Knowledge Base Explorer

A detailed explorer for:

```text
Knowledge Base
├── Domain
├── Requirements
├── Sources
├── Documents
├── Chunks
└── Evaluation
```

---

## Retrieval Playground

Allow users to enter arbitrary questions and inspect:

```text
Query
 ↓
Retrieved Chunks
 ↓
Similarity Scores
 ↓
Source
 ↓
Page / Section
```

---

## Experiment Comparison

Provide side-by-side comparisons of:

```text
Strategy A
vs
Strategy B
```

using:

```text
Recall@K
Precision@K
MRR
NDCG
Latency
Coverage
```

---

## Vector Index Comparison

Once TurboVec support is implemented:

```text
              Vector Index

        ┌──────────┴──────────┐
        ↓                     ↓
      Qdrant               TurboVec
        │                     │
        └──────────┬──────────┘
                   ↓
              Evaluation
```

The frontend can visualize the trade-offs between retrieval quality and computational efficiency.

---

## MCP Integration

The eventual MCP architecture will allow external AI agents to operate RAGForge.

For example:

```text
Claude
   │
   │ MCP
   ▼
RAGForge
   │
   ├── Build Knowledge Base
   ├── Retrieve
   ├── Evaluate
   └── Inspect
```

The frontend remains useful as the human-facing interface, while MCP provides an AI-agent-facing interface.

---

# Status

### Implemented

* RAGForge web interface
* Domain workflow
* Source workflow
* Knowledge-base workflow
* Retrieval/evaluation visualization
* Backend integration

### Planned

* Rich knowledge-base explorer
* Advanced retrieval playground
* Experiment comparison
* Qdrant vs TurboVec visualization
* Automated optimization visualization
* MCP-aware agent workflow
