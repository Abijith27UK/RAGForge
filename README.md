# RAGForge

## Automated Domain-Specific RAG Knowledge Base Engineering

RAGForge is a platform for **constructing, evaluating, and managing domain-specific knowledge bases for Retrieval-Augmented Generation (RAG) systems**.

It supports two fundamentally different workflows:

| Workflow | You give it | It builds |
|---|---|---|
| **External knowledge** | a domain description | a KB from discovered, scored, authoritative public sources |
| **User knowledge** | your own lecture decks, PDFs, notes, URLs | a private KB from your material, with per-page/per-slide citations |

```text
"Build a knowledge base about automobile engineering."        → EXTERNAL
"Build a Naval Architecture KB from my lecture notes."        → USER_PROVIDED
"Use my documents plus discovered sources."                  → MIXED
```

**A knowledge base needs no ground truth.** A KB becomes `READY` once it is
indexed. An evaluation benchmark is a *separate, optional* instrument you can add
later to measure retrieval quality. See [docs/user-knowledge-workflow.md](docs/user-knowledge-workflow.md).

> ### ⚠️ A new domain does not automatically get ground truth
>
> RAGForge does **not** generate verified ground truth for you, and no benchmark
> ships with a new knowledge base.
>
> The only benchmark here is the **Automobile Engineering** one (28 questions),
> and it exists because a human read the indexed chunks, recorded the passage
> answering each question, and reviewed the set before freezing it.
>
> A brand-new knowledge base — a Naval Architecture one, say — is fully usable for
> retrieval, but its retrieval quality is **unmeasured**, not "good". The UI says
> `Evaluation: Not configured`, metrics are `null` rather than placeholders, and
> building the benchmark is a deliberate human process:
>
> ```
> author questions → human review → APPROVED → snapshot → human review
>                  → FROZEN benchmark → run evaluation → real metrics
> ```
>
> Auto-generated "ground truth" would be a fabricated evaluation instrument, which
> is the exact failure mode RAGForge exists to avoid.

---

## What Problem Does RAGForge Solve?

Building a reliable RAG system is not simply a matter of putting documents into a vector database.

A high-quality domain-specific RAG system requires decisions about:

* What knowledge is required for the domain?
* Which sources are relevant?
* Which sources are trustworthy?
* Which sources provide sufficient domain coverage?
* How should documents be processed?
* How should documents be chunked?
* Which embedding model should be used?
* How should the vectors be indexed?
* Does the resulting knowledge base actually cover the domain?
* Does retrieval return the correct information?
* How can the resulting knowledge base be quantitatively evaluated?

RAGForge addresses this **knowledge-base engineering problem**.

### Core Research Question

> **Can the construction of a high-quality domain-specific RAG knowledge base be automated from a high-level domain specification while maintaining source quality, domain coverage, retrieval quality, provenance, and computational efficiency?**

---

# RAGForge in One Picture

```text
                         DOMAIN
                           │
                           ▼
                  ┌─────────────────┐
                  │  Domain Analysis│
                  └────────┬────────┘
                           │
                           ▼
                  Domain Knowledge Map
                           │
                           ▼
                  ┌─────────────────┐
                  │ Source Discovery│
                  └────────┬────────┘
                           │
                           ▼
                  Candidate Sources
                           │
                           ▼
             ┌───────────────────────────┐
             │ Source Quality & Selection│
             └─────────────┬─────────────┘
                           │
                           ▼
                    Selected Sources
                           │
                           ▼
                  ┌─────────────────┐
                  │    Ingestion    │
                  └────────┬────────┘
                           │
                           ▼
                  Structured Documents
                           │
                           ▼
                  ┌─────────────────┐
                  │     Chunking    │
                  └────────┬────────┘
                           │
                           ▼
                    Chunks + Metadata
                           │
                           ▼
                  ┌─────────────────┐
                  │    Embeddings   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     Qdrant      │
                  │  Vector Database│
                  └────────┬────────┘
                           │
                           ▼
                       Retrieval
                           │
                           ▼
                      Evaluation
                           │
                           ▼
                    Quality Report
```

---

# Why Is This Different From a Normal RAG Application?

A conventional RAG application generally assumes that the knowledge corpus already exists.

```text
Documents
   ↓
Chunk
   ↓
Embed
   ↓
Vector DB
   ↓
Retrieve
   ↓
LLM
```

RAGForge focuses on the **construction and evaluation of the knowledge layer itself**.

```text
Domain
   ↓
What knowledge is required?
   ↓
Which sources should be used?
   ↓
Which sources are trustworthy?
   ↓
How should they be processed?
   ↓
How should they be indexed?
   ↓
Does the resulting knowledge base work?
   ↓
Can it be improved?
```

In short:

> **LLMs such as Claude or GPT primarily consume knowledge to answer questions. RAGForge is designed to engineer the domain-specific knowledge layer that those AI systems can consume.**

The resulting vector database is therefore **an output of the knowledge-engineering pipeline, not the entire product**.

---

# Example Use Case

Suppose a user wants:

```text
Underwater Marine Robotics
```

RAGForge can construct a domain-specific knowledge base covering areas such as:

```text
Underwater Marine Robotics
│
├── Autonomous Underwater Vehicles (AUVs)
├── Remotely Operated Vehicles (ROVs)
├── Hydrodynamics
├── Propulsion
├── Navigation
├── Sensors
├── Control Systems
├── Underwater Communication
└── Power and Battery Systems
```

The system discovers and evaluates candidate sources, selects relevant sources, processes their content, generates embeddings, and creates a searchable vector index.

The resulting knowledge base can then be queried:

```text
"What are the main differences between an AUV and an ROV?"
```

The retrieval layer returns relevant passages together with metadata and provenance.

An external LLM can then use those retrieved passages to generate a grounded answer.

---

# What Is the Final Output?

RAGForge does not only produce a vector index.

A generated knowledge base can conceptually contain:

```text
Knowledge Base
│
├── Domain Specification
├── Domain Knowledge Map
├── Selected Sources
├── Source Quality Information
├── Documents
├── Chunks
├── Embeddings
├── Vector Index
├── Metadata
├── Provenance
├── Retrieval Configuration
├── Evaluation Benchmark
└── Evaluation Results
```

This makes the knowledge base **inspectable, reproducible, and reusable**.

---

# Core Features

## 1. Domain Analysis

RAGForge starts from a high-level domain description and creates a structured representation of the knowledge that should be covered.

Example:

```text
Input:
Automobile Engineering
```

can be represented through requirements such as:

```text
Engine
Transmission
Braking
Steering
Suspension
Chassis
Vehicle Dynamics
Electric Vehicles
Fundamentals
```

These requirements can subsequently guide source selection and coverage evaluation.

---

## 2. Source Discovery

RAGForge supports discovery of candidate knowledge sources.

Current source discovery includes:

* User-provided URLs
* arXiv-based research discovery

Candidate sources are subsequently validated and assessed before being incorporated into the knowledge base.

---

## 3. Content-Aware Source Selection

Source selection is one of the central components of RAGForge.

Rather than blindly ingesting every discovered source, RAGForge evaluates sources using multiple signals, including:

* Content relevance
* Domain coverage
* Source authority
* Accessibility
* Recency
* Source type
* Evidence quality
* Duplication

The content-aware approach evaluates the **actual content of sources against the requirements of the target domain**.

This allows the system to construct a more focused corpus rather than simply collecting a large number of documents.

---

## 4. Document Ingestion

RAGForge processes supported documents into a common internal representation.

| Format | Parser | Structural provenance |
|---|---|---|
| PDF | `PdfParser` (pypdf) | `page` |
| PPTX | `PptxParser` (python-pptx) | `slide`, `slide_title`, speaker notes, tables |
| DOCX | `DocxParser` (python-docx) | `section_path` (heading hierarchy), tables as pipe rows |
| HTML | `HtmlParser` (BeautifulSoup) | cleaned text |
| Markdown | `MarkdownParser` | heading structure preserved |
| TXT | `TextParser` | normalized text |

Legacy binary `.ppt` is accepted by the uploader but explicitly rejected by the
parser with an actionable message ("Re-save the deck as .pptx").

The ingestion pipeline includes:

* Upload validation: file-name sanitisation, size caps, magic-byte sniffing,
  OOXML container integrity (a `.docx` renamed to `.pptx` is rejected)
* Content hashing and duplicate detection
* Timeout handling and graceful per-document failure
* Provenance preservation at document / page / slide / section / chunk level

**Page, slide and section numbers are read from the file, never inferred.** A
deck with no title placeholder yields `slide_title = None`; a scanned PDF fails
with an explicit "OCR is not implemented" message rather than producing an empty
document.

---

## 5. Section-Aware Chunking

RAGForge currently supports section-aware document chunking.

For example:

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

The system preserves structural information such as section paths as chunk metadata.

This allows retrieved information to retain context about where it originated within the source document.

---

## 6. Embedding Generation

The current default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

Pipeline:

```text
Document
   ↓
Chunk
   ↓
Embedding Model
   ↓
384-dimensional Vector
```

---

## 7. Vector Database

RAGForge currently uses **Qdrant** as its primary vector database.

The vector index stores embeddings together with relevant metadata and provenance.

Conceptually:

```text
Vector
│
├── Chunk
├── Document
├── Source
├── Section / Slide / Page
├── Document version
├── Knowledge-base version
└── Chunking strategy + config
```

Qdrant is currently the primary production/development vector backend. A
`VectorStore` abstraction and a backend factory exist so an experimental backend
(e.g. TurboVec) can be registered without changing call sites; none is
implemented yet.

---

## 7b. Document Library & Incremental Ingestion

Every knowledge base has a **document library** with per-document status:

```text
UPLOADED → PARSING → PARSED → CHUNKING → INDEXING → READY
                   ↘ FAILED
```

Adding the 101st document to a 100-document knowledge base does **not** re-embed
the other 100:

```text
add document 101 → parse → chunk → embed → index (only that document)
replace a document → delete old vectors → ingest new version → index new chunks
delete a document → remove its vectors (never leave stale points behind)
```

Both the full build and the incremental path call one shared function, so
stale-vector handling cannot drift between them. `Document.document_version` and
`KnowledgeBase.version` are stamped onto every chunk, so any retrieved chunk can
be traced back to the exact index that produced it.

---

## 8. Retrieval

Retrieval is a pluggable layer with four strategies behind one interface. The
strategy is selected by name (registry), never hard-coded at the call site.

```text
User Query
    ↓
Candidate retrieval      dense (Qdrant vectors)   +   lexical (BM25 over indexed chunks)
    ↓
Normalization            min-max (default) or rank, per candidate pool
    ↓
Fusion                   weighted (dense 0.65 / bm25 0.35, configurable) or RRF
    ↓
Diversification          optional: per-document cap, or MMR
    ↓
Reranking                optional cross-encoder (honest fallback when unavailable)
    ↓
Top-K chunks + provenance + score breakdown
```

| strategy | what runs | needs embeddings |
|---|---|---|
| `dense` | Qdrant vector similarity (the V1–V5 baseline, unchanged) | yes |
| `bm25` | real BM25 (`k1=1.2`, `b=0.75`, configurable) over SQLite chunks; loads no embedding model | no |
| `hybrid` | dense + BM25, normalized and fused (weighted or RRF) | yes |
| `hybrid_reranked` | hybrid + cross-encoder rerank | yes |

Every result reports how it was selected: raw and normalized dense/lexical scores,
the fused score and each source's contribution, the pipeline stages it survived,
and a deterministic `why` sentence. Each request is recorded as a retrieval run
(strategy, parameters, applied weights, corpus version, measured stage timings) and
returns a `retrieval_run_id`.

The retrieved context can then be supplied to an external LLM-based RAG application.

Details and honest limitations: [docs/retrieval-architecture.md](docs/retrieval-architecture.md).

---

## 9. Retrieval Evaluation (OPTIONAL)

RAGForge evaluates the resulting knowledge base rather than assuming that successful indexing means successful retrieval.

**Evaluation is never required.** A knowledge base with no questions, no benchmark
and no runs is a valid, fully usable artifact; the UI reports
`Evaluation: Not configured` and the KB is still `READY`.

Current evaluation metrics include:

* Recall@K
* Precision@K
* Mean Reciprocal Rank (MRR)
* Normalized Discounted Cumulative Gain (NDCG)

The evaluation framework supports:

* Chunk-level ground truth
* Document-level ground truth
* Question lifecycle DRAFT → REVIEW → APPROVED → FROZEN with reviewer attribution
* Immutable benchmark snapshots; only FROZEN versions are usable for official runs
* Strict evaluation (explicit ground truth only)
* Diagnostic keyword evaluation (opt-in and labelled)
* Provenance tracking
* Experiment artifacts

Metrics are `null` when ground truth is missing — never `0`, never a fabricated
percentage. **See the warning at the top of this file: a new domain has no
benchmark until a human builds one.**

---

# Research & Experimental Framework

RAGForge is designed as a **research platform**, not simply as a production RAG application.

The system supports controlled experiments of the form:

```text
             Experiment
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
   Strategy A          Strategy B
       │                   │
       └─────────┬─────────┘
                 ▼
          Same Benchmark
                 │
                 ▼
              Metrics
                 │
                 ▼
        Research Conclusion
```

This allows individual components of RAG construction to be studied independently.

---

# Source Selection Experiment

An initial source-selection strategy based primarily on metadata was evaluated on an Automobile Engineering corpus.

The experiment exposed an important limitation:

```text
36 candidate sources
        ↓
34 sources received identical scores
```

This meant the scoring mechanism was not sufficiently discriminative.

Instead of treating this as a successful result, the limitation motivated a content-aware source-selection approach.

The revised approach produced differentiated scores across the candidate sources and improved tested domain coverage under the experimental conditions.

However, the retrieval-quality superiority hypothesis requires further controlled experimentation.

This iterative process is an important part of the research methodology of RAGForge.

---

# Current Research Directions

RAGForge is being developed toward a more complete automated RAG knowledge-engineering system.

## 1. Source Selection

Compare:

```text
Random Selection
       vs
Metadata-Based Selection
       vs
Content-Aware Selection
       vs
Domain-Aware Selection
```

Potential measurements:

```text
Domain Coverage
Recall@K
MRR
NDCG
Corpus Size
```

---

## 2. Chunking

Compare:

```text
Fixed-Size Chunking
        vs
Section-Aware Chunking
        vs
Domain-Aware Chunking
```

---

## 3. Retrieval

Extend the current dense retrieval baseline with:

```text
Dense Retrieval
      ↓
BM25
      ↓
Hybrid Retrieval
      ↓
Hybrid + Reranking
```

The objective is to experimentally determine whether each additional technique actually improves retrieval quality.

---

# TurboVec: Experimental Vector Index

RAGForge is designed so that vector storage can eventually become a pluggable component.

The current architecture is:

```text
Embedding
    ↓
Vector Index Layer
    │
    ├── Qdrant
    │
    └── TurboVec [planned]
    ↓
Retrieval
    ↓
Evaluation
```

TurboVec is being considered as an **experimental high-efficiency vector index**, not as an immediate replacement for Qdrant.

A controlled experiment can compare:

```text
Qdrant
   vs
TurboVec
```

using the same:

* Corpus
* Documents
* Chunks
* Embeddings
* Benchmark
* Ground truth

Potential metrics include:

```text
Recall@K
MRR
NDCG
Query Latency
p50 / p95 Latency
Index Construction Time
Memory Usage
Index Size
```

This allows RAGForge to study the trade-off between **retrieval quality and vector-index efficiency**.

---

# Automated RAG Optimization

A longer-term direction is to make RAGForge capable of improving its own knowledge-base construction process.

The intended workflow is:

```text
Build Knowledge Base
        ↓
Evaluate
        ↓
Identify Weakness
        ↓
Diagnose Possible Cause
        │
        ├── Poor Sources
        ├── Poor Coverage
        ├── Poor Chunking
        ├── Embedding Mismatch
        └── Retrieval Configuration
        ↓
Modify Strategy
        ↓
Rebuild
        ↓
Evaluate Again
        ↓
Keep Improved Version
```

This would move RAGForge from a **RAG construction system** toward an **automated RAG knowledge-base optimization system**.

---

# MCP Integration

A future version of RAGForge can expose its functionality through the **Model Context Protocol (MCP)**.

This would allow AI agents such as Claude and other MCP-compatible systems to interact with RAGForge directly.

The architecture would be:

```text
                 AI Agent
              Claude / GPT
                    │
                    │ MCP
                    ▼
           ┌──────────────────┐
           │ RAGForge MCP     │
           │ Server           │
           └────────┬─────────┘
                    │
                    ▼
             RAGForge Backend
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Sources     Retrieval   Evaluation
        │           │           │
        └───────────┼───────────┘
                    ▼
               Vector Index
```

Potential MCP tools include:

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

This would allow an external AI agent to perform operations such as:

```text
"Create a knowledge base for Underwater Marine Robotics."
```

RAGForge could then construct the knowledge base.

The same agent could subsequently call:

```text
"Retrieve information about AUV navigation from the
Underwater Marine Robotics knowledge base."
```

RAGForge would return the relevant passages and provenance, allowing the AI agent to use them as grounded context for its response.

### Important distinction

MCP is an **integration layer**.

It does not replace:

* Embeddings
* Qdrant
* TurboVec
* Chunking
* Retrieval
* Evaluation

Instead:

```text
External AI Agent
        ↕
       MCP
        ↕
     RAGForge
```

---

# End-to-End Vision

The long-term RAGForge architecture is:

```text
                         USER / AI AGENT
                                │
                   ┌────────────┴────────────┐
                   │                         │
                 Web UI                    MCP
                   │                         │
                   └────────────┬────────────┘
                                ▼
                       RAGForge Orchestrator
                                │
                                ▼
                         Domain Analysis
                                │
                                ▼
                      Domain Knowledge Map
                                │
                                ▼
                       Source Discovery
                                │
                                ▼
                    Source Quality Analysis
                                │
                                ▼
                       Source Selection
                                │
                                ▼
                           Ingestion
                                │
                                ▼
                       Document Parsing
                                │
                                ▼
                     Domain-Aware Chunking
                                │
                                ▼
                          Embeddings
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                 Qdrant                 TurboVec
                    │                       │
                    └───────────┬───────────┘
                                ▼
                        Retrieval Engine
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                 Dense                 Hybrid/Rerank
                    │                       │
                    └───────────┬───────────┘
                                ▼
                           Evaluation
                                │
                                ▼
                      Quality Diagnostics
                                │
                                ▼
                     Optimization Loop
                                │
                                └──────────► Rebuild
```

---

# Repository Structure

```text
RAGForge/
│
├── README.md
│
├── backend/
│   ├── README.md
│   ├── app/
│   ├── scripts/
│   ├── tests/
│   ├── requirements.txt
│   ├── .env.example
│   └── ...
│
├── frontend/
│   ├── README.md
│   ├── package.json
│   └── ...
│
├── docs/
│   ├── architecture.md
│   ├── user-knowledge-workflow.md
│   ├── v3-product-architecture.md
│   ├── roadmap.md
│   └── verification.md
│
├── benchmarks/            frozen benchmark + experiment artifacts (read-only)
│
└── qdrant/
    └── qdrant.exe
```

---

# Technology Stack

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

### Backend

* Python
* FastAPI
* Uvicorn
* Sentence Transformers

### Data & Retrieval

* Qdrant
* arXiv
* `sentence-transformers/all-MiniLM-L6-v2`
* `python-pptx` / `python-docx` / `pypdf` / BeautifulSoup for document parsing

### Implemented (V6 retrieval layer)

* BM25 retrieval (real term weighting, persisted index, exact staleness detection)
* Hybrid retrieval (dense + BM25) with weighted fusion **and** Reciprocal Rank Fusion
* Optional cross-encoder reranking with an honest `unavailable_fallback` status
* Optional diversification (per-document cap, MMR)
* Per-KB retrieval configuration + retrieval run records (observability)

### Implemented (V7 grounded answer engine)

* Deterministic query processing (heuristic-labelled query plans; original question preserved)
* Evidence assembly with full provenance and auditable deduplication
* Multi-signal evidence gate: `ANSWER / PARTIAL_ANSWER / ABSTAIN / ASK_CLARIFICATION`
  (categorical confidence only — never a fabricated percentage)
* Provider-agnostic grounded generation (OpenAI-compatible, deterministic mock for tests)
* Claim → Evidence → Source citation chain with deterministic citation validation
* First-class abstention: "I don't have enough evidence" beats a plausible guess
* Answer trace: every stage inspectable (`GET /answer-traces/{id}`)
* Prompt-injection defense (evidence wrapped as untrusted data; marker scan)
* Answer UI with the "How was this answer produced?" panel

### Implemented (V7.2 grounded knowledge assistant)

* `POST /chat` — grounded chat that **never** hides the retrieval process:
  answer, citations, grounding, evidence, `query_trace`, warnings all in one response
* Five explicit grounding states: `ANSWERED`, `PARTIALLY_SUPPORTED`,
  `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, `NO_RELEVANT_EVIDENCE` —
  each with machine-readable reasons, never a blended confidence score
* `QueryTrace` proving what was done to your question (rewrites, decomposition,
  expansions, processor version, measured timing) — and provably *nothing*
  when the question passed through unchanged
* Full citation provenance: `source_title`, `source_type`, `page_number`,
  `slide_number`, `section_path`, `content_hash` (never fabricated — a missing
  page renders as "not recorded")
* Minimal conversation memory for reference resolution ("What about the previous
  case?") that is **never** promoted to knowledge, and never crosses knowledge
  bases
* `AnswerRun` observability: measured latencies, grounding reasons, selected
  evidence ids and every version string for reproducible, auditable answers
* No silent fallback: a degraded generator is disclosed on the answer itself
* Three-pane grounded chat UI (corpus / conversation / evidence) where clicking a
  citation opens the exact supporting chunk

See [docs/grounding-and-citations.md](docs/grounding-and-citations.md) and
[docs/chat-architecture.md](docs/chat-architecture.md).

### Implemented (V8 answer-quality evaluation)

* `benchmarks/answer-quality-automobile-v1.json` — an answer benchmark whose
  ground truth is **evidence-based, not answer-based**. 28 questions inherit the
  frozen retrieval benchmark's human-selected chunks; `expected_answer` and
  `key_points` are deliberately **empty** because no human wrote them
* 13 answer-quality metrics, each reported as a `Measured` value **or** an
  explicit reason it could not be measured — UNKNOWN is never rendered as `0`
* **`correctness` and `key_point_recall` are UNKNOWN for all 28 questions.**
  The frozen benchmark has no reference answers, and scoring against retrieved
  text would re-measure retrieval while calling it answer quality
* Claim-level citation audit: every claim is checked against the evidence it
  actually cites, with polarity/negation conflict detection
* Retrieval misses are scored **separately** from citation quality, so a
  retrieval failure is never blamed on the answerer
* Immutable run records recording every producer (generator, model, mock flag,
  prompt/answerer/evaluator versions, entailment provider) and the exact
  question subset; runs over different subsets **cannot** be compared
* Answer Quality UI: provenance, measured-vs-UNKNOWN grid, and per-question
  failure cards with the full claim audit

Measured on the real 28-question corpus (dense, mock generator): `pass_rate`
0.821, `citation_recall` 0.821, `retrieval_hit_rate` 0.964,
`abstention_accuracy` 1.000. Correctness UNKNOWN. This is a measurement
apparatus, **not** a claim that answer quality is solved.

See [docs/answer-evaluation.md](docs/answer-evaluation.md),
[docs/answer-benchmark-design.md](docs/answer-benchmark-design.md) and
[docs/v8-continuation-checkpoint.md](docs/v8-continuation-checkpoint.md).

### Extended (V8 continuation — answer evaluation & reliability)

* **Answer relevance is now measured**: IDF-weighted question-term coverage
  (plain coverage was *measured to be inverted* on known off-domain answers
  and is kept only as a labelled fallback). Abstentions are exempt; the
  threshold's thin margin is documented, not oversold
* **Claim-level five-state verdicts** (`SUPPORTED` / `PARTIALLY_SUPPORTED` /
  `UNSUPPORTED` / `CONTRADICTED` / `UNVERIFIABLE`) with supported/partial/
  unsupported ratios over ALL claims
* **Fabrication is measured, not assumed**: `fabricated_citation_rate` and
  `unsupported_citation_rate` (judged citations only)
* **Completeness from human labels** when they exist (`key_point_recall` /
  `expected_information_coverage`, `reference_answer_similarity`);
  `correctness` stays UNKNOWN until a human reviews or an explicitly
  model-based judge scores it
* **Human review workflow**: append-only, attributed `answer_reviews` with
  closed verdict/label vocabularies; a derivation pass turns reviews into a
  NEW run's `correctness` without mutating anything
* **Evaluator abstraction**: deterministic (default) / human / LLM — each
  kind must be requested explicitly, LLM-as-judge stores model + prompt
  version + raw output and is never ground truth
* **Benchmark lifecycle** `draft → approved → frozen`; `official: true` runs
  are refused unless the benchmark is FROZEN (a new domain never gets
  implicit ground truth)
* **Comparability verdicts**: `COMPARABLE` (identical subsets),
  `INCONCLUSIVE` (intersection only), `NOT_COMPARABLE` (zero overlap or
  changed content)
* **Reliability dashboard** (`/knowledge-bases/[id]/reliability`): strategies
  compared across SEPARATE dimensions — deliberately **no combined score**
* Experiment: [`docs/answer-evaluation-v1.md`](docs/answer-evaluation-v1.md) —
  dense/bm25/hybrid on 28 questions (hybrid best: `pass_rate` 0.893,
  `hallucination_rate` 0.107; all comparisons COMPARABLE; Qdrant 14→14;
  non-official because the source benchmark is `draft`)

Full metric definitions and limits:
[docs/answer-evaluation-architecture.md](docs/answer-evaluation-architecture.md).

### Planned / Experimental

* **Human-authored reference answers** for a question subset (plus genuinely
  unanswerable questions) — the only thing blocking `correctness` from being
  measured rather than UNKNOWN. **This is the recommended next phase.**
* Fixing the extractive mock generator's rank truncation: it walks evidence in
  rank order with a 5-claim budget, making evidence ranked 4th+ structurally
  uncitable (measured: the required chunk was retrieved at rank 5 and never
  cited in 4 of 5 real failures)
* Model-based (LLM) entailment judge — `LLMCitationEntailment` exists and
  refuses to run without a provider, but is not the default
* Benchmark authoring UI so a non-Automobile domain can acquire real ground truth
* Query decomposition (multi-part questions are currently retrieved as one query,
  and the trace says so)
* Retrieval Lab V2 parameter controls, strategy-comparison experiment runner
* TurboVec (optional experimental vector backend)
* Domain-Aware Chunking
* Automated RAG Optimization
* MCP (deliberately not implemented yet)

---

# Running RAGForge Locally

## Prerequisites

Install the following:

* Python
* Node.js
* npm
* Git

Qdrant is included in the repository as a local executable.

---

## 1. Start Qdrant

From the RAGForge root directory:

```bash
qdrant\qdrant.exe
```

Keep this terminal running.

---

## 2. Start the Backend

Open a new terminal:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
copy .env.example .env
```

Start the FastAPI backend:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

FastAPI API documentation:

```text
http://localhost:8000/docs
```

---

## 3. Start the Frontend

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL shown by Next.js.

---

# Quick Start

The complete local setup requires three running processes.

### Terminal 1 — Qdrant

```bash
qdrant\qdrant.exe
```

### Terminal 2 — Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --reload --port 8000
```

### Terminal 3 — Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# Typical Workflows

## A. External knowledge

```text
1. Create a knowledge base (source mode: External)
        ↓
2. Analyze the domain
        ↓
3. Discover candidate sources (user URLs / arXiv)
        ↓
4. Review explainable quality scores; accept sources
        ↓
5. Ingest → chunk → embed → index
        ↓
6. Retrieve and inspect provenance
        ↓
7. (optional) Author questions, freeze a benchmark, evaluate
```

## B. Your own knowledge

```text
1. Create a knowledge base (source mode: My files)
        ↓
2. Drag & drop lecture PPTs / PDFs / notes
        ↓
3. Files are validated, parsed and de-duplicated locally
        ↓
4. Documents become chunks → embeddings → vectors
        ↓
5. KB status: READY  ← no benchmark required
        ↓
6. Retrieval Lab returns passages with document / slide / page / section
        ↓
7. (later) Add documents any time — only the new one is indexed
```

---

# Current Status

## Implemented

**Knowledge building**

* Domain analysis (LLM provider abstraction, structured DomainSpec, labelled dev mock)
* Source discovery (user URLs, arXiv) with SSRF-safe validation and HTTP probing
* Content-aware source selection (v2) with explainable, overridable quality decisions
* **Source modes: `EXTERNAL` / `USER_PROVIDED` / `MIXED`**
* **User document upload**: drag & drop, progress, validation, duplicate detection, failure reporting
* **Parsers**: PDF, PPTX, DOCX, HTML, Markdown, TXT (+ explicit legacy `.ppt` rejection)
* **Document library** with statuses, inspection, rebuild, replace and delete
* **Incremental ingestion** with stale-vector safety
* Duplicate detection and content hashing
* Section-aware + fixed-size chunking (registry)

**Retrieval & index**

* Sentence Transformers `all-MiniLM-L6-v2` embeddings (384d)
* Qdrant vector indexing via a `VectorStore` abstraction + backend factory
* Dense retrieval via a `Retriever` abstraction + registry
* Provenance across document / page / slide / section / chunk / versions

**Evaluation (optional)**

* Recall@K, Precision@K, MRR, NDCG — chunk- and document-level
* Question lifecycle with reviewer attribution, revision-on-approved-edit, freeze protection
* Immutable frozen benchmark snapshots; only FROZEN versions usable for official runs
* Experiment artifacts, rendered read-only

**Interface**

* Next.js dashboard: guided Create KB wizard, KB overview, Document Library,
  Sources (user-provided vs discovered), Processing, Chunks, Retrieval Lab,
  Answer (grounded answers + "How was this answer produced?" trace),
  Grounded Chat (three-pane: corpus / conversation / evidence), Evaluation, Experiments

*Verification: 511 backend tests, `npx tsc --noEmit` clean, `npm run build` clean (16 routes).*

## In Development / Planned

* Source-selection experiment v3 (infrastructure ready; **prepared, not executed**)
* Background/queued ingestion for very large uploads
* OCR for scanned PDFs
* CSV / XLSX ingestion (the parser registry already accepts them)
* Domain-aware chunking
* Answer-level evaluation + answer benchmark (retrieval answering is implemented; answer quality is unmeasured until ground truth exists)
* TurboVec vector-index backend (the factory is the extension point; Qdrant stays the default)
* Automated RAG optimization loop
* MCP integration
* Retrieval Lab V2 parameter controls; streaming answers; conversation history

---

# Research Vision

RAGForge aims to evolve from:

```text
"Build a vector database"
```

into:

```text
"Automatically engineer the best possible
knowledge base for a given domain."
```

The long-term objective is to create a system where:

```text
Domain
   ↓
Knowledge Requirements
   ↓
Source Discovery
   ↓
Source Selection
   ↓
Corpus Construction
   ↓
Chunking
   ↓
Embedding
   ↓
Indexing
   ↓
Retrieval
   ↓
Evaluation
   ↓
Diagnosis
   ↓
Optimization
   ↓
Improved Knowledge Base
```

This provides a framework for studying **automated, measurable, reproducible, and reusable RAG knowledge-base construction**.

---

# Project

**RAGForge — Automated Domain-Specific RAG Knowledge Base Engineering**

> Build the knowledge layer. Evaluate it. Improve it. Make it usable by AI.

---

## Corpus engineering (V5)

RAGForge is a **corpus-engineering platform**, not a chatbot with an upload box.
The product is the knowledge base; the job is to make that corpus trustworthy,
inspectable, reproducible and repairable.

Full documentation: **[docs/corpus-reliability.md](docs/corpus-reliability.md)**.

### Corpus Command Center

`/knowledge-bases/{id}/corpus` — bulk ingestion for 50–200 file corpora, a corpus
map, a read-only integrity scan, explicit repair, and corpus version fingerprints.

### What is guaranteed

| Guarantee | How |
|---|---|
| Every file is processed or clearly reported as failed | persistent per-file batch items with `error_code` / `error_message` |
| Nothing silently disappears | a rejected file stays a batch item with a reason; a failed parse keeps a `FAILED` document row **and** its bytes |
| Duplicates are detected | content-hash match against the corpus; the existing document is kept, never replaced silently |
| Documents can be replaced safely | superseded versions kept as history, old vectors removed |
| Indexing can resume after interruption | `POST .../ingestion-batches/{id}/resume` — idempotent by `UNIQUE(batch_id, item_key)` |
| Vectors stay consistent with chunk versions | vectors deleted *before* upsert, chunk rows written *after* |
| Provenance is preserved | page / slide / section / version / hash on every chunk and vector |
| Orphan and stale vectors are detectable | integrity checks B and C |
| The user can inspect what entered the KB | corpus manifest, per document and per corpus totals |
| Results trace back to the source | provenance on every retrieval hit |
| Corpus changes can be audited | corpus versions, fingerprints, document diffs |

### Corpus coverage vs retrieval quality

These are different questions and are never conflated:

- **Coverage** — *does the corpus contain the information?* (manifest, corpus map)
- **Retrieval quality** — *can retrieval find it?* (only measurable against ground truth)

> **A domain does not automatically have ground truth.** Entering a domain,
> analysing it, and indexing 200 documents produces **no** ground truth.
> `ground_truth_status` stays `NOT_AVAILABLE` until humans author, review, approve
> and freeze benchmark questions. No Recall@K / MRR / NDCG is reported until then.
> The Automobile Engineering benchmark is a hand-built research artifact.

### Integrity checks (read-only)

`missing_vectors` · `orphan_vectors` · `stale_vectors` · `embedding_mismatches` ·
`chunking_mismatches` · `duplicate_documents` · `duplicate_chunks` ·
`failed_documents` · `partial_documents` · `provenance_gaps` ·
`broken_source_references`

A scan **never** modifies anything. Repair is a separate, explicit operation, and
destructive repairs require `confirm_action`.

### Honesty conventions

Any count that could not be *measured* is reported as `unknown` / `-1`, never `0`.
This applies to vector totals, deletion counts and expected repair blast radius.

### Scale benchmark

```bash
cd backend && .venv/Scripts/python.exe scripts/corpus_benchmark.py
```

Measures the real pipeline at 10/50/100/200 documents using deterministic synthetic
fixtures and writes `backend/data/benchmarks/corpus-scale.json`. It measures
RAGForge's orchestration — **not** transformer inference, Qdrant throughput, or
retrieval quality.
