# RAGForge Frontend UI Guide

This guide covers how to use the RAGForge web interface to create and manage knowledge bases, from document upload to question answering and evaluation.

## 📋 Table of Contents

1. [Creating a Knowledge Base](#creating-a-knowledge-base)
2. [Adding Documents](#adding-documents)
3. [Working with External Sources](#working-with-external-sources)
4. [Building and Indexing](#building-and-indexing)
5. [Retrieval and QA](#retrieval-and-qa)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)

---

## 🏗️ Creating a Knowledge Base

### Where to Go
- **URL**: `http://localhost:3000` (frontend) + `http://localhost:8000` (backend API)
- Start screen: **`/knowledge-bases/new`** — the guided KB creation wizard

### Step 1 — Domain Specification
| Field | Purpose | Example |
|---|---|---|
| **Name** | Human-readable KB identifier | `OE6008 Course Materials` |
| **Domain** | Academic/professional domain | `Engineering` or `OE6008` |
| **Purpose / use case** | Intended application | `Study assistant for OE6008 exam prep` |
| **Target audience** | Who this is for | `Graduate engineering students` |
| **Depth** | Complexity level | `intermediate` (beginner/expert options available) |

**Why these fields exist**: The domain, purpose, and target audience drive the domain analyzer's source recommendations, chunking strategy selection, and evaluation question generation. Depth influences chunk size and embedding model selection.

**After submitting**: You're routed to Step 2 (sources) with the KB ID in the URL (`/knowledge-bases/{kbId}`).

### Step 2 — Source Mode
Three modes are available:

| Mode | When to Use | What Happens |
|---|---|---|
| **`external`** | You want RAGForge to find public sources | After KB creation, you'll see a Sources page with discovery forms for URL submission and arXiv search. Quality scoring is automatic. |
| **`user_provided`** | You have your own documents (PDFs, notes, slides) | Upload files directly. No external discovery. Your files are parsed, chunked, and indexed locally. |
| **`mixed`** | Both your documents + external discovery | Upload files AND submit URLs. Best of both worlds. |

**Why these modes exist**: The source mode determines what provenance metadata appears on every chunk and answer. `user_provided` material is assessed for file integrity (not publication authority), while `external` sources receive authority/relevance scoring.

### Step 3 — Ingest Sources
After mode selection, you'll see a staging panel:

**File upload**:
- Drag & drop or browse for: `.pdf`, `.pptx`, `.ppt`, `.docx`, `.txt`, `.md`, `.markdown`, `.html`, `.htm`
- Max: `100MB` per file
- Files are parsed locally; duplicate content is detected by SHA-256 hash and skipped

**URL submission**:
- Paste one URL per line
- URLs are marked as `user_provided` (no authority penalty)
- After KB creation, you can re-discover and re-assess these sources

**Why the staging step exists**: Prevents partial KB state on failure. The KB container is created first, then sources are added. If ingestion fails, the KB remains in a valid (empty) state.

### Step 4 — Create & Index
- Click **Create Knowledge Base** — the KB container is persisted
- The **StagedUpload** panel appears showing upload progress
- After upload, chunks are created, embeddings generated, and vectors indexed in Qdrant
- **KB is READY** when `vector_store_status = "reachable"` and `chunks > 0`

**Why indexing is separate**: You can upload documents without immediately embedding them. This lets you:
- Review parsed text before committing to embeddings
- Re-index individual documents without re-processing the whole KB
- Control costs by batching embedding generation

---

## 📄 Adding Documents to an Existing KB

### Where to Go
- **URL**: `http://localhost:3000/knowledge-bases/{kbId}/documents`

### Upload New Documents
1. Click **Upload documents** button or drag & drop files
2. Supported formats: same as KB creation (PDF, PPTX, DOCX, TXT, MD, etc.)
3. Max 100MB per file

**What happens after upload**:
- Files are parsed locally (text extraction, table preservation)
- Duplicate detection by content hash — identical files skip creation
- Each document gets a `document_version` increment
- Chunks are created only for new/changed documents (incremental)
- If `index: true` is set, chunks are embedded and vectors updated immediately

**Why incremental indexing**: Adding one document never re-embeds the rest. This means:
- Large KBs can grow organically without full re-processing
- Individual document errors don't affect the whole corpus
- You can re-index a single problematic document

### Managing Existing Documents
On the document library page, each document has actions:

| Action | When to Use | Effect |
|---|---|---|
| **Inspect** | You want to see extracted text + chunks | Opens side panel with full text and chunk provenance (source, section, page, trust score) |
| **Re-index** | Chunks are stale or missing | Re-runs chunking + embedding for that document only |
| **Re-parse** | Text extraction failed or you want a do-over | Re-runs the parser on the stored original |
| **Remove** | You want to delete a document + its vectors | Deletes from library + issues vector store deletion. Qdrant confirmation is verified afterward. |

**Why "remove" has a confirmation**: Vector store deletion is best-effort. If Qdrant can't confirm the count, you'll get an honest warning + recommendation to re-index.

### Adding URLs to an Existing KB
- Go to the **Sources** page of your KB
- Use the "Add user URLs" form (bottom of the Sources page)
- Paste one URL per line
- These are treated as `user_provided` sources
- After submission, they appear in the source list with quality assessment
- You can then run ingestion to download and parse them

---

## 🌐 Working with External Sources

### Discovery
Navigate to **`/knowledge-bases/{kbId}/sources`** to access the source intelligence page.

**Discovery forms**:
- **Provider: `user-url`** — Paste URLs (one per line). These are marked as user-provided.
- **Provider: `arxiv`** — Search phrase (e.g., `"electric vehicle battery management"`). Queries the arXiv API.

**What happens after discovery**:
1. Candidate sources are registered as `Source` rows in the SQLite database
2. Each candidate receives an **explainable quality assessment** with:
   - Composite `trust_score` (0-100 scale, or `null` if unscored)
   - Signal breakdown: authority, relevance, recency, source type, accessibility, duplication, evidence quality
   - Reasons and warnings for each assessment (why it got that score)
   - Decision: `ACCEPT`, `REVIEW`, or `REJECT`
3. Sources are ranked by trust_score (high to low)

**Why explainable quality**: Every score has visible reasons. This is a core research integrity principle — you should never silently treat a random website as authoritative. The signals map is:
- **authority**: Domain reputation, publisher credibility
- **relevance**: Query-term match, topic modeling
- **recency**: Publication date freshness
- **source_type**: Peer-reviewed, technical report, blog, etc.
- **accessibility**: Whether full text is reachable
- **duplication**: Already in the KB?
- **evidence_quality**: Quality of supporting evidence

### Assessing & Deciding
For each source, you can set a decision:

| Decision | When to Use | Effect |
|---|---|---|
| **`ACCEPT`** | Quality is sufficient for ingestion | Source is marked accepted; can be ingested via the Sources page |
| **`REVIEW`** | You need to human-verify | Source stays in REVIEW state; must be decided before ingestion |
| **`REJECT`** | Quality is insufficient | Source is excluded from ingestion |

After deciding, you can ingest accepted sources (see below).

---

## ⚙️ Building and Indexing

### The Build Process
After sources are ingested (documents parsed), you need to **build** the KB before it's queryable.

**Where**: **`/knowledge-bases/{kbId}/processing`** or click the build button from the KB overview.

**Build parameters**:
- **Chunker** (default: `section-aware`): How text is split into chunks
  - `section-aware`: Preserves headings/section metadata (recommended)
  - `fixed-size`: Fixed chunk size with overlap
  - `semantic`: Domain-aware semantic splitting (experimental)
- **Target size** (default: `1200` tokens): Target chunk size
- **Overlap** (default: `150 tokens`): Overlap between adjacent chunks

**What the build does** (in order):
1. **Chunking**: Text split into chunks with preserved metadata (sections, pages, slides)
2. **Embedding**: Each chunk gets a vector via the configured embedding model (SentenceTransformers local by default)
3. **Indexing**: Vectors + provenance stored in Qdrant vector store
4. **KB status**: Updated from `BUILDING` to `READY` (or `FAILED`)

**Why the multi-stage build**: Each stage can fail independently, and you get granular diagnostics:
- If chunking fails: text format issues, unsupported content
- If embedding fails: model mismatch, API key problems
- If indexing fails: Qdrant connection, schema conflicts

### After Build — KB Readiness
A KB is **ready for retrieval** when ALL of these are true:
- `chunk_count > 0` (documents have been chunked)
- `vector_points` matches `chunk_count` (or discrepancies are noted)
- `embedding_identity` matches the configured provider/model
- No stale vector warnings

You can check this at **`/knowledge-bases/{kbId}`** (overview page) or via **`get_knowledge_base_status`** MCP tool.

---

## 🔍 Retrieval and QA

### Searching Your KB
**URL**: `http://localhost:3000/knowledge-bases/{kbId}/retrieval` or use the MCP tool `search_knowledge_base`.

**Parameters**:
- **Query**: Your natural-language question
- **Strategy** (optional): 
  - `dense` (default): Vector similarity search
  - `bm25`: Keyword/lexical search
  - `hybrid`: Combined dense + BM25 (if configured)
  - `dense_reranked`: With cross-encoder reranking (if model available)
  - `bm25_reranked`: With reranking
- **top_k**: Number of results to return (default: 5-10)
- **params**: Override top_k or other strategy-specific params

**Results show**:
- Ranked chunks with provenance (source URL, title, section, trust score)
- Retrieval method + score breakdown
- Embedding model used
- Any warnings (embedding mismatch, index/corpus disagreement)

### Grounded Answers
**URL**: `http://localhost:3000/knowledge-bases/{kbId}/answer` (via API) or use the MCP tool `answer_from_knowledge_base`.

**Parameters**:
- **Question**: Your question
- **mode** (`answer_mode`): 
  - `grounded`: Answer must be supported by evidence (may abstain if insufficient)
  - `abstain_if_unsupported`: Always abstain if can't ground (default)
- **strategy**: Retrieval strategy (default: configured KB strategy)
- **top_k**: How many chunks to retrieve (default: KB config)
- **unsupported_claim_action**: What to do with unsupported claims
  - `remove`: Remove unsupported claims from answer
  - `downgrade`: Weaken claim support status
  - `abstain`: Abstain from answering entirely

**Answer result includes**:
- `status`: `grounded`, `partial`, `abstained`, or `generation_failed`
- `answer_text`: The synthesized answer (or empty if abstained)
- `citations`: List of evidence citations with chunk IDs, document IDs, source info
- `claims`: Structured claims with support status (supported/partially-supported/unsupported)
- `grounding_state`: Five-state outcome (ANSWERED/PARTIALLY_SUPPORTED/INSUFFICIENT_EVIDENCE/CONFLICTING_EVIDENCE/NO_RELEVANT_EVIDENCE)
- `unsupported_aspects`: What couldn't be grounded
- `missing_information`: What's needed for a complete answer

**Why grounded answers matter**: The system never hallucinates. If the KB doesn't contain relevant information, the answer will either:
- Say it cannot answer (abstain)
- Give a partial answer with clear marking
- Cite the specific evidence that supports each claim

### Chat Interface
**URL**: `http://localhost:3000/knowledge-bases/{kbId}/chat` — a full conversational interface.

Features:
- Conversation history preserved
- Each message gets a grounded assessment
- Retrieval + generation in one turn
- Conversation summaries and export
- Answer run tracking (latencies, models used, etc.)

---

## 📊 Evaluation

### Adding Evaluation Questions
**URL**: `http://localhost:3000/knowledge-bases/{kbId}/evaluation-questions` or use the API `addQuestion`.

**Question schema**:
- `question`: The natural-language question
- `expected_chunk_ids`: Chunks that contain the answer (optional)
- `expected_document_ids`: Documents that contain the answer (optional)
- `expected_keywords`: Keywords for lexical matching (optional)
- `notes`: Free text

**Why questions are separate from documents**: Evaluation questions provide ground truth for metrics computation. Without them, the system can compute retrieval metrics (recall, precision) but not answer correctness.

### Creating a Benchmark Version
1. Add questions via the UI or API
2. Set question lifecycle status: `DRAFT` → `REVIEW` → `APPROVED` → `FROZEN`
3. **Freeze** the benchmark version (only FROZEN versions can be evaluated officially)

### Running Evaluation
**URL**: `http://localhost:3000/knowledge-bases/{kbId}/evaluate` or API `evaluate`.

**Parameters**:
- `top_k`: Retrieval cutoff (default: KB config)
- `label`: Free-text label for this run (e.g., `baseline-v1`, `with-rerank`)
- `allow_keyword_fallback`: Allow BM25 when dense retrieval fails (diagnostic only)
- `benchmark_version`: Which frozen version to evaluate (optional)

**Metrics reported**:
- **Retrieval metrics**: Recall@K, Precision@K, MRR, nDCG@K
- **Question-level**: Per-question recall, precision, MRR, nDCG
- **Skipped/fallback**: Questions with no ground truth, scored by keyword fallback
- **Status**: Questions skipped vs. scored

**Why evaluation is gated**: Only FROZEN benchmark versions may be evaluated. This prevents accidental use of draft/DRAFT questions and ensures metric stability for comparison.

### Evaluation Results Page
- Aggregate metrics at the top
- Per-question breakdown table
- Filter by question status ( answered, skipped, fallback)
- Export/download options
- Comparison between multiple runs (via `/compare` endpoint)

---

## 🔧 Troubleshooting

### Common Issues

| Symptom | Likely Cause | Fix |
|---|---|---|
| **KB shows 0 chunks** | Documents uploaded but not indexed | Go to processing page and run build |
| **Retrieval returns no results** | Embedding mismatch (provider/model changed) | Check KB overview → embedding identity section. Re-index if model changed. |
| **Vector count mismatch** | Build succeeded but Qdrant count differs | Use `validate_knowledge_base` from the processing page, or re-index |
| **All answers abstain** | No relevant chunks in KB, or embedding/config mismatch | Check KB status; ensure build completed successfully. Use `search_knowledge_base` to verify retrieval works first. |
| **Source quality scores look wrong** | For user-provided files: assessed for integrity, not authority. For external: check signal values. | Review individual source assessments on the Sources page. |
| **Upload fails (413 or timeout)** | File > 100MB | Compress or split the document. Max is 100MB per file. |
| **arXiv search returns nothing** | External service unreachable or query too specific | Try a broader query. arXiv availability varies by day/time. |
| **MCP server won't start** | Port conflict or venv not activated | Ensure `.venv\Scripts\python.exe` is in PATH. Try `cd backend && .venv\Scripts\python.exe -m app.mcp` |

### When to Contact Support / Escalate
- Qdrant is unreachable and restart doesn't help
- Embedding model consistently fails (likely API key or model name mismatch)
- Benchmark lifecycle gets stuck (DRAFT won't progress to REVIEW)
- Corrupt data detected by integrity checks (run `corpus integrity` API endpoint)

---

## 📚 Feature Rationale Summary

| Feature | Why It Exists |
|---|---|
| **Domain-aware chunking** | Preserves section headings and structure for multi-hop reasoning |
| **Explainable source quality** | Research integrity: never treat random websites as authoritative without visibility into why |
| **Incremental indexing** | Large course corpora can't be re-embedding on every addition |
| **Five-state grounding** | Honest communication: partially supported/insufficient evidence is better than fake certainty |
| **Benchmark lifecycle (DRAFT→REVIEW→APPROVED→FROZEN)** | Prevents accidental use of unvalidated questions; enables official metric reporting |
| **MCP stdio server** | Local single-user boundary: no auth, no network exposure, client controls the corpus |
| **User-provided vs. external source distinction** | Different quality signals: integrity vs. authority; different decision workflows |
| **Grounded answers over free-form generation** | Core research principle: every claim must be traceable to source provenance |

---

## 🚀 Quick Start Checklist

For a new user (like your OE6008 integration):

- [ ] Start frontend: `cd frontend && npm run dev` (port 3000)
- [ ] Start backend: `cd backend && .venv\Scripts\python.exe -m uvicorn app.main:app --reload` (port 8000)
- [ ] Start Qdrant (Docker: `docker run -p 6333:6333 qdrant/qdrant`)
- [ ] Create KB: domain = `OE6008`, purpose = `Course study assistant`, target audience = `OE6008 students`
- [ ] Upload your OE6008 PDFs/notes via the "Upload documents" flow
- [ ] Click **Build** when upload completes (chunk + embed + index)
- [ ] Verify KB status shows `READY` and `chunk_count > 0`
- [ ] Go to `/knowledge-bases/{kbId}/retrieval` and test a query
- [ ] Go to `/knowledge-bases/{kbId}/answer` and ask a question
- [ ] (Optional) Add evaluation questions and freeze a benchmark version
- [ ] Connect Claude Desktop MCP to `.venv\Scripts\python.exe -m app.mcp`

---

*Guide generated from codebase inspection as of October 2026. For the latest state, check the RAGForge dashboard at http://localhost:3000.*