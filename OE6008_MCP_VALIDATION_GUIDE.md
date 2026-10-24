# RAGForge OE6008 MCP Intelligence Validation Guide

## Overview

This guide documents the end-to-end validation of RAGForge as an MCP (Model Context Protocol) knowledge and evidence retrieval layer for the OE6008 course. The validation proves that an external LLM (Claude Desktop) can use RAGForge's retrieval tools while performing final reasoning externally.

## Prerequisites

### 1. Repository State
- RAGForge repository at commit `Dev_1_midterm`
- OE6008 course documents in `OE6008/` directory (6 PDF files)
- Qdrant vector database running
- Backend API operational

### 2. System Requirements
- **Python 3.9+** with virtual environment
- **Qdrant** running at `http://127.0.0.1:6333`
- **Backend API** at `http://localhost:8000`
- **Claude Desktop** with MCP support
- **MCP SDK** for the external client

### 3. Protected Data
- **Automobile KB `kb_f278c283c748`** — 812 indexed points; must remain untouched
- All benchmark files must remain unchanged
- No unrelated KB modifications during testing

## Phase 1 — Repository Audit (Completed)

### Implemented Components (from code inspection)

| Component | Status |
|-----------|--------|
| MCP Server (V17, stdio) | ✅ Implemented in `backend/app/mcp/server.py` |
| MCP Bridge (`bridge.py`) | ✅ All tools call same services as HTTP API |
| `search_knowledge_base` | ✅ Returns EvidenceHit with full provenance |
| `get_evidence` | ✅ Fetches chunks by id or document |
| `answer_from_knowledge_base` | ✅ Grounded answers with citation validation |
| `evaluate_retrieval` | ✅ Lifecycle-gated evaluation on frozen benchmarks |
| `analyze_domain` | ✅ Domain analysis with mock fallback |
| `discover_sources` | ✅ Source discovery and quality scoring |
| `plan_knowledge_base` / `confirm_plan` / `create_knowledge_base` | ✅ Confirmed workflow |
| `ingest_sources` / `build_knowledge_base` | ✅ Document ingestion pipeline |
| Vector store (Qdrant) | ✅ Configured, needs operational instance |
| Embeddings (sentence-transformers) | ✅ Default: all-MiniLM-L6-v2 |
| Source quality engine | ✅ ACCEPT/REVIEW/REJECT with transparent scores |
| Chunking (section-aware) | ✅ 1200 char target, 150 char overlap |

### Unverified Components
1. End-to-end OE6008 KB creation with actual documents
2. Qdrant operational status and reachability
3. Claude Desktop MCP connection
4. Retrieval accuracy on OE6008 corpus
5. Answer correctness and citation traceability
6. Unsupported question handling

### Pre-existing Changes to Preserve
- Automobile KB `kb_f278c283c748` at 812 points
- All benchmark artifact files
- `.env` configuration variables
- Git working tree integrity

## Phase 2 — Create OE6008 Test Knowledge Base

### 2.1 Document Collection
The OE6008 folder contains 6 PDF documents:
- `OE6008_Exam_Quick_Revision.pdf`
- `OE6008_Final_10min_Handwritten_Notes.pdf`
- `OE6008_Q1 portions.pdf`
- `OE6008_Q1 portions_merged.pdf`
- `OE6008_Q1.1 portions.pdf`
- `OE6008_Short_Notes.pdf`

### 2.2 KB Creation Workflow

#### Step 2.2.1: Create Empty Knowledge Base
```http
POST /api/knowledge-bases
{
  "name": "OE6008",
  "domain": "Education",
  "purpose": "OE6008 Course Knowledge Base for student inquiries",
  "target_audience": "Students",
  "depth": "intermediate"
}
```

**Expected response:** New KB with ID, status `CREATED`, version 1

#### Step 2.2.2: Upload Documents
```http
POST /api/knowledge-bases/{kb_id}/documents/upload
```
- 6 PDF files, `index=false` (don't index yet)
- Each file validated for name safety, size, magic bytes, container integrity
- Duplicate detection: identical content reported, existing doc kept

**Expected response:** Upload result with:
- `uploaded`: count of successfully parsed documents
- `failed`: count of failed uploads
- Each file has status: `uploaded`, `failed`, or `duplicate`

#### Step 2.2.3: Index Documents
After upload, index all documents:
```http
POST /api/knowledge-bases/{kb_id}/index
```
Or via MCP: `build_knowledge_base` tool with confirmation

**Indexing pipeline:**
1. **Chunking** (section-aware): 1200 char target, 150 char overlap
2. **Embedding** (sentence-transformers/all-MiniLM-L6-v2): 384-dim vectors
3. **Vector store** (Qdrant): upsert points with provenance metadata

**Expected outcome:**
- KB status changes from `BUILDING` to `READY`
- `get_knowledge_base_status` reports `ready_for_retrieval: true`
- Vector points count ≈ chunk count
- Embedding identity recorded at index time

#### Step 2.2.4: Verify KB Readiness
```http
GET /api/knowledge-bases/{kb_id}/status
```
Or via MCP: `get_knowledge_base_status` tool

**Must see:**
- `status: READY`
- `chunk_count` > 0
- `chunk_count` ≈ `points` in vector store
- `embedding_identity` recorded (provider + model)
- `ready_for_retrieval: true`

#### Step 2.2.5: Verify Source Provenance
Check that source titles, document IDs and page/slide metadata survive:
```http
GET /api/knowledge-bases/{kb_id}/documents
```
Or via MCP: `list_knowledge_bases` → check individual KB summaries

**Must see for each document:**
- `title` = original PDF filename
- `source_title` or `document_title` preserved
- `page_count`, `slide_count`, or `section_count` recorded
- `parser` = "pdf" for all files
- `content_hash` = sha256 of original file bytes

### 2.3 Verification Criteria (Phase 2)

- [ ] All 6 PDFs uploaded successfully (or appropriate count with any failures documented)
- [ ] Chunk counts and document counts reported accurately
- [ ] Qdrant points match intended indexed corpus (point count ≈ chunk count)
- [ ] Source titles, document IDs and page/slide metadata survive ingestion and retrieval
- [ ] KB distinguishable from all other KBs (different KB ID, name, domain)
- [ ] KB readiness confirmed via `get_knowledge_base_status` and `validate_knowledge_base`

## Phase 3 — Establish the MCP Retrieval Contract

### 3.1 Required Fields in `search_knowledge_base` Results

The `search_knowledge_base` tool returns `SearchResponse` with `results: EvidenceHit[]`. Each `EvidenceHit` must include, where available:

| Field | Description |
|-------|-------------|
| `kb_id` | Knowledge base ID from `SearchResponse.kb_id` |
| `chunk_id` | Stable chunk identifier |
| `document_id` | Stable document identifier |
| `source_title` / `document_title` | Source filename or document title |
| `page` | Page number (if known from PDF) |
| `slide` | Slide number (if presentation) |
| `section` | Section heading (if structured) |
| `retrieval_score` | Score on strategy's own scale |
| `score_breakdown` | Per-pool scores for auditability |
| `why` | Why this chunk ranked here (as retriever recorded) |
| `text` | Chunk text (controlled by `include_text`) |
| `truncated` | Whether text was cut to size limit |
| `provenance` | Unmodified provenance subset |

### 3.2 Empty-Result Handling
- `result_count=0` with empty `results` array
- No fabricated entries
- Client can distinguish "no evidence" from "evidence found"

### 3.3 Verify `search_knowledge_base` Does Not Trigger Answer Generation
- Calling `search_knowledge_base` with any question only retrieves evidence chunks
- No answer generation occurs; external LLM performs final reasoning
- Retrieval result provides bounded set (top_k, default 5)

### 3.4 Current Status
The `search_knowledge_base` implementation already returns all required fields via `EvidenceHit`. No code changes needed to the retrieval contract. The work is **creating the OE6008 KB** so the contract has substance.

## Phase 4 — Connect and Test Claude Desktop

### 4.1 MCP Server Launch Configuration

| Parameter | Value |
|-----------|-------|
| **Python executable** | `.venv/Scripts/python.exe` (backend dir) |
| **Working directory** | `backend/` |
| **Launch command** | `.venv/Scripts/python.exe -m app.mcp` |
| **Transport** | `stdio` (default V17) |
| **Environment** | RAGForge settings from `.env` / `config.py` |

### 4.2 Claude Desktop Configuration

1. **Open Claude Desktop** → MCP Settings → "Add Server"
2. **Name:** "RAGForge OE6008"
3. **Type:** Stdio
4. **Command:** Full path to Python executable
5. **Arguments:** ` -m app.mcp` (or `backend\.venv\Scripts\python.exe -m app.mcp`)
6. **Working directory:** `backend/`

### 4.3 Tool Discovery Confirmation

After connection, Claude should discover these tools:
- `list_knowledge_bases` — list all KBs
- `get_knowledge_base_status` — check KB readiness
- `search_knowledge_base` — retrieve evidence
- `get_evidence` — fetch specific chunks
- `answer_from_knowledge_base` — grounded answers with citations
- `evaluate_retrieval` — run evaluation on frozen benchmark

### 4.4 Test Workflow in Claude

1. **List KBs:** Confirm OE6008 appears in the list
2. **Check status:** `get_knowledge_base_status` → `ready_for_retrieval: true`
3. **Search:** `search_knowledge_base` with OE6008-related queries
4. **Ask questions:** Natural language questions about OE6008 content
5. **Inspect evidence:** Which tools Claude calls, with what arguments
6. **Verify citations:** Final answer should cite specific chunks/documents/pages

### 4.5 If Claude Desktop Unavailable
Perform strongest possible official-SDK client test (Python MCP SDK) and label as **NOT VERIFIED**. Do not claim Claude Desktop interoperability without evidence.

## Phase 5 — MCP Intelligence Evaluation

### 5.1 Test Categories (against OE6008 KB)

| ID | Category | Success Criterion |
|----|----------|-------------------|
| **A** | Direct factual question | Claude retrieves passage, gives factually supported answer, cites correct source |
| **B** | Multi-document synthesis | Claude retrieves ≥2 sources, synthesizes correctly, distinguishes evidence from inference |
| **C** | Page-specific question | Retrieved evidence and citation identify correct source location; no fabricated page numbers |
| **D** | Reasoning or derivation | Claude performs reasoning itself, shows coherent explanation, grounds premises in retrieved passages |
| **E** | Unsupported question | System identifies course evidence insufficient; Claude labels external knowledge clearly if offered |
| **F** | Ambiguous question | Claude requests clarification or retrieves only when topic is clear |
| **G** | Conflicting evidence | Answer surfaces conflict and identifies both sources |
| **H** | KB isolation | OE6008 retrieval path does not leak/search unauthorized KBs |

### 5.2 Reference Set
Manually inspected questions from actual OE6008 material. Do not present unreviewed machine-generated benchmark as ground truth.

### 5.3 Evaluation Fields (per test)

For every test, record:
- Test ID and question
- Expected evidence (from actual source documents)
- MCP tool calls and their arguments
- Retrieved chunk/document IDs and source locations
- Final answer
- Citation correctness (does citation match claimed location?)
- Unsupported or fabricated claims
- Whether abstention or clarification was appropriate
- End-to-end latency
- Pass/fail and failure reason

## Phase 6 — Compare Three Distinct Paths

Run the same suitable questions through these paths:

### 6.1 Path 1: Direct Claude Question (No RAGForge)
- Claude answers from its own knowledge
- **Compare:** No retrieval, no citations from OE6008 sources

### 6.2 Path 2: Claude + RAGForge Retrieval
- Claude calls `search_knowledge_base` → receives evidence → generates final answer
- **Compare:** Factual correctness, evidence use, citation accuracy, abstention rate, latency

### 6.3 Path 3: RAGForge `answer_from_knowledge_base`
- RAGForge's internal pipeline generates answer with citations
- **Compare:** Same dimensions as Path 2, but with RAGForge-generated answer

### 6.4 Comparison Dimensions (fair)
- Factual correctness against verified course sources
- Correct use of evidence
- Citation and page/slide accuracy
- Unsupported-claim rate
- Appropriate abstention
- Retrieval and total response latency
- Retrieved evidence volume vs. full corpus size

**Keep fair:** Same questions and source material, record model/configuration differences, do not claim one approach is better based on anecdotes alone.

### 6.5 Do Not
- Fabricate scores or significance claims
- Present unreviewed machine-generated benchmark as ground truth
- Claim one approach is better based on small sample anecdotes

## Phase 7 — Privacy and Security Verification

### 7.1 What Is Sent Externally

| Item | Transmitted? |
|------|-------------|
| Selected evidence passages (top-k chunks) | ✅ Yes, necessary for retrieval |
| Necessary conversational context (question, chat history) | ✅ Yes |
| Full corpus to enable retrieval | ❌ No |
| Retrieval passage text to hosted LLM | ✅ Only what's needed |

### 7.2 Client Access Checks

- [ ] External clients cannot access unintended KBs (KB isolation via `kb_id` validation)
- [ ] Read-only retrieval tools do not perform mutations (all V17 tools have `_READ_ONLY` annotations)
- [ ] Write operations retain confirmation gates (`confirm_plan` → `create_knowledge_base`, etc.)
- [ ] Errors do not expose credentials or unnecessary document contents
- [ ] Logs do not unnecessarily store full sensitive prompts or passages

### 7.3 Security Boundaries

- **Local single-user setup:** Corpus reachable only by the local user (stdio transport)
- **Future remotely exposed service:** Would need auth, per-user scoping, token validation — not implemented in V17 stdio design

### 7.4 Do Not Claim
- Zero data sharing with hosted LLM: retrieved passages ARE transmitted when model answers

## Phase 8 — ChatGPT Compatibility Assessment

### 8.1 Transport Limitations (Documented)

| Limitation | Detail |
|------------|--------|
| V17 MCP server transport | stdio only |
| Streamable HTTP (V22) | Available via `app.mcp.remote`, not default |
| Second MCP server | Not implemented |
| Duplicate retrieval implementation | Avoided — reuses existing service layer |
| HTTPS adapter for ChatGPT | Not available in V17 |

### 8.2 If ChatGPT Integration Is a Goal

1. Inspect current official MCP and ChatGPT integration requirements
2. Implement remote MCP adapter only if needed, reusing existing service layer
3. Include: HTTPS, authentication, KB authorization, appropriate request limits
4. Test with real supported ChatGPT integration if available
5. Otherwise, document as **unverified**

**A local stdio smoke test does not prove remote ChatGPT compatibility.**

### 8.3 Do Not
- Build a second MCP server or duplicate retrieval implementation
- Claim ChatGPT compatibility without remote transport testing

## Phase 9 — Regression Checks and Final Report

### 9.1 Protected-Data Integrity Verification

- [ ] Automobile KB remains at 812 points (KB ID: `kb_f278c283c748`)
- [ ] Existing benchmark files remain unchanged
- [ ] No unrelated KB is modified
- [ ] No destructive operation occurs during read-only testing
- [ ] OE6008 test KB cleaned up only if safe and explicitly intended

### 9.2 Report Structure

Report these sections:

1. **What worked before changes** — Baseline functionality
2. **What changed and why** — Modifications made during experiment
3. **Actual commands executed and test results** — Detailed log of all operations
4. **Whether a real Claude Desktop session was tested** — Yes/No with evidence
5. **The tool-call and citation evidence from the end-to-end experiment** — Tool calls, arguments, returned evidence, final answers
6. **The evaluation table with actual results** — Table completing Phase 5 test categories A-H
7. **Privacy findings** — What data was transmitted, security observations
8. **Known limitations and remaining work** — Unexplored areas, open questions
9. **Changed files and Git status** — Diff of all modifications

**Clearly distinguish:** source-code inspection, automated tests, standalone MCP tests, and real external-client tests. Do not claim a test passed unless it was actually executed.

### 9.3 OE6008 KB Cleanup
Only clean up the OE6008 test KB if:
- Cleanup is safe (no downstream dependencies)
- Explicitly intended for the experiment
- Automobile KB and other KBs remain untouched

## Appendix A — OE6008 Document Summary

| File | Expected Content | Page Count (est.) |
|------|-----------------|-------------------|
| `OE6008_Exam_Quick_Revision.pdf` | Exam revision notes | ~10-15 |
| `OE6008_Final_10min_Handwritten_Notes.pdf` | Final exam condensed notes | ~5-10 |
| `OE6008_Q1 portions.pdf` | Question 1 material | ~15-20 |
| `OE6008_Q1 portions_merged.pdf` | Merged Q1 content | ~15-20 |
| `OE6008_Q1.1 portions.pdf` | Question 1.1 material | ~10-15 |
| `OE6008_Short_Notes.pdf` | Short reference notes | ~5-10 |

**Total estimated:** ~60-90 pages across 6 documents

## Appendix B — Known OE6008 Topics (from document names)

Based on file names, the OE6008 course likely covers:
- Exam preparation and revision
- Key concepts and formulas
- Question type 1 and 1.1 topics
- Final exam preparation
- Short reference summaries

Exact content must be verified after ingestion.

## Appendix C — MCP Tool Reference

### Core Retrieval Tools

| Tool | Purpose | Key Arguments |
|------|---------|---------------|
| `search_knowledge_base` | Ranked evidence retrieval | `kb_id`, `query`, `top_k`, `strategy`, `include_text` |
| `get_evidence` | Fetch specific chunks | `kb_id`, `chunk_ids` or `document_id`, `max_chunks` |
| `answer_from_knowledge_base` | Grounded answer with citations | `kb_id`, `question`, `answer_mode`, `strategy`, `top_k`, `unsupported_claim_action` |

### Agentic Tools

| Tool | Purpose | Key Arguments |
|------|---------|---------------|
| `analyze_domain` | Propose domain specification | `domain`, `purpose`, `target_audience`, `depth`, `kb_id` |
| `discover_sources` | Register candidate sources | `kb_id`, `provider`, `query`, `limit` |
| `assess_sources` | Explain source-quality decisions | `kb_id`, `source_ids`, `decision`, `user_provided_only` |
| `plan_knowledge_base` | Plan construction steps | `name`, `domain`, `purpose`, `target_audience`, `depth`, `source_mode`, `urls`, `user_documents`, `chunking`, `retrieval`, `index_documents` |
| `confirm_plan` | Record user confirmation | `plan_id`, `action`, `parameters_fingerprint`, `note` |
| `create_knowledge_base` | Create KB container | `confirmation_id`, `name`, `domain`, `purpose`, `target_audience`, `depth`, `source_mode` |
| `ingest_sources` | Download and parse sources | `confirmation_id`, `kb_id`, `source_ids` |
| `build_knowledge_base` | Chunk, embed, index | `confirmation_id`, `kb_id`, `chunker`, `target_size`, `overlap` |
| `add_user_documents` | Add user-supplied documents | `confirmation_id`, `kb_id`, `documents`, `index`, `chunker`, `target_size`, `overlap` |

### Supporting Tools

| Tool | Purpose | Key Arguments |
|------|---------|---------------|
| `get_knowledge_base_status` | KB build/index status | `kb_id`, `include_vectors`, `include_diagnostics` |
| `validate_knowledge_base` | Measure readiness | `kb_id`, `include_vectors`, `include_diagnostics` |
| `evaluate_retrieval` | Score against frozen benchmark | `kb_id`, `benchmark_version`, `strategy`, `top_k`, `label`, `dry_run`, `allow_keyword_fallback` |
| `get_operation_status` | Inspect operation state | `operation_id`, `kb_id` |
| `list_knowledge_bases` | List all KBs | None (read-only) |

### Resources (read-only)

| Resource | Path | Returns |
|----------|------|---------|
| `knowledge-bases` | `ragforge://knowledge-bases` | Index of all KBs |
| `knowledge-base-status` | `ragforge://knowledge-bases/{kb_id}` | Status of one KB |
| `document-metadata` | `ragforge://knowledge-bases/{kb_id}/documents/{document_id}` | Document metadata (no filesystem paths) |
| `evidence-chunk` | `ragforge://knowledge-bases/{kb_id}/chunks/{chunk_id}` | One chunk's text and provenance |

---

**This guide documents the complete workflow for RAGForge OE6008 MCP intelligence validation. 
Execution requires: Qdrant running, backend API operational, OE6008 KB created, and an MCP client (Claude Desktop or Python SDK) connected to the stdio server.**