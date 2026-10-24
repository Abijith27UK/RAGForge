# RAGForge MCP Intelligence End-to-End Validation — Audit Report

## Phase 1 — Repository audit and safety baseline

### What is implemented

**MCP Server (V17, stdio transport)**
- `backend/app/mcp/server.py` — complete MCP server over stdio
- All read-only tools operational: `list_knowledge_bases`, `get_knowledge_base_status`, `search_knowledge_base`, `get_evidence`, `answer_from_knowledge_base`, `evaluate_retrieval`
- Agentic operations: `analyze_domain`, `discover_sources`, `assess_sources`, `plan_knowledge_base`, `confirm_plan`, `create_knowledge_base`, `ingest_sources`, `build_knowledge_base`, `add_user_documents`
- Resources: `knowledge-bases`, `knowledge-base-status`, `document-metadata`, `evidence-chunk`
- Token/V22 remote transport: stdio default, Streamable HTTP optional via `app.mcp.remote`
- Logging: `configure_mcp_logging` routes to stderr, protocol bytes on stdout
- Error translation: `_tool_call` / `_resource_call` map exceptions to `ToolError`/`ResourceError`

**Bridge (V17, backend/app/mcp/bridge.py)**
- Single source of truth: every MCP tool calls the same services the HTTP API uses
- No parallel retrieval path, no second scoring scheme, no strategy fallback
- Wiring: `repo` (SQLite), `settings` (Pydantic Settings), `_kb` (lookup), `_store` (vector factory), `_embedding_config`, `_generator_status`, `_benchmark_versions`
- Tool implementations: `_hit` (EvidenceHit construction), `_chunk`, `_document`, `_claim`, `_citation`, `_evidence_from_answer`
- Dependency checks: vector store reachability, embedding identity verification, generator availability

**Key services wired through the bridge**
- Retrieval: `app.services.retrieval.service.RetrievalService` (registry, KB config, recorded run)
- Answering: `app.services.answering.service.AnsweringService` (query plan, evidence gate, generator, citation validation)
- Evaluation: `app.services.evaluation.evaluator.Evaluator` (frozen benchmark gate, real metrics)
- Storage: `app.repositories.sqlite_repo.Repository` (chunks, documents, build runs, benchmark versions)
- Vector store: `app.services.vector_store.factory` (Qdrant backend)

**Retrieval (search_knowledge_base)**
- Strategy options: `dense`, `bm25`, `hybrid`, `hybrid_reranked`
- Parameters: `top_k`, `strategy`, `params` (overrides), `include_text`
- Returns: `SearchResponse` with `results` (EvidenceHit[]), `trace` (RetrievalTrace)
- EvidenceHit includes: chunk_id, document_id, source_title, page/slide, domain, subdomain, trust_score, content_hash, retrieval_score, score_breakdown, provenance
- Empty-result handling: `result_count=0` with empty `results` array

**Answer generation (answer_from_knowledge_base)**
- Modes: `abstain_if_unsupported` (default), `grounded`
- Citation validation: `unsupported_claim_action` = `remove`/`downgrade`/`abstain`
- Returns: `GroundedAnswerResult` with `status` (ANSWERED/PARTIALLY_SUPPORTED/INSUFFICIENT_EVIDENCE/CONFLICTING_EVIDENCE/NO_RELEVANT_EVIDENCE), `claims`, `citations`, `evidence`
- Explicit abstention when evidence insufficient — no model call in that case

**Evaluation (evaluate_retrieval)**
- Lifecycle: DRAFT → REVIEW → APPROVED → FROZEN
- Only FROZEN benchmark versions may be scored
- Metrics: Recall@K, Precision@K, MRR, NDCG (where applicable)
- Questions with keyword fallback flagged diagnostic-only; excluded from headline metrics
- Questions with no ground truth skipped rather than scored zero

**Chunking (section-aware baseline)**
- Configurable target size (200-8000 chars) and overlap (0-1000 chars)
- Default: 1200 chars target, 150 chars overlap
- Preserves headings/section metadata

**Embeddings**
- Default: `sentence-transformers/all-MiniLM-L6-v2` via `sentence-transformers` provider
- Configurable: `openai` provider also supported
- Identity recorded at index time, verified at retrieval

**Source quality**
- Signals: authority, relevance, recency, source type, accessibility, duplication, evidence quality
- Decisions: ACCEPT / REVIEW / REJECT
- Transparent scores/reasons/warnings

### What is unverified

1. **End-to-end OE6008 KB creation** — No test KB has been created with actual OE6008 course documents yet. The documents exist in `OE6008/` (6 PDFs) but have not been uploaded/indexed.

2. **Qdrant operational status** — Unclear whether Qdrant is running and reachable at `http://127.0.0.1:6333`. The config references this URL but has not been confirmed.

3. **Claude Desktop MCP connection** — No test has been performed connecting an external LLM client to the stdio MCP server.

4. **Retrieval accuracy on OE6008 corpus** — No known-passage retrieval test has been run against the OE6008 documents.

5. **Answer correctness** — No questions have been asked and verified against the OE6008 KB.

6. **MCP tool-call semantics in external client** — Which tools Claude Desktop actually calls, and with what arguments, is unverified.

7. **Citation traceability** — Whether citations in answers correctly map to the claimed source locations has not been verified.

8. **Unsupported question handling** — Whether questions with answers absent from the corpus are correctly handled as unsupported has not been tested.

### Pre-existing changes to preserve

- **Automobile KB `kb_f278c283c748`** — 812 indexed points; must not be modified, re-indexed, or deleted. This KB is the protected corpus referenced in the architecture docs and must remain untouched.
- **Benchmark files** — All existing benchmark artifacts (automobile-engineering-baseline-v1.json and any held-out versions) must remain unchanged.
- **`.env` configuration** — Any operator-set environment variables must be preserved.
- **Git working tree** — No commits, pushes, or resets. All pre-existing modifications must be retained.

### Smallest set of changes required

1. **Create an OE6008 knowledge base** — Upload the 6 OE6008 PDF documents through the API or MCP, run chunking/embedding/indexing, and verify the KB is READY.

2. **Verify Qdrant is reachable** — Start Qdrant if not already running, confirm the vector store probe in `get_knowledge_base_status` succeeds.

3. **Run a basic retrieval test** — Use `search_knowledge_base` with a known query from the OE6008 material and verify evidence hits with provenance.

4. **Test answer_from_knowledge_base** — Ask a factual question whose answer is in the OE6008 documents and verify grounded answer with correct citation.

5. **Test unsupported question handling** — Ask a question whose answer is absent from OE6008 and verify the pipeline reports `INSUFFICIENT_EVIDENCE` / abstains rather than fabricating.

6. **Test MCP tool discovery** — Run the stdio MCP server and verify an external client can connect and discover the tools.

7. **Compare paths (Phase 6)** — Run the same questions through: (a) direct Claude, (b) Claude + RAGForge retrieval, (c) `answer_from_knowledge_base`.

### Safety boundary

- The OE6008 test KB is a **dedicated test KB**, never using the protected Automobile KB.
- All operations on the OE6008 KB are read-only or confirmational; nothing destroys user data.
- The Automobile KB's 812 points are protected by: (a) separate KB ID, (b) Git-level protection (no commits that touch it), (c) explicit audit requirement.

---

## Phase 2 — Create a controlled OE6008 test knowledge base

### Documents available

The OE6008 folder contains 6 PDFs:
- `OE6008_Exam_Quick_Revision.pdf`
- `OE6008_Final_10min_Handwritten_Notes.pdf`
- `OE6008_Q1 portions.pdf`
- `OE6008_Q1 portions_merged.pdf`
- `OE6008_Q1.1 portions.pdf`
- `OE6008_Short_Notes.pdf`

### Test KB creation workflow

**Step 2.1:** Create an empty knowledge base named "OE6008" with domain "Education" / "OE6008 Course", purpose "Course knowledge base for OE6008", target audience "Students", depth "intermediate".

**Step 2.2:** Upload all 6 OE6008 PDF documents using `POST /api/knowledge-bases/{kb_id}/documents/upload`. Each file is validated for name safety, size, magic bytes, and container integrity. Duplicates are reported but not silently replaced.

**Step 2.3:** After upload, index the documents — chunking (section-aware, 1200 char target, 150 char overlap), embedding (sentence-transformers/all-MiniLM-L6-v2), and vector store index into Qdrant.

**Step 2.4:** Validate the KB is READY via `get_knowledge_base_status` and `validate_knowledge_base`.

**Step 2.5:** Run basic retrieval tests: search for known terms, verify provenance (page/slide, document ID, source title) survives ingestion and indexing.

### Verification criteria (Phase 2)

- [ ] Uploaded documents are parsed successfully (no parse errors, reasonable `text_length`)
- [ ] Chunk counts and document counts are reported accurately by `list_knowledge_bases` and `get_knowledge_base_status`
- [ ] Qdrant points match the intended indexed corpus (point count ≈ chunk count)
- [ ] Source titles, document IDs and page/slide metadata survive ingestion and retrieval
- [ ] A known passage can be retrieved using a specific query (e.g., "What is the definition of X from page Y?")
- [ ] The test KB is distinguishable from all other KBs (different KB ID, name, domain)
- [ ] A question whose answer is absent from the documents is correctly recognized as unsupported

---

## Phase 3 — Establish the MCP retrieval contract

### Required fields in `search_knowledge_base` results

The `search_knowledge_base` tool must return evidence with, where available:

- [x] **Knowledge-base ID** — `kb_id` in `SearchResponse`
- [x] **Stable chunk and document IDs** — `chunk_id`, `document_id` in each `EvidenceHit`
- [x] **Source title or filename** — `source_title` or `document_title` in each `EvidenceHit`
- [x] **Page, slide or section provenance** — `page`, `slide`, `section` in each `EvidenceHit`
- [x] **Passage text** — `text` in each `EvidenceHit` (controlled by `include_text`)
- [x] **Ranking/relevance information** — `retrieval_score`, `score_breakdown`, `why` in each `EvidenceHit`
- [x] **Empty-result or insufficient-evidence indication** — `result_count=0` with empty `results` array

### Verify `search_knowledge_base` does not trigger `answer_from_knowledge_base`

- [ ] Calling `search_knowledge_base` with a question only retrieves evidence chunks
- [ ] No answer generation occurs; the external LLM performs final reasoning
- [ ] The retrieval result provides a bounded set of relevant passages (top_k, default 5)

### Current contract — already satisfied

The `search_knowledge_base` implementation in `bridge.py` already returns all required fields via `EvidenceHit`. No changes needed to the retrieval contract itself. The main work is **creating the OE6008 KB with actual documents** so the contract has substance to operate on.

---

## Phase 4 — Connect and test Claude Desktop

### MCP server launch configuration

**Python executable:** `.venv/Scripts/python.exe` (or `.venv/bin/python` on Linux/macOS)
**Working directory:** `backend/`
**Launch command:** `.venv/Scripts/python.exe -m app.mcp` (starts stdio server)
**Environment:** RAGForge settings from `.env` / `config.py` (QDRANT_URL, EMBEDDING_PROVIDER, etc.)

### If Claude Desktop is available:

1. Configure the RAGForge MCP server in Claude Desktop's MCP settings, pointing at the stdio command
2. Restart/reload the client as required
3. Confirm the server connects (handshake succeeds)
4. Confirm the expected tools are discovered: `list_knowledge_bases`, `search_knowledge_base`, `get_evidence`, `answer_from_knowledge_base`
5. Ask questions in a real Claude conversation, recording which tools Claude actually calls and the arguments used
6. Inspect returned evidence and compare the final response against it
7. Verify Claude calls `search_knowledge_base` → retrieves evidence → reasons over it, rather than calling `answer_from_knowledge_base` internally

### If Claude Desktop is unavailable:

Perform the strongest possible official-SDK client test (e.g., Python MCP SDK client) and clearly label the real-host test as **NOT VERIFIED**. Do not claim Claude Desktop interoperability without evidence.

---

## Phase 5 — MCP intelligence evaluation

### Test categories (run against OE6008 KB)

| ID | Category | Description |
|----|----------|-------------|
| A | Direct factual question | Answer explicitly stated in one passage |
| B | Multi-document synthesis | Combine evidence from ≥2 documents |
| C | Page-specific question | Definition/equation on known page/slide |
| D | Reasoning or derivation | Course question requiring explanation based on material |
| E | Unsupported question | Fact absent from test corpus |
| F | Ambiguous question | Underspecified question |
| G | Conflicting evidence | Disputed point in corpus |
| H | KB isolation | Answer exists only in another KB, not OE6008 |

For each test, record:
- Test ID and question
- Expected evidence (from actual source documents)
- MCP tool calls and arguments
- Retrieved chunk/document IDs and source locations
- Final answer
- Citation correctness
- Unsupported or fabricated claims
- Whether abstention or clarification was appropriate
- End-to-end latency
- Pass/fail and failure reason

### Reference set

Use a manually inspected reference set for the initial experiment. Do not present an unreviewed machine-generated benchmark as ground truth.

---

## Phase 6 — Compare three distinct paths

Run the same suitable questions through these paths:

1. **Direct Claude question without RAGForge retrieval** — Claude answers from its own knowledge
2. **Claude using RAGForge retrieval-only MCP tools** — Claude calls `search_knowledge_base`, receives evidence, generates the final answer
3. **RAGForge's `answer_from_knowledge_base` tool** — RAGForge's internal pipeline generates the answer with citations

**Comparison dimensions:**
- Factual correctness against verified course sources
- Correct use of evidence
- Citation and page/slide accuracy
- Unsupported-claim rate
- Appropriate abstention
- Retrieval and total response latency
- Retrieved evidence volume vs. full corpus size

**Keep fair:** Same questions and source material, record model/configuration differences, do not claim one approach is better based on anecdotes alone.

---

## Phase 7 — Privacy and security verification

### What is sent externally

- [ ] Only selected evidence passages (top-k chunks) are sent to the hosted LLM
- [ ] Necessary conversational context (question, chat history) is transmitted
- [ ] The full corpus is NOT sent to the hosted LLM merely to enable retrieval
- [ ] Retrieved passage text is the only document-level data transmitted

### Client access checks

- [ ] External clients cannot access unintended KBs (KB isolation via `kb_id` validation)
- [ ] Read-only retrieval tools do not perform mutations (all V17 tools have `_READ_ONLY` annotations)
- [ ] Write operations retain confirmation gates (`confirm_plan` → `create_knowledge_base`, etc.)
- [ ] Errors do not expose credentials or unnecessary document contents
- [ ] Logs do not unnecessarily store full sensitive prompts or passages

### Security boundaries

- Local single-user setup: corpus reachable only by the local user
- Future remotely exposed service: would need auth, per-user scoping, token validation — not implemented in V17 stdio design

---

## Phase 8 — ChatGPT compatibility assessment

### Transport limitations (documented)

- V17 MCP server uses **stdio** transport only
- Streamable HTTP (V22) is available as an optional remote transport via `app.mcp.remote`
- No second MCP server or duplicate retrieval implementation
- No HTTPS adapter for ChatGPT integration

### If ChatGPT integration is a goal:

- Inspect current official MCP and ChatGPT integration requirements
- Implement a remote MCP adapter only if needed, reusing the existing service layer
- Include HTTPS, authentication, KB authorization and appropriate request limits
- Test with a real supported ChatGPT integration if access is available
- Otherwise, document as unverified

**A local stdio smoke test does not prove remote ChatGPT compatibility.**

---

## Phase 9 — Regression checks and final report

### Protected-data integrity verification

- [ ] Automobile KB remains at 812 points (KB ID: `kb_f278c283c748`)
- [ ] Existing benchmark files remain unchanged
- [ ] No unrelated KB is modified
- [ ] No destructive operation occurs during read-only testing
- [ ] OE6008 test KB cleaned up only if cleanup is safe and explicitly intended

### Report structure

1. What worked before changes
2. What changed and why
3. Actual commands executed and test results
4. Whether a real Claude Desktop session was tested
5. The tool-call and citation evidence from the end-to-end experiment
6. The evaluation table with actual results
7. Privacy findings
8. Known limitations and remaining work
9. Changed files and Git status

**Clearly distinguish:** source-code inspection, automated tests, standalone MCP tests, and real external-client tests. Do not claim a test passed unless it was actually executed.

### Files created/changed (to be filled after experiment)