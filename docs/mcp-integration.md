# RAGForge MCP integration (V17 read surface + V18 agentic operations)

**What this adds.** A working Model Context Protocol server that exposes RAGForge
to any MCP-compatible AI client (Claude Desktop, Cursor, VS Code MCP clients, the
`mcp` CLI, custom agents):

* **V17 — reading** (6 read-only tools): discover knowledge bases, inspect their
  real index state, retrieve evidence, get grounded answers with citations, and
  run official retrieval evaluations;
* **V18 — constructing** (12 agentic tools): propose a domain specification,
  discover and explain source-quality decisions, plan a corpus, then — only after
  an explicit, parameter-bound user confirmation — create the knowledge base,
  ingest approved sources, add user-provided documents, chunk/embed/index, and
  validate readiness from measured evidence. See
  [`docs/v18-agentic-operations.md`](v18-agentic-operations.md) for the operation
  lifecycle, the confirmation gates and the demonstrated runs.

**What it is not.** It is not a second retrieval system, a second answer
pipeline, a second orchestrator or an agent framework. There is no MCP code path
that ranks, scores, generates, parses or validates anything itself:
`app/mcp/bridge.py` and `app/mcp/agent_ops.py` call `RetrievalService`,
`AnsweringService`, `Evaluator`, the ingestion/document/build pipelines and
`Repository` — the exact services behind `/api/knowledge-bases/...`. If a
behaviour is not in RAGForge, it is not in the MCP surface either.

---

## 1. Architecture

```
 MCP client (Claude Desktop, Cursor, custom agent)
        │  JSON-RPC over stdio (stdout)
        ▼
 backend/scripts/run_mcp_server.py      ← absolute-path launcher
        │
 backend/app/mcp/server.py              ← MCPServer: tools, resources, stdio
        │
 backend/app/mcp/bridge.py              ← validation, limits, error vocabulary
 backend/app/mcp/planning.py            ← V18: what WOULD happen, fingerprints
 backend/app/mcp/operations.py          ← V18: operation rows, confirmation gate
 backend/app/mcp/agent_ops.py           ← V18: the execution half of the bridge
        │
 backend/app/services/…                 ← the EXISTING services
   retrieval.service.RetrievalService   (registry, persisted config, run record)
   answering.service.AnsweringService   (query plan → gate → generator → validation)
   evaluation.evaluator.Evaluator       (FROZEN-version gate, real metrics)
   ingestion / documents / build         (V18: parse, intake, chunk, embed, index)
   repositories.sqlite_repo.Repository  (chunks, documents, versions, runs, operations)
        │
 SQLite (backend/data/ragforge.db) + Qdrant (vectors)
```

| Module | Responsibility |
|---|---|
| [`app/mcp/server.py`](../backend/app/mcp/server.py) | MCP wiring only: 18 tools (6 read-only + 12 agentic), 4 read-only resources, stdio transport, stderr logging |
| [`app/mcp/bridge.py`](../backend/app/mcp/bridge.py) | The only RAGForge-facing code: argument validation, resource lookups, service calls, result shaping |
| [`app/mcp/planning.py`](../backend/app/mcp/planning.py) | V18 plans: exact action parameters, fingerprints, URL/document/chunker/strategy validation |
| [`app/mcp/operations.py`](../backend/app/mcp/operations.py) | V18 lifecycle: operation rows, confirmation issuance/validation/consumption/expiry, audit reads |
| [`app/mcp/agent_ops.py`](../backend/app/mcp/agent_ops.py) | V18 execution: creation, discovery, ingestion, document intake, build, validation, comparison |
| [`app/mcp/schemas.py`](../backend/app/mcp/schemas.py) | The structured payloads a client receives (measured-or-UNKNOWN metrics, provenance copied verbatim) |
| [`app/mcp/limits.py`](../backend/app/mcp/limits.py) | Input/result bounds for untrusted callers |
| [`app/mcp/errors.py`](../backend/app/mcp/errors.py) | Error codes whose messages are safe to show a client |
| [`scripts/run_mcp_server.py`](../backend/scripts/run_mcp_server.py) | Launcher that works from any cwd (clients pass absolute paths, no cwd) |
| [`scripts/mcp_smoke_client.py`](../backend/scripts/mcp_smoke_client.py) | A real client that runs the documented read flow for manual checks |
| [`scripts/mcp_agent_workflow.py`](../backend/scripts/mcp_agent_workflow.py) | V18: the live end-to-end construction workflow against the real workspace |

**Transport: stdio only, deliberately.** RAGForge is a local, single-user
application with no authentication, no accounts and no tenant isolation. A stdio
server is launched *by* the client process, so it inherits exactly that boundary.
A Streamable-HTTP listener, by contrast, would expose every knowledge base on the
machine to anything that can reach the port, with no auth layer to secure it —
so HTTP transport is **not implemented** in this milestone rather than shipped
unauthenticated. (`MCPServer.run_streamable_http_async` exists in the SDK; the
documents and tests here do not claim it works.)

**Separation from the API.** The FastAPI app never imports `app.mcp`, and the MCP
process starts no web server: both can run at once, on different ports or none.
The MCP server reads the same SQLite database and Qdrant collections, so a
knowledge base built in the UI is immediately visible to MCP clients.

---

## 2. Install and run

The MCP SDK is an **optional** dependency: the API and the UI do not need it.

```bash
cd backend
.venv\Scripts\python.exe -m pip install -r requirements-mcp.txt
```

Run the server by hand (it then waits for JSON-RPC on stdin):

```bash
backend\.venv\Scripts\python.exe -m app.mcp
# or, equivalently, from any working directory:
backend\.venv\Scripts\python.exe backend\scripts\run_mcp_server.py
```

Verify with the bundled client (starts the server itself and walks the flow):

```bash
backend\.venv\Scripts\python.exe backend\scripts\mcp_smoke_client.py --kb kb_f278c283c748 --query "What is an engine?"
```

**Logging rule.** stdout is the protocol channel, so the MCP entry point installs
a **stderr** handler (`configure_mcp_logging`) and never calls
`app.utils.logging_config.setup_logging`, which writes to stdout. Nothing else in
the MCP path prints to stdout.

### Client configuration (Windows, absolute paths)

Claude Desktop: `%APPDATA%\Claude\claude_desktop_config.json` (other clients use
the same `command`/`args`/`env` shape under their own key):

```json
{
  "mcpServers": {
    "ragforge": {
      "command": "C:\\Users\\abiji\\Documents\\ID4100\\RAGForge\\backend\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\abiji\\Documents\\ID4100\\RAGForge\\backend\\scripts\\run_mcp_server.py"
      ],
      "env": {
        "DATA_DIR": "C:\\Users\\abiji\\Documents\\ID4100\\RAGForge\\backend\\data"
      }
    }
  }
}
```

Notes:

* `DATA_DIR` is optional (it defaults to `backend/data`). Set it explicitly if
  you keep the corpus elsewhere — it is the only switch needed to point a client
  at a different workspace.
* Never put provider API keys in a committed config file. If a client needs one,
  set it in that client's own config, which lives outside this repository.
* Restart the client after editing the config; MCP servers are started at client
  startup and are not reloaded.

---

## 3. Tools

**Read-only (6)** — annotated `read_only_hint=true`, `destructive_hint=false`.
Nothing in this group changes workspace state beyond recording the retrieval run
a search would record in the UI.

**Agentic (12, V18)** — annotated by what they do: planning tools are not
read-only (some register candidate rows) but none of them require confirmation;
the four that change a corpus (`create_knowledge_base`, `ingest_sources`,
`build_knowledge_base`, `add_user_documents`) are `destructive_hint=false` but
require a recorded, parameter-bound confirmation. There is still no delete,
bulk-destructive, benchmark-edit, review or configuration tool. The full table,
with the lifecycle and the gates, is in
[`docs/v18-agentic-operations.md`](v18-agentic-operations.md#3-tools).

### `list_knowledge_bases`
Returns id, name, domain/purpose/audience/depth, status, real document/chunk
counts, vector backend, recorded embedding model, benchmark version counts, and
capability flags (`search`, `answer`, `evaluate_retrieval`) each backed by a
`basis` string. Also states the access model honestly: a single-user local
install with no per-user scoping.

### `get_knowledge_base_status(kb_id)`
Live probe of the configured vector backend (`points_count`), the local chunk
row count, whether they agree, the embedding identity recorded at index time vs
the configured provider/model, the registered retrieval strategies with an
explicit usability basis, and dependency states (vector store, embedding
provider, answer generator). `ready_for_retrieval` is true only when chunks
exist, the vector store answered and the embedding configuration is not known to
be inconsistent — metadata alone never makes a knowledge base "ready".
Unmeasurable fields are `null` with a reason in `unavailable`, never `0`.

### `search_knowledge_base(kb_id, query, top_k?, strategy?, params?, include_text?)`
Calls `RetrievalService.run()`: the KB's persisted configuration, the registered
strategy, the recorded run. Returns ranked chunks with chunk/document ids, real
scores (`retrieval_score` on the strategy's own scale, `rerank_score`,
per-pool `score_breakdown`), the retriever's own `why`, and the provenance the
corpus has (`source_title`, `source_url`, `publisher`, `section`, `section_path`,
`page`/`slide`, `content_hash`, `trust_score`). The trace carries the applied
strategy, resolved params, config fingerprint, stages, timings, corpus version
and the retrieval run id. `top_k` is validated (1–50) and an unknown strategy is
an error listing the registry — **no fallback between strategies, ever**.

### `get_evidence(kb_id, chunk_ids? | document_id?, max_chunks?, include_text?)`
Fetches specific evidence from the workspace by id, preserving the requested
order and naming any id that does not exist. Document metadata excludes
filesystem paths by construction, and an id that looks like a path is simply
"not found" — nothing is opened from disk.

### `answer_from_knowledge_base(kb_id, question, answer_mode?, strategy?, top_k?, unsupported_claim_action?, include_evidence?)`
Calls `AnsweringService.answer()` — query plan, retrieval, evidence gate,
generation, deterministic citation validation. Returns the pipeline's own
`status` (`grounded` / `partial` / `abstained` / `clarification_required` /
`generation_failed`) and `grounding_state` (`ANSWERED` /
`PARTIALLY_SUPPORTED` / `INSUFFICIENT_EVIDENCE` / `CONFLICTING_EVIDENCE` /
`NO_RELEVANT_EVIDENCE`), the claims with their support verdicts, validated
citations, the evidence used, the abstention reason when it could not answer, and
the generation identity (`generated_by`, `model`, `is_mock`, `prompt_version`).
In `abstain_if_unsupported` mode an insufficient gate makes **no model call at
all**. If the generator is unavailable the pipeline's failure answer is returned
with `status=generation_failed` and warnings, or a capability error is raised —
never a fabricated grounded answer.

### `evaluate_retrieval(kb_id, benchmark_version?, strategy?, top_k?, question_ids?, label?, dry_run?, allow_keyword_fallback?)`
Honours the existing lifecycle: only a **FROZEN** benchmark version is scored (a
draft/review/approved version is refused with its actual status), and with no
version requested the most recent FROZEN version is used, naming what was
skipped. `dry_run=true` reports the version states and executes nothing. Results
carry the real aggregate and per-question metrics, the evaluator identity
(implementation, ground-truth grains, strict mode, `k` = retrieval cut-off), the
per-question lifecycle statuses, and `unavailable_metrics` (answer correctness,
key-point recall, citation support, abstention quality) with the reason — the
answer-quality benchmark is still in `draft`, so those metrics stay UNKNOWN.

### Agentic operations (V18, summary)

| Tool | Needs confirmation |
|---|---|
| `analyze_domain` | no — returns a proposal, persists nothing |
| `discover_sources` | no — registers candidate rows and scores them; downloads nothing |
| `assess_sources` | no — explains stored decisions, signals, reasons, warnings |
| `plan_knowledge_base` | no — proposes actions, parameters and fingerprints (`mutates=false`) |
| `confirm_plan` | n/a — issues the single-use confirmation the next four tools require |
| `create_knowledge_base` | **yes** |
| `ingest_sources` | **yes** — ACCEPTED sources only; REVIEW needs a recorded human decision |
| `build_knowledge_base` | **yes** — the existing chunk → embed → index pipeline |
| `add_user_documents` | **yes** — content, never paths; bound to each document's sha256 |
| `get_operation_status` | no — lifecycle, authorized parameters, run and stages |
| `validate_knowledge_base` | no — readiness from measured evidence, `UNKNOWN` never a pass |
| `compare_retrieval_strategies` | no — lifecycle-gated on the benchmark instrument; promotes nothing |

### Resources (read-only)

| URI | Content |
|---|---|
| `ragforge://knowledge-bases` | the knowledge-base index |
| `ragforge://knowledge-bases/{kb_id}` | one knowledge base's status report |
| `ragforge://knowledge-bases/{kb_id}/documents/{document_id}` | document metadata/provenance (no paths) |
| `ragforge://knowledge-bases/{kb_id}/chunks/{chunk_id}` | one evidence chunk with its provenance |

Resources are targeted lookups by id: there is no resource that dumps a corpus.

---

## 4. Example flow (real output)

The transcript below was produced on 2026-10-14 by
`scripts/mcp_smoke_client.py --kb kb_f278c283c748` against the real workspace
(real Qdrant, real `all-MiniLM-L6-v2` embeddings, development mock LLM because no
`LLM_API_KEY` is configured). It is copied from the run, not reconstructed.

```
tools: list_knowledge_bases, get_knowledge_base_status, search_knowledge_base,
       get_evidence, answer_from_knowledge_base, evaluate_retrieval
resources: ['ragforge://knowledge-bases']
resource templates: ['ragforge://knowledge-bases/{kb_id}',
                     'ragforge://knowledge-bases/{kb_id}/documents/{document_id}',
                     'ragforge://knowledge-bases/{kb_id}/chunks/{chunk_id}']
is_error: False
```

`list_knowledge_bases` returns every knowledge base with real counts and capability
flags, e.g. for the protected corpus `kb_f278c283c748`: `status: ready`, 11
documents, 812 chunks, `frozen_benchmark_versions: 0`, and a `basis` list stating
what each capability was granted from — while a knowledge base with an empty
corpus reports `search: false` with the basis `"0 chunk row(s) and 0 document(s) in
the corpus"`.

Search on the protected corpus (query `"What is an engine?"`, `top_k: 3`):

```
search is_error: False
  #1 chk_00fe4e273d0e score=0.66    'table kinds of engines include:…'
  #2 chk_936027ecaa9f score=0.6263  '2 ] Internal combustion engine [ edit ] A three-horsepower internal combustion engine th'
  #3 chk_d15901e67e35 score=0.6064  'Jump to content From Wikipedia, the free encyclopedia Machine that converts one or more '
```

`get_evidence` with the returned chunk id then gives the full passage with its
provenance (`document_id: doc_90752e50cc19`, `chunk_index: 35`, source title and
URL) — no filesystem path is included:

```
get_evidence {
  "kb_id": "kb_f278c283c748",
  "requested_chunk_ids": ["chk_00fe4e273d0e"],
  "chunks": [{"chunk_id": "chk_00fe4e273d0e", "document_id": "doc_90752e50cc19",
              "chunk_index": 35, "text": "table kinds of engines include:…", …}]
}
```

Grounded answer for the same question:

```
answer: status=grounded grounded=True grounding_state=ANSWERED citations=3
(answer text and the three validated citations are returned in full)
```

Evaluation lifecycle (this KB has no frozen retrieval benchmark version):

```
evaluate_retrieval (dry run): allowed=False
reason=No FROZEN benchmark version exists for this knowledge base (versions: no versions exist).
```

Errors are actionable and coded, e.g.:

```
[KB_NOT_FOUND] Knowledge base 'kb_missing' does not exist in this workspace.
Hint: Call list_knowledge_bases to see the available knowledge base ids.

[INVALID_ARGUMENT] top_k must be between 1 and 50; got 0. Hint: Ask for at most 50 chunks per call.

[CAPABILITY_UNAVAILABLE] Retrieval could not run: Vector store error: … 
Hint: Call get_knowledge_base_status to see whether the vector store and the
embedding configuration are usable. No other strategy was tried on your behalf.

[LIFECYCLE_BLOCKED] Benchmark version 'draft-v1' is DRAFT, not FROZEN, so it may not be scored.
```

---

## 5. Security and reliability

| Boundary | How it is enforced |
|---|---|
| Untrusted arguments | Every field is schema-validated by the SDK (types, required, ranges) and re-validated in the bridge: query ≤ 2000 chars, `top_k` 1–50, ≤ 50 chunk ids per call, ≤ 200 questions per evaluation, `max_chunks` 1–50, `max_chunks` order-of-checks independent of workspace state |
| Result size | chunk text capped at 4000 chars (`text_truncated` flags it and `char_count` reports the original), ≤ 10 evidence chunks per answer, ≤ 100 claims, with counts always reported |
| Filesystem | The tools only look ids up in SQLite/Qdrant. No tool accepts a path; document metadata omits `file_path`/`raw_file_path`; a path-shaped id is simply NOT_FOUND |
| Secrets | No tool returns settings, keys, environment values or server configuration. A test asserts that `api_key`/`secret`/`password` never appear in any payload |
| Errors | Anticipated failures become coded tool errors the model can act on; unexpected ones are logged with a traceback on stderr and reported to the client as a generic failure with no internals |
| Dependencies | An unreachable vector store, a missing embedding model or a failed generation produces a capability error naming what is missing. There is no silent fallback to another strategy, another corpus or another model |
| Prompt injection | Retrieved text is returned as data inside `text`/`content` fields, every text-bearing payload carries `content_warning`, and the server `instructions` tell the client that corpus text is untrusted. Tests index a document containing "IGNORE ALL PREVIOUS INSTRUCTIONS…" and assert it comes back as quotable evidence with the warning intact, not as an instruction |
| V18 upload/write path | No tool accepts a path; documents travel as content (≤ 20 per call, ≤ 8 MiB each, ≤ 24 MiB per call), file names are sanitised, extensions **and** magic bytes are validated, and every URL passes the existing SSRF validation before any request |
| Read-only surface | Six read-only tools plus four read-only resources; every state-changing tool is allow-listed by name and requires a parameter-bound confirmation; tests pin the exact public method set of the bridge, assert no destructive tool exists, and assert that only the four allow-listed actions can be authorized |
| Protected data | Tests use a scratch `DATA_DIR` and an in-memory vector store; `kb_f278c283c748` is never opened, listed or written by the test suite |

**Multi-user:** RAGForge has no accounts or per-user isolation, so the tools do
not pretend to scope anything by user. `list_knowledge_bases` says so in
`access_model`, and the server instructions repeat it. Do not expose this server
to a shared or networked environment as if it were per-user.

---

## 6. Tests

[`tests/test_v17_mcp_server.py`](../backend/tests/test_v17_mcp_server.py) — 53
tests, hermetic (scratch DATA_DIR, in-memory cosine store, hashing embeddings,
mock LLM):

* **Protocol**: initialization/handshake, tool discovery, published input and
  output schemas, required arguments, tool annotations, resource and template
  discovery, unknown-tool handling. (V18 extends the same class of checks to the
  18-tool surface and adds the confirmation-argument requirement, in
  [`tests/test_v18_mcp_agentic.py`](../backend/tests/test_v18_mcp_agentic.py).)
* **Tools**: success and failure paths for all six; provenance and score
  semantics; retrieval trace and run persistence; strategy honoured without
  fallback; grounding statuses preserved; abstention behaviour; citation
  resolution to real chunks; generation-failure disclosure.
* **Lifecycle**: draft/unknown versions refused with their real states;
  `dry_run` executes nothing (asserted by row counts); a frozen version runs and
  reports measured metrics; a frozen version whose questions have only keyword
  ground truth reports UNKNOWN metrics (never `0.0`); answer-level metrics listed
  as unavailable with the human-review reason.
* **Security**: `top_k`/query/count bounds, both-sources and neither-source
  requests, path-shaped ids, no path or secret leakage, read-only surface guard,
  prompt-injection handling.
* **Real stdio**: a subprocess launched through `StdioServerParameters` with the
  project's launcher script completes the handshake, lists the tools, calls tools
  successfully and receives coded errors — proving the transport and the stdout
  protocol channel. (V18's suite repeats this against the 18-tool surface and
  drives V18 tools through it.)

V18's suite is hermetic in the same way and adds deterministic network doubles,
with the real parser still exercised over fixture bytes:
[`tests/test_v18_mcp_agentic.py`](../backend/tests/test_v18_mcp_agentic.py) — 95
tests covering planning, the confirmation gates (missing, wrong fingerprint,
wrong action, replay, expiry, changed document bytes, a source downgraded after
confirmation), KB creation, ingestion, the build, user-document intake and its
provenance, operation status, validation, strategy comparison, and the safety
assertions (allow-listed surface, no path or secret leakage, untrusted text stays
data, protected artifacts untouched). Together with the V17 suite that is 149
tests.

A defect this suite caught and V17 fixed: `AnsweringService._failure_answer` was
called with `kb_id=` while its parameter is `kb=`, so **any** generation failure
raised `TypeError` instead of returning the documented `generation_failed`
answer. Fixed in [`app/services/answering/service.py`](../backend/app/services/answering/service.py);
the MCP test now covers that path.

---

## 7. Limitations (stated, not hidden)

* **stdio only.** No streamable-HTTP transport, because securing it properly
  (auth, per-client scoping) is beyond this milestone; shipping it unauthenticated
  would expose the whole corpus.
* **No prompts** are registered: the value is in the tools and the honest
  resource views, and inventing prompt templates was not part of the objective.
  Elicitation/sampling are unused.
* **Writes exist in V18, but only four of them and only behind a confirmation:**
  create a KB, ingest approved sources, add user-provided documents, build the
  index. There is still no delete, rename, review, benchmark-edit or
  configuration tool, and no bulk destructive operation.
* **No cancellation and no background progress.** Every operation runs
  synchronously inside the tool call, so there is nothing safe to cancel;
  `get_operation_status` reports what a run did rather than a live ticker.
* **Uploading a file needs the host.** stdio has no file channel, so
  `add_user_documents` takes content: the client reads the bytes and passes them.
  The tool refuses path-shaped arguments instead of pretending to read disk.
* **No per-user access control** (single-user local install), and no rate
  limiting: a client can issue as many searches as it likes. Each search records
  a retrieval run, exactly like a UI query, which is what makes the usage
  auditable.
* **MCP tool results feed another model's context**, so the size caps exist for
  cost and privacy reasons: a client that wants a whole document should page
  through `get_evidence`, not expect one giant dump.
* **Retrieval quality is whatever RAGForge's retrieval is.** The MCP layer adds
  no ranking, no reranking and no fusion.
