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

**V20 — the answer surface behaves honestly off-topic.** The off-domain gate
defect (an Automobile KB answering "How many moons does Jupiter have?") is fixed:
the gate now applies a measured relevance floor and a corpus vocabulary-fit rule,
and the same call over this server returns `abstained` / `LOW_EVIDENCE_RELEVANCE`
with zero citations. See [`docs/v20-offdomain-gate.md`](v20-offdomain-gate.md) for
the measured before/after record and the residual limits.

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
| [`scripts/check_claude_desktop_connection.py`](../backend/scripts/check_claude_desktop_connection.py) | Reads a client's own `claude_desktop_config.json`, launches the configured command verbatim and runs the acceptance sequence (§2) |

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

### Client configuration (any MCP client, Windows)

Every MCP client uses the same `command` / `args` / `env` shape. Use **absolute
paths** and point `args` at the launcher script, which puts `backend/` on
`sys.path` itself and therefore works from any working directory:

```json
{
  "mcpServers": {
    "ragforge": {
      "command": "C:\\path\\to\\RAGForge\\backend\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\RAGForge\\backend\\scripts\\run_mcp_server.py"],
      "env": { "DATA_DIR": "C:\\path\\to\\RAGForge\\backend\\data" }
    }
  }
}
```

* `env` is optional. `DATA_DIR` defaults to the absolute `backend/data` (the
  settings module resolves it from its own file location, and `backend/.env` is
  loaded the same way), so the server finds the same SQLite database whatever
  directory the client launches it from. Set it explicitly only to point a
  client at a different workspace.
* A `cwd` is deliberately **not** required. Do not add one.
* Never put provider API keys in a client configuration file: the server reads
  the git-ignored `backend/.env` itself, and a client config may be shared,
  synced or committed by accident.

### Claude Desktop on Windows — setup and verification

The steps below were executed on this machine except where a step is marked
**manual** (the host application itself is not scriptable from here).

#### Where the configuration file lives

| Item | Value |
|---|---|
| Configuration file | `%APPDATA%\Claude\claude_desktop_config.json` |
| Absolute form | `C:\Users\<you>\AppData\Roaming\Claude\claude_desktop_config.json` |
| Created by | Claude Desktop on first run — create the folder yourself if it is absent |
| Reloaded | only at client start, so **restart Claude Desktop** after editing |

`%APPDATA%` is a per-user path and lives outside this repository on purpose: it
is user-specific and must never be committed.

#### Step 1 — find the two absolute paths

```bat
cd <your RAGForge checkout>\backend
.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
echo %CD%
```

The first command prints the `command` value; the second gives the repository
root used to build `args` and `DATA_DIR`.

#### Step 2 — merge, never overwrite

The file may already configure other MCP servers. Back it up, then add (or
replace) **only** the `ragforge` key inside `mcpServers`, leaving every other
entry untouched:

```bat
copy "%APPDATA%\Claude\claude_desktop_config.json" "%APPDATA%\Claude\claude_desktop_config.json.bak"
```

```json
{
  "mcpServers": {
    "some-other-server": { "command": "...", "args": ["..."] },
    "ragforge": {
      "command": "C:\\path\\to\\RAGForge\\backend\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\RAGForge\\backend\\scripts\\run_mcp_server.py"],
      "env": { "DATA_DIR": "C:\\path\\to\\RAGForge\\backend\\data" }
    }
  }
}
```

The file must be valid JSON, and on Windows the backslashes inside JSON strings
must be **doubled** (`C:\\Users\\...`). A single `\U` makes the document invalid
and Claude Desktop will fail to start the server without saying why.

#### Step 3 — verify the connection before blaming the client

RAGForge ships a checker that reads your own client configuration, launches the
**exact** command it contains (no `cwd` is set, because the host sets none) and
runs the acceptance sequence: handshake, tool/resource discovery, knowledge-base
lookup, search with provenance, evidence by chunk id, a grounded answer, an
unsupported question that must abstain, and an off-domain question:

```bat
backend\.venv\Scripts\python.exe backend\scripts\check_claude_desktop_connection.py

:: or with explicit paths and questions:
backend\.venv\Scripts\python.exe backend\scripts\check_claude_desktop_connection.py ^
  --config "%APPDATA%\Claude\claude_desktop_config.json" --server ragforge ^
  --kb kb_f278c283c748 --supported "What is an engine?" ^
  --unsupported "Describe the courtship display of the Australian lyrebird." ^
  --offdomain "How many moons does Jupiter have?" --json report.json
```

Exit code 0 means no check failed. Results are labelled `PASS`, `WARN`,
`KNOWN-DEFECT` (a defect this project has already documented and left open —
**not** a pass) or `FAIL`. The off-domain check used to be the `KNOWN-DEFECT`
case; since V20 it is a plain PASS/FAIL criterion, so a reappearance of that
defect now fails the run.

`--env KEY=VALUE` (repeatable) adds variables to the launched server's
environment, **merged over** the config's own `env`. `command` and `args` always
come from the client configuration verbatim, so the transport under test is the
configured one. Its purpose is to pin the answer generator for an acceptance run
without editing `backend/.env` or the client config:

```bat
backend\.venv\Scripts\python.exe backend\scripts\check_claude_desktop_connection.py ^
  --kb kb_f278c283c748 --env LLM_PROVIDER=mock --json reports\mcp-check.json
```

Every override is printed and written into the report, so a run is never silently
different from the configured one.

**Measured on 2026-10-15** (V20, `--env LLM_PROVIDER=mock`): **25 checks — 23
PASS, 0 FAIL, 2 warnings, 0 known defects.** The two warnings are the pinned
`LLM_PROVIDER` override and the disclosed mock generation identity
(`is_mock=true`, `model="mock/mock-1"`) — retrieval, scores, citations and the
abstention are real. The protected KB reported 812 chunk rows = 812 measured
vector points, `match=true`.

#### Integration checklist (repeatable)

Run this after any change to the launcher, the server, the gate or the provider
configuration. Each line is a check the script above performs, so a green script
run and a completed checklist say the same thing.

| # | Check | Expected |
|---|---|---|
| 1 | config preflight | `command`, launcher and `DATA_DIR` all exist |
| 2 | MCP handshake | `serverInfo.name='ragforge'`, protocol negotiated |
| 3 | tool discovery | 18 tools, the 6 read tools present |
| 4 | annotations | `search_knowledge_base` read-only, non-destructive |
| 5 | resources | 1 resource + 3 templates |
| 6 | knowledge-base list | the Automobile KB listed with real counts (11 docs / 812 chunks) |
| 7 | live index probe | chunk rows = measured vector points, `match=true` |
| 8 | search | ranked chunks with chunk id, document id, URL, content hash |
| 9 | no path leak | no filesystem path in any payload |
| 10 | evidence by id | resolves to a document id and chunk index |
| 11 | grounded answer | `grounded`, citations all resolve to real chunks |
| 12 | generation identity | `is_mock` reported honestly (true with the mock) |
| 13 | unsupported question | `abstained`, 0 citations, 0 claims |
| 14 | off-domain question | `abstained`, `LOW_EVIDENCE_RELEVANCE`, 0 citations |
| 15 | lifecycle gate | `evaluate_retrieval` refuses without a FROZEN benchmark |
| 16 | manual, in the client | RAGForge listed; Claude *calls* the tool rather than answering from memory |

Check 16 is the only one that cannot be automated from this repository, and it is
the only one that proves *host* integration.

#### Verification status of the host test

**UNVERIFIED — Claude Desktop is not installed on this machine.** Verified
2026-10-15 by: `reg query HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall
/s /f Claude` → 0 matches; no `%LOCALAPPDATA%\AnthropicClaude`,
`%LOCALAPPDATA%\Programs\claude-desktop` or `%PROGRAMFILES%\Claude`; `claude` not
on `PATH`; no Start-menu entry. What **is** verified is the configured command
speaking MCP correctly and returning real RAGForge evidence (the script above).
These are different claims and must not be substituted for each other.

#### Step 4 — restart and verify in the client (manual)

1. Quit Claude Desktop completely (tray icon → Quit; a closed window is not a
   quit), then start it again — MCP servers are spawned at start-up.
2. Open the tools / connector indicator in the message box. **RAGForge**
   (`ragforge`) must be listed with 18 tools and 4 resources.
3. If it is not listed, read the client's own server log:
   `%APPDATA%\Claude\logs\mcp-server-ragforge.log` (older builds: `mcp.log`). A
   successful start writes `RAGForge MCP server starting on stdio (stdout is the
   protocol channel; logs go to stderr)`.
4. Ask a question that can only be answered from the corpus and confirm in the
   transcript that Claude **called the tool** instead of answering from its own
   knowledge.

#### Step 5 — Automobile questions worth trying

| Intent | Ask | What to check |
|---|---|---|
| Direct retrieval | `What is an engine?` | hits cite `https://en.wikipedia.org/wiki/Engine` with chunk ids and real scores |
| Grounded answer | `What are the main types of suspension systems?` | each claim carries a citation that resolves to a real chunk |
| Provenance | `Show me the full text of the first chunk` | `get_evidence` returns document id, chunk index and content hash |
| Abstention | `Describe the courtship display of the Australian lyrebird.` | `abstained`, `INSUFFICIENT_EVIDENCE`, **zero citations**, no model call |
| Off-domain refusal (V20) | `How many moons does Jupiter have?` | `abstained`, `LOW_EVIDENCE_RELEVANCE`, zero citations — it must **not** be answered from engine text |
| Provable topic mismatch (V20) | `What is the capital of Peru?` | `abstained`; the terms occur in none of the 812 chunks |
| Lifecycle gate | `Evaluate retrieval quality on this knowledge base` | refused with the real reason (no FROZEN benchmark version exists) |

#### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Server missing from the connector list | invalid JSON (single backslashes) or the client was not fully restarted. |
| Server exits immediately / `ENOENT` | `command` is not a real `python.exe`. Print it with `.venv\Scripts\python.exe -c "import sys;print(sys.executable)"` and paste the exact text. |
| `ModuleNotFoundError: app` | `args` does not point at `scripts/run_mcp_server.py`, or an added `cwd` broke it. The launcher needs no `cwd`. |
| `ModuleNotFoundError: mcp` | the MCP SDK is optional: `cd backend && .venv\Scripts\python.exe -m pip install -r requirements-mcp.txt`. |
| Wrong or empty knowledge base list | wrong `DATA_DIR` (the database is `backend\data\ragforge.db`) or a different checkout. |
| Vector-store / embedding error on search | Qdrant is not running (`curl http://localhost:6333/collections`) or `QDRANT_URL` is wrong. The first run also downloads `all-MiniLM-L6-v2`. |
| Answers come back with `is_mock: true` | no usable LLM provider is configured, so the deterministic extractive generator is used and **disclosed** in `generation.is_mock` / `generated_by` / `model`. Retrieval and citations are still real. Set `LLM_PROVIDER=openai` (+ `LLM_MODEL`, `LLM_API_KEY`, optional `LLM_BASE_URL`) in `backend/.env` for real generation. |
| An off-domain question is answered anyway | this was a real gate defect, fixed in V20. If it reappears, check `evidence_relevance_floor` in the answer signals: it is NOT APPLICABLE for `bm25`/`hybrid` runs (pool-normalized score scale) and for the development hashing embedder, so those runs have one fewer guard. See [`docs/v20-offdomain-gate.md`](v20-offdomain-gate.md). |
| An in-domain question abstains after changing `EMBEDDING_MODEL` | `ANSWER_MIN_RELEVANCE` (default `0.35`) is calibrated for `all-MiniLM-L6-v2`. Recalibrate per model with `backend\scripts\v20_calibrate_relevance.py --kb <kb_id>` and set the measured value in `backend/.env`. |
| `LLM_PROVIDER=gemini` used to fail with "declared but not yet implemented" | fixed in V20: `gemini` is routed through the existing OpenAI-compatible provider to Google's documented endpoint `https://generativelanguage.googleapis.com/v1beta/openai/`. Check the model ids your key can actually see with `GET https://generativelanguage.googleapis.com/v1beta/openai/models`. |
| Every call hangs | a hung provider call; `LLM_TIMEOUT_S` and `LLM_MAX_RETRIES` in `backend/.env` bound it. |

#### Security and single-user limits

* **stdio only.** The server is a child of the client process, so the corpus is
  reachable only by a program you started. There is no HTTP port and no
  authentication — do not expose an RAGForge MCP server to a shared or networked
  environment.
* **No per-user scoping.** Every knowledge base in the workspace is visible to
  every client on this machine.
* Tools never accept a filesystem path, never return one, and never return
  settings, keys or environment values. Corpus text is returned as untrusted data
  with a `content_warning`, and the server instructions tell the model to quote
  it rather than obey it.
* Write tools exist (V18) but only four, and each requires a recorded,
  parameter-bound confirmation. There is no delete, no bulk mutation, no
  benchmark edit and no override of a source decision.
* Keep the client configuration file out of version control.

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
(real Qdrant, real `all-MiniLM-L6-v2` embeddings, and the deterministic
**fallback** generator for answer text because the configured provider was not
usable — the answer payload says so in `generation.is_mock`). It is copied from
the run, not reconstructed. Re-run on 2026-10-15 with
`scripts/check_claude_desktop_connection.py` (§2), which reproduces every step
there and additionally exercises the abstention and off-domain cases.

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

A second defect was found and fixed on 2026-10-15, while verifying the Claude
Desktop setup. The "real stdio" test in the V18 suite launched its server with
`StdioServerParameters(...)` and no `env`, on the assumption that the child would
inherit the test's environment. It does not: the SDK's `env` field is documented
as *"extra environment variables, merged over `get_default_environment()`"*, and
that helper inherits a **whitelist** (`PATH`, `HOME`, …), not `os.environ`. The
nested server therefore lost `DATA_DIR` (and the embedding/LLM overrides) and
opened the developer's real `backend/data/ragforge.db`, writing two
`mcp_operations` rows into the real workspace on **every** run — measured by
counting rows before and after a single test invocation. The test now forwards
the scratch variables explicitly and **asserts** the database path the launched
server resolved, so the suite is hermetic as documented and cannot regress
silently. It also runs faster, because the nested server no longer loads the real
embedding model. (The same SDK behaviour is why a Claude Desktop `env` block must
give `DATA_DIR` explicitly when the corpus does not live in `backend/data` — see
§2.)

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
* **The off-domain gate defect was reachable through MCP and is now FIXED (V20).**
  It was reproduced over this server on 2026-10-15: `How many moons does Jupiter
  have?` against the Automobile corpus returned `status=grounded`,
  `reason_code=SUFFICIENT`, 1 citation, because the gate measured lexical
  alignment over *unweighted* question terms so generic vocabulary (`many`,
  `2018`, `regulations`) cleared `MIN_ALIGNMENT_TO_PARTIAL = 0.15`, and the
  SUFFICIENT rule consulted alignment only for the *confidence* label. V20 adds
  two measured rules — a raw-similarity relevance floor and a corpus
  vocabulary-fit check — and the same call through this server now returns
  `status=abstained` / `reason_code=LOW_EVIDENCE_RELEVANCE` / 0 citations. Full
  record: [`docs/v20-offdomain-gate.md`](v20-offdomain-gate.md). Two residual
  gaps are documented there and are NOT hidden: the floor is calibrated per
  embedding model, and `bm25`/`hybrid` runs have no relevance floor because their
  scores are pool-normalized.
* **Answer text can come from a disclosed fallback generator.** With no usable
  LLM provider configured, `answer_from_knowledge_base` still returns retrieval,
  citations and provenance, but the answer *text* is produced by the
  deterministic extractive generator and is labelled
  `generation.is_mock = true` / `generated_by = "fallback"` /
  `model = "extractive-mock"`. In that state claims are marked
  `support_check = "lexical_overlap"` with the explicit note that semantic
  entailment was **not** performed. Never quote such an answer as if a real
  model produced it.
* **A legacy knowledge base may not validate as READY.**
  `validate_knowledge_base` on the protected corpus `kb_f278c283c748` returns
  `NOT_READY` while its index is intact (812 chunk rows = 812 *measured* vector
  points, collection green, embedding identity matching) and 0 missing/orphan/
  stale vectors. The two blocking findings come from the corpus's age, not from
  damage: its 11 documents were serialised before the document `status` field was
  persisted (so the audit reads them as `parsed` rather than indexed → 11
  `partial_documents`), and its chunks carry `document_id` and `content_hash` but
  no `source_id` (→ 812 chunk-metadata gaps recorded twice, once for the chunk
  row and once for the vector payload = 1624 `provenance_gaps`;
  `_REQUIRED_PROVENANCE = (source_id, document_id, content_hash)`). The tool
  reports what it measures rather than softening the verdict, and the corpus is
  deliberately not rebuilt to make the number prettier.
