# V19 — Real-LLM, Second-Domain Validation

Status: **plan pre-registered; results appended after execution** (this file is
written *before* the corpus is downloaded, so the plan, the acceptance criteria
and the evaluation methodology below are declared up front and cannot be
quietly adjusted after the numbers are known).

V18 proved the agentic MCP construction loop end-to-end, but on the same
Automobile Engineering corpus the project had used for every earlier
experiment, and with the labelled development mock LLM because no provider was
configured. V19 answers a narrower, harder question:

> Does the automated construction loop — domain specification → source
> discovery → quality decisions → ingestion → chunking → embedding → retrieval
> → evaluation — produce a usable, provenance-rich knowledge base in a
> **materially different domain**, and can the pipeline be validated with a
> **real LLM** whose identity is honestly recorded at every step?

## Phase 1 — Provider readiness (measured, not assumed)

Measured on this machine before any experiment work (no secret values printed):

| Check | Result |
|---|---|
| `LLM_PROVIDER` (env / `backend/.env`) | unset / empty |
| `LLM_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` | all unset |
| `LLM_MODEL` | `gpt-4o-mini` present in `backend/.env` as an unused placeholder |
| `ALLOW_MOCK_LLM` | `true` → today every LLM-backed step silently uses the labelled dev mock |
| `openai` Python package | installed (so any OpenAI-compatible endpoint works immediately) |
| Local OpenAI-compatible server (Ollama 11434 / LM Studio 1234) | none listening |

Provider identity rules (verified in code, unchanged by V19 except the timeout
fix below):

* `create_llm_provider()` selects the provider from `Settings`; with no
  provider configured and `ALLOW_MOCK_LLM=true` it returns `MockLLMProvider`
  (`is_mock=True`) **with a logged warning** — never a silent real-looking call.
* `analyze_domain` returns `generated_by` + `is_mock` and emits the warning
  `is_mock=true: ... development mock provider, not a real model`.
* `answer_from_knowledge_base` returns `generation.{generated_by, model,
  is_mock}` in every payload.
* `anthropic` / `gemini` provider names raise `LLMError` ("declared but not
  yet implemented"); only `openai` (incl. any OpenAI-compatible endpoint via
  `LLM_BASE_URL`) and `mock` exist.

**Gap found and fixed (V19 product change):** `OpenAICompatibleProvider` used
the SDK defaults — a 600 s timeout and 2 automatic retries — so one hung call
could stall an MCP tool for half an hour. `Settings` now exposes
`LLM_TIMEOUT_S` (default 120 s) and `LLM_MAX_RETRIES` (default 2), both passed
to the client; `.env.example` documents them.

**Configuration needed for a real provider** (kept in the git-ignored
`backend/.env`; never committed):

```
LLM_PROVIDER=openai            # any OpenAI-compatible endpoint
LLM_MODEL=<model name>
LLM_API_KEY=<key>
LLM_BASE_URL=<optional; e.g. https://api.deepseek.com/v1 or http://localhost:11434/v1>
LLM_TIMEOUT_S=120
LLM_MAX_RETRIES=2
```

The operator elected to supply a key in `backend/.env`. The V19 run re-probes
the resolved provider at start-up and records `name`/`model`/`is_mock` (and the
base-URL host only) in the run artifact, so the report states the real identity
or the exact reason a mock was used.

## Phase 2 — Domain, corpus and evaluation plan

### Domain and intended use case

**Internet Protocols / Network Engineering** — a course-reference knowledge
base for network-engineering students and junior engineers: what a protocol
actually specifies (field semantics, parameter values, message flows, state
machines), answerable from the primary standards with citations.

Why this domain: it is materially different from Automobile Engineering
(terminology, subdomains, source *type*: primary standards documents rather
than lecture notes and web pages), it is bounded, and its primary sources are
unambiguously authoritative — the corpus is the IETF standards themselves,
published by the RFC Editor, immutable once published, and freely
redistributable. A claim in this corpus can be checked against a paragraph of
the specification, which makes grounding and abstention measurable.

### Knowledge boundaries

In scope (subdomains → RFCs, all fetched as plain text from
`https://www.rfc-editor.org/rfc/<n>.txt`, reachability verified 12/12 before
planning):

| # | Subdomain | RFC | Title | ~Size |
|---|---|---|---|---|
| 1 | IPv4 layering, datagram, TOS/TTL | 791 | Internet Protocol | 95 KB |
| 2 | Control messages, ICMP | 792 | Internet Control Message Protocol | 29 KB |
| 3 | Address resolution | 826 | ARP | 22 KB |
| 4 | Private address allocation | 1918 | Address Allocation for Private Internets | 22 KB |
| 5 | Documentation address blocks | 5737 | IPv4 Address Blocks Reserved for Documentation | 7 KB |
| 6 | IPv6 base header, flow labels | 8200 | Internet Protocol, Version 6 (IPv6) | 93 KB |
| 7 | Transport, connections, 3W handshake | 9293 | Transmission Control Protocol | 264 KB |
| 8 | DNS concepts, resolver model | 1034 | Domain Names — Concepts and Facilities | 126 KB |
| 9 | DNS messages, resource records | 1035 | Domain Names — Implementation and Specification | ~110 KB |
| 10 | Dynamic host configuration | 2131 | DHCP | 114 KB |
| 11 | Inter-domain routing, FSM | 4271 | BGP-4 | 223 KB |
| 12 | Link-state intra-domain routing | 2328 | OSPF Version 2 | ~150 KB |
| 13 | TLS 1.3 handshake and records | 8446 | The Transport Layer Security (TLS) Protocol Version 1.3 | 338 KB |
| 14 | HTTP semantics (methods, status, fields) | 9110 | HTTP Semantics | 503 KB |

Expected corpus: 14 documents, ≈2.1 MB of parsed text, ≈1 700–2 200 chunks at
the pre-declared chunking configuration. Out of scope by design: vendor
configuration (Cisco/Juniper CLI), non-IETF protocols (MQTT, Modbus, CAN),
physical/wireless layers, RFCs not ingested (e.g. 7540 HTTP/2, 6749 OAuth),
programming how-tos, and anything historical outside the ingested documents.

### Candidate sources and trustworthiness

Every source is the RFC Editor's own rendering of a published RFC. Authority
is asserted from the *publisher and document type* (RFC Editor publishes IETF
RFCs; each RFC is an immutable, numbered specification with a stable
identifier), not from an LLM's opinion. The quality engine still scores each
URL with its own signals (domain authority, accessibility, source type,
recency) and may return REVIEW; a REVIEW decision is a **human approval
gate**: the operator approves the exact URL list before ingestion runs. The
approval (by URL, the stable identifier) is recorded in the run artifact.

### Ingestion, chunking, embedding, retrieval configuration (pre-declared)

* **Discovery**: MCP `discover_sources`, provider `user-url`, the 14 URLs above
  (SSRF-validated; ≤ 20 per call).
* **Decision**: engine scores; operator approves REVIEW-band sources by URL;
  only ACCEPTED sources are ingested (`ingest_sources` refuses others).
* **Ingestion**: `ingest_sources` — real HTTP download (30 s timeout, 50 MB
  cap), `TextParser`, content hash, per-document error isolation.
* **Chunking**: `section-aware`, `target_size=1200`, `overlap=150` — the
  pipeline defaults, deliberately *not* tuned for this domain. RFC plain text
  carries no Markdown heading markers, so the section-aware chunker will fall
  back to fixed windows; this is a *measured* limitation, recorded per chunk
  as `chunking_strategy`, and a candidate Phase-5 investigation — not silently
  patched mid-run.
* **Embedding**: local SentenceTransformers `all-MiniLM-L6-v2` (CPU), the
  same model as the Automobile KB, so the embedding identity is comparable.
* **Retrieval**: dense (the documented baseline) with the KB's stored
  configuration; `top_k=10`, candidate probe depth 20 for measurement.
* **No strategy promotion**: BM25/hybrid/reranking are *measured* on this
  corpus as exploratory numbers only; nothing is promoted from Automobile
  results or from these measurements.

### Evaluation methodology (pre-declared)

1. **Draft retrieval benchmark** `benchmarks/internet-protocols-draft-v1.json`
   in the existing retrieval-benchmark schema, with
   `authorship.tier="draft"`, `authorship.human_review="pending"`, and a
   machine-drafting disclosure. Questions are drafted **from the built corpus**
   (question ← passage), each grounded in ≥1 real chunk id whose text contains
   the expected terminology; a grounding self-check verifies every expected
   chunk id exists in the corpus and matches its keywords before the artifact
   is written. ~24–30 supported questions across ≥ 6 subdomains with
   terminology variations (TTL vs hop limit, "3-way handshake", "private
   addresses"/RFC 1918 blocks, record types, DORA, BGP FSM states, TLS
   handshake messages, HTTP method semantics).
2. **Unsupported questions** (~6–8): well-formed, in-domain-looking questions
   whose answer is provably absent (e.g. protocols/RFCs not ingested,
   vendor-specific behaviour). They carry **no expected ids** — the instrument
   classifies them UNKNOWN rather than scoring them 0 — and abstention is
   measured separately through `answer_from_knowledge_base`.
3. **Measurements** (all real, from the existing runners over the draft
   instrument):
   * corpus/ingestion coverage: planned vs ingested vs parsed documents,
     chunk rows, index points, `validate_knowledge_base` verdict and corpus
     integrity status;
   * evidence availability (the diagnostics runner probes the live corpus and
     index for every expected chunk id — this is the per-question coverage
     measurement);
   * retrieval: recall@5/@10, MRR, nDCG@10, precision@10 (lower-bound caveat
     recorded), per-question latency — dense, and (exploratory) bm25 + hybrid;
   * one pre-declared V13-style controlled experiment (baseline dense k=10 vs
     dense k=5 and bm25/hybrid candidates, primary metric MRR, recall as
     guardrail) with hypothesis and rejection conditions in the script.
4. **Answer-level properties stay UNKNOWN**: answer correctness, citation
   support and abstention quality are reported UNKNOWN until reference answers
   and human review exist. No percentage is fabricated; no benchmark question
   is frozen or human-approved by the agent.

### Acceptance criteria (declared before the run)

* A1 — the MCP workflow completes with every consequential step preceded by a
  recorded confirmation; gate refusals (no confirmation, plan-id-as-
  confirmation, replay, changed parameters) still refuse.
* A2 — all 14 sources are either ingested or explicitly reported failed; no
  silent drop. `validate_knowledge_base` returns verdict READY with
  chunk rows == vector points and corpus integrity HEALTHY.
* A3 — ≥ 90 % of labelled questions' expected chunks are present in both the
  corpus and the index (evidence availability); anything below is investigated
  as a Phase-5 failure, not tuned away.
* A4 — dense recall@10 ≥ 0.70 on the draft set. A miss is reported as a
  finding with falsifiable hypotheses; no tuning against the same set.
* A5 — ≥ 6/7 unsupported questions end in abstention or an explicit
  insufficient-evidence status (never a confident grounded answer).
* A6 — every LLM-backed step records its provider identity; if the mock was
  used the report says so and no real-LLM claim is made.
* A7 — protection: protected KB `kb_f278c283c748` stays green with exactly
  812 points; SQLite integrity ok; the four frozen benchmark files and all
  historical experiment artifacts byte-identical; zero `mcp_operations` rows
  referencing the protected KB; cleanup verified for anything created and not
  kept.

### Protection and cleanup

Before and after: SQLite `integrity_check`/`foreign_key_check`, protected
collection point count (812, green), sha256 of the four frozen benchmark
artifacts, KB count, `mcp_operations` rows per KB. The second-domain KB and its
collection are **kept** (they are the validation artifact the draft benchmark
is bound to, and deleting them would orphan it — recorded explicitly); every
other resource this run creates is inventoried, and the report lists what was
created, what was kept and why.

## Phase 3 — MCP workflow execution

(results appended after execution)

## Phase 4 — Measurements

(results appended after execution)

## Phase 5 — Failure investigation

(results appended after execution)

## Phase 6 — Protection checks

(results appended after execution)

## Phase 7 — Verification and delivery

(results appended after execution)

---

## Phase 8 — Executed 2026-10-15: what actually happened

This section is written **after** the run and states results, not intentions. The
plan above is left visible and unchanged, so the pre-registration can be compared
with the outcome.

### Phase 1 re-measured: the real-LLM gate did NOT open → Phases 2–5 not executed

| Check (2026-10-15, no secret value printed) | Result |
|---|---|
| `backend/.env` mtime | `2026-10-14T16:27:58` — unchanged since the session in which the key was exposed |
| `LLM_PROVIDER` | `gemini` |
| `LLM_MODEL` | `gemini-2.5-flash` |
| `LLM_API_KEY` | present, 53 chars (no value, prefix or hash recorded) |
| `LLM_BASE_URL` | empty |
| `LLM_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` in the process environment | unset |
| `create_llm_provider(Settings())` | **raises `LLMError`**: `Provider 'gemini' is declared but not yet implemented. Use 'openai' (or any OpenAI-compatible endpoint via LLM_BASE_URL) or 'mock'.` |
| External LLM calls made during this run | **0** |

Two independent blockers, both configuration:

1. **No rotated replacement key is configured.** The local `.env` still holds the
   key from the session in which it was exposed, and this run's instruction was
   explicit: do not use that key, and make no external LLM call until a
   replacement is present. It is treated as compromised.
2. **Even with a valid key the provider cannot be constructed.** `gemini` is a
   declared-but-unimplemented provider name; the working path to Gemini is
   `LLM_PROVIDER=openai` + `LLM_MODEL=gemini-2.5-flash` +
   `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, exactly as the
   provider docstring says. The second blocker is therefore **not** the malformed
   model string this plan warned about; it is the provider name.

Consequences, stated rather than worked around:

* **Phases 2–5 are not executed.** No second-domain corpus was discovered,
  downloaded, ingested, chunked, embedded, indexed or evaluated, because the
  workflow's first step (`analyze_domain`) is LLM-backed and A6 requires every
  LLM-backed step to record a real provider identity. Falling back to the labelled
  development mock and calling it a real-LLM run is exactly what the plan above
  forbids, so nothing was run.
* **No second-domain benchmark artifact was created.** A draft instrument whose
  expected chunks cannot exist would be a fabricated artifact, and the plan
  requires every supported label to be grounded in a real corpus chunk.
* **A2/A3/A4 are therefore not evaluated.** A1/A6/A7 were evaluated in the form
  available to this run (below).
* The Phase-1 "one small structured-output call" was **not** made: the
  instructions permit it only once a rotated key is configured.

### Phase 9 — Claude Desktop connection (the other deliverable of this task)

Recorded in [`docs/mcp-integration.md`](mcp-integration.md) §2, with the reusable
checker
[`backend/scripts/check_claude_desktop_connection.py`](../backend/scripts/check_claude_desktop_connection.py),
which reads the client's own configuration file and launches the exact command it
contains.

| Item | Result |
|---|---|
| MCP server as configured | 18 tools, 1 resource, 3 resource templates; handshake `ragforge 0.1.0`, protocol `2026-07-28`; 1476 chars of server instructions |
| Transport hygiene | stdout carried **only** JSON-RPC (0 non-JSON lines); all diagnostics on stderr; **no listening socket** on the server process |
| Working-directory independence | launched with `cwd=C:\` it still resolved the real workspace, because the launcher and `Settings` use absolute paths |
| Claude Desktop configuration | `%APPDATA%\Claude\claude_desktop_config.json` written with a `ragforge` entry; pre-flight (`command`, `args[0]`, `DATA_DIR` all exist) passes |
| Automobile retrieval | `What is an engine?` → 3 ranked chunks from `https://en.wikipedia.org/wiki/Engine`, scores 0.66 / 0.6263 / 0.6064, provenance (`chunk_id`, `document_id`, `source_url`, `content_hash`) present, no filesystem path |
| Evidence by id | `chk_00fe4e273d0e` → `doc_90752e50cc19`, `chunk_index 35`, 1160 chars |
| Grounded answer | `status=grounded`, `grounding_state=ANSWERED`, 3 validated citations — **but** `generation.is_mock=true`, `generated_by=fallback`, `model=extractive-mock`: the answer *text* is the disclosed deterministic fallback, not a model |
| Unsupported question (terms absent) | `Describe the courtship display of the Australian lyrebird.` → `abstained`, `INSUFFICIENT_EVIDENCE`, `LOW_LEXICAL_ALIGNMENT`, **0 citations, 0 claims**, no model call |
| Off-domain question (generic terms) | `How many moons does Jupiter have?` → `grounded` / `SUFFICIENT`, 1 citation. This is the **pre-existing off-domain gate defect**, reproduced through MCP (§A5 is the criterion it violates), not a new finding: `docs/progress-report-2026-10-03.md`, open item in `docs/roadmap.md` |
| Acceptance sequence | 24 checks: **22 passed, 1 warning** (disclosed fallback generator), **1 known defect**, 0 failed |
| Real Claude Desktop host test | **NOT PERFORMED** — Claude Desktop is not installed on this machine (`%APPDATA%\Claude` did not exist, no registry uninstall entry, no install directory). Only protocol and configuration checks were run, and they are **not** reported as a host integration test |

### Phase 10 — Defect found and fixed during this run

`tests/test_v18_mcp_agentic.py::TestProtocolV18::test_real_stdio_subprocess_exposes_v18_tools`
wrote **into the developer's real workspace database**. The nested stdio launch
omitted `env`, and `mcp`'s `StdioServerParameters.env` is only *extra* variables
merged over `get_default_environment()`, which inherits a whitelist rather than
`os.environ` — so the child lost `DATA_DIR` and opened the real
`backend/data/ragforge.db`. Measured, not inferred: running that single test
added 2 `mcp_operations` rows (name `Stdio KB`) to the real database in three
separate invocations. The test now forwards the scratch variables explicitly and
**asserts** the database path the launched server resolved; after the fix the same
invocation leaves the real row count unchanged (45 → 45), and the suite runs
faster because the nested server no longer loads the real embedding model. See
[`docs/mcp-integration.md`](mcp-integration.md) §6.

Pollution this created in the real workspace, reported rather than quietly
deleted: **8** inert `mcp_operations` rows (`plan_*` + `op_*`, `kb_id=''`, status
`proposed`, `action=create_knowledge_base`, all named `Stdio KB`, created
2026-10-15) out of 45 total. No knowledge base, document, chunk, vector, benchmark
or review row was created. The table held 33 rows before this run.

### Phase 11 — Protection checks (before and after)

| Invariant | Before | After | Verdict |
|---|---|---|---|
| SQLite `integrity_check` / `foreign_key_check` | ok / ok | ok / ok | unchanged |
| Knowledge bases | 21 | 21 | unchanged |
| `kb_f278c283c748` documents / chunk rows | 11 / 812 | 11 / 812 | unchanged |
| `kb_f278c283c748` collection | green, 812 points | green, 812 points | unchanged |
| `kb_f278c283c748` `updated_at` | `2026-09-15T10:58:31Z` | same | unchanged |
| `mcp_operations` for the protected KB | 0 | 0 | unchanged |
| sha256 of all 15 files in `benchmarks/` | — | — | **all 15 identical** |
| Test-suite write into the real workspace | yes (defect) | no | **fixed** |

`validate_knowledge_base` on the protected KB returns `NOT_READY`, and did so
before this run: its 11 documents were serialised before the document `status`
field existed (11 `partial_documents`) and its chunks carry no `source_id` (1624
`provenance_gaps`, counted once for the chunk row and once for the vector
payload). Index integrity itself is clean: 812 measured points, 0
missing/orphan/stale vectors, 0 embedding mismatches, embedding identity
matching. The corpus was deliberately **not** rebuilt to make the verdict
prettier; the finding is recorded in
[`docs/mcp-integration.md`](mcp-integration.md) §7.

### Phase 12 — Verification

| Check | Result |
|---|---|
| `tests/test_v17_mcp_server.py`, `tests/test_v18_mcp_agentic.py`, `tests/test_v19_second_domain.py` | **162 passed** |
| Full backend suite (`pytest` in `backend/`) | **1481 passed**, 3 warnings |
| Frontend | not run: no frontend source changed (`frontend/tsconfig.tsbuildinfo` is a generated artifact and is excluded from the commit) |
| Real-workspace side effects of the suite | none after the Phase-10 fix (row count verified stable across a full MCP-suite run) |

### Status

* **Claude Desktop deliverable: complete up to the host test.** Configuration,
  protocol verification and the full acceptance sequence passed; the actual
  Claude Desktop UI test is blocked on the application not being installed and is
  handed back as documented manual steps in
  [`docs/mcp-integration.md`](mcp-integration.md) §2.
* **Real-LLM V19 deliverable: BLOCKED at the gate**, before any external call.
  Unblock by rotating the key and setting, in `backend/.env`:
  `LLM_PROVIDER=openai`, `LLM_MODEL=gemini-2.5-flash`,
  `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, plus the
  replacement `LLM_API_KEY`. Then re-run Phase 1 (one structured-output call) and
  continue to Phase 2 only if `is_mock=false`.
* **No claim is made anywhere in this document about a second domain's corpus,
  index, retrieval or grounding quality**, because none was built.
