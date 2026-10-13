# V18 — Agentic RAGForge: knowledge-base operations through MCP

**What this adds.** V17 exposed the *reading* half of RAGForge to MCP clients
(six read-only tools). V18 adds the *construction* half: an MCP client can take a
domain description, obtain a proposed domain specification, discover and assess
sources, plan a corpus, and then — only after an explicit, parameter-bound user
confirmation — create the knowledge base, ingest approved sources, ingest
user-provided documents, chunk/embed/index it, validate readiness from measured
evidence, and report the whole lifecycle back.

**What it is not.** There is no second orchestrator, queue, scheduler or job
runner. Every tool calls the same services the HTTP API calls
(`SourceQualityScorer`, `ingest_source`, the document intake pipeline, the
chunk/embed/index pipeline, the corpus integrity audit, the retrieval diagnostics
runner, `Repository`). What V18 adds is *control*: a recorded operation row, a
plan/execution split, and a confirmation gate that the model cannot satisfy on
its own behalf.

---

## 1. The operation lifecycle

```
 REQUEST            the client calls a tool
   │
 PLAN               plan_knowledge_base / analyze_domain / discover_sources /
   │                assess_sources → a row with status=proposed, which
   │                authorizes NOTHING (mutates=false)
   │
 USER CONFIRMATION  confirm_plan(plan_id, action, parameters_fingerprint)
   │                → status=confirmed + a single-use confirmation_id,
   │                  bound to one action and one exact parameter set, with an
   │                  expiry. A plan id is not a confirmation.
   │
 EXECUTION          create_knowledge_base / ingest_sources /
   │                build_knowledge_base / add_user_documents
   │                → the confirmation is consumed, status=running → done|error,
   │                  result_summary records what the service actually returned
   │
 VALIDATION         validate_knowledge_base → measured evidence, never a
   │                metadata claim (status/verdict READY | NOT_READY | UNKNOWN)
   │
 REPORT             get_operation_status → lifecycle stage, authorized
                    parameters, consumed confirmation, result summary, warnings,
                    and the underlying BuildRun / diagnostics record with stages
```

Two properties are load-bearing and are explicit in the payloads:

* **`mutates`** — every planning result says `false`, so a client can never
  confuse a plan with an execution.
* **`parameters_fingerprint`** — sha256 over the canonical JSON of the exact
  parameter set a confirmation would authorize. The caller confirms what it was
  shown; changing any parameter produces a different fingerprint and the call is
  refused (`CONFIRMATION_INVALID`) rather than silently executed.

Operations are stored in the SQLite table `mcp_operations` (see
[`app/schemas/agent_ops.py`](../backend/app/schemas/agent_ops.py)), next to the
rest of RAGForge's metadata, so an operator can audit agent-driven changes with
the same tooling they use for the UI. Statuses: `proposed`, `confirmed`,
`running`, `done`, `error`, `expired`.

### Idempotency and duplicate protection

* A KB name + domain that already exists is refused **without** burning the
  approval (`DUPLICATE_OPERATION`), and planning warns about it up front.
* Re-running an executed confirmation is refused: it was already consumed.
* Document intake skips content whose sha256 already exists in the KB and reports
  `duplicates_skipped` instead of creating a second document.
* A build is a rebuild of the documents it indexes (the same semantics as
  `POST /{kb_id}/index`): the re-indexed documents' vectors are deleted before
  the new ones are upserted and orphan points are swept afterwards. It does not
  silently delete or recreate the knowledge base.

---

## 2. Modules

| Module | Responsibility |
|---|---|
| [`app/mcp/server.py`](../backend/app/mcp/server.py) | MCP wiring only: 18 tools (6 read-only + 12 agentic), 4 read-only resources, stdio transport, stderr logging |
| [`app/mcp/planning.py`](../backend/app/mcp/planning.py) | Plan computation: what *would* happen, the exact action parameters, fingerprints, and the validation of URLs/documents/chunkers/strategies |
| [`app/mcp/operations.py`](../backend/app/mcp/operations.py) | The lifecycle itself: operation rows, confirmation issuance/validation/consumption, expiry, audit reads |
| [`app/mcp/agent_ops.py`](../backend/app/mcp/agent_ops.py) | The execution half of the bridge: creation, discovery, ingestion, intake, build, validation, comparison |
| [`app/mcp/bridge.py`](../backend/app/mcp/bridge.py) | The only RAGForge-facing code; the public surface is an explicit allow-list asserted by tests |
| [`app/schemas/agent_ops.py`](../backend/app/schemas/agent_ops.py) | `AgentOperation` / `OperationStatus` records |
| [`scripts/mcp_agent_workflow.py`](../backend/scripts/mcp_agent_workflow.py) | The live end-to-end workflow against the real workspace (see §6) |

---

## 3. Tools

Read-only (unchanged from V17, annotations `read_only_hint=true`,
`destructive_hint=false`): `list_knowledge_bases`, `get_knowledge_base_status`,
`search_knowledge_base`, `get_evidence`, `answer_from_knowledge_base`,
`evaluate_retrieval`.

Agentic (V18):

| Tool | Mutates | Needs confirmation | What it really does |
|---|---|---|---|
| `analyze_domain(domain?, purpose, target_audience, depth, kb_id?)` | no | no | Runs the existing domain analyzer and returns a **proposal**: subdomains, key concepts, entities, terminology, knowledge requirements, recommended source categories, plus `generated_by` and `is_mock`. Persists nothing. |
| `discover_sources(kb_id, provider, query, limit)` | yes (candidate rows) | no | Runs the existing discovery providers and the real quality scorer, registering candidates with their assessment. Never downloads, embeds or indexes. A failed provider is a capability error, not an empty success. |
| `assess_sources(kb_id, source_ids?, decision?, user_provided_only?)` | no | no | Returns the stored scores, component signals, weights, reasons, warnings and probe evidence per source. Unassessed sources are listed as unassessed, never scored 0. |
| `plan_knowledge_base(...)` | no (plan rows only) | no | Proposes the corpus, ingestion plan, chunking and retrieval configuration as one confirmable action per step, with exact parameters and fingerprints. |
| `confirm_plan(plan_id, action, parameters_fingerprint, note)` | yes (confirmation) | n/a | Records the user's approval of ONE action with ONE parameter set; issues a single-use, expiring `confirmation_id`. |
| `create_knowledge_base(confirmation_id, name, domain, purpose, target_audience, depth, source_mode)` | yes | **yes** | Creates an empty KB through the same route the API uses. Refuses a duplicate name+domain. Grants no capability claim. |
| `ingest_sources(confirmation_id, kb_id, source_ids)` | yes | **yes** | Downloads and parses APPROVED sources. REVIEW needs a human decision recorded in the API/UI first; REJECT is never overridden; user-provided material is not downloaded. Nothing is embedded. |
| `build_knowledge_base(confirmation_id, kb_id, chunker, target_size, overlap)` | yes | **yes** | Runs the existing chunk → embed → index pipeline and reports the real counts from the run record. |
| `add_user_documents(confirmation_id, kb_id, documents, index, chunker, target_size, overlap)` | yes | **yes** | Adds the user's own documents **as content** through the same validation/parse/hash/provenance path as an upload. Bound to the sha256 of every document. |
| `get_operation_status(operation_id, kb_id?)` | no | no | Lifecycle stage, authorized parameters, confirmation state, result summary, warnings, and the underlying run (e.g. the BuildRun with `chunking`/`embedding`/`indexing` stages). |
| `validate_knowledge_base(kb_id, include_vectors, include_diagnostics)` | no | no | Runs the corpus audit: documents parsed, chunk rows, vector points vs chunk rows, recorded embedding identity, and the full integrity audit. A measurement that could not be taken is `UNKNOWN`, never a pass. |
| `compare_retrieval_strategies(kb_id, benchmark_path?, strategies?, top_k?, label?, acknowledge_exploratory?, candidate_depth?)` | no (records a run) | no | Uses the existing diagnostics runner, gated on the instrument's lifecycle: an instrument authored for another KB, outside the allowed roots, or without completed human review is refused (the last only unless the caller explicitly acknowledges that no confirmatory claim follows). Nothing is promoted. |

Nothing was added for deletion, bulk operations, arbitrary SQL, shell execution
or unapproved network access: no such tool exists, and a test pins the bridge's
public method set so one cannot be added unnoticed.

---

## 4. Confirmation gates

Rules the implementation enforces (each has a test, and each was reproduced
against the live server — see §6):

1. **No confirmation, no execution.** `create_knowledge_base`, `ingest_sources`,
   `build_knowledge_base` and `add_user_documents` refuse with
   `CONFIRMATION_REQUIRED`.
2. **A plan is not a confirmation.** Passing a `plan_…` id as the confirmation is
   refused with `CONFIRMATION_INVALID`.
3. **Bound to the exact parameters.** Any difference between the confirmed
   fingerprint and the call's fingerprint is refused, and the error names both
   fingerprints.
4. **Bound to the action.** A confirmation issued for `ingest_sources` cannot run
   a `build_knowledge_base`.
5. **Single use.** The first execution consumes it; a replay is refused with the
   original consumption timestamp.
6. **Expiry.** A confirmation window that passes unused
   (`DEFAULT_CONFIRMATION_TTL`, 30 minutes) authorizes nothing; the row is marked
   `expired`.
7. **Bound to the content, not just the call.** `add_user_documents` is bound to
   the sha256 of every document, so different bytes with the same file name are
   refused.
8. **The plan can go stale.** A source downgraded from ACCEPT to REVIEW (or
   removed) after the confirmation was recorded is refused or reported when the
   ingest runs — the corpus state at execution time is what counts.
9. **The server cannot verify who approved.** `confirm_plan` records a note and
   the host client owns that responsibility; the tool says so in its description.
10. **No destructive recovery.** Nothing deletes or recreates a knowledge base
    automatically; the only cleanup in the workflow script is an explicit,
    operator-visible step.

---

## 5. Security boundaries

| Boundary | How it is enforced |
|---|---|
| Untrusted arguments | Schema-validated by the SDK and re-validated in the bridge: ≤ 20 documents per call, ≤ 8 MiB per document, ≤ 24 MiB per call, ≤ 50 URLs per plan, ≤ 20 discovered candidates, `top_k` 1–50, query ≤ 2000 chars, `target_size` 200–8000, `overlap` 0–1000 and `< target_size` |
| SSRF | Every URL goes through the existing SSRF validation (`app/utils/http_probe.py` / URL validation); unreachable or invalid URLs are kept visible as invalid candidates rather than fetched |
| Filesystem | No tool accepts a path. `add_user_documents` takes content; a `path` field is rejected with an argument error naming `content_text`. File names are sanitised (`../../../etc/passwd.md` → `passwd.md`) and written only under the KB's own upload directory |
| File type | Declared extension **and** magic bytes are checked (a mislabelled PDF is rejected), unsupported types are rejected inside the operation and reported per item |
| Secrets | No payload contains settings, keys, environment values or server configuration; a test asserts `api_key`/`secret`/`password` never appear |
| Network | Only two paths make outbound requests — `discover_sources` (probe/scoring, incl. arXiv) and `ingest_sources` (download) — both report it in `warnings`, and neither runs without the corresponding tool call |
| Retrieved text | Corpus text stays untrusted data: it is returned inside `text` fields with `content_warning`, and a test indexes "IGNORE ALL PREVIOUS INSTRUCTIONS…" and asserts it comes back as quotable evidence |
| Errors | Coded, actionable, leak-free; unexpected failures are logged on stderr and reported generically |
| Protected artifacts | The V18 tests use a scratch `DATA_DIR` and an in-memory store; `kb_f278c283c748` and every frozen benchmark artifact are never written |

---

## 6. Demonstrated end-to-end workflow

[`scripts/mcp_agent_workflow.py`](../backend/scripts/mcp_agent_workflow.py)
starts the real MCP server over stdio — the way a client does — and runs the whole
lifecycle against the **real workspace**: real SQLite, real Qdrant, real local
`all-MiniLM-L6-v2` embeddings, and the labelled development mock LLM because no
`LLM_API_KEY` is configured. It refuses to start unless the protected KB measures
exactly 812 points, measures it again at the end, and deletes only the scratch KB
it created.

```bash
cd backend
.venv\Scripts\python.exe scripts/mcp_agent_workflow.py                       # offline
.venv\Scripts\python.exe scripts/mcp_agent_workflow.py --external            # + real URLs
```

### Run A — offline run, 2026-10-14 (18 tools discovered, 0 problems)

| Step | Result as returned |
|---|---|
| `analyze_domain` | `is_mock=true`, `generated_by=mock:mock-1`, `persisted=false`; 4 subdomains, 4 key concepts, 4 recommended source categories; warning: "this domain specification came from RAGForge's development mock provider… treat it as a placeholder" |
| `plan_knowledge_base` (create) | `mutates=false`; one `create_knowledge_base` action; the KB count was identical before and after planning |
| gate: no confirmation | `[CONFIRMATION_REQUIRED]` |
| gate: plan id as confirmation | `[CONFIRMATION_INVALID] Confirmation 'plan_…' is not known to this workspace` |
| gate: parameters changed after approval | `[CONFIRMATION_INVALID] Confirmed fingerprint 14e011aca20d, call fingerprint b53ac3efff05` |
| gate: replay after execution | `[CONFIRMATION_INVALID] Confirmation conf_… was already used (2026-10-13T21:35:57…)` |
| `create_knowledge_base` | `kb_109e3d6447fb`, `status=draft`, 0 documents, `search: false` with basis "0 chunk row(s) and 0 document(s) in the corpus" |
| `plan_knowledge_base` (documents) | proposed 1 document with its sha256; chunking `section-aware 700/80` |
| gate: document bytes changed after approval | `[CONFIRMATION_INVALID]` and nothing was written |
| `add_user_documents` | `accepted=1`, `indexed=1`, `chunks_created=4`, `vectors_indexed=4`, `content_hash=0c70f816…`, `user_provided=true`, `decision=ACCEPT`, `score=0.95` |
| `build_knowledge_base` | run `build_b6999fca2132`, statuses `chunking → embedding → indexing`, `run_status=done`, `chunk_rows=4`, `vector_points=4`, `vectors_match_chunks=true` |
| `validate_knowledge_base` | `verdict=READY`; basis: 1 document, 4 chunk rows, 4 vector points measured, integrity `HEALTHY`; all 11 integrity counters 0, `integrity_unknown=[]`; diagnostics refused with the real reason ("no FROZEN benchmark version exists for this knowledge base") |
| `search_knowledge_base` | `applied_strategy=dense`, model `sentence-transformers/all-MiniLM-L6-v2 (384d)`; #1 `Radiator` 0.7053, #2 `Coolant circuit` 0.6369, #3 `Engine cooling system` 0.4999 |
| `get_evidence` | `found=1`; a chunk with `section_path=Engine cooling system > Radiator`, `chunking_strategy=section-aware`, `user_provided=true`, `trust_score=0.95`, `source_url=upload://engine-cooling-notes.md` |
| `answer_from_knowledge_base` (in corpus) | `status=grounded`, `grounding_state=ANSWERED`, 3 validated citations (`validation=provenance_valid`), 3 claims all `supported`; `generation={generated_by: llm, model: mock/mock-1, is_mock: true}` |
| `answer_from_knowledge_base` (out of corpus) | `status=abstained`, `grounding_state=INSUFFICIENT_EVIDENCE`, 0 citations, reason naming the missing question terms |
| `compare_retrieval_strategies` | `permitted=false` — "benchmark 'automobile-engineering-baseline-v1' was authored for knowledge base 'kb_f278c283c748'; scoring it here would measure a corpus it was not written against" |
| `get_operation_status` ×3 | creation/intake/build all at `lifecycle_stage=report`, `status=done`, `confirmation_used=true`, with the build's stages listed |
| cleanup | scratch KB row and its Qdrant collection deleted; 21 KBs before and after; protected KB 812 points before and 812 after |

### Run B — with `--external` (real network), same day

| Step | Result as returned |
|---|---|
| `discover_sources` | 2 real candidates registered from Wikipedia URLs, both `decision=REVIEW`, `trust_score=0.5475`, `assessed_by=heuristic-v1`, signals/weights/reasons stored (`authority 0.6`, `relevance 0.3`, `recency 1.0` from a real `Last-Modified`, …) |
| `assess_sources` | `total=3`, `counts_by_decision={ACCEPT: 1, REVIEW: 2}`, `unassessed=[]` — the REVIEW decisions are the honest result for a general encyclopedia page, not an error |
| human decision (via the API's own path) | the two REVIEW sources set to ACCEPT by the operator — the MCP surface deliberately cannot do this |
| `plan_knowledge_base` → `ingest_sources` | real downloads (190 818 and 191 465 bytes), `2 ingested, 0 duplicate skipped, 0 failed` |
| `build_knowledge_base` | `chunk_rows=118`, `vector_points=118`, `vectors_match_chunks=true` over 3 documents |
| `validate_knowledge_base` | `verdict=READY`, integrity `HEALTHY` |
| cleanup | scratch KB removed; protected KB still 812 points |

Both runs exited 0 with an empty `problems` list. The full JSON records were
produced by `--json-out` and are not committed (they are run artifacts); every
number above is copied from them.

---

## 7. Tests

[`tests/test_v18_mcp_agentic.py`](../backend/tests/test_v18_mcp_agentic.py) — 95
tests, hermetic (scratch `DATA_DIR`, in-memory cosine store, hashing embeddings,
mock LLM, deterministic probe/arXiv doubles, and a local-fixture download
replacement so the **real** parser still runs):

* **Protocol**: all 18 tools discoverable, schemas published, annotations
  separating reads from writes, consequential tools requiring a `confirmation_id`
  argument, and a **real stdio subprocess** that handshakes, lists the 18 tools
  and calls V18 tools.
* **Planning**: creation/extension plans mutate nothing, parameters and
  fingerprints are stable, URL previews are validated, REVIEW sources are
  excluded from ingest actions while ACCEPT sources get one, invalid/duplicate
  documents are reported, and unknown chunkers/strategies/overlap are refused.
* **Confirmation gates**: every rule in §4, including expiry (via an injected
  clock) and action scoping.
* **Execution**: empty KB creation, duplicate refusal without burning the
  approval, ingestion of accepted sources, rebuild counts, build without
  documents reporting the API's own reason, and a source downgraded after
  confirmation being refused.
* **Documents**: intake parses/indexes, preserves section provenance, skips
  duplicates, refuses different bytes, rejects unsupported/mislabelled files,
  neutralises path-shaped names, cannot read the filesystem, and enforces the
  count/size caps.
* **Status/validation**: plan vs confirmed vs executed reporting, polling a
  foreign run id, unknown ids listing the id families, and an unreachable vector
  store reporting `UNKNOWN` rather than a pass.
* **Comparison**: wrong-KB instrument, out-of-root path, missing file, unscorable
  instrument, provisional instrument needing explicit acknowledgement, and a run
  that neither promotes nor writes anything.
* **Surface/safety**: the bridge's public surface equals its allow-list, only the
  four allow-listed actions can be authorized, no destructive tool exists, no
  secret or path appears in any payload, retrieved text stays untrusted data, and
  no operation can mutate a protected artifact.

The V17 suite now holds 54 tests: most are unchanged, and the ones that encoded
the old invariant were restated for the intended change (the surface is no longer
read-only-only, stdio exposes 18 tools, and "nothing can change state except
through the four allow-listed confirmed actions" is now the checked property).
Both suites pass together: **149 tests**, and the full backend suite is **1468
passed**.

---

## 8. Limitations (stated, not hidden)

* **stdlib transport only.** Uploading a file through MCP is not possible because
  stdio has no file channel: the host reads the bytes and passes them as content.
  The tool documents this, and `add_user_documents` refuses path-shaped
  arguments instead of inventing one.
* **No cancellation, no background jobs.** Every operation runs synchronously
  inside the tool call, so there is no safe cancel to expose; `get_operation_status`
  reports what a run did, not a live progress ticker.
* **Confirmation is a protocol contract, not proof of a human.** The server
  records that a confirmation was requested, with the note the client supplied.
  A client that calls `confirm_plan` immediately after planning has not involved a
  user; the host is responsible for that, and the tool description says so.
* **`discover_sources` writes candidate rows.** It is annotated as a write (it
  registers candidates and scores them) even though it downloads nothing; it does
  not require a confirmation because it cannot change a corpus.
* **The article/encyclopedia case still needs a human.** A general reference page
  scores 0.5475 → REVIEW. V18 makes that decision explicit and auditable; it does
  not pretend the automation is better than it is.
* **`analyze_domain` is a placeholder without an LLM key.** With no provider
  configured it returns the labelled development mock (`is_mock=true`), and the
  script prints that rather than presenting the proposal as research output.
* **Retrieval quality is unchanged.** V18 adds construction control, not ranking
  improvements; `compare_retrieval_strategies` inherits the existing lifecycle
  gates and promotes nothing.
* **Answer-quality metrics remain unavailable.** The answer benchmark still needs
  human-authored references and review; no V18 path fabricates them, turns
  UNKNOWN into 0, or makes a confirmatory claim from the provisional instrument.
