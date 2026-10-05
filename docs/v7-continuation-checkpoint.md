# V7 continuation checkpoint — Grounded Knowledge Assistant

Written at the end of the session that turned RAGForge from a retrieval system
into a **grounded knowledge assistant**. **Read this before doing anything
else** so the state is not re-derived from scratch.

Companion docs:
[grounding-and-citations.md](grounding-and-citations.md) (the rules),
[chat-architecture.md](chat-architecture.md) (chat + conversation memory),
[answering-architecture.md](answering-architecture.md) (pipeline design),
[v6-continuation-checkpoint.md](v6-continuation-checkpoint.md) (retrieval layer),
[verification.md](verification.md) (commands + bugs).

---

## Repository state

* Branch: `Dev_1_midterm`. Working tree contains **V6, V7 and V7.2 changes
  uncommitted** (V1–V5 are committed). Do not bundle tracked-generated-file
  housekeeping (`frontend/tsconfig.tsbuildinfo`, `package-lock.json` autocrlf
  noise, `.qdrant-initialized`) into a feature commit — handle separately.
* Backend tests: **511 passed, 0 failed** (244 V1–V5 + 74 V6 + 193 V7/V7.2).
* `python -m compileall -q app tests` → exit 0.
* Frontend: `npx tsc --noEmit` → clean; `npm run build` → **16 routes**, exit 0.
* Qdrant: **14 collections** before and after the live smoke (13 real KB
  collections + pre-existing `kb_live_itest`). The smoke created its own scratch
  KB, verified it, deleted it → back to 14.
* Frozen benchmarks: `git diff --stat HEAD -- benchmarks/` is **empty**.
* No real KB was modified or deleted; no data-impacting operation was performed.

---

## What existed before this phase (do not rebuild)

* **V1–V6**: FastAPI + Next.js + SQLite + Qdrant; ingestion
  (PDF/PPTX/DOCX/TXT/MD/HTML) with provenance; chunking; embedding; source
  quality; benchmark lifecycle; the retrieval registry
  (`dense | bm25 | hybrid | hybrid_reranked`), fusion/rerank/diversity,
  `RetrievalParams` persistence, `RetrievalRun` observability. 318 tests.
* **V7.1 (previous session)**: the grounded answering pipeline —
  `QueryProcessor`, evidence assembly, `EvidenceGate`, `AnswerGenerator`,
  prompting + injection scan, citation validation, `AnswerTrace`,
  `POST /answer`. 96 tests → 414 total.

This phase **extended** that pipeline behind new interfaces. It did not
redesign it, and every pre-existing test still passes unchanged.

---

## What this phase implemented

| # | deliverable | file(s) |
|---|---|---|
| 1 | `QueryTrace` (original/normalized/**rewritten**/**subqueries**/processor/**processor_version**/warnings/**timing_ms**) + `QueryNature` detection (conversational, non-knowledge, underspecified, multi-hop) + opt-in additive expansion + `passthrough_trace` for disabled processing | `app/schemas/answer.py`, `app/services/answering/query_processor.py` |
| 2 | `EvidenceSelector` ABC + `ProvenancePreservingSelector` (behaviour identical to `build_evidence_set`); `Evidence.source_type` | `app/services/answering/evidence.py` |
| 3 | `AnswerGenerator` gains `version`, `unavailable`, `unavailable_reason`, `describe()`; `UnavailableAnswerGenerator`; **`FallbackAnswerGenerator` makes fallback explicit** (disclosure on the answer, never silent) | `app/services/answering/generator.py` |
| 4 | `Citation` extended: `source_title`, `source_type`, `page_number`, `slide_number`, `section_path`, `content_hash` | `app/schemas/answer.py`, `app/services/answering/validation.py` |
| 5 | `GroundingGate` ABC + **five explicit states** (`ANSWERED`, `PARTIALLY_SUPPORTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, `NO_RELEVANT_EVIDENCE`) via exhaustive `GATE_TO_GROUNDING`; non-knowledge turns short-circuit | `app/services/answering/gate.py` |
| 6 | `Conversation`, `Message`, `ConversationSummary`, `ConversationDetail`, `AnswerRun` schemas | `app/schemas/answer.py` |
| 7 | `GroundedChatService`: reference resolution, conversation persistence, `AnswerRun` record; delegates to `AnsweringService.answer` | `app/services/answering/chat.py` |
| 8 | `POST /chat` + conversation + answer-run endpoints | `app/api/routes_chat.py`, `app/main.py` |
| 9 | Tables `conversations`, `messages`, `answer_runs` + repo methods + `delete_kb` cleanup | `app/repositories/sqlite_repo.py` |
| 10 | Service wired to the new interfaces (`process_with_trace`, `selector.select`, `assess_with_trace`) | `app/services/answering/service.py` |
| 11 | 3-pane grounded chat UI + client types/methods + nav entry | `frontend/src/app/knowledge-bases/[id]/chat/page.tsx`, `lib/api.ts`, `AppShell.tsx` |
| 12 | 99 new tests (65 unit + 34 API) | `tests/test_answering_v7_2.py`, `tests/test_chat_api_v7.py` |
| 13 | Docs: `grounding-and-citations.md`, `chat-architecture.md` (new); `architecture.md`, `roadmap.md`, `verification.md`, `README.md` (updated) | `docs/` |

---

## APIs added

```
POST /api/knowledge-bases/{kb_id}/chat
     body: {message, retrieval_strategy?, retrieval_params?,
            conversation_id?, answer_mode?}
     200 → {answer, answer_id, status, citations, claims, grounding, evidence,
            retrieval_run_id, answer_run_id, answer_trace_id, query_trace,
            conversation_id, user_message_id, assistant_message_id,
            generation, warnings, created_at}
     404 unknown KB / conversation · 400 bad mode/strategy · 422 malformed · 503 backend

GET    /api/knowledge-bases/{kb_id}/conversations              → summaries
POST   /api/knowledge-bases/{kb_id}/conversations              → create thread
GET    /api/knowledge-bases/{kb_id}/conversations/{cid}        → thread + messages
DELETE /api/knowledge-bases/{kb_id}/conversations/{cid}        → 204
GET    /api/knowledge-bases/{kb_id}/answer-runs                → history, newest first
GET    /api/knowledge-bases/{kb_id}/answer-runs/{run_id}       → one run
```

All pre-existing endpoints unchanged and backwards-compat tested.

---

## Schema changes

New SQLite tables (all keyed by `kb_id`, all removed on `delete_kb`):

| Table | Columns |
|---|---|
| `conversations` | `id` PK, `kb_id`, `created_at`, `updated_at`, `data` |
| `messages` | `id` PK, `conversation_id`, `kb_id`, `created_at`, `data` |
| `answer_runs` | `id` PK, `kb_id`, `created_at`, `data` |

New Pydantic models: `QueryTrace`, `QueryNature`, `GroundingState`,
`GATE_TO_GROUNDING`, `AnswerRun`, `Message`, `MessageRole`, `Conversation`,
`ConversationSummary`, `ConversationDetail`, `ChatRequest`, `ChatResponse`.
Extended: `Citation` (+6 provenance fields), `Evidence` (+`source_type`),
`EvidenceAssessment` (+`grounding_state`), `AnswerTrace` (+`query_trace`),
`AnswerGenerator` (+`version`, `unavailable`, `unavailable_reason`, `describe`).

**No existing table or column was altered**, so existing databases keep working
(`CREATE TABLE IF NOT EXISTS`).

---

## Key behavioural contracts (tests depend on these)

* `rewritten_query` is **`null`** when no rewrite happened — never `""`.
* Query processing disabled ⇒ original query passes through byte-for-byte,
  `enabled=false` recorded.
* Query decomposition is **NOT IMPLEMENTED**; a multi-part question is retrieved
  as one query and the trace carries that as a warning.
* Conversation history is used for **reference resolution only**. It is never
  embedded, indexed or retrieved, and `Message.used_as_knowledge` is permanently
  `False`.
* A conversation is scoped to **one** KB; using its id against another KB is 404.
* **Unmeasured latency is `null`, never `0`.** A skipped stage has no latency.
* Gate thresholds unchanged: `MIN_ALIGNMENT_TO_ANSWER=0.34`,
  `MIN_ALIGNMENT_TO_PARTIAL=0.15`, `ASPECT_MIN_COVERAGE=0.5`, `TRUST_LOW=0.3`,
  containment dedup 0.9, lexical support 0.34.
* Abstention in `abstain_if_unsupported` mode makes **no** generator call.
* There is **no silent fallback** from a real LLM to a mock: the fallback is
  disclosed via `unavailable`, an answer-level warning and `generator_status`.
* `PARTIAL_ANSWER` does **not** abstain — it answers what is supported and states
  what is missing.
* A claim citing one valid **and** one invalid evidence id is downgraded, never
  reported as fully supported.
* The recorded validation action always reflects what actually happened (a claim
  kept under `downgrade` logs `downgrade`, not `remove_claim`).
* Missing page/slide/section stay `None` → UI renders "not recorded".

---

## Tests

**511 passed, 0 failed** (full suite, run in two chunks).

* `tests/test_answering_v7_2.py` — **65 unit**: `QueryTrace` (pass-through,
  conversational/non-knowledge/underspecified/multi-hop detection, no false
  positives on short real questions, expansion opt-in), `EvidenceSelector`
  (provenance preserved, nothing invented, no silent score floor),
  `GroundingGate` (all five states, mapping exhaustiveness, non-knowledge
  short-circuit, underspecified still using evidence), generator honesty
  (fallback disclosure, mock labelling, no secrets), `Citation` fields,
  validation cases 1–10, `AnswerRun`, conversation isolation.
* `tests/test_chat_api_v7.py` — **34 API**: response shape, conversation scoping,
  grounding states, abstention skips generation, reference resolution recording,
  history never used as knowledge, citation validation cases, prompt injection,
  answer runs (measured latencies, newest-first, unmeasured `None`), errors
  (404/400/422), backwards compatibility.
* Pre-existing: `test_answering_v7.py` (73), `test_answer_api_v7.py` (23),
  V6 retrieval (74), V1–V5 (244) — all unchanged and passing.

### Bugs found and fixed this phase

| # | Bug | Fix |
|---|---|---|
| 1 | `test_incremental_documents.py` permanently replaced the real `bm25` retriever in the process-global registry, breaking `test_retrieval_v6.py` depending on run order (8 failures in the full suite) | Added `snapshot_retrievers()` / `restore_retrievers()`; both polluting tests now restore the registry in a `finally` block |
| 2 | The mock provider answered "What happens to GM when the center of gravity rises?" with a **propulsion** sentence | `len(term) > 3` substring matching dropped "GM" and kept the stop-word "when". Now uses the stop-word-aware `extract_terms` and ranks sentences by term-hit count |
| 3 | A claim citing one valid and one fabricated evidence id was reported as fully **supported** | Downgraded to `partially_supported` with the dropped ids named |
| 4 | Under `policy=downgrade` the action log recorded `remove_claim` although the claim was kept | Log now records the action actually taken |
| 5 | `_nature_of` reported all-stop-word phrases ("who are you") as *underspecified* instead of *non-knowledge* | Conversational/non-knowledge patterns are checked before the content-term check |
| 6 | The underspecified check counted non-topic words, so "What about the previous case?" looked specific | `_NON_TOPIC` exclusion; acronyms (GM, KG) no longer filtered by length |
| 7 | The gate short-circuited underspecified turns, masking the true `NO_EVIDENCE` reason on an empty KB | Only conversational/non-knowledge turns short-circuit |

---

## Verification commands

```bash
cd backend
./.venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/test_retrieval_integration.py -p no:randomly
./.venv/Scripts/python.exe -m pytest tests/test_retrieval_integration.py -q -p no:randomly
./.venv/Scripts/python.exe -m compileall -q app tests

cd ../frontend
npx tsc --noEmit
npm run build

# live smoke (creates and deletes its own scratch KB)
cd ../backend
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8012
./.venv/Scripts/python.exe scripts/smoke_chat_v7.py http://localhost:8012
```

---

## What remains / known limitations

* **No semantic entailment.** Claim support is a lexical-overlap heuristic, and
  every claim says so. No NLI model is used or claimed.
* **Conflict detection is a heuristic** (`term = value` across documents).
  Prose-form contradictions are not detected.
* **Prompt-injection detection is best-effort** — a marker scan, not a filter.
* **Query decomposition is not implemented**; multi-part questions are retrieved
  as one query and disclosed as a warning.
* **Query expansion is opt-in and programmatic only** (`expand_query`), not
  exposed in the UI.
* **Answer quality is still unmeasured.** There is no answer benchmark, so
  "grounded and cited" is demonstrated, not scored.
* **Naval Architecture answer quality is unmeasured** — no ground truth exists
  for that domain and none is invented.
* No streaming, no regeneration UI, no multi-turn answer reuse.
* History rehydration cannot recover evidence text (stored answers keep citation
  snippets and chunk ids); it is labelled `HISTORY_UNAVAILABLE` rather than
  filled in.

---

## Frozen artifact integrity

* `benchmarks/` — **untouched**. `git diff --stat HEAD -- benchmarks/` is empty.
* V1–V5 frozen benchmark artifacts — unmodified.
* No experiment was executed in this phase.
* No existing knowledge base was deleted or modified.
* The only data-touching operation was the live smoke's own scratch KB, created
  and deleted within the script; Qdrant returned to 14 collections.

---

## Next recommended phase

**Evaluating the grounded answerer.** Everything needed already exists: real
chunks, real citations, machine-readable grounding states and per-turn
`AnswerRun` records. The missing piece is measurement — a human-reviewed
answer benchmark (question → required evidence chunk ids → acceptable answers),
so grounded-ness and abstention behaviour can be scored rather than asserted.

Suggested order:

1. Answer-quality metrics: citation precision/recall against required evidence,
   abstention accuracy (does it abstain when it should, and answer when it can?).
2. A reviewed benchmark for **one** existing domain (Automobile Engineering,
   which already has reviewed questions) before attempting Naval Architecture.
3. Report grounding-state distribution and failure modes over real questions.
4. Only then consider decomposition, expansion in the UI, or streaming.

Explicitly still out of scope: TurboVec, MCP, autonomous corpus optimization,
automatic benchmark generation, OCR, CSV/XLSX ingestion, authentication, cloud
deployment, multi-user permissions, agent tool execution.