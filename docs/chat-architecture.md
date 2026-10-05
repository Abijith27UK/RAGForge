# Chat Architecture

How the grounded chat layer is built, and why it is a thin addition rather than a
second answer pipeline.

Related: [grounding-and-citations.md](grounding-and-citations.md),
[answering-architecture.md](answering-architecture.md),
[architecture.md](architecture.md).

---

## 1. What chat adds — and what it deliberately does not

`POST /knowledge-bases/{kb_id}/chat` is a **front end for the existing answering
pipeline**, not a new one. Every turn delegates to `AnsweringService.answer`,
so chat and `/answer` cannot drift apart in behaviour: an answer reached through
chat is grounded exactly like one reached directly, using the same gate, the
same citation validation and the same prompt.

Chat adds precisely two things:

1. **Conversation memory** — minimal, and explicitly *not* knowledge.
2. **Observability** — an `AnswerRun` record per turn.

Anything else (answer modes, retrieval strategies, policies) is already
parameterised on `/answer` and is reused rather than reimplemented.

```
                    ┌─────────────────────────────┐
   POST /chat ──────►  GroundedChatService        │
                    │  1. resolve conversation   │
                    │  2. resolve references     │
                    │  3. store user message     │
                    │  4. delegate ↓             │
                    │  5. store reply + AnswerRun│
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  AnsweringService.answer()  │  ← unchanged
                    └──────────────┬──────────────┘
                                   ▼
        QueryProcessor → Retrieval → EvidenceSelector → GroundingGate
                        → AnswerGenerator → CitationValidator → Answer
```

---

## 2. Conversation memory

### 2.1 It is not knowledge

This is the most important constraint in the design. Conversation history is
stored in `messages`, entirely separate from `documents` and `chunks`, and is:

- never chunked;
- never embedded;
- never indexed;
- never retrieved by the retriever.

`Message.used_as_knowledge` exists and is permanently `False`. The field is
there so the UI and any audit can *assert* this rather than take it on trust.

### 2.2 What history is used for

One purpose: resolving references like *"What about the previous case?"* into a
standalone question. `resolve_reference` borrows topic terms from earlier **user**
turns and appends them as explicit context.

Rules that keep this safe:

| Rule | Why |
|---|---|
| Only **user** messages are used | The assistant's own words must never be fed back as if they were fact |
| Original wording is preserved verbatim at the front | The user can always see what was actually asked |
| The borrowed source is recorded in `Message.resolved_from` | The transformation is auditable, not invisible |
| At most `MAX_HISTORY_TURNS` (3) prior turns are consulted | A long thread cannot drag unrelated topics into a question |
| Only reference-shaped turns are expanded | A standalone question is passed through byte-for-byte unchanged |

The expanded question is then answered **only** from retrieved KB evidence, or
not at all. Expanding the question changes what is *searched for*; it never
changes what is *allowed as an answer*.

### 2.3 Reference detection

`is_reference_turn` matches an explicit, conservative set of shapes: `what about
…`, `tell me more`, `explain that/this/it`, a bare follow-up, `the previous/last
…`. It is deliberately narrow — a real question that merely contains "that" is
not treated as a reference, because over-triggering would inject unrelated
context into a well-formed question.

---

## 3. Query processing

`QueryProcessor` (`services/answering/query_processor.py`) is the pipeline's
first stage. It is deterministic and LLM-free.

### 3.1 The trace

Every turn produces a `QueryTrace`:

| Field | Meaning |
|---|---|
| `original_query` | Byte-for-byte what the caller sent |
| `normalized_query` | NFC-normalised, whitespace collapsed |
| `rewritten_query` | **null** when no rewrite happened |
| `subqueries` | Empty — decomposition is not implemented |
| `processor` / `processor_version` | Which processor, which revision |
| `query_type` / `nature` | Answer shape / kind of turn |
| `transformations` | Ordered list of transformations actually applied |
| `warnings` | Non-fatal disclosures (e.g. decomposition not implemented) |
| `timing_ms` | Measured |

`rewritten_query` is `null` rather than `""` when unchanged, so "we did not
rewrite" can never be confused with "we rewrote it to nothing".

### 3.2 Turn nature

| Nature | Meaning | Gate behaviour |
|---|---|---|
| `knowledge` | A real question | Normal evidence assessment |
| `conversational` | "hi", "thanks" | Short-circuits to `INSUFFICIENT_EVIDENCE` |
| `non_knowledge` | "help", "what can you do" | Short-circuits to `INSUFFICIENT_EVIDENCE` |
| `underspecified` | No searchable content | Normal evidence assessment |
| `multi_hop` | "how does X affect Y" | Normal assessment + warning that multi-step retrieval is not implemented |

Detection is an explicit rule set. Accuracy is not claimed; the matching rule is
recorded in `classification_signals`.

### 3.3 Conservatism

- **No blind rewriting.** With `expand=False` (the default) the query sent to
  retrieval is the normalised question, `rewritten_query` stays `null` and
  `transformations` stays empty — a caller can *prove* the question was not
  altered.
- **Expansion is opt-in and additive.** Terms are added as an extra retrieval
  query; the original question is never replaced.
- **Decomposition is not implemented.** A multi-part question is retrieved as
  one query, and the trace records that as a warning rather than pretending.
- **When disabled, the query passes through unchanged.**
  `passthrough_trace` records `enabled=false` with the original query intact.

---

## 4. Answer runs

`AnswerRun` is the per-turn observability record, stored in `answer_runs`. It is
deliberately smaller than `AnswerTrace`: it is the row you list and aggregate,
while full evidence text and raw generator output stay in the trace.

Recorded: ids (`answer_run_id`, `retrieval_run_id`, `kb_id`, `conversation_id`,
`answer_id`, `answer_trace_id`), the query, strategy and retrieval parameters,
selected evidence ids, the grounding decision + reasons + sufficiency, citation
and claim counts, **measured** latencies, the model/provider, and every version
string (`prompt_version`, `answerer_version`, `query_processor`,
`query_processor_version`, `evidence_selector`, `grounding_gate`).

**A latency that was not measured is `null`, never `0`.** A stage that was
skipped has no latency, and reporting `0 ms` would imply a measurement that never
happened.

---

## 5. API

Base path `/api/knowledge-bases`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/{kb_id}/chat` | One grounded chat turn |
| `GET` | `/{kb_id}/conversations` | List threads (summaries, no message bodies) |
| `POST` | `/{kb_id}/conversations` | Create a thread explicitly |
| `GET` | `/{kb_id}/conversations/{cid}` | Thread + messages, oldest first |
| `DELETE` | `/{kb_id}/conversations/{cid}` | Delete a thread |
| `GET` | `/{kb_id}/answer-runs` | Answer-run history, newest first |
| `GET` | `/{kb_id}/answer-runs/{run_id}` | One answer run |

### 5.1 Request

```json
{
  "message": "What happens to GM when the center of gravity rises?",
  "retrieval_strategy": "hybrid",
  "retrieval_params": { "top_k": 8 },
  "conversation_id": "conv_…",
  "answer_mode": "abstain_if_unsupported"
}
```

`retrieval_strategy` and `retrieval_params` are optional; when omitted the KB's
persisted retrieval configuration applies, exactly as in `/answer`.

### 5.2 Response

```json
{
  "answer": "The metacentric height GM must be positive…",
  "answer_id": "ans_…",
  "status": "grounded",
  "citations": [ { "citation_id": "cite_ev_0001", "evidence_id": "ev_0001",
                   "source_title": "Ship Stability Notes", "source_type": "pdf",
                   "page_number": 23, "section_path": "Initial Stability",
                   "content_hash": "a1b2…", "validation": "provenance_valid" } ],
  "grounding": {
    "state": "ANSWERED",
    "decision": "ANSWER",
    "sufficient": true,
    "confidence": "high",
    "reason_code": "SUFFICIENT",
    "reason": "Evidence status: SUFFICIENT. Basis: 2 retrieved chunk(s) from 1 document(s); …",
    "evidence_count": 2,
    "document_count": 1,
    "signals": [ { "name": "lexical_alignment", "measured": true, "value": 0.6, … } ]
  },
  "evidence": [ { "evidence_id": "ev_0001", "chunk_id": "ch…", "content": "…", … } ],
  "retrieval_run_id": "rr_…",
  "answer_run_id": "arun_…",
  "answer_trace_id": "atr_…",
  "query_trace": { "original_query": "…", "rewritten_query": null,
                   "processor": "heuristic-rules", "processor_version": "v7.2",
                   "nature": "knowledge", "transformations": [], "timing_ms": 0.074 },
  "conversation_id": "conv_…",
  "user_message_id": "msg_…",
  "assistant_message_id": "msg_…",
  "generation": { "generated_by": "llm", "model": "…", "is_mock": false,
                  "degraded": false, "generator_status": "live" },
  "warnings": []
}
```

Nothing about the retrieval or grounding process is hidden — that is the point of
the endpoint. `evidence` carries full provenance so the UI can render source
cards without a second retrieval, and `warnings` carries every disclosure
(injection markers, generator fallback, unmeasured heuristics).

### 5.3 Errors

| Code | Cause |
|---|---|
| `400` | Unknown answer mode, unknown strategy, bad retrieval params |
| `404` | Unknown KB, unknown conversation, unknown answer run |
| `422` | Malformed body (missing/empty `message`) |
| `503` | Retrieval or generation backend failure |

---

## 6. Conversation scoping

A conversation belongs to exactly **one** knowledge base. Using a conversation id
against a different KB returns `404` rather than silently answering from the
wrong corpus — grounding must never be attributed to the wrong knowledge base.

---

## 7. Storage

Four tables, all keyed by `kb_id` and all removed when the KB is deleted:

| Table | Contents |
|---|---|
| `conversations` | Thread metadata (`id`, `kb_id`, `title`, `updated_at`) |
| `messages` | User + assistant turns with answer/trace/run links |
| `answer_runs` | Per-turn observability record |
| `answers` / `answer_traces` | Final answer and full audit trail (V7.1) |

`Message.resolved_from` records which prior message a reference was resolved
from; `Message.used_as_knowledge` is permanently `false`.

---

## 8. Frontend

`/knowledge-bases/{id}/chat` is a **three-pane investigation interface**, not a
chat clone:

- **Left** — knowledge base and corpus facts, retrieval strategy selector,
  conversation list.
- **Centre** — the conversation. Every answer shows its grounding state,
  answer text (or the abstention message), per-claim support badges, numbered
  citations, and an expandable *"Why this answer?"*.
- **Right** — evidence and sources. Clicking a citation or an evidence item
  opens the exact supporting chunk with its full provenance.

### 8.1 How the UI stays honest

- The grounding state is a categorical label — **no fake confidence percentage**
  anywhere.
- A missing page renders **"page not recorded"**, never a guess.
- A mock or fallback generator is labelled, including a disclosure banner when
  the configured generator was unavailable.
- *"Why this answer?"* shows the real gate signals **including the ones that were
  not measured, and why**.
- Scores are shown with their scale and explicitly described as pool-relative,
  not probabilities.
- When evidence is insufficient the UI says *"I couldn't find enough information
  in this knowledge base to answer that reliably"* plus the actual reason — it
  never falls back to general knowledge.

### 8.2 Historical turns

Reopening a stored conversation rehydrates each turn from its stored answer and
answer run. Anything that genuinely cannot be recovered — a query trace from
before traces were stored — is marked unavailable rather than reconstructed from
memory, and the response is labelled `HISTORY_UNAVAILABLE`. A past turn must
never display a fabricated trace.

---

## 9. Known limitations

1. **No streaming.** A turn completes before the response returns; long
   retrievals show a progress state.
2. **No regeneration UI.** `AnswerPolicy.regenerate_attempts` exists but no
   client control is exposed.
3. **Conversation history is display-only beyond reference resolution.** It
   cannot be searched, cited or used as evidence.
4. **Reference resolution is heuristic and lexical** — it borrows topic terms, so
   a pronoun referring to a *computed value* may not resolve well.
5. **No multi-turn answer reuse.** Each turn retrieves independently; the prior
   answer's evidence is not carried forward.
6. **Prompt decomposition for multi-part questions is not implemented.**
7. **Query expansion is opt-in** and currently only reachable programmatically
   (`expand_query`), not from the UI.