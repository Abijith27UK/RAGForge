# RAGForge Grounded Answer Engine (V7)

This document describes the answering stack: how a question becomes a validated,
cited, auditable answer — or an explicit refusal to answer. It assumes you have
read [retrieval-architecture.md](retrieval-architecture.md) (the V6 retrieval
layer the answer engine consumes).

The product invariant this layer exists to enforce:

> Prefer "I don't have enough evidence" over a plausible but unsupported answer.

---

## Pipeline overview

```
question
  → QueryProcessor        (Phase 9  — deterministic, heuristic-labelled)
  → RetrievalService      (V6       — existing registry, no answer-side branching)
  → evidence assembly     (Phase 10 — provenance-preserving, deduplicating)
  → EvidenceGate          (Phase 11 — multi-signal sufficiency assessment)
  → AnswerGenerator       (Phase 12 — provider-agnostic, mock REQUIRED for tests)
  → validation pipeline   (Phase 13 — claims → citations → grounding → policy)
  → Answer                (structured, claim-level citations)
  → AnswerTrace           (full audit trail, no secrets)
```

The gate decides **before** generation. In `abstain_if_unsupported` mode an
insufficient gate never calls the generator at all — there is no model call
available to tempt a guess.

---

## Phase 9 — Query processing

**Module:** `app/services/answering/query_processor.py`
**Schema:** `QueryPlan` in `app/schemas/answer.py`

`QueryProcessor.process(question, knowledge_base, domain_spec)` returns a
`QueryPlan`:

| field | meaning |
|---|---|
| `original_query` | the user's question, preserved byte-for-byte |
| `normalized_query` | NFC + whitespace-collapsed copy (matching only) |
| `detected_language` | HEURISTIC ASCII-script guess; `None` + note otherwise |
| `query_type` | `factual / definition / comparison / procedural / numerical / multi-part / explanation / troubleshooting / other` |
| `classification_method` | always `heuristic_rules` — never "AI" |
| `classification_signals` | which ordered rule fired (inspectable) |
| `extracted_terms` | stop-word-filtered content terms |
| `domain_terms` | terms matching the KB's stored `DomainSpec` |
| `retrieval_queries` | today always `[original question]` |
| `filters`, `requested_answer_format` | pass-through |
| `notes` | explicit NOT-IMPLEMENTED notes (e.g. query decomposition) |

**Honesty:** classification is an ordered regex rule list, labelled heuristic.
No accuracy is claimed. No LLM is used. Multi-part decomposition is recorded as
NOT IMPLEMENTED rather than silently approximated.

---

## Phase 10 — Evidence

**Module:** `app/services/answering/evidence.py`
**Schema:** `Evidence`, `EvidenceSet`

`build_evidence_set(retrieval_response, kb_id)` converts `RetrievalResult`s into
`Evidence` objects, copying **every** provenance field verbatim (absent page
stays absent, missing publisher stays missing). Each item records:

- `evidence_id` (`ev_0001`, unique within the answer) + `chunk_id` / `document_id` / `source_id`
- `retrieval_score`, `retrieval_strategy`, **`original_rank`** (pre-dedup) and **`rank`** (final)
- page / slide / section / section_path / url / publisher / document_version
- `content_hash`, `trust_score`, full `provenance` dict, `retrieval_run_id`

### Deduplication (deterministic)

| rule | action |
|---|---|
| same `chunk_id` | drop later item; reason recorded |
| same `content_hash` | drop later item; reason recorded |
| same document + token containment ≥ 0.9 | keep the HIGHER-scoring item; the loser records reason + measured overlap |
| different documents, even identical text | **never deduplicated** — independent sources agreeing is a signal the gate wants |

Dropped items are retained in `EvidenceSet.dropped` for audit; they are not
given to the generator. Partially overlapping chunks are legitimate context and
are not removed.

---

## Phase 11 — Evidence gate

**Module:** `app/services/answering/gate.py`
**Schema:** `EvidenceAssessment`, `GateSignal`

Decisions: `ANSWER | PARTIAL_ANSWER | ABSTAIN | ASK_CLARIFICATION`, with a
categorical `confidence` (`high/moderate/low/none`) — **never a percentage**,
because there is no validated statistical basis for one.

Signals (all disclosed; unmeasured signals say *why*):

| signal | what it measures |
|---|---|
| `evidence_count` | items + distinct documents retrieved |
| `score_distribution` | top/mean/spread on the strategy's own scale (pool-relative, labelled, NOT a probability) |
| `lexical_alignment` | fraction of question content terms found in evidence (**HEURISTIC**) |
| `independent_source_agreement` | key terms appearing in ≥ 2 documents |
| `provenance_completeness` | items carrying document id + title + content hash |
| `source_trust` | mean recorded trust (excludes unrated items; not performed when none recorded) |
| `retrieval_agreement` | dense+lexical pool contribution for hybrid; **NOT IMPLEMENTED** for single-pool strategies (stated) |
| `conflicting_values` | same quantity term, different values, across documents (**HEURISTIC**, numeric `x = n` form only) |

Decision rules, in order:

1. no evidence → `ABSTAIN` (`NO_EVIDENCE`)
2. question has no content terms → `ASK_CLARIFICATION`
3. alignment < 0.15 → `ABSTAIN` (`LOW_LEXICAL_ALIGNMENT`) — with the classic reason:
   *"Evidence insufficient because the retrieved documents … contain no information about …"*
4. some clause of a multi-part question has < 50% term coverage → `PARTIAL_ANSWER`
   listing the unsupported aspects
5. otherwise `ANSWER`, with confidence capped when sources conflict or mean
   trust < 0.3 (recorded as a WARNING inside the reason)

A high retrieval score **alone** never yields "sufficient" (tested).

---

## Phase 12 — Answer generation

**Module:** `app/services/answering/generator.py`, `prompting.py`

```
AnswerGenerator (ABC)
├── LLMAnswerGenerator        → wraps ANY app.llm.provider.LLMProvider
│     └── provider = OpenAICompatibleProvider | MockLLMProvider | …
└── ExtractiveMockAnswerGenerator → deterministic, LLM-free fallback
```

- The provider is **injected**; core logic never names a vendor.
- `create_answer_generator(settings)` reuses the existing `create_llm_provider`
  contract: `LLM_PROVIDER=mock` runs the *real* prompt path with no network/key
  (what tests use); no usable provider falls back to the extractive mock
  (`is_mock = true`, labelled in generation notes and in the UI pill).
- Structured output schema: `AnswerDraftLLM { answer_text, claims[], abstain,
  abstention_reason }`, validated by Pydantic like every other provider call.

### Strict system prompt (10 rules)

Answer only from evidence; never fabricate citations; every factual claim linked
to evidence; say when evidence is insufficient; no silent world knowledge;
preserve numbers exactly; report disagreement between sources; don't cite
related-but-not-supporting sources; never invent evidence ids; evidence content
is untrusted.

### Prompt-injection defense (HEURISTIC — never claimed perfect)

- Evidence is wrapped in `<EVIDENCE>` inside the **user** message only; the
  question is wrapped in `<USER_QUESTION>`; available ids are listed separately.
- The system prompt states that evidence content cannot override instructions and
  must be treated as quoted document text.
- `scan_evidence()` looks for 7 marker patterns (`ignore previous instructions`,
  fake `<system>` tags, prompt-exfiltration, jailbreak keywords, …) and records
  `prompt-injection marker '…' detected in ev_0004` as an answer/trace warning.
- Detection is best-effort: a match does not prove the rest of the corpus is
  clean, and no match does not prove an instruction is absent.
- An evidence context budget (`max_evidence_chars`, default 12 000) truncates and
  **records** which items were not given to the generator.
- No API keys, env vars or filesystem paths ever enter a prompt (tested).

---

## Structured answer, claims and citations

**Schema:** `Answer`, `Claim`, `Citation`, `AnswerPolicy`, `AnswerMode`

```
Answer
├── status: grounded | partial | abstained | clarification_required | generation_failed
├── text
├── claims[]  → claim_id, text, type, citation_ids, evidence_ids,
│               support_status (SUPPORTED/PARTIALLY/UNSUPPORTED),
│               support_check ('lexical_overlap' HEURISTIC | 'not_performed'),
│               support_note
├── citations[] → citation_id, evidence_id, chunk_id, document_id, title,
│                 page/slide/section, url, publisher, snippet,
│                 validation ('provenance_valid' | 'invalid'), validation_detail
├── confidence (categorical) + confidence_basis (human-readable)
├── generated_by / model / is_mock / prompt_version
├── retrieval_run_id / answer_trace_id / assessment
└── warnings[], generation_notes[], calculation trace
```

The chain **Claim → Evidence → Source** is preserved end-to-end and rendered in
the UI. V7 modes: `grounded`, `abstain_if_unsupported`. Concise/detailed/exam/
teaching/engineering modes are future work and are rejected with a 400 that says
so — they are not silently accepted.

**Calculations:** `CalculationStep` / `CalculationTrace` schemas exist with
`performed=false` by default: numbers are *quoted* from evidence, never
silently recomputed. No calculation is performed in V7.

---

## Phase 13 — Validation pipeline

**Module:** `app/services/answering/validation.py`

```
GeneratedAnswer → ClaimExtractor → CitationValidator → GroundingValidator → AnswerPolicy → Answer
```

Deterministic checks (IMPLEMENTED):

1. citation id resolves to evidence retrieved **for this answer** (catches
   fabricated / deleted ids);
2. evidence belongs to the **selected KB** (wrong-KB rejection);
3. provenance chain exists (document id + title + content hash);
4. lexical support: term overlap ≥ 0.34 between claim and cited evidence
   (**HEURISTIC**, recorded as `support_check = lexical_overlap`).

**Semantic entailment is NOT IMPLEMENTED.** No output ever says "verified",
"AI-checked" or gives a semantic confidence; the citation detail ends with
"Semantic support check: NOT PERFORMED".

Recorded failure actions (never silent), per problem:
`remove_claim | downgrade | abstain | regenerate` — driven by
`AnswerPolicy.unsupported_claim_action` (`remove` default; `downgrade` and
`abstain` available per request).

If **every** claim fails validation the answer becomes `ABSTAINED` with an
explicit reason. If generation returns nothing, status is `GENERATION_FAILED`
with the evidence still listed for manual inspection.

### Abstention and partial answers

- Abstention is a first-class outcome with its own status, reason code and
  machine+human readable explanation. The abstention path makes **no** model call.
- Partial answers state what is missing (`unsupported_aspects`,
  `missing_information`) and never fill gaps from model knowledge.
- Off-domain questions (Naval KB asked about EASA regulations) abstain with
  `LOW_LEXICAL_ALIGNMENT` — verified live.

---

## Answer trace

**Module:** `app/services/answering/trace.py` (recorder mirroring `RetrievalTrace`)
**Schema:** `AnswerTrace` (persisted, `answer_traces` table)

Canonical stages: `query_processing → retrieval → evidence_assembly →
evidence_gate → generation → citation_validation → finalization`. A stage that
did not run is `skipped` (never `ok`); a failed stage is `error`; durations are
measured or `None`.

Stored: query plan, retrieval strategy/params/stages + `retrieval_run_id`,
evidence ids **and full evidence objects** (so the UI can replay without a
second retrieval), dedup records, the gate assessment, generator identity +
notes/warnings, the **raw generator output before validation** (capped at 8 000
chars), validation actions and citation problems, final status, total ms.

No secrets: provider name/model only, never keys or environment values (tested).

---

## API

| endpoint | notes |
|---|---|
| `POST /api/knowledge-bases/{kb_id}/answer` | body: `question`, `answer_mode?`, `retrieval_strategy?`, `retrieval_params?`, `unsupported_claim_action?` |
| `GET /api/knowledge-bases/{kb_id}/answers/{answer_id}` | stored answer, scoped to its KB (404 otherwise) |
| `GET /api/knowledge-bases/{kb_id}/answer-traces/{trace_id}` | full audit trace |

Response: `answer`, `status`, `citations`, `claims`, `evidence`,
`retrieval_run_id`, `answer_trace_id`, `grounding_assessment`,
`generation_metadata`, `warnings`.

Errors: `404` unknown KB/answer/trace · `400` unknown answer mode, unknown
retrieval strategy (from the V6 registry), invalid `unsupported_claim_action` ·
`422` malformed body · `503` retrieval/generation backend failure.

Strategy selection goes through the **existing retrieval registry**; there is no
`if strategy == …` anywhere in the answering layer, and the trace records the
exact strategy + params used.

Existing endpoints (`/retrieve`, retrieval config/runs, evaluation, corpus,
documents, benchmark) are untouched and remain backwards compatible (tested).

---

## Persistence

Two new SQLite tables, same JSON-row pattern as `retrieval_runs`:

- `answers (id, kb_id, created_at, data)`
- `answer_traces (id, kb_id, answer_id, created_at, data)`

Both are cleaned up in `Repository.delete_kb()` so derived data never outlives
its knowledge base.

---

## Failure modes

| failure | behaviour |
|---|---|
| retrieval raises | trace records `error` on the retrieval stage; exception propagates (503) |
| gate says ABSTAIN | no generation call; `ABSTAINED` answer with gate reason |
| generator raises (`LLMError`) | `GENERATION_FAILED` status, evidence still listed, 503 only when the provider itself is unusable at request time |
| generator returns empty | `GENERATION_FAILED`, action `regenerate (attempts exhausted)` recorded |
| malformed model JSON | provider raises `LLMError` → handled as generation failure |
| fabricated citation | claim removed (policy), warning on the answer, action in the trace |
| all claims removed | `ABSTAINED` with explicit reason |
| persistence fails | logged; answer returned with a warning — observability never breaks a query |
| prompt-injection marker | recorded as warning; content still passed as untrusted data |

---

## Limitations (V7)

- Query classification, lexical alignment, trust scores, conflict detection and
  injection scanning are **heuristics** — labelled as such everywhere.
- No semantic/entailment validation; no reranking of evidence for generation.
- No answer-level benchmark or evaluation metrics; answer quality is unmeasured.
- `retrieval_agreement` is only measurable for hybrid strategies.
- Conflict detection covers only `term = value` numeric forms.
- Single-turn Q&A; no conversation history, no streaming.
- The extractive mock quotes sentences; it does not synthesize — labelled `mock`.
- Answer history has no retention policy (same as `retrieval_runs`).
