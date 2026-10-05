# Grounding and Citations

How RAGForge decides whether an answer may be given, and how every factual claim
is tied back to a retrieved chunk.

This document is the reference for the rules the rest of the system is
accountable to. Where the implementation cannot do something, that is stated
here rather than left for a user to discover.

Related: [answering-architecture.md](answering-architecture.md),
[chat-architecture.md](chat-architecture.md),
[architecture.md](architecture.md).

---

## 1. The core rule

> RAGForge answers **only** from retrieved evidence, shows where every factual
> claim came from, and abstains when the evidence is insufficient.

Two consequences that shape the whole design:

1. **The generator cannot see the database.** It receives exactly the evidence
   an `EvidenceSelector` chose. There is no code path that hands a model a
   corpus, a vector store or a file path, so "use outside knowledge" is not a
   matter of willpower — it is not reachable.
2. **Nothing is presented as verified unless it is.** `provenance_valid` means
   an evidence id resolved to a retrieved chunk with a complete provenance
   chain. It does **not** mean the source proves the claim.

---

## 2. The grounding gate

`GroundingGate` (`app/services/answering/gate.py`) is deterministic. No LLM is
involved in deciding whether an answer is permitted.

### 2.1 The five states

| State | Meaning | Answer permitted? |
|---|---|---|
| `ANSWERED` | Evidence supports the question. | Yes |
| `PARTIALLY_SUPPORTED` | Evidence covers part of the question only. | Yes — what is supported, with what is missing stated |
| `INSUFFICIENT_EVIDENCE` | Something was retrieved but it does not ground an answer. | No |
| `CONFLICTING_EVIDENCE` | Sources disagree on a quantity. | Yes — the disagreement is reported, never silently resolved |
| `NO_RELEVANT_EVIDENCE` | Nothing usable was retrieved at all. | No |

`GroundingState` maps onto the coarser `GateDecision` used internally
(`ANSWER` / `PARTIAL_ANSWER` / `ABSTAIN` / `ASK_CLARIFICATION`) through the
exhaustive, unit-tested `GATE_TO_GROUNDING` table. A new `GateDecision` without
a mapping is a programming error, not a runtime fallback.

### 2.2 Why there is no confidence percentage

`confidence` is categorical (`high` / `moderate` / `low` / `none`). A single
blended percentage would hide *which* signal failed, and there is no validated
statistical basis for one here. The UI shows the state and the signals instead
of a number that implies more than was measured.

### 2.3 Signals

Every signal is a measurement with a recorded threshold, or it is explicitly
marked `measured=False` with the reason it was not performed.

| Signal | What it measures |
|---|---|
| `evidence_count` | How many items, from how many documents |
| `score_distribution` | Top / mean / spread on this run's score scale |
| `lexical_alignment` | Fraction of question terms present in the evidence (heuristic) |
| `independent_source_agreement` | Distinct documents containing the same key term |
| `provenance_completeness` | Share of items with document id + title + content hash |
| `source_trust` | Mean recorded trust score, when the KB records one |
| `retrieval_agreement` | Dense/lexical pool agreement — hybrid strategies only |
| `conflicting_values` | Same quantity, different value, across documents (heuristic) |

Scores are **pool-relative**. A dense cosine score of 0.5 and a BM25 score of 0.5
are not the same quantity; they are only comparable within one run. The UI says
so wherever a score is displayed.

### 2.4 Decision order

```
NO EVIDENCE ──────────────► NO_RELEVANT_EVIDENCE (abstain)
QUESTION HAS NO TERMS ────► INSUFFICIENT_EVIDENCE   (ask for clarification)
ALIGNMENT < 0.15 ─────────► INSUFFICIENT_EVIDENCE   (corpus does not discuss this)
PARTIAL COVERAGE ─────────► PARTIALLY_SUPPORTED
CONFLICTING VALUES FOUND ─► CONFLICTING_EVIDENCE   (answer, flag the conflict)
otherwise ─────────────────► ANSWERED
```

Thresholds are named constants (`MIN_ALIGNMENT_TO_ANSWER`,
`MIN_ALIGNMENT_TO_PARTIAL`, `ASPECT_MIN_COVERAGE`, `TOP_SCORE_MODERATE`,
`INDEPENDENT_SOURCES_STRONG`, `TRUST_LOW`) so tests and docs reference the same
values the code uses.

### 2.5 Non-knowledge turns

`GroundingGate.assess_with_trace` short-circuits **conversational** and
**non-knowledge** turns ("hi", "thanks", "help"). Judging whether a greeting's
evidence was sufficient would be a category error, and reporting "the corpus does
not cover this" would be misleading.

`UNDERSPECIFIED` and `MULTI_HOP` are deliberately **not** short-circuited: both
are real questions, so the real evidence signals decide — which is what keeps an
empty knowledge base reporting the honest `NO_EVIDENCE` reason instead of a
classification artefact.

### 2.6 Gate ordering in the pipeline

The gate runs **before** generation. In `abstain_if_unsupported` mode an
abstained question never reaches the generator, so there is no model call to
tempt a guess. The trace records the generation stage as `skipped`, with the
reason — a stage that did not run is never shown as `ok`.

---

## 3. Citations

### 3.1 Structure

The LLM is never trusted to produce citation strings. It may only reference
evidence ids from the `<AVAILABLE_EVIDENCE_IDS>` list it was given; anything
else is a fabrication and is caught.

```
Evidence  ev_0001 → chunk ch1 → document d1 → page 23 → hash a1b2…
Claim     "The metacentric height GM must be positive."
Citation  cite_ev_0001 → ev_0001
```

`Citation` fields (`backend/app/schemas/answer.py`):

`citation_id`, `evidence_id`, `document_id`, `chunk_id`, `source_title`,
`source_type`, `page_number`, `slide_number`, `section_path`, `content_hash`,
plus `url`, `publisher`, `snippet`, `validation`, `validation_detail`.

`page`/`page_number` and `slide`/`slide_number` are the same measurement under
two names; both are copied from the retrieved chunk. `source_title` falls back to
the document id when the chunk has no title — never to the content.

### 3.2 Never fabricated

A page number that was not recorded stays `None` and renders as *"page not
recorded"*. This is asserted by tests
(`test_missing_page_stays_none_and_is_never_invented`) because a plausible
guess is worse than an honest gap: a student must be able to trust that "p. 23"
means page 23.

### 3.3 Validation

`CitationValidator` is deterministic. For every claim it checks that each cited
evidence id:

1. exists in **this** answer's evidence set (catches fabricated ids);
2. belongs to **this** knowledge base (catches wrong-KB ids);
3. carries a complete provenance chain (missing → downgrade, not removal);
4. is lexically supported by the claim text (heuristic; see below).

A claim that cites one valid and one invalid id is kept but **downgraded**: the
generator demonstrably invented part of its provenance, so it cannot be reported
as fully supported. Both the downgrade and the dropped ids are recorded in the
trace.

### 3.4 What "supported" means

`Claim.support_check` is explicit:

- `lexical_overlap` — a deterministic heuristic. A claim is `supported` when at
  least `MIN_LEXICAL_SUPPORT` (0.34) of its content terms appear in the cited
  evidence; below that it is `partially_supported`.
- `not_performed` — nothing was checked.

**Semantic entailment is NOT IMPLEMENTED.** No LLM or NLI model verifies that a
source truly proves a claim, so nothing in the system may say "verified",
"AI-checked" or "fact-checked". `support_note` says so on every claim.

### 3.5 Remediation actions

`AnswerPolicy.unsupported_claim_action` selects the response, and the action
taken is recorded per problem:

| Action | Effect |
|---|---|
| `remove` | Claim removed; if all claims go, the answer abstains |
| `downgrade` | Claim kept, marked partially supported |
| `abstain` | Any claim problem abstains the whole answer |

The recorded action always reflects what **actually happened**. A claim kept
under `downgrade` is logged as `downgrade`, never as `remove_claim`.

---

## 4. Prompt injection

Retrieved documents are **untrusted data**.

* Evidence is wrapped in `<EVIDENCE>` tags and placed **only** in the user
  message; document content never enters the system prompt.
* The system prompt states that nothing inside those tags can change the
  instructions.
* `scan_evidence` runs 7 heuristics over the evidence and records markers, so a
  trace can show *"prompt-injection marker detected in ev_0004"*. The marker is
  surfaced as a warning on the answer.

**This defense is heuristic and is never claimed as complete.** Detection is
best-effort: a match does not prove the corpus is otherwise safe, and the absence
of a match does not prove no instruction is present.

No API keys, environment values, internal prompts or filesystem paths are placed
in prompts, traces, answers or answer runs. The system prompt is not echoed back
to the user — asserted by
`test_system_prompt_is_never_echoed_back`.

---

## 5. Generator honesty

| Configuration | What runs | How it is labelled |
|---|---|---|
| `LLM_PROVIDER=openai` (+ key) | `LLMAnswerGenerator(OpenAICompatibleProvider)` | live |
| `LLM_PROVIDER=mock` | `LLMAnswerGenerator(MockLLMProvider)` — the **real** prompt path, deterministic stand-in | `is_mock=true` |
| provider cannot be constructed | `FallbackAnswerGenerator` wrapping the extractive mock | `generator_status="fallback"`, `unavailable=true`, warning on the answer |

There is **no silent fallback**. When the configured generator is unavailable:

- `FallbackAnswerGenerator.unavailable = True` with the reason;
- the answer carries a warning naming the fallback and the reason;
- the UI renders a disclosure banner.

Two mock paths are deliberately distinguished, because they are not the same
thing: `LLM_PROVIDER=mock` means the prompt-building and structured-output path
genuinely ran against a deterministic stand-in, while a fallback means the
configured provider could not be constructed at all.

`UnavailableAnswerGenerator` records what was *wanted*, so an audit can say "you
asked for openai and got the extractive mock" instead of quietly losing that
fact.

---

## 6. Retrieval scores

Retrieval scores are reported as measured, in their original units, and are
always pool-relative. They are **not** probabilities and **not** confidence. The
UI shows them next to the score scale of the run that produced them, and the
grounding gate compares them only against thresholds documented for that scale.

---

## 7. Where each rule is enforced and tested

| Rule | Code | Tests |
|---|---|---|
| Five grounding states | `services/answering/gate.py` | `test_answering_v7_2.py::TestGroundingGateStates` |
| Gate is deterministic | same | `test_every_assessment_carries_machine_readable_reasons` |
| Non-knowledge turns | same | `test_conversational_turn_is_not_judged_on_evidence` |
| Citation fields | `schemas/answer.py`, `validation.py` | `TestCitationFields` |
| No fabricated pages | `validation.py` | `test_missing_page_stays_none_and_is_never_invented` |
| Nonexistent evidence rejected | `validation.py` | `TestCitationValidationCases::test_case_2_*` |
| Wrong-KB citation rejected | `validation.py` | `test_case_9_wrong_kb_citation_rejected` |
| Out-of-run citation rejected | `validation.py` | `test_case_10_*` |
| Outside knowledge prevented | gate + validator | `test_case_8_outside_knowledge_cannot_survive_the_gate` |
| No silent fallback | `generator.py` | `TestGeneratorHonesty` |
| Prompt injection | `prompting.py` | `test_chat_api_v7.py::TestPromptInjection` |
| Mock never mimics an LLM | `llm/provider.py` | `test_mock_provider_*` |

---

## 8. Known limitations

1. **Lexical support is a heuristic.** A claim can be marked `supported` on term
   overlap while the evidence does not genuinely entail it. This is stated on
   every claim, not hidden.
2. **Conflict detection is a heuristic.** It finds `term = value` disagreements
   across documents. Prose-form contradictions ("X improves, Y degrades") are
   **not** detected.
3. **No semantic entailment.** No NLI model is used or claimed.
4. **Injection detection is best-effort.** It is a marker scan, not a filter.
5. **Query decomposition is not implemented.** A multi-part question is retrieved
   as one query; the trace records this as a warning rather than pretending.
6. **Conversation history is not knowledge.** It resolves references only, and is
   never embedded, indexed or retrieved from.
---

# V8 — Answer-quality evaluation

This document describes how answers are *produced* and grounded. V8 adds how
they are *scored*. Full detail in
[answer-evaluation.md](answer-evaluation.md).

## Citation correctness is now measurable, not asserted

A citation existing is not a citation being correct. V8 measures, per claim:

- did the claim cite anything?
- does the cited evidence actually support it (entailment)?
- does the cited evidence **contradict** it (including polarity/negation)?
- does the citation point at evidence that was never retrieved?

and per answer:

- was all the *necessary* evidence cited?
- did the system abstain exactly when it should?
- did it label its own grounding state correctly?

## Two things deliberately not claimed

1. **Correctness is UNKNOWN.** The benchmark defines which chunk answers each
   question, not what the answer should say. No human has written a reference
   answer, so nothing is scored against one.

2. **`citation_precision` is a lower bound.** The benchmark's `required` flag
   marks *necessary* evidence, not an exhaustive list of permitted evidence.
   Answering thoroughly with additional on-topic context lowers this number
   without the answer being wrong. It is reported, never used as a hard failure.

## Retrieval misses are separate

If the required evidence was never retrieved, the answerer was never able to
cite it. `retrieval_hit_rate` is computed independently so the answer score is
never used to blame (or credit) the retrieval stage.

## The evaluator changed, and that is recorded

Scoring semantics are versioned (`deterministic-evidence` **v8.2**). The
v8.1 → v8.2 change corrected a level-of-judgement bug that had made the real
corpus score 0.000. Because every result carries its evaluator name and
version, two runs scored by different semantics are never silently mixed.
