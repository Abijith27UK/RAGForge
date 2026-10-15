# V20 — Off-domain abstention: reproduced failure, root cause, fix, limits

**Status:** fixed and regression-covered on 2026-10-15 · **Baseline:** `ef8538f` (V19)
**Corpus used for every measurement:** the protected Automobile KB `kb_f278c283c748`
(11 documents, 812 chunk rows, 812 Qdrant points, collection `kb_kb_f278c283c748`,
`all-MiniLM-L6-v2`, strategy `dense`). **Nothing in that corpus was modified.**

---

## 1. The failure, reproduced

Command (read-only; runs the same pipeline the MCP tool runs):

```bat
backend\.venv\Scripts\python.exe backend\scripts\v20_reproduce_offdomain.py ^
  --kb kb_f278c283c748
```

Before the fix, on the real corpus:

| Question | reported | why it was wrong |
|---|---|---|
| `How many moons does Jupiter have?` | `status=grounded`, `grounding_state=ANSWERED`, `reason_code=SUFFICIENT`, **1 citation**, 1 claim | an Automobile corpus cannot answer a question about Jupiter's moons |
| `Describe the courtship display of the Australian lyrebird.` | `abstained`, `INSUFFICIENT_EVIDENCE`, `LOW_LEXICAL_ALIGNMENT`, 0 citations | correct — this is the contrast case that hid the defect |

## 2. Root cause — three separate mistakes, not one threshold

The gate's decision path (`app/services/answering/gate.py`) had:

1. **`lexical_alignment` was unweighted.** It is the raw fraction of question
   content terms found *anywhere* in the retrieved text, so a corpus-ubiquitous
   English word counts exactly as much as the term that carries the question's
   meaning. For the Jupiter question the terms are `{jupiter, many, moons}`;
   `jupiter` and `moons` occur in **none** of the 812 chunks while `many` occurs in
   **62** of them, giving `alignment = 1/3 = 0.3333`.
2. **The only real floor was 0.15.** `MIN_ALIGNMENT_TO_PARTIAL = 0.15`, so
   `0.3333` cleared it and the rule fell through to the next one.
3. **The SUFFICIENT rule had no alignment requirement at all.** Rule 5 used
   `align >= MIN_ALIGNMENT_TO_ANSWER` only to choose between `HIGH` and
   `MODERATE` *confidence*; it never gated the decision. A single-clause question
   that cleared 0.15 and produced no unsupported aspect therefore reached
   `decision=ANSWER`, `sufficient=True` — and the generator was then asked to
   produce an answer from evidence about engines.

So raising one global threshold could not fix this: the number that was supposed
to mean "the corpus discusses this" did not mean that, and the decision rule that
mattered never consulted it.

Meanwhile the gate was already **recording** the signal that does separate the two
situations — `score_distribution`, the top retrieval score — and never reading it.

## 3. The fix — two rules, both measured against the corpus

### Rule 4: relevance floor (`LOW_EVIDENCE_RELEVANCE`)

`MIN_RELEVANCE_TO_ANSWER = TOP_SCORE_MODERATE = 0.35`, applied to the top reported
score. Measured separation on the real corpus (`scripts/v20_calibrate_relevance.py`):

| Class | n | top cosine, min → max |
|---|---|---|
| supported in-domain questions | 12 | **0.4365 → 0.6946** |
| off-domain questions | 6 | **0.2317 → 0.3411** |

The bands do not overlap and `0.35` — an **existing** V7 constant, not a new
number — sits inside the gap. Two guard rails keep it honest:

* it is applied **only** to strategies whose reported score is an unnormalized
  similarity (`dense`). For `bm25` / `hybrid` / `hybrid_reranked` the score is
  min-max normalized inside the candidate pool, so the top hit is ~1.0 by
  construction; the gate records the check as **NOT APPLICABLE with the reason**
  rather than applying a threshold that means something else there.
* it is applied **only** to semantic embeddings. The development hashing fallback
  (`hashing-dev-fallback`) scores character overlap, not meaning — measured, it
  scored a supported naval question at **0.274** while scoring an off-domain
  question at **0.4543**, so a semantic floor there would both refuse answerable
  questions and pass unanswerable ones. `RetrievalResponse.embedding_is_fallback`
  now carries that fact through to `EvidenceSet`, and the gate records the check
  as NOT APPLICABLE for it.
* it is configuration (`ANSWER_MIN_RELEVANCE`, default `0.35`), because a cosine
  scale is embedding-model dependent. **Recalibrate when `EMBEDDING_MODEL`
  changes.**

### Rule 5: corpus vocabulary fit (`TOPIC_OUTSIDE_CORPUS`)

Scale-free, so it still protects where rule 4 does not apply. It fires only when
**both** hold:

* **at least half** of the question's content terms occur in **none** of the
  corpus's chunks (`MIN_ABSENT_TERM_SHARE = 0.5`), and
* the corpus holds **no rare term** from the question (`MIN_DISTINCTIVE_IDF = 3.0`
  measured with the same Lucene IDF the BM25 retriever scores with).

The statistical source is the project's own BM25 statistics (`lexical_indexes`):
`df`, `n_docs`, `avgdl`. `LexicalIndexStatsProvider` reads the persisted index when
one exists and otherwise builds the statistics **read-only, in memory, cached per
process** — because a fresh KB that has only ever used dense retrieval has no
persisted index, and the check would otherwise be silently unavailable on exactly
the corpora a new user builds. Measured cost of that build on the real corpus:
**~82 ms** for 812 chunks / 548 K characters, once per process. It is bounded
(`MAX_TERM_STATS_CHUNKS = 50 000`) and it never writes: persisting an index stays
retrieval's job.

### What was deliberately NOT done

A first attempt used IDF alone ("abstain when the question's rare terms are
missing") and was **measured and rejected**: rare is not the same as topical.
Question-framing words that technical prose rarely uses land in the same band —
`specification` (IDF 6.295), `purpose` (5.448), `difference` (4.686), `work`
(3.316), `components` (3.131) — and `explain`/`describe` do not occur in the
corpus at all. That rule refused **two questions this corpus answers**
(`How do brakes work?`, `Explain the steering system.`). Rule 5 is shaped by that
measurement, and a regression test (`test_rare_framing_words_do_not_cause_abstention`)
exists specifically to keep it from coming back. The band table is in the source
next to `MIN_DISTINCTIVE_IDF`.

## 4. Measured results (real corpus, before → after)

| # | Question | before | after |
|---|---|---|---|
| 1 | What is an engine? | grounded | grounded |
| 2 | How do brakes work? | grounded | grounded |
| 3 | Explain the steering system. | grounded | grounded |
| 4 | What are the main components of a suspension system? | grounded | grounded |
| 5 | What is a turbocharger and how does it work? | partial | partial |
| 6 | What is the firing order of a 4-cylinder engine? | grounded | grounded |
| 7 | What is the difference between a diesel and a petrol engine? | partial | partial |
| 8 | What is the purpose of a catalytic converter? | grounded | grounded |
| 9 | What is the torque specification for the cylinder head bolts? | grounded | grounded |
| 10 | How does a transmission work? | grounded | grounded |
| 11 | What is buoyancy? | abstained (`LOW_LEXICAL_ALIGNMENT`) | abstained (`LOW_LEXICAL_ALIGNMENT`) |
| 12 | **How many moons does Jupiter have?** | **grounded, 1 citation** | **abstained (`LOW_EVIDENCE_RELEVANCE`), 0 citations, 0 claims** |
| 13 | Describe the courtship display of the Australian lyrebird. | abstained | abstained |
| 14 | **What are the EASA regulations for commercial drone certification?** | **grounded** | **abstained (`LOW_EVIDENCE_RELEVANCE`)** |
| 15 | Who won the 2018 FIFA World Cup? | grounded | abstained (`LOW_EVIDENCE_RELEVANCE`) |
| 16 | What is the capital of Peru? | grounded | abstained (`LOW_LEXICAL_ALIGNMENT`) |

**10/10 supported in-domain questions still answer; 6/6 off-domain questions now
abstain with zero citations and zero claims.** The pre-existing `partial`
distinction is preserved, not flattened into "answer".

Through the **public MCP tool** (`scripts/check_claude_desktop_connection.py`
against the configured client command): `answer_from_knowledge_base` on
`How many moons does Jupiter have?` returns `status=abstained`,
`grounding_state=INSUFFICIENT_EVIDENCE`, `reason_code=LOW_EVIDENCE_RELEVANCE`,
`citation_count=0`. The acceptance run went from *23 passed / 1 known defect* to
**23 passed / 0 failed / 2 warnings / 0 known defects** (the warnings are the
pinned `LLM_PROVIDER` override and the disclosed mock generator identity).

## 5. Regression coverage

`backend/tests/test_v20_offdomain_gate.py` (43 tests) and
`backend/tests/test_v20_provider_gate.py` (9 tests). Cases, in the order the user
asked for them:

| Case | Where |
|---|---|
| the two known off-domain examples | `TestReproducedDefect` (real measured cosine + real df values) |
| high superficial lexical overlap, irrelevant evidence | `test_generic_overlap_alone_is_never_sufficient`, `test_rare_framing_words_do_not_cause_abstention` |
| in-domain questions supported by the corpus | `TestSupportedInDomainStillAnswers` (5 parametrised), plus `TestThroughTheMCPTool` |
| in-domain questions with weak/missing evidence | `test_no_evidence_abstains_without_any_measurement`, `TestThroughTheAnswerService` |
| partial support stays distinct | `test_partially_covered_question_keeps_partial_state`, `test_conflicting_evidence_state_still_wins` |
| ambiguous / underspecified questions | `TestAmbiguityAndTurns` (conversational turn, `and?`) |
| answer present but ranked lower | `test_lower_ranked_evidence_is_still_considered` |
| prompt-injection text inside retrieved documents | `test_injected_instruction_inside_evidence_is_not_a_question_term` (gate) and `test_prompt_injection_in_retrieved_text_is_not_obeyed` (service) |
| an unmeasured check never abstains | `TestUnmeasuredChecks` (no provider / unavailable / no kb id / pool-normalized strategy / dev fallback) |
| statistics source, bounds and read-only-ness | `TestStatisticsSource` (a store whose `save_lexical_index` raises if called) |
| no hidden fallback reporting success | `test_unusable_provider_is_disclosed_not_silent`, `test_a_failed_real_call_is_an_error_never_a_mock_success` |
| the fix reaches a real client | `TestThroughTheMCPTool` (real MCP protocol, in-memory transport) |
| default wiring really uses the statistics | `TestDefaultWiringUsesPersistedCorpusStatistics` (no injected gate) |

Gate-level cases use **synthetic** evidence with clearly-labelled synthetic
statistics; cases that depend on the real corpus reuse its measured numbers and
say so. This is a regression suite, not a benchmark: **it does not and cannot show
that off-domain abstention generalizes** beyond the cases measured here. The
held-out benchmark lifecycle owns that question, and it is still open.

## 6. Limitations (stated, not hidden)

* **The relevance floor is calibrated per embedding model.** `0.35` is measured for
  `all-MiniLM-L6-v2` on this corpus. A different `EMBEDDING_MODEL` needs its own
  calibration (`scripts/v20_calibrate_relevance.py` reproduces the table above);
  the value is configuration, not a secret constant, for exactly that reason.
* **Rule 5 fires only after at least half the question's terms are absent from the
  corpus.** A question built entirely from words the corpus happens to know
  (`How many moons does the vehicle have?`) is not caught by it. That wording is
  caught by rule 4 instead.
* **Rule 5 needs SQLite-backed chunk text.** A store that can neither provide a
  persisted lexical index nor list chunk text leaves it NOT PERFORMED, and the run
  says so. In that state only rule 4 and the pre-existing alignment rule apply.
* **`bm25` / `hybrid` runs have no relevance floor**, by design (their scale is
  pool-relative). Rule 5 and the alignment rule still apply; a hybrid run has one
  fewer guard than a dense run, and says so in its signals.
* **Rule 3 (alignment floor) still reports the retrieved ids as
  `supporting_evidence_ids` while abstaining.** That behaviour is pre-existing and
  was left unchanged rather than quietly altered; the V20 rules report `[]`,
  because those chunks genuinely do not support the question. What matters for a
  user is `claims` and `citations`, and both are empty on every abstention.
* **This is not a statistical result.** 16 questions were measured, chosen to
  cover the reported defect and its neighbours. No significance is claimed, and no
  threshold was fitted to a held-out set that does not exist yet.
* **No live model was involved.** The answer *text* in these measurements came
  from the disclosed deterministic mock (`is_mock=true`); retrieval, gate
  decisions, citations and abstentions are real.
