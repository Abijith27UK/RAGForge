# Failure Analysis — the V9 deterministic taxonomy

**Status: IMPLEMENTED and TESTED. Classifications are DETERMINISTIC and
falsifiable, not semantic truth.**

Module: `backend/app/services/answer_eval/failure.py`
Tests: `backend/tests/test_failure_taxonomy.py` (8 tests)

---

## Why this exists

V8 measured answer quality: citation precision/recall, fabricated-citation rate,
unsupported-claim rate, relevance, key-point recall. All of that answers *how
good* an answer was. None of it answers **why it failed**, and without "why"
there is no route from measurement to improvement — you cannot recommend a
configuration change from a number that says only `citation_recall = 0.42`.

The taxonomy converts a failed question into one dominant, inspectable code.

---

## The closed set

Exactly eleven labels. Adding one is a deliberate breaking change, because the
UI and every persisted experiment record key on this set.

| Code | Meaning |
|---|---|
| `NO_RELEVANT_RETRIEVAL` | Retrieval ran, ground truth exists, and none of the required evidence came back |
| `LOW_RETRIEVAL_RECALL` | Required evidence WAS retrieved, but beyond the front window |
| `WRONG_DOCUMENT` | Evidence was drawn from documents other than the ones that can answer |
| `WRONG_CHUNK` | Required evidence was near the top but was outranked |
| `INSUFFICIENT_EVIDENCE` | The gate said so, or retrieval returned nothing at all |
| `CONFLICTING_EVIDENCE` | A cited passage contradicts the claim drawn from it |
| `CITATION_ERROR` | A citation resolves to nothing retrieved, or a claim is uncited |
| `UNSUPPORTED_CLAIM` | A claim was asserted with no supporting citation |
| `ANSWER_RELEVANCE_FAILURE` | Grounded, but about the wrong subject |
| `GENERATION_FAILURE` | The pipeline reported it, or the answer is empty |
| `UNKNOWN` | No deterministic rule fired. **Preferred over a guess** |

---

## Every classification carries four things

```python
FailureClassification(
    classification="LOW_RETRIEVAL_RECALL",   # or "" when nothing failed
    confidence=FailureConfidence.DETERMINISTIC,   # deterministic | partial | unknown
    evidence=["required chunk at rank 8 (front window = 3)", "retrieval_hit_rate=1.0"],
    reason="...",
)
```

`confidence` is **categorical, never numeric**. A fabricated percentage
("72% confident it's a recall failure") would be exactly the kind of number this
project forbids. The three values mean:

- `deterministic` — a rule fired on a value the pipeline or benchmark **states**.
- `partial` — a rule fired on an **inference** (e.g. empty answer text).
- `unknown` — nothing fired.

---

## Rule precedence, and why it is that order

Two principles decide the order:

1. **A failure the pipeline itself stated outranks a failure the module infers.**
   If the pipeline reported `answer_status = generation_failed`, that is a fact
   about what happened. If the answer text is merely empty, that is an inference.
2. **A retrieval fault outranks an answer symptom,** because a generator cannot
   answer from evidence it never received. Diagnosing a citation problem when
   retrieval never supplied the passage sends the reader to the wrong fix.

```
 1. answer_status == GENERATION_FAILED            -> GENERATION_FAILURE (deterministic)
 2. retrieved non-empty, hit_rate == 0, no evidence-> NO_RELEVANT_RETRIEVAL
 3. grounding_state == INSUFFICIENT_EVIDENCE      -> INSUFFICIENT_EVIDENCE (deterministic)
    retrieval returned nothing at all             -> INSUFFICIENT_EVIDENCE
 4. required rank > FRONT_RANK_WINDOW (3)         -> LOW_RETRIEVAL_RECALL
 5. 1 < required rank <= 3                        -> WRONG_CHUNK
 6. evidence from disjoint documents, hit_rate 0  -> WRONG_DOCUMENT
 7. a claim judged CONTRADICTED                   -> CONFLICTING_EVIDENCE
 8. a claim judged UNSUPPORTED                    -> UNSUPPORTED_CLAIM
 9. invalid/absent citation, or fabrication > 0   -> CITATION_ERROR
10. relevance_passed is False                     -> ANSWER_RELEVANCE_FAILURE
11. empty answer, no stated status                -> GENERATION_FAILURE (partial)
12. otherwise                                     -> UNKNOWN
```

### `FRONT_RANK_WINDOW = 3` is a CONVENTION, not a measurement

The split between `LOW_RETRIEVAL_RECALL` and `WRONG_CHUNK` is a fixed rank
threshold, and it is named as a constant precisely so it cannot drift silently.
Both are retrieval faults; the distinction only directs which knob to try:

- rank inside the window → **ranking** problem (something else outranked it)
- rank beyond the window → **depth** problem (a smaller `top_k` drops it)

It is not presented as truth.

---

## Ground truth is required before blaming retrieval

This is the most important property, and it comes from a real defect in the
first draft.

V8's evaluator sets `retrieval_hit_rate` to **UNKNOWN** (`Measured.measured is
False`) whenever a benchmark question carries **no required-evidence label** —
see `retrieval_hit_rate = Measured.unknown("no required-evidence label")` in
`services/answer_eval/evaluator.py`.

So `retrieval_hit_rate is None` means *"we have no reference evidence for this
question"*, and the taxonomy therefore **refuses to classify a retrieval fault
from it**. Those questions fall through to `UNKNOWN`.

The first draft got this backwards: an `INSUFFICIENT_EVIDENCE` rule fired on
`answerability == answerable and rank is None and not evidence`, which matches
every question with **no ground truth at all**. It would have reported a
retrieval or corpus defect on the basis of *missing labels*, which is a
fabricated finding. Four tests now pin the corrected behaviour.

> **Zero and unknown are different findings.** "The right passage was ranked
> last" and "we don't know which passage is right" must never render the same,
> because only the first is a retrieval bug.

---

## Two entry points

```python
# 1. Direct, from measured signals (used by unit tests and diagnostics)
classify_failure(question=..., answer_text=..., evidence=[...],
                 retrieved_chunk_ids=[...], retrieved_rank_of_required=...,
                 required_chunk_ids=[...], retrieval_hit_rate=...,
                 question_answer_relevance=...,
                 relevance_passed=..., claim_verdicts=[...],
                 fabrication_rate=..., unsupported_citation_rate=...,
                 answerability=..., answer_status="", grounding_state="",
                 abstained=False)

# 2. From a real V8 AnswerQualityResult — reads ONLY its measured values
classify_result(result)
```

`classify_result` is the integration point: it reads the same `Measured` triples
the UI displays, so an unmeasured metric arrives as `None` and cannot justify a
classification.

---

## What this layer does NOT do

- It does **not** claim semantic truth. `CONFLICTING_EVIDENCE` means a cited
  passage contradicts a claim under the entailment provider's judgement; it does
  not mean the corpus is objectively contradictory.
- It does **not** replace human review. `UNKNOWN` is an explicit hand-off.
- It does **not** assign a numeric confidence.
- It does **not** decide anything about retrieval configuration. That is the
  recommendation engine's job, and it requires an experiment.
