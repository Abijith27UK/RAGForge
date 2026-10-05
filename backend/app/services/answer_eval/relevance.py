"""Answer relevance: does the answer actually address the question? (V8 STEP 8)

WHY THIS EXISTS
---------------
Groundedness and citation correctness cannot detect a *grounded answer to the
wrong question*. An answer can quote real retrieved evidence perfectly and still
be useless because it is about a different subject. Every other V8 metric
rewards exactly that behaviour, so without this check a system can score well
on citations and grounding while answering nothing that was asked.

METHOD — AND THE EVIDENCE BEHIND IT
-----------------------------------
Two candidate signals were MEASURED on the real corpus (286 stored answers in
KB `kb_f278c283c748`, plus one known off-domain answer produced by the
documented gate defect):

| method                     | on-domain min | known off-domain | verdict      |
|----------------------------|---------------|------------------|--------------|
| plain term coverage        | 0.333         | 0.400            | INVERTED     |
| IDF-weighted term coverage | 0.300         | 0.260            | separates    |

Plain coverage is **inverted**: the off-domain answer scored HIGHER than every
on-domain answer, because generic technical words ("regulations",
"certification") appear in any engineering text. A threshold on plain coverage
would therefore have failed correct answers and passed the wrong one. It is not
the shipped method; it remains only as the fallback when no corpus statistics
exist, and the fallback is labelled as weaker on every result.

IDF-weighted coverage weights each question content term by the corpus's own
inverse document frequency (the same Lucene IDF the BM25 index uses), so the
*rare* terms — the ones that actually name the subject, like "easa" or "drone"
— dominate the score, and generic words barely move it.

WHAT THIS IS NOT
----------------
* It is NOT semantic relevance. No embedding, no model, no LLM is involved;
  `is_model_based` is False on every result. A paraphrase that uses none of the
  question's terms scores low, and that limitation is reported, not hidden.
* It is NOT a substitute for the retrieval gate. A question whose terms are
  absent from the whole corpus is an upstream (retrieval/gate) problem; this
  metric only says whether the ANSWER text is about the QUESTION.
* The threshold is a policy constant chosen from the measured distribution
  above (on-domain min 0.300, known off-domain 0.260). The margin is THIN —
  0.04 on a single negative sample — and that is stated here rather than
  presented as a validated separation. `close_call` marks results near the
  threshold so a reviewer can see when a verdict was marginal.

`excess_information` (answer sentences sharing no question term) is reported as
an OBSERVATION only and never fails an answer: background sentences
legitimately explain context without repeating the question's words.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.services.answering.query_processor import extract_terms

#: Published method labels. Recorded on every result so a lexical proxy is
#: never mistaken for a model's judgement.
METHOD_IDF = "idf-weighted-question-term-coverage"
METHOD_PLAIN = "lexical-question-term-coverage"

#: Policy constant: an answer at or below this relevance is treated as not
#: addressing the question. Chosen from the measured distribution documented in
#: the module docstring (on-domain min 0.300, known off-domain 0.260).
MIN_RELEVANCE_TO_PASS = 0.30

#: Within this distance of the threshold a verdict is flagged as marginal so a
#: reviewer can see it was decided by a thin margin.
CLOSE_CALL_MARGIN = 0.05

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class TermWeights(BaseModel):
    """Corpus term statistics used to weight question terms.

    Built from the KB's own BM25 lexical index when one exists. `source`
    records where the numbers came from, so a run with corpus statistics is
    distinguishable from a run without.
    """

    n_docs: int = 0
    df: dict[str, int] = Field(default_factory=dict)
    source: str = ""

    @property
    def available(self) -> bool:
        return self.n_docs > 0 and bool(self.df)

    def idf(self, term: str) -> float:
        from app.services.retrieval.lexical import idf

        return idf(self.n_docs, self.df.get(term, 0))


def term_weights_from_lexical_index(index_data: dict) -> TermWeights | None:
    """Build weights from a persisted lexical index row, or None if unusable.

    A missing/older index simply means the plain (weaker, labelled) method is
    used — it never silently degrades a weighted score into an unweighted one
    under the same name.
    """
    if not isinstance(index_data, dict):
        return None
    n_docs = index_data.get("n_docs")
    df = index_data.get("df")
    if not isinstance(n_docs, int) or n_docs <= 0 or not isinstance(df, dict) or not df:
        return None
    revision = index_data.get("revision", "")
    return TermWeights(
        n_docs=n_docs,
        df={str(k): int(v) for k, v in df.items() if isinstance(v, int)},
        source=f"lexical-index:n_docs={n_docs}" + (f",rev={revision}" if revision != "" else ""),
    )


class RelevanceResult(BaseModel):
    """Relevance of one answer to one question, with its method and limits."""

    value: float | None = None
    measured: bool = True
    reason: str = ""
    method: str = METHOD_PLAIN
    is_model_based: bool = False
    weight_source: str = ""

    question_terms: list[str] = Field(default_factory=list)
    present_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)

    answer_sentences: int = 0
    sentences_without_question_terms: int = 0
    excess_information_ratio: float | None = None

    threshold: float = MIN_RELEVANCE_TO_PASS
    passes: bool | None = None
    close_call: bool = False

    @classmethod
    def not_measured(cls, reason: str, *, method: str = METHOD_PLAIN) -> "RelevanceResult":
        return cls(
            value=None,
            measured=False,
            reason=reason,
            method=method,
            passes=None,
        )


def _sentences(text: str) -> list[str]:
    """Simple deterministic sentence split; documented as a heuristic."""
    parts = [p.strip() for p in _SENTENCE_RE.split((text or "").strip())]
    return [p for p in parts if p]


def compute_relevance(
    question: str,
    answer_text: str,
    weights: TermWeights | None = None,
) -> RelevanceResult:
    """Measure how much of the question's subject the answer actually contains.

    Returns a `RelevanceResult` that is either measured (a value plus its
    method) or explicitly not measured with a reason — never a fabricated 0.
    """
    q_terms = extract_terms(question or "")
    answer = (answer_text or "").strip()

    if not answer:
        return RelevanceResult.not_measured(
            "no answer text to assess (abstention or empty answer); "
            "relevance is not applicable to a refusal"
        )
    if not q_terms:
        return RelevanceResult.not_measured(
            "question has no content terms after stop-word removal, so "
            "relevance cannot be computed"
        )

    answer_terms = set(extract_terms(answer))
    present = [t for t in q_terms if t in answer_terms]
    missing = [t for t in q_terms if t not in answer_terms]

    method = METHOD_PLAIN
    weight_source = "uniform"
    if weights is not None and weights.available:
        method = METHOD_IDF
        weight_source = weights.source
        total = sum(weights.idf(t) for t in q_terms)
        covered = sum(weights.idf(t) for t in present)
        value = (covered / total) if total > 0 else None
    else:
        value = len(present) / len(q_terms)

    if value is None:
        return RelevanceResult.not_measured(
            "question terms carry zero total weight in the corpus statistics",
            method=method,
        )
    value = round(float(value), 6)

    # excess information: observations only, never a failure
    sentences = _sentences(answer)
    q_set = set(q_terms)
    off_topic = sum(1 for s in sentences if not (q_set & set(extract_terms(s))))
    excess = round(off_topic / len(sentences), 6) if sentences else None

    passes = value >= MIN_RELEVANCE_TO_PASS
    close = abs(value - MIN_RELEVANCE_TO_PASS) <= CLOSE_CALL_MARGIN

    reason = (
        f"{len(present)}/{len(q_terms)} question content terms present in the "
        f"answer ({method})"
    )
    if missing:
        reason += f"; missing: {', '.join(missing[:8])}"
    if method == METHOD_PLAIN:
        reason += (
            " — WARNING: no corpus IDF statistics were available; plain "
            "coverage is known to be non-discriminative for off-domain "
            "answers (measured: off-domain 0.400 vs on-domain min 0.333)"
        )

    return RelevanceResult(
        value=value,
        measured=True,
        reason=reason,
        method=method,
        is_model_based=False,
        weight_source=weight_source,
        question_terms=q_terms,
        present_terms=present,
        missing_terms=missing,
        answer_sentences=len(sentences),
        sentences_without_question_terms=off_topic,
        excess_information_ratio=excess,
        threshold=MIN_RELEVANCE_TO_PASS,
        passes=passes,
        close_call=close,
    )


__all__ = [
    "CLOSE_CALL_MARGIN",
    "METHOD_IDF",
    "METHOD_PLAIN",
    "MIN_RELEVANCE_TO_PASS",
    "RelevanceResult",
    "TermWeights",
    "compute_relevance",
    "term_weights_from_lexical_index",
]
