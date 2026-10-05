"""Phase 9 — deterministic query processing.

Design rules:
* The ORIGINAL question is preserved byte-for-byte in `QueryPlan.original_query`;
  every other field is derived from it.
* No LLM is used. Classification is a small ordered rule list, honestly labelled
  `classification_method = "heuristic_rules"` with the matching rules recorded in
  `classification_signals`. Accuracy is not claimed.
* The processor never invents retrieval queries: `retrieval_queries` starts as
  exactly the original question (multi-query expansion / decomposition is a
  future extension point, recorded as a note rather than silently faked).
"""
from __future__ import annotations

import re
import time
import unicodedata
from abc import ABC, abstractmethod

from app.schemas.answer import QueryNature, QueryPlan, QueryTrace, QueryType
from app.schemas.models import DomainSpec, KnowledgeBase

#: Bump when a rule changes behaviour, so stored traces remain interpretable.
PROCESSOR_VERSION = "v7.2"

#: Queries that are conversation, not a request for knowledge. Retrieval for
#: these is wasted work and, worse, invites an answer with no question behind
#: it. Matched against the WHOLE normalized query (anchored), so a real
#: question that merely contains "thanks" is unaffected.
CONVERSATIONAL_RE = re.compile(
    r"^\s*(?:hi|hello|hey|yo|good\s+(?:morning|afternoon|evening)|thanks|thank\s+you|ok|okay|cool|great|nice|bye|goodbye)\b[\s!.,?]*$",
    re.I,
)

#: Short, content-free requests. These are NOT answered and NOT retrieved for;
#: the caller is asked for a specific question instead.
NON_KNOWLEDGE_RE = re.compile(
    r"^\s*(?:help|help\s+me|what\s+can\s+you\s+do|who\s+are\s+you|what\s+are\s+you|test|ping)\b[\s!.,?]*$",
    re.I,
)

#: A question this short, with no domain term and no content term, cannot be
#: searched usefully. Threshold is explicit and deliberately low — it only
#: catches genuine one-word or pronoun-only turns ("and?", "this one", "why").
MIN_SPECIFIC_TERMS = 2

#: Words that make a question unanswerable on their own: they reference
#: something the corpus cannot contain.
_DEICTIC_RE = re.compile(
    r"\b(?:this|that|these|those|it|them|the\s+previous|the\s+last|the\s+former|the\s+latter)\b",
    re.I,
)

#: Words that carry no topic. Used ONLY to judge whether a question is specific
#: enough to search — never to modify the question itself. "What about the
#: previous case?" consists entirely of these, so it is correctly recognised as
#: referring to something outside itself.
_NON_TOPIC = frozenset(
    {
        "about", "above", "again", "also", "another", "any", "anything",
        "case", "certain", "else", "example", "following", "former", "given",
        "latter", "like", "many", "more", "much", "one", "other", "others",
        "previous", "same", "something", "such", "thing", "things", "time",
        "kind", "sort", "way", "part", "parts",
    }
)

#: A question is treated as multi-hop when it asks for a relationship between
#: two named things ("how does X affect Y", "why does X change Y"). This is a
#: HEURISTIC signal that one chunk may not contain the whole answer; it does
#: NOT change retrieval in v7.2 (decomposition is not implemented) — it is
#: recorded so the limitation is visible.
_MULTI_HOP_RE = re.compile(
    r"\b(?:how|why)\s+(?:does|do|is|are|did)\b[\s\S]{0,120}?\b(?:affect|influence|change|relate|interact|depend)\w*\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Tokenization helpers (deterministic, Unicode-safe — V6 lesson: do not split
# on non-ASCII boundaries)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: Minimal English stop-word list. Deliberately small and explicit: a larger
#: list would be a hidden behaviour change, and no stemming is applied (stated
#: limitation, same as BM25).
STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "did",
        "do", "does", "for", "from", "had", "has", "have", "how", "i", "if",
        "in", "into", "is", "it", "its", "of", "on", "or", "our", "s", "should",
        "that", "the", "their", "them", "then", "there", "these", "they",
        "this", "to", "was", "we", "were", "what", "when", "where", "which",
        "who", "why", "will", "with", "you", "your",
    }
)

# ---------------------------------------------------------------------------
# Query-type rules: ordered, each records why it fired.
# ---------------------------------------------------------------------------

_RULES: list[tuple[QueryType, list[re.Pattern[str]]]] = [
    (
        QueryType.MULTI_PART,
        [
            # Two explicit questions …
            re.compile(r"\?[\s\S]{0,400}\?"),
            # … or two interrogative clauses joined by a conjunction.
            re.compile(
                r"\b(?:what|why|how|when|where|who|which)\b[\s\S]{0,200}"
                r"\b(?:and|also|additionally)\b[\s\S]{0,200}"
                r"\b(?:what|why|how|when|where|who|which)\b",
                re.I,
            ),
        ],
    ),
    (
        QueryType.TROUBLESHOOTING,
        [
            re.compile(r"\b(?:not\s+work(?:ing|s)?|fail(?:s|ed|ure)?|error|broken|crash(?:es|ed)?)\b", re.I),
            re.compile(r"\b(?:fix|troubleshoot|diagnose|repair)\b", re.I),
        ],
    ),
    (
        QueryType.COMPARISON,
        [
            re.compile(r"\bcompare\b", re.I),
            re.compile(r"\bversus\b|\bvs\.?\b", re.I),
            re.compile(r"\bdifference\s+between\b", re.I),
            re.compile(r"\bbetter\s+than\b", re.I),
        ],
    ),
    (
        QueryType.PROCEDURAL,
        [
            re.compile(r"\bhow\s+(?:do|does|can|to)\b", re.I),
            re.compile(r"\bsteps?\b", re.I),
            re.compile(r"\bprocedure\b", re.I),
            re.compile(r"\bprocess\s+of\b", re.I),
        ],
    ),
    (
        QueryType.NUMERICAL,
        [
            re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|kg|kn|nm|n\.m\.|m|km|mm|cm|kg/m3|kg\/m\^?3|knots?|hp|kw|w|v|a|pa|mpa|kpa|psi|mm2|mm\^?2)\b", re.I),
            re.compile(r"\b(?:calculate|compute|how\s+many|how\s+much|quantity|amount)\b", re.I),
            re.compile(r"\b\d+(?:[.,]\d+)?"),
        ],
    ),
    (
        QueryType.DEFINITION,
        [
            re.compile(r"\bwhat\s+(?:is|are)\b", re.I),
            re.compile(r"\bdefine\b|\bdefinition\s+of\b", re.I),
            re.compile(r"\bmeaning\s+of\b", re.I),
        ],
    ),
    (
        QueryType.EXPLANATION,
        [
            re.compile(r"\bwhy\b", re.I),
            re.compile(r"\bexplain\b", re.I),
            re.compile(r"\breason\s+for\b", re.I),
            re.compile(r"\bhow\s+does\b[\s\S]*\bwork\b", re.I),
        ],
    ),
]

_DEFAULT_TYPE = QueryType.FACTUAL

#: Phrases that mark a question as open/other rather than a clean fact request.
_OTHER_HINT = re.compile(r"^\s*(?:opinion|thought|prefer|recommend)\b", re.I)


class QueryProcessor(ABC):
    """Produces a structured QueryPlan for one question in one knowledge base.

    Two entry points, deliberately:
    * `process` returns the `QueryPlan` the pipeline consumes (unchanged since
      v7.1, so existing callers keep working).
    * `process_with_trace` returns the `QueryTrace` an auditor reads. The
      default implementation derives it from `process`, so a custom processor
      only has to implement one method and still gets a truthful trace.
    """

    name: str = "base"
    version: str = PROCESSOR_VERSION

    @abstractmethod
    def process(
        self,
        question: str,
        knowledge_base: KnowledgeBase,
        domain_spec: DomainSpec | None = None,
        *,
        filters: dict | None = None,
        requested_answer_format: str = "standard",
    ) -> QueryPlan: ...

    def process_with_trace(
        self,
        question: str,
        knowledge_base: KnowledgeBase,
        domain_spec: DomainSpec | None = None,
        *,
        filters: dict | None = None,
        requested_answer_format: str = "standard",
        expand: bool = False,
    ) -> QueryTrace:
        """Default: derive a truthful trace from `process`.

        Reports that no rewrite/expansion/decomposition happened, because this
        implementation cannot know otherwise. Overriders that DO transform the
        question must record it here.
        """
        started = time.perf_counter()
        plan = self.process(
            question,
            knowledge_base,
            domain_spec,
            filters=filters,
            requested_answer_format=requested_answer_format,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        return QueryTrace(
            original_query=plan.original_query,
            normalized_query=plan.normalized_query,
            rewritten_query=None,
            subqueries=[],
            processor=self.name,
            processor_version=self.version,
            query_type=plan.query_type,
            nature=_nature_of(plan),
            classification_method=plan.classification_method,
            classification_signals=list(plan.classification_signals),
            transformations=[],
            expanded_terms=[],
            warnings=[],
            notes=[
                *plan.notes,
                "query processing trace: derived from QueryPlan by the base "
                "implementation (no rewrite/expansion/decomposition recorded)",
            ],
            timing_ms=round(elapsed, 3),
        )


def _nature_of(plan: QueryPlan) -> QueryNature:
    """Classify the KIND of turn. Deterministic; used by the gate to avoid
    retrieving for conversational or empty turns.

    Order matters: the conversational / non-knowledge patterns are checked
    BEFORE the content-term check, because phrases like "who are you" and
    "thanks" consist entirely of stop-words and would otherwise be reported as
    merely underspecified instead of correctly identified as chit-chat.
    """
    q = plan.normalized_query
    if CONVERSATIONAL_RE.match(q):
        return QueryNature.CONVERSATIONAL
    if NON_KNOWLEDGE_RE.match(q):
        return QueryNature.NON_KNOWLEDGE
    if not q or not plan.extracted_terms:
        return QueryNature.UNDERSPECIFIED
    if _MULTI_HOP_RE.search(q):
        return QueryNature.MULTI_HOP
    return QueryNature.KNOWLEDGE


def normalize(question: str) -> str:
    """Deterministic normalization: NFC unicode, collapse whitespace, trim.

    Casing and punctuation are preserved (a normalized query is for matching,
    not for display; the original stays authoritative).
    """
    text = unicodedata.normalize("NFC", question or "")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens (Unicode word chars, same rule as BM25 tokenizer)."""
    return [t.lower() for t in _WORD_RE.findall(text)]


def extract_terms(text: str) -> list[str]:
    """Content terms: stop-words removed, order preserved, de-duplicated."""
    seen: dict[str, None] = {}
    for tok in _WORD_RE.findall(text.lower()):
        if tok in STOP_WORDS or len(tok) < 2:
            continue
        seen.setdefault(tok, None)
    return list(seen)


def detect_language(text: str) -> tuple[str | None, str]:
    """HEURISTIC script-based language guess.

    Returns (code, note). Uses only Unicode script ranges: ASCII/Latin-dominant
    text is reported as 'en' ONLY when it is exclusively ASCII letters; anything
    else is None with an explicit note. This is a guess, not a language
    detector, and is labelled as such everywhere it surfaces.
    """
    stripped = re.sub(r"[\s\d\W_]+", "", text, flags=re.UNICODE)
    if not stripped:
        return None, "language detection: not attempted (question has no letters)"
    if stripped.isascii():
        return "en", "language detection: HEURISTIC (ASCII-only text assumed English)"
    return None, "language detection: NOT IMPLEMENTED for non-ASCII text"


class HeuristicQueryProcessor(QueryProcessor):
    """Rule-based, deterministic, LLM-free query processor (Phase 9)."""

    name = "heuristic-rules"

    def process(
        self,
        question: str,
        knowledge_base: KnowledgeBase,
        domain_spec: DomainSpec | None = None,
        *,
        filters: dict | None = None,
        requested_answer_format: str = "standard",
    ) -> QueryPlan:
        original = question if question is not None else ""
        normalized = normalize(original)
        notes: list[str] = []

        language, lang_note = detect_language(normalized)
        notes.append(lang_note)

        # --- classification ------------------------------------------------
        query_type = _DEFAULT_TYPE
        signals: list[str] = []
        if _OTHER_HINT.search(normalized):
            query_type = QueryType.OTHER
            signals.append("rule:open-ended-prefix")
        else:
            for qtype, patterns in _RULES:
                for pattern in patterns:
                    if pattern.search(normalized):
                        query_type = qtype
                        signals.append(f"rule:{qtype.value}:{pattern.pattern[:40]}")
                        break
                if query_type is qtype:
                    break
        if query_type is _DEFAULT_TYPE:
            signals.append("rule:default-factual")

        # --- terms ---------------------------------------------------------
        extracted = extract_terms(normalized)
        domain_terms = _match_domain_terms(normalized, domain_spec, knowledge_base)
        if domain_spec is None:
            notes.append(
                "domain terms: no domain specification stored for this knowledge base; "
                "matching limited to the question text"
            )

        # --- retrieval queries ---------------------------------------------
        retrieval_queries = [normalized] if normalized else []
        if query_type is QueryType.MULTI_PART:
            notes.append(
                "query decomposition: NOT IMPLEMENTED; the multi-part question is "
                "retrieved as one query and the plan records that this happened"
            )
        if not retrieval_queries:
            notes.append("retrieval queries: empty after normalization; retrieval will be skipped")

        return QueryPlan(
            original_query=original,
            normalized_query=normalized,
            detected_language=language,
            query_type=query_type,
            classification_method="heuristic_rules",
            classification_signals=signals,
            extracted_terms=extracted,
            domain_terms=domain_terms,
            retrieval_queries=retrieval_queries,
            filters=dict(filters or {}),
            requested_answer_format=requested_answer_format or "standard",
            notes=notes,
        )

    # -- traced variant ------------------------------------------------------

    def process_with_trace(
        self,
        question: str,
        knowledge_base: KnowledgeBase,
        domain_spec: DomainSpec | None = None,
        *,
        filters: dict | None = None,
        requested_answer_format: str = "standard",
        expand: bool = False,
    ) -> QueryTrace:
        """Process a question and record EXACTLY what was done to it.

        Guarantees, in order:
        1. With `expand=False` (the default) the query sent to retrieval is the
           normalized question — `rewritten_query` stays None and
           `transformations` stays empty, so a caller can prove the question was
           not silently altered.
        2. Expansion only ever ADDS terms; the original question is never
           replaced, and every added term is listed in `expanded_terms`.
        3. Decomposition is NOT implemented; a multi-part question is retrieved
           as one query and that limitation is recorded as a warning.
        """
        started = time.perf_counter()
        plan = self.process(
            question,
            knowledge_base,
            domain_spec,
            filters=filters,
            requested_answer_format=requested_answer_format,
        )

        notes: list[str] = list(plan.notes)
        warnings: list[str] = []
        transformations: list[str] = []
        expanded_terms: list[str] = []
        rewritten: str | None = None

        nature = _nature_of(plan)

        # -- underspecified detection ----------------------------------------
        # Counted only when the query is NOT already flagged conversational /
        # non-knowledge, so "hi" is reported as conversation rather than as an
        # underspecified question.
        if nature is QueryNature.KNOWLEDGE and _is_underspecified(plan):
            nature = QueryNature.UNDERSPECIFIED

        # -- expansion (opt-in, additive only) --------------------------------
        if expand and nature is QueryNature.KNOWLEDGE and domain_spec is not None:
            additions = _expansion_terms(plan, domain_spec)
            if additions:
                expanded_terms = additions
                # The question text is NOT modified; the additions are appended
                # as a separate retrieval query so the original wording is
                # still searched verbatim.
                rewritten = None
                transformations.append("expansion:domain-terms")
                notes.append(
                    "query expansion: "
                    f"{len(additions)} domain term(s) added as an extra retrieval "
                    "query; the original question was left unchanged"
                )
            else:
                notes.append(
                    "query expansion: requested but no additional domain term was "
                    "found for this question, so nothing was added"
                )
        elif expand:
            notes.append(
                "query expansion: requested but not applied (no domain specification "
                "or the turn is not a knowledge question)"
            )

        # -- decomposition (explicitly not implemented) ------------------------
        if plan.query_type is QueryType.MULTI_PART:
            warnings.append(
                "multi-part question: query decomposition is NOT IMPLEMENTED, so "
                "every part is retrieved with one query. Parts may therefore be "
                "answered unevenly."
            )
        if nature is QueryNature.MULTI_HOP:
            warnings.append(
                "multi-hop question detected (HEURISTIC): the answer may require "
                "evidence from more than one chunk, and multi-step retrieval is NOT "
                "IMPLEMENTED. Coverage will be reported honestly by the gate."
            )
        if nature is QueryNature.UNDERSPECIFIED:
            warnings.append(
                "question appears underspecified (HEURISTIC): it contains too few "
                "content terms to search the corpus meaningfully."
            )
        if nature in (QueryNature.CONVERSATIONAL, QueryNature.NON_KNOWLEDGE):
            warnings.append(
                f"turn classified as {nature.value}: no knowledge question was asked, "
                "so no evidence can ground an answer."
            )

        elapsed = (time.perf_counter() - started) * 1000.0
        return QueryTrace(
            original_query=plan.original_query,
            normalized_query=plan.normalized_query,
            rewritten_query=rewritten,
            subqueries=[],
            processor=self.name,
            processor_version=self.version,
            query_type=plan.query_type,
            nature=nature,
            classification_method=plan.classification_method,
            classification_signals=list(plan.classification_signals),
            transformations=transformations,
            expanded_terms=expanded_terms,
            warnings=warnings,
            notes=notes,
            timing_ms=round(elapsed, 3),
            enabled=True,
        )

    # -- helpers -------------------------------------------------------------

    def passthrough_trace(self, question: str, *, reason: str) -> QueryTrace:
        """Trace for when query processing is DISABLED by configuration.

        Records the original query byte-for-byte and states plainly that nothing
        was transformed. The caller then uses the original query for retrieval.
        """
        original = question if question is not None else ""
        normalized = normalize(original)
        return QueryTrace(
            original_query=original,
            normalized_query=normalized,
            rewritten_query=None,
            subqueries=[],
            processor="passthrough",
            processor_version=PROCESSOR_VERSION,
            query_type=QueryType.OTHER,
            nature=QueryNature.KNOWLEDGE,
            classification_method="disabled",
            classification_signals=[],
            transformations=[],
            expanded_terms=[],
            warnings=[],
            notes=[reason, "query processing disabled: the original query was used unchanged"],
            timing_ms=0.0,
            enabled=False,
        )


def _is_underspecified(plan: QueryPlan) -> bool:
    """HEURISTIC: too little to search on.

    Deliberately conservative, because the cost of a false positive is refusing
    a legitimate question. Fires only when:

    * the question carries NO content term at all ("and?", "ok so?"), or
    * it leans on a deictic reference AND carries at most MIN_SPECIFIC_TERMS
      content words ("what about the previous case"),

    and in both cases the question matches no domain term. A short but specific
    question ("what is buoyancy?", "metacentric height?") never trips this.
    """
    if plan.domain_terms:
        return False
    # NOTE: no minimum length filter here. Engineering questions are full of
    # two-letter acronyms (GM, KG, KB, LCG) that a length filter would wrongly
    # discard, turning "What is GM?" into an underspecified turn. Instead,
    # non-topic words are excluded so "the previous case" counts as zero.
    content = [t for t in plan.extracted_terms if t not in _NON_TOPIC]
    if not content:
        return True
    return bool(_DEICTIC_RE.search(plan.normalized_query)) and len(content) <= MIN_SPECIFIC_TERMS


def _expansion_terms(plan: QueryPlan, spec: DomainSpec) -> list[str]:
    """Domain terms related to the question that the question did not already
    contain. Deterministic substring/term matching — no LLM, no thesaurus.

    Returns at most 3 terms, ordered by the specification's own order, so the
    result is reproducible.
    """
    question_lower = plan.normalized_query.lower()
    question_terms = {t.lower() for t in plan.extracted_terms}
    candidates = [*spec.terminology, *spec.key_concepts]
    additions: list[str] = []
    for candidate in candidates:
        term = candidate.strip()
        low = term.lower()
        if not term or len(low) < 4:
            continue
        if low in question_lower or low in question_terms:
            continue
        # Only add terms that are lexically adjacent to something already in
        # the question (share a token). Otherwise "expansion" would just be
        # dumping the whole domain into the query.
        tokens = set(tokenize(term))
        if not (tokens & question_terms):
            continue
        if term in additions:
            continue
        additions.append(term)
        if len(additions) >= 3:
            break
    return additions


def _match_domain_terms(
    question: str, spec: DomainSpec | None, kb: KnowledgeBase
) -> list[str]:
    """Question terms that appear in the KB's domain specification.

    Deterministic substring matching against terminology/concepts/entities —
    no fuzzy matching, no LLM.
    """
    if spec is None:
        return []
    haystack_phrases = [
        *spec.terminology,
        *spec.key_concepts,
        *spec.entities,
        *spec.subdomains,
    ]
    lowered = question.lower()
    found: dict[str, None] = {}
    for phrase in haystack_phrases:
        p = phrase.strip().lower()
        if p and p in lowered:
            found.setdefault(phrase.strip(), None)
    # Also mark single extracted terms that appear in the terminology list.
    term_set = {t.lower() for t in spec.terminology}
    for tok in tokenize(question):
        if tok in term_set:
            found.setdefault(tok, None)
    return list(found)


def create_query_processor() -> QueryProcessor:
    """Factory. Single deterministic implementation today; the ABC exists so an
    LLM-backed or learned processor can be added without touching callers."""
    return HeuristicQueryProcessor()


__all__ = [
    "CONVERSATIONAL_RE",
    "HeuristicQueryProcessor",
    "MIN_SPECIFIC_TERMS",
    "NON_KNOWLEDGE_RE",
    "PROCESSOR_VERSION",
    "QueryProcessor",
    "create_query_processor",
    "detect_language",
    "extract_terms",
    "normalize",
    "tokenize",
]
