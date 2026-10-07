"""The answer evaluator: scores a grounded answer against evidence-based truth.

Core rule: **"a citation exists" is not "a citation is correct"**. Every metric
here is computed from the claim→evidence mapping and the benchmark's required
evidence, never from the fact that a citation object was emitted.

What is measured, and how:

* `citation_precision`   — of the chunks the answer CITED, what fraction are
                           labelled as required for this question. This is a
                           LOWER BOUND: the required set is a human's selection
                           of *necessary* evidence, not an exhaustive whitelist,
                           so citing extra retrieved context lowers this number
                           without necessarily being an error.
* `citation_recall`      — of the chunks REQUIRED, what fraction were cited.
                           An answer that cites nothing scores 0, not "n/a".
* `evidence_support_rate`— of claims with resolved citations, what fraction the
                           entailment provider judged SUPPORTED.
* `unsupported_claim_rate`— claims asserting fact with no resolving citation.
* `supported/partial/unsupported_claim_ratio` — claims bucketed into the
                           five-state vocabulary (SUPPORTED, PARTIALLY_SUPPORTED,
                           UNSUPPORTED, CONTRADICTED, UNVERIFIABLE), computed
                           over ALL claims so unverifiable claims cannot hide.
* `fabricated_citation_rate` — evidence references that do not resolve to
                           retrieved evidence or fail provenance validation.
* `unsupported_citation_rate` — citations on claims the entailment provider
                           judged NOT_SUPPORTED (over JUDGED citations only).
* `contradiction_rate`   — claims whose cited evidence was judged NOT_SUPPORTED.
* `abstention_correct`   — did the system abstain exactly when it should?
* `retrieval_hit_rate`   — was the required evidence even retrieved? Reported
                           SEPARATELY from citation quality so a retrieval miss
                           is never blamed on the answerer (or vice versa).
* `question_answer_relevance` — does the answer text actually contain the
                           question's subject terms? Groundedness alone cannot
                           detect a grounded answer to the WRONG question. Uses
                           corpus IDF weights when a lexical index exists
                           (measured to separate on- vs off-domain answers);
                           without them the plain fallback is reported and its
                           shortfall is a WARNING, never a failure, because
                           plain coverage was measured to be non-discriminative
                           (see `relevance.py` for the evidence table).
                           Abstentions are NOT scored for relevance — a refusal
                           is not "irrelevant".

What is NOT measured, and reported as UNKNOWN with a reason:

* `correctness` — requires human review or an explicitly model-based judge.
  When a HUMAN reference answer exists, lexical similarity to it IS reported
  (`reference_answer_similarity`), but it is deliberately not renamed
  "correctness": a proxy must not be promoted to a verdict.
* `key_point_recall` / `expected_information_coverage` — measured ONLY when a
  human authored key points (lexical coverage proxy, threshold published).
  Without human labels they stay UNKNOWN rather than being approximated with
  fuzzy string matching against retrieved text — that would measure "did we
  retrieve the expected chunk", which is already reported separately.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from app.schemas.answer import Answer, AnswerStatus, Claim, Evidence, GroundingState
from app.services.answering.query_processor import extract_terms
from app.services.answer_eval.benchmark import AnswerBenchmarkQuestion, Answerability
from app.services.answer_eval.entailment import (
    CitationEntailmentEvaluator,
    SupportJudgement,
)
from app.services.answer_eval.metrics import (
    AnswerQualityResult,
    ClaimCitationVerdict,
    Measured,
    compute_final_score,
    grounding_states_compatible,
)
from app.services.answer_eval.relevance import (
    CLOSE_CALL_MARGIN,
    METHOD_PLAIN,
    MIN_RELEVANCE_TO_PASS,
    RelevanceResult,
    TermWeights,
    compute_relevance,
)
from app.services.answer_eval.review import AnswerReview, verdict_score

logger = logging.getLogger(__name__)

EVALUATOR_VERSION = "v8.3"

#: Fraction of a human key point's content terms that must appear in the
#: answer text for that point to count as covered. Matches the heuristic
#: entailment support threshold (0.6) so one published number governs both
#: lexical judgements — it is a policy constant, not a tuned-per-run value.
KEY_POINT_COVERAGE_THRESHOLD = 0.6


class AnswerEvaluator(ABC):
    """Scores one answer against one benchmark question."""

    name: str = "base"
    version: str = EVALUATOR_VERSION
    #: True when THIS evaluator's judgements come from a model (LLM-as-judge).
    #: Persisted on every result so model-based evaluation is never
    #: indistinguishable from a deterministic one.
    is_model_based: bool = False
    #: Which model / reviewers produced this evaluator's judgements.
    detail: str = ""

    @abstractmethod
    def evaluate(
        self,
        question: AnswerBenchmarkQuestion,
        answer: Answer,
        evidence: list[Evidence],
    ) -> AnswerQualityResult: ...

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


class DeterministicAnswerEvaluator(AnswerEvaluator):
    """Evidence-based, offline, deterministic evaluator. The default path.

    Requires no network, no model and no API key, so every metric it reports is
    reproducible. Its citation-support judgements come from the injected
    `CitationEntailmentEvaluator`, whose identity is recorded on every result so
    a heuristic verdict is never mistaken for a model's.
    """

    name = "deterministic-evidence"
    version = EVALUATOR_VERSION

    #: Answer statuses that do not produce an answer to be relevant. Relevance
    #: is NOT scored for these — a refusal is not an irrelevant answer.
    _NON_ANSWER_STATUSES = (
        AnswerStatus.ABSTAINED,
        AnswerStatus.CLARIFICATION_REQUIRED,
        AnswerStatus.GENERATION_FAILED,
    )

    def __init__(
        self,
        entailment: CitationEntailmentEvaluator | None = None,
        *,
        weights: TermWeights | None = None,
    ) -> None:
        from app.services.answer_eval.entailment import create_entailment_evaluator

        self._entailment = entailment or create_entailment_evaluator("heuristic")
        self._weights = weights

    def set_weights(self, weights: TermWeights | None) -> None:
        """Inject corpus term statistics for IDF-weighted relevance.

        Set by the evaluation service per knowledge base. When None, relevance
        falls back to the plain method, whose results carry an explicit warning
        (see `relevance.py`).
        """
        self._weights = weights

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _required_chunk_ids(question: AnswerBenchmarkQuestion) -> set[str]:
        return {e.chunk_id for e in question.required_evidence if e.required}

    @staticmethod
    def _evidence_text_for_chunk(evidence: list[Evidence], chunk_id: str) -> str:
        for item in evidence:
            if item.chunk_id == chunk_id:
                return item.content or ""
        return ""

    def _audit_claim(
        self,
        claim: Claim,
        question: AnswerBenchmarkQuestion,
        evidence_by_id: dict[str, Evidence],
        required: set[str],
    ) -> ClaimCitationVerdict:
        """Deterministic per-claim citation audit."""
        resolved: list[str] = []
        has_invalid = False
        problems: list[str] = []

        for eid in claim.evidence_ids:
            item = evidence_by_id.get(eid)
            if item is None:
                # The pipeline already rejects these; if one reaches here the
                # answer was produced outside the validated path.
                has_invalid = True
                problems.append(
                    f"claim {claim.claim_id} cites evidence {eid}, which is not in "
                    f"the retrieved evidence set"
                )
                continue
            resolved.append(item.chunk_id)

        cited = set(resolved)
        missing_required = sorted(required - cited)
        irrelevant = sorted(cited - required) if required else []

        entailment = SupportJudgement.UNKNOWN
        entailment_detail = ""
        method = ""

        if claim.evidence_ids and evidence_by_id:
            # Judge the claim against the evidence it actually cited, combined.
            # A claim supported by ANY cited chunk counts as supported; a claim
            # supported by NONE is not_supported.
            per_item: list[SupportJudgement] = []
            details: list[str] = []
            methods: set[str] = set()
            for eid in claim.evidence_ids:
                item = evidence_by_id.get(eid)
                if item is None:
                    continue
                res = self._entailment.evaluate(claim.text, item.content or "")
                per_item.append(res.judgement)
                details.append(f"{eid}: {res.judgement.value} ({res.detail})")
                methods.add(res.method)
            if per_item:
                if any(j is SupportJudgement.SUPPORTED for j in per_item):
                    entailment = SupportJudgement.SUPPORTED
                elif all(j is SupportJudgement.NOT_SUPPORTED for j in per_item):
                    entailment = SupportJudgement.NOT_SUPPORTED
                else:
                    entailment = SupportJudgement.UNKNOWN
                entailment_detail = "; ".join(details)
                method = ",".join(sorted(methods))

        # Deliberately NOT enforced per claim:
        #   * must_cite_all_required_evidence — the required set is the evidence a
        #     human judged NECESSARY. Completeness is a property of the answer as
        #     a whole; a claim about alternative fuels has no reason to cite the
        #     engine-definition chunk. Enforcing it per claim manufactured a
        #     problem for every claim of every multi-claim answer.
        #   * must_not_cite_irrelevant_chunks — the benchmark never established
        #     that an unlabelled chunk is irrelevant, only that a human picked a
        #     different canonical chunk. Calling it "irrelevant" asserts ground
        #     truth we do not have, so the per-claim signal is recorded on the
        #     verdict and left to the answer-level precision metric.
        # Both fields stay populated so failure analysis can still show them.
        if question.citation_requirements.require_at_least_one_citation and not claim.evidence_ids:
            problems.append(
                f"claim {claim.claim_id} states a factual assertion with no citation"
            )
        if entailment is SupportJudgement.NOT_SUPPORTED:
            problems.append(
                f"claim {claim.claim_id} is not supported by the evidence it cites "
                f"({self._entailment.name}, lexical coverage only)"
            )

        # -- five-state verdict (V8 STEP 6) ----------------------------------
        # Entailment WINS over the generator's own say-so: a claim the
        # provider could not contradict but the generator declared partial (or
        # even unsupported) lands in the honest middle ground rather than
        # being upgraded to SUPPORTED. An uncited factual assertion is
        # UNSUPPORTED regardless of how plausible it sounds; a claim whose
        # support could not be determined (thin evidence, ambiguous coverage,
        # no evidence to check against) is UNVERIFIABLE, never silently
        # 'supported'.
        factual_uncited = not claim.evidence_ids and claim.claim_type.value not in (
            "missing_information", "abstention"
        )
        if entailment is SupportJudgement.NOT_SUPPORTED:
            evaluated_state = "CONTRADICTED"
        elif factual_uncited:
            evaluated_state = "UNSUPPORTED"
        elif entailment is SupportJudgement.SUPPORTED:
            declared = claim.support_status.value
            evaluated_state = (
                "PARTIALLY_SUPPORTED"
                if declared in ("partially_supported", "unsupported")
                else "SUPPORTED"
            )
        else:
            evaluated_state = "UNVERIFIABLE"

        return ClaimCitationVerdict(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            support_status=claim.support_status.value,
            evaluated_state=evaluated_state,
            cited_evidence_ids=list(claim.evidence_ids),
            resolved_chunk_ids=sorted(cited),
            missing_required_chunk_ids=missing_required,
            irrelevant_chunk_ids=irrelevant,
            entailment=entailment,
            entailment_detail=entailment_detail,
            entailment_method=method,
            has_invalid_citation=has_invalid,
            uncited=not claim.evidence_ids and claim.claim_type.value
            not in ("missing_information", "abstention"),
            problems=problems,
        )

    # -- main entry point ----------------------------------------------------

    def evaluate(
        self,
        question: AnswerBenchmarkQuestion,
        answer: Answer,
        evidence: list[Evidence],
    ) -> AnswerQualityResult:
        required = self._required_chunk_ids(question)
        evidence_by_id = {e.evidence_id: e for e in evidence}
        retrieved_chunk_ids = [e.chunk_id for e in evidence]
        cited_chunk_ids = sorted({c.chunk_id for c in answer.citations})

        warnings: list[str] = []
        problems: list[str] = []

        verdicts = [self._audit_claim(c, question, evidence_by_id, required)
                    for c in answer.claims]
        for v in verdicts:
            problems.extend(v.problems)

        # -- citation precision / recall ------------------------------------
        if cited_chunk_ids and required:
            precision = len(set(cited_chunk_ids) & required) / len(set(cited_chunk_ids))
            citation_precision = Measured.of(
                precision,
                reason="fraction of CITED chunks that are required for this question",
                sample_size=len(set(cited_chunk_ids)),
            )
        elif not cited_chunk_ids and answer.status is AnswerStatus.ABSTAINED:
            citation_precision = Measured.unknown(
                "abstained: no citations expected, precision is not applicable"
            )
        elif not cited_chunk_ids:
            citation_precision = Measured.of(
                0.0, reason="answer cited nothing while producing an answer", sample_size=0
            )
        else:
            citation_precision = Measured.unknown(
                "question has no required-evidence label, so citation precision "
                "cannot be computed"
            )

        if required:
            hit = len(set(cited_chunk_ids) & required)
            citation_recall = Measured.of(
                hit / len(required),
                reason="fraction of REQUIRED chunks that were cited",
                sample_size=len(required),
            )
            citation_completeness = Measured.of(
                hit / len(required),
                reason="alias of citation_recall; both required chunks cited means complete",
                sample_size=len(required),
            )
        elif answer.status is AnswerStatus.ABSTAINED:
            citation_recall = Measured.unknown(
                "abstained: no citations expected, recall is not applicable"
            )
            citation_completeness = citation_recall
        else:
            citation_recall = Measured.unknown(
                "no required-evidence label for this question"
            )
            citation_completeness = citation_recall

        # -- evidence support rate -------------------------------------------
        judged = [v for v in verdicts if v.entailment is not SupportJudgement.UNKNOWN]
        if judged:
            supported = sum(1 for v in judged if v.entailment is SupportJudgement.SUPPORTED)
            evidence_support_rate = Measured.of(
                supported / len(judged),
                reason=f"fraction of judged claims judged SUPPORTED by {self._entailment.name}",
                sample_size=len(judged),
            )
            contradiction_rate = Measured.of(
                sum(1 for v in judged if v.entailment is SupportJudgement.NOT_SUPPORTED) / len(judged),
                reason="fraction of judged claims contradicted by their cited evidence",
                sample_size=len(judged),
            )
        else:
            evidence_support_rate = Measured.unknown(
                f"no claim received a {self._entailment.name} verdict (UNKNOWN judgements only)"
            )
            contradiction_rate = Measured.unknown("no claim was judged")

        # -- unsupported claim rate --------------------------------------------
        factual = [
            v for v in verdicts if v.uncited
        ]
        if verdicts:
            unsupported_claim_rate = Measured.of(
                len(factual) / len(verdicts),
                reason="fraction of claims asserting fact without a resolving citation",
                sample_size=len(verdicts),
            )
        else:
            unsupported_claim_rate = Measured.unknown("answer produced no claims")

        # -- claim-state ratios (V8 STEP 6) ---------------------------------
        if verdicts:
            states = [v.evaluated_state for v in verdicts]
            n = len(states)
            supported_claim_ratio = Measured.of(
                states.count("SUPPORTED") / n,
                reason="fraction of ALL claims whose evidence chain was judged "
                       "SUPPORTED (UNVERIFIABLE claims count against it, not "
                       "out of it)",
                sample_size=n,
            )
            partial_claim_ratio = Measured.of(
                states.count("PARTIALLY_SUPPORTED") / n,
                reason="fraction of ALL claims only partially backed by their "
                       "evidence chain",
                sample_size=n,
            )
            unsupported_claim_ratio = Measured.of(
                (states.count("UNSUPPORTED") + states.count("CONTRADICTED")) / n,
                reason="fraction of ALL claims that are uncited (UNSUPPORTED) "
                       "or contradicted by their own evidence (CONTRADICTED). "
                       "UNVERIFIABLE claims stay in the denominator, so "
                       "supported + partial + unsupported + unverifiable = 1",
                sample_size=n,
            )
        else:
            supported_claim_ratio = Measured.unknown("answer produced no claims")
            partial_claim_ratio = Measured.unknown("answer produced no claims")
            unsupported_claim_ratio = Measured.unknown("answer produced no claims")

        # -- citation fabrication / support (V8 STEP 7) ---------------------
        refs: set[str] = {eid for c in answer.claims for eid in c.evidence_ids}
        refs |= {c.evidence_id for c in answer.citations}
        unresolved = {eid for eid in refs if eid not in evidence_by_id}
        provenance_invalid = {
            c.evidence_id for c in answer.citations if c.validation == "invalid"
        }
        fabricated = (unresolved | provenance_invalid) & refs
        if refs:
            fabricated_citation_rate = Measured.of(
                len(fabricated) / len(refs),
                reason=(
                    f"{len(fabricated)}/{len(refs)} distinct evidence references "
                    f"are unresolvable to retrieved evidence "
                    f"({len(unresolved)}) or provenance-invalid "
                    f"({len(provenance_invalid)})"
                ),
                sample_size=len(refs),
            )
        elif answer.status is AnswerStatus.ABSTAINED:
            fabricated_citation_rate = Measured.unknown(
                "abstained: no citations expected, fabrication rate not applicable"
            )
        else:
            fabricated_citation_rate = Measured.unknown(
                "answer referenced no evidence, so there is nothing that could "
                "be fabricated (citation_recall reports the absence instead)"
            )

        judged_ref_count = 0
        unsupported_ref_count = 0
        for v in verdicts:
            if v.entailment is SupportJudgement.UNKNOWN:
                continue
            cited_n = len(v.resolved_chunk_ids)
            judged_ref_count += cited_n
            if v.entailment is SupportJudgement.NOT_SUPPORTED:
                unsupported_ref_count += cited_n
        if judged_ref_count:
            unsupported_citation_rate = Measured.of(
                unsupported_ref_count / judged_ref_count,
                reason=(
                    f"citations on claims judged NOT_SUPPORTED by "
                    f"{self._entailment.name}, over {judged_ref_count} judged "
                    f"citation(s); unjudged citations are excluded, not counted "
                    f"as supported"
                ),
                sample_size=judged_ref_count,
            )
        else:
            unsupported_citation_rate = Measured.unknown(
                f"no citation received a {self._entailment.name} verdict "
                f"(UNKNOWN judgements only, or no citations at all)"
            )

        # -- retrieval hit rate (kept separate from citation quality) ------------
        rank_of_required: int | None = None
        if required and retrieved_chunk_ids:
            for i, cid in enumerate(retrieved_chunk_ids, start=1):
                if cid in required:
                    rank_of_required = i
                    break
            retrieval_hit_rate = Measured.of(
                1.0 if rank_of_required else 0.0,
                reason="was the required evidence retrieved at all (separate from citation quality)",
                sample_size=1,
            )
        elif required:
            retrieval_hit_rate = Measured.of(
                0.0, reason="required evidence was not retrieved at all", sample_size=1
            )
        else:
            retrieval_hit_rate = Measured.unknown("no required-evidence label")

        # -- answer-level citation completeness ----------------------------------
        # Necessary-evidence recall is enforced here, once per answer.
        missing_at_answer = sorted(required - set(cited_chunk_ids))
        if (
            question.citation_requirements.must_cite_all_required_evidence
            and missing_at_answer
        ):
            problems.append(
                f"answer does not cite the required evidence "
                f"{', '.join(missing_at_answer)}"
            )

        # Chunks cited that the benchmark does not label as required. Reported as
        # a warning, never as a failure: the required set is a necessary-evidence
        # selection, so extra retrieved context is not provably wrong.
        extra_cited = sorted(set(cited_chunk_ids) - required) if required else []
        if question.citation_requirements.must_not_cite_irrelevant_chunks and extra_cited:
            warnings.append(
                f"cites {len(extra_cited)} chunk(s) not labelled as required for this "
                f"question ({', '.join(extra_cited)}). The benchmark records which "
                f"evidence is NECESSARY, not an exhaustive list of permitted evidence, "
                f"so this lowers citation_precision without necessarily being an error."
            )

        # -- completeness from HUMAN labels (V8 STEP 2/F) ---------------------
        # Only a human-authored label can say what the answer SHOULD contain.
        # The check is lexical coverage, published as such: a point counts as
        # covered when >= KEY_POINT_COVERAGE_THRESHOLD of its content terms
        # appear in the answer text. A deterministic proxy — never called a
        # semantic judgement, and never called "correctness".
        produced_content = bool(answer.text.strip()) and answer.status not in self._NON_ANSWER_STATUSES
        answer_terms = set(extract_terms(answer.text)) if produced_content else set()

        if question.key_points:
            n_points = len(question.key_points)
            if produced_content:
                covered_points = 0
                for point in question.key_points:
                    terms = extract_terms(point)
                    if not terms:
                        continue
                    hits = sum(1 for t in terms if t in answer_terms)
                    if hits / len(terms) >= KEY_POINT_COVERAGE_THRESHOLD:
                        covered_points += 1
                coverage_value = covered_points / n_points
                coverage_reason = (
                    f"{covered_points}/{n_points} human-authored key points covered "
                    f"(>= {KEY_POINT_COVERAGE_THRESHOLD:.0%} of each point's content "
                    f"terms present in the answer; LEXICAL PROXY, not semantic "
                    f"entailment)"
                )
            else:
                coverage_value = 0.0
                coverage_reason = (
                    f"answer produced no content ({answer.status.value}); none of "
                    f"the {n_points} human-authored key points were provided"
                )
            key_point_recall = Measured.of(
                coverage_value, reason=coverage_reason, sample_size=n_points
            )
            expected_information_coverage = Measured.of(
                coverage_value,
                reason=coverage_reason + "; identical to key_point_recall",
                sample_size=n_points,
            )
        else:
            key_point_recall = Measured.unknown(
                "no human-authored key points for this question; the frozen "
                "benchmark defines which CHUNK answers the question, not what "
                "the answer must say"
            )
            expected_information_coverage = Measured.unknown(
                "no human-authored key points/required facts for this question"
            )

        # -- reference answer similarity (deliberately NOT correctness) --------
        if question.expected_answer:
            ref_terms = extract_terms(question.expected_answer)
            if not produced_content:
                reference_answer_similarity = Measured.of(
                    0.0,
                    reason="answer produced no content; none of the reference "
                           "answer's terms could be present",
                )
            elif not ref_terms:
                reference_answer_similarity = Measured.unknown(
                    "reference answer has no content terms to compare against"
                )
            else:
                hits = sum(1 for t in ref_terms if t in answer_terms)
                reference_answer_similarity = Measured.of(
                    hits / len(ref_terms),
                    reason=(
                        f"{hits}/{len(ref_terms)} reference-answer content terms "
                        f"present in the answer (LEXICAL similarity to a human "
                        f"reference; NOT a correctness judgement)"
                    ),
                    sample_size=len(ref_terms),
                )
        else:
            reference_answer_similarity = Measured.unknown(
                "no human-authored reference answer for this question"
            )

        # Correctness stays UNKNOWN even when a reference exists: a lexical
        # similarity number is reported as `reference_answer_similarity`, and
        # relabelling it "correctness" would upgrade a proxy into a verdict.
        # A correctness verdict needs human review (STEP 4) or an explicitly
        # model-based judge (STEP 5).
        correctness = Measured.unknown(
            "a human reference answer exists; lexical similarity to it is "
            "reported separately as reference_answer_similarity. Calling that "
            "proxy 'correctness' would overstate it — correctness requires "
            "human review or an explicitly model-based judge."
            if question.expected_answer
            else "no human-authored reference answer or key points for this question. "
            "The frozen benchmark defines which CHUNK answers the question, not what "
            "the answer should say. Scoring correctness against retrieved text would "
            "re-measure retrieval and overstate answer quality."
        )

        # -- abstention behaviour --------------------------------------------------
        abstained = answer.status is AnswerStatus.ABSTAINED or not answer.citations
        expects_abstention = question.abstention_required
        if question.answerability is Answerability.UNANSWERABLE:
            abstention_correct = abstained
        elif question.answerability is Answerability.ANSWERABLE:
            abstention_correct = not abstained
        else:
            abstention_correct = None  # UNKNOWN: no human answerability determination

        actual_state = (
            answer.assessment.grounding_state.value
            if answer.assessment is not None
            else "UNKNOWN"
        )
        expected_state = question.expected_grounding_state.value
        state_correct: bool | None = None
        if expected_state != "ANY_ACCEPTABLE":
            state_correct = grounding_states_compatible(expected_state, actual_state)
        elif actual_state != "UNKNOWN":
            state_correct = True  # no specific expectation; not penalised

        # "false supported" = answered confidently without the required evidence.
        false_supported = bool(
            question.answerability is Answerability.ANSWERABLE
            and not abstained
            and required
            and not (set(cited_chunk_ids) & required)
        )
        false_unsupported = bool(
            question.answerability is Answerability.UNANSWERABLE and not abstained
        )

        if false_supported:
            problems.append(
                "answered without citing the evidence required for this question "
                "(confident answer, unsupported corpus coverage)"
            )
        if false_unsupported:
            problems.append(
                "the corpus does not contain the answer to this question, but the "
                "system answered anyway (hallucination risk)"
            )

        # -- question/answer relevance (V8 STEP 8) ------------------------------
        # Groundedness cannot catch an answer that is perfectly grounded about
        # the WRONG subject. Abstentions are exempt: a refusal is not an
        # irrelevant answer, and scoring it would invent a failure.
        if answer.status in self._NON_ANSWER_STATUSES:
            relevance = RelevanceResult.not_measured(
                f"answer status is {answer.status.value}: relevance applies to "
                f"answers that speak to the question, not to refusals/abstentions"
            )
        else:
            relevance = compute_relevance(question.question, answer.text, self._weights)

        if relevance.measured and relevance.value is not None:
            question_answer_relevance = Measured.of(
                relevance.value,
                reason=relevance.reason,
                sample_size=len(relevance.question_terms),
            )
            if relevance.excess_information_ratio is not None:
                excess_information = Measured.of(
                    relevance.excess_information_ratio,
                    reason=(
                        f"{relevance.sentences_without_question_terms}/"
                        f"{relevance.answer_sentences} answer sentences share no "
                        f"question content term (OBSERVATION ONLY, never a failure)"
                    ),
                    sample_size=relevance.answer_sentences,
                )
            else:
                excess_information = Measured.unknown("no answer sentence to split")
        else:
            question_answer_relevance = Measured.unknown(
                relevance.reason or "relevance was not measured"
            )
            excess_information = Measured.unknown(
                relevance.reason or "relevance was not measured"
            )

        if relevance.passes is False:
            shortfall = (
                f"question/answer relevance {relevance.value} is below the "
                f"{relevance.threshold} threshold ({relevance.method}): "
                f"{relevance.reason}"
            )
            if relevance.method == METHOD_PLAIN:
                # The plain method was measured to be INVERTED on known
                # off-domain answers, so its shortfall is reported but must not
                # flip a verdict — a non-discriminative signal cannot be a judge.
                warnings.append(
                    "relevance shortfall reported but NOT failed: " + shortfall
                    + ". Plain coverage is measured to be non-discriminative "
                    "for off-domain answers, so corpus IDF statistics are "
                    "required before a relevance verdict can fail an answer."
                )
            else:
                problems.append(shortfall)
        if relevance.close_call and relevance.measured:
            warnings.append(
                f"relevance verdict is a close call: {relevance.value} vs "
                f"threshold {relevance.threshold} (within {CLOSE_CALL_MARGIN}); "
                f"the pass/fail was decided by a thin margin"
            )

        result = AnswerQualityResult(
            question_id=question.question_id,
            question=question.question,
            answerability=question.answerability.value,
            answer_id=answer.answer_id,
            answer_trace_id=answer.answer_trace_id,
            answer_status=answer.status.value,
            answer_text=answer.text,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            citation_completeness=citation_completeness,
            evidence_support_rate=evidence_support_rate,
            unsupported_claim_rate=unsupported_claim_rate,
            contradiction_rate=contradiction_rate,
            retrieval_hit_rate=retrieval_hit_rate,
            correctness=correctness,
            key_point_recall=key_point_recall,
            expected_information_coverage=expected_information_coverage,
            reference_answer_similarity=reference_answer_similarity,
            abstention_expected=expects_abstention,
            abstention_performed=abstained,
            abstention_correct=abstention_correct,
            expected_grounding_state=expected_state,
            actual_grounding_state=actual_state,
            grounding_state_correct=state_correct,
            false_supported=false_supported,
            false_unsupported=false_unsupported,
            claim_verdicts=verdicts,
            required_chunk_ids=sorted(required),
            cited_chunk_ids=cited_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_rank_of_required=rank_of_required,
            warnings=warnings,
            problems=problems,
            evaluator_version=self.version,
            evaluator_name=self.name,
            evaluator_is_model_based=self.is_model_based,
            evaluator_detail=self.detail,
            entailment_provider=self._entailment.name,
            entailment_is_model_based=self._entailment.is_model_based,
            question_answer_relevance=question_answer_relevance,
            excess_information=excess_information,
            supported_claim_ratio=supported_claim_ratio,
            partial_claim_ratio=partial_claim_ratio,
            unsupported_claim_ratio=unsupported_claim_ratio,
            fabricated_citation_rate=fabricated_citation_rate,
            unsupported_citation_rate=unsupported_citation_rate,
            relevance_method=relevance.method,
            relevance_is_model_based=relevance.is_model_based,
            relevance_close_call=relevance.close_call,
            relevance_weight_source=relevance.weight_source,
            relevance_passed=relevance.passes,
        )
        score, computable = compute_final_score(result)
        result.final_score = score
        result.final_score_computable = computable
        return result


def apply_reviews(
    result: AnswerQualityResult,
    reviews: list[AnswerReview],
    *,
    evaluator_name: str = "human-reviews",
) -> AnswerQualityResult:
    """Overlay human review verdicts onto a stored per-question result.

    Used by both `HumanAnswerEvaluator` (during evaluation) and the service's
    derived human pass (from a stored run). Returns a NEW result — the input is
    never mutated, and the source run stays byte-identical.

    Only `correctness` changes: human verdicts are the one ground truth we have
    for whether an answer was actually right. Every mechanical metric keeps the
    value the deterministic evaluator measured.
    """
    if not reviews:
        result.correctness = Measured.unknown(
            "no human reviews recorded for this answer yet; correctness stays "
            "UNKNOWN until a reviewer judges it"
        )
        return result

    scores = [verdict_score(r.verdict) for r in reviews]
    reviewers = sorted({r.reviewer for r in reviews})
    verdicts = ", ".join(f"{r.reviewer}={r.verdict.value}" for r in reviews)
    result.correctness = Measured.of(
        sum(scores) / len(scores),
        reason=(
            f"mean of {len(scores)} human review verdict(s): {verdicts} "
            f"(ordinal convention correct=1.0, mostly=0.75, partial=0.5, "
            f"incorrect/should-abstain=0.0 — a policy mapping, not a measurement)"
        ),
        sample_size=len(scores),
    )
    result.evaluator_name = evaluator_name
    result.evaluator_is_model_based = False
    result.evaluator_detail = f"reviewers: {', '.join(reviewers)} ({len(scores)} review(s))"
    result.warnings.append(
        f"correctness populated from {len(scores)} human review(s) by "
        f"{', '.join(reviewers)}; raw verdicts remain on the review records"
    )
    return result


class HumanAnswerEvaluator(AnswerEvaluator):
    """Turns stored human reviews into correctness metrics (V8 STEP 5/14).

    Not a model: reviews are the human's own verdicts, so
    `is_model_based` stays False — but `evaluator_detail` records WHO, because
    an attributed review is evidence and an anonymous one is not.
    """

    name = "human-reviews"
    version = EVALUATOR_VERSION
    is_model_based = False

    def __init__(self, reviews_provider, base: DeterministicAnswerEvaluator | None = None) -> None:
        """`reviews_provider(question_id, answer_id) -> list[AnswerReview]`.

        Raises when absent: a "human" evaluator with no access to reviews
        would silently produce nothing while claiming human oversight.
        """
        if reviews_provider is None:
            raise ValueError(
                "HumanAnswerEvaluator requires a reviews_provider callable; "
                "without access to stored reviews there is nothing human to evaluate"
            )
        self._reviews_provider = reviews_provider
        self._base = base or DeterministicAnswerEvaluator()
        self.detail = "human review verdicts"

    def evaluate(
        self,
        question: AnswerBenchmarkQuestion,
        answer: Answer,
        evidence: list[Evidence],
    ) -> AnswerQualityResult:
        result = self._base.evaluate(question, answer, evidence)
        reviews = list(
            self._reviews_provider(question.question_id, answer.answer_id) or []
        )
        # A provider that ignores answer ids must not get reviews for a
        # DIFFERENT answer attached to this one.
        reviews = [r for r in reviews if not r.answer_id or r.answer_id == answer.answer_id]
        apply_reviews(result, reviews, evaluator_name=self.name)
        result.evaluator_version = self.version
        return result


class LLMAnswerEvaluator(AnswerEvaluator):
    """Optional LLM-as-judge for correctness, behind the same interface.

    Three rules, all testable:

    * **Never silent.** Constructing it without a provider raises, so a run can
      never attribute model judgements to a heuristic by accident.
    * **Never crashes the run.** If the provider fails at judge time, or its
      output cannot be parsed as structured JSON, correctness stays UNKNOWN
      with an explanatory warning. A failed judge is not a verdict.
    * **Never ground truth.** When it does produce a score, the score carries
      "LLM-as-judge" + model + prompt version in its reason, the raw output is
      stored verbatim in `judge_raw`, and a warning marks it model-based.
    """

    name = "llm-judge"
    version = EVALUATOR_VERSION
    is_model_based = True
    PROMPT_VERSION = "aej-v1"

    def __init__(
        self,
        provider,
        model: str = "",
        prompt_version: str = PROMPT_VERSION,
        base: DeterministicAnswerEvaluator | None = None,
    ) -> None:
        if provider is None:
            raise ValueError(
                "LLMAnswerEvaluator requires a provider; use "
                "DeterministicAnswerEvaluator for the offline default path"
            )
        self._provider = provider
        self._model = model or getattr(provider, "model", "unknown-model")
        self._prompt_version = prompt_version
        self._base = base or DeterministicAnswerEvaluator()
        self.detail = f"{self._model} (prompt {self._prompt_version})"

    @staticmethod
    def _build_prompt(
        question: AnswerBenchmarkQuestion,
        answer: Answer,
        evidence: list[Evidence],
    ) -> str:
        ref = question.expected_answer or "(none provided)"
        points = "; ".join(question.key_points) or "(none provided)"
        chunks = "\n".join(
            f"- [{e.chunk_id}] {(e.content or '')[:600]}" for e in evidence[:6]
        ) or "(no evidence)"
        return (
            "You are evaluating one RAG answer. Judge only how correct the "
            "answer is relative to the reference and evidence. Output ONLY "
            "one JSON object, no prose:\n"
            '{"correctness": <0..1>, "confidence": <0..1>, "reason": "<short>"}\n\n'
            f"QUESTION: {question.question}\n"
            f"REFERENCE ANSWER: {ref}\n"
            f"KEY POINTS: {points}\n"
            f"EVIDENCE:\n{chunks}\n"
            f"ANSWER: {answer.text}"
        )

    @staticmethod
    def _parse(raw: str) -> tuple[float, float, str] | None:
        """Extract {correctness, confidence, reason} or None — never guess."""
        text = (raw or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "correctness" not in data:
            return None
        try:
            correctness = float(data["correctness"])
        except (TypeError, ValueError):
            return None
        if not 0.0 <= correctness <= 1.0:
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return correctness, max(0.0, min(1.0, confidence)), str(data.get("reason", ""))

    def evaluate(
        self,
        question: AnswerBenchmarkQuestion,
        answer: Answer,
        evidence: list[Evidence],
    ) -> AnswerQualityResult:
        result = self._base.evaluate(question, answer, evidence)
        result.evaluator_name = self.name
        result.evaluator_version = self.version
        result.evaluator_is_model_based = True
        result.evaluator_detail = self.detail

        prompt = self._build_prompt(question, answer, evidence)
        try:
            raw = self._provider.generate_text(prompt)
        except Exception as exc:  # provider failure must degrade, not crash
            result.warnings.append(
                f"LLM judge UNAVAILABLE ({self.detail}): {exc}; correctness left "
                f"as the deterministic evaluator reported it (UNKNOWN). A failed "
                f"judge is never recorded as a verdict."
            )
            return result

        result.judge_raw = (raw or "")[:4000]
        parsed = self._parse(raw)
        if parsed is None:
            result.warnings.append(
                f"LLM judge returned unparseable output ({self.detail}); "
                f"correctness NOT recorded — no verdict was guessed"
            )
            return result

        correctness_value, confidence, reason = parsed
        result.correctness = Measured.of(
            correctness_value,
            reason=(
                f"LLM-as-judge ({self.detail}): {reason or 'no reason given'}; "
                f"confidence={confidence:.2f}. MODEL-BASED JUDGEMENT — stored "
                f"alongside its raw output, never ground truth."
            ),
        )
        result.warnings.append(
            "correctness here is an LLM-as-judge judgement (model-based, "
            f"{self.detail}), not ground truth; human review or a validated "
            "benchmark answer outranks it"
        )
        return result


class ReferenceAnswerEvaluator(DeterministicAnswerEvaluator):
    """Measures `correctness` from human-reviewed reference labels (V10 STEP 9).

    V8 deliberately kept `correctness` UNKNOWN even when a reference answer
    existed, because lexical similarity to a reference is not a correctness
    judgement. That refusal stays in force for the DEFAULT evaluator, and the
    V8 tests that pin it are unchanged.

    This evaluator is the explicit instrument that unlocks correctness once
    the benchmark has been through the V10 authoring + review + freeze
    workflow, under a published policy:

        reference-key-point-coverage-v1

    `correctness` = fraction of the question's HUMAN-REVIEWED key points the
    answer expresses (the same >= 60% content-term coverage rule used for
    `key_point_recall`), or — when a reviewed reference answer exists but no
    key points were authored — the reference-answer term coverage.

    It is a *completeness against human-approved content* measure. It is NOT
    semantic truth and NOT a substitute for answer reviews or a model-based
    judge; every value it produces says so in its `reason`. Questions without
    reviewed labels stay UNKNOWN, so a partially reviewed benchmark cannot
    silently score the unreviewed remainder.
    """

    name = "reference-labels"
    version = "v10.1"

    #: Published policy name. Recorded on every correctness value so a number
    #: can never be read without knowing which rule produced it.
    CORRECTNESS_POLICY = "reference-key-point-coverage-v1"

    def evaluate(
        self,
        question: AnswerBenchmarkQuestion,
        answer: Answer,
        evidence: list[Evidence],
    ) -> AnswerQualityResult:
        result = super().evaluate(question, answer, evidence)
        result.correctness = self._correctness_from_labels(question, answer, result)
        return result

    def _correctness_from_labels(
        self,
        question: AnswerBenchmarkQuestion,
        answer: Answer,
        result: AnswerQualityResult,
    ) -> Measured:
        policy = self.CORRECTNESS_POLICY
        if (
            question.abstention_required
            or question.answerability is Answerability.UNANSWERABLE
        ):
            return Measured.unknown(
                f"{policy}: correctness is not scored for a question whose correct "
                f"behaviour is abstention; abstention_accuracy covers it"
            )
        key_points = [p for p in question.key_points if p.strip()]
        reference = (question.expected_answer or "").strip()
        if not key_points and not reference:
            return Measured.unknown(
                f"{policy}: this question carries no human-reviewed reference "
                f"answer or key points (it has not been through the V10 review "
                f"workflow), so correctness cannot be measured"
            )
        produced = (
            bool(answer.text.strip())
            and answer.status not in self._NON_ANSWER_STATUSES
        )
        if not produced:
            return Measured.of(
                0.0,
                reason=(
                    f"{policy}: human-reviewed ground truth exists for this "
                    f"question but the system produced no answer (status "
                    f"{answer.status.value}); scored 0.0 by policy"
                ),
                sample_size=1,
            )
        if key_points:
            coverage = result.key_point_recall
            if not coverage.measured or coverage.value is None:
                return Measured.unknown(
                    f"{policy}: key-point coverage was not measurable "
                    f"({coverage.reason})"
                )
            return Measured.of(
                coverage.value,
                reason=(
                    f"{policy}: {coverage.value:.3f} of {len(key_points)} "
                    f"human-reviewed key point(s) expressed by the answer "
                    f"(content-term coverage >= {KEY_POINT_COVERAGE_THRESHOLD:.0%}). "
                    f"This is completeness against human-approved content, NOT "
                    f"semantic truth."
                ),
                sample_size=len(key_points),
            )
        similarity = result.reference_answer_similarity
        if not similarity.measured or similarity.value is None:
            return Measured.unknown(
                f"{policy}: reference-answer similarity was not measurable "
                f"({similarity.reason})"
            )
        return Measured.of(
            similarity.value,
            reason=(
                f"{policy}: {similarity.value:.3f} of the human-reviewed "
                f"reference answer's content terms are present (no key points "
                f"were authored for this question). Lexical completeness against "
                f"human-approved content, NOT semantic truth."
            ),
            sample_size=similarity.sample_size,
        )


def create_answer_evaluator(
    kind: str = "deterministic",
    entailment_kind: str = "heuristic",
    entailment_provider=None,
    *,
    weights: TermWeights | None = None,
    reviews_provider=None,
    llm_provider=None,
    model: str = "",
    prompt_version: str = LLMAnswerEvaluator.PROMPT_VERSION,
) -> AnswerEvaluator:
    """Evaluator factory. Every kind must be requested EXPLICITLY — no kind of
    judge is ever substituted silently for another."""
    key = (kind or "deterministic").strip().lower()
    if key in ("deterministic", "heuristic", "offline"):
        from app.services.answer_eval.entailment import create_entailment_evaluator

        return DeterministicAnswerEvaluator(
            entailment=create_entailment_evaluator(entailment_kind, entailment_provider),
            weights=weights,
        )
    if key in ("human", "human-reviews", "reviews"):
        return HumanAnswerEvaluator(
            reviews_provider=reviews_provider,
            base=DeterministicAnswerEvaluator(weights=weights),
        )
    if key in ("llm", "llm-judge", "model"):
        return LLMAnswerEvaluator(
            provider=llm_provider,
            model=model,
            prompt_version=prompt_version,
            base=DeterministicAnswerEvaluator(weights=weights),
        )
    if key in ("reference", "reference-labels", "reviewed-reference"):
        from app.services.answer_eval.entailment import create_entailment_evaluator

        return ReferenceAnswerEvaluator(
            entailment=create_entailment_evaluator(entailment_kind, entailment_provider),
            weights=weights,
        )
    raise ValueError(
        f"Unknown answer evaluator {kind!r}. Supported: 'deterministic', "
        f"'reference' (correctness from a human-reviewed benchmark), "
        f"'human' (requires reviews_provider), 'llm' (requires llm_provider). "
        f"A model-based judge must be added explicitly, never substituted silently."
    )


__all__ = [
    "AnswerEvaluator",
    "DeterministicAnswerEvaluator",
    "EVALUATOR_VERSION",
    "GroundingState",
    "HumanAnswerEvaluator",
    "LLMAnswerEvaluator",
    "ReferenceAnswerEvaluator",
    "apply_reviews",
    "create_answer_evaluator",
]