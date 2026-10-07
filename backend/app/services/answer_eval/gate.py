"""Off-domain gate review (V9 Phase 8).

V8 left the off-domain gate using unweighted thresholds (a known defect:
IDF-weighted relevance detected it downstream, but the gate itself was
unchanged).

This module does **not** rewrite the gate. It first measures the on/off
distribution, false positives and false negatives, and only then recommends a
threshold change — with the change gated behind a reviewer decision, never an
auto-optimize.

The flow:
    1. Build on/off score distributions from stored runs.
    2. Report FP/FN at the current unweighted threshold.
    3. Recommend a threshold only if the data AND the sample size support one.
    4. Never apply it: the Phase 5 recommendation engine is the only authorizer
       and `answer_eval` does not touch production configuration.

## Polarity of the score

The score axis must be declared. This project's measured signal
(`question_answer_relevance`) is an ON-DOMAIN-ness score, so HIGHER means the
answer is more clearly about the question, and this module defaults to that.
Comparing the two distributions without stating the polarity is how a threshold
recommendation gets inverted, so an unknown polarity is refused rather than
assumed.

## Sample size

No threshold is proposed on fewer than
`MIN_SAMPLES_PER_SIDE_FOR_A_RECOMMENDATION` cases per side. The published V8
measurement has exactly ONE known off-domain answer, so today the honest output
is "cannot be measured yet", not a number.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)

#: Minimum samples PER SIDE before a threshold recommendation is offered.
#: A distribution built from one or two negative examples cannot support a
#: threshold, and offering one anyway is how a gate gets tuned to a single
#: anecdote. The published V8 measurement has exactly ONE known off-domain
#: answer, so this floor is the reason no threshold is recommended today.
MIN_SAMPLES_PER_SIDE_FOR_A_RECOMMENDATION = 5

#: The minimum median gap between the two distributions that is treated as
#: separable. A smaller gap means the distributions overlap and no single
#: threshold can separate them.
MIN_MEDIAN_GAP = 0.05


class GateStudyError(RuntimeError):
    """The study could not be computed as written."""


def _median(values: list[float]) -> float | None:
    """True median (mean of the two middle values for even n).

    The first draft took ``sorted(values)[len // 2]``, which is the UPPER of the
    two middle values, not the median — for an even-sized sample that is a
    different number and it biases the recommended threshold upward.
    """
    if not values:
        return None
    return float(statistics.median(values))


#: Which end of the score axis means "on domain".
#:
#: The first draft of this module silently assumed an OFF-domain-ness score
#: (higher = more likely off-domain) and compared ``off_median > on_median``.
#: The signal this project actually measures is the opposite: V8's
#: ``question_answer_relevance`` is an ON-DOMAIN-ness score, where a HIGHER value
#: means the answer is more clearly about the question. Wiring a real relevance
#: score into an off-domain-ness assumption would have inverted the separability
#: test and recommended LOWERING the threshold — the exact wrong direction.
#: The polarity is therefore an explicit, echoed parameter.
HIGHER_MEANS_ON_DOMAIN = "on_domain"
HIGHER_MEANS_OFF_DOMAIN = "off_domain"


def _normalise_polarity(higher_means: str) -> str:
    if higher_means not in (HIGHER_MEANS_ON_DOMAIN, HIGHER_MEANS_OFF_DOMAIN):
        raise GateStudyError(
            f"higher_means must be {HIGHER_MEANS_ON_DOMAIN!r} or "
            f"{HIGHER_MEANS_OFF_DOMAIN!r}; got {higher_means!r}. The polarity "
            f"cannot be assumed because assuming it wrong inverts the advice"
        )
    return higher_means


def on_off_distribution(
    scores: list[float],
    labels: list[str],
    *,
    higher_means: str = HIGHER_MEANS_ON_DOMAIN,
) -> dict[str, Any]:
    """Return the empirical on/off score distributions.

    ``higher_means`` states the polarity of the score, and the separation is
    computed in that direction. The default is ``on_domain`` because that is what
    this project's measured relevance signal is; a caller comparing an
    off-domain-ness signal must say so explicitly.
    """
    higher_means = _normalise_polarity(higher_means)
    on = [s for s, l in zip(scores, labels) if l == "on"]
    off = [s for s, l in zip(scores, labels) if l == "off"]
    on_median = _median(on)
    off_median = _median(off)

    signed_gap: float | None = None
    if on_median is not None and off_median is not None:
        signed_gap = (
            on_median - off_median
            if higher_means == HIGHER_MEANS_ON_DOMAIN
            else off_median - on_median
        )

    return {
        "on": on,
        "off": off,
        "n_on": len(on),
        "n_off": len(off),
        "on_mean": sum(on) / len(on) if on else None,
        "off_mean": sum(off) / len(off) if off else None,
        "on_median": on_median,
        "off_median": off_median,
        "on_min": min(on) if on else None,
        "on_max": max(on) if on else None,
        "off_min": min(off) if off else None,
        "off_max": max(off) if off else None,
        "higher_means": higher_means,
        #: Separation measured in the direction that means "more on-domain".
        #: Positive = the two groups are separable in the correct direction;
        #: negative = they are INVERTED (the off-domain group scores higher).
        "separation_gap": round(signed_gap, 6) if signed_gap is not None else None,
        "inverted": signed_gap is not None and signed_gap < 0,
        "separable": signed_gap is not None and signed_gap > MIN_MEDIAN_GAP,
    }


def false_positives(scores: list[float], labels: list[str], threshold: float) -> list[dict[str, Any]]:
    """Off-domain cases the gate would ACCEPT, on an on-domain-ness axis.

    A false positive is a question with no business being answered from this
    corpus that the gate would answer anyway. On a higher-is-more-on-domain
    score, that is an off-domain case scoring AT OR ABOVE the threshold.
    """
    return [
        {"score": s, "label": l}
        for s, l in zip(scores, labels)
        if l == "off" and s >= threshold
    ]


def false_negatives(scores: list[float], labels: list[str], threshold: float) -> list[dict[str, Any]]:
    """On-domain cases the gate would REFUSE, on an on-domain-ness axis."""
    return [
        {"score": s, "label": l}
        for s, l in zip(scores, labels)
        if l == "on" and s < threshold
    ]


def gate_study(
    *,
    scores: list[float],
    labels: list[str],
    threshold: float,
    higher_means: str = HIGHER_MEANS_ON_DOMAIN,
) -> dict[str, Any]:
    """One measurement pass for the off-domain gate.

    Returns on/off counts, FP/FN at the given threshold, and a recommendation
    only when the data supports one AND the score polarity was declared.
    """
    if len(scores) != len(labels):
        raise GateStudyError("scores and labels must be the same length")

    dist = on_off_distribution(scores, labels, higher_means=higher_means)
    fp = false_positives(scores, labels, threshold)
    fn = false_negatives(scores, labels, threshold)

    recommendation = None
    reason = ""
    floor = MIN_SAMPLES_PER_SIDE_FOR_A_RECOMMENDATION
    gap = dist["separation_gap"]

    if dist["n_off"] >= floor and dist["n_on"] >= floor:
        if dist["inverted"]:
            # The off-domain group scores HIGHER in the on-domain direction. No
            # threshold can fix a signal that ranks the wrong answer first.
            recommendation = {
                "action": "no_threshold_possible",
                "rationale": (
                    f"the score is INVERTED: the off-domain group's median is "
                    f"{abs(gap):.4f} HIGHER on the on-domain axis than the "
                    f"on-domain group's (separation_gap {gap:+.4f}). A threshold "
                    f"cannot separate a signal that ranks the wrong examples "
                    f"first; the method itself must change"
                ),
                "exploratory": True,
                "n_on": dist["n_on"],
                "n_off": dist["n_off"],
            }
            reason = "measured signal is inverted; no threshold is proposed"
        elif dist["separable"]:
            recommendation = {
                "action": "threshold",
                "new_threshold": round((dist["off_median"] + dist["on_median"]) / 2, 4),
                "rationale": (
                    f"the on-domain median exceeds the off-domain median by "
                    f"{gap:+.4f} (> {MIN_MEDIAN_GAP}) on an on-domain axis; the "
                    f"midpoint is a STARTING POINT for a controlled experiment, "
                    f"not a validated threshold"
                ),
                "exploratory": True,
                "n_on": dist["n_on"],
                "n_off": dist["n_off"],
            }
            reason = "measured distributions are separable enough to justify testing a threshold"
        else:
            recommendation = {
                "action": "no_change",
                "rationale": (
                    f"separation gap {gap!r} does not exceed {MIN_MEDIAN_GAP}; the "
                    f"two distributions overlap too much for a single threshold to "
                    f"separate them"
                ),
                "exploratory": True,
                "n_on": dist["n_on"],
                "n_off": dist["n_off"],
            }
            reason = "data does not support a threshold change"
    else:
        reason = (
            f"insufficient samples to recommend a threshold: {dist['n_on']} on-domain "
            f"and {dist['n_off']} off-domain, against a floor of {floor} per side. "
            f"No threshold is proposed, because tuning a gate to fewer "
            f"observations than this fits the sample rather than the problem"
        )

    return {
        "distribution": dist,
        "higher_means": dist["higher_means"],
        "threshold": threshold,
        "false_positives": fp,
        "false_negatives": fn,
        "fp_count": len(fp),
        "fn_count": len(fn),
        "recommendation": recommendation,
        "reason": reason,
        "sample_floor_per_side": floor,
        "exploratory": True,
    }
