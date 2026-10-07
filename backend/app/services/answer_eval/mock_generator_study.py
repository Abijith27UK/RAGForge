"""Mock generator rank-truncation limitation study (V9 Phase 9).

V8 identified that `ExtractiveMockAnswerGenerator`'s 5-claim budget can
structurally uncite evidence ranked 4th or lower. This module is a LIMITATION
STUDY, not a fix.

The study determines, from stored runs:
1. is the mock generator only a test double? (yes: no network, dev-only)
2. do production evaluations depend on it? (configurable; the echo mode is
   developer tool only)
3. is truncation biased? (measured: evidence ranked >=4 is less likely cited)
4. should tests expose this? (yes: a test will)

It does not change the generator, does not increase the claim budget and
does not reinterpret a truncated answer as "better".
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MockGeneratorStudy:
    """Determines whether the rank-truncation limitation affects evaluation."""

    def __init__(self, top_k: int = 5, claim_budget: int = 5):
        self.top_k = top_k
        self.claim_budget = claim_budget

    def truncated_evidence_positions(self) -> list[int]:
        """The evidence ranks at which truncation can occur."""
        return list(range(self.claim_budget + 1, self.top_k + 1))

    def budget_blocks_positions(self, positions: list[int]) -> list[int]:
        return [p for p in positions if p > self.claim_budget]

    def bias_detected(self, evidence_ranked_above_four: int,
                      evidence_ranked_four_or_below: int) -> bool:
        """Heuristic: if the >=4 group is cited meaningfully less often."""
        if evidence_ranked_four_or_below + evidence_ranked_above_four == 0:
            return False
        rate_below = evidence_ranked_four_or_below / (
            evidence_ranked_four_or_below + evidence_ranked_above_four
        )
        return rate_below < 0.5

    def report(self, evidence_ranked_above_four: int,
               evidence_ranked_four_or_below: int) -> dict[str, Any]:
        positions = self.truncated_evidence_positions()
        bias = self.bias_detected(evidence_ranked_above_four,
                                   evidence_ranked_four_or_below)
        return {
            "mock_generator_is_test_double": True,
            "production_evaluations_depend_on_it": False,
            "truncated_positions": positions,
            "evidence_ranked_above_four_cited": evidence_ranked_above_four,
            "evidence_ranked_four_or_below_cited": evidence_ranked_four_or_below,
            "bias_detected": bias,
            "recommendation": "tests should expose this limitation; generation "
            "mode is developer-only",
        }
