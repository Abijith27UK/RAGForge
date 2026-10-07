"""V9 Phase 9 regression tests for the mock generator rank-truncation study.

The study documents that the extractive mock generator (5-claim budget) can
structurally uncite evidence ranked 4th or lower. It does not change the
generator or increase the claim budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.answer_eval.mock_generator_study import (  # noqa: E402
    MockGeneratorStudy,
)


class TestMockGeneratorStudy:
    def test_truncated_positions_are_r4_and_below(self):
        study = MockGeneratorStudy(top_k=5, claim_budget=5)
        assert study.truncated_evidence_positions() == [], (
            "with budget=5 and top_k=5 there is no truncation gap"
        )

    def test_study_reports_test_double_only(self):
        study = MockGeneratorStudy(top_k=10, claim_budget=5)
        report = study.report(evidence_ranked_above_four=10,
                              evidence_ranked_four_or_below=2)
        assert report["mock_generator_is_test_double"] is True
        assert report["production_evaluations_depend_on_it"] is False
        assert report["bias_detected"] is True

    def test_study_recommendation_does_not_change_generator(self):
        study = MockGeneratorStudy(top_k=10, claim_budget=5)
        report = study.report(evidence_ranked_above_four=10,
                              evidence_ranked_four_or_below=2)
        assert report["recommendation"] == "tests should expose this limitation; generation mode is developer-only"
