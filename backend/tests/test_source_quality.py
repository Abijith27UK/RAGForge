"""Source quality scoring tests: explainability, decisions, signals."""
from __future__ import annotations

import pytest

from app.schemas.models import DomainSpec, Source, SourceDecision, SourceType
from app.services.source_quality.scorer import DEFAULT_WEIGHTS, SourceQualityScorer
from app.utils.ids import new_id


def make_source(url: str, **kwargs) -> Source:
    return Source(id=new_id("src"), kb_id="kb_test", url=url, **kwargs)


def test_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-6
    with pytest.raises(ValueError):
        SourceQualityScorer(weights={"authority": 0.5, "relevance": 0.9})


def test_standards_body_scores_higher_than_random_site():
    scorer = SourceQualityScorer()
    sae = scorer.assess(make_source("https://www.sae.org/standards/j3016"))
    random = scorer.assess(make_source("https://some-random-blog.example.com/post"))
    assert sae.signals.authority > random.signals.authority
    assert sae.score > random.score


def test_gov_and_edu_domains():
    scorer = SourceQualityScorer()
    gov = scorer.assess(make_source("https://www.nhtsa.gov/recalls"))
    edu = scorer.assess(make_source("https://web.mit.edu/course/automotive"))
    assert gov.signals.authority >= 0.85
    assert edu.signals.authority >= 0.8


def test_invalid_url_is_rejected():
    scorer = SourceQualityScorer()
    src = make_source("https://bad.example.com/x")
    src.notes = "INVALID: host not allowed"
    assessment = scorer.assess(src)
    assert assessment.decision == SourceDecision.REJECT
    assert any("invalid" in w.lower() for w in assessment.warnings)


def test_decision_thresholds():
    scorer = SourceQualityScorer()
    good = scorer.assess(make_source("https://www.iso.org/standard/iso-26262.html", title="ISO 26262 standard reference"))
    assert good.decision in (SourceDecision.ACCEPT, SourceDecision.REVIEW)
    assert good.reasons, "Every assessment must include reasons"


def test_assessment_is_explainable():
    scorer = SourceQualityScorer()
    a = scorer.assess(make_source("https://arxiv.org/abs/2401.12345", title="EV battery paper", notes="Published: 2024"))
    # All signals present, weights recorded, reasons non-empty
    assert a.signals.model_dump().keys() >= {"authority", "relevance", "recency", "accessibility", "duplication"}
    assert a.weights == DEFAULT_WEIGHTS
    assert len(a.reasons) >= 3
    assert a.assessed_by


def test_relevance_uses_domain_spec_terms():
    scorer = SourceQualityScorer()
    spec = DomainSpec(
        kb_id="kb_test",
        domain="Automobile Engineering",
        subdomains=["Electric Vehicles", "Batteries", "Vehicle Dynamics"],
        key_concepts=["battery management system", "regenerative braking"],
    )
    relevant = scorer.assess(
        make_source("https://edu.example.com/ev-batteries", title="EV Batteries and Battery Management System lecture notes"),
        spec=spec,
    )
    irrelevant = scorer.assess(
        make_source("https://cooking.example.com/recipes", title="Chocolate cake recipes"),
        spec=spec,
    )
    assert relevant.signals.relevance > irrelevant.signals.relevance


def test_duplication_penalty():
    scorer = SourceQualityScorer()
    src = make_source("https://blog.example.com/post-2")
    seen = {"https://blog.example.com/post-1"}
    a = scorer.assess(src, seen_urls=seen)
    assert a.signals.duplication < 1.0
    assert any("same host" in w.lower() for w in a.warnings)


def test_accessibility_unknown_warns_instead_of_silent_neutral():
    """No probe performed must produce an explicit warning, not silent neutrality."""
    scorer = SourceQualityScorer()
    a = scorer.assess(make_source("https://example.com/never-probed"), url_accessible=None)
    assert a.signals.accessibility == 0.5
    assert any("not verified" in w.lower() for w in a.warnings)


def test_accessibility_verified_uses_http_status():
    scorer = SourceQualityScorer()
    ok = scorer.assess(make_source("https://example.com/ok"), url_accessible=True, http_status=200)
    dead = scorer.assess(make_source("https://example.com/dead"), url_accessible=False, http_status=404)
    assert ok.signals.accessibility == 1.0
    assert any("HTTP 200" in r for r in ok.reasons)
    assert dead.signals.accessibility == 0.1
    assert any("HTTP 404" in w for w in dead.warnings)


def test_recency_from_last_modified_precedes_text_heuristic():
    """A verified Last-Modified date must win over a year found in the title."""
    from datetime import datetime, timedelta, timezone as tz

    scorer = SourceQualityScorer()
    src = make_source("https://example.com/updated", title="Guide 1999")
    recent = datetime.now(tz.utc) - timedelta(days=30)
    a = scorer.assess(src, last_modified=recent)
    assert a.signals.recency == 1.0, "30-day-old content must score max recency"
    assert any("Last-Modified" in r for r in a.reasons)


def test_recency_old_last_modified_warns():
    from datetime import datetime, timedelta, timezone as tz

    scorer = SourceQualityScorer()
    src = make_source("https://example.com/old", title="Guide 1999")
    ancient = datetime.now(tz.utc) - timedelta(days=365 * 15)
    a = scorer.assess(src, last_modified=ancient)
    assert a.signals.recency < 0.5
    assert any("outdated" in w.lower() for w in a.warnings)
