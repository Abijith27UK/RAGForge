"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.schemas.models import KnowledgeBase, Source, SourceType  # noqa: E402
from app.utils.ids import new_id  # noqa: E402


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository(str(tmp_path / "test.db"))


@pytest.fixture
def kb() -> KnowledgeBase:
    return KnowledgeBase(
        id=new_id("kb"),
        name="Test KB",
        domain="Automobile Engineering",
        purpose="Engineering education",
        target_audience="Engineering students",
        depth="technical",
    )


@pytest.fixture
def source(kb) -> Source:
    return Source(
        id=new_id("src"),
        kb_id=kb.id,
        url="https://www.sae.org/standards/some-standard",
        title="SAE Standard Example",
        source_type=SourceType.WEB_PAGE,
        publisher="SAE International",
    )
