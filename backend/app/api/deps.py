"""Shared FastAPI dependencies (singleton wiring)."""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.repositories.sqlite_repo import Repository


@lru_cache
def get_repo() -> Repository:
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    return Repository(str(settings.db_path))
