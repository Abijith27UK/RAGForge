"""Read-only endpoints serving real frozen experiment artifacts (no mutation).

The Experiments page displays v1/v2 source-selection results verbatim from the
benchmark JSON files on disk. These endpoints are read-only and never rewrite
artifacts; the UI must not hardcode any metric it can read from here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/experiments", tags=["experiments"])

# Frozen artifacts live in <repo>/benchmarks (backend/ is the app root's parent).
_BENCHMARKS_DIR = Path(__file__).resolve().parents[3] / "benchmarks"

_EXPERIMENTS: dict[str, Path] = {
    "source-selection-v1": _BENCHMARKS_DIR / "source-selection-experiment-v1-results.json",
    "source-selection-v2": _BENCHMARKS_DIR / "source-selection-experiment-v2-results.json",
}

# Auxiliary analysis artifacts — fetchable, but not listed as standalone experiments.
_ANALYSES: dict[str, Path] = {
    "source-selection-v2-paired": _BENCHMARKS_DIR / "source-selection-experiment-v2-paired-analysis.json",
}


def _load(name: str) -> dict:
    path = _ANALYSES.get(name) or _EXPERIMENTS.get(name)
    if path is None or not path.is_file():
        raise HTTPException(404, f"No results artifact found for experiment {name!r}")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read experiment artifact %s: %s", path, exc)
        raise HTTPException(503, f"Experiment artifact {name!r} exists but could not be read") from exc


@router.get("")
def list_experiments() -> list[dict]:
    """Index of frozen experiments that have readable results artifacts."""
    out = []
    for name in _EXPERIMENTS:
        try:
            data = _load(name)
        except HTTPException:
            continue
        runs = data.get("runs", [])
        out.append({
            "id": name,
            "research_question": data.get("research_question"),
            "random_seed": data.get("random_seed"),
            "run_at": data.get("run_at"),
            "n_runs": len(runs),
            "caveat": data.get("frozen_baseline_caveat"),
        })
    return out


@router.get("/{experiment_id}/results")
def get_experiment_results(experiment_id: str) -> dict:
    """Full frozen results artifact, verbatim."""
    return _load(experiment_id)
