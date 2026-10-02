"""Run the corpus scale benchmark and write a machine-readable artifact.

    cd backend && .venv/Scripts/python.exe scripts/corpus_benchmark.py

Writes ``backend/data/benchmarks/corpus-scale.json``. Deliberately NOT under
the repository's ``benchmarks/`` directory, whose frozen Automobile
Engineering artifacts must never be modified.

Options:
    --scales 10,50,100,200   Comma-separated document counts.
    --out PATH              Override the artifact path.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "tests"))

from app.services.corpus.perf import SCALE_POINTS, run_all  # noqa: E402

DEFAULT_OUT = BACKEND_DIR / "data" / "benchmarks" / "corpus-scale.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGForge corpus scale benchmark")
    parser.add_argument("--scales", default=",".join(str(s) for s in SCALE_POINTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    scales = tuple(int(s) for s in args.scales.split(",") if s.strip())
    results = run_all(scales)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scales": [r.model_dump(mode="json") for r in results],
        "methodology": {
            "fixtures": "deterministic synthetic Markdown (app/services/corpus/perf.py)",
            "embedding": "HashingEmbeddingProvider (deterministic, local)",
            "vector_store": "InMemoryVectorStore (hermetic cosine fake)",
            "measures": "RAGForge pipeline orchestration only",
            "does_not_measure": [
                "real transformer inference speed",
                "real Qdrant network throughput",
                "retrieval quality of any kind",
            ],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'scale':>7} {'docs':>5} {'chunks':>7} {'docs/min':>9} "
          f"{'chunks/s':>9} {'vectors/s':>10} {'total s':>8}")
    print("-" * 62)
    for r in results:
        t = r.throughput
        print(f"{r.documents:>7} {r.documents:>5} {r.chunks:>7} "
              f"{t['documents_per_minute']:>9.1f} {t['chunks_per_second']:>9.1f} "
              f"{t['vectors_per_second']:>10.1f} {r.timings.total:>8.2f}")
    print(f"\nArtifact written to: {out}")
    print("NOTE: hashing embeddings + in-memory store. This measures RAGForge's own")
    print("      orchestration, NOT transformer inference or Qdrant throughput.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())