"""v2 scorer instrument check over the FROZEN 36-source candidate pool.

Research requirement: the improved scorer must demonstrate meaningful score
variation among the 36 Automobile Engineering candidate sources before any v2
experiment runs. This script reports:
  - score distribution (histogram over content_relevance and composite)
  - number of unique scores
  - top 10 / bottom 10 sources with reasons
  - coverage per domain requirement area
  - limitations

STOP condition: if content_relevance values remain degenerate (fewer than 15
unique values across 36 sources), the report says so and v2 must be improved
further instead of running an experiment.

Usage:
  python scripts/report_v2_pool_scores.py \
      --experiment ../benchmarks/source-selection-experiment-v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.schemas.models import Source, SourceType
from app.services.embeddings.provider import create_embedding_provider
from app.services.source_quality.content_scorer import (
    COVER_THRESHOLD,
    ContentAwareSourceScorer,
    fetch_source_content,
    load_domain_map,
)
from app.utils.ids import new_id


def make_source(url: str, kb_id: str = "kb_poolcheck") -> Source:
    slug = url.rsplit("/", 1)[-1].replace("_", " ").replace("%28", "(").replace("%29", ")")
    return Source(id=new_id("src"), kb_id=kb_id, url=url, title=slug, source_type=SourceType.WEB_PAGE)


def build_domain_scorer():
    """Embedding provider + v2 scorer for the Automobile Engineering domain map."""
    settings = get_settings()
    embedder = create_embedding_provider(settings)
    dom_map = load_domain_map("Automobile Engineering")
    scorer = ContentAwareSourceScorer(embedder, dom_map)
    return embedder, scorer, {
        "domain": dom_map["domain"],
        "version": dom_map["version"],
        "provenance": dom_map["provenance"],
    }


def load_pool(exp: dict, exp_path: Path | None = None) -> list[str]:
    """Pool URLs from an experiment JSON: inline, or resolved from 'urls_ref'
    ("<file>#/json/pointer") so the pool is never duplicated across files."""
    cp = exp["candidate_pool"]
    if "urls" in cp:
        return list(cp["urls"])
    ref: str = cp["urls_ref"]
    file_part, _, pointer = ref.partition("#")
    path = (exp_path.parent / file_part).resolve() if exp_path else Path(file_part)
    data = json.loads(path.read_text(encoding="utf-8"))
    for part in [p for p in pointer.split("/") if p]:
        data = data[part]
    return list(data)


def score_pool(pool, embedder, scorer, cache_dir=None, seen=None, verbose=True):
    """Score every pool URL: fetch (cached) content, then v2 assess. Never raises."""
    if cache_dir is None:
        cache_dir = get_settings().data_dir / "content_cache"
    if seen is None:
        seen = set(pool)
    records = []
    for url in pool:
        src = make_source(url)
        content, status, err = fetch_source_content(url, cache_dir)
        rec = scorer.assess_with_content(src, content, http_status=status, seen_urls=seen)
        rec["url"] = url
        rec["fetch_error"] = err
        records.append(rec)
        if verbose:
            print(f"  {rec['score']:.4f} (rel={rec['signals']['content_relevance']:.3f} "
                  f"cov={rec['signals']['domain_coverage']:.2f} conf={rec['confidence']:.2f}) {url.split('/wiki/')[-1]}")
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, type=Path, help="frozen v1 experiment JSON (candidate pool)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exp = json.loads(args.experiment.read_text(encoding="utf-8"))
    pool: list[str] = load_pool(exp, args.experiment)
    settings = get_settings()
    embedder = create_embedding_provider(settings)
    dom_map = load_domain_map("Automobile Engineering")
    scorer = ContentAwareSourceScorer(embedder, dom_map)
    cache_dir = settings.data_dir / "content_cache"
    seen = set(pool)

    print(f"v2 pool instrument check: {len(pool)} sources, threshold tau={COVER_THRESHOLD}, "
          f"embedder={embedder.identity().describe()}")
    t0 = time.perf_counter()
    records = score_pool(pool, embedder, scorer, cache_dir, seen=seen)
    elapsed = round(time.perf_counter() - t0, 1)

    # ---- Distribution analysis ------------------------------------------------
    rel_scores = [r["signals"]["content_relevance"] for r in records]
    comp_scores = [r["score"] for r in records]
    n_unique_rel = len(set(rel_scores))
    n_unique_comp = len(set(comp_scores))
    rel_dist = Counter(round(x, 2) for x in rel_scores)

    print("\n--- score distribution (content_relevance, rounded to 2dp) ---")
    for bucket in sorted(rel_dist):
        bar = "#" * rel_dist[bucket]
        print(f"  {bucket:.2f}: {rel_dist[bucket]:2d} {bar}")

    ranked = sorted(records, key=lambda r: -r["score"])

    def show(rows, label):
        print(f"\n--- {label} ---")
        for r in rows:
            covered = [p["area_id"] for p in r["per_requirement"] if p["covered"]]
            print(f"  {r['score']:.4f} rel={r['signals']['content_relevance']:.3f} "
                  f"cov={r['signals']['domain_coverage']:.2f} [{','.join(covered) or '-'}] "
                  f"{r['url'].split('/wiki/')[-1]}")

    show(ranked[:10], "TOP 10")
    show(sorted(records, key=lambda r: r["score"])[:10], "BOTTOM 10")

    # ---- Coverage per requirement area ----------------------------------------
    print("\n--- coverage by requirement area (sources covering the area at tau) ---")
    areas = [a["id"] for a in dom_map["requirement_areas"]]
    for area in areas:
        n_covered = sum(1 for r in records for p in r["per_requirement"] if p["area_id"] == area and p["covered"])
        avg_sim = (
            sum(p["similarity"] for r in records for p in r["per_requirement"] if p["area_id"] == area)
            / len(records)
        )
        bar = "#" * n_covered
        print(f"  {area:16s} {n_covered:2d}/36 {bar}  (avg sim {avg_sim:.3f})")

    degenerate = n_unique_rel < 15
    print("\n--- verdict ---")
    print(f"unique content_relevance values: {n_unique_rel}/36 | unique composite: {n_unique_comp}/36")
    if degenerate:
        print("VERDICT: DEGENERATE - the v2 scorer does NOT yet discriminate the pool. STOP; improve scoring before any v2 experiment.")
    else:
        print("VERDICT: DISCRIMINATING - the scorer separates the pool; v2 experiment may be proposed for approval.")
    print(f"total wall time: {elapsed}s (content cached at {cache_dir})")

    out = args.out or args.experiment.parent / "source-quality-v2-pool-report.json"
    out.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pool": exp["candidate_pool"]["urls"],
                "scorer_version": scorer.version,
                "domain_map_version": dom_map["version"],
                "embedder": embedder.identity().describe(),
                "cover_threshold": COVER_THRESHOLD,
                "unique_content_relevance": n_unique_rel,
                "unique_composite": n_unique_comp,
                "degenerate": degenerate,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
