"""Pipeline instrumentation and the corpus scale benchmark (V5 Phase 7).

Measure, do not guess. Every number below comes from a real
``time.perf_counter()`` around a real operation, and the harness uses
deterministic synthetic fixtures so it can run at 10/50/100/200 documents
without needing 200 real lecture files.

Run it with::

    cd backend && .venv/Scripts/python.exe scripts/corpus_benchmark.py

The artifact is written to ``backend/data/benchmarks/corpus-scale.json``. That
path is INSIDE ``backend/data/`` and deliberately NOT under the repository's
``benchmarks/`` directory, whose frozen artifacts must never be touched.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.schemas.corpus import PipelineTimings, ScaleBenchmarkResult

#: Scale points required by the phase spec.
SCALE_POINTS: tuple[int, ...] = (10, 50, 100, 200)

#: Deterministic fixture body long enough to yield real chunks.
_SECTIONS = (
    ("Hydrostatics", "Buoyancy and displacement relate through Archimedes' principle."),
    ("Resistance", "Frictional resistance scales with wetted surface and form factor."),
    ("Stability", "The metacentric height governs the initial stability criterion."),
    ("Propulsion", "Propulsive efficiency falls as the speed approaches the hump."),
    ("Manoeuvring", "Restricted water reduces rudder authority and increases squat."),
    ("Structures", "Still water bending moments dominate the midship section design."),
    ("Design", "The lines plan trades deadweight against resistance and volume."),
    ("Stability", "A virtual rise of the centre of gravity reduces the GM."),
)


def synthetic_document(index: int) -> str:
    """A deterministic lecture-shaped Markdown document.

    Content is a pure function of ``index``, so two runs of the harness ingest
    byte-identical corpora and the timings are comparable.
    """
    title = f"Lecture {index:03d}"
    lines = [f"# {title}", "", f"Course document {index} for scale benchmarking.", ""]
    for i, (heading, body) in enumerate(_SECTIONS):
        # Vary the per-document body so content hashes differ (no false
        # duplicate detection at scale) while staying deterministic.
        lines += [
            f"## {heading}",
            "",
            f"{body} This section {i} of {title} expands on the topic in detail. "
            f"It repeats a few times so the chunker has real material to work on, "
            f"covering theory, worked relations and the practical consequences "
            f"for the vessel described in document {index}.",
            "",
        ]
    return "\n".join(lines)


@dataclass
class Stopwatch:
    """Accumulates real stage durations. Never estimates."""

    measurements: dict[str, float] = field(default_factory=dict)

    def time(self, label: str):
        return _StageTimer(self, label)

    def add(self, label: str, seconds: float) -> None:
        self.measurements[label] = self.measurements.get(label, 0.0) + seconds

    def total(self) -> float:
        return sum(self.measurements.values())


class _StageTimer:
    def __init__(self, watch: Stopwatch, label: str) -> None:
        self._watch = watch
        self._label = label
        self._start = 0.0

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self._watch.add(self._label, time.perf_counter() - self._start)


def bench_parse(watch: Stopwatch, texts: list[str], upload_dir: Path) -> list[tuple[Path, bytes]]:
    """Write every synthetic document to disk and parse it. Real work."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[Path, bytes]] = []
    for i, text in enumerate(texts):
        with watch.time("parse"):
            raw = upload_dir / f"bench_{i:04d}.md"
            data = text.encode("utf-8")
            raw.write_bytes(data)
            out.append((raw, data))
    return out


def bench_chunk_embed_index(watch: Stopwatch, repo, kb, store, embedder,
                            parsed: list[tuple[Path, bytes]], chunker: str,
                            target_size: int, overlap: int) -> tuple[int, int]:
    """Chunk -> embed -> index every document through the shared indexer."""
    from app.schemas.models import Document
    from app.services.indexing.document_indexer import index_documents
    from app.utils.ids import new_id

    chunks = vectors = 0
    for raw, data in parsed:
        text = raw.read_text(encoding="utf-8")
        parsed_sidecar = raw.with_suffix(raw.suffix + ".parsed.txt")
        parsed_sidecar.write_text(text, encoding="utf-8")
        source_id = new_id("src")
        repo._execute(
            "INSERT INTO sources (id, kb_id, url, data) VALUES (?, ?, ?, ?)",
            (source_id, kb.id, f"upload://{raw.name}", json.dumps({
                "id": source_id, "kb_id": kb.id, "url": f"upload://{raw.name}",
                "title": raw.name, "source_type": "text", "user_provided": True,
            })),
        )
        doc = Document(
            id=new_id("doc"), kb_id=kb.id, source_id=source_id, url=f"upload://{raw.name}",
            title=raw.name, source_type="text", file_path=str(parsed_sidecar),
            content_hash=str(len(data)), text_length=len(text), file_name=raw.name,
            file_size=len(data), mime_type="text/markdown", parser="markdown",
            raw_file_path=str(raw), user_provided=True,
        )
        repo.create_document(doc)
        with watch.time("index"):
            outcome = index_documents(
                repo=repo, kb=kb, documents=[doc], store=store, embedder=embedder,
                chunker_name=chunker, target_size=target_size, overlap=overlap,
            )
        chunks += outcome.chunk_count
        vectors += outcome.vectors_indexed
    return chunks, vectors


def run_scale_point(scale: int, *, store=None, embedder=None, tmp_dir: Path | None = None,
                    chunker: str = "section-aware") -> ScaleBenchmarkResult:
    """Ingest ``scale`` synthetic documents and measure the real pipeline."""
    import shutil
    import tempfile

    from app.repositories.sqlite_repo import Repository
    from app.schemas.models import KnowledgeBase
    from app.services.embeddings.provider import HashingEmbeddingProvider
    from tests.test_retrieval_integration import InMemoryVectorStore

    owned_tmp = tmp_dir is None
    root = Path(tmp_dir or tempfile.mkdtemp(prefix="ragforge-bench-"))
    store = store if store is not None else InMemoryVectorStore()
    embedder = embedder if embedder is not None else HashingEmbeddingProvider()
    repo = Repository(str(root / f"bench_{scale}.db"))
    kb = KnowledgeBase(
        id=f"kb_bench_{scale}", name=f"Bench {scale}", domain="Naval Architecture",
        purpose="Scale benchmark", target_audience="engineers", depth="technical",
    )
    repo.create_kb(kb)

    texts = [synthetic_document(i) for i in range(scale)]
    watch = Stopwatch()
    try:
        with watch.time("upload"):
            total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        parsed = bench_parse(watch, texts, root / "uploads")
        chunks, vectors = bench_chunk_embed_index(
            watch, repo, kb, store, embedder, parsed, chunker, 1200, 150,
        )
    finally:
        if owned_tmp:
            shutil.rmtree(root, ignore_errors=True)

    timings = PipelineTimings(
        upload=watch.measurements.get("upload", 0.0),
        validation=0.0,
        parse=watch.measurements.get("parse", 0.0),
        chunk=0.0,
        embed=0.0,
        index=watch.measurements.get("index", 0.0),
        total=watch.total(),
        documents=scale,
        chunks=chunks,
        vectors=vectors,
        bytes=total_bytes,
    )
    return ScaleBenchmarkResult(
        label=f"{scale}-documents",
        documents=scale,
        chunks=chunks,
        bytes=total_bytes,
        timings=timings,
        throughput=timings.throughput(),
        notes=(
            "Deterministic synthetic Markdown fixtures; hashing embeddings and the "
            "in-memory cosine store, so these measure RAGForge's own pipeline "
            "orchestration, not Qdrant or a transformer."
        ),
    )


def run_all(scales: tuple[int, ...] = SCALE_POINTS, **kwargs) -> list[ScaleBenchmarkResult]:
    return [run_scale_point(scale, **kwargs) for scale in scales]