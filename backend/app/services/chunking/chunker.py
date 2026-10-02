"""Section-aware chunking.

Design goals:
- Never blindly split by character count; respect headings/section structure.
- Preserve provenance: section path, document title, page (PDF), domain info.
- Configurable target size + overlap so experiments can compare strategies.
- Interface supports future fixed-size / semantic strategies.

All strategies emit the same Chunk model so downstream code is agnostic.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas.models import Chunk, Document, Source
from app.utils.ids import new_id
from app.utils.text import sha256_text

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SIZE = 1200
DEFAULT_OVERLAP = 150
MIN_CHUNK_CHARS = 60  # drop tiny fragments

# Markdown/HTML-ish headings: # Foo, **Foo**, 1.2 Foo (numbered), ALL CAPS lines
_HEADING_MD = re.compile(r"^(#{1,6})\s+(.+)$")
_HEADING_NUMBERED = re.compile(r"^(\d+(\.\d+)*)\s+([A-Z].{2,120})$")
_HEADING_CAPS = re.compile(r"^([A-Z][A-Z0-9 ,\-/&]{4,80})$")

# Page markers emitted by the PDF parser: "\f[PAGE 12]". Stripped from chunk
# text but recorded so every chunk carries its source page for provenance.
_PAGE_MARKER = re.compile(r"^[\f\s]*\[PAGE (\d+)\]\s*$")


@dataclass
class Section:
    title: str
    level: int
    path: str  # e.g. "3. Vehicle Dynamics > 3.2 Suspension"
    page: int | None
    text: str


def split_into_sections(text: str) -> list[Section]:
    """Split flat text into sections using heading heuristics.

    PDF page markers ([PAGE n]) are removed from section text and recorded as
    the page number of every section started after the marker (a section that
    spans pages reports its FIRST page).
    """
    sections: list[Section] = []
    current = Section(title="Introduction", level=0, path="", page=None, text="")
    stack: list[tuple[int, str]] = []  # (level, title)
    current_page: int | None = None

    def flush():
        if current.text.strip():
            sections.append(Section(current.title, current.level, current.path, current.page, current.text))

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            current.text += "\n"
            continue
        m_page = _PAGE_MARKER.match(line)
        if m_page:
            flush()
            current_page = int(m_page.group(1))
            current = Section(title=current.title, level=current.level, path=current.path, page=current_page, text="")
            continue
        m_md = _HEADING_MD.match(line)
        m_num = _HEADING_NUMBERED.match(line)
        m_caps = _HEADING_CAPS.match(line)
        is_heading = False
        level, title = 1, line
        if m_md:
            is_heading, level, title = True, len(m_md.group(1)), m_md.group(2).strip()
        elif m_num and len(line) < 140:
            is_heading, level, title = True, m_num.group(1).count(".") + 1, f"{m_num.group(1)} {m_num.group(3)}"
        elif m_caps and len(line) < 100 and len(line.split()) <= 12:
            is_heading, level, title = True, 1, m_caps.group(1).title()
        if is_heading:
            flush()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = " > ".join(t for _, t in stack)
            current = Section(title=title, level=level, path=path, page=current_page, text="")
        else:
            current.text += line + "\n"
    flush()
    return sections


def _split_long_text(text: str, target: int, overlap: int) -> list[str]:
    """Split a long section into overlapping windows, preferring paragraph
    boundaries; hard-splitting paragraphs that alone exceed the target."""
    if len(text) <= target:
        return [text]

    # Break paragraphs that individually exceed the target.
    paragraphs: list[str] = []
    for para in text.split("\n\n"):
        if len(para) <= target:
            paragraphs.append(para)
            continue
        step = max(1, target - overlap)
        for start in range(0, len(para), step):
            piece = para[start : start + target]
            if piece.strip():
                paragraphs.append(piece)

    parts: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 > target and buf:
            parts.append(buf.strip())
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + "\n\n" + para).strip()
        else:
            buf = (buf + "\n\n" + para).strip()
    if buf.strip():
        parts.append(buf.strip())
    return parts


class Chunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(
        self,
        document: Document,
        text: str,
        source: Source,
        target_size: int = DEFAULT_TARGET_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        domain: str | None = None,
    ) -> list[Chunk]:
        ...


class SectionAwareChunker(Chunker):
    """Baseline: group content by detected sections, then window long sections."""

    name = "section-aware"

    def chunk(
        self,
        document: Document,
        text: str,
        source: Source,
        target_size: int = DEFAULT_TARGET_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        domain: str | None = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        sections = split_into_sections(text)
        if not sections:
            sections = [Section(title=document.title or "Document", level=0, path="", page=None, text=text)]

        for section in sections:
            idx = 0
            for piece in _split_long_text(section.text.strip(), target_size, overlap):
                if len(piece.strip()) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(self._make_chunk(
                    document=document,
                    source=source,
                    chunk_index=idx,
                    text=piece.strip(),
                    section_title=section.title,
                    section_path=section.path,
                    page=section.page,
                    domain=domain,
                ))
                idx += 1
        return chunks

    def _make_chunk(
        self,
        document: Document,
        source: Source,
        chunk_index: int,
        text: str,
        section_title: str,
        section_path: str,
        page: int | None,
        domain: str | None,
    ) -> Chunk:
        return Chunk(
            id=new_id("chk"),
            document_id=document.id,
            kb_id=document.kb_id,
            chunk_index=chunk_index,
            text=text,
            source_id=source.id,
            source_url=source.url,
            source_title=source.title,
            source_type=source.source_type.value,
            publisher=source.publisher,
            document_title=document.title or source.title,
            section=section_title,
            section_path=section_path or None,
            page=page,
            domain=domain,
            subdomain=None,
            trust_score=source.trust_score,
            content_hash=sha256_text(text),
            ingestion_timestamp=document.ingestion_timestamp,
            char_count=len(text),
            chunking_strategy=self.name,
        )


class FixedSizeChunker(Chunker):
    """Comparison baseline: naive fixed-size windows (for experiments)."""

    name = "fixed-size"

    def chunk(
        self,
        document: Document,
        text: str,
        source: Source,
        target_size: int = DEFAULT_TARGET_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        domain: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = max(1, target_size - overlap)
        idx = 0
        for start in range(0, len(text), step):
            piece = text[start : start + target_size].strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(SectionAwareChunker._make_chunk(
                self,
                document=document,
                source=source,
                chunk_index=idx,
                text=piece,
                section_title=None,
                section_path=None,
                page=None,
                domain=domain,
            ))
            idx += 1
        return chunks


class ChunkingStrategyRegistry:
    """V3 Step 5: explicit registry for chunking strategies.

    Existing strategies register here; future ones (e.g. DomainAwareChunker)
    plug in without touching call sites. Names are stable strategy ids.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, type[Chunker]] = {}

    def register(self, cls: type[Chunker]) -> type[Chunker]:
        self._strategies[cls.name] = cls
        return cls

    def get(self, name: str) -> Chunker:
        cls = self._strategies.get(name)
        if cls is None:
            raise ValueError(f"Unknown chunking strategy '{name}'. Available: {self.names()}")
        return cls()

    def names(self) -> list[str]:
        return sorted(self._strategies)


CHUNKING_REGISTRY = ChunkingStrategyRegistry()
CHUNKING_REGISTRY.register(SectionAwareChunker)
CHUNKING_REGISTRY.register(FixedSizeChunker)


def get_chunker(name: str = "section-aware") -> Chunker:
    """Backward-compatible accessor (routes_build calls this)."""
    return CHUNKING_REGISTRY.get(name)
