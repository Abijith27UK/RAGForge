"""Document parsers: one class per format, behind a small registry.

Every parser converts a document into NORMALIZED plain text plus metadata, and
raises `IngestionError` (never a silent empty result) when the file cannot be
read. Provenance markers emitted here are consumed by the section-aware chunker
so that every resulting chunk can report a real page / slide / section number:

    [PAGE 12]          -> Chunk.page     (PDF)
    [SLIDE 17]         -> Chunk.slide    (PPTX) + Chunk.slide_title
    "# Heading" lines  -> Chunk.section / section_path (DOCX, Markdown, TXT)

Marker numbers are ALWAYS derived from the file itself. They are never
estimated, interpolated or invented.
"""
from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.utils.text import clean_text

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Parse/ingest failure with a user-facing message.

    ``raw_path`` carries the location the bytes were written to, when one was
    already stored. The corpus batch service uses it so a FAILED document can
    be retried later from disk instead of forcing the user to re-upload.
    """

    def __init__(self, message: str, raw_path: str | None = None) -> None:
        super().__init__(message)
        self.raw_path = raw_path
    """Per-document ingestion failure with a human-readable message."""


def page_marker(page_number: int) -> str:
    return f"[PAGE {page_number}]"


def slide_marker(slide_number: int) -> str:
    return f"[SLIDE {slide_number}]"


def heading_line(text: str, level: int = 1) -> str:
    """Render a heading as Markdown so the section-aware chunker can rebuild it."""
    return f"{'#' * max(1, min(level, 6))} {text.strip()}"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class DocumentParser(ABC):
    """Converts one binary/encoded document into (normalized_text, metadata)."""

    name: str = "base"
    extensions: tuple[str, ...] = ()
    #: MIME types this parser claims when the extension is inconclusive.
    content_types: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        """Return (text, metadata). Raise IngestionError on any failure."""

    def _require_text(self, text: str, path: Path, minimum: int = 40) -> str:
        cleaned = clean_text(text)
        if len(cleaned) < minimum:
            raise IngestionError(
                f"{path.name}: no extractable text found "
                f"({len(cleaned)} chars < {minimum}). Scanned/image-only documents "
                "are not supported — OCR is not implemented."
            )
        return cleaned


# ---------------------------------------------------------------------------
# Text-ish formats
# ---------------------------------------------------------------------------

class TextParser(DocumentParser):
    extensions = (".txt",)
    content_types = ("text/plain",)
    name = "txt"

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - latin-1 decodes anything
            raise IngestionError(f"Could not decode {path.name} as text")
        cleaned = clean_text(text)
        if not cleaned:
            raise IngestionError(f"{path.name} is empty")
        return cleaned, {"format": "txt"}


class MarkdownParser(DocumentParser):
    """Markdown keeps its own heading structure (the chunker consumes it)."""

    extensions = (".md", ".markdown", ".mdown")
    content_types = ("text/markdown",)
    name = "markdown"

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        for encoding in ("utf-8", "latin-1"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover
            raise IngestionError(f"Could not decode {path.name} as markdown")
        cleaned = clean_text(text)
        if not cleaned:
            raise IngestionError(f"{path.name} is empty")
        return cleaned, {"format": "markdown"}


class HtmlParser(DocumentParser):
    extensions = (".html", ".htm", ".xhtml")
    content_types = ("text/html",)
    name = "html"

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - replace never raises
            raise IngestionError(f"Could not decode {path.name}: {exc}") from exc
        try:
            soup = BeautifulSoup(text, "lxml")
        except Exception as exc:
            raise IngestionError(f"Could not parse HTML {path.name}: {exc}") from exc
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        body = soup.body or soup
        lines = [line.strip() for line in body.get_text("\n").split("\n")]
        cleaned = self._require_text("\n".join(lines), path)
        meta: dict = {"format": "html"}
        if title:
            meta["title"] = title
        return cleaned, meta


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class PdfParser(DocumentParser):
    extensions = (".pdf",)
    content_types = ("application/pdf",)
    name = "pdf"

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as exc:
            raise IngestionError(f"Could not parse PDF {path.name}: {exc}") from exc
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise IngestionError(
                    f"PDF {path.name} is password-protected and cannot be parsed"
                ) from exc
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("PDF %s page %d extraction failed: %s", path.name, i + 1, exc)
                page_text = ""
            pages.append(f"\n\f{page_marker(i + 1)}\n{page_text}")
        text = clean_text("\n\n".join(pages))
        if not text.replace("[PAGE", "").strip():
            raise IngestionError(
                f"PDF {path.name} contained no extractable text (scanned/image PDF? "
                "OCR is not implemented)"
            )
        meta: dict = {"format": "pdf", "page_count": len(reader.pages), "unit": "page"}
        if reader.metadata and reader.metadata.title:
            meta["title"] = str(reader.metadata.title)
        return text, meta


# ---------------------------------------------------------------------------
# PowerPoint (PPTX)
# ---------------------------------------------------------------------------

class PptxParser(DocumentParser):
    """Slide-aware PowerPoint parsing.

    Emits `[SLIDE n]` markers followed by the slide title as a Markdown
    heading, so every chunk derived from a slide carries the real slide number
    and slide title. Speaker notes are included (valuable for lecture decks).
    Legacy binary `.ppt` is accepted by the upload validator but rejected here
    with an explicit, honest message rather than silently producing garbage.
    """

    extensions = (".pptx", ".pptm")
    content_types = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    )
    name = "pptx"

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        from pptx import Presentation

        try:
            prs = Presentation(io.BytesIO(data))
        except Exception as exc:
            raise IngestionError(f"Could not parse PowerPoint {path.name}: {exc}") from exc

        blocks: list[str] = []
        slide_titles: list[str | None] = []
        for index, slide in enumerate(prs.slides, start=1):
            title = ""
            try:
                title_shape = slide.shapes.title
                if title_shape is not None and title_shape.has_text_frame:
                    title = title_shape.text_frame.text.strip()
            except Exception:
                title = ""
            # Only the slide's real title is recorded. A deck whose layout has
            # no title placeholder yields None — never an invented "Slide N".
            title = title.split("\n")[0].strip() if title else None
            slide_titles.append(title)

            blocks.append(f"\n\f{slide_marker(index)}\n")
            if title:
                blocks.append(heading_line(title, level=1) + "\n")

            for shape in self._iter_shapes(slide.shapes):
                if getattr(shape, "has_table", False):
                    table = shape.table
                    rows: list[str] = []
                    for row in table.rows:
                        cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                        rows.append("| " + " | ".join(cells) + " |")
                    if rows:
                        blocks.append("\n".join(rows) + "\n")
                    continue
                if not getattr(shape, "has_text_frame", False):
                    continue
                for para in shape.text_frame.paragraphs:
                    line = "".join(r.text for r in para.runs) if para.runs else para.text
                    line = line.strip()
                    if line:
                        # Sub-heading paragraphs inside a slide become level-2
                        # headings so the section path is richer than the title.
                        blocks.append(f"{line}\n")

            try:
                if slide.has_notes_slide:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        blocks.append(f"\nSpeaker notes: {notes}\n")
            except Exception:
                pass  # notes are optional; never fail a slide over them

        text = clean_text("".join(blocks))
        if not text.replace("[SLIDE", "").strip():
            raise IngestionError(f"PowerPoint {path.name} contained no extractable text")

        meta: dict = {
            "format": "pptx",
            "slide_count": len(prs.slides),
            "unit": "slide",
            "slide_titles": slide_titles,
        }
        core_title = ""
        try:
            core_title = (prs.core_properties.title or "").strip()
        except Exception:
            core_title = ""
        if core_title:
            meta["title"] = core_title
        elif slide_titles and slide_titles[0]:
            meta["title"] = slide_titles[0]
        return text, meta

    @staticmethod
    def _iter_shapes(shapes):
        """Yield shapes, descending into groups (recursively)."""
        for shape in shapes:
            if getattr(shape, "shape_type", None) is not None and shape.shape_type == 6:  # GROUP
                try:
                    yield from PptxParser._iter_shapes(shape.shapes)
                    continue
                except Exception:
                    pass
            yield shape


class LegacyPptParser(DocumentParser):
    """Explicit, honest rejection of the binary PowerPoint 97-2003 format."""

    extensions = (".ppt",)
    content_types = ("application/vnd.ms-powerpoint",)
    name = "ppt-legacy"

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        raise IngestionError(
            f"{path.name}: legacy binary PowerPoint (.ppt) is not supported. "
            "Re-save the deck as .pptx and upload it again."
        )


# ---------------------------------------------------------------------------
# Word (DOCX)
# ---------------------------------------------------------------------------

class DocxParser(DocumentParser):
    """Heading-path aware Word parsing.

    Walks the document body in true order (paragraphs AND tables) and renders
    headings as Markdown so the chunker can rebuild the heading/section path
    (e.g. "Ship Stability > Free Surface Effect"). Tables are preserved as
    pipe-rows rather than flattened into prose.
    """

    extensions = (".docx", ".docm")
    content_types = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    name = "docx"

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        import docx
        from docx.document import Document as _DocxDocument
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table, _Cell
        from docx.text.paragraph import Paragraph

        try:
            document = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise IngestionError(f"Could not parse Word document {path.name}: {exc}") from exc

        blocks: list[str] = []
        heading_paths: list[str] = []
        stack: list[tuple[int, str]] = []

        def emit_paragraph(paragraph: Paragraph) -> None:
            text = paragraph.text.strip()
            if not text:
                return
            level = self._heading_level(paragraph)
            if level:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                heading_paths.append(" > ".join(t for _, t in stack))
                blocks.append(heading_line(text, level) + "\n")
            else:
                blocks.append(text + "\n")

        def emit_table(table: Table) -> None:
            rows: list[str] = []
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                if any(cells):
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                blocks.append("\n".join(rows) + "\n")

        def walk(parent) -> None:
            if isinstance(parent, _DocxDocument):
                element = parent.element.body
            elif isinstance(parent, _Cell):
                element = parent._tc
            else:  # pragma: no cover - defensive
                return
            for child in element.iterchildren():
                if isinstance(child, CT_P):
                    emit_paragraph(Paragraph(child, parent))
                elif isinstance(child, CT_Tbl):
                    emit_table(Table(child, parent))

        try:
            walk(document)
        except Exception as exc:
            raise IngestionError(f"Could not read Word body of {path.name}: {exc}") from exc

        text = clean_text("\n".join(blocks))
        if not text:
            raise IngestionError(f"Word document {path.name} contained no extractable text")

        meta: dict = {
            "format": "docx",
            "section_count": len(heading_paths),
            "unit": "section",
        }
        if heading_paths:
            meta["section_paths"] = heading_paths[:200]
        try:
            core_title = (document.core_properties.title or "").strip()
        except Exception:
            core_title = ""
        if core_title:
            meta["title"] = core_title
        elif heading_paths:
            meta["title"] = heading_paths[0].split(" > ")[0]
        return text, meta

    @staticmethod
    def _heading_level(paragraph) -> int | None:
        try:
            style_name = (paragraph.style.name or "") if paragraph.style else ""
        except Exception:
            return None
        if style_name.lower().startswith("heading "):
            try:
                return max(1, min(6, int(style_name.split()[-1])))
            except ValueError:
                return None
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ParserRegistry:
    """Explicit registry of document parsers (mirrors the chunking registry).

    Adding CSV/XLSX/OCR later means registering a class here — no call sites
    change.
    """

    def __init__(self) -> None:
        self._by_extension: dict[str, DocumentParser] = {}
        self._classes: list[type[DocumentParser]] = []

    def register(self, cls: type[DocumentParser]) -> type[DocumentParser]:
        instance = cls()
        self._classes.append(cls)
        for ext in instance.extensions:
            self._by_extension[ext.lower()] = instance
        return cls

    def by_extension(self, ext: str) -> DocumentParser | None:
        return self._by_extension.get(ext.lower())

    def by_content_type(self, content_type: str) -> DocumentParser | None:
        ct = (content_type or "").split(";")[0].strip().lower()
        for parser in self._by_extension.values():
            if ct and ct in parser.content_types:
                return parser
        return None

    def extensions(self) -> list[str]:
        return sorted(self._by_extension)

    def names(self) -> list[str]:
        return sorted({p.name for p in self._by_extension.values()})

    def resolve(self, path: Path, content_type: str = "") -> DocumentParser:
        parser = self.by_extension(path.suffix)
        if parser is not None:
            return parser
        parser = self.by_content_type(content_type)
        if parser is not None:
            return parser
        raise IngestionError(
            f"Unsupported file type '{path.suffix or content_type or 'unknown'}'. "
            f"Supported: {', '.join(self.extensions())}"
        )


PARSER_REGISTRY = ParserRegistry()
PARSER_REGISTRY.register(PdfParser)
PARSER_REGISTRY.register(PptxParser)
PARSER_REGISTRY.register(LegacyPptParser)
PARSER_REGISTRY.register(DocxParser)
PARSER_REGISTRY.register(HtmlParser)
PARSER_REGISTRY.register(MarkdownParser)
PARSER_REGISTRY.register(TextParser)

#: Extensions accepted by the user-upload endpoint.
UPLOAD_EXTENSIONS: tuple[str, ...] = tuple(PARSER_REGISTRY.extensions())


def supported_upload_extensions() -> list[str]:
    return list(UPLOAD_EXTENSIONS)


def get_parser(path: Path, content_type: str = "") -> DocumentParser:
    """Resolve a parser by extension, then by content type, else fail loudly."""
    return PARSER_REGISTRY.resolve(path, content_type)