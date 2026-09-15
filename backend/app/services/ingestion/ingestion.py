"""Document ingestion: download, validate, parse, clean, hash.

Formats supported in MVP: PDF (pypdf), HTML/web pages (BeautifulSoup),
TXT/Markdown. DOCX deliberately deferred. All external fetches go through
SSRF-checked URLs with timeouts, and failures are per-document: one bad
document must not abort the batch.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.schemas.models import Document, Source, SourceType
from app.utils.ids import new_id
from app.utils.text import clean_text, sha256_bytes, sha256_text
from app.utils.url_validation import validate_public_http_url

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB cap
USER_AGENT = "RAGForge/0.1 (+local research tool)"


class IngestionError(RuntimeError):
    """Per-document ingestion failure with a human-readable message."""


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def download_document(source: Source, documents_dir: Path) -> tuple[Path, bytes]:
    """Download the source content to documents_dir. Raises IngestionError."""
    url = validate_public_http_url(source.url)
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise IngestionError(f"HTTP {exc.response.status_code} fetching {url}") from exc
    except httpx.HTTPError as exc:
        raise IngestionError(f"Network error fetching {url}: {exc}") from exc

    data = resp.content
    if len(data) == 0:
        raise IngestionError(f"Empty response from {url}")
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise IngestionError(f"Document at {url} exceeds {MAX_DOWNLOAD_BYTES // (1024*1024)} MB cap")

    content_type = resp.headers.get("content-type", "").lower()
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext not in (".pdf", ".html", ".htm", ".txt", ".md", ".markdown") and "pdf" not in content_type:
        ext = ".html" if "html" in content_type else ".txt"

    filename = f"{source.id}{ext}"
    out_path = documents_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    logger.info("Downloaded %s -> %s (%d bytes)", url, out_path.name, len(data))
    return out_path, data


# ---------------------------------------------------------------------------
# Parsers (one per supported format, selected by extension/content type)
# ---------------------------------------------------------------------------

class DocumentParser(ABC):
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        """Return (text, metadata). Raise IngestionError on failure."""


class TextParser(DocumentParser):
    extensions = (".txt", ".md", ".markdown")

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except Exception as exc:
                raise IngestionError(f"Could not decode {path.name} as text") from exc
        return clean_text(text), {"format": "text"}


class HtmlParser(DocumentParser):
    extensions = (".html", ".htm")

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception as exc:
            raise IngestionError(f"Could not decode {path.name}: {exc}") from exc
        soup = BeautifulSoup(text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        # Use heading structure for readability; get_text keeps natural flow.
        body = soup.body or soup
        lines = [line.strip() for line in body.get_text("\n").split("\n")]
        cleaned = clean_text("\n".join(lines))
        if len(cleaned) < 40:
            raise IngestionError(f"Page {path.name} contained no extractable text")
        meta = {"format": "html"}
        if title:
            meta["title"] = title
        return cleaned, meta


class PdfParser(DocumentParser):
    extensions = (".pdf",)

    def parse(self, path: Path, data: bytes) -> tuple[str, dict]:
        import io

        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as exc:
            raise IngestionError(f"Could not parse PDF {path.name}: {exc}") from exc
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise IngestionError(f"PDF {path.name} is encrypted and cannot be parsed") from exc
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("PDF %s page %d extraction failed: %s", path.name, i + 1, exc)
                page_text = ""
            # Page marker: consumed by the section-aware chunker so every chunk
            # derived from this page carries page=i+1 in its provenance.
            pages.append(f"\n\f[PAGE {i + 1}]\n{page_text}")
        text = clean_text("\n\n".join(pages))
        if not text.replace("[PAGE", "").strip():
            raise IngestionError(f"PDF {path.name} contained no extractable text (scanned/image PDF?)")
        meta = {"format": "pdf", "page_count": len(reader.pages)}
        if reader.metadata and reader.metadata.title:
            meta["title"] = str(reader.metadata.title)
        return text, meta


PARSERS: list[DocumentParser] = [PdfParser(), HtmlParser(), TextParser()]


def get_parser(path: Path, content_type: str = "") -> DocumentParser:
    ext = path.suffix.lower()
    for parser in PARSERS:
        if ext in parser.extensions:
            return parser
    if "pdf" in content_type:
        return PdfParser()
    if "html" in content_type:
        return HtmlParser()
    return TextParser()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ingest_source(
    source: Source,
    kb_id: str,
    documents_dir: Path,
) -> Document:
    """Download + parse one source into a Document record.

    Raises IngestionError with a user-facing message on failure.
    """
    file_path, data = download_document(source, documents_dir)
    parser = get_parser(file_path)
    text, meta = parser.parse(file_path, data)
    doc = Document(
        id=new_id("doc"),
        kb_id=kb_id,
        source_id=source.id,
        url=source.url,
        title=meta.get("title") or source.title,
        source_type=source.source_type,
        publisher=source.publisher,
        file_path=str(file_path),
        content_hash=sha256_bytes(data) or sha256_text(text),
        text_length=len(text),
        page_count=meta.get("page_count"),
        ingestion_timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    # Keep the parsed text alongside for chunking (stored as .txt next to raw file)
    text_path = file_path.with_suffix(file_path.suffix + ".parsed.txt")
    text_path.write_text(text, encoding="utf-8")
    doc.file_path = str(text_path)
    return doc
