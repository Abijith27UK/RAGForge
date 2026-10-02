"""Document ingestion: download, validate, parse, clean, hash.

Two entry points share one parser registry (app/services.ingestion.parsers):

* ``ingest_source``  - EXTERNAL knowledge: download a discovered URL (SSRF-safe,
  size-capped, per-document error isolation) and parse it.
* ``ingest_uploaded_bytes`` - USER knowledge: parse a file the user supplied,
  with no network access at all (privacy: private course material never leaves
  the machine unless explicitly configured).

Every parser failure raises IngestionError with a user-facing message; one bad
document must never abort a batch.
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.schemas.models import Document, Source, SourceType
from app.services.ingestion.parsers import (
    PARSER_REGISTRY,
    DocumentParser,
    HtmlParser,
    IngestionError,
    MarkdownParser,
    PdfParser,
    TextParser,
    get_parser,
    supported_upload_extensions,
)
from app.utils.ids import new_id
from app.utils.text import clean_text, sha256_bytes, sha256_text
from app.utils.url_validation import validate_public_http_url

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB cap for downloaded sources
USER_AGENT = "RAGForge/0.1 (+local research tool)"

_DOWNLOADABLE_EXT = (".pdf", ".html", ".htm", ".txt", ".md", ".markdown")

__all__ = [
    "IngestionError",
    "MAX_DOWNLOAD_BYTES",
    "PARSER_REGISTRY",
    "DocumentParser",
    "HtmlParser",
    "MarkdownParser",
    "PdfParser",
    "TextParser",
    "download_document",
    "get_parser",
    "ingest_source",
    "ingest_uploaded_bytes",
    "parsed_text_path",
    "store_raw_bytes",
    "supported_upload_extensions",
]


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
    if ext not in _DOWNLOADABLE_EXT and "pdf" not in content_type:
        ext = ".html" if "html" in content_type else ".txt"

    filename = f"{source.id}{ext}"
    out_path = documents_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    logger.info("Downloaded %s -> %s (%d bytes)", url, out_path.name, len(data))
    return out_path, data


# ---------------------------------------------------------------------------
# Raw + parsed file storage
# ---------------------------------------------------------------------------

def store_raw_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def parsed_text_path(raw_path: Path) -> Path:
    """Sidecar file holding the normalized text extracted from ``raw_path``."""
    return raw_path.with_suffix(raw_path.suffix + ".parsed.txt")


def _write_parsed_text(raw_path: Path, text: str) -> Path:
    text_path = parsed_text_path(raw_path)
    text_path.write_text(text, encoding="utf-8")
    return text_path


def _document_from_parse(
    *,
    kb_id: str,
    source: Source,
    raw_path: Path,
    data: bytes,
    text: str,
    meta: dict,
    file_name: str,
    mime_type: str | None,
    user_provided: bool,
) -> Document:
    return Document(
        id=new_id("doc"),
        kb_id=kb_id,
        source_id=source.id,
        url=source.url,
        title=meta.get("title") or source.title or file_name,
        source_type=source.source_type,
        publisher=source.publisher,
        file_path=str(_write_parsed_text(raw_path, text)),
        content_hash=sha256_bytes(data) or sha256_text(text),
        text_length=len(text),
        page_count=meta.get("page_count"),
        parse_metadata=dict(meta),
        parser=meta.get("format"),
        file_name=file_name,
        file_size=len(data),
        mime_type=mime_type,
        raw_file_path=str(raw_path),
        slide_count=meta.get("slide_count"),
        section_count=meta.get("section_count"),
        user_provided=user_provided,
        status="parsed",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ingest_source(
    source: Source,
    kb_id: str,
    documents_dir: Path,
) -> Document:
    """Download + parse one EXTERNAL source into a Document record.

    Raises IngestionError with a user-facing message on failure.
    """
    file_path, data = download_document(source, documents_dir)
    parser = get_parser(file_path)
    text, meta = parser.parse(file_path, data)
    return _document_from_parse(
        kb_id=kb_id,
        source=source,
        raw_path=file_path,
        data=data,
        text=text,
        meta=meta,
        file_name=file_path.name,
        mime_type=None,
        user_provided=False,
    )


def ingest_uploaded_bytes(
    *,
    kb_id: str,
    source: Source,
    upload_dir: Path,
    file_name: str,
    data: bytes,
    mime_type: str | None = None,
) -> Document:
    """Parse bytes the user supplied. No network access is performed.

    ``file_name`` is used only for the on-disk extension and as a display
    title; callers must have already sanitized it (see upload.validate_upload).
    """
    extension = Path(file_name).suffix.lower()
    raw_path = upload_dir / f"{new_id('raw')}{extension}"
    store_raw_bytes(raw_path, data)
    parser = get_parser(raw_path, mime_type or "")
    try:
        text, meta = parser.parse(raw_path, data)
    except IngestionError as exc:
        # The bytes are on disk; say where, so the document can be retried
        # without the user re-uploading it.
        if exc.raw_path is None:
            exc.raw_path = str(raw_path)
        raise
    return _document_from_parse(
        kb_id=kb_id,
        source=source,
        raw_path=raw_path,
        data=data,
        text=text,
        meta=meta,
        file_name=file_name,
        mime_type=mime_type,
        user_provided=True,
    )


def reparse_document(doc: Document) -> tuple[str, dict]:
    """Re-run the parser over a document's stored original bytes.

    Used by the document library "rebuild" action. Raises IngestionError.
    """
    if not doc.raw_file_path or not Path(doc.raw_file_path).exists():
        raise IngestionError(
            f"Original file for document {doc.id} is no longer on disk; "
            "re-upload the document instead of rebuilding it."
        )
    raw = Path(doc.raw_file_path)
    data = raw.read_bytes()
    parser = get_parser(raw, doc.mime_type or "")
    text, meta = parser.parse(raw, data)
    _write_parsed_text(raw, text)
    return clean_text(text), meta