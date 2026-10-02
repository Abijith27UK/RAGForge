"""User upload validation: file-name safety, size, format sniffing, corruption.

Runs BEFORE anything is written to disk or parsed. Every rejection carries a
user-facing message. Nothing is ever "silently accepted then fails later".
"""
from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.services.ingestion.parsers import supported_upload_extensions

logger = logging.getLogger(__name__)

#: Magic-byte signatures per container format.
_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK\x03\x04"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

#: OOXML formats are ZIP containers; the real format lives inside. This catches
#: a .docx that is actually a .pptx, or an arbitrary ZIP renamed to .docx.
_OOXML_REQUIRED_PART = {
    ".docx": "word/document.xml",
    ".docm": "word/document.xml",
    ".pptx": "ppt/presentation.xml",
    ".pptm": "ppt/presentation.xml",
}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9 ._\-()\[\]]+")
MAX_FILE_NAME_LENGTH = 180


@dataclass
class UploadValidation:
    ok: bool
    file_name: str
    size_bytes: int = 0
    extension: str = ""
    detected_format: str = ""
    error: str = ""
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def sanitize_file_name(raw: str | None) -> str:
    """Reduce an arbitrary client-supplied name to a safe display name.

    Strips directory components (including Windows drive/UNC forms and POSIX
    traversal), then removes characters that are unsafe on any platform.
    """
    name = (raw or "").strip()
    # Normalise both separators before dropping the path component.
    name = name.replace("\\", "/")
    name = PurePosixPath(name).name
    name = _SAFE_NAME.sub("_", name).strip(" .")
    if not name:
        return "upload"
    if len(name) > MAX_FILE_NAME_LENGTH:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 10:
            name = stem[: MAX_FILE_NAME_LENGTH - len(ext) - 1] + "." + ext
        else:
            name = name[:MAX_FILE_NAME_LENGTH]
    return name


def detect_container(data: bytes) -> str:
    """Identify the container format from magic bytes."""
    if data.startswith(_PDF_MAGIC):
        return "pdf"
    if data.startswith(_ZIP_MAGIC):
        return "ooxml"
    if data.startswith(_OLE2_MAGIC):
        return "ole2"
    return "plain-text"


def _validate_ooxml(data: bytes, extension: str, errors: list[str]) -> tuple[str, list[str]]:
    """Confirm an OOXML ZIP actually contains the part its extension promises."""
    warnings: list[str] = []
    required = _OOXML_REQUIRED_PART.get(extension)
    if required is None:
        return "", warnings
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            bad = zf.testzip()
            if bad is not None:
                errors.append(f"corrupt ZIP entry '{bad}'")
            names = set(zf.namelist())
            if required not in names:
                others = sorted(n for n in names if n.endswith(".xml"))[:5]
                errors.append(
                    f"ZIP container does not contain '{required}' — this file is not a "
                    f"valid {extension.lstrip('.')} document"
                    + (f" (found: {', '.join(others)})" if others else "")
                )
    except zipfile.BadZipFile:
        errors.append("File claims to be an Office document but is not a readable ZIP container")
    return "", warnings


def validate_upload(
    file_name: str,
    data: bytes,
    max_bytes: int = 100 * 1024 * 1024,
    allowed_extensions: list[str] | None = None,
) -> UploadValidation:
    """Validate one uploaded file. Returns a result; never raises."""
    safe_name = sanitize_file_name(file_name)
    allowed = [e.lower() for e in (allowed_extensions or supported_upload_extensions())]
    warnings: list[str] = []

    if not data:
        return UploadValidation(False, safe_name, 0, error="File is empty (0 bytes).")

    size = len(data)
    if size > max_bytes:
        return UploadValidation(
            False, safe_name, size,
            error=f"File is {size / (1024*1024):.1f} MB, above the {max_bytes // (1024*1024)} MB limit.",
        )

    from pathlib import Path

    extension = Path(safe_name).suffix.lower()
    if not extension:
        return UploadValidation(
            False, safe_name, size,
            error=f"No file extension. Supported: {', '.join(allowed)}",
        )
    if extension not in allowed:
        return UploadValidation(
            False, safe_name, size, extension=extension,
            error=f"Unsupported file type '{extension}'. Supported: {', '.join(allowed)}",
        )

    container = detect_container(data)

    if extension in (".txt", ".md", ".markdown", ".mdown", ".html", ".htm", ".xhtml"):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            if container == "plain-text":
                warnings.append("File is not valid UTF-8; it will be decoded as Latin-1.")
        if extension in (".html", ".htm", ".xhtml"):
            return UploadValidation(True, safe_name, size, extension, "html", warnings=warnings)
        return UploadValidation(True, safe_name, size, extension, "text", warnings=warnings)

    if extension == ".pdf":
        if not data.startswith(_PDF_MAGIC):
            return UploadValidation(
                False, safe_name, size, extension, "pdf",
                error="File has a .pdf extension but is not a PDF (missing %PDF header).",
            )
        return UploadValidation(True, safe_name, size, extension, "pdf", warnings=warnings)

    if extension in (".pptx", ".pptm", ".docx", ".docm"):
        errors: list[str] = []
        _validate_ooxml(data, extension, errors)
        if errors:
            return UploadValidation(
                False, safe_name, size, extension, "ooxml",
                error="; ".join(errors) + ".",
            )
        return UploadValidation(True, safe_name, size, extension, "ooxml", warnings=warnings)

    if extension == ".ppt":
        if container != "ole2":
            return UploadValidation(
                False, safe_name, size, extension, "ole2",
                error="File has a .ppt extension but is not a legacy PowerPoint binary.",
            )
        return UploadValidation(True, safe_name, size, extension, "ole2", warnings=warnings)

    return UploadValidation(True, safe_name, size, extension, container, warnings=warnings)