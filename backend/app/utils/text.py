"""Content hashing and text cleaning utilities."""

import hashlib
import re
import unicodedata


_WS = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def sha256_text(text: str) -> str:
    """SHA-256 hash of text, normalized to NFC with collapsed whitespace."""
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_running_headers(text: str) -> str:
    """Strip running headers/footers from PDF text before sectioning.

    Removes common running header/footer patterns such as:
    - Plain ``Page N`` lines or lines starting with ``Page N`` (without the
      form-feed prefix), which are often running footers.
    - Repeated text that appears at the top/bottom of every page.

    The structural ``\f[PAGE n]`` and ``\f[SLIDE n]`` markers emitted by
    PDF/PPTX parsers are preserved, as the chunker uses them to track page/slide
    provenance. The document's own parsed text is left on disk unchanged; only
    the working copy passed to the chunker is modified.
    """
    # Remove lines that start with "Page N" (with optional trailing text) -
    # these are often running footers, not structural section markers
    text = re.sub(r"^[Pp]age\s+\d+\s.*$", "", text, flags=re.MULTILINE)

    # Remove 3+ consecutive identical non-empty lines (common running footers)
    # Be conservative: only strip if they span more than a reasonable chunk boundary
    text = re.sub(r"(^[^\n]{1,80}\n)\1{2,}", "", text, flags=re.MULTILINE)

    return text


def clean_text(text: str) -> str:
    """Lightweight cleanup: normalize unicode, collapse runs of spaces,
    trim trailing whitespace per line, collapse 3+ blank lines to 2."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
