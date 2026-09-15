"""Content hashing and text cleaning utilities."""
import hashlib
import re
import unicodedata


def sha256_text(text: str) -> str:
    """SHA-256 hash of text, normalized to NFC with collapsed whitespace."""
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_WS = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Lightweight cleanup: normalize unicode, collapse runs of spaces,
    trim trailing whitespace per line, collapse 3+ blank lines to 2."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
