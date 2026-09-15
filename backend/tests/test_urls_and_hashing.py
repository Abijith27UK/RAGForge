"""URL validation and document hashing tests."""
from __future__ import annotations

import pytest

from app.utils.text import clean_text, sha256_bytes, sha256_text
from app.utils.url_validation import InvalidURLError, validate_public_http_url


def test_valid_public_url():
    url = validate_public_http_url("https://www.sae.org/standards")
    assert url.startswith("https://")


def test_rejects_non_http_schemes():
    with pytest.raises(InvalidURLError):
        validate_public_http_url("ftp://example.com/file")
    with pytest.raises(InvalidURLError):
        validate_public_http_url("file:///etc/passwd")


def test_rejects_private_and_loopback_hosts():
    with pytest.raises(InvalidURLError):
        validate_public_http_url("http://localhost:8000/api")
    with pytest.raises(InvalidURLError):
        validate_public_http_url("http://127.0.0.1/admin")
    with pytest.raises(InvalidURLError):
        validate_public_http_url("http://192.168.1.10/config")


def test_rejects_unresolvable_host():
    with pytest.raises(InvalidURLError):
        validate_public_http_url("http://this-host-does-not-exist-abcxyz.invalid/")


def test_sha256_text_stable_and_normalizing():
    a = sha256_text("Hello   World")
    b = sha256_text("Hello \n World")
    assert a == b
    assert a != sha256_text("Hello World!")


def test_sha256_bytes_differs():
    assert sha256_bytes(b"a") != sha256_bytes(b"b")


def test_clean_text():
    assert clean_text("  a   b  \n\n\n\n c ") == "a b\n\nc"
