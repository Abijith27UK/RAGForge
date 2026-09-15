"""URL validation utilities with basic SSRF protections."""
import ipaddress
import socket
from urllib.parse import urlparse

BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal", "169.254.169.254"}


class InvalidURLError(ValueError):
    pass


def validate_public_http_url(url: str) -> str:
    """Validate a user-supplied URL: must be http(s) with a public DNS host.

    Raises InvalidURLError with a human-readable message otherwise.
    Returns the validated URL string.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError as exc:
        raise InvalidURLError(f"Could not parse URL: {exc}") from exc

    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError("URL must use http:// or https://")
    host = parsed.hostname
    if not host:
        raise InvalidURLError("URL has no hostname")
    if host.lower() in BLOCKED_HOSTNAMES:
        raise InvalidURLError(f"Host '{host}' is not allowed")

    # Reject obviously private IPs (literal addresses or resolved names).
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError as exc:
        raise InvalidURLError(f"Could not resolve host '{host}': {exc}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise InvalidURLError(f"Host '{host}' resolves to a private address")

    return url.strip()
