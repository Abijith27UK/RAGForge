"""Stable ID generation."""
import uuid


def new_id(prefix: str) -> str:
    """Generate a prefixed, URL-safe unique id, e.g. kb_a1b2c3d4."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
