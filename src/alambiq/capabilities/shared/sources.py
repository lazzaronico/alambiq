"""Reading a supplied source descriptor.

A descriptor carries either `text` or `path`, never both, with optional
provenance beside it. This lived in the protein dossier plugin, where 27
modules reached for it and where it closed an import cycle through the
provider registry.

New capabilities do not use this: a request names a path and the execution
context reads it. It stays for the modules not yet ported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# A supplied source is read whole, so it has to be small enough to hold.
MAX_BYTES = 20_000_000


def read_input(spec: dict) -> str:
    if not isinstance(spec, dict) or ("text" in spec) == ("path" in spec):
        raise ValueError("Each input requires exactly one of text and path")
    if not spec.keys() <= {
        "text",
        "path",
        "sha256",
        "source_url",
        "license",
        "receipt",
    }:
        raise ValueError("Unknown source descriptor field")
    if "receipt" in spec and not isinstance(spec["receipt"], dict):
        raise ValueError("Source receipt must be a record")
    if any(
        not isinstance(value, str) or not value
        for key, value in spec.items()
        if key != "receipt"
    ):
        raise ValueError("Supplied source fields must be non-empty strings")
    if "path" in spec:
        path = Path(spec["path"])
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError("The input must be a regular file of at most 20 MB")
        content = path.read_bytes()
    else:
        content = spec["text"].encode("utf-8")
    if len(content) > MAX_BYTES or (
        "sha256" in spec and hashlib.sha256(content).hexdigest() != spec["sha256"]
    ):
        raise ValueError("The input exceeds 20 MB, or its SHA-256 digest differs from sha256")
    return content.decode("utf-8")


def source_text(spec: dict) -> str:
    """Read a supplied source descriptor that already carries its text.

    The science accepts `{"text": ...}`. A capability resolves any path before
    calling, so nothing here touches the filesystem.
    """
    if not isinstance(spec, dict) or "text" not in spec:
        raise ValueError("Each source requires supplied text")
    text = spec["text"]
    if not isinstance(text, str):
        raise ValueError("Source text must be a string")
    return text


def load_record(value):
    """Read a JSON input given as a path or text descriptor, or inline."""
    return (
        json.loads(read_input(value))
        if isinstance(value, dict) and ("path" in value or "text" in value)
        else value
    )
