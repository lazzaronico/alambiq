"""Content-addressed storage for files a step reads or writes.

Keeping files by digest means an unchanged input has an unchanged name, which
is what lets the step cache recognize work it has already done.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Artifact:
    """A stored file: its digest, its media type and where it now lives."""

    digest: str
    media_type: str
    path: Path

    @property
    def id(self) -> str:
        return f"artifact-{self.digest[:16]}"


def _replace(value: Any, old: str, new: str) -> Any:
    """Replace `old` with `new` in every string inside plain data."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, dict):
        return {k: _replace(v, old, new) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace(v, old, new) for v in value]
    return value


def digest_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def digest_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Store:
    """A local store under one directory."""

    # Stands for the store's location wherever a record is written down, so
    # that a run directory can be moved and its saved results still resolve.
    TOKEN = "{store}"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def portable(self, value: Any) -> Any:
        """The value with this store's location replaced by a token."""
        return _replace(value, str(self.root), self.TOKEN)

    def local(self, value: Any) -> Any:
        """The value with the token replaced by this store's location."""
        return _replace(value, self.TOKEN, str(self.root))

    def _target(self, digest: str) -> Path:
        return self.root / "sha256" / digest[:2] / digest

    def _write_once(
        self, digest: str, content: bytes | None, source: Path | None
    ) -> Path:
        target = self._target(digest)
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=f".{digest}.", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(handle, "wb") as stream:
                if content is not None:
                    stream.write(content)
                else:
                    assert source is not None
                    with source.open("rb") as origin:
                        for chunk in iter(lambda: origin.read(1 << 20), b""):
                            stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def put_bytes(self, content: bytes, media_type: str) -> Artifact:
        digest = digest_bytes(content)
        return Artifact(digest, media_type, self._write_once(digest, content, None))

    def put_file(self, path: str | Path, media_type: str) -> Artifact:
        source = Path(path)
        digest = digest_file(source)
        return Artifact(digest, media_type, self._write_once(digest, None, source))

    def read(self, artifact: Artifact | str) -> bytes:
        digest = artifact if isinstance(artifact, str) else artifact.digest
        target = self._target(digest)
        if not target.is_file():
            raise FileNotFoundError(f"The store has no artifact {digest}")
        return target.read_bytes()
