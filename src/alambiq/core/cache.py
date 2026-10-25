"""Reuse of stored step results.

Running a campaign again with a step added computes that step and reuses the
others. The key of a step covers every input that can change its result: the
capability and its version, the request after bindings are resolved (each input
file named in the request is represented by its contents), and, for a step that
runs an engine or reads a model, the engine environment and the pinned
resources.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alambiq.core.loader import as_data
from alambiq.core.store import Store, digest_file


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def step_key(
    capability_id: str,
    version: str,
    request: Any,
    engine: dict[str, Any] | None = None,
) -> str:
    """A step's identity: its capability, its version, its request and its engine.

    The request is the one `request_identity` describes, so the key follows
    what the inputs contain and not where they happen to live. `engine` is what
    `engine_identity` reports, and is part of the key only when there is one.
    """
    parts: dict[str, Any] = {
        "capability": capability_id,
        "version": version,
        "request": request,
    }
    if engine:
        parts["engine"] = engine
    return hashlib.sha256(_canonical(parts).encode()).hexdigest()


def engine_identity(capability: Any, request: Any = None) -> dict[str, Any]:
    """The engine environment and pinned resources a capability depends on.

    Empty for a capability that runs in process and reads no model, so its key
    is unaffected. Raises `EnvironmentMissing` when its environment is not
    configured: a result cannot be shown to still apply without knowing what
    would produce it.
    """
    from alambiq.core import envs, resources

    found: dict[str, Any] = {}
    environment = capability.environment_used(request)
    if environment:
        found["environment"] = envs.spec(environment).identity()
    used = (
        capability.resources_used(request)
        if request is not None
        else capability.resources
    )
    if used:
        found["resources"] = {name: resources.identity(name) for name in used}
    return found


def request_identity(request: Any, base: Path, store: Store) -> Any:
    """The request as the reuse key sees it, independent of where things live.

    Anything under the store becomes a token, so moving a run directory changes
    nothing. Every string that names an existing file is replaced by the file's
    name and digest, so two campaigns that point at identical content in
    different places share a result, and editing the content invalidates it.
    """

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            if value and len(value) < 4096:
                candidate = Path(store.local(value))
                located = candidate if candidate.is_absolute() else base / candidate
                try:
                    if located.is_file():
                        return {"file": located.name, "sha256": digest_file(located)}
                except OSError:
                    pass
            return value
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(v) for v in value]
        return value

    return walk(store.portable(as_data(request)))


@dataclass(slots=True)
class Cache:
    """Step results on disk, by key."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            # A damaged entry is a cache miss, never an execution failure.
            return None

    def put(self, key: str, record: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=str))
