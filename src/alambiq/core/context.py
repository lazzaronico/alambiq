"""What a capability is given while it runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alambiq.core import envs
from alambiq.core.store import Artifact, Store


@dataclass(slots=True)
class StepContext:
    """Somewhere to put files, a way to read inputs, a way to reach an engine."""

    step_id: str
    store: Store
    workdir: Path
    base: Path = field(default_factory=Path)
    notes: list[str] = field(default_factory=list)

    def resolve(self, path: str | Path) -> Path:
        """Find an input file, relative to the campaign that named it."""
        candidate = Path(path)
        if candidate.is_absolute():
            located = candidate
        else:
            located = (self.base / candidate).resolve()
        if not located.is_file():
            raise FileNotFoundError(f"{self.step_id}: no input file at {path}")
        return located

    def read_text(self, path: str | Path) -> str:
        return self.resolve(path).read_text()

    def read_json(self, path: str | Path) -> Any:
        return json.loads(self.read_text(path))

    def keep(self, path: str | Path, media_type: str) -> Artifact:
        """Store a file this step produced."""
        return self.store.put_file(path, media_type)

    def keep_text(self, content: str, media_type: str) -> Artifact:
        return self.store.put_bytes(content.encode(), media_type)

    def keep_json(self, value: Any) -> str:
        """Store a JSON record and return its path.

        Records pass between steps in this form: a binding passes the path, so
        outputs and the step cache hold the path and not the record.
        """
        text = json.dumps(value, allow_nan=False)
        return str(self.keep_text(text, "application/json").path)

    def note(self, message: str) -> None:
        """Record an observation for the report.

        Observations are findings, not failures: strained geometry, sparse
        measurements, a fit at its bounds. They belong in the result where a
        researcher will see them, never in input validation.
        """
        self.notes.append(message)

    def run_in_environment(
        self, environment: str, entry: str, request: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        return envs.run_in_environment(
            environment, entry, request, workdir=self.workdir, **kwargs
        )
