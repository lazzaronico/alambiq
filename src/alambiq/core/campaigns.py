"""Reading a campaign from YAML.

```yaml
id: kinetics
steps:
  - id: fit
    capability: assay.fit
    params:
      measurements: rates.json
      model: michaelis_menten
  - id: operate
    capability: operation.simulate
    bind:
      rate_parameters: fit.parameters
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from alambiq.core.loader import RequestError
from alambiq.core.model import Binding, Campaign, Step

STEP_KEYS = {"id", "capability", "preset", "params", "bind"}


def campaign_from_data(data: Any, *, base: Path | None = None) -> Campaign:
    if not isinstance(data, dict):
        raise RequestError(
            f"campaign: found {type(data).__name__}; expected a mapping with id and steps"
        )
    unknown = sorted(set(data) - {"id", "steps", "base"})
    if unknown:
        raise RequestError(
            f"campaign: unknown field{'s' if len(unknown) > 1 else ''} "
            f"{', '.join(unknown)}; a campaign accepts id, steps and base"
        )
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RequestError("campaign.steps: found no steps; at least 1 step is required")

    steps: list[Step] = []
    for index, raw in enumerate(raw_steps):
        where = f"campaign.steps[{index}]"
        if not isinstance(raw, dict):
            raise RequestError(f"{where}: found {type(raw).__name__}; expected a mapping")
        extra = sorted(set(raw) - STEP_KEYS)
        if extra:
            raise RequestError(
                f"{where}: unknown field{'s' if len(extra) > 1 else ''} "
                f"{', '.join(extra)}; a step accepts {', '.join(sorted(STEP_KEYS))}"
            )
        for required in ("id", "capability"):
            if not isinstance(raw.get(required), str) or not raw[required]:
                raise RequestError(
                    f"{where}.{required}: found no name; a non-empty text is required"
                )
        bind_raw = raw.get("bind") or {}
        if not isinstance(bind_raw, dict):
            raise RequestError(
                f"{where}.bind: found {type(bind_raw).__name__}; "
                "expected a mapping of request field to 'step.output'"
            )
        bind: dict[str, Binding] = {}
        for name, target in bind_raw.items():
            if not isinstance(target, str):
                raise RequestError(
                    f"{where}.bind.{name}: found {type(target).__name__}; "
                    "expected text of the form 'step.output'"
                )
            try:
                bind[name] = Binding.parse(target)
            except ValueError as error:
                raise RequestError(f"{where}.bind.{name}: {error}") from None
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise RequestError(
                f"{where}.params: found {type(params).__name__}; expected a mapping"
            )
        steps.append(
            Step(
                id=raw["id"],
                capability=raw["capability"],
                preset=raw.get("preset"),
                params=params,
                bind=bind,
            )
        )

    try:
        return Campaign(
            steps=tuple(steps),
            id=str(data.get("id", "campaign")),
            base=Path(base or data.get("base") or Path()),
        )
    except ValueError as error:
        raise RequestError(f"campaign: {error}") from None


def load_campaign(path: str | Path) -> Campaign:
    """Read a campaign file; relative inputs resolve against its directory."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"No campaign file at {source}")
    return campaign_from_data(yaml.safe_load(source.read_text()), base=source.parent)
