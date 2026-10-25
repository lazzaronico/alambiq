"""The Python way in.

```python
from alambiq.api import run, step

result = run("examples/kinetics/campaign.yaml")
result.outputs("fit")["parameters"]
result.report("report.html")

step("assay.fit", measurements="rates.json", model="linear")
```

YAML and Python build the same campaign object and take the same path, so a
result does not depend on which one asked for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alambiq import capabilities
from alambiq.core import registry
from alambiq.core.campaigns import campaign_from_data, load_campaign
from alambiq.core.execution import RunResult, execute
from alambiq.core.loader import build
from alambiq.core.model import Binding, Campaign, Step

__all__ = [
    "Binding",
    "Campaign",
    "RunResult",
    "Step",
    "capability_list",
    "describe",
    "history",
    "load_campaign",
    "replay",
    "run",
    "step",
    "validate",
]


def _ready() -> None:
    capabilities.load()


def run(
    campaign: Campaign | str | Path | dict[str, Any],
    *,
    run_dir: str | Path | None = None,
    fresh: bool = False,
) -> RunResult:
    """Run a campaign given as an object, a path or plain data."""
    _ready()
    if isinstance(campaign, Campaign):
        plan = campaign
    elif isinstance(campaign, dict):
        plan = campaign_from_data(campaign)
    else:
        plan = load_campaign(campaign)
    return execute(plan, run_dir=run_dir, fresh=fresh)


def step(
    capability: str,
    *,
    preset: str | None = None,
    run_dir: str | Path | None = None,
    base: str | Path | None = None,
    **params: Any,
) -> Any:
    """Run one capability and return its result."""
    _ready()
    plan = Campaign(
        steps=(Step(id="step", capability=capability, preset=preset, params=params),),
        id=capability.replace(".", "-"),
        base=Path(base or Path()),
    )
    outcome = execute(plan, run_dir=run_dir)
    record = outcome.steps["step"]
    if record.status == "failed":
        raise RuntimeError(f"{capability} failed: {record.error}")
    return record.result


def validate(campaign: Campaign | str | Path | dict[str, Any]) -> list[str]:
    """Check a campaign without running it; returns the problems found."""
    _ready()
    problems: list[str] = []
    try:
        if isinstance(campaign, Campaign):
            plan = campaign
        elif isinstance(campaign, dict):
            plan = campaign_from_data(campaign)
        else:
            plan = load_campaign(campaign)
    except Exception as error:  # noqa: BLE001 - reported to the caller
        return [str(error)]
    for item in plan.ordered():
        try:
            capability = registry.get(item.capability)
        except KeyError as error:
            problems.append(f"{item.id}: {error}")
            continue
        # The shape of the request, as far as it is known before anything has run: an
        # unknown field, a missing one, or a value of the wrong kind. What a binding will
        # supply is checked when the step runs.
        try:
            data = {**capability.preset(item.preset), **item.params}
            build(
                capability.request_type,
                data,
                f"step {item.id}",
                deferred=frozenset(name.split(".")[0] for name in item.bind),
            )
        except Exception as error:  # noqa: BLE001 - reported to the caller
            problems.append(f"{type(error).__name__}: {error}")
    return problems


def history(run_dir: str | Path) -> list[dict[str, Any]]:
    """The runs recorded in a run directory, oldest first."""
    from alambiq.core import records

    return records.history(run_dir)


def replay(run_dir: str | Path, step: str, out_dir: str | Path) -> Path:
    """Write one step of a recorded run as a campaign that runs it again.

    The campaign and the files it reads are written into `out_dir`; run it with
    `alambiq run`. Returns the campaign file.
    """
    from alambiq.core import records

    return records.replay(run_dir, step, out_dir)


def capability_list() -> list[dict[str, str]]:
    """Every capability, for `alambiq capability list` and the docs index."""
    _ready()
    return [
        {
            "id": c.id,
            "family": c.family,
            "summary": c.summary,
            "environment": c.environment or "",
        }
        for c in registry.registered()
    ]


def describe(capability: str) -> dict[str, Any]:
    """One capability's fields and documentation, read from its own code."""
    import dataclasses

    _ready()
    cap = registry.get(capability)

    def fields(kind: type) -> list[dict[str, str]]:
        return [
            {
                "name": f.name,
                "type": getattr(f.type, "__name__", str(f.type)),
                "default": (
                    "required"
                    if f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
                    else str(
                        f.default
                        if f.default is not dataclasses.MISSING
                        else f.default_factory()  # type: ignore[misc]
                    )
                ),
            }
            for f in dataclasses.fields(kind)
        ]

    return {
        "id": cap.id,
        "version": cap.version,
        "family": cap.family,
        "summary": cap.summary,
        "environment": cap.environment,
        "resources": list(cap.resources),
        "presets": sorted(cap.presets),
        "documentation": cap.doc,
        "request": fields(cap.request_type),
        "result": fields(cap.result_type),
    }
