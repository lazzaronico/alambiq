"""What a run leaves behind: one record, a history of earlier runs, and replays.

Every run writes `run.json` into its directory and a copy under `history/`, so a
run directory that is used for repeated runs keeps the list of what ran when. The
record names the campaign as it was asked for and, for each step, the capability and
version, the request after bindings were resolved, the reuse key, the outcome and the
duration. The location of the store is replaced by a token so that the directory can be
moved.

A replay writes one step's resolved request back out as a campaign of its own, with the
files it reads copied beside it, so that the step can be run again anywhere and compared
with what the record says it produced.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

import alambiq
from alambiq.core.store import Store

SCHEMA = 1


def campaign_data(campaign: Any) -> dict[str, Any]:
    """The campaign as it was asked for, in the form a campaign file takes."""
    steps = []
    for step in campaign.steps:
        row: dict[str, Any] = {"id": step.id, "capability": step.capability}
        if step.preset:
            row["preset"] = step.preset
        if step.params:
            row["params"] = step.params
        if step.bind:
            row["bind"] = {k: f"{b.step}.{b.output}" for k, b in step.bind.items()}
        steps.append(row)
    return {"id": campaign.id, "steps": steps}


def build_record(
    campaign: Any,
    steps: dict[str, Any],
    store: Store,
    started: datetime,
    seconds: float,
) -> dict[str, Any]:
    rows = []
    for step in campaign.steps:
        result = steps.get(step.id)
        if result is None:
            continue
        rows.append(
            {
                "id": step.id,
                "capability": result.capability,
                "version": result.version,
                "status": result.status,
                "reused": result.reused,
                "seconds": round(result.seconds, 3),
                "key": result.key,
                "request": store.portable(result.request),
                "notes": store.portable(result.notes),
                "error": result.error,
            }
        )
    return {
        "schema": SCHEMA,
        "campaign": campaign_data(campaign),
        "base": str(Path(campaign.base).resolve()),
        "started": started.isoformat(timespec="seconds"),
        "seconds": round(seconds, 3),
        "ok": all(r["status"] in ("completed", "reused") for r in rows),
        "alambiq": alambiq.__version__,
        "python": platform.python_version(),
        "system": sys.platform,
        "steps": rows,
    }


def _finite(value: Any) -> Any:
    """The value with non-finite numbers written as text, so a record is always valid JSON.

    A request can legitimately hold a NaN or an infinity (the step that received it may
    well have refused it), and recording the run must never fail because of one.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def write_record(root: Path, record: dict[str, Any]) -> Path:
    """Write `run.json` and a copy in `history/`; return the copy's path."""
    root = Path(root)
    history = root / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = record["started"].replace(":", "").replace("-", "").replace("+0000", "Z")
    stamp = stamp.replace("+00:00", "Z")
    target = history / f"run-{stamp}.json"
    counter = 1
    while target.exists():
        counter += 1
        target = history / f"run-{stamp}-{counter}.json"
    text = json.dumps(_finite(record), indent=2, allow_nan=False, default=str) + "\n"
    target.write_text(text)
    (root / "run.json").write_text(text)
    return target


def read_record(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"{root} holds no run record (run.json)")
    return json.loads(path.read_text())


def history(root: str | Path, campaign: str | None = None) -> list[dict[str, Any]]:
    """Summaries of the runs recorded in a directory, oldest first."""
    rows = []
    for path in sorted((Path(root) / "history").glob("run-*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if campaign is not None and record.get("campaign", {}).get("id") != campaign:
            continue
        steps = record.get("steps", [])
        rows.append(
            {
                "file": path.name,
                "campaign": record.get("campaign", {}).get("id"),
                "started": record.get("started"),
                "seconds": record.get("seconds"),
                "ok": record.get("ok"),
                "steps": len(steps),
                "reused": sum(1 for s in steps if s.get("reused")),
                "computed": sum(1 for s in steps if s.get("status") == "completed"),
                "failed": sum(1 for s in steps if s.get("status") == "failed"),
                "skipped": sum(1 for s in steps if s.get("status") == "skipped"),
                "alambiq": record.get("alambiq"),
            }
        )
    return rows


def _is_json(path: Path) -> bool:
    """Whether a file with no extension (as the store keeps them) holds JSON."""
    try:
        if path.stat().st_size > 50_000_000:
            return False
        json.loads(path.read_text())
        return True
    except (OSError, ValueError):
        return False


def _localise(value: Any, base: Path, inputs: Path, copied: dict[str, str]) -> Any:
    """Copy each existing file the request names into `inputs` and point at the copy."""
    if isinstance(value, str):
        if value and len(value) < 4096:
            candidate = Path(value)
            located = candidate if candidate.is_absolute() else base / candidate
            try:
                if located.is_file():
                    key = str(located.resolve())
                    if key not in copied:
                        inputs.mkdir(parents=True, exist_ok=True)
                        name = located.name
                        if "." not in name and _is_json(located):
                            name += ".json"
                        while (inputs / name).exists():
                            name = f"{len(copied)}-{name}"
                        shutil.copy2(located, inputs / name)
                        copied[key] = f"inputs/{name}"
                    return copied[key]
            except OSError:
                pass
        return value
    if isinstance(value, dict):
        return {k: _localise(v, base, inputs, copied) for k, v in value.items()}
    if isinstance(value, list):
        return [_localise(v, base, inputs, copied) for v in value]
    return value


def replay(root: str | Path, step_id: str, out_dir: str | Path) -> Path:
    """Write one step's request as a campaign that runs it again; return its path.

    The campaign and the files it reads go into `out_dir`. Nothing in it refers back
    to the run directory, so it can be moved and run on its own.
    """
    root = Path(root)
    record = read_record(root)
    match = next((s for s in record["steps"] if s["id"] == step_id), None)
    if match is None:
        known = ", ".join(s["id"] for s in record["steps"])
        raise KeyError(f"The run has no step {step_id!r}; its steps are {known}")
    if match["request"] is None:
        raise ValueError(
            f"Step {step_id!r} ended before its request was built, so there is nothing "
            f"to replay ({match['error'] or 'it was not run'})"
        )
    store = Store(root / "artifacts")
    request = store.local(match["request"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    request = _localise(request, Path(record["base"]), out / "inputs", {})
    campaign = {
        "id": f"replay-{record['campaign']['id']}-{step_id}",
        "steps": [
            {"id": step_id, "capability": match["capability"], "params": request}
        ],
    }
    header = (
        f"# Replays step {step_id!r} ({match['capability']} {match['version']}) of run "
        f"{record['started']} of campaign {record['campaign']['id']!r}.\n"
        f"# Reuse key of the original: {match['key']}\n"
        "# Run with: alambiq run campaign.yaml\n"
    )
    target = out / "campaign.yaml"
    target.write_text(header + yaml.safe_dump(campaign, sort_keys=False))
    return target


def now() -> datetime:
    return datetime.now(UTC)
