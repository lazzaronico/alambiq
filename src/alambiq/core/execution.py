"""Running a campaign.

Prefect owns execution. One flow is defined here, at module level, rather than
rebuilt on every call, and steps that do not depend on each other may run at
the same time.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.futures import wait
from prefect.task_runners import ThreadPoolTaskRunner

from alambiq._native_process import native_process_scope
from alambiq.core import records, registry
from alambiq.core.cache import Cache, engine_identity, request_identity, step_key
from alambiq.core.context import StepContext
from alambiq.core.loader import as_data, build
from alambiq.core.model import Campaign, Step
from alambiq.core.store import Store


@dataclass(slots=True)
class StepResult:
    """What one step produced."""

    step_id: str
    capability: str
    status: str
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    artifacts: list[dict[str, str]] = field(default_factory=list)
    seconds: float = 0.0
    reused: bool = False
    error: str | None = None
    result: Any = None
    version: str = ""
    key: str = ""
    request: dict[str, Any] | None = None
    """The request after bindings were resolved, as the capability received it."""


@dataclass(slots=True)
class RunResult:
    """What a campaign produced."""

    campaign: str
    steps: dict[str, StepResult]
    run_dir: Path
    record: dict[str, Any] = field(default_factory=dict)
    """What was written to `run.json`."""

    @property
    def ok(self) -> bool:
        return all(s.status in ("completed", "reused") for s in self.steps.values())

    def outputs(self, step_id: str) -> dict[str, Any]:
        return self.steps[step_id].outputs

    def report(self, path: str | Path) -> Path:
        from alambiq.report.page import write_report

        return write_report(self, path)


def _assign(data: dict[str, Any], target: str, value: Any) -> None:
    """Set `target` in the request data; `a.2.b` reaches into lists and mappings.

    A bound value can fill one field of one item, so a request that takes a list
    of records can be assembled from several steps' outputs. Missing containers
    are created; a list index may be at most one past the end.
    """
    parts = target.split(".")
    cursor: Any = data
    for part, following in zip(parts, parts[1:], strict=False):
        fresh = [] if following.isdigit() else {}
        if isinstance(cursor, list):
            if not part.isdigit() or int(part) > len(cursor):
                raise RuntimeError(f"Cannot bind into {target!r}: the list has no item {part}")
            if int(part) == len(cursor):
                cursor.append(fresh)
            cursor = cursor[int(part)]
        else:
            cursor = cursor.setdefault(part, fresh)
        if not isinstance(cursor, list | dict):
            raise RuntimeError(
                f"Cannot bind into {target!r}: {part} is neither a mapping nor a list"
            )
    last = parts[-1]
    if isinstance(cursor, list):
        if not last.isdigit() or int(last) > len(cursor):
            raise RuntimeError(f"Cannot bind into {target!r}: the list has no item {last}")
        if int(last) == len(cursor):
            cursor.append(value)
        else:
            cursor[int(last)] = value
    else:
        cursor[last] = value


class UpstreamIncomplete(RuntimeError):
    """A step that this one binds to did not finish, so this one was not run."""

    def __init__(self, step: str, upstream: str, status: str | None) -> None:
        self.step, self.upstream, self.state = step, upstream, status
        super().__init__(
            f"{step} was not run because {upstream} "
            + ("failed" if status == "failed" else "was not run")
        )


def _resolve_bindings(step: Step, done: dict[str, StepResult]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, binding in step.bind.items():
        upstream = done.get(binding.step)
        if upstream is None or upstream.status not in ("completed", "reused"):
            raise UpstreamIncomplete(
                step.id, binding.step, upstream.status if upstream else None
            )
        cursor: Any = upstream.outputs
        reached = binding.step
        for part in binding.output.split("."):
            if isinstance(cursor, list) and part.isdigit() and int(part) < len(cursor):
                cursor = cursor[int(part)]
            elif isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                if isinstance(cursor, dict) and cursor:
                    available = ", ".join(sorted(cursor))
                elif isinstance(cursor, list):
                    available = (
                        f"items 0 to {len(cursor) - 1}" if cursor else "no items"
                    )
                else:
                    available = "none"
                raise RuntimeError(
                    f"Step {step.id} binds {name} to {binding.step}.{binding.output}, "
                    f"but {reached} has no {part!r}. Available at {reached}: {available}"
                )
            reached += f".{part}"
        values[name] = cursor
    return values


def run_step(
    step: Step,
    campaign: Campaign,
    done: dict[str, StepResult],
    store: Store,
    cache: Cache | None,
    workdir: Path,
) -> StepResult:
    """Build the request, reuse or run, and record what happened."""
    cap = registry.get(step.capability)
    # Building the request and resolving bindings can fail on bad input. That
    # is this step's failure, recorded like any other, so one mistyped field
    # does not abort the rest of the campaign.
    try:
        data = copy.deepcopy({**cap.preset(step.preset), **step.params})
        for name, value in _resolve_bindings(step, done).items():
            _assign(data, name, value)
        request = build(cap.request_type, data, f"step {step.id}")
        identity = request_identity(request, campaign.base, store)
        engine = engine_identity(cap, request)
    except UpstreamIncomplete as error:
        # The cause is the upstream step's own error; saying it again for every step
        # that depends on it only buries it.
        return StepResult(
            step_id=step.id,
            capability=cap.id,
            status="skipped",
            notes=[str(error)],
        )
    except Exception as error:  # noqa: BLE001 - reported as a step failure
        return StepResult(
            step_id=step.id,
            capability=cap.id,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )

    key = step_key(cap.id, cap.version, identity, engine)
    described = {"version": cap.version, "key": key, "request": as_data(request)}

    if cache is not None:
        found = cache.get(key)
        if found is not None:
            reused = _reuse(cap, step, found, store)
            if reused is not None:
                reused.version, reused.key = cap.version, key
                reused.request = described["request"]
                return reused

    started = time.monotonic()
    # A step's working directory is scratch: whatever it kept is in the store. A
    # leftover from an earlier attempt would otherwise collide with this one.
    place = workdir / step.id
    shutil.rmtree(place, ignore_errors=True)
    place.mkdir(parents=True)
    ctx = StepContext(step_id=step.id, store=store, workdir=place, base=campaign.base)
    try:
        result = cap.run(request, ctx)
    except Exception as error:  # noqa: BLE001 - reported as a step failure
        return StepResult(
            step_id=step.id,
            capability=cap.id,
            status="failed",
            notes=ctx.notes,
            seconds=time.monotonic() - started,
            error=f"{type(error).__name__}: {error}",
            **described,
        )

    outputs = as_data(result)
    record = StepResult(
        step_id=step.id,
        capability=cap.id,
        status="completed",
        outputs=outputs,
        notes=ctx.notes,
        seconds=time.monotonic() - started,
        result=result,
        **described,
    )
    if cache is not None:
        cache.put(
            key,
            {
                "capability": cap.id,
                "version": cap.version,
                "step": step.id,
                "outputs": store.portable(outputs),
                "notes": store.portable(ctx.notes),
                "artifacts": store.portable(record.artifacts),
            },
        )
    return record


def _reuse(
    cap: Any, step: Step, found: dict[str, Any], store: Store
) -> StepResult | None:
    """Rebuild a step's result from the cache, or None if it no longer fits.

    Records name the store by a token, so they stay valid when a run directory
    is moved. The result object is rebuilt from its saved outputs so that a
    reused step is presented exactly like one that just ran. A record that no
    longer fits the result type, because the capability changed, is a miss and
    the step runs again.
    """
    outputs = store.local(found.get("outputs", {}))
    try:
        result = build(cap.result_type, outputs, f"saved result of step {step.id}")
    except Exception:  # noqa: BLE001 - a stale record is a cache miss
        return None
    return StepResult(
        step_id=step.id,
        capability=cap.id,
        status="reused",
        outputs=outputs,
        notes=store.local(found.get("notes", [])),
        artifacts=store.local(found.get("artifacts", [])),
        reused=True,
        result=result,
    )


@task(name="alambiq-step")
def _step_task(
    step: Step,
    campaign: Campaign,
    done: dict[str, StepResult],
    store: Store,
    cache: Cache | None,
    workdir: Path,
) -> StepResult:
    return run_step(step, campaign, done, store, cache, workdir)


@flow(name="alambiq-campaign", task_runner=ThreadPoolTaskRunner(max_workers=4))
def _campaign_flow(
    campaign: Campaign, store: Store, cache: Cache | None, workdir: Path
) -> dict[str, StepResult]:
    # Inside the flow, so that this scope's signal handlers replace the ones Prefect
    # installs when the flow starts; otherwise SIGTERM would not reach the commands.
    with native_process_scope():
        return _run_steps(campaign, store, cache, workdir)


def _run_steps(
    campaign: Campaign, store: Store, cache: Cache | None, workdir: Path
) -> dict[str, StepResult]:
    done: dict[str, StepResult] = {}
    pending: dict[str, Any] = {}
    for step in campaign.ordered():
        upstream = [pending[b.step] for b in step.bind.values() if b.step in pending]
        if upstream:
            wait(upstream)
            for b in step.bind.values():
                if b.step in pending:
                    done[b.step] = pending[b.step].result()
        pending[step.id] = _step_task.submit(
            step, campaign, dict(done), store, cache, workdir
        )
    for step_id, future in pending.items():
        done[step_id] = future.result()
    return done


def execute(
    campaign: Campaign,
    *,
    run_dir: str | Path | None = None,
    fresh: bool = False,
) -> RunResult:
    """Run a campaign, reusing unchanged steps unless asked not to."""
    root = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix="alambiq-"))
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root / "artifacts")
    cache = None if fresh else Cache(root / "cache")
    workdir = root / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    started, clock = records.now(), time.monotonic()
    steps = _campaign_flow(campaign, store, cache, workdir)
    record = records.build_record(
        campaign, steps, store, started, time.monotonic() - clock
    )
    records.write_record(root, record)
    return RunResult(campaign=campaign.id, steps=steps, run_dir=root, record=record)
