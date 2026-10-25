"""Compare simulation records with observations, one at a time or as an ensemble.

A simulation record is the stored record of an `operation.simulate` result.
`operation.compare` reads one simulation record and matches each observation to the
recorded state with the same time, the same `state` and the same quantity. The `state`
is `before` (the state before an exchange; requires an exchange at that time) or
`after` (the state at a recorded time, or after the exchange at that time). A quantity
is one of `concentration`, `conversion` (against the reference concentration
`reference_M`), `removed` (amount removed, in mol), `effluent-concentration` (the
flow-weighted mean concentration over a collection window), `catalyst-total` and
`catalyst-active`. An `effluent-concentration` observation uses the state `after`, a
`collection_id` and the end time of that collection window. The target of an
observation is the species or the catalyst to which the quantity refers. Residuals
(prediction minus measurement) are grouped into residual groups by partition, quantity,
target, unit and, for conversions, `reference_M`. The partition is the label
`calibration`, `evaluation` or `reference` copied from the observation file; it is
a grouping key. Each residual group is summarized as RMSE (the root mean
square of its residuals, in the unit of the observations). The result holds the RMSE of
each residual group; the stored record holds the residual of each observation.

`operation.compare_ensemble` reads an explicit set of simulation records, for
example one per row of source parameters, and the same observations. For every
member it computes the residuals as `operation.compare` does. The result holds the RMSE
of the pointwise median over the members and the RMSE of each member; the stored record
also holds the median, the minimum and the maximum over the members at each observation.

Limits. The prediction is the one stored in the
simulation record, so a comparison uses the rate values and conditions of that
simulation. The observation file for `operation.compare` names the simulated definition
(`operation_id`) and the conditions to which it applies; a file that names a different
definition or different conditions is refused. The file for
`operation.compare_ensemble` names the conditions only and contains no `operation_id`,
because each member supplies its own definition. A reported standard deviation (SD) or
standard error (SE) is stored in the record with its kind, unconverted;
the RMSE is unweighted and a missing error stays missing. An ensemble range
describes the supplied members. The format of the observation file is described in
`docs/operation-formats.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alambiq.capabilities.operate.compare_report import render, render_ensemble
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class CompareRequest:
    """The simulation record and the observations to compare with it.

    operation:
        A stored `operation.simulate` record.
    comparison:
        Path to the observation file. Each observation has an `id`, a time
        (`time_s`), a `state` (`before` or `after`), a quantity, a target, a unit,
        a value, a partition and a source. A `conversion` observation also has
        `reference_M`, an `effluent-concentration` observation also has
        `collection_id`, and any observation may have `reported_error`.
    """

    operation: str
    comparison: str


@dataclass(frozen=True, slots=True)
class EnsembleRequest:
    """The simulation records of an ensemble and the observations to compare with them.

    members:
        Stored `operation.simulate` records keyed by member name, reported in the
        order listed.
    comparison:
        Path to the observation file, shared by every member. The file does not
        name a simulated definition, because each member has its own.
    ensemble_id, source:
        The identifier of the set and a text stating where the members come from.
    """

    members: dict[str, str]
    comparison: str
    ensemble_id: str
    source: str


@dataclass(frozen=True, slots=True)
class ErrorGroup:
    """The RMSE of the residuals of one residual group (partition, quantity, target, unit) and the count of the group.

    Conversion observations with different `reference_M` form separate residual groups.
    The reference concentration is a field of the group in the stored record and not of
    this type.
    """

    partition: str
    quantity: str
    target: str
    unit: str
    count: int
    rmse: float


@dataclass(frozen=True, slots=True)
class CompareResult:
    """The RMSE of the simulation for each residual group of observations, and the stored record."""

    observations: int
    errors: list[ErrorGroup]
    record: str
    """Path to the stored record: every matched observation with its residual."""


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """The RMSE of the pointwise median of the members and of each member. The stored record holds the minimum and maximum over the members for every observation."""

    members: int
    observations: int
    errors: list[ErrorGroup]
    """Errors of the pointwise median across members."""
    member_errors: dict[str, list[ErrorGroup]]
    """Errors of each member, keyed by member name."""
    record: str
    """Path to the stored record."""


def _groups(rows: list[dict[str, Any]]) -> list[ErrorGroup]:
    return [
        ErrorGroup(
            partition=r["partition"],
            quantity=r["quantity"],
            target=r["target"],
            unit=r["unit"],
            count=r["count"],
            rmse=r["rmse"],
        )
        for r in rows
    ]


@capability(
    "operation.compare",
    version="1.0",
    family="operate",
    summary="Compare a simulation record with measurements",
    render=render,
)
def compare(request: CompareRequest, ctx: StepContext) -> CompareResult:
    """Match the observations to the simulation record and report the RMSE of each residual group."""
    from alambiq.capabilities.operate.science.comparison import compare_operation

    record = compare_operation(
        ctx.read_json(request.operation), ctx.read_json(request.comparison)
    )
    return CompareResult(
        observations=record["counts"]["observations"],
        errors=_groups(record["errors"]),
        record=ctx.keep_json(record),
    )


@capability(
    "operation.compare_ensemble",
    version="1.0",
    family="operate",
    summary="Compare a set of simulation records with the same measurements",
    render=render_ensemble,
)
def compare_ensemble(request: EnsembleRequest, ctx: StepContext) -> EnsembleResult:
    """Compare every member with the observations and report the RMSE of the median prediction and of each member."""
    from alambiq.capabilities.operate.science.comparison import (
        compare_ensemble as compare_members,
    )

    ensemble = {
        "id": request.ensemble_id,
        "source": request.source,
        "members": [
            {"id": name, "operation": ctx.read_json(path)}
            for name, path in request.members.items()
        ],
    }
    record = compare_members(ensemble, ctx.read_json(request.comparison))
    return EnsembleResult(
        members=record["counts"]["members"],
        observations=record["counts"]["observations"],
        errors=_groups(record["errors"]),
        member_errors={m["id"]: _groups(m["errors"]) for m in record["members"]},
        record=ctx.keep_json(record),
    )
