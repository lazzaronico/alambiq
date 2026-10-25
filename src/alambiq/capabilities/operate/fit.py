"""Fit operating definitions to measured progress or recovery.

The fit adjusts named parameters of one or more operating definitions (the files that
`operation.simulate` integrates, listed under `experiments` in the fit specification)
to minimize the standardized residuals between the predictions of
`operation.simulate` and the supplied observations: concentrations, conversions,
removed amounts, collected effluent concentrations or catalyst amounts. A fitted
parameter is a field value of the operating definitions that the fit adjusts. A
parameter can target several fields, so that one physical constant is shared across
operating definitions, exchanges or both reaction directions. Fields that no parameter
targets retain their supplied values.

The optimizer is bounded trust-region least squares on scaled parameters, started
from every supplied start. Each residual evaluation integrates every operating
definition that has `fit` observations. The number of residual evaluations of each
start is limited by `max_nfev` (1 to 2000). Each evaluation of the Jacobian needs 2
residual evaluations per free parameter, which do not count against `max_nfev`. The
reported parameters are those of the converged start (the optimizer reports success)
with the lowest sum of squared standardized residuals; the fit fails when no start
converges. Each observation carries the partition label `fit` or `evaluation` in the
fit specification. The parameters are fitted to the `fit` observations only, and the
error of both partitions is reported as a standardized RMSE: the root mean square
of the residuals (prediction minus value), each divided by the `scale` of its
observation. For each parameter that has `profile_values`, the capability fixes
the parameter at each of these values, re-fits the other parameters and stores
the sum of squared standardized residuals at each value in the stored record.

Limits. The fit specification has 1 to 20 operating definitions, 1 to 12 parameters,
1 to 10 starts, at most 50 `profile_values` per parameter, 1 to 10000 observations and
more `fit` observations than parameters. `descriptive` uncertainty reports the
diagnostics and the profiles only. `independent-residuals` adds local 95 % intervals,
computed from the Jacobian of the standardized residuals at the optimum. They assume
independent residuals with one common variance after division by the observation
`scale`, and they are computed only if the rank equals the number of parameters and no
parameter is at a bound. Residuals within one progress curve are correlated,
which violates the independence assumption; the intervals are valid for fits whose
observations are independent of each other, which excludes several observations
along one progress curve. The rank counts the singular values of that Jacobian
above 1e-7 times the largest. A parameter is at a bound when it is within 1e-4 of
the bound in scaled units. The result holds the rank, the scaled condition number of the Jacobian and the
parameters at a bound; the stored record also holds the profiles. A
rank below the number of parameters means that the data do not determine at least
one combination of parameters. A parameter at a bound is a limit of the search
range. If an active fraction and a catalytic capacity are
estimated together, the data determine only their product. Recovery and
deactivation parameters are separable only with direct measurements of recovery
and of activity. The format of the fit specification is described in
`docs/operation-formats.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.operate.fit_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class OperationFitRequest:
    """The fit specification and the solver.

    definition:
        Path to the fit specification: the operating definitions (`experiments`),
        the parameters with bounds, scales and target fields, the starts, the
        observations with their fit or evaluation partition, and the uncertainty
        basis.
    solver:
        The integrator of each forward simulation, as in `operation.simulate`.
        Default `Radau`.
    """

    definition: str
    solver: Literal["Radau", "DOP853"] = "Radau"


@dataclass(frozen=True, slots=True)
class FittedOperationParameter:
    """One fitted parameter with its unit, bounds, local interval and bound flag."""

    name: str
    value: float
    unit: str
    lower: float
    upper: float
    interval: list[float] | None
    """Local 95 % interval; present only under `independent-residuals`."""
    at_bound: bool


@dataclass(frozen=True, slots=True)
class OperationFitResult:
    """The fitted parameters, the fit and evaluation errors, the diagnostics and the paths of the fitted operating definitions."""

    parameters: list[FittedOperationParameter]
    fitted: int
    fit_error: float | None
    """Standardized RMSE over the fitted observations."""
    evaluated: int
    evaluation_error: float | None
    """Standardized RMSE over the held-out observations."""
    rank: int
    scaled_condition_number: float | None
    uncertainty: str
    starts: int
    converged_starts: int
    operations: dict[str, str]
    """Path to each fitted operating definition, keyed by id, for use by `operation.simulate`."""
    record: str
    """Path to the stored fit record."""


@capability(
    "operation.fit",
    version="1.0",
    family="operate",
    summary="Fit operating definitions to measured progress or recovery",
    render=render,
)
def fit(request: OperationFitRequest, ctx: StepContext) -> OperationFitResult:
    """Fit the named parameters and report the fit with its diagnostics."""
    from alambiq.capabilities.operate.science.fitting import fit_operation

    record = fit_operation(ctx.read_json(request.definition), method=request.solver)

    diagnostics = record["diagnostics"]
    intervals = diagnostics["local_95_percent_intervals"] or {}
    bounds = set(diagnostics["parameters_at_bounds"])
    parameters = [
        FittedOperationParameter(
            name=p["id"],
            value=record["parameters"][p["id"]],
            unit=p["unit"],
            lower=p["lower"],
            upper=p["upper"],
            interval=intervals.get(p["id"]),
            at_bound=p["id"] in bounds,
        )
        for p in record["request"]["parameters"]
    ]
    if bounds:
        ctx.note(
            f"These parameters finished at a bound: {', '.join(sorted(bounds))}."
        )
    if diagnostics["rank"] < len(parameters):
        ctx.note(
            f"The fit determines {diagnostics['rank']} of {len(parameters)} "
            "parameter directions."
        )
    errors = record["errors"]
    return OperationFitResult(
        parameters=parameters,
        fitted=errors["fit"]["count"],
        fit_error=errors["fit"]["standardized_rmse"],
        evaluated=errors["evaluation"]["count"],
        evaluation_error=errors["evaluation"]["standardized_rmse"],
        rank=diagnostics["rank"],
        scaled_condition_number=diagnostics["scaled_condition_number"],
        uncertainty=diagnostics["uncertainty_basis"],
        starts=len(record["starts"]),
        converged_starts=sum(1 for s in record["starts"] if s["success"]),
        operations={
            name: ctx.keep_json(operation["request"])
            for name, operation in record["operations"].items()
        },
        record=ctx.keep_json(record),
    )
