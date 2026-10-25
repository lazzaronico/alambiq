"""Fit a response model to the measurements of one sample and one metric.

The fit estimates the coefficients of an explicit response model from supplied
measurements. An observation is the value that enters the fit: one measurement
under the `supplied-values` basis, or the mean of the technical readings of one
independent unit under the `independent-means` basis. Five models are
available, identified by the value of `model`:

- `linear`: intercept + slope * x, with the coefficients `intercept` and `slope`.
- `michaelis_menten`: Vmax * x / (Km + x), the initial-rate relation, with the
  coefficients `Km` and `Vmax`.
- `arrhenius`: k_ref * exp(-Ea / R * (1 / T - 1 / T_ref)), the dependence of a
  rate constant on temperature, with the coefficients `k_ref` and `Ea`. T is
  the temperature in K, T_ref is the reference temperature in K (`reference_x`)
  and R is 8.314462618 J/(mol K).
- `first_order_decay`: A0 * exp(-k * x), first-order decay in time (thermal
  inactivation), with the coefficients `A0` and `k`.
- `ph_bell`: ymax / (1 + 10^(pKa1 - pH) + 10^(pH - pKa2)), a bell-shaped
  activity profile over pH (the independent quantity x is the pH), with the
  coefficients `ymax`, `pKa1` and `pKa2`.

Coefficients can be bounded or held fixed at supplied values, and predictions
can be requested at chosen values of the independent quantity. Quantities that
follow from the coefficients are reported with a standard error and a 95 %
interval from the delta method: the half-life for decay, the activation energy
in kJ/mol (the coefficient `Ea` is in J/mol) and the pre-exponential factor for
Arrhenius, and the optimum pH and the activity at the optimum for the pH
profile. The reference temperature of an Arrhenius fit is reported without a
standard error.

Three intervals are reported. The standard error and 95 % interval of a
coefficient describe how precisely the data determine it. The mean-response
interval is the 95 % interval for the expected value of the response at a point
of the independent quantity. The future-observation interval is the 95 %
interval for a single new measurement at that point and is wider; it is not
estimated with `cluster_bootstrap`. Measurements listed in `exclude` are omitted from the fit, and their measured
values can be compared with the predictions to assess the error on held-out
measurements.

Limits. Every interval is conditional on the recorded experiment and the
chosen model; its width reflects the scatter of the data about that model. With
`uncertainty` set to `residual` (the default) the intervals come from the
covariance of the model linearized at the fitted coefficients and from
Student-t quantiles; they assume independent residuals with equal variance, and
for nonlinear models they are approximate and can extend beyond the physical
bounds of a coefficient. A cluster bootstrap resamples whole independent units
instead and gives percentile intervals for the coefficients and the mean
response; derived quantities use the delta method with the bootstrap
covariance, and no future-observation interval is computed. The cluster
bootstrap is valid only when the independent units are the sampling units.

A fixed coefficient has no standard error or interval. When any free
coefficient ends at a bound, when the scaled Jacobian is rank deficient or has
a condition number of 1e8 or more, when the response is flat across x for a
model other than `linear` with 2 or more free coefficients, or when any
bootstrap resample fails, no coefficient, derived quantity or prediction has a
standard error or interval, and a diagnostic states this.

The fit requires at least 4 observations at 3 or more distinct values of x.
Measurements without a value (status other than `measured`) and measurements
whose condition lacks the x quantity are omitted and listed with the reason in
the record. `supplied-values` rejects measurements that have an
`independent_id`; `independent-means` requires an `independent_id` on every
included measurement. `arrhenius` requires `x_unit` K or degC, temperatures
above absolute zero and positive responses. `michaelis_menten` and
`first_order_decay` require x of 0 or more. A prediction outside the measured
range is labeled as extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from alambiq.capabilities.measure.assay_fit_report import render
from alambiq.core.context import StepContext
from alambiq.core.loader import as_data
from alambiq.core.registry import capability

Model = Literal[
    "linear", "michaelis_menten", "arrhenius", "first_order_decay", "ph_bell"
]

_PROFILE = {
    "linear": "response-linear-v1",
    "michaelis_menten": "initial-rate-michaelis-menten-v1",
    "arrhenius": "temperature-arrhenius-v1",
    "first_order_decay": "first-order-decay-v1",
    "ph_bell": "ph-bell-v1",
}

# A coefficient is at a bound when it sits within this relative distance of it.
_AT_BOUND = 1e-6


@dataclass(frozen=True, slots=True)
class RateBasis:
    """The relation between the measured rate and the moles of substrate consumed.

    kind:
        `product-formation` or `substrate-consumption`.
    measured_species, substrate:
        The species the assay measures and the substrate it reports on.
    measured_per_substrate:
        Moles of measured species per mole of substrate consumed, including any
        reporter yield that the assay specifies. For `substrate-consumption` the
        assay measures the substrate itself, so the ratio is 1.
    """

    kind: Literal["product-formation", "substrate-consumption"]
    measured_species: str
    substrate: str
    measured_per_substrate: float


@dataclass(frozen=True, slots=True)
class FitRequest:
    """The measurements, the response model and the fit settings.

    measurements:
        Path to a stored `assay.inspect` record (JSON).
    sample_id, metric:
        The sample and the measured quantity to fit.
    x_quantity, x_unit, y_unit:
        The independent quantity (a name from `condition_values` of the
        `assay.inspect` record) and the units of both axes. The units must equal
        the units stored in the record; no conversion is applied.
    observation_basis:
        `supplied-values` fits the values as given, as for published individual
        measurements. `independent-means` fits the mean of the technical readings
        of each named independent unit. Default `supplied-values`.
    model:
        The response model to fit: `linear`, `michaelis_menten`, `arrhenius`,
        `first_order_decay` or `ph_bell`. Default `michaelis_menten`.
    condition_ids:
        Restrict the fit to these conditions. When omitted, all conditions of the
        record are used. A condition quantity other than `x_quantity` that differs
        between the included conditions is an error unless `pool_over` names it.
    pool_over:
        Names of condition quantities (other than `x_quantity`) that differ, or
        are given for only some conditions, across the included measurements.
        The model has no term for them; a note of the step and a diagnostic in
        the stored record state that they were pooled.
    exclude:
        Ids of measurements of the fitted sample and metric that are omitted from
        the fit. Their measured values can be compared with the predictions to
        assess the error on held-out measurements.
    parameter_bounds:
        Bounds on coefficients, as `{"Vmax": [0, 150]}`: a lower and an upper
        value in the coefficient's own units, with `null` for unbounded. The
        coefficients are `Km` and `Vmax` for Michaelis-Menten, `intercept` and
        `slope` for the linear model, `k_ref` (the rate constant at the reference
        temperature) and `Ea` (the activation energy, J/mol) for Arrhenius, `A0`
        and `k` for decay, and `ymax`, `pKa1` and `pKa2` for the pH profile.
        The coefficients `Km`, `Vmax`, `k_ref`, `A0`, `k` and `ymax` cannot be
        negative: a bound given for one of them needs a lower value of 0 or more,
        and a `null` lower value is rejected.
    fixed_parameters:
        Coefficients held constant at the supplied values, as `{"Km": 2.5}`. A
        value must lie within the bounds of its coefficient. At least one
        coefficient must remain to be fitted (the two-coefficient models allow
        one fixed). The intervals for the remaining coefficients are conditional
        on the fixed values.
    reference_x:
        For the Arrhenius model, the temperature (in `x_unit`) at which the
        rate constant `k_ref` is stated. Defaults to the mean of the measured
        temperatures.
    prediction_x:
        1 to 200 values of the independent quantity, in `x_unit`, at which to
        predict. When omitted, the stored record contains predictions at 51
        evenly spaced values across the measured range, and the result contains
        none.
    uncertainty:
        `residual` gives standard errors and intervals from the covariance of the
        model linearized at the fitted coefficients, with Student-t quantiles; it
        assumes independent residuals with equal variance. `cluster_bootstrap`
        resamples whole independent units; it is valid only when those units are
        the sampling units, and requires the `independent-means` basis with at
        least 4 independent units.
    seed:
        An integer from 0 to 2,147,483,647. Required by the bootstrap; the same
        seed reproduces the reported intervals.
    resamples:
        Number of bootstrap resamples, 100 to 2,000. Default 1,000. Used only
        with `cluster_bootstrap`.
    rate_basis:
        The relation between the measured rate and the substrate consumed, for the
        `michaelis_menten` model with a concentration `x_unit` and a
        concentration-rate `y_unit`. An object with `kind` (`substrate-consumption`
        or `product-formation`), `measured_species`, `substrate` and
        `measured_per_substrate` (moles of measured species per mole of substrate
        consumed; 1, with the substrate as measured species, for
        `substrate-consumption`). Required when the fit is used by `assay.balance`.
    """

    measurements: str
    sample_id: str
    metric: str
    x_quantity: str
    x_unit: str
    y_unit: str
    observation_basis: Literal["supplied-values", "independent-means"] = (
        "supplied-values"
    )
    model: Model = "michaelis_menten"
    condition_ids: list[str] = field(default_factory=list)
    pool_over: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    parameter_bounds: dict[str, list[float | None]] = field(default_factory=dict)
    fixed_parameters: dict[str, float] = field(default_factory=dict)
    prediction_x: list[float] = field(default_factory=list)
    uncertainty: Literal["residual", "cluster_bootstrap"] = "residual"
    seed: int | None = None
    resamples: int = 1000
    rate_basis: RateBasis | None = None
    reference_x: float | None = None


@dataclass(frozen=True, slots=True)
class DerivedQuantity:
    """A quantity that follows from the coefficients, with its own uncertainty."""

    name: str
    value: float
    unit: str
    standard_error: float | None
    interval_95: list[float] | None


@dataclass(frozen=True, slots=True)
class FittedParameter:
    """One coefficient, in the units of the measurement from which it was estimated."""

    name: str
    value: float
    standard_error: float | None
    interval_95: list[float] | None
    fixed: bool
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class Prediction:
    """The fitted curve at one requested value of the independent quantity."""

    x: float
    y: float
    mean_interval_95: list[float] | None
    """95 % interval for the expected value of the response at this x."""
    prediction_interval_95: list[float] | None
    """95 % interval for a single new measurement at this x; wider than the mean interval. Not estimated with `cluster_bootstrap`."""
    extrapolation: bool
    """True when x lies outside the measured range."""


@dataclass(frozen=True, slots=True)
class FitResult:
    """The coefficients with their uncertainty, the counts of observations used and measurements omitted, the measured range, the requested predictions and the path of the stored record."""

    model: Model
    parameters: list[FittedParameter]
    observations_used: int
    observations_excluded: int
    termination: str
    uncertainty_method: str
    measured_range: list[float]
    """The lowest and highest x of the fitted observations."""
    predictions: list[Prediction]
    """One prediction per requested x; empty when `prediction_x` is not given."""
    record: str
    """Path to the stored fit record, which includes the residuals and the predictions."""
    derived: list[DerivedQuantity] = field(default_factory=list)
    """Half-life, optimum pH and similar, with delta-method uncertainty."""


def _parameters(fit: dict[str, Any]) -> list[FittedParameter]:
    return [
        FittedParameter(
            name=str(row["name"]),
            value=float(row["value"]),
            standard_error=(
                float(row["standard_error"])
                if row.get("standard_error") is not None
                else None
            ),
            interval_95=row.get("interval_95"),
            fixed=bool(row.get("fixed")),
            unit=row.get("unit"),
        )
        for row in fit.get("parameters") or []
    ]


def _predictions(fit: dict[str, Any]) -> list[Prediction]:
    return [
        Prediction(
            x=float(row["x"]),
            y=float(row["y"]),
            mean_interval_95=row.get("mean_interval_95"),
            prediction_interval_95=row.get("prediction_interval_95"),
            extrapolation=bool(row["extrapolation"]),
        )
        for row in fit.get("predictions") or []
    ]


def _bound_reached(value: float, pair: list[float | None]) -> tuple[str, float] | None:
    """Return the supplied bound, if any, at which a coefficient lies."""
    for label, edge in zip(("lower", "upper"), pair, strict=True):
        if edge is not None and abs(value - edge) <= _AT_BOUND * max(1.0, abs(edge)):
            return label, edge
    return None


@capability(
    "assay.fit",
    version="1.1",
    family="measure",
    summary="Fit a measured series: linear, Michaelis-Menten, Arrhenius, decay or pH profile",
    render=render,
    presets={
        "initial-rates": {"model": "michaelis_menten", "uncertainty": "residual"},
        "linear-range": {"model": "linear", "uncertainty": "residual"},
        "temperature": {"model": "arrhenius", "uncertainty": "residual"},
        "inactivation": {"model": "first_order_decay", "uncertainty": "residual"},
        "ph-profile": {"model": "ph_bell", "uncertainty": "residual"},
    },
)
def fit(request: FitRequest, ctx: StepContext) -> FitResult:
    """Fit the supplied measurements and report the fit with its uncertainty."""
    from alambiq.capabilities.measure.science.fitting import fit_assay

    record = ctx.read_json(request.measurements)
    parameters: dict[str, Any] = {
        "sample_id": request.sample_id,
        "metric": request.metric,
        "x_quantity": request.x_quantity,
        "x_unit": request.x_unit,
        "y_unit": request.y_unit,
        "observation_basis": request.observation_basis,
    }
    if request.condition_ids:
        parameters["condition_ids"] = list(request.condition_ids)
    if request.pool_over:
        parameters["pool_over"] = list(request.pool_over)
    if request.exclude:
        parameters["exclude_measurement_ids"] = list(request.exclude)
    if request.parameter_bounds:
        parameters["parameter_bounds"] = {
            name: list(pair) for name, pair in request.parameter_bounds.items()
        }
    if request.fixed_parameters:
        parameters["fixed_parameters"] = dict(request.fixed_parameters)
    if request.prediction_x:
        parameters["prediction_x"] = list(request.prediction_x)
    if request.rate_basis is not None:
        parameters["rate_basis"] = as_data(request.rate_basis)
    if request.reference_x is not None:
        parameters["reference_x"] = request.reference_x
    if request.uncertainty == "cluster_bootstrap":
        if request.seed is None:
            raise ValueError(
                "uncertainty cluster_bootstrap requires seed, which makes the intervals "
                "reproducible."
            )
        parameters["uncertainty"] = {
            "method": "cluster-bootstrap",
            "resamples": request.resamples,
            "seed": request.seed,
        }

    fitted = fit_assay(record, _PROFILE[request.model], parameters)

    optimizer = fitted.get("optimizer") or {}
    termination = str(
        optimizer.get("message")
        or ("converged" if optimizer.get("success") else "did not converge")
    )
    used = int((fitted.get("statistics") or {}).get("n", 0))
    excluded = len(fitted.get("excluded") or request.exclude)
    found = _parameters(fitted)
    domain = fitted.get("domain") or {}
    measured_range = [domain.get("minimum_x"), domain.get("maximum_x")]
    predictions = _predictions(fitted) if request.prediction_x else []

    if not found:
        ctx.note(
            "The solver returned no parameter estimates."
        )
    for message in fitted.get("diagnostics") or []:
        ctx.note(message)
    for parameter in found:
        if parameter.fixed:
            continue
        reached = _bound_reached(
            parameter.value, request.parameter_bounds.get(parameter.name, [None, None])
        )
        if reached:
            label, edge = reached
            ctx.note(
                f"{parameter.name} finished at its {label} bound {edge:g}: the bound, "
                "not the data, limits its value. A fit with a coefficient at a bound "
                "reports no standard error and no interval for any coefficient."
            )
    outside = sum(1 for p in predictions if p.extrapolation)
    if outside:
        ctx.note(
            f"{outside} of {len(predictions)} requested predictions lie outside the "
            f"measured range {measured_range[0]:g} to {measured_range[1]:g} "
            f"{request.x_unit}."
        )

    return FitResult(
        model=request.model,
        parameters=found,
        observations_used=used,
        observations_excluded=excluded,
        termination=termination,
        uncertainty_method=request.uncertainty,
        measured_range=measured_range,
        predictions=predictions,
        record=ctx.keep_json(fitted),
        derived=[
            DerivedQuantity(
                name=row["name"],
                value=float(row["value"]),
                unit=row["unit"],
                standard_error=row.get("standard_error"),
                interval_95=row.get("interval_95"),
            )
            for row in fitted.get("derived") or []
        ],
    )
