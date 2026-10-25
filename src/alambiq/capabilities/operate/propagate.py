"""Propagate the uncertainty of a fitted rate to the predicted concentrations.

A stored `assay.fit` record of the Michaelis-Menten model holds a Km and a Vmax and,
depending on the fit, a covariance or bootstrap draws. The capability draws coefficient
sets from the covariance of the fit (`normal`) or resamples the coefficient sets of a
cluster bootstrap fit (`bootstrap`). With `normal`, a draw whose Km or Vmax is not above
0 is discarded and redrawn. With `bootstrap`, the resampled sets have the skew of the
bootstrap distribution. For each set the capability builds the operating definition as
`operation.define` does, simulates it, and reports quantiles of the predicted
concentrations at the recorded times. The result holds the spread of each species at
each recorded time and the mean and standard deviation of each coefficient over the
draws; the stored record holds every draw and the concentrations that it produced.

Limits. Only the fitted coefficients vary. The enzyme concentration of the assay,
its active fraction, the other rates of the template, the dependences on
temperature and pH, and the model form are held fixed. The fields `temperature_fit`,
`ph_fit`, `assay_temperature_K` and `assay_pH` of `kinetics` are ignored. The spread
therefore includes only the uncertainty of the fitted coefficients. The Michaelis-Menten
form is assumed to hold at the operating conditions. The normal
approximation assumes jointly normal coefficients, which a nonlinear fit
satisfies only approximately. `normal` fails when the fit has no covariance (a
coefficient at a bound, an ill-conditioned fit, or a failed bootstrap resample) and when
50 batches of `samples` draws do not yield `samples` accepted draws. `bootstrap` fails
when the fit has no bootstrap draws. The number of discarded draws is reported, and a
note is issued when it exceeds 20 % of `samples`. The sampling error of a quantile is
proportional to 1/sqrt(n) for n samples.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Literal

from alambiq.capabilities.operate.define import ChannelKinetics
from alambiq.capabilities.operate.propagate_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class PropagateRequest:
    """The operating definition, the fit whose uncertainty is propagated, and the sampling settings.

    template:
        Path to the operating definition, as for `operation.define`.
    kinetics:
        The channel, its Michaelis-Menten fit and the assay's enzyme and its active
        fraction. The fit's coefficients are the uncertain quantities. The fields
        `temperature_fit`, `ph_fit`, `assay_temperature_K` and `assay_pH` are
        ignored.
    species:
        The species whose concentrations are reported; all species when empty.
    samples:
        Number of coefficient sets to simulate, 20 to 2000. Default 200.
    seed:
        The random seed; the same seed reproduces the draws. Default 1.
    distribution:
        `normal` draws from the fit's covariance; `bootstrap` resamples the draws of a
        `cluster_bootstrap` fit. Default `normal`.
    quantiles:
        The quantile levels to report, each strictly between 0 and 1. Default 0.05,
        0.5 and 0.95.
    solver:
        `Radau` or `DOP853`, as for `operation.simulate`. Default `Radau`.
    """

    template: str
    kinetics: ChannelKinetics
    species: list[str] = field(default_factory=list)
    samples: int = 200
    seed: int = 1
    distribution: Literal["normal", "bootstrap"] = "normal"
    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.5, 0.95])
    solver: Literal["Radau", "DOP853"] = "Radau"


@dataclass(frozen=True, slots=True)
class SampledCoefficient:
    """A fitted coefficient with the mean and standard deviation of its draws."""

    name: str
    value: float
    unit: str
    draw_mean: float
    draw_standard_deviation: float


@dataclass(frozen=True, slots=True)
class PredictedSpread:
    """The predicted concentration of one species at one recorded time."""

    species: str
    time_s: float
    nominal_M: float
    """The prediction with the fit's coefficients exactly as fitted."""
    mean_M: float
    standard_deviation_M: float
    quantiles_M: dict[str, float]


@dataclass(frozen=True, slots=True)
class PropagateResult:
    """The spread of the predicted output and the draws on which it is based."""

    distribution: str
    samples: int
    discarded_draws: int
    """Number of normal draws discarded and redrawn because Km or Vmax was negative."""
    coefficients: list[SampledCoefficient]
    spreads: list[PredictedSpread]
    record: str
    """Path to the stored record: every draw and the concentrations that it produced."""


@capability(
    "operation.propagate",
    version="1.0",
    family="operate",
    summary="Propagate the uncertainty of a fitted rate to the predicted concentrations",
    render=render,
)
def propagate_uncertainty(
    request: PropagateRequest, ctx: StepContext
) -> PropagateResult:
    """Draw coefficient sets, simulate each, and report the spread of the output."""
    from alambiq.capabilities.operate.science.definition import define
    from alambiq.capabilities.operate.science.propagation import (
        draw_bootstrap,
        draw_normal,
        propagate,
    )
    from alambiq.capabilities.operate.science.simulation import simulate
    from alambiq.capabilities.shared.records import digest

    if not 20 <= request.samples <= 2000:
        raise ValueError(
            f"samples must be between 20 and 2,000; found {request.samples}."
        )
    if not request.quantiles or not all(0 < q < 1 for q in request.quantiles):
        raise ValueError(
            "Each quantile must lie strictly between 0 and 1; found "
            f"{list(request.quantiles)}."
        )
    template = ctx.read_json(request.template)
    fit = ctx.read_json(request.kinetics.fit)
    entry = {
        "channel": request.kinetics.channel,
        "fit": fit,
        "fit_sha256": digest(fit),
        "species": request.kinetics.species,
        "assay_enzyme": {
            "value": request.kinetics.assay_enzyme.value,
            "unit": request.kinetics.assay_enzyme.unit,
        },
        "assay_active_fraction": request.kinetics.assay_active_fraction,
    }
    if fit.get("model") != "initial-rate-michaelis-menten-v1":
        raise ValueError(
            "kinetics.fit must be an assay.fit record of a Michaelis-Menten fit "
            f"(model initial-rate-michaelis-menten-v1); found model {fit.get('model')!r}."
        )
    if request.distribution == "normal":
        names, draws, discarded = draw_normal(
            fit, request.samples, request.seed, positive=("Km", "Vmax")
        )
    else:
        names, draws = draw_bootstrap(fit, request.samples, request.seed)
        discarded = 0
    nominal = simulate(define(template, [entry])["definition"], method=request.solver)
    species = request.species or [s["id"] for s in template["species"]]
    unknown = set(species) - {s["id"] for s in template["species"]}
    if unknown:
        raise ValueError(
            "species names species that the template does not declare: "
            f"{', '.join(sorted(unknown))}."
        )

    trajectories = []
    for row in draws:
        sampled = copy.deepcopy(fit)
        for parameter in sampled["parameters"]:
            parameter["value"] = float(row[names.index(parameter["name"])])
        built = define(template, [{**entry, "fit": sampled}])["definition"]
        trajectories.append(simulate(built, method=request.solver)["trajectory"])

    spreads = []
    values: dict[str, list[list[float]]] = {}
    times = [state["time_s"] for state in nominal["trajectory"]]
    for name in species:
        values[name] = []
        for index, time in enumerate(times):
            column = [t[index]["concentrations_M"][name] for t in trajectories]
            stats = propagate(column, lambda value: value, request.quantiles)
            values[name].append(stats["values"])
            spreads.append(
                PredictedSpread(
                    species=name,
                    time_s=time,
                    nominal_M=nominal["trajectory"][index]["concentrations_M"][name],
                    mean_M=stats["mean"],
                    standard_deviation_M=stats["standard_deviation"],
                    quantiles_M=stats["quantiles"],
                )
            )
    if discarded > request.samples * 0.2:
        ctx.note(
            f"{discarded} draws were discarded because a coefficient was negative, "
            "which is more than 20 % of the requested samples."
        )
    units = {p["name"]: p.get("unit", "") for p in fit["parameters"]}
    reported = {p["name"]: float(p["value"]) for p in fit["parameters"]}
    record = {
        "distribution": request.distribution,
        "seed": request.seed,
        "coefficient_names": names,
        "draws": draws.tolist(),
        "times_s": times,
        "concentrations_M": values,
    }
    return PropagateResult(
        distribution=request.distribution,
        samples=int(len(draws)),
        discarded_draws=discarded,
        coefficients=[
            SampledCoefficient(
                name=n,
                value=reported[n],
                unit=units[n],
                draw_mean=float(draws[:, i].mean()),
                draw_standard_deviation=float(draws[:, i].std(ddof=1)),
            )
            for i, n in enumerate(names)
        ],
        spreads=spreads,
        record=ctx.keep_json(record),
    )
