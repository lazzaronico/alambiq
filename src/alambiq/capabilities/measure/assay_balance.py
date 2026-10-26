"""Compute the conversion and the volumetric activity that a batch product target requires.

The capability takes a fitted Michaelis-Menten initial-rate curve and a stated batch
(substrate, volume, duration, product, and the formation, recovery and accepted
fractions). For each scenario it returns the conversion that the target requires and
the volumetric activity (the required Vmax) that reaches it in the stated time. A
scenario is the set of request values with the overrides of one `sensitivity` entry.
The scenario `base` has no overrides and is always computed.

Method. The calculation integrates irreversible single-substrate Michaelis-Menten
depletion at constant activity and volume:

    required Vmax = measured_per_substrate * (S0 * X - Km * ln(1 - X)) / t

S0 is `substrate_initial`. X is the required conversion, the fraction of the initial
substrate consumed: `target_product` divided by the accepted product at full conversion.
Km is the fitted Michaelis constant and t is `duration`. `measured_per_substrate` is
the moles of measured species per mole of substrate in the rate basis of the fit.

The rate basis is the `rate_basis` request field of `assay.fit`. It converts the
measured rate to substrate consumed. The required Vmax is in the measured units of the
fit.

Limits.
- The batch, the target, the duration and the three fractions are request values of
  the scenario, not measured outcomes.
- The equation assumes that the assay kinetics hold at the batch conditions, with
  product inhibition, enzyme decay, reversibility and transport limits absent.
- A scenario is flagged as an extrapolation when its initial substrate concentration is
  above the highest concentration of the fit. A scenario without a `limitation` is also
  flagged when the concentration remaining after the required conversion is below the
  lowest concentration of the fit.
- A scenario has a `limitation` message and no required Vmax when no activity can reach
  the target. This happens when a fraction is 0 (no accepted product), when the target
  exceeds the accepted product at full conversion (conversion above 1), and when the
  conversion is 1 with Km above 0 (infinite time).
- Differences between scenarios describe the sensitivity to the request values.
- The scenarios use the fitted coefficients as point values. Their uncertainty is in
  the source `assay.fit` record, and the `assay.balance` record copies the fitted
  coefficients and diagnostics of that record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alambiq.capabilities.measure.balance_report import render
from alambiq.core.context import StepContext
from alambiq.core.loader import as_data
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class Quantity:
    """A number with its unit, for example `{"value": 10, "unit": "mM"}`."""

    value: float
    unit: str


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named variation of the request values. `id` names it and `overrides` replaces the listed base values."""

    id: str
    overrides: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BalanceRequest:
    """The fit and the batch onto which it is projected.

    fit:
        Path to a stored `assay.fit` record that uses the Michaelis-Menten model
        and has a `rate_basis`.
    substrate_initial, volume, duration:
        The batch. Each is an object with a positive `value` and a `unit`, for
        example `{"value": 10, "unit": "mM"}`. Units: concentration M, mM, uM
        (micromolar) or nM; volume L or mL; time s, min or h.
    product_molar_mass, product_per_substrate:
        Molar mass of the product, an object with `value` and `unit` (g/mol), and
        moles of product formed per mole of substrate (a positive number).
    target_product:
        Accepted product to reach, an object with `value` and `unit` (g, mg or kg).
    formation_fraction, recovery_fraction, accepted_fraction:
        Each is from 0 to 1 inclusive. `formation_fraction` is the fraction of the
        theoretical product that forms. `recovery_fraction` is the fraction of the
        formed product that is recovered. `accepted_fraction` is the fraction of the
        recovered product that meets specification.
    assumption_note:
        Why the assay kinetics are expected to apply to this batch, and which assay
        conditions were not reported. Nonblank, at most 3,000 characters. Stored in
        the record.
    sensitivity:
        Up to 20 scenarios. Each is an object with an `id` and `overrides`. The `id`
        is nonempty, at most 200 characters, distinct and not `base`. `overrides` is a
        nonempty object that replaces any of `substrate_initial`, `volume`,
        `duration`, `product_molar_mass`, `target_product`, `product_per_substrate`
        and the three fractions. `fit` and `assumption_note` cannot be overridden.
        The fitted coefficients are not changed.
    """

    fit: str
    substrate_initial: Quantity
    volume: Quantity
    duration: Quantity
    product_molar_mass: Quantity
    target_product: Quantity
    product_per_substrate: float
    formation_fraction: float
    recovery_fraction: float
    accepted_fraction: float
    assumption_note: str
    sensitivity: list[Scenario] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """The projection for one scenario: maximum accepted product, required conversion, required Vmax and flags."""

    id: str
    maximum_accepted_product_g: float
    required_conversion: float | None
    required_vmax: float | None
    required_vmax_unit: str | None
    activity_multiple: float | None
    """Required Vmax over the fitted Vmax."""
    concentration_extrapolation: bool
    limitation: str | None


@dataclass(frozen=True, slots=True)
class BalanceResult:
    """The projection for every scenario and the path of the stored record."""

    scenarios: list[ScenarioResult]
    record: str
    """Path to the stored record, with its inputs and the source fit."""


@capability(
    "assay.balance",
    version="1.0",
    family="measure",
    summary="Compute the conversion and activity that a batch product target requires",
    render=render,
)
def balance(request: BalanceRequest, ctx: StepContext) -> BalanceResult:
    """Compute the conversion and activity that the target of each scenario requires."""
    from alambiq.capabilities.measure.science.balance import balance as project

    parameters = as_data(request)
    del parameters["fit"]
    record = project(ctx.read_json(request.fit), parameters)

    scenarios = []
    for row in record["scenarios"]:
        vmax = row["required_vmax"] or {}
        scenarios.append(
            ScenarioResult(
                id=row["id"],
                maximum_accepted_product_g=row["maximum_accepted_product_g"],
                required_conversion=row["required_conversion"],
                required_vmax=vmax.get("value"),
                required_vmax_unit=vmax.get("unit"),
                activity_multiple=row["activity_multiple"],
                concentration_extrapolation=row["concentration_extrapolation"],
                limitation=row["limitation"],
            )
        )
    outside = [s.id for s in scenarios if s.concentration_extrapolation]
    if outside:
        ctx.note(
            "These scenarios reach substrate concentrations outside the fitted "
            f"range: {', '.join(outside)}."
        )
    return BalanceResult(scenarios=scenarios, record=ctx.keep_json(record))
