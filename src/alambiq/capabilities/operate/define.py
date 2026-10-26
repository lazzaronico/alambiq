"""Build an operating definition from a template and measured kinetics.

The capability sets the rate values of the channels named in `kinetics` from stored
`assay.fit` records and writes the operating definition that `operation.simulate`
reads. It returns the capacity and saturation scale of each channel.

An operating definition is a JSON file. Each of its rate channels is one reaction in one
direction, with a stoichiometry, a capacity and rate factors. A channel needs a capacity
in M/s at the reference amount of its catalyst and the initial volume. It also needs a
saturation scale, the concentration at which the saturation factor is half of its
maximum.

Method. A stored Michaelis-Menten `assay.fit` record gives a Vmax and a Km for the
enzyme concentration of the assay. It also gives a rate basis: the species on which the
rate was measured, the substrate, and the molar ratio `measured_per_substrate` of the
measured species to the substrate. For each channel named in `kinetics`:

1. Divide the Vmax by `measured_per_substrate`. The result is the substrate-equivalent
   rate.
2. Divide by the absolute value of the stoichiometric coefficient of the substrate in
   the channel. The result is the rate of the channel, in reaction extent per volume and
   time, at the enzyme concentration of the assay.
3. Divide by the enzyme concentration of the assay and by the active fraction of that
   enzyme. The result is a rate per amount of fully active enzyme.
4. Multiply by the reference amount of the catalyst divided by the initial volume. The
   result is the capacity in M/s.
5. Set the saturation scale equal to the Km.

The other values are set as described for their fields: the capacity of the reverse
channel and the saturation scales of each pair in `equilibria`, the deactivation of
each catalyst in `inactivation`, and the dependences on temperature and pH from
`temperature_fit` and `ph_fit`. Every other value of the template is unchanged, except
the `source` text of each channel whose capacity is set.

Limits.
- The result assumes the amount of enzyme in the assay and its active fraction, both as
  stated in `kinetics`, and that the temperature, pH and composition of the reactor
  equal those of the assay.
- A Km that is not above 0 is refused.
- A change of conditions is modeled only for a channel whose `kinetics` entry supplies
  `temperature_fit` or `ph_fit`.
- Inhibition, reversibility and deactivation come from the template, except for the
  reverse channels set by `equilibria` and the deactivation set by `inactivation`.
- A channel is flagged as an extrapolation when the largest substrate concentration
  stated in the definition (the initial concentration and the feed concentration of each
  phase) exceeds the highest substrate concentration covered by the fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from alambiq.capabilities.operate.define_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class AssayEnzyme:
    """The enzyme concentration in the assay from which the fit was obtained.

    value, unit:
        The concentration and its unit: an amount per volume such as `mg/mL`. When
        the catalyst is counted in a mass (g, mg, ug), any mass per volume is
        converted; otherwise the unit must be the unit of the catalyst, per
        liter, such as `U/L`.
    """

    value: float
    unit: str


@dataclass(frozen=True, slots=True)
class ChannelKinetics:
    """The channel that a fit sets, and the assay conditions of the fit.

    channel:
        The identifier of a template channel. The channel must name a catalyst and
        either have a first-power saturation factor on `species` and no
        `denominator`, or be the forward channel of a pair in `equilibria`.
    fit:
        Path to a stored `assay.fit` record of the Michaelis-Menten model with a
        rate basis.
    species:
        The substrate that the channel consumes and the assay measured (a template
        species identifier).
    assay_enzyme:
        The enzyme concentration of the assay.
    assay_active_fraction:
        The fraction of that enzyme that was active, above 0 and at most 1. A rate
        measured with partly active enzyme corresponds to a higher rate at full
        activity.
    temperature_fit:
        Optional. Path to a stored `assay.fit` record of the Arrhenius model for the
        rate constant of this channel. Its activation energy makes the capacity
        of the channel depend on the temperature that the operation states.
    ph_fit:
        Optional. Path to a stored `assay.fit` record of the pH-profile model. Its
        two pKa values make the capacity depend on the pH that the operation
        states.
    assay_temperature_K, assay_pH:
        The temperature and pH of the kinetics assay, at which the capacity is
        stated. Required with `temperature_fit` and `ph_fit` respectively.
    """

    channel: str
    fit: str
    species: str
    assay_enzyme: AssayEnzyme
    assay_active_fraction: float = 1.0
    temperature_fit: str | None = None
    ph_fit: str | None = None
    assay_temperature_K: float | None = None
    assay_pH: float | None = None


@dataclass(frozen=True, slots=True)
class Quantity:
    """A concentration with its unit: `M`, `mM`, `uM` or `nM`."""

    value: float
    unit: str


@dataclass(frozen=True, slots=True)
class ReversiblePair:
    """A forward and a reverse channel linked by the equilibrium constant.

    forward, reverse:
        Template channel identifiers. Both are written as a mass-action factor on
        their own substrate over the denominator `1 + S/Ks + P/Kp`, and share one
        catalyst. The forward channel must also have a kinetics entry.
    substrate, product:
        The template species that the forward channel converts into each other.
    product_km:
        The Michaelis constant of the product in the reverse direction.
    equilibrium_constant, ln_equilibrium_constant:
        Exactly one: [P]/[S] at equilibrium, or its natural logarithm, such as the
        `ln_k_prime` of `reaction.estimate_thermodynamics`.
    equilibrium_temperature_K:
        The temperature of the constant. The estimate applies at 298.15 K. The
        constant is used unchanged when the operation states another temperature.
    """

    forward: str
    reverse: str
    substrate: str
    product: str
    product_km: Quantity
    equilibrium_constant: float | None = None
    ln_equilibrium_constant: float | None = None
    equilibrium_temperature_K: float = 298.15


@dataclass(frozen=True, slots=True)
class Inactivation:
    """The loss of activity of a catalyst, from a first-order decay fit.

    catalyst:
        The template catalyst whose `deactivation_per_s` is set.
    decay_fit:
        Path to a stored `assay.fit` record of the first-order decay model, with
        time in s, min or h.
    decay_temperature_K:
        The temperature of the decay measurement.
    temperature_fit:
        Optional. Path to an Arrhenius fit of decay rate constants measured at
        several temperatures. Its activation energy makes the deactivation depend on
        the stated temperature.
    """

    catalyst: str
    decay_fit: str
    decay_temperature_K: float
    temperature_fit: str | None = None


@dataclass(frozen=True, slots=True)
class DefineRequest:
    """The template and the kinetics to set in it.

    template:
        Path to an operating definition, as `operation.simulate` reads it. The
        capacity and saturation scale of the channels named in `kinetics` are
        replaced; their stored values are placeholders.
    kinetics:
        One entry per channel to set, 1 to 50.
    equilibria:
        Reversible pairs. The capacity of the reverse channel follows from the forward
        kinetics, the Michaelis constant of the product and the equilibrium constant
        by the Haldane relation. The equilibrium ratio [P]/[S] of the rate law then
        equals the equilibrium constant.
    inactivation:
        Catalysts whose loss of activity is set from decay fits.
    """

    template: str
    kinetics: list[ChannelKinetics]
    equilibria: list[ReversiblePair] = field(default_factory=list)
    inactivation: list[Inactivation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DefinedChannel:
    """One channel as set from its fit."""

    channel: str
    catalyst: str
    species: str
    capacity_M_s: float
    """At the catalyst's reference amount, full activity and the initial volume."""
    scale_M: float
    """The saturation scale, equal to the Km of the fit."""
    vmax_substrate_equivalent_M_s: float
    assay_enzyme_per_L: float
    """In the catalyst's own amount unit."""
    fit_domain_M: list[float]
    """The lowest and highest substrate concentration covered by the fit."""
    substrate_maximum_M: float
    """The highest substrate concentration in the initial state or the feed of the definition."""
    extrapolates: bool
    activation_energy_J_mol: float | None = None
    """Set when a temperature fit makes the capacity depend on the stated temperature."""
    ph_pKa: list[float] | None = None
    """The two pKa values, when a pH fit makes the capacity depend on the stated pH."""


@dataclass(frozen=True, slots=True)
class DefinedEquilibrium:
    """A reversible pair as set."""

    forward: str
    reverse: str
    equilibrium_constant: float
    product_km_M: float
    reverse_capacity_M_s: float
    implied_equilibrium_constant: float
    """The equilibrium constant of the finished rate law, recomputed from its parameters."""


@dataclass(frozen=True, slots=True)
class DefinedInactivation:
    """The deactivation of a catalyst as set."""

    catalyst: str
    deactivation_per_s: float
    half_life_s: float | None
    decay_temperature_K: float
    activation_energy_J_mol: float | None


@dataclass(frozen=True, slots=True)
class DefineResult:
    """The definition and the values set in it."""

    operation_id: str
    channels: list[DefinedChannel]
    unchanged_channels: list[str]
    """Template channels that are unchanged."""
    equilibria: list[DefinedEquilibrium]
    inactivation: list[DefinedInactivation]
    definition: str
    """Path to the operating definition, for use by `operation.simulate`."""


@capability(
    "operation.define",
    version="1.0",
    family="operate",
    summary="Set the rate values of an operating definition from measured kinetics",
    render=render,
)
def define_operation(request: DefineRequest, ctx: StepContext) -> DefineResult:
    """Convert each fit to a capacity and a saturation scale and write the definition."""
    from alambiq.capabilities.operate.science.definition import define
    from alambiq.capabilities.shared.records import digest

    if not 1 <= len(request.kinetics) <= 50:
        raise ValueError(
            f"kinetics must have 1 to 50 entries; found {len(request.kinetics)}."
        )
    template = ctx.read_json(request.template)
    entries: list[dict[str, Any]] = []
    for item in request.kinetics:
        fit = ctx.read_json(item.fit)
        entries.append(
            {
                "channel": item.channel,
                "fit": fit,
                "fit_sha256": digest(fit),
                "species": item.species,
                "assay_enzyme": {
                    "value": item.assay_enzyme.value,
                    "unit": item.assay_enzyme.unit,
                },
                "assay_active_fraction": item.assay_active_fraction,
                "temperature_fit": (
                    ctx.read_json(item.temperature_fit)
                    if item.temperature_fit is not None
                    else None
                ),
                "ph_fit": (
                    ctx.read_json(item.ph_fit) if item.ph_fit is not None else None
                ),
                "assay_temperature_K": item.assay_temperature_K,
                "assay_pH": item.assay_pH,
            }
        )
    equilibria = []
    for pair in request.equilibria:
        if (pair.equilibrium_constant is None) == (
            pair.ln_equilibrium_constant is None
        ):
            raise ValueError(
                f"Reversible pair {pair.forward}/{pair.reverse}: set exactly one of "
                "equilibrium_constant and ln_equilibrium_constant."
            )
        constant = (
            pair.equilibrium_constant
            if pair.equilibrium_constant is not None
            else math.exp(pair.ln_equilibrium_constant)
        )
        if not math.isfinite(constant) or constant <= 0:
            raise ValueError(
                f"The equilibrium constant of reversible pair {pair.forward}/"
                f"{pair.reverse} must be a finite number above 0."
            )
        factor = _CONCENTRATION.get(pair.product_km.unit)
        if factor is None or pair.product_km.value <= 0:
            raise ValueError(
                f"product_km of reversible pair {pair.forward}/{pair.reverse} must be a "
                f"concentration above 0 in one of the units {', '.join(_CONCENTRATION)}."
            )
        state = (template.get("state") or {}).get("temperature_K")
        if state is not None and abs(state - pair.equilibrium_temperature_K) > 0.01:
            ctx.note(
                f"The equilibrium constant is for {pair.equilibrium_temperature_K:g} K "
                f"and the operation is stated at {state:g} K. The constant is not "
                "corrected for temperature."
            )
        equilibria.append(
            {
                "forward": pair.forward,
                "reverse": pair.reverse,
                "substrate": pair.substrate,
                "product": pair.product,
                "product_km_M": pair.product_km.value * factor,
                "equilibrium_constant": constant,
            }
        )
    inactivations = [
        {
            "catalyst": item.catalyst,
            "decay_fit": ctx.read_json(item.decay_fit),
            "decay_temperature_K": item.decay_temperature_K,
            "temperature_fit": (
                ctx.read_json(item.temperature_fit)
                if item.temperature_fit is not None
                else None
            ),
        }
        for item in request.inactivation
    ]
    built = define(template, entries, equilibria, inactivations)
    for row in built["channels"]:
        if row["extrapolates"]:
            ctx.note(
                f"Channel {row['channel']} reaches {row['substrate_maximum_M']:g} M of "
                f"{row['species']}, above the highest measured concentration of the fit, "
                f"{row['fit_domain_M'][1]:g} M."
            )
    definition = built["definition"]
    return DefineResult(
        operation_id=definition["id"],
        channels=[
            DefinedChannel(**{k: row[k] for k in _FIELDS}) for row in built["channels"]
        ],
        unchanged_channels=built["unchanged"],
        equilibria=[DefinedEquilibrium(**row) for row in built["equilibria"]],
        inactivation=[DefinedInactivation(**row) for row in built["inactivation"]],
        definition=ctx.keep_json(definition),
    )


_CONCENTRATION = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9}

_FIELDS = (
    "channel",
    "catalyst",
    "species",
    "capacity_M_s",
    "scale_M",
    "vmax_substrate_equivalent_M_s",
    "assay_enzyme_per_L",
    "fit_domain_M",
    "substrate_maximum_M",
    "extrapolates",
    "activation_energy_J_mol",
    "ph_pKa",
)
