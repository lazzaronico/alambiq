"""Simulate a reaction over time under stated operating conditions.

The capability integrates the rate channels of an operating definition in a well-mixed
liquid reactor. It returns the state at the recorded times, with the ledger of each
species, and the maximum balance error. The operating definition is a JSON file whose
format is in `docs/operation-formats.md`.

Method. A rate channel is one reaction in one direction, with a stoichiometry, a
capacity and rate factors. The rate of a channel is its capacity multiplied by the factor
of each listed species. It is divided by the optional coupled denominator (a constant
plus terms of species concentrations) and by the optional inhibitor term (1 plus the sum
of each inhibitor concentration divided by its constant). A factor is a mass-action
factor or a saturation factor of one species. A saturation factor can include
competitive, uncompetitive and substrate inhibition.

The model also includes:
- conserved pools of cofactors and chemical moieties;
- feed and withdrawal, with separate inlet and outlet flows;
- catalyst feed, deactivation and washout;
- discrete exchange events.

The capacity of a channel is in M/s at the reference amount of its catalyst, full
activity and the initial volume. Three ratios scale it: the rate factors, the active
amount over the reference amount, and the initial volume over the current volume.

Every recorded state in the stored record has the ledger of each species: the balance of
its initial, fed, formed, consumed, removed and final amounts. A state fails, and no
trajectory is returned, in two cases:
- the residual of the ledger of a species exceeds 1e-9 times the larger of 1 M and the
  largest concentration expected from the ledger;
- the residual of the ledger of a catalyst exceeds 1e-9 times its `reference_amount`.

A definition can state its operating conditions (`state`: temperature and pH). It can
declare that the capacity of a channel depends on temperature (an Arrhenius factor) or on
pH (a bell-shaped factor, `1 / (1 + 10^(pKa1 - pH) + 10^(pH - pKa2))` divided by the same
factor at the reference pH). It can declare that the deactivation of a catalyst depends
on temperature (an Arrhenius factor). Each factor is relative to the conditions at which
the value was measured. The capacities and deactivation rates are scaled before the
integration, and the result holds each multiplier.

Limits.
- The model is a well-mixed reactor at constant temperature and pH, with the supplied
  rate laws, flows and events.
- The values of the rate laws are supplied in the definition or set by `operation.fit`
  or `operation.define`.
- A channel depends on temperature or pH only if the definition declares a dependence. A
  channel with no declared dependence is the same at every stated temperature and pH.
- A declared dependence comes from a measured series. The data constrain it only within
  the conditions of that series, and it is applied at any stated condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alambiq.capabilities.operate.simulate_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

Solver = Literal["Radau", "DOP853"]


@dataclass(frozen=True, slots=True)
class SimulateRequest:
    """The operating definition and the solver.

    definition:
        Path to the operating definition, whose format is described in
        `docs/operation-formats.md`. A fitted definition from `operation.fit` can be
        bound here.
    solver:
        The integration method. `Radau` (the default) is an implicit method for stiff
        systems. `DOP853` is an explicit method for non-stiff systems.
    """

    definition: str
    solver: Solver = "Radau"


@dataclass(frozen=True, slots=True)
class ConditionMultiplier:
    """The factor by which the stated conditions scaled one rate parameter."""

    kind: str
    """`temperature`, `pH` or `deactivation temperature`."""
    target: str
    """The channel or catalyst to which the multiplier applied."""
    multiplier: float


@dataclass(frozen=True, slots=True)
class SimulateResult:
    """The trajectory summary and the maximum mass-balance error."""

    operation_id: str
    species: list[str]
    duration_s: float
    final_concentrations_M: dict[str, float]
    maximum_balance_error_M: float
    recorded_states: int
    events: int
    record: str
    """Path to the stored trajectory, with its ledgers and the solver log."""
    condition_multipliers: list[ConditionMultiplier] = field(default_factory=list)
    """Every scaling applied by the stated temperature and pH; empty without a `state`."""


@capability(
    "operation.simulate",
    version="1.0",
    family="operate",
    summary="Integrate a reaction under stated operating conditions",
    render=render,
)
def simulate(request: SimulateRequest, ctx: StepContext) -> SimulateResult:
    """Integrate the supplied operating definition and report the trajectory."""
    from alambiq.capabilities.operate.science.simulation import simulate as integrate

    record = integrate(ctx.read_json(request.definition), method=request.solver)

    trajectory = record["trajectory"]
    final = trajectory[-1]
    error = max(state["maximum_balance_error_M"] for state in trajectory)
    return SimulateResult(
        operation_id=record["request"]["id"],
        species=[s["id"] for s in record["request"]["species"]],
        duration_s=float(final["time_s"]),
        final_concentrations_M=dict(final["concentrations_M"]),
        maximum_balance_error_M=float(error),
        recorded_states=len(trajectory),
        events=len(record["events"]),
        record=ctx.keep_json(record),
        condition_multipliers=[
            ConditionMultiplier(
                kind=m["kind"],
                target=m.get("channel") or m.get("catalyst"),
                multiplier=m["multiplier"],
            )
            for m in (record.get("conditions") or {}).get("multipliers", [])
        ],
    )
