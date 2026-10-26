"""Report the product amounts and the catalyst use of a simulation record.

The capability reads a stored `operation.simulate` record and an explicit assessment. It
returns the net formed, selected and useful amounts of the product, the productivity,
the product per supplied active site of each catalyst, and the time at which each
catalyst falls below its usefulness threshold. The format of the assessment is in
`docs/operation-formats.md`.

Method.
- Product ledger: the initial, fed, formed, consumed, removed and final amounts of the
  product species. The stored record holds the ledger. The result holds the net formed
  amount, formed minus consumed.
- Selected output: the sum of the amounts that the assessment counts (final reactor
  contents, discrete exchanges and continuous collections).
- Useful output: the selected output multiplied by the stated recovery and accepted
  fractions.
- Productivity: the useful output divided by the reaction time, in mol/s. When the
  assessment gives an additional handling time, the useful output is also divided by the
  sum of the reaction time and the handling time.
- Catalyst use, for each catalyst listed in the assessment: the supplied active sites,
  and the net product and the useful output per supplied active site. These need the
  active-site count and, for a catalyst counted in a mass unit, the molecular mass. The
  stored record also holds the extent of each channel of the catalyst and its turnovers
  per supplied active site.
- Usefulness threshold: a minimum active amount of a catalyst, in the amount unit of the
  catalyst. The result is the interval between two recorded states in which the active
  amount first falls below the threshold, or the time of the exchange when the fall
  occurs at a discrete event.

Limits.
- The assessment names the simulated definition (`operation_id`) and its conditions. A
  different definition or different conditions are refused.
- The selected collection windows do not overlap. The selected output does not exceed the
  recorded available amount of the product (final plus removed).
- The recovery fraction and the accepted fraction are stated handling assumptions.
- The turnover per active site is unresolved when any of these holds: the molecular mass
  is missing for a catalyst counted in a mass unit; the amount unit of the catalyst is
  not one of `mol`, `mmol`, `umol`, `nmol`, `kg`, `g`, `mg` or `ug`; the supplied active
  sites number zero. A catalyst without an entry in the assessment has no turnover
  figures.
- The net product of the product species is the sum over channels of the stoichiometric
  coefficient times the extent. It equals the extent of one channel when that channel is
  the only one that changes the species and the coefficient is 1.
- A threshold crossing between two recorded states is located within that interval only.
  For a threshold that is not crossed, the result states whether the active amount is
  below it at time zero or stays at or above it up to the last recorded time.
"""

from __future__ import annotations

from dataclasses import dataclass

from alambiq.capabilities.operate.summarize_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class SummarizeRequest:
    """The simulation record and the assessment.

    operation:
        A stored `operation.simulate` record.
    assessment:
        Path to the assessment. It states:
        - the product species and its recovery and accepted fractions;
        - which outputs count;
        - the active-site count of each catalyst and, for a catalyst counted in a mass
          unit, its molecular mass;
        - the usefulness thresholds;
        - optionally, an additional handling time, which is added to the reaction
          time.
    """

    operation: str
    assessment: str


@dataclass(frozen=True, slots=True)
class CatalystUse:
    """The supplied active sites of one catalyst and the product per active site."""

    id: str
    supplied_active_site_mol: float | None
    product_per_active_site: float | None
    """Net product equivalents per supplied active site."""
    useful_output_per_active_site: float | None


@dataclass(frozen=True, slots=True)
class Lifetime:
    """The interval in which the active amount of one catalyst first falls below a threshold, with the recorded horizon."""

    id: str
    catalyst: str
    status: str
    first_below_interval_s: list[float] | None
    horizon_s: float


@dataclass(frozen=True, slots=True)
class SummarizeResult:
    """The product amounts, the catalyst use and the path of the stored record."""

    product: str
    net_formed_mol: float
    selected_output_mol: float
    useful_output_mol: float
    reaction_time_productivity_mol_s: float | None
    process_time_productivity_mol_s: float | None
    catalysts: list[CatalystUse]
    lifetimes: list[Lifetime]
    record: str
    """Path to the stored record, with every threshold point."""


@capability(
    "operation.summarize",
    version="1.0",
    family="operate",
    summary="Account for product output and catalyst use in a simulation record",
    render=render,
)
def summarize(request: SummarizeRequest, ctx: StepContext) -> SummarizeResult:
    """Compute the product amounts and the catalyst use from the recorded states."""
    from alambiq.capabilities.operate.science.performance import summarize_operation

    record = summarize_operation(
        ctx.read_json(request.operation), ctx.read_json(request.assessment)
    )
    for message in record["diagnostics"]:
        ctx.note(message)
    product = record["product"]
    return SummarizeResult(
        product=product["species"],
        net_formed_mol=product["net_formed_mol"],
        selected_output_mol=product["selected_output_mol"],
        useful_output_mol=product["useful_output_mol"],
        reaction_time_productivity_mol_s=product["reaction_time_productivity_mol_s"],
        process_time_productivity_mol_s=product.get("process_time_productivity_mol_s"),
        catalysts=[
            CatalystUse(
                id=c["id"],
                supplied_active_site_mol=c.get("supplied_active_site_mol"),
                product_per_active_site=c.get(
                    "net_target_equivalents_per_supplied_active_site"
                ),
                useful_output_per_active_site=c.get(
                    "useful_output_per_supplied_active_site"
                ),
            )
            for c in record["catalysts"]
        ],
        lifetimes=[
            Lifetime(
                id=t["id"],
                catalyst=t["catalyst"],
                status=t["status"],
                first_below_interval_s=t["first_below_interval_s"],
                horizon_s=t["horizon_s"],
            )
            for t in record["lifetimes"]
        ],
        record=ctx.keep_json(record),
    )
