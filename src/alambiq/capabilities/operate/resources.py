"""Compare the consumables that each operating alternative needs per useful product.

A consumable is a priced item that an alternative consumes, such as an enzyme, a
substrate or a cofactor. The comparison file lists the consumables under the key
`resources`. Each alternative declares a product amount, a recovery fraction, an
accepted fraction, a duration and an amount of every listed consumable, each
fraction as a low, base and high value. The useful product is the product amount
multiplied by the recovery fraction and the accepted fraction. The cost of the listed
consumables is the sum over the consumables of amount multiplied by the shared price,
reported per useful millimole at the low, base and high levels of the declared price
and fraction ranges. The duration converts the useful product to a rate in mmol/h in
the stored record.

For every pair of alternatives the capability reports the cost difference (first
alternative minus second) per useful millimole as a lower bound, a base value and an
upper bound. The base value uses the base prices and fractions. The bounds are the
minimum and maximum of the difference over the declared price and fraction ranges, with
one price per consumable shared by both alternatives and the fractions of each
alternative varied independently. For each consumable the capability also reports the
price at which the two alternatives cost the same when all other prices and the product
fractions are at their base values. A crossover price is reported only if both
alternatives have a base useful product above 0 and the cost difference depends on that
price.

Limits. The cost covers the consumables listed in the comparison file. The `scope` field of the comparison file is a single-line text of at
most 3000 characters that states which items the list includes. The file lists at
most 64 consumables and at most 20 alternatives. Every alternative declares every
consumable, including explicit zeros, and all alternatives share one price per
consumable and one cost unit. Each range requires low <= base <= high. The low cost of
an alternative uses the lowest price of every consumable and the highest recovery and
accepted fractions; the high cost uses the highest prices and the lowest fractions.
Amounts and durations are fixed. The ranges are conditional scenarios. The format of the comparison file is described in
`docs/operation-formats.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

from alambiq.capabilities.operate.resources_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class ResourcesRequest:
    """The comparison of operating alternatives.

    comparison:
        Path to the comparison file: the product identity, the cost unit, the
        scope, the consumables with their unit and price bounds (key `resources`),
        and the alternatives with product amount, fractions, duration and
        consumable amounts.
    """

    comparison: str


@dataclass(frozen=True, slots=True)
class Range:
    """A low, base and high value."""

    low: float | None
    base: float | None
    high: float | None


@dataclass(frozen=True, slots=True)
class Alternative:
    """One alternative: its useful product and its cost per useful millimole, each as a range."""

    id: str
    useful_product_mmol: Range
    cost_per_useful_mmol: Range


@dataclass(frozen=True, slots=True)
class Crossover:
    """The price of one consumable (field `resource`) at which two alternatives cost the same, and whether it lies in the declared price range."""

    resource: str
    price: float | None
    inside_declared_range: bool


@dataclass(frozen=True, slots=True)
class Pair:
    """The cost difference between two alternatives and the price crossovers per consumable."""

    left: str
    right: str
    difference_per_useful_mmol: Range
    """Left minus right: the lower bound, the base value and the upper bound of the pair difference."""
    crossovers: list[Crossover]


@dataclass(frozen=True, slots=True)
class ResourcesResult:
    """The consumable cost per useful product of each alternative and the pairwise differences."""

    cost_unit: str
    alternatives: list[Alternative]
    pairs: list[Pair]
    record: str
    """Path to the stored record, with the amounts and costs of every consumable."""


def _range(values: dict | None) -> Range:
    values = values or {}
    return Range(values.get("low"), values.get("base"), values.get("high"))


@capability(
    "operation.compare_resources",
    version="1.0",
    family="operate",
    summary="Compare listed consumable costs per useful product across alternatives",
    render=render,
)
def compare_resources(request: ResourcesRequest, ctx: StepContext) -> ResourcesResult:
    """Compute the cost per useful product of each alternative and the pairwise differences."""
    from alambiq.capabilities.operate.science.resources import (
        compare_resources as account,
    )

    comparison = ctx.read_json(request.comparison)
    record = account(comparison)
    for alternative in record["alternatives"]:
        for message in alternative["diagnostics"]:
            ctx.note(f"{alternative['id']}: {message}")
    return ResourcesResult(
        cost_unit=comparison["cost_unit"],
        alternatives=[
            Alternative(
                id=a["id"],
                useful_product_mmol=_range(a["useful_product_mmol"]),
                cost_per_useful_mmol=_range(a["cost_per_mmol"]),
            )
            for a in record["alternatives"]
        ],
        pairs=[
            Pair(
                left=p["left"],
                right=p["right"],
                difference_per_useful_mmol=_range(p["difference"]),
                crossovers=[
                    Crossover(c["resource"], c["price"], c["inside_declared_range"])
                    for c in p["price_crossovers"]
                ],
            )
            for p in record["pairs"]
        ],
        record=ctx.keep_json(record),
    )
