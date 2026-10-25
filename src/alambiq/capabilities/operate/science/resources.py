"""Selected-input accounting on an explicit useful-product amount basis."""

import copy
import itertools
import math

from alambiq.capabilities.operate.science.simulation import (
    _fields,
    _number,
    _records,
)
from alambiq.capabilities.shared.records import digest, label

PROFILE = "explicit-resource-comparison-v1"
UNITS = {
    "mol": ("amount", 1000),
    "mmol": ("amount", 1),
    "umol": ("amount", 0.001),
    "kg": ("mass", 1000),
    "g": ("mass", 1),
    "mg": ("mass", 0.001),
    "L": ("volume", 1),
    "mL": ("volume", 0.001),
    "uL": ("volume", 0.000001),
    "h": ("time", 1),
    "min": ("time", 1 / 60),
    "s": ("time", 1 / 3600),
    "kWh": ("energy", 1),
    "Wh": ("energy", 0.001),
    "kJ": ("energy", 1 / 3600),
}
LEVELS = ("low", "base", "high")


def _range(value, *, fraction=False):
    _fields(value, {*LEVELS, "source"})
    label(value["source"], "range source", 3000)
    numbers = [_number(value[k], k, fraction=fraction) for k in LEVELS]
    if numbers != sorted(numbers):
        raise ValueError("Each range must satisfy low <= base <= high.")


def _quantity(value, unit, *, positive=False):
    _fields(value, {"value", "unit", "source"})
    label(value["source"], "quantity source", 3000)
    source_unit = value["unit"]
    if source_unit not in UNITS or UNITS[source_unit][0] != UNITS[unit][0]:
        raise ValueError(
            f"A quantity must have a unit of the same dimension as {unit}; found "
            f"{source_unit!r}."
        )
    number = _number(value["value"], "quantity", positive=positive)
    result = number * UNITS[source_unit][1] / UNITS[unit][1]
    _number(result, "converted quantity", positive=positive or number > 0)
    return result


def validate_request(request):
    _fields(
        request, {"id", "product", "cost_unit", "scope", "resources", "alternatives"}
    )
    for key in ("id", "cost_unit"):
        label(request[key], key)
    for key in ("product", "scope"):
        label(request[key], key, 3000)
    resources = _records(request["resources"], "resources", 64)
    _records(request["alternatives"], "alternatives", 20)
    for r in request["resources"]:
        _fields(r, {"id", "unit", "price"})
        if r["unit"] not in UNITS:
            raise ValueError(
                f"The unit of the consumable {r['id']} must be a supported unit of "
                f"amount, mass, volume, energy or time; found {r['unit']!r}."
            )
        _range(r["price"])
    for a in request["alternatives"]:
        _fields(
            a,
            {"id", "basis", "product", "recovery", "accepted", "duration", "resources"},
        )
        label(a["basis"], "batch basis", 3000)
        _quantity(a["product"], "mmol")
        _quantity(a["duration"], "h", positive=True)
        _range(a["recovery"], fraction=True)
        _range(a["accepted"], fraction=True)
        if not isinstance(a["resources"], dict) or set(a["resources"]) != resources:
            raise ValueError(
                f"Alternative {a['id']} must give an amount for every consumable in "
                "resources, including explicit zeros, and for no other."
            )
        for r in request["resources"]:
            _quantity(a["resources"][r["id"]], r["unit"])


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _difference(a, b, prices, *, low=False, high=False):
    # Both alternatives see the same resource prices. Recovery ranges are independent.
    ua = a["useful_product_mmol"]["high" if low else "low" if high else "base"]
    ub = b["useful_product_mmol"]["low" if low else "high" if high else "base"]
    if ua == 0 or ub == 0:
        return None
    coefficients = [
        x["amount"] / ua - y["amount"] / ub
        for x, y in zip(a["resources"], b["resources"], strict=True)
    ]
    return math.fsum(
        c
        * p[
            "low"
            if (low and c >= 0) or (high and c < 0)
            else "high"
            if low or high
            else "base"
        ]
        for c, p in zip(coefficients, prices, strict=True)
    )


def compare_resources(request):
    validate_request(request)
    resources = request["resources"]
    prices = [r["price"] for r in resources]
    alternatives = []
    for a in request["alternatives"]:
        product = _quantity(a["product"], "mmol")
        duration = _quantity(a["duration"], "h", positive=True)
        useful = {k: product * a["recovery"][k] * a["accepted"][k] for k in LEVELS}
        for k in LEVELS:
            _number(
                useful[k],
                "useful product",
                positive=all(
                    v > 0 for v in (product, a["recovery"][k], a["accepted"][k])
                ),
            )
        rows = []
        for r in resources:
            amount = _quantity(a["resources"][r["id"]], r["unit"])
            rows.append(
                dict(
                    id=r["id"],
                    unit=r["unit"],
                    amount=amount,
                    base_cost=amount * r["price"]["base"],
                    per_useful_mmol={
                        k: _ratio(amount, useful[v])
                        for k, v in (("low", "high"), ("base", "base"), ("high", "low"))
                    },
                )
            )
        subtotal = {
            k: math.fsum(r["amount"] * p[k] for r, p in zip(rows, prices, strict=True))
            for k in LEVELS
        }
        alternatives.append(
            dict(
                id=a["id"],
                product_mmol=product,
                duration_h=duration,
                useful_product_mmol=useful,
                useful_product_mmol_h={k: v / duration for k, v in useful.items()},
                resources=rows,
                subtotal=subtotal,
                cost_per_mmol={
                    k: _ratio(subtotal[k], useful[v])
                    for k, v in (("low", "high"), ("base", "base"), ("high", "low"))
                },
                diagnostics=[
                    "The range of the useful product includes 0."
                ]
                if useful["low"] == 0
                else [],
            )
        )
    pairs = []
    for a, b in itertools.combinations(alternatives, 2):
        difference = {
            k: _difference(a, b, prices, low=k == "low", high=k == "high")
            for k in LEVELS
        }
        crossovers = []
        if (
            a["useful_product_mmol"]["base"] > 0
            and b["useful_product_mmol"]["base"] > 0
        ):
            slopes = [
                x["per_useful_mmol"]["base"] - y["per_useful_mmol"]["base"]
                for x, y in zip(a["resources"], b["resources"], strict=True)
            ]
            for index, (resource, slope) in enumerate(
                zip(resources, slopes, strict=True)
            ):
                offset = math.fsum(
                    s * p["base"]
                    for i, (s, p) in enumerate(zip(slopes, prices, strict=True))
                    if i != index
                )
                price = -offset / slope if slope else None
                crossovers.append(
                    dict(
                        resource=resource["id"],
                        slope=slope,
                        offset=offset,
                        price=price,
                        inside_declared_range=price is not None
                        and resource["price"]["low"]
                        <= price
                        <= resource["price"]["high"],
                        status="single_price"
                        if slope
                        else "equal_at_all_prices"
                        if offset == 0
                        else "constant_difference",
                    )
                )
        pairs.append(
            dict(
                left=a["id"],
                right=b["id"],
                difference=difference,
                price_crossovers=crossovers,
            )
        )
    result = dict(
        schema_version="0.1",
        profile=PROFILE,
        request=copy.deepcopy(request),
        request_sha256=digest(request),
        alternatives=alternatives,
        pairs=pairs,
        method="The useful product is the declared product amount multiplied by the recovery fraction and the accepted fraction. The subtotal of the listed consumables is the sum of the amount of each consumable multiplied by its price. All alternatives share the price of each consumable. The bounds vary the prices, and the recovery and accepted fractions of each alternative independently, within the declared ranges. A price crossover holds all other prices and the product fractions at their base values. Amounts and durations are fixed. The ranges describe conditional scenarios.",
    )
    try:
        digest(result)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            "The cost arithmetic gives values that are not finite; check the quantities "
            "and the ranges."
        ) from exc
    return result
