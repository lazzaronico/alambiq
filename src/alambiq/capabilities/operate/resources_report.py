"""How a resource comparison appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table


def render(result) -> str:
    unit = esc(result.cost_unit)
    parts = [
        table(
            [
                "Alternative",
                "Useful product (mmol, base)",
                f"Cost per useful mmol, low ({unit})",
                "Base",
                "High",
            ],
            [
                [
                    a.id,
                    a.useful_product_mmol.base,
                    a.cost_per_useful_mmol.low,
                    a.cost_per_useful_mmol.base,
                    a.cost_per_useful_mmol.high,
                ]
                for a in result.alternatives
            ],
            numeric={1, 2, 3, 4},
        )
    ]
    for p in result.pairs:
        d = p.difference_per_useful_mmol
        crossings = "; ".join(
            f"{number(c.price)} for {esc(c.resource)}"
            + ("" if c.inside_declared_range else " (outside its declared range)")
            for c in p.crossovers
            if c.price is not None
        )
        parts.append(
            f'<p class="sub">{esc(p.left)} minus {esc(p.right)}: '
            f"{number(d.low)} to {number(d.high)}, base {number(d.base)} {unit} per "
            f"useful mmol. "
            + (
                f"The costs are equal at a price of {crossings}, with all other "
                "inputs at their base values."
                if crossings
                else "No change of a single price equalizes the costs."
            )
            + "</p>"
        )
    return "".join(parts)
