"""How a batch balance appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import table


def render(result) -> str:
    rows = [
        [
            s.id,
            s.maximum_accepted_product_g,
            s.required_conversion,
            s.required_vmax,
            s.required_vmax_unit or "—",
            s.activity_multiple,
            s.limitation or ("extrapolated" if s.concentration_extrapolation else ""),
        ]
        for s in result.scenarios
    ]
    return "".join(
        [
            table(
                [
                    "Scenario",
                    "Maximum accepted product (g)",
                    "Conversion needed",
                    "Vmax needed",
                    "Unit",
                    "Multiple of fitted Vmax",
                    "Limitation",
                ],
                rows,
                numeric={1, 2, 3, 5},
            ),
        ]
    )
