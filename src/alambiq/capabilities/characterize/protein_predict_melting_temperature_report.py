"""How melting temperature estimates appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table


def render(result) -> str:
    rows = [
        [
            esc(p.id),
            p.length + 0,
            p.melting_temperature_c,
            p.spread_c,
            *p.replica_values_c,
            "yes" if p.truncated else "",
        ]
        for p in result.proteins
    ]
    parts = [
        f"<p>{len(result.proteins)} proteins with {esc(result.model)} on "
        f"{esc(result.device)} in {number(result.seconds, 3)} s.</p>",
        table(
            [
                "Protein",
                "Length",
                "Tm (°C)",
                "Spread (°C)",
                "Replica 1",
                "Replica 2",
                "Replica 3",
                "Truncated",
            ],
            rows,
            numeric={1, 2, 3, 4, 5, 6},
        ),
    ]
    return "".join(parts)
