"""How panel hits appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import number, table


def render(result) -> str:
    rows = [
        [h.target, h.evalue, h.identity, h.coverage, h.bits, h.alignment_length]
        for h in result.hits
    ]
    thresholds = " · ".join(
        f"{k.replace('_', ' ')} {number(v)}" for k, v in result.thresholds.items()
    )
    return "".join(
        [
            table(
                [
                    "Target",
                    "E-value",
                    "Identity",
                    "Coverage",
                    "Bits",
                    "Alignment length (residues)",
                ],
                rows,
                numeric={1, 2, 3, 4, 5},
            ),
            '<p class="sub">'
            f"{len(result.hits)} of {result.considered} hits reported from a panel of "
            f"{result.panel_size} proteins · {thresholds}</p>",
        ]
    )
