"""How one free-energy leg appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    rows = [
        [
            r.id[:12],
            r.delta_g_kcal_mol,
            r.mbar_error_kcal_mol,
            r.minimum_adjacent_overlap,
            r.minimum_effective_samples,
        ]
        for r in result.repeats
    ]
    spread = (
        "one replica, so no spread"
        if result.replica_sd_kcal_mol is None
        else f"spread {result.replica_sd_kcal_mol:.3f} kcal/mol"
    )
    return "".join(
        [
            f"<p>{esc(result.phase)} leg of edge {esc(result.edge.id)} "
            f"({esc(result.edge.a)} to {esc(result.edge.b)}) on the {esc(result.platform)}, "
            f"OpenFE {esc(result.versions.get('openfe', ''))}; mean "
            f"{result.mean_delta_g_kcal_mol:.3f} kcal/mol, {spread}; "
            f"{result.seconds:.0f} s.</p>",
            table(
                [
                    "Replica",
                    "ΔG (kcal/mol)",
                    "MBAR error",
                    "Min overlap",
                    "Min effective n",
                ],
                rows,
                numeric={1, 2, 3, 4},
            ),
        ]
    )
