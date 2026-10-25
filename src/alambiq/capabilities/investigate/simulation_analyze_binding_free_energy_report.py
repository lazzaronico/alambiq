"""How a relative binding free-energy analysis appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    edges = table(
        [
            "Edge",
            "A to B",
            "ΔΔG bind (kcal/mol)",
            "Replica SEM",
            "MBAR error",
            "Bound ΔG",
            "Solvent ΔG",
            "Min overlap",
            "Min effective n",
            "Reference ΔΔG",
        ],
        [
            [
                e.id,
                f"{e.a} to {e.b}",
                e.delta_delta_g_kcal_mol,
                e.replica_sem_kcal_mol,
                e.mbar_error_kcal_mol,
                e.bound.mean_kcal_mol,
                e.solvent.mean_kcal_mol,
                min(
                    e.bound.minimum_adjacent_overlap,
                    e.solvent.minimum_adjacent_overlap,
                ),
                min(
                    e.bound.minimum_effective_samples,
                    e.solvent.minimum_effective_samples,
                ),
                ""
                if e.reference_delta_delta_g_kcal_mol is None
                else e.reference_delta_delta_g_kcal_mol,
            ]
            for e in result.edges
        ],
        numeric={2, 3, 4, 5, 6, 7, 8, 9},
    )
    parts = [
        f"<p>{esc(result.title)} at {result.temperature_kelvin:g} K. "
        f"{esc(result.direction)}.</p>",
        edges,
    ]
    if result.cycles:
        parts.append(
            table(
                ["Cycle", "Closure", "Replica SEM", "MBAR error", "Reference closure"],
                [
                    [
                        c.id,
                        c.closure_kcal_mol,
                        c.replica_sem_kcal_mol,
                        c.mbar_error_kcal_mol,
                        ""
                        if c.reference_closure_kcal_mol is None
                        else c.reference_closure_kcal_mol,
                    ]
                    for c in result.cycles
                ],
                numeric={1, 2, 3, 4},
            )
        )
    if result.checks:
        parts.append(
            f"<p>Checks: {esc(result.status)} overall.</p>"
            + table(
                ["Diagnostic", "Where", "Observed", "Limit", "Status"],
                [
                    [
                        c.diagnostic,
                        c.identity,
                        "" if c.observed is None else c.observed,
                        c.limit,
                        c.status + (f" ({c.reason})" if c.reason else ""),
                    ]
                    for c in result.checks
                ],
                numeric={2, 3},
            )
        )
    return "".join(parts)
