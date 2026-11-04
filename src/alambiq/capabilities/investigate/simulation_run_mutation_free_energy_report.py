"""How one leg of a substitution free energy appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    sd = "" if result.replica_sd_kcal_mol is None else f", replica spread {result.replica_sd_kcal_mol:.2f}"
    return (
        f"<p>{esc(result.leg)} leg, {esc(result.mutation)}: mean "
        f"<strong>{result.mean_delta_g_kcal_mol:.2f} kcal/mol</strong>{sd} over "
        f"{len(result.repeats)} replica(s), {esc(result.platform)} ({esc(result.precision)}).</p>"
        + table(
            ["Replica", "ΔG (kcal/mol)", "MBAR error", "Minimum overlap", "Minimum effective samples", "Sampling (s)"],
            [
                [
                    r.id,
                    r.delta_g_kcal_mol,
                    r.mbar_error_kcal_mol,
                    r.minimum_adjacent_overlap,
                    r.minimum_effective_samples,
                    r.sampling_seconds,
                ]
                for r in result.repeats
            ],
            numeric={1, 2, 3, 4, 5},
        )
    )
