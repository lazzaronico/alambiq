"""How the change in folding free energy appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    se = (
        "not available (a leg has one preparation)"
        if result.preparation_standard_error_kcal_mol is None
        else f"{result.preparation_standard_error_kcal_mol:.2f} kcal/mol"
    )
    parts = [
        f"<p>{esc(result.mutation)}: ΔΔG(fold) <strong>{result.ddg_fold_kcal_mol:+.2f} kcal/mol</strong>. "
        f"Standard error from the spread across preparations: {se}; from MBAR: {result.mbar_error_kcal_mol:.2f} kcal/mol. "
        f"{esc(result.convention)}.</p>",
        table(
            [
                "Leg",
                "Preparations",
                "Replicas",
                "Mean ΔG",
                "Preparation sd",
                "Replica sd",
                "MBAR error (rms)",
                "Min overlap",
                "Min effective samples",
                "Forward-reverse gap",
            ],
            [
                [
                    leg.leg,
                    leg.preparations,
                    leg.replicas,
                    leg.mean_delta_g_kcal_mol,
                    leg.preparation_sd_kcal_mol,
                    leg.replica_sd_kcal_mol,
                    leg.mbar_error_rms_kcal_mol,
                    leg.minimum_adjacent_overlap,
                    leg.minimum_effective_samples,
                    leg.forward_reverse_gap_kcal_mol,
                ]
                for leg in result.legs
            ],
            numeric={1, 2, 3, 4, 5, 6, 7, 8, 9},
        ),
    ]
    if result.verdicts:
        parts.append(
            f"<p>Checks: <strong>{esc(result.status)}</strong>.</p>"
            + table(
                ["Diagnostic", "Subject", "Value", "Limit", "Status"],
                [
                    [v.diagnostic, v.subject, v.value, v.limit, v.status]
                    for v in result.verdicts
                ],
                numeric={2, 3},
            )
        )
    return "".join(parts)
