"""How a scalar-coupling comparison appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    rows = [
        [
            c.id,
            c.measured_hz,
            c.reported_error_hz,
            c.calculated_hz,
            c.replica_standard_error_hz,
            c.residual_hz,
            ""
            if c.residual_over_reported_error is None
            else c.residual_over_reported_error,
            "" if c.split_rhat is None else c.split_rhat,
            "" if c.effective_samples is None else c.effective_samples,
            c.sampling or "",
        ]
        for c in result.couplings
    ]
    status = (
        ""
        if result.sampling_status is None
        else f" The sampling limits were judged {result.sampling_status}."
    )
    swapped = (
        f" {result.swapped_beta_hydrogens} residue(s) had HB2 and HB3 exchanged."
        if result.swapped_beta_hydrogens
        else ""
    )
    return "".join(
        [
            f"<p>{len(result.couplings)} couplings over {len(result.replicas)} replicas "
            f"({esc(', '.join(result.replicas))}), {result.frames_per_replica} frames each, "
            f"sampled every {result.sampling_interval_ps:g} ps. RMSE {result.rmse_hz:.3f} Hz, "
            f"mean absolute error {result.mae_hz:.3f} Hz. Sampled at "
            f"{result.mean_sampled_temperature_kelvin:.1f} K, measured at "
            f"{result.measured_temperature_kelvin:g} K; atom naming "
            f"{esc(result.atom_naming)}.{esc(swapped)}{esc(status)}</p>",
            table(
                [
                    "Coupling",
                    "Measured (Hz)",
                    "Reported error (Hz)",
                    "Calculated (Hz)",
                    "SE (replicas)",
                    "Residual (Hz)",
                    "Residual / error",
                    "Split R-hat",
                    "Effective n",
                    "Sampling",
                ],
                rows,
                numeric={1, 2, 3, 4, 5, 6, 7, 8},
            ),
        ]
    )
