"""How a trajectory inspection appears in the dossier."""

from __future__ import annotations

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, table


def _series(times, values, label: str, unit: str) -> str:
    """One quantity against time, with labeled axes."""
    if len(times) < 2:
        return ""
    return line_chart(
        [Series(label, times, values)],
        x_label="time (ps)",
        y_label=f"{label} ({unit})",
        title=label,
        height=260,
    )


def render(result) -> str:
    rows = [
        [
            o.id,
            o.unit,
            o.mean,
            o.standard_deviation,
            o.minimum,
            o.maximum,
            o.first,
            o.last,
        ]
        for o in result.observables
    ]
    parts = [
        f"<p>{result.frames} frames of {result.atoms} atoms, {result.first_time_ps:g} to "
        f"{result.last_time_ps:g} ps; fitted on {result.fit_atoms} alpha carbons to the "
        f"{esc(result.reference)}; imaging {esc(result.imaging)}; MDTraj "
        f"{esc(result.mdtraj_version)}.</p>",
        _series(
            result.plot_time_ps,
            result.plot_rmsd_angstrom,
            "Alpha-carbon RMSD against time",
            "Å RMSD",
        ),
        _series(
            result.plot_time_ps,
            result.plot_radius_gyration_angstrom,
            "Alpha-carbon radius of gyration against time",
            "Å Rg",
        ),
        table(
            ["Column", "Unit", "Mean", "SD", "Min", "Max", "First", "Last"],
            rows,
            numeric={2, 3, 4, 5, 6, 7},
        ),
        f"<p>Alpha-carbon fluctuation: mean {result.rmsf_mean_angstrom:.2f} Å, largest "
        f"{result.rmsf_max_angstrom:.2f} Å at {esc(result.rmsf_max_residue)}. State averages: "
        f"{result.mean_temperature_kelvin:.1f} K, potential energy "
        f"{result.mean_potential_kj_mol:.1f} kJ/mol"
        + (
            f", volume {result.mean_volume_nm3:.2f} nm³"
            if result.mean_volume_nm3 is not None
            else ""
        )
        + ".</p>",
    ]
    if result.ligand_contacts is not None:
        parts.append(
            table(
                ["Residue", "Chain", "Contact fraction", "Closest (Å)"],
                [
                    [
                        f"{c.residue_name}{c.residue_number}",
                        c.chain_id,
                        c.contact_fraction,
                        c.minimum_distance_angstrom,
                    ]
                    for c in result.ligand_contacts
                ],
                numeric={2, 3},
            )
        )
    if result.image_check is not None:
        c = result.image_check
        verdict = "was beyond the cutoff from" if c.passed else "came within the cutoff of"
        parts.append(
            f"<p>The protein {verdict} its periodic image (cutoff {c.cutoff_nm} nm) in "
            f"{c.checked_frames} of {c.total_frames} frames checked; closest approach "
            f"{c.minimum_distance_nm:.3f} nm.</p>"
        )
    return "".join(parts)
