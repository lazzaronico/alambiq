"""How a ligand pose measurement appears in the dossier."""

from __future__ import annotations

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, table


def _chart(result) -> str:
    if len(result.plot_rmsd_angstrom) < 2:
        return ""
    if result.plot_time_ps is not None:
        x, label = result.plot_time_ps, "time (ps)"
    else:
        x, label = result.plot_frame, "frame"
    return line_chart(
        [
            Series("RMSD to the reference pose", x, result.plot_rmsd_angstrom),
            Series(
                "Centroid shift",
                x,
                result.plot_centroid_shift_angstrom,
                color="#999999",
                dashed=True,
            ),
        ],
        x_label=label,
        y_label="distance (Å)",
        title="Ligand pose against the reference",
        height=300,
    )


def _beyond(result) -> str:
    frame = result.first_frame_beyond_threshold
    if frame is None:
        return "no frame exceeds it"
    return f"the first frame above it is frame {frame}"


def render(result) -> str:
    stats = result.rmsd
    shift = result.centroid_shift
    rows = [
        ["RMSD (Å)", stats.first, stats.last, stats.mean, stats.median, stats.maximum],
        ["Centroid shift (Å)", shift.first, shift.last, shift.mean, shift.median, shift.maximum],
    ]
    return "".join(
        [
            f"<p>{esc(result.ligand)} against {esc(result.reference_ligand)} in the "
            f"reference: {result.heavy_atoms} heavy atoms, {result.pairings} "
            f"symmetry-equivalent pairing(s), {result.fit_atoms} alpha carbons fitted "
            f"within {result.fit_cutoff_angstrom:g} Å of the ligand "
            f"(mean fit RMSD {result.fit_rmsd_mean_angstrom:.2f} Å).</p>",
            f"<p>{result.frames_within_threshold} of {result.frames} frames "
            f"({result.fraction_within_threshold:.0%}) have an RMSD of at most "
            f"{result.threshold_angstrom:g} Å; {_beyond(result)}.</p>",
            _chart(result),
            table(
                ["Measure", "First", "Last", "Mean", "Median", "Maximum"],
                rows,
                numeric={1, 2, 3, 4, 5},
            ),
            f'<p class="sub">MDTraj {esc(result.mdtraj_version)}, RDKit '
            f"{esc(result.rdkit_version)}; imaging {esc(result.imaging)}.</p>",
        ]
    )
