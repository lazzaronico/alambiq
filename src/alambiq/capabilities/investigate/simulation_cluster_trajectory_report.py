"""How a trajectory clustering appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def render(result) -> str:
    parts = [
        f"<p>{result.frames_read} of {result.frames_in_file} frames compared on "
        f"{result.atoms_compared} atoms with an RMSD cutoff of {result.rmsd_cutoff_angstrom:g} Å: "
        f"<strong>{result.cluster_count} clusters</strong>, {result.singleton_count} of 1 frame.</p>",
        table(
            ["Cluster", "Frames", "Share", "Center frame", "Mean RMSD (Å)", "Max RMSD (Å)"],
            [
                [
                    c.index,
                    c.size,
                    f"{c.fraction:.1%}",
                    c.center_frame,
                    c.mean_rmsd_to_center_angstrom,
                    c.max_rmsd_to_center_angstrom,
                ]
                for c in result.clusters
            ],
            numeric={0, 1, 3, 4, 5},
        ),
    ]
    first = next((c for c in result.clusters if c.structure), None)
    if first is not None:
        parts.append(
            structure_file(
                first.structure,
                caption=f"The center of the largest cluster (frame {first.center_frame}).",
            )
        )
    parts.append(f'<p class="sub">{esc(result.mdtraj_version)}</p>')
    return "".join(parts)
