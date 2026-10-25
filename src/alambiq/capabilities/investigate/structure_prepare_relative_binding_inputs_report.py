"""How prepared relative-binding inputs appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def render(result) -> str:
    ligands = table(
        [
            "Ligand",
            "SMILES",
            "Atoms",
            "Charge sum (e)",
            "Pose",
            "Clash score",
            "Nearest (Å)",
        ],
        [
            [
                lig.id,
                lig.smiles,
                lig.atoms,
                lig.charge_sum_e,
                "anchor"
                if lig.selected_pose_index is None
                else lig.selected_pose_index,
                ""
                if lig.overlap_penalty_angstrom2 is None
                else lig.overlap_penalty_angstrom2,
                ""
                if lig.minimum_substituent_distance_angstrom is None
                else lig.minimum_substituent_distance_angstrom,
            ]
            for lig in result.ligands
        ],
        numeric={2, 3, 5, 6},
    )
    edges = table(
        ["Edge", "A to B", "Mapped atoms", "RMS mapped distance (Å)"],
        [
            [e.id, f"{e.a} to {e.b}", e.mapped_atoms, e.rms_mapped_distance_angstrom]
            for e in result.edges
        ],
        numeric={2, 3},
    )
    return "".join(
        [
            structure_file(result.protein_pdb, caption="The prepared protein chain."),
            f"<p>Protein of {result.protein_residues} residues "
            f"({result.deposited_residues} deposited), hydrogens at pH {result.ph:g}.</p>",
            ligands,
            edges,
            f'<p class="sub">Versions: '
            f"{esc(', '.join(f'{k} {v}' for k, v in sorted(result.versions.items())[:4]))}.</p>",
        ]
    )
