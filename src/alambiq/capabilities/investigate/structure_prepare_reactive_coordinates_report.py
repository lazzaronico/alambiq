"""How a reactive-coordinate model appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def render(result) -> str:
    termini = table(
        [
            "Chain",
            "Terminus",
            "Modeled residues",
            "Donor",
            "Fit RMSD (Å)",
            "Peptide bond (Å)",
            "Donors considered (count)",
        ],
        [
            [
                t.target_chain,
                t.terminus,
                ", ".join(map(str, t.modeled_residues)),
                t.donor_chain,
                t.backbone_fit_rmsd_angstrom,
                t.junction_bond_angstrom,
                t.candidates_considered,
            ]
            for t in result.modeled_termini
        ],
        numeric={4, 5, 6},
    )
    substrates = table(
        [
            "Chain",
            "Heavy atoms",
            "Atoms",
            "Analogue RMSD (Å)",
            "Overlap score",
            "SMILES",
            "Charge",
        ],
        [
            [
                s.target_chain,
                s.heavy_atoms,
                s.atoms,
                s.analogue_rmsd_angstrom,
                s.protein_overlap_score_angstrom2,
                s.canonical_smiles,
                s.formal_charge,
            ]
            for s in result.substrates
        ],
        numeric={1, 2, 3, 4, 6},
    )
    return "".join(
        [
            structure_file(
                result.display_structure,
                "cif",
                caption="The modeled protein with its positioned substrate.",
            ),
            f"<p>{result.protein_residues} residues, {result.protein_heavy_atoms} protein heavy "
            f"atoms, {result.water_count} waters retained and {result.removed_waters} removed.</p>",
            termini if result.modeled_termini else "<p>No terminus is modeled.</p>",
            substrates,
            f'<p class="sub">Gemmi {esc(result.versions.get("gemmi", ""))}, RDKit '
            f"{esc(result.versions.get('rdkit', ''))}.</p>",
        ]
    )
