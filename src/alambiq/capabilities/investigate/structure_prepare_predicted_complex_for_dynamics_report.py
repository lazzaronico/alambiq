"""How a prepared predicted complex appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def render(result) -> str:
    composition = [
        ["Protein chains", result.protein_chains],
        ["Protein residues", result.protein_residues],
        ["Ligand heavy atoms", result.ligand_heavy_atoms],
        ["Ligand formal charge (e)", result.ligand_formal_charge],
        ["Sum of AshGC charges (e)", result.ligand_charge_sum_e],
        ["Disulfides", ", ".join(result.disulfides) or "none"],
        ["Atoms", result.atom_count],
        ["Water molecules", result.water_molecules],
        ["Sodium / chloride ions", f"{result.sodium_ions} / {result.chloride_ions}"],
        ["Net charge (e)", result.net_charge_e],
        ["Box volume (nm³)", result.box_volume_nm3],
    ]
    energies = [
        ["Potential energy before (kJ/mol)", result.potential_energy_before_kj_mol],
        ["Potential energy after (kJ/mol)", result.potential_energy_after_kj_mol],
        ["RMS force after (kJ/mol/nm)", result.rms_force_after_kj_mol_nm],
        ["Largest force after (kJ/mol/nm)", result.max_force_after_kj_mol_nm],
        [
            "Largest peptide planarity deviation (degrees)",
            ""
            if result.max_peptide_planarity_deviation_degrees is None
            else result.max_peptide_planarity_deviation_degrees,
        ],
        [
            "Inverted alpha carbons",
            ", ".join(map(str, result.inverted_alpha_residues)) or "none",
        ],
    ]
    field = result.force_field
    return "".join(
        [
            structure_file(
                result.topology_pdb,
                solute_only=True,
                caption="Solute only; the ligand is drawn as sticks.",
            ),
            f"<p>Protein {esc(field['protein'])}, water {esc(field['water'])}, ligand "
            f"{esc(field['ligand'])} with {esc(field['charges'])} charges.</p>",
            table(["Composition", "Value"], composition, numeric={1}),
            table(["Minimization", "Value"], energies, numeric={1}),
        ]
    )
