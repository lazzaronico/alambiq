"""How a prepared protein-ligand system appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def render(result) -> str:
    composition = [
        ["Atoms", result.atom_count],
        ["Solute atoms (before solvent)", result.solute_atoms],
        ["Water molecules", result.water_molecules],
        ["Sodium / chloride ions", f"{result.sodium_ions} / {result.chloride_ions}"],
        ["Net charge (e)", result.net_charge_e],
        [
            "Box volume (nm³)",
            "" if result.box_volume_nm3 is None else result.box_volume_nm3,
        ],
    ]
    relaxation = [
        ["Observed heavy atoms held", result.observed_heavy_atoms_held],
        ["Modeled heavy atoms relaxed", result.modeled_heavy_atoms_relaxed],
        [
            "Largest modeled displacement (Å)",
            result.maximum_modeled_displacement_angstrom,
        ],
        ["Vacuum energy before (kJ/mol)", result.vacuum_energy_before_kj_mol],
        ["Vacuum energy after (kJ/mol)", result.vacuum_energy_after_kj_mol],
        ["Solvent stage energy before (kJ/mol)", result.solvent_energy_before_kj_mol],
        ["Solvent stage energy after (kJ/mol)", result.solvent_energy_after_kj_mol],
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
    ligands = [
        [
            s.chain,
            s.residue_name,
            s.residue_number,
            f"{s.first_atom}–{s.first_atom + s.atom_count - 1}",
            s.formal_charge,
            s.charge_sum_e,
        ]
        for s in result.ligand_sites
    ]
    field = result.force_field
    return "".join(
        [
            structure_file(
                result.topology_pdb,
                solute_only=True,
                caption="Solute only; ligands are drawn as sticks.",
            ),
            f"<p>Protein {esc(field['protein'])}, water {esc(field['water'])}, ligand "
            f"{esc(field['ligand'])} with {esc(field['charges'])} charges.</p>",
            table(["Composition", "Value"], composition, numeric={1}),
            table(
                [
                    "Ligand chain",
                    "Residue",
                    "Residue number",
                    "Atoms",
                    "Charge (e)",
                    "AshGC sum (e)",
                ],
                ligands,
                numeric={2, 4, 5},
            ),
            table(["Relaxation", "Value"], relaxation, numeric={1}),
        ]
    )
