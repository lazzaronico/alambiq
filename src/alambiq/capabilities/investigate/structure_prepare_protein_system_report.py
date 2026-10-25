"""How a prepared protein system appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def render(result) -> str:
    waters = result.residue_counts.get("HOH", 0)
    ions = {k: v for k, v in result.residue_counts.items() if k in ("NA", "CL")}
    box = (
        ""
        if result.box_volume_nm3 is None
        else f", box {result.box_volume_nm3:.1f} nm³"
    )
    solvent = (
        f"explicit TIP3P water ({waters} waters, ions {esc(str(ions) or 'none')}{box})"
        if result.solvent == "tip3p"
        else "implicit GBn2 solvent"
    )
    removed = (
        ", ".join(f"{k} {v}" for k, v in result.removed_source_atoms.items()) or "none"
    )
    rows = [
        ["Protein residues", result.protein_residues],
        ["Protein atoms (with hydrogens)", result.protein_atoms],
        ["Atoms in the system", result.atom_count],
        [
            "Potential energy before minimization (kJ/mol)",
            result.potential_energy_before_kj_mol,
        ],
        ["Potential energy after (kJ/mol)", result.potential_energy_after_kj_mol],
        ["RMS force component after (kJ/mol/nm)", result.rms_force_after_kj_mol_nm],
        ["Largest atom force after (kJ/mol/nm)", result.max_force_after_kj_mol_nm],
        ["Heavy atoms added by completion", result.added_heavy_atoms],
        ["Substitutions applied", result.substitutions_applied],
        [
            "Largest peptide planarity deviation (degrees)",
            "not assessed"
            if result.max_peptide_planarity_deviation_degrees is None
            else result.max_peptide_planarity_deviation_degrees,
        ],
        [
            "Residues with inverted alpha chirality",
            ", ".join(map(str, result.inverted_alpha_residues)) or "none",
        ],
    ]
    return "".join(
        [
            structure_file(
                result.topology_pdb,
                solute_only=True,
                caption="Prepared solute without its water and ions.",
            ),
            f"<p>Prepared in {solvent} with {esc(result.versions.get('force_field', 'the force field'))}. "
            f"Source atoms removed: {esc(removed)}.</p>",
            table(["Quantity", "Value"], rows, numeric={1}),
        ]
    )
