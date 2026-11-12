"""How a ligand-site measurement appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def render(result) -> str:
    fit = result.fit
    ligand = result.ligand
    pairing = ", ".join(f"{m}→{r}" for m, r in fit.chain_assignment.items())
    runner_up = (
        f" {fit.equivalent_fits} of {fit.assignments_tested} assignments of the chains "
        "fit within the tolerance of the smallest RMSD."
        if fit.assignments_tested > 1
        else ""
    )
    parts = [
        f"<p>{fit.fitted_residues} alpha carbons fitted, RMSD "
        f"<strong>{fit.rmsd_angstrom:.3f} Å</strong>; chains (mobile→reference): "
        f"{esc(pairing)}. The fit covers {fit.mobile_coverage:.0%} of the mobile and "
        f"{fit.reference_coverage:.0%} of the reference alpha carbons.{runner_up}</p>",
        f"<p>Ligand {esc(ligand.residue_name)} {ligand.residue_number} in chain "
        f"{esc(ligand.chain_id)}, {len(ligand.atoms)} heavy atoms.</p>",
        structure_file(
            result.fitted_structure,
            caption="The mobile structure after superposition onto the reference.",
        ),
    ]
    for m in result.measurements:
        subject = (
            f"atom {esc(m.ligand_atom_requested)}"
            if m.ligand_atom_requested
            else "the nearest ligand atom"
        )
        parts.append(
            f"<h4>{esc(m.label)}</h4><p>Distance of {subject} to {esc(m.atom)} of "
            f"{esc(m.residue_name)}: <strong>{m.nearest.distance_angstrom:.2f} Å</strong> "
            f"({esc(m.nearest.ligand_atom)} to chain {esc(m.nearest.chain_id)} "
            f"{m.nearest.residue_number}). Over the {fit.equivalent_fits} equivalent fits the "
            f"nearest distance ranges from {m.nearest_range_angstrom[0]:.2f} to "
            f"{m.nearest_range_angstrom[1]:.2f} Å.</p>"
        )
        parts.append(
            table(
                ["Chain", "Residue", "Ligand atom", "Distance (Å)"],
                [
                    [i.chain_id, i.residue_number, i.ligand_atom, i.distance_angstrom]
                    for i in m.instances
                ],
                numeric={1, 3},
            )
        )
    parts.append("<p>Ligand atoms and the atoms they are bonded to:</p>")
    parts.append(
        table(
            ["Atom", "Element", "Bonded to"],
            [[a.name, a.element, " ".join(a.bonded_to)] for a in ligand.atoms],
        )
    )
    parts.append(f'<p class="sub">gemmi {esc(result.gemmi_version)}</p>')
    return "".join(parts)
