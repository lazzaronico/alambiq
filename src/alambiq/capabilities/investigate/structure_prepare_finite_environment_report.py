"""How a finite environment appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def render(result) -> str:
    rows = [
        [
            "Atoms retained / in the periodic source",
            f"{result.atoms} / {result.source_atoms}",
        ],
        [
            "Solute atoms (residues)",
            f"{result.solute_atoms} ({result.solute_residues})",
        ],
        ["Waters and ions retained", result.solvent_and_ion_molecules],
        ["Net charge (e)", result.net_charge_e],
        ["Restrained solute heavy atoms", result.restrained_heavy_atoms],
        ["RMS force on the source positions (kJ/mol/nm)", result.rms_force_kj_mol_nm],
        ["Largest atomic force (kJ/mol/nm)", result.max_force_kj_mol_nm],
        [
            "Inverted alpha carbons",
            ", ".join(map(str, result.inverted_alpha_residues)) or "none",
        ],
        ["Source frame time (ps)", result.source_frame_time_ps],
    ]
    versions = ", ".join(f"{k} {v}" for k, v in sorted(result.versions.items()))
    return "".join(
        [
            structure_file(
                result.topology_pdb, caption="The droplet: solute and retained solvent."
            ),
            table(["Quantity", "Value"], rows, numeric={1}),
            f'<p class="sub">{esc(versions)}</p>',
        ]
    )
