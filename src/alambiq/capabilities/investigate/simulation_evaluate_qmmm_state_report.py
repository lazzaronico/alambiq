"""How a QM/MM state evaluation appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    energies = table(
        ["Component", "Energy (kJ/mol)"],
        [
            [
                "Electronic (PBE/6-31G**, in the classical field)",
                result.electronic_energy_kj_mol,
            ],
            [
                "Classical (MM and QM–MM Lennard-Jones, bonded)",
                result.classical_energy_kj_mol,
            ],
            ["External potential", result.external_energy_kj_mol],
            ["Total", result.total_energy_kj_mol],
        ],
        numeric={1},
    )
    quality = table(
        ["Quantity", "Value"],
        [
            [
                "Atoms (physical / electronic)",
                f"{result.physical_atoms} / {result.electronic_atoms}",
            ],
            ["Basis functions", result.basis_functions],
            ["SCF cycles", result.scf_cycles],
            ["Converged", "yes" if result.converged else "no"],
            ["Orbital gradient norm", result.orbital_gradient_norm],
            [
                "Electrons (found / expected)",
                f"{result.electrons_found:.8f} / {result.electrons_expected}",
            ],
            [
                "Largest atomic force (kJ/mol/nm)",
                result.maximum_physical_force_kj_mol_nm,
            ],
            [
                "Net force (kJ/mol/nm)",
                ", ".join(f"{x:.2e}" for x in result.net_force_kj_mol_nm),
            ],
        ],
        numeric={1},
    )
    return "".join(
        [
            f"<p>PySCF {esc(result.pyscf_version)}; {result.seconds:.1f} s.</p>",
            energies,
            quality,
        ]
    )
