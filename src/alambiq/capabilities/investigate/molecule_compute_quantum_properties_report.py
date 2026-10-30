"""How quantum-chemical properties of a molecule appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table


def render(result) -> str:
    state = f"charge {result.charge}, multiplicity {result.multiplicity}"
    medium = "in water (IEF-PCM)" if result.solvent == "water" else "in the gas phase"
    parts = [
        f"<p>{esc(result.formula)} ({esc(result.canonical_smiles)}), {state}, "
        f"{result.heavy_atoms} heavy atoms. Optimized {medium} with "
        f"{esc(result.functional)}/{esc(result.basis)} in {result.optimization_steps} steps; "
        f"energy {number(result.energy_hartree, 8)} hartree.</p>"
    ]
    if result.solvent_polarization_energy_kcal_mol is not None:
        parts.append(
            f"<p>The continuum lowers the energy by "
            f"{number(-result.solvent_polarization_energy_kcal_mol, 4)} kcal/mol at this "
            "geometry.</p>"
        )
    if result.spin_squared is not None:
        parts.append(
            f"<p>S squared is {number(result.spin_squared, 4)} against "
            f"{number(result.expected_spin_squared, 4)} for a pure spin state.</p>"
        )
    if result.orbitals:
        o = result.orbitals
        parts.append(
            table(
                ["Orbital", "Energy (eV)"],
                [
                    ["Highest occupied", o.homo_ev],
                    ["Lowest unoccupied", o.lumo_ev],
                    ["Gap", o.gap_ev],
                ],
                numeric={1},
            )
        )
    if result.charges:
        c = result.charges
        spin = any(a.spin_population is not None for a in c.mulliken)
        headers = ["Atom", "Element", "Mulliken charge"] + (
            ["Spin population"] if spin else []
        )
        rows = [
            [a.atom, a.element, a.charge] + ([a.spin_population] if spin else [])
            for a in c.mulliken
        ]
        parts.append(
            table(headers, rows, numeric=set(range(len(headers))) - {1})
            + f'<p class="sub">Dipole moment {number(c.dipole_debye, 4)} D, about the '
            f"{esc(c.dipole_origin)}.</p>"
        )
    if result.thermochemistry:
        t = result.thermochemistry
        parts.append(
            table(
                ["Quantity", "Value", "Unit"],
                [
                    ["Zero-point energy", t.zero_point_energy_kcal_mol, "kcal/mol"],
                    ["Enthalpy", t.enthalpy_hartree, "hartree"],
                    ["Gibbs energy", t.gibbs_energy_hartree, "hartree"],
                    ["Entropy", t.entropy_cal_mol_k, "cal/(mol K)"],
                    ["Heat capacity at constant volume", t.cv_cal_mol_k, "cal/(mol K)"],
                    [
                        "Heat capacity at constant pressure",
                        t.cp_cal_mol_k,
                        "cal/(mol K)",
                    ],
                    ["Imaginary frequencies", t.imaginary_modes, ""],
                ],
                numeric={1},
            )
            + f'<p class="sub">At {number(t.temperature_k)} K and {number(t.pressure_pa)} Pa; '
            f"symmetry number {t.symmetry_number}.</p>"
        )
    if result.excitations:
        parts.append(
            table(
                ["State", "Energy (eV)", "Wavelength (nm)", "Oscillator strength"],
                [
                    [e.state, e.energy_ev, e.wavelength_nm, e.oscillator_strength]
                    for e in result.excitations
                ],
                numeric={0, 1, 2, 3},
            )
        )
    return "".join(parts)
