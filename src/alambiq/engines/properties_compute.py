"""Quantum-chemical properties of one small molecule from its SMILES.

Runs inside the molecular-properties environment (PySCF, RDKit, PyBerny). Imports the standard
library and those packages only. The molecule is embedded in 3D, relaxed with a force field,
optimized with density functional theory, and the requested properties are computed at the
optimized geometry. Energies are returned in hartree and the derived values in the units named
in each key.

A closed-shell molecule uses restricted Kohn-Sham theory and an open-shell one (a multiplicity
above 1) uses unrestricted theory. Water is modelled as an IEF-PCM polarizable continuum, which
is a purely electrostatic model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_PER_KELVIN_TO_CAL_MOL_K = HARTREE_TO_KCAL_MOL * 1000.0
HC_EV_NM = 1239.841984332
CONFORMERS = 10
GRID_LEVEL = 4
PRESSURE_PA = 101325.0
WATER_PERMITTIVITY = 78.3553
# An optimized geometry is symmetric to about 1e-3 angstrom; PySCF's default of 1e-5 misses the
# point group of ethylene, benzene and carbon dioxide and so halves their symmetry number.
SYMMETRY_TOLERANCE = 3e-3


def conformer(
    smiles: str, seed: int, request_elements: list[str], max_heavy_atoms: int
) -> dict[str, Any]:
    """The lowest-energy of `CONFORMERS` embedded conformers after a force-field relaxation."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"{smiles!r} is not a SMILES string that RDKit can read")
    unsupported = sorted(
        {a.GetSymbol() for a in molecule.GetAtoms()} - set(request_elements)
    )
    if unsupported:
        raise ValueError(
            f"the elements {', '.join(unsupported)} are not supported; "
            f"supported are {', '.join(request_elements)}"
        )
    if molecule.GetNumHeavyAtoms() > max_heavy_atoms:
        raise ValueError(
            f"the molecule has {molecule.GetNumHeavyAtoms()} heavy atoms; "
            f"at most {max_heavy_atoms} are accepted"
        )
    radicals = sum(a.GetNumRadicalElectrons() for a in molecule.GetAtoms())
    molecule = Chem.AddHs(molecule)
    ids = list(AllChem.EmbedMultipleConfs(molecule, CONFORMERS, randomSeed=seed))
    if not ids:
        raise ValueError(f"RDKit could not embed {smiles!r} in three dimensions")
    outcomes = AllChem.MMFFOptimizeMoleculeConfs(molecule, maxIters=5000)
    best = min(range(len(ids)), key=lambda i: outcomes[i][1])
    positions = molecule.GetConformer(ids[best]).GetPositions()
    atoms = [
        (a.GetSymbol(), tuple(float(x) for x in positions[a.GetIdx()]))
        for a in molecule.GetAtoms()
    ]
    return {
        "atoms": atoms,
        "charge": Chem.GetFormalCharge(molecule),
        "radical_electrons": radicals,
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule)),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "embedded": len(ids),
        "force_field_energy_kcal_mol": float(outcomes[best][1]),
    }


def xyz(atoms: list[tuple[str, tuple[float, float, float]]], comment: str) -> str:
    lines = [str(len(atoms)), comment]
    lines += [f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for s, (x, y, z) in atoms]
    return "\n".join(lines) + "\n"


def check_multiplicity(
    electrons: int, multiplicity: int, radical_electrons: int
) -> None:
    """Refuse a multiplicity that the electron count or the SMILES contradicts."""
    unpaired = multiplicity - 1
    if radical_electrons and not unpaired:
        raise ValueError(
            f"the SMILES has unpaired electrons ({radical_electrons}), so the molecule is "
            f"open-shell; set multiplicity to {radical_electrons + 1} for the lowest-spin state"
        )
    if unpaired > electrons or (electrons - unpaired) % 2:
        raise ValueError(
            f"a multiplicity of {multiplicity} is impossible for {electrons} electrons"
        )


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Compute the requested properties and return them with the optimized geometry."""
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib
    from pyscf.geomopt import berny_solver
    from pyscf.hessian import thermo
    from pyscf.symm import geom

    started = time.time()
    lib.num_threads(int(request["threads"]))
    prepared = conformer(
        request["smiles"],
        int(request["seed"]),
        request["elements"],
        int(request["max_heavy_atoms"]),
    )
    electrons = sum(gto.charge(s) for s, _ in prepared["atoms"]) - prepared["charge"]
    multiplicity = int(request["multiplicity"])
    check_multiplicity(electrons, multiplicity, prepared["radical_electrons"])
    open_shell = multiplicity > 1
    water = request["solvent"] == "water"

    def model(atoms, continuum=water):
        molecule = gto.M(
            atom=atoms,
            basis=request["basis"],
            charge=prepared["charge"],
            spin=multiplicity - 1,
            cart=True,
            verbose=0,
        )
        mean_field = dft.UKS(molecule) if open_shell else dft.RKS(molecule)
        mean_field.xc = request["functional"]
        mean_field.grids.level = GRID_LEVEL
        if continuum:
            mean_field = mean_field.PCM()
            mean_field.with_solvent.method = "IEF-PCM"
            mean_field.with_solvent.eps = WATER_PERMITTIVITY
        return mean_field

    steps: list[int] = []
    converged, relaxed = berny_solver.kernel(
        model(prepared["atoms"]),
        maxsteps=int(request["max_steps"]),
        callback=lambda envs: steps.append(1),
    )
    optimization_seconds = time.time() - started
    if not converged:
        raise RuntimeError(
            f"the geometry optimization did not converge in {request['max_steps']} steps; "
            "no properties are returned for an unconverged geometry"
        )
    coordinates = relaxed.atom_coords(unit="Angstrom")
    atoms = [
        (relaxed.atom_symbol(i), tuple(float(x) for x in coordinates[i]))
        for i in range(relaxed.natm)
    ]
    mean_field = model(atoms)
    energy = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError(
            "the self-consistent field did not converge at the optimized geometry"
        )
    level = f"{request['functional']}/{request['basis']}"
    result: dict[str, Any] = {
        "canonical_smiles": prepared["canonical_smiles"],
        "formula": prepared["formula"],
        "charge": prepared["charge"],
        "multiplicity": multiplicity,
        "solvent": request["solvent"],
        "heavy_atoms": prepared["heavy_atoms"],
        "atoms": relaxed.natm,
        "conformers_embedded": prepared["embedded"],
        "force_field_energy_kcal_mol": prepared["force_field_energy_kcal_mol"],
        "optimization_steps": len(steps),
        "energy_hartree": energy,
        "geometry_xyz": xyz(
            atoms,
            f"{prepared['canonical_smiles']} {level} multiplicity {multiplicity} "
            f"solvent {request['solvent']}",
        ),
        "seconds": {"optimization": optimization_seconds},
        "versions": {"pyscf": pyscf.__version__, "numpy": np.__version__},
    }
    if open_shell:
        spin_squared = float(mean_field.spin_square()[0])
        expected = (multiplicity - 1) / 2 * ((multiplicity - 1) / 2 + 1)
        result["spin_squared"] = spin_squared
        result["expected_spin_squared"] = expected
    if water:
        vacuum = model(atoms, continuum=False)
        vacuum.kernel()
        result["gas_phase_energy_hartree"] = float(vacuum.e_tot)
        result["solvent_polarization_energy_kcal_mol"] = (
            energy - float(vacuum.e_tot)
        ) * HARTREE_TO_KCAL_MOL

    wanted = set(request["properties"])
    if "orbitals" in wanted:
        if open_shell:
            channels = {}
            for name, energies, occupation in zip(
                ("alpha", "beta"), mean_field.mo_energy, mean_field.mo_occ, strict=True
            ):
                occupied, virtual = energies[occupation > 0], energies[occupation == 0]
                channels[name] = (occupied, virtual)
            occupied = np.concatenate([channels[n][0] for n in channels])
            virtual = np.concatenate([channels[n][1] for n in channels])
        else:
            occupied = mean_field.mo_energy[mean_field.mo_occ > 0]
            virtual = mean_field.mo_energy[mean_field.mo_occ == 0]
        homo, lumo = float(occupied.max()), float(virtual.min())
        orbitals = {
            "homo_ev": homo * HARTREE_TO_EV,
            "lumo_ev": lumo * HARTREE_TO_EV,
            "gap_ev": (lumo - homo) * HARTREE_TO_EV,
            "occupied_orbitals": int(occupied.size),
            "highest_occupied_ev": [
                float(x) * HARTREE_TO_EV for x in sorted(occupied)[-5:]
            ],
            "lowest_virtual_ev": [
                float(x) * HARTREE_TO_EV for x in sorted(virtual)[:5]
            ],
        }
        if open_shell:
            for name, (occ, virt) in channels.items():
                orbitals[f"{name}_homo_ev"] = float(occ.max()) * HARTREE_TO_EV
                orbitals[f"{name}_lumo_ev"] = float(virt.min()) * HARTREE_TO_EV
        result["orbitals"] = orbitals
    if "charges" in wanted:
        masses = relaxed.atom_mass_list(isotope_avg=True)
        centre = tuple(float(x) for x in masses @ relaxed.atom_coords() / masses.sum())
        dipole = np.asarray(
            mean_field.dip_moment(unit="Debye", origin=centre, verbose=0), dtype=float
        )
        _, charges = mean_field.mulliken_pop(verbose=0)
        spin = (
            np.asarray(mean_field.mulliken_spin_pop(verbose=0)[1], dtype=float)
            if open_shell
            else None
        )
        result["charges"] = {
            "dipole_debye": float(np.linalg.norm(dipole)),
            "dipole_components_debye": [float(x) for x in dipole],
            "dipole_origin": "center of mass",
            "mulliken": [
                {
                    "atom": i + 1,
                    "element": relaxed.atom_symbol(i),
                    "charge": float(charges[i]),
                    "spin_population": None if spin is None else float(spin[i]),
                }
                for i in range(relaxed.natm)
            ],
        }
    if "thermochemistry" in wanted:
        lap = time.time()
        hessian = mean_field.Hessian().kernel()
        vibrations = thermo.harmonic_analysis(relaxed, hessian)
        default_tolerance = geom.TOLERANCE
        geom.TOLERANCE = SYMMETRY_TOLERANCE
        try:
            values = thermo.thermo(
                mean_field,
                vibrations["freq_au"],
                float(request["temperature_k"]),
                PRESSURE_PA,
            )
        finally:
            geom.TOLERANCE = default_tolerance
        wavenumbers = np.asarray(vibrations["freq_wavenumber"])
        real = (
            wavenumbers.real[wavenumbers.imag == 0]
            if np.iscomplexobj(wavenumbers)
            else wavenumbers
        )
        imaginary = int(vibrations["freq_error"])
        energy_unit = lambda key: float(values[key][0])  # noqa: E731
        per_kelvin = lambda key: float(values[key][0]) * HARTREE_PER_KELVIN_TO_CAL_MOL_K  # noqa: E731
        result["thermochemistry"] = {
            "temperature_k": float(request["temperature_k"]),
            "pressure_pa": PRESSURE_PA,
            "zero_point_energy_hartree": energy_unit("ZPE"),
            "zero_point_energy_kcal_mol": energy_unit("ZPE") * HARTREE_TO_KCAL_MOL,
            "energy_0k_hartree": energy_unit("E_0K"),
            "internal_energy_hartree": energy_unit("E_tot"),
            "enthalpy_hartree": energy_unit("H_tot"),
            "gibbs_energy_hartree": energy_unit("G_tot"),
            "entropy_cal_mol_k": per_kelvin("S_tot"),
            "cv_cal_mol_k": per_kelvin("Cv_tot"),
            "cp_cal_mol_k": per_kelvin("Cp_tot"),
            "symmetry_number": int(values["sym_number"][0]),
            "vibrational_modes": int(real.size),
            "imaginary_modes": imaginary,
            "lowest_frequencies_cm1": [float(x) for x in sorted(real)[:3]],
        }
        result["seconds"]["thermochemistry"] = time.time() - lap
    if "excitations" in wanted:
        lap = time.time()
        excited = mean_field.TDDFT()
        excited.nstates = int(request["excited_states"])
        excited.kernel()
        strengths = excited.oscillator_strength()
        result["excitations"] = [
            {
                "state": i + 1,
                "energy_ev": float(e) * HARTREE_TO_EV,
                "wavelength_nm": HC_EV_NM / (float(e) * HARTREE_TO_EV),
                "oscillator_strength": float(f),
            }
            for i, (e, f) in enumerate(zip(excited.e, strengths, strict=True))
        ]
        result["seconds"]["excitations"] = time.time() - lap
    result["seconds"]["total"] = time.time() - started
    return result


if __name__ == "__main__":  # pragma: no cover - exercised through the runner
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
