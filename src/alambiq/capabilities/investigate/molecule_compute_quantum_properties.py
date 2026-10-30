"""Compute orbital energies, thermochemistry, excitations and charges of one small molecule.

Returns the density-functional energy of a molecule given as SMILES at its optimized geometry,
with the properties that the request names: the energies of the frontier orbitals and the
gap between them, the vibrational thermochemistry (zero-point energy, enthalpy, Gibbs energy,
entropy and heat capacities), the lowest vertical excitation energies with their oscillator
strengths, and the dipole moment with atomic charges. The molecule can be charged, in a spin
state above the singlet, and in water.

The method:

- RDKit embeds 10 conformers (ETKDG, from the request's `seed`), relaxes each with the MMFF94
  force field, and keeps the one of lowest force-field energy.
- PySCF optimizes that geometry with the request's functional and basis, with PyBerny, on a
  numerical grid of level 4. The optimization must converge; a geometry that has not converged
  returns no properties.
- At the optimized geometry, the orbital energies come from the Kohn-Sham calculation. The
  thermochemistry comes from the analytic Hessian, the harmonic oscillator and rigid rotor
  approximations and the ideal gas at the requested temperature and 101,325 Pa; modes that PySCF
  classes as translations and rotations are excluded. The excitations come from time-dependent
  density functional theory (TDDFT) with the request's functional.
- The basis functions are Cartesian, the convention of Gaussian for the Pople bases.
- A molecule with `multiplicity` 1 uses restricted Kohn-Sham theory. A higher multiplicity uses
  unrestricted theory, and the result gives the expectation value of S squared against the value
  S(S+1) of a pure spin state; a difference of more than 10 % of S(S+1) carries a note.
- With `solvent` `water`, the optimization and every property use the IEF-PCM polarizable
  continuum with a relative permittivity of 78.3553. The result also gives the polarization
  energy of the continuum: the energy in water minus the gas-phase energy at the same geometry.
- The dipole moment is taken about the center of mass, and the charges are Mulliken charges. For
  an open-shell molecule the result adds the Mulliken spin population of each atom.
- Energies are in hartree, orbital and excitation energies in eV, thermochemical energies in
  kcal/mol, entropies and heat capacities in cal/(mol K), dipole moments in debye.

Limits.

- The charge is the formal charge of the SMILES, and the multiplicity is a request field. A
  radical in the SMILES requires a multiplicity above 1, and a multiplicity that the number of
  electrons cannot have is refused. An unrestricted calculation can converge to a state other
  than the lowest of that multiplicity, and the result does not check this.
- IEF-PCM is an electrostatic model. It has no cavity or dispersion term, so its polarization
  energy is not a solvation free energy: polar molecules come within about 1 kcal/mol of the
  experimental values and nonpolar ones differ by 1.5 kcal/mol or more (see
  `docs/methods/molecules.md`). It is a continuum without hydrogen bonds to individual waters.
- Mulliken charges depend strongly on the basis set and are not transferable between levels of
  theory. The dipole of a charged molecule depends on its origin.
- The geometry is the lowest-energy conformer of 10, which is not necessarily the global minimum.
  Properties of a flexible molecule depend on the conformer.
- Vibrations are harmonic. Internal rotations, such as a methyl group, are treated as vibrations,
  which affects the entropy and the heat capacity of flexible molecules.
- The rotational symmetry number comes from the point group that PySCF detects at the optimized
  geometry, with a tolerance of 0.003 angstrom. Entropy and Gibbs energy depend on it. A
  conformer that is asymmetric by more than the tolerance is given the lower symmetry.
- A modest number of imaginary frequencies means that the optimized geometry is a saddle point;
  the result states the number, and the thermochemistry leaves those modes out.
- TDDFT excitation energies are vertical. In water the continuum is equilibrated with the
  ground state. They differ from coupled-cluster references by amounts that depend on the
  functional and the state; charge-transfer and Rydberg states are described poorly by B3LYP
  and PBE. For an open-shell molecule they come from unrestricted TDDFT and are spin
  contaminated to the extent that the ground state is.
- A molecule of 1 to 12 heavy atoms from H, C, N, O, F, P, S, Cl and Br is accepted. The run time
  grows steeply with size; the table in `docs/methods/molecules.md` gives measured times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from alambiq.capabilities.investigate.molecule_compute_quantum_properties_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

MAX_HEAVY_ATOMS = 12
ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br"})
GRID_LEVEL = 4
CONFORMERS = 10
PRESSURE_PA = 101325.0
TIMEOUT_SECONDS = 4 * 3600


@dataclass(frozen=True, slots=True)
class QuantumRequest:
    """The molecule, the level of theory and the properties to compute.

    smiles:
        The molecule as SMILES, 1 to 500 characters with no whitespace. Its net charge is the
        formal charge of the SMILES.
    multiplicity:
        The spin multiplicity 2S+1, 1 to 7; default 1 (closed-shell). Radicals and triplets
        need a higher value.
    solvent:
        `none` (default, the gas phase) or `water` (the IEF-PCM continuum).
    functional:
        The density functional: `b3lyp` (default), `pbe0`, `pbe` or `camb3lyp`.
    basis:
        The basis set: `6-31G*` (default), `6-31G(2df,p)`, `def2-SVP` or `def2-TZVP`.
    properties:
        Any of `orbitals`, `charges`, `thermochemistry` and `excitations`; default `orbitals`.
        The energy and the geometry are always returned. `charges` gives the dipole moment and
        the Mulliken charges at no extra cost. `thermochemistry` adds an analytic Hessian, and
        `excitations` adds a TDDFT calculation; both cost more than the optimization.
    excited_states:
        The number of excited states for `excitations`, 1 to 10; default 3.
    temperature_k:
        The temperature of the thermochemistry in kelvin, 1 to 2,000; default 298.15.
    max_steps:
        The most geometry optimization steps, 1 to 300; default 100.
    seed:
        The random seed of the conformer embedding, 0 to 2,147,483,647; default 1.
    threads:
        Threads used by PySCF, 1 to 64; default 8.
    """

    smiles: str
    multiplicity: int = 1
    solvent: Literal["none", "water"] = "none"
    functional: Literal["b3lyp", "pbe0", "pbe", "camb3lyp"] = "b3lyp"
    basis: Literal["6-31G*", "6-31G(2df,p)", "def2-SVP", "def2-TZVP"] = "6-31G*"
    properties: list[
        Literal["orbitals", "charges", "thermochemistry", "excitations"]
    ] = field(default_factory=lambda: ["orbitals"])
    excited_states: int = 3
    temperature_k: float = 298.15
    max_steps: int = 100
    seed: int = 1
    threads: int = 8


@dataclass(frozen=True, slots=True)
class Orbitals:
    """The frontier orbitals. For an open-shell molecule they pool the two spins."""

    homo_ev: float
    lumo_ev: float
    gap_ev: float
    """The energy of the lowest unoccupied orbital minus that of the highest occupied one."""
    occupied_orbitals: int
    """The occupied orbitals, counted as spin-orbitals in an open-shell molecule."""
    highest_occupied_ev: list[float]
    """The energies of the 5 highest occupied orbitals, ascending."""
    lowest_virtual_ev: list[float]
    """The energies of the 5 lowest unoccupied orbitals, ascending."""
    alpha_homo_ev: float | None = None
    alpha_lumo_ev: float | None = None
    beta_homo_ev: float | None = None
    beta_lumo_ev: float | None = None
    """The frontier orbitals of each spin; only for an open-shell molecule."""


@dataclass(frozen=True, slots=True)
class AtomCharge:
    """The Mulliken charge of one atom."""

    atom: int
    """The one-based position of the atom in the geometry file."""
    element: str
    charge: float
    spin_population: float | None
    """The Mulliken spin population; only for an open-shell molecule."""


@dataclass(frozen=True, slots=True)
class Charges:
    """The dipole moment and the atomic charges of the optimized geometry."""

    dipole_debye: float
    dipole_components_debye: list[float]
    """The components in the frame of the geometry file."""
    dipole_origin: str
    mulliken: list[AtomCharge]


@dataclass(frozen=True, slots=True)
class Thermochemistry:
    """The ideal-gas thermochemistry of the optimized geometry."""

    temperature_k: float
    pressure_pa: float
    zero_point_energy_hartree: float
    zero_point_energy_kcal_mol: float
    energy_0k_hartree: float
    """The electronic energy plus the zero-point energy."""
    internal_energy_hartree: float
    enthalpy_hartree: float
    gibbs_energy_hartree: float
    entropy_cal_mol_k: float
    cv_cal_mol_k: float
    cp_cal_mol_k: float
    symmetry_number: int
    vibrational_modes: int
    """The number of real vibrational frequencies used."""
    imaginary_modes: int
    lowest_frequencies_cm1: list[float]
    """The 3 lowest real frequencies, ascending."""


@dataclass(frozen=True, slots=True)
class Excitation:
    """One vertical singlet excitation."""

    state: int
    energy_ev: float
    wavelength_nm: float
    oscillator_strength: float


@dataclass(frozen=True, slots=True)
class QuantumResult:
    """The optimized geometry, its energy and the requested properties."""

    smiles: str
    canonical_smiles: str
    formula: str
    charge: int
    multiplicity: int
    solvent: str
    heavy_atoms: int
    atoms: int
    functional: str
    basis: str
    grid_level: int
    cartesian_basis: bool
    conformers_embedded: int
    force_field_energy_kcal_mol: float
    """The MMFF94 energy of the conformer that was optimized."""
    optimization_steps: int
    energy_hartree: float
    """The Kohn-Sham energy at the optimized geometry, in the solvent when there is one."""
    geometry: str
    """The optimized geometry as an XYZ file, in angstrom."""
    orbitals: Orbitals | None
    charges: Charges | None
    thermochemistry: Thermochemistry | None
    excitations: list[Excitation]
    seconds: dict[str, float]
    record: str
    engine_versions: dict[str, Any]
    spin_squared: float | None = None
    """The expectation value of S squared; only for an open-shell molecule."""
    expected_spin_squared: float | None = None
    """S(S+1) of a pure spin state of the requested multiplicity."""
    gas_phase_energy_hartree: float | None = None
    """The energy at the same geometry without the continuum; only with a solvent."""
    solvent_polarization_energy_kcal_mol: float | None = None
    """The energy with the continuum minus the gas-phase energy, at the same geometry."""


def _check(request: QuantumRequest) -> None:
    import re

    if (
        not request.smiles
        or len(request.smiles) > 500
        or re.search(r"\s", request.smiles)
    ):
        raise ValueError("smiles must be 1 to 500 characters with no whitespace")
    if not request.properties or len(set(request.properties)) != len(
        request.properties
    ):
        raise ValueError(
            "properties must list each of orbitals, charges, thermochemistry, "
            "excitations once"
        )
    if not 1 <= request.multiplicity <= 7:
        raise ValueError("multiplicity must be an integer from 1 to 7")
    if not 1 <= request.excited_states <= 10:
        raise ValueError("excited_states must be an integer from 1 to 10")
    if (
        not math.isfinite(request.temperature_k)
        or not 1 <= request.temperature_k <= 2000
    ):
        raise ValueError("temperature_k must be between 1 and 2000")
    for name, value, low, high in (
        ("max_steps", request.max_steps, 1, 300),
        ("seed", request.seed, 0, 2147483647),
        ("threads", request.threads, 1, 64),
    ):
        if not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")


@capability(
    "molecule.compute_quantum_properties",
    version="1.0",
    family="investigate",
    summary="Compute orbitals, charges, thermochemistry and excitations of a molecule",
    render=render,
    environment="molecular_properties",
)
def compute_quantum_properties(
    request: QuantumRequest, ctx: StepContext
) -> QuantumResult:
    """Optimize the molecule with density functional theory and compute the properties."""
    _check(request)
    found = ctx.run_in_environment(
        "molecular_properties",
        "properties_compute:run",
        {
            "smiles": request.smiles,
            "functional": request.functional,
            "basis": request.basis,
            "properties": list(request.properties),
            "multiplicity": request.multiplicity,
            "solvent": request.solvent,
            "excited_states": request.excited_states,
            "temperature_k": request.temperature_k,
            "max_steps": request.max_steps,
            "seed": request.seed,
            "threads": request.threads,
            "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "elements": sorted(ELEMENTS),
        },
        timeout=TIMEOUT_SECONDS,
    )
    thermochemistry = (
        Thermochemistry(**found["thermochemistry"])
        if "thermochemistry" in found
        else None
    )
    if thermochemistry and thermochemistry.imaginary_modes:
        ctx.note(
            f"The optimized geometry has {thermochemistry.imaginary_modes} imaginary "
            "frequencies and is a saddle point; the thermochemistry leaves those modes out."
        )
    spin_squared = found.get("spin_squared")
    expected = found.get("expected_spin_squared")
    if (
        spin_squared is not None
        and expected
        and abs(spin_squared - expected) > 0.1 * expected
    ):
        ctx.note(
            f"The expectation value of S squared is {spin_squared:.3f} against {expected:.3f} "
            "for a pure spin state; the unrestricted solution is spin contaminated."
        )
    excitations = [Excitation(**e) for e in found.get("excitations", [])]
    charges = None
    if "charges" in found:
        atoms = [AtomCharge(**a) for a in found["charges"]["mulliken"]]
        charges = Charges(**{**found["charges"], "mulliken": atoms})
    geometry = ctx.keep_text(found["geometry_xyz"], "chemical/x-xyz")
    record = {
        "smiles": request.smiles,
        "functional": request.functional,
        "basis": request.basis,
        "cartesian_basis": True,
        "grid_level": GRID_LEVEL,
        "conformers": CONFORMERS,
        "pressure_pa": PRESSURE_PA,
        "seed": request.seed,
        "multiplicity": request.multiplicity,
        "solvent": request.solvent,
        **{key: value for key, value in found.items() if key != "geometry_xyz"},
    }
    return QuantumResult(
        smiles=request.smiles,
        canonical_smiles=found["canonical_smiles"],
        formula=found["formula"],
        charge=found["charge"],
        multiplicity=request.multiplicity,
        solvent=request.solvent,
        heavy_atoms=found["heavy_atoms"],
        atoms=found["atoms"],
        functional=request.functional,
        basis=request.basis,
        grid_level=GRID_LEVEL,
        cartesian_basis=True,
        conformers_embedded=found["conformers_embedded"],
        force_field_energy_kcal_mol=found["force_field_energy_kcal_mol"],
        optimization_steps=found["optimization_steps"],
        energy_hartree=found["energy_hartree"],
        geometry=str(geometry.path),
        orbitals=Orbitals(**found["orbitals"]) if "orbitals" in found else None,
        charges=charges,
        thermochemistry=thermochemistry,
        excitations=excitations,
        seconds=found["seconds"],
        record=ctx.keep_json(record),
        engine_versions=found["versions"],
        spin_squared=spin_squared,
        expected_spin_squared=expected,
        gas_phase_energy_hartree=found.get("gas_phase_energy_hartree"),
        solvent_polarization_energy_kcal_mol=found.get(
            "solvent_polarization_energy_kcal_mol"
        ),
    )
