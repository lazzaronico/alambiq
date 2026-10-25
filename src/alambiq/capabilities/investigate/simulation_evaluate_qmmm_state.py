"""Evaluate the quantum-mechanical and classical energy and forces of one molecular geometry.

A prepared molecular system (its serialized OpenMM system, OpenMM state file and
topology, from `structure.prepare_finite_environment` or from another preparation that
gives a system as defined under Limits) is divided into a quantum region, whose atoms
are listed in the request, and a classical remainder. The quantum region is solved with
PySCF as closed-shell PBE in the 6-31G** basis, in the field of the charges of the
classical atoms, which enter as Gaussian-smeared point charges of the radius given in
the request. Where one covalent bond joins the quantum region to the classical
remainder, the bond is capped with a hydrogen at the distance 1/`alpha` of the bond
length from the quantum parent atom, with `alpha` from the request. The classical
remainder keeps its charges, the Lennard-Jones terms between classical and quantum
atoms, and every bonded term that has at least one atom outside the set formed by the
quantum region and the classical parent atom of the bond; these are evaluated
classically. An optional external potential (the boundary of a finite droplet) is added.
The result is the electronic, classical and external energies, their sum, and the force
on every atom. The capability refuses a result with a non-finite force.

The result gives the energy and forces of this geometry under this division. The
capability refuses an evaluation unless the self-consistent field has converged, the
electron count agrees with the expected count within 1e-7, and the norm of the orbital
gradient is at most 1e-7.

Limits. The functional is PBE, a generalized-gradient functional without dispersion
correction and without exact exchange; the basis is 6-31G**. The Lennard-Jones terms
between quantum and classical atoms are the dispersion treatment of the interaction
between the quantum region and its surroundings. Barrier heights and reaction energies
are those of PBE in the 6-31G** basis. The classical charges are fixed values that do
not change with the quantum region. The result depends on the division in the request:
the atoms of the quantum region, the hydrogen cap and the classical atoms whose charges
are excluded from the field of the quantum region (`cancelled_coupling_atoms`). Energies
are comparable between geometries that share one division. The capability refuses a
quantum region that is cut by a covalent bond other than the declared link. The system
must contain only all-pair nonbonded terms, harmonic bonds, angles and periodic
torsions, with no cutoff and no cell. The capability supports closed-shell systems on
the CPU. One call evaluates one geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from alambiq.capabilities.investigate.simulation_evaluate_qmmm_state_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class Link:
    """The one covalent bond between the quantum region and the classical rest.

    quantum_parent, classical_parent:
        The one-based indices of the atoms at the two ends of the bond.
    alpha:
        The cap hydrogen is placed at the distance 1/`alpha` of the bond length from the
        quantum parent along the bond to the classical parent; greater than 1.
    cancelled_coupling_atoms:
        The one-based indices of the classical atoms whose charges are excluded from the
        field of the quantum region, to avoid overpolarization of the cap; for example
        the classical parent and the atoms bonded to it.
    """

    quantum_parent: int
    classical_parent: int
    alpha: float
    cancelled_coupling_atoms: list[int]


@dataclass(frozen=True, slots=True)
class QuantumRegion:
    """The atoms of the quantum region.

    atoms:
        The one-based indices of the quantum atoms, in the order of the topology; a
        proper subset of the atoms.
    charge:
        The net charge of the quantum region, in elementary charges. With the cap
        hydrogen counted, the electrons must pair.
    link:
        The covalent bond to the classical rest, if the quantum region cuts one.
    """

    atoms: list[int]
    charge: int
    link: Link | None = None


@dataclass(frozen=True, slots=True)
class QmmmRequest:
    """The prepared system and the division and solution settings.

    system, state, topology:
        Paths to the serialized OpenMM system, the OpenMM state file and the topology
        PDB.
    quantum:
        The quantum region: the atom indices, the net charge and, where a covalent bond
        is cut, the link with the cap setting `alpha`.
    external_system:
        Optional. An OpenMM system holding only external, centroid or compound-bond
        forces over the same particles, added unchanged (for example a droplet wall).
    initial_density:
        Optional. A density matrix (an NPY, or the electronic state NPZ of an earlier
        evaluation) used as the starting point of the self-consistent field.
    grid:
        The PySCF quadrature level, 0 to 9.
    gaussian_radius_angstrom:
        The radius, in angstrom, of the Gaussian into which each classical charge is
        smeared.
    threads, memory_mb:
        The number of threads and the memory, in MB, for PySCF.
    classical_coulomb_ratio:
        A positive factor by which the classical Coulomb interaction energies are
        multiplied; the default 1.0 leaves them unchanged.
    timeout_seconds:
        The longest permitted duration of the evaluation, in seconds.
    """

    system: str
    state: str
    topology: str
    quantum: QuantumRegion
    external_system: str | None = None
    initial_density: str | None = None
    grid: int = 7
    gaussian_radius_angstrom: float = 0.8
    threads: int = 2
    memory_mb: int = 6000
    classical_coulomb_ratio: float = 1.0
    timeout_seconds: int = 86400


@dataclass(frozen=True, slots=True)
class QmmmResult:
    """The energy components, the forces and the convergence diagnostics of the electronic solution."""

    electronic_energy_kj_mol: float
    classical_energy_kj_mol: float
    external_energy_kj_mol: float
    total_energy_kj_mol: float
    """The sum of the three."""
    physical_atoms: int
    electronic_atoms: int
    """Quantum atoms plus the cap hydrogen, if any."""
    basis_functions: int
    scf_cycles: int
    converged: bool
    orbital_gradient_norm: float
    electrons_found: float
    electrons_expected: int
    maximum_physical_force_kj_mol_nm: float
    net_force_kj_mol_nm: list[float]
    """The sum of the forces on all atoms, which is zero for forces derived from a potential."""
    classical_ledger: dict[str, dict[str, int]]
    """For each kind of term, the number in the source, the number retained and the number removed because the term lies wholly in the quantum region."""
    forces: str
    """An NPY file of the force on each physical atom, kJ/mol/nm."""
    electronic_state: str
    """An NPZ file of the density, overlap and orbitals, usable as the starting point of a later evaluation."""
    summary: str
    atom_map: str
    seconds: float
    pyscf_version: str


@capability(
    "simulation.evaluate_qmmm_state",
    version="1.0",
    family="investigate",
    summary="Evaluate the QM/MM energy and forces of one molecular geometry with PySCF",
    render=render,
    environment="qmmm",
)
def evaluate_qmmm_state(request: QmmmRequest, ctx: StepContext) -> QmmmResult:
    """Solve the quantum region in the classical field and compose the energies and forces."""
    from alambiq.capabilities.investigate.science.qmmm import (
        check_qmmm_settings,
        region_dict,
    )

    check_qmmm_settings(
        grid=request.grid,
        gaussian_radius_angstrom=request.gaussian_radius_angstrom,
        threads=request.threads,
        memory_mb=request.memory_mb,
        classical_coulomb_ratio=request.classical_coulomb_ratio,
        timeout_seconds=request.timeout_seconds,
    )
    output = ctx.workdir / "qmmm"
    found = ctx.run_in_environment(
        "qmmm",
        "qmmm_runner:run",
        {
            "system_xml": str(ctx.resolve(request.system)),
            "state_xml": str(ctx.resolve(request.state)),
            "topology_pdb": str(ctx.resolve(request.topology)),
            "external_system_xml": str(ctx.resolve(request.external_system))
            if request.external_system
            else None,
            "initial_density": str(ctx.resolve(request.initial_density))
            if request.initial_density
            else None,
            "quantum": region_dict(request.quantum),
            "parameters": {
                "backend": "cpu",
                "grid": request.grid,
                "gaussian_radius_angstrom": request.gaussian_radius_angstrom,
                "threads": request.threads,
                "memory_mb": request.memory_mb,
                "classical_coulomb_ratio": request.classical_coulomb_ratio,
            },
            "output": str(output),
        },
        timeout=request.timeout_seconds + 600,
    )
    electronic = found["electronic"]
    if request.quantum.link is not None:
        ctx.note("The quantum region cuts a covalent bond and is capped with a hydrogen.")
    seconds = electronic["elapsed_seconds"]
    return QmmmResult(
        electronic_energy_kj_mol=found["electronic_energy_kj_mol"],
        classical_energy_kj_mol=found["classical_energy_kj_mol"],
        external_energy_kj_mol=found["external_energy_kj_mol"],
        total_energy_kj_mol=found["total_energy_kj_mol"],
        physical_atoms=found["physical_atoms"],
        electronic_atoms=found["electronic_atoms"],
        basis_functions=electronic["basis_functions"],
        scf_cycles=electronic["scf_cycles"],
        converged=electronic["converged"],
        orbital_gradient_norm=electronic["orbital_gradient_norm"],
        electrons_found=electronic["electrons"],
        electrons_expected=electronic["expected_electrons"],
        maximum_physical_force_kj_mol_nm=found["maximum_physical_force_kj_mol_nm"],
        net_force_kj_mol_nm=found["net_force_kj_mol_nm"],
        classical_ledger=found["classical_ledger"],
        forces=str(
            ctx.keep(
                output / "physical-forces-kj-mol-nm.npy", "application/octet-stream"
            ).path
        ),
        electronic_state=str(
            ctx.keep(
                output / "electronic" / "electronic-state.npz", "application/zip"
            ).path
        ),
        summary=ctx.keep_json(json.loads((output / "summary.json").read_text())),
        atom_map=str(ctx.keep(output / "atom-map.json", "application/json").path),
        seconds=seconds,
        pyscf_version=electronic["pyscf_version"],
    )
