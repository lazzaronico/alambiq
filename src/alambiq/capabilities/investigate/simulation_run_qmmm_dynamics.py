"""Run constant-temperature QM/MM dynamics on a prepared molecular system.

The system is divided as in `simulation.evaluate_qmmm_state`: a quantum region solved
with closed-shell PBE/6-31G** in PySCF, in the field of the Gaussian-smeared charges of
the classical atoms, and a classical remainder (with an optional external potential). A
Langevin thermostat, with the temperature, friction, timestep and seed given in the
request, propagates every atom, with an electronic calculation at each new geometry; the
density of the previous integration step is the starting point of each solution.
Equilibration steps are run first. The table records the initial geometry and every
reporting interval of the equilibration and production steps (column `phase`); the
trajectory holds the production frames only. An optional umbrella bias adds a harmonic
potential on the difference of two interatomic distances, which holds the reaction
coordinate near a chosen value. The result reports the bias energy separately from the
unbiased potential energy.

The result gives a seeded trajectory of this system, started from these positions,
under this Hamiltonian and thermostat, with the energy components at each reporting interval.

Limits. Each integration step requires a self-consistent-field solution, which limits
the length of a trajectory; the trajectory samples the processes that occur within its length.
A trajectory depends on the starting geometry, and an error estimate needs independent
trajectories. The Hamiltonian has the limits of `simulation.evaluate_qmmm_state`: PBE
without dispersion, fixed classical charges, and the division and cap defined in the
request. The thermostat is Langevin, so the dynamics are stochastic. The kinetic energy
is the native expression of the OpenMM LangevinMiddle integrator, which uses half-step
velocities. The instantaneous temperature of a system with N degrees of freedom has a
relative standard deviation of sqrt(2/N), which is 26 % for 30 degrees of freedom. The
velocities are drawn from the seed, so each simulation starts from the supplied
positions. The capability runs on the OpenMM Reference platform on the CPU, and the
system must contain no cell and only the terms listed in
`simulation.evaluate_qmmm_state`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from alambiq.capabilities.investigate.simulation_evaluate_qmmm_state import (
    QuantumRegion,
)
from alambiq.capabilities.investigate.simulation_run_qmmm_dynamics_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class Umbrella:
    """A harmonic bias on the difference of two distances.

    pairs:
        Two pairs of one-based atom indices, [[a, b], [c, d]]. The coordinate is the
        distance a-b minus the distance c-d, in angstrom.
    center_angstrom:
        The value, in angstrom, near which the bias holds the coordinate.
    coefficient_kj_mol_angstrom2:
        The force constant K of the bias K (s - center)^2, in kJ/mol/angstrom^2.
    """

    pairs: list[list[int]]
    center_angstrom: float
    coefficient_kj_mol_angstrom2: float


@dataclass(frozen=True, slots=True)
class QmmmDynamicsRequest:
    """The prepared system, the division and the sampling settings.

    system, state, topology:
        Paths to the serialized OpenMM system, the OpenMM state file and the topology PDB.
    quantum:
        The quantum region, as in `simulation.evaluate_qmmm_state`.
    temperature_kelvin, friction_per_ps, timestep_fs:
        The temperature and friction of the Langevin thermostat, and the timestep.
    seed:
        The seed for the starting velocities and the thermostat, at least 1.
    equilibration_steps, production_steps, report_interval:
        The number of integration steps run before production, the number of production
        integration steps, and the number of integration steps between records of the
        table; the production steps must be a whole number of intervals.
    external_system:
        Optional external potential, as in `simulation.evaluate_qmmm_state`.
    initial_density:
        Optional density used as the starting point of the first solution.
    umbrella:
        Optional bias on a distance difference.
    grid, gaussian_radius_angstrom, threads, memory_mb, classical_coulomb_ratio:
        The electronic settings, as in `simulation.evaluate_qmmm_state`.
    timeout_seconds:
        The longest permitted duration of the simulation, in seconds.
    """

    system: str
    state: str
    topology: str
    quantum: QuantumRegion
    temperature_kelvin: float = 300.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 0.5
    seed: int = 1
    equilibration_steps: int = 0
    production_steps: int = 10
    report_interval: int = 1
    external_system: str | None = None
    initial_density: str | None = None
    umbrella: Umbrella | None = None
    grid: int = 7
    gaussian_radius_angstrom: float = 0.8
    threads: int = 2
    memory_mb: int = 6000
    classical_coulomb_ratio: float = 1.0
    timeout_seconds: int = 604800


@dataclass(frozen=True, slots=True)
class QmmmDynamicsResult:
    """The size of the trajectory, its averages and the location of its files."""

    production_frames: int
    equilibration_steps: int
    production_steps: int
    frame_interval_fs: float
    electronic_calls: int
    """Distinct geometries solved; a geometry seen twice is solved once."""
    cache_hits: int
    mean_temperature_kelvin: float
    mean_potential_kj_mol: float
    mean_unbiased_potential_kj_mol: float
    mean_total_energy_kj_mol: float
    total_energy_drift_kj_mol: float
    """The last production total energy minus the first."""
    mean_coordinate_angstrom: float | None
    coordinate_sd_angstrom: float | None
    maximum_constraint_error_angstrom: float
    constraints: int
    trajectory: str
    """A DCD of the production frames; every atom."""
    table: str
    """A CSV, one row per record, with the energy components of every force group."""
    final_state: str
    final_density: str
    summary: str
    hamiltonian_sha256: str
    """A digest of the Hamiltonian definition, used to test whether two runs share a Hamiltonian."""
    seconds: float


@capability(
    "simulation.run_qmmm_dynamics",
    version="1.0",
    family="investigate",
    summary="Run Langevin QM/MM dynamics with PySCF, optionally under an umbrella bias",
    render=render,
    environment="qmmm",
)
def run_qmmm_dynamics(
    request: QmmmDynamicsRequest, ctx: StepContext
) -> QmmmDynamicsResult:
    """Check the settings, sample, and summarize the table."""
    import csv
    import time

    from alambiq.capabilities.investigate.science.qmmm import (
        check_qmmm_settings,
        check_sampling,
        region_dict,
        umbrella_dict,
    )

    check_qmmm_settings(
        grid=request.grid,
        gaussian_radius_angstrom=request.gaussian_radius_angstrom,
        threads=request.threads,
        memory_mb=request.memory_mb,
        classical_coulomb_ratio=request.classical_coulomb_ratio,
        timeout_seconds=request.timeout_seconds,
    )
    sampling = {
        "temperature_kelvin": request.temperature_kelvin,
        "friction_per_ps": request.friction_per_ps,
        "timestep_fs": request.timestep_fs,
        "seed": request.seed,
        "equilibration_steps": request.equilibration_steps,
        "production_steps": request.production_steps,
        "report_interval": request.report_interval,
        "umbrella": umbrella_dict(request.umbrella),
    }
    check_sampling(sampling)
    started = time.perf_counter()
    output = ctx.workdir / "qmmm-dynamics"
    found = ctx.run_in_environment(
        "qmmm",
        "qmmm_dynamics_runner:run",
        {
            "prepared": {
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
            },
            "sampling": sampling,
            "output": str(output),
        },
        timeout=request.timeout_seconds + 600,
    )
    with (output / "state.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["phase"] == "production"]
    if not rows:
        raise ValueError("The run recorded no production steps")

    import numpy as np

    def column(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows])

    coordinate = column("coordinate_angstrom") if request.umbrella is not None else None
    total = column("total_kj_mol")
    if request.umbrella is not None:
        ctx.note(
            "The potential includes the umbrella bias; the unbiased potential is reported "
            "beside it."
        )
    return QmmmDynamicsResult(
        production_frames=found["production_frames"],
        equilibration_steps=request.equilibration_steps,
        production_steps=request.production_steps,
        frame_interval_fs=found["frame_interval_fs"],
        electronic_calls=found["electronic_calls"],
        cache_hits=found["cache_hits"],
        mean_temperature_kelvin=float(column("temperature_kelvin").mean()),
        mean_potential_kj_mol=float(column("potential_kj_mol").mean()),
        mean_unbiased_potential_kj_mol=float(
            column("unbiased_potential_kj_mol").mean()
        ),
        mean_total_energy_kj_mol=float(total.mean()),
        total_energy_drift_kj_mol=float(total[-1] - total[0]),
        mean_coordinate_angstrom=float(coordinate.mean())
        if coordinate is not None
        else None,
        coordinate_sd_angstrom=float(coordinate.std(ddof=1))
        if coordinate is not None and len(coordinate) > 1
        else None,
        maximum_constraint_error_angstrom=float(
            column("maximum_constraint_error_angstrom").max()
        ),
        constraints=found["constraints"],
        trajectory=str(
            ctx.keep(output / "trajectory.dcd", "application/octet-stream").path
        ),
        table=str(ctx.keep(output / "state.csv", "text/csv").path),
        final_state=str(ctx.keep(output / "final-state.xml", "application/xml").path),
        final_density=str(
            ctx.keep(output / "final-density.npz", "application/zip").path
        ),
        summary=ctx.keep_json(_summary(found)),
        hamiltonian_sha256=found["hamiltonian_sha256"],
        seconds=time.perf_counter() - started,
    )


def _summary(found: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(found))
