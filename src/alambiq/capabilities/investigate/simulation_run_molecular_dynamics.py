"""Run molecular dynamics on a prepared system with OpenMM.

A serialized system, OpenMM state file and topology, such as those written by
`structure.prepare_protein_system`, are run through the following phases, whose lengths
are set in picoseconds:

1. Optional minimization.
2. Velocities drawn at the target temperature, or taken from the state file.
3. Thermalization at constant volume with the protein heavy atoms restrained.
4. For `npt`, release: the restraints decrease to zero while the barostat relaxes the
   box.
5. Readiness windows. Each window runs unrestrained dynamics in 4 blocks and tests the
   drift of temperature, density and potential energy against the thresholds given
   under Limits. Readiness windows repeat up to `max_readiness_windows`.
6. Production, which writes a trajectory and the thermodynamic data.

The ensembles are `npt` and `nvt` for explicit solvent, `implicit` for a GBn2 system
and `nve` for energy-conserving dynamics, which has no thermostat and so no
equilibration.

The result gives a trajectory that started from the prepared system, with the
statistics of every phase, the readiness verdict, and the platform and versions used.

Limits. A readiness window passes when four conditions hold: the mean temperature of the
window is within 5 % of the target, the means of the 4 blocks span at most 3 % of the
target temperature, the block means of the potential energy span at most 0.1 kJ/mol per
atom, and (for a periodic system) the block means of the density span at most 2 % of the
mean density. A passed last window means that none of these thresholds was exceeded in
that window, which indicates stable temperature, density and potential energy over the
window. A simulation samples the motions that occur within its length, and an average
over the simulation describes that sampling. A difference between two conditions is
tested with replicas. A simulation samples the ensemble that its thermostat and barostat
settings define. The force field is that of the supplied system; for a system from
`structure.prepare_protein_system` it is Amber ff14SB with TIP3P water (explicit
solvent) or GBn2 (implicit solvent), and the limits of that force field apply. Timesteps
above 1 fs require constrained hydrogen bonds. On one CPU thread, the default, the same
seed gives the same trajectory for the implicit-solvent system in
`tests/core/test_simulation_run_molecular_dynamics.py`. With more threads, the order of
floating-point sums varies, simulations diverge from each other, and results agree only
statistically. Results on different platforms also agree only statistically. A
trajectory holds every atom, solvent included. A request that would write more than
20,000 frames is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from alambiq.capabilities.investigate.simulation_run_molecular_dynamics_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class DynamicsRequest:
    """The prepared system, the ensemble and the length of each phase.

    system, state, topology:
        Paths to the serialized OpenMM system, the OpenMM state file and the topology PDB, as
        produced by `structure.prepare_protein_system`.
    ensemble:
        `npt` or `nvt` for a periodic system, `implicit` for a non-periodic one,
        `nve` for constant energy (no thermostat, no equilibration).
    temperature_kelvin, friction_per_ps, timestep_fs:
        The target temperature and friction of the Langevin thermostat, and the
        timestep (a timestep above 1 fs requires constrained hydrogen bonds).
    pressure_bar, barostat_interval:
        The pressure of the barostat for `npt`, and the number of integration steps between
        volume-move attempts.
    minimize_iterations:
        The maximum number of minimization iterations before dynamics; 0 skips
        minimization (in `structure.prepare_predicted_complex_for_dynamics` and
        `structure.prepare_protein_ligand_system`, 0 means until converged).
    velocities:
        `maxwell-boltzmann` draws new velocities at the temperature; `state` uses
        the velocities of the supplied state file, for a continuation or for `nve`.
    thermalize_ps, restraint_kj_mol_nm2:
        The length of the restrained constant-volume phase, and the force constant on
        the protein heavy atoms. A zero restraint thermalizes without restraints. The
        default length is 50 ps, and 0 ps (the phase is not run) for `nve`.
    release_ps:
        For `npt`: the length of the phase in which the restraints decrease to zero
        while the box relaxes. The default is 100 ps for `npt` and 0 ps (the phase is
        not run) otherwise.
    readiness_window_ps, max_readiness_windows:
        The length of each readiness window (0 skips the check, otherwise at least
        20 ps) and the maximum number of windows tried. The default length is 200 ps,
        and 0 ps (no check) for `nve`.
    require_ready:
        If true (the default), production is skipped when no readiness window passes.
        If false, production runs whether or not a window passes.
    production_ps, report_interval:
        The production length, and the number of integration steps between trajectory
        frames and rows of the state data.
    platform, platform_properties, seed:
        The OpenMM platform (`CPU`, `Reference`, `CUDA`, `OpenCL`; no fallback), its
        properties (1 CPU thread by default, the setting under which a seed reproduces
        a trajectory), and the seed for velocities and the thermostat.
    """

    system: str
    state: str
    topology: str
    ensemble: Literal["npt", "nvt", "implicit", "nve"]
    temperature_kelvin: float = 300.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 2.0
    pressure_bar: float = 1.0
    barostat_interval: int = 25
    minimize_iterations: int = 0
    velocities: Literal["maxwell-boltzmann", "state"] = "maxwell-boltzmann"
    thermalize_ps: float | None = None
    restraint_kj_mol_nm2: float = 1000.0
    release_ps: float | None = None
    readiness_window_ps: float | None = None
    max_readiness_windows: int = 3
    require_ready: bool = True
    production_ps: float = 100.0
    report_interval: int = 500
    platform: Literal["CPU", "Reference", "CUDA", "OpenCL"] = "CPU"
    platform_properties: dict[str, str] = field(default_factory=dict)
    seed: int = 1


@dataclass(frozen=True, slots=True)
class Phase:
    """One phase that ran, and its statistics."""

    name: str
    steps: int | None
    detail: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DynamicsResult:
    """The phases that ran, the readiness verdict and the location of the trajectory."""

    ensemble: str
    particles: int
    phases: list[Phase]
    ready_for_production: bool
    """True if no readiness check was requested, or a readiness window passed."""
    production_run: bool
    frames: int
    final_step: int
    final_time_ps: float
    mean_temperature_kelvin: float | None
    mean_volume_nm3: float | None
    mean_density_g_ml: float | None
    total_energy_first_kj_mol: float | None
    total_energy_last_kj_mol: float | None
    total_energy_min_kj_mol: float | None
    total_energy_max_kj_mol: float | None
    max_force_kj_mol_nm: float
    trajectory: str | None
    state_data: str | None
    final_state: str
    final_structure: str
    summary: str
    seed: int
    platform: str
    versions: dict[str, Any]


@capability(
    "simulation.run_molecular_dynamics",
    version="1.0",
    family="investigate",
    summary="Run molecular dynamics on a prepared system with staged equilibration",
    render=render,
    environment="openmm",
)
def run_molecular_dynamics(
    request: DynamicsRequest, ctx: StepContext
) -> DynamicsResult:
    """Equilibrate in stages, check for drift, then run production."""
    from alambiq.capabilities.investigate.science.simulation import (
        check_dynamics,
        platform_properties,
    )

    nve = request.ensemble == "nve"
    thermalize_ps = (
        request.thermalize_ps
        if request.thermalize_ps is not None
        else (0.0 if nve else 50.0)
    )
    release_ps = (
        request.release_ps
        if request.release_ps is not None
        else (100.0 if request.ensemble == "npt" else 0.0)
    )
    readiness_window_ps = (
        request.readiness_window_ps
        if request.readiness_window_ps is not None
        else (0.0 if nve else 200.0)
    )
    check_dynamics(
        ensemble=request.ensemble,
        temperature_kelvin=request.temperature_kelvin,
        friction_per_ps=request.friction_per_ps,
        timestep_fs=request.timestep_fs,
        pressure_bar=request.pressure_bar,
        barostat_interval=request.barostat_interval,
        minimize_iterations=request.minimize_iterations,
        thermalize_ps=thermalize_ps,
        release_ps=release_ps,
        restraint_kj_mol_nm2=request.restraint_kj_mol_nm2,
        readiness_window_ps=readiness_window_ps,
        max_readiness_windows=request.max_readiness_windows,
        production_ps=request.production_ps,
        report_interval=request.report_interval,
        seed=request.seed,
    )
    if request.ensemble == "nve" and request.velocities == "maxwell-boltzmann":
        ctx.note(
            "NVE with fresh Maxwell-Boltzmann velocities starts from a random kinetic "
            "energy."
        )
    output = ctx.workdir / "dynamics"
    found = ctx.run_in_environment(
        "openmm",
        "openmm_md:run",
        {
            "system_xml": str(ctx.resolve(request.system)),
            "state_xml": str(ctx.resolve(request.state)),
            "topology_pdb": str(ctx.resolve(request.topology)),
            "ensemble": request.ensemble,
            "temperature_kelvin": request.temperature_kelvin,
            "friction_per_ps": request.friction_per_ps,
            "timestep_fs": request.timestep_fs,
            "pressure_bar": request.pressure_bar,
            "barostat_interval": request.barostat_interval,
            "minimize_iterations": request.minimize_iterations,
            "velocities": request.velocities,
            "thermalize_ps": thermalize_ps,
            "restraint_kj_mol_nm2": request.restraint_kj_mol_nm2,
            "release_ps": release_ps,
            "readiness_window_ps": readiness_window_ps,
            "max_readiness_windows": request.max_readiness_windows,
            "require_ready": request.require_ready,
            "production_ps": request.production_ps,
            "report_interval": request.report_interval,
            "platform": request.platform,
            "platform_properties": platform_properties(
                request.platform, request.platform_properties
            ),
            "seed": request.seed,
            "output": str(output),
        },
        timeout=7 * 24 * 3600,
    )

    ready = found["ready_for_production"]
    if not ready:
        ctx.note(
            "No readiness window passed, so the temperature, density or energy was still "
            "drifting. "
            + (
                found["production_skipped"]
                if "production_skipped" in found
                else "Production was run regardless of the readiness result."
            )
        )
    stats = found["production"]
    production_run = stats.get("samples", 0) > 0

    def keep(name: str, media: str) -> str | None:
        path = output / name
        return str(ctx.keep(path, media).path) if path.is_file() else None

    def mean(key: str) -> float | None:
        return stats[key]["mean"] if key in stats else None

    energy = stats.get("total_energy_kj_mol")
    phases = [
        Phase(
            name=p["name"],
            steps=p.get("steps"),
            detail={k: v for k, v in p.items() if k not in ("name", "steps")},
        )
        for p in found["phases"]
    ]
    return DynamicsResult(
        ensemble=found["ensemble"],
        particles=found["particles"],
        phases=phases,
        ready_for_production=ready,
        production_run=production_run,
        frames=found["frames"],
        final_step=found["final"]["step"],
        final_time_ps=found["final"]["time_ps"],
        mean_temperature_kelvin=stats["temperature_kelvin"]["mean"]
        if production_run
        else None,
        mean_volume_nm3=mean("volume_nm3"),
        mean_density_g_ml=mean("density_g_ml"),
        total_energy_first_kj_mol=energy["first"] if energy else None,
        total_energy_last_kj_mol=energy["last"] if energy else None,
        total_energy_min_kj_mol=energy["min"] if energy else None,
        total_energy_max_kj_mol=energy["max"] if energy else None,
        max_force_kj_mol_nm=found["final"]["forces"]["maximum_atom_kj_mol_nm"],
        trajectory=keep("trajectory.dcd", "application/octet-stream"),
        state_data=keep("state_data.csv", "text/csv"),
        final_state=str(ctx.keep(output / "final_state.xml", "application/xml").path),
        final_structure=str(ctx.keep(output / "final.pdb", "chemical/x-pdb").path),
        summary=ctx.keep_json(found),
        seed=request.seed,
        platform=found["platform"],
        versions=found["versions"],
    )
