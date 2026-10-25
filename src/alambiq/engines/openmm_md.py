"""Run molecular dynamics on a prepared OpenMM system.

Runs inside the openmm environment. The standard library is imported at the top;
numpy and OpenMM are loaded when a request runs. A request names a serialised
system, state and topology (as `structure.prepare_protein_system` writes them),
an ensemble, and the length of each phase in picoseconds. The phases, in order:

1. optional minimization;
2. velocities, from a Maxwell-Boltzmann draw at the target temperature or from the
   supplied state;
3. thermalisation: Langevin dynamics at constant volume, with harmonic restraints
   on the protein heavy atoms if asked;
4. release (constant pressure, explicit solvent): the restraints fall linearly to
   zero while the barostat relaxes the box;
5. readiness windows: unrestrained dynamics assessed in four blocks for drift in
   temperature, density and potential energy, repeated up to a limit until one
   passes;
6. production, with a trajectory and thermodynamic data written.

Equilibration is judged on solvent and thermal drift, so a passing window indicates
temperature, density and potential-energy stability.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import openmm_common as diagnostics
from openmm_xml import deserialize_data

np = openmm = app = unit = None

RESTRAINT_PARAMETER = "alambiq_restraint_k"
SOLVENT = {"HOH", "WAT", "NA", "CL"}
STATE_COLUMNS = (
    "step,time_ps,potential_kj_mol,kinetic_kj_mol,total_kj_mol,temperature_kelvin"
)
TOLERANCES = {
    "temperature_relative": 0.05,
    "temperature_block_relative": 0.03,
    "density_block_relative": 0.02,
    "energy_block_kj_mol_per_atom": 0.1,
}


def _load() -> None:
    global np, openmm, app, unit
    import numpy
    import openmm as _openmm
    from openmm import app as _app
    from openmm import unit as _unit

    np, openmm, app, unit = numpy, _openmm, _app, _unit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _steps(picoseconds: float, timestep_fs: float, interval: int) -> int:
    """Whole report intervals that cover at least the requested time."""
    if picoseconds <= 0:
        return 0
    return math.ceil(picoseconds * 1000 / timestep_fs / interval) * interval


def _sample(simulation, periodic: bool) -> list[int | float]:
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    dof = (
        3 * simulation.system.getNumParticles() - simulation.system.getNumConstraints()
    )
    if any(
        type(f).__name__ == "CMMotionRemover" for f in simulation.system.getForces()
    ):
        dof -= 3
    kelvin = 2 * kinetic / (dof * 0.00831446261815324) if dof > 0 else 0.0
    row = [
        simulation.currentStep,
        simulation.context.getTime().value_in_unit(unit.picosecond),
        potential,
        kinetic,
        potential + kinetic,
        kelvin,
    ]
    if periodic:
        box = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
        row.append(float(np.linalg.det(box.value_in_unit(unit.nanometer))))
    return [int(row[0]), *[float(v) for v in row[1:]]]


def _run_steps(
    simulation, steps: int, interval: int, periodic: bool, rows=None, writer=None
):
    """Advance in report intervals, recording a state sample after each."""
    for _ in range(steps // interval):
        simulation.step(interval)
        row = _sample(simulation, periodic)
        if not all(math.isfinite(v) for v in row):
            raise ValueError("The simulation produced a non-finite energy or box")
        if rows is not None:
            rows.append(row)
        if writer is not None:
            writer.writerow(row)


def assess(rows, temperature, mass, atom_count):
    """Four block means expose gross thermal, density and energy drift."""
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or len(data) < 20:
        raise ValueError("A readiness window needs at least 20 state samples")
    series = {
        "temperature_kelvin": data[:, 5],
        "potential_energy_kj_mol_per_atom": data[:, 2] / atom_count,
    }
    if mass is not None:
        series["density_g_ml"] = mass * 0.00166053906660 / data[:, 6]
    blocks = {
        k: [float(b.mean()) for b in np.array_split(v, 4)] for k, v in series.items()
    }
    observed = {
        "temperature_relative_error": abs(
            float(series["temperature_kelvin"].mean()) - temperature
        )
        / temperature,
        "temperature_block_relative_range": (
            max(blocks["temperature_kelvin"]) - min(blocks["temperature_kelvin"])
        )
        / temperature,
        "energy_block_range_kj_mol_per_atom": max(
            blocks["potential_energy_kj_mol_per_atom"]
        )
        - min(blocks["potential_energy_kj_mol_per_atom"]),
    }
    limits = {
        "temperature_relative_error": TOLERANCES["temperature_relative"],
        "temperature_block_relative_range": TOLERANCES["temperature_block_relative"],
        "energy_block_range_kj_mol_per_atom": TOLERANCES[
            "energy_block_kj_mol_per_atom"
        ],
    }
    if mass is not None:
        observed["density_block_relative_range"] = (
            max(blocks["density_g_ml"]) - min(blocks["density_g_ml"])
        ) / float(series["density_g_ml"].mean())
        limits["density_block_relative_range"] = TOLERANCES["density_block_relative"]
    failed = [k for k, v in observed.items() if v > limits[k]]
    return {
        "samples": len(data),
        "block_means": blocks,
        "observed": observed,
        "limits": limits,
        "failed": failed,
        "passed": not failed,
    }


def _statistics(rows, mass):
    data = np.asarray(rows, dtype=float)
    if data.size == 0:
        return {"samples": 0}
    out = {
        "samples": len(data),
        "temperature_kelvin": {
            "mean": float(data[:, 5].mean()),
            "sd": float(data[:, 5].std()),
        },
        "potential_energy_kj_mol": {
            "mean": float(data[:, 2].mean()),
            "sd": float(data[:, 2].std()),
        },
        "total_energy_kj_mol": {
            "first": float(data[0, 4]),
            "last": float(data[-1, 4]),
            "min": float(data[:, 4].min()),
            "max": float(data[:, 4].max()),
        },
    }
    if data.shape[1] > 6:
        out["volume_nm3"] = {
            "mean": float(data[:, 6].mean()),
            "sd": float(data[:, 6].std()),
        }
        if mass is not None:
            density = mass * 0.00166053906660 / data[:, 6]
            out["density_g_ml"] = {
                "mean": float(density.mean()),
                "sd": float(density.std()),
            }
    return out


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run the phases and write the trajectory, state data and final state."""
    _load()
    started = time.perf_counter()
    cfg = request
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)

    system = deserialize_data(
        Path(cfg["system_xml"]).read_text(encoding="utf-8"),
        "System",
        openmm.XmlSerializer.deserialize,
    )
    state = deserialize_data(
        Path(cfg["state_xml"]).read_text(encoding="utf-8"),
        "State",
        openmm.XmlSerializer.deserialize,
    )
    topology = app.PDBFile(str(cfg["topology_pdb"])).topology
    ensemble = cfg["ensemble"]
    explicit = system.usesPeriodicBoundaryConditions()
    if topology.getNumAtoms() != system.getNumParticles():
        raise ValueError("The topology and the system have different numbers of atoms")
    if ensemble in ("npt", "nvt") and not explicit:
        raise ValueError(
            f"The {ensemble} ensemble needs a periodic system; this one is not"
        )
    if ensemble == "implicit" and explicit:
        raise ValueError(
            "The implicit ensemble needs a non-periodic system; this one is periodic"
        )
    if system.getNumParticles() > 250000:
        raise ValueError(
            f"The system has {system.getNumParticles()} particles; at most 250,000 are supported"
        )
    diagnostics.validate_hydrogen_timestep(
        system, topology, cfg["timestep_fs"], unit, app
    )
    for force in system.getForces():
        if type(force).__name__ in (
            "MonteCarloBarostat",
            "MonteCarloAnisotropicBarostat",
        ):
            raise ValueError(
                "The system already contains a barostat; supply the unmodified prepared system"
            )
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    if (
        positions.shape != (system.getNumParticles(), 3)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("The state positions are missing, have the wrong shape or are not finite")

    temperature = float(cfg["temperature_kelvin"]) * unit.kelvin
    dt = float(cfg["timestep_fs"]) * unit.femtosecond
    interval = int(cfg["report_interval"])
    seed = int(cfg["seed"])
    mass = sum(
        system.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(system.getNumParticles())
    )
    mass = mass if explicit else None

    protein = [
        a.index
        for a in topology.atoms()
        if a.residue.name not in SOLVENT
        and a.element is not None
        and a.element != app.element.hydrogen
    ]
    restrained = (
        ensemble != "nve"
        and float(cfg["restraint_kj_mol_nm2"]) > 0
        and cfg["thermalize_ps"] + cfg["release_ps"] > 0
    )
    if restrained:
        restraint = openmm.CustomExternalForce(
            f"0.5*{RESTRAINT_PARAMETER}*periodicdistance(x,y,z,x0,y0,z0)^2"
            if explicit
            else f"0.5*{RESTRAINT_PARAMETER}*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        restraint.addGlobalParameter(
            RESTRAINT_PARAMETER, float(cfg["restraint_kj_mol_nm2"])
        )
        for name in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(name)
        for index in protein:
            restraint.addParticle(index, [float(v) for v in positions[index]])
        system.addForce(restraint)
    barostat = None
    if ensemble == "npt":
        barostat = openmm.MonteCarloBarostat(
            float(cfg["pressure_bar"]) * unit.bar,
            temperature,
            int(cfg["barostat_interval"]),
        )
        barostat.setRandomNumberSeed(seed)
        barostat.setFrequency(0)
        system.addForce(barostat)

    if ensemble == "nve":
        integrator = openmm.VerletIntegrator(dt)
    else:
        integrator = openmm.LangevinMiddleIntegrator(
            temperature, float(cfg["friction_per_ps"]) / unit.picosecond, dt
        )
        integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(cfg["platform"])
    simulation = app.Simulation(
        topology, system, integrator, platform, cfg["platform_properties"]
    )
    simulation.context.setState(state)
    if cfg["velocities"] == "maxwell-boltzmann":
        simulation.context.setVelocitiesToTemperature(temperature, seed)
    simulation.context.setTime(0.0 * unit.picosecond)
    simulation.currentStep = 0

    summary: dict[str, Any] = {
        "ensemble": ensemble,
        "particles": system.getNumParticles(),
        "periodic": explicit,
        "platform": platform.getName(),
        "tolerances": TOLERANCES,
        "phases": [],
        "inputs": {
            key: _sha256(cfg[key])
            for key in ("system_xml", "state_xml", "topology_pdb")
        },
    }
    if cfg["minimize_iterations"] > 0:
        before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        simulation.minimizeEnergy(maxIterations=int(cfg["minimize_iterations"]))
        after = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        summary["phases"].append(
            {
                "name": "minimization",
                "potential_before_kj_mol": before.value_in_unit(
                    unit.kilojoule_per_mole
                ),
                "potential_after_kj_mol": after.value_in_unit(unit.kilojoule_per_mole),
            }
        )

    def block(picoseconds):
        return _steps(picoseconds, cfg["timestep_fs"], interval)

    thermal_steps = block(cfg["thermalize_ps"])
    if thermal_steps:
        rows: list[list[float]] = []
        _run_steps(simulation, thermal_steps, interval, explicit, rows)
        summary["phases"].append(
            {
                "name": "thermalization",
                "ensemble": "nvt" if explicit else "implicit",
                "steps": thermal_steps,
                "restraint_kj_mol_nm2": float(cfg["restraint_kj_mol_nm2"])
                if restrained
                else 0.0,
                "statistics": _statistics(rows, mass),
            }
        )
    if barostat is not None:
        barostat.setFrequency(int(cfg["barostat_interval"]))
        simulation.context.reinitialize(preserveState=True)
    release_steps = block(cfg["release_ps"])
    if release_steps and ensemble == "npt":
        rows = []
        pieces = 20
        chunk = max(interval, (release_steps // pieces) // interval * interval)
        done = 0
        while done < release_steps:
            fraction = done / release_steps
            if restrained:
                simulation.context.setParameter(
                    RESTRAINT_PARAMETER,
                    float(cfg["restraint_kj_mol_nm2"]) * (1 - fraction),
                )
            take = min(chunk, release_steps - done)
            _run_steps(simulation, take, interval, explicit, rows)
            done += take
        summary["phases"].append(
            {
                "name": "release",
                "ensemble": "npt",
                "steps": release_steps,
                "statistics": _statistics(rows, mass),
            }
        )
    if restrained:
        simulation.context.setParameter(RESTRAINT_PARAMETER, 0.0)

    ready = None
    window_steps = block(cfg["readiness_window_ps"])
    if window_steps and ensemble != "nve":
        windows = []
        for number in range(1, int(cfg["max_readiness_windows"]) + 1):
            rows = []
            _run_steps(simulation, window_steps, interval, explicit, rows)
            verdict = assess(
                rows, float(cfg["temperature_kelvin"]), mass, system.getNumParticles()
            )
            windows.append({"window": number, "steps": window_steps, **verdict})
            if verdict["passed"]:
                break
        ready = windows[-1]["passed"]
        summary["phases"].append(
            {"name": "readiness", "windows": windows, "ready": ready}
        )
    summary["ready_for_production"] = True if ready is None else ready

    production_steps = block(cfg["production_ps"])
    stats: dict[str, Any] = {"samples": 0}
    frames = 0
    if production_steps and (ready is None or ready or not cfg["require_ready"]):
        if production_steps // interval > 20000:
            raise ValueError(
                "Production would write more than 20,000 frames; increase report_interval"
            )
        trajectory = out / "trajectory.dcd"
        simulation.reporters.append(app.DCDReporter(str(trajectory), interval))
        rows = []
        with (out / "state_data.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                STATE_COLUMNS.split(",") + (["volume_nm3"] if explicit else [])
            )
            first = _sample(simulation, explicit)
            rows.append(first)
            writer.writerow(first)
            _run_steps(simulation, production_steps, interval, explicit, rows, writer)
        simulation.reporters.clear()
        frames = production_steps // interval
        stats = _statistics(rows, mass)
        summary["phases"].append(
            {
                "name": "production",
                "steps": production_steps,
                "frames": frames,
                "statistics": stats,
            }
        )
    elif production_steps:
        summary["production_skipped"] = (
            "The readiness windows did not pass, so production was not run. "
            "Set require_ready to false to run it anyway."
        )

    final = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, getForces=True
    )
    (out / "final_state.xml").write_text(openmm.XmlSerializer.serialize(final))
    with (out / "final.pdb").open("w") as stream:
        app.PDBFile.writeFile(topology, final.getPositions(), stream, keepIds=True)
    summary["final"] = {
        "step": simulation.currentStep,
        "time_ps": simulation.context.getTime().value_in_unit(unit.picosecond),
        "forces": diagnostics.force_diagnostics(final, unit, topology),
        "cell": diagnostics.cell_diagnostics(final, unit, 0.9) if explicit else None,
    }
    summary["production"] = stats
    summary["frames"] = frames
    summary["seconds"] = time.perf_counter() - started
    summary["versions"] = {
        "openmm": openmm.version.short_version,
        "numpy": np.__version__,
    }
    summary["platform_properties"] = {
        key: platform.getPropertyValue(simulation.context, key)
        for key in platform.getPropertyNames()
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
