"""Well-tempered torsion metadynamics on a prepared OpenMM system.

Runs inside the openmm environment. The standard library is imported at the top;
numpy and OpenMM are loaded when a request runs. One or two periodic torsions are
biased with OpenMM's own `Metadynamics`. The run has three phases:

1. adaptation: Gaussians are deposited, and the free-energy estimate they imply
   is kept at quarter intervals so its settling can be seen;
2. frozen equilibration: the bias stays as it was at the end of adaptation (plain
   dynamics keeps the tabulated bias, which only `Metadynamics.step` updates);
3. production under that fixed bias, recording each torsion and the bias energy at
   every report. Because the bias no longer changes, the samples can be reweighted
   exactly: frame i has weight exp(+beta * bias_i), and the weighted histogram is
   the unbiased distribution of the torsions.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import openmm_common as diagnostics
from openmm_xml import deserialize_data

np = openmm = app = unit = None

FORCE_NAME = "AlambiqTorsionMetadynamics-v1"
GAS_CONSTANT = 0.00831446261815324  # kJ/mol/K


def _load() -> None:
    global np, openmm, app, unit
    import numpy
    import openmm as _openmm
    from openmm import app as _app
    from openmm import unit as _unit

    np, openmm, app, unit = numpy, _openmm, _app, _unit


def _dihedral(p: Any) -> float:
    """The IUPAC dihedral of four points, in (-pi, pi]."""
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _resolve(topology, selectors):
    atoms: dict[tuple, list] = {}
    for atom in topology.atoms():
        residue = atom.residue
        key = (residue.chain.id, residue.id, residue.insertionCode.strip(), atom.name)
        atoms.setdefault(key, []).append(atom)
    indices = []
    for selector in selectors:
        key = (
            selector["chain_id"],
            selector["residue_id"],
            selector["insertion_code"],
            selector["atom_name"],
        )
        found = atoms.get(key, [])
        if len(found) != 1:
            raise ValueError(
                f"Torsion atom must match exactly once: {key}; found {len(found)}"
            )
        indices.append(found[0].index)
    bonds = {frozenset((a.index, b.index)) for a, b in topology.bonds()}
    if any(
        frozenset(pair) not in bonds for pair in zip(indices, indices[1:], strict=False)
    ):
        raise ValueError("Torsion atoms must form a bonded three-bond path")
    return indices


def _histogram(angles, weights, bins, dims):
    edges = np.linspace(-math.pi, math.pi, bins + 1)
    if dims == 1:
        counts, _ = np.histogram(angles[:, 0], bins=edges, weights=weights)
    else:
        counts, _, _ = np.histogram2d(
            angles[:, 0], angles[:, 1], bins=[edges, edges], weights=weights
        )
    return counts, edges


def _free_energy(counts, kt):
    total = counts.sum()
    if total <= 0:
        raise ValueError("No production samples are available to build a free-energy surface")
    with np.errstate(divide="ignore"):
        f = -kt * np.log(counts / total)
    finite = np.isfinite(f)
    f = np.where(finite, f - f[finite].min(), np.nan)
    return f


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run adaptation, frozen equilibration and fixed-bias production."""
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
    if topology.getNumAtoms() != system.getNumParticles():
        raise ValueError("The topology and the system have different numbers of atoms")
    explicit = system.usesPeriodicBoundaryConditions()
    ensemble = cfg["ensemble"]
    if ensemble in ("npt", "nvt") and not explicit:
        raise ValueError(
            f"The {ensemble} ensemble needs a periodic system; this one is not"
        )
    if ensemble == "implicit" and explicit:
        raise ValueError(
            "The implicit ensemble needs a non-periodic system; this one is periodic"
        )
    if any(f.getName() == FORCE_NAME for f in system.getForces()):
        raise ValueError(
            "The system already contains a torsion bias; supply the unbiased system"
        )
    diagnostics.validate_hydrogen_timestep(
        system, topology, cfg["timestep_fs"], unit, app
    )
    if (
        cfg["platform"] == "CUDA"
        and cfg["platform_properties"].get("DisablePmeStream") != "true"
    ):
        raise ValueError(
            "CUDA torsion metadynamics requires DisablePmeStream=true; combined PME and bias forces are verified only with it"
        )

    temperature = float(cfg["temperature_kelvin"])
    kt = GAS_CONSTANT * temperature
    seed = int(cfg["seed"])
    variables, mappings = [], []
    for row in cfg["torsions"]:
        indices = _resolve(topology, row["atoms"])
        if any(
            system.getParticleMass(i).value_in_unit(unit.dalton) <= 0 for i in indices
        ):
            raise ValueError("Torsion coordinates require massive particles")
        force = openmm.CustomTorsionForce("theta")
        force.addTorsion(*indices)
        force.setUsesPeriodicBoundaryConditions(explicit)
        variables.append(
            app.BiasVariable(
                force, -math.pi, math.pi, row["width_radians"], True, row["grid_width"]
            )
        )
        mappings.append({**row, "atom_indices_zero_based": indices})
    dims = len(variables)
    bins = int(cfg["histogram_bins"])
    meta = app.Metadynamics(
        system,
        variables,
        temperature * unit.kelvin,
        float(cfg["bias_factor"]),
        float(cfg["height_kj_mol"]) * unit.kilojoule_per_mole,
        int(cfg["deposition_interval"]),
    )
    bias_force = system.getForce(system.getNumForces() - 1)
    bias_force.setName(FORCE_NAME)
    group = bias_force.getForceGroup()

    barostat = None
    if ensemble == "npt":
        barostat = openmm.MonteCarloBarostat(
            float(cfg["pressure_bar"]) * unit.bar,
            temperature * unit.kelvin,
            int(cfg["barostat_interval"]),
        )
        barostat.setRandomNumberSeed(seed)
        system.addForce(barostat)
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        float(cfg["friction_per_ps"]) / unit.picosecond,
        float(cfg["timestep_fs"]) * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(cfg["platform"])
    simulation = app.Simulation(
        topology, system, integrator, platform, cfg["platform_properties"]
    )
    simulation.context.setState(state)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)
    simulation.context.setTime(0.0 * unit.picosecond)
    simulation.currentStep = 0

    interval = int(cfg["report_interval"])
    adaptation = int(cfg["adaptation_steps"])
    quarter = adaptation // 4
    estimates = []
    done = 0
    for fraction in (1, 2, 3, 4):
        target = quarter * fraction if fraction < 4 else adaptation
        meta.step(simulation, target - done)
        done = target
        grid = np.array(meta.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole))
        grid = grid - np.nanmin(grid)
        estimates.append(
            {
                "fraction": fraction / 4,
                "step": done,
                "free_energy_kj_mol": grid.tolist(),
            }
        )
    if cfg["frozen_equilibration_steps"]:
        simulation.step(int(cfg["frozen_equilibration_steps"]))

    production = int(cfg["production_steps"])
    samples = []
    with (out / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        names = [t["name"] for t in cfg["torsions"]]
        writer.writerow(
            ["step", "time_ps", *[f"{n}_radians" for n in names], "bias_energy_kj_mol"]
        )
        for _ in range(production // interval):
            simulation.step(interval)
            positions = simulation.context.getState(getPositions=True).getPositions(
                asNumpy=True
            )
            xyz = positions.value_in_unit(unit.nanometer)
            angles = [_dihedral(xyz[m["atom_indices_zero_based"]]) for m in mappings]
            bias = (
                simulation.context.getState(getEnergy=True, groups={group})
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
            row = [
                simulation.currentStep,
                simulation.context.getTime().value_in_unit(unit.picosecond),
                *angles,
                bias,
            ]
            if not all(math.isfinite(v) for v in row):
                raise ValueError("The simulation produced a non-finite value")
            samples.append(row)
            writer.writerow(row)

    data = np.array(samples)
    angles = data[:, 2 : 2 + dims]
    bias = data[:, -1]
    log_w = bias / kt
    weights = np.exp(log_w - log_w.max())
    ess = float(weights.sum() ** 2 / (weights**2).sum())
    counts, edges = _histogram(angles, weights, bins, dims)
    reweighted = _free_energy(counts, kt)
    half = len(data) // 2
    halves = []
    for part in (slice(0, half), slice(half, None)):
        c, _ = _histogram(angles[part], weights[part], bins, dims)
        halves.append(_free_energy(c, kt))
    both = (
        np.isfinite(halves[0])
        & np.isfinite(halves[1])
        & (halves[0] < 6 * kt)
        & (halves[1] < 6 * kt)
    )
    half_difference = (
        float(np.sqrt(np.mean((halves[0][both] - halves[1][both]) ** 2)))
        if both.any()
        else None
    )
    visited = float(np.mean(counts > 0))
    centres = ((edges[:-1] + edges[1:]) / 2).tolist()

    final = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True
    )
    (out / "final_state.xml").write_text(openmm.XmlSerializer.serialize(final))
    result = {
        "ensemble": ensemble,
        "torsions": mappings,
        "temperature_kelvin": temperature,
        "kt_kj_mol": kt,
        "bias_factor": float(cfg["bias_factor"]),
        "adaptation_estimates": estimates,
        "production": {
            "samples": len(data),
            "steps": production,
            "report_interval": interval,
            "effective_sample_size": ess,
            "effective_sample_fraction": ess / len(data),
            "bias_energy_range_kj_mol": [float(bias.min()), float(bias.max())],
        },
        "free_energy": {
            "axes_radians": centres,
            "bins": bins,
            "reweighted_kj_mol": np.where(
                np.isfinite(reweighted), reweighted, None
            ).tolist(),
            "bins_visited_fraction": visited,
            "half_to_half_rms_kj_mol": half_difference,
            "half_to_half_scope": "Over bins below 6 kT in both halves; the RMS difference between the free-energy profiles of the first and second half of the samples of one run",
        },
        "settings": {
            "friction_per_ps": float(cfg["friction_per_ps"]),
            "timestep_fs": float(cfg["timestep_fs"]),
            "pressure_bar": float(cfg["pressure_bar"]) if ensemble == "npt" else None,
            "barostat_interval": int(cfg["barostat_interval"])
            if ensemble == "npt"
            else None,
            "seed": seed,
        },
        "platform": platform.getName(),
        "platform_properties": {
            key: platform.getPropertyValue(simulation.context, key)
            for key in platform.getPropertyNames()
        },
        "versions": {"openmm": openmm.version.short_version, "numpy": np.__version__},
        "seconds": time.perf_counter() - started,
    }
    (out / "metadynamics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
