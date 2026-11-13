"""Diagnostic probe, not part of the test suite: are the CPU forces of a prepared system reproducible?

Prepares the complex of the rerun test once, then in the OpenFF engine environment
(1) evaluates the forces 12 times in one context, (2) evaluates them in 12 fresh contexts,
(3) minimizes from the same start in two fresh contexts, all with one thread and
DeterministicForces on, and prints a JSON summary.
"""

import json
import os
import platform
import subprocess
from pathlib import Path

from core.test_structure_prepare_predicted_complex_for_dynamics import (  # noqa: F401
    ENV,
    SEQUENCE,
    complex_files,
    request,
    run,
)

PROBE = r"""
import json, os, sys
import numpy as np
import openmm
from openmm import app, unit

directory = sys.argv[1]
prmtop = app.AmberPrmtopFile(directory + "/complex.prmtop")
inpcrd = app.AmberInpcrdFile(directory + "/complex.inpcrd")
if inpcrd.boxVectors is not None:
    prmtop.topology.setPeriodicBoxVectors(inpcrd.boxVectors)

def simulation():
    system = prmtop.createSystem(
        nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
        constraints=app.HBonds, rigidWater=True,
    )
    sim = app.Simulation(
        prmtop.topology, system, openmm.VerletIntegrator(1.0 * unit.femtosecond),
        openmm.Platform.getPlatformByName("CPU"), {"Threads": "1", "DeterministicForces": "true"},
    )
    sim.context.setPositions(inpcrd.positions)
    sim.context.applyConstraints(1e-8)
    return sim

def forces(sim):
    state = sim.context.getState(getForces=True)
    return state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

def summary(arrays):
    first = arrays[0]
    return {
        "arrays": len(arrays),
        "distinct": len({a.tobytes() for a in arrays}),
        "max_abs_difference": max(float(np.abs(a - first).max()) for a in arrays),
    }

out = {"openmm": openmm.version.version, "atoms": prmtop.topology.getNumAtoms()}
sim = simulation()
out["platform_properties"] = {
    k: sim.context.getPlatform().getPropertyValue(sim.context, k)
    for k in sim.context.getPlatform().getPropertyNames()
}
out["one_context_12_evaluations"] = summary([forces(sim) for _ in range(12)])
out["twelve_fresh_contexts"] = summary([forces(simulation()) for _ in range(12)])

positions, energies = [], []
for _ in range(2):
    sim = simulation()
    openmm.LocalEnergyMinimizer.minimize(sim.context, maxIterations=200)
    state = sim.context.getState(getPositions=True, getEnergy=True)
    positions.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    energies.append(float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)))
out["two_minimizations_200_iterations"] = {
    "energies_kj_mol": energies,
    "positions_identical": bool(positions[0].tobytes() == positions[1].tobytes()),
    "max_abs_position_difference_nm": float(np.abs(positions[0] - positions[1]).max()),
}
print("PROBE " + json.dumps(out))
"""


def test_probe(complex_files, tmp_path_factory):
    os.environ["ALAMBIQ_ENV_OPENFF"] = ENV
    root = tmp_path_factory.mktemp("probe")
    outcome = run(
        request(
            complex_files,
            protein_sequences=[SEQUENCE],
            ligand_formal_charge=-1,
            box_shape="dodecahedron",
        ),
        root,
    )
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    prmtops = sorted(root.rglob("complex.prmtop"))
    assert prmtops, "the preparation left no complex.prmtop in its work directory"
    cpu = ""
    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    except OSError:
        pass
    keep = [l for l in cpu.splitlines() if l.split(":")[0] in ("Architecture", "Model name", "CPU(s)", "Thread(s) per core", "Vendor ID")]
    flags = [l for l in cpu.splitlines() if l.startswith("Flags")]
    finished = subprocess.run(
        [ENV, "-c", PROBE, str(prmtops[0].parent)], capture_output=True, text=True
    )
    print("MACHINE", platform.machine(), os.cpu_count(), keep)
    print("SIMD", [w for w in (flags[0].split() if flags else []) if w in ("avx", "avx2", "avx512f", "fma", "sse4_2", "neon", "asimd")])
    print(finished.stdout[-3000:])
    print(finished.stderr[-1500:])
    assert finished.returncode == 0
