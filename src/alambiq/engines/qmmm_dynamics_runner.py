"""Native constant-temperature molecular QM/MM sampling from prepared data."""

import copy
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "alambiq_qmmm_coupling", Path(__file__).with_name("qmmm_coupling.py")
)
coupling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coupling)
runner = coupling.runner


def sampling_settings(value):
    """Resolve explicit sampling choices independently of electronic parameters."""
    required = {
        "temperature_kelvin",
        "friction_per_ps",
        "timestep_fs",
        "seed",
        "equilibration_steps",
        "production_steps",
        "report_interval",
        "umbrella",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "Supply temperature, friction, timestep, seed, equilibration/production "
            "steps, report interval and optional umbrella explicitly"
        )
    for key in ("temperature_kelvin", "friction_per_ps", "timestep_fs"):
        if (
            type(value[key]) not in (int, float)
            or not math.isfinite(value[key])
            or value[key] <= 0
        ):
            raise ValueError(f"{key} requires a positive finite number")
    for key, low in (
        ("seed", 1),
        ("equilibration_steps", 0),
        ("production_steps", 1),
        ("report_interval", 1),
    ):
        if type(value[key]) is not int or not low <= value[key] <= 2147483647:
            raise ValueError(f"{key} requires an integer in [{low}, 2147483647]")
    if value["production_steps"] % value["report_interval"]:
        raise ValueError("Production steps must contain complete reporting intervals")
    umbrella = value["umbrella"]
    if umbrella is not None:
        if not isinstance(umbrella, dict) or set(umbrella) != {
            "pairs",
            "center_angstrom",
            "coefficient_kj_mol_angstrom2",
        }:
            raise ValueError(
                "An umbrella requires ordered pairs, center and coefficient"
            )
        coupling._pairs(umbrella["pairs"], 2147483647)
        for key in ("center_angstrom", "coefficient_kj_mol_angstrom2"):
            v = umbrella[key]
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError("Finite umbrella center and coefficient are required")
        if umbrella["coefficient_kj_mol_angstrom2"] <= 0:
            raise ValueError("The umbrella coefficient must be positive")
    return copy.deepcopy(value)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sample(request, output):
    """Run one seeded LangevinMiddle segment and retain every electronic call."""
    import numpy as np
    import openmm
    from openmm import app, unit

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = integrator = None
    try:
        runner._record(output / "request.json", request)
        if not isinstance(request, dict) or set(request) != {"prepared", "sampling"}:
            raise ValueError(
                "Supply one prepared molecular request and explicit sampling settings"
            )
        settings = sampling_settings(request["sampling"])
        source = request["prepared"]
        paths = {
            k: source[k]
            for k in (
                "system_xml",
                "state_xml",
                "topology_pdb",
                "external_system_xml",
                "initial_density",
            )
            if source[k] is not None
        }
        identities = {k: runner._sha(p) for k, p in paths.items()}
        prepared = runner.prepare(source)

        def exact_sources():
            if any(runner._sha(p) != identities[k] for k, p in paths.items()):
                raise ValueError(
                    "Every prepared source must be unchanged throughout sampling"
                )

        exact_sources()
        runner._record(output / "atom-map.json", prepared["atom_labels"])
        (output / "classical-system.xml").write_text(
            openmm.XmlSerializer.serialize(prepared["classical_system"])
        )
        initial = None
        if source["initial_density"] is not None:
            with np.load(source["initial_density"], allow_pickle=False) as saved:
                initial = saved["density"].copy()
        evaluator = coupling.MolecularElectronicForce(
            prepared, source["parameters"], output / "calls", initial
        )
        system, cache = coupling.compose_system(
            prepared, evaluator, settings["umbrella"]
        )
        integrator = openmm.LangevinMiddleIntegrator(
            settings["temperature_kelvin"] * unit.kelvin,
            settings["friction_per_ps"] / unit.picosecond,
            settings["timestep_fs"] * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(settings["seed"])
        integrator.setConstraintTolerance(1e-10)
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(prepared["positions_angstrom"] * unit.angstrom)
        context.applyConstraints(1e-10)
        context.setVelocitiesToTemperature(
            settings["temperature_kelvin"], settings["seed"]
        )
        context.applyVelocityConstraints(1e-10)
        topology = app.PDBFile(str(source["topology_pdb"])).topology
        atoms = system.getNumParticles()
        dof = 3 * atoms - system.getNumConstraints()
        if dof <= 0:
            raise ValueError(
                "The molecular system requires positive unconstrained degrees of freedom"
            )
        gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
            unit.kilojoule_per_mole / unit.kelvin
        )
        columns = [
            "phase",
            "step",
            "production_step",
            "time_fs",
            "electronic_call",
            "coordinate_angstrom",
            "potential_kj_mol",
            "kinetic_kj_mol",
            "total_kj_mol",
            "classical_kj_mol",
            "external_kj_mol",
            "umbrella_kj_mol",
            "electronic_kj_mol",
            "unbiased_potential_kj_mol",
            "temperature_kelvin",
            "maximum_constraint_error_angstrom",
        ]
        last = None
        frames = 0
        with (
            (output / "state.csv").open("x", newline="") as table,
            (output / "trajectory.dcd").open("xb") as stream,
        ):
            writer = csv.DictWriter(table, fieldnames=columns)
            writer.writeheader()
            dcd = app.DCDFile(
                stream,
                topology,
                settings["timestep_fs"] * unit.femtosecond,
                firstStep=settings["equilibration_steps"] + settings["report_interval"],
                interval=settings["report_interval"],
            )

            def observe(phase, step):
                nonlocal frames
                state = context.getState(
                    getEnergy=True, getPositions=True, getVelocities=True
                )
                xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                velocity = state.getVelocities(asNumpy=True).value_in_unit(
                    unit.nanometer / unit.picosecond
                )
                potential = state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
                kinetic = state.getKineticEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
                energies = [
                    context.getState(getEnergy=True, groups={group})
                    .getPotentialEnergy()
                    .value_in_unit(unit.kilojoule_per_mole)
                    for group in range(4)
                ]
                if (
                    xyz.shape != (atoms, 3)
                    or velocity.shape != xyz.shape
                    or not np.isfinite(xyz).all()
                    or not np.isfinite(velocity).all()
                    or not np.isfinite([potential, kinetic, *energies]).all()
                    or not math.isclose(
                        sum(energies), potential, rel_tol=0, abs_tol=1e-7
                    )
                ):
                    raise ValueError(
                        "Finite complete positions, velocities and reconciled energy components are required"
                    )
                errors = []
                for i in range(system.getNumConstraints()):
                    a, b, distance = system.getConstraintParameters(i)
                    errors.append(
                        abs(
                            np.linalg.norm(xyz[a] - xyz[b])
                            - distance.value_in_unit(unit.nanometer)
                        )
                        * 10
                    )
                row = dict(
                    zip(
                        columns,
                        [
                            phase,
                            step,
                            max(0, step - settings["equilibration_steps"]),
                            step * settings["timestep_fs"],
                            evaluator.calls - 1,
                            coupling.coordinate_angstrom(
                                xyz, settings["umbrella"]["pairs"]
                            )
                            if settings["umbrella"]
                            else "",
                            potential,
                            kinetic,
                            potential + kinetic,
                            *energies,
                            potential - energies[2],
                            2 * kinetic / (dof * gas_constant),
                            max(errors, default=0.0),
                        ],
                        strict=True,
                    )
                )
                writer.writerow(row)
                table.flush()
                if phase == "production":
                    dcd.writeModel(xyz * unit.nanometer)
                    stream.flush()
                    frames += 1
                return state, row

            last, _ = observe("initial", 0)
            step = 0
            for phase, length in (
                ("equilibration", settings["equilibration_steps"]),
                ("production", settings["production_steps"]),
            ):
                if phase == "production":
                    (output / "production-start-state.xml").write_text(
                        openmm.XmlSerializer.serialize(last)
                    )
                remaining = length
                while remaining:
                    count = min(remaining, settings["report_interval"])
                    integrator.step(count)
                    step += count
                    remaining -= count
                    last, final = observe(phase, step)
        exact_sources()
        (output / "final-state.xml").write_text(openmm.XmlSerializer.serialize(last))
        (output / "checkpoint.bin").write_bytes(context.createCheckpoint())
        np.savez(output / "final-density.npz", density=evaluator.density)
        p = source["parameters"]
        hamiltonian = {
            "system_sha256": identities["system_xml"],
            "atom_map_sha256": runner._sha(output / "atom-map.json"),
            "external_system_sha256": identities.get("external_system_xml"),
            "quantum": source["quantum"],
            "method": "PBE",
            "basis": "6-31g**",
            "spin": 0,
            **{
                k: p[k]
                for k in ("grid", "gaussian_radius_angstrom", "classical_coulomb_ratio")
            },
        }
        result = {
            "schema_version": "0.1",
            "quantity": "molecular-qmmm-trajectory",
            "source_sha256": identities,
            "parameters": p,
            "sampling": settings,
            "quantum": source["quantum"],
            "physical_atoms": atoms,
            "electronic_atoms": len(prepared["electronic_elements"]),
            "classical_ledger": prepared["ledger"],
            "hamiltonian": hamiltonian,
            "hamiltonian_sha256": _digest(hamiltonian),
            "platform": "Reference",
            "integrator": "LangevinMiddle",
            "constraint_tolerance": 1e-10,
            "constraints": system.getNumConstraints(),
            "kinetic_degrees_of_freedom": dof,
            "initialization": "Source positions projected to retained classical constraints; independent seeded thermal velocities and stochastic stream; native time starts at zero",
            "velocity_convention": "Native LangevinMiddle half-step velocities; kinetic energy uses the native integrator expression",
            "ensemble": "NVT",
            "equilibration_steps_completed": settings["equilibration_steps"],
            "production_steps_completed": settings["production_steps"],
            "production_frames": frames,
            "frame_interval_fs": settings["report_interval"] * settings["timestep_fs"],
            "electronic_calls": evaluator.calls,
            "cache_hits": cache.hits,
            "final_total_energy_kj_mol": final["total_kj_mol"],
            "outputs_sha256": {
                p.name: runner._sha(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        }
        runner._record(output / "summary.json", result)
        return result
    except Exception as exc:
        runner._record(
            output / "failure.json",
            {"exception": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        context = integrator = None


def run(cfg):
    """Engine entry: run one segment; `cfg` is the request plus `output`."""
    return sample({k: v for k, v in cfg.items() if k != "output"}, cfg["output"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Supply a native sampling request and a new output directory")
    sample(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
