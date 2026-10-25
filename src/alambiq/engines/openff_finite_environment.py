"""Complete-molecule finite environments cut from a periodic solvated system.

Runs inside the OpenFF environment (OpenMM, ParmEd and NumPy). Starting from the Amber
topology of a periodic system and an OpenMM state of it, bonds are unwrapped so every
molecule is whole, whole molecules are imaged around an anchor molecule's heavy atom
nearest its mass center, and every solvent or ion molecule with any atom inside a
sphere is kept whole. The kept atoms get a new Amber topology with every parameter
checked to be the source's, a separate OpenMM system of boundary potentials (outer
solute atoms tied to where they were, and a wall on the mass center of each molecule of
solvent or ion), and the positions.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import openmm_common as diagnostics


def physical_identity(system, atom_indices=None):
    """Full native force parameters in particle order, independent of table order."""
    from openmm import unit

    def number(value):
        return (
            value.value_in_unit_system(unit.md_unit_system)
            if unit.is_quantity(value)
            else value
        )

    count = system.getNumParticles()
    selected = list(range(count)) if atom_indices is None else atom_indices
    if (
        not isinstance(selected, list)
        or not selected
        or any(type(i) is not int or not 0 <= i < count for i in selected)
        or selected != sorted(set(selected))
    ):
        raise ValueError("Declare a nonempty ordered subset of source particles")
    inverse = {old: new for new, old in enumerate(selected)}
    forces = {}
    for force in system.getForces():
        kind = type(force).__name__
        if kind in forces:
            raise ValueError(
                "This OpenFF profile requires one force of each native kind"
            )
        if kind in {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"}:
            n, count, get = {
                "HarmonicBondForce": (2, "getNumBonds", "getBondParameters"),
                "HarmonicAngleForce": (3, "getNumAngles", "getAngleParameters"),
                "PeriodicTorsionForce": (4, "getNumTorsions", "getTorsionParameters"),
            }[kind]
            rows = []
            for i in range(getattr(force, count)()):
                values = list(getattr(force, get)(i))
                # Exact zero-amplitude torsions contribute zero energy and force;
                # their native bookkeeping atom permutations can differ.
                if kind == "PeriodicTorsionForce" and number(values[-1]) == 0:
                    continue
                atoms = values[:n]
                included = [i in inverse for i in atoms]
                if any(included) and not all(included):
                    raise ValueError(
                        "Physical projection must retain complete bonded molecules"
                    )
                if all(included):
                    mapped = [inverse[i] for i in atoms]
                    rows.append([*min(mapped, mapped[::-1]), *map(number, values[n:])])
            forces[kind] = sorted(rows)
        elif kind == "NonbondedForce":
            if (
                force.getNonbondedMethod() != force.NoCutoff
                or force.getNumParticleParameterOffsets()
                or force.getNumExceptionParameterOffsets()
                or force.getNumGlobalParameters()
            ):
                raise ValueError(
                    "Source identity requires fixed-charge nonperiodic forces"
                )
            exceptions = []
            for i in range(force.getNumExceptions()):
                a, b, *values = force.getExceptionParameters(i)
                if (a in inverse) != (b in inverse):
                    raise ValueError(
                        "Physical projection must retain complete exception pairs"
                    )
                if a in inverse:
                    exceptions.append(
                        [*sorted([inverse[a], inverse[b]]), *map(number, values)]
                    )
            forces[kind] = {
                "particles": [
                    list(map(number, force.getParticleParameters(i))) for i in selected
                ],
                "exceptions": sorted(exceptions),
            }
        else:
            raise ValueError(
                f"Unsupported source force for this native identity: {kind}"
            )
    if (
        set(forces)
        != {
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "PeriodicTorsionForce",
            "NonbondedForce",
        }
        or system.getNumConstraints()
    ):
        raise ValueError(
            "Source identity requires the complete unconstrained Hamiltonian"
        )
    result = {
        "masses": [number(system.getParticleMass(i)) for i in selected],
        "forces": forces,
    }
    return hashlib.sha256(
        json.dumps(result, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def image_components(xyz, cell, bonds, masses, anchor, heavy):
    import numpy as np

    xyz, cell, masses = (
        np.asarray(xyz, float),
        np.asarray(cell, float),
        np.asarray(masses, float),
    )
    n = len(xyz)
    if (
        xyz.shape != (n, 3)
        or n == 0
        or not np.isfinite(xyz).all()
        or cell.shape != (3, 3)
        or not np.isfinite(cell).all()
        or not np.allclose(cell, np.diag(np.diag(cell)), rtol=0, atol=1e-12)
        or np.any(np.diag(cell) <= 0)
    ):
        raise ValueError(
            "Finite positions and a positive orthorhombic periodic cell are required"
        )
    if masses.shape != (n,) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Every atom requires its positive source mass")
    adjacency = [[] for _ in range(n)]
    for a, b in bonds:
        if (
            type(a) is not int
            or type(b) is not int
            or not 0 <= a < n
            or not 0 <= b < n
            or a == b
        ):
            raise ValueError("Source bonds require distinct valid atom indices")
        adjacency[a].append(b)
        adjacency[b].append(a)
    imaged = np.empty_like(xyz)
    seen, groups = set(), []
    lengths = np.diag(cell)
    for first in range(n):
        if first in seen:
            continue
        imaged[first] = xyz[first]
        seen.add(first)
        stack, group = [first], []
        while stack:
            a = stack.pop()
            group.append(a)
            for b in adjacency[a]:
                v = xyz[b] - xyz[a]
                v -= lengths * np.rint(v / lengths)
                expected = imaged[a] + v
                if b in seen:
                    if not np.allclose(imaged[b], expected, rtol=0, atol=1e-7):
                        raise ValueError(
                            "Source bonds have inconsistent periodic image cycles"
                        )
                else:
                    imaged[b] = expected
                    seen.add(b)
                    stack.append(b)
        groups.append(sorted(group))
    if (
        not anchor
        or anchor != sorted(set(anchor))
        or anchor not in groups
        or not heavy
        or heavy != sorted(set(heavy))
        or not set(heavy).issubset(anchor)
    ):
        raise ValueError(
            "The anchor must be one complete molecule with explicit heavy atoms"
        )
    com = np.average(imaged[anchor], axis=0, weights=masses[anchor])
    center = heavy[int(np.argmin(np.linalg.norm(imaged[heavy] - com, axis=1)))]
    origin = imaged[center].copy()
    for group in groups:
        imaged[group] -= lengths * np.rint(
            (np.average(imaged[group], axis=0, weights=masses[group]) - origin)
            / lengths
        )
    return imaged, groups, center


def select_components(xyz, groups, solute, center, radius):
    import numpy as np

    xyz = np.asarray(xyz, float)
    flat = [i for group in groups for i in group]
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not np.isfinite(xyz).all()
        or sorted(flat) != list(range(len(xyz)))
        or any(not group for group in groups)
        or not 0 <= center < len(xyz)
        or not math.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError(
            "Finite positions, complete atom groups, center and radius are required"
        )
    required = set(solute)
    if not required or len(required) != len(solute) or not required.issubset(flat):
        raise ValueError("Declare distinct valid solute atoms")
    distances = np.linalg.norm(xyz - xyz[center], axis=1)
    selected = []
    for group in groups:
        if required.intersection(group) and not set(group).issubset(required):
            raise ValueError("Solute residues must retain complete bonded molecules")
        if required.intersection(group) or np.min(distances[group]) <= radius:
            selected.extend(group)
    return sorted(selected)


def boundary_system(
    masses, elements, xyz, solute_residues, solvent_groups, protected, center, values
):
    import numpy as np
    import openmm

    xyz, masses, center = (
        np.asarray(xyz, float),
        np.asarray(masses, float),
        np.asarray(center, float),
    )
    elements = np.asarray(elements)
    solute = [i for residue in solute_residues for i in residue]
    solvent = [i for group in solvent_groups for i in group]
    if (
        elements.shape != masses.shape
        or not np.issubdtype(elements.dtype, np.integer)
        or np.any(elements < 1)
        or xyz.shape != (len(masses), 3)
        or not np.isfinite(xyz).all()
        or not np.isfinite(masses).all()
        or np.any(masses <= 0)
        or center.shape != (3,)
        or not np.isfinite(center).all()
        or any(not r for r in [*solute_residues, *solvent_groups])
        or sorted(solute + solvent) != list(range(len(masses)))
        or not protected
        or len(set(protected)) != len(protected)
        or not set(protected).issubset(solute)
    ):
        raise ValueError(
            "Complete disjoint solute/solvent groups, source masses and protected atoms are required"
        )
    distance = np.linalg.norm(xyz - center, axis=1)
    restrained = []
    for residue in solute_residues:
        if np.all(distance[residue] > values["inner_solute_radius_angstrom"] * 0.1):
            if set(residue).intersection(protected):
                raise ValueError(
                    "Protected atoms must remain inside the unrestrained solute region"
                )
            restrained.extend(i for i in residue if elements[i] > 1)
    system = openmm.System()
    for mass in masses:
        system.addParticle(float(mass))
    position = openmm.CustomExternalForce("k_position*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    position.setName("Outer-solute positional restraint")
    position.addGlobalParameter("k_position", values["position_coefficient_kj_mol_nm2"])
    for name in ["x0", "y0", "z0"]:
        position.addPerParticleParameter(name)
    for i in restrained:
        position.addParticle(i, xyz[i].tolist())
    system.addForce(position)
    wall = openmm.CustomCentroidBondForce(
        1, "k_wall*max(0,r-radius)^2; r=pointdistance(x1,y1,z1,cx,cy,cz)"
    )
    wall.setName("Complete solvent/ion mass-center wall")
    for name, value in [
        ("k_wall", values["wall_coefficient_kj_mol_nm2"]),
        ("radius", values["solvent_radius_angstrom"] * 0.1),
        ("cx", center[0]),
        ("cy", center[1]),
        ("cz", center[2]),
    ]:
        wall.addGlobalParameter(name, float(value))
    for group in solvent_groups:
        i = wall.addGroup(group, masses[group].tolist())
        wall.addBond([i], [])
    system.addForce(wall)
    return system, restrained


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Cut the droplet, check it against the source, and write every file."""
    import numpy as np
    import openmm
    import parmed
    from openmm import app, unit
    from parmed.amber import Rst7

    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    values = cfg
    source = parmed.load_file(cfg["amber_topology"])
    if source.box is None or source.parm_data.get("POINTERS", [0] * 28)[27] == 0:
        raise ValueError("The source topology must declare its periodic solvent box")
    topology = app.AmberPrmtopFile(cfg["amber_topology"])
    state = openmm.XmlSerializer.deserialize(Path(cfg["state"]).read_text())
    if not isinstance(state, openmm.State):
        raise ValueError("The state input must contain an OpenMM State")
    xyz = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    atoms = list(topology.topology.atoms())
    if (
        xyz.shape != (len(atoms), 3)
        or len(atoms) != len(source.atoms)
        or max(values["solute_residues"]) > len(source.residues)
        or max(values["protected_atoms"]) > len(atoms)
    ):
        raise ValueError(
            "The declared residue and atom indices and the state must match the complete topology"
        )
    residues = [source.residues[i - 1] for i in values["solute_residues"]]
    solute = sorted(a.idx for residue in residues for a in residue.atoms)
    if not set(i - 1 for i in values["protected_atoms"]).issubset(solute):
        raise ValueError("Every protected atom must belong to the declared solute")
    required = set(solute)
    for residue in source.residues:
        if residue in residues:
            continue
        elements = sorted(a.atomic_number for a in residue.atoms)
        if elements != [1, 1, 8] and not (len(elements) == 1 and elements[0] > 2):
            raise ValueError(
                "Include every molecular solute residue; the remaining residues must be complete water or monatomic ions"
            )
    anchor = sorted(a.idx for a in source.residues[values["anchor_residue"] - 1].atoms)
    heavy = [i for i in anchor if source.atoms[i].atomic_number > 1]
    masses = np.array([a.mass for a in source.atoms])
    imaged, groups, center = image_components(
        xyz,
        cell,
        [(a.index, b.index) for a, b in topology.topology.bonds()],
        masses,
        anchor,
        heavy,
    )
    selected = select_components(
        imaged, groups, solute, center, values["solvent_radius_angstrom"]
    )
    if len(selected) > 100000:
        raise ValueError("At most 100,000 atoms can be included in a finite environment")
    inverse = {old: new for new, old in enumerate(selected)}
    mask = np.zeros(len(atoms), dtype=bool)
    mask[selected] = True
    subset = source[mask]
    subset.box = None
    projected = imaged[selected] - imaged[center] + 64.0
    subset.coordinates = projected
    subset.save(str(output / "complex.prmtop"), format="amber", overwrite=False)
    restart = Rst7(natom=len(selected), title="Alambiq finite environment")
    restart.coordinates = projected
    restart.write(str(output / "complex.inpcrd"))
    prepared_top = app.AmberPrmtopFile(str(output / "complex.prmtop"))
    system = prepared_top.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    signature = physical_identity(system)
    source_system = topology.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    if physical_identity(source_system, selected) != signature:
        raise ValueError(
            "Every retained source particle and physical force parameter must be unchanged"
        )
    del source_system
    source_map = {
        "source_atom_count": len(atoms),
        "source_atom_indices_one_based": [i + 1 for i in selected],
        "protected_atoms_one_based": [
            inverse[i - 1] + 1 for i in values["protected_atoms"]
        ],
        "center_atom_one_based": inverse[center] + 1,
    }
    (output / "source-atom-map.json").write_text(
        json.dumps(source_map, indent=2) + "\n"
    )
    solute_groups = [[inverse[a.idx] for a in residue.atoms] for residue in residues]
    solvent_groups = [
        [inverse[i] for i in group]
        for group in groups
        if not required.intersection(group) and group[0] in inverse
    ]
    external, restrained = boundary_system(
        masses[selected],
        np.array([a.atomic_number for a in subset.atoms]),
        projected * 0.1,
        solute_groups,
        solvent_groups,
        [inverse[i - 1] for i in values["protected_atoms"]],
        np.array([6.4] * 3),
        values,
    )
    (output / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    (output / "external.xml").write_text(openmm.XmlSerializer.serialize(external))
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("CPU"),
        {"Threads": str(values["threads"])},
    )
    context.setPositions(projected * unit.angstrom)
    (output / "state.xml").write_text(
        openmm.XmlSerializer.serialize(context.getState(getPositions=True))
    )
    physical_forces = diagnostics.force_diagnostics(
        context.getState(getForces=True), unit, prepared_top.topology
    )
    del context, integrator
    with (output / "topology.pdb").open("w") as stream:
        app.PDBFile.writeFile(prepared_top.topology, projected * unit.angstrom, stream)
    summary = {
        "geometry_diagnostics": diagnostics.protein_geometry_diagnostics(
            prepared_top.topology, projected * 0.1, system
        ),
        "physical_forces": physical_forces,
        "parameters": {
            k: values[k]
            for k in (
                "solute_residues",
                "anchor_residue",
                "protected_atoms",
                "solvent_radius_angstrom",
                "inner_solute_radius_angstrom",
                "position_coefficient_kj_mol_nm2",
                "wall_coefficient_kj_mol_nm2",
            )
        },
        "source_atom_count": len(atoms),
        "atom_count": len(selected),
        "solute_atom_count": len(solute),
        "solute_residue_count": len(residues),
        "source_frame_time_ps": state.getTime().value_in_unit(unit.picosecond),
        "source_box_vectors_angstrom": cell.tolist(),
        "particle_map": source_map,
        "physical_hamiltonian_sha256": signature,
        "net_charge_e": sum(a.charge for a in subset.atoms),
        "restrained_heavy_atoms_one_based": [i + 1 for i in restrained],
        "solvent_ion_groups_one_based": [
            [i + 1 for i in group] for group in solvent_groups
        ],
        "boundary_energy_convention": "V_position=k_position*sum(|r-r0|^2); V_wall=k_wall*sum(max(0,|COM-center|-R)^2)",
        "exported_hamiltonian": "Unconstrained supplied Amber parameters with all-pair nonperiodic electrostatics and Lennard-Jones; the separate external System holds the boundary potentials",
        "scope": "Complete declared solute and whole water and ion molecules from the supplied periodic state. Bond graphs are unwrapped, whole molecules are imaged around an anchor, and an explicit particle map records source identity. Outer-solute heavy atoms whose whole residue is outside the inner sphere are restrained; protected atoms are unrestrained.",
        "versions": {
            "openmm": openmm.__version__,
            "parmed": parmed.__version__,
            "numpy": np.__version__,
        },
    }
    (output / "preparation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return summary
