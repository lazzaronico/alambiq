"""Native molecular QM/MM evaluation from explicit prepared OpenMM data."""

import functools
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path


@functools.lru_cache
def _component(name):
    spec = importlib.util.spec_from_file_location(
        f"alambiq_native_{name}", Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _indices(values, atoms):
    if (
        not isinstance(values, list)
        or not values
        or any(type(i) is not int or not 1 <= i <= atoms for i in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(
            "Distinct ordered one-based physical atom indices are required"
        )
    return [i - 1 for i in values]


def quantum_specification(value, atoms):
    """Translate the user-facing physical atom convention at one explicit boundary."""
    if (
        not isinstance(value, dict)
        or set(value) != {"atoms", "charge", "link"}
        or type(value["charge"]) is not int
    ):
        raise ValueError(
            "A quantum region requires explicit atoms, integer charge and optional link"
        )
    q = _indices(value["atoms"], atoms)
    if len(q) == atoms:
        raise ValueError("A molecular QM/MM state requires classical embedding atoms")
    link = value["link"]
    if link is not None:
        if not isinstance(link, dict) or set(link) != {
            "quantum_parent",
            "classical_parent",
            "alpha",
            "cancelled_coupling_atoms",
        }:
            raise ValueError(
                "One contracted link requires explicit parents, alpha and coupling cancellations"
            )
        parents = _indices([link["quantum_parent"], link["classical_parent"]], atoms)
        cancelled = link["cancelled_coupling_atoms"]
        if cancelled == []:
            cancelled = []
        else:
            cancelled = _indices(cancelled, atoms)
        alpha = link["alpha"]
        if (
            parents[0] not in q
            or parents[1] in q
            or set(cancelled).intersection(q)
            or type(alpha) not in (int, float)
            or not math.isfinite(alpha)
            or alpha <= 1
        ):
            raise ValueError(
                "The link joins quantum and classical atoms with alpha > 1 and classical coupling cancellations"
            )
        link = {
            "quantum_parent": parents[0],
            "classical_parent": parents[1],
            "alpha": float(alpha),
            "cancelled_coupling_indices": cancelled,
        }
    return q, value["charge"], link


def prepare(request):
    """Resolve exact atom order, positions, classical terms and electronic embedding."""
    import numpy as np
    import openmm
    from openmm import app, unit

    if openmm.__version__ != "8.6":
        raise ValueError("The native molecular composition requires OpenMM 8.6")
    required = {
        "system_xml",
        "state_xml",
        "topology_pdb",
        "external_system_xml",
        "initial_density",
        "quantum",
        "parameters",
    }
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError(
            "Supply the explicit prepared system, state, topology, optional external potential/density, quantum region and parameters"
        )
    p = request["parameters"]
    if not isinstance(p, dict) or set(p) != {
        "backend",
        "grid",
        "gaussian_radius_angstrom",
        "threads",
        "memory_mb",
        "classical_coulomb_ratio",
    }:
        raise ValueError(
            "Explicit electronic backend, quadrature, embedding, resource and classical Coulomb settings are required"
        )
    native = _component("qmmm_native")
    system = native.deserialize_data(
        Path(request["system_xml"]).read_text(),
        "System",
        openmm.XmlSerializer.deserialize,
    )
    state = native.deserialize_data(
        Path(request["state_xml"]).read_text(),
        "State",
        openmm.XmlSerializer.deserialize,
    )
    topology = app.PDBFile(str(request["topology_pdb"])).topology
    atoms = list(topology.atoms())
    n = system.getNumParticles()
    if len(atoms) != n or not n or any(a.element is None for a in atoms):
        raise ValueError(
            "The source topology requires one resolved element per physical system particle"
        )
    for i, atom in enumerate(atoms):
        mass = system.getParticleMass(i).value_in_unit(unit.dalton)
        if not math.isfinite(mass) or not math.isclose(
            mass,
            atom.element.mass.value_in_unit(unit.dalton),
            rel_tol=0.01,
            abs_tol=0.2,
        ):
            raise ValueError(
                "The prepared topology element order must match conventional force-field particle masses"
            )
    q, charge, link = quantum_specification(request["quantum"], n)
    qset = set(q)
    crossings = set()
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for j in range(force.getNumBonds()):
                a, b, *_ = force.getBondParameters(j)
                if (a in qset) != (b in qset):
                    crossings.add(tuple(sorted((a, b))))
    expected = (
        set()
        if link is None
        else {tuple(sorted((link["quantum_parent"], link["classical_parent"])))}
    )
    if crossings != expected:
        raise ValueError(
            "Every covalent boundary must match the explicitly contracted quantum/classical link"
        )
    classical, ledger = native.classical_complement(
        system,
        q,
        None if link is None else link["classical_parent"],
        p["classical_coulomb_ratio"],
    )
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    charges = np.array(
        [
            nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
            for i in range(n)
        ]
    )
    positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    )
    if positions.shape != (n, 3):
        raise ValueError(
            "The source state must contain one Cartesian triple per exact physical atom"
        )
    mapping = native.partition(positions, charges, q, link)
    elements = [a.element.atomic_number for a in atoms]
    electronic_elements = [elements[i] for i in q] + ([1] if link is not None else [])
    _component("qmmm_pyscf").validate_inputs(
        electronic_elements,
        mapping["qm_coordinates"],
        mapping["mm_coordinates"],
        mapping["mm_charges"],
        charge=charge,
        **{k: v for k, v in p.items() if k != "classical_coulomb_ratio"},
    )
    external = None
    if request["external_system_xml"] is not None:
        external = native.deserialize_data(
            Path(request["external_system_xml"]).read_text(),
            "System",
            openmm.XmlSerializer.deserialize,
        )
        allowed = (
            openmm.CustomExternalForce,
            openmm.CustomCentroidBondForce,
            openmm.CustomCompoundBondForce,
        )
        if (
            external.getNumParticles() != n
            or external.getNumConstraints()
            or any(
                external.isVirtualSite(i)
                or external.getParticleMass(i) != system.getParticleMass(i)
                for i in range(n)
            )
            or any(
                type(f) not in allowed or f.usesPeriodicBoundaryConditions()
                for f in external.getForces()
            )
        ):
            raise ValueError(
                "The external potential requires matching fixed atoms and explicit nonperiodic external, centroid or compound-bond forces"
            )
    return {
        "source_system": system,
        "classical_system": classical,
        "external_system": external,
        "positions_angstrom": positions.copy(),
        "elements": elements,
        "electronic_elements": electronic_elements,
        "charges": charges,
        "quantum_indices": q,
        "charge": charge,
        "link": link,
        "mapping": mapping,
        "ledger": ledger,
        "atom_labels": [
            {
                "atom": i + 1,
                "name": a.name,
                "residue": a.residue.name,
                "residue_id": a.residue.id,
                "chain": a.residue.chain.id,
                "element": a.element.symbol,
            }
            for i, a in enumerate(atoms)
        ],
    }


def _classical_observation(system, xyz):
    import numpy as np
    import openmm
    from openmm import unit

    if system is None:
        return 0.0, np.zeros_like(xyz)
    method = openmm.VerletIntegrator(0.0005)
    context = openmm.Context(
        system, method, openmm.Platform.getPlatformByName("Reference")
    )
    try:
        context.setPositions(xyz * unit.angstrom)
        state = context.getState(getEnergy=True, getForces=True)
        energy = float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        force = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
        )
        if (
            not math.isfinite(energy)
            or force.shape != xyz.shape
            or not np.isfinite(force).all()
        ):
            raise ValueError("Every classical energy and physical force must be finite")
        return energy, force
    finally:
        del context, method


def _record(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_initial_density(path):
    """Read a native NPY density or the density member of an electronic NPZ state."""
    import numpy as np

    saved = np.load(path, allow_pickle=False)
    if isinstance(saved, np.ndarray):
        return saved
    with saved:
        if "density" not in saved:
            raise ValueError("Starting density archive requires a density array")
        return saved["density"]


def evaluate(request, output):
    """Combine native electronic, classical and explicit external contributions."""
    import numpy as np
    import openmm
    from pyscf.data import nist

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        _record(output / "request.json", request)
        sources = {
            key: _sha(request[key])
            for key in (
                "system_xml",
                "state_xml",
                "topology_pdb",
                "external_system_xml",
                "initial_density",
            )
            if request[key] is not None
        }
        prepared = prepare(request)
        if any(_sha(request[k]) != digest for k, digest in sources.items()):
            raise ValueError(
                "Every prepared source must be unchanged throughout native preparation"
            )
        _record(output / "atom-map.json", prepared["atom_labels"])
        (output / "classical-system.xml").write_text(
            openmm.XmlSerializer.serialize(prepared["classical_system"])
        )
        np.save(output / "positions-angstrom.npy", prepared["positions_angstrom"])
        p, mapping = request["parameters"], prepared["mapping"]
        initial = None
        if request["initial_density"] is not None:
            initial = _load_initial_density(request["initial_density"])
        result = _component("qmmm_pyscf").evaluate_molecular(
            prepared["electronic_elements"],
            mapping["qm_coordinates"],
            mapping["mm_coordinates"],
            mapping["mm_charges"],
            charge=prepared["charge"],
            directory=output / "electronic",
            initial_density=initial,
            **{k: v for k, v in p.items() if k != "classical_coulomb_ratio"},
        )
        eq = result["observation"]["energy_hartree"]
        fq = _component("qmmm_native").scatter_forces(
            mapping,
            result["quantum_force_hartree_bohr"],
            result["embedding_force_hartree_bohr"],
        )
        ec, fc = _classical_observation(
            prepared["classical_system"], prepared["positions_angstrom"]
        )
        ex, fx = _classical_observation(
            prepared["external_system"], prepared["positions_angstrom"]
        )
        hartree = nist.HARTREE2J * nist.AVOGADRO / 1000
        force = fq * hartree / (nist.BOHR * 0.1) + fc + fx
        if not np.isfinite(force).all():
            raise ValueError("Every composed physical force must be finite")
        np.save(output / "physical-forces-kj-mol-nm.npy", force)
        np.save(output / "classical-forces-kj-mol-nm.npy", fc)
        np.save(output / "external-forces-kj-mol-nm.npy", fx)
        if any(_sha(request[k]) != digest for k, digest in sources.items()):
            raise ValueError(
                "Every prepared source must be unchanged throughout native evaluation"
            )
        record = {
            "schema_version": "0.1",
            "quantity": "molecular-qmmm-state",
            "source_sha256": sources,
            "parameters": p,
            "quantum_atoms": request["quantum"]["atoms"],
            "quantum_charge": prepared["charge"],
            "link": request["quantum"]["link"],
            "physical_atoms": len(force),
            "electronic_atoms": len(prepared["electronic_elements"]),
            "electronic_energy_hartree": eq,
            "electronic_energy_kj_mol": eq * hartree,
            "classical_energy_kj_mol": ec,
            "external_energy_kj_mol": ex,
            "total_energy_kj_mol": eq * hartree + ec + ex,
            "hartree_kj_mol": hartree,
            "bohr_angstrom": nist.BOHR,
            "force_unit": "kJ/mol/nm",
            "electronic": result["observation"],
            "classical_ledger": prepared["ledger"],
            "maximum_physical_force_kj_mol_nm": float(
                np.max(np.linalg.norm(force, axis=1))
            ),
            "net_force_kj_mol_nm": force.sum(0).tolist(),
        }
        _record(output / "summary.json", record)
        return record
    except Exception as exc:
        _record(
            output / "failure.json",
            {"exception": type(exc).__name__, "message": str(exc)},
        )
        raise


def run(cfg):
    """Engine entry: evaluate one prepared state; `cfg` is the request plus `output`."""
    return evaluate({k: v for k, v in cfg.items() if k != "output"}, cfg["output"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "Supply a native prepared-state request and a new output directory"
        )
    evaluate(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
