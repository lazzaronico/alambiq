"""Explicit molecular QM/MM partitions and native OpenMM callback boundaries.

Internal atom indices are zero-based. Electronic coordinates use angstrom;
the callback receives nanometres and returns kJ/mol and kJ/mol/nm. Molecular
complements retain all physical particles and all-pair classical interactions.
"""

import math
import runpy
from pathlib import Path

deserialize_data = runpy.run_path(str(Path(__file__).with_name("openmm_xml.py")))[
    "deserialize_data"
]


def _indices(values, atoms):
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(type(i) is not int or not 0 <= i < atoms for i in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "An ordered nonempty set of distinct zero-based physical atom indices is required"
        )
    return list(values)


def partition(coordinates_angstrom, charges, quantum_indices, link=None):
    """Map real atoms and one optional contracted hydrogen cap into an electronic state."""
    import numpy as np

    xyz, charges = np.asarray(coordinates_angstrom, float), np.asarray(charges, float)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not len(xyz)
        or charges.shape != (len(xyz),)
        or not np.isfinite(xyz).all()
        or not np.isfinite(charges).all()
    ):
        raise ValueError(
            "Complete finite physical coordinates and one embedding charge per atom are required"
        )
    quantum = _indices(quantum_indices, len(xyz))
    selected = set(quantum)
    mm = [i for i in range(len(xyz)) if i not in selected]
    result = {
        "atoms": len(xyz),
        "quantum_indices": quantum,
        "mm_indices": mm,
        "qm_coordinates": xyz[quantum].copy(),
        "mm_coordinates": xyz[mm].copy(),
        "mm_charges": charges[mm].copy(),
    }
    if link is None:
        return result
    if not isinstance(link, dict) or set(link) != {
        "quantum_parent",
        "classical_parent",
        "alpha",
        "cancelled_coupling_indices",
    }:
        raise ValueError(
            "One link requires both parents, contraction alpha and explicit coupling cancellations"
        )
    qp, cp, alpha = link["quantum_parent"], link["classical_parent"], link["alpha"]
    cancelled = link["cancelled_coupling_indices"]
    if (
        type(qp) is not int
        or type(cp) is not int
        or qp not in selected
        or cp not in mm
        or type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or alpha <= 1
        or np.linalg.norm(xyz[qp] - xyz[cp]) <= 1e-12
        or not isinstance(cancelled, (list, tuple))
        or any(type(i) is not int or i not in mm for i in cancelled)
        or len(set(cancelled)) != len(cancelled)
    ):
        raise ValueError(
            "A resolved quantum/classical bond, alpha > 1 and distinct classical cancellation indices are required"
        )
    cap = xyz[qp] + (xyz[cp] - xyz[qp]) / alpha
    result.update(
        quantum_parent=qp,
        classical_parent=cp,
        alpha=float(alpha),
        qm_coordinates=np.vstack([result["qm_coordinates"], cap]),
        cancelled_coupling_indices=list(cancelled),
    )
    for i in cancelled:
        result["mm_charges"][mm.index(i)] = 0.0
    return result


def scatter_forces(mapping, quantum_force, embedding_force):
    """Apply the link-coordinate chain rule while retaining every physical force."""
    import numpy as np

    fq, fm = np.asarray(quantum_force, float), np.asarray(embedding_force, float)
    linked = "classical_parent" in mapping
    if (
        fq.shape != (len(mapping["quantum_indices"]) + int(linked), 3)
        or fm.shape != (len(mapping["mm_indices"]), 3)
        or not np.isfinite(fq).all()
        or not np.isfinite(fm).all()
    ):
        raise ValueError(
            "Every finite real-quantum, cap and embedding force is required"
        )
    physical = np.zeros((mapping["atoms"], 3))
    physical[mapping["quantum_indices"]] = fq[:-1] if linked else fq
    physical[mapping["mm_indices"]] = fm
    if linked:
        weight = 1 / mapping["alpha"]
        physical[mapping["quantum_parent"]] += (1 - weight) * fq[-1]
        physical[mapping["classical_parent"]] += weight * fq[-1]
    return physical


class ExactElectronicCache:
    """Reuse only the last exact geometry of one fixed-cell, fixed-Hamiltonian instance."""

    def __init__(self, function, atoms, cell_nm):
        import numpy as np

        cell = np.asarray(cell_nm, float)
        if (
            not callable(function)
            or type(atoms) is not int
            or atoms <= 0
            or cell.shape != (3, 3)
            or not np.isfinite(cell).all()
        ):
            raise ValueError(
                "An electronic evaluator, positive atom count and finite fixed cell are required"
            )
        self.function, self.atoms, self.cell = function, atoms, cell.copy()
        self.position = self.value = None
        self.hits = 0

    def evaluate(self, position_nm, cell_nm):
        import numpy as np

        position = np.asarray(position_nm, float)
        if (
            position.shape != (self.atoms, 3)
            or not np.isfinite(position).all()
            or not np.array_equal(cell_nm, self.cell)
        ):
            raise ValueError(
                "Exact fixed cell and complete finite physical positions are required"
            )
        if self.position is not None and np.array_equal(position, self.position):
            self.hits += 1
        else:
            energy, force = self.function(position * 10)
            force = np.asarray(force, float)
            if (
                not np.isfinite(energy)
                or force.shape != position.shape
                or not np.isfinite(force).all()
            ):
                raise ValueError(
                    "A finite native electronic energy and complete physical forces are required"
                )
            self.position = position.copy()
            self.value = float(energy), force.copy()
        return self.value[0], self.value[1].copy()


def classical_complement(source, quantum_indices, link_parent=None, coulomb_ratio=1.0):
    """Retain MM electrostatics, QM–MM Lennard-Jones and mixed bonded terms.

    Terms entirely inside the real quantum region and optional classical link parent
    are represented by the electronic calculation. Source classical charges persist
    independently of electronic coupling cancellations. The explicit Coulomb ratio
    matches the requested physical-unit convention.
    """
    import openmm
    from openmm import unit

    atoms = source.getNumParticles()
    quantum = set(_indices(quantum_indices, atoms))
    if link_parent is not None and (
        type(link_parent) is not int
        or not 0 <= link_parent < atoms
        or link_parent in quantum
    ):
        raise ValueError("The optional link parent is one distinct real classical atom")
    if (
        type(coulomb_ratio) not in (int, float)
        or not math.isfinite(coulomb_ratio)
        or coulomb_ratio <= 0
    ):
        raise ValueError("A finite positive Coulomb convention ratio is required")
    allowed = (
        openmm.NonbondedForce,
        openmm.HarmonicBondForce,
        openmm.HarmonicAngleForce,
        openmm.PeriodicTorsionForce,
    )
    if any(
        type(force) not in allowed or force.usesPeriodicBoundaryConditions()
        for force in source.getForces()
    ):
        raise ValueError(
            "The molecular partition uses explicit all-pair nonbonded, harmonic bonded and periodic-torsion data"
        )
    if any(
        source.isVirtualSite(i)
        or source.getParticleMass(i).value_in_unit(unit.dalton) <= 0
        for i in range(atoms)
    ):
        raise ValueError(
            "Every physical particle requires a positive mass and explicit position"
        )
    found = [f for f in source.getForces() if isinstance(f, openmm.NonbondedForce)]
    if (
        len(found) != 1
        or found[0].getNonbondedMethod() != openmm.NonbondedForce.NoCutoff
    ):
        raise ValueError("One all-pair molecular nonbonded force is required")
    original = found[0]
    if (
        original.getNumParticles() != atoms
        or original.getNumGlobalParameters()
        or original.getNumParticleParameterOffsets()
        or original.getNumExceptionParameterOffsets()
    ):
        raise ValueError(
            "One fixed nonbonded parameter set per physical particle is required"
        )
    result = openmm.System()
    for i in range(atoms):
        result.addParticle(source.getParticleMass(i))
    nb = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(original))
    nb.setUseDispersionCorrection(False)
    nb.setUseSwitchingFunction(False)
    for i in range(atoms):
        charge, sigma, epsilon = nb.getParticleParameters(i)
        nb.setParticleParameters(
            i,
            0.0 if i in quantum else charge * math.sqrt(coulomb_ratio),
            sigma,
            epsilon,
        )
    excluded = quantum | ({link_parent} if link_parent is not None else set())
    for i in range(nb.getNumExceptions()):
        a, b, charge, sigma, epsilon = nb.getExceptionParameters(i)
        pair = {a, b}
        nb.setExceptionParameters(
            i,
            a,
            b,
            0.0 if pair & quantum else charge * coulomb_ratio,
            sigma,
            0.0 if pair <= excluded else epsilon,
        )
    ordered = sorted(excluded)
    for offset, a in enumerate(ordered):
        for b in ordered[offset + 1 :]:
            nb.addException(a, b, 0.0, 1.0, 0.0, replace=True)
    result.addForce(nb)
    ledger = {}
    for cls, count, get, add, arity in (
        (openmm.HarmonicBondForce, "getNumBonds", "getBondParameters", "addBond", 2),
        (
            openmm.HarmonicAngleForce,
            "getNumAngles",
            "getAngleParameters",
            "addAngle",
            3,
        ),
        (
            openmm.PeriodicTorsionForce,
            "getNumTorsions",
            "getTorsionParameters",
            "addTorsion",
            4,
        ),
    ):
        selected = [f for f in source.getForces() if isinstance(f, cls)]
        if len(selected) > 1:
            raise ValueError("One exact source force for each bonded kind is required")
        if not selected:
            continue
        original = selected[0]
        force = cls()
        kept = 0
        force.setForceGroup(original.getForceGroup())
        force.setUsesPeriodicBoundaryConditions(False)
        total = getattr(original, count)()
        for i in range(total):
            parameters = getattr(original, get)(i)
            if set(parameters[:arity]) <= excluded:
                continue
            getattr(force, add)(*parameters)
            kept += 1
        result.addForce(force)
        ledger[cls.__name__] = {
            "source": total,
            "retained": kept,
            "removed": total - kept,
        }
    total, kept = source.getNumConstraints(), 0
    for i in range(total):
        a, b, distance = source.getConstraintParameters(i)
        if {a, b} <= excluded:
            continue
        if {a, b} & quantum:
            raise ValueError(
                "Retained classical constraints require flexible real quantum atoms"
            )
        result.addConstraint(a, b, distance)
        kept += 1
    ledger["constraints"] = {"source": total, "retained": kept, "removed": total - kept}
    return result, ledger
