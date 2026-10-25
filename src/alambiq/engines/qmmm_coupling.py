"""Native molecular electronic callbacks and explicit reaction-coordinate biases."""

import copy
import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "alambiq_prepared_qmmm", Path(__file__).with_name("qmmm_runner.py")
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def _pairs(value, atoms):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(
            "Two ordered one-based atom pairs define the distance difference"
        )
    pairs = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                "Each distance requires two distinct physical atom indices"
            )
        pairs.append(runner._indices(pair, atoms))
    if set(pairs[0]) == set(pairs[1]):
        raise ValueError("The two distances require distinct physical atom pairs")
    return pairs


def coordinate_angstrom(positions_nm, pairs):
    import numpy as np

    xyz = np.asarray(positions_nm, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError(
            "Complete finite Cartesian positions in nanometres are required"
        )
    first, second = _pairs(pairs, len(xyz))
    return float(
        10
        * (
            np.linalg.norm(xyz[first[0]] - xyz[first[1]])
            - np.linalg.norm(xyz[second[0]] - xyz[second[1]])
        )
    )


def umbrella_force(specification, atoms):
    import openmm

    if not isinstance(specification, dict) or set(specification) != {
        "pairs",
        "center_angstrom",
        "coefficient_kj_mol_angstrom2",
    }:
        raise ValueError(
            "An umbrella requires ordered pairs, an angstrom center and K in kJ/mol/angstrom²"
        )
    pairs = _pairs(specification["pairs"], atoms)
    center, coefficient = (
        specification[k] for k in ("center_angstrom", "coefficient_kj_mol_angstrom2")
    )
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in (center, coefficient)
        )
        or coefficient <= 0
    ):
        raise ValueError(
            "A finite umbrella center and positive finite energy coefficient are required"
        )
    force = openmm.CustomCompoundBondForce(
        4, "K*(10*(distance(p1,p2)-distance(p3,p4))-center)^2"
    )
    force.addPerBondParameter("K")
    force.addPerBondParameter("center")
    force.addBond(pairs[0] + pairs[1], [coefficient, center])
    force.setForceGroup(2)
    return force


class MolecularElectronicForce:
    """Evaluate each distinct geometry using the preceding converged density guess.

    Every native call retains its geometry, initial density, complete electronic
    state and derivatives. SCF convergence and observed stationarity apply to each
    invocation. The exact-geometry cache is applied by the surrounding context.
    """

    def __init__(self, prepared, parameters, directory, initial_density):
        import numpy as np
        from pyscf.data import nist

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.parameters = copy.deepcopy(parameters)
        self.quantum = list(prepared["quantum_indices"])
        self.link = copy.deepcopy(prepared["link"])
        self.charge = prepared["charge"]
        self.charges = np.array(prepared["charges"], copy=True)
        self.elements = list(prepared["electronic_elements"])
        self._density = (
            None if initial_density is None else np.array(initial_density, copy=True)
        )
        self.hartree_kj_mol = nist.HARTREE2J * nist.AVOGADRO / 1000
        self.bohr_angstrom = nist.BOHR
        self.calls, self.failed = 0, False

    @property
    def density(self):
        return None if self._density is None else self._density.copy()

    def __call__(self, xyz_angstrom):
        import numpy as np

        if self.failed:
            raise ValueError(
                "A failed electronic trajectory retains its last successful state and diagnostics"
            )
        directory = self.directory / f"{self.calls:06d}"
        directory.mkdir(exist_ok=False)
        try:
            native = runner._component("qmmm_native")
            mapping = native.partition(
                xyz_angstrom, self.charges, self.quantum, self.link
            )
            np.save(directory / "positions-angstrom.npy", xyz_angstrom)
            result = runner._component("qmmm_pyscf").evaluate_molecular(
                self.elements,
                mapping["qm_coordinates"],
                mapping["mm_coordinates"],
                mapping["mm_charges"],
                charge=self.charge,
                directory=directory / "electronic",
                initial_density=self.density,
                **{
                    k: v
                    for k, v in self.parameters.items()
                    if k != "classical_coulomb_ratio"
                },
            )
            energy = result["observation"]["energy_hartree"] * self.hartree_kj_mol
            force = (
                native.scatter_forces(
                    mapping,
                    result["quantum_force_hartree_bohr"],
                    result["embedding_force_hartree_bohr"],
                )
                * self.hartree_kj_mol
                / (self.bohr_angstrom * 0.1)
            )
            if not math.isfinite(energy) or not np.isfinite(force).all():
                raise ValueError(
                    "Finite composed electronic energy and all physical forces are required"
                )
            self._density = np.array(result["density"], copy=True)
            self.calls += 1
            return energy, force
        except Exception as exc:
            self.failed = True
            runner._record(
                directory / "failure.json",
                {"exception": type(exc).__name__, "message": str(exc)},
            )
            raise


def compose_system(prepared, electronic, umbrella):
    """Compose a fixed molecular context from exact classical, external and QM terms.

    Groups 0/1/2/3 identify classical/external/umbrella/electronic contributions.
    The caller supplies the native OpenMM integrator and sampling request.
    """
    import numpy as np
    import openmm
    from openmm import unit

    system = openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(prepared["classical_system"])
    )
    for force in system.getForces():
        force.setForceGroup(0)
    if prepared["external_system"] is not None:
        for source in prepared["external_system"].getForces():
            force = openmm.XmlSerializer.deserialize(
                openmm.XmlSerializer.serialize(source)
            )
            force.setForceGroup(1)
            system.addForce(force)
    if umbrella is not None:
        system.addForce(umbrella_force(umbrella, system.getNumParticles()))
    cell = np.array(
        [v.value_in_unit(unit.nanometer) for v in system.getDefaultPeriodicBoxVectors()]
    )
    cache = runner._component("qmmm_native").ExactElectronicCache(
        electronic, system.getNumParticles(), cell
    )

    def callback(state):
        # Molecular callback states supply positions; the nonperiodic model keeps
        # the prepared placeholder cell used by the exact-geometry cache.
        return cache.evaluate(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), cell
        )

    force = openmm.PythonForce(callback)
    force.setForceGroup(3)
    system.addForce(force)
    return system, cache
