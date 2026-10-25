"""simulation.evaluate_qmmm_state on a water molecule beside two ions, with answers known independently.

The system is one water as the quantum region and a sodium and a chloride ion as the
classical rest, with simple fixed charges and Lennard-Jones parameters. That makes the
following checkable without trusting the capability:

* with the ions' charges set to zero there is no field, so the electronic energy must be
  the energy of the isolated water computed directly with PySCF at the same level;
* the classical energy is the ions' Coulomb energy plus the Lennard-Jones energy of every
  pair that is not inside the quantum region, written out here from the formulas;
* the force on any atom is minus the derivative of the total energy, so a central
  finite difference of the reported energy must reproduce the reported force, for a
  quantum atom and for a classical one;
* a potential has zero net force (translation invariance);
* the energy components add to the total and the electron count is exact.

The evaluation runs PySCF 2.14.0 in the pinned QM/MM environment. Set ALAMBIQ_ENV_QMMM
to its Python.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ENV = os.environ.get("ALAMBIQ_ENV_QMMM", "")
COULOMB = 138.93545764438198  # kJ mol^-1 nm e^-2 (OpenMM's ONE_4PI_EPS0)
HARTREE = 2625.4996394799  # kJ/mol
CHARGES = [-0.8, 0.4, 0.4, 0.2, -0.2]
SIGMA = [0.3, 0.1, 0.1, 0.25, 0.35]  # nm
EPSILON = [0.6, 0.0, 0.0, 0.5, 0.4]  # kJ/mol
POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0757, 0.0586, 0.0],
        [-0.0757, 0.0586, 0.0],
        [0.45, 0.10, 0.05],
        [-0.30, 0.50, -0.10],
    ]
)

pytestmark = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_QMMM to run the QM/MM evaluation",
)

BUILD = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    import openmm
    from openmm import app, unit
    spec = json.loads(sys.argv[1])
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    topology = app.Topology()
    chain = topology.addChain("A")
    water = topology.addResidue("HOH", chain)
    atoms = [topology.addAtom(n, e, water) for n, e in
             [("O", app.element.oxygen), ("H1", app.element.hydrogen), ("H2", app.element.hydrogen)]]
    topology.addBond(atoms[0], atoms[1]); topology.addBond(atoms[0], atoms[2])
    for name, element in [("NA", app.element.sodium), ("CL", app.element.chlorine)]:
        topology.addAtom(name, element, topology.addResidue(name, chain))
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for atom, q, s, e in zip(topology.atoms(), spec["charges"], spec["sigma"], spec["epsilon"]):
        system.addParticle(atom.element.mass)
        nb.addParticle(q, s, e)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        nb.addException(a, b, 0.0, 1.0, 0.0)
    system.addForce(nb)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.0957, 4.0e5); bonds.addBond(0, 2, 0.0957, 4.0e5)
    system.addForce(bonds)
    angles = openmm.HarmonicAngleForce()
    angles.addAngle(1, 0, 2, 1.824, 400.0)
    system.addForce(angles)
    if spec.get("motion_remover"):
        system.addForce(openmm.CMMotionRemover())
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(spec["positions"] * unit.nanometer)
    (out / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    (out / "state.xml").write_text(openmm.XmlSerializer.serialize(context.getState(getPositions=True)))
    with (out / "topology.pdb").open("w") as stream:
        app.PDBFile.writeFile(topology, spec["positions"] * unit.nanometer, stream)
    """
)

ISOLATED = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    from pyscf import dft, gto
    positions = np.array(json.loads(sys.argv[1])) * 10
    mol = gto.M(atom=[("O", positions[0]), ("H", positions[1]), ("H", positions[2])],
                unit="Angstrom", basis="6-31g**", charge=0, spin=0, verbose=0)
    method = dft.RKS(mol); method.xc = "PBE"; method.grids.level = 7
    method.conv_tol = 1e-11
    print(json.dumps(float(method.kernel())))
    """
)


def build(
    directory: Path, charges=None, positions=None, motion_remover=False
) -> dict[str, str]:
    spec = {
        "motion_remover": motion_remover,
        "charges": charges or CHARGES,
        "sigma": SIGMA,
        "epsilon": EPSILON,
        "positions": (POSITIONS if positions is None else positions).tolist(),
    }
    subprocess.run(
        [ENV, "-I", "-c", BUILD, json.dumps(spec), str(directory)],
        check=True,
        capture_output=True,
    )
    return {
        "system": str(directory / "system.xml"),
        "state": str(directory / "state.xml"),
        "topology": str(directory / "topology.pdb"),
    }


def run(params, tmp_path, name="run"):
    from alambiq.api import run as execute

    step = {
        "id": "qm",
        "capability": "simulation.evaluate_qmmm_state",
        "params": params,
    }
    return execute({"id": "qm", "steps": [step]}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("qm")


def request(files, **changes):
    return {
        **files,
        "quantum": {"atoms": [1, 2, 3], "charge": 0},
        "threads": 4,
        **changes,
    }


def classical_energy(positions: np.ndarray, charges=CHARGES) -> float:
    """Ion Coulomb energy plus every Lennard-Jones pair outside the quantum water."""
    total = 0.0
    r = np.linalg.norm(positions[3] - positions[4])
    total += COULOMB * charges[3] * charges[4] / r
    for i in range(5):
        for j in range(i + 1, 5):
            if i < 3 and j < 3:
                continue  # inside the quantum region
            d = np.linalg.norm(positions[i] - positions[j])
            sigma = 0.5 * (SIGMA[i] + SIGMA[j])
            epsilon = math.sqrt(EPSILON[i] * EPSILON[j])
            total += 4 * epsilon * ((sigma / d) ** 12 - (sigma / d) ** 6)
    return total


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    os.environ["ALAMBIQ_ENV_QMMM"] = ENV
    directory = tmp_path_factory.mktemp("qmmm")
    files = build(directory / "base")
    outcome = run(request(files), directory)
    return directory, files, outcome, ok(outcome)


class TestEnergiesAndForces:
    def test_the_components_add_to_the_total_and_the_run_is_converged(self, base):
        out = base[3]
        assert out["converged"] is True
        assert out["total_energy_kj_mol"] == pytest.approx(
            out["electronic_energy_kj_mol"]
            + out["classical_energy_kj_mol"]
            + out["external_energy_kj_mol"],
            abs=1e-8,
        )
        assert out["external_energy_kj_mol"] == 0.0

    def test_the_electron_count_is_exact_and_the_orbitals_stationary(self, base):
        out = base[3]
        assert out["electrons_expected"] == 10
        assert out["electrons_found"] == pytest.approx(10.0, abs=1e-7)
        assert out["orbital_gradient_norm"] < 1e-7
        assert out["basis_functions"] == 24 and out["electronic_atoms"] == 3

    def test_the_classical_energy_is_the_ion_coulomb_and_outside_lennard_jones_energy(
        self, base
    ):
        assert base[3]["classical_energy_kj_mol"] == pytest.approx(
            classical_energy(POSITIONS), rel=1e-9
        )

    def test_the_ledger_shows_the_bonded_terms_inside_the_quantum_water_were_removed(
        self, base
    ):
        ledger = base[3]["classical_ledger"]
        assert ledger["HarmonicBondForce"] == {"source": 2, "retained": 0, "removed": 2}
        assert ledger["HarmonicAngleForce"] == {
            "source": 1,
            "retained": 0,
            "removed": 1,
        }

    def test_a_potential_has_no_net_force(self, base):
        net = np.array(base[3]["net_force_kj_mol_nm"])
        forces = np.load(base[3]["forces"])
        assert forces.shape == (5, 3)
        assert np.abs(net).max() < 1e-3 * np.linalg.norm(forces, axis=1).max()
        assert np.allclose(forces.sum(axis=0), net, atol=1e-6)
        assert base[3]["maximum_physical_force_kj_mol_nm"] == pytest.approx(
            np.linalg.norm(forces, axis=1).max()
        )

    @pytest.mark.parametrize(
        ("atom", "axis", "label"),
        [(0, 0, "quantum oxygen"), (3, 1, "classical sodium")],
    )
    def test_the_force_is_minus_the_derivative_of_the_total_energy(
        self, base, tmp_path, atom, axis, label
    ):
        h = 0.0005  # nm
        energies = []
        for sign in (+1, -1):
            moved = POSITIONS.copy()
            moved[atom, axis] += sign * h
            files = build(tmp_path / f"s{sign}", positions=moved)
            energies.append(
                ok(run(request(files), tmp_path, f"r{sign}"))["total_energy_kj_mol"]
            )
        numerical = -(energies[0] - energies[1]) / (2 * h)
        reported = np.load(base[3]["forces"])[atom, axis]
        assert reported == pytest.approx(numerical, rel=5e-3, abs=0.5), label

    def test_the_saved_electronic_state_starts_the_next_evaluation_in_fewer_cycles(
        self, base, tmp_path
    ):
        again = ok(
            run(request(base[1], initial_density=base[3]["electronic_state"]), tmp_path)
        )
        assert again["scf_cycles"] < base[3]["scf_cycles"]
        assert again["total_energy_kj_mol"] == pytest.approx(
            base[3]["total_energy_kj_mol"], abs=1e-6
        )


class TestTheFieldOfTheSurroundings:
    def test_with_no_classical_charges_the_electronic_energy_is_that_of_the_isolated_water(
        self, tmp_path
    ):
        os.environ["ALAMBIQ_ENV_QMMM"] = ENV
        files = build(tmp_path / "neutral", charges=[-0.8, 0.4, 0.4, 0.0, 0.0])
        out = ok(run(request(files), tmp_path))
        isolated = float(
            subprocess.run(
                [ENV, "-I", "-c", ISOLATED, json.dumps(POSITIONS[:3].tolist())],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()[-1]
        )
        assert out["electronic_energy_kj_mol"] == pytest.approx(
            isolated * HARTREE, abs=2e-3
        )

    def test_charges_near_the_water_change_its_electronic_energy(self, base, tmp_path):
        neutral = build(tmp_path / "n", charges=[-0.8, 0.4, 0.4, 0.0, 0.0])
        free = ok(run(request(neutral), tmp_path, "n"))
        assert (
            abs(base[3]["electronic_energy_kj_mol"] - free["electronic_energy_kj_mol"])
            > 0.5
        )


class TestRefusals:
    def fails(self, base, tmp_path, message, **changes):
        outcome = run(request(base[1], **changes), tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["qm"].error

    def test_a_region_with_every_atom_leaves_nothing_classical(self, base, tmp_path):
        self.fails(
            base,
            tmp_path,
            "requires classical embedding atoms",
            quantum={"atoms": [1, 2, 3, 4, 5], "charge": 0},
        )

    def test_an_odd_number_of_electrons_is_refused(self, base, tmp_path):
        self.fails(
            base,
            tmp_path,
            "closed-shell population",
            quantum={"atoms": [1, 2, 3], "charge": 1},
        )

    def test_atoms_outside_the_system_are_refused(self, base, tmp_path):
        self.fails(
            base,
            tmp_path,
            "Distinct ordered one-based physical atom indices",
            quantum={"atoms": [1, 2, 9], "charge": 0},
        )

    def test_a_cut_bond_without_a_declared_link_is_refused(self, base, tmp_path):
        self.fails(
            base,
            tmp_path,
            "covalent boundary must match",
            quantum={"atoms": [1, 2], "charge": 0},
        )

    def test_a_declared_link_where_no_bond_is_cut_is_refused(self, base, tmp_path):
        link = {
            "quantum_parent": 1,
            "classical_parent": 4,
            "alpha": 1.4,
            "cancelled_coupling_atoms": [4],
        }
        self.fails(
            base,
            tmp_path,
            "covalent boundary must match",
            quantum={"atoms": [1, 2, 3], "charge": 0, "link": link},
        )

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"grid": 12}, "grid requires an integer in \\[0, 9\\]"),
            ({"threads": 0}, "threads requires"),
            ({"memory_mb": 10}, "memory_mb requires"),
            (
                {"gaussian_radius_angstrom": 0.0},
                "gaussian_radius_angstrom requires a positive",
            ),
            (
                {"classical_coulomb_ratio": float("nan")},
                "classical_coulomb_ratio requires a positive",
            ),
            ({"timeout_seconds": 0}, "timeout_seconds requires"),
        ],
    )
    def test_settings_outside_their_limits_are_refused_by_name(
        self, base, tmp_path, changes, message
    ):
        outcome = run(request(base[1], **changes), tmp_path)
        assert not outcome.ok
        import re

        assert re.search(message, outcome.steps["qm"].error)

    def test_a_force_the_partition_cannot_split_is_refused(self, tmp_path):
        files = build(tmp_path / "other", motion_remover=True)
        outcome = run(request(files), tmp_path)
        assert not outcome.ok
        assert (
            "explicit all-pair nonbonded, harmonic bonded and periodic-torsion data"
            in (outcome.steps["qm"].error)
        )


class TestReporting:
    def test_the_report_labels_the_electronic_and_classical_components(
        self, base, tmp_path
    ):
        outcome = base[2]
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "Electronic (PBE/6-31G**, in the classical field)" in page
        assert "Classical (MM and QM–MM Lennard-Jones, bonded)" in page
