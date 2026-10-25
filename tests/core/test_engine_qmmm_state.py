"""Exact prepared-state joins and complete molecular QM/MM energy/force composition."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "qmmm_runner",
    Path(__file__).resolve().parents[2] / "src/alambiq/engines/qmmm_runner.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required")
class InitialDensityTests(unittest.TestCase):
    def test_native_array_and_electronic_archive_preserve_same_density(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            density = np.array([[1.0, 0.125], [0.125, 0.5]])
            array = root / "starting-density.npy"
            archive = root / "electronic-state.npz"
            np.save(array, density)
            np.savez(archive, density=density, orbital_energies=np.array([-0.5, 0.5]))
            for source in [array, archive]:
                # Artifact-store inputs can have digest names without extensions.
                content = source.read_bytes()
                retained = root / hashlib.sha256(content).hexdigest()
                retained.write_bytes(content)
                with self.subTest(format=source.suffix):
                    np.testing.assert_array_equal(
                        runner._load_initial_density(retained), density
                    )
                    self.assertEqual(retained.read_bytes(), content)
                    self.assertEqual(source.read_bytes(), content)

    def test_missing_density_and_pickled_arrays_are_rejected(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.npz"
            np.savez(missing, orbital_energies=np.array([-0.5, 0.5]))
            with self.assertRaisesRegex(ValueError, "requires a density array"):
                runner._load_initial_density(missing)
            for suffix in ["npy", "npz"]:
                path = root / f"object.{suffix}"
                values = np.array([{"density": 1}], dtype=object)
                if suffix == "npy":
                    np.save(path, values)
                else:
                    np.savez(path, density=values)
                with self.subTest(format=suffix), self.assertRaisesRegex(
                    ValueError, "allow_pickle=False"
                ):
                    runner._load_initial_density(path)


class QuantumSpecificationTests(unittest.TestCase):
    def test_one_based_region_and_link_translate_once(self):
        region = {
            "atoms": [3, 1],
            "charge": 0,
            "link": {
                "quantum_parent": 3,
                "classical_parent": 4,
                "alpha": 1.4,
                "cancelled_coupling_atoms": [4, 5],
            },
        }
        q, charge, link = runner.quantum_specification(region, 5)
        self.assertEqual(q, [2, 0])
        self.assertEqual(charge, 0)
        self.assertEqual(
            link,
            {
                "quantum_parent": 2,
                "classical_parent": 3,
                "alpha": 1.4,
                "cancelled_coupling_indices": [3, 4],
            },
        )
        self.assertEqual(region["atoms"], [3, 1])
        for changed in (
            {"atoms": [0]},
            {"atoms": [True]},
            {"atoms": [1, 1]},
            {"atoms": [[1]]},
            {"atoms": [6]},
            {"charge": True},
            {"unexpected": 1},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner.quantum_specification({**region, **changed}, 5)


@unittest.skipUnless(importlib.util.find_spec("openmm"), "OpenMM 8.6 required")
class PreparedStateTests(unittest.TestCase):
    @staticmethod
    def source(directory):
        import openmm
        from openmm import app, unit

        topology = app.Topology()
        chain = topology.addChain("A")
        water = topology.addResidue("HOH", chain)
        atoms = [
            topology.addAtom(name, element, water)
            for name, element in [
                ("O", app.element.oxygen),
                ("H1", app.element.hydrogen),
                ("H2", app.element.hydrogen),
            ]
        ]
        topology.addBond(atoms[0], atoms[1])
        topology.addBond(atoms[0], atoms[2])
        for name, element in [("NA", app.element.sodium), ("CL", app.element.chlorine)]:
            residue = topology.addResidue(name, chain)
            topology.addAtom(name, element, residue)
        system = openmm.System()
        nb = openmm.NonbondedForce()
        for atom, charge in zip(
            topology.atoms(), [-0.8, 0.4, 0.4, 0.2, -0.2], strict=True
        ):
            system.addParticle(atom.element.mass)
            nb.addParticle(charge, 0.15, 0.1)
        system.addForce(nb)
        bonds = openmm.HarmonicBondForce()
        bonds.addBond(0, 1, 0.09, 1000.0)
        bonds.addBond(0, 2, 0.09, 1000.0)
        system.addForce(bonds)
        system.addConstraint(0, 1, 0.09)
        positions = [
            [0.0, 0.0, 0.0],
            [0.075, 0.0, 0.05],
            [-0.075, 0.0, 0.05],
            [0.3, 0.1, 0.2],
            [-0.4, -0.2, -0.3],
        ]
        integrator = openmm.VerletIntegrator(0.0005)
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName("Reference")
        )
        try:
            context.setPositions(positions)
            state = context.getState(getPositions=True)
            (directory / "system.xml").write_text(
                openmm.XmlSerializer.serialize(system)
            )
            (directory / "state.xml").write_text(openmm.XmlSerializer.serialize(state))
        finally:
            del context, integrator
        with (directory / "topology.pdb").open("w") as stream:
            app.PDBFile.writeFile(topology, positions * unit.nanometer, stream)
        external = openmm.System()
        for atom in topology.atoms():
            external.addParticle(atom.element.mass)
        restraint = openmm.CustomExternalForce("100*x*x")
        restraint.addParticle(3, [])
        external.addForce(restraint)
        (directory / "external.xml").write_text(
            openmm.XmlSerializer.serialize(external)
        )
        return {
            "system_xml": str(directory / "system.xml"),
            "state_xml": str(directory / "state.xml"),
            "topology_pdb": str(directory / "topology.pdb"),
            "external_system_xml": str(directory / "external.xml"),
            "initial_density": None,
            "quantum": {"atoms": [1, 2, 3], "charge": 0, "link": None},
            "parameters": {
                "backend": "cpu",
                "grid": 3,
                "gaussian_radius_angstrom": 0.8,
                "threads": 1,
                "memory_mb": 1000,
                "classical_coulomb_ratio": 1.0,
            },
        }

    def test_source_order_charges_positions_and_constraint_ledger(self):
        import numpy as np
        import openmm

        with tempfile.TemporaryDirectory() as tmp:
            request = self.source(Path(tmp))
            before = Path(request["system_xml"]).read_bytes()
            prepared = runner.prepare(request)
            self.assertEqual(prepared["elements"], [8, 1, 1, 11, 17])
            self.assertEqual(prepared["quantum_indices"], [0, 1, 2])
            np.testing.assert_allclose(prepared["charges"], [-0.8, 0.4, 0.4, 0.2, -0.2])
            np.testing.assert_allclose(
                prepared["positions_angstrom"][3], [3.0, 1.0, 2.0]
            )
            self.assertEqual(
                prepared["ledger"]["constraints"],
                {"source": 1, "retained": 0, "removed": 1},
            )
            self.assertEqual(Path(request["system_xml"]).read_bytes(), before)
            self.assertEqual(prepared["source_system"].getNumConstraints(), 1)
            self.assertEqual(prepared["classical_system"].getNumConstraints(), 0)
            bad = {**request, "quantum": {"atoms": [1], "charge": 0, "link": None}}
            with self.assertRaisesRegex(ValueError, "covalent boundary"):
                runner.prepare(bad)
            system = openmm.XmlSerializer.deserialize(before.decode())
            system.setParticleMass(3, 12.0)
            Path(request["system_xml"]).write_text(
                openmm.XmlSerializer.serialize(system)
            )
            with self.assertRaisesRegex(ValueError, "element.*mass"):
                runner.prepare(request)

    def test_external_potential_uses_matching_fixed_atoms_and_native_data(self):
        import openmm

        with tempfile.TemporaryDirectory() as tmp:
            request = self.source(Path(tmp))
            external = openmm.XmlSerializer.deserialize(
                Path(request["external_system_xml"]).read_text()
            )
            external.addConstraint(3, 4, 0.4)
            Path(request["external_system_xml"]).write_text(
                openmm.XmlSerializer.serialize(external)
            )
            with self.assertRaisesRegex(ValueError, "external potential"):
                runner.prepare(request)
            Path(request["system_xml"]).write_text(
                '<System type="System"><Force type="PythonForce"/></System>'
            )
            with self.assertRaisesRegex(ValueError, "not an allowed native data type"):
                runner.prepare(request)


@unittest.skipUnless(
    importlib.util.find_spec("pyscf") and importlib.util.find_spec("openmm"),
    "PySCF 2.14.0 and OpenMM 8.6 required",
)
class CompleteCompositionTests(unittest.TestCase):
    def test_preparation_failures_and_changed_sources_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request = PreparedStateTests.source(directory)
            original = runner.prepare

            def change_source(value):
                prepared = original(value)
                with Path(value["topology_pdb"]).open("a") as stream:
                    stream.write("REMARK modified input\n")
                return prepared

            with patch.object(runner, "prepare", side_effect=change_source):
                with self.assertRaisesRegex(ValueError, "source must be unchanged throughout native preparation"):
                    runner.evaluate(request, directory / "changed")
            self.assertTrue((directory / "changed/failure.json").is_file())
            self.assertFalse((directory / "changed/electronic").exists())
            with self.assertRaises(FileExistsError):
                runner.evaluate(request, directory / "changed")
            Path(request["system_xml"]).write_text(
                '<System type="System"><Force type="PythonForce"/></System>'
            )
            with self.assertRaisesRegex(ValueError, "not an allowed native data type"):
                runner.evaluate(request, directory / "invalid")
            failure = json.loads((directory / "invalid/failure.json").read_text())
            self.assertEqual(failure["exception"], "ValueError")
            self.assertTrue((directory / "invalid/request.json").is_file())

    def test_complete_energy_and_physical_force_derivative(self):
        import numpy as np
        import openmm
        from openmm import unit

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request = PreparedStateTests.source(directory)
            record = runner.evaluate(request, directory / "base")
            self.assertAlmostEqual(record["external_energy_kj_mol"], 9.0, delta=1e-12)
            self.assertAlmostEqual(
                record["total_energy_kj_mol"],
                record["electronic_energy_kj_mol"]
                + record["classical_energy_kj_mol"]
                + record["external_energy_kj_mol"],
                delta=1e-10,
            )
            self.assertEqual(record["quantum_atoms"], [1, 2, 3])
            self.assertEqual(record["force_unit"], "kJ/mol/nm")
            self.assertEqual(
                json.loads((directory / "base/summary.json").read_text()), record
            )
            full = np.load(
                directory / "base/physical-forces-kj-mol-nm.npy", allow_pickle=False
            )
            self.assertEqual(full.shape, (5, 3))
            with np.load(
                directory / "base/electronic/electronic-state.npz", allow_pickle=False
            ) as saved:
                density = saved["density"]
            np.save(directory / "starting-density.npy", density)
            request["initial_density"] = str(directory / "starting-density.npy")
            restarted = runner.evaluate(request, directory / "array-restart")
            self.assertAlmostEqual(
                restarted["electronic_energy_hartree"],
                record["electronic_energy_hartree"],
                delta=1e-9,
            )
            np.testing.assert_allclose(
                np.load(
                    directory / "array-restart/physical-forces-kj-mol-nm.npy",
                    allow_pickle=False,
                ),
                full,
                rtol=1e-8,
                atol=1e-6,
            )
            request["initial_density"] = str(
                directory / "base/electronic/electronic-state.npz"
            )
            original = Path(request["state_xml"]).read_text()
            energies = []
            for sign in (1, -1):
                prepared = runner.prepare(request)
                xyz = prepared["positions_angstrom"] * 0.1
                xyz[3, 0] += sign * 1e-5
                method = openmm.VerletIntegrator(0.0005)
                context = openmm.Context(
                    prepared["source_system"],
                    method,
                    openmm.Platform.getPlatformByName("Reference"),
                )
                try:
                    context.setPositions(xyz * unit.nanometer)
                    Path(request["state_xml"]).write_text(
                        openmm.XmlSerializer.serialize(
                            context.getState(getPositions=True)
                        )
                    )
                finally:
                    del context, method
                shifted = runner.evaluate(request, directory / f"shift-{sign}")
                energies.append(shifted["total_energy_kj_mol"])
                Path(request["state_xml"]).write_text(original)
            self.assertAlmostEqual(
                full[3, 0], -(energies[0] - energies[1]) / 2e-5, delta=2e-3
            )


if __name__ == "__main__":
    unittest.main()
