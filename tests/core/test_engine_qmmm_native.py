"""Native data boundaries, explicit partitions and complete quantum force mapping."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    "qmmm_native",
    Path(__file__).resolve().parents[2] / "src/alambiq/engines/qmmm_native.py",
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class DataBoundaryTests(unittest.TestCase):
    def test_types_and_complete_document_precede_native_deserialization(self):
        for text in (
            '<System type="PythonForce"/>',
            '<State type="System"/>',
            '<System type="System"><Force type="PythonForce"/></System>',
            '<System type="System"><Force type="CustomCVForce"><Force type="Pyth&#111;nForce"/></Force></System>',
            '<System type="System"><Force type="UncommissionedForce"/></System>',
            '<System type="System"><Broken></System>',
            '<!DOCTYPE System [<!ENTITY x "System">]><System type="&x;"/>',
        ):
            deserialize = Mock()
            with self.subTest(text=text), self.assertRaises(ValueError):
                native.deserialize_data(text, "System", deserialize)
            deserialize.assert_not_called()

    def test_original_data_bytes_pass_to_native_once(self):
        text = '<System type="System"><!-- PythonForce --><Particle type="2" name="PythonForce"/></System>'
        deserialize = Mock(return_value="system")
        self.assertEqual(native.deserialize_data(text, "System", deserialize), "system")
        deserialize.assert_called_once_with(text)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required")
class QuantumMappingTests(unittest.TestCase):
    def test_geometry_charges_and_link_derivative_preserve_physical_inputs(self):
        import numpy as np

        xyz = np.array(
            [[0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [2.5, 1.0, 0.3], [3.0, 2.0, 1.0]]
        )
        charges = np.array([0.1, 0.2, -0.3, 0.0])
        before = xyz.copy(), charges.copy()
        link = {
            "quantum_parent": 1,
            "classical_parent": 2,
            "alpha": 1.4,
            "cancelled_coupling_indices": [2],
        }
        mapping = native.partition(xyz, charges, [0, 1], link)
        np.testing.assert_array_equal(xyz, before[0])
        np.testing.assert_array_equal(charges, before[1])
        np.testing.assert_allclose(
            mapping["qm_coordinates"][-1], xyz[1] + (xyz[2] - xyz[1]) / 1.4
        )
        np.testing.assert_array_equal(mapping["mm_charges"], [0.0, 0.0])
        fq = -2 * mapping["qm_coordinates"]
        fm = -4 * mapping["mm_coordinates"]
        force = native.scatter_forces(mapping, fq, fm)

        def energy(positions):
            value = native.partition(positions, charges, [0, 1], link)
            return (value["qm_coordinates"] ** 2).sum() + 2 * (
                value["mm_coordinates"] ** 2
            ).sum()

        for atom in range(4):
            for axis in range(3):
                plus, minus = xyz.copy(), xyz.copy()
                plus[atom, axis] += 1e-5
                minus[atom, axis] -= 1e-5
                self.assertAlmostEqual(
                    force[atom, axis],
                    -(energy(plus) - energy(minus)) / 2e-5,
                    delta=1e-8,
                )
        np.testing.assert_allclose(force.sum(axis=0), fq.sum(axis=0) + fm.sum(axis=0))

    def test_uncapped_mapping_and_invalid_regions(self):
        import numpy as np

        xyz = np.arange(12.0).reshape(4, 3)
        mapping = native.partition(xyz, [0.0, 0.1, -0.1, 0.0], [2, 0])
        np.testing.assert_array_equal(mapping["qm_coordinates"], xyz[[2, 0]])
        result = native.scatter_forces(mapping, np.ones((2, 3)), np.full((2, 3), 2.0))
        np.testing.assert_array_equal(result[[2, 0]], 1.0)
        np.testing.assert_array_equal(result[[1, 3]], 2.0)
        for quantum in ([], [0, 0], [True], [4], [-1]):
            with self.assertRaises(ValueError):
                native.partition(xyz, [0.0] * 4, quantum)
        for link in (
            {},
            {
                "quantum_parent": 0,
                "classical_parent": 1,
                "alpha": 1.0,
                "cancelled_coupling_indices": [],
            },
            {
                "quantum_parent": 0,
                "classical_parent": 1,
                "alpha": 1.4,
                "cancelled_coupling_indices": [0],
            },
        ):
            with self.assertRaises(ValueError):
                native.partition(xyz, [0.0] * 4, [0], link)
        with self.assertRaises(ValueError):
            native.scatter_forces(mapping, np.ones((1, 3)), np.ones((2, 3)))

    def test_exact_callback_cache_keeps_cell_geometry_and_result_ownership(self):
        import numpy as np

        calculate = Mock(side_effect=lambda x: (float((x * x).sum()), -20 * x))
        cache = native.ExactElectronicCache(calculate, 2, np.eye(3) * 3)
        xyz = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        energy, force = cache.evaluate(xyz, np.eye(3) * 3)
        self.assertAlmostEqual(energy, 91.0)
        force[:] = 500.0
        np.testing.assert_allclose(cache.evaluate(xyz, np.eye(3) * 3)[1], -200 * xyz)
        self.assertEqual(calculate.call_count, 1)
        self.assertEqual(cache.hits, 1)
        moved = xyz.copy()
        moved[0, 0] += 1e-12
        cache.evaluate(moved, np.eye(3) * 3)
        cache.evaluate(xyz, np.eye(3) * 3)
        self.assertEqual(calculate.call_count, 3)
        with self.assertRaises(ValueError):
            cache.evaluate(xyz, np.eye(3) * 3.00001)
        broken = native.ExactElectronicCache(
            lambda x: (0.0, np.zeros((1, 3))), 2, np.eye(3)
        )
        with self.assertRaises(ValueError):
            broken.evaluate(xyz, np.eye(3))


@unittest.skipUnless(importlib.util.find_spec("openmm"), "OpenMM 8.6 required")
class ClassicalComplementTests(unittest.TestCase):
    @staticmethod
    def system():
        import openmm

        system = openmm.System()
        for _ in range(6):
            system.addParticle(12.0)
        nonbonded = openmm.NonbondedForce()
        for i in range(6):
            nonbonded.addParticle((i - 2) * 0.1, 0.2, 0.3)
        system.addForce(nonbonded)
        bonds = openmm.HarmonicBondForce()
        bonds.addBond(0, 1, 0.14, 200.0)
        bonds.addBond(1, 2, 0.15, 300.0)
        bonds.addBond(2, 3, 0.16, 400.0)
        angles = openmm.HarmonicAngleForce()
        angles.addAngle(0, 1, 2, 1.9, 10.0)
        angles.addAngle(1, 2, 3, 2.0, 20.0)
        torsions = openmm.PeriodicTorsionForce()
        torsions.addTorsion(0, 1, 2, 4, 3, 0.1, 2.0)
        torsions.addTorsion(1, 2, 3, 5, 2, 0.2, 3.0)
        for force in (bonds, angles, torsions):
            system.addForce(force)
        system.addConstraint(0, 1, 0.14)
        system.addConstraint(2, 3, 0.16)
        return system

    def test_partition_ledger_source_preservation_and_charge_convention(self):
        import openmm
        from openmm import unit

        source = self.system()
        original = openmm.XmlSerializer.serialize(source)
        complement, ledger = native.classical_complement(source, [0, 1, 4], 2, 4.0)
        self.assertEqual(openmm.XmlSerializer.serialize(source), original)
        for name in (
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "PeriodicTorsionForce",
            "constraints",
        ):
            self.assertEqual(
                ledger[name]["source"],
                ledger[name]["retained"] + ledger[name]["removed"],
            )
            self.assertEqual(ledger[name]["retained"], 1)
        nonbonded = complement.getForce(0)
        self.assertEqual(nonbonded.getNumExceptions(), 6)
        self.assertEqual(
            nonbonded.getParticleParameters(1)[0].value_in_unit(unit.elementary_charge),
            0.0,
        )
        self.assertAlmostEqual(
            nonbonded.getParticleParameters(3)[0].value_in_unit(unit.elementary_charge),
            0.2,
        )
        restored = native.deserialize_data(
            openmm.XmlSerializer.serialize(complement),
            "System",
            openmm.XmlSerializer.deserialize,
        )
        self.assertEqual(
            openmm.XmlSerializer.serialize(restored),
            openmm.XmlSerializer.serialize(complement),
        )

    def test_explicit_static_molecular_force_inventory(self):
        import openmm

        def offset(system):
            nb = system.getForce(0)
            nb.addGlobalParameter("lambda", 0.0)
            nb.addParticleParameterOffset("lambda", 3, 0.1, 0.0, 0.0)

        mutations = [
            lambda s: s.addForce(openmm.CustomExternalForce("0")),
            lambda s: s.getForce(0).setNonbondedMethod(
                openmm.NonbondedForce.CutoffNonPeriodic
            ),
            lambda s: s.addConstraint(0, 5, 0.2),
            offset,
            lambda s: s.setParticleMass(5, 0.0),
        ]
        for mutate in mutations:
            source = self.system()
            mutate(source)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                native.classical_complement(source, [0, 1, 4], 2, 1.0)

    def test_complete_complement_derivatives(self):
        import numpy as np
        import openmm
        from openmm import unit

        system, _ = native.classical_complement(self.system(), [0, 1, 4], 2, 1.0)
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName("Reference")
        )
        xyz = (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.4, 0.1, 0.0],
                    [2.0, 1.5, 0.2],
                    [3.0, 1.9, 1.3],
                    [0.1, 2.0, 1.0],
                    [4.2, 0.6, 0.8],
                ]
            )
            * 0.1
        )
        try:
            context.setPositions(xyz)
            forces = (
                context.getState(getForces=True)
                .getForces(asNumpy=True)
                .value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            )
            for atom in range(6):
                for axis in range(3):
                    energies = []
                    for sign in (1, -1):
                        moved = xyz.copy()
                        moved[atom, axis] += sign * 1e-6
                        context.setPositions(moved)
                        energies.append(
                            context.getState(getEnergy=True)
                            .getPotentialEnergy()
                            .value_in_unit(unit.kilojoule_per_mole)
                        )
                    self.assertAlmostEqual(
                        forces[atom, axis],
                        -(energies[0] - energies[1]) / 2e-6,
                        delta=2e-4,
                    )
        finally:
            del context, integrator


if __name__ == "__main__":
    unittest.main()
