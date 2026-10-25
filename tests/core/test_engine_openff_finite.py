"""Whole molecules, explicit periodic imaging and finite physical boundaries.

Unit tests of `src/alambiq/engines/openff_finite_environment.py`. They need NumPy and OpenMM,
which only the pinned OpenFF environment has, and skip elsewhere; `test_engine_suites.py`
runs them there.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ENGINES = Path(__file__).resolve().parents[2] / "src/alambiq/engines"
sys.path.insert(0, str(ENGINES))

try:
    import numpy  # noqa: F401
    import openmm  # noqa: F401
except ImportError:
    finite = None
else:
    spec = importlib.util.spec_from_file_location(
        "openff_finite_environment", ENGINES / "openff_finite_environment.py"
    )
    finite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finite)

VALUES = dict(
    solute_residues=[1, 2],
    anchor_residue=1,
    protected_atoms=[1, 2],
    solvent_radius_angstrom=25.0,
    inner_solute_radius_angstrom=20.0,
    position_coefficient_kj_mol_nm2=2092.0,
    wall_coefficient_kj_mol_nm2=4184.0,
    threads=2,
)


def parameters(supplied):
    return {**VALUES, **supplied}


def request_parameters():
    return dict(solute_residues=[1, 2], anchor_residue=1, protected_atoms=[1, 2])


@unittest.skipIf(finite is None, "NumPy and OpenMM required (the OpenFF environment)")
class FiniteEnvironmentTests(unittest.TestCase):
    def test_unwrapping_preserves_complete_molecules_and_translation(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy optional numerical dependency")
        xyz = np.array(
            [
                [9.8, 0, 0],
                [0.2, 0, 0],
                [9.9, 0.3, 0],
                [0.1, 1.2, 0],
                [0.15, 1.3, 0],
                [0.2, 1.2, 0],
                [3.0, 0, 0],
            ]
        )
        masses = np.array([12.0, 16.0, 1.0, 16.0, 1.0, 1.0, 23.0])
        bonds = [(0, 1), (1, 2), (2, 0), (3, 4), (3, 5)]
        before = xyz.copy()
        imaged, groups, center = finite.image_components(
            xyz, np.eye(3) * 10, bonds, masses, [0, 1, 2], [0, 1]
        )
        self.assertEqual(groups, [[0, 1, 2], [3, 4, 5], [6]])
        self.assertEqual(
            finite.select_components(imaged, groups, [0, 1, 2], center, 1.5),
            list(range(6)),
        )
        np.testing.assert_allclose(
            imaged[1] - imaged[0], [0.4, 0, 0], atol=1e-14, rtol=0
        )
        np.testing.assert_array_equal(xyz, before)
        shift = np.array([0.3, 0.1, -0.2])
        translated, translated_groups, translated_center = finite.image_components(
            xyz + shift, np.eye(3) * 10, bonds, masses, [0, 1, 2], [0, 1]
        )
        np.testing.assert_allclose(translated, imaged + shift, atol=1e-14, rtol=0)
        self.assertEqual(groups, translated_groups)
        self.assertEqual(center, translated_center)
        with self.assertRaisesRegex(ValueError, "complete bonded molecules"):
            finite.select_components(imaged, groups, [0], center, 1.5)

    def test_invalid_cells_anchor_cycles_and_partial_denominators_are_rejected(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy optional numerical dependency")
        xyz = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0]], float)
        with self.assertRaisesRegex(ValueError, "inconsistent periodic image"):
            finite.image_components(
                xyz, np.eye(3) * 10, [(0, 1), (1, 2), (2, 0)], [12] * 3, [0, 1, 2], [0]
            )
        for cell, masses, anchor, heavy in [
            (np.array([[10, 1, 0], [0, 10, 0], [0, 0, 10]]), [12] * 3, [0, 1, 2], [0]),
            (np.eye(3) * 10, [0, 12, 12], [0, 1, 2], [0]),
            (np.eye(3) * 10, [12] * 3, [0, 1], [0]),
            (np.eye(3) * 10, [12] * 3, [0, 1, 2], []),
        ]:
            with (
                self.subTest(cell=cell, masses=masses, anchor=anchor, heavy=heavy),
                self.assertRaises(ValueError),
            ):
                finite.image_components(
                    xyz, cell, [(0, 1), (1, 2)], masses, anchor, heavy
                )
        with self.assertRaisesRegex(ValueError, "complete atom groups"):
            finite.select_components(xyz, [[0], [1]], [0], 0, 2)

    def test_native_boundary_matches_independent_energy_force_and_translation(self):
        try:
            import numpy as np
            import openmm
            from openmm import unit
        except ImportError:
            self.skipTest("OpenMM optional native numerical dependency")
        masses = np.array([12.0, 1.0, 12.0, 1.0, 16.0, 1.0, 1.0, 23.0])
        elements = np.array([6, 1, 6, 1, 8, 1, 1, 11])
        reference = np.array(
            [
                [0, 0, 0],
                [0.1, 0, 0],
                [1, 0, 0],
                [1.1, 0, 0],
                [1.1, 0.1, 0],
                [1.18, 0.1, 0],
                [1.1, 0.18, 0],
                [1.5, 0, 0],
            ]
        )
        values = parameters(
            {
                **request_parameters(),
                "protected_atoms": [1],
                "inner_solute_radius_angstrom": 5,
                "solvent_radius_angstrom": 10,
            }
        )
        observed = reference + np.random.default_rng(94317).normal(
            0, 0.01, reference.shape
        )
        groups = [[4, 5, 6], [7]]
        expected_force = np.zeros_like(reference)
        delta = observed[2] - reference[2]
        expected_energy = values["position_coefficient_kj_mol_nm2"] * np.sum(
            delta * delta
        )
        expected_force[2] = -2 * values["position_coefficient_kj_mol_nm2"] * delta
        for group in groups:
            weights = masses[group] / sum(masses[group])
            vector = np.sum(observed[group] * weights[:, None], axis=0)
            radius = np.linalg.norm(vector)
            excess = max(0, radius - 1.0)
            expected_energy += values["wall_coefficient_kj_mol_nm2"] * excess * excess
            if excess > 0:
                expected_force[group] -= (
                    2
                    * values["wall_coefficient_kj_mol_nm2"]
                    * excess
                    * weights[:, None]
                    * vector
                    / radius
                )
        for shift in [np.zeros(3), np.array([0.37, -0.28, 0.91])]:
            system, restrained = finite.boundary_system(
                masses,
                elements,
                reference + shift,
                [[0, 1], [2, 3]],
                groups,
                [0],
                shift,
                values,
            )
            self.assertEqual(restrained, [2])
            integrator = openmm.VerletIntegrator(0.001)
            context = openmm.Context(
                system, integrator, openmm.Platform.getPlatformByName("Reference")
            )
            context.setPositions(observed + shift)
            state = context.getState(getEnergy=True, getForces=True)
            self.assertAlmostEqual(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                expected_energy,
                delta=1e-7,
            )
            np.testing.assert_allclose(
                state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                ),
                expected_force,
                atol=1e-7,
                rtol=0,
            )
            del context, integrator
        with self.assertRaisesRegex(ValueError, "unrestrained solute"):
            finite.boundary_system(
                masses,
                elements,
                reference,
                [[0, 1], [2, 3]],
                groups,
                [0, 2],
                [0, 0, 0],
                values,
            )

    def test_physical_projection_rejects_cut_bonds_and_changed_retained_terms(self):
        try:
            import openmm
        except ImportError:
            self.skipTest("OpenMM optional native numerical dependency")

        def model(count):
            system = openmm.System()
            for _ in range(count):
                system.addParticle(12)
            bond = openmm.HarmonicBondForce()
            bond.addBond(0, 1, 0.15, 500)
            if count == 4:
                bond.addBond(2, 3, 0.15, 500)
            system.addForce(bond)
            system.addForce(openmm.HarmonicAngleForce())
            system.addForce(openmm.PeriodicTorsionForce())
            force = openmm.NonbondedForce()
            for _ in range(count):
                force.addParticle(0.1, 0.2, 0.3)
            force.addException(0, 1, 0.01, 0.2, 0.1)
            if count == 4:
                force.addException(2, 3, 0.01, 0.2, 0.1)
            system.addForce(force)
            return system

        complete, subset = model(4), model(2)
        self.assertEqual(
            finite.physical_identity(complete, [0, 1]), finite.physical_identity(subset)
        )
        self.assertEqual(
            finite.physical_identity(complete, [2, 3]), finite.physical_identity(subset)
        )
        self.assertEqual(
            finite.physical_identity(complete),
            finite.physical_identity(complete, [0, 1, 2, 3]),
        )
        with self.assertRaisesRegex(ValueError, "complete bonded molecules"):
            finite.physical_identity(complete, [0])
        subset.getForce(0).setBondParameters(0, 0, 1, 0.15, 501)
        self.assertNotEqual(
            finite.physical_identity(complete, [0, 1]), finite.physical_identity(subset)
        )
        with self.assertRaisesRegex(ValueError, "ordered subset"):
            finite.physical_identity(complete, [2, 0])


if __name__ == "__main__":
    unittest.main()
