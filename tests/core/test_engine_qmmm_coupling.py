"""Native molecular force coupling, retained density guesses and umbrella derivatives."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "qmmm_coupling",
    Path(__file__).resolve().parents[2]
    / "src/alambiq/engines/qmmm_coupling.py",
)
coupling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coupling)


@unittest.skipUnless(
    importlib.util.find_spec("openmm") and importlib.util.find_spec("pyscf"),
    "OpenMM 8.6 and PySCF 2.14.0 required",
)
class MolecularCouplingTests(unittest.TestCase):
    def test_ordered_distance_difference_and_harmonic_derivative(self):
        import numpy as np
        import openmm
        from openmm import unit

        xyz = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.5, 0.2, 0.1], [0.8, 0.3, 0.2]]
        )
        specification = {
            "pairs": [[1, 2], [2, 3]],
            "center_angstrom": -0.4,
            "coefficient_kj_mol_angstrom2": 20.0,
        }
        force = coupling.umbrella_force(specification, 4)
        system = openmm.System()
        for _ in xyz:
            system.addParticle(12.0)
        system.addForce(force)
        method = openmm.VerletIntegrator(0.0001)
        context = openmm.Context(
            system, method, openmm.Platform.getPlatformByName("Reference")
        )
        try:
            context.setPositions(xyz)
            state = context.getState(getEnergy=True, getForces=True)
            q = 10 * (np.linalg.norm(xyz[0] - xyz[1]) - np.linalg.norm(xyz[1] - xyz[2]))
            self.assertAlmostEqual(
                coupling.coordinate_angstrom(xyz, specification["pairs"]), q
            )
            self.assertAlmostEqual(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                20 * (q + 0.4) ** 2,
            )
            native = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            for i in range(4):
                for axis in range(3):

                    def energy(delta, i=i, axis=axis):
                        x = xyz.copy()
                        x[i, axis] += delta
                        return (
                            20
                            * (
                                coupling.coordinate_angstrom(x, specification["pairs"])
                                + 0.4
                            )
                            ** 2
                        )

                    self.assertAlmostEqual(
                        native[i, axis],
                        -(energy(1e-6) - energy(-1e-6)) / 2e-6,
                        delta=2e-7,
                    )
        finally:
            del context, method
        for invalid in (
            [[0, 2], [2, 3]],
            [[True, 2], [2, 3]],
            [[1, 1], [2, 3]],
            [[1, 2], [2, 1]],
            [[1, 5], [2, 3]],
        ):
            with self.assertRaises(ValueError):
                coupling.umbrella_force({**specification, "pairs": invalid}, 4)
        for k in (0.0, -1.0, float("nan"), True):
            with self.assertRaises(ValueError):
                coupling.umbrella_force(
                    {**specification, "coefficient_kj_mol_angstrom2": k}, 4
                )

    def test_exact_context_composition_cache_and_density_ownership(self):
        import numpy as np
        import openmm
        from openmm import unit
        from test_engine_qmmm_state import PreparedStateTests

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = PreparedStateTests.source(root)
            prepared = coupling.runner.prepare(request)
            original = [
                openmm.XmlSerializer.serialize(prepared[k])
                for k in ("classical_system", "external_system")
            ]
            electronic = coupling.runner._component("qmmm_pyscf")
            guesses = []

            def calculate(elements, q, m, c, **kwargs):
                guesses.append(
                    None
                    if kwargs["initial_density"] is None
                    else kwargs["initial_density"].copy()
                )
                self.assertEqual(elements, [8, 1, 1])
                return {
                    "observation": {"energy_hartree": -1.0},
                    "density": np.eye(2) * len(guesses),
                    "quantum_force_hartree_bohr": np.full_like(q, 1e-4),
                    "embedding_force_hartree_bohr": np.full_like(m, -2e-4),
                }

            with patch.object(electronic, "evaluate_molecular", side_effect=calculate):
                callback = coupling.MolecularElectronicForce(
                    prepared, request["parameters"], root / "calls", None
                )
                system, cache = coupling.compose_system(prepared, callback, None)
                method = openmm.VerletIntegrator(0.0001)
                context = openmm.Context(
                    system, method, openmm.Platform.getPlatformByName("Reference")
                )
                try:
                    xyz = prepared["positions_angstrom"] * 0.1
                    context.setPositions(xyz)
                    state = context.getState(getEnergy=True, getForces=True)
                    eq = -callback.hartree_kj_mol
                    ec, fc = coupling.runner._classical_observation(
                        prepared["classical_system"], prepared["positions_angstrom"]
                    )
                    ex, fx = coupling.runner._classical_observation(
                        prepared["external_system"], prepared["positions_angstrom"]
                    )
                    self.assertAlmostEqual(
                        state.getPotentialEnergy().value_in_unit(
                            unit.kilojoule_per_mole
                        ),
                        eq + ec + ex,
                        delta=1e-8,
                    )
                    fq = (
                        np.array([[1e-4] * 3] * 3 + [[-2e-4] * 3] * 2)
                        * callback.hartree_kj_mol
                        / (callback.bohr_angstrom * 0.1)
                    )
                    np.testing.assert_allclose(
                        state.getForces(asNumpy=True).value_in_unit(
                            unit.kilojoule_per_mole / unit.nanometer
                        ),
                        fq + fc + fx,
                        rtol=0,
                        atol=1e-9,
                    )
                    context.getState(getEnergy=True, getForces=True)
                    self.assertEqual(callback.calls, 1)
                    self.assertGreater(cache.hits, 0)
                    xyz[0, 0] += 1e-5
                    context.setPositions(xyz)
                    context.getState(getEnergy=True)
                    self.assertEqual(callback.calls, 2)
                    self.assertIsNone(guesses[0])
                    np.testing.assert_array_equal(guesses[1], np.eye(2))
                    self.assertTrue(
                        (root / "calls/000000/positions-angstrom.npy").is_file()
                    )
                    self.assertTrue(
                        (root / "calls/000001/positions-angstrom.npy").is_file()
                    )
                finally:
                    del context, method
            self.assertEqual(
                original,
                [
                    openmm.XmlSerializer.serialize(prepared[k])
                    for k in ("classical_system", "external_system")
                ],
            )
            with self.assertRaises(FileExistsError):
                coupling.MolecularElectronicForce(
                    prepared, request["parameters"], root / "calls", None
                )

    def test_failed_electronic_call_retains_geometry_and_previous_density(self):
        import json

        import numpy as np
        from test_engine_qmmm_state import PreparedStateTests

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = PreparedStateTests.source(root)
            prepared = coupling.runner.prepare(request)
            initial = np.eye(2)
            callback = coupling.MolecularElectronicForce(
                prepared, request["parameters"], root / "calls", initial
            )
            initial[0, 0] = 99
            with patch.object(
                coupling.runner._component("qmmm_pyscf"),
                "evaluate_molecular",
                side_effect=ValueError("SCF failed"),
            ):
                with self.assertRaisesRegex(ValueError, "SCF failed"):
                    callback(prepared["positions_angstrom"])
            np.testing.assert_array_equal(callback.density, np.eye(2))
            self.assertEqual(callback.calls, 0)
            self.assertTrue((root / "calls/000000/positions-angstrom.npy").is_file())
            self.assertEqual(
                json.loads((root / "calls/000000/failure.json").read_text())["message"],
                "SCF failed",
            )
            with self.assertRaisesRegex(ValueError, "failed electronic"):
                callback(prepared["positions_angstrom"])

    def test_real_tiny_molecular_native_integration(self):
        import numpy as np
        import openmm
        from openmm import unit
        from test_engine_qmmm_state import PreparedStateTests

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = PreparedStateTests.source(root)
            prepared = coupling.runner.prepare(request)
            errors = []
            for timestep, steps in ((0.000125, 4), (0.0000625, 8)):
                callback = coupling.MolecularElectronicForce(
                    prepared, request["parameters"], root / str(steps), None
                )
                system, cache = coupling.compose_system(prepared, callback, None)
                method = openmm.VerletIntegrator(timestep)
                context = openmm.Context(
                    system, method, openmm.Platform.getPlatformByName("Reference")
                )
                try:
                    context.setPositions(prepared["positions_angstrom"] * 0.1)
                    context.setVelocities(np.zeros((5, 3)))
                    start = context.getState(getEnergy=True)
                    initial = (
                        start.getPotentialEnergy() + start.getKineticEnergy()
                    ).value_in_unit(unit.kilojoule_per_mole)
                    method.step(steps)
                    last = context.getState(getEnergy=True, getPositions=True)
                    final = (
                        last.getPotentialEnergy() + last.getKineticEnergy()
                    ).value_in_unit(unit.kilojoule_per_mole)
                    errors.append(abs(final - initial))
                    self.assertTrue(
                        np.isfinite(
                            last.getPositions(asNumpy=True).value_in_unit(
                                unit.nanometer
                            )
                        ).all()
                    )
                    self.assertEqual(callback.calls, steps + 1)
                    self.assertGreater(cache.hits, 0)
                finally:
                    del context, method
            self.assertLess(errors[1], 0.01)
            self.assertLess(errors[1], errors[0] / 2)


if __name__ == "__main__":
    unittest.main()
