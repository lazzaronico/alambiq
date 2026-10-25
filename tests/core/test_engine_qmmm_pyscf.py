"""Molecular electronic inputs, Gaussian embedding and full moving-grid derivatives."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "qmmm_pyscf",
    Path(__file__).resolve().parents[2] / "src/alambiq/engines/qmmm_pyscf.py",
)
electronic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(electronic)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required")
class InputTests(unittest.TestCase):
    def test_diis_conditioning_is_local_and_restores_the_native_metric(self):
        import numpy as np

        class Native:
            def get_num_vec(self):
                return 2

            def extrapolate(self, count):
                self.observed = self._H.copy()
                return count

        first, second = (SimpleNamespace(DIIS=Native) for _ in range(2))
        history = electronic.condition_diis(first)
        self.assertIs(second.DIIS, Native)
        solver = first.DIIS()
        solver._H = np.array([[0, 1, 1], [1, 1e-20, 0], [1, 0, 4e-20]])
        original = solver._H.copy()
        self.assertEqual(solver.extrapolate(), 2)
        np.testing.assert_array_equal(solver._H, original)
        np.testing.assert_array_equal(solver.observed[0], original[0])
        np.testing.assert_array_equal(solver.observed[:, 0], original[:, 0])
        np.testing.assert_array_equal(solver.observed[1:, 1:], [[0.25, 0], [0, 1]])
        self.assertEqual(history, [4e-20])

    def test_diis_native_failure_restores_the_complete_history(self):
        import numpy as np

        class Native:
            def extrapolate(self, count):
                self.observed = self._H.copy()
                raise RuntimeError("retained native failure")

        method = SimpleNamespace(DIIS=Native)
        electronic.condition_diis(method)
        solver = method.DIIS()
        solver._H = np.array([[0, 1, 1], [1, 1e-20, 0], [1, 0, 4e-20]])
        original = solver._H.copy()
        with self.assertRaisesRegex(RuntimeError, "retained native failure"):
            solver.extrapolate(2)
        np.testing.assert_array_equal(solver._H, original)
        np.testing.assert_array_equal(solver.observed[1:, 1:], [[0.25, 0], [0, 1]])

    def test_diis_zero_vectors_and_nonfinite_metric_are_explicit(self):
        from unittest.mock import Mock

        import numpy as np

        called = Mock(return_value="native empty history")

        class Native:
            extrapolate = called

        method = SimpleNamespace(DIIS=Native)
        history = electronic.condition_diis(method)
        solver = method.DIIS()
        self.assertEqual(solver.extrapolate(0), "native empty history")
        self.assertEqual(history, [])
        called.assert_called_once_with(0)
        for value in (float("nan"), float("inf")):
            solver._H = np.array([[0, 1], [1, value]])
            original = solver._H.copy()
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                solver.extrapolate(1)
            np.testing.assert_array_equal(solver._H, original)
        self.assertEqual(history, [])
        called.assert_called_once_with(0)

    def test_complete_density_updates_use_the_native_full_matrix(self):
        from unittest.mock import Mock

        method = Mock()
        original = method.get_veff
        electronic.full_density_updates(method)
        method.get_veff(
            mol="molecule", dm="current", dm_last="previous", vhf_last="old", hermi=2
        )
        original.assert_called_once_with(
            mol="molecule", dm="current", dm_last=0, vhf_last=0, hermi=2
        )

    def test_resolved_finite_inputs_and_closed_shell_population(self):
        import numpy as np

        valid = dict(
            elements=[8, 1, 1],
            quantum_xyz=[[0, 0, 0], [0.75, 0, 0.5], [-0.75, 0, 0.5]],
            embedding_xyz=[[3.0, 1.0, 0.5]],
            embedding_charges=[-0.2],
            charge=0,
            grid=3,
            gaussian_radius_angstrom=0.8,
            backend="cpu",
            threads=1,
            memory_mb=1000,
        )
        self.assertEqual(electronic.validate_inputs(**valid)["electrons"], 10)
        for change in (
            {"elements": [8, 1]},
            {"elements": [8, True, 1]},
            {"elements": [8, 1, 119]},
            {"quantum_xyz": np.full((3, 3), np.nan)},
            {"embedding_charges": [float("inf")]},
            {"embedding_xyz": []},
            {"charge": 1},
            {"charge": True},
            {"charge": 10},
            {"grid": True},
            {"grid": -1},
            {"grid": 10},
            {"backend": "automatic"},
            {"gaussian_radius_angstrom": 0},
            {"threads": 0},
            {"memory_mb": float("inf")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                electronic.validate_inputs(**{**valid, **change})

    def test_stationarity_checks_native_convergence_population_and_observed_gradient(
        self,
    ):
        self.assertTrue(electronic.stationary(True, -75.0, 10.0, 10, 1e-8))
        for changes in (
            (False, -75, 10, 10, 1e-8),
            (True, -75, 9.9, 10, 1e-8),
            (True, -75, 10, 10, 1.1e-7),
            (True, float("nan"), 10, 10, 0),
            (True, -75, 10, 10, -1),
        ):
            self.assertFalse(electronic.stationary(*changes))


@unittest.skipUnless(importlib.util.find_spec("pyscf"), "PySCF 2.14.0 required")
class NativeElectronicTests(unittest.TestCase):
    def test_native_gradient_capture_preserves_the_single_kernel_result(self):
        import json
        from unittest.mock import patch

        import numpy as np

        observer = electronic.gradient_components
        calls = []

        def observed(gradient, method, backend, directory):
            native = gradient.kernel

            def record():
                value = native()
                calls.append(value)
                return value

            gradient.kernel = record
            result = observer(gradient, method, backend, directory)
            self.assertIs(result, calls[-1])
            return result

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(electronic, "gradient_components", side_effect=observed),
        ):
            directory = Path(tmp) / "water"
            result = electronic.evaluate_molecular(
                [8, 1, 1],
                [[0, 0, 0], [0.75, 0, 0.5], [-0.75, 0, 0.5]],
                [[2.1, 1.0, 0.4], [-1.7, 0.3, 0.8]],
                [0.3, -0.2],
                charge=0,
                grid=3,
                gaussian_radius_angstrom=0.8,
                backend="cpu",
                threads=1,
                memory_mb=1000,
                directory=directory,
            )
            self.assertEqual(len(calls), 1)
            with np.load(directory / "gradient-components-hartree-bohr.npz") as arrays:
                np.testing.assert_array_equal(
                    arrays["total"], -result["quantum_force_hartree_bohr"]
                )
                np.testing.assert_allclose(
                    arrays["nuclear"] + arrays["electronic"],
                    arrays["total"],
                    rtol=0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    arrays["one_electron_overlap_and_embedding"]
                    + arrays["coulomb"]
                    + arrays["exchange_correlation_with_grid_response"],
                    arrays["electronic"],
                    rtol=0,
                    atol=1e-12,
                )
            record = json.loads((directory / "gradient-components.json").read_text())
            self.assertEqual(record["native_calls"]["extra_force"], 3)
            self.assertEqual(record["native_gradient_kernel_calls"], 1)
            self.assertEqual(record["additional_electronic_or_gradient_evaluations"], 0)

    def test_native_diis_solution_preserves_residual_scale_invariance(self):
        import numpy as np
        from pyscf.scf.diis import CDIIS

        for scale in (1.0, 1e-8, 1e-20, 1e-30, 0.0):
            method = SimpleNamespace(DIIS=CDIIS)
            history = electronic.condition_diis(method)
            solver = method.DIIS()
            solver._H = np.array([[0, 1, 1], [1, scale, 0], [1, 0, 4 * scale]])
            solver.get_num_vec = lambda: 2
            solver.get_vec = lambda i: np.array([1, 2]) if i == 0 else np.array([3, 8])
            original = solver._H.copy()
            # Minimize c1^2 + 4*c2^2 with c1+c2=1: c=(0.8,0.2).
            expected = [1.4, 3.2] if scale else [2.0, 5.0]
            with self.subTest(scale=scale):
                np.testing.assert_allclose(
                    solver.extrapolate(), expected, atol=1e-13, rtol=0
                )
                np.testing.assert_array_equal(solver._H, original)
                self.assertEqual(history, [4 * scale])

    def test_cpu_routes_each_potential_through_full_density_adapter(self):
        from unittest.mock import patch

        import numpy as np

        calls = []
        original = electronic.full_density_updates

        def observed(method):
            result = original(method)
            native = result.get_veff

            def track(*args, **kwargs):
                calls.append((args, kwargs))
                return native(*args, **kwargs)

            result.get_veff = track
            return result

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                electronic, "full_density_updates", side_effect=observed
            ) as adapter,
        ):
            result = electronic.evaluate_molecular(
                [8, 1, 1],
                [[0.0, 0.0, 0.0], [0.75, 0.0, 0.5], [-0.75, 0.0, 0.5]],
                [[2.1, 1.0, 0.4], [-1.7, 0.3, 0.8]],
                [0.3, -0.2],
                charge=0,
                grid=3,
                gaussian_radius_angstrom=0.8,
                backend="cpu",
                threads=1,
                memory_mb=1000,
                directory=Path(tmp) / "water",
            )
            adapter.assert_called_once()
            self.assertGreater(len(calls), 2)
            self.assertTrue(result["observation"]["stationarity_passed"])
            np.testing.assert_allclose(
                result["quantum_force_hartree_bohr"].sum(0)
                + result["embedding_force_hartree_bohr"].sum(0),
                0,
                rtol=0,
                atol=2e-10,
            )

    @staticmethod
    def embedded_model(radii=(0.8, 0.7)):
        from pyscf import dft, gto, qmmm

        mol = gto.M(atom="O 0 0 0; H .76 0 .5; H -.76 0 .5", basis="6-31g**", verbose=0)
        return qmmm.mm_charge(
            dft.RKS(mol),
            [[0.3, 0.4, 0.2], [3.0, -1.0, 2.0]],
            [0.2, -0.3],
            radii=radii,
            unit="Angstrom",
        )

    def test_complete_gaussian_forces_and_saved_stationary_state(self):
        import numpy as np

        xyz = np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.5], [-0.75, 0.0, 0.5]])
        mm = np.array([[2.1, 1.0, 0.4], [-1.7, 0.3, 0.8]])
        charges = [0.3, -0.2]
        with tempfile.TemporaryDirectory() as tmp:

            def run(name, quantum=xyz, classical=mm, density=None):
                return electronic.evaluate_molecular(
                    [8, 1, 1],
                    quantum,
                    classical,
                    charges,
                    charge=0,
                    grid=3,
                    gaussian_radius_angstrom=0.8,
                    backend="cpu",
                    threads=1,
                    memory_mb=1000,
                    directory=Path(tmp) / name,
                    initial_density=density,
                )

            base = run("base")
            self.assertTrue(base["observation"]["stationarity_passed"])
            conditioning = base["observation"]["diis_conditioning"]
            self.assertEqual(conditioning["method"], "unit-maximum-residual-gram-v1")
            self.assertTrue(conditioning["scales"])
            self.assertTrue(
                all(np.isfinite(s) and s >= 0 for s in conditioning["scales"])
            )
            self.assertEqual(base["quantum_force_hartree_bohr"].shape, (3, 3))
            self.assertEqual(base["embedding_force_hartree_bohr"].shape, (2, 3))
            with np.load(
                Path(tmp) / "base/electronic-state.npz", allow_pickle=False
            ) as state:
                density = state["density"]
                np.testing.assert_allclose(
                    (state["mo_coeff"] * state["mo_occ"]) @ state["mo_coeff"].T,
                    density,
                    rtol=0,
                    atol=1e-12,
                )
                self.assertAlmostEqual(
                    np.einsum("ij,ji", density, state["overlap"]), 10.0, delta=1e-8
                )
            repeat = run("repeat", density=density)
            self.assertAlmostEqual(
                base["observation"]["energy_hartree"],
                repeat["observation"]["energy_hartree"],
                delta=1e-9,
            )
            for part, atom, axis in (("quantum", 1, 2), ("embedding", 0, 0)):
                energies = []
                for sign in (1, -1):
                    q, m = xyz.copy(), mm.copy()
                    (q if part == "quantum" else m)[atom, axis] += sign * 1e-4
                    result = run(f"{part}-{sign}", q, m, density)
                    energies.append(result["observation"]["energy_hartree"])
                derivative = -(energies[0] - energies[1]) / 2e-4 * 0.52917721092
                self.assertAlmostEqual(
                    base[f"{part}_force_hartree_bohr"][atom, axis],
                    derivative,
                    delta=2e-6,
                )
            net = base["quantum_force_hartree_bohr"].sum(0) + base[
                "embedding_force_hartree_bohr"
            ].sum(0)
            np.testing.assert_allclose(net, 0.0, rtol=0, atol=2e-10)
            with self.assertRaises(FileExistsError):
                run("base")

    def test_invalid_density_retains_native_failure_record(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "failed"
            with self.assertRaisesRegex(ValueError, "density"):
                electronic.evaluate_molecular(
                    [8, 1, 1],
                    [[0.0, 0.0, 0.0], [0.75, 0.0, 0.5], [-0.75, 0.0, 0.5]],
                    [[2.0, 0.0, 0.0]],
                    [0.2],
                    charge=0,
                    grid=3,
                    gaussian_radius_angstrom=0.8,
                    backend="cpu",
                    threads=1,
                    memory_mb=1000,
                    directory=directory,
                    initial_density=np.zeros((2, 2)),
                )
            self.assertTrue((directory / "failure.json").is_file())
            self.assertTrue((directory / "native.log").is_file())


if __name__ == "__main__":
    unittest.main()
