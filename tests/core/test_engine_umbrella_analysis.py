"""Known-density umbrella reconstruction and identity/numerical failure checks.

Unit tests of `src/alambiq/engines/umbrella_analysis.py`; they need PyMBAR and skip where it
is missing (`test_engine_suites.py` runs them in the pinned analysis environment).
"""

import copy
import hashlib
import importlib.util
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "umbrella_runner",
    Path(__file__).resolve().parents[2] / "src/alambiq/engines/umbrella_analysis.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(samples=4000):
    import numpy as np

    rng = np.random.default_rng(62419)
    temperature = 300.0
    rt = 0.00831446261815324 * temperature
    windows = [
        {"id": f"w{i}", "center_angstrom": center, "k_kj_mol_angstrom2": 5.0}
        for i, center in enumerate([-1.5, -0.75, 0.0, 0.75, 1.5])
    ]
    # Exact Gaussian samples of U(q)=q^2 + 5*(q-center)^2 in kJ/mol.
    replicas = []
    for replica in range(2):
        trajectories = []
        for window in windows:
            q = rng.normal(
                5 * window["center_angstrom"] / 6, math.sqrt(rt / 12), samples
            )
            trajectories.append(
                {
                    "window": window["id"],
                    "trajectory_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
                    "discard_frames": 0,
                    "frame_interval_fs": 1.0,
                    "coordinate_angstrom": q.tolist(),
                }
            )
        replicas.append({"id": f"r{replica}", "windows": trajectories})
    return {
        "schema_version": "0.1",
        "title": "Analytic harmonic-density fixture",
        "system_sha256": "a" * 64,
        "hamiltonian_sha256": "b" * 64,
        "temperature_kelvin": temperature,
        "coordinate": {
            "unit": "angstrom",
            "definition": "distance(1,2)-distance(3,4)",
            "positive_distance_atoms": [1, 2],
            "negative_distance_atoms": [3, 4],
        },
        "bias_convention": "K*(q-center)^2",
        "windows": windows,
        "replicas": replicas,
    }


def parameters():
    return {
        "bin_edges_angstrom": [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
        "reference_bin": 3,
        "reactant_bins": [2, 3],
        "product_bins": [5, 6],
        "transition_bins": [4],
        "block_counts": [2],
    }


def paired_fixture(samples=4000):
    data = fixture(samples)
    data["schema_version"] = "0.2"
    data["target"] = {
        "hamiltonian_sha256": "c" * 64,
        "energy_unit": "kJ/mol",
        "energy_convention": "unbiased-potential-energy",
        "configuration_measure": "same-positions-cell-composition-and-constraints",
    }
    for replica in data["replicas"]:
        for trace in replica["windows"]:
            q = trace["coordinate_angstrom"]
            frames = [hashlib.sha256(float(x).hex().encode()).hexdigest() for x in q]
            reference = [x * x for x in q]
            target = [2 * (x - 0.2) ** 2 for x in q]
            trace["energy_pairs"] = {
                "reference_source_sha256": hashlib.sha256(
                    json.dumps(reference).encode()
                ).hexdigest(),
                "target_source_sha256": hashlib.sha256(
                    json.dumps(target).encode()
                ).hexdigest(),
                "reference_frame_sha256": frames,
                "target_frame_sha256": frames.copy(),
                "reference_kj_mol": reference,
                "target_kj_mol": target,
            }
    return data


@unittest.skipUnless(importlib.util.find_spec("numpy"), "quantitative extra required")
class PairedUmbrellaInputTests(unittest.TestCase):
    def test_explicit_profile_and_schema(self):
        data = paired_fixture(100)
        runner.validate_umbrella(data, parameters(), runner.TARGET_PROFILE)
        for source, profile in (
            (data, runner.PROFILE),
            (fixture(100), runner.TARGET_PROFILE),
        ):
            with self.assertRaisesRegex(ValueError, "schema"):
                runner.validate_umbrella(source, parameters(), profile)
        with self.assertRaisesRegex(ValueError, "profile"):
            runner.validate_umbrella(data, parameters(), "unknown")

    def test_pair_identity_length_units_and_energy_validation(self):
        mutations = [
            lambda d: d["target"].update(energy_unit="hartree"),
            lambda d: d["target"].update(energy_convention="includes-umbrella"),
            lambda d: d["target"].update(
                configuration_measure="relaxed-target-geometry"
            ),
            lambda d: d["target"].update(hamiltonian_sha256="invalid"),
        ]
        for key, value in (
            ("reference_source_sha256", "invalid"),
            ("target_frame_sha256", ["d" * 64] * 100),
            ("reference_kj_mol", [0.0] * 99),
            ("target_kj_mol", [float("nan")] * 100),
            ("target_kj_mol", [True] * 100),
        ):
            mutations.append(
                lambda d, k=key, v=value: d["replicas"][0]["windows"][0][
                    "energy_pairs"
                ].update({k: v})
            )
        for mutate in mutations:
            data = paired_fixture(100)
            mutate(data)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                runner.validate_umbrella(data, parameters(), runner.TARGET_PROFILE)


@unittest.skipUnless(
    importlib.util.find_spec("pymbar"), "PyMBAR 4.2.0 environment required"
)
class PairedUmbrellaNumericalTests(unittest.TestCase):
    def analyze(self, data):
        return runner.analyze_umbrella(data, parameters(), runner.TARGET_PROFILE)

    def test_shifted_target_matches_analytic_density_and_basin(self):
        import numpy as np
        from scipy.special import ndtr

        result = self.analyze(paired_fixture())
        p = parameters()
        rt = runner.GAS_CONSTANT_KJ_MOL_K * 300
        edges = np.array(p["bin_edges_angstrom"])
        mass = np.diff(ndtr((edges - 0.2) / math.sqrt(rt / 4)))
        expected = -rt * np.log(mass / np.diff(edges) / (mass[3] / np.diff(edges)[3]))
        np.testing.assert_allclose(
            result["summary"]["pmf_mean_kj_mol"], expected, atol=0.35, rtol=0
        )
        self.assertAlmostEqual(
            result["summary"]["basin_delta_g_mean_kj_mol"],
            -rt * math.log(mass[5:7].sum() / mass[2:4].sum()),
            delta=0.3,
        )
        self.assertEqual(result["hamiltonian_sha256"], "c" * 64)
        self.assertEqual(result["sampling_hamiltonian_sha256"], "b" * 64)
        self.assertEqual(result["method_profile"], runner.TARGET_PROFILE)
        for replica in result["replicas"]:
            self.assertEqual(len(replica["state_effective_samples"]), 5)
            self.assertGreater(
                replica["reweighting"]["reference_effective_samples"], 100
            )
            self.assertGreater(replica["reweighting"]["normalized_entropy"], 0)
            self.assertEqual(len(replica["reweighting"]["bins"]), 8)

    def test_zero_correction_matches_original_estimator(self):
        import numpy as np

        data = fixture(1000)
        samples, _ = runner._sample(
            data["replicas"][0]["windows"], data["windows"], (0, 1)
        )
        original = runner._estimate(samples, data["windows"], 300, parameters())
        paired = runner._estimate(
            samples,
            data["windows"],
            300,
            parameters(),
            np.zeros(sum(map(len, samples))),
        )
        for key in (
            "pmf_kj_mol",
            "pmf_asymptotic_error_kj_mol",
            "bin_probability_covariance",
            "bin_effective_samples",
            "overlap_matrix",
        ):
            np.testing.assert_allclose(paired[key], original[key], atol=1e-10)
        self.assertAlmostEqual(paired["reweighting"]["target_reference_ess_ratio"], 1)

    def test_target_covariance_matches_direct_importance_sampling(self):
        import numpy as np

        rng = np.random.default_rng(91384)
        samples = [rng.normal(size=1000) for _ in range(2)]
        q = np.concatenate(samples)
        delta = 0.5 * q**2 - q
        windows = [
            {"id": str(i), "center_angstrom": center, "k_kj_mol_angstrom2": 1e-12}
            for i, center in enumerate((-1, 1))
        ]
        result = runner._estimate(samples, windows, 300, parameters(), delta)
        weights = np.exp(-delta / (runner.GAS_CONSTANT_KJ_MOL_K * 300))
        weights /= weights.sum()
        bins = np.searchsorted(parameters()["bin_edges_angstrom"], q, side="right") - 1
        indicators = np.array([bins == i for i in range(8)], dtype=float)
        mass = indicators @ weights
        centered = indicators - mass[:, None]
        covariance = (centered * weights**2) @ centered.T
        np.testing.assert_allclose(result["bin_probability"], mass, atol=1e-11)
        np.testing.assert_allclose(
            result["bin_probability_covariance"], covariance, atol=1e-11
        )
        np.testing.assert_allclose(
            result["target_effective_samples"], 1 / np.sum(weights**2), rtol=1e-10
        )

    def test_empty_weight_diagnostics_and_invalid_corrections(self):
        import numpy as np

        self.assertEqual(
            runner._weight_diagnostics(np.array([])),
            {
                "samples": 0,
                "effective_samples": 0.0,
                "maximum_weight": None,
                "normalized_entropy": None,
            },
        )
        self.assertEqual(
            runner._weight_diagnostics(np.array([0.4]))["normalized_entropy"], 0
        )
        data = fixture(100)
        samples, _ = runner._sample(
            data["replicas"][0]["windows"], data["windows"], (0, 1)
        )
        for delta in ([0], [float("inf")] * sum(map(len, samples))):
            with self.assertRaisesRegex(ValueError, "finite paired"):
                runner._estimate(samples, data["windows"], 300, parameters(), delta)

    def test_constant_energy_offset_leaves_density_and_weights_unchanged(self):
        import numpy as np

        data = paired_fixture(1000)
        result = self.analyze(data)
        for replica in data["replicas"]:
            for trace in replica["windows"]:
                trace["energy_pairs"]["target_kj_mol"] = [
                    x + 1e6 for x in trace["energy_pairs"]["target_kj_mol"]
                ]
        shifted = self.analyze(data)
        np.testing.assert_allclose(
            shifted["summary"]["pmf_mean_kj_mol"],
            result["summary"]["pmf_mean_kj_mol"],
            atol=1e-8,
        )
        for a, b in zip(result["replicas"], shifted["replicas"], strict=True):
            self.assertAlmostEqual(
                a["target_effective_samples"], b["target_effective_samples"], places=6
            )
            self.assertAlmostEqual(
                b["reweighting"]["subtracted_delta_energy_kj_mol"]
                - a["reweighting"]["subtracted_delta_energy_kj_mol"],
                1e6,
            )

    def test_concentrated_weights_remain_visible(self):
        import numpy as np

        data = fixture(1000)
        samples, _ = runner._sample(
            data["replicas"][0]["windows"], data["windows"], (0, 1)
        )
        delta = np.zeros(sum(map(len, samples)))
        delta[0] = -50
        result = runner._estimate(samples, data["windows"], 300, parameters(), delta)
        self.assertGreater(result["reweighting"]["maximum_weight"], 0.99)
        self.assertLess(result["target_effective_samples"], 1.01)
        self.assertLess(result["reweighting"]["normalized_entropy"], 0.01)

    def test_rare_target_bins_preserve_finite_uncertainty_and_effective_counts(self):
        import numpy as np

        rng = np.random.default_rng(91384)
        samples = [rng.normal(size=500) for _ in range(2)]
        q = np.concatenate(samples)
        windows = [
            {"id": str(i), "center_angstrom": c, "k_kj_mol_angstrom2": 1e-12}
            for i, c in enumerate((-1, 1))
        ]
        result = runner._estimate(
            samples, windows, 300, parameters(), np.where(q > 0, 600.0, 0.0)
        )
        json.dumps(result, allow_nan=False)
        bins = np.searchsorted(parameters()["bin_edges_angstrom"], q, side="right") - 1
        np.testing.assert_allclose(
            result["bin_effective_samples"],
            [np.count_nonzero(bins == i) for i in range(8)],
            rtol=1e-10,
        )
        self.assertTrue(np.isfinite(result["pmf_asymptotic_error_kj_mol"]).all())

    def test_target_covariance_underflow_has_a_numerical_diagnostic(self):
        import numpy as np

        rng = np.random.default_rng(91384)
        samples = [rng.normal(size=500) for _ in range(2)]
        q = np.concatenate(samples)
        windows = [
            {"id": str(i), "center_angstrom": c, "k_kj_mol_angstrom2": 1e-12}
            for i, c in enumerate((-1, 1))
        ]
        # Native observable covariance is scaled by products of probabilities.
        # These target states reach subnormal and zero products respectively.
        for penalty in (900.0, 1200.0):
            with (
                self.subTest(penalty=penalty),
                self.assertRaisesRegex(ValueError, "normal-range covariance"),
            ):
                runner._estimate(
                    samples, windows, 300, parameters(), np.where(q > 0, penalty, 0.0)
                )

    def test_energy_decorrelation_and_exact_selected_frame_pairing(self):
        data = paired_fixture(100)
        traces = data["replicas"][0]["windows"]
        with patch(
            "pymbar.timeseries.statistical_inefficiency",
            side_effect=[1.0, 1.0, 5.0, 7.0] * 5,
        ):
            samples, diagnostics = runner._sample(traces, data["windows"], (0, 1))
        self.assertTrue(all(d["stride_frames"] == 7 for d in diagnostics))
        difference = runner._paired_differences(traces, diagnostics)
        expected = []
        for trace, diagnostic in zip(traces, diagnostics, strict=True):
            for i in diagnostic["selected_source_frame_indices"]:
                expected.append(
                    trace["energy_pairs"]["target_kj_mol"][i]
                    - trace["energy_pairs"]["reference_kj_mol"][i]
                )
        self.assertEqual(difference, expected)
        self.assertEqual(len(difference), sum(map(len, samples)))


def double_well_fixture(samples=2000):
    """Inverse-CDF independent samples of a defined quartic test Hamiltonian."""
    import numpy as np

    data = fixture(100)
    data["title"] = (
        "Double-well numerical reference: replicated umbrella reconstruction"
    )
    data["system_sha256"] = hashlib.sha256(
        b"One-dimensional coordinate q in angstrom; quartic numerical reference"
    ).hexdigest()
    data["hamiltonian_sha256"] = hashlib.sha256(
        b"U(q)=8*(q^2-1)^2 kJ/mol, q numerical value in angstrom; T=300 K"
    ).hexdigest()
    data["windows"] = [
        {"id": f"w{i}", "center_angstrom": float(center), "k_kj_mol_angstrom2": 10.0}
        for i, center in enumerate(np.linspace(-1.5, 1.5, 7))
    ]
    grid = np.linspace(-2.2, 2.2, 22001)
    potential = 8 * (grid**2 - 1) ** 2
    rt = runner.GAS_CONSTANT_KJ_MOL_K * 300
    rng = np.random.default_rng(71623)
    data["replicas"] = []
    for replica in range(2):
        traces = []
        for window in data["windows"]:
            density = np.exp(
                -(potential + 10 * (grid - window["center_angstrom"]) ** 2) / rt
            )
            cdf = np.concatenate(
                [[0], np.cumsum((density[:-1] + density[1:]) / 2 * (grid[1] - grid[0]))]
            )
            cdf /= cdf[-1]
            q = np.interp(rng.uniform(size=samples), cdf, grid).tolist()
            traces.append(
                {
                    "window": window["id"],
                    "trajectory_sha256": hashlib.sha256(
                        json.dumps(q, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "discard_frames": 0,
                    "frame_interval_fs": 1.0,
                    "coordinate_angstrom": q,
                }
            )
        data["replicas"].append({"id": f"replica-{replica + 1}", "windows": traces})
    p = {
        "bin_edges_angstrom": np.linspace(-1.75, 1.75, 15).tolist(),
        "reference_bin": 3,
        "reactant_bins": [2, 3, 4],
        "product_bins": [9, 10, 11],
        "transition_bins": [6, 7],
        "block_counts": [2, 4],
    }
    return data, p


@unittest.skipUnless(importlib.util.find_spec("numpy"), "quantitative extra required")
class UmbrellaInputTests(unittest.TestCase):
    def validate(self, data=None, p=None):
        return runner.validate_umbrella(
            data if data is not None else fixture(100), p or parameters()
        )

    def test_valid_source_and_parameters(self):
        self.validate()

    def test_identity_duplicate_and_unmatched_windows(self):
        for mutate in (
            lambda d: d.update(hamiltonian_sha256="wrong"),
            lambda d: d["replicas"][1].update(id="r0"),
            lambda d: d["replicas"][1]["windows"][0].update(
                trajectory_sha256=d["replicas"][0]["windows"][0]["trajectory_sha256"]
            ),
            lambda d: d["replicas"][1]["windows"][0].update(window="missing"),
            lambda d: d.update(bias_convention="0.5*K*(q-center)^2"),
            lambda d: d["coordinate"].update(positive_distance_atoms=[1, 1]),
        ):
            data = fixture(100)
            mutate(data)
            with self.subTest(data=str(data)[:200]), self.assertRaises(ValueError):
                self.validate(data)

    def test_invalid_frames_discard_and_bounds(self):
        for key, value in [
            ("discard_frames", True),
            ("discard_frames", 99),
            ("frame_interval_fs", 0),
        ]:
            data = fixture(100)
            data["replicas"][0]["windows"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.validate(data)
        for value in [float("nan"), float("inf"), True, "0.5"]:
            data = fixture(100)
            data["replicas"][0]["windows"][0]["coordinate_angstrom"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(data)

    def test_bin_and_basin_definitions(self):
        for key, value in [
            ("reference_bin", 0),
            ("product_bins", [3, 5]),
            ("transition_bins", []),
            ("block_counts", [2, 2]),
            ("block_counts", [10]),
            ("bin_edges_angstrom", [0, 1, 1, 2]),
            ("automatic_rate", True),
        ]:
            p = parameters()
            p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(p=p)


@unittest.skipUnless(
    importlib.util.find_spec("pymbar"), "PyMBAR 4.2.0 environment required"
)
class UmbrellaNumericalTests(unittest.TestCase):
    def analyze(self, data=None, p=None):
        return runner.analyze_umbrella(
            data if data is not None else fixture(), p or parameters()
        )

    def test_gaussian_density_and_basin_balance(self):
        import numpy as np
        from scipy.special import ndtr

        result = self.analyze()
        edges = np.array(parameters()["bin_edges_angstrom"])
        rt = 0.00831446261815324 * 300
        p = np.diff(ndtr(edges / math.sqrt(rt / 2)))
        expected = -rt * np.log((p / np.diff(edges)) / (p[3] / np.diff(edges)[3]))
        observed = np.array(result["summary"]["pmf_mean_kj_mol"])
        np.testing.assert_allclose(observed, expected, atol=0.35, rtol=0)
        self.assertAlmostEqual(
            result["summary"]["basin_delta_g_mean_kj_mol"],
            -rt * math.log(p[5:7].sum() / p[2:4].sum()),
            delta=0.3,
        )
        for replica in result["replicas"]:
            self.assertAlmostEqual(replica["pmf_kj_mol"][3], 0)
            self.assertAlmostEqual(replica["pmf_asymptotic_error_kj_mol"][3], 0)
            self.assertGreater(replica["outside_probability"], 0)
            self.assertEqual(len(replica["time_blocks"][0]["blocks"]), 2)
            np.testing.assert_allclose(
                np.sum(replica["overlap_matrix"], axis=1), 1, atol=1e-7
            )
            self.assertGreater(min(replica["bin_effective_samples"]), 10)
        self.assertNotIn("rate", result)

    def test_unequal_bin_width_density(self):
        import numpy as np
        from scipy.special import ndtr

        p = parameters()
        p["bin_edges_angstrom"] = [-2, -1.5, -1, -0.5, 0, 0.25, 0.75, 1.5, 2]
        result = self.analyze(p=p)
        edges = np.array(p["bin_edges_angstrom"])
        rt = 0.00831446261815324 * 300
        mass = np.diff(ndtr(edges / math.sqrt(rt / 2)))
        expected = -rt * np.log(mass / np.diff(edges) / (mass[3] / np.diff(edges)[3]))
        np.testing.assert_allclose(
            result["summary"]["pmf_mean_kj_mol"], expected, atol=0.35, rtol=0
        )

    def test_empty_bin_stays_missing(self):
        p = parameters()
        p["bin_edges_angstrom"] = [-20, -10, -1, -0.5, 0, 0.5, 1, 10, 20]
        result = self.analyze(p=p)
        self.assertIsNone(result["summary"]["pmf_mean_kj_mol"][0])
        self.assertIsNone(result["summary"]["pmf_mean_kj_mol"][-1])
        self.assertEqual(result["replicas"][0]["bin_effective_samples"][0], 0)

    def test_constant_trace_and_empty_reference_are_errors(self):
        data = fixture(100)
        data["replicas"][0]["windows"][0]["coordinate_angstrom"] = [0.0] * 100
        with self.assertRaisesRegex(ValueError, "varies"):
            self.analyze(data)
        p = parameters()
        p["bin_edges_angstrom"] = [-20, -10, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        p["reference_bin"] = 0
        p["reactant_bins"] = [0]
        with self.assertRaisesRegex(ValueError, "reference"):
            self.analyze(p=p)

    def test_discarded_prefix_does_not_enter_estimate(self):
        data = fixture(1000)
        baseline = self.analyze(data)
        changed = copy.deepcopy(data)
        for replica in changed["replicas"]:
            for trajectory in replica["windows"]:
                trajectory["coordinate_angstrom"] = [1000.0] * 20 + trajectory[
                    "coordinate_angstrom"
                ]
                trajectory["discard_frames"] = 20
        retained = self.analyze(changed)
        self.assertEqual(baseline["summary"], retained["summary"])

    def test_unbiased_limit_has_multinomial_covariance(self):
        import numpy as np

        rng = np.random.default_rng(47218)
        samples = [rng.normal(size=1000) for _ in range(2)]
        windows = [
            {"id": "a", "center_angstrom": -1, "k_kj_mol_angstrom2": 1e-12},
            {"id": "b", "center_angstrom": 1, "k_kj_mol_angstrom2": 1e-12},
        ]
        p = parameters()
        p["bin_edges_angstrom"] = [-20, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 20]
        result = runner._estimate(samples, windows, 300, p)
        mass = np.array(result["bin_probability"])
        expected = (np.diag(mass) - np.outer(mass, mass)) / 2000
        np.testing.assert_allclose(
            result["bin_probability_covariance"], expected, atol=1e-12, rtol=1e-6
        )
        np.testing.assert_allclose(
            np.sum(result["bin_probability_covariance"], axis=1), 0, atol=1e-12
        )

    def test_disconnected_windows_require_connected_sampling(self):
        import numpy as np

        samples = [np.linspace(-20.01, -19.99, 100), np.linspace(19.99, 20.01, 100)]
        windows = [
            {"id": str(i), "center_angstrom": center, "k_kj_mol_angstrom2": 500}
            for i, center in enumerate([-20, 20])
        ]
        with self.assertRaisesRegex(ValueError, "disconnected"):
            runner._estimate(samples, windows, 300, parameters())

    def test_failed_time_segment_stays_visible(self):
        data = fixture(1000)
        for replica in data["replicas"]:
            for trace in replica["windows"]:
                trace["coordinate_angstrom"][:500] = [
                    trace["coordinate_angstrom"][0]
                ] * 500
        # The full trace remains variable; the constant first segment is retained.
        result = self.analyze(data)
        for replica in result["replicas"]:
            first = replica["time_blocks"][0]["blocks"][0]
            self.assertIsNone(first["estimate"])
            self.assertIn("varies", first["error"])

    def test_native_segment_failure_preserves_other_estimates(self):
        from pymbar.utils import ConvergenceError

        estimate = runner._estimate
        calls = 0

        def fail_one_segment(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ConvergenceError("Native segment solver did not converge")
            return estimate(*args)

        with patch.object(runner, "_estimate", side_effect=fail_one_segment):
            result = self.analyze(fixture(1000))
        blocks = result["replicas"][0]["time_blocks"][0]["blocks"]
        self.assertIsNone(blocks[0]["estimate"])
        self.assertIn("did not converge", blocks[0]["error"])
        self.assertIsNotNone(blocks[1]["estimate"])
        self.assertIsNotNone(result["summary"]["basin_delta_g_mean_kj_mol"])
        with patch.object(runner, "_estimate", side_effect=ConvergenceError("full")):
            with self.assertRaisesRegex(ConvergenceError, "full"):
                self.analyze(fixture(1000))

    def test_double_well_peak_matches_independent_quadrature(self):
        import numpy as np
        from scipy.integrate import quad

        data, p = double_well_fixture()
        result = self.analyze(data, p)
        rt = runner.GAS_CONSTANT_KJ_MOL_K * 300
        edges = p["bin_edges_angstrom"]
        mass = np.array(
            [
                quad(
                    lambda q: math.exp(-8 * (q * q - 1) ** 2 / rt), a, b, epsabs=1e-12
                )[0]
                for a, b in zip(edges[:-1], edges[1:], strict=True)
            ]
        )
        density = mass / np.diff(edges)
        expected = -rt * np.log(density / density[p["reference_bin"]])
        observed = result["summary"]["pmf_mean_kj_mol"]
        populated = [i for i, value in enumerate(observed) if value is not None]
        self.assertTrue(
            set(p["reactant_bins"] + p["product_bins"] + p["transition_bins"])
            <= set(populated)
        )
        np.testing.assert_allclose(
            [observed[i] for i in populated], expected[populated], atol=0.8, rtol=0
        )
        self.assertAlmostEqual(
            result["summary"]["basin_delta_g_mean_kj_mol"], 0, delta=0.4
        )
        self.assertGreater(
            min(r["peak_minus_reference_kj_mol"] for r in result["replicas"]), 6
        )


if __name__ == "__main__":
    unittest.main()
