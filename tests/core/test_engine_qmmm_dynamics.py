"""Explicit molecular sampling, phase boundaries and retained native failures."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "qmmm_dynamics_runner",
    Path(__file__).resolve().parents[2]
    / "src/alambiq/engines/qmmm_dynamics_runner.py",
)
dynamics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamics)


def sampling():
    return {
        "temperature_kelvin": 300.0,
        "friction_per_ps": 5.0,
        "timestep_fs": 0.0625,
        "seed": 741,
        "equilibration_steps": 2,
        "production_steps": 4,
        "report_interval": 2,
        "umbrella": {
            "pairs": [[1, 2], [1, 3]],
            "center_angstrom": 0.0,
            "coefficient_kj_mol_angstrom2": 10.0,
        },
    }


class SamplingRequestTests(unittest.TestCase):
    def test_explicit_settings_and_finite_bounds(self):
        supplied = sampling()
        before = copy.deepcopy(supplied)
        self.assertEqual(dynamics.sampling_settings(supplied), supplied)
        self.assertEqual(supplied, before)
        for change in (
            {"timestep_fs": 0},
            {"temperature_kelvin": float("nan")},
            {"friction_per_ps": True},
            {"seed": 0},
            {"production_steps": False},
            {"report_interval": 3},
            {"equilibration_steps": -1},
            {"ensemble": "NPT"},
            {"umbrella": {"pairs": [[1, 2], [2, 1]]}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dynamics.sampling_settings({**supplied, **change})
        with self.assertRaises(ValueError):
            dynamics.sampling_settings({})


@unittest.skipUnless(
    importlib.util.find_spec("openmm") and importlib.util.find_spec("pyscf"),
    "OpenMM 8.6 and PySCF 2.14.0 required",
)
class NativeSamplingTests(unittest.TestCase):
    @staticmethod
    def request(root):
        from test_engine_qmmm_state import PreparedStateTests

        root.mkdir()
        return {"prepared": PreparedStateTests.source(root), "sampling": sampling()}

    @staticmethod
    def electronic(elements, q, m, c, *, directory, initial_density=None, **kwargs):
        import numpy as np

        directory.mkdir(parents=True)
        density = np.eye(2)
        np.savez(directory / "electronic-state.npz", density=density)
        return {
            "observation": {"energy_hartree": -1.0},
            "density": density,
            "quantum_force_hartree_bohr": np.zeros_like(q),
            "embedding_force_hartree_bohr": np.zeros_like(m),
        }

    def test_native_phase_artifacts_and_seeded_repeat(self):
        import csv

        import numpy as np
        import openmm
        from openmm import unit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self.request(root / "input")
            original = {p.name: p.read_bytes() for p in (root / "input").iterdir()}
            results = []
            for label in ("first", "second"):
                out = root / label
                with patch.object(
                    dynamics.coupling.runner._component("qmmm_pyscf"),
                    "evaluate_molecular",
                    side_effect=self.electronic,
                ):
                    result = dynamics.sample(request, out)
                results.append(result)
                self.assertEqual(result["production_frames"], 2)
                self.assertEqual(result["equilibration_steps_completed"], 2)
                self.assertEqual(result["production_steps_completed"], 4)
                self.assertEqual(result["electronic_calls"], 7)
                self.assertEqual(result["sampling"], request["sampling"])
                with (out / "state.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(
                    [r["phase"] for r in rows],
                    ["initial", "equilibration", "production", "production"],
                )
                self.assertEqual([int(r["step"]) for r in rows], [0, 2, 4, 6])
                self.assertTrue((out / "trajectory.dcd").is_file())
                state = openmm.XmlSerializer.deserialize(
                    (out / "final-state.xml").read_text()
                )
                self.assertAlmostEqual(
                    state.getTime().value_in_unit(unit.femtosecond), 0.375
                )
                with np.load(out / "final-density.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["density"], np.eye(2))
                self.assertEqual(
                    result["outputs_sha256"]["trajectory.dcd"],
                    dynamics.runner._sha(out / "trajectory.dcd"),
                )
                self.assertEqual(
                    result["hamiltonian_sha256"], results[0]["hamiltonian_sha256"]
                )
            self.assertEqual(
                (root / "first/state.csv").read_bytes(),
                (root / "second/state.csv").read_bytes(),
            )
            self.assertEqual(
                original, {p.name: p.read_bytes() for p in (root / "input").iterdir()}
            )
            with self.assertRaises(FileExistsError):
                dynamics.sample(request, root / "first")

    def test_failure_preserves_partial_native_state_and_sources(self):
        import openmm

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self.request(root / "input")
            with (
                patch.object(
                    dynamics.coupling.runner._component("qmmm_pyscf"),
                    "evaluate_molecular",
                    side_effect=ValueError("native SCF failure"),
                ),
                self.assertRaisesRegex(openmm.OpenMMException, "native SCF failure"),
            ):
                dynamics.sample(request, root / "failed")
            self.assertTrue((root / "failed/request.json").is_file())
            self.assertTrue((root / "failed/calls/000000/failure.json").is_file())
            self.assertEqual(
                json.loads((root / "failed/failure.json").read_text())["message"],
                "native SCF failure",
            )
            self.assertFalse((root / "failed/summary.json").exists())

    def test_source_mutation_retains_trace_without_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self.request(root / "input")
            source = Path(request["prepared"]["topology_pdb"])

            def changed(*args, **kwargs):
                result = self.electronic(*args, **kwargs)
                source.write_text(source.read_text() + "REMARK source changed\n")
                return result

            with (
                patch.object(
                    dynamics.coupling.runner._component("qmmm_pyscf"),
                    "evaluate_molecular",
                    side_effect=changed,
                ),
                self.assertRaisesRegex(ValueError, "source must be unchanged throughout sampling"),
            ):
                dynamics.sample(request, root / "changed")
            self.assertTrue((root / "changed/state.csv").is_file())
            self.assertTrue((root / "changed/failure.json").is_file())
            self.assertFalse((root / "changed/summary.json").exists())

    def test_real_quantum_thermal_sampling(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self.request(root / "input")
            request["sampling"].update(
                equilibration_steps=0,
                production_steps=2,
                report_interval=1,
                umbrella=None,
            )
            result = dynamics.sample(request, root / "real")
            self.assertEqual(result["production_frames"], 2)
            self.assertEqual(result["electronic_calls"], 3)
            self.assertEqual(result["platform"], "Reference")
            self.assertTrue(np.isfinite(result["final_total_energy_kj_mol"]))
            self.assertTrue(
                (root / "real/calls/000002/electronic/result.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
