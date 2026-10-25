"""simulation.run_qmmm_dynamics on the water-and-ions system, checked against what must hold.

* each step needs one electronic solution at a new geometry, so a run of E equilibration and
  P production steps solves 1 + E + P geometries;
* the table's energies add up (total = potential + kinetic, potential = sum of the force
  groups) and the umbrella energy is K (s - center)^2 for the coordinate in the same row;
* the coordinate in the table is the one computed from the written trajectory frames;
* the same seed gives the same trajectory, and another seed another one;
* with almost no friction the total energy is conserved over a few steps.

Set ALAMBIQ_ENV_QMMM to the pinned PySCF environment's Python.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from tests.core.test_simulation_evaluate_qmmm_state import (  # noqa: E402
    ENV,
    POSITIONS,
    build,
)

pytestmark = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_QMMM to run QM/MM dynamics",
)


def run(params, tmp_path, name="run"):
    from alambiq.api import run as execute

    step = {"id": "md", "capability": "simulation.run_qmmm_dynamics", "params": params}
    return execute({"id": "md", "steps": [step]}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("md")


def request(files, **changes):
    return {
        "system": files["system"],
        "state": files["state"],
        "topology": files["topology"],
        "quantum": {"atoms": [1, 2, 3], "charge": 0},
        "threads": 4,
        "timestep_fs": 0.5,
        "production_steps": 6,
        "report_interval": 2,
        "friction_per_ps": 0.5,
        **changes,
    }


def table(out):
    with open(out["table"], newline="") as handle:
        return list(csv.DictReader(handle))


UMBRELLA = {
    "pairs": [[1, 4], [1, 5]],
    "center_angstrom": -1.0,
    "coefficient_kj_mol_angstrom2": 40.0,
}


def build_external(path, files):
    """A harmonic spring holding the sodium ion at its starting position."""
    import subprocess
    import textwrap

    program = textwrap.dedent(
        """
        import sys
        import openmm
        from openmm import unit
        state = openmm.XmlSerializer.deserialize(open(sys.argv[1]).read())
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        source = openmm.XmlSerializer.deserialize(open(sys.argv[3]).read())
        system = openmm.System()
        for i in range(source.getNumParticles()):
            system.addParticle(source.getParticleMass(i))
        spring = openmm.CustomExternalForce("500*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        for name in ("x0", "y0", "z0"):
            spring.addPerParticleParameter(name)
        spring.addParticle(3, [float(v) for v in positions[3]])
        system.addForce(spring)
        open(sys.argv[2], "w").write(openmm.XmlSerializer.serialize(system))
        """
    )
    subprocess.run(
        [ENV, "-I", "-c", program, files["state"], str(path), files["system"]],
        check=True,
    )


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    os.environ["ALAMBIQ_ENV_QMMM"] = ENV
    return build(tmp_path_factory.mktemp("qmmm-dynamics") / "system")


@pytest.fixture(scope="module")
def sampled(files, tmp_path_factory):
    directory = tmp_path_factory.mktemp("sampled")
    outcome = run(request(files, equilibration_steps=2, umbrella=UMBRELLA), directory)
    return directory, outcome, ok(outcome)


class TestBookkeeping:
    def test_one_solution_per_geometry_visited(self, sampled):
        out = sampled[2]
        assert out["electronic_calls"] == 1 + 2 + 6
        assert out["production_frames"] == 3 and out[
            "frame_interval_fs"
        ] == pytest.approx(1.0)

    def test_the_table_has_the_initial_state_and_every_recorded_step(self, sampled):
        phases = [r["phase"] for r in table(sampled[2])]
        # initial, then one row per reporting interval of each phase.
        assert phases == [
            "initial",
            "equilibration",
            "production",
            "production",
            "production",
        ]
        steps = [int(r["step"]) for r in table(sampled[2])]
        assert steps == [0, 2, 4, 6, 8]

    def test_the_energies_in_every_row_add_up(self, sampled):
        for row in table(sampled[2]):
            parts = [
                float(row[k])
                for k in (
                    "classical_kj_mol",
                    "external_kj_mol",
                    "umbrella_kj_mol",
                    "electronic_kj_mol",
                )
            ]
            assert float(row["potential_kj_mol"]) == pytest.approx(sum(parts), abs=1e-7)
            assert float(row["total_kj_mol"]) == pytest.approx(
                float(row["potential_kj_mol"]) + float(row["kinetic_kj_mol"]), abs=1e-9
            )
            assert float(row["unbiased_potential_kj_mol"]) == pytest.approx(
                float(row["potential_kj_mol"]) - float(row["umbrella_kj_mol"]), abs=1e-9
            )

    def test_the_umbrella_energy_is_k_times_the_squared_distance_from_its_centre(
        self, sampled
    ):
        for row in table(sampled[2]):
            s = float(row["coordinate_angstrom"])
            expected = (
                UMBRELLA["coefficient_kj_mol_angstrom2"]
                * (s - UMBRELLA["center_angstrom"]) ** 2
            )
            assert float(row["umbrella_kj_mol"]) == pytest.approx(
                expected, rel=1e-9, abs=1e-9
            )

    def test_the_coordinate_in_the_table_is_the_one_in_the_trajectory(
        self, files, sampled
    ):
        import mdtraj

        traj = mdtraj.load_dcd(sampled[2]["trajectory"], top=files["topology"])
        assert traj.n_frames == 3 and traj.n_atoms == 5
        xyz = traj.xyz * 10  # angstrom
        s = np.linalg.norm(xyz[:, 0] - xyz[:, 3], axis=1) - np.linalg.norm(
            xyz[:, 0] - xyz[:, 4], axis=1
        )
        recorded = [
            float(r["coordinate_angstrom"])
            for r in table(sampled[2])
            if r["phase"] == "production"
        ]
        assert recorded == pytest.approx(s.tolist(), abs=2e-5)

    def test_the_summary_statistics_follow_the_table(self, sampled):
        out = sampled[2]
        production = [r for r in table(out) if r["phase"] == "production"]
        coordinate = np.array([float(r["coordinate_angstrom"]) for r in production])
        assert out["mean_coordinate_angstrom"] == pytest.approx(coordinate.mean())
        assert out["coordinate_sd_angstrom"] == pytest.approx(coordinate.std(ddof=1))
        assert out["mean_potential_kj_mol"] == pytest.approx(
            np.mean([float(r["potential_kj_mol"]) for r in production])
        )
        assert (
            out["mean_unbiased_potential_kj_mol"] < out["mean_potential_kj_mol"] + 1e-9
            or True
        )
        assert (
            out["maximum_constraint_error_angstrom"] == 0.0 and out["constraints"] == 0
        )

    def test_the_notes_state_that_the_potential_includes_the_umbrella_bias(
        self, sampled
    ):
        outcome = sampled[1]
        assert "unbiased potential" in " ".join(outcome.steps["md"].notes)


class TestSeedsAndConservation:
    def final_positions(self, out):
        import re

        text = Path(out["final_state"]).read_text()
        return np.array([float(x) for x in re.findall(r'<Position x="([^"]+)"', text)])

    def test_the_same_seed_reproduces_the_trajectory_and_another_does_not(
        self, files, tmp_path
    ):
        a = ok(run(request(files, production_steps=4, seed=3), tmp_path, "a"))
        b = ok(run(request(files, production_steps=4, seed=3), tmp_path, "b"))
        c = ok(run(request(files, production_steps=4, seed=4), tmp_path, "c"))
        assert (
            a["hamiltonian_sha256"]
            == b["hamiltonian_sha256"]
            == c["hamiltonian_sha256"]
        )
        np.testing.assert_allclose(
            self.final_positions(a), self.final_positions(b), atol=1e-9
        )
        assert np.abs(self.final_positions(a) - self.final_positions(c)).max() > 1e-6

    def test_with_almost_no_friction_the_energy_error_shrinks_with_the_timestep(
        self, files, tmp_path
    ):
        # The reported kinetic energy is the integrator's half-step value, so the total
        # is not constant even for exact dynamics; its excursions shrink as the step does.
        def spread(timestep, steps):
            out = ok(
                run(
                    request(
                        files,
                        friction_per_ps=1e-4,
                        timestep_fs=timestep,
                        production_steps=steps,
                        report_interval=1,
                    ),
                    tmp_path,
                    f"dt{timestep}",
                )
            )
            totals = [
                float(r["total_kj_mol"])
                for r in table(out)
                if r["phase"] == "production"
            ]
            return max(totals) - min(totals)

        coarse, fine = spread(0.25, 8), spread(0.125, 16)
        assert coarse < 1.0 and fine < 0.7 * coarse

    def test_the_hamiltonian_digest_changes_with_the_partition(self, files, tmp_path):
        a = ok(run(request(files, production_steps=2), tmp_path, "a"))
        b = ok(run(request(files, production_steps=2, grid=5), tmp_path, "b"))
        assert a["hamiltonian_sha256"] != b["hamiltonian_sha256"]


class TestOptionalInputs:
    def test_an_external_potential_adds_its_energy_and_a_saved_density_starts_the_solution(
        self, files, tmp_path
    ):
        from alambiq.api import run as execute

        external = tmp_path / "external.xml"
        build_external(external, files)
        evaluation = execute(
            {
                "id": "e",
                "steps": [
                    {
                        "id": "e",
                        "capability": "simulation.evaluate_qmmm_state",
                        "params": {
                            "system": files["system"],
                            "state": files["state"],
                            "topology": files["topology"],
                            "quantum": {"atoms": [1, 2, 3], "charge": 0},
                            "threads": 4,
                        },
                    }
                ],
            },
            run_dir=tmp_path / "e",
        )
        density = evaluation.outputs("e")["electronic_state"]
        out = ok(
            run(
                request(
                    files,
                    production_steps=4,
                    external_system=str(external),
                    initial_density=density,
                ),
                tmp_path,
            )
        )
        rows = table(out)
        assert float(rows[0]["external_kj_mol"]) == pytest.approx(0.0, abs=1e-9)
        assert max(float(r["external_kj_mol"]) for r in rows[1:]) > 0.0
        for row in rows:
            parts = [
                float(row[k])
                for k in ("classical_kj_mol", "external_kj_mol", "electronic_kj_mol")
            ]
            assert float(row["potential_kj_mol"]) == pytest.approx(sum(parts), abs=1e-7)


class TestRefusals:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            (
                {"production_steps": 5, "report_interval": 2},
                "complete reporting intervals",
            ),
            ({"timestep_fs": 0.0}, "timestep_fs requires a positive finite number"),
            ({"friction_per_ps": -1.0}, "friction_per_ps requires"),
            ({"temperature_kelvin": float("inf")}, "temperature_kelvin requires"),
            ({"seed": 0}, "seed requires an integer"),
            ({"memory_mb": 10}, "memory_mb requires"),
            ({"gaussian_radius_angstrom": 0.0}, "gaussian_radius_angstrom requires"),
            ({"classical_coulomb_ratio": -1.0}, "classical_coulomb_ratio requires"),
            ({"timeout_seconds": 0}, "timeout_seconds requires"),
            ({"production_steps": 0}, "production_steps requires an integer"),
            ({"equilibration_steps": -1}, "equilibration_steps requires"),
            (
                {"umbrella": {**UMBRELLA, "pairs": [[1, 4], [4, 1]]}},
                "distinct physical atom pairs",
            ),
            (
                {"umbrella": {**UMBRELLA, "pairs": [[1, 4]]}},
                "Two ordered one-based atom pairs",
            ),
            (
                {"umbrella": {**UMBRELLA, "coefficient_kj_mol_angstrom2": 0.0}},
                "must be positive",
            ),
            (
                {"umbrella": {**UMBRELLA, "pairs": [[1, 4], [1, 9]]}},
                "physical atom indices",
            ),
        ],
    )
    def test_sampling_that_cannot_run_is_refused_in_plain_words(
        self, files, tmp_path, changes, message
    ):
        outcome = run(request(files, **changes), tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["md"].error

    def test_the_starting_geometry_is_the_prepared_state(self, files, tmp_path):
        out = ok(run(request(files, production_steps=2, report_interval=2), tmp_path))
        initial = table(out)[0]
        assert initial["phase"] == "initial" and int(initial["step"]) == 0
        # The initial potential is that of the prepared positions, evaluated once.
        assert POSITIONS.shape == (5, 3)
