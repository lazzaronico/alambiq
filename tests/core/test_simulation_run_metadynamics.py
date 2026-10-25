"""simulation.run_metadynamics against a torsion potential whose answer is exact.

A butane-like chain of four carbons has stiff bonds and angles and one torsion
potential V(θ) = k(1 + cos 3θ), with k = 7.5 kJ/mol: three equal wells at ±60° and
180° and barriers of 15 kJ/mol (six kT at 300 K) at 0° and ±120°. Nothing else in
the system depends on θ, so its unbiased distribution is exactly exp(-V/kT), and
the free-energy surface that reweighted metadynamics returns must be V(θ) minus
its minimum. A five-carbon chain with two such torsions gives the two-dimensional
case, whose surface is the sum V(φ) + V(ψ).

The runs are about a minute each: they are real simulations of a few million
steps. Set ALAMBIQ_ENV_OPENMM to the pinned openmm-ensemble environment's Python.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

ENV = os.environ.get("ALAMBIQ_ENV_OPENMM", "")
K_ONE = 7.5

pytestmark = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_OPENMM to run the OpenMM tests",
)

CHAIN = textwrap.dedent(
    """
    import math, sys
    from pathlib import Path
    import openmm
    from openmm import app, unit
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    count, k = int(sys.argv[2]), float(sys.argv[3])
    system = openmm.System()
    for _ in range(count):
        system.addParticle(12.0 * unit.amu)
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    torsion = openmm.CustomTorsionForce("k*(1+cos(3*theta))")
    torsion.addGlobalParameter("k", k)
    for i in range(count - 1):
        bonds.addBond(i, i + 1, 0.153, 250000.0)
    for i in range(count - 2):
        angles.addAngle(i, i + 1, i + 2, math.radians(111.0), 500.0)
    for i in range(count - 3):
        torsion.addTorsion(i, i + 1, i + 2, i + 3, [])
    for force in (bonds, angles, torsion):
        system.addForce(force)
    topology = app.Topology()
    residue = topology.addResidue("CHN", topology.addChain("A"))
    atoms = [topology.addAtom(f"C{i + 1}", app.element.carbon, residue) for i in range(count)]
    for a, b in zip(atoms, atoms[1:]):
        topology.addBond(a, b)
    r, turn = 0.153, math.radians(180 - 111)
    x, y, heading, positions = 0.0, 0.0, 0.0, [(0.0, 0.0, 0.0)]
    for i in range(count - 1):
        x, y = x + r * math.cos(heading), y + r * math.sin(heading)
        positions.append((x, y, 0.0))
        heading += turn if i % 2 == 0 else -turn
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions * unit.nanometer)
    openmm.LocalEnergyMinimizer.minimize(context)
    state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    (out / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    (out / "state.xml").write_text(openmm.XmlSerializer.serialize(state))
    with (out / "topology.pdb").open("w") as stream:
        app.PDBFile.writeFile(topology, state.getPositions(), stream)
    """
)


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    if ENV:
        monkeypatch.setenv("ALAMBIQ_ENV_OPENMM", ENV)


def build(directory: Path, name: str, atoms: int, k: float) -> Path:
    script = directory / "chain.py"
    script.write_text(CHAIN)
    target = directory / name
    subprocess.run(
        [ENV, str(script), str(target), str(atoms), str(k)],
        check=True,
        capture_output=True,
    )
    return target


@pytest.fixture(scope="module")
def butane(tmp_path_factory):
    return build(tmp_path_factory.mktemp("butane"), "butane", 4, K_ONE)


def torsion(name: str, first: int, width: float = 0.35, grid: int = 128) -> dict:
    return {
        "name": name,
        "atoms": [
            {"chain_id": "A", "residue": "1", "atom_name": f"C{i}"}
            for i in range(first, first + 4)
        ],
        "width_radians": width,
        "grid_width": grid,
    }


def step(files: Path, torsions: list[dict], **params) -> dict:
    return {
        "id": "meta",
        "capability": "simulation.run_metadynamics",
        "params": {
            "system": str(files / "system.xml"),
            "state": str(files / "state.xml"),
            "topology": str(files / "topology.pdb"),
            "torsions": torsions,
            "ensemble": "implicit",
            "adaptation_steps": 400_000,
            "frozen_equilibration_steps": 50_000,
            "production_steps": 1_200_000,
            "report_interval": 100,
            "deposition_interval": 1000,
            "timestep_fs": 1.0,
            "seed": 3,
            **params,
        },
    }


def run(steps: list[dict], tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    return execute({"id": "meta", "steps": steps}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("meta")


def potential(axes: np.ndarray, k: float) -> np.ndarray:
    return k * (1 + np.cos(3 * axes))


class TestOneTorsion:
    @pytest.fixture(scope="class")
    def result(self, butane, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_OPENMM"] = ENV
        outcome = run(
            [step(butane, [torsion("phi", 1)])], tmp_path_factory.mktemp("one")
        )
        return outcome, ok(outcome)

    def surface(self, result):
        out = result[1]
        axes = np.array(out["surface"]["axes_radians"])
        f = np.array(
            [np.nan if v is None else v for v in out["surface"]["free_energy_kj_mol"]]
        )
        exact = potential(axes, K_ONE)
        return axes, f, exact - exact.min()

    def test_the_reweighted_surface_is_the_torsion_potential_to_half_a_kilojoule(
        self, result
    ):
        _, f, exact = self.surface(result)
        low = np.isfinite(f) & (exact < 10)
        assert np.abs(f[low] - exact[low]).max() < 1.0
        assert np.sqrt(np.mean((f[low] - exact[low]) ** 2)) < 0.5

    def test_the_three_wells_are_equally_deep_and_the_barriers_are_fifteen(
        self, result
    ):
        axes, f, _ = self.surface(result)

        def at(angle):
            return float(
                np.nanmean(
                    f[
                        np.abs(((axes - angle + math.pi) % (2 * math.pi)) - math.pi)
                        < 0.08
                    ]
                )
            )

        wells = [at(math.radians(a)) for a in (-60, 60, 180)]
        barriers = [at(math.radians(a)) for a in (0, 120, -120)]
        assert max(wells) - min(wells) < 1.0
        assert all(abs(b - 15.0) < 2.0 for b in barriers), barriers

    def test_the_surface_has_the_threefold_symmetry_of_the_potential(self, result):
        axes, f, _ = self.surface(result)
        shifted = np.roll(f, len(f) // 3)
        both = np.isfinite(f) & np.isfinite(shifted)
        assert np.abs(f[both] - shifted[both]).max() < 1.5

    def test_the_diagnostics_say_the_sampling_was_good(self, result):
        out = result[1]
        assert out["bins_visited_fraction"] == 1.0
        assert (
            out["effective_sample_fraction"] > 0.2
            and out["effective_sample_size"] > 1000
        )
        assert out["half_to_half_rms_kj_mol"] < 1.5
        assert out["samples"] == 12000
        assert out["kt_kj_mol"] == pytest.approx(0.00831446261815324 * 300)

    def test_the_stored_samples_are_the_production_frames_with_their_bias_energies(
        self, result
    ):
        out = result[1]
        rows = list(csv.reader(open(out["samples_table"])))
        assert rows[0] == ["step", "time_ps", "phi_radians", "bias_energy_kj_mol"]
        assert len(rows) == 12001
        angles = np.array([float(r[2]) for r in rows[1:]])
        assert angles.min() >= -math.pi and angles.max() <= math.pi

    def test_the_adaptation_estimate_is_recorded_at_four_checkpoints(self, result):
        record = json.loads(Path(result[1]["record"]).read_text())
        assert [e["fraction"] for e in record["adaptation_estimates"]] == [
            0.25,
            0.5,
            0.75,
            1.0,
        ]
        assert record["adaptation_estimates"][-1]["step"] == 400_000
        assert record["torsions"][0]["atom_indices_zero_based"] == [0, 1, 2, 3]

    def test_the_report_draws_the_surface(self, result, tmp_path):
        page = result[0].report(tmp_path / "r.html").read_text()
        assert "<svg" in page


class TestTwoTorsions:
    def test_two_independent_torsions_give_the_sum_of_their_potentials(
        self, tmp_path_factory
    ):
        directory = tmp_path_factory.mktemp("pentane")
        files = build(directory, "pentane", 5, 4.0)
        outcome = run(
            [
                step(
                    files,
                    [
                        torsion("phi", 1, grid=64, width=0.7),
                        torsion("psi", 2, grid=64, width=0.7),
                    ],
                    bias_factor=8.0,
                    adaptation_steps=800_000,
                    production_steps=2_000_000,
                    histogram_bins=36,
                )
            ],
            directory,
        )
        out = ok(outcome)
        axes = np.array(out["surface"]["axes_radians"])
        f = np.array(
            [
                [np.nan if v is None else v for v in row]
                for row in out["surface"]["free_energy_kj_mol"]
            ]
        )
        assert f.shape == (36, 36)
        exact = potential(axes[:, None], 4.0) + potential(axes[None, :], 4.0)
        exact = exact - exact.min()
        low = np.isfinite(f) & (exact < 8)
        # A two-dimensional surface from two nanoseconds is noisier than a one-dimensional one:
        # single bins on the shoulders of the barriers can be off by several kJ/mol, so the
        # whole surface is judged by its rms and by the marginal along each torsion, which
        # averages that noise away.
        assert np.sqrt(np.mean((f[low] - exact[low]) ** 2)) < 1.5
        kt = out["kt_kj_mol"]
        weights = np.where(np.isfinite(f), np.exp(-f / kt), 0.0)
        for axis in (1, 0):
            marginal = -kt * np.log(weights.sum(axis=axis))
            marginal -= marginal.min()
            truth = potential(axes, 4.0)
            truth -= truth.min()
            assert np.abs(marginal - truth).max() < 1.5
        assert out["bins_visited_fraction"] > 0.9


STRUCTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/1L2Y.cif"


class TestProteinInWater:
    def test_a_backbone_torsion_runs_under_a_barostat_with_the_settings_recorded(
        self, tmp_path
    ):
        backbone = {
            "name": "phi2",
            "atoms": [
                {"chain_id": "A", "residue": "1", "atom_name": "C"},
                {"chain_id": "A", "residue": "2", "atom_name": "N"},
                {"chain_id": "A", "residue": "2", "atom_name": "CA"},
                {"chain_id": "A", "residue": "2", "atom_name": "C"},
            ],
            "width_radians": 0.35,
            "grid_width": 128,
        }
        steps = [
            {
                "id": "prep",
                "capability": "structure.prepare_protein_system",
                "params": {
                    "structure": str(STRUCTURE),
                    "chain_id": "A",
                    "solvent": "tip3p",
                    "padding_nm": 1.0,
                },
            },
            {
                "id": "meta",
                "capability": "simulation.run_metadynamics",
                "bind": {
                    "system": "prep.system_xml",
                    "state": "prep.state_xml",
                    "topology": "prep.topology_pdb",
                },
                "params": {
                    "torsions": [backbone],
                    "ensemble": "npt",
                    "adaptation_steps": 4000,
                    "deposition_interval": 500,
                    "production_steps": 2000,
                    "report_interval": 100,
                    "pressure_bar": 2.0,
                    "barostat_interval": 10,
                    "friction_per_ps": 2.0,
                    "platform": "CPU",
                    "platform_properties": {"Threads": "2"},
                },
            },
        ]
        out = ok(run(steps, tmp_path))
        record = json.loads(Path(out["record"]).read_text())
        assert out["samples"] == 20 and out["platform"] == "CPU"
        assert (
            record["settings"]["pressure_bar"] == 2.0
            and record["settings"]["barostat_interval"] == 10
        )
        assert record["settings"]["friction_per_ps"] == 2.0
        assert record["platform_properties"]["Threads"] == "2"

    def test_an_implicit_run_records_no_barostat(self, butane, tmp_path):
        out = ok(
            run(
                [
                    step(
                        butane,
                        [torsion("phi", 1)],
                        adaptation_steps=4000,
                        production_steps=2000,
                        frozen_equilibration_steps=0,
                    )
                ],
                tmp_path,
            )
        )
        record = json.loads(Path(out["record"]).read_text())
        assert (
            record["settings"]["pressure_bar"] is None
            and record["settings"]["barostat_interval"] is None
        )


class TestRefusals:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"bias_factor": 1.5}, "bias_factor must be between 2 and 100"),
            ({"height_kj_mol": 50.0}, "height_kj_mol must be between 0.001 and 20"),
            ({"deposition_interval": 0}, "deposition_interval must be an integer"),
            ({"adaptation_steps": 400_500}, "multiple of deposition_interval"),
            ({"adaptation_steps": 2000}, "at least four deposition intervals"),
            ({"production_steps": 1_200_050}, "multiple of report_interval"),
            ({"histogram_bins": 4}, "histogram_bins must be an integer from 8 to 180"),
            (
                {"histogram_bins": 500},
                "histogram_bins must be an integer from 8 to 180",
            ),
            ({"temperature_kelvin": 900}, "temperature_kelvin"),
            ({"timestep_fs": 10}, "timestep_fs"),
            ({"seed": 0}, "seed"),
            ({"ensemble": "npt"}, "needs a periodic system"),
            ({"ensemble": "langevin"}, "expected one of"),
            ({"report_interval": 0}, "report_interval"),
            ({"barostat_interval": 0}, "barostat_interval"),
        ],
    )
    def test_a_run_that_cannot_mean_what_it_says_is_refused_in_plain_words(
        self, butane, tmp_path, changes, message
    ):
        outcome = run([step(butane, [torsion("phi", 1)], **changes)], tmp_path)
        assert outcome.steps["meta"].status == "failed"
        assert message in outcome.steps["meta"].error

    @pytest.mark.parametrize(
        ("torsions", "message"),
        [
            ([], "one or two periodic torsions"),
            (
                [torsion("a", 1), torsion("b", 1), torsion("c", 1)],
                "one or two periodic torsions",
            ),
            ([torsion("1bad", 1)], "unique identifiers"),
            ([torsion("a", 1), torsion("a", 1)], "unique identifiers"),
            ([torsion("a", 1, width=0.01)], "width_radians"),
            ([torsion("a", 1, grid=8)], "grid_width"),
            (
                [torsion("a", 1, width=0.35, grid=40)],
                "four intervals per Gaussian width",
            ),
        ],
    )
    def test_torsions_that_cannot_be_biased_are_refused(
        self, butane, tmp_path, torsions, message
    ):
        outcome = run([step(butane, torsions)], tmp_path)
        assert message in outcome.steps["meta"].error

    def test_a_torsion_on_atoms_that_are_not_bonded_in_a_path_is_refused(
        self, butane, tmp_path
    ):
        crooked = torsion("phi", 1)
        crooked["atoms"] = [crooked["atoms"][i] for i in (0, 2, 1, 3)]
        outcome = run([step(butane, [crooked])], tmp_path)
        assert "bonded three-bond path" in outcome.steps["meta"].error

    def test_an_atom_the_topology_does_not_have_is_named(self, butane, tmp_path):
        missing = torsion("phi", 1)
        missing["atoms"][3] = {"chain_id": "A", "residue": "1", "atom_name": "C9"}
        outcome = run([step(butane, [missing])], tmp_path)
        assert "must match exactly once" in outcome.steps["meta"].error

    def test_the_same_atom_twice_is_refused(self, butane, tmp_path):
        repeated = torsion("phi", 1)
        repeated["atoms"][3] = repeated["atoms"][0]
        assert (
            "four distinct atoms"
            in run([step(butane, [repeated])], tmp_path).steps["meta"].error
        )

    def test_a_residue_label_that_is_not_a_residue_number_is_refused(
        self, butane, tmp_path
    ):
        bad = torsion("phi", 1)
        bad["atoms"][0]["residue"] = "A1"
        assert (
            "author residue number"
            in run([step(butane, [bad])], tmp_path).steps["meta"].error
        )
