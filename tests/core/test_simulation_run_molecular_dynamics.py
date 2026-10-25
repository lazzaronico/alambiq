"""simulation.run_molecular_dynamics with the real OpenMM.

The analytic case is two argon atoms joined by a harmonic bond (k = 1000 kJ/mol/nm²,
r0 = 0.25 nm), released at rest at 0.30 nm. Their relative coordinate is a harmonic
oscillator with reduced mass μ = m/2, so ω = sqrt(k/μ) = 7.0797 per ps, the total
energy stays at ½k(0.05)² = 1.25 kJ/mol, and the potential energy follows
1.25·cos²(ωt) with period π/ω = 0.4437 ps. A correct integrator reproduces that
curve and conserves the energy; no force field is involved.

The protein cases run Trp-cage, prepared by `structure.prepare_protein_system`, for
a few picoseconds: they check the structure of the run (phases, frames, restraints,
reproducibility), not the physics, which a few picoseconds cannot test.

Set ALAMBIQ_ENV_OPENMM to the pinned openmm-ensemble environment's Python.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

ENV = os.environ.get("ALAMBIQ_ENV_OPENMM", "")
STRUCTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/1L2Y.cif"
K, MASS, R0, R_START = 1000.0, 39.9, 0.25, 0.30
OMEGA = math.sqrt(K / (MASS / 2))  # per ps
ENERGY = 0.5 * K * (R_START - R0) ** 2

pytestmark = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_OPENMM to run the OpenMM tests",
)

HARMONIC = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    import openmm
    from openmm import app, unit
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    system = openmm.System()
    for _ in range(2):
        system.addParticle(39.9 * unit.amu)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.25 * unit.nanometer, 1000 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond)
    topology = app.Topology()
    residue = topology.addResidue("DUM", topology.addChain("A"))
    topology.addAtom("A", app.element.argon, residue)
    topology.addAtom("B", app.element.argon, residue)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    positions = [[0, 0, 0], [0.30, 0, 0]] * unit.nanometer
    context.setPositions(positions)
    context.setVelocities([[0, 0, 0], [0, 0, 0]] * unit.nanometer / unit.picosecond)
    (out / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    (out / "state.xml").write_text(openmm.XmlSerializer.serialize(context.getState(getPositions=True, getVelocities=True, getEnergy=True)))
    with (out / "topology.pdb").open("w") as stream:
        app.PDBFile.writeFile(topology, positions, stream)
    """
)


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    if ENV:
        monkeypatch.setenv("ALAMBIQ_ENV_OPENMM", ENV)


def run(steps: list[dict], tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    return execute({"id": "md", "steps": steps}, run_dir=tmp_path / name)


def ok(outcome, step="md"):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs(step)


@pytest.fixture(scope="module")
def harmonic(tmp_path_factory):
    directory = tmp_path_factory.mktemp("harmonic")
    script = directory / "build.py"
    script.write_text(HARMONIC)
    subprocess.run(
        [ENV, str(script), str(directory / "system")], check=True, capture_output=True
    )
    return directory / "system"


def harmonic_step(files: Path, **params) -> dict:
    return {
        "id": "md",
        "capability": "simulation.run_molecular_dynamics",
        "params": {
            "system": str(files / "system.xml"),
            "state": str(files / "state.xml"),
            "topology": str(files / "topology.pdb"),
            "ensemble": "nve",
            "velocities": "state",
            "production_ps": 1.0,
            "timestep_fs": 1.0,
            "report_interval": 10,
            "platform": "Reference",
            **params,
        },
    }


def samples(out: dict) -> list[list[float]]:
    return [
        [float(v) for v in row] for row in list(csv.reader(open(out["state_data"])))[1:]
    ]


class TestHarmonicOscillator:
    @pytest.fixture(scope="class")
    def result(self, harmonic, tmp_path_factory):
        return ok(run([harmonic_step(harmonic)], tmp_path_factory.mktemp("h")))

    def test_the_potential_energy_follows_the_analytic_curve(self, result):
        # Leapfrog treats the starting velocity as that of half a step earlier, so the
        # numerical curve is the analytic one shifted by dt/2 = 0.5 fs (0.0035 rad,
        # which moves the energy by at most 0.35 % of 1.25 kJ/mol). Allow that and no more.
        rows = samples(result)
        worst = max(abs(r[2] - ENERGY * math.cos(OMEGA * r[1]) ** 2) for r in rows)
        shifted = max(
            abs(r[2] - ENERGY * math.cos(OMEGA * (r[1] + 0.0005)) ** 2) for r in rows
        )
        assert worst < 6e-3 and shifted < worst and shifted < 2e-3, (worst, shifted)

    def test_the_total_energy_is_conserved_to_a_part_in_ten_thousand(self, result):
        assert result["total_energy_first_kj_mol"] == pytest.approx(ENERGY, rel=1e-4)
        spread = result["total_energy_max_kj_mol"] - result["total_energy_min_kj_mol"]
        assert spread / ENERGY < 1e-4

    def test_the_potential_energy_hits_zero_a_quarter_period_in_and_returns_after_half(
        self, result
    ):
        by_time = {round(row[1], 2): row[2] for row in samples(result)}
        quarter = round(math.pi / (2 * OMEGA), 2)  # 0.22 ps
        half = round(math.pi / OMEGA, 2)  # 0.44 ps
        assert by_time[quarter] < 0.01
        assert by_time[half] > 0.99 * ENERGY

    def test_the_run_records_one_hundred_frames_and_the_step_count(self, result):
        assert result["frames"] == 100 and result["final_step"] == 1000
        assert result["final_time_ps"] == pytest.approx(1.0)
        assert result["production_run"] and result["ready_for_production"]

    def test_a_longer_timestep_conserves_energy_less_well(self, harmonic, tmp_path):
        coarse = ok(
            run([harmonic_step(harmonic, timestep_fs=4.0, report_interval=1)], tmp_path)
        )
        fine = ok(run([harmonic_step(harmonic)], tmp_path, "fine"))
        coarse_spread = (
            coarse["total_energy_max_kj_mol"] - coarse["total_energy_min_kj_mol"]
        )
        fine_spread = fine["total_energy_max_kj_mol"] - fine["total_energy_min_kj_mol"]
        assert coarse_spread > 5 * fine_spread

    def test_nve_with_fresh_velocities_warns_of_a_random_kinetic_energy(
        self, harmonic, tmp_path
    ):
        outcome = run(
            [harmonic_step(harmonic, velocities="maxwell-boltzmann")], tmp_path
        )
        assert outcome.ok
        assert "random kinetic energy" in " ".join(outcome.steps["md"].notes)

    def test_the_summary_names_the_platform_and_the_versions(self, result):
        summary = json.loads(Path(result["summary"]).read_text())
        assert summary["platform"] == "Reference" and summary["versions"][
            "openmm"
        ].startswith("8.")
        assert len(summary["inputs"]["system_xml"]) == 64


class TestHarmonicRefusals:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"thermalize_ps": 5.0}, "nve ensemble has no thermostat"),
            ({"release_ps": 5.0}, "no thermostat"),
            ({"ensemble": "npt"}, "needs a periodic system"),
            ({"ensemble": "nvt"}, "needs a periodic system"),
            (
                {"production_ps": 0.0},
                "Nothing to run",
            ),
            (
                {"timestep_fs": 10.0},
                "timestep_fs must be between 0.1 and 4",
            ),
            ({"temperature_kelvin": 900}, "temperature_kelvin"),
            ({"timestep_fs": 0.01}, "timestep_fs"),
            ({"report_interval": 0}, "report_interval"),
            ({"seed": 0}, "seed"),
            ({"readiness_window_ps": 5.0, "ensemble": "implicit"}, "at least 20"),
            ({"max_readiness_windows": 11}, "max_readiness_windows"),
            ({"platform": "TPU"}, "expected one of"),
            ({"ensemble": "langevin"}, "expected one of"),
            ({"production_ps": 100.0, "report_interval": 1}, "20,000 frames"),
        ],
    )
    def test_a_run_that_cannot_mean_what_it_says_is_refused_in_plain_words(
        self, harmonic, tmp_path, changes, message
    ):
        outcome = run([harmonic_step(harmonic, **changes)], tmp_path)
        assert outcome.steps["md"].status == "failed"
        assert message in outcome.steps["md"].error

    def test_a_topology_with_the_wrong_number_of_atoms_is_refused(
        self, harmonic, tmp_path
    ):
        three = tmp_path / "three.pdb"
        three.write_text(
            "ATOM      1  A   DUM A   1       0.000   0.000   0.000  1.00  0.00           AR\n"
            "ATOM      2  B   DUM A   1       3.000   0.000   0.000  1.00  0.00           AR\n"
            "ATOM      3  C   DUM A   1       6.000   0.000   0.000  1.00  0.00           AR\n"
            "END\n"
        )
        step = harmonic_step(harmonic)
        step["params"]["topology"] = str(three)
        assert "different numbers of atoms" in run([step], tmp_path).steps["md"].error

    def test_a_system_that_is_not_xml_data_is_refused(self, harmonic, tmp_path):
        bad = tmp_path / "bad.xml"
        bad.write_text("<System type='System'><Evil type='os.system'/></System>")
        step = harmonic_step(harmonic)
        step["params"]["system"] = str(bad)
        assert (
            "not an allowed native data type" in run([step], tmp_path).steps["md"].error
        )


def protein_steps(
    solvent: str, dynamics: dict, extra: dict | None = None
) -> list[dict]:
    return [
        {
            "id": "prep",
            "capability": "structure.prepare_protein_system",
            "params": {
                "structure": str(STRUCTURE),
                "chain_id": "A",
                "solvent": solvent,
                **(extra or {}),
            },
        },
        {
            "id": "md",
            "capability": "simulation.run_molecular_dynamics",
            "bind": {
                "system": "prep.system_xml",
                "state": "prep.state_xml",
                "topology": "prep.topology_pdb",
            },
            "params": dynamics,
        },
    ]


def load_trajectory(out: dict):
    """Read the stored DCD against the stored structure; neither file name carries an extension."""
    import mdtraj

    return mdtraj.load_dcd(
        out["trajectory"], top=mdtraj.load_pdb(out["final_structure"]).topology
    )


IMPLICIT = {
    "ensemble": "implicit",
    "thermalize_ps": 1.0,
    "readiness_window_ps": 0,
    "production_ps": 2.0,
    "report_interval": 100,
}


class TestImplicitProtein:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_OPENMM"] = ENV
        return run(protein_steps("gbn2", IMPLICIT), tmp_path_factory.mktemp("implicit"))

    def test_the_phases_run_in_order_and_production_writes_the_expected_frames(
        self, outcome
    ):
        out = ok(outcome)
        assert [p["name"] for p in out["phases"]] == ["thermalization", "production"]
        assert out["frames"] == 10 and out["final_step"] == 1500
        assert out["ensemble"] == "implicit" and out["particles"] == 304

    def test_the_trajectory_has_every_atom_and_every_frame(self, outcome):
        out = ok(outcome)
        trajectory = load_trajectory(out)
        assert (trajectory.n_frames, trajectory.n_atoms) == (10, 304)

    def test_restrained_thermalisation_keeps_the_protein_close_to_where_it_started(
        self, outcome
    ):
        import mdtraj

        out = ok(outcome)
        start = mdtraj.load_pdb(ok(outcome, "prep")["topology_pdb"])
        trajectory = load_trajectory(out)
        heavy = start.topology.select("not element H")
        rmsd = mdtraj.rmsd(trajectory, start, atom_indices=heavy)
        assert rmsd.max() < 0.3  # nm; a restrained start and 2 ps of free dynamics

    def test_the_temperature_sits_near_the_target(self, outcome):
        assert 220 < ok(outcome)["mean_temperature_kelvin"] < 380

    def test_the_same_seed_reproduces_the_trajectory_and_another_seed_does_not(
        self, outcome, tmp_path
    ):
        first = samples(ok(outcome))
        again = ok(run(protein_steps("gbn2", IMPLICIT), tmp_path, "again"))
        other = ok(
            run(protein_steps("gbn2", {**IMPLICIT, "seed": 2}), tmp_path, "other")
        )
        assert samples(again) == first
        assert samples(other) != first

    def test_a_run_without_restraints_is_allowed_and_recorded(self, outcome, tmp_path):
        free = ok(
            run(
                protein_steps("gbn2", {**IMPLICIT, "restraint_kj_mol_nm2": 0}), tmp_path
            )
        )
        phase = next(p for p in free["phases"] if p["name"] == "thermalization")
        assert phase["detail"]["restraint_kj_mol_nm2"] == 0.0

    def test_minimising_first_is_recorded_with_its_energies(self, outcome, tmp_path):
        out = ok(
            run(
                protein_steps("gbn2", {**IMPLICIT, "minimize_iterations": 50}), tmp_path
            )
        )
        phase = out["phases"][0]
        assert phase["name"] == "minimization"
        assert (
            phase["detail"]["potential_after_kj_mol"]
            <= phase["detail"]["potential_before_kj_mol"]
        )

    def test_a_system_that_never_warms_up_is_not_ready_and_production_is_withheld(
        self, tmp_path
    ):
        # With almost no friction the thermostat takes far longer than a window to warm a system
        # that starts at rest, so the first window cannot pass the temperature check.
        dynamics = {
            "ensemble": "implicit",
            "velocities": "state",
            "friction_per_ps": 0.01,
            "thermalize_ps": 0,
            "readiness_window_ps": 20,
            "max_readiness_windows": 1,
            "production_ps": 1.0,
            "report_interval": 100,
        }
        outcome = run(protein_steps("gbn2", dynamics), tmp_path)
        out = ok(outcome)
        assert out["ready_for_production"] is False and out["production_run"] is False
        assert out["frames"] == 0 and out["trajectory"] is None
        assert "No readiness window passed" in " ".join(outcome.steps["md"].notes)
        readiness = next(p for p in out["phases"] if p["name"] == "readiness")
        assert (
            "temperature_relative_error" in readiness["detail"]["windows"][0]["failed"]
        )

    def test_the_report_names_the_ensemble(self, outcome, tmp_path):
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "implicit" in page


class TestExplicitProtein:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_OPENMM"] = ENV
        dynamics = {
            "ensemble": "npt",
            "thermalize_ps": 1.0,
            "release_ps": 1.0,
            "readiness_window_ps": 20,
            "max_readiness_windows": 1,
            "require_ready": False,
            "production_ps": 2.0,
            "report_interval": 250,
        }
        return run(
            protein_steps("tip3p", dynamics, {"padding_nm": 1.0}),
            tmp_path_factory.mktemp("explicit"),
        )

    def test_the_four_phases_ran_in_order(self, outcome):
        out = ok(outcome)
        assert [p["name"] for p in out["phases"]] == [
            "thermalization",
            "release",
            "readiness",
            "production",
        ]
        assert out["frames"] == 4 and out["final_step"] == 12000

    def test_the_box_relaxed_under_the_barostat_to_a_water_like_density(self, outcome):
        out = ok(outcome)
        prepared = ok(outcome, "prep")["box_volume_nm3"]
        assert out["mean_volume_nm3"] != pytest.approx(prepared, rel=1e-3)
        assert 0.95 < out["mean_density_g_ml"] < 1.08

    def test_the_thermalisation_held_the_volume_fixed(self, outcome):
        out = ok(outcome)
        thermal = next(p for p in out["phases"] if p["name"] == "thermalization")
        assert thermal["detail"]["ensemble"] == "nvt"
        assert thermal["detail"]["statistics"]["volume_nm3"]["sd"] == pytest.approx(
            0, abs=1e-9
        )

    def test_the_readiness_window_reports_its_blocks_and_limits(self, outcome):
        window = next(p for p in ok(outcome)["phases"] if p["name"] == "readiness")[
            "detail"
        ]["windows"][0]
        assert set(window["observed"]) >= {
            "temperature_relative_error",
            "density_block_relative_range",
        }
        assert window["limits"]["temperature_relative_error"] == 0.05
        assert len(window["block_means"]["temperature_kelvin"]) == 4

    def test_a_constant_volume_run_keeps_the_prepared_box(self, outcome, tmp_path):
        dynamics = {
            "ensemble": "nvt",
            "thermalize_ps": 1.0,
            "readiness_window_ps": 0,
            "production_ps": 1.0,
            "report_interval": 250,
        }
        out = ok(run(protein_steps("tip3p", dynamics, {"padding_nm": 1.0}), tmp_path))
        prepared = ok(
            run(protein_steps("tip3p", dynamics, {"padding_nm": 1.0}), tmp_path), "prep"
        )
        assert out["mean_volume_nm3"] == pytest.approx(
            prepared["box_volume_nm3"], rel=1e-6
        )

    def test_a_higher_pressure_gives_a_smaller_box(self, tmp_path):
        short = {
            "ensemble": "npt",
            "thermalize_ps": 0.5,
            "release_ps": 0.5,
            "readiness_window_ps": 0,
            "production_ps": 2.0,
            "report_interval": 250,
            "barostat_interval": 10,
        }
        low = ok(
            run(
                protein_steps(
                    "tip3p", {**short, "pressure_bar": 1.0}, {"padding_nm": 1.0}
                ),
                tmp_path,
                "low",
            )
        )
        high = ok(
            run(
                protein_steps(
                    "tip3p", {**short, "pressure_bar": 3000.0}, {"padding_nm": 1.0}
                ),
                tmp_path,
                "high",
            )
        )
        # Water compresses by about 4.5e-5 per bar, so 3000 bar asks for a box several percent smaller.
        assert high["mean_volume_nm3"] < 0.99 * low["mean_volume_nm3"]

    def test_the_platform_properties_are_passed_to_openmm_and_recorded(self, tmp_path):
        out = ok(
            run(
                protein_steps(
                    "gbn2", {**IMPLICIT, "platform_properties": {"Threads": "2"}}
                ),
                tmp_path,
            )
        )
        assert (
            json.loads(Path(out["summary"]).read_text())["platform_properties"][
                "Threads"
            ]
            == "2"
        )

    def test_an_implicit_system_cannot_run_under_a_barostat(self, tmp_path):
        outcome = run(
            protein_steps("gbn2", {**IMPLICIT, "ensemble": "npt", "release_ps": 0}),
            tmp_path,
        )
        assert "needs a periodic system" in outcome.steps["md"].error

    def test_an_explicit_system_cannot_run_as_implicit(self, tmp_path):
        outcome = run(
            protein_steps(
                "tip3p", {**IMPLICIT, "readiness_window_ps": 0}, {"padding_nm": 1.0}
            ),
            tmp_path,
        )
        assert "needs a non-periodic system" in outcome.steps["md"].error
