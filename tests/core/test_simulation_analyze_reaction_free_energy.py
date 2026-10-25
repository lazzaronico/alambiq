"""simulation.analyze_reaction_free_energy on windows sampled from a known density.

Each window is written as the files `simulation.run_qmmm_dynamics` leaves: a summary and a
state table. Their coordinate values are exact Gaussian draws from the biased density
exp(-(q^2 + K (q - c)^2) / RT), with q^2 the unbiased energy, so that the unbiased density is a
Gaussian of variance RT/2 and everything the analysis returns has an analytic value:

* the profile relative to the reference bin is -RT ln(P_bin / P_reference), P from the normal
  distribution function, within a few of the reported replica standard errors;
* the basin difference is -RT ln(P_product / P_reactant) over the declared bins;
* the classical flux follows the closed form for the density at the dividing surface;
* every way the recorded windows can disagree with each other is refused before the analysis.

The analysis runs in the pinned PyMBAR environment (ALAMBIQ_ENV_PYMBAR).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
norm = pytest.importorskip("scipy.stats").norm

ENV = os.environ.get("ALAMBIQ_ENV_PYMBAR", "")
RT = 0.00831446261815324 * 300.0
CENTERS = [-1.5, -0.75, 0.0, 0.75, 1.5]
K = 5.0
EDGES = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
SAMPLES = 4000

pytestmark = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_PYMBAR to run the umbrella analysis",
)


def sample_window(rng, center):
    # exp(-(q^2 + K (q-c)^2)/RT) is Gaussian with variance RT/(2(1+K)), mean K c/(1+K).
    return rng.normal(K * center / (1 + K), math.sqrt(RT / (2 * (1 + K))), SAMPLES)


def write_window(
    directory: Path,
    name,
    q,
    center,
    *,
    seed,
    hamiltonian="b" * 64,
    temperature=300.0,
    pairs=((1, 4), (2, 5)),
    umbrella=True,
    system="a" * 64,
    coefficient=K,
):
    table = directory / f"{name}.csv"
    rows = ["phase,step,coordinate_angstrom"] + [f"initial,0,{float(q[0])!r}"]
    rows += [f"production,{i + 1},{float(value)!r}" for i, value in enumerate(q)]
    table.write_text("\n".join(rows) + "\n")
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    summary = {
        "quantity": "molecular-qmmm-trajectory",
        "ensemble": "NVT",
        "source_sha256": {"system_xml": system},
        "hamiltonian_sha256": hamiltonian,
        "sampling": {
            "temperature_kelvin": temperature,
            "seed": seed,
            "report_interval": 2,
            "timestep_fs": 0.5,
            "umbrella": {
                "pairs": [list(p) for p in pairs],
                "center_angstrom": center,
                "coefficient_kj_mol_angstrom2": coefficient,
            }
            if umbrella
            else None,
        },
        "frame_interval_fs": 1.0,
        "production_frames": len(q),
        "outputs_sha256": {"state.csv": digest},
    }
    path = directory / f"{name}.json"
    path.write_text(json.dumps(summary))
    return {
        "window": f"w{CENTERS.index(center)}" if center in CENTERS else name,
        "summary": str(path),
        "table": str(table),
        "discard_frames": 0,
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("windows")
    rng = np.random.default_rng(62419)
    replicas = []
    for r in range(2):
        windows = []
        for index, center in enumerate(CENTERS):
            windows.append(
                write_window(
                    directory,
                    f"r{r}w{index}",
                    sample_window(rng, center),
                    center,
                    seed=100 * r + index + 1,
                )
            )
        replicas.append({"id": f"r{r}", "windows": windows})
    return directory, replicas


def request(replicas, **changes):
    return {
        "title": "analytic harmonic density",
        "replicas": replicas,
        "bin_edges_angstrom": EDGES,
        "reference_bin": 3,
        "reactant_bins": [2, 3],
        "product_bins": [5, 6],
        "transition_bins": [4],
        "block_counts": [2],
        **changes,
    }


def run(params, tmp_path, name="run"):
    from alambiq.api import run as execute

    os.environ["ALAMBIQ_ENV_PYMBAR"] = ENV
    step = {
        "id": "rx",
        "capability": "simulation.analyze_reaction_free_energy",
        "params": params,
    }
    return execute({"id": "rx", "steps": [step]}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("rx")


def bin_probability(low, high):
    sigma = math.sqrt(RT / 2)
    return norm.cdf(high, scale=sigma) - norm.cdf(low, scale=sigma)


@pytest.fixture(scope="module")
def analysed(runs, tmp_path_factory):
    outcome = run(request(runs[1]), tmp_path_factory.mktemp("analysis"))
    return outcome, ok(outcome)


class TestTheProfile:
    def test_the_profile_is_minus_rt_ln_of_the_normal_bin_probabilities(self, analysed):
        out = analysed[1]
        reference = bin_probability(0.0 - 0.5, 0.0)  # bin 3 is [-0.5, 0)
        for i, (low, high) in enumerate(zip(EDGES[:-1], EDGES[1:], strict=True)):
            expected = -RT * math.log(bin_probability(low, high) / reference)
            mean, sem = out["pmf_mean_kj_mol"][i], out["pmf_replica_sem_kj_mol"][i]
            assert mean is not None
            assert abs(mean - expected) < max(4 * (sem or 0.0), 0.35), (
                i,
                mean,
                expected,
            )

    def test_the_reference_bin_is_zero_with_no_error(self, analysed):
        assert analysed[1]["pmf_mean_kj_mol"][3] == 0.0
        assert analysed[1]["pmf_replica_sd_kj_mol"][3] == 0.0

    def test_the_profile_is_symmetric_because_the_density_is(self, analysed):
        pmf = analysed[1]["pmf_mean_kj_mol"]
        # Bins 0..3 mirror bins 7..4 about q = 0 (reference bin 3 mirrors bin 4).
        assert pmf[4] == pytest.approx(pmf[3], abs=0.5)
        assert pmf[2] == pytest.approx(pmf[5], abs=0.5)
        assert pmf[0] == pytest.approx(pmf[7], abs=1.0)

    def test_the_basin_difference_is_minus_rt_ln_of_the_basin_probabilities(
        self, analysed
    ):
        out = analysed[1]
        product = bin_probability(0.5, 1.5)
        reactant = bin_probability(-1.0, 0.0)
        expected = -RT * math.log(product / reactant)
        assert out["basin_delta_g_mean_kj_mol"] == pytest.approx(
            expected, abs=max(4 * out["basin_delta_g_replica_sem_kj_mol"], 0.35)
        )

    def test_two_independent_replicas_agree_and_every_window_overlaps_its_neighbours(
        self, analysed
    ):
        out = analysed[1]
        assert [r["id"] for r in out["replicas"]] == ["r0", "r1"]
        for replica in out["replicas"]:
            assert replica["minimum_adjacent_overlap"] > 0.01
            assert replica["minimum_state_effective_samples"] > 100
            assert replica["blocks_failed"] == 0 and replica["blocks_total"] == 2
        a, b = (r["basin_delta_g_kj_mol"] for r in out["replicas"])
        assert abs(a - b) < 1.0

    def test_the_run_describes_what_it_analysed(self, analysed):
        out = analysed[1]
        assert out["windows"] == 5 and out["frames"] == 2 * 5 * SAMPLES
        assert out["temperature_kelvin"] == 300.0
        assert out["coordinate"] == "distance(1,4)-distance(2,5)"
        assert out["bin_centres_angstrom"][0] == pytest.approx(-1.75)
        assert out["pymbar_version"] == "4.2.0"

    def test_the_details_keep_the_blocks_and_the_sampling_record(self, analysed):
        details = json.loads(Path(analysed[1]["details"]).read_text())
        replica = details["replicas"][0]
        assert replica["time_blocks"][0]["count"] == 2
        assert all(b["error"] is None for b in replica["time_blocks"][0]["blocks"])
        assert replica["sampling"][0]["selected_samples"] >= 5

    def test_the_report_has_the_profile_chart(self, analysed, tmp_path):
        page = analysed[0].report(tmp_path / "r.html").read_text()
        assert "<svg" in page


class TestTheFlux:
    def test_the_classical_flux_follows_the_closed_form_for_the_surface_density(
        self, runs, tmp_path
    ):
        inverse_mass = 0.5
        out = ok(
            run(
                request(
                    runs[1],
                    transition_state_flux={
                        "transition_bin": 4,
                        "inverse_mass_dalton": inverse_mass,
                    },
                ),
                tmp_path,
            )
        )
        reactant = bin_probability(-1.0, 0.0)
        surface = bin_probability(0.0, 0.5)
        velocity = 10 * math.sqrt(RT * inverse_mass / (2 * math.pi))
        log_rate = (
            math.log(surface / 0.5)
            - math.log(reactant)
            + math.log(velocity)
            + math.log(1e12)
        )
        frequency = 1.380649e-23 * 300.0 / 6.62607015e-34
        expected = -RT * (log_rate - math.log(frequency))
        flux = out["flux"]
        assert flux["log_rate_per_second_mean"] == pytest.approx(log_rate, abs=0.25)
        assert flux["activation_free_energy_kj_mol_mean"] == pytest.approx(
            expected, abs=0.6
        )
        assert flux["activation_free_energy_kj_mol_replica_sem"] >= 0


class TestRefusals:
    def fails(self, params, tmp_path, message):
        outcome = run(params, tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["rx"].error

    def copy_runs(self, runs, **per_window):
        """Rewrite every window's files with changes for chosen (replica, window) pairs."""
        directory, replicas = runs
        return directory, replicas

    def test_one_replica_is_refused(self, runs, tmp_path):
        self.fails(
            request(runs[1][:1]), tmp_path, "2–20 independently sampled replicas are required"
        )

    def test_replicas_with_different_windows_are_refused(self, runs, tmp_path):
        shorter = {"id": "r1", "windows": runs[1][1]["windows"][:4]}
        self.fails(
            request([runs[1][0], shorter]), tmp_path, "same complete ordered windows"
        )

    def test_a_table_that_is_not_the_one_the_run_wrote_is_refused(self, runs, tmp_path):
        replicas = json.loads(json.dumps(runs[1]))
        replicas[0]["windows"][0]["table"] = replicas[1]["windows"][0]["table"]
        self.fails(request(replicas), tmp_path, "is not the one its run wrote")

    def test_a_window_run_without_an_umbrella_is_refused(self, runs, tmp_path):
        rng = np.random.default_rng(5)
        replicas = json.loads(json.dumps(runs[1]))
        replicas[0]["windows"][0] = write_window(
            tmp_path,
            "bare",
            sample_window(rng, CENTERS[0]),
            CENTERS[0],
            seed=901,
            umbrella=False,
        )
        replicas[0]["windows"][0]["window"] = "w0"
        self.fails(request(replicas), tmp_path, "run without an umbrella bias")

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"hamiltonian": "c" * 64}, "same Hamiltonian, temperature and coordinate"),
            ({"temperature": 310.0}, "same Hamiltonian, temperature and coordinate"),
            (
                {"pairs": ((1, 4), (3, 5))},
                "same Hamiltonian, temperature and coordinate",
            ),
            ({"seed": 2}, "own random seed"),
            ({"coefficient": 9.0}, "same recorded centers and coefficients"),
        ],
    )
    def test_windows_that_do_not_belong_together_are_refused(
        self, runs, tmp_path, changes, message
    ):
        rng = np.random.default_rng(6)
        replicas = json.loads(json.dumps(runs[1]))
        arguments = {"seed": 777, **changes}
        replicas[1]["windows"][2] = write_window(
            tmp_path, "odd", sample_window(rng, CENTERS[2]), CENTERS[2], **arguments
        )
        replicas[1]["windows"][2]["window"] = "w2"
        self.fails(request(replicas), tmp_path, message)

    def test_a_table_with_a_different_frame_count_than_its_run_is_refused(
        self, runs, tmp_path
    ):
        replicas = json.loads(json.dumps(runs[1]))
        window = replicas[0]["windows"][1]
        summary = json.loads(Path(window["summary"]).read_text())
        table = tmp_path / "shorter.csv"
        lines = Path(window["table"]).read_text().splitlines()
        table.write_text("\n".join(lines[:-3]) + "\n")
        summary["outputs_sha256"]["state.csv"] = hashlib.sha256(
            table.read_bytes()
        ).hexdigest()
        edited = tmp_path / "edited.json"
        edited.write_text(json.dumps(summary))
        window["summary"], window["table"] = str(edited), str(table)
        self.fails(
            request(replicas), tmp_path, "production frames but the run recorded"
        )

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"bin_edges_angstrom": [0, 1, 2]}, "Supply 3–200 histogram bins"),
            ({"bin_edges_angstrom": [-2, -1, -1.5, 0, 1, 2]}, "strictly increasing"),
            ({"reference_bin": 6}, "reference bin must belong to the reactant basin"),
            ({"reactant_bins": [2, 3, 4]}, "must be distinct"),
            ({"reactant_bins": [2, 99]}, "explicit zero-based histogram bins"),
            ({"block_counts": [1]}, "time-block counts between 2 and 10"),
            ({"block_counts": [2, 2]}, "time-block counts between 2 and 10"),
            (
                {
                    "transition_state_flux": {
                        "transition_bin": 0,
                        "inverse_mass_dalton": 1.0,
                    }
                },
                "dividing-surface bin must belong to the transition region",
            ),
            (
                {
                    "transition_state_flux": {
                        "transition_bin": 4,
                        "inverse_mass_dalton": -1.0,
                    }
                },
                "Constant inverse coordinate mass",
            ),
        ],
    )
    def test_bins_and_blocks_that_cannot_be_analysed_are_refused_in_plain_words(
        self, runs, tmp_path, changes, message
    ):
        self.fails(request(runs[1], **changes), tmp_path, message)

    def test_a_discarded_prefix_that_leaves_too_few_frames_is_refused(
        self, runs, tmp_path
    ):
        replicas = json.loads(json.dumps(runs[1]))
        for replica in replicas:
            for window in replica["windows"]:
                window["discard_frames"] = SAMPLES - 10
        self.fails(
            request(replicas), tmp_path, "leave at least 20 frames in every time block"
        )

    def test_windows_that_do_not_overlap_are_refused(self, runs, tmp_path):
        # Windows with very stiff restraints far apart share no samples.
        rng = np.random.default_rng(8)
        replicas = []
        for r in range(2):
            windows = []
            for index, center in enumerate([-1.5, 1.5]):
                q = rng.normal(center, 0.01, SAMPLES)
                windows.append(
                    write_window(
                        tmp_path,
                        f"d{r}{index}",
                        q,
                        center,
                        seed=500 + 10 * r + index,
                        coefficient=1.0e6,
                    )
                )
                windows[-1]["window"] = f"w{index}"
            replicas.append({"id": f"r{r}", "windows": windows})
        self.fails(
            request(
                replicas,
                bin_edges_angstrom=[-2, -1.5, -1, 0, 1, 1.5, 2],
                reference_bin=0,
                reactant_bins=[0, 1],
                product_bins=[4, 5],
                transition_bins=[2],
            ),
            tmp_path,
            "disconnected",
        )
