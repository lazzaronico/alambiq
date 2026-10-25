"""simulation.compare_replica_structures against replicas with known statistics.

Each replica is a Trp-cage trajectory whose frames are the deposited structure scaled
about its alpha-carbon centroid by a factor s(t), so the alpha-carbon radius of
gyration of frame t is exactly s(t) times the original. The factors are drawn
from known processes, which fixes what every statistic must be:

* the mean over replicas and its standard error follow from the replica means
  with the t quantile for three degrees of freedom (3.1824463);
* independent factors give a split R-hat near 1 and an effective size near the
  number of frames, while an AR(1) process with coefficient 0.9 has an
  effective size of N (1 - 0.9) / (1 + 0.9), about a nineteenth of N;
* replicas whose means are separated by many standard deviations cannot have an
  R-hat near 1;
* a reference structure scaled by s0 has radius of gyration s0 times the original, so
  the offset of the mean from it is known.

The replicas are inspected by `simulation.inspect_trajectory` and compared in one
campaign whose bindings assemble the list of replicas from the inspection steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
md = pytest.importorskip("mdtraj")
pytest.importorskip("gemmi")
pytest.importorskip("scipy")

from tests.core.test_simulation_inspect_trajectory import (  # noqa: E402, F401
    ok,
    radius_of_gyration,
    run,
    transform,
    trpcage,
    write_dcd,
    write_pdb,
    write_state,
)

T975_3 = 3.1824463
OBSERVABLE = "ca_radius_gyration_angstrom"
REFERENCE_SCALE = 1.05


def scales(kind: str, replica: int, frames: int = 80) -> np.ndarray:
    rng = np.random.default_rng(100 + replica)
    noise = rng.standard_normal(frames)
    if kind == "iid":
        return 1 + 0.02 * noise
    if kind == "ar1":
        x = np.empty(frames)
        x[0] = noise[0] / np.sqrt(1 - 0.81)
        for t in range(1, frames):
            x[t] = 0.9 * x[t - 1] + noise[t]
        return 1 + 0.02 * x / np.sqrt(1 / (1 - 0.81))
    if kind == "constant":
        return np.ones(frames)
    if kind.startswith("shift"):
        return 1 + 0.02 * noise + float(kind[5:]) * replica
    raise ValueError(kind)


class Study:
    """Write the replicas' trajectories and build the campaign steps for them."""

    def __init__(self, files: dict, directory: Path):
        self.files = files
        self.directory = directory
        self.top = files["protein_topology"]
        self.ca = self.top.select("protein and name CA")
        self.base = files["protein_xyz"]
        self.rg0 = radius_of_gyration(self.base, self.ca)
        reference = directory / "reference.pdb"
        write_pdb(reference, transform(self.base, REFERENCE_SCALE, self.ca), self.top)
        self.reference = reference
        self.series: list[np.ndarray] = []

    def replica(self, s: np.ndarray, name: str, *, volume: bool = False) -> dict:
        frames = [transform(self.base, float(v), self.ca) for v in s]
        write_dcd(self.directory / f"{name}.dcd", frames)
        write_state(
            self.directory / f"{name}.csv", len(frames), extra_first=True, volume=volume
        )
        self.series.append(np.asarray(s))
        return {
            "topology": str(self.files["protein"]),
            "trajectory": str(self.directory / f"{name}.dcd"),
            "state_data": str(self.directory / f"{name}.csv"),
        }

    def steps(self, inspections: list[dict], compare: dict, seeds=True) -> list[dict]:
        out = []
        for i, params in enumerate(inspections):
            out.append(
                {
                    "id": f"r{i}",
                    "capability": "simulation.inspect_trajectory",
                    "params": {
                        "reference_structure": str(self.reference),
                        "imaging": "as-stored",
                        **params,
                    },
                }
            )
        replicas = [
            {"id": f"rep{i}", **({"seed": 10 + i} if seeds else {})}
            for i in range(len(inspections))
        ]
        bind = {}
        for i in range(len(inspections)):
            bind[f"replicas.{i}.details"] = f"r{i}.details"
            bind[f"replicas.{i}.arrays"] = f"r{i}.arrays"
        out.append(
            {
                "id": "cmp",
                "capability": "simulation.compare_replica_structures",
                "params": {"replicas": replicas, **compare},
                "bind": bind,
            }
        )
        return out


@pytest.fixture(scope="module")
def study(trpcage, tmp_path_factory):  # noqa: F811
    return Study(trpcage, tmp_path_factory.mktemp("replicas"))


def compare(study, kinds, tmp_path, criteria=None, **extra):
    inspections = [
        study.replica(
            scales(kind, i), f"{kinds[0]}{i}_{abs(hash(tuple(kinds))) % 10**6}"
        )
        for i, kind in enumerate(kinds)
    ]
    criteria = criteria or [{"observable": OBSERVABLE, "precision": 0.05}]
    outcome = run(
        study.steps(inspections, {"criteria": criteria, **extra}),
        tmp_path,
    )
    return outcome


def verdict(out, name=OBSERVABLE):
    return next(o for o in out["observables"] if o["observable"] == name)


class TestIndependentReplicas:
    @pytest.fixture(scope="class")
    def result(self, study, tmp_path_factory):
        study.series.clear()
        outcome = compare(
            study,
            ["iid"] * 4,
            tmp_path_factory.mktemp("iid"),
            criteria=[
                {"observable": OBSERVABLE, "precision": 0.05, "tolerance": 0.5},
            ],
        )
        return ok(outcome, "cmp"), [s.copy() for s in study.series[-4:]], outcome

    def test_the_mean_and_its_standard_error_follow_the_replica_means(
        self, result, study
    ):
        out, series, _ = result
        means = np.array([study.rg0 * s.mean() for s in series])
        se = means.std(ddof=1) / np.sqrt(4)
        row = verdict(out)
        assert row["mean"] == pytest.approx(means.mean(), abs=2e-3)
        assert row["replica_standard_error"] == pytest.approx(se, abs=2e-3)
        assert row["ci95_low"] == pytest.approx(means.mean() - T975_3 * se, abs=1e-2)
        assert row["ci95_high"] == pytest.approx(means.mean() + T975_3 * se, abs=1e-2)

    def test_independent_draws_give_an_r_hat_near_one_and_a_large_effective_size(
        self, result
    ):
        row = verdict(result[0])
        assert row["diagnostic"] == "estimated"
        assert 1.0 <= row["split_rhat"] < 1.1
        assert 0.6 * 320 < row["effective_samples"] <= 320

    def test_the_offset_from_the_reference_is_the_mean_minus_its_radius_of_gyration(
        self, result, study
    ):
        row = verdict(result[0])
        assert row["reference_value"] == pytest.approx(
            REFERENCE_SCALE * study.rg0, abs=5e-3
        )
        assert row["mean_offset"] == pytest.approx(
            row["mean"] - row["reference_value"], abs=1e-9
        )
        # Replica means sit near 1.0 x Rg0, 0.05 x Rg0 below the reference.
        assert row["mean_offset"] == pytest.approx(-0.05 * study.rg0, abs=0.1)

    def test_a_tolerance_wider_than_the_offset_plus_the_interval_agrees(self, result):
        out = result[0]
        row = verdict(out)
        assert row["agrees"] is True and out["agrees_with_reference"] is True
        assert out["precision_passed"] and row["precision_passed"]

    def test_the_sampling_is_reported_in_frames_and_simulated_time(self, result):
        out = result[0]
        assert out["replicas"] == ["rep0", "rep1", "rep2", "rep3"]
        assert out["frames_per_replica"] == 80 and out["trimmed_frames"] == 0
        # The state samples are 100 steps apart; the interval is 100 steps at 2 fs.
        assert out["sampling_interval_ps"] == pytest.approx(0.2)
        assert out["simulated_ps_per_replica"] == pytest.approx(80 * 0.2)
        assert out["total_simulated_ps"] == pytest.approx(4 * 80 * 0.2)

    def test_the_checkpoints_end_at_the_full_length_and_grow(self, result):
        out = result[0]
        frames = [c["frames_per_replica"] for c in out["prefix_checks"]]
        assert frames == sorted(frames) and frames[0] == 40 and frames[-1] == 80
        assert all(f % 20 == 0 for f in frames)
        assert out["prefix_checks"][-1]["precision_passed"] is True
        assert out["first_sustained_checkpoint"]["frames_per_replica"] <= 80

    def test_the_fluctuation_is_the_distance_from_the_centre_times_the_mean_spread(
        self, result, study
    ):
        out, series, _ = result
        x = study.base[study.ca]
        radial = np.linalg.norm(x - x.mean(axis=0), axis=1)
        per_replica = np.array([radial * s.std() for s in series])
        assert out["rmsf_mean_angstrom"] == pytest.approx(
            per_replica.mean(axis=0), abs=3e-3
        )
        assert out["rmsf_standard_error_angstrom"] == pytest.approx(
            per_replica.std(axis=0, ddof=1) / 2, abs=3e-3
        )

    def test_the_details_file_holds_blocking_and_cumulative_means(self, result):
        details = json.loads(Path(result[0]["details"]).read_text())
        stats = details["statistics"]
        assert [b["block_count_per_replica"] for b in stats["blocking"]] == [5, 10, 20]
        assert len(stats["cumulative"]) == 10
        assert stats["replica_count"] == 4 and stats["frames_per_replica"] == 80


class TestJudgements:
    def test_a_tolerance_tighter_than_the_offset_does_not_agree(self, study, tmp_path):
        outcome = compare(
            study,
            ["iid"] * 4,
            tmp_path,
            criteria=[{"observable": OBSERVABLE, "precision": 0.05, "tolerance": 0.05}],
        )
        out = ok(outcome, "cmp")
        assert verdict(out)["agrees"] is False
        assert out["agrees_with_reference"] is False

    def test_a_precision_no_run_could_meet_fails_and_has_no_sustained_checkpoint(
        self, study, tmp_path
    ):
        outcome = compare(
            study,
            ["iid"] * 4,
            tmp_path,
            criteria=[{"observable": OBSERVABLE, "precision": 1e-6}],
        )
        out = ok(outcome, "cmp")
        assert out["precision_passed"] is False
        assert out["first_sustained_checkpoint"] is None
        assert "did not meet its precision" in " ".join(outcome.steps["cmp"].notes)
        assert out["agrees_with_reference"] is None

    def test_a_generous_precision_is_sustained_from_the_first_checkpoint(
        self, study, tmp_path
    ):
        out = ok(
            compare(
                study,
                ["iid"] * 4,
                tmp_path,
                criteria=[{"observable": OBSERVABLE, "precision": 5.0}],
            ),
            "cmp",
        )
        assert out["first_sustained_checkpoint"]["frames_per_replica"] == 40

    def test_correlated_samples_have_a_small_effective_size_and_fail_the_limit(
        self, study, tmp_path
    ):
        # AR(1) with coefficient 0.9: N (1 - phi) / (1 + phi) per replica.
        inspections = [study.replica(scales("ar1", i, 400), f"ar{i}") for i in range(4)]
        outcome = run(
            study.steps(
                inspections,
                {
                    "criteria": [{"observable": OBSERVABLE, "precision": 5.0}],
                    "min_effective_samples": 200.0,
                },
            ),
            tmp_path,
        )
        row = verdict(ok(outcome, "cmp"))
        expected = 4 * 400 * (1 - 0.9) / (1 + 0.9)  # about 84
        assert 0.4 * expected < row["effective_samples"] < 2.0 * expected
        assert row["precision_passed"] is False

    def test_replicas_with_separated_means_do_not_converge(self, study, tmp_path):
        out = ok(
            compare(
                study,
                ["shift0.1"] * 4,
                tmp_path,
                criteria=[{"observable": OBSERVABLE, "precision": 5.0}],
            ),
            "cmp",
        )
        row = verdict(out)
        # Means 0.1 apart against a within-replica spread of 0.02 x Rg0.
        assert row["split_rhat"] > 3
        assert row["precision_passed"] is False

    def test_a_replica_that_never_moves_is_reported_not_estimated(
        self, study, tmp_path
    ):
        inspections = [
            study.replica(scales(k, i), f"mix{i}")
            for i, k in enumerate(["iid", "iid", "iid", "constant"])
        ]
        outcome = run(
            study.steps(
                inspections,
                {"criteria": [{"observable": OBSERVABLE, "precision": 5.0}]},
            ),
            tmp_path,
        )
        row = verdict(ok(outcome, "cmp"))
        assert row["diagnostic"] == "zero-within-chain-variance"
        assert row["split_rhat"] is None and row["precision_passed"] is False

    def test_frames_beyond_a_multiple_of_twenty_are_left_out_and_noted(
        self, study, tmp_path
    ):
        inspections = [study.replica(scales("iid", i, 90), f"odd{i}") for i in range(3)]
        outcome = run(
            study.steps(
                inspections,
                {"criteria": [{"observable": OBSERVABLE, "precision": 5.0}]},
            ),
            tmp_path,
        )
        out = ok(outcome, "cmp")
        assert out["frames_per_replica"] == 80 and out["trimmed_frames"] == 10
        assert "left out" in " ".join(outcome.steps["cmp"].notes)

    def test_the_other_limits_and_block_counts_are_used(self, study, tmp_path):
        out = ok(
            compare(
                study,
                ["iid"] * 3,
                tmp_path,
                criteria=[{"observable": OBSERVABLE, "precision": 5.0}],
                max_split_rhat=1.001,
                min_effective_samples=5000.0,
                block_counts=[4, 8],
            ),
            "cmp",
        )
        details = json.loads(Path(out["details"]).read_text())
        assert [
            b["block_count_per_replica"] for b in details["statistics"]["blocking"]
        ] == [4, 8]
        assert details["limits"]["max_split_rhat"] == 1.001
        assert details["limits"]["min_effective_samples"] == 5000.0
        assert verdict(out)["precision_passed"] is False

    def test_block_counts_that_do_not_divide_the_frames_are_refused(
        self, study, tmp_path
    ):
        outcome = compare(
            study,
            ["iid"] * 3,
            tmp_path,
            criteria=[{"observable": OBSERVABLE, "precision": 5.0}],
            block_counts=[7],
        )
        assert not outcome.ok
        assert "do not divide the 80 frames" in outcome.steps["cmp"].error

    def test_a_measured_distance_is_judged_with_its_unit(self, study, tmp_path):
        from tests.core.test_simulation_inspect_trajectory import selector

        _, a = selector(study.top, 3, "CA")
        _, b = selector(study.top, 15, "CA")
        observable = {"id": "span", "kind": "distance", "atoms": [a, b]}
        inspections = [
            {
                **study.replica(scales("iid", i), f"dist{i}"),
                "observables": [observable],
            }
            for i in range(4)
        ]
        out = ok(
            run(
                study.steps(
                    inspections,
                    {
                        "criteria": [
                            {"observable": "span", "precision": 0.5, "tolerance": 3.0}
                        ]
                    },
                ),
                tmp_path,
            ),
            "cmp",
        )
        x = study.base
        i, _ = selector(study.top, 3, "CA")
        j, _ = selector(study.top, 15, "CA")
        d0 = float(np.linalg.norm(x[i] - x[j]))
        row = verdict(out, "span")
        assert row["unit"] == "angstrom"
        assert row["mean"] == pytest.approx(d0, rel=0.02)
        assert row["reference_value"] == pytest.approx(REFERENCE_SCALE * d0, rel=1e-3)


class TestRefusals:
    def test_one_replica_cannot_show_convergence(self, study, tmp_path):
        inspections = [study.replica(scales("iid", 0), "single")]
        outcome = run(
            study.steps(
                inspections,
                {"criteria": [{"observable": OBSERVABLE, "precision": 0.1}]},
            ),
            tmp_path,
        )
        assert not outcome.ok
        assert "2–30 replicas are required" in outcome.steps["cmp"].error

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            (
                {"criteria": [{"observable": OBSERVABLE, "precision": -1.0}]},
                "precision must be positive",
            ),
            (
                {
                    "criteria": [
                        {"observable": OBSERVABLE, "precision": 1.0, "tolerance": 0.0}
                    ]
                },
                "tolerance must be positive",
            ),
            (
                {
                    "criteria": [
                        {"observable": OBSERVABLE, "precision": 1.0},
                        {"observable": OBSERVABLE, "precision": 2.0},
                    ]
                },
                "only one criterion",
            ),
            ({"criteria": []}, "Supply 1–100 criteria"),
            ({"max_split_rhat": 2.0}, "max_split_rhat must be between"),
            ({"min_effective_samples": 5.0}, "min_effective_samples must be between"),
            ({"block_counts": [1]}, "block counts between 2 and 100"),
            ({"block_counts": [5, 5]}, "block counts between 2 and 100"),
            (
                {"criteria": [{"observable": "nope", "precision": 1.0}]},
                "do not record the observables",
            ),
        ],
    )
    def test_criteria_that_cannot_be_judged_are_refused_in_plain_words(
        self, study, tmp_path, changes, message
    ):
        inspections = [study.replica(scales("iid", i), f"ref{i}") for i in range(3)]
        params = {"criteria": [{"observable": OBSERVABLE, "precision": 1.0}], **changes}
        outcome = run(study.steps(inspections, params), tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["cmp"].error

    def test_replicas_that_share_a_seed_are_refused(self, study, tmp_path):
        inspections = [study.replica(scales("iid", i), f"seed{i}") for i in range(3)]
        steps = study.steps(
            inspections, {"criteria": [{"observable": OBSERVABLE, "precision": 1.0}]}
        )
        for replica in steps[-1]["params"]["replicas"]:
            replica["seed"] = 7
        outcome = run(steps, tmp_path)
        assert not outcome.ok
        assert "distinct seeds" in outcome.steps["cmp"].error

    def test_a_seed_for_some_replicas_only_is_refused(self, study, tmp_path):
        inspections = [study.replica(scales("iid", i), f"half{i}") for i in range(3)]
        steps = study.steps(
            inspections, {"criteria": [{"observable": OBSERVABLE, "precision": 1.0}]}
        )
        del steps[-1]["params"]["replicas"][1]["seed"]
        outcome = run(steps, tmp_path)
        assert not outcome.ok
        assert "for every replica, or for none" in outcome.steps["cmp"].error

    def test_two_replicas_with_identical_samples_are_refused(self, study, tmp_path):
        same = study.replica(scales("iid", 1), "same")
        inspections = [same, same, study.replica(scales("iid", 2), "other")]
        outcome = run(
            study.steps(
                inspections,
                {"criteria": [{"observable": OBSERVABLE, "precision": 1.0}]},
            ),
            tmp_path,
        )
        assert not outcome.ok
        assert "identical samples" in outcome.steps["cmp"].error

    def test_replicas_of_different_lengths_are_refused(self, study, tmp_path):
        inspections = [
            study.replica(scales("iid", 0, 80), "len0"),
            study.replica(scales("iid", 1, 60), "len1"),
        ]
        outcome = run(
            study.steps(
                inspections,
                {"criteria": [{"observable": OBSERVABLE, "precision": 1.0}]},
            ),
            tmp_path,
        )
        assert not outcome.ok
        assert "equal numbers of frames" in outcome.steps["cmp"].error

    def test_replicas_that_are_too_short_are_refused(self, study, tmp_path):
        inspections = [
            study.replica(scales("iid", i, 30), f"short{i}") for i in range(2)
        ]
        outcome = run(
            study.steps(
                inspections,
                {"criteria": [{"observable": OBSERVABLE, "precision": 1.0}]},
            ),
            tmp_path,
        )
        assert not outcome.ok
        assert "at least 40 frames" in outcome.steps["cmp"].error

    def test_replicas_inspected_with_different_observables_are_refused(
        self, study, tmp_path
    ):
        from tests.core.test_simulation_inspect_trajectory import selector

        _, a = selector(study.top, 3, "CA")
        _, b = selector(study.top, 15, "CA")
        inspections = [
            study.replica(scales("iid", 0), "def0"),
            {
                **study.replica(scales("iid", 1), "def1"),
                "observables": [{"id": "span", "kind": "distance", "atoms": [a, b]}],
            },
        ]
        outcome = run(
            study.steps(
                inspections,
                {"criteria": [{"observable": OBSERVABLE, "precision": 1.0}]},
            ),
            tmp_path,
        )
        assert not outcome.ok
        assert (
            "different protein atoms or observable definitions"
            in outcome.steps["cmp"].error
        )

    def test_a_tolerance_needs_every_inspection_fitted_to_the_same_reference(
        self, study, tmp_path
    ):
        inspections = [study.replica(scales("iid", i), f"noref{i}") for i in range(2)]
        steps = study.steps(
            inspections,
            {
                "criteria": [
                    {"observable": OBSERVABLE, "precision": 1.0, "tolerance": 1.0}
                ]
            },
        )
        del steps[1]["params"]["reference_structure"]
        outcome = run(steps, tmp_path)
        assert not outcome.ok
        assert "same supplied reference_structure" in outcome.steps["cmp"].error

    def test_a_periodic_trajectory_inspected_as_stored_is_refused(
        self, study, tmp_path
    ):
        inspections = [
            study.replica(scales("iid", i), f"box{i}", volume=True) for i in range(2)
        ]
        outcome = run(
            study.steps(
                inspections,
                {"criteria": [{"observable": OBSERVABLE, "precision": 1.0}]},
            ),
            tmp_path,
        )
        assert not outcome.ok
        assert "imaging whole-molecules" in outcome.steps["cmp"].error
