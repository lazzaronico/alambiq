"""simulation.analyze_binding_free_energy against legs whose arithmetic can be done by hand.

The legs are written as the records `simulation.run_relative_free_energy` produces, with
numbers chosen so each result follows by arithmetic:

* two replicas at x - d and x + d have mean x, standard deviation d sqrt(2) and
  standard error d; the edge's bound-minus-solvent difference is the difference of the
  two means and its standard error the hypotenuse of the two legs';
* MBAR errors add in quadrature over replicas and divide by their number;
* the closure of a cycle is the signed sum of its edges, and its errors add in quadrature;
* the reference's edge values are the signed differences of the ligand values, and a
  cycle through them closes to zero when the values are consistent;
* the overlap of neighbouring states is the first off-diagonal of a row-stochastic
  matrix, and a half-data forward-reverse difference is read at the 0.5 fraction.

No engine is involved: this is the analysis of results, and it has no environment.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from alambiq.api import run as execute  # noqa: E402

METHOD = hashlib.sha256(b"method").hexdigest()
PROTEIN = hashlib.sha256(b"protein").hexdigest()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def overlap(k: int = 4, adjacent: float = 0.3) -> list[list[float]]:
    """A row-stochastic matrix whose first off-diagonals are all `adjacent`."""
    matrix = np.zeros((k, k))
    for i in range(k):
        for j in (i - 1, i + 1):
            if 0 <= j < k:
                matrix[i, j] = adjacent
        matrix[i, i] = 1 - matrix[i].sum()
    return matrix.tolist()


def replica(name, value, error=0.2, k=4, adjacent=0.3, ess=500.0, forward_reverse=None):
    return {
        "id": name,
        "trajectory_sha256": sha("trajectory-" + name),
        "delta_g_kcal_mol": value,
        "mbar_error_kcal_mol": error,
        "overlap_matrix": overlap(k, adjacent),
        "effective_samples": [ess] * k,
        "forward_reverse": forward_reverse,
    }


def fractional(final, half_forward, half_reverse):
    """Forward and reverse estimates at 25, 50 and 100 per cent of the data."""
    return {
        "fractions": [0.25, 0.5, 1.0],
        "forward_kcal_mol": [final + 1.0, half_forward, final],
        "reverse_kcal_mol": [final - 1.0, half_reverse, final],
        "forward_error_kcal_mol": [0.5, 0.3, 0.1],
        "reverse_error_kcal_mol": [0.5, 0.3, 0.1],
    }


def leg(
    edge,
    phase,
    replicas,
    mapping="map",
    ligands=None,
    method=METHOD,
    temperature=298.15,
):
    return {
        "schema_version": "0.1",
        "edge": edge,
        "phase": phase,
        "temperature_kelvin": temperature,
        "method_sha256": method,
        "mapping_sha256": sha(mapping + edge["id"]),
        "ligand_sha256": ligands
        or {"a": sha("ligand-" + edge["a"]), "b": sha("ligand-" + edge["b"])},
        "protein_sha256": PROTEIN if phase == "bound" else None,
        "replicas": replicas,
    }


EDGES = [
    {"id": "ab", "a": "A", "b": "B"},
    {"id": "bc", "a": "B", "b": "C"},
    {"id": "ac", "a": "A", "b": "C"},
]
# (bound mean, bound half-width, solvent mean, solvent half-width) per edge.
LEGS = {
    "ab": (-3.1, 0.1, -1.1, 0.05),
    "bc": (-2.0, 0.2, -0.5, 0.05),
    "ac": (-5.0, 0.1, -1.7, 0.05),
}
# Differences: ab -2.0, bc -1.5, ac -3.3; the cycle ab + bc - ac closes to -0.2.


def write_legs(directory: Path, changes=None) -> list[str]:
    changes = changes or {}
    paths = []
    for edge in EDGES:
        bound, bd, solvent, sd = LEGS[edge["id"]]
        for phase, mean, d in (("bound", bound, bd), ("solvent", solvent, sd)):
            replicas = [
                replica(
                    f"{edge['id']}-{phase}-{n}",
                    mean + s * d,
                    forward_reverse=fractional(mean + s * d, -1.0, -1.4),
                )
                for n, s in enumerate((-1, 1))
            ]
            record = leg(edge, phase, replicas)
            record.update(changes.get((edge["id"], phase), {}))
            path = directory / f"{edge['id']}-{phase}.json"
            path.write_text(json.dumps(record))
            paths.append(str(path))
    return paths


def request(paths, **changes):
    return {
        "title": "toy series",
        "edges": EDGES,
        "legs": paths,
        "cycles": [{"id": "triangle", "coefficients": {"ab": 1, "bc": 1, "ac": -1}}],
        **changes,
    }


def run(params, tmp_path, name="run"):
    step = {
        "id": "bfe",
        "capability": "simulation.analyze_binding_free_energy",
        "params": params,
    }
    return execute({"id": "bfe", "steps": [step]}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("bfe")


@pytest.fixture
def legs(tmp_path):
    return write_legs(tmp_path)


class TestEdgesAndCycles:
    @pytest.fixture
    def out(self, legs, tmp_path):
        return ok(run(request(legs), tmp_path))

    def edge(self, out, name):
        return next(e for e in out["edges"] if e["id"] == name)

    def test_an_edge_is_the_bound_mean_minus_the_solvent_mean(self, out):
        assert self.edge(out, "ab")["delta_delta_g_kcal_mol"] == pytest.approx(-2.0)
        assert self.edge(out, "bc")["delta_delta_g_kcal_mol"] == pytest.approx(-1.5)
        assert self.edge(out, "ac")["delta_delta_g_kcal_mol"] == pytest.approx(-3.3)

    def test_each_leg_reports_its_mean_spread_and_standard_error(self, out):
        bound = self.edge(out, "ab")["bound"]
        assert bound["replicas"] == 2 and bound["mean_kcal_mol"] == pytest.approx(-3.1)
        assert bound["replica_sd_kcal_mol"] == pytest.approx(0.1 * math.sqrt(2))
        assert bound["replica_sem_kcal_mol"] == pytest.approx(0.1)

    def test_an_edges_standard_error_is_the_hypotenuse_of_its_legs(self, out):
        assert self.edge(out, "ab")["replica_sem_kcal_mol"] == pytest.approx(
            math.hypot(0.1, 0.05)
        )

    def test_mbar_errors_add_in_quadrature_and_divide_by_the_replica_count(self, out):
        leg_error = math.sqrt(2 * 0.2**2) / 2
        assert self.edge(out, "ab")["bound"]["mbar_error_kcal_mol"] == pytest.approx(
            leg_error
        )
        assert self.edge(out, "ab")["mbar_error_kcal_mol"] == pytest.approx(
            math.hypot(leg_error, leg_error)
        )

    def test_the_cycle_closure_is_the_signed_sum_of_its_edges(self, out):
        (cycle,) = out["cycles"]
        assert cycle["closure_kcal_mol"] == pytest.approx(-2.0 - 1.5 + 3.3)
        sems = [math.hypot(0.1, 0.05), math.hypot(0.2, 0.05), math.hypot(0.1, 0.05)]
        assert cycle["replica_sem_kcal_mol"] == pytest.approx(
            math.sqrt(sum(s**2 for s in sems))
        )

    def test_overlap_and_effective_samples_are_the_worst_over_replicas(self, out):
        bound = self.edge(out, "ab")["bound"]
        assert bound["minimum_adjacent_overlap"] == pytest.approx(0.3)
        assert bound["minimum_effective_samples"] == 500.0

    def test_the_half_data_forward_reverse_difference_is_read_at_the_half_fraction(
        self, out
    ):
        assert self.edge(out, "ab")["bound"][
            "largest_half_forward_reverse_difference_kcal_mol"
        ] == pytest.approx(0.4)

    def test_the_direction_and_temperature_are_stated(self, out):
        assert out["temperature_kelvin"] == 298.15
        assert "ΔΔG_bind(B−A) = ΔG_bound(A→B) − ΔG_solvent(A→B)" == out["direction"]
        assert out["status"] is None and out["checks"] == []

    def test_the_details_file_keeps_every_replica_and_identity(self, out):
        details = json.loads(Path(out["details"]).read_text())
        assert len(details["edges"][0]["bound"]["replicas"]) == 2
        assert (
            details["method_sha256"] == METHOD and details["protein_sha256"] == PROTEIN
        )


class TestReference:
    def reference(self, **changes):
        values = {"A": -4.0, "B": -5.5, "C": -7.0}
        return {
            "temperature_kelvin": 298.15,
            "citation": "constructed for this test",
            "license": "test fixture",
            "conditions": "none; made up",
            "error_description": "invented standard deviations",
            "observations": [
                {"ligand": k, "delta_g_kcal_mol": v, "reported_error_kcal_mol": 0.2}
                for k, v in values.items()
            ],
            **changes,
        }

    def test_reference_edges_are_signed_differences_of_the_ligand_values(
        self, legs, tmp_path
    ):
        out = ok(run(request(legs, reference=self.reference()), tmp_path))
        ref = {e["id"]: e["reference_delta_delta_g_kcal_mol"] for e in out["edges"]}
        assert ref == {
            "ab": pytest.approx(-1.5),
            "bc": pytest.approx(-1.5),
            "ac": pytest.approx(-3.0),
        }

    def test_a_consistent_reference_closes_its_cycle_to_zero_with_propagated_error(
        self, legs, tmp_path
    ):
        out = ok(run(request(legs, reference=self.reference()), tmp_path))
        (cycle,) = out["cycles"]
        assert cycle["reference_closure_kcal_mol"] == pytest.approx(0.0, abs=1e-12)
        # ab + bc - ac = (B-A) + (C-B) - (C-A): every ligand cancels, errors as well.
        assert cycle["reference_propagated_error_kcal_mol"] == pytest.approx(
            0.0, abs=1e-12
        )

    def test_a_reference_at_another_temperature_is_refused(self, legs, tmp_path):
        outcome = run(
            request(legs, reference=self.reference(temperature_kelvin=310.0)), tmp_path
        )
        assert not outcome.ok
        assert "matched temperature" in outcome.steps["bfe"].error

    def test_a_reference_missing_a_ligand_is_refused(self, legs, tmp_path):
        short = self.reference()
        short["observations"] = short["observations"][:2]
        outcome = run(request(legs, reference=short), tmp_path)
        assert "one observation per series ligand" in outcome.steps["bfe"].error

    def test_a_reference_naming_an_unknown_ligand_is_refused(self, legs, tmp_path):
        wrong = self.reference()
        wrong["observations"][2]["ligand"] = "Z"
        outcome = run(request(legs, reference=wrong), tmp_path)
        assert "must match the series exactly" in outcome.steps["bfe"].error


class TestChecks:
    def checks(self, legs, tmp_path, **checks):
        return ok(run(request(legs, checks=checks), tmp_path))

    def rows(self, out, diagnostic):
        return [c for c in out["checks"] if c["diagnostic"] == diagnostic]

    def test_edge_standard_errors_pass_or_fail_against_the_limit(self, legs, tmp_path):
        out = self.checks(legs, tmp_path, maximum_edge_replica_sem_kcal_mol=0.15)
        statuses = {
            c["identity"]: c["status"] for c in self.rows(out, "edge_replica_sem")
        }
        # hypot(0.1, 0.05) = 0.112 passes; hypot(0.2, 0.05) = 0.206 fails.
        assert statuses == {"ab": "pass", "bc": "fail", "ac": "pass"}
        assert out["status"] == "fail" and out["sampling_status"] == "fail"

    def test_leg_spread_overlap_and_effective_samples_are_checked_per_leg_and_replica(
        self, legs, tmp_path
    ):
        out = self.checks(
            legs,
            tmp_path,
            maximum_leg_replica_sd_kcal_mol=0.2,
            minimum_adjacent_overlap_each_replica=0.25,
            minimum_effective_samples_each_state=600.0,
        )
        spreads = {c["identity"]: c["status"] for c in self.rows(out, "leg_replica_sd")}
        # SD = d sqrt(2): bc/bound 0.283 fails, the rest are below 0.2.
        assert spreads["bc/bound"] == "fail" and spreads["ab/bound"] == "pass"
        assert all(c["status"] == "pass" for c in self.rows(out, "overlap"))
        assert all(c["status"] == "fail" for c in self.rows(out, "effective_samples"))
        assert len(self.rows(out, "overlap")) == 12

    def test_the_half_data_check_uses_the_stored_fraction_nearest_the_target(
        self, legs, tmp_path
    ):
        out = self.checks(
            legs,
            tmp_path,
            half_fraction_target=0.5,
            half_fraction_maximum_distance=0.1,
            maximum_half_forward_reverse_difference_kcal_mol=0.5,
        )
        rows = self.rows(out, "half_forward_reverse")
        assert rows and all(r["selected_fraction"] == 0.5 for r in rows)
        assert all(r["status"] == "pass" for r in rows)
        tight = self.checks(
            legs,
            tmp_path,
            half_fraction_target=0.5,
            half_fraction_maximum_distance=0.1,
            maximum_half_forward_reverse_difference_kcal_mol=0.3,
        )
        assert all(
            r["status"] == "fail" for r in self.rows(tight, "half_forward_reverse")
        )

    def test_a_target_fraction_the_run_did_not_store_is_unassessed_not_failed(
        self, legs, tmp_path
    ):
        out = self.checks(
            legs,
            tmp_path,
            half_fraction_target=0.9,
            half_fraction_maximum_distance=0.01,
            maximum_half_forward_reverse_difference_kcal_mol=0.5,
        )
        rows = self.rows(out, "half_forward_reverse")
        assert all(
            r["status"] == "unassessed" and "declared target tolerance" in r["reason"]
            for r in rows
        )
        assert out["status"] == "unassessed"

    def test_a_cycle_closes_when_within_the_larger_of_the_floor_and_the_multiple_of_its_error(
        self, legs, tmp_path
    ):
        # Closure 0.2 in magnitude; SEM sqrt(0.0125 + 0.0425 + 0.0125) = 0.26.
        loose = self.checks(
            legs,
            tmp_path,
            cycle_absolute_floor_kcal_mol=0.1,
            cycle_replica_sem_multiplier=1.0,
        )
        (row,) = self.rows(loose, "cycle_closure")
        assert row["status"] == "pass" and row["limit"] == pytest.approx(
            math.sqrt(0.0125 + 0.0425 + 0.0125)
        )
        tight = self.checks(
            legs,
            tmp_path,
            cycle_absolute_floor_kcal_mol=0.1,
            cycle_replica_sem_multiplier=0.1,
        )
        assert self.rows(tight, "cycle_closure")[0]["status"] == "fail"

    def test_no_cycle_to_check_is_unassessed(self, legs, tmp_path):
        outcome = run(
            request(
                legs,
                cycles=[],
                checks={
                    "cycle_absolute_floor_kcal_mol": 0.5,
                    "cycle_replica_sem_multiplier": 2.0,
                },
            ),
            tmp_path,
        )
        (row,) = self.rows(ok(outcome), "cycle_closure")
        assert (
            row["status"] == "unassessed"
            and row["reason"] == "No closed cycle was supplied"
        )

    @pytest.mark.parametrize(
        ("checks", "message"),
        [
            ({}, "named binding checks"),
            (
                {"half_fraction_target": 0.5},
                "complete fractional or cycle check settings",
            ),
            (
                {"cycle_absolute_floor_kcal_mol": 0.5},
                "complete fractional or cycle check settings",
            ),
            (
                {"maximum_edge_replica_sem_kcal_mol": -1.0},
                "finite nonnegative thresholds",
            ),
            (
                {
                    "half_fraction_target": 1.5,
                    "half_fraction_maximum_distance": 0.1,
                    "maximum_half_forward_reverse_difference_kcal_mol": 1.0,
                },
                "probability ranges",
            ),
            ({"minimum_adjacent_overlap_each_replica": 1.5}, "probability ranges"),
        ],
    )
    def test_checks_that_cannot_be_judged_are_refused(
        self, legs, tmp_path, checks, message
    ):
        outcome = run(request(legs, checks=checks), tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["bfe"].error


class TestStructuralDiagnostics:
    def drift_file(self, directory, peak, states=4, frames=10):
        path = directory / f"drift-{peak}.npz"
        drift = np.zeros((states, frames))
        drift[0, 3] = peak
        np.savez(path, time_ps=np.arange(frames, dtype=float), ligand_COM_drift=drift)
        return str(path)

    def test_the_largest_ligand_drift_is_checked_against_the_limit(
        self, legs, tmp_path
    ):
        files = {
            "ab-bound-0": self.drift_file(tmp_path, 1.5),
            "ab-bound-1": self.drift_file(tmp_path, 4.0),
        }
        out = ok(
            run(
                request(
                    legs,
                    structural_diagnostics=files,
                    checks={"maximum_ligand_com_drift_angstrom": 3.0},
                ),
                tmp_path,
            )
        )
        rows = {
            c["identity"]: c
            for c in out["checks"]
            if c["diagnostic"] == "ligand_com_drift"
        }
        assert rows["ab/bound/ab-bound-0"]["status"] == "pass"
        assert rows["ab/bound/ab-bound-1"]["status"] == "fail"
        assert rows["ab/bound/ab-bound-1"]["observed"] == pytest.approx(4.0)
        # A bound replica with no file is not failed, it is not assessed.
        missing = rows["bc/bound/bc-bound-0"]
        assert (
            missing["status"] == "unassessed"
            and "was not supplied" in missing["reason"]
        )
        assert out["structural_review_status"] == "review-required"

    def test_a_diagnostic_for_a_replica_that_is_not_bound_is_refused(
        self, legs, tmp_path
    ):
        outcome = run(
            request(
                legs,
                structural_diagnostics={"ab-solvent-0": self.drift_file(tmp_path, 1.0)},
            ),
            tmp_path,
        )
        assert not outcome.ok
        assert "not bound replicas" in outcome.steps["bfe"].error

    def test_a_diagnostic_with_the_wrong_number_of_states_is_refused(
        self, legs, tmp_path
    ):
        outcome = run(
            request(
                legs,
                structural_diagnostics={
                    "ab-bound-0": self.drift_file(tmp_path, 1.0, states=3)
                },
            ),
            tmp_path,
        )
        assert not outcome.ok
        assert "nonnegative finite state-by-time" in outcome.steps["bfe"].error


class TestRefusals:
    def fails(self, params, tmp_path, message):
        outcome = run(params, tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["bfe"].error

    def test_a_missing_leg_is_refused(self, legs, tmp_path):
        self.fails(
            request(legs[:-1]), tmp_path, "exactly one bound and one solvent record"
        )

    def test_a_repeated_leg_is_refused(self, legs, tmp_path):
        self.fails(
            request([*legs[:-1], legs[0]]),
            tmp_path,
            "exactly one bound and one solvent record",
        )

    def test_legs_run_with_another_method_are_refused(self, tmp_path):
        paths = write_legs(
            tmp_path, {("ab", "solvent"): {"method_sha256": sha("other")}}
        )
        self.fails(request(paths), tmp_path, "same exact method and temperature")

    def test_legs_at_another_temperature_are_refused(self, tmp_path):
        paths = write_legs(tmp_path, {("ab", "solvent"): {"temperature_kelvin": 300.0}})
        self.fails(request(paths), tmp_path, "same exact method and temperature")

    def test_legs_of_one_edge_with_different_atom_mappings_are_refused(self, tmp_path):
        paths = write_legs(
            tmp_path, {("ab", "solvent"): {"mapping_sha256": sha("different")}}
        )
        self.fails(request(paths), tmp_path, "identical atom mapping")

    def test_bound_legs_with_different_proteins_are_refused(self, tmp_path):
        paths = write_legs(
            tmp_path, {("bc", "bound"): {"protein_sha256": sha("another protein")}}
        )
        self.fails(request(paths), tmp_path, "same exact protein input")

    def test_a_ligand_that_changes_between_legs_is_refused(self, tmp_path):
        changed = {"a": sha("changed"), "b": sha("ligand-C")}
        paths = write_legs(tmp_path, {("bc", "bound"): {"ligand_sha256": changed}})
        self.fails(request(paths), tmp_path, "same exact molecular input")

    def test_a_solvent_leg_that_names_a_protein_is_refused(self, tmp_path):
        paths = write_legs(tmp_path, {("ab", "solvent"): {"protein_sha256": PROTEIN}})
        self.fails(request(paths), tmp_path, "protein identity must be null")

    def test_one_replica_is_not_a_leg(self, tmp_path):
        paths = write_legs(tmp_path)
        record = json.loads(Path(paths[0]).read_text())
        record["replicas"] = record["replicas"][:1]
        Path(paths[0]).write_text(json.dumps(record))
        self.fails(request(paths), tmp_path, "at least two independent replicas")

    def test_two_replicas_sharing_a_trajectory_are_refused(self, tmp_path):
        paths = write_legs(tmp_path)
        record = json.loads(Path(paths[0]).read_text())
        record["replicas"][1]["trajectory_sha256"] = record["replicas"][0][
            "trajectory_sha256"
        ]
        Path(paths[0]).write_text(json.dumps(record))
        self.fails(
            request(paths), tmp_path, "independent replica and trajectory identities"
        )

    def test_an_overlap_matrix_that_is_not_row_stochastic_is_refused(self, tmp_path):
        paths = write_legs(tmp_path)
        record = json.loads(Path(paths[0]).read_text())
        record["replicas"][0]["overlap_matrix"][0][0] += 0.2
        Path(paths[0]).write_text(json.dumps(record))
        self.fails(request(paths), tmp_path, "square row-stochastic matrix")

    def test_a_full_fraction_estimate_that_differs_from_the_replica_estimate_is_refused(
        self, tmp_path
    ):
        paths = write_legs(tmp_path)
        record = json.loads(Path(paths[0]).read_text())
        record["replicas"][0]["forward_reverse"]["forward_kcal_mol"][-1] += 0.5
        Path(paths[0]).write_text(json.dumps(record))
        self.fails(request(paths), tmp_path, "must reproduce the replica estimate")

    def test_a_non_finite_energy_is_refused(self, tmp_path):
        paths = write_legs(tmp_path)
        record = json.loads(Path(paths[0]).read_text())
        record["replicas"][0]["delta_g_kcal_mol"] = float("nan")
        Path(paths[0]).write_text(json.dumps(record))
        self.fails(
            request(paths), tmp_path, "Replica free energy requires a finite number"
        )

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"edges": []}, "Supply 1–100 directed ligand edges"),
            (
                {"edges": [EDGES[0], {"id": "ab2", "a": "B", "b": "A"}, *EDGES[1:]]},
                "distinct undirected ligand pairs",
            ),
            (
                {"edges": [EDGES[0], {"id": "ab", "a": "A", "b": "D"}, *EDGES[2:]]},
                "distinct IDs",
            ),
            (
                {"cycles": [{"id": "open", "coefficients": {"ab": 1, "bc": 1}}]},
                "closed over ligand identities",
            ),
            (
                {
                    "cycles": [
                        {"id": "weighted", "coefficients": {"ab": 2, "bc": 1, "ac": -1}}
                    ]
                },
                "signed edge coefficients",
            ),
            (
                {"cycles": [{"id": "ghost", "coefficients": {"zz": 1}}]},
                "signed edge coefficients",
            ),
            (
                {
                    "cycles": [
                        {"id": "t", "coefficients": {"ab": 1, "bc": 1, "ac": -1}},
                        {"id": "t", "coefficients": {"ab": 1, "bc": 1, "ac": -1}},
                    ]
                },
                "distinct IDs",
            ),
        ],
    )
    def test_a_series_that_cannot_be_analysed_is_refused_in_plain_words(
        self, legs, tmp_path, changes, message
    ):
        self.fails(request(legs, **changes), tmp_path, message)
