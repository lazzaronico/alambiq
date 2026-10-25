"""reaction.analyze_enzyme_panel on a synthetic panel with known structure, and a real one.

The synthetic panel has eight enzymes whose sequences differ by how many alanines
are swapped for valines, and two reactions, an ester hydrolysis and the same with
an ethyl group. An observation's value is the enzyme index times (reaction + 1), so
every answer can be worked out by hand.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("sklearn")

from tests.core.test_reaction_inspect import hydrolysis  # noqa: E402

METHODS = {
    "training-mean",
    "nearest-reference",
    "sequence-ridge",
    "reaction-ridge",
    "joint-ridge",
}
EXAMPLE = Path(__file__).resolve().parents[1] / "fixtures/enzyme-reaction-panel"


def fixture() -> tuple[dict, dict]:
    reactions = []
    for i, smiles in enumerate(("[CH3:5]", "[CH2:5][CH3:7]")):
        r = {**hydrolysis(), "convention": "explicit-species"}
        r["id"] = f"r{i}"
        if i:
            for p in r["participants"]:
                p["smiles"] = p["smiles"].replace("[CH3:5]", smiles)
        reactions.append(
            {"reaction": r, "feature_participants": ["ester", "acid", "alcohol"]}
        )
    enzymes, observations = [], []
    for i in range(8):
        sequence = "M" + "A" * (12 - i) + "V" * i
        enzymes.append(
            {
                "id": f"e{i}",
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "form": "Synthetic test sequence",
                "source_id": "s",
                "source_locator": str(i),
            }
        )
        for j in range(2):
            observations.append(
                {
                    "id": f"o{i}-{j}",
                    "enzyme_id": f"e{i}",
                    "reaction_id": f"r{j}",
                    "condition_id": "c",
                    "status": "measured",
                    "value": float(i * (j + 1)),
                    "reported_sd": 0.2,
                    "independent_id": None,
                    "technical_id": None,
                    "source_id": "s",
                    "source_locator": f"row-{i}-{j}",
                    "note": "Synthetic control",
                }
            )
    panel = {
        "schema_version": "0.1",
        "id": "synthetic",
        "title": "Synthetic paired assay",
        "endpoint": {
            "metric": "response",
            "unit": "units",
            "description": "Synthetic values",
        },
        "sources": [
            {
                "id": "s",
                "citation": "Alambiq synthetic fixture",
                "license": "Apache-2.0",
            }
        ],
        "enzymes": enzymes,
        "reactions": reactions,
        "conditions": {"c": {"description": "Synthetic fixed condition"}},
        "observations": observations,
    }
    params = {
        "alpha": 0.01,
        "candidate_ids": ["e7"],
        "splits": [
            {
                "id": "held-sequences",
                "novelty": "sequence",
                "train": [r["id"] for r in observations[:12]],
                "evaluation": [r["id"] for r in observations[12:]],
            }
        ],
    }
    return panel, params


def run(panel: dict, params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    (tmp_path / f"{name}-panel.json").write_text(json.dumps(panel))
    step = {
        "id": "panel",
        "capability": "reaction.analyze_enzyme_panel",
        "params": {"panel": str(tmp_path / f"{name}-panel.json"), **params},
    }
    return execute({"id": "p", "steps": [step]}, run_dir=tmp_path / name)


def analyze(panel: dict, params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(panel, params, tmp_path, name)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def record(outcome) -> dict:
    return json.loads(Path(outcome.outputs("panel")["record"]).read_text())


def split_by_reaction(panel: dict, params: dict) -> None:
    params["splits"][0].update(
        novelty="reaction",
        train=[r["id"] for r in panel["observations"] if r["reaction_id"] == "r0"],
        evaluation=[r["id"] for r in panel["observations"] if r["reaction_id"] == "r1"],
    )


class TestSyntheticPanel:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        panel, params = fixture()
        return analyze(panel, params, tmp_path_factory.mktemp("panel"))

    def test_the_counts_the_splits_and_the_predictors_are_what_the_panel_implies(
        self, outcome
    ):
        out = outcome.outputs("panel")
        assert out["observation_counts"] == {"measured": 16}
        [split] = out["splits"]
        assert (split["training_groups"], split["evaluation_groups"]) == (12, 4)
        assert set(split["metrics"]) == METHODS
        assert split["metrics"]["joint-ridge"]["n"] == 4
        assert out["endpoint"]["metric"] == "response"

    def test_the_candidate_export_carries_the_named_enzyme_as_a_pool_candidate(
        self, outcome
    ):
        panel, _ = fixture()
        exported = json.loads(
            Path(outcome.outputs("panel")["candidate_panel"]).read_text()
        )
        [candidate] = exported["candidates"]
        assert candidate["sample_id"] == "e7" and candidate["partition"] == "pool"
        assert candidate["sequence"] == panel["enzymes"][-1]["sequence"]
        assert Path(outcome.outputs("panel")["candidate_fasta"]).read_text() == (
            f">e7\n{candidate['sequence']}\n"
        )

    def test_the_record_keeps_every_group_and_reaction_and_the_library_versions(
        self, outcome
    ):
        stored = record(outcome)
        assert len(stored["groups"]) == 16
        assert len(stored["reactions"]) == 2
        assert set(stored["versions"]) == {"rdkit", "numpy", "scipy", "scikit-learn"}

    def test_the_candidate_panel_feeds_candidate_select(self, tmp_path):
        from alambiq.api import run as execute

        panel, params = fixture()
        (tmp_path / "panel.json").write_text(json.dumps(panel))
        outcome = execute(
            {
                "id": "chain",
                "steps": [
                    {
                        "id": "panel",
                        "capability": "reaction.analyze_enzyme_panel",
                        "params": {"panel": str(tmp_path / "panel.json"), **params},
                    },
                    {
                        "id": "choose",
                        "capability": "candidate.select",
                        "bind": {"panel": "panel.candidate_panel"},
                        "params": {"count": 1, "seed": 0, "strategy": "random"},
                    },
                ],
            },
            run_dir=tmp_path / "chain",
        )
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        assert [c["sample_id"] for c in outcome.outputs("choose")["selected"]] == ["e7"]

    def test_the_report_has_a_table_per_split(self, outcome, tmp_path):
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "held-sequences" in page and "joint-ridge" in page


class TestWhatTheEvaluationMayUse:
    def test_changing_the_held_out_values_never_changes_a_fit_or_a_prediction(
        self, tmp_path
    ):
        panel, params = fixture()
        first = record(analyze(panel, params, tmp_path, "a"))
        for row in panel["observations"][12:]:
            row["value"] += 1000
        second_outcome = analyze(panel, params, tmp_path, "b")
        second = record(second_outcome)
        assert first["evaluations"][0]["models"] == second["evaluations"][0]["models"]
        assert [r["predictions"] for r in first["evaluations"][0]["rows"]] == [
            r["predictions"] for r in second["evaluations"][0]["rows"]
        ]
        assert (
            second_outcome.outputs("panel")["splits"][0]["metrics"]["joint-ridge"][
                "mae"
            ]
            > 900
        )

    def test_the_nearest_reference_is_always_a_training_group(self, tmp_path):
        panel, params = fixture()
        evaluation = record(analyze(panel, params, tmp_path))["evaluations"][0]
        training = {g["id"] for g in evaluation["training"]}
        for row in evaluation["rows"]:
            assert row["nearest_reference"]["group_id"] in training
            assert 0 <= row["nearest_reference"]["joint_similarity"] <= 1.000000001

    def test_the_same_panel_gives_the_same_record_however_it_is_ordered(self, tmp_path):
        panel, params = fixture()
        expected = record(analyze(panel, params, tmp_path, "a"))["evaluations"]
        panel["enzymes"].reverse()
        panel["observations"].reverse()
        panel["reactions"].reverse()
        params["splits"][0]["train"].reverse()
        params["splits"][0]["evaluation"].reverse()
        assert record(analyze(panel, params, tmp_path, "b"))["evaluations"] == expected

    def test_the_fitted_coefficients_reconstruct_the_predictions(self, tmp_path):
        import numpy as np

        from alambiq.capabilities.investigate.science.enzyme_panel import (
            pair_kernels,
            validate_panel,
        )

        panel, params = fixture()
        stored = record(analyze(panel, params, tmp_path))
        enzymes, reactions, _, groups = validate_panel(panel)
        kernels, _ = pair_kernels(
            groups, list(enzymes.values()), list(reactions.values())
        )
        kernel = kernels["joint-ridge"]
        lookup = {g["id"]: i for i, g in enumerate(groups)}
        evaluation = stored["evaluations"][0]
        train = [lookup[g["id"]] for g in evaluation["training"]]
        model = evaluation["models"]["joint-ridge"]
        for row in evaluation["rows"]:
            value = (
                kernel[lookup[row["group_id"]], train]
                @ np.asarray(model["dual_coefficients"])
                + model["training_mean"]
            )
            assert value == pytest.approx(row["predictions"]["joint-ridge"], abs=1e-9)

    def test_a_held_out_reaction_gives_every_reaction_only_prediction_one_value_and_no_rank(
        self, tmp_path
    ):
        panel, params = fixture()
        split_by_reaction(panel, params)
        outcome = analyze(panel, params, tmp_path)
        evaluation = record(outcome)["evaluations"][0]
        assert (
            len({r["predictions"]["reaction-ridge"] for r in evaluation["rows"]}) == 1
        )
        assert (
            outcome.outputs("panel")["splits"][0]["metrics"]["reaction-ridge"][
                "spearman_rho"
            ]
            is None
        )

    def test_values_that_never_vary_have_an_unavailable_rank_not_nan(self, tmp_path):
        panel, params = fixture()
        for row in panel["observations"]:
            row["value"] = 2
        outcome = analyze(panel, params, tmp_path)
        for score in outcome.outputs("panel")["splits"][0]["metrics"].values():
            assert score["mae"] == 0 and score["spearman_rho"] is None
        json.dumps(record(outcome), allow_nan=False)

    def test_a_model_that_does_not_beat_a_lookup_is_called_out(self, tmp_path):
        panel, params = fixture()
        for row in panel["observations"]:
            row["value"] = 2
        outcome = analyze(panel, params, tmp_path)
        # Every predictor is exact on constant data; the first by error is the training mean.
        assert "a lookup baseline" in " ".join(outcome.steps["panel"].notes)


class TestKernels:
    def test_each_kernel_is_symmetric_with_unit_diagonal_and_nonnegative_eigenvalues(
        self,
    ):
        import numpy as np

        from alambiq.capabilities.investigate.science.enzyme_panel import (
            pair_kernels,
            validate_panel,
        )

        panel, _ = fixture()
        enzymes, reactions, _, groups = validate_panel(panel)
        kernels, _ = pair_kernels(
            groups, list(enzymes.values()), list(reactions.values())
        )
        for kernel in kernels.values():
            np.testing.assert_allclose(kernel, kernel.T)
            np.testing.assert_allclose(np.diag(kernel), 1)
            assert float(np.linalg.eigvalsh(kernel).min()) >= -1e-10
        np.testing.assert_allclose(
            kernels["joint-ridge"],
            kernels["reaction-ridge"] * kernels["sequence-ridge"],
        )

    def test_exact_sequence_counts_match_a_sparse_integer_reference_at_the_length_limit(
        self,
    ):
        from alambiq.capabilities.investigate.science.enzyme_panel import (
            pair_kernels,
            validate_panel,
        )

        panel, _ = fixture()
        for enzyme, sequence in zip(
            panel["enzymes"], ("A" * 10000, "AVA" * 3333), strict=False
        ):
            enzyme.update(
                sequence=sequence,
                sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
            )
        enzymes, reactions, _, groups = validate_panel(panel)
        _, exact = pair_kernels(
            groups, list(enzymes.values()), list(reactions.values())
        )
        vectors = [
            Counter(s[i : i + 3] for i in range(len(s) - 2))
            for s in (enzymes[g["enzyme_ids"][0]]["sequence"] for g in groups)
        ]
        for i, first in enumerate(vectors):
            assert int(exact["sequence_norm2"][i]) == sum(n * n for n in first.values())
            for j, second in enumerate(vectors):
                assert int(exact["sequence_dot"][i, j]) == sum(
                    n * second.get(k, 0) for k, n in first.items()
                )

    def test_nearest_reference_breaks_exact_ties_by_order_and_sees_tiny_real_differences(
        self,
    ):
        import numpy as np

        from alambiq.capabilities.investigate.science.enzyme_panel import (
            nearest_reference,
        )

        counts = {
            "sequence_dot": np.array([[3, 6, 3]], dtype=np.int64),
            "sequence_norm2": np.array([9, 16, 9], dtype=np.int64),
            "reaction_intersection": np.array([[2, 4, 2]], dtype=np.int64),
            "reaction_union": np.array([[4, 12, 4]], dtype=np.int64),
        }
        # Squared cosine × Tanimoto: the first two scores are algebraically equal.
        assert nearest_reference(0, [0, 1, 2], counts) == 0
        assert nearest_reference(0, [2, 1, 0], counts) == 0
        counts["sequence_dot"][:] = 99_960_004
        counts["sequence_norm2"][:] = 99_960_004
        counts["reaction_intersection"][:] = 1023
        counts["reaction_union"][:] = 1024
        counts["sequence_dot"][0, 1] -= 1
        counts["sequence_norm2"][1] -= 2
        assert nearest_reference(0, [0, 1, 2], counts) == 1
        counts["sequence_dot"][:] = 0
        assert nearest_reference(0, [2, 0, 1], counts) == 0


class TestStatesAndReplicates:
    def test_inactive_failed_untested_and_unreported_are_counted_and_never_fitted(
        self, tmp_path
    ):
        panel, params = fixture()
        for row, status in zip(
            panel["observations"][8:12],
            ("inactive", "preparation-failed", "untested", "unreported"),
            strict=True,
        ):
            row.update(status=status, value=None, reported_sd=None)
        outcome = analyze(panel, params, tmp_path)
        counts = outcome.outputs("panel")["observation_counts"]
        assert counts == {
            "measured": 12,
            "inactive": 1,
            "preparation-failed": 1,
            "untested": 1,
            "unreported": 1,
        }
        assert len(record(outcome)["evaluations"][0]["training"]) == 8

    def test_a_value_on_an_observation_that_was_not_measured_is_refused(self, tmp_path):
        panel, params = fixture()
        panel["observations"][8].update(status="inactive", value=0, reported_sd=None)
        assert (
            "finite response belongs"
            in run(panel, params, tmp_path).steps["panel"].error
        )

    def test_nested_replicates_average_independent_groups_equally(self, tmp_path):
        panel, params = fixture()
        base = panel["observations"][0]
        base.update(independent_id="a", technical_id="one", value=2.0)
        for name, independent, value in (("two", "a", 4.0), ("three", "b", 9.0)):
            panel["observations"].append(
                {
                    **base,
                    "id": name,
                    "technical_id": name,
                    "independent_id": independent,
                    "value": value,
                }
            )
            params["splits"][0]["train"].append(name)
        stored = record(analyze(panel, params, tmp_path))
        group = next(g for g in stored["groups"] if base["id"] in g["observation_ids"])
        # Technical replicates 2 and 4 average to 3, then the independent values 3 and 9 average to 6.
        assert (group["value"], group["independent_n"], group["standard_error"]) == (
            6.0,
            2,
            3.0,
        )

    def test_replicates_of_one_pair_cannot_straddle_train_and_evaluation(
        self, tmp_path
    ):
        panel, params = fixture()
        base = panel["observations"][0]
        base.update(independent_id="a", technical_id="one")
        for name, independent in (("two", "a"), ("three", "b")):
            panel["observations"].append(
                {
                    **base,
                    "id": name,
                    "technical_id": name,
                    "independent_id": independent,
                }
            )
        params["splits"][0]["train"] += ["two"]
        params["splits"][0]["evaluation"].append("three")
        assert (
            "replicate" in run(panel, params, tmp_path).steps["panel"].error
            or "pair" in run(panel, params, tmp_path, "b").steps["panel"].error
        )

    def test_a_failed_independent_replicate_stays_attached_to_its_measured_partner(
        self, tmp_path
    ):
        panel, params = fixture()
        panel["observations"][0].update(independent_id="rep1", technical_id=None)
        panel["observations"].append(
            {
                **panel["observations"][0],
                "id": "failed",
                "independent_id": "rep2",
                "status": "preparation-failed",
                "value": None,
                "reported_sd": None,
            }
        )
        params["splits"][0]["train"].append("failed")
        stored = record(analyze(panel, params, tmp_path))
        group = next(g for g in stored["groups"] if "failed" in g["observation_ids"])
        assert group["observation_state_counts"] == {
            "measured": 1,
            "preparation-failed": 1,
        }
        assert (group["value"], group["independent_n"]) == (0, 1)

    def test_an_unassigned_duplicate_aggregate_is_refused(self, tmp_path):
        panel, params = fixture()
        panel["observations"].append({**panel["observations"][0], "id": "duplicate"})
        assert "replicate" in run(panel, params, tmp_path).steps["panel"].error

    def test_conflicting_states_for_one_pair_are_refused(self, tmp_path):
        panel, params = fixture()
        panel["observations"].append(
            {
                **panel["observations"][0],
                "id": "conflict",
                "status": "inactive",
                "value": None,
                "reported_sd": None,
            }
        )
        assert "conflicting" in run(panel, params, tmp_path).steps["panel"].error


class TestNoveltyClaims:
    def test_a_sequence_that_reappears_under_another_name_breaks_a_sequence_split(
        self, tmp_path
    ):
        panel, params = fixture()
        panel["enzymes"][-1].update(
            {k: panel["enzymes"][0][k] for k in ("sequence", "sequence_sha256")}
        )
        panel["enzymes"][-1]["form"] = "Different purification tag context"
        assert "overlap" in run(panel, params, tmp_path).steps["panel"].error

    def test_a_reaction_split_holds_while_the_reactions_differ_and_breaks_when_they_do_not(
        self, tmp_path
    ):
        panel, params = fixture()
        split_by_reaction(panel, params)
        assert (
            analyze(panel, params, tmp_path).outputs("panel")["splits"][0]["metrics"][
                "joint-ridge"
            ]["n"]
            == 8
        )
        panel["reactions"][1]["reaction"]["participants"] = copy.deepcopy(
            panel["reactions"][0]["reaction"]["participants"]
        )
        error = run(panel, params, tmp_path, "b").steps["panel"].error
        assert "overlap" in error or "pair" in error

    def test_the_same_equation_scaled_is_still_the_same_held_out_chemistry(
        self, tmp_path
    ):
        panel, params = fixture()
        split_by_reaction(panel, params)
        panel["reactions"][1]["reaction"]["participants"] = copy.deepcopy(
            panel["reactions"][0]["reaction"]["participants"]
        )
        for participant in panel["reactions"][1]["reaction"]["participants"]:
            participant["coefficient"] = 2
        assert "overlap" in run(panel, params, tmp_path).steps["panel"].error

    def test_a_both_claim_needs_both_sequence_and_reaction_to_be_new(self, tmp_path):
        panel, params = fixture()
        params["splits"][0]["novelty"] = "both"
        assert "overlap" in run(panel, params, tmp_path).steps["panel"].error


class TestRefusals:
    @pytest.mark.parametrize(
        "field", ["enzyme_id", "source_id", "condition_id", "reaction_id"]
    )
    def test_an_observation_naming_something_the_panel_does_not_have_is_refused(
        self, tmp_path, field
    ):
        panel, params = fixture()
        panel["observations"][0][field] = "unknown"
        assert (
            f"Unknown observation {field}"
            in run(panel, params, tmp_path).steps["panel"].error
        )

    @pytest.mark.parametrize(
        ("field", "value"), [("value", float("nan")), ("reported_sd", -1)]
    )
    def test_a_nonfinite_value_or_a_negative_spread_is_refused(
        self, tmp_path, field, value
    ):
        panel, params = fixture()
        panel["observations"][0][field] = value
        assert run(panel, params, tmp_path).steps["panel"].status == "failed"

    def test_a_sequence_that_does_not_match_its_digest_is_refused(self, tmp_path):
        panel, params = fixture()
        panel["enzymes"][0]["sequence_sha256"] = "0" * 64
        assert "digest" in run(panel, params, tmp_path).steps["panel"].error

    def test_a_reaction_that_does_not_balance_is_refused(self, tmp_path):
        panel, params = fixture()
        panel["reactions"][0]["reaction"]["participants"].pop(1)
        assert "balance" in run(panel, params, tmp_path).steps["panel"].error

    def test_a_feature_participant_the_reaction_lacks_is_refused(self, tmp_path):
        panel, params = fixture()
        panel["reactions"][0]["feature_participants"] = ["unknown"]
        assert (
            "feature participant" in run(panel, params, tmp_path).steps["panel"].error
        )

    @pytest.mark.parametrize("edit", ["unknown", "overlap", "short"])
    def test_a_split_must_name_known_observations_not_share_them_and_have_enough_targets(
        self, tmp_path, edit
    ):
        panel, params = fixture()
        split = params["splits"][0]
        if edit == "unknown":
            split["train"].append("missing")
        if edit == "overlap":
            split["evaluation"].append("o0-0")
        if edit == "short":
            split["train"] = ["o0-0"]
        assert run(panel, params, tmp_path).steps["panel"].status == "failed"

    @pytest.mark.parametrize(("alpha", "message"), [(0, "alpha"), (1e7, "alpha")])
    def test_a_ridge_penalty_outside_its_range_is_refused(
        self, tmp_path, alpha, message
    ):
        panel, params = fixture()
        params["alpha"] = alpha
        assert message in run(panel, params, tmp_path).steps["panel"].error

    def test_a_candidate_that_is_not_an_enzyme_of_the_panel_is_refused(self, tmp_path):
        panel, params = fixture()
        params["candidate_ids"] = ["e99"]
        assert "candidate_ids" in run(panel, params, tmp_path).steps["panel"].error

    def test_a_panel_with_a_repeated_json_field_is_refused(self, tmp_path):
        panel, params = fixture()
        text = json.dumps(panel).replace(
            '"id": "synthetic"', '"id": "synthetic", "id": "again"', 1
        )
        path = tmp_path / "dup.json"
        path.write_text(text)
        from alambiq.api import run as execute

        step = {
            "id": "panel",
            "capability": "reaction.analyze_enzyme_panel",
            "params": {"panel": str(path), **params},
        }
        outcome = execute({"id": "p", "steps": [step]}, run_dir=tmp_path / "dup")
        assert "repeats the field" in outcome.steps["panel"].error

    def test_a_split_novelty_that_is_not_offered_is_named(self, tmp_path):
        panel, params = fixture()
        params["splits"][0]["novelty"] = "family"
        assert "expected one of" in run(panel, params, tmp_path).steps["panel"].error


class TestRealPanel:
    """The tutorial's imine-reductase panel: 241 enzymes, 723 observations, four splits."""

    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        if not (EXAMPLE / "panel.json").is_file():
            pytest.skip("the imine-reductase panel is not in this checkout")
        tmp_path = tmp_path_factory.mktemp("real")
        panel = json.loads((EXAMPLE / "panel.json").read_text())
        params = json.loads((EXAMPLE / "parameters.json").read_text())
        return analyze(panel, params, tmp_path), panel

    def test_the_counts_match_the_source_accounting(self, outcome):
        result, panel = outcome
        accounting = json.loads((EXAMPLE / "source-accounting.json").read_text())
        assert result.outputs("panel")["observation_counts"] == accounting["states"]
        assert len(panel["enzymes"]) == 241 and len(panel["observations"]) == 723
        assert len(result.outputs("panel")["splits"]) == 4

    def test_held_out_reactions_give_each_reaction_one_reaction_only_prediction(
        self, outcome
    ):
        result, _ = outcome
        for evaluation in record(result)["evaluations"]:
            by_reaction: dict[tuple, set] = {}
            for row in evaluation["rows"]:
                by_reaction.setdefault(tuple(row["reaction_ids"]), set()).add(
                    row["predictions"]["reaction-ridge"]
                )
            assert all(len(values) == 1 for values in by_reaction.values())
            if evaluation["novelty"] == "reaction":
                assert evaluation["metrics"]["reaction-ridge"]["spearman_rho"] is None

    def test_every_reaction_of_the_panel_balances(self, outcome):
        result, _ = outcome
        assert all(r["balance"]["balanced"] for r in record(result)["reactions"])
