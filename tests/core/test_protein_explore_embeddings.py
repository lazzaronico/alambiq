"""protein.explore_embeddings on vectors chosen so the answers are known.

Four-dimensional vectors stand in for a model's output; nothing here needs a
model. The panel has two tight groups and one outlier:

    a1 (1, 0, 0, 0)       b1 (0, 1, 0, 0)       c1 (0, 0, 1, 0)
    a2 (0.9, 0.1, 0, 0)   b2 (0.1, 0.9, 0, 0)
    a3 (0.8, 0.2, 0, 0)

Cosine distance between a1 and a2 is 1 - 0.9 / sqrt(0.82), about 0.006116.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from alambiq.capabilities.characterize.science.embeddings import merge_records

PANEL = {
    "a1": [1.0, 0.0, 0.0, 0.0],
    "a2": [0.9, 0.1, 0.0, 0.0],
    "a3": [0.8, 0.2, 0.0, 0.0],
    "b1": [0.0, 1.0, 0.0, 0.0],
    "b2": [0.1, 0.9, 0.0, 0.0],
    "c1": [0.0, 0.0, 1.0, 0.0],
}


def record(vectors: dict[str, list[float]], **overrides) -> dict:
    base = {
        "model": "toy/model",
        "revision": "r1",
        "layer": "last",
        "pooling": "mean",
        "input_tracks": ["sequence"],
        "dimension": 4,
        "proteins": [
            {"id": k, "sequence": "ACDEFG", "pooled_vector": v}
            for k, v in vectors.items()
        ],
    }
    return {**base, **overrides}


def write(tmp_path: Path, name: str, data: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {
        "id": "explore",
        "capability": "protein.explore_embeddings",
        "params": params,
    }
    return execute({"id": "e", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


@pytest.fixture
def panel(tmp_path):
    return write(tmp_path, "panel.json", record(PANEL))


class TestPca:
    def test_each_proteins_nearest_neighbour_is_the_one_a_person_would_pick(
        self, panel, tmp_path
    ):
        outcome = run_ok({"embeddings": panel}, tmp_path)
        nearest = {n["id"]: n for n in outcome.outputs("explore")["nearest"]}
        assert nearest["a1"]["neighbour"] == "a2"
        assert nearest["a3"]["neighbour"] == "a2"
        assert nearest["b1"]["neighbour"] == "b2"
        assert nearest["b2"]["neighbour"] == "b1"

    def test_the_cosine_distance_is_the_one_worked_out_by_hand(self, panel, tmp_path):
        outcome = run_ok({"embeddings": panel}, tmp_path)
        a1 = {n["id"]: n for n in outcome.outputs("explore")["nearest"]}["a1"]
        assert a1["cosine_distance"] == pytest.approx(
            1 - 0.9 / math.sqrt(0.82), abs=1e-9
        )

    def test_the_two_axes_explain_a_share_of_the_variance_largest_first(self, tmp_path):
        outcome = run_ok(
            {"embeddings": write(tmp_path, "p.json", record(PANEL))}, tmp_path
        )
        explained = outcome.outputs("explore")["explained_variance_ratio"]
        assert len(explained) == 2
        assert 0 < sum(explained) <= 1 + 1e-12
        assert explained[0] >= explained[1]

    def test_points_on_a_line_are_explained_entirely_by_one_axis(self, tmp_path):
        line = {f"p{i}": [1.0 + i, 2.0 * (1.0 + i), 0.0, 0.0] for i in range(6)}
        outcome = run_ok(
            {"embeddings": write(tmp_path, "line.json", record(line))}, tmp_path
        )
        first, second = outcome.outputs("explore")["explained_variance_ratio"]
        assert first == pytest.approx(1.0, abs=1e-9)
        assert second == pytest.approx(0.0, abs=1e-9)

    def test_the_layout_has_one_point_per_protein_in_input_order(self, panel, tmp_path):
        points = run_ok({"embeddings": panel}, tmp_path).outputs("explore")["points"]
        assert [p["id"] for p in points] == list(PANEL)

    def test_representatives_start_at_the_medoid_and_spread_out(self, panel, tmp_path):
        recommended = run_ok(
            {"embeddings": panel, "representatives": 3}, tmp_path
        ).outputs("explore")["recommended"]
        assert len(recommended) == 3
        # One from each of the three directions, not three from the a group.
        groups = {name[0] for name in recommended}
        assert groups == {"a", "b", "c"}

    def test_asking_for_more_representatives_than_proteins_gives_each_protein_once(
        self, panel, tmp_path
    ):
        outcome = run_ok({"embeddings": panel, "representatives": 20}, tmp_path)
        # The panel has six proteins: no more than six can be chosen, none twice.
        names = outcome.outputs("explore")["recommended"]
        assert len(names) == len(set(names)) <= len(PANEL)

    def test_identical_vectors_are_not_proposed_twice(self, tmp_path):
        twins = {"x": [1.0, 0, 0, 0], "y": [1.0, 0, 0, 0], "z": [0, 1.0, 0, 0]}
        outcome = run_ok(
            {
                "embeddings": write(tmp_path, "t.json", record(twins)),
                "representatives": 3,
            },
            tmp_path,
        )
        assert len(outcome.outputs("explore")["recommended"]) == 2

    def test_the_stored_record_carries_the_model_and_the_hierarchy(
        self, panel, tmp_path
    ):
        outcome = run_ok({"embeddings": panel}, tmp_path)
        stored = json.loads(Path(outcome.outputs("explore")["record"]).read_text())
        assert stored["representation"]["model"] == "toy/model"
        assert len(stored["hierarchy"]) == len(PANEL) - 1
        assert stored["distance"].startswith("cosine")

    def test_a_layout_that_hides_most_of_the_variance_says_so(self, tmp_path):
        # Six orthogonal directions: two axes cannot explain much.
        spread = {f"p{i}": [1.0 if j == i else 0.0 for j in range(6)] for i in range(6)}
        outcome = run_ok(
            {"embeddings": write(tmp_path, "s.json", record(spread, dimension=6))},
            tmp_path,
        )
        assert "The two principal components explain" in " ".join(outcome.steps["explore"].notes)

    def test_a_tiny_panel_is_flagged(self, tmp_path):
        outcome = run_ok(
            {
                "embeddings": write(
                    tmp_path, "t.json", record({"a": PANEL["a1"], "b": PANEL["b1"]})
                )
            },
            tmp_path,
        )
        assert "2 proteins, fewer than 5" in " ".join(outcome.steps["explore"].notes)

    def test_the_report_shows_the_layout(self, panel, tmp_path):
        outcome = run_ok({"embeddings": panel}, tmp_path)
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "<svg" in page
        assert "toy/model" in page


class TestUmap:
    @pytest.fixture
    def many(self, tmp_path):
        rng_vectors = {}
        for group in range(3):
            for member in range(5):
                vector = [0.0] * 4
                vector[group] = 1.0
                vector[3] = 0.05 * member
                rng_vectors[f"g{group}m{member}"] = vector
        return write(tmp_path, "many.json", record(rng_vectors))

    def test_a_seeded_layout_is_reproducible_and_keeps_groups_together(
        self, many, tmp_path
    ):
        pytest.importorskip("umap")
        first = run_ok(
            {
                "embeddings": many,
                "projection": "umap",
                "seed": 3,
                "n_neighbors": 4,
                "n_epochs": 100,
            },
            tmp_path,
            "a",
        )
        again = run_ok(
            {
                "embeddings": many,
                "projection": "umap",
                "seed": 3,
                "n_neighbors": 4,
                "n_epochs": 100,
            },
            tmp_path,
            "b",
        )
        assert first.outputs("explore")["points"] == again.outputs("explore")["points"]
        assert first.outputs("explore")["explained_variance_ratio"] is None
        assert "UMAP" in first.outputs("explore")["projection"]
        # Each point's nearest neighbour is in its own group.
        for entry in first.outputs("explore")["nearest"]:
            assert entry["id"][:2] == entry["neighbour"][:2]

    def test_a_neighbour_count_not_smaller_than_the_panel_is_refused(
        self, panel, tmp_path
    ):
        outcome = run(
            {"embeddings": panel, "projection": "umap", "n_neighbors": 6}, tmp_path
        )
        assert (
            "n_neighbors must be smaller than the panel"
            in outcome.steps["explore"].error
        )

    def test_min_dist_is_checked(self, many, tmp_path):
        pytest.importorskip("umap")
        outcome = run(
            {"embeddings": many, "projection": "umap", "min_dist": 2.0}, tmp_path
        )
        assert "min_dist" in outcome.steps["explore"].error


class TestSeveralRecords:
    def test_records_of_one_space_are_analysed_together(self, tmp_path):
        first = write(
            tmp_path, "one.json", record({k: PANEL[k] for k in ("a1", "a2", "a3")})
        )
        second = write(
            tmp_path, "two.json", record({k: PANEL[k] for k in ("b1", "b2", "c1")})
        )
        outcome = run_ok({"embeddings": first, "more_embeddings": [second]}, tmp_path)
        assert outcome.outputs("explore")["proteins"] == 6
        assert [p["id"] for p in outcome.outputs("explore")["points"]] == list(PANEL)

    @pytest.mark.parametrize(
        ("change", "named"),
        [
            ({"model": "other/model"}, "model"),
            ({"revision": "r2"}, "revision"),
            ({"layer": "first"}, "layer"),
            ({"pooling": "max"}, "pooling"),
            ({"input_tracks": ["sequence", "structure"]}, "input_tracks"),
        ],
    )
    def test_records_from_different_spaces_are_refused_and_the_difference_named(
        self, tmp_path, change, named
    ):
        first = write(tmp_path, "one.json", record({"a1": PANEL["a1"]}))
        second = write(tmp_path, "two.json", record({"b1": PANEL["b1"]}, **change))
        outcome = run({"embeddings": first, "more_embeddings": [second]}, tmp_path)
        error = outcome.steps["explore"].error
        assert "different representation space" in error and named in error

    def test_a_different_dimension_is_refused(self, tmp_path):
        first = write(tmp_path, "one.json", record({"a1": PANEL["a1"]}))
        second = write(tmp_path, "two.json", record({"b1": [1.0, 0.0]}, dimension=2))
        outcome = run({"embeddings": first, "more_embeddings": [second]}, tmp_path)
        assert "dimension" in outcome.steps["explore"].error

    def test_a_protein_in_two_records_is_refused(self, tmp_path):
        first = write(tmp_path, "one.json", record({"a1": PANEL["a1"]}))
        second = write(tmp_path, "two.json", record({"a1": PANEL["a2"]}))
        outcome = run({"embeddings": first, "more_embeddings": [second]}, tmp_path)
        assert "unique across the embedding records" in outcome.steps["explore"].error

    def test_merging_keeps_the_first_records_identity(self):
        merged = merge_records(
            [record({"a1": PANEL["a1"]}), record({"b1": PANEL["b1"]})]
        )
        assert merged["model"] == "toy/model"
        assert [p["id"] for p in merged["proteins"]] == ["a1", "b1"]

    def test_nothing_to_merge_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            merge_records([])


class TestRefusals:
    @pytest.mark.parametrize(
        ("bad", "message"),
        [
            (record(PANEL, dimension=3), "Vector dimensions"),
            (record({"a": [0.0, 0.0, 0.0, 0.0]}), "nonzero vector norm"),
            (record({"a": [float("nan"), 0, 0, 0]}), "Vector dimensions"),
            ({k: v for k, v in record(PANEL).items() if k != "model"}, "needs model"),
            (record({}), "between 1 and 500"),
        ],
    )
    def test_a_record_that_cannot_be_trusted_is_refused(self, tmp_path, bad, message):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        outcome = run({"embeddings": str(path)}, tmp_path)
        assert message in outcome.steps["explore"].error

    def test_a_protein_whose_vector_is_from_another_model_is_refused(self, tmp_path):
        data = record(PANEL)
        data["proteins"][2]["model"] = "other/model"
        outcome = run({"embeddings": write(tmp_path, "mixed.json", data)}, tmp_path)
        assert (
            "Do not mix different representation spaces"
            in outcome.steps["explore"].error
        )

    def test_a_sequence_that_does_not_match_its_hash_is_refused(self, tmp_path):
        data = record(PANEL)
        data["proteins"][0]["sequence_sha256"] = "0" * 64
        outcome = run({"embeddings": write(tmp_path, "h.json", data)}, tmp_path)
        assert "SHA-256 mismatch" in outcome.steps["explore"].error

    def test_representatives_outside_one_to_sixty_four_are_refused(
        self, panel, tmp_path
    ):
        outcome = run({"embeddings": panel, "representatives": 65}, tmp_path)
        assert (
            "representatives must be an integer from 1 to 64"
            in outcome.steps["explore"].error
        )
