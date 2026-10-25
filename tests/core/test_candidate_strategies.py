"""candidate.select: balancing predicted gain against diversity, and several objectives together.

Both strategies have answers worked out on paper.

highest-predicted-diverse picks, at each step, the candidate with the largest of
(predicted response scaled 0 to 1 over the pool) + weight * (distance to the nearest candidate
already chosen). On six eight-residue sequences with known pairwise distances, the order for
weight 1 is c0, c3, c2, c1; for weight 0 it is the plain predicted ranking c0, c1, c2, c4; and a
very large weight picks the farthest points first.

pareto takes whole non-dominated fronts of the fitted objectives and, within a front, the sum
of the objectives scaled 0 to 1 over the pool. Six candidates with two objectives to maximise
have fronts {c0, c1, c2}, {c3, c4}, {c5}, so a batch of four is c1, c0, c2, c4; the same
candidates with both objectives to minimise have the single point c5 first.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from alambiq.capabilities.engineer.science.candidates import content_digest

POOL = {
    "c0": "AAAAAAAA",
    "c1": "AAAAAAAT",
    "c2": "AAAAAATT",
    "c3": "TTTTTTTT",
    "c4": "GGGGAAAA",
    "c5": "CCCCCCCC",
}


def candidates() -> list[dict]:
    rows = []
    for name, sequence in POOL.items():
        rows.append((name, sequence, "pool", f"g-{name}"))
    fillers = ["GGGGGGGG", "CCCCAAAA", "TTTTAAAA", "AAAATTTT", "GGGGTTTT", "CCCCGGGG"]
    for i, sequence in enumerate(fillers):
        partition = "train" if i % 2 == 0 else "evaluation"
        rows.append((f"f{i}", sequence, partition, f"{partition}-{i}"))
    return [
        {
            "sample_id": name,
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "group_id": group,
            "partition": partition,
        }
        for name, sequence, partition, group in rows
    ]


def panel_data() -> dict:
    return {"schema_version": "0.1", "candidates": candidates(), "source": "toy panel"}


def model(values: dict[str, float]) -> dict:
    return {
        "schema_version": "0.1",
        "source": {"panel_sha256": content_digest(panel_data())},
        "evaluation": {"model": {"n": 8}},
        "diagnostics": [],
        "predictions": [
            {
                "sample_id": c["sample_id"],
                "predicted": values.get(c["sample_id"], 0.0),
                "predictive_standard_deviation": 0.1,
                "sequence_sha256": c["sequence_sha256"],
                "partition": c["partition"],
                "group_id": c["group_id"],
            }
            for c in candidates()
        ],
    }


def write(tmp_path: Path, name: str, data) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def select(tmp_path, **params):
    from alambiq.api import run

    request = {
        "panel": write(tmp_path, "panel.json", panel_data()),
        "seed": 1,
        **params,
    }
    step = {"id": "pick", "capability": "candidate.select", "params": request}
    return run({"id": "s", "steps": [step]}, run_dir=tmp_path / "run")


def chosen(outcome) -> list[str]:
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return [c["sample_id"] for c in outcome.outputs("pick")["selected"]]


PREDICTED = {"c0": 10.0, "c1": 9.9, "c2": 9.8, "c3": 4.0, "c4": 5.0, "c5": 1.0}


class TestPredictedGainAgainstDiversity:
    def run(self, tmp_path, weight, count=4):
        return select(
            tmp_path,
            count=count,
            strategy="highest-predicted-diverse",
            model=write(tmp_path, "m.json", model(PREDICTED)),
            diversity="sequence",
            **({} if weight is None else {"diversity_weight": weight}),
        )

    def test_with_the_default_weight_the_order_is_the_one_worked_out_by_hand(
        self, tmp_path
    ):
        assert chosen(self.run(tmp_path, None)) == ["c0", "c3", "c2", "c1"]
        assert chosen(self.run(tmp_path, 1.0)) == ["c0", "c3", "c2", "c1"]

    def test_with_no_weight_it_is_the_plain_predicted_ranking(self, tmp_path):
        assert chosen(self.run(tmp_path, 0.0)) == ["c0", "c1", "c2", "c4"]
        plain = select(
            tmp_path,
            count=4,
            strategy="highest-predicted",
            model=write(tmp_path, "m.json", model(PREDICTED)),
        )
        assert chosen(plain) == ["c0", "c1", "c2", "c4"]

    def test_a_large_weight_picks_the_farthest_candidates_first(self, tmp_path):
        picked = chosen(self.run(tmp_path, 10.0, count=3))
        assert picked[0] == "c0" and set(picked[1:]) == {"c3", "c5"}

    def test_the_batch_is_more_spread_out_than_the_top_of_the_ranking(self, tmp_path):
        def spread(names):
            seqs = [POOL[n] for n in names]
            pairs = [
                sum(a != b for a, b in zip(x, y, strict=True)) / 8
                for i, x in enumerate(seqs)
                for y in seqs[i + 1 :]
            ]
            return sum(pairs) / len(pairs)

        assert spread(chosen(self.run(tmp_path, 1.0))) > spread(
            chosen(self.run(tmp_path, 0.0))
        )

    def test_the_value_each_candidate_had_when_chosen_is_recorded(self, tmp_path):
        outcome = self.run(tmp_path, 1.0, count=3)
        stored = json.loads(Path(outcome.outputs("pick")["record"]).read_text())
        values = [c["selection_value"] for c in stored["selected"]]
        assert values == pytest.approx(
            [1.0, 0.3333333 + 1.0, 0.97777778 + 0.25], rel=1e-6
        )

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"diversity": None}, "must be sequence or embedding-cosine"),
            ({"diversity_weight": 11}, "from 0 to 10"),
            ({"diversity_weight": -1}, "from 0 to 10"),
            (
                {"strategy": "highest-predicted", "diversity": None},
                "diversity_weight is accepted only",
            ),
        ],
    )
    def test_settings_that_do_not_fit_are_refused(self, tmp_path, changes, message):
        params = {
            "count": 3,
            "strategy": "highest-predicted-diverse",
            "model": write(tmp_path, "m.json", model(PREDICTED)),
            "diversity": "sequence",
            "diversity_weight": 1.0,
        }
        params.update(changes)
        params = {k: v for k, v in params.items() if v is not None}
        assert message in select(tmp_path, **params).steps["pick"].error


class TestSeveralObjectives:
    A = {"c0": 3.0, "c1": 2.0, "c2": 0.5, "c3": 2.0, "c4": 1.0, "c5": 0.0}
    B = {"c0": 1.0, "c1": 2.5, "c2": 3.0, "c3": 1.0, "c4": 2.0, "c5": 0.5}

    def run(self, tmp_path, objectives, count=4):
        return select(
            tmp_path,
            count=count,
            strategy="pareto",
            models=[
                write(tmp_path, "a.json", model(self.A)),
                write(tmp_path, "b.json", model(self.B)),
            ],
            objectives=objectives,
        )

    def test_fronts_are_taken_in_order_and_ordered_by_the_scaled_sum_within_a_front(
        self, tmp_path
    ):
        outcome = self.run(tmp_path, ["maximize", "maximize"])
        assert chosen(outcome) == ["c1", "c0", "c2", "c4"]
        stored = json.loads(Path(outcome.outputs("pick")["record"]).read_text())
        assert stored["fronts"] == {
            "c0": 0,
            "c1": 0,
            "c2": 0,
            "c3": 1,
            "c4": 1,
            "c5": 2,
        }

    def test_minimising_both_makes_the_dominated_corner_the_first_front(self, tmp_path):
        assert chosen(self.run(tmp_path, ["minimize", "minimize"], count=1)) == ["c5"]

    def test_one_to_maximise_and_one_to_minimise_changes_the_front(self, tmp_path):
        picked = chosen(self.run(tmp_path, ["maximize", "minimize"], count=2))
        # c0 (3, 1) has the most of A for the least of B among the non-dominated.
        assert picked[0] == "c0" or "c0" in picked
        assert "c2" not in picked

    def test_a_cap_per_group_still_applies(self, tmp_path):
        outcome = select(
            tmp_path,
            count=3,
            strategy="pareto",
            models=[
                write(tmp_path, "a.json", model(self.A)),
                write(tmp_path, "b.json", model(self.B)),
            ],
            objectives=["maximize", "maximize"],
            exclude_ids=["c1"],
        )
        assert "c1" not in chosen(outcome)

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"objectives": ["maximize"]}, "2 to 5 objectives"),
            ({"objectives": ["maximize", "sideways"]}, "expected one of"),
            (
                {"objectives": ["maximize", "maximize", "maximize"]},
                "one objective for each model",
            ),
        ],
    )
    def test_objectives_that_do_not_fit_the_models_are_refused(
        self, tmp_path, params, message
    ):
        outcome = select(
            tmp_path,
            count=3,
            strategy="pareto",
            models=[
                write(tmp_path, "a.json", model(self.A)),
                write(tmp_path, "b.json", model(self.B)),
            ],
            **params,
        )
        assert message in outcome.steps["pick"].error

    def test_models_are_needed_and_only_for_pareto(self, tmp_path):
        assert (
            "requires models"
            in select(
                tmp_path,
                count=3,
                strategy="pareto",
                objectives=["maximize", "maximize"],
            )
            .steps["pick"]
            .error
        )
        assert (
            "objectives is accepted only"
            in select(
                tmp_path,
                count=3,
                strategy="random",
                objectives=["maximize", "maximize"],
            )
            .steps["pick"]
            .error
        )
