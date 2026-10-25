"""Candidate fitting and selection, end to end through the new core.

The panel's measured property is an exact linear function of one amino-acid
fraction, so the fit must predict held-out candidates almost exactly and must
beat the training-mean baseline. Selection is seeded, so its batch is a fixed
answer that the constraints change in a known direction.
"""

from __future__ import annotations

import hashlib
import json
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("sklearn")),
    reason="the quantitative extra is required"
)

SELECT = {"metric": "activity", "condition_id": "standard", "unit": "umol/min/mg"}


def sequence(alanines: int) -> str:
    """A 60-residue sequence whose alanine fraction is the only thing that varies."""
    return "A" * alanines + "G" * (60 - alanines)


def response(alanines: int) -> float:
    """The property the panel measures: linear in the alanine fraction."""
    return 10.0 + 120.0 * (alanines / 60)


def panel_candidates() -> list[dict]:
    """Twelve training, six evaluation and six pool candidates, in separate groups."""
    rows = []
    for index, alanines in enumerate(range(5, 53, 4)):
        if index % 4 == 3:
            partition, group = "evaluation", f"eval-{index}"
        elif index % 4 == 2:
            partition, group = "pool", f"pool-{index}"
        else:
            partition, group = "train", f"train-{index // 4}"
        rows.append((f"c{index:02d}", alanines, partition, group))
    # Eight training candidates in three groups, and four evaluation candidates
    # in two groups, are the stated minimum; widen the panel to clear it.
    for index, alanines in enumerate(range(7, 55, 4), start=100):
        partition = "train" if index % 2 else "evaluation"
        group = f"{partition}-{index}"
        rows.append((f"c{index}", alanines, partition, group))
    return [
        {
            "sample_id": sample_id,
            "sequence": sequence(alanines),
            "sequence_sha256": hashlib.sha256(sequence(alanines).encode()).hexdigest(),
            "group_id": group,
            "partition": partition,
        }
        for sample_id, alanines, partition, group in rows
    ]


@pytest.fixture
def panel(tmp_path: Path) -> Path:
    candidates = panel_candidates()
    path = tmp_path / "panel.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "candidates": candidates,
                "source": "Synthetic panel with a known linear response",
            }
        )
    )
    return path


@pytest.fixture
def assay(tmp_path: Path) -> Path:
    """Measurements for every candidate that is not in the pool."""
    from alambiq.capabilities.measure.science.inspection import inspect_assay

    header = (
        "measurement_id,sample_id,condition_id,metric,value,unit,"
        "independent_id,technical_id,role,status\n"
    )
    rows = []
    for candidate in panel_candidates():
        if candidate["partition"] == "pool":
            continue
        alanines = candidate["sequence"].count("A")
        rows.append(
            f"m-{candidate['sample_id']},{candidate['sample_id']},standard,"
            f"activity,{response(alanines)},umol/min/mg,,,sample,measured\n"
        )
    record = inspect_assay(
        header + "".join(rows),
        {"standard": "Synthetic standard condition for a known linear response"},
        None,
    )
    path = tmp_path / "assay.json"
    path.write_text(json.dumps(record))
    return path


def run_ok(campaign: dict, tmp_path: Path):
    from alambiq.api import run

    outcome = run(campaign, run_dir=tmp_path / "run")
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def fit_step(panel: Path, assay: Path) -> dict:
    return {
        "id": "model",
        "capability": "candidate.fit",
        "params": {"panel": str(panel), "assay": str(assay), **SELECT},
    }


class TestCandidateFit:
    def test_a_linear_property_is_recovered_on_held_out_candidates(
        self, panel, assay, tmp_path
    ):
        out = run_ok(
            {"id": "engineering", "steps": [fit_step(panel, assay)]}, tmp_path
        ).outputs("model")
        # The response is exactly linear in a supplied feature.
        assert out["model"]["rmse"] < 1.0
        assert out["model"]["r_squared"] > 0.99
        # It must beat predicting the training mean every time.
        assert out["model"]["rmse"] < out["baseline"]["rmse"] / 10
        assert out["evaluated_on"] == out["model"]["n"] >= 4
        assert out["evaluation_groups"] >= 2
        assert out["indistinguishable_groups"] == 0

    def test_predictions_cover_the_unmeasured_pool(self, panel, assay, tmp_path):
        outcome = run_ok(
            {"id": "engineering", "steps": [fit_step(panel, assay)]}, tmp_path
        )
        record = json.loads(Path(outcome.outputs("model")["record"]).read_text())
        predicted = {row["sample_id"]: row for row in record["predictions"]}
        pool = [c for c in panel_candidates() if c["partition"] == "pool"]
        assert pool and all(c["sample_id"] in predicted for c in pool)
        for candidate in pool:
            row = predicted[candidate["sample_id"]]
            expected = response(candidate["sequence"].count("A"))
            low, high = row["model_interval_95"]
            assert low <= expected <= high

    def test_a_measured_pool_candidate_is_refused(self, panel, assay, tmp_path):
        from alambiq.api import run

        record = json.loads(assay.read_text())
        # A pool candidate's outcome must not reach a fit: it would make the
        # pool's predictions a restatement of its measurements.
        pool = next(c for c in panel_candidates() if c["partition"] == "pool")
        text = record["source_text"] + (
            f"m-leak,{pool['sample_id']},standard,activity,42.0,"
            "umol/min/mg,,,sample,measured\n"
        )
        leaked = tmp_path / "leaked.json"
        leaked.write_text(
            json.dumps(
                {
                    **record,
                    "source_text": text,
                    "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
        )
        outcome = run(
            {"id": "engineering", "steps": [fit_step(panel, leaked)]},
            run_dir=tmp_path / "run",
        )
        assert outcome.steps["model"].status == "failed"

    def test_an_evaluation_candidate_in_a_training_group_is_refused(
        self, assay, tmp_path
    ):
        from alambiq.api import run

        candidates = panel_candidates()
        training = next(c for c in candidates if c["partition"] == "train")
        evaluation = next(c for c in candidates if c["partition"] == "evaluation")
        crossed = [
            {**c, "group_id": training["group_id"]}
            if c["sample_id"] == evaluation["sample_id"]
            else c
            for c in candidates
        ]
        path = tmp_path / "crossed.json"
        path.write_text(json.dumps({"schema_version": "0.1", "candidates": crossed}))
        outcome = run(
            {"id": "engineering", "steps": [fit_step(path, assay)]},
            run_dir=tmp_path / "run",
        )
        assert outcome.steps["model"].status == "failed"
        assert "group" in outcome.steps["model"].error


class TestCandidateSelect:
    def campaign(self, panel: Path, **params) -> dict:
        return {
            "id": "batch",
            "steps": [
                {
                    "id": "next-batch",
                    "capability": "candidate.select",
                    "params": {"panel": str(panel), "count": 3, "seed": 7, **params},
                }
            ],
        }

    def test_only_pool_candidates_are_chosen_and_the_seed_fixes_them(
        self, panel, tmp_path
    ):
        first = run_ok(self.campaign(panel), tmp_path).outputs("next-batch")
        again = run_ok(self.campaign(panel), tmp_path / "second").outputs("next-batch")
        pool = {c["sample_id"] for c in panel_candidates() if c["partition"] == "pool"}
        chosen = [c["sample_id"] for c in first["selected"]]
        assert len(chosen) == 3
        assert set(chosen) <= pool
        assert [c["sample_id"] for c in again["selected"]] == chosen
        # Every candidate outside the pool is reported with its partition.
        assert first["reasons"]["partition-train"] > 0
        assert first["reasons"]["partition-evaluation"] > 0

    def test_the_highest_predicted_batch_is_the_top_of_the_fit(
        self, panel, assay, tmp_path
    ):
        campaign = self.campaign(panel, strategy="highest-predicted")
        campaign["steps"] = [fit_step(panel, assay), *campaign["steps"]]
        campaign["steps"][1]["bind"] = {"model": "model.record"}
        outcome = run_ok(campaign, tmp_path)
        chosen = [c["sample_id"] for c in outcome.outputs("next-batch")["selected"]]
        record = json.loads(Path(outcome.outputs("model")["record"]).read_text())
        pool = {c["sample_id"] for c in panel_candidates() if c["partition"] == "pool"}
        ranked = sorted(
            (row for row in record["predictions"] if row["sample_id"] in pool),
            key=lambda row: -row["predicted"],
        )
        assert chosen == [row["sample_id"] for row in ranked[:3]]

    def test_a_budget_shortens_the_batch_and_says_why(self, panel, tmp_path):
        pool = [c["sample_id"] for c in panel_candidates() if c["partition"] == "pool"]
        out = run_ok(
            self.campaign(
                panel,
                costs={sample_id: 40.0 for sample_id in pool},
                budget=100.0,
                cost_unit="illustrative-credit",
            ),
            tmp_path,
        ).outputs("next-batch")
        assert len(out["selected"]) == 2
        assert out["total_cost"] == pytest.approx(80.0)
        assert out["reasons"]["remaining-budget"] > 0

    def test_a_group_cap_limits_one_group(self, panel, tmp_path):
        out = run_ok(self.campaign(panel, max_per_group=1), tmp_path).outputs(
            "next-batch"
        )
        groups = [c["group_id"] for c in out["selected"]]
        assert len(groups) == len(set(groups))

    def test_a_model_strategy_without_a_model_is_refused(self, panel, tmp_path):
        from alambiq.api import run

        outcome = run(
            self.campaign(panel, strategy="uncertain"), run_dir=tmp_path / "run"
        )
        assert outcome.steps["next-batch"].status == "failed"
        assert "model" in outcome.steps["next-batch"].error


CODING_DNA = "ATGAAAGCGTTCACCGGTAACGACGTTCTGGAACTGTACGGTCTGAAAGGTTCTGACGCTTAA"
CODING_PROTEIN = "MKAFTGNDVLELYGLKGSDA"


@pytest.mark.skipif(
    not find_spec("dnachisel"), reason="the coding-design extra is required"
)
class TestCodingDesign:
    """Recoding keeps the protein and removes what it was asked to remove."""

    def campaign(self, tmp_path: Path, **params) -> dict:
        dna = tmp_path / "cds.fa"
        dna.write_text(f">cds\n{CODING_DNA}\n")
        protein = tmp_path / "cds.faa"
        protein.write_text(f">cds\n{CODING_PROTEIN}\n")
        return {
            "id": "recode",
            "steps": [
                {
                    "id": "design",
                    "capability": "dna.optimize_coding",
                    "params": {
                        "coding_dna": str(dna),
                        "expected_protein": str(protein),
                        "seed": 7,
                        **params,
                    },
                }
            ],
        }

    def test_a_motif_is_removed_and_the_protein_is_unchanged(self, tmp_path):
        from Bio.Seq import Seq

        # GGTTCT occurs in this sequence and has a synonymous way out.
        assert "GGTTCT" in CODING_DNA
        outcome = run_ok(self.campaign(tmp_path, avoid_patterns=["GGTTCT"]), tmp_path)
        out = outcome.outputs("design")
        assert out["status"] == "completed"
        assert out["before"]["motif_matches"] == 1
        assert out["after"]["motif_matches"] == 0
        assert out["after"]["satisfied"] is True
        assert out["changed_codons"] >= 1

        record = json.loads(Path(out["record"]).read_text())
        after = record["dna_after"]
        assert "GGTTCT" not in after
        assert len(after) == len(CODING_DNA)
        # Synonymous: the protein and the terminal codons are untouched.
        assert str(Seq(after).translate(table=11, cds=True)) == CODING_PROTEIN
        assert after[:3] == CODING_DNA[:3] and after[-3:] == CODING_DNA[-3:]
        assert all(
            change["before"] != change["after"]
            for change in record["changed_codons"]
        )

    def test_the_same_seed_gives_the_same_sequence(self, tmp_path):
        first = run_ok(self.campaign(tmp_path, avoid_patterns=["GGTTCT"]), tmp_path)
        again = run_ok(
            self.campaign(tmp_path, avoid_patterns=["GGTTCT"]), tmp_path / "again"
        )
        assert (
            json.loads(Path(first.outputs("design")["record"]).read_text())["dna_after"]
            == json.loads(Path(again.outputs("design")["record"]).read_text())[
                "dna_after"
            ]
        )

    def test_recoding_does_not_disturb_the_callers_random_state(self, tmp_path):
        import numpy as np

        np.random.seed(1234)
        expected = np.random.random()
        np.random.seed(1234)
        run_ok(self.campaign(tmp_path, avoid_patterns=["GGTTCT"]), tmp_path)
        # The solver seeds NumPy's global state; it must give it back.
        assert np.random.random() == expected

    def test_an_impossible_constraint_is_reported_not_returned(self, tmp_path):
        out = run_ok(
            self.campaign(tmp_path, gc_min=0.95, gc_max=1.0), tmp_path
        ).outputs("design")
        assert out["status"] == "failed"
        assert out["after"]["satisfied"] is False
        record = json.loads(Path(out["record"]).read_text())
        # A rejected candidate is kept for inspection, never as a usable sequence.
        assert "dna_after" not in record
        assert record["rejected_candidate"]

    def test_a_protein_the_sequence_does_not_encode_is_refused(self, tmp_path):
        from alambiq.api import run

        campaign = self.campaign(tmp_path)
        other = tmp_path / "other.faa"
        other.write_text(">cds\n" + "M" + "A" * 19 + "\n")
        campaign["steps"][0]["params"]["expected_protein"] = str(other)
        outcome = run(campaign, run_dir=tmp_path / "run")
        assert outcome.steps["design"].status == "failed"
        assert "translate" in outcome.steps["design"].error
