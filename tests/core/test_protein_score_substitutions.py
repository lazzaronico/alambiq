"""protein.score_substitutions on ubiquitin, with the real ESM C 300M.

Ubiquitin is one of the most conserved proteins there is, so a language model that has learned
anything should name the wild-type residue as its first choice at most masked positions.
Beyond that, the scores must obey the arithmetic they are defined by, whatever the model says:

* the log-probabilities at a position are those of a distribution over the twenty canonical
  residues, so their exponentials sum to at most one, and to nearly one;
* a substitution's log-likelihood ratio is its log-probability minus the wild type's at the
  same position, so each ratio is recomputed here from the stored log-probabilities;
* scoring a subset of positions gives the same numbers as scoring all of them, and the batch
  size changes nothing beyond float noise;
* the sequence, the positions and the device are checked before any model runs.

The model runs in the pinned esm environment (ALAMBIQ_ENV_ESM) with the esmc-300m resource.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

ENV = os.environ.get("ALAMBIQ_ENV_ESM", "")


def installed() -> bool:
    from alambiq.core import resources

    return (
        bool(ENV) and Path(ENV).is_file() and resources.location("esmc-300m").is_dir()
    )


pytestmark = pytest.mark.skipif(
    not installed(),
    reason="set ALAMBIQ_ENV_ESM and install the esmc-300m resource to run",
)

UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_ESM", ENV)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def fasta(tmp_path: Path, sequence: str = UBIQUITIN, name: str = "ubq") -> str:
    path = tmp_path / f"{name}.faa"
    path.write_text(f">{name}\n{sequence}\n")
    return str(path)


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {
        "id": "score",
        "capability": "protein.score_substitutions",
        "params": params,
    }
    return execute({"id": "s", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def table(outcome) -> dict:
    return json.loads(Path(outcome.outputs("score")["record"]).read_text())


@pytest.fixture(scope="module")
def everything(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ubq")
    return run_ok({"protein": fasta(tmp), "batch_size": 16, "threads": 4}, tmp)


class TestTheScores:
    def test_every_position_gets_the_other_nineteen_residues(self, everything):
        record = table(everything)
        out = everything.outputs("score")
        assert out["length"] == len(UBIQUITIN) == out["positions_scored"]
        assert out["substitutions"] == 19 * len(UBIQUITIN) == len(record["rows"])
        for row in record["rows"][:200]:
            assert row["wild_type"] == UBIQUITIN[row["position"] - 1] != row["mutant"]
            assert (
                row["variant"] == f"{row['wild_type']}{row['position']}{row['mutant']}"
            )

    def test_each_ratio_is_the_mutants_log_probability_minus_the_wild_types(
        self, everything
    ):
        for row in table(everything)["rows"]:
            assert row["llr"] == pytest.approx(
                row["log_prob"] - row["wild_type_log_prob"], abs=1e-9
            )

    def test_the_log_probabilities_are_those_of_a_distribution_over_the_twenty_residues(
        self, everything
    ):
        rows = table(everything)["rows"]
        by_position: dict[int, float] = {}
        wild: dict[int, float] = {}
        for row in rows:
            by_position[row["position"]] = by_position.get(
                row["position"], 0.0
            ) + math.exp(row["log_prob"])
            wild[row["position"]] = row["wild_type_log_prob"]
        for position, mutants in by_position.items():
            total = mutants + math.exp(wild[position])
            assert total <= 1.0 + 1e-5
            assert total > 0.9

    def test_the_model_names_the_wild_type_first_at_most_positions_of_a_conserved_protein(
        self, everything
    ):
        assert everything.outputs("score")["wild_type_is_top"] > 0.5

    def test_the_least_certain_positions_are_the_ones_with_the_highest_entropy(
        self, everything
    ):
        out = everything.outputs("score")
        entropies = [p["entropy_bits"] for p in out["positions"]]
        assert min(entropies) >= 0.0 and max(entropies) <= math.log2(20) + 1e-9
        for p in out["positions"]:
            assert p["wild_type"] == UBIQUITIN[p["position"] - 1]
            assert p["top_log_prob"] >= p["wild_type_log_prob"] - 1e-12

    def test_a_subset_of_positions_gives_the_same_numbers_as_all_of_them(
        self, everything, tmp_path
    ):
        subset = run_ok(
            {"protein": fasta(tmp_path), "positions": [5, 20, 48]}, tmp_path
        )
        full = {r["variant"]: r["llr"] for r in table(everything)["rows"]}
        rows = table(subset)["rows"]
        assert {r["position"] for r in rows} == {5, 20, 48} and len(rows) == 57
        for row in rows:
            assert row["llr"] == pytest.approx(full[row["variant"]], abs=2e-4)

    def test_the_batch_size_changes_nothing_but_float_noise(self, tmp_path):
        a = run_ok(
            {"protein": fasta(tmp_path), "positions": [3, 10, 30, 40], "batch_size": 1},
            tmp_path,
            "a",
        )
        b = run_ok(
            {"protein": fasta(tmp_path), "positions": [3, 10, 30, 40], "batch_size": 4},
            tmp_path,
            "b",
        )
        for x, y in zip(table(a)["rows"], table(b)["rows"], strict=True):
            assert x["variant"] == y["variant"]
            assert x["llr"] == pytest.approx(y["llr"], abs=2e-4)

    def test_a_charged_residue_in_the_hydrophobic_core_is_less_natural_than_a_conservative_one(
        self, tmp_path
    ):
        # Ile3 is buried in ubiquitin's core: Leu or Val is a mild change, Asp is not.
        out = run_ok({"protein": fasta(tmp_path), "positions": [3]}, tmp_path)
        llr = {r["mutant"]: r["llr"] for r in table(out)["rows"]}
        assert UBIQUITIN[2] == "I"
        assert llr["V"] > llr["D"] and llr["L"] > llr["D"]

    def test_the_record_names_the_model_and_the_sequence(self, everything):
        record = table(everything)
        assert record["model"] == "biohub/ESMC-300M" and record["sequence"] == UBIQUITIN
        assert record["license"] and record["resource_identity"]

    def test_the_report_draws_the_wild_types_log_probability_along_the_sequence(
        self, everything, tmp_path
    ):
        page = everything.report(tmp_path / "r.html").read_text()
        assert "Log-probability of the wild type at each position" in page
        assert "10 positions with the highest entropy" in page


class TestRefusals:
    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"positions": [0]}, "positions must lie between"),
            ({"positions": [999]}, "positions must lie between"),
            ({"threads": 0}, "threads must be an integer"),
            ({"batch_size": 100}, "batch_size must be an integer"),
        ],
    )
    def test_requests_outside_their_range_are_refused_before_a_model_runs(
        self, tmp_path, params, message
    ):
        outcome = run({"protein": fasta(tmp_path), **params}, tmp_path)
        assert message in outcome.steps["score"].error

    def test_a_sequence_with_a_non_canonical_residue_is_refused(self, tmp_path):
        outcome = run({"protein": fasta(tmp_path, "MQIFVKXLTG")}, tmp_path)
        assert "canonical residues" in outcome.steps["score"].error

    def test_a_device_that_is_not_there_is_an_error_not_a_fallback(self, tmp_path):
        outcome = run(
            {"protein": fasta(tmp_path), "positions": [1], "device": "cuda"}, tmp_path
        )
        assert "CUDA is unavailable" in outcome.steps["score"].error


class TestVariants:
    @pytest.fixture(scope="class")
    def scored(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("variants")
        return run_ok(
            {
                "protein": fasta(tmp),
                "variants": ["I3V", "K6N:T7A", "I3V:K6N:T7A", "Q2L:I3V"],
                "batch_size": 16,
                "threads": 4,
            },
            tmp,
        )

    def test_the_score_of_a_variant_is_the_sum_of_its_substitutions(self, scored):
        rows = {r["variant"]: r["llr"] for r in table(scored)["rows"]}
        scores = {
            v["variant"]: v["llr"] for v in scored.outputs("score")["variant_scores"]
        }
        assert scores["I3V"] == pytest.approx(rows["I3V"])
        assert scores["K6N:T7A"] == pytest.approx(rows["K6N"] + rows["T7A"])
        assert scores["I3V:K6N:T7A"] == pytest.approx(
            rows["I3V"] + rows["K6N"] + rows["T7A"]
        )

    def test_a_substitution_in_two_variants_has_one_score(self, scored):
        scores = {
            v["variant"]: v["llr"] for v in scored.outputs("score")["variant_scores"]
        }
        shared = scores["Q2L:I3V"] - scores["I3V"]
        rows = {r["variant"]: r["llr"] for r in table(scored)["rows"]}
        assert shared == pytest.approx(rows["Q2L"])

    def test_variants_are_returned_in_the_order_of_the_request(self, scored):
        names = [v["variant"] for v in scored.outputs("score")["variant_scores"]]
        assert names == ["I3V", "K6N:T7A", "I3V:K6N:T7A", "Q2L:I3V"]
        assert scored.outputs("score")["variant_scores"][1]["substitutions"] == [
            "K6N",
            "T7A",
        ]

    def test_only_the_positions_of_the_variants_are_scored_without_positions(
        self, scored
    ):
        assert table(scored)["positions"] == [2, 3, 6, 7]
        assert scored.outputs("score")["positions_scored"] == 4

    def test_given_positions_are_scored_in_addition(self, tmp_path):
        outcome = run_ok(
            {"protein": fasta(tmp_path), "positions": [10], "variants": ["I3V"]},
            tmp_path,
        )
        assert table(outcome)["positions"] == [3, 10]

    def test_the_record_lists_the_variants(self, scored):
        record = table(scored)
        assert [v["variant"] for v in record["variants"]] == [
            "I3V",
            "K6N:T7A",
            "I3V:K6N:T7A",
            "Q2L:I3V",
        ]

    def test_the_report_lists_the_variant_scores(self, scored):
        from alambiq.capabilities.characterize.protein_score_substitutions_report import (
            render,
        )

        html = render(scored.steps["score"].result)
        assert "K6N:T7A" in html
        assert "Log-likelihood ratio" in html

    def test_a_request_without_variants_has_no_variant_scores(self, tmp_path):
        outcome = run_ok({"protein": fasta(tmp_path), "positions": [3]}, tmp_path)
        assert outcome.outputs("score")["variant_scores"] == []

    @pytest.mark.parametrize(
        ("variants", "message"),
        [
            (["A3V"], "position 3 is I in the sequence, not A"),
            (["I3I"], "must be one of the 20 amino acids and differ"),
            (["I3V:I3L"], "two substitutions are at one position"),
            (["I300V"], "outside the sequence"),
            (["I3"], "is not a substitution"),
        ],
    )
    def test_an_inconsistent_variant_is_refused_before_a_model_runs(
        self, tmp_path, variants, message
    ):
        outcome = run({"protein": fasta(tmp_path), "variants": variants}, tmp_path)
        assert message in outcome.steps["score"].error

    def test_more_than_a_thousand_variants_are_refused(self, tmp_path):
        outcome = run(
            {"protein": fasta(tmp_path), "variants": ["I3V"] * 1001}, tmp_path
        )
        assert "At most 1000 variants" in outcome.steps["score"].error
