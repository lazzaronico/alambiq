"""variant.propose: ranking and filtering substitutions by the evidence for them.

The score tables here are written by hand, so every ranking has an answer worked out on
paper: the order by log-likelihood ratio, the order by predicted stability, and the combined
order, which is the mean of the two percentile ranks (best 0, worst 1). The filters are
checked by what they leave out and the reason each reports, homolog counts by an alignment
whose columns can be read by eye, and the shortlist by whether it can be handed on to the
stability predictor as it stands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SEQUENCE = "ACDEFG"


def score_record(llrs: dict[str, float], sequence: str = SEQUENCE) -> dict:
    rows = []
    for variant, llr in llrs.items():
        wild, position, mutant = variant[0], int(variant[1:-1]), variant[-1]
        assert sequence[position - 1] == wild
        rows.append(
            {
                "variant": variant,
                "position": position,
                "wild_type": wild,
                "mutant": mutant,
                "log_prob": llr - 1.0,
                "wild_type_log_prob": -1.0,
                "llr": llr,
            }
        )
    return {
        "schema_version": "0.1",
        "protein_id": "toy",
        "sequence": sequence,
        "model": "toy/model",
        "rows": rows,
    }


def stability_record(ddgs: dict[str, float], sequence: str = SEQUENCE) -> dict:
    return {
        "model": "toy-stability",
        "sequence": sequence,
        "rows": [
            {
                "position": int(v[1:-1]) - 1,
                "wild_type": v[0],
                "mutant": v[-1],
                "predicted_ddg_kcal_mol": d,
            }
            for v, d in ddgs.items()
        ],
    }


LLRS = {
    "A1G": 2.0,
    "A1V": 1.0,
    "C2S": 3.0,
    "C2A": -1.0,
    "D3E": 0.5,
    "D3N": 0.5,
    "E4Q": -3.0,
    "F5Y": 1.5,
}


def write(tmp_path: Path, name: str, data) -> str:
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return str(path)


def propose(tmp_path, scores=None, stability=None, **params):
    from alambiq.api import run

    request = {"scores": write(tmp_path, "scores.json", scores or score_record(LLRS))}
    if stability is not None:
        request["stability"] = write(tmp_path, "stability.json", stability)
    request.update(params)
    step = {"id": "p", "capability": "variant.propose", "params": request}
    return run({"id": "v", "steps": [step]}, run_dir=tmp_path / "run")


def proposed(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("p")


def names(out) -> list[str]:
    return [p["variant"] for p in out["proposals"]]


class TestRankingByLanguageModel:
    def test_the_best_log_likelihood_ratios_come_first(self, tmp_path):
        out = proposed(propose(tmp_path, count=4))
        assert names(out) == ["C2S", "A1G", "F5Y", "A1V"]
        assert [p["rank"] for p in out["proposals"]] == [1, 2, 3, 4]
        assert out["candidates"] == 8

    def test_ties_are_broken_by_position_then_residue(self, tmp_path):
        out = proposed(propose(tmp_path, count=8))
        assert names(out)[4:6] == ["D3E", "D3N"]

    def test_percentile_ranks_run_from_best_to_worst(self, tmp_path):
        out = proposed(propose(tmp_path, count=8))
        ranks = {p["variant"]: p["llr_rank"] for p in out["proposals"]}
        assert ranks["C2S"] == 0.0 and ranks["E4Q"] == 1.0
        assert ranks["D3E"] == ranks["D3N"] == pytest.approx(4.5 / 7)

    def test_asking_for_more_than_there_is_returns_what_there_is_and_says_so(
        self, tmp_path
    ):
        outcome = propose(tmp_path, count=50)
        out = proposed(outcome)
        assert len(out["proposals"]) == 8
        assert "8 of the 50" in " ".join(outcome.steps["p"].notes)


class TestFilters:
    def test_positions_can_be_chosen_or_excluded(self, tmp_path):
        only = proposed(propose(tmp_path, positions=[1, 5], count=10))
        assert set(names(only)) == {"A1G", "A1V", "F5Y"}
        without = proposed(propose(tmp_path, exclude_positions=[2, 4], count=10))
        assert set(names(without)) == {"A1G", "A1V", "D3E", "D3N", "F5Y"}
        assert without["excluded"] == {"position-excluded": 3}

    def test_a_floor_on_the_score_leaves_out_what_falls_below_it(self, tmp_path):
        out = proposed(propose(tmp_path, minimum_llr=0.5, count=10))
        assert set(names(out)) == {"C2S", "A1G", "F5Y", "A1V", "D3E", "D3N"}
        assert out["excluded"] == {"below-minimum-llr": 2}

    def test_a_cap_per_position_spreads_the_proposals(self, tmp_path):
        out = proposed(propose(tmp_path, max_per_position=1, count=10))
        assert names(out) == ["C2S", "A1G", "F5Y", "D3E", "E4Q"]
        assert out["excluded"]["position-cap"] == 3

    def test_nothing_passing_the_filters_is_refused(self, tmp_path):
        outcome = propose(tmp_path, minimum_llr=99.0)
        assert "No substitution passes the filters" in outcome.steps["p"].error


class TestStability:
    DDG = {
        "A1G": 0.5,
        "A1V": -1.0,
        "C2S": 0.0,
        "C2A": 2.0,
        "D3E": -0.2,
        "D3N": 1.0,
        "E4Q": 0.3,
        "F5Y": -2.0,
    }

    def test_ranking_by_stability_puts_the_most_stabilising_first(self, tmp_path):
        out = proposed(
            propose(
                tmp_path,
                stability=stability_record(self.DDG),
                ranking="stability",
                count=3,
            )
        )
        assert names(out) == ["F5Y", "A1V", "D3E"]
        assert [p["ddg_kcal_mol"] for p in out["proposals"]] == [-2.0, -1.0, -0.2]

    def test_the_combined_rank_is_the_mean_of_the_two_percentile_ranks(self, tmp_path):
        llrs = {"A1G": 4.0, "C2S": 3.0, "D3E": 2.0, "E4Q": 1.0}
        ddgs = {"A1G": 0.5, "C2S": -1.0, "D3E": 0.0, "E4Q": 2.0}
        out = proposed(
            propose(
                tmp_path,
                scores=score_record(llrs),
                stability=stability_record(ddgs),
                ranking="combined",
                count=4,
            )
        )
        # LLR ranks 0, 1/3, 2/3, 1; stability ranks (by -ddg) 2/3, 0, 1/3, 1.
        by = {p["variant"]: p for p in out["proposals"]}
        assert by["A1G"]["combined_rank"] == pytest.approx(1 / 3)
        assert by["C2S"]["combined_rank"] == pytest.approx(1 / 6)
        assert by["D3E"]["combined_rank"] == pytest.approx(1 / 2)
        assert by["E4Q"]["combined_rank"] == pytest.approx(1.0)
        assert names(out) == ["C2S", "A1G", "D3E", "E4Q"]

    def test_a_ceiling_on_destabilisation_removes_what_exceeds_it(self, tmp_path):
        out = proposed(
            propose(
                tmp_path,
                stability=stability_record(self.DDG),
                maximum_ddg_kcal_mol=0.4,
                count=10,
            )
        )
        assert "C2A" not in names(out) and "D3N" not in names(out)
        assert out["excluded"]["destabilizing"] == 3

    def test_candidates_without_a_prediction_are_left_out_of_a_stability_ranking(
        self, tmp_path
    ):
        partial = {k: v for k, v in self.DDG.items() if k != "C2S"}
        out = proposed(
            propose(
                tmp_path,
                stability=stability_record(partial),
                ranking="stability",
                count=10,
            )
        )
        assert "C2S" not in names(out)
        assert out["excluded"] == {"no-stability-prediction": 1}

    def test_a_language_model_ranking_keeps_them_and_leaves_their_stability_empty(
        self, tmp_path
    ):
        partial = {k: v for k, v in self.DDG.items() if k != "C2S"}
        out = proposed(propose(tmp_path, stability=stability_record(partial), count=2))
        first = out["proposals"][0]
        assert first["variant"] == "C2S" and first["ddg_kcal_mol"] is None
        assert first["stability_rank"] is None

    def test_a_prediction_for_another_sequence_is_refused(self, tmp_path):
        other = stability_record(self.DDG, sequence="ACDEFH")
        outcome = propose(tmp_path, stability=other, ranking="stability")
        assert "differs from the sequence" in outcome.steps["p"].error

    def test_ranking_by_stability_needs_a_prediction(self, tmp_path):
        assert (
            "requires a stability record"
            in propose(tmp_path, ranking="combined").steps["p"].error
        )


class TestHomologs:
    ALIGNMENT = ">query\nACDEFG\n>h1\nAVDEFG\n>h2\nASDQFG\n>h3\nA-DEYG\n>h4\nGCNEFA\n"

    def test_each_substitution_counts_the_homologs_that_carry_it(self, tmp_path):
        out = proposed(
            propose(
                tmp_path,
                homologs=write(tmp_path, "h.fa", self.ALIGNMENT),
                count=8,
                scores=score_record(
                    {
                        "C2S": 3.0,
                        "C2V": 1.0,
                        "C2A": 0.0,
                        "E4Q": 0.2,
                        "F5Y": 1.0,
                        "A1G": 2.0,
                    }
                ),
            )
        )
        by = {p["variant"]: p for p in out["proposals"]}
        assert by["C2S"]["homolog_count"] == 1 and by["C2V"]["homolog_count"] == 1
        assert by["C2A"]["homolog_count"] == 0
        assert by["E4Q"]["homolog_count"] == 1 and by["F5Y"]["homolog_count"] == 1
        assert by["A1G"]["homolog_count"] == 1
        assert by["C2S"]["homolog_fraction"] == pytest.approx(0.25)

    def test_restricting_to_homolog_residues_drops_the_rest(self, tmp_path):
        scores = score_record({"C2S": 3.0, "C2V": 1.0, "C2A": 0.0, "E4Q": 0.2})
        out = proposed(
            propose(
                tmp_path,
                scores=scores,
                homologs=write(tmp_path, "h.fa", self.ALIGNMENT),
                homolog_only=True,
                count=10,
            )
        )
        assert set(names(out)) == {"C2S", "C2V", "E4Q"}
        assert out["excluded"] == {"not-seen-in-homologs": 1}

    def test_a3m_insertions_are_ignored(self, tmp_path):
        a3m = ">query\nACDEFG\n>h1\nAVdddDEFG\n"
        out = proposed(
            propose(
                tmp_path,
                scores=score_record({"C2V": 1.0, "C2S": 2.0}),
                homologs=write(tmp_path, "h.a3m", a3m),
                homologs_format="a3m",
                count=2,
            )
        )
        by = {p["variant"]: p for p in out["proposals"]}
        assert by["C2V"]["homolog_count"] == 1 and by["C2S"]["homolog_count"] == 0

    def test_a_first_row_that_is_not_the_starting_sequence_is_refused(self, tmp_path):
        outcome = propose(
            tmp_path, homologs=write(tmp_path, "h.fa", ">q\nACDEFH\n>h\nACDEFG\n")
        )
        assert "must be the starting sequence" in outcome.steps["p"].error

    def test_restricting_without_an_alignment_is_refused(self, tmp_path):
        assert (
            "requires an alignment"
            in propose(tmp_path, homolog_only=True).steps["p"].error
        )


class TestHandOff:
    def test_the_shortlist_is_in_the_form_the_stability_predictor_takes(self, tmp_path):
        from alambiq.capabilities.characterize.protein_predict_stability_change import (
            Substitution,
        )
        from alambiq.core.loader import build

        out = proposed(propose(tmp_path, count=3, author_offset=100))
        assert out["stability_request"] == [
            {"residue": "102", "wild_type": "C", "mutant": "S"},
            {"residue": "101", "wild_type": "A", "mutant": "G"},
            {"residue": "105", "wild_type": "F", "mutant": "Y"},
        ]
        for item in out["stability_request"]:
            built = build(Substitution, item, "shortlist")
            assert built.mutant in "ACDEFGHIKLMNPQRSTVWY"

    def test_the_report_charts_the_proposals_in_rank_order(self, tmp_path):
        import re

        outcome = propose(tmp_path, count=4)
        page = outcome.report(tmp_path / "r.html").read_text()
        labels = re.findall(r'class="bar-label"[^>]*>([^<]+)<', page)
        assert labels == ["C2S", "A1G", "F5Y", "A1V"]


class TestRefusals:
    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"count": 0}, "count must be from 1 to 2,000"),
            ({"count": 5000}, "count must be from 1 to 2,000"),
            ({"max_per_position": 0}, "at least 1"),
        ],
    )
    def test_settings_outside_their_range_are_refused(self, tmp_path, params, message):
        assert message in propose(tmp_path, **params).steps["p"].error

    def test_a_record_that_is_not_a_score_table_is_refused(self, tmp_path):
        outcome = propose(tmp_path, scores={"rows": []})
        assert "must be a protein.score_substitutions record" in outcome.steps["p"].error
