"""sequence.inspect_msa against alignments small enough to count by hand.

The reference alignment has five rows and six columns:

    q   ACDEFG      column 1  A A A A A     conserved, entropy 0
    r1  ACDEFG      column 2  C C C C G     C 4, G 1
    r2  ACDQFG      column 3  D D D - D     one gap, the rest agree
    r3  AC-EWG      column 4  E E Q E E     E 4, Q 1
    r4  AGDEWG      column 5  F F F W W     F 3, W 2
                    column 6  G G G G G     conserved, entropy 0
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROWS = {"q": "ACDEFG", "r1": "ACDEFG", "r2": "ACDQFG", "r3": "AC-EWG", "r4": "AGDEWG"}


def entropy(*counts: int) -> float:
    total = sum(counts)
    return -sum(n / total * math.log2(n / total) for n in counts)


def write_fasta(tmp_path: Path, rows: dict[str, str], name: str = "msa.fasta") -> str:
    path = tmp_path / name
    path.write_text("".join(f">{k}\n{v}\n" for k, v in rows.items()))
    return str(path)


def step(**params) -> dict:
    return {"id": "msa", "capability": "sequence.inspect_msa", "params": params}


def run(s: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    return execute({"id": "m", "steps": [s]}, run_dir=tmp_path / name)


def run_ok(s: dict, tmp_path: Path, name: str = "run"):
    outcome = run(s, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def record(outcome) -> dict:
    return json.loads(Path(outcome.outputs("msa")["record"]).read_text())


class TestColumnStatistics:
    @pytest.fixture
    def outcome(self, tmp_path):
        return run_ok(step(alignment=write_fasta(tmp_path, ROWS), format="fasta"), tmp_path)

    def test_the_shape_of_the_alignment(self, outcome):
        out = outcome.outputs("msa")
        assert (out["rows"], out["columns"], out["query_id"]) == (5, 6, "q")

    def test_each_column_has_the_entropy_its_counts_imply(self, outcome):
        by_position = {c["query_position"]: c for c in record(outcome)["columns"]}
        assert by_position[1]["entropy_bits"] == 0.0
        assert by_position[2]["entropy_bits"] == pytest.approx(entropy(4, 1))
        assert by_position[4]["entropy_bits"] == pytest.approx(entropy(4, 1))
        assert by_position[5]["entropy_bits"] == pytest.approx(entropy(3, 2))
        assert by_position[5]["entropy_bits"] == pytest.approx(0.970951, abs=1e-6)
        assert by_position[6]["entropy_bits"] == 0.0

    def test_a_gap_is_left_out_of_the_distribution_and_counted_as_a_gap(self, outcome):
        column = record(outcome)["columns"][2]
        assert column["gap_fraction"] == pytest.approx(0.2)
        assert column["informative_count"] == 4
        assert (column["consensus"], column["consensus_fraction"]) == ("D", 1.0)
        assert column["entropy_bits"] == 0.0

    def test_the_consensus_is_the_most_common_residue_and_its_share(self, outcome):
        column = record(outcome)["columns"][4]
        assert column["counts"] == {"F": 3, "W": 2}
        assert (column["consensus"], column["consensus_fraction"]) == ("F", 0.6)

    def test_only_columns_without_gaps_and_without_variation_are_fully_conserved(
        self, outcome
    ):
        # Column 3 agrees everywhere it has a residue, but one row has a gap.
        assert outcome.outputs("msa")["fully_conserved_positions"] == [1, 6]

    def test_the_most_variable_positions_come_in_order_of_entropy_then_position(
        self, outcome
    ):
        order = [p["query_position"] for p in outcome.outputs("msa")["most_variable"]]
        assert order == [5, 2, 4, 1, 3, 6]

    def test_the_averages_follow_from_the_columns(self, outcome):
        out = outcome.outputs("msa")
        assert out["mean_entropy_bits"] == pytest.approx((2 * entropy(4, 1) + entropy(3, 2)) / 6)
        assert out["mean_gap_fraction"] == pytest.approx(0.2 / 6)


class TestRedundancy:
    def test_an_exact_repeat_is_flagged_and_counted(self, tmp_path):
        outcome = run_ok(step(alignment=write_fasta(tmp_path, ROWS), format="fasta"), tmp_path)
        out = outcome.outputs("msa")
        # r1 repeats the query's match columns.
        assert (out["unique_rows"], out["duplicate_rows"]) == (4, 1)
        assert record(outcome)["rows"][1]["duplicate_of"] == "q"
        assert any("repeat an earlier row" in n for n in outcome.steps["msa"].notes)

    def test_a_single_row_is_noted(self, tmp_path):
        outcome = run_ok(
            step(alignment=write_fasta(tmp_path, {"q": "ACDEFG"}), format="fasta"), tmp_path
        )
        assert any("has one row" in n for n in outcome.steps["msa"].notes)


class TestA3M:
    def test_inserted_residues_are_kept_and_do_not_add_columns(self, tmp_path):
        # Lower case marks residues inserted after the second match column.
        path = tmp_path / "msa.a3m"
        path.write_text(">q\nACDEFG\n>r\nACwwDEFG\n")
        outcome = run_ok(step(alignment=str(path), format="a3m"), tmp_path)
        out = outcome.outputs("msa")
        assert (out["columns"], out["inserted_residues"]) == (6, 2)
        row = record(outcome)["rows"][1]
        assert row["sequence"] == "ACWWDEFG"
        assert row["insertions"][0]["after_column"] == 2
        assert row["insertions"][0]["sequence"] == "WW"

    def test_the_query_must_come_first_ungapped_and_in_capitals(self, tmp_path):
        path = tmp_path / "bad.a3m"
        path.write_text(">q\nAC-EFG\n>r\nACDEFG\n")
        outcome = run(step(alignment=str(path), format="a3m"), tmp_path)
        assert outcome.steps["msa"].status == "failed"
        assert "ungapped uppercase query" in outcome.steps["msa"].error

    def test_the_refusal_of_a_bad_first_row_names_the_format_and_the_row(self, tmp_path):
        path = tmp_path / "bad.a3m"
        path.write_text(">q\nAC-EFG\n>r\nACDEFG\n")
        error = run(step(alignment=str(path), format="a3m"), tmp_path).steps["msa"].error
        assert "first row 'q' of the A3M input" in error


class TestAlignedFastaFirstRow:
    @pytest.mark.parametrize("query", ["AC-EFG", "ACdEFG"])
    def test_a_first_row_with_gaps_or_lowercase_letters_is_refused(self, tmp_path, query):
        path = write_fasta(tmp_path, {"q": query, "r": "ACDEFG"})
        outcome = run(step(alignment=path, format="fasta"), tmp_path)
        assert outcome.steps["msa"].status == "failed"
        error = outcome.steps["msa"].error
        assert "first row 'q' of the aligned FASTA input" in error
        assert "ungapped uppercase query" in error

    def test_gaps_and_lowercase_letters_after_the_first_row_are_accepted(self, tmp_path):
        path = write_fasta(tmp_path, {"q": "ACDEFG", "r": "AC-eFG"})
        assert run(step(alignment=path, format="fasta"), tmp_path).ok


class TestFullSequences:
    FULL = {
        name: {"sequence": "MMM" + seq.replace("-", "G") + "KKK", "start": 4}
        for name, seq in ROWS.items()
        if "-" not in seq
    }

    def test_a_rows_positions_map_back_into_its_full_protein(self, tmp_path):
        rows = {k: v for k, v in ROWS.items() if "-" not in v}
        outcome = run_ok(
            step(
                alignment=write_fasta(tmp_path, rows),
                format="fasta",
                full_sequences={k: self.FULL[k] for k in rows},
            ),
            tmp_path,
        )
        first = record(outcome)["rows"][0]
        # The row begins at position 4 of its full sequence.
        assert first["full_sequence_positions"] == [4, 5, 6, 7, 8, 9]

    def test_a_row_that_is_not_in_its_stated_interval_is_refused(self, tmp_path):
        rows = {"q": "ACDEFG", "r": "ACDEFG"}
        wrong = {
            "q": {"sequence": "MMMACDEFGKKK", "start": 4},
            "r": {"sequence": "MMMACDEFGKKK", "start": 5},
        }
        outcome = run(step(alignment=write_fasta(tmp_path, rows), format="fasta", full_sequences=wrong), tmp_path)
        assert outcome.steps["msa"].status == "failed"
        assert "full-protein interval" in outcome.steps["msa"].error

    def test_the_full_sequences_must_cover_exactly_the_rows(self, tmp_path):
        rows = {"q": "ACDEFG", "r": "ACDEFG"}
        outcome = run(
            step(
                alignment=write_fasta(tmp_path, rows),
                format="fasta",
                full_sequences={"q": {"sequence": "MMMACDEFGKKK", "start": 4}},
            ),
            tmp_path,
        )
        assert outcome.steps["msa"].status == "failed"
        assert "exactly the supplied rows" in outcome.steps["msa"].error


class TestRefusals:
    def test_rows_of_different_widths_are_refused(self, tmp_path):
        outcome = run(
            step(alignment=write_fasta(tmp_path, {"q": "ACDEFG", "r": "ACDE"}), format="fasta"),
            tmp_path,
        )
        assert "same match-column count" in outcome.steps["msa"].error

    def test_a_repeated_identifier_is_refused(self, tmp_path):
        path = tmp_path / "dup.fasta"
        path.write_text(">q\nACDEFG\n>q\nACDEFG\n")
        outcome = run(step(alignment=str(path), format="fasta"), tmp_path)
        assert "unique" in outcome.steps["msa"].error

    def test_a_dot_is_not_a_gap_in_aligned_fasta(self, tmp_path):
        outcome = run(
            step(alignment=write_fasta(tmp_path, {"q": "ACDEFG", "r": "AC.EFG"}), format="fasta"),
            tmp_path,
        )
        assert "dots require A3M" in outcome.steps["msa"].error

    def test_a_format_that_is_not_offered_is_refused(self, tmp_path):
        outcome = run(
            step(alignment=write_fasta(tmp_path, ROWS), format="stockholm"), tmp_path
        )
        assert "'fasta'" in outcome.steps["msa"].error and "'a3m'" in outcome.steps["msa"].error

    def test_more_than_two_hundred_rows_are_refused(self, tmp_path):
        rows = {f"r{i}": "ACDEFG" for i in range(201)}
        outcome = run(step(alignment=write_fasta(tmp_path, rows), format="fasta"), tmp_path)
        assert "1–200 alignment rows" in outcome.steps["msa"].error


class TestPresentation:
    def test_the_report_names_the_conserved_positions(self, tmp_path):
        outcome = run_ok(step(alignment=write_fasta(tmp_path, ROWS), format="fasta"), tmp_path)
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "2 query positions are identical in every row" in page
        assert "Consensus residue" in page
