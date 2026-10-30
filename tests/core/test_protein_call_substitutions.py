"""protein.call_substitutions names the differences between a variant and its closest reference."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from alambiq.engines.swissprot_annotations import COLUMNS

AMINO = "ACDEFGHIKLMNPQRSTVWY"
MMSEQS = os.environ.get("ALAMBIQ_ENV_MMSEQS")
BINARY = (Path(MMSEQS).parent / "mmseqs") if MMSEQS else None


def mmseqs_binary() -> str | None:
    if BINARY and BINARY.is_file():
        return str(BINARY)
    return shutil.which("mmseqs")


def protein(seed: int, length: int = 160) -> str:
    return "".join(random.Random(seed).choices(AMINO, k=length))


REF = protein(1)
OTHER = protein(2, 200)


def swap(residue: str) -> str:
    return "W" if residue != "W" else "Y"


def edited() -> str:
    """REF with substitutions at 21 and 51, GGG after 70, residues 98 to 100 removed, and ends."""
    residues = list(REF)
    residues[20] = swap(residues[20])
    residues[50] = swap(residues[50])
    sequence = "".join(residues)
    sequence = sequence[:70] + "GGG" + sequence[70:]
    sequence = sequence[:100] + sequence[103:]
    return "MKV" + sequence + "HHHH"


def point_mutant() -> str:
    residues = list(REF)
    residues[119] = swap(residues[119])
    return "".join(residues)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", MMSEQS or sys.executable)


def write(path: Path, records: dict[str, str]) -> str:
    path.write_text("".join(f">{k}\n{v}\n" for k, v in records.items()))
    return str(path)


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "call", "capability": "protein.call_substitutions", "params": params}
    return execute({"id": "c", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path):
    outcome = run(params, tmp_path)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def refusal(params: dict, tmp_path: Path) -> str:
    outcome = run(params, tmp_path)
    assert not outcome.ok
    return " ".join(x.error or "" for x in outcome.steps.values())


@pytest.fixture
def panel(tmp_path) -> str:
    return write(tmp_path / "refs.faa", {"REF1": REF, "OTHER": OTHER})


@pytest.fixture
def variants(tmp_path) -> str:
    return write(
        tmp_path / "vars.faa",
        {
            "edited": edited(),
            "point": point_mutant(),
            "same": REF,
            "stranger": protein(99, 180),
        },
    )


@pytest.mark.usefixtures("engine")
class TestRefusals:
    def test_neither_source_is_refused(self, tmp_path, variants):
        assert "exactly one of references and database" in refusal(
            {"variants": variants}, tmp_path
        )

    def test_both_sources_are_refused(self, tmp_path, variants, panel):
        message = refusal(
            {
                "variants": variants,
                "references": panel,
                "database": "swissprot-2026_03",
            },
            tmp_path,
        )
        assert "exactly one of references and database" in message

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"min_identity": 2}, "min_identity must be between 0 and 1"),
            ({"min_coverage": -1}, "min_coverage must be between 0 and 1"),
            ({"max_evalue": 0}, "max_evalue must be above 0"),
            ({"threads": 0}, "threads must be an integer from 1 to 64"),
            ({"sensitivity": 8}, "sensitivity must be between 1 and 7.5"),
        ],
    )
    def test_a_setting_out_of_range_is_refused(
        self, tmp_path, variants, panel, change, message
    ):
        params = {"variants": variants, "references": panel, **change}
        assert message in refusal(params, tmp_path)

    def test_a_reference_that_is_too_short_is_refused_with_its_name(
        self, tmp_path, variants
    ):
        short = write(tmp_path / "short.faa", {"tiny": "ACDE"})
        message = refusal({"variants": variants, "references": short}, tmp_path)
        assert "reference tiny" in message


class TestAgainstAPanel:
    @pytest.fixture
    def outcome(self, tmp_path, engine, variants, panel):
        return run_ok(
            {"variants": variants, "references": panel, "threads": 2}, tmp_path
        )

    @pytest.fixture
    def calls(self, outcome):
        return {c["id"]: c for c in outcome.outputs("call")["variants"]}

    def test_the_closest_reference_is_the_panel_member_it_derives_from(self, calls):
        assert calls["edited"]["reference"] == "REF1"
        assert calls["point"]["reference"] == "REF1"

    def test_substitutions_are_named_by_reference_residue_and_position(self, calls):
        edit = calls["edited"]
        assert (
            edit["notation"] == f"{REF[20]}21{swap(REF[20])}:{REF[50]}51{swap(REF[50])}"
        )
        assert calls["point"]["notation"] == f"{REF[119]}120{swap(REF[119])}"

    def test_an_insertion_follows_the_reference_residue_before_it(self, calls):
        assert calls["edited"]["insertions"] == [
            {"after_position": 70, "residues": "GGG"}
        ]

    def test_a_deletion_gives_the_reference_residues_the_variant_lacks(self, calls):
        assert calls["edited"]["deletions"] == [
            {"start": 98, "end": 100, "residues": REF[97:100]}
        ]

    def test_residues_beyond_the_alignment_are_counted(self, calls):
        edit = calls["edited"]
        assert edit["unaligned_variant_start"] == 3
        assert edit["unaligned_variant_end"] == 4
        assert edit["unaligned_reference_start"] == 0

    def test_a_variant_equal_to_its_reference_is_identical(self, calls):
        assert calls["same"]["identical"] is True
        assert calls["same"]["notation"] == ""
        assert calls["point"]["identical"] is False

    def test_a_variant_without_a_reference_is_returned_empty(self, outcome, calls):
        stranger = calls["stranger"]
        assert stranger["reference"] is None and stranger["identity"] is None
        out = outcome.outputs("call")
        assert (out["matched"], out["unmatched"]) == (3, 1)
        assert any("stranger" in n for n in outcome.steps["call"].notes)

    def test_the_source_and_settings_are_recorded(self, outcome):
        record = json.loads(Path(outcome.outputs("call")["record"]).read_text())
        assert record["source"] == "references 2 sequences"
        assert record["settings"]["min_identity"] == 0.5

    def test_the_report_lists_each_variant_with_its_differences(self, outcome):
        from alambiq.capabilities.characterize.protein_call_substitutions_report import (
            render,
        )

        html = render(outcome.steps["call"].result)
        assert "3 of 4 variants have a closest reference" in html
        assert "insertion GGG after 70" in html
        assert "deletion 98-100" in html


class TestAgainstSwissProt:
    @pytest.fixture
    def toy(self, tmp_path, monkeypatch, engine):
        from alambiq.core import resources

        binary = mmseqs_binary()
        if binary is None:
            pytest.skip("mmseqs is not available to build a test database")
        root = tmp_path / "swissprot"
        root.mkdir()
        entries = {"P00010": REF, "P00020": OTHER}
        fasta = tmp_path / "sp.faa"
        fasta.write_text(
            "".join(f">sp|{a}|{a}_TEST x\n{s}\n" for a, s in entries.items())
        )
        subprocess.run(
            [binary, "createdb", str(fasta), str(root / "swissprot-db")],
            check=True,
            capture_output=True,
        )
        lines = ["\t".join(COLUMNS)] + [
            "\t".join(
                [
                    a,
                    f"Protein {a}",
                    "Homo sapiens (Human)",
                    str(len(s)),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
            for a, s in entries.items()
        ]
        (root / "annotations.tsv").write_text("\n".join(lines) + "\n")
        monkeypatch.setattr(resources, "verify", lambda name: root)
        monkeypatch.setattr(
            resources,
            "spec",
            lambda name: {"logical_id": "swissprot-2026_03-v1", "version": "2026_03"},
        )
        monkeypatch.setattr(resources, "identity", lambda name: "cd" * 32)
        return root

    def test_the_reference_carries_its_accession_and_recommended_name(
        self, tmp_path, toy, variants
    ):
        outcome = run_ok(
            {"variants": variants, "database": "swissprot-2026_03", "threads": 2},
            tmp_path,
        )
        point = next(
            c for c in outcome.outputs("call")["variants"] if c["id"] == "point"
        )
        assert point["reference"] == "P00010"
        assert point["reference_name"] == "Protein P00010"
        assert point["notation"] == f"{REF[119]}120{swap(REF[119])}"

    def test_the_source_names_the_database(self, tmp_path, toy, variants):
        outcome = run_ok(
            {"variants": variants, "database": "swissprot-2026_03", "threads": 2},
            tmp_path,
        )
        assert outcome.outputs("call")["source"] == "database swissprot-2026_03-v1"


class TestThresholds:
    def test_a_stricter_identity_removes_the_distant_variant(
        self, tmp_path, engine, variants, panel
    ):
        outcome = run_ok(
            {
                "variants": variants,
                "references": panel,
                "min_identity": 0.99,
                "threads": 2,
            },
            tmp_path,
        )
        calls = {c["id"]: c for c in outcome.outputs("call")["variants"]}
        assert calls["same"]["reference"] == "REF1"
        assert calls["edited"]["reference"] is None


class TestTheClosestReference:
    @pytest.fixture
    def outcome(self, tmp_path, engine):
        references = write(
            tmp_path / "ties.faa",
            {"A_LONGER": "MKVLAAGIVG" + REF, "B_EXACT": REF, "C_OTHER": OTHER},
        )
        variants = write(tmp_path / "v.faa", {"same": REF, "point": point_mutant()})
        return run_ok(
            {"variants": variants, "references": references, "threads": 2}, tmp_path
        )

    @pytest.fixture
    def calls(self, outcome):
        return {c["id"]: c for c in outcome.outputs("call")["variants"]}

    def test_a_reference_with_extra_residues_loses_to_the_exact_one_at_equal_score(
        self, calls
    ):
        assert calls["same"]["reference"] == "B_EXACT"
        assert calls["same"]["differences"] == 0

    def test_the_closest_has_the_fewest_differences_and_the_others_are_alternatives(
        self, calls
    ):
        point = calls["point"]
        assert point["reference"] == "B_EXACT"
        assert point["differences"] == 1
        alternative = point["alternatives"][0]
        assert alternative["reference"] == "A_LONGER"
        assert alternative["differences"] == 11

    def test_a_variant_without_a_reference_has_no_difference_count(
        self, tmp_path, engine, panel
    ):
        stranger = write(tmp_path / "s.faa", {"stranger": protein(99, 180)})
        outcome = run_ok(
            {"variants": stranger, "references": panel, "threads": 2}, tmp_path
        )
        assert outcome.outputs("call")["variants"][0]["differences"] is None
