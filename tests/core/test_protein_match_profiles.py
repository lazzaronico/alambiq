"""protein.match_profiles: Stockholm reading, hit reading, and real searches.

The searches use a small family built for the test: eight variants of a
60-residue motif, aligned without gaps. A protein that carries the motif
between flanks of unrelated residues must match it, inside the span the motif
occupies; one that does not carry it must not. Searches against Pfam 38.2 run
only where that resource is installed.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

import pytest

from alambiq.capabilities.characterize.science.profiles import (
    parse_hits,
    parse_stockholm,
)

AMINO = "ACDEFGHIKLMNPQRSTVWY"
STO = """# STOCKHOLM 1.0
#=GF ID Example
#=GF AC PF99999.1
#=GF DE Example profile
a ACDEFGHIKL
b ACDEYGHIKL
//
"""


def residues(seed: int, length: int) -> str:
    return "".join(random.Random(seed).choices(AMINO, k=length))


MOTIF = residues(11, 60)


def variant(seed: int) -> str:
    rng = random.Random(seed)
    return "".join(
        rng.choice([a for a in AMINO if a != c]) if rng.random() < 0.12 else c
        for c in MOTIF
    )


def family(accession: str, name: str, members: list[str], extra: str = "") -> str:
    lines = ["# STOCKHOLM 1.0", f"#=GF ID {name}", f"#=GF AC {accession}", f"#=GF DE {name} family"]
    lines += [f"#=GF {e}" for e in extra.split(";") if e]
    lines += [f"m{i} {m}" for i, m in enumerate(members)]
    return "\n".join(lines) + "\n//\n"


FAMILY_A = family("PF00001.1", "Motif", [MOTIF] + [variant(s) for s in range(1, 8)], "CL CL0001;TP Domain")
FAMILY_B = family("PF00002.1", "Other", [residues(99, 60)] + [residues(99, 60)[:30] + residues(s, 30) for s in range(5)])
FLANK_LEFT, FLANK_RIGHT = residues(21, 3), residues(22, 20)
CARRIER = FLANK_LEFT + variant(40) + FLANK_RIGHT


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", os.environ.get("ALAMBIQ_ENV_MMSEQS") or sys.executable)


def files(tmp_path: Path, protein: str, stockholm: str | None = None) -> dict:
    (tmp_path / "q.faa").write_text(f">query\n{protein}\n")
    out = {"protein": str(tmp_path / "q.faa")}
    if stockholm is not None:
        (tmp_path / "p.sto").write_text(stockholm)
        out["profiles"] = str(tmp_path / "p.sto")
    return out


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "match", "capability": "protein.match_profiles", "params": params}
    return execute({"id": "m", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


needs_mmseqs = pytest.mark.skipif(
    not (os.environ.get("ALAMBIQ_ENV_MMSEQS") or shutil.which("mmseqs")),
    reason="set ALAMBIQ_ENV_MMSEQS or put mmseqs on PATH to run the search",
)

class TestStockholm:
    def test_the_identity_and_shape_of_a_profile(self):
        row = parse_stockholm(STO)[0]
        assert (row["accession"], row["name"], row["description"]) == ("PF99999.1", "Example", "Example profile")
        assert (row["sequence_count"], row["alignment_columns"]) == (2, 10)
        assert (row["clan_accession"], row["entry_type"]) == (None, None)

    def test_clan_and_type_are_optional_metadata(self):
        source = STO.replace("#=GF DE", "#=GF CL CL0090\n#=GF TP Domain\n#=GF DE")
        row = parse_stockholm(source)[0]
        assert (row["clan_accession"], row["entry_type"]) == ("CL0090", "Domain")

    def test_interleaved_blocks_annotations_and_lower_case_are_read(self):
        source = (
            "# STOCKHOLM 1.0\n#=GF ID X\n#=GF AC PF1.1\n#=GF DE x\n"
            "#=GS a AC P12345\na acdef\nb ACDEY\n#=GR a SS HHHHH\n#=GC SS_cons HHHHH\n\n"
            "a ghikl\nb GHIKL\n//\n"
        )
        row = parse_stockholm(source)[0]
        assert (row["sequence_count"], row["alignment_columns"]) == (2, 10)

    def test_ambiguous_residues_are_counted(self):
        source = STO.replace("ACDEYGHIKL", "ACDEXGHBZL")
        assert parse_stockholm(source)[0]["ambiguous_residues"] == 3

    def test_several_alignments_are_read_in_order(self):
        rows = parse_stockholm(STO + STO.replace("PF99999.1", "PF99998.1"))
        assert [r["accession"] for r in rows] == ["PF99999.1", "PF99998.1"]

    @pytest.mark.parametrize(
        "bad",
        [
            STO + STO,
            STO.replace("//", ""),
            STO.replace("ACDEYGHIKL", "ACDE"),
            STO.replace("PF99999.1", ""),
            STO.replace("# STOCKHOLM 1.0", "# STOCKHOLM 1.1"),
            STO.replace("#=GF DE", "#=GF CL CL0090\n#=GF CL CL0090\n#=GF DE"),
            STO.replace("#=GF DE", "#=GF CL  \n#=GF DE"),
            STO.replace("#=GF DE", "#=GF CL ../CL0090\n#=GF DE"),
            STO.replace("#=GF DE", "#=GF TP Domain\n#=GF TP Family\n#=GF DE"),
            STO.replace("PF99999.1", "PF/99999"),
            STO.replace("ACDEYGHIKL", "ACDE*GHIKL"),
            STO.replace("a ACDEFGHIKL", "a ACDEF GHIKL"),
            STO.replace("b ACDEYGHIKL\n", ""),
            "",
        ],
    )
    def test_anything_a_profile_cannot_use_is_refused(self, bad):
        with pytest.raises(ValueError):
            parse_stockholm(bad)


CATALOGUE = {
    "PF99999.1": {"accession": "PF99999.1", "name": "Example", "description": "Example", "clan_accession": None, "entry_type": None},
}
ONE = ["q0", "PF99999.1", "1e-9", "40", "2", "9", "1", "8", "12", "10"]


def hits(rows, sequence="M" * 12, **overrides):
    settings = {"evalue": 0.001, "min_profile_coverage": 0.5, "alternatives": 0, **overrides}
    return parse_hits(rows, sequence, CATALOGUE, **settings)


class TestHits:
    def test_coordinates_scores_and_coverage_come_through(self):
        row = hits([ONE])[0]
        assert (row["query_start"], row["query_end"], row["profile_start"], row["profile_end"]) == (2, 9, 1, 8)
        assert (row["evalue"], row["bits"], row["profile_coverage"], row["status"]) == (1e-9, 40.0, 0.8, "selected")

    def test_a_match_covering_too_little_of_the_profile_is_filtered_for_that_reason(self):
        row = hits([ONE], min_profile_coverage=0.81)[0]
        assert (row["status"], row["reasons"]) == ("filtered", ["profile_coverage"])

    def test_a_match_above_the_expectation_threshold_is_filtered_for_that_reason(self):
        row = hits([ONE], evalue=1e-10)[0]
        assert (row["status"], row["reasons"]) == ("filtered", ["evalue"])

    def test_a_profile_that_returned_nothing_has_no_row(self):
        assert hits([]) == []

    def test_a_repeated_family_keeps_each_span_in_protein_order(self):
        second = ["q0", "PF99999.1", "1e-5", "25", "20", "23", "1", "4", "30", "10"]
        rows = hits([second, ["q0", *ONE[1:8], "30", "10"]], sequence="M" * 30, alternatives=1)
        assert [r["query_start"] for r in rows] == [2, 20]
        assert [r["status"] for r in rows] == ["selected", "filtered"]

    @pytest.mark.parametrize(
        "bad",
        [
            [ONE, ONE],
            [["q1", *ONE[1:]]],
            [[*ONE[:2], "nan", *ONE[3:]]],
            [[*ONE[:4], "0", *ONE[5:]]],
            [[*ONE[:8], "11", "10"]],
            [["q0", "PF00000.1", *ONE[2:]]],
            [ONE[:9]],
        ],
    )
    def test_malformed_or_inconsistent_rows_are_refused(self, bad):
        with pytest.raises(ValueError):
            hits(bad)


@needs_mmseqs
class TestSearch:
    @pytest.fixture
    def outcome(self, tmp_path):
        return run_ok({**files(tmp_path, CARRIER, FAMILY_A + FAMILY_B)}, tmp_path)

    def test_a_protein_carrying_the_motif_matches_its_family_inside_the_motif(self, outcome):
        out = outcome.outputs("match")
        assert out["library"] == "supplied Stockholm alignments"
        assert out["profiles_searched"] == 2
        [match] = out["matches"]
        assert match["accession"] == "PF00001.1"
        assert match["status"] == "selected"
        # The motif sits at residues 4 to 63 of the 83-residue query.
        assert 4 <= match["query_start"] <= 12
        assert 55 <= match["query_end"] <= 63
        assert match["profile_coverage"] >= 0.8
        assert match["evalue"] < 1e-5

    def test_the_family_that_is_not_carried_is_not_returned(self, outcome):
        assert [m["accession"] for m in outcome.outputs("match")["matches"]] == ["PF00001.1"]
        assert outcome.outputs("match")["families_returned"] == 1

    def test_the_source_clan_and_type_travel_with_the_match(self, outcome):
        match = outcome.outputs("match")["matches"][0]
        assert (match["clan_accession"], match["entry_type"]) == ("CL0001", "Domain")

    def test_a_protein_without_the_motif_matches_nothing_and_says_so(self, tmp_path):
        outcome = run_ok(files(tmp_path, residues(77, 90), FAMILY_A + FAMILY_B), tmp_path)
        assert outcome.outputs("match")["matches"] == []
        assert "matched at an expectation value" in " ".join(outcome.steps["match"].notes)

    def test_a_fragment_of_the_motif_is_filtered_when_coverage_is_demanded(self, tmp_path):
        fragment = residues(23, 5) + variant(41)[:36] + residues(24, 30)
        lenient = run_ok({**files(tmp_path, fragment, FAMILY_A), "min_profile_coverage": 0.5}, tmp_path, "lenient")
        strict = run_ok({**files(tmp_path, fragment, FAMILY_A), "min_profile_coverage": 0.9}, tmp_path, "strict")
        [kept] = lenient.outputs("match")["matches"]
        [dropped] = strict.outputs("match")["matches"]
        # Only 36 of the 60 motif positions are present, so coverage stays well under
        # 0.9 (a little of the flank may align too) and over 0.5.
        assert kept["status"] == "selected" and 0.5 <= kept["profile_coverage"] <= 0.8
        assert (dropped["status"], dropped["reasons"]) == ("filtered", ["profile_coverage"])
        assert strict.outputs("match")["selected"] == 0 and strict.outputs("match")["filtered"] == 1
        assert "filtered by the expectation or coverage" in " ".join(strict.steps["match"].notes)

    def test_a_repeated_motif_gives_one_alignment_unless_more_are_asked_for(self, tmp_path):
        repeat = residues(25, 4) + variant(42) + residues(26, 25) + variant(43) + residues(27, 4)
        one = run_ok({**files(tmp_path, repeat, FAMILY_A)}, tmp_path, "one")
        two = run_ok({**files(tmp_path, repeat, FAMILY_A), "alternative_alignments": 1}, tmp_path, "two")
        assert len(one.outputs("match")["matches"]) == 1
        spans = sorted((m["query_start"], m["query_end"]) for m in two.outputs("match")["matches"])
        assert len(spans) == 2
        # One copy in each half of the protein; neither span reaches into the other copy.
        assert spans[0][1] < spans[1][0]
        assert spans[0][0] >= 3 and spans[1][1] <= 4 + 60 + 25 + 60

    def test_the_stored_record_names_the_source_construction_and_settings(self, outcome):
        record = json.loads(Path(outcome.outputs("match")["record"]).read_text())
        assert record["profile_source"]["kind"] == "stockholm"
        assert record["construction"] == {"match_mode": 1, "match_ratio": 0.5, "filter_msa": 0, "msa_type": 2}
        assert record["settings"]["evalue"] == 0.001
        assert record["engine_versions"]["mmseqs"]
        assert "one-based inclusive" in record["coordinate_convention"].lower()

    def test_changing_a_supplied_alignment_makes_the_step_run_again(self, tmp_path):
        first = run_ok(files(tmp_path, CARRIER, FAMILY_A), tmp_path, "same")
        (tmp_path / "p.sto").write_text(FAMILY_A.replace("m1 ", "m1 ", 1).replace("Motif family", "Motif family changed"))
        second = run_ok({"protein": str(tmp_path / "q.faa"), "profiles": str(tmp_path / "p.sto")}, tmp_path, "same")
        assert first.steps["match"].status == "completed"
        assert second.steps["match"].status == "completed"
        assert second.outputs("match")["matches"][0]["description"] == "Motif family changed"

    def test_the_report_names_the_library(self, outcome, tmp_path):
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "supplied Stockholm alignments" in page
        assert "PF00001.1" in page


class TestRefusals:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"library": "pfam-38.2"}, "not both and not neither"),
            ({"evalue": 0}, "evalue must be between"),
            ({"sensitivity": 8}, "sensitivity must be between"),
            ({"min_profile_coverage": 1.5}, "min_profile_coverage must be between"),
            ({"alternative_alignments": 21}, "between 0 and 20"),
            ({"alternative_alignments": -1}, "between 0 and 20"),
        ],
    )
    def test_a_request_outside_the_supported_range_is_refused(self, tmp_path, changes, message):
        outcome = run({**files(tmp_path, CARRIER, FAMILY_A), **changes}, tmp_path)
        assert outcome.steps["match"].status == "failed"
        assert message in outcome.steps["match"].error

    def test_neither_profiles_nor_a_library_is_refused(self, tmp_path):
        outcome = run(files(tmp_path, CARRIER), tmp_path)
        assert "not both and not neither" in outcome.steps["match"].error

    def test_a_query_that_is_too_short_is_refused(self, tmp_path):
        outcome = run(files(tmp_path, "MKV", FAMILY_A), tmp_path)
        assert "10–5000 standard protein residues" in outcome.steps["match"].error

    def test_a_malformed_alignment_is_refused_before_any_search(self, tmp_path):
        outcome = run(files(tmp_path, CARRIER, STO.replace("//", "")), tmp_path)
        assert "complete Stockholm alignments" in outcome.steps["match"].error


PFAM = os.environ.get("ALAMBIQ_RESOURCE_PFAM_38_2_PROFILES")
LYSOZYME = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)
GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


def pfam_installed() -> bool:
    from alambiq.core import resources

    return resources.location("pfam-38.2-profiles").is_dir()


@pytest.mark.skipif(not pfam_installed(), reason="the pinned Pfam 38.2 profile library is not installed")
@needs_mmseqs
class TestPfamLibrary:
    @pytest.mark.parametrize(
        ("protein", "accession"),
        [(LYSOZYME, "PF00062"), (GFP, "PF01353")],
        ids=["hen-lysozyme", "avGFP"],
    )
    def test_a_textbook_protein_matches_its_textbook_family(self, tmp_path, protein, accession):
        outcome = run_ok({**files(tmp_path, protein), "library": "pfam-38.2"}, tmp_path)
        out = outcome.outputs("match")
        assert out["profiles_searched"] == 30134
        assert out["library"].startswith("Pfam 38.2")
        selected = [m for m in out["matches"] if m["status"] == "selected"]
        assert any(m["accession"].startswith(accession) for m in selected), [m["accession"] for m in out["matches"]]

    def test_the_record_names_the_library_version_and_identity(self, tmp_path):
        outcome = run_ok({**files(tmp_path, GFP), "library": "pfam-38.2"}, tmp_path)
        record = json.loads(Path(outcome.outputs("match")["record"]).read_text())
        assert record["profile_source"]["version"] == "Pfam38.2"
        assert record["profile_source"]["license"] == "CC0-1.0"
        assert len(record["profile_source"]["identity"]) == 64
