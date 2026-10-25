"""genome.inspect_context against a genome small enough to read at a glance.

    contig ctg1, bases 1 to 20,000
      cdsA  1,000- 1,999  +   protein pA
      cdsB  3,000- 3,999  -   protein pB
      trnaC 4,500- 4,575  +   (an RNA, no protein)
      cdsD  9,000- 9,999  +   protein pD
      cdsE 19,500-19,999  +   protein pE

Every window below is worked out by hand from those coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

GFF = "\n".join(
    [
        "##gff-version 3",
        "##sequence-region ctg1 1 20000",
        "ctg1\ttest\tCDS\t1000\t1999\t.\t+\t0\tID=cdsA;protein_id=pA;locus_tag=LA",
        "ctg1\ttest\tCDS\t3000\t3999\t.\t-\t0\tID=cdsB;protein_id=pB;locus_tag=LB",
        "ctg1\ttest\ttRNA\t4500\t4575\t.\t+\t.\tID=trnaC",
        "ctg1\ttest\tCDS\t9000\t9999\t.\t+\t0\tID=cdsD;protein_id=pD;locus_tag=LD",
        "ctg1\ttest\tCDS\t19500\t19999\t.\t+\t0\tID=cdsE;protein_id=pE;locus_tag=LE",
    ]
) + "\n"
SEQUENCES = {"pA": "MKVAAA", "pB": "MKVCCC", "pD": "MKVDDD", "pE": "MKVEEE"}


def fasta(sequences: dict[str, str]) -> str:
    return "".join(f">{k}\n{v}\n" for k, v in sequences.items())


def write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def step(tmp_path: Path, gff: str = GFF, **params) -> dict:
    return {
        "id": "ctx",
        "capability": "genome.inspect_context",
        "params": {
            "gff": write(tmp_path, "genome.gff3", gff),
            "proteins": write(tmp_path, "proteins.faa", fasta(SEQUENCES)),
            "genome_id": "demo",
            "protein_id_attribute": "protein_id",
            **params,
        },
    }


def run(s: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    return execute({"id": "g", "steps": [s]}, run_dir=tmp_path / name)


def run_ok(s: dict, tmp_path: Path, name: str = "run"):
    outcome = run(s, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def windows(outcome) -> dict[str, dict]:
    return {w["anchor_id"]: w for w in outcome.outputs("ctx")["windows"]}


class TestWindows:
    def test_a_window_reaches_flank_bases_either_side_and_clips_at_the_contig(
        self, tmp_path
    ):
        outcome = run_ok(step(tmp_path, anchor_ids=["pA", "pE"], flank_bp=1100), tmp_path)
        w = windows(outcome)
        # pA spans 1000-1999: 1000 - 1100 falls off the start, 1999 + 1100 = 3099.
        assert (w["pA"]["start"], w["pA"]["end"]) == (1, 3099)
        assert (w["pA"]["left_clipped"], w["pA"]["right_clipped"]) == (True, False)
        # pE spans 19500-19999: 19500 - 1100 = 18400; the contig ends at 20000.
        assert (w["pE"]["start"], w["pE"]["end"]) == (18400, 20000)
        assert (w["pE"]["left_clipped"], w["pE"]["right_clipped"]) == (False, True)
        assert w["pA"]["contig"] == "ctg1"

    def test_only_features_that_overlap_the_window_are_shown(self, tmp_path):
        outcome = run_ok(step(tmp_path, anchor_ids=["pA"], flank_bp=1100), tmp_path)
        features = json.loads(Path(outcome.outputs("ctx")["record"]).read_text())[
            "neighbourhoods"
        ][0]["features"]
        # cdsB starts at 3000, inside 1-3099; the tRNA at 4500 is out of range.
        assert [f["key"] for f in features] == ["id:cdsA", "id:cdsB"]

    def test_no_flank_shows_only_the_anchor_itself(self, tmp_path):
        outcome = run_ok(step(tmp_path, anchor_ids=["pD"], flank_bp=0), tmp_path)
        w = windows(outcome)["pD"]
        assert (w["start"], w["end"], w["features"]) == (9000, 9999, 1)
        assert not (w["left_clipped"] or w["right_clipped"])

    def test_a_larger_flank_reaches_the_rna_between_the_genes(self, tmp_path):
        outcome = run_ok(step(tmp_path, anchor_ids=["pB"], flank_bp=1000), tmp_path)
        # 3000 - 1000 = 2000 to 3999 + 1000 = 4999: cdsA (1000-1999) just misses,
        # cdsB is the anchor, and the tRNA at 4500-4575 is inside.
        assert windows(outcome)["pB"]["features"] == 2

    def test_a_flank_beyond_the_supported_range_is_refused(self, tmp_path):
        outcome = run(step(tmp_path, anchor_ids=["pA"], flank_bp=100_001), tmp_path)
        assert "flank_bp must be an integer from 0 to 100000" in outcome.steps["ctx"].error


class TestAnchors:
    def test_an_identifier_no_cds_carries_is_unmapped_and_gets_no_window(self, tmp_path):
        outcome = run_ok(step(tmp_path, anchor_ids=["pA", "pZ"], flank_bp=100), tmp_path)
        status = {a["id"]: a["status"] for a in outcome.outputs("ctx")["anchors"]}
        assert status == {"pA": "mapped", "pZ": "unmapped"}
        assert set(windows(outcome)) == {"pA"}
        assert any("pZ" in n and "match no CDS feature" in n for n in outcome.steps["ctx"].notes)

    def test_an_identifier_on_two_features_is_ambiguous_and_gets_a_window_for_each(
        self, tmp_path
    ):
        gff = GFF + "ctg1\ttest\tCDS\t12000\t12500\t.\t+\t0\tID=cdsF;protein_id=pA\n"
        outcome = run_ok(step(tmp_path, gff, anchor_ids=["pA"], flank_bp=100), tmp_path)
        anchor = outcome.outputs("ctx")["anchors"][0]
        assert (anchor["status"], anchor["loci"]) == ("ambiguous", 2)
        assert len(outcome.outputs("ctx")["windows"]) == 2

    def test_the_chosen_attribute_decides_what_an_identifier_means(self, tmp_path):
        by_id = run_ok(
            step(tmp_path, anchor_ids=["cdsA"], protein_id_attribute="ID", flank_bp=0), tmp_path, "id"
        )
        assert by_id.outputs("ctx")["anchors"][0]["status"] == "mapped"
        wrong = run_ok(
            step(tmp_path, anchor_ids=["cdsA"], protein_id_attribute="protein_id", flank_bp=0),
            tmp_path,
            "wrong",
        )
        assert wrong.outputs("ctx")["anchors"][0]["status"] == "unmapped"
        by_locus = run_ok(
            step(tmp_path, anchor_ids=["LD"], protein_id_attribute="locus_tag", flank_bp=0),
            tmp_path,
            "locus",
        )
        assert by_locus.outputs("ctx")["anchors"][0]["status"] == "mapped"

    def test_candidate_proteins_are_checked_against_the_panel(self, tmp_path):
        candidates = write(
            tmp_path, "cand.faa", fasta({"pA": SEQUENCES["pA"], "pB": "MKVWYYYY"})
        )
        outcome = run_ok(step(tmp_path, candidates=candidates, flank_bp=100), tmp_path)
        status = {a["id"]: a["status"] for a in outcome.outputs("ctx")["anchors"]}
        # pB is a candidate whose sequence differs from the genome's pB.
        assert status == {"pA": "mapped", "pB": "sequence_mismatch"}
        assert set(windows(outcome)) == {"pA"}

    def test_anchors_and_candidates_together_are_refused(self, tmp_path):
        candidates = write(tmp_path, "cand.faa", fasta({"pA": SEQUENCES["pA"]}))
        outcome = run(step(tmp_path, anchor_ids=["pA"], candidates=candidates), tmp_path)
        assert "not both" in outcome.steps["ctx"].error

    def test_no_anchors_at_all_is_refused(self, tmp_path):
        outcome = run(step(tmp_path), tmp_path)
        assert "Supply 1–50 unique anchor_ids" in outcome.steps["ctx"].error

    def test_repeated_anchors_are_refused(self, tmp_path):
        outcome = run(step(tmp_path, anchor_ids=["pA", "pA"]), tmp_path)
        assert outcome.steps["ctx"].status == "failed"


class TestProteinExport:
    def test_the_neighbourhoods_proteins_come_back_as_a_fasta(self, tmp_path):
        outcome = run_ok(step(tmp_path, anchor_ids=["pA"], flank_bp=1100), tmp_path)
        out = outcome.outputs("ctx")
        # The window holds cdsA and cdsB: their two proteins, in panel order.
        assert out["neighbourhood_proteins"] == 2
        assert Path(out["neighbourhood_fasta"]).read_text() == ">pA\nMKVAAA\n>pB\nMKVCCC\n"

    def test_nothing_is_exported_when_no_anchor_was_found(self, tmp_path):
        outcome = run_ok(step(tmp_path, anchor_ids=["pZ"]), tmp_path)
        assert outcome.outputs("ctx")["neighbourhood_fasta"] is None
        assert outcome.outputs("ctx")["neighbourhood_proteins"] == 0

    def test_the_exported_proteins_feed_a_later_step(self, tmp_path):
        """The FASTA is an ordinary file, so it binds into another capability."""
        alignment_step = {
            "id": "msa",
            "capability": "sequence.inspect_msa",
            "bind": {"alignment": "ctx.neighbourhood_fasta"},
            "params": {"format": "fasta"},
        }
        # pA and pB are both six residues, so the exported file is a valid
        # two-row alignment and the second step can read it.
        from alambiq.api import run as execute

        outcome = execute(
            {
                "id": "chain",
                "steps": [step(tmp_path, anchor_ids=["pA"], flank_bp=1100), alignment_step],
            },
            run_dir=tmp_path / "chain",
        )
        assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
        assert (outcome.outputs("msa")["rows"], outcome.outputs("msa")["columns"]) == (2, 6)


class TestInputsAndPresentation:
    def test_a_gff_that_is_not_gff3_is_refused(self, tmp_path):
        outcome = run(step(tmp_path, "not a gff\n", anchor_ids=["pA"]), tmp_path)
        assert outcome.steps["ctx"].status == "failed"

    def test_a_circular_contig_is_labelled_and_does_not_wrap(self, tmp_path):
        gff = (
            "##gff-version 3\n##sequence-region ring 1 5000\n"
            "ring\ttest\tregion\t1\t5000\t.\t+\t.\tID=ring;Is_circular=true\n"
            "ring\ttest\tCDS\t100\t400\t.\t+\t0\tID=cdsR;protein_id=pR\n"
        )
        s = step(tmp_path, gff, anchor_ids=["pR"], flank_bp=300)
        s["params"]["proteins"] = write(tmp_path, "r.faa", fasta({"pR": "MKVRRR"}))
        outcome = run_ok(s, tmp_path)
        w = windows(outcome)["pR"]
        assert w["circular"] is True
        # 100 - 300 would wrap below the origin; the window stops at base 1.
        assert w["start"] == 1

    def test_the_report_states_the_coordinate_basis(self, tmp_path):
        outcome = run_ok(step(tmp_path, anchor_ids=["pA"], flank_bp=1100), tmp_path)
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "one-based and inclusive" in page
        assert "1–3,099" in page
