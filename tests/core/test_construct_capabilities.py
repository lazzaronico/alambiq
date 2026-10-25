"""Composing an insert and assembling a construct, through the new core.

The sequences are short enough to check by hand: the composed insert must be
the parts concatenated plus a stop codon, and it must translate to the parts'
proteins joined in order. The assembly uses the real pUC19 vector, so the
product's length is the backbone plus the insert, and its coding sequence must
translate to the expected protein.
"""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

import pytest

VECTOR = Path(__file__).resolve().parents[1] / "fixtures/constructs/puc19.gb"

# Twenty residues, with a terminal stop that composition removes before joining.
PROTEIN_DNA = "ATGAAAGCGTTCACCGGTAACGACGTTCTGGAACTGTACGGTCTGAAAGGTTCTGACGCTTAA"
PROTEIN = "MKAFTGNDVLELYGLKGSDA"
LINKER_DNA = "GGTTCT"
LINKER = "GS"
TAG_DNA = "CACCAC"
TAG = "HH"

CODONS = dict(
    zip(
        "ACDEFGHIKLMNPQRSTVWY",
        "GCT TGT GAT GAA TTT GGT CAT ATT AAA CTG ATG AAT CCT CAA CGT TCT ACT GTT TGG TAT".split(),
        strict=True,
    )
)


def fasta(tmp_path: Path, name: str, label: str, sequence: str) -> str:
    path = tmp_path / name
    path.write_text(f">{label}\n{sequence}\n")
    return str(path)


def run_ok(campaign: dict, tmp_path: Path):
    from alambiq.api import run

    outcome = run(campaign, run_dir=tmp_path / "run")
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


@pytest.mark.skipif(not find_spec("Bio"), reason="Biopython is required")
class TestComposeCoding:
    def coding_parts(self, tmp_path: Path) -> list[dict]:
        return [
            {
                "id": "protein",
                "role": "protein",
                "coding_dna": fasta(tmp_path, "p.fa", "p", PROTEIN_DNA),
                "expected_protein": fasta(tmp_path, "p.faa", "p", PROTEIN),
                "remove_terminal_stop": True,
            },
            {
                "id": "linker",
                "role": "linker",
                "coding_dna": fasta(tmp_path, "l.fa", "l", LINKER_DNA),
                "expected_protein": fasta(tmp_path, "l.faa", "l", LINKER),
            },
            {
                "id": "tag",
                "role": "tag",
                "coding_dna": fasta(tmp_path, "t.fa", "t", TAG_DNA),
                "expected_protein": fasta(tmp_path, "t.faa", "t", TAG),
            },
        ]

    def campaign(self, tmp_path: Path, **params) -> dict:
        return {
            "id": "insert",
            "steps": [
                {
                    "id": "compose",
                    "capability": "construct.compose_coding",
                    "params": {
                        "parts": self.coding_parts(tmp_path),
                        "identifier": "tagged",
                        "stop_codon": "TAA",
                        **params,
                    },
                }
            ],
        }

    def test_the_insert_is_the_parts_joined_with_one_stop_codon(self, tmp_path):
        outcome = run_ok(self.campaign(tmp_path), tmp_path)
        out = outcome.outputs("compose")
        record = json.loads(Path(out["record"]).read_text())
        # The protein part's own stop is removed, then one stop ends the insert.
        assert record["dna"] == PROTEIN_DNA[:-3] + LINKER_DNA + TAG_DNA + "TAA"
        assert record["protein"] == PROTEIN + LINKER + TAG
        assert out["length_bp"] == len(record["dna"])
        assert out["protein_length"] == len(PROTEIN) + len(LINKER) + len(TAG)
        assert out["encoded_from_codon_table"] is False

    def test_each_part_reports_where_it_sits(self, tmp_path):
        out = run_ok(self.campaign(tmp_path), tmp_path).outputs("compose")
        spans = {p["id"]: p for p in out["parts"]}
        assert spans["protein"]["dna_start"] == 1
        assert spans["protein"]["protein_start"] == 1
        assert spans["protein"]["protein_end"] == len(PROTEIN)
        # The record names the codon that was dropped, not just that one was.
        assert spans["protein"]["removed_stop_codon"] == "TAA"
        assert spans["linker"]["removed_stop_codon"] is None
        # Spans are one-based and inclusive, so each follows the last exactly.
        assert spans["linker"]["dna_start"] == spans["protein"]["dna_end"] + 1
        assert spans["tag"]["dna_start"] == spans["linker"]["dna_end"] + 1
        assert spans["tag"]["protein_end"] == out["protein_length"]

    def test_dna_that_encodes_another_protein_is_refused(self, tmp_path):
        from alambiq.api import run

        campaign = self.campaign(tmp_path)
        campaign["steps"][0]["params"]["parts"][1]["expected_protein"] = fasta(
            tmp_path, "wrong.faa", "l", "GG"
        )
        outcome = run(campaign, run_dir=tmp_path / "run")
        assert outcome.steps["compose"].status == "failed"
        assert "translation" in outcome.steps["compose"].error

    def test_protein_parts_are_encoded_from_the_supplied_codon_table(self, tmp_path):
        choices = tmp_path / "codons.json"
        choices.write_text(json.dumps(CODONS))
        parts = [
            {
                "id": part["id"],
                "role": part["role"],
                "expected_protein": part["expected_protein"],
            }
            for part in self.coding_parts(tmp_path)
        ]
        campaign = self.campaign(tmp_path, codon_choices=str(choices))
        campaign["steps"][0]["params"]["parts"] = parts
        out = run_ok(campaign, tmp_path).outputs("compose")
        record = json.loads(Path(out["record"]).read_text())
        assert out["encoded_from_codon_table"] is True
        # Every residue uses exactly the codon the table names.
        expected = "".join(CODONS[residue] for residue in PROTEIN + LINKER + TAG)
        assert record["dna"] == expected + "TAA"

    def test_a_codon_that_encodes_another_residue_is_refused(self, tmp_path):
        from alambiq.api import run

        choices = tmp_path / "bad.json"
        # GCC is alanine; claiming it for cysteine contradicts genetic code 11.
        choices.write_text(json.dumps({**CODONS, "C": "GCC"}))
        parts = [
            {
                "id": part["id"],
                "role": part["role"],
                "expected_protein": part["expected_protein"],
            }
            for part in self.coding_parts(tmp_path)
        ]
        campaign = self.campaign(tmp_path, codon_choices=str(choices))
        campaign["steps"][0]["params"]["parts"] = parts
        outcome = run(campaign, run_dir=tmp_path / "run")
        assert outcome.steps["compose"].status == "failed"
        assert "amino acid" in outcome.steps["compose"].error

    def test_dna_supplied_together_with_a_codon_table_is_refused(self, tmp_path):
        from alambiq.api import run

        choices = tmp_path / "codons.json"
        choices.write_text(json.dumps(CODONS))
        campaign = self.campaign(tmp_path, codon_choices=str(choices))
        outcome = run(campaign, run_dir=tmp_path / "run")
        assert outcome.steps["compose"].status == "failed"
        error = outcome.steps["compose"].error
        assert "Part protein has coding_dna" in error
        assert "together with codon_choices" in error

    def test_remove_terminal_stop_is_not_read_when_a_codon_table_is_given(self, tmp_path):
        choices = tmp_path / "codons.json"
        choices.write_text(json.dumps(CODONS))
        outcomes = {}
        for flag in (False, True):
            parts = [
                {
                    "id": part["id"],
                    "role": part["role"],
                    "expected_protein": part["expected_protein"],
                    "remove_terminal_stop": flag,
                }
                for part in self.coding_parts(tmp_path)
            ]
            campaign = self.campaign(tmp_path, codon_choices=str(choices))
            campaign["steps"][0]["params"]["parts"] = parts
            outcomes[flag] = run_ok(campaign, tmp_path / str(flag)).outputs("compose")
        records = {
            flag: json.loads(Path(out["record"]).read_text())
            for flag, out in outcomes.items()
        }
        assert records[True]["dna"] == records[False]["dna"]
        assert all(p["removed_stop_codon"] is None for p in outcomes[True]["parts"])

    def test_a_part_without_dna_and_without_a_codon_table_is_refused(self, tmp_path):
        from alambiq.api import run

        campaign = self.campaign(tmp_path)
        del campaign["steps"][0]["params"]["parts"][1]["coding_dna"]
        outcome = run(campaign, run_dir=tmp_path / "run")
        assert outcome.steps["compose"].status == "failed"
        assert "codon_choices" in outcome.steps["compose"].error


@pytest.mark.skipif(not find_spec("pydna"), reason="the construct extra is required")
class TestAssemble:
    def campaign(self, tmp_path: Path, **params) -> dict:
        return {
            "id": "construct",
            "steps": [
                {
                    "id": "build",
                    "capability": "construct.assemble",
                    "params": {
                        "vector": str(VECTOR),
                        "insert": fasta(tmp_path, "i.fa", "insert", PROTEIN_DNA),
                        "expected_protein": fasta(tmp_path, "i.faa", "insert", PROTEIN),
                        **params,
                    },
                }
            ],
        }

    def test_the_product_is_the_backbone_plus_the_insert(self, tmp_path):
        from Bio import SeqIO

        out = run_ok(self.campaign(tmp_path), tmp_path).outputs("build")
        vector = SeqIO.read(VECTOR.open(), "genbank")
        # SmaI cuts pUC19 once, bluntly, so the backbone keeps every base.
        assert out["length"] == len(vector) + len(PROTEIN_DNA)
        assert out["protein_id"] == "insert"
        record = json.loads(Path(out["record"]).read_text())
        assert record["topology"] == "circular"
        assert record["translation"] == PROTEIN
        assert record["unique_circular_products"] == 1

    def test_the_designed_primers_carry_their_melting_temperatures(self, tmp_path):
        out = run_ok(self.campaign(tmp_path), tmp_path).outputs("build")
        assert len(out["primers"]) >= 2
        for primer in out["primers"]:
            assert set(primer["sequence"]) <= set("ACGT")
            assert primer["footprint"] in primer["sequence"]
            assert 45 <= primer["footprint_tm_degC"] <= 80

    def test_a_protein_the_product_does_not_encode_is_refused(self, tmp_path):
        from alambiq.api import run

        campaign = self.campaign(tmp_path)
        campaign["steps"][0]["params"]["expected_protein"] = fasta(
            tmp_path, "other.faa", "insert", "M" + "A" * 19
        )
        outcome = run(campaign, run_dir=tmp_path / "run")
        assert outcome.steps["build"].status == "failed"

    def test_an_enzyme_that_does_not_cut_the_vector_is_refused(self, tmp_path):
        from alambiq.api import run

        outcome = run(self.campaign(tmp_path, enzyme="NotI"), run_dir=tmp_path / "run")
        assert outcome.steps["build"].status == "failed"
