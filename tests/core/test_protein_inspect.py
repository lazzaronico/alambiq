"""protein.inspect, through the new core, against answers known independently.

The structure is 1L2Y, the Trp-cage miniprotein, a 20 residue NMR model whose
deposited file declares its own helices. DSSP, computed here from coordinates,
is checked against that independent annotation rather than against itself.
"""

from __future__ import annotations

import hashlib
import json
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not find_spec("gemmi"), reason="the dossier extra is required")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "structures" / "1L2Y.cif"
SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"
THREE = dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    strict=True,
))


def write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def fasta(tmp_path: Path, sequence: str = SEQUENCE, name: str = "q.fa") -> str:
    return write(tmp_path, name, f">trpcage\n{sequence}\n")


def inspect_step(**params) -> dict:
    return {"id": "look", "capability": "protein.inspect", "params": params}


def run(step: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    return execute({"id": "inspect", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(step: dict, tmp_path: Path, name: str = "run"):
    outcome = run(step, tmp_path, name)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def dossier(outcome) -> dict:
    return json.loads(Path(outcome.outputs("look")["record"]).read_text())


class TestSequenceOnly:
    def test_the_query_is_identified_by_its_digest(self, tmp_path):
        out = run_ok(inspect_step(fasta=fasta(tmp_path)), tmp_path).outputs("look")
        assert out["identifier"] == "trpcage"
        assert out["sequence_length"] == 20
        assert out["sequence_sha256"] == hashlib.sha256(SEQUENCE.encode()).hexdigest()
        assert out["chains"] == 0 and out["reference_features"] == 0

    def test_nothing_is_requested_means_nothing_is_run(self, tmp_path):
        outcome = run(inspect_step(), tmp_path)
        assert outcome.steps["look"].status == "failed"
        assert "fasta file, a UniProt record, or both" in outcome.steps["look"].error

    def test_an_ambiguous_letter_is_refused_not_guessed(self, tmp_path):
        outcome = run(inspect_step(fasta=fasta(tmp_path, "NLY1QWL")), tmp_path)
        assert outcome.steps["look"].status == "failed"


def reference(sequence: str) -> dict:
    def location(a, b, end="EXACT"):
        return {"start": {"value": a, "modifier": "EXACT"}, "end": {"value": b, "modifier": end}}

    return {
        "primaryAccession": "P99999",
        "sequence": {"value": sequence},
        "features": [
            {"type": "Domain", "description": "inside", "location": location(5, 10)},
            {"type": "Region", "description": "before the query", "location": location(1, 2)},
            {"type": "Site", "description": "uncertain end", "location": location(6, 8, "UNKNOWN")},
        ],
    }


class TestReferenceFeatures:
    REF = "MKTAYIAKQRQISFVKSHFSRQ"

    def test_a_feature_moves_by_the_query_offset_and_unmappable_ones_are_withheld(
        self, tmp_path
    ):
        query = self.REF[2:16]  # starts at reference position 3
        outcome = run_ok(
            inspect_step(
                fasta=fasta(tmp_path, query),
                uniprot=write(tmp_path, "ref.json", json.dumps(reference(self.REF))),
            ),
            tmp_path,
        )
        out = outcome.outputs("look")
        assert (out["reference_features"], out["mapped_features"]) == (3, 1)
        domain, region, site = dossier(outcome)["features"]
        # Reference 5-10 with the query beginning at reference 3: query 3-8.
        assert (domain["query_start"], domain["query_end"]) == (3, 8)
        assert region["mapping"] == "withheld" and site["mapping"] == "withheld"
        assert any("withheld" in n for n in outcome.steps["look"].notes)

    def test_the_reference_alone_defines_the_query(self, tmp_path):
        outcome = run_ok(
            inspect_step(uniprot=write(tmp_path, "ref.json", json.dumps(reference(self.REF)))),
            tmp_path,
        )
        assert outcome.outputs("look")["identifier"] == "P99999"
        assert outcome.outputs("look")["sequence_length"] == len(self.REF)

    def test_attribution_travels_with_the_record(self, tmp_path):
        outcome = run_ok(
            inspect_step(
                uniprot=write(tmp_path, "ref.json", json.dumps(reference(self.REF))),
                source_notes={
                    "uniprot": {
                        "source_url": "https://rest.uniprot.org/uniprotkb/P99999",
                        "license": "CC-BY-4.0",
                        "citation": "UniProt Consortium",
                    }
                },
            ),
            tmp_path,
        )
        source = dossier(outcome)["sources"]["uniprot"]
        assert source["license"] == "CC-BY-4.0"
        assert source["source_url"].endswith("P99999")
        assert source["receipt"] == {"citation": "UniProt Consortium"}

    def test_a_note_for_an_input_that_does_not_exist_is_refused(self, tmp_path):
        outcome = run(
            inspect_step(fasta=fasta(tmp_path), source_notes={"structur": {"license": "x"}}),
            tmp_path,
        )
        assert outcome.steps["look"].status == "failed"
        assert "structur" in outcome.steps["look"].error


class TestStructureMapping:
    def test_a_chain_that_contains_the_query_exactly_maps_every_residue(self, tmp_path):
        out = run_ok(
            inspect_step(fasta=fasta(tmp_path), structure=str(FIXTURE)), tmp_path
        ).outputs("look")
        assert (out["chains"], out["mapped_chains"]) == (1, 1)
        assert out["chain_mappings"][0]["mapping"] == "exact"
        assert out["chain_mappings"][0]["query_offset"] == 0
        assert out["mapped_residues"] == out["observed_residues"] == 20

    def test_a_query_inside_the_chain_maps_only_the_residues_it_covers(self, tmp_path):
        out = run_ok(
            inspect_step(fasta=fasta(tmp_path, SEQUENCE[4:14]), structure=str(FIXTURE)),
            tmp_path,
        ).outputs("look")
        assert out["chain_mappings"][0]["reason"] == "unique-exact-query-within-polymer"
        # Residues 5 to 14 are in the query; the other ten are kept but withheld.
        assert (out["mapped_residues"], out["observed_residues"]) == (10, 20)

    def test_one_substitution_withholds_the_chain_and_says_so(self, tmp_path):
        mutated = SEQUENCE[:4] + "A" + SEQUENCE[5:]  # Q5A
        outcome = run_ok(
            inspect_step(fasta=fasta(tmp_path, mutated), structure=str(FIXTURE)), tmp_path
        )
        out = outcome.outputs("look")
        assert out["mapped_chains"] == 0 and out["mapped_residues"] == 0
        assert out["chain_mappings"][0]["mapping"] == "withheld"
        assert any("No chain of the structure maps onto the query" in n for n in outcome.steps["look"].notes)

    def test_a_model_the_file_does_not_have_is_refused(self, tmp_path):
        outcome = run(
            inspect_step(fasta=fasta(tmp_path), structure=str(FIXTURE), model_number=99),
            tmp_path,
        )
        assert outcome.steps["look"].status == "failed"
        assert "model number is absent" in outcome.steps["look"].error

    def test_a_later_model_of_an_nmr_ensemble_can_be_chosen(self, tmp_path):
        first = run_ok(inspect_step(fasta=fasta(tmp_path), structure=str(FIXTURE)), tmp_path, "m1")
        fifth = run_ok(
            inspect_step(fasta=fasta(tmp_path), structure=str(FIXTURE), model_number=5),
            tmp_path,
            "m5",
        )
        # Same residues, different coordinates.
        first_x = [a["x"] for a in dossier(first)["atoms"]]
        fifth_x = [a["x"] for a in dossier(fifth)["atoms"]]
        assert len(first_x) == len(fifth_x) and first_x != fifth_x

    def test_a_structure_is_not_needed_for_annotations_to_be_refused_without_one(
        self, tmp_path
    ):
        outcome = run(inspect_step(fasta=fasta(tmp_path), coordinate_annotations=True), tmp_path)
        assert outcome.steps["look"].status == "failed"
        assert "need a structure" in outcome.steps["look"].error


@pytest.mark.skipif(not find_spec("mdtraj"), reason="the trajectory extra is required")
class TestCoordinateAnnotations:
    def declared_helices(self) -> list[tuple[str, int, int]]:
        """The helices the deposited file itself declares, with their class."""
        import gemmi

        block = gemmi.cif.read_string(FIXTURE.read_text()).sole_block()
        table = block.get_mmcif_category("_struct_conf.")
        return [
            (table["pdbx_PDB_helix_class"][i], int(table["beg_label_seq_id"][i]), int(table["end_label_seq_id"][i]))
            for i in range(len(table["id"]))
        ]

    def test_dssp_recovers_the_helices_the_file_declares(self, tmp_path):
        outcome = run_ok(
            inspect_step(
                fasta=fasta(tmp_path), structure=str(FIXTURE), coordinate_annotations=True
            ),
            tmp_path,
        )
        rows = json.loads(Path(outcome.outputs("look")["annotation_record"]).read_text())["residues"]
        assigned = {r["label_seq_id"]: r["secondary_structure"] for r in rows}

        (alpha_class, a_start, a_end), (tenth_class, t_start, t_end) = self.declared_helices()
        # Class 1 is a right-handed alpha helix; class 5 is a 3-10 helix.
        assert (alpha_class, tenth_class) == ("1", "5")
        alpha = [assigned[i] for i in range(a_start, a_end + 1)]
        tenth = [assigned[i] for i in range(t_start, t_end + 1)]
        # Strands and ends fray: most, not all, of each declared span is recovered.
        assert alpha.count("H") >= 7 and "E" not in alpha
        assert tenth.count("G") >= 4

    def test_the_composition_and_surface_are_summarised(self, tmp_path):
        out = run_ok(
            inspect_step(
                fasta=fasta(tmp_path), structure=str(FIXTURE), coordinate_annotations=True
            ),
            tmp_path,
        ).outputs("look")
        assert sum(out["secondary_structure"].values()) == 20
        assert out["secondary_structure"]["H"] >= 7
        # A 20 residue miniprotein buries little: every residue is partly exposed.
        assert 20 < out["mean_surface_area"] < 250

    def test_the_library_that_computed_it_is_recorded(self, tmp_path):
        import mdtraj

        outcome = run_ok(
            inspect_step(
                fasta=fasta(tmp_path), structure=str(FIXTURE), coordinate_annotations=True
            ),
            tmp_path,
        )
        record = json.loads(Path(outcome.outputs("look")["annotation_record"]).read_text())
        assert record["mdtraj_version"] == mdtraj.__version__


class TestDeclaredConfidence:
    def modelcif(self, tmp_path: Path) -> str:
        """1L2Y with a declared per-residue pLDDT of 50 + 2 x position."""
        rows = "".join(
            f"{i} 1 A {i} {THREE[aa]} 1 {50 + 2 * i}\n" for i, aa in enumerate(SEQUENCE, 1)
        )
        metrics = (
            "\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.name\n_ma_qa_metric.type\n"
            "_ma_qa_metric.mode\n1 pLDDT pLDDT local\n"
            "loop_\n_ma_qa_metric_local.ordinal_id\n_ma_qa_metric_local.model_id\n"
            "_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_seq_id\n"
            "_ma_qa_metric_local.label_comp_id\n_ma_qa_metric_local.metric_id\n"
            "_ma_qa_metric_local.metric_value\n" + rows
        )
        return write(tmp_path, "model.cif", FIXTURE.read_text().rstrip("\n") + "\n" + metrics)

    def test_the_declared_values_are_summarised_exactly(self, tmp_path):
        out = run_ok(
            inspect_step(fasta=fasta(tmp_path), structure=self.modelcif(tmp_path)), tmp_path
        ).outputs("look")
        c = out["confidence"]
        # Values 52, 54, ..., 90 over positions 1 to 20.
        assert (c["residues"], c["missing"]) == (20, 0)
        assert (c["lowest"], c["highest"]) == (52.0, 90.0)
        assert c["mean"] == pytest.approx(71.0)

    def test_an_experimental_structure_declares_none_and_none_is_invented(self, tmp_path):
        out = run_ok(
            inspect_step(fasta=fasta(tmp_path), structure=str(FIXTURE)), tmp_path
        ).outputs("look")
        # B-factors are present in the file; they are not read as confidence.
        assert out["confidence"] is None and out["confidence_record"] is None


class TestPresentation:
    def test_the_report_says_what_mapped(self, tmp_path):
        outcome = run_ok(inspect_step(fasta=fasta(tmp_path), structure=str(FIXTURE)), tmp_path)
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "unique exact subsequence" in page
        assert "20 of 20 observed polymer residues" in page

    def test_a_second_run_reuses_the_result_and_still_renders_it(self, tmp_path):
        step = inspect_step(fasta=fasta(tmp_path), structure=str(FIXTURE))
        run_ok(step, tmp_path)
        again = run_ok(step, tmp_path)
        assert again.steps["look"].reused
        assert "20 of 20 observed polymer residues" in again.report(tmp_path / "r.html").read_text()
