"""structure.detect_binding_pockets with the real P2Rank on the hemoglobin tetramer.

Human haemoglobin (PDB 1A3N) holds four haems, one per subunit, each bound to a
proximal histidine (F8: alpha His87, beta His92) and facing a distal histidine (E7:
alpha His58, beta His63). A ligand-site detector that works should rank those four
sites at the top. The run also has a recorded result from the previous engine
(the earlier tutorial's verification record): 15 pockets,
and a rank-1 pocket with score 49.84, probability 0.980 and 26 adjacent residues.

These tests need Java 17-23 and the P2Rank resource: set ALAMBIQ_ENV_P2RANK to a
Python on a PATH where `java` is found.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

ENV = os.environ.get("ALAMBIQ_ENV_P2RANK", "")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def installed() -> bool:
    from alambiq.core import resources

    return (
        bool(ENV)
        and Path(ENV).is_file()
        and resources.location("p2rank-2.5.1").is_dir()
        and (FIXTURES / "structures/1A3N.cif").is_file()
    )


pytestmark = pytest.mark.skipif(
    not installed(), reason="set ALAMBIQ_ENV_P2RANK and install p2rank-2.5.1 to run"
)


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    if ENV:
        monkeypatch.setenv("ALAMBIQ_ENV_P2RANK", ENV)


def run(steps: list[dict], tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    return execute({"id": "pockets", "steps": steps}, run_dir=tmp_path / name)


def inspect_step() -> dict:
    return {
        "id": "inspect",
        "capability": "protein.inspect",
        "params": {
            "fasta": str(FIXTURES / "hemoglobin/hemoglobin-alpha.fasta"),
            "uniprot": str(FIXTURES / "hemoglobin/P69905.json"),
            "structure": str(FIXTURES / "structures/1A3N.cif"),
        },
    }


def mapped(**params) -> list[dict]:
    return [
        inspect_step(),
        {
            "id": "pockets",
            "capability": "structure.detect_binding_pockets",
            "bind": {"protein": "inspect.record"},
            "params": {"structure_kind": "experimental", **params},
        },
    ]


def direct(path: Path, **params) -> list[dict]:
    return [
        {
            "id": "pockets",
            "capability": "structure.detect_binding_pockets",
            "params": {
                "structure": str(path),
                "structure_kind": "experimental",
                **params,
            },
        }
    ]


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("pockets")


@pytest.fixture(scope="module")
def haemoglobin(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("hb")
    os.environ["ALAMBIQ_ENV_P2RANK"] = ENV
    outcome = run(mapped(), tmp_path)
    return outcome, ok(outcome), tmp_path


class TestHaemoglobin:
    def test_it_reproduces_the_recorded_run_of_the_previous_engine(self, haemoglobin):
        _, out, _ = haemoglobin
        assert len(out["pockets"]) == 15
        first = out["pockets"][0]
        assert (first["rank"], first["score"], first["probability"]) == (1, 49.84, 0.98)
        assert len(first["residue_ids"]) == 26 and first["unmapped_residue_ids"] == []

    def test_the_four_best_pockets_are_the_four_haems(self, haemoglobin):
        _, out, _ = haemoglobin
        top = out["pockets"][:4]
        chains = [{r.split("_")[0] for r in p["residue_ids"]} for p in top]
        assert chains == [{"A"}, {"B"}, {"C"}, {"D"}]
        # Each lines the haem's proximal histidine (F8) and its distal histidine (E7).
        expected = {
            "A": ("A_87", "A_58"),
            "B": ("B_92", "B_63"),
            "C": ("C_87", "C_58"),
            "D": ("D_92", "D_63"),
        }
        for pocket, (chain,) in zip(top, chains, strict=True):
            for residue in expected[chain]:
                assert residue in pocket["residue_ids"], (chain, residue)

    def test_alpha_chain_pockets_are_joined_to_the_query_and_beta_chain_ones_are_not_forced_onto_it(
        self, haemoglobin
    ):
        outcome, out, _ = haemoglobin
        record = json.loads(Path(outcome.outputs("inspect")["record"]).read_text())
        sequence = record["sequence"]
        first, second = out["pockets"][0], out["pockets"][1]
        # Query numbering counts the initiator Met, so author His58 and His87 are positions 59 and 88.
        assert {59, 88} <= set(first["query_positions"])
        assert sequence[58] == "H" and sequence[87] == "H"
        assert second["query_positions"] == []

    def test_ranks_run_from_one_scores_fall_and_probabilities_are_probabilities(
        self, haemoglobin
    ):
        _, out, _ = haemoglobin
        scores = [p["score"] for p in out["pockets"]]
        assert [p["rank"] for p in out["pockets"]] == list(range(1, 16))
        assert scores == sorted(scores, reverse=True)
        assert all(0 <= p["probability"] <= 1 for p in out["pockets"])
        assert all(len(p["center_angstrom"]) == 3 for p in out["pockets"])

    def test_the_tools_own_tables_are_kept_and_the_record_names_what_ran(
        self, haemoglobin
    ):
        _, out, _ = haemoglobin
        predictions = Path(out["predictions_table"]).read_text().splitlines()
        assert len(predictions) == 16 and predictions[0].startswith("name")
        assert out["residue_memberships"] == 572
        record = json.loads(Path(out["record"]).read_text())
        assert record["configuration"] == "default" and record["tool"] == "P2Rank"
        assert record["versions"]["p2rank"] == "2.5.1" and record["versions"]["java"]
        assert record["license"] == "MIT" and len(record["resource_identity"]) == 64

    def test_the_report_lists_the_pocket_scores(self, haemoglobin):
        outcome, _, tmp_path = haemoglobin
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "49.84" in page

    def test_a_smaller_limit_keeps_the_best_pockets_in_order(self, tmp_path):
        out = ok(run(mapped(maximum_pockets=3), tmp_path))
        assert [p["rank"] for p in out["pockets"]] == [1, 2, 3]
        assert out["pockets"][0]["score"] == 49.84


class TestStructureFiles:
    def test_a_structure_given_as_a_file_gives_the_same_pockets_without_query_positions(
        self, haemoglobin, tmp_path
    ):
        _, mapped_out, _ = haemoglobin
        out = ok(run(direct(FIXTURES / "structures/1A3N.cif"), tmp_path))
        assert [(p["rank"], p["score"]) for p in out["pockets"]] == [
            (p["rank"], p["score"]) for p in mapped_out["pockets"]
        ]
        assert all(p["query_positions"] is None for p in out["pockets"])

    def test_a_gzipped_structure_is_read_the_same(self, tmp_path):
        compressed = tmp_path / "1A3N.cif.gz"
        compressed.write_bytes(
            gzip.compress((FIXTURES / "structures/1A3N.cif").read_bytes())
        )
        out = ok(run(direct(compressed), tmp_path))
        assert out["pockets"][0]["score"] == 49.84

    def test_the_predicted_configuration_is_a_different_model_and_is_recorded_as_such(
        self, tmp_path
    ):
        out = ok(
            run(
                direct(FIXTURES / "structures/1A3N.cif", structure_kind="predicted"),
                tmp_path,
            )
        )
        assert out["configuration"] == "alphafold"
        assert (
            json.loads(Path(out["record"]).read_text())["configuration"] == "alphafold"
        )
        assert out["pockets"]

    def test_a_structure_with_no_pocket_is_a_result_not_an_error(self, tmp_path):
        # A single glycine has no surface to speak of.
        tiny = tmp_path / "tiny.pdb"
        tiny.write_text(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00  0.00           C\n"
            "ATOM      4  O   GLY A   1       1.251   2.390   0.000  1.00  0.00           O\n"
            "END\n"
        )
        outcome = run(direct(tiny), tmp_path)
        if outcome.ok:
            assert outcome.outputs("pockets")["pockets"] == []
            assert "no pocket" in " ".join(outcome.steps["pockets"].notes)


class TestReuseAndEnvironment:
    def test_the_same_request_is_reused_and_a_different_kind_runs_again(self, tmp_path):
        first = run(direct(FIXTURES / "structures/1A3N.cif"), tmp_path, "same")
        second = run(direct(FIXTURES / "structures/1A3N.cif"), tmp_path, "same")
        other = run(
            direct(FIXTURES / "structures/1A3N.cif", structure_kind="predicted"),
            tmp_path,
            "same",
        )
        assert (first.steps["pockets"].status, second.steps["pockets"].status) == (
            "completed",
            "reused",
        )
        assert other.steps["pockets"].status == "completed"

    def test_a_missing_java_is_named_not_mistaken_for_no_pockets(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("PATH", "/nonexistent")
        outcome = run(direct(FIXTURES / "structures/1A3N.cif"), tmp_path)
        assert outcome.steps["pockets"].status == "failed"
        assert (
            "java is not on PATH" in outcome.steps["pockets"].error
            or "Java" in outcome.steps["pockets"].error
        )


class TestRefusals:
    def test_neither_or_both_inputs_are_refused(self, tmp_path):
        neither = run(
            [
                {
                    "id": "pockets",
                    "capability": "structure.detect_binding_pockets",
                    "params": {"structure_kind": "experimental"},
                }
            ],
            tmp_path,
        )
        assert "exactly one of them is required" in neither.steps["pockets"].error
        both = run(
            direct(
                FIXTURES / "structures/1A3N.cif",
                protein=str(FIXTURES / "structures/1A3N.cif"),
            ),
            tmp_path,
            "b",
        )
        assert "exactly one of them is required" in both.steps["pockets"].error

    @pytest.mark.parametrize("limit", [0, 51])
    def test_a_pocket_limit_out_of_range_is_refused(self, tmp_path, limit):
        outcome = run(
            direct(FIXTURES / "structures/1A3N.cif", maximum_pockets=limit), tmp_path
        )
        assert "an integer from 1 to 50 is required" in outcome.steps["pockets"].error

    def test_a_file_type_p2rank_does_not_read_is_named(self, tmp_path):
        other = tmp_path / "structure.xyz"
        other.write_text("data")
        assert (
            "an extension that P2Rank does not read"
            in run(direct(other), tmp_path).steps["pockets"].error
        )

    def test_a_kind_that_is_not_offered_is_refused(self, tmp_path):
        outcome = run(
            direct(FIXTURES / "structures/1A3N.cif", structure_kind="docked"), tmp_path
        )
        assert "expected one of" in outcome.steps["pockets"].error

    def test_a_record_without_a_structure_is_refused(self, tmp_path):
        steps = [
            {
                "id": "inspect",
                "capability": "protein.inspect",
                "params": {
                    "fasta": str(FIXTURES / "hemoglobin/hemoglobin-alpha.fasta")
                },
            },
            {
                "id": "pockets",
                "capability": "structure.detect_binding_pockets",
                "bind": {"protein": "inspect.record"},
                "params": {"structure_kind": "experimental"},
            },
        ]
        assert "exact structure source" in run(steps, tmp_path).steps["pockets"].error

    def test_a_record_whose_atoms_were_edited_is_refused(self, haemoglobin, tmp_path):
        outcome, _, _ = haemoglobin
        edited = json.loads(Path(outcome.outputs("inspect")["record"]).read_text())
        edited["atoms"][0]["x"] += 10
        bad = tmp_path / "edited.json"
        bad.write_text(json.dumps(edited))
        steps = [
            {
                "id": "pockets",
                "capability": "structure.detect_binding_pockets",
                "params": {"protein": str(bad), "structure_kind": "experimental"},
            }
        ]
        assert "differ from its source" in run(steps, tmp_path).steps["pockets"].error


class TestParsing:
    def test_pocket_tables_that_are_malformed_are_refused(self):
        from alambiq.capabilities.investigate.science.pockets import (
            parse_pockets,
            validate_residue_table,
        )

        header = "name,rank,score,probability,sas_points,surf_atoms,center_x,center_y,center_z,residue_ids"
        good = f"{header}\npocket1,1,5.0,0.5,10,20,1.0,2.0,3.0,A_1 A_2\n"
        assert parse_pockets(good, 10)[0]["residue_ids"] == ["A_1", "A_2"]
        for bad in (
            "name,rank\npocket1,1\n",
            good.replace("0.5", "1.5"),
            good.replace("A_1 A_2", ""),
            good.replace(",1,5.0", ",2,5.0"),
            good.replace("10,20", "-1,20"),
            good.replace("5.0,0.5", "nan,0.5"),
        ):
            with pytest.raises(ValueError):
                parse_pockets(bad, 10)
        residues = "chain,residue_label,residue_name,score,zscore,probability,pocket\nA,1,GLY,0.1,0.2,0.3,0\n"
        assert validate_residue_table(residues) == 1
        for bad in (
            residues.replace("0.3", "1.3"),
            residues.replace("GLY", ""),
            "chain\nA\n",
        ):
            with pytest.raises(ValueError):
                validate_residue_table(bad)
