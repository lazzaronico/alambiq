"""protein.predict_stability_change: structure mapping, and ThermoMPNN on GB3.

The mapping tests need no model: they check how a structure and a list of
substitutions become the backbone and positions the network reads, and every
way that could silently go wrong. The model tests run the real ThermoMPNN
checkpoint in its pinned environment (set ALAMBIQ_ENV_THERMOMPNN to that
environment's Python, with the `thermompnn` resource installed).

GB3 (PDB 1P7E, 56 residues) is the reference. Two predictions come with a record
from an earlier, independent run of the previous engine on this machine
(`environments/thermompnn/verification.osx-arm64.json`): T55A is predicted at
0.8774051070213318 kcal/mol and the zero-change control T55T at exactly 0. The
network is deterministic, so the port must reproduce both.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")

from alambiq.capabilities.characterize.science.stability import (  # noqa: E402
    normalize_structure,
    parse_residue,
)

STRUCTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/1P7E.cif"
SEQUENCE = "MQYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTE"
RECORDED_T55A = 0.8774051070213318
ENV = os.environ.get("ALAMBIQ_ENV_THERMOMPNN", "")


def row(residue: str = "55", wild: str = "T", mutant: str = "A") -> dict[str, str]:
    number, insertion = parse_residue(residue)
    return {
        "residue_id": number,
        "insertion_code": insertion,
        "wild_type": wild,
        "mutant": mutant,
    }


def structure():
    return gemmi.make_structure_from_block(
        gemmi.cif.read_file(str(STRUCTURE)).sole_block()
    )


def mapped(model=None, **overrides):
    parameters = {
        "text": (model or structure()).make_mmcif_document().as_string(),
        "fmt": "mmcif",
        "chain": "A",
        "model_index": 0,
        "sequence": SEQUENCE,
        "mutations": [row()],
    }
    parameters.update(overrides)
    return normalize_structure(
        parameters["text"],
        parameters["fmt"],
        parameters["chain"],
        parameters["model_index"],
        parameters["sequence"],
        parameters["mutations"],
    )


class TestResidueLabels:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("55", ("55", "")),
            ("-3", ("-3", "")),
            ("55A", ("55", "A")),
            (" 7 ", ("7", "")),
        ],
    )
    def test_an_author_label_splits_into_number_and_insertion_letter(
        self, label, expected
    ):
        assert parse_residue(label) == expected

    @pytest.mark.parametrize("label", ["", "A55", "5.5", "55AB", "T55"])
    def test_anything_else_is_refused(self, label):
        with pytest.raises(ValueError, match="author residue number"):
            parse_residue(label)


class TestMapping:
    def test_the_position_is_the_residue_index_not_the_author_number(self):
        request, _ = mapped()
        # Residue 55 is the 55th residue of this chain, so its zero-based index is 54.
        assert request["mutations"] == [
            {"position": 54, "wild_type": "T", "mutant": "A"}
        ]
        assert request["residues"][54]["residue_id"] == "55"
        assert len(request["coordinates_angstrom"]) == 56
        assert all(
            len(r) == 4 and all(len(a) == 3 for a in r)
            for r in request["coordinates_angstrom"]
        )

    def test_the_backbone_is_the_deposited_one(self):
        request, _ = mapped()
        first_ca = structure()[0]["A"][0]["CA"][0].pos
        assert request["coordinates_angstrom"][0][1] == pytest.approx(
            [first_ca.x, first_ca.y, first_ca.z]
        )

    def test_an_insertion_letter_and_negative_numbering_are_kept(self):
        model = structure()
        model[0]["A"][54].seqid = gemmi.SeqId(-3, "B")
        request, _ = mapped(model, mutations=[row("-3B")])
        assert request["mutations"][0]["position"] == 54
        with pytest.raises(ValueError, match="absent"):
            mapped(model, mutations=[row("-3")])

    def test_a_pdb_file_maps_to_the_same_request_as_the_mmcif(self):
        pdb = mapped(fmt="pdb", text=structure().make_pdb_string())
        cif = mapped()
        assert pdb[0]["sequence"] == cif[0]["sequence"]
        assert pdb[0]["mutations"] == cif[0]["mutations"]
        import numpy as np

        assert np.allclose(
            pdb[0]["coordinates_angstrom"], cif[0]["coordinates_angstrom"], atol=1e-3
        )

    def test_other_chains_and_omitted_residues_are_listed_as_not_seen(self):
        _, omitted = mapped()
        assert omitted["other_chains"] == []
        assert (
            "other chains, ligands, waters and side-chain coordinates are excluded"
            in omitted["scope"]
        )

    def test_a_split_author_chain_leaves_out_an_explicit_nonpolymer_residue(self):
        model = structure()
        ligand = gemmi.Residue()
        ligand.name = "ALA"
        ligand.seqid = gemmi.SeqId(900, " ")
        ligand.entity_type = gemmi.EntityType.NonPolymer
        ligand.entity_id = "2"
        ligand.subchain = "Z"
        ligand.het_flag = "H"
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        ligand.add_atom(atom)
        part = gemmi.Chain("A")
        part.add_residue(ligand)
        model[0].add_chain(part)
        entity = gemmi.Entity("2")
        entity.entity_type = gemmi.EntityType.NonPolymer
        entity.subchains = ["Z"]
        model.entities.append(entity)
        request, omitted = mapped(model)
        assert (
            request["sequence"] == SEQUENCE
            and len(request["coordinates_angstrom"]) == 56
        )
        assert omitted["selected_chain_omitted_residues"][0]["residue_id"] == "900"
        assert omitted["selected_chain_omitted_residues"][0]["residue_name"] == "ALA"


class TestMappingRefusals:
    def test_a_missing_backbone_atom_is_refused(self):
        model = structure()
        model[0]["A"][12].remove_atom("O", "\x00")
        with pytest.raises(ValueError, match="N/CA/C/O"):
            mapped(model)

    def test_a_chain_break_is_refused(self):
        model = structure()
        model[0]["A"][12]["N"][0].pos.x += 20
        with pytest.raises(ValueError, match="chain break"):
            mapped(model)

    def test_an_alternate_backbone_position_is_refused(self):
        model = structure()
        model[0]["A"][12]["CA"][0].altloc = "A"
        with pytest.raises(ValueError, match="unambiguous"):
            mapped(model)

    def test_a_duplicate_residue_identity_is_refused(self):
        model = structure()
        model[0]["A"][12].seqid = model[0]["A"][11].seqid
        with pytest.raises(ValueError, match="residue identity"):
            mapped(model)

    def test_a_modified_residue_needs_explicit_preparation(self):
        model = structure()
        model[0]["A"][0].name = "MSE"
        with pytest.raises(ValueError, match="canonical"):
            mapped(model)

    def test_a_construct_that_differs_from_the_chain_is_refused(self):
        with pytest.raises(ValueError, match="observed chain sequence differs"):
            mapped(sequence=SEQUENCE[:-1])

    def test_a_wild_type_the_structure_does_not_have_is_named(self):
        with pytest.raises(
            ValueError, match="names wild type K, but the structure has T"
        ):
            mapped(mutations=[row(wild="K")])

    def test_a_residue_the_chain_lacks_is_named(self):
        with pytest.raises(ValueError, match="residue 999 is absent from chain A"):
            mapped(mutations=[row("999")])

    def test_a_mutant_that_is_not_a_canonical_residue_is_refused(self):
        with pytest.raises(ValueError, match="20 canonical"):
            mapped(mutations=[row(mutant="X")])

    def test_a_duplicate_substitution_is_refused_and_a_self_substitution_is_a_control(
        self,
    ):
        with pytest.raises(ValueError, match="Duplicate"):
            mapped(mutations=[row(), row()])
        request, _ = mapped(mutations=[row(mutant="T")])
        assert request["mutations"][0]["mutant"] == "T"

    def test_a_chain_that_is_not_there_is_refused(self):
        with pytest.raises(ValueError, match="resolve uniquely"):
            mapped(chain="Q")

    def test_a_model_that_is_not_there_is_refused(self):
        with pytest.raises(ValueError, match="model is absent"):
            mapped(model_index=5)

    def test_no_substitutions_is_refused(self):
        with pytest.raises(ValueError, match="1–20000"):
            mapped(mutations=[])


# --- through the capability ------------------------------------------------


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {
        "id": "st",
        "capability": "protein.predict_stability_change",
        "params": params,
    }
    return execute({"id": "s", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def params(tmp_path: Path, mutations=None, **extra) -> dict:
    (tmp_path / "construct.faa").write_text(f">gb3\n{SEQUENCE}\n")
    return {
        "structure": str(STRUCTURE),
        "sequence": str(tmp_path / "construct.faa"),
        "chain_id": "A",
        "mutations": mutations or [{"residue": "55", "wild_type": "T", "mutant": "A"}],
        **extra,
    }


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_THERMOMPNN", ENV or sys.executable)


def installed() -> bool:
    from alambiq.core import resources

    return (
        bool(ENV) and Path(ENV).is_file() and resources.location("thermompnn").is_dir()
    )


class TestRefusalsThroughTheCapability:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"model_index": 1000}, "model_index must be an integer from 0 to 999"),
            ({"seed": -1}, "seed must be an integer"),
            ({"threads": 0}, "threads must be an integer from 1 to 32"),
            ({"chain_id": "Q"}, "resolve uniquely"),
            (
                {"mutations": [{"residue": "55", "wild_type": "K", "mutant": "A"}]},
                "names wild type K",
            ),
            (
                {"mutations": [{"residue": "T55", "wild_type": "T", "mutant": "A"}]},
                "author residue number",
            ),
        ],
    )
    def test_a_request_that_cannot_be_trusted_is_refused_before_the_model_runs(
        self, tmp_path, changes, message
    ):
        outcome = run({**params(tmp_path), **changes}, tmp_path)
        assert outcome.steps["st"].status == "failed"
        assert message in outcome.steps["st"].error

    def test_a_format_that_does_not_match_the_file_fails_in_plain_terms(self, tmp_path):
        outcome = run({**params(tmp_path), "structure_format": "pdb"}, tmp_path)
        assert outcome.steps["st"].status == "failed"
        assert "Incorrect file format" in outcome.steps["st"].error

    def test_a_file_whose_format_is_unclear_must_say_so(self, tmp_path):
        odd = tmp_path / "structure.dat"
        odd.write_text(STRUCTURE.read_text())
        outcome = run({**params(tmp_path), "structure": str(odd)}, tmp_path)
        assert "set structure_format to pdb or mmcif" in outcome.steps["st"].error


pytestmark_model = pytest.mark.skipif(
    not installed(),
    reason="set ALAMBIQ_ENV_THERMOMPNN and install the thermompnn resource to run",
)


@pytestmark_model
class TestOnGB3:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("gb3")
        os.environ["ALAMBIQ_ENV_THERMOMPNN"] = ENV
        mutations = [
            {"residue": "55", "wild_type": "T", "mutant": "A"},
            {"residue": "55", "wild_type": "T", "mutant": "T"},
            {"residue": "7", "wild_type": "I", "mutant": "D"},
            {"residue": "30", "wild_type": "F", "mutant": "A"},
        ]
        return run_ok(params(tmp_path, mutations), tmp_path), tmp_path

    def by_label(self, outcome) -> dict[str, float]:
        return {
            f"{p['wild_type']}{p['residue']}{p['mutant']}": p["ddg_kcal_mol"]
            for p in outcome[0].outputs("st")["predictions"]
        }

    def test_t55a_reproduces_the_recorded_prediction_of_the_previous_engine(
        self, outcome
    ):
        assert self.by_label(outcome)["T55A"] == pytest.approx(RECORDED_T55A, abs=1e-6)

    def test_a_substitution_to_the_same_residue_is_exactly_zero(self, outcome):
        assert self.by_label(outcome)["T55T"] == 0.0

    def test_a_buried_hydrophobic_made_charged_is_far_more_destabilizing_than_a_surface_change(
        self, outcome
    ):
        values = self.by_label(outcome)
        # I7 and F30 are core residues of GB3; T55 is on the surface.
        assert values["I7D"] > values["T55A"] + 1.0
        assert values["F30A"] > values["T55A"]
        assert values["I7D"] > 1.0

    def test_predictions_come_back_in_request_order_with_their_residue_names(
        self, outcome
    ):
        predictions = outcome[0].outputs("st")["predictions"]
        assert [(p["wild_type"], p["residue"], p["mutant"]) for p in predictions] == [
            ("T", "55", "A"),
            ("T", "55", "T"),
            ("I", "7", "D"),
            ("F", "30", "A"),
        ]
        assert [p["residue_name"] for p in predictions] == ["THR", "THR", "ILE", "PHE"]

    def test_the_counts_follow_from_the_signs(self, outcome):
        out = outcome[0].outputs("st")
        values = [p["ddg_kcal_mol"] for p in out["predictions"]]
        assert out["destabilizing"] == sum(v > 0 for v in values)
        assert out["stabilizing"] == sum(v < 0 for v in values)
        assert out["unchanged"] == 1  # the zero-change control
        assert out["destabilizing"] + out["stabilizing"] + out["unchanged"] == 4

    def test_the_record_states_sign_scope_model_and_what_ran(self, outcome):
        record = json.loads(Path(outcome[0].outputs("st")["record"]).read_text())
        assert record["sign_convention"].endswith("positive destabilizing")
        assert record["model"] == "ThermoMPNN-default"
        assert record["source_revision"] == "370f76ec62bd929f7425e311d8df04a0d094990f"
        installed_versions = subprocess.run(
            [
                ENV,
                "-c",
                "import importlib.metadata as m, json;"
                "print(json.dumps({n: m.version(n) for n in ('torch', 'numpy')}))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert record["runtime"]["versions"] == json.loads(installed_versions)
        assert record["runtime"]["versions"]["numpy"] == "1.26.4"
        assert (
            record["source"]["chain_id"] == "A"
            and len(record["source"]["sha256"]) == 64
        )
        assert record["rows"][0]["source_residue"]["residue_id"] == "55"
        assert "side-chain coordinates are excluded" in record["model_context"]["scope"]

    def test_the_seed_and_thread_count_do_not_change_a_deterministic_network(
        self, outcome, tmp_path
    ):
        again = run_ok(
            params(
                tmp_path,
                [{"residue": "55", "wild_type": "T", "mutant": "A"}],
                seed=7,
                threads=1,
            ),
            tmp_path,
        )
        assert again.outputs("st")["predictions"][0]["ddg_kcal_mol"] == pytest.approx(
            RECORDED_T55A, abs=1e-6
        )

    def test_a_pdb_file_gives_the_same_predictions_as_the_mmcif(self, tmp_path):
        pdb = tmp_path / "gb3.pdb"
        pdb.write_text(structure().make_pdb_string())
        result = run_ok({**params(tmp_path), "structure": str(pdb)}, tmp_path)
        assert result.outputs("st")["predictions"][0]["ddg_kcal_mol"] == pytest.approx(
            RECORDED_T55A, abs=1e-4
        )

    def test_a_file_with_an_unhelpful_extension_reads_once_its_format_is_stated(
        self, tmp_path
    ):
        odd = tmp_path / "structure.dat"
        odd.write_text(STRUCTURE.read_text())
        result = run_ok(
            {**params(tmp_path), "structure": str(odd), "structure_format": "mmcif"},
            tmp_path,
        )
        assert result.outputs("st")["predictions"][0]["ddg_kcal_mol"] == pytest.approx(
            RECORDED_T55A, abs=1e-6
        )

    def test_the_same_request_is_reused_and_a_new_substitution_runs_again(
        self, tmp_path
    ):
        first = run_ok(params(tmp_path), tmp_path, "same")
        second = run_ok(params(tmp_path), tmp_path, "same")
        changed = run_ok(
            params(tmp_path, [{"residue": "55", "wild_type": "T", "mutant": "V"}]),
            tmp_path,
            "same",
        )
        assert (first.steps["st"].status, second.steps["st"].status) == (
            "completed",
            "reused",
        )
        assert changed.steps["st"].status == "completed"

    def test_the_report_states_the_sign_convention(self, outcome):
        result, tmp_path = outcome
        page = result.report(tmp_path / "report.html").read_text()
        assert "positive destabilizing" in page and "T55A" in page
