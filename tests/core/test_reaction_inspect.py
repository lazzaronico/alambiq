"""reaction.inspect on equations small enough to balance by hand.

The reference equation is methyl acetate hydrolysis with every atom mapped:

    CH3(1) C(2)(=O(3)) O(4) CH3(5)  +  H2O(6)  ->  CH3(1) C(2)(=O(3)) O(6)H  +  O(4)H CH3(5)

The ester bond C2-O4 is broken and a bond C2-O6 is made, and nothing else changes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("rdkit")


def hydrolysis() -> dict:
    return {
        "id": "ester-hydrolysis",
        "title": "Methyl acetate hydrolysis",
        "source": {
            "kind": "researcher-specified",
            "citation": "Balanced illustrative equation",
        },
        "conditions": {"description": "Explicit neutral species for atom accounting"},
        "participants": [
            {
                "id": "ester",
                "side": "reactant",
                "role": "substrate",
                "smiles": "[CH3:1][C:2](=[O:3])[O:4][CH3:5]",
                "coefficient": 1,
            },
            {
                "id": "water",
                "side": "reactant",
                "role": "water",
                "smiles": "[OH2:6]",
                "coefficient": 1,
            },
            {
                "id": "acid",
                "side": "product",
                "role": "product",
                "smiles": "[CH3:1][C:2](=[O:3])[OH:6]",
                "coefficient": 1,
            },
            {
                "id": "alcohol",
                "side": "product",
                "role": "byproduct",
                "smiles": "[OH:4][CH3:5]",
                "coefficient": 1,
            },
        ],
    }


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "rxn", "capability": "reaction.inspect", "params": params}
    return execute({"id": "r", "steps": [step]}, run_dir=tmp_path / name)


def inspect(params: dict, tmp_path: Path):
    outcome = run(params, tmp_path)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def codes(outcome) -> set[str]:
    return {d["code"] for d in outcome.outputs("rxn")["diagnostics"]}


def edit(**changes) -> dict:
    params = hydrolysis()
    params.update(changes)
    return params


class TestBalanceAndMaps:
    def test_the_hydrolysis_balances_and_its_bond_changes_are_read_from_the_maps(
        self, tmp_path
    ):
        out = inspect(hydrolysis(), tmp_path).outputs("rxn")
        assert out["balance"] == {
            "balanced": True,
            "elements": {},
            "isotopes": {},
            "charge": "0",
        }
        assert out["mapping"]["status"] == "complete-heavy-atoms"
        assert out["mapping"]["bond_changes"] == [
            {"atoms": [2, 4], "reactant_order": 1.0, "product_order": None},
            {"atoms": [2, 6], "reactant_order": None, "product_order": 1.0},
        ]
        assert out["diagnostics"] == []

    def test_each_species_is_identified_by_its_canonical_smiles_with_a_drawing(
        self, tmp_path
    ):
        from rdkit import Chem

        species = inspect(hydrolysis(), tmp_path).outputs("rxn")["species"]
        assert [s["id"] for s in species] == ["ester", "water", "acid", "alcohol"]
        for s in species:
            assert (
                Chem.MolToSmiles(Chem.MolFromSmiles(s["canonical_smiles"]))
                == s["canonical_smiles"]
            )
            assert "<svg" in s["drawing_svg"]
        assert {s["id"]: s["formula"] for s in species} == {
            "ester": "C3H6O2",
            "water": "H2O",
            "acid": "C2H4O2",
            "alcohol": "CH4O",
        }

    def test_the_same_equation_gives_the_same_reaction_identity_and_a_different_one_for_another(
        self, tmp_path
    ):
        first = inspect(hydrolysis(), tmp_path).outputs("rxn")["reaction_id"]
        again = inspect(hydrolysis(), tmp_path).outputs("rxn")["reaction_id"]
        other = edit()
        other["participants"][3]["smiles"] = "[OH:4][CH2:5]C"
        changed = inspect(other, tmp_path).outputs("rxn")["reaction_id"]
        assert first == again and first.startswith("reaction-") and changed != first

    def test_a_missing_water_is_an_exact_residual(self, tmp_path):
        params = hydrolysis()
        params["participants"].pop(1)
        out = inspect(params, tmp_path).outputs("rxn")
        assert not out["balance"]["balanced"] and out["balance"]["elements"] == {
            "H": "2",
            "O": "1",
        }
        assert "equation-imbalance" in codes(inspect(params, tmp_path))

    def test_a_deprotonated_product_needs_its_proton_to_balance(self, tmp_path):
        params = hydrolysis()
        params["participants"][2]["smiles"] = "[CH3:1][C:2](=[O:3])[O-:6]"
        out = inspect(params, tmp_path).outputs("rxn")
        assert (
            out["balance"]["elements"] == {"H": "-1"}
            and out["balance"]["charge"] == "-1"
        )
        params["participants"].append(
            {
                "id": "proton",
                "side": "product",
                "role": "proton",
                "smiles": "[H+]",
                "coefficient": 1,
            }
        )
        assert inspect(params, tmp_path).outputs("rxn")["balance"]["balanced"]

    def test_an_isotope_label_unbalances_only_the_isotopes(self, tmp_path):
        params = hydrolysis()
        params["participants"][0]["smiles"] = "[13CH3:1][C:2](=[O:3])[O:4][CH3:5]"
        out = inspect(params, tmp_path).outputs("rxn")
        assert out["balance"]["elements"] == {} and out["balance"]["isotopes"] == {
            "13C": "-1"
        }
        assert not out["balance"]["balanced"]

    def test_fractional_coefficients_balance_exactly_but_blur_the_maps(self, tmp_path):
        params = hydrolysis()
        for p in params["participants"]:
            p["coefficient"] = "1/3"
        out = inspect(params, tmp_path).outputs("rxn")
        assert out["balance"]["balanced"]
        assert out["mapping"]["status"] == "stoichiometric-ambiguity"
        assert out["species"][0]["coefficient"] == "1/3"

    def test_an_equation_that_cancels_itself_is_flagged_as_a_zero_net_change(
        self, tmp_path
    ):
        params = hydrolysis()
        params["participants"] = [
            {
                "id": "left",
                "side": "reactant",
                "role": "substrate",
                "smiles": "CCO",
                "coefficient": 1,
            },
            {
                "id": "right",
                "side": "product",
                "role": "product",
                "smiles": "OCC",
                "coefficient": 1,
            },
        ]
        outcome = inspect(params, tmp_path)
        assert outcome.outputs("rxn")["balance"]["balanced"]
        assert "zero-net-transformation" in codes(outcome)

    @pytest.mark.parametrize(
        ("water_smiles", "code"),
        [("[OH2:1]", "duplicate-atom-map"), ("[NH3:6]", "mapped-atom-identity")],
    )
    def test_maps_that_repeat_or_exchange_an_element_are_inconsistent_and_give_no_bond_changes(
        self, tmp_path, water_smiles, code
    ):
        params = hydrolysis()
        params["participants"][1]["smiles"] = water_smiles
        outcome = inspect(params, tmp_path)
        assert outcome.outputs("rxn")["mapping"]["status"] == "inconsistent"
        assert code in codes(outcome)
        assert outcome.outputs("rxn")["mapping"]["bond_changes"] == []

    def test_a_species_without_maps_makes_the_correspondence_partial(self, tmp_path):
        params = hydrolysis()
        params["participants"][0]["smiles"] = "COC(C)=O"
        outcome = inspect(params, tmp_path)
        assert outcome.outputs("rxn")["mapping"]["status"] == "partial"
        assert "atom-correspondence-coverage" in codes(outcome)

    def test_an_equation_without_any_maps_reports_them_absent(self, tmp_path):
        params = hydrolysis()
        for p, plain in zip(
            params["participants"], ["COC(C)=O", "O", "CC(=O)O", "CO"], strict=True
        ):
            p["smiles"] = plain
        assert inspect(params, tmp_path).outputs("rxn")["mapping"]["status"] == "absent"


class TestSpeciesIdentity:
    @pytest.mark.parametrize(
        ("left", "right"),
        [("OCC", "CCO"), ("[CH3:9][OH:2]", "CO"), ("[H]OC", "CO")],
    )
    def test_notation_and_maps_and_explicit_hydrogens_do_not_change_identity(
        self, left, right
    ):
        from alambiq.capabilities.investigate.science.reactions import inspect_compound

        assert (
            inspect_compound(left)["chemical_id"]
            == inspect_compound(right)["chemical_id"]
        )

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"),
            ("CC(=O)O", "CC(=O)[O-]"),
            ("CO", "[13CH3]O"),
            ("[Cl-]", "[Na+].[Cl-]"),
            ("C/C=C/C", "C/C=C\\C"),
        ],
    )
    def test_stereochemistry_charge_isotope_fragments_and_double_bond_geometry_do(
        self, left, right
    ):
        from alambiq.capabilities.investigate.science.reactions import inspect_compound

        assert (
            inspect_compound(left)["chemical_id"]
            != inspect_compound(right)["chemical_id"]
        )

    def test_a_reference_enantiomer_that_differs_is_flagged(self, tmp_path):
        params = hydrolysis()
        params["participants"][0].update(
            smiles="C[C@H](O)C(=O)O", reference_smiles="C[C@@H](O)C(=O)O"
        )
        assert "reference-structure-mismatch" in codes(inspect(params, tmp_path))

    def test_unspecified_stereochemistry_is_flagged(self, tmp_path):
        params = hydrolysis()
        params["participants"][0]["smiles"] = "CC(O)C(=O)O"
        assert "unspecified-stereochemistry" in codes(inspect(params, tmp_path))

    def test_a_radical_is_flagged(self, tmp_path):
        params = hydrolysis()
        params["participants"][1]["smiles"] = "[OH]"
        assert "radical-species" in codes(inspect(params, tmp_path))

    def test_one_database_entry_given_to_two_different_species_is_flagged(
        self, tmp_path
    ):
        params = hydrolysis()
        source = {"database": "local-reference", "accession": "x", "version": "1"}
        params["participants"][1]["role"] = "cosubstrate"
        params["participants"][0]["sources"] = [source]
        params["participants"][2]["sources"] = [source]
        outcome = inspect(params, tmp_path)
        assert outcome.outputs("rxn")["balance"][
            "balanced"
        ]  # a cosubstrate role still counts
        assert "source-identity-ambiguous" in codes(outcome)

    def test_chorismate_to_prephenate_matches_its_curated_reference_and_a_wrong_enantiomer_does_not(
        self, tmp_path
    ):
        from rdkit import Chem

        from alambiq.capabilities.investigate.science.reactions import inspect_compound

        manifest = (
            Path(__file__).resolve().parents[1]
            / "fixtures/reactions/chorismate/manifest.json"
        )
        fixture = json.loads(manifest.read_text())
        params = hydrolysis()
        params.update(id="RHEA:13898", title="Chorismate to prephenate")
        params["source"] = {
            "kind": "curated-reference",
            "citation": fixture["citation"],
            "url": fixture["url"],
            "version": fixture["snapshot_date"],
        }
        params["participants"] = []
        for index, item in enumerate(fixture["species"]):
            found = inspect_compound(item["smiles"])
            assert found["formula"] == fixture["expected_formula"]
            assert found["formal_charge"] == fixture["expected_charge"]
            assert (
                Chem.MolToInchiKey(Chem.MolFromSmiles(found["canonical_smiles"]))
                == item["inchi_key"]
            )
            params["participants"].append(
                {
                    "id": item["name"],
                    "side": "reactant" if index == 0 else "product",
                    "role": "substrate" if index == 0 else "product",
                    "coefficient": 1,
                    "smiles": item["smiles"],
                    "reference_smiles": item["mol_smiles"],
                    "sources": [
                        {
                            "database": "ChEBI",
                            "accession": item["accession"],
                            "version": fixture["snapshot_date"],
                        }
                    ],
                }
            )
        right = inspect(copy.deepcopy(params), tmp_path)
        assert (
            right.outputs("rxn")["balance"]["balanced"]
            and right.outputs("rxn")["diagnostics"] == []
        )
        params["participants"][0]["smiles"] = params["participants"][0][
            "smiles"
        ].replace("@@", "@", 1)
        wrong = inspect(params, tmp_path)
        assert wrong.outputs("rxn")["balance"]["balanced"]
        assert "reference-structure-mismatch" in codes(wrong)
        assert (
            right.outputs("rxn")["reaction_id"] != wrong.outputs("rxn")["reaction_id"]
        )


class TestRefusals:
    @pytest.mark.parametrize(
        "value", [True, 0, -1, 0.5, "nan", "1/0", "2**4", "0" * 100 + "1", "1e99999999"]
    )
    def test_a_coefficient_that_is_not_an_exact_positive_rational_is_refused(
        self, tmp_path, value
    ):
        params = hydrolysis()
        params["participants"][0]["coefficient"] = value
        outcome = run(params, tmp_path)
        assert outcome.steps["rxn"].status == "failed"

    @pytest.mark.parametrize(
        "smiles", ["", "*", "C title", "C |$foo$|", "C(C)(C)(C)(C)C", "C1CC"]
    )
    def test_a_species_that_is_not_a_valid_explicit_molecule_is_refused(
        self, tmp_path, smiles
    ):
        params = hydrolysis()
        params["participants"][0]["smiles"] = smiles
        assert run(params, tmp_path).steps["rxn"].status == "failed"

    def test_an_equation_with_only_one_side_is_refused(self, tmp_path):
        params = hydrolysis()
        params["participants"] = params["participants"][:2]
        assert "both equation sides" in run(params, tmp_path).steps["rxn"].error

    def test_a_substrate_on_the_product_side_is_refused(self, tmp_path):
        params = hydrolysis()
        params["participants"][2]["role"] = "substrate"
        assert (
            "roles must agree with their equation side"
            in run(params, tmp_path).steps["rxn"].error
        )

    def test_a_repeated_participant_id_is_refused(self, tmp_path):
        params = hydrolysis()
        params["participants"][1]["id"] = "ester"
        assert "unique" in run(params, tmp_path).steps["rxn"].error

    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (("conditions", "pH"), 20, "pH"),
            (("conditions", "temperature_K"), 0, "temperature_K"),
            (("conditions", "ionic_strength_M"), -1, "ionic_strength_M"),
            (("source", "sha256"), "abc", "64 lowercase hexadecimal"),
        ],
    )
    def test_context_outside_its_range_or_form_is_refused(
        self, tmp_path, path, value, message
    ):
        params = hydrolysis()
        params[path[0]][path[1]] = value
        assert message in run(params, tmp_path).steps["rxn"].error

    def test_a_field_the_capability_does_not_have_is_named(self, tmp_path):
        outcome = run(edit(convention="transformed-pH7"), tmp_path)
        assert "unknown field convention" in outcome.steps["rxn"].error

    def test_a_role_that_is_not_offered_is_named(self, tmp_path):
        params = hydrolysis()
        params["participants"][0]["role"] = "enzyme"
        assert "expected one of" in run(params, tmp_path).steps["rxn"].error


class TestPresentation:
    def test_the_report_gives_the_verdict_and_the_species(self, tmp_path):
        outcome = inspect(hydrolysis(), tmp_path)
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "balances in elements, isotopes and charge" in page
        assert "C3H6O2" in page and "<svg" in page
        assert "2–4" in page and "2–6" in page

    def test_an_imbalance_is_stated_with_its_residual(self, tmp_path):
        params = hydrolysis()
        params["participants"].pop(1)
        page = inspect(params, tmp_path).report(tmp_path / "report.html").read_text()
        assert "does not balance" in page and "H 2" in page

    def test_labels_are_escaped(self, tmp_path):
        params = hydrolysis()
        params["participants"][0]["id"] = "<script>alert(1)</script>"
        page = inspect(params, tmp_path).report(tmp_path / "report.html").read_text()
        assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page

    def test_the_stored_record_keeps_the_request_and_the_method(self, tmp_path):
        outcome = inspect(hydrolysis(), tmp_path)
        record = json.loads(Path(outcome.outputs("rxn")["record"]).read_text())
        assert record["request"]["id"] == "ester-hydrolysis"
        assert record["balance"]["basis"] == "products-minus-reactants"
        assert "canonical isomeric SMILES" in record["method"]
