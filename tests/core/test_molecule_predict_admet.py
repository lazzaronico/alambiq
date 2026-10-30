"""molecule.predict_admet estimates ADMET and physicochemical properties of molecules."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ENV = os.environ.get("ALAMBIQ_ENV_ADMET", "")
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def installed() -> bool:
    return bool(ENV) and Path(ENV).is_file()


engine_only = pytest.mark.skipif(
    not installed(), reason="set ALAMBIQ_ENV_ADMET to run the engine tests"
)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_ADMET", ENV or sys.executable)


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "admet", "capability": "molecule.predict_admet", "params": params}
    return execute({"id": "a", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def stored_predictions(count: int) -> tuple[list[str], list[dict[str, float]]]:
    """The first approved drugs of the DrugBank table that ADMET-AI ships, with its predictions."""
    import csv
    import subprocess

    package = subprocess.run(
        [ENV, "-c", "import admet_ai; print(admet_ai.__file__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    table = Path(package).parent / "resources" / "data" / "drugbank_approved.csv"
    with table.open() as handle:
        rows = list(csv.DictReader(handle))[:count]
    endpoints = [
        name
        for name in rows[0]
        if name not in {"smiles", "name", "id", "atc"}
        and not name.startswith("atc_name")
    ]
    return [r["smiles"] for r in rows], [
        {name: float(r[name]) for name in endpoints} for r in rows
    ]


def refusal(params: dict, tmp_path: Path) -> str:
    outcome = run(params, tmp_path)
    assert not outcome.ok
    return " ".join(x.error or "" for x in outcome.steps.values())


@pytest.mark.usefixtures("engine")
class TestRequestsAreRefusedBeforeAnythingRuns:
    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"smiles": []}, "smiles must hold 1 to 10000 molecules"),
            ({"smiles": ["C"] * 10001}, "smiles must hold 1 to 10000 molecules"),
            ({"smiles": ["C", ""]}, "smiles 2 must be 1 to 1000 characters"),
            ({"smiles": ["C C"]}, "smiles 1 must be 1 to 1000 characters"),
            ({"smiles": ["C" * 1001]}, "smiles 1 must be 1 to 1000 characters"),
            ({"smiles": ["C"], "threads": 0}, "threads must be an integer"),
            ({"smiles": ["C"], "threads": 65}, "threads must be an integer"),
        ],
    )
    def test_a_request_out_of_range_is_refused(self, tmp_path, params, message):
        assert message in refusal(params, tmp_path)


@engine_only
@pytest.mark.usefixtures("engine")
class TestTheEstimates:
    def test_a_molecule_gets_the_52_endpoints_with_a_table_that_describes_them(
        self, tmp_path
    ):
        result = (
            run_ok({"smiles": [ASPIRIN], "threads": 4}, tmp_path).steps["admet"].result
        )
        endpoint_ids = [e.id for e in result.endpoints]
        assert len(endpoint_ids) == 52
        assert set(result.molecules[0].values) == set(endpoint_ids)
        kinds = {e.id: e.kind for e in result.endpoints}
        assert sum(k == "descriptor" for k in kinds.values()) == 11
        assert sum(k == "classification" for k in kinds.values()) == 31
        assert sum(k == "regression" for k in kinds.values()) == 10

    def test_the_descriptors_are_the_rdkit_values_of_aspirin(self, tmp_path):
        values = (
            run_ok({"smiles": [ASPIRIN], "threads": 4}, tmp_path)
            .steps["admet"]
            .result.molecules[0]
            .values
        )
        assert values["molecular_weight"] == pytest.approx(180.159, abs=1e-3)
        assert values["tpsa"] == pytest.approx(63.6, abs=1e-6)
        assert values["hydrogen_bond_donors"] == 1
        assert values["hydrogen_bond_acceptors"] == 3
        assert values["stereo_centers"] == 0

    def test_a_classification_endpoint_is_a_probability(self, tmp_path):
        result = (
            run_ok({"smiles": [ASPIRIN, "CCO", "c1ccc2[nH]ccc2c1"]}, tmp_path)
            .steps["admet"]
            .result
        )
        classification = [e.id for e in result.endpoints if e.kind == "classification"]
        for molecule in result.molecules:
            assert all(0.0 <= molecule.values[i] <= 1.0 for i in classification)

    def test_the_endpoint_table_carries_units_and_the_reported_metrics(self, tmp_path):
        endpoints = {
            e.id: e
            for e in run_ok({"smiles": ["CCO"]}, tmp_path)
            .steps["admet"]
            .result.endpoints
        }
        assert endpoints["Solubility_AqSolDB"].units == "log(mol/L)"
        assert endpoints["Solubility_AqSolDB"].reported_r2 == pytest.approx(
            0.8166, abs=1e-3
        )
        assert endpoints["HIA_Hou"].reported_auroc == pytest.approx(0.994, abs=1e-3)
        assert endpoints["HIA_Hou"].reported_r2 is None
        assert endpoints["molecular_weight"].reported_r2 is None
        assert endpoints["Half_Life_Obach"].reported_r2 < 0

    def test_results_follow_the_request_and_a_molecule_written_twice_is_predicted_once(
        self, tmp_path
    ):
        result = (
            run_ok({"smiles": ["CCO", ASPIRIN, "OCC"]}, tmp_path).steps["admet"].result
        )
        assert [m.smiles for m in result.molecules] == ["CCO", ASPIRIN, "OCC"]
        assert result.distinct_molecules == 2
        assert result.molecules[0].canonical_smiles == "CCO"
        assert result.molecules[0].values == result.molecules[2].values

    def test_percentiles_appear_only_when_asked_for_and_lie_between_0_and_100(
        self, tmp_path
    ):
        plain = run_ok({"smiles": [ASPIRIN]}, tmp_path, "plain").steps["admet"].result
        asked = (
            run_ok(
                {"smiles": [ASPIRIN], "drugbank_percentiles": True}, tmp_path, "asked"
            )
            .steps["admet"]
            .result
        )
        assert plain.molecules[0].drugbank_percentiles is None
        percentiles = asked.molecules[0].drugbank_percentiles
        assert set(percentiles) == set(asked.molecules[0].values)
        assert all(0 <= p <= 100 for p in percentiles.values())

    def test_the_estimates_reproduce_the_predictions_stored_for_approved_drugs(
        self, tmp_path
    ):
        smiles, stored = stored_predictions(20)
        result = run_ok({"smiles": smiles}, tmp_path).steps["admet"].result
        for molecule, row in zip(result.molecules, stored, strict=True):
            for name, value in row.items():
                assert molecule.values[name] == pytest.approx(value, abs=1e-3), name

    def test_versions_and_the_record_are_kept(self, tmp_path):
        result = run_ok({"smiles": ["CCO"]}, tmp_path).steps["admet"].result
        assert result.engine_versions["admet-ai"] == "2.0.1"
        assert result.engine_versions["chemprop"].startswith("2.")
        assert Path(result.record).is_file()


@engine_only
@pytest.mark.usefixtures("engine")
class TestUnreadableMolecules:
    def test_every_smiles_that_rdkit_cannot_read_is_named_with_its_position(
        self, tmp_path
    ):
        message = refusal({"smiles": ["CCO", "C(C", ASPIRIN, "xx"]}, tmp_path)
        assert "2 SMILES cannot be read by RDKit" in message
        assert "position 2 (C(C), 4 (xx)" in message

    def test_the_report_shows_the_first_molecules_and_says_how_many_more_there_are(
        self, tmp_path
    ):
        from alambiq.capabilities.investigate.molecule_predict_admet_report import (
            render,
        )

        smiles = ["C" * n for n in range(1, 11)]
        outcome = run_ok({"smiles": smiles}, tmp_path)
        html = render(outcome.steps["admet"].result)
        assert "10 molecules (10 distinct), 52 estimates each" in html
        assert "Aqueous Solubility" in html
        assert "The first 8 of 10 molecules are shown" in html
