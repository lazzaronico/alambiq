"""molecule.compute_hydration_free_energy runs the OpenFE absolute solvation protocol."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ENV = os.environ.get("ALAMBIQ_ENV_OPENFE", "")
CAPABILITY = "molecule.compute_hydration_free_energy"


def installed() -> bool:
    return bool(ENV) and Path(ENV).is_file()


engine_only = pytest.mark.skipif(
    not installed(), reason="set ALAMBIQ_ENV_OPENFE to run the engine tests"
)
gpu_only = pytest.mark.skipif(
    not installed() or os.environ.get("ALAMBIQ_TEST_CUDA") != "1",
    reason="set ALAMBIQ_ENV_OPENFE and ALAMBIQ_TEST_CUDA=1 on a CUDA host to run the simulation",
)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_OPENFE", ENV or sys.executable)


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "hfe", "capability": CAPABILITY, "params": params}
    return execute({"id": "h", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def refusal(params: dict, tmp_path: Path) -> str:
    outcome = run(params, tmp_path)
    assert not outcome.ok
    return " ".join(x.error or "" for x in outcome.steps.values())


@pytest.mark.usefixtures("engine")
class TestRequestsAreRefusedBeforeAnythingRuns:
    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"smiles": ""}, "smiles must be 1 to 500 characters"),
            ({"smiles": "C C"}, "smiles must be 1 to 500 characters"),
            ({"smiles": "C", "repeats": 0}, "repeats must be an integer"),
            ({"smiles": "C", "repeats": 11}, "repeats must be an integer"),
            ({"smiles": "C", "seed": -1}, "seed must be an integer"),
            ({"smiles": "C", "timeout_seconds": 59}, "timeout_seconds must be"),
            ({"smiles": "C", "production_ns": 0.04}, "production_ns must be between"),
            ({"smiles": "C", "production_ns": 101}, "production_ns must be between"),
            (
                {"smiles": "C", "equilibration_ns": 11},
                "equilibration_ns must be between",
            ),
            (
                {"smiles": "C", "temperature_kelvin": float("nan")},
                "temperature_kelvin must be between",
            ),
            ({"smiles": "C", "temperature_kelvin": 249}, "temperature_kelvin must be"),
            ({"smiles": "C", "pressure_bar": 0}, "pressure_bar must be between"),
            ({"smiles": "C", "production_ns": 1.0003}, "whole number of 2.5 ps"),
            ({"smiles": "C", "production_ns": 0.0525}, "(the vacuum leg)"),
            ({"smiles": "C", "equilibration_ns": 0.0625}, "(the vacuum leg)"),
        ],
    )
    def test_a_setting_that_cannot_run_is_refused_by_name(
        self, tmp_path, params, message
    ):
        assert message in refusal(params, tmp_path)


def leg(decoupling: float, error: float, overlap: float = 0.2) -> dict:
    return {
        "id": "r",
        "decoupling_kcal_mol": decoupling,
        "mbar_error_kcal_mol": error,
        "overlap_matrix": [
            [0.5, overlap, 0.0],
            [overlap, 0.5, overlap],
            [0.0, overlap, 0.5],
        ],
        "effective_samples": [40.0, 25.0, 33.0],
        "forward_reverse": None,
    }


def found(repeats: int = 3, overlap: float = 0.2) -> dict:
    solvent = [leg(5.0 + 0.1 * i, 0.3, overlap) for i in range(repeats)]
    vacuum = [leg(1.0 - 0.1 * i, 0.4, overlap) for i in range(repeats)]
    return {
        "canonical_smiles": "C",
        "legs": {"solvent": solvent, "vacuum": vacuum},
        "versions": {"openfe": "1.12.0"},
        "charges": [-0.4, 0.1, 0.1, 0.1, 0.1],
        "settings": {"protocol_repeats": repeats},
        "seconds": 12.5,
        "estimate_kcal_mol": -3.9,
        "uncertainty_kcal_mol": 0.2,
    }


@pytest.fixture
def stand_in(monkeypatch):
    """Replace the simulation by a recorded answer; the capability's own logic runs."""
    from alambiq.core import envs

    calls = []
    answer = {}

    def fake(environment, entry, request, **kwargs):
        calls.append((environment, entry, request, kwargs))
        output = Path(request["output"])
        output.mkdir(parents=True)
        (output / "settings.json").write_text("{}")
        return answer["value"]

    monkeypatch.setattr(envs, "run_in_environment", fake)
    monkeypatch.setenv("ALAMBIQ_ENV_OPENFE", ENV or sys.executable)
    return calls, answer


class TestTheResultIsAssembledFromTheLegs:
    def test_a_repeat_is_the_vacuum_leg_minus_the_solvent_leg(self, tmp_path, stand_in):
        calls, answer = stand_in
        answer["value"] = found()
        result = run_ok({"smiles": "C"}, tmp_path).steps["hfe"].result
        first = result.repeats[0]
        assert first.hydration_free_energy_kcal_mol == pytest.approx(1.0 - 5.0)
        assert first.mbar_error_kcal_mol == pytest.approx(0.5)
        assert first.solvent.decoupling_kcal_mol == 5.0
        assert first.vacuum.decoupling_kcal_mol == 1.0
        assert result.hydration_free_energy_kcal_mol == pytest.approx(
            (-4.0 - 4.2 - 4.4) / 3
        )
        assert result.mean_mbar_error_kcal_mol == pytest.approx(0.5)
        assert result.uncertainty_kcal_mol == 0.2

    def test_the_overlap_and_the_effective_samples_are_read_from_the_legs(
        self, tmp_path, stand_in
    ):
        _, answer = stand_in
        answer["value"] = found()
        repeat = run_ok({"smiles": "C"}, tmp_path).steps["hfe"].result.repeats[0]
        assert repeat.solvent.minimum_adjacent_overlap == pytest.approx(0.2)
        assert repeat.solvent.minimum_effective_samples == 25.0

    def test_the_request_reaches_the_engine_with_every_setting(
        self, tmp_path, stand_in
    ):
        calls, answer = stand_in
        answer["value"] = found(repeats=2)
        run_ok(
            {
                "smiles": "CCO",
                "repeats": 2,
                "production_ns": 2.5,
                "equilibration_ns": 0.5,
                "temperature_kelvin": 310.0,
                "pressure_bar": 2.0,
                "seed": 7,
                "platform": "CPU",
                "timeout_seconds": 3600,
            },
            tmp_path,
        )
        environment, entry, request, kwargs = calls[0]
        assert (environment, entry) == ("openfe", "openfe_hydration:run")
        assert {k: v for k, v in request.items() if k != "output"} == {
            "smiles": "CCO",
            "repeats": 2,
            "production_ns": 2.5,
            "equilibration_ns": 0.5,
            "temperature_kelvin": 310.0,
            "pressure_bar": 2.0,
            "seed": 7,
            "platform": "CPU",
        }
        assert kwargs["timeout"] == 3600

    def test_one_repeat_has_no_uncertainty(self, tmp_path, stand_in):
        _, answer = stand_in
        answer["value"] = found(repeats=1)
        result = run_ok({"smiles": "C", "repeats": 1}, tmp_path).steps["hfe"].result
        assert result.uncertainty_kcal_mol is None

    def test_a_short_run_and_a_low_overlap_each_carry_a_note(self, tmp_path, stand_in):
        _, answer = stand_in
        answer["value"] = found(repeats=2, overlap=0.01)
        outcome = run_ok({"smiles": "C", "repeats": 2, "production_ns": 1.0}, tmp_path)
        notes = " ".join(outcome.steps["hfe"].notes)
        assert "below the 3 repeats and 5 ns" in notes
        assert "overlap between neighboring lambda windows falls to 0.010" in notes

    def test_a_converged_looking_run_has_no_note(self, tmp_path, stand_in):
        _, answer = stand_in
        answer["value"] = found()
        outcome = run_ok({"smiles": "C"}, tmp_path)
        assert not outcome.steps["hfe"].notes

    def test_a_run_that_returns_too_few_repeats_is_refused(self, tmp_path, stand_in):
        _, answer = stand_in
        answer["value"] = found(repeats=2)
        assert "repeat count" in refusal({"smiles": "C", "repeats": 3}, tmp_path)

    def test_native_records_are_archived_only_when_asked_for(self, tmp_path, stand_in):
        _, answer = stand_in
        answer["value"] = found()
        plain = run_ok({"smiles": "C"}, tmp_path, "plain").steps["hfe"].result
        kept = (
            run_ok({"smiles": "C", "keep_native_records": True}, tmp_path, "kept")
            .steps["hfe"]
            .result
        )
        assert plain.native_records is None
        assert kept.native_records and Path(kept.native_records).is_file()
        assert Path(kept.record).is_file()

    def test_the_report_lists_each_repeat(self, tmp_path, stand_in):
        from alambiq.capabilities.investigate.molecule_compute_hydration_free_energy_report import (
            render,
        )

        _, answer = stand_in
        answer["value"] = found()
        html = render(run_ok({"smiles": "C"}, tmp_path).steps["hfe"].result)
        assert "hydration free energy -4.2" in html
        assert "over 3 repeats" in html
        assert "Minimum effective samples" in html


@engine_only
@pytest.mark.usefixtures("engine")
class TestTheMoleculeIsChecked:
    @pytest.mark.parametrize(
        ("smiles", "message"),
        [
            ("C(C", "is not a SMILES string that RDKit can read"),
            ("[Na+].[Cl-]", "one molecule"),
            ("CC.O", "one molecule"),
            ("C[NH3+]", "charged atoms"),
            ("[NH3+]CC([O-])=O", "charged atoms"),
            ("[CH3]", "unpaired electrons"),
            ("C[Si](C)(C)C", "elements Si are not supported"),
            ("CC(F)Cl", "stereocenter with no specified configuration"),
            ("C" * 51, "51 heavy atoms; at most 50 are accepted"),
        ],
    )
    def test_a_molecule_the_method_cannot_treat_is_refused_with_the_reason(
        self, tmp_path, smiles, message
    ):
        params = {"smiles": smiles, "repeats": 1, "production_ns": 0.05}
        assert message in refusal(params, tmp_path)


@gpu_only
@pytest.mark.usefixtures("engine")
class TestTheSimulation:
    def test_methane_has_a_small_positive_hydration_free_energy(self, tmp_path):
        outcome = run_ok(
            {
                "smiles": "C",
                "repeats": 2,
                "production_ns": 0.05,
                "equilibration_ns": 0.05,
                "platform": "CUDA",
            },
            tmp_path,
        )
        result = outcome.steps["hfe"].result
        assert 0.0 < result.hydration_free_energy_kcal_mol < 5.0
        assert result.uncertainty_kcal_mol is not None
        assert len(result.repeats) == 2
        assert result.versions["openfe"] == "1.12.0"
