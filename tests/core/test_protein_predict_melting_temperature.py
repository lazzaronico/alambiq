"""protein.predict_melting_temperature estimates melting temperatures with TemBERTure Tm.

The request and aggregation tests need no model: a stand-in replaces the engine. The model tests
run the three pinned TemBERTure Tm replicas (set ALAMBIQ_ENV_THERMOSTABILITY to the Python of the
thermostability environment, with the `temberture-tm` resource installed).
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import pytest

ENV = os.environ.get("ALAMBIQ_ENV_THERMOSTABILITY", "")
CAPABILITY = "protein.predict_melting_temperature"
# A shikimate dehydrogenase; the TemBERTure repository uses it as its example.
SEQUENCE = (
    "MEKVYGLIGFPVEHSLSPLMHNDAFARLGIPARYHLFSVEPGQVGAAIAGVRALGIAGVNVTIPHKLAVIPFLDEVDEHARRIGAVNTII"
    "NNDGRLIGFNTDGPGYVQALEEEMNITLDGKRILVIGAGGGARGIYFSLLSTAAERIDMANRTVEKAERLVREGEGGRSAYFSLAEAETR"
    "LDEYDIIINTTSVGMHPRVEVQPLSLERLRPGVIVSNIIYNPLETKWLKEAKARGARVQNGVGMLVYQGALAFEKWTGQWPDVNRMKQLV"
    "IEALRR"
)
OTHER = "MQYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTE"


def model_installed() -> bool:
    if not ENV or not Path(ENV).is_file():
        return False
    from alambiq.core import resources

    try:
        resources.verify("temberture-tm")
    except Exception:
        return False
    return True


model_only = pytest.mark.skipif(
    not model_installed(),
    reason="set ALAMBIQ_ENV_THERMOSTABILITY and install the temberture-tm resource",
)


def fasta(tmp_path: Path, records: dict[str, str], name: str = "p.faa") -> str:
    path = tmp_path / name
    path.write_text("".join(f">{k}\n{v}\n" for k, v in records.items()))
    return str(path)


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "tm", "capability": CAPABILITY, "params": params}
    return execute({"id": "m", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def refusal(params: dict, tmp_path: Path) -> str:
    outcome = run(params, tmp_path)
    assert not outcome.ok
    return " ".join(x.error or "" for x in outcome.steps.values())


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """Replace the resource check and the engine by recorded answers."""
    from alambiq.core import envs, resources

    calls = []
    answer = {"values": None}

    def fake(environment, entry, request, **kwargs):
        calls.append((environment, entry, request, kwargs))
        return {
            "replica_values": answer["values"],
            "versions": {"torch": "2.15.0", "adapters": "1.3.0"},
            "device": request["device"],
            "seconds": 1.5,
        }

    monkeypatch.setattr(envs, "run_in_environment", fake)
    monkeypatch.setattr(resources, "verify", lambda name: tmp_path)
    monkeypatch.setenv("ALAMBIQ_ENV_THERMOSTABILITY", ENV or sys.executable)
    return calls, answer


class TestRequestsAreRefusedBeforeAnythingRuns:
    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"batch_size": 0}, "batch_size must be an integer from 1 to 256"),
            ({"batch_size": 257}, "batch_size must be an integer from 1 to 256"),
            ({"threads": 0}, "threads must be an integer from 1 to 64"),
            ({"threads": 65}, "threads must be an integer from 1 to 64"),
        ],
    )
    def test_a_setting_out_of_range_is_refused(
        self, stand_in, tmp_path, params, message
    ):
        path = fasta(tmp_path, {"a": SEQUENCE})
        assert message in refusal({"proteins": path, **params}, tmp_path)

    @pytest.mark.parametrize(
        ("records", "message"),
        [
            ({"a": "MKV"}, "a must have 10 to 5,000 residues"),
            ({"a": SEQUENCE[:50] + "B" + SEQUENCE[51:]}, "a must have 10 to 5,000"),
            ({"a": SEQUENCE[:50] + "U" + SEQUENCE[51:]}, "a must have 10 to 5,000"),
            ({"a": "M" * 5001}, "a must have 10 to 5,000 residues"),
        ],
    )
    def test_a_protein_the_model_cannot_read_is_refused_by_name(
        self, stand_in, tmp_path, records, message
    ):
        assert message in refusal({"proteins": fasta(tmp_path, records)}, tmp_path)

    def test_a_request_of_more_than_1000_proteins_is_refused(self, stand_in, tmp_path):
        records = {f"p{i}": SEQUENCE[:60] for i in range(1001)}
        message = refusal({"proteins": fasta(tmp_path, records)}, tmp_path)
        assert "holds 1001 proteins; at most 1000" in message

    def test_duplicate_identifiers_are_refused(self, stand_in, tmp_path):
        path = tmp_path / "d.faa"
        path.write_text(f">a\n{SEQUENCE}\n>a\n{OTHER}\n")
        assert "appears twice" in refusal({"proteins": str(path)}, tmp_path)


class TestTheResultIsAssembledFromTheReplicas:
    def test_the_estimate_is_the_mean_and_the_spread_the_standard_deviation(
        self, stand_in, tmp_path
    ):
        _, answer = stand_in
        answer["values"] = [[70.0, 74.0, 78.0], [40.0, 41.0, 42.0]]
        result = (
            run_ok(
                {"proteins": fasta(tmp_path, {"a": SEQUENCE, "b": OTHER + "AAAA"})},
                tmp_path,
            )
            .steps["tm"]
            .result
        )
        first, second = result.proteins
        assert (first.id, second.id) == ("a", "b")
        assert first.melting_temperature_c == pytest.approx(74.0)
        assert first.spread_c == pytest.approx(statistics.stdev([70.0, 74.0, 78.0]))
        assert first.replica_values_c == [70.0, 74.0, 78.0]
        assert second.melting_temperature_c == pytest.approx(41.0)
        assert first.length == len(SEQUENCE) and not first.truncated

    def test_the_engine_receives_the_sequences_the_device_and_the_resource(
        self, stand_in, tmp_path
    ):
        calls, answer = stand_in
        answer["values"] = [[60.0, 61.0, 62.0]]
        run_ok(
            {
                "proteins": fasta(tmp_path, {"a": SEQUENCE}),
                "batch_size": 4,
                "device": "cuda",
                "threads": 3,
            },
            tmp_path,
        )
        environment, entry, request, kwargs = calls[0]
        assert (environment, entry) == (
            "thermostability",
            "thermostability_predict:run",
        )
        assert request == {
            "sequences": [SEQUENCE],
            "model_root": str(tmp_path),
            "batch_size": 4,
            "device": "cuda",
            "threads": 3,
        }
        assert kwargs["timeout"] > 0

    def test_a_protein_beyond_510_residues_is_marked_and_noted(
        self, stand_in, tmp_path
    ):
        _, answer = stand_in
        answer["values"] = [[60.0, 61.0, 62.0], [60.0, 61.0, 62.0]]
        long = (SEQUENCE * 3)[:511]
        outcome = run_ok(
            {"proteins": fasta(tmp_path, {"long": long, "edge": long[:510]})},
            tmp_path,
        )
        long_one, edge = outcome.steps["tm"].result.proteins
        assert long_one.truncated and not edge.truncated
        assert "1 protein(s) are longer than 510 residues" in " ".join(
            outcome.steps["tm"].notes
        )

    def test_a_wide_spread_carries_a_note_and_a_narrow_one_does_not(
        self, stand_in, tmp_path
    ):
        _, answer = stand_in
        answer["values"] = [[40.0, 60.0, 80.0]]
        outcome = run_ok({"proteins": fasta(tmp_path, {"a": SEQUENCE})}, tmp_path)
        assert "spread of more than 15" in " ".join(outcome.steps["tm"].notes)
        answer["values"] = [[60.0, 61.0, 62.0]]
        quiet = run_ok({"proteins": fasta(tmp_path, {"a": SEQUENCE})}, tmp_path, "q")
        assert not quiet.steps["tm"].notes

    def test_the_record_keeps_every_replica_value(self, stand_in, tmp_path):
        import json

        _, answer = stand_in
        answer["values"] = [[60.0, 61.0, 62.0]]
        result = (
            run_ok({"proteins": fasta(tmp_path, {"a": SEQUENCE})}, tmp_path)
            .steps["tm"]
            .result
        )
        record = json.loads(Path(result.record).read_text())
        assert record["predictions"][0]["replica_values_c"] == [60.0, 61.0, 62.0]
        assert record["model"] == "TemBERTure Tm"
        assert record["resource_identity"] == result.resource_identity

    def test_the_report_lists_each_protein_with_its_replicas(self, stand_in, tmp_path):
        from alambiq.capabilities.characterize.protein_predict_melting_temperature_report import (
            render,
        )

        _, answer = stand_in
        answer["values"] = [[60.0, 61.0, 62.0]]
        result = (
            run_ok({"proteins": fasta(tmp_path, {"enzyme1": SEQUENCE})}, tmp_path)
            .steps["tm"]
            .result
        )
        html = render(result)
        assert "enzyme1" in html and "Replica 3" in html and "61" in html


@model_only
class TestTheModel:
    @pytest.fixture(scope="class")
    def predicted(self, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_THERMOSTABILITY"] = ENV
        tmp = tmp_path_factory.mktemp("tm")
        records = {"enzyme": SEQUENCE, "domain": SEQUENCE[:120], "gb3": OTHER}
        outcome = run_ok(
            {"proteins": fasta(tmp, records), "batch_size": 2, "threads": 4}, tmp
        )
        return tmp, records, outcome.steps["tm"].result

    def test_every_protein_gets_three_finite_values_in_a_plausible_range(
        self, predicted
    ):
        _, records, result = predicted
        assert [p.id for p in result.proteins] == list(records)
        for protein in result.proteins:
            assert len(protein.replica_values_c) == 3
            assert all(0.0 < v < 120.0 for v in protein.replica_values_c)

    def test_the_prediction_does_not_depend_on_the_batch_it_is_in(
        self, predicted, tmp_path
    ):
        _, _, together = predicted
        alone = (
            run_ok(
                {"proteins": fasta(tmp_path, {"enzyme": SEQUENCE}), "batch_size": 1},
                tmp_path,
            )
            .steps["tm"]
            .result
        )
        mine = together.proteins[0].replica_values_c
        for a, b in zip(mine, alone.proteins[0].replica_values_c, strict=True):
            assert a == pytest.approx(b, abs=0.05)

    def test_a_repeated_run_gives_the_same_values(self, predicted, tmp_path):
        _, _, first = predicted
        second = (
            run_ok(
                {
                    "proteins": fasta(
                        tmp_path,
                        {"enzyme": SEQUENCE, "domain": SEQUENCE[:120], "gb3": OTHER},
                    ),
                    "batch_size": 2,
                    "threads": 4,
                },
                tmp_path,
            )
            .steps["tm"]
            .result
        )
        for a, b in zip(first.proteins, second.proteins, strict=True):
            assert a.replica_values_c == pytest.approx(b.replica_values_c, abs=1e-4)

    def test_residues_beyond_the_first_510_do_not_change_the_prediction(self, tmp_path):
        os.environ["ALAMBIQ_ENV_THERMOSTABILITY"] = ENV
        base = (SEQUENCE * 3)[:510]
        records = {"cut": base, "longer": base + "WWWWWWWWWWWWWWWWWWWW"}
        result = (
            run_ok({"proteins": fasta(tmp_path, records)}, tmp_path).steps["tm"].result
        )
        cut, longer = result.proteins
        assert longer.truncated and not cut.truncated
        assert longer.replica_values_c == pytest.approx(cut.replica_values_c, abs=0.05)

    def test_the_versions_of_the_libraries_are_reported(self, predicted):
        _, _, result = predicted
        assert result.engine_versions["adapters"]
        assert result.engine_versions["torch"]
