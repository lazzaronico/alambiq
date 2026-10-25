"""The run record, the history of a run directory, and the replay of a step.

The answers are known from how the pieces are built: a run writes one record naming each
step's capability, version, resolved request and reuse key; running the same campaign
again in the same directory adds an entry to the history and reuses every step; the
record and a replay survive moving the directory; and a replay, run in a fresh directory,
returns what the original step returned.
"""

from __future__ import annotations

import json
import shutil
from importlib.util import find_spec
from pathlib import Path

import pytest

from alambiq import cli
from alambiq.api import history, replay, run

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy")),
    reason="the quantitative extra is required",
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/kinetics-from-measurements/campaign.yaml"
FIXTURES = ROOT / "tests/fixtures/operation"


def run_example(run_dir: Path, **options):
    outcome = run(EXAMPLE, run_dir=run_dir, **options)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def stored(outcome, step: str) -> dict:
    return json.loads(Path(outcome.outputs(step)["record"]).read_text())


class TestRunRecord:
    def test_it_names_each_step_its_capability_version_request_and_key(self, tmp_path):
        outcome = run_example(tmp_path / "run")
        record = json.loads((tmp_path / "run/run.json").read_text())
        assert record == outcome.record
        assert record["campaign"]["id"] == "kinetics-from-measurements" and record["ok"] is True
        assert [s["id"] for s in record["steps"]] == ["intake", "fit"]
        intake, fit = record["steps"]
        assert (intake["capability"], fit["capability"]) == (
            "assay.inspect",
            "assay.fit",
        )
        assert fit["version"] and len(fit["key"]) == 64 and fit["status"] == "completed"
        # The fit's request is what the capability received: the bound record, the
        # model, and the defaults it filled in.
        assert fit["request"]["model"] == "michaelis_menten"
        assert fit["request"]["measurements"].startswith("{store}")
        assert record["alambiq"] and record["python"]

    def test_the_same_campaign_has_the_same_keys_wherever_it_runs(self, tmp_path):
        first = run_example(tmp_path / "a").record
        second = run_example(tmp_path / "b").record
        assert [s["key"] for s in first["steps"]] == [s["key"] for s in second["steps"]]
        assert [s["request"] for s in first["steps"]] == [
            s["request"] for s in second["steps"]
        ]

    def test_a_failed_step_is_recorded_with_its_error(self, tmp_path):
        outcome = run(
            {
                "id": "broken",
                "steps": [
                    {
                        "id": "fit",
                        "capability": "assay.fit",
                        "params": {"measurements": "nowhere.json", "no_such_field": 1},
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        assert not outcome.ok
        (step,) = json.loads((tmp_path / "run/run.json").read_text())["steps"]
        assert step["status"] == "failed" and "no_such_field" in step["error"]
        assert step["request"] is None
        assert outcome.record["ok"] is False

    def test_a_request_holding_a_nan_is_recorded_not_fatal(self, tmp_path):
        outcome = run(
            {
                "id": "odd",
                "steps": [
                    {
                        "id": "pockets",
                        "capability": "operation.simulate",
                        "params": {
                            "definition": str(FIXTURES / "first-order.json"),
                            "solver": float("nan"),
                        },
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        assert not outcome.ok
        record = json.loads((tmp_path / "run/run.json").read_text())
        assert record["steps"][0]["status"] == "failed"

    def test_the_record_follows_the_directory_when_it_is_moved(self, tmp_path):
        run_example(tmp_path / "run")
        shutil.move(tmp_path / "run", tmp_path / "moved")
        assert len(history(tmp_path / "moved")) == 1
        target = replay(tmp_path / "moved", "fit", tmp_path / "replay")
        assert str(tmp_path / "run") not in target.read_text()


class TestHistory:
    def test_each_run_in_a_directory_adds_an_entry_and_the_second_reuses_everything(
        self, tmp_path
    ):
        run_example(tmp_path / "run")
        again = run_example(tmp_path / "run")
        rows = history(tmp_path / "run")
        assert [(r["computed"], r["reused"]) for r in rows] == [(2, 0), (0, 2)]
        assert all(r["campaign"] == "kinetics-from-measurements" and r["ok"] for r in rows)
        assert rows[0]["started"] <= rows[1]["started"]
        assert all(s.reused for s in again.steps.values())

    def test_the_report_lists_earlier_runs_only_when_there_were_some(self, tmp_path):
        first = run_example(tmp_path / "run")
        assert "History" not in first.report(tmp_path / "one.html").read_text()
        second = run_example(tmp_path / "run")
        page = second.report(tmp_path / "two.html").read_text()
        assert 'id="history"' in page and "2 runs of kinetics-from-measurements" in page
        assert page.count("<tr>") >= 3

    def test_an_edit_that_changes_a_step_shows_up_as_a_computed_step(self, tmp_path):
        campaign = json.loads(
            json.dumps(__import__("yaml").safe_load(EXAMPLE.read_text()))
        )
        for step in campaign["steps"]:
            if step["id"] == "intake":
                step["params"]["table"] = str(EXAMPLE.parent / "rates.csv")
        run(campaign, run_dir=tmp_path / "run")
        campaign["steps"][1]["params"]["model"] = "linear"
        run(campaign, run_dir=tmp_path / "run")
        rows = history(tmp_path / "run")
        assert (rows[1]["computed"], rows[1]["reused"]) == (1, 1)

    def test_a_directory_with_no_runs_has_no_history(self, tmp_path):
        assert history(tmp_path) == []


def comparable(outputs: dict) -> dict:
    """Outputs without the paths of stored files, which differ by run directory."""
    return {
        k: comparable(v) if isinstance(v, dict) else v
        for k, v in outputs.items()
        if not (isinstance(v, str) and Path(v).is_absolute() and Path(v).exists())
    }


class TestReplay:
    def test_each_step_of_the_example_reproduces_its_result_in_a_fresh_directory(
        self, tmp_path
    ):
        original = run_example(tmp_path / "run")
        for step in ("intake", "fit"):
            target = replay(tmp_path / "run", step, tmp_path / f"replay-{step}")
            again = run(target, run_dir=tmp_path / f"again-{step}")
            assert again.ok, [s.error for s in again.steps.values() if s.error]
            assert not again.steps[step].reused
            assert comparable(again.outputs(step)) == comparable(original.outputs(step))
        # The fitted record itself, not only its summary, is the same.
        fitted = run(
            tmp_path / "replay-fit/campaign.yaml", run_dir=tmp_path / "again-fit-2"
        )
        assert (
            stored(fitted, "fit")["parameters"] == stored(original, "fit")["parameters"]
        )
        assert (
            stored(fitted, "fit")["observations"]
            == stored(original, "fit")["observations"]
        )

    def test_a_simulation_replay_gives_the_same_trajectory(self, tmp_path):
        campaign = {
            "id": "decay",
            "steps": [
                {
                    "id": "batch",
                    "capability": "operation.simulate",
                    "params": {"definition": str(FIXTURES / "first-order.json")},
                }
            ],
        }
        original = run(campaign, run_dir=tmp_path / "run")
        target = replay(tmp_path / "run", "batch", tmp_path / "replay")
        again = run(target, run_dir=tmp_path / "again")
        assert again.ok
        assert (
            stored(again, "batch")["trajectory"]
            == stored(original, "batch")["trajectory"]
        )

    def test_the_replay_carries_its_own_inputs_and_names_none_of_the_original(
        self, tmp_path
    ):
        run_example(tmp_path / "run")
        target = replay(tmp_path / "run", "fit", tmp_path / "replay")
        text = target.read_text()
        assert str(tmp_path / "run") not in text and "{store}" not in text
        assert text.startswith("# Replays step 'fit'")
        copies = list((tmp_path / "replay/inputs").iterdir())
        assert len(copies) == 1 and copies[0].suffix == ".json"
        moved = tmp_path / "elsewhere"
        shutil.copytree(tmp_path / "replay", moved)
        shutil.rmtree(tmp_path / "run")
        again = run(moved / "campaign.yaml", run_dir=tmp_path / "again")
        assert again.ok

    def test_an_unknown_step_is_refused_with_the_steps_there_are(self, tmp_path):
        run_example(tmp_path / "run")
        with pytest.raises(KeyError, match="intake, fit"):
            replay(tmp_path / "run", "nope", tmp_path / "out")

    def test_a_step_that_failed_before_its_request_existed_cannot_be_replayed(
        self, tmp_path
    ):
        run(
            {
                "id": "broken",
                "steps": [
                    {
                        "id": "fit",
                        "capability": "assay.fit",
                        "params": {"measurements": "x.json", "no_such_field": 1},
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        with pytest.raises(ValueError, match="nothing to replay"):
            replay(tmp_path / "run", "fit", tmp_path / "out")

    def test_a_directory_without_a_record_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run.json"):
            replay(tmp_path, "fit", tmp_path / "out")


class TestCommands:
    def test_history_and_replay_are_commands(self, tmp_path, capsys):
        run_example(tmp_path / "run")
        assert cli.main(["history", str(tmp_path / "run")]) == 0
        assert "kinetics-from-measurements" in capsys.readouterr().out
        assert (
            cli.main(
                ["replay", str(tmp_path / "run"), "fit", "--out", str(tmp_path / "r")]
            )
            == 0
        )
        assert (tmp_path / "r/campaign.yaml").is_file()

    def test_they_say_what_is_missing(self, tmp_path, capsys):
        assert cli.main(["history", str(tmp_path)]) == 1
        assert "no recorded runs" in capsys.readouterr().err
        assert (
            cli.main(["replay", str(tmp_path), "fit", "--out", str(tmp_path / "o")])
            == 1
        )
        assert "run.json" in capsys.readouterr().err
