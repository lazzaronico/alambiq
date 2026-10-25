"""Saved results survive being moved, and reuse follows contents, not locations.

A researcher moves a run directory, copies a project to another machine, or
points a campaign at the same data in a new place. None of that should cost
the results already computed, and none of it should leave a step reading a
path that no longer exists. These tests run a real two-step campaign, move it,
and run it again.
"""

from __future__ import annotations

import json
import shutil
from importlib.util import find_spec
from pathlib import Path

import pytest

from alambiq.core.store import Store

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy")),
    reason="the quantitative extra is required",
)

XS = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 8, 20, 30]
HEAD = (
    "measurement_id,sample_id,condition_id,metric,value,unit,"
    "independent_id,technical_id,role,status\n"
)


def write_table(directory: Path, scale: float = 1.0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rates.csv"
    path.write_text(
        HEAD
        + "".join(
            f"m{i},enzyme,c{i},initial-rate,{scale * 180 * x / (2.5 + x)},uM/min,,,sample,measured\n"
            for i, x in enumerate(XS)
        )
    )
    return path


def campaign(table: str) -> dict:
    return {
        "id": "kinetics",
        "steps": [
            {
                "id": "intake",
                "capability": "assay.inspect",
                "params": {
                    "table": table,
                    "conditions": {f"c{i}": f"S {x} mM" for i, x in enumerate(XS)},
                    "condition_values": {
                        f"c{i}": {"substrate": {"value": x, "unit": "mM"}}
                        for i, x in enumerate(XS)
                    },
                },
            },
            {
                "id": "fit",
                "capability": "assay.fit",
                "bind": {"measurements": "intake.record"},
                "params": {
                    "sample_id": "enzyme",
                    "metric": "initial-rate",
                    "x_quantity": "substrate",
                    "x_unit": "mM",
                    "y_unit": "uM/min",
                },
            },
        ],
    }


def run(data: dict, run_dir: Path):
    from alambiq.api import run as execute

    outcome = execute(data, run_dir=run_dir)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def states(outcome) -> dict[str, bool]:
    return {name: step.reused for name, step in outcome.steps.items()}


class TestTheStoreNamesItself:
    def test_a_location_becomes_a_token_and_back(self, tmp_path):
        store = Store(tmp_path / "artifacts")
        record = {
            "path": f"{store.root}/sha256/ab/abc",
            "nested": [{"also": f"see {store.root}/sha256/cd/cde"}],
            "number": 3,
        }
        portable = store.portable(record)
        assert str(store.root) not in json.dumps(portable)
        assert Store.TOKEN in portable["path"]
        assert store.local(portable) == record

    def test_the_token_resolves_to_wherever_the_store_now_is(self, tmp_path):
        old, new = Store(tmp_path / "old"), Store(tmp_path / "new")
        portable = old.portable({"path": f"{old.root}/sha256/ab/abc"})
        assert new.local(portable)["path"] == f"{new.root}/sha256/ab/abc"


class TestMovingARunDirectory:
    def test_every_step_is_reused_and_points_into_the_new_location(self, tmp_path):
        table = write_table(tmp_path / "data")
        first = run(campaign(str(table)), tmp_path / "a" / "run")
        assert states(first) == {"intake": False, "fit": False}

        shutil.move(tmp_path / "a" / "run", tmp_path / "b" / "run")
        second = run(campaign(str(table)), tmp_path / "b" / "run")

        assert states(second) == {"intake": True, "fit": True}
        for step in ("intake", "fit"):
            record = Path(second.outputs(step)["record"])
            assert record.is_file()
            assert str((tmp_path / "b" / "run").resolve()) in str(record)
            assert "/a/run" not in str(record)

    def test_the_answer_is_the_one_computed_before_the_move(self, tmp_path):
        table = write_table(tmp_path / "data")
        first = run(campaign(str(table)), tmp_path / "a" / "run")
        shutil.move(tmp_path / "a" / "run", tmp_path / "b" / "run")
        second = run(campaign(str(table)), tmp_path / "b" / "run")
        assert second.outputs("fit")["parameters"] == first.outputs("fit")["parameters"]

    def test_saved_records_hold_no_absolute_location(self, tmp_path):
        run_dir = tmp_path / "run"
        run(campaign(str(write_table(tmp_path / "data"))), run_dir)
        records = list((run_dir / "cache").rglob("*.json"))
        assert len(records) == 2
        for path in records:
            text = path.read_text()
            assert str(run_dir.resolve()) not in text
            assert Store.TOKEN in text

    def test_a_whole_project_can_be_copied_elsewhere(self, tmp_path):
        """A campaign with relative paths, its data and its run, copied together."""
        project = tmp_path / "project"
        write_table(project)
        (project / "campaign.yaml").write_text(json.dumps(campaign("rates.csv")))
        from alambiq.api import run as execute

        first = execute(str(project / "campaign.yaml"), run_dir=project / "run")
        assert first.ok

        shutil.copytree(project, tmp_path / "elsewhere")
        again = execute(
            str(tmp_path / "elsewhere" / "campaign.yaml"),
            run_dir=tmp_path / "elsewhere" / "run",
        )
        assert again.ok and states(again) == {"intake": True, "fit": True}


class TestReuseFollowsContents:
    def test_the_same_file_in_another_place_is_the_same_input(self, tmp_path):
        run_dir = tmp_path / "run"
        run(campaign(str(write_table(tmp_path / "one"))), run_dir)
        again = run(campaign(str(write_table(tmp_path / "two"))), run_dir)
        assert states(again) == {"intake": True, "fit": True}

    def test_a_renamed_copy_is_a_different_input(self, tmp_path):
        run_dir = tmp_path / "run"
        table = write_table(tmp_path / "data")
        run(campaign(str(table)), run_dir)
        renamed = tmp_path / "data" / "other-name.csv"
        shutil.copy(table, renamed)
        again = run(campaign(str(renamed)), run_dir)
        # The file's name is part of what it is; the intake therefore runs again,
        # while a step downstream that receives the identical record does not.
        assert again.steps["intake"].reused is False
        assert again.steps["fit"].reused is True

    def test_a_changed_file_is_not_reused(self, tmp_path):
        run_dir = tmp_path / "run"
        table = write_table(tmp_path / "data")
        first = run(campaign(str(table)), run_dir)
        write_table(tmp_path / "data", scale=1.1)
        again = run(campaign(str(table)), run_dir)
        assert states(again) == {"intake": False, "fit": False}
        # 10 % higher readings are a 10 % higher Vmax.
        before = {p["name"]: p["value"] for p in first.outputs("fit")["parameters"]}
        after = {p["name"]: p["value"] for p in again.outputs("fit")["parameters"]}
        assert after["Vmax"] == pytest.approx(1.1 * before["Vmax"], rel=1e-4)

    def test_only_the_downstream_step_recomputes_when_only_it_changes(self, tmp_path):
        run_dir = tmp_path / "run"
        table = str(write_table(tmp_path / "data"))
        run(campaign(table), run_dir)
        changed = campaign(table)
        changed["steps"][1]["params"]["model"] = "linear"
        again = run(changed, run_dir)
        assert states(again) == {"intake": True, "fit": False}


class TestReusedStepsAreShownLikeFreshOnes:
    def test_the_report_of_a_reused_step_is_the_capabilitys_own_rendering(
        self, tmp_path
    ):
        run_dir = tmp_path / "run"
        table = str(write_table(tmp_path / "data"))
        run(campaign(table), run_dir)
        again = run(campaign(table), run_dir)
        assert states(again) == {"intake": True, "fit": True}

        page = again.report(tmp_path / "report.html").read_text()
        # The coefficient table, not a raw dump of the saved outputs.
        assert "95 % interval" in page
        assert "<summary>Result</summary>" not in page

    def test_a_reused_result_is_an_object_not_a_dictionary(self, tmp_path):
        run_dir = tmp_path / "run"
        table = str(write_table(tmp_path / "data"))
        run(campaign(table), run_dir)
        again = run(campaign(table), run_dir)
        result = again.steps["fit"].result
        assert result is not None
        assert {p.name for p in result.parameters} == {"Km", "Vmax"}


class TestAStaleRecordIsAMiss:
    def test_a_saved_result_that_no_longer_fits_its_type_is_recomputed(self, tmp_path):
        run_dir = tmp_path / "run"
        table = str(write_table(tmp_path / "data"))
        run(campaign(table), run_dir)
        for path in (run_dir / "cache").rglob("*.json"):
            record = json.loads(path.read_text())
            if record["capability"] == "assay.fit":
                del record["outputs"]["model"]
                path.write_text(json.dumps(record))

        again = run(campaign(table), run_dir)
        # Not a failure and not a reuse of something that cannot be presented.
        assert states(again) == {"intake": True, "fit": False}
        assert again.steps["fit"].status == "completed"
