"""The pilot capabilities, end to end through the new core.

These run real science through the new path: YAML to loader to cache to
execution to report. The numerical assertion is the one that matters — a known
Michaelis-Menten curve must come back with the parameters that generated it —
so this is a check on the core's plumbing and on the science together.
"""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy")),
    reason="the quantitative extra is required",
)

XS = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 8, 20, 30]
KM, VMAX = 2.5, 180.0

SELECT = {
    "sample_id": "enzyme",
    "metric": "initial-rate",
    "x_quantity": "substrate",
    "x_unit": "mM",
    "y_unit": "uM/min",
}


@pytest.fixture
def measurements(tmp_path: Path) -> Path:
    """A saved assay record on a known curve."""
    from alambiq.capabilities.measure.science.inspection import inspect_assay

    header = (
        "measurement_id,sample_id,condition_id,metric,value,unit,"
        "independent_id,technical_id,role,status\n"
    )
    rows = "".join(
        f"m{i},enzyme,c{i},initial-rate,{VMAX * x / (KM + x)},uM/min,,,sample,measured\n"
        for i, x in enumerate(XS)
    )
    record = inspect_assay(
        header + rows,
        {f"c{i}": f"Substrate {x} mM" for i, x in enumerate(XS)},
        None,
        condition_values={
            f"c{i}": {"substrate": {"value": x, "unit": "mM"}}
            for i, x in enumerate(XS)
        },
    )
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(record))
    return path


class TestAssayFit:
    def test_a_known_curve_comes_back_through_the_new_core(self, measurements, tmp_path):
        from alambiq.api import run

        campaign = {
            "id": "kinetics",
            "steps": [
                {
                    "id": "fit",
                    "capability": "assay.fit",
                    "params": {
                        "measurements": str(measurements),
                        "model": "michaelis_menten",
                        **SELECT,
                    },
                }
            ],
        }
        outcome = run(campaign, run_dir=tmp_path / "run")
        assert outcome.ok, outcome.steps["fit"].error

        found = {p["name"]: p for p in outcome.outputs("fit")["parameters"]}
        assert found["Km"]["value"] == pytest.approx(KM, abs=1e-6)
        assert found["Vmax"]["value"] == pytest.approx(VMAX, abs=1e-4)
        assert found["Km"]["unit"] == "mM"
        assert outcome.outputs("fit")["observations_used"] == len(XS)

    def test_the_same_campaign_as_yaml_gives_the_same_answer(
        self, measurements, tmp_path
    ):
        from alambiq.api import run

        document = {
            "id": "kinetics",
            "steps": [
                {
                    "id": "fit",
                    "capability": "assay.fit",
                    "preset": "initial-rates",
                    "params": {"measurements": str(measurements), **SELECT},
                }
            ],
        }
        path = tmp_path / "campaign.yaml"
        path.write_text(yaml.safe_dump(document))

        from_yaml = run(path, run_dir=tmp_path / "a")
        from_python = run(document, run_dir=tmp_path / "b")
        assert from_yaml.outputs("fit")["parameters"] == (
            from_python.outputs("fit")["parameters"]
        )

    def test_a_second_run_reuses_the_fit(self, measurements, tmp_path):
        from alambiq.api import run

        campaign = {
            "id": "kinetics",
            "steps": [
                {
                    "id": "fit",
                    "capability": "assay.fit",
                    "params": {"measurements": str(measurements), **SELECT},
                }
            ],
        }
        run(campaign, run_dir=tmp_path / "run")
        again = run(campaign, run_dir=tmp_path / "run")
        assert again.steps["fit"].reused is True

    def test_editing_the_measurements_invalidates_the_reuse(
        self, measurements, tmp_path
    ):
        from alambiq.api import run

        campaign = {
            "id": "kinetics",
            "steps": [
                {
                    "id": "fit",
                    "capability": "assay.fit",
                    "params": {"measurements": str(measurements), **SELECT},
                }
            ],
        }
        run(campaign, run_dir=tmp_path / "run")
        # The cache key covers input file digests, not just the request.
        record = json.loads(measurements.read_text())
        measurements.write_text(json.dumps({**record, "note": "changed"}))
        again = run(campaign, run_dir=tmp_path / "run")
        assert again.steps["fit"].reused is False

    def test_the_report_states_what_the_numbers_mean(self, measurements, tmp_path):
        from alambiq.api import run

        outcome = run(
            {
                "id": "kinetics",
                "steps": [
                    {
                        "id": "fit",
                        "capability": "assay.fit",
                        "params": {"measurements": str(measurements), **SELECT},
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "Km" in page and "Vmax" in page
        assert "michaelis-menten" in page
        assert "http" not in page

    def test_a_missing_selection_field_names_itself(self, measurements, tmp_path):
        from alambiq.api import run

        outcome = run(
            {
                "id": "kinetics",
                "steps": [
                    {
                        "id": "fit",
                        "capability": "assay.fit",
                        "params": {"measurements": str(measurements)},
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        assert outcome.steps["fit"].status == "failed"
        assert "sample_id is required" in outcome.steps["fit"].error


class TestCapabilityIntrospection:
    def test_every_capability_documents_itself(self):
        from alambiq.api import capability_list, describe

        for row in capability_list():
            described = describe(row["id"])
            assert described["summary"], row["id"]
            # The documentation lives on the function, so it cannot drift.
            assert len(described["documentation"]) > 200, row["id"]
            assert described["request"], row["id"]

    def test_documentation_states_limits_not_only_behaviour(self):
        from alambiq.api import describe

        for capability, phrase in [
            ("assay.fit", "conditional on the recorded experiment and the chosen model"),
            ("operation.simulate", "no declared dependence"),
            ("protein.search", "homologous proteins can differ in function"),
        ]:
            assert phrase in " ".join(describe(capability)["documentation"].lower().split())


class TestMeasurementChain:
    """Intake and fitting compose: one step's record is the next step's input."""

    @pytest.fixture
    def table(self, tmp_path: Path) -> Path:
        rows = "".join(
            f"m{i},enzyme,c{i},initial-rate,{VMAX * x / (KM + x)},uM/min,,,sample,measured\n"
            for i, x in enumerate(XS)
        )
        # One reading that failed, which must not be fitted as a value.
        rows += f"m{len(XS)},enzyme,c0,initial-rate,,uM/min,,,sample,failed\n"
        path = tmp_path / "rates.csv"
        path.write_text(
            "measurement_id,sample_id,condition_id,metric,value,unit,"
            "independent_id,technical_id,role,status\n" + rows
        )
        return path

    def campaign(self, table: Path) -> dict:
        return {
            "id": "kinetics",
            "steps": [
                {
                    "id": "intake",
                    "capability": "assay.inspect",
                    "params": {
                        "table": str(table),
                        "conditions": {
                            f"c{i}": f"Substrate {x} mM" for i, x in enumerate(XS)
                        },
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
                    "params": SELECT | {"model": "michaelis_menten"},
                },
            ],
        }

    def test_intake_feeds_the_fit_and_the_curve_is_recovered(self, table, tmp_path):
        from alambiq.api import run

        outcome = run(self.campaign(table), run_dir=tmp_path / "run")
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]

        intake = outcome.outputs("intake")
        assert intake["measurements"] == len(XS) + 1
        assert intake["failed"] == 1

        found = {p["name"]: p["value"] for p in outcome.outputs("fit")["parameters"]}
        assert found["Km"] == pytest.approx(KM, abs=1e-4)
        assert found["Vmax"] == pytest.approx(VMAX, abs=1e-2)
        # The failed reading is not fitted as a value.
        assert outcome.outputs("fit")["observations_used"] == len(XS)

    def test_a_record_is_stored_rather_than_carried_inline(self, table, tmp_path):
        from alambiq.api import run

        outcome = run(self.campaign(table), run_dir=tmp_path / "run")
        stored = Path(outcome.outputs("intake")["record"])
        # The step output names where the record is, not the record itself, so
        # the outputs and the cache stay small.
        assert stored.is_file()
        assert json.loads(stored.read_text())["rows"]
        assert "sha256" in str(stored)

    def test_the_failed_reading_is_reported_to_the_reader(self, table, tmp_path):
        from alambiq.api import run

        outcome = run(self.campaign(table), run_dir=tmp_path / "run")
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "1 reading failed" in page
        assert "Km" in page

    def test_changing_only_the_fit_reuses_the_intake(self, table, tmp_path):
        from alambiq.api import run

        run(self.campaign(table), run_dir=tmp_path / "run")
        changed = self.campaign(table)
        changed["steps"][1]["params"] = SELECT | {"model": "linear"}
        again = run(changed, run_dir=tmp_path / "run")
        assert again.steps["intake"].reused is True
        assert again.steps["fit"].reused is False
