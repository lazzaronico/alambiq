"""Every option of `assay.inspect` and `assay.fit`, run through the new core.

Each test uses data generated from a known curve, Michaelis-Menten with
Km 2.5 and Vmax 180, so the right answer is known without running the code.
The options here are the ones that were once declared without ever being run,
which is how a bootstrap that could not be requested and an `exclude` that sent
the wrong key went unnoticed.
"""

from __future__ import annotations

import json
import math
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy")),
    reason="the quantitative extra is required",
)

XS = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 8, 20, 30]
KM, VMAX = 2.5, 180.0
HEAD = (
    "measurement_id,sample_id,condition_id,metric,value,unit,"
    "independent_id,technical_id,role,status\n"
)
SELECT = {
    "sample_id": "enzyme",
    "metric": "initial-rate",
    "x_quantity": "substrate",
    "x_unit": "mM",
    "y_unit": "uM/min",
}


def curve(x: float, factor: float = 1.0) -> float:
    return VMAX * x / (KM + x) * factor


def conditions(xs: list[float]) -> dict:
    return {f"c{i}": f"Substrate {x} mM" for i, x in enumerate(xs)}


def condition_values(xs: list[float]) -> dict:
    return {f"c{i}": {"substrate": {"value": x, "unit": "mM"}} for i, x in enumerate(xs)}


def write_table(tmp_path: Path, rows: list[str], name: str = "rates.csv") -> Path:
    path = tmp_path / name
    path.write_text(HEAD + "".join(rows))
    return path


def plain_rows(noise: float = 0.0, xs: list[float] = XS) -> list[str]:
    return [
        f"m{i},enzyme,c{i},initial-rate,{curve(x, 1 + noise * math.sin(7 * i + 1))},"
        "uM/min,,,sample,measured\n"
        for i, x in enumerate(xs)
    ]


def intake(table: Path, xs: list[float] = XS, **extra) -> dict:
    return {
        "id": "intake",
        "capability": "assay.inspect",
        "params": {
            "table": str(table),
            "conditions": conditions(xs),
            "condition_values": condition_values(xs),
            **extra,
        },
    }


def fit_step(**params) -> dict:
    return {
        "id": "fit",
        "capability": "assay.fit",
        "bind": {"measurements": "intake.record"},
        "params": {**SELECT, "model": "michaelis_menten", **params},
    }


def run_campaign(steps: list[dict], tmp_path: Path, name: str = "run"):
    from alambiq.api import run

    return run({"id": "options", "steps": steps}, run_dir=tmp_path / name)


def run_ok(steps: list[dict], tmp_path: Path, name: str = "run"):
    outcome = run_campaign(steps, tmp_path, name)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def coefficient(outcome, name: str) -> dict:
    return next(p for p in outcome.outputs("fit")["parameters"] if p["name"] == name)


def stored(outcome, step: str = "fit") -> dict:
    return json.loads(Path(outcome.outputs(step)["record"]).read_text())


class TestParameterBounds:
    def test_a_bound_below_the_truth_pins_the_coefficient_to_it(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_ok(
            [intake(table), fit_step(parameter_bounds={"Vmax": [0, 150]})], tmp_path
        )
        vmax, km = coefficient(outcome, "Vmax"), coefficient(outcome, "Km")
        # The unconstrained optimum is 180, outside the bound, so the
        # constrained optimum lies on it.
        assert vmax["value"] == pytest.approx(150.0, rel=1e-6)
        # Km moves to compensate for the capped plateau.
        assert km["value"] < KM
        # A coefficient set by a constraint has no standard error to report.
        assert vmax["standard_error"] is None
        notes = " ".join(outcome.steps["fit"].notes)
        assert "Vmax finished at its upper bound 150" in notes

    def test_an_unordered_pair_is_refused(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_campaign(
            [intake(table), fit_step(parameter_bounds={"Vmax": [150, 100]})], tmp_path
        )
        assert outcome.steps["fit"].status == "failed"
        assert "ordered" in outcome.steps["fit"].error

    def test_a_bound_that_does_not_bind_changes_nothing(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_ok(
            [intake(table), fit_step(parameter_bounds={"Vmax": [0, 1000]})], tmp_path
        )
        assert coefficient(outcome, "Vmax")["value"] == pytest.approx(VMAX, abs=1e-2)
        assert not any("bound" in n for n in outcome.steps["fit"].notes)


class TestFixedParameters:
    def test_fixing_a_coefficient_at_its_true_value_recovers_the_other(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        free = run_ok([intake(table), fit_step()], tmp_path, "free")
        fixed = run_ok(
            [intake(table), fit_step(fixed_parameters={"Km": 2.5})], tmp_path, "fixed"
        )
        km = coefficient(fixed, "Km")
        assert km["value"] == 2.5 and km["fixed"] is True
        assert km["standard_error"] is None
        assert coefficient(fixed, "Vmax")["value"] == pytest.approx(VMAX, abs=1e-2)
        # Fixing does not change which observations were fitted.
        assert (
            fixed.outputs("fit")["observations_used"]
            == free.outputs("fit")["observations_used"]
            == len(XS)
        )
        # A coefficient that was fixed is not "at a bound".
        assert not any("finished at" in n for n in fixed.steps["fit"].notes)

    def test_a_wrong_fixed_value_pulls_the_other_coefficient_and_worsens_the_fit(
        self, tmp_path
    ):
        table = write_table(tmp_path, plain_rows())
        free = run_ok([intake(table), fit_step()], tmp_path, "free")
        wrong = run_ok(
            [intake(table), fit_step(fixed_parameters={"Km": 5.0})], tmp_path, "wrong"
        )
        # With Km held at twice its true value, the best Vmax is larger.
        assert coefficient(wrong, "Vmax")["value"] == pytest.approx(212.26, abs=0.1)
        # The free fit of exact data has no residual; the wrong one does.
        assert stored(free)["statistics"]["rmse"] < 1e-6
        assert stored(wrong)["statistics"]["rmse"] == pytest.approx(11.71, abs=0.05)

    def test_fixing_both_coefficients_is_refused(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_campaign(
            [intake(table), fit_step(fixed_parameters={"Km": 2.5, "Vmax": 180.0})],
            tmp_path,
        )
        assert outcome.steps["fit"].status == "failed"
        assert "at least 1 coefficient must remain free" in outcome.steps["fit"].error

    def test_a_fixed_value_outside_the_bounds_is_refused(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_campaign(
            [
                intake(table),
                fit_step(
                    fixed_parameters={"Km": 2.5}, parameter_bounds={"Km": [3, 10]}
                ),
            ],
            tmp_path,
        )
        assert outcome.steps["fit"].status == "failed"
        assert "lies outside its bounds" in outcome.steps["fit"].error


class TestPredictions:
    ASKED = [0.4, 4.0, 100.0]

    def test_predictions_follow_the_curve_and_only_the_far_one_extrapolates(
        self, tmp_path
    ):
        table = write_table(tmp_path, plain_rows())
        outcome = run_ok([intake(table), fit_step(prediction_x=self.ASKED)], tmp_path)
        out = outcome.outputs("fit")
        assert out["measured_range"] == [0.05, 30.0]
        by_x = {p["x"]: p for p in out["predictions"]}
        for x in self.ASKED:
            assert by_x[x]["y"] == pytest.approx(curve(x), rel=1e-3)
        # The data end at 30, so only 100 lies outside what was measured.
        assert [by_x[x]["extrapolation"] for x in self.ASKED] == [False, False, True]
        assert any("outside the measured range" in n for n in outcome.steps["fit"].notes)

    def test_a_new_measurement_is_less_certain_than_the_curve(self, tmp_path):
        table = write_table(tmp_path, plain_rows(noise=0.02))
        outcome = run_ok([intake(table), fit_step(prediction_x=self.ASKED)], tmp_path)

        def width(pair):
            return pair[1] - pair[0]

        for p in outcome.outputs("fit")["predictions"]:
            assert width(p["prediction_interval_95"]) > width(p["mean_interval_95"]) > 0
        # With noise the fitted curve is no longer the generating one, so the
        # intervals are what carries the information, not the exact value.
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "extrapolated" in page
        assert "95 % interval of a new measurement" in page

    def test_no_predictions_are_in_the_result_unless_asked_for(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_ok([intake(table), fit_step()], tmp_path)
        assert outcome.outputs("fit")["predictions"] == []
        # The stored record keeps the default grid across the measured range.
        assert len(stored(outcome)["predictions"]) == 51


class TestExclude:
    def test_an_excluded_outlier_stops_distorting_the_fit(self, tmp_path):
        # Ten exact points, plus one more at 2.5 mM reading three times too high.
        xs = XS + [2.5]
        rows = plain_rows() + [
            f"mout,enzyme,c{len(XS)},initial-rate,{3 * curve(2.5)},uM/min,,,sample,measured\n"
        ]
        table = write_table(tmp_path, rows)
        kept = run_ok([intake(table, xs), fit_step()], tmp_path, "kept")
        dropped = run_ok(
            [intake(table, xs), fit_step(exclude=["mout"])], tmp_path, "dropped"
        )
        assert abs(coefficient(kept, "Vmax")["value"] - VMAX) > 1.0
        # The remaining ten points are exact, so the exact answer comes back.
        assert coefficient(dropped, "Km")["value"] == pytest.approx(KM, abs=1e-4)
        assert coefficient(dropped, "Vmax")["value"] == pytest.approx(VMAX, abs=1e-2)
        assert dropped.outputs("fit")["observations_used"] == 10
        assert kept.outputs("fit")["observations_used"] == 11

    def test_excluding_a_measurement_that_does_not_exist_is_refused(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_campaign([intake(table), fit_step(exclude=["nope"])], tmp_path)
        assert outcome.steps["fit"].status == "failed"
        assert "not in the series of the sample and the metric" in outcome.steps["fit"].error


class TestOtherFitOptions:
    def test_a_linear_model_recovers_a_line(self, tmp_path):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        rows = [
            f"m{i},enzyme,c{i},initial-rate,{4.0 + 3.0 * x},uM/min,,,sample,measured\n"
            for i, x in enumerate(xs)
        ]
        table = write_table(tmp_path, rows)
        outcome = run_ok([intake(table, xs), fit_step(model="linear")], tmp_path)
        assert coefficient(outcome, "intercept")["value"] == pytest.approx(4.0, abs=1e-6)
        assert coefficient(outcome, "slope")["value"] == pytest.approx(3.0, abs=1e-6)

    def test_restricting_the_conditions_fits_only_those(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_ok(
            [intake(table), fit_step(condition_ids=[f"c{i}" for i in range(8)])],
            tmp_path,
        )
        assert outcome.outputs("fit")["observations_used"] == 8
        assert outcome.outputs("fit")["measured_range"] == [0.05, 8.0]

    def test_conditions_that_differ_in_an_unmodelled_quantity_need_pooling_stated(
        self, tmp_path
    ):
        table = write_table(tmp_path, plain_rows())
        values = {
            c: {**v, "temperature": {"value": 25 + (i % 2), "unit": "degC"}}
            for i, (c, v) in enumerate(condition_values(XS).items())
        }
        base = intake(table)
        base["params"]["condition_values"] = values
        refused = run_campaign([base, fit_step()], tmp_path, "refused")
        assert refused.steps["fit"].status == "failed"
        assert "pool_over" in refused.steps["fit"].error
        # Naming the quantity says the variation is accepted, and the fit runs.
        accepted = run_ok([base, fit_step(pool_over=["temperature"])], tmp_path, "ok")
        assert coefficient(accepted, "Vmax")["value"] == pytest.approx(VMAX, abs=1e-2)

    def test_replicates_are_fitted_through_their_independent_unit_means(self, tmp_path):
        rows = [
            f"m{i}-{r},enzyme,c{i},initial-rate,{curve(x)},uM/min,u{r},t0,sample,measured\n"
            for i, x in enumerate(XS)
            for r in range(4)
        ]
        table = write_table(tmp_path, rows)
        outcome = run_ok(
            [
                intake(table, independent_unit="preparation"),
                fit_step(observation_basis="independent-means"),
            ],
            tmp_path,
        )
        assert coefficient(outcome, "Km")["value"] == pytest.approx(KM, abs=1e-4)
        assert coefficient(outcome, "Vmax")["value"] == pytest.approx(VMAX, abs=1e-2)


class TestClusterBootstrap:
    def replicate_rows(self, noise: float) -> list[str]:
        return [
            f"m{i}-{r},enzyme,c{i},initial-rate,"
            f"{curve(x, 1 + noise * math.sin(5 * i + r))},uM/min,u{r},t0,sample,measured\n"
            for i, x in enumerate(XS)
            for r in range(4)
        ]

    def steps(self, table: Path, **params) -> list[dict]:
        return [
            intake(table, independent_unit="preparation"),
            fit_step(
                observation_basis="independent-means",
                uncertainty="cluster_bootstrap",
                **params,
            ),
        ]

    def test_units_that_agree_exactly_give_a_collapsed_interval_on_the_truth(
        self, tmp_path
    ):
        table = write_table(tmp_path, self.replicate_rows(0.0))
        outcome = run_ok(self.steps(table, seed=7, resamples=100), tmp_path)
        vmax = coefficient(outcome, "Vmax")
        low, high = vmax["interval_95"]
        # Identical units leave nothing to resample: the interval is the value.
        assert high - low < 1e-3
        assert low == pytest.approx(VMAX, abs=1e-2)

    def test_the_interval_brackets_the_estimate_and_follows_the_seed(self, tmp_path):
        table = write_table(tmp_path, self.replicate_rows(0.03))
        first = run_ok(self.steps(table, seed=7, resamples=200), tmp_path, "a")
        again = run_ok(self.steps(table, seed=7, resamples=200), tmp_path, "b")
        other = run_ok(self.steps(table, seed=8, resamples=200), tmp_path, "c")
        low, high = coefficient(first, "Vmax")["interval_95"]
        assert low < coefficient(first, "Vmax")["value"] < high
        assert coefficient(again, "Vmax")["interval_95"] == [low, high]
        assert coefficient(other, "Vmax")["interval_95"] != [low, high]

    @pytest.mark.parametrize("resamples", [50, 5000])
    def test_a_resample_count_outside_the_supported_range_is_refused(
        self, resamples, tmp_path
    ):
        table = write_table(tmp_path, self.replicate_rows(0.03))
        outcome = run_campaign(self.steps(table, seed=7, resamples=resamples), tmp_path)
        assert outcome.steps["fit"].status == "failed"
        assert "from 100 to 2,000" in outcome.steps["fit"].error

    def test_a_bootstrap_without_a_seed_is_refused(self, tmp_path):
        table = write_table(tmp_path, self.replicate_rows(0.03))
        outcome = run_campaign(self.steps(table), tmp_path)
        assert outcome.steps["fit"].status == "failed"
        assert "seed" in outcome.steps["fit"].error


class TestInspectOptions:
    OWN = "id,sample,condition,assay,reading,units\n"

    def own_table(self, tmp_path: Path) -> Path:
        path = tmp_path / "own.csv"
        path.write_text(
            self.OWN
            + "".join(
                f"r{i},enzyme,c{i},initial-rate,{curve(x)},uM/min\n"
                for i, x in enumerate(XS)
            )
        )
        return path

    COLUMNS = {
        "measurement_id": "id",
        "sample_id": "sample",
        "condition_id": "condition",
        "metric": "assay",
        "value": "reading",
        "unit": "units",
    }
    DEFAULTS = {"independent_id": "", "technical_id": "", "role": "sample"}

    def rows_of(self, outcome) -> list[dict]:
        return json.loads(Path(outcome.outputs("intake")["record"]).read_text())["rows"]

    def test_your_own_column_names_read_the_same_as_the_canonical_ones(self, tmp_path):
        canonical = run_ok([intake(write_table(tmp_path, plain_rows()))], tmp_path, "c")
        own = run_ok(
            [
                intake(
                    self.own_table(tmp_path),
                    columns=self.COLUMNS,
                    defaults=self.DEFAULTS,
                )
            ],
            tmp_path,
            "o",
        )
        assert [r["value"] for r in self.rows_of(own)] == [
            r["value"] for r in self.rows_of(canonical)
        ]
        assert own.outputs("intake")["measured"] == len(XS)

    def test_a_table_read_through_columns_can_be_fitted(self, tmp_path):
        steps = [
            intake(
                self.own_table(tmp_path), columns=self.COLUMNS, defaults=self.DEFAULTS
            ),
            fit_step(),
        ]
        outcome = run_ok(steps, tmp_path)
        assert coefficient(outcome, "Vmax")["value"] == pytest.approx(VMAX, abs=1e-2)

    def test_a_missing_source_column_is_named(self, tmp_path):
        outcome = run_campaign(
            [
                intake(
                    self.own_table(tmp_path),
                    columns={**self.COLUMNS, "value": "no-such-column"},
                    defaults=self.DEFAULTS,
                )
            ],
            tmp_path,
        )
        error = outcome.steps["intake"].error
        assert "no-such-column" in error
        # The table's real columns are listed so the right one can be chosen.
        assert "reading" in error

    def test_one_source_column_cannot_serve_two_fields(self, tmp_path):
        outcome = run_campaign(
            [
                intake(
                    self.own_table(tmp_path),
                    columns={**self.COLUMNS, "unit": "reading"},
                    defaults=self.DEFAULTS,
                )
            ],
            tmp_path,
        )
        assert "may occur only once in columns" in outcome.steps["intake"].error

    def test_a_required_field_with_no_source_or_default_is_named(self, tmp_path):
        outcome = run_campaign(
            [
                intake(
                    self.own_table(tmp_path),
                    columns=self.COLUMNS,
                    defaults={"role": "sample"},
                )
            ],
            tmp_path,
        )
        error = outcome.steps["intake"].error
        assert "independent_id" in error and "technical_id" in error
        assert "role" not in error.split("required table fields")[1]

    def test_a_misnamed_header_points_to_the_columns_option(self, tmp_path):
        path = tmp_path / "misnamed.csv"
        path.write_text(HEAD.replace("sample_id", "sample") + "".join(plain_rows()).replace("enzyme", "enzyme"))
        outcome = run_campaign([intake(path)], tmp_path)
        error = outcome.steps["intake"].error
        assert "sample" in error and "columns" in error

    def test_a_field_cannot_come_from_both_a_column_and_a_default(self, tmp_path):
        outcome = run_campaign(
            [
                intake(
                    self.own_table(tmp_path),
                    columns=self.COLUMNS,
                    defaults={**self.DEFAULTS, "unit": "uM/min"},
                )
            ],
            tmp_path,
        )
        assert "both a source column and a default" in outcome.steps["intake"].error

    def test_a_json_table_reads_like_the_csv_one(self, tmp_path):
        records = [
            {
                "measurement_id": f"m{i}", "sample_id": "enzyme", "condition_id": f"c{i}",
                "metric": "initial-rate", "value": curve(x), "unit": "uM/min",
                "independent_id": None, "technical_id": None, "role": "sample",
            }
            for i, x in enumerate(XS)
        ]
        path = tmp_path / "rates.json"
        path.write_text(json.dumps(records))
        as_json = run_ok([intake(path, table_format="json")], tmp_path, "j")
        as_csv = run_ok([intake(write_table(tmp_path, plain_rows()))], tmp_path, "c")
        assert [r["value"] for r in self.rows_of(as_json)] == [
            r["value"] for r in self.rows_of(as_csv)
        ]

    def test_tab_separated_input_is_not_a_listed_format(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_campaign([intake(table, table_format="tsv")], tmp_path)
        error = outcome.steps["intake"].error
        assert "'csv'" in error and "'json'" in error

    def test_a_requested_unit_conversion_is_applied_and_the_original_kept(
        self, tmp_path
    ):
        rows = [
            f"t{i},enzyme,c{i},melting,{40 + i},degC,,,sample,measured\n"
            for i in range(3)
        ]
        table = write_table(tmp_path, rows)
        step = {
            "id": "intake",
            "capability": "assay.inspect",
            "params": {
                "table": str(table),
                "conditions": {f"c{i}": f"Run {i}" for i in range(3)},
                "units": {"melting": "K"},
            },
        }
        outcome = run_ok([step], tmp_path)
        rows_out = self.rows_of(outcome)
        assert [r["unit"] for r in rows_out] == ["K"] * 3
        assert [r["value"] for r in rows_out] == pytest.approx([313.15, 314.15, 315.15])
        assert [r["original"]["value"] for r in rows_out] == [40.0, 41.0, 42.0]

    def test_sample_descriptions_are_kept_with_the_record(self, tmp_path):
        table = write_table(tmp_path, plain_rows())
        outcome = run_ok(
            [intake(table, sample_descriptions={"enzyme": "A test enzyme"})], tmp_path
        )
        record = Path(outcome.outputs("intake")["record"]).read_text()
        assert "A test enzyme" in record
