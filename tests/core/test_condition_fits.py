"""assay.fit on how a rate depends on temperature, time at temperature, and pH.

Each series is generated from known coefficients with no noise, so the fit must return them:

* Arrhenius: a rate constant of 0.5 per second at 300 K with an activation energy of
  60 kJ/mol, over 280 to 320 K. The fit returns both, the activation energy in kJ/mol, and
  the pre-exponential factor k_ref * exp(Ea / (R T_ref)) computed here.
* First-order inactivation: activity 100 decaying with a half-life of exactly 30 minutes,
  so k is ln 2 / 30 per minute and the derived half-life is 30.
* A bell-shaped pH profile with pKa1 = 5.5 and pKa2 = 8.5 and a maximum of 10, which has its
  optimum at pH 7 (the midpoint of the pKas) with activity 10 / (1 + 2 * 10^-1.5) there, at a
  pH that is not among the measured ones.

Noisy series show that the intervals cover the truth, and each way a request can be wrong
for these models is refused by name.
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

R = 8.314462618
HEAD = (
    "measurement_id,sample_id,condition_id,metric,value,unit,"
    "independent_id,technical_id,role,status\n"
)


def series(tmp_path: Path, xs, ys, quantity: str, unit: str, y_unit: str) -> dict:
    """Campaign steps that read a table of (x, y) and are ready to fit."""
    table = tmp_path / "series.csv"
    table.write_text(
        HEAD
        + "".join(
            f"m{i},enzyme,c{i},rate,{y!r},{y_unit},,,sample,measured\n"
            for i, y in enumerate(ys)
        )
    )
    return {
        "id": "intake",
        "capability": "assay.inspect",
        "params": {
            "table": str(table),
            "conditions": {f"c{i}": f"{quantity} {x}" for i, x in enumerate(xs)},
            "condition_values": {
                f"c{i}": {quantity: {"value": x, "unit": unit}}
                for i, x in enumerate(xs)
            },
        },
    }


def fit_step(quantity: str, unit: str, y_unit: str, **params) -> dict:
    return {
        "id": "fit",
        "capability": "assay.fit",
        "bind": {"measurements": "intake.record"},
        "params": {
            "sample_id": "enzyme",
            "metric": "rate",
            "x_quantity": quantity,
            "x_unit": unit,
            "y_unit": y_unit,
            **params,
        },
    }


def run_fit(tmp_path, xs, ys, quantity, unit, y_unit, name="run", **params):
    from alambiq.api import run

    steps = [
        series(tmp_path, xs, ys, quantity, unit, y_unit),
        fit_step(quantity, unit, y_unit, **params),
    ]
    return run({"id": "conditions", "steps": steps}, run_dir=tmp_path / name)


def fitted(outcome) -> dict:
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    out = outcome.outputs("fit")
    return {
        "parameters": {p["name"]: p for p in out["parameters"]},
        "derived": {d["name"]: d for d in out["derived"]},
        "outcome": outcome,
    }


def arrhenius(t, k_ref=0.5, ea=60_000.0, t_ref=300.0):
    return k_ref * math.exp(-ea / R * (1.0 / t - 1.0 / t_ref))


TEMPERATURES = [280.0, 285.0, 290.0, 295.0, 300.0, 305.0, 310.0, 315.0, 320.0]


class TestArrhenius:
    def test_the_coefficients_that_generated_the_rates_are_returned(self, tmp_path):
        found = fitted(
            run_fit(
                tmp_path,
                TEMPERATURES,
                [arrhenius(t) for t in TEMPERATURES],
                "temperature",
                "K",
                "1/s",
                model="arrhenius",
                reference_x=300.0,
            )
        )
        assert found["parameters"]["k_ref"]["value"] == pytest.approx(0.5, rel=1e-6)
        assert found["parameters"]["Ea"]["value"] == pytest.approx(60_000.0, rel=1e-6)
        assert found["parameters"]["Ea"]["unit"] == "J/mol"
        assert found["derived"]["activation_energy_kJ_mol"]["value"] == pytest.approx(
            60.0, rel=1e-6
        )
        assert found["derived"]["pre_exponential_factor"]["value"] == pytest.approx(
            0.5 * math.exp(60_000.0 / (R * 300.0)), rel=1e-6
        )
        assert found["derived"]["reference_temperature"]["value"] == 300.0

    def test_temperatures_in_celsius_give_the_same_coefficients(self, tmp_path):
        celsius = [t - 273.15 for t in TEMPERATURES]
        found = fitted(
            run_fit(
                tmp_path,
                celsius,
                [arrhenius(t) for t in TEMPERATURES],
                "temperature",
                "degC",
                "1/s",
                model="arrhenius",
                reference_x=300.0 - 273.15,
            )
        )
        assert found["parameters"]["k_ref"]["value"] == pytest.approx(0.5, rel=1e-6)
        assert found["parameters"]["Ea"]["value"] == pytest.approx(60_000.0, rel=1e-6)

    def test_the_default_reference_is_the_mean_temperature_and_the_curve_is_unchanged(
        self, tmp_path
    ):
        ys = [arrhenius(t) for t in TEMPERATURES]
        default = fitted(
            run_fit(
                tmp_path, TEMPERATURES, ys, "temperature", "K", "1/s", model="arrhenius"
            )
        )
        mean = sum(TEMPERATURES) / len(TEMPERATURES)
        assert default["derived"]["reference_temperature"]["value"] == pytest.approx(
            mean
        )
        # The same curve, restated at the mean: k at the mean temperature is the rate there.
        assert default["parameters"]["k_ref"]["value"] == pytest.approx(
            arrhenius(mean), rel=1e-6
        )
        assert default["parameters"]["Ea"]["value"] == pytest.approx(60_000.0, rel=1e-6)

    def test_noisy_rates_give_intervals_that_cover_the_truth(self, tmp_path):
        noisy = [
            arrhenius(t) * (1 + 0.04 * math.sin(3.1 * i + 0.4))
            for i, t in enumerate(TEMPERATURES)
        ]
        found = fitted(
            run_fit(
                tmp_path,
                TEMPERATURES,
                noisy,
                "temperature",
                "K",
                "1/s",
                model="arrhenius",
                reference_x=300.0,
            )
        )
        for name, truth in (("k_ref", 0.5), ("Ea", 60_000.0)):
            low, high = found["parameters"][name]["interval_95"]
            assert low < truth < high
            assert found["parameters"][name]["standard_error"] > 0
        low, high = found["derived"]["activation_energy_kJ_mol"]["interval_95"]
        assert low < 60.0 < high

    def test_a_fixed_activation_energy_leaves_the_rate_constant_to_fit(self, tmp_path):
        found = fitted(
            run_fit(
                tmp_path,
                TEMPERATURES,
                [arrhenius(t) for t in TEMPERATURES],
                "temperature",
                "K",
                "1/s",
                model="arrhenius",
                reference_x=300.0,
                fixed_parameters={"Ea": 60_000.0},
            )
        )
        assert found["parameters"]["Ea"]["fixed"] is True
        assert found["parameters"]["k_ref"]["value"] == pytest.approx(0.5, rel=1e-6)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"unit": "mM"}, "K or degC"),
            ({"ys": [0.0] + [1.0] * 8}, "must be positive"),
            ({"xs": [-300.0] + TEMPERATURES[1:]}, "above absolute zero"),
            ({"reference_x": -10.0}, "above absolute zero"),
        ],
    )
    def test_requests_that_cannot_be_an_arrhenius_fit_are_refused(
        self, tmp_path, changes, message
    ):
        xs = changes.pop("xs", TEMPERATURES)
        ys = changes.pop("ys", [arrhenius(t) for t in TEMPERATURES])
        unit = changes.pop("unit", "K")
        outcome = run_fit(
            tmp_path,
            xs,
            ys,
            "temperature",
            unit,
            "1/s",
            model="arrhenius",
            **({"reference_x": 300.0} | changes),
        )
        assert not outcome.ok
        error = " ".join(s.error or "" for s in outcome.steps.values())
        assert message in error

    def test_a_reference_is_refused_for_other_models(self, tmp_path):
        outcome = run_fit(
            tmp_path,
            [1, 2, 3, 4],
            [1, 2, 3, 4.5],
            "dose",
            "mM",
            "1/s",
            model="linear",
            reference_x=2.0,
        )
        assert "reference_x is accepted only with the model arrhenius" in outcome.steps["fit"].error


class TestThermalInactivation:
    K = math.log(2) / 30.0

    def times(self):
        return [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0]

    def test_a_thirty_minute_half_life_is_returned_as_one(self, tmp_path):
        found = fitted(
            run_fit(
                tmp_path,
                self.times(),
                [100.0 * math.exp(-self.K * t) for t in self.times()],
                "time",
                "min",
                "%",
                model="first_order_decay",
            )
        )
        assert found["parameters"]["A0"]["value"] == pytest.approx(100.0, rel=1e-6)
        assert found["parameters"]["k"]["value"] == pytest.approx(self.K, rel=1e-6)
        assert found["parameters"]["k"]["unit"] == "1/min"
        half = found["derived"]["half_life"]
        assert half["value"] == pytest.approx(30.0, rel=1e-6)
        assert half["unit"] == "min"

    def test_the_half_life_interval_is_the_delta_method_of_the_rate_constants(
        self, tmp_path
    ):
        ys = [
            100.0 * math.exp(-self.K * t) * (1 + 0.03 * math.sin(2.3 * i))
            for i, t in enumerate(self.times())
        ]
        found = fitted(
            run_fit(
                tmp_path,
                self.times(),
                ys,
                "time",
                "min",
                "%",
                model="first_order_decay",
            )
        )
        k = found["parameters"]["k"]
        half = found["derived"]["half_life"]
        assert half["standard_error"] == pytest.approx(
            math.log(2) / k["value"] ** 2 * k["standard_error"], rel=1e-4
        )
        assert half["interval_95"][0] < 30.0 < half["interval_95"][1]

    def test_the_same_decay_in_seconds_has_a_half_life_in_seconds(self, tmp_path):
        times = [t * 60.0 for t in self.times()]
        found = fitted(
            run_fit(
                tmp_path,
                times,
                [100.0 * math.exp(-self.K * t / 60.0) for t in times],
                "time",
                "s",
                "%",
                model="first_order_decay",
            )
        )
        assert found["derived"]["half_life"]["value"] == pytest.approx(1800.0, rel=1e-6)

    def test_activity_that_does_not_fall_is_flagged_not_fitted_confidently(
        self, tmp_path
    ):
        found = fitted(
            run_fit(
                tmp_path,
                self.times(),
                [100.0] * 8,
                "time",
                "min",
                "%",
                model="first_order_decay",
            )
        )
        assert found["parameters"]["k"]["value"] == pytest.approx(0.0, abs=1e-6)
        assert any(
            "bound" in n or "flat" in n for n in found["outcome"].steps["fit"].notes
        )

    def test_the_inactivation_rate_at_several_temperatures_follows_arrhenius(
        self, tmp_path
    ):
        """Half-lives measured at three temperatures give back the activation energy."""
        ea = 150_000.0
        rates = {}
        for temperature in (305.0, 310.0, 315.0, 320.0):
            k = 0.01 * math.exp(-ea / R * (1.0 / temperature - 1.0 / 310.0))
            times = [0.0, 10.0, 20.0, 40.0, 80.0]
            found = fitted(
                run_fit(
                    tmp_path,
                    times,
                    [100.0 * math.exp(-k * t) for t in times],
                    "time",
                    "min",
                    "%",
                    model="first_order_decay",
                    name=f"t{int(temperature)}",
                )
            )
            rates[temperature] = found["parameters"]["k"]["value"]
        found = fitted(
            run_fit(
                tmp_path,
                list(rates),
                list(rates.values()),
                "temperature",
                "K",
                "1/min",
                model="arrhenius",
                reference_x=310.0,
                name="arrhenius",
            )
        )
        assert found["parameters"]["Ea"]["value"] == pytest.approx(ea, rel=1e-4)


class TestPhProfile:
    PH = [4.5, 5.0, 5.5, 6.0, 6.5, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

    @staticmethod
    def activity(ph, ymax=10.0, pka1=5.5, pka2=8.5):
        return ymax / (1 + 10 ** (pka1 - ph) + 10 ** (ph - pka2))

    def test_the_pkas_and_the_optimum_are_returned(self, tmp_path):
        found = fitted(
            run_fit(
                tmp_path,
                self.PH,
                [self.activity(p) for p in self.PH],
                "ph",
                "pH",
                "U/mg",
                model="ph_bell",
            )
        )
        assert found["parameters"]["ymax"]["value"] == pytest.approx(10.0, rel=1e-5)
        assert found["parameters"]["pKa1"]["value"] == pytest.approx(5.5, abs=1e-5)
        assert found["parameters"]["pKa2"]["value"] == pytest.approx(8.5, abs=1e-5)
        assert found["derived"]["optimum_pH"]["value"] == pytest.approx(7.0, abs=1e-5)
        assert 7.0 not in self.PH
        assert found["derived"]["activity_at_optimum"]["value"] == pytest.approx(
            10.0 / (1 + 2 * 10**-1.5), rel=1e-5
        )

    def test_the_optimum_is_where_the_fitted_curve_is_highest(self, tmp_path):
        found = fitted(
            run_fit(
                tmp_path,
                self.PH,
                [self.activity(p) for p in self.PH],
                "ph",
                "pH",
                "U/mg",
                model="ph_bell",
            )
        )
        p = found["parameters"]
        grid = [4 + i / 1000 for i in range(7001)]
        best = max(
            grid,
            key=lambda ph: self.activity(
                ph, p["ymax"]["value"], p["pKa1"]["value"], p["pKa2"]["value"]
            ),
        )
        assert best == pytest.approx(found["derived"]["optimum_pH"]["value"], abs=1e-3)

    def test_noisy_activities_give_intervals_that_cover_the_truth(self, tmp_path):
        ys = [
            self.activity(p) * (1 + 0.03 * math.sin(2.1 * i + 0.3))
            for i, p in enumerate(self.PH)
        ]
        found = fitted(
            run_fit(tmp_path, self.PH, ys, "ph", "pH", "U/mg", model="ph_bell")
        )
        for name, truth in (("pKa1", 5.5), ("pKa2", 8.5)):
            low, high = found["parameters"][name]["interval_95"]
            assert low < truth < high
        low, high = found["derived"]["optimum_pH"]["interval_95"]
        assert low < 7.0 < high

    def test_a_profile_with_too_few_points_to_fix_three_coefficients_is_refused(
        self, tmp_path
    ):
        outcome = run_fit(
            tmp_path,
            [6.0, 7.0, 8.0],
            [1.0, 2.0, 1.0],
            "ph",
            "pH",
            "U/mg",
            model="ph_bell",
        )
        assert not outcome.ok

    def test_all_three_coefficients_cannot_be_held_fixed(self, tmp_path):
        outcome = run_fit(
            tmp_path,
            self.PH,
            [self.activity(p) for p in self.PH],
            "ph",
            "pH",
            "U/mg",
            model="ph_bell",
            fixed_parameters={"ymax": 10.0, "pKa1": 5.5, "pKa2": 8.5},
        )
        assert "at least 1 coefficient must remain free" in outcome.steps["fit"].error

    def test_two_held_fixed_leave_the_third_to_be_fitted(self, tmp_path):
        found = fitted(
            run_fit(
                tmp_path,
                self.PH,
                [self.activity(p) for p in self.PH],
                "ph",
                "pH",
                "U/mg",
                model="ph_bell",
                fixed_parameters={"pKa1": 5.5, "pKa2": 8.5},
            )
        )
        assert found["parameters"]["ymax"]["value"] == pytest.approx(10.0, rel=1e-6)


class TestReportAndRecord:
    def test_the_record_carries_a_correlation_matrix_and_the_report_the_derived_table(
        self, tmp_path
    ):
        ph = TestPhProfile.PH
        ys = [
            TestPhProfile.activity(p) * (1 + 0.02 * math.sin(1.7 * i))
            for i, p in enumerate(ph)
        ]
        outcome = run_fit(tmp_path, ph, ys, "ph", "pH", "U/mg", model="ph_bell")
        record = json.loads(Path(outcome.outputs("fit")["record"]).read_text())
        matrix = record["parameter_dependence"]["correlation_matrix"]
        assert len(matrix) == 3 and all(
            row[i] == pytest.approx(1.0) for i, row in enumerate(matrix)
        )
        assert record["equation"].startswith("ymax /")
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "Derived quantity" in page and "optimum pH" in page
        assert "ph (pH)" in page
