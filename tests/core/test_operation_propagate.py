"""operation.propagate: a fit's uncertainty carried to the predicted concentrations.

Three kinds of answer are known without running the capability:

* for a function linear in the coefficients the spread is exact: the coefficients' covariance
  gives the standard deviation sqrt(g' C g), which is also the mean-response standard error
  `assay.fit` reports for a linear fit, and normal draws reproduce its interval;
* for a Michaelis-Menten operation, each draw's prediction is the integrated solution
  Km ln(S0/S) + (S0 - S) = Vmax t, solved here independently, so the reported quantiles must
  equal the quantiles of those solutions over the same draws;
* the spread falls when the data are less noisy, a seed fixes the draws, and the counts of
  samples and discarded draws are as stated.
"""

from __future__ import annotations

import json
import math
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy")),
    reason="the quantitative extra is required",
)

from tests.core.test_operation_define import (  # noqa: E402
    BASIS,
    HEAD,
    KM_MM,
    VMAX_UM_MIN,
    XS,
    template,
    write,
)

T_LEVELS = [0.05, 0.5, 0.95]


def noisy_table(tmp_path: Path, noise: float, name: str = "rates.csv") -> Path:
    rows = "".join(
        f"m{i},enzyme,c{i},initial-rate,"
        f"{VMAX_UM_MIN * x / (KM_MM + x) * (1 + noise * math.sin(7 * i + 1))},"
        "uM/min,,,sample,measured\n"
        for i, x in enumerate(XS)
    )
    path = tmp_path / name
    path.write_text(HEAD + rows)
    return path


def fit_steps(
    tmp_path: Path, noise: float, name: str = "n", **fit_params
) -> list[dict]:
    return [
        {
            "id": f"{name}-intake",
            "capability": "assay.inspect",
            "params": {
                "table": str(noisy_table(tmp_path, noise, f"{name}.csv")),
                "conditions": {f"c{i}": f"Substrate {x} mM" for i, x in enumerate(XS)},
                "condition_values": {
                    f"c{i}": {"substrate": {"value": x, "unit": "mM"}}
                    for i, x in enumerate(XS)
                },
            },
        },
        {
            "id": f"{name}-fit",
            "capability": "assay.fit",
            "bind": {"measurements": f"{name}-intake.record"},
            "params": {
                "sample_id": "enzyme",
                "metric": "initial-rate",
                "x_quantity": "substrate",
                "x_unit": "mM",
                "y_unit": "uM/min",
                "rate_basis": BASIS,
                **fit_params,
            },
        },
    ]


def propagate_step(tmp_path: Path, name: str = "n", **params) -> dict:
    return {
        "id": "spread",
        "capability": "operation.propagate",
        "bind": {"kinetics.fit": f"{name}-fit.record"},
        "params": {
            "template": write(tmp_path, "template.json", template()),
            "kinetics": {
                "channel": "hydrolysis",
                "species": "S",
                "assay_enzyme": {"value": 20, "unit": "mg/L"},
            },
            **params,
        },
    }


def run_all(tmp_path, steps, name="run"):
    from alambiq.api import run

    return run({"id": "propagate", "steps": steps}, run_dir=tmp_path / name)


def propagated(tmp_path, noise=0.03, name="run", fit_params=None, **params):
    steps = fit_steps(tmp_path, noise, **(fit_params or {})) + [
        propagate_step(tmp_path, **params)
    ]
    outcome = run_all(tmp_path, steps, name)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def mm_solution(s0: float, km: float, vmax: float, t: float) -> float:
    from scipy.optimize import brentq

    return brentq(
        lambda s: km * math.log(s0 / s) + (s0 - s) - vmax * t, 1e-15, s0, xtol=1e-18
    )


class TestLinearCase:
    """A line fitted to 60 points: the propagated interval is the analytic one."""

    @pytest.fixture(scope="class")
    def line(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("line")
        xs = [i / 6 for i in range(60)]
        ys = [2.0 + 1.5 * x + 0.4 * math.sin(5.3 * i) for i, x in enumerate(xs)]
        rows = "".join(
            f"m{i},s,c{i},resp,{y!r},U,,,sample,measured\n" for i, y in enumerate(ys)
        )
        (tmp_path / "t.csv").write_text(HEAD + rows)
        steps = [
            {
                "id": "intake",
                "capability": "assay.inspect",
                "params": {
                    "table": str(tmp_path / "t.csv"),
                    "conditions": {f"c{i}": f"x {x}" for i, x in enumerate(xs)},
                    "condition_values": {
                        f"c{i}": {"dose": {"value": x, "unit": "mM"}}
                        for i, x in enumerate(xs)
                    },
                },
            },
            {
                "id": "fit",
                "capability": "assay.fit",
                "bind": {"measurements": "intake.record"},
                "params": {
                    "sample_id": "s",
                    "metric": "resp",
                    "x_quantity": "dose",
                    "x_unit": "mM",
                    "y_unit": "U",
                    "model": "linear",
                    "prediction_x": [4.0],
                },
            },
        ]
        outcome = run_all(tmp_path, steps)
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        return json.loads(Path(outcome.outputs("fit")["record"]).read_text())

    def test_the_analytic_standard_deviation_is_the_fits_own_mean_standard_error(
        self, line
    ):
        from alambiq.capabilities.operate.science.propagation import (
            covariance_of,
            linear_interval,
        )

        names, covariance = covariance_of(line)
        assert names == ["intercept", "slope"]
        values = [p["value"] for p in line["parameters"]]
        centre, low, high = linear_interval(values, covariance, [1.0, 4.0], 0.95)
        prediction = line["predictions"][0]
        assert centre == pytest.approx(prediction["y"], rel=1e-9)
        sd = math.sqrt(np.array([1, 4]) @ covariance @ np.array([1, 4]))
        assert sd == pytest.approx(prediction["mean_standard_error"], rel=1e-9)
        # The fit uses a Student-t quantile; the normal one differs by exactly their ratio.
        from scipy.stats import norm, t

        df = line["statistics"]["residual_degrees_of_freedom"]
        half = (
            prediction["mean_interval_95"][1] - prediction["mean_interval_95"][0]
        ) / 2
        assert half / (high - centre) == pytest.approx(
            t.ppf(0.975, df) / norm.ppf(0.975), rel=1e-9
        )

    def test_normal_draws_reproduce_the_analytic_interval(self, line):
        from alambiq.capabilities.operate.science.propagation import (
            covariance_of,
            draw_normal,
            linear_interval,
            propagate,
        )

        names, covariance = covariance_of(line)
        values = [p["value"] for p in line["parameters"]]
        _, draws, rejected = draw_normal(line, 40_000, seed=11)
        assert rejected == 0
        spread = propagate(draws, lambda q: q[0] + 4.0 * q[1], [0.025, 0.5, 0.975])
        centre, low, high = linear_interval(values, covariance, [1.0, 4.0], 0.95)
        width = high - low
        assert spread["mean"] == pytest.approx(centre, abs=0.01 * width)
        assert spread["quantiles"]["0.025"] == pytest.approx(low, abs=0.02 * width)
        assert spread["quantiles"]["0.975"] == pytest.approx(high, abs=0.02 * width)
        assert spread["quantiles"]["0.5"] == pytest.approx(centre, abs=0.01 * width)

    def test_the_draws_have_the_fits_covariance(self, line):
        from alambiq.capabilities.operate.science.propagation import (
            covariance_of,
            draw_normal,
        )

        _, covariance = covariance_of(line)
        _, draws, _ = draw_normal(line, 60_000, seed=3)
        sample = np.cov(draws, rowvar=False)
        assert np.allclose(sample, covariance, rtol=0.03, atol=0)


class TestOperationalPrediction:
    def test_each_quantile_is_that_of_the_independently_integrated_solutions(
        self, tmp_path
    ):
        outcome = propagated(tmp_path, samples=300, seed=5)
        out = outcome.outputs("spread")
        stored = json.loads(Path(out["record"]).read_text())
        draws = np.array(stored["draws"])
        names = stored["coefficient_names"]
        km = draws[:, names.index("Km")] * 1e-3  # mM to M
        vmax = draws[:, names.index("Vmax")] * 1e-6 / 60.0  # uM/min to M/s
        # 20 mg/L of assay enzyme scaled to 10 mg in 0.5 L (20 mg/L) leaves the rate unchanged.
        for row in out["spreads"]:
            if row["species"] != "S":
                continue
            t = row["time_s"]
            expected = (
                np.array(
                    [mm_solution(0.01, k, v, t) for k, v in zip(km, vmax, strict=True)]
                )
                if t
                else np.full(len(km), 0.01)
            )
            wanted = np.quantile(expected, T_LEVELS)
            got = [row["quantiles_M"][f"{q:g}"] for q in T_LEVELS]
            assert got == pytest.approx(list(wanted), rel=1e-4, abs=1e-9)
            assert row["mean_M"] == pytest.approx(expected.mean(), rel=1e-4, abs=1e-9)

    def test_the_nominal_prediction_is_the_integrated_solution_at_the_fitted_values(
        self, tmp_path
    ):
        outcome = propagated(tmp_path, samples=40)
        fitted = {p["name"]: p["value"] for p in outcome.outputs("n-fit")["parameters"]}
        for row in outcome.outputs("spread")["spreads"]:
            if row["species"] == "S" and row["time_s"]:
                expected = mm_solution(
                    0.01,
                    fitted["Km"] * 1e-3,
                    fitted["Vmax"] * 1e-6 / 60.0,
                    row["time_s"],
                )
                assert row["nominal_M"] == pytest.approx(expected, rel=1e-4)

    def test_substrate_and_product_spreads_mirror_each_other(self, tmp_path):
        out = propagated(tmp_path, samples=100).outputs("spread")
        by = {(r["species"], r["time_s"]): r for r in out["spreads"]}
        for (species, t), row in by.items():
            if species == "S":
                other = by[("P", t)]
                assert row["mean_M"] + other["mean_M"] == pytest.approx(0.01, rel=1e-9)
                assert row["standard_deviation_M"] == pytest.approx(
                    other["standard_deviation_M"], rel=1e-6, abs=1e-12
                )

    def test_a_seed_fixes_the_draws_and_another_changes_them(self, tmp_path):
        a = propagated(tmp_path, samples=60, seed=2, name="a").outputs("spread")
        b = propagated(tmp_path, samples=60, seed=2, name="b").outputs("spread")
        c = propagated(tmp_path, samples=60, seed=3, name="c").outputs("spread")
        draws = [json.loads(Path(o["record"]).read_text())["draws"] for o in (a, b, c)]
        assert draws[0] == draws[1] and draws[0] != draws[2]

    def test_less_noisy_data_give_a_narrower_spread(self, tmp_path):
        noisy = propagated(tmp_path, noise=0.06, samples=200, name="a").outputs(
            "spread"
        )
        quiet = propagated(tmp_path, noise=0.006, samples=200, name="b").outputs(
            "spread"
        )

        def width(out):
            row = next(
                r
                for r in out["spreads"]
                if r["species"] == "S" and r["time_s"] == 1800.0
            )
            return row["quantiles_M"]["0.95"] - row["quantiles_M"]["0.05"]

        assert width(quiet) < width(noisy) / 3

    def test_the_counts_and_the_coefficients_are_reported(self, tmp_path):
        out = propagated(tmp_path, samples=80).outputs("spread")
        assert out["samples"] == 80 and out["discarded_draws"] == 0
        assert {c["name"] for c in out["coefficients"]} == {"Km", "Vmax"}
        for c in out["coefficients"]:
            assert c["draw_standard_deviation"] > 0
            assert c["draw_mean"] == pytest.approx(c["value"], rel=0.2)

    def test_the_other_solver_gives_the_same_spread(self, tmp_path):
        implicit = propagated(tmp_path, samples=40, seed=6, name="a").outputs("spread")
        explicit = propagated(
            tmp_path, samples=40, seed=6, name="b", solver="DOP853"
        ).outputs("spread")
        for one, other in zip(implicit["spreads"], explicit["spreads"], strict=True):
            for level in ("0.05", "0.5", "0.95"):
                assert one["quantiles_M"][level] == pytest.approx(
                    other["quantiles_M"][level], rel=1e-5, abs=1e-10
                )

    def test_the_species_can_be_chosen(self, tmp_path):
        out = propagated(tmp_path, samples=30, species=["P"]).outputs("spread")
        assert {r["species"] for r in out["spreads"]} == {"P"}

    def test_the_report_draws_a_band_around_the_nominal_prediction(self, tmp_path):
        outcome = propagated(tmp_path, samples=30)
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "Predicted S with the uncertainty of the fit" in page and "Nominal prediction" in page


class TestBootstrap:
    def bootstrap_steps(self, tmp_path):
        units = 6
        rows = []
        for u in range(units):
            for i, x in enumerate(XS):
                y = (
                    VMAX_UM_MIN
                    * x
                    / (KM_MM + x)
                    * (1 + 0.04 * math.sin(3.7 * u + 1.9 * i))
                )
                rows.append(
                    f"m{u}-{i},enzyme,c{i},initial-rate,{y!r},uM/min,u{u},r1,sample,measured\n"
                )
        path = tmp_path / "units.csv"
        path.write_text(HEAD + "".join(rows))
        return [
            {
                "id": "n-intake",
                "capability": "assay.inspect",
                "params": {
                    "table": str(path),
                    "conditions": {
                        f"c{i}": f"Substrate {x} mM" for i, x in enumerate(XS)
                    },
                    "condition_values": {
                        f"c{i}": {"substrate": {"value": x, "unit": "mM"}}
                        for i, x in enumerate(XS)
                    },
                    "independent_unit": "preparation",
                },
            },
            {
                "id": "n-fit",
                "capability": "assay.fit",
                "bind": {"measurements": "n-intake.record"},
                "params": {
                    "sample_id": "enzyme",
                    "metric": "initial-rate",
                    "x_quantity": "substrate",
                    "x_unit": "mM",
                    "y_unit": "uM/min",
                    "rate_basis": BASIS,
                    "observation_basis": "independent-means",
                    "uncertainty": "cluster_bootstrap",
                    "seed": 4,
                    "resamples": 200,
                },
            },
        ]

    def test_a_cluster_bootstrap_fit_is_propagated_from_its_own_draws(self, tmp_path):
        steps = self.bootstrap_steps(tmp_path) + [
            propagate_step(tmp_path, distribution="bootstrap", samples=60, seed=8)
        ]
        outcome = run_all(tmp_path, steps)
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        out = outcome.outputs("spread")
        fit = json.loads(Path(outcome.outputs("n-fit")["record"]).read_text())
        pool = {
            tuple(round(v, 12) for v in row)
            for row in fit["uncertainty"]["parameter_draws"]
        }
        stored = json.loads(Path(out["record"]).read_text())
        assert out["distribution"] == "bootstrap" and out["discarded_draws"] == 0
        assert all(tuple(round(v, 12) for v in row) in pool for row in stored["draws"])

    def test_a_fit_without_bootstrap_draws_cannot_be_propagated_that_way(
        self, tmp_path
    ):
        steps = fit_steps(tmp_path, 0.03) + [
            propagate_step(tmp_path, distribution="bootstrap")
        ]
        outcome = run_all(tmp_path, steps)
        assert "contains no bootstrap draws" in outcome.steps["spread"].error


class TestRefusals:
    def fails(self, tmp_path, message, noise=0.03, fit_params=None, **params):
        steps = fit_steps(tmp_path, noise, **(fit_params or {})) + [
            propagate_step(tmp_path, **params)
        ]
        outcome = run_all(tmp_path, steps)
        assert not outcome.ok
        assert message in outcome.steps["spread"].error

    def test_a_fit_at_a_bound_has_no_covariance_to_propagate(self, tmp_path):
        self.fails(
            tmp_path,
            "no parameter covariance",
            fit_params={"parameter_bounds": {"Vmax": [0, 100]}},
        )

    @pytest.mark.parametrize("samples", [5, 5000])
    def test_the_number_of_samples_is_bounded(self, tmp_path, samples):
        self.fails(tmp_path, "samples must be between", samples=samples)

    def test_quantiles_must_lie_between_zero_and_one(self, tmp_path):
        self.fails(tmp_path, "strictly between 0 and 1", quantiles=[0.0, 0.5])

    def test_an_unknown_species_is_named(self, tmp_path):
        self.fails(tmp_path, "species that the template does not declare: Q", species=["Q"])

    def test_only_a_michaelis_menten_fit_can_be_propagated(self, tmp_path):
        self.fails(
            tmp_path,
            "Michaelis-Menten",
            fit_params={"model": "linear", "rate_basis": None},
        )
