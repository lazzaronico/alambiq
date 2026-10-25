"""operation.define: measured kinetics become the rate parameters of an operating definition.

The assay is a synthetic Michaelis-Menten series with Vmax = 180 uM/min and Km = 2.5 mM
(noise-free, so the fit returns them), measured with 20 mg/L of enzyme. The template is a
0.5 L reactor holding 10 mg of that enzyme, which is the catalyst's reference amount. The
answers are known without running the model:

* the capacity is the assay rate per mg of enzyme times the reference amount over the
  volume: 180e-6 / 60 / 20 * 10 / 0.5 = 3e-6 M/s, and the saturation scale is 2.5e-3 M;
* mg/mL and mg/L are the same concentration, a catalyst counted in activity units takes
  them per litre, a product-formation basis of two measured per substrate halves the
  substrate rate, a stoichiometric coefficient of two halves the extent rate, and an assay
  with half the enzyme active doubles the rate at full activity;
* a campaign from the measurement table through the fit and this capability to
  `operation.simulate` reproduces the integrated Michaelis-Menten solution
  Km ln(S0/S) + (S0 - S) = Vmax t, solved here independently;
* each way the request can disagree with the template or the fit is refused by name.
"""

from __future__ import annotations

import copy
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
KM_MM, VMAX_UM_MIN = 2.5, 180.0
HEAD = (
    "measurement_id,sample_id,condition_id,metric,value,unit,"
    "independent_id,technical_id,role,status\n"
)
CAPACITY = VMAX_UM_MIN * 1e-6 / 60 / 20 * 10 / 0.5
BASIS = {
    "kind": "product-formation",
    "measured_species": "P",
    "substrate": "S",
    "measured_per_substrate": 1,
}


def template(**changes) -> dict:
    value = {
        "id": "hydrolysis",
        "volume_L": 0.5,
        "conditions": "Assay conditions: buffer, 30 C",
        "species": [
            {"id": "S", "initial_M": 0.01, "pools": {"moiety": 1}},
            {"id": "P", "initial_M": 0, "pools": {"moiety": 1}},
        ],
        "catalysts": [
            {
                "id": "E",
                "amount": 10,
                "amount_unit": "mg",
                "active_fraction": 1,
                "reference_amount": 10,
                "deactivation_per_s": 0,
                "washout_fraction": 0,
            }
        ],
        "channels": [
            {
                "id": "hydrolysis",
                "reaction_id": "S-to-P",
                "stoichiometry": {"S": -1, "P": 1},
                "capacity_M_s": 0.0,
                "catalyst": "E",
                "factors": [{"species": "S", "kind": "saturation", "scale_M": 1.0}],
                "source": "placeholder",
            },
            {
                "id": "side",
                "reaction_id": "S-to-P-slow",
                "stoichiometry": {"S": -1, "P": 1},
                "capacity_M_s": 1e-9,
                "factors": [{"species": "S", "kind": "mass-action", "scale_M": 0.01}],
                "source": "kept as written",
            },
        ],
        "phases": [
            {
                "end_s": 3600.0,
                "dilution_per_s": 0.0,
                "feed_M": {},
                "active_channels": ["hydrolysis"],
            }
        ],
        "events": [],
        "times_s": [0.0, 600.0, 1800.0, 3600.0],
    }
    value.update(changes)
    return value


def fit_record(tmp_path: Path, **changes) -> dict:
    """A stored fit, as `assay.fit` writes one, with the synthetic Vmax and Km."""
    record = {
        "schema_version": "0.1",
        "model": "initial-rate-michaelis-menten-v1",
        "optimizer": {"success": True},
        "request": {"x_unit": "mM", "y_unit": "uM/min", "rate_basis": BASIS},
        "parameters": [
            {"name": "Km", "value": KM_MM, "unit": "mM"},
            {"name": "Vmax", "value": VMAX_UM_MIN, "unit": "uM/min"},
        ],
        "domain": {"minimum_x": 0.05, "maximum_x": 30.0},
    }
    record.update(changes)
    return record


def kinetics(fit: str, **changes) -> dict:
    value = {
        "channel": "hydrolysis",
        "fit": fit,
        "species": "S",
        "assay_enzyme": {"value": 20, "unit": "mg/L"},
    }
    value.update(changes)
    return value


def write(tmp_path: Path, name: str, data) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def run_define(tmp_path, tpl=None, fit=None, entry=None, name="run", **request):
    from alambiq.api import run

    params = {
        "template": write(tmp_path, f"{name}-template.json", tpl or template()),
        "kinetics": [
            kinetics(
                write(tmp_path, f"{name}-fit.json", fit or fit_record(tmp_path)),
                **(entry or {}),
            )
        ],
        **request,
    }
    step = {"id": "define", "capability": "operation.define", "params": params}
    return run({"id": "d", "steps": [step]}, run_dir=tmp_path / name)


def defined(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("define")


def failure(outcome) -> str:
    assert not outcome.ok
    return outcome.steps["define"].error


class TestConversion:
    def test_the_capacity_and_saturation_scale_are_the_hand_computed_values(
        self, tmp_path
    ):
        out = defined(run_define(tmp_path))
        (channel,) = out["channels"]
        assert channel["capacity_M_s"] == pytest.approx(CAPACITY, rel=1e-12)
        assert channel["capacity_M_s"] == pytest.approx(3e-6, rel=1e-12)
        assert channel["scale_M"] == pytest.approx(2.5e-3, rel=1e-12)
        assert channel["vmax_substrate_equivalent_M_s"] == pytest.approx(3e-6)
        definition = json.loads(Path(out["definition"]).read_text())
        stored = definition["channels"][0]
        assert stored["capacity_M_s"] == channel["capacity_M_s"]
        assert stored["factors"][0]["scale_M"] == channel["scale_M"]
        assert "Michaelis-Menten fit" in stored["source"]

    def test_everything_the_fit_does_not_set_is_exactly_as_the_template_wrote_it(
        self, tmp_path
    ):
        tpl = template()
        out = defined(run_define(tmp_path, tpl=tpl))
        definition = json.loads(Path(out["definition"]).read_text())
        expected = copy.deepcopy(tpl)
        expected["channels"][0]["capacity_M_s"] = definition["channels"][0][
            "capacity_M_s"
        ]
        expected["channels"][0]["factors"][0]["scale_M"] = definition["channels"][0][
            "factors"
        ][0]["scale_M"]
        expected["channels"][0]["source"] = definition["channels"][0]["source"]
        assert definition == expected
        assert out["unchanged_channels"] == ["side"]

    @pytest.mark.parametrize(
        "unit_value", [(0.02, "mg/mL"), (20000, "ug/L"), (0.02, "g/L")]
    )
    def test_the_same_enzyme_concentration_in_other_mass_units_gives_the_same_capacity(
        self, tmp_path, unit_value
    ):
        value, unit = unit_value
        out = defined(
            run_define(
                tmp_path,
                entry={"assay_enzyme": {"value": value, "unit": unit}},
                name=f"u{unit.replace('/', '')}",
            )
        )
        assert out["channels"][0]["capacity_M_s"] == pytest.approx(CAPACITY, rel=1e-12)

    def test_a_catalyst_counted_in_activity_units_takes_the_concentration_per_litre(
        self, tmp_path
    ):
        tpl = template()
        tpl["catalysts"][0].update(amount_unit="U", amount=5, reference_amount=5)
        out = defined(
            run_define(
                tmp_path,
                tpl=tpl,
                entry={"assay_enzyme": {"value": 10, "unit": "U/L"}},
            )
        )
        # 3e-6 M/s per 10 U/L, times 5 U over 0.5 L = 10 U/L: the same 3e-6 M/s.
        assert out["channels"][0]["capacity_M_s"] == pytest.approx(
            VMAX_UM_MIN * 1e-6 / 60 / 10 * 5 / 0.5, rel=1e-12
        )

    def test_two_measured_per_substrate_halves_the_substrate_rate(self, tmp_path):
        fit = fit_record(
            tmp_path,
            request={
                "x_unit": "mM",
                "y_unit": "uM/min",
                "rate_basis": {**BASIS, "measured_per_substrate": 2},
            },
        )
        out = defined(run_define(tmp_path, fit=fit))
        assert out["channels"][0]["capacity_M_s"] == pytest.approx(CAPACITY / 2)

    def test_a_stoichiometric_coefficient_of_two_halves_the_extent_rate(self, tmp_path):
        tpl = template()
        tpl["channels"][0]["stoichiometry"] = {"S": -2, "P": 2}
        out = defined(run_define(tmp_path, tpl=tpl))
        assert out["channels"][0]["capacity_M_s"] == pytest.approx(CAPACITY / 2)

    def test_half_the_enzyme_active_in_the_assay_doubles_the_rate_at_full_activity(
        self, tmp_path
    ):
        out = defined(run_define(tmp_path, entry={"assay_active_fraction": 0.5}))
        assert out["channels"][0]["capacity_M_s"] == pytest.approx(2 * CAPACITY)

    def test_a_larger_reference_amount_scales_the_capacity_with_it(self, tmp_path):
        tpl = template()
        tpl["catalysts"][0].update(amount=40, reference_amount=40)
        out = defined(run_define(tmp_path, tpl=tpl))
        assert out["channels"][0]["capacity_M_s"] == pytest.approx(4 * CAPACITY)

    def test_the_units_of_the_fit_do_not_change_the_answer(self, tmp_path):
        # 180 uM/min is 10.8 mM/h, and 2.5 mM is 2500 uM.
        fit = fit_record(
            tmp_path,
            request={"x_unit": "uM", "y_unit": "mM/h", "rate_basis": BASIS},
            parameters=[
                {"name": "Km", "value": 2500.0, "unit": "uM"},
                {"name": "Vmax", "value": 10.8, "unit": "mM/h"},
            ],
            domain={"minimum_x": 50.0, "maximum_x": 30000.0},
        )
        out = defined(run_define(tmp_path, fit=fit))
        assert out["channels"][0]["capacity_M_s"] == pytest.approx(CAPACITY, rel=1e-9)
        assert out["channels"][0]["scale_M"] == pytest.approx(2.5e-3, rel=1e-12)


class TestExtrapolation:
    def test_a_start_within_the_fitted_range_is_not_flagged(self, tmp_path):
        out = defined(run_define(tmp_path))
        channel = out["channels"][0]
        assert channel["fit_domain_M"] == pytest.approx([5e-5, 0.03])
        assert channel["substrate_maximum_M"] == pytest.approx(0.01)
        assert channel["extrapolates"] is False

    def test_a_start_above_the_fitted_range_is_flagged_and_noted(self, tmp_path):
        tpl = template()
        tpl["species"][0]["initial_M"] = 0.1
        outcome = run_define(tmp_path, tpl=tpl)
        out = defined(outcome)
        assert out["channels"][0]["extrapolates"] is True
        assert "above the highest measured concentration" in " ".join(
            outcome.steps["define"].notes
        )

    def test_a_fed_substrate_above_the_range_counts_too(self, tmp_path):
        tpl = template()
        tpl["phases"][0]["feed_M"] = {"S": 0.2}
        tpl["phases"][0]["dilution_per_s"] = 0.001
        out = defined(run_define(tmp_path, tpl=tpl))
        assert out["channels"][0]["substrate_maximum_M"] == pytest.approx(0.2)
        assert out["channels"][0]["extrapolates"] is True


class TestRefusals:
    def test_an_unknown_channel_is_refused(self, tmp_path):
        assert "no channel 'nope'" in failure(
            run_define(tmp_path, entry={"channel": "nope"})
        )

    def test_a_channel_without_a_catalyst_is_refused(self, tmp_path):
        tpl = template()
        del tpl["channels"][0]["catalyst"]
        assert "has no catalyst" in failure(run_define(tmp_path, tpl=tpl))

    def test_a_species_the_channel_does_not_consume_is_refused(self, tmp_path):
        assert "is not consumed" in failure(
            run_define(tmp_path, entry={"species": "P"})
        )

    def test_a_channel_without_a_saturation_factor_is_refused(self, tmp_path):
        tpl = template()
        tpl["channels"][0]["factors"][0]["kind"] = "mass-action"
        assert "saturation factor" in failure(run_define(tmp_path, tpl=tpl))

    def test_an_active_fraction_outside_zero_to_one_is_refused(self, tmp_path):
        for bad in (0, 1.5):
            assert "assay_active_fraction" in failure(
                run_define(
                    tmp_path, entry={"assay_active_fraction": bad}, name=f"a{bad}"
                )
            )

    def test_an_enzyme_unit_that_is_not_an_amount_per_volume_is_refused(self, tmp_path):
        assert "amount per volume" in failure(
            run_define(tmp_path, entry={"assay_enzyme": {"value": 20, "unit": "mg"}})
        )

    def test_a_unit_that_cannot_reach_the_catalysts_is_refused(self, tmp_path):
        assert "cannot be converted" in failure(
            run_define(tmp_path, entry={"assay_enzyme": {"value": 20, "unit": "U/mL"}})
        )

    def test_a_non_positive_enzyme_concentration_is_refused(self, tmp_path):
        assert "number above 0" in failure(
            run_define(tmp_path, entry={"assay_enzyme": {"value": 0, "unit": "mg/L"}})
        )

    def test_a_fit_without_a_rate_basis_is_refused(self, tmp_path):
        fit = fit_record(tmp_path, request={"x_unit": "mM", "y_unit": "uM/min"})
        assert "rate_basis" in failure(run_define(tmp_path, fit=fit))

    def test_a_fit_with_a_zero_km_is_refused(self, tmp_path):
        fit = fit_record(
            tmp_path,
            parameters=[
                {"name": "Km", "value": 0.0, "unit": "mM"},
                {"name": "Vmax", "value": 180.0, "unit": "uM/min"},
            ],
        )
        assert "positive Km" in failure(run_define(tmp_path, fit=fit))

    def test_the_same_channel_twice_is_refused(self, tmp_path):
        from alambiq.api import run

        fit = write(tmp_path, "fit.json", fit_record(tmp_path))
        params = {
            "template": write(tmp_path, "t.json", template()),
            "kinetics": [kinetics(fit), kinetics(fit)],
        }
        step = {"id": "define", "capability": "operation.define", "params": params}
        outcome = run({"id": "d", "steps": [step]}, run_dir=tmp_path / "twice")
        assert "twice" in failure(outcome)

    def test_a_template_that_is_not_a_valid_definition_is_refused(self, tmp_path):
        tpl = template()
        del tpl["times_s"]
        assert "A record must have the fields" in failure(run_define(tmp_path, tpl=tpl))

    def test_no_kinetics_is_refused(self, tmp_path):
        from alambiq.api import run

        params = {"template": write(tmp_path, "t.json", template()), "kinetics": []}
        step = {"id": "define", "capability": "operation.define", "params": params}
        outcome = run({"id": "d", "steps": [step]}, run_dir=tmp_path / "none")
        assert "1 to 50" in failure(outcome)


def michaelis_menten_substrate(s0: float, km: float, vmax: float, t: float) -> float:
    """Solve Km ln(S0/S) + (S0 - S) = Vmax t for S by bracketing."""
    from scipy.optimize import brentq

    return brentq(
        lambda s: km * math.log(s0 / s) + (s0 - s) - vmax * t, 1e-15, s0, xtol=1e-18
    )


class TestFromMeasurementToSimulation:
    def campaign(self, tmp_path: Path) -> dict:
        rows = "".join(
            f"m{i},enzyme,c{i},initial-rate,{VMAX_UM_MIN * x / (KM_MM + x)},uM/min,,,sample,measured\n"
            for i, x in enumerate(XS)
        )
        table = tmp_path / "rates.csv"
        table.write_text(HEAD + rows)
        return {
            "id": "fit-to-operation",
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
                    "params": {
                        "sample_id": "enzyme",
                        "metric": "initial-rate",
                        "x_quantity": "substrate",
                        "x_unit": "mM",
                        "y_unit": "uM/min",
                        "rate_basis": BASIS,
                    },
                },
                {
                    "id": "define",
                    "capability": "operation.define",
                    "bind": {"kinetics.0.fit": "fit.record"},
                    "params": {
                        "template": write(tmp_path, "template.json", template()),
                        "kinetics": [
                            {
                                "channel": "hydrolysis",
                                "species": "S",
                                "assay_enzyme": {"value": 0.02, "unit": "mg/mL"},
                            }
                        ],
                    },
                },
                {
                    "id": "simulate",
                    "capability": "operation.simulate",
                    "bind": {"definition": "define.definition"},
                    "params": {},
                },
            ],
        }

    def test_the_simulated_substrate_follows_the_integrated_solution(self, tmp_path):
        from alambiq.api import run

        outcome = run(self.campaign(tmp_path), run_dir=tmp_path / "run")
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        capacity = outcome.outputs("define")["channels"][0]["capacity_M_s"]
        assert capacity == pytest.approx(3e-6, rel=1e-5)
        record = json.loads(Path(outcome.outputs("simulate")["record"]).read_text())
        for state in record["trajectory"]:
            t = state["time_s"]
            expected = (
                michaelis_menten_substrate(0.01, 2.5e-3, capacity, t) if t else 0.01
            )
            # Only the enzyme channel is active in the phase, so the solution is exact.
            assert state["concentrations_M"]["S"] == pytest.approx(expected, rel=1e-5)
            assert state["concentrations_M"]["S"] + state["concentrations_M"]["P"] == (
                pytest.approx(0.01, rel=1e-9)
            )

    def test_the_fit_to_operation_campaign_reuses_its_steps_on_a_rerun(self, tmp_path):
        from alambiq.api import run

        campaign = self.campaign(tmp_path)
        run(campaign, run_dir=tmp_path / "run")
        again = run(campaign, run_dir=tmp_path / "run")
        assert all(s.reused for s in again.steps.values())
