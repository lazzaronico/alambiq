"""Temperature and pH in an operating definition, and a reversible rate law that obeys equilibrium.

Every answer is computed here from the formula it must follow:

* a channel declared to follow Arrhenius is scaled by exp(-Ea/R (1/T - 1/Tref)), and a
  first-order decay in it is exp(-k(T) t) with that k(T);
* a pH dependence scales by the bell-shaped activity at the stated pH over that at the
  reference pH, which is exactly one at the reference and at no other pH;
* a catalyst's deactivation follows Arrhenius too, and its active amount is A0 exp(-kd t);
* from fitted series (Arrhenius, pH profile, decay) through `operation.define` to
  `operation.simulate`, the simulated substrate follows the integrated Michaelis-Menten
  solution with the capacity those fits and the stated conditions imply;
* a reversible pair set by the Haldane relation relaxes to the equilibrium constant it was
  given, whatever the product's Michaelis constant, and the constant can come from its
  logarithm as `reaction.estimate_thermodynamics` reports it.
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

R = 8.314462618
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "operation"
S0, K0 = 0.01, 0.2


def first_order() -> dict:
    return json.loads((FIXTURES / "first-order.json").read_text())


def activity(ph: float, pka1: float = 5.5, pka2: float = 8.5) -> float:
    return 1.0 / (1.0 + 10 ** (pka1 - ph) + 10 ** (ph - pka2))


def arrhenius_factor(t: float, t_ref: float, ea: float) -> float:
    return math.exp(-ea / R * (1.0 / t - 1.0 / t_ref))


def simulate(definition: dict):
    from alambiq.capabilities.operate.science.simulation import simulate as integrate

    return integrate(definition)


def concentrations(record: dict) -> dict[float, float]:
    return {s["time_s"]: s["concentrations_M"]["S"] for s in record["trajectory"]}


class TestTemperature:
    def test_a_channel_follows_arrhenius_at_the_stated_temperature(self):
        definition = first_order()
        definition["state"] = {"temperature_K": 310.0}
        definition["channels"][0]["temperature"] = {
            "activation_energy_J_mol": 50_000.0,
            "reference_K": 300.0,
        }
        record = simulate(definition)
        k = K0 * arrhenius_factor(310.0, 300.0, 50_000.0)
        for t, s in concentrations(record).items():
            assert s == pytest.approx(S0 * math.exp(-k * t), rel=1e-6)
        (applied,) = record["conditions"]["multipliers"]
        assert applied["multiplier"] == pytest.approx(
            arrhenius_factor(310.0, 300.0, 50_000.0), rel=1e-12
        )

    def test_at_the_reference_temperature_nothing_changes(self):
        plain = simulate(first_order())
        definition = first_order()
        definition["state"] = {"temperature_K": 300.0}
        definition["channels"][0]["temperature"] = {
            "activation_energy_J_mol": 50_000.0,
            "reference_K": 300.0,
        }
        assert concentrations(simulate(definition)) == concentrations(plain)

    def test_a_colder_operation_is_slower_by_the_same_law(self):
        definition = first_order()
        definition["state"] = {"temperature_K": 280.0}
        definition["channels"][0]["temperature"] = {
            "activation_energy_J_mol": 50_000.0,
            "reference_K": 300.0,
        }
        record = simulate(definition)
        k = K0 * arrhenius_factor(280.0, 300.0, 50_000.0)
        assert k < K0
        assert concentrations(record)[10.0] == pytest.approx(
            S0 * math.exp(-k * 10), rel=1e-6
        )

    def test_the_request_in_the_record_is_the_one_that_was_stated(self):
        definition = first_order()
        definition["state"] = {"temperature_K": 310.0}
        definition["channels"][0]["temperature"] = {
            "activation_energy_J_mol": 50_000.0,
            "reference_K": 300.0,
        }
        record = simulate(copy.deepcopy(definition))
        assert record["request"]["channels"][0]["capacity_M_s"] == 0.002
        assert record["request"] == definition


class TestPh:
    def test_the_capacity_follows_the_bell_shaped_activity(self):
        definition = first_order()
        definition["state"] = {"pH": 6.0}
        definition["channels"][0]["ph"] = {
            "pKa1": 5.5,
            "pKa2": 8.5,
            "reference_pH": 7.0,
        }
        record = simulate(definition)
        factor = activity(6.0) / activity(7.0)
        assert 0 < factor < 1
        k = K0 * factor
        for t, s in concentrations(record).items():
            assert s == pytest.approx(S0 * math.exp(-k * t), rel=1e-6)

    def test_the_optimum_ph_is_the_midpoint_of_the_pkas_and_scales_by_more_than_one(
        self,
    ):
        definition = first_order()
        definition["state"] = {"pH": 7.0}
        definition["channels"][0]["ph"] = {
            "pKa1": 5.5,
            "pKa2": 8.5,
            "reference_pH": 6.0,
        }
        (applied,) = simulate(definition)["conditions"]["multipliers"]
        assert applied["multiplier"] == pytest.approx(activity(7.0) / activity(6.0))
        assert applied["multiplier"] > 1
        grid = [4 + i / 100 for i in range(701)]
        assert max(grid, key=activity) == pytest.approx(7.0, abs=0.01)

    def test_temperature_and_ph_multiply(self):
        definition = first_order()
        definition["state"] = {"temperature_K": 310.0, "pH": 6.0}
        definition["channels"][0]["temperature"] = {
            "activation_energy_J_mol": 50_000.0,
            "reference_K": 300.0,
        }
        definition["channels"][0]["ph"] = {
            "pKa1": 5.5,
            "pKa2": 8.5,
            "reference_pH": 7.0,
        }
        record = simulate(definition)
        k = (
            K0
            * arrhenius_factor(310.0, 300.0, 50_000.0)
            * activity(6.0)
            / activity(7.0)
        )
        assert concentrations(record)[10.0] == pytest.approx(
            S0 * math.exp(-k * 10), rel=1e-6
        )
        assert len(record["conditions"]["multipliers"]) == 2


class TestThermalDeactivation:
    def test_active_enzyme_decays_with_the_temperature_dependent_rate(self):
        definition = first_order()
        definition["catalysts"] = [
            {
                "id": "E",
                "amount": 10,
                "amount_unit": "mg",
                "active_fraction": 1,
                "reference_amount": 10,
                "deactivation_per_s": 1e-3,
                "washout_fraction": 0,
                "deactivation_temperature": {
                    "activation_energy_J_mol": 80_000.0,
                    "reference_K": 300.0,
                },
            }
        ]
        definition["channels"][0]["catalyst"] = "E"
        definition["state"] = {"temperature_K": 310.0}
        record = simulate(definition)
        kd = 1e-3 * arrhenius_factor(310.0, 300.0, 80_000.0)
        assert kd > 1e-3
        for state in record["trajectory"]:
            assert state["catalysts"]["E"]["active_amount"] == pytest.approx(
                10.0 * math.exp(-kd * state["time_s"]), rel=1e-6
            )


class TestRefusals:
    def fails(self, definition, message):
        with pytest.raises(ValueError, match=message):
            simulate(definition)

    def test_a_dependence_needs_the_operating_conditions(self):
        definition = first_order()
        definition["channels"][0]["temperature"] = {
            "activation_energy_J_mol": 5e4,
            "reference_K": 300.0,
        }
        self.fails(definition, "requires the operating conditions")

    def test_a_temperature_dependence_needs_a_temperature(self):
        definition = first_order()
        definition["state"] = {"pH": 7.0}
        definition["channels"][0]["temperature"] = {
            "activation_energy_J_mol": 5e4,
            "reference_K": 300.0,
        }
        self.fails(definition, "state.temperature_K")

    def test_a_ph_dependence_needs_a_ph(self):
        definition = first_order()
        definition["state"] = {"temperature_K": 300.0}
        definition["channels"][0]["ph"] = {
            "pKa1": 5.0,
            "pKa2": 9.0,
            "reference_pH": 7.0,
        }
        self.fails(definition, "state.pH")

    def test_a_malformed_dependence_is_refused(self):
        definition = first_order()
        definition["state"] = {"temperature_K": 300.0}
        definition["channels"][0]["temperature"] = {"activation_energy_J_mol": 5e4}
        self.fails(definition, "A record must have the fields")

    def test_a_temperature_that_is_not_positive_is_refused(self):
        definition = first_order()
        definition["state"] = {"temperature_K": 0.0}
        self.fails(definition, "temperature_K")


# ------------------------------------------------------------------ from fitted series

HEAD = (
    "measurement_id,sample_id,condition_id,metric,value,unit,"
    "independent_id,technical_id,role,status\n"
)
MM_XS = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 8, 20, 30]
KM_MM, VMAX = 2.5, 180.0
TEMPS = [280.0, 285.0, 290.0, 295.0, 300.0, 305.0, 310.0, 315.0, 320.0]
PHS = [4.5, 5.0, 5.5, 6.0, 6.5, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
EA = 60_000.0
BASIS = {
    "kind": "product-formation",
    "measured_species": "P",
    "substrate": "S",
    "measured_per_substrate": 1,
}


def table(
    tmp_path: Path, name: str, xs, ys, quantity: str, unit: str, y_unit: str
) -> dict:
    path = tmp_path / f"{name}.csv"
    path.write_text(
        HEAD
        + "".join(
            f"m{i},enzyme,c{i},rate,{y!r},{y_unit},,,sample,measured\n"
            for i, y in enumerate(ys)
        )
    )
    return {
        "id": f"{name}-intake",
        "capability": "assay.inspect",
        "params": {
            "table": str(path),
            "conditions": {f"c{i}": f"{quantity} {x}" for i, x in enumerate(xs)},
            "condition_values": {
                f"c{i}": {quantity: {"value": x, "unit": unit}}
                for i, x in enumerate(xs)
            },
        },
    }


def fit(name: str, quantity: str, unit: str, y_unit: str, **params) -> dict:
    return {
        "id": name,
        "capability": "assay.fit",
        "bind": {"measurements": f"{name}-intake.record"},
        "params": {
            "sample_id": "enzyme",
            "metric": "rate",
            "x_quantity": quantity,
            "x_unit": unit,
            "y_unit": y_unit,
            **params,
        },
    }


def template(tmp_path: Path, **changes) -> str:
    value = {
        "id": "hydrolysis",
        "volume_L": 0.5,
        "conditions": "Assay conditions",
        "state": {"temperature_K": 310.0, "pH": 6.5},
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
            }
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
    path = tmp_path / "template.json"
    path.write_text(json.dumps(value))
    return str(path)


def mm_series():
    return [VMAX * x / (KM_MM + x) for x in MM_XS]


def run_steps(steps, tmp_path, name="run"):
    from alambiq.api import run

    outcome = run({"id": "conditions", "steps": steps}, run_dir=tmp_path / name)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def mm_solution(s0: float, km: float, vmax: float, t: float) -> float:
    from scipy.optimize import brentq

    return brentq(
        lambda s: km * math.log(s0 / s) + (s0 - s) - vmax * t, 1e-15, s0, xtol=1e-18
    )


class TestFromFittedSeriesToSimulation:
    def campaign(self, tmp_path: Path) -> list[dict]:
        k_ref = 0.5
        arrhenius = [k_ref * arrhenius_factor(t, 300.0, EA) for t in TEMPS]
        ph = [10.0 * activity(p) for p in PHS]
        return [
            table(tmp_path, "mm", MM_XS, mm_series(), "substrate", "mM", "uM/min"),
            fit("mm", "substrate", "mM", "uM/min", rate_basis=BASIS),
            table(tmp_path, "temperature", TEMPS, arrhenius, "temperature", "K", "1/s"),
            fit(
                "temperature",
                "temperature",
                "K",
                "1/s",
                model="arrhenius",
                reference_x=300.0,
            ),
            table(tmp_path, "ph", PHS, ph, "ph", "pH", "U/mg"),
            fit("ph", "ph", "pH", "U/mg", model="ph_bell"),
            {
                "id": "define",
                "capability": "operation.define",
                "bind": {
                    "kinetics.0.fit": "mm.record",
                    "kinetics.0.temperature_fit": "temperature.record",
                    "kinetics.0.ph_fit": "ph.record",
                },
                "params": {
                    "template": template(tmp_path),
                    "kinetics": [
                        {
                            "channel": "hydrolysis",
                            "species": "S",
                            "assay_enzyme": {"value": 20, "unit": "mg/L"},
                            "assay_temperature_K": 300.0,
                            "assay_pH": 7.0,
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
        ]

    def test_the_simulation_follows_the_capacity_the_fits_and_the_conditions_imply(
        self, tmp_path
    ):
        outcome = run_steps(self.campaign(tmp_path), tmp_path)
        factor = arrhenius_factor(310.0, 300.0, EA) * activity(6.5) / activity(7.0)
        base = 3e-6  # 180 uM/min per 20 mg/L, scaled to 10 mg in 0.5 L
        defined = outcome.outputs("define")["channels"][0]
        assert defined["capacity_M_s"] == pytest.approx(base, rel=1e-5)
        assert defined["activation_energy_J_mol"] == pytest.approx(EA, rel=1e-5)
        assert defined["ph_pKa"] == pytest.approx([5.5, 8.5], abs=1e-4)
        multipliers = outcome.outputs("simulate")["condition_multipliers"]
        assert sorted((m["kind"], round(m["multiplier"], 6)) for m in multipliers) == (
            sorted(
                [
                    ("temperature", round(arrhenius_factor(310.0, 300.0, EA), 6)),
                    ("pH", round(activity(6.5) / activity(7.0), 6)),
                ]
            )
        )
        stored = json.loads(Path(outcome.outputs("simulate")["record"]).read_text())
        for state in stored["trajectory"]:
            t = state["time_s"]
            expected = mm_solution(0.01, 2.5e-3, base * factor, t) if t else 0.01
            assert state["concentrations_M"]["S"] == pytest.approx(expected, rel=1e-4)

    def test_a_dependence_fit_of_the_wrong_kind_is_refused(self, tmp_path):
        steps = self.campaign(tmp_path)
        steps[6]["bind"]["kinetics.0.temperature_fit"] = "ph.record"
        from alambiq.api import run

        outcome = run({"id": "c", "steps": steps}, run_dir=tmp_path / "wrong")
        assert "temperature-arrhenius-v1" in outcome.steps["define"].error

    def test_the_assay_conditions_are_required_with_a_dependence_fit(self, tmp_path):
        steps = self.campaign(tmp_path)
        del steps[6]["params"]["kinetics"][0]["assay_temperature_K"]
        from alambiq.api import run

        outcome = run({"id": "c", "steps": steps}, run_dir=tmp_path / "missing")
        assert "assay_temperature_K" in outcome.steps["define"].error


class TestInactivationFromFits:
    def test_decay_and_its_temperature_dependence_set_the_catalysts_deactivation(
        self, tmp_path
    ):
        from alambiq.api import run

        half_life_min = 30.0
        k = math.log(2) / half_life_min
        times = [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0]
        decay = [100.0 * math.exp(-k * t) for t in times]
        # Rate constants of inactivation at four temperatures, Ea = 150 kJ/mol about 300 K.
        ts = [290.0, 295.0, 300.0, 305.0]
        kds = [k * arrhenius_factor(t, 300.0, 150_000.0) for t in ts]
        steps = [
            table(tmp_path, "decay", times, decay, "time", "min", "%"),
            fit("decay", "time", "min", "%", model="first_order_decay"),
            table(tmp_path, "kd", ts, kds, "temperature", "K", "1/min"),
            fit(
                "kd", "temperature", "K", "1/min", model="arrhenius", reference_x=300.0
            ),
            table(tmp_path, "mm", MM_XS, mm_series(), "substrate", "mM", "uM/min"),
            fit("mm", "substrate", "mM", "uM/min", rate_basis=BASIS),
            {
                "id": "define",
                "capability": "operation.define",
                "bind": {
                    "kinetics.0.fit": "mm.record",
                    "inactivation.0.decay_fit": "decay.record",
                    "inactivation.0.temperature_fit": "kd.record",
                },
                "params": {
                    "template": template(tmp_path, state={"temperature_K": 310.0}),
                    "kinetics": [
                        {
                            "channel": "hydrolysis",
                            "species": "S",
                            "assay_enzyme": {"value": 20, "unit": "mg/L"},
                        }
                    ],
                    "inactivation": [{"catalyst": "E", "decay_temperature_K": 300.0}],
                },
            },
            {
                "id": "simulate",
                "capability": "operation.simulate",
                "bind": {"definition": "define.definition"},
                "params": {},
            },
        ]
        outcome = run({"id": "c", "steps": steps}, run_dir=tmp_path / "run")
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        row = outcome.outputs("define")["inactivation"][0]
        assert row["half_life_s"] == pytest.approx(1800.0, rel=1e-6)
        assert row["deactivation_per_s"] == pytest.approx(k / 60.0, rel=1e-6)
        assert row["activation_energy_J_mol"] == pytest.approx(150_000.0, rel=1e-4)
        kd_310 = k / 60.0 * arrhenius_factor(310.0, 300.0, 150_000.0)
        stored = json.loads(Path(outcome.outputs("simulate")["record"]).read_text())
        for state in stored["trajectory"]:
            assert state["catalysts"]["E"]["active_amount"] == pytest.approx(
                10.0 * math.exp(-kd_310 * state["time_s"]), rel=1e-4
            )


# ------------------------------------------------------------------ the equilibrium limit


def reversible_template(tmp_path: Path, **changes) -> str:
    def channel(identifier, source, target, reaction):
        return {
            "id": identifier,
            "reaction_id": reaction,
            "stoichiometry": {source: -1, target: 1},
            "capacity_M_s": 0.0,
            "catalyst": "E",
            "factors": [{"species": source, "kind": "mass-action", "scale_M": 1.0}],
            "denominator": {
                "constant": 1,
                "terms": [
                    {
                        "id": "s",
                        "coefficient": 1,
                        "factors": [{"species": "S", "scale_M": 1.0}],
                    },
                    {
                        "id": "p",
                        "coefficient": 1,
                        "factors": [{"species": "P", "scale_M": 1.0}],
                    },
                ],
            },
            "source": "placeholder",
        }

    value = json.loads(Path(template(tmp_path)).read_text())
    value["id"] = "reversible"
    value["state"] = {"temperature_K": 298.15}
    value["channels"] = [
        channel("forward", "S", "P", "S-to-P"),
        channel("reverse", "P", "S", "P-to-S"),
    ]
    value["phases"][0]["end_s"] = 500_000.0
    value["phases"][0]["active_channels"] = ["forward", "reverse"]
    value["times_s"] = [0.0, 50_000.0, 500_000.0]
    value.update(changes)
    path = tmp_path / "reversible.json"
    path.write_text(json.dumps(value))
    return str(path)


def pair(**changes) -> dict:
    value = {
        "forward": "forward",
        "reverse": "reverse",
        "substrate": "S",
        "product": "P",
        "product_km": {"value": 5.0, "unit": "mM"},
        "equilibrium_constant": 4.0,
    }
    value.update(changes)
    return value


def define_reversible(tmp_path, name="run", pair_changes=None, **extra):
    from alambiq.api import run

    steps = [
        table(tmp_path, "mm", MM_XS, mm_series(), "substrate", "mM", "uM/min"),
        fit("mm", "substrate", "mM", "uM/min", rate_basis=BASIS),
        {
            "id": "define",
            "capability": "operation.define",
            "bind": {"kinetics.0.fit": "mm.record"},
            "params": {
                "template": reversible_template(tmp_path),
                "kinetics": [
                    {
                        "channel": "forward",
                        "species": "S",
                        "assay_enzyme": {"value": 20, "unit": "mg/L"},
                    }
                ],
                "equilibria": [pair(**(pair_changes or {}))],
                **extra,
            },
        },
        {
            "id": "simulate",
            "capability": "operation.simulate",
            "bind": {"definition": "define.definition"},
            "params": {},
        },
    ]
    return run({"id": "reversible", "steps": steps}, run_dir=tmp_path / name)


def settled(outcome) -> tuple[float, float]:
    stored = json.loads(Path(outcome.outputs("simulate")["record"]).read_text())
    last = stored["trajectory"][-1]["concentrations_M"]
    return last["S"], last["P"]


class TestEquilibriumLimit:
    def test_the_haldane_relation_holds_for_the_parameters_that_were_set(
        self, tmp_path
    ):
        outcome = define_reversible(tmp_path)
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        (pair_out,) = outcome.outputs("define")["equilibria"]
        vf, ks, kp, k_eq = 3e-6, 2.5e-3, 5e-3, 4.0
        assert pair_out["reverse_capacity_M_s"] == pytest.approx(
            vf * kp / (k_eq * ks), rel=1e-5
        )
        assert pair_out["implied_equilibrium_constant"] == pytest.approx(4.0, rel=1e-12)
        assert pair_out["product_km_M"] == pytest.approx(5e-3)

    def test_the_simulated_reaction_relaxes_to_the_equilibrium_constant(self, tmp_path):
        outcome = define_reversible(tmp_path)
        assert outcome.ok
        s, p = settled(outcome)
        assert p / s == pytest.approx(4.0, rel=1e-4)
        assert s + p == pytest.approx(0.01, rel=1e-9)
        assert s == pytest.approx(0.01 / 5.0, rel=1e-4)

    def test_the_product_michaelis_constant_changes_the_speed_not_the_destination(
        self, tmp_path
    ):
        slow = define_reversible(
            tmp_path, "a", pair_changes={"product_km": {"value": 50.0, "unit": "mM"}}
        )
        fast = define_reversible(
            tmp_path, "b", pair_changes={"product_km": {"value": 0.5, "unit": "mM"}}
        )
        assert slow.ok and fast.ok
        for outcome in (slow, fast):
            s, p = settled(outcome)
            assert p / s == pytest.approx(4.0, rel=1e-3)
        reverse = [
            o.outputs("define")["equilibria"][0]["reverse_capacity_M_s"]
            for o in (slow, fast)
        ]
        assert reverse[0] / reverse[1] == pytest.approx(100.0, rel=1e-9)

    def test_the_constant_can_be_given_as_its_logarithm_like_the_thermodynamic_estimate(
        self, tmp_path
    ):
        outcome = define_reversible(
            tmp_path,
            pair_changes={
                "equilibrium_constant": None,
                "ln_equilibrium_constant": math.log(0.25),
            },
        )
        assert outcome.ok
        s, p = settled(outcome)
        assert p / s == pytest.approx(0.25, rel=1e-4)

    def test_early_on_the_forward_reaction_is_the_michaelis_menten_rate(self, tmp_path):
        outcome = define_reversible(tmp_path)
        stored = json.loads(Path(outcome.outputs("simulate")["record"]).read_text())
        # With no product the rational rate law reduces to Vf S / (Ks + S).
        capacity = outcome.outputs("define")["channels"][0]["capacity_M_s"]
        assert capacity == pytest.approx(3e-6, rel=1e-5)
        first = stored["trajectory"][1]
        assert first["time_s"] == 50_000.0
        assert first["concentrations_M"]["P"] > 0

    def test_a_constant_at_another_temperature_is_flagged_not_corrected(self, tmp_path):
        from alambiq.api import run

        steps = [
            table(tmp_path, "mm", MM_XS, mm_series(), "substrate", "mM", "uM/min"),
            fit("mm", "substrate", "mM", "uM/min", rate_basis=BASIS),
            {
                "id": "define",
                "capability": "operation.define",
                "bind": {"kinetics.0.fit": "mm.record"},
                "params": {
                    "template": reversible_template(
                        tmp_path, state={"temperature_K": 310.0}
                    ),
                    "kinetics": [
                        {
                            "channel": "forward",
                            "species": "S",
                            "assay_enzyme": {"value": 20, "unit": "mg/L"},
                        }
                    ],
                    "equilibria": [pair()],
                },
            },
        ]
        outcome = run({"id": "c", "steps": steps}, run_dir=tmp_path / "warm")
        assert outcome.ok
        assert "not corrected for temperature" in " ".join(
            outcome.steps["define"].notes
        )

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"equilibrium_constant": None}, "exactly one"),
            ({"ln_equilibrium_constant": 1.0}, "exactly one"),
            ({"equilibrium_constant": -1.0}, "finite number above 0"),
            ({"product_km": {"value": 5.0, "unit": "mol"}}, "product_km"),
            ({"product_km": {"value": 0.0, "unit": "mM"}}, "product_km"),
            ({"reverse": "forward"}, "channels of a reversible pair must differ"),
            ({"substrate": "P", "product": "S"}, "must consume"),
        ],
    )
    def test_a_pair_that_cannot_be_set_is_refused_by_name(
        self, tmp_path, changes, message
    ):
        outcome = define_reversible(tmp_path, pair_changes=changes)
        assert not outcome.ok
        assert message in outcome.steps["define"].error

    def test_a_forward_channel_that_is_not_in_rational_form_is_refused(self, tmp_path):
        from alambiq.api import run

        steps = [
            table(tmp_path, "mm", MM_XS, mm_series(), "substrate", "mM", "uM/min"),
            fit("mm", "substrate", "mM", "uM/min", rate_basis=BASIS),
            {
                "id": "define",
                "capability": "operation.define",
                "bind": {"kinetics.0.fit": "mm.record"},
                "params": {
                    "template": template(tmp_path),
                    "kinetics": [
                        {
                            "channel": "hydrolysis",
                            "species": "S",
                            "assay_enzyme": {"value": 20, "unit": "mg/L"},
                        }
                    ],
                    "equilibria": [pair(forward="hydrolysis", reverse="hydrolysis")],
                },
            },
        ]
        outcome = run({"id": "c", "steps": steps}, run_dir=tmp_path / "plain")
        assert not outcome.ok

    def test_kinetics_for_the_reverse_channel_do_not_set_a_reversible_pair(
        self, tmp_path
    ):
        from alambiq.api import run

        steps = [
            table(tmp_path, "mm", MM_XS, mm_series(), "substrate", "mM", "uM/min"),
            fit("mm", "substrate", "mM", "uM/min", rate_basis=BASIS),
            {
                "id": "define",
                "capability": "operation.define",
                "bind": {"kinetics.0.fit": "mm.record"},
                "params": {
                    "template": reversible_template(tmp_path),
                    "kinetics": [
                        {
                            "channel": "reverse",
                            "species": "P",
                            "assay_enzyme": {"value": 20, "unit": "mg/L"},
                        }
                    ],
                    "equilibria": [pair()],
                },
            },
        ]
        outcome = run({"id": "c", "steps": steps}, run_dir=tmp_path / "none")
        assert not outcome.ok
        assert "saturation factor" in outcome.steps["define"].error

    def test_a_forward_channel_with_no_kinetics_entry_is_refused(self, tmp_path):
        from alambiq.capabilities.operate.science.definition import define

        definition = json.loads(Path(reversible_template(tmp_path)).read_text())
        with pytest.raises(ValueError, match="has no kinetics entry"):
            define(
                definition,
                [],
                [
                    {
                        "forward": "forward",
                        "reverse": "reverse",
                        "substrate": "S",
                        "product": "P",
                        "product_km_M": 5e-3,
                        "equilibrium_constant": 4.0,
                    }
                ],
            )

    def test_the_report_names_the_reverse_capacity_and_the_constant_the_law_gives(
        self, tmp_path
    ):
        outcome = define_reversible(tmp_path)
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "Equilibrium constant of the rate law" in page and "Reverse capacity" in page


class TestFromThermodynamicsToEquilibrium:
    """Glucose 6-phosphate to fructose 6-phosphate: the estimate sets where the operation settles."""

    def test_the_operation_settles_at_the_estimated_equilibrium_constant(
        self, tmp_path
    ):
        pytest.importorskip("equilibrator_api")
        pytest.importorskip("rdkit")
        from alambiq.api import run

        conditions = {
            "pH": 7.0,
            "pMg": 3.0,
            "ionic_strength_M": 0.25,
            "temperature_K": 298.15,
        }
        thermo = {
            "id": "thermo",
            "capability": "reaction.estimate_thermodynamics",
            "params": {
                "id": "g6p-to-f6p",
                "conditions": conditions,
                "assumption_note": "Ideal dilute aqueous activities, concentration divided by 1 M.",
                "compounds": [
                    {"id": "g6p", "accession": "kegg:C00092", "activity": 0.001},
                    {"id": "f6p", "accession": "kegg:C05345", "activity": 0.001},
                ],
                "reactions": [
                    {
                        "id": "isomerisation",
                        "stoichiometry": {"g6p": -1, "f6p": 1},
                        "source": {
                            "kind": "curated-reference",
                            "citation": "eQuilibrator",
                        },
                    }
                ],
                "route_weights": {"isomerisation": 1},
            },
        }
        steps = [
            thermo,
            table(tmp_path, "mm", MM_XS, mm_series(), "substrate", "mM", "uM/min"),
            fit("mm", "substrate", "mM", "uM/min", rate_basis=BASIS),
            {
                "id": "define",
                "capability": "operation.define",
                "bind": {
                    "kinetics.0.fit": "mm.record",
                    "equilibria.0.ln_equilibrium_constant": "thermo.reactions.0.ln_k_prime",
                },
                "params": {
                    "template": reversible_template(tmp_path),
                    "kinetics": [
                        {
                            "channel": "forward",
                            "species": "S",
                            "assay_enzyme": {"value": 20, "unit": "mg/L"},
                        }
                    ],
                    "equilibria": [
                        {
                            "forward": "forward",
                            "reverse": "reverse",
                            "substrate": "S",
                            "product": "P",
                            "product_km": {"value": 5.0, "unit": "mM"},
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
        ]
        outcome = run({"id": "c", "steps": steps}, run_dir=tmp_path / "run")
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        ln_k = outcome.outputs("thermo")["reactions"][0]["ln_k_prime"]
        dg0 = outcome.outputs("thermo")["reactions"][0]["dg0_prime_kj_mol"]
        rt = outcome.outputs("thermo")["rt_kj_mol"]
        assert ln_k == pytest.approx(-dg0 / rt, rel=1e-9)
        s, p = settled(outcome)
        assert p / s == pytest.approx(math.exp(ln_k), rel=1e-3)
        assert s + p == pytest.approx(0.01, rel=1e-9)
