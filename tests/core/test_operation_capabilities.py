"""The batch balance and the operation family, end to end through the new core.

Each check has an answer known without the code: a first-order batch
(k = 0.2/s, S0 = 10 mM) has S(t) = S0 exp(-kt), its fit must recover the rate
that generated the data, and the balance and resource ledgers are short enough
to compute by hand.
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

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "operation"
S0, K = 0.01, 0.2


def fixture(name: str) -> str:
    return str(FIXTURES / name)


def run_ok(campaign: dict, tmp_path: Path):
    from alambiq.api import run

    outcome = run(campaign, run_dir=tmp_path / "run")
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


class TestSimulate:
    def test_first_order_decay_matches_the_closed_form(self, tmp_path):
        outcome = run_ok(
            {
                "id": "decay",
                "steps": [
                    {
                        "id": "batch",
                        "capability": "operation.simulate",
                        "params": {"definition": fixture("first-order.json")},
                    }
                ],
            },
            tmp_path,
        )
        out = outcome.outputs("batch")
        assert out["duration_s"] == 10.0
        assert out["final_concentrations_M"]["S"] == pytest.approx(
            S0 * math.exp(-K * 10), rel=1e-6
        )
        assert out["maximum_balance_error_M"] < 1e-12
        stored = json.loads(Path(out["record"]).read_text())
        assert len(stored["trajectory"]) == out["recorded_states"] == 4


class TestFitSimulateCompare:
    """A fit's operating definition feeds a simulation, which feeds a comparison."""

    def campaign(self) -> dict:
        return {
            "id": "progress",
            "steps": [
                {
                    "id": "fit",
                    "capability": "operation.fit",
                    "params": {"definition": fixture("first-order-fit.json")},
                },
                {
                    "id": "fitted",
                    "capability": "operation.simulate",
                    "bind": {"definition": "fit.operations.first-order-reference"},
                },
                {
                    "id": "against-measurements",
                    "capability": "operation.compare",
                    "bind": {"operation": "fitted.record"},
                    "params": {"comparison": fixture("first-order-observations.json")},
                },
            ],
        }

    def test_the_generating_rate_is_recovered(self, tmp_path):
        out = run_ok(self.campaign(), tmp_path).outputs("fit")
        (capacity,) = out["parameters"]
        # capacity = k * S0 at the reference scale of 10 mM.
        assert capacity["value"] == pytest.approx(K * S0, rel=1e-6)
        assert capacity["at_bound"] is False
        assert out["fitted"] == 4 and out["evaluated"] == 2
        assert out["fit_error"] < 1e-6 and out["evaluation_error"] < 1e-6
        assert out["converged_starts"] == out["starts"] == 2

    def test_the_fitted_definition_reproduces_the_measurements(self, tmp_path):
        outcome = run_ok(self.campaign(), tmp_path)
        fitted = outcome.outputs("fitted")
        assert fitted["final_concentrations_M"]["S"] == pytest.approx(
            S0 * math.exp(-K * 10), rel=1e-6
        )
        comparison = outcome.outputs("against-measurements")
        assert comparison["observations"] == 4
        assert {g["partition"] for g in comparison["errors"]} == {
            "calibration",
            "evaluation",
        }
        assert max(g["rmse"] for g in comparison["errors"]) < 1e-8

    def test_new_measurements_reuse_the_fit_and_the_simulation(self, tmp_path):
        run_ok(self.campaign(), tmp_path)
        changed = json.loads(Path(fixture("first-order-observations.json")).read_text())
        for row in changed["observations"]:
            row["value"] *= 1.1
        path = tmp_path / "later.json"
        path.write_text(json.dumps(changed))
        campaign = self.campaign()
        campaign["steps"][2]["params"]["comparison"] = str(path)
        again = run_ok(campaign, tmp_path)
        assert again.steps["fit"].reused and again.steps["fitted"].reused
        assert not again.steps["against-measurements"].reused
        assert max(g["rmse"] for g in again.outputs("against-measurements")["errors"]) > 1e-5

    def test_measurements_for_another_operation_are_refused(self, tmp_path):
        from alambiq.api import run

        other = json.loads(Path(fixture("first-order-observations.json")).read_text())
        other["operation_id"] = "a-different-batch"
        path = tmp_path / "other.json"
        path.write_text(json.dumps(other))
        campaign = self.campaign()
        campaign["steps"][2]["params"]["comparison"] = str(path)
        outcome = run(campaign, run_dir=tmp_path / "run")
        assert outcome.steps["fit"].status == "completed"
        assert outcome.steps["against-measurements"].status == "failed"


class TestEnsemble:
    def test_each_member_and_the_median_are_reported(self, tmp_path):
        steps = [
            {
                "id": f"member-{i}",
                "capability": "operation.simulate",
                "params": {"definition": fixture(f"member-{i}.json")},
            }
            for i in range(3)
        ]
        steps.append(
            {
                "id": "ensemble",
                "capability": "operation.compare_ensemble",
                "params": {
                    "comparison": fixture("ensemble-observations.json"),
                    "ensemble_id": "three-rates",
                    "source": "Analytic rate panel, k = 0.1, 0.2 and 0.4 per second",
                    # Members are stored records; their paths come from the runs.
                    "members": {},
                },
            }
        )
        from alambiq.api import run

        first = run({"id": "panel", "steps": steps[:3]}, run_dir=tmp_path / "run")
        assert first.ok
        steps[3]["params"]["members"] = {
            f"k{i}": first.outputs(f"member-{i}")["record"] for i in range(3)
        }
        out = run_ok({"id": "panel", "steps": steps}, tmp_path).outputs("ensemble")
        assert out["members"] == 3 and out["observations"] == 4
        # The observations are the k = 0.2 member's curve, which is also the median.
        assert max(g["rmse"] for g in out["member_errors"]["k1"]) < 1e-10
        assert max(g["rmse"] for g in out["errors"]) < 1e-10
        assert min(g["rmse"] for g in out["member_errors"]["k0"]) > 1e-4


class TestSummarize:
    def test_turnover_per_active_site_is_the_ledger_ratio(self, tmp_path):
        outcome = run_ok(
            {
                "id": "turnover",
                "steps": [
                    {
                        "id": "batch",
                        "capability": "operation.simulate",
                        "params": {"definition": fixture("enzyme-batch.json")},
                    },
                    {
                        "id": "account",
                        "capability": "operation.summarize",
                        "bind": {"operation": "batch.record"},
                        "params": {"assessment": fixture("enzyme-assessment.json")},
                    },
                ],
            },
            tmp_path,
        )
        out = outcome.outputs("account")
        # Half of 2 mg is active: rate 0.1/s over 10 s in 10 mL at 10 mM.
        formed = S0 * 0.01 * -math.expm1(-0.1 * 10)
        assert out["net_formed_mol"] == pytest.approx(formed, rel=1e-6)
        assert out["useful_output_mol"] == pytest.approx(formed * 0.8 * 0.9, rel=1e-6)
        (enzyme,) = out["catalysts"]
        # 1 mg active / 50 kDa, one site per molecule.
        sites = 1e-3 / 50000
        assert enzyme["supplied_active_site_mol"] == pytest.approx(sites)
        assert enzyme["product_per_active_site"] == pytest.approx(formed / sites, rel=1e-6)
        (lifetime,) = out["lifetimes"]
        assert lifetime["status"] == "above_at_all_recorded_states"


class TestResources:
    def test_equal_base_costs_cross_at_the_base_enzyme_price(self, tmp_path):
        out = run_ok(
            {
                "id": "inputs",
                "steps": [
                    {
                        "id": "compare",
                        "capability": "operation.compare_resources",
                        "params": {"comparison": fixture("resources.json")},
                    }
                ],
            },
            tmp_path,
        ).outputs("compare")
        a, b = out["alternatives"]
        # A: 2 g enzyme x 2 + 10 mmol feed x 0.2 = 6 over 8 useful mmol; B likewise.
        assert a["cost_per_useful_mmol"]["base"] == pytest.approx(0.75)
        assert b["cost_per_useful_mmol"]["base"] == pytest.approx(0.75)
        (pair,) = out["pairs"]
        assert pair["difference_per_useful_mmol"]["base"] == pytest.approx(0.0, abs=1e-12)
        enzyme = next(c for c in pair["crossovers"] if c["resource"] == "enzyme")
        assert enzyme["price"] == pytest.approx(2.0)
        assert enzyme["inside_declared_range"] is True


class TestBalance:
    KM, VMAX = 2.5, 180.0
    XS = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 8, 20, 30]

    def campaign(self, tmp_path: Path) -> dict:
        table = tmp_path / "rates.csv"
        table.write_text(
            "measurement_id,sample_id,condition_id,metric,value,unit,"
            "independent_id,technical_id,role\n"
            + "".join(
                f"m{i},enzyme,c{i},initial-rate,{self.VMAX * x / (self.KM + x)},"
                "uM/min,,,sample\n"
                for i, x in enumerate(self.XS)
            )
        )
        return {
            "id": "target",
            "steps": [
                {
                    "id": "intake",
                    "capability": "assay.inspect",
                    "params": {
                        "table": str(table),
                        "conditions": {
                            f"c{i}": f"Substrate {x} mM" for i, x in enumerate(self.XS)
                        },
                        "condition_values": {
                            f"c{i}": {"substrate": {"value": x, "unit": "mM"}}
                            for i, x in enumerate(self.XS)
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
                        "rate_basis": {
                            "kind": "product-formation",
                            "measured_species": "P",
                            "substrate": "S",
                            "measured_per_substrate": 1,
                        },
                    },
                },
                {
                    "id": "balance",
                    "capability": "assay.balance",
                    "bind": {"fit": "fit.record"},
                    "params": {
                        "substrate_initial": {"value": 10, "unit": "mM"},
                        "volume": {"value": 1, "unit": "L"},
                        "duration": {"value": 60, "unit": "min"},
                        "product_molar_mass": {"value": 100, "unit": "g/mol"},
                        "target_product": {"value": 500, "unit": "mg"},
                        "product_per_substrate": 1,
                        "formation_fraction": 1,
                        "recovery_fraction": 1,
                        "accepted_fraction": 1,
                        "assumption_note": "Synthetic batch for a known answer.",
                        "sensitivity": [
                            {"id": "too-much", "overrides": {"target_product": {"value": 2, "unit": "g"}}}
                        ],
                    },
                },
            ],
        }

    def test_required_activity_follows_the_integrated_rate_law(self, tmp_path):
        out = run_ok(self.campaign(tmp_path), tmp_path).outputs("balance")
        base, too_much = out["scenarios"]
        # 1 g theoretical, 0.5 g target: half conversion in 60 min.
        assert base["required_conversion"] == pytest.approx(0.5)
        integral_mM = 10 * 0.5 - self.KM * math.log(0.5)
        assert base["required_vmax"] == pytest.approx(integral_mM / 60 * 1000, rel=1e-4)
        assert base["required_vmax_unit"] == "uM/min"
        assert base["activity_multiple"] == pytest.approx(
            integral_mM / 60 * 1000 / self.VMAX, rel=1e-4
        )
        assert base["concentration_extrapolation"] is False
        assert too_much["required_conversion"] == pytest.approx(2.0)
        assert too_much["required_vmax"] is None
        assert "exceeds" in too_much["limitation"]

    def test_a_fit_without_a_rate_basis_is_refused(self, tmp_path):
        from alambiq.api import run

        campaign = self.campaign(tmp_path)
        del campaign["steps"][1]["params"]["rate_basis"]
        outcome = run(campaign, run_dir=tmp_path / "run")
        assert outcome.steps["fit"].status == "completed"
        assert outcome.steps["balance"].status == "failed"
        assert "rate_basis" in outcome.steps["balance"].error
