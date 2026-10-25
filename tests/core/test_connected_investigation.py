"""The connected investigation: a protein carried through to an operating comparison.

`examples/connected-investigation` chains protein inspection, variant proposal, a returned
round of measurements, a fitted model, the choice of the next variant, its kinetics, the
operating definition, a simulation and a comparison with a measured progress curve. Its inputs
are generated from a stated truth (see `prepare.py`), so every stage has an answer:

* the proposals are the language-model ranking with the catalytic residues left out and one
  per position;
* the fitted model beats predicting the training mean on the held-out variants, and picks the
  variant the truth says is best of the pool;
* the kinetics intervals contain the constants that generated the rates;
* the operating definition's capacity is the fitted Vmax in M/s for the enzyme the assay held,
  and the simulated substrate follows the integrated Michaelis-Menten solution for it;
* the simulation agrees with the measured progress curve to about one percent of the
  starting concentration;
* a second run in the same directory reuses every step.
"""

from __future__ import annotations

import json
import math
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy") and find_spec("sklearn")),
    reason="the quantitative extra is required",
)

HERE = Path(__file__).resolve().parents[2] / "examples/connected-investigation"
STEPS = [
    "first-look",
    "propose",
    "round",
    "model",
    "next-batch",
    "kinetics-intake",
    "kinetics",
    "define",
    "simulate",
    "compare",
]


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    from alambiq.api import run

    run_dir = tmp_path_factory.mktemp("connected") / "run"
    first = run(HERE / "campaign.yaml", run_dir=run_dir)
    assert first.ok, [s.error for s in first.steps.values() if s.error]
    return first, run_dir


@pytest.fixture(scope="module")
def truth():
    return json.loads((HERE / "truth.json").read_text())


def stored(result, step: str) -> dict:
    return json.loads(Path(result.outputs(step)["record"]).read_text())


def test_every_stage_runs_and_the_report_tells_the_whole_story(outcome, tmp_path):
    result, _ = outcome
    assert list(result.steps) == STEPS or set(result.steps) == set(STEPS)
    page = result.report(tmp_path / "r.html").read_text()
    for step in STEPS:
        assert f'id="{step}"' in page


def test_the_proposals_are_the_language_models_ranking_without_the_catalytic_residues(
    outcome, truth
):
    result, _ = outcome
    proposals = result.outputs("propose")["proposals"]
    assert [p["variant"] for p in proposals] == truth["proposals"]
    assert all(p["position"] not in (35, 52) for p in proposals)
    assert len({p["position"] for p in proposals}) == 24
    llrs = [p["llr"] for p in proposals]
    assert llrs == sorted(llrs, reverse=True)
    scores = json.loads((HERE / "scores.json").read_text())
    assert scores["model"] == "biohub/ESMC-300M" and len(scores["rows"]) == 19 * 129


def test_the_fitted_model_beats_the_training_mean_on_held_out_variants(outcome):
    result, _ = outcome
    fit = result.outputs("model")
    assert fit["evaluated_on"] == 6 and fit["trained_on"] == 12
    assert fit["model"]["rmse"] < fit["baseline"]["rmse"] / 2
    assert fit["model"]["r_squared"] > 0.5


def test_the_next_variant_is_the_one_the_investigation_goes_on_to_measure(
    outcome, truth
):
    result, _ = outcome
    (picked,) = result.outputs("next-batch")["selected"]
    assert picked["sample_id"] == truth["chosen_variant"]
    assert picked["sample_id"] in truth["pool"]
    kinetics = result.steps["kinetics"].request
    assert kinetics["sample_id"] == truth["chosen_variant"]


def test_the_kinetics_intervals_contain_the_constants_that_generated_the_rates(
    outcome, truth
):
    result, _ = outcome
    found = {p["name"]: p for p in result.outputs("kinetics")["parameters"]}
    low, high = found["Km"]["interval_95"]
    assert low < truth["km_mM"] < high
    low, high = found["Vmax"]["interval_95"]
    assert low < truth["vmax_uM_min"] < high


def test_the_operating_capacity_is_the_fitted_rate_for_this_much_enzyme(outcome):
    result, _ = outcome
    vmax = {p["name"]: p["value"] for p in result.outputs("kinetics")["parameters"]}[
        "Vmax"
    ]
    channel = result.outputs("define")["channels"][0]
    # 20 mg/L in the assay, 10 mg in 0.5 L: the same concentration, so the same rate.
    assert channel["capacity_M_s"] == pytest.approx(vmax * 1e-6 / 60.0, rel=1e-9)
    assert channel["scale_M"] == pytest.approx(
        {p["name"]: p["value"] for p in result.outputs("kinetics")["parameters"]}["Km"]
        * 1e-3,
        rel=1e-9,
    )
    assert channel["extrapolates"] is False


def test_the_simulation_follows_the_integrated_solution_for_the_fitted_constants(
    outcome,
):
    from scipy.optimize import brentq

    result, _ = outcome
    channel = result.outputs("define")["channels"][0]
    record = stored(result, "simulate")
    for state in record["trajectory"]:
        t = state["time_s"]
        if t == 0:
            continue
        expected = brentq(
            lambda s, t=t: (
                channel["scale_M"] * math.log(0.01 / s)
                + (0.01 - s)
                - channel["capacity_M_s"] * t
            ),
            1e-15,
            0.01,
            xtol=1e-18,
        )
        assert state["concentrations_M"]["S"] == pytest.approx(expected, rel=1e-4)


def test_the_simulation_agrees_with_the_measured_progress_curve(outcome):
    result, _ = outcome
    (group,) = result.outputs("compare")["errors"]
    assert group["count"] == 7 and group["unit"] == "M"
    assert group["rmse"] < 2e-4  # two percent of the 0.01 M starting concentration


def test_a_second_run_reuses_every_step(outcome):
    from alambiq.api import run

    _, run_dir = outcome
    again = run(HERE / "campaign.yaml", run_dir=run_dir)
    assert again.ok and all(s.reused for s in again.steps.values())
