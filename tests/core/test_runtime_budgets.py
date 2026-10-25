"""No step of an engine-free example takes longer than a researcher would expect to wait.

Each budget is about ten times the time measured on a laptop while other work was running,
with a floor of five seconds, so it fails when a step becomes an order of magnitude slower
and not when a machine is busy. A budget that has to be raised is a change to report.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy") and find_spec("sklearn")),
    reason="the quantitative extra is required",
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

# Seconds per step, over a floor of five. Only the steps that take noticeable time are listed.
BUDGETS = {
    "kinetics-from-measurements": {"fit": 10},
    "representative-panel": {"layout": 10},
    "measured-variant-rounds": {"model": 10},
    "operation-from-progress-curves": {
        "hydrogel-saturation": 250,
        "free-saturation": 150,
        "hydrogel-first-order": 70,
        "free-first-order": 30,
    },
    "cofactor-regeneration": {"with": 10},
    "connected-investigation": {"simulate": 10},
}


@pytest.mark.parametrize("name", sorted(BUDGETS))
def test_steps_stay_within_their_budget(name, tmp_path):
    from alambiq.api import run

    outcome = run(EXAMPLES / name / "campaign.yaml", run_dir=tmp_path / "run")
    assert outcome.ok
    for step_id, result in outcome.steps.items():
        budget = BUDGETS[name].get(step_id, 5)
        assert result.seconds < budget, (name, step_id, result.seconds, budget)
