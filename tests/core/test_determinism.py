"""A seeded capability gives the same result when it is run again from nothing.

Each case runs a campaign twice in two fresh run directories, so nothing is reused, and
compares the results after replacing the run directory in every stored path. The cases are
the ones that draw random numbers or start from random points: the random and
uncertainty-driven selections, codon recoding and the optimiser's starts. A different seed
must give a different result, or the seed is not doing
anything.
"""

from __future__ import annotations

import json
import shutil
from importlib.util import find_spec
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy") and find_spec("sklearn")),
    reason="the quantitative extra is required",
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def results(example: str, root: Path, mutate=None, steps=None):
    from alambiq.api import run

    # The example's inputs sit beside its campaign, so run a copy of the directory with the
    # changed campaign in it; nothing is written into the examples tree.
    directory = root / "example"
    shutil.copytree(EXAMPLES / example, directory)
    campaign = yaml.safe_load((directory / "campaign.yaml").read_text())
    if mutate:
        mutate(campaign)
    path = directory / "campaign.yaml"
    path.write_text(yaml.safe_dump(campaign, sort_keys=False))
    outcome = run(path, run_dir=root / "run")
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    text = json.dumps(
        {
            step: result.outputs
            for step, result in outcome.steps.items()
            if steps is None or step in steps
        },
        sort_keys=True,
        default=str,
    )
    return text.replace(str(root / "run"), "RUN")


def twice(example, tmp_path, mutate=None, steps=None):
    return (
        results(example, tmp_path / "one", mutate, steps),
        results(example, tmp_path / "two", mutate, steps),
    )


def test_selection_strategies_repeat_exactly(tmp_path):
    first, second = twice(
        "measured-variant-rounds",
        tmp_path,
        steps={"model", "highest-predicted", "most-uncertain", "random"},
    )
    assert first == second


def test_the_selection_seed_changes_a_random_batch(tmp_path):
    def reseed(campaign):
        for step in campaign["steps"]:
            if step["id"] == "random":
                step["params"]["seed"] = 2

    base = results("measured-variant-rounds", tmp_path / "a", steps={"random"})
    other = results("measured-variant-rounds", tmp_path / "b", reseed, {"random"})
    assert base != other


def test_codon_recoding_repeats_exactly(tmp_path):
    pytest.importorskip("dnachisel")
    pytest.importorskip("pydna")
    first, second = twice("variant-to-construct", tmp_path, steps={"recode", "build"})
    assert first == second


def test_progress_fits_repeat_exactly(tmp_path):
    first, second = twice(
        "operation-from-progress-curves", tmp_path, steps={"hydrogel-saturation"}
    )
    assert first == second
