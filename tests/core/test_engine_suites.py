"""Run the engine unit suites inside the environments whose libraries they need.

The files `test_engine_*.py` exercise the science modules in `src/alambiq/engines`
directly, with PySCF, OpenMM or ParmEd. The core test environment has none of
them, so those files skip themselves there, and this module runs each in the pinned
environment that does, as the engine itself would be run, and requires all of its tests
to pass. Set the environment variable named in each case to that environment's Python.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
CASES = {
    "ALAMBIQ_ENV_QMMM": "test_engine_qmmm_*.py",
    "ALAMBIQ_ENV_OPENFF": "test_engine_openff_*.py",
    "ALAMBIQ_ENV_PYMBAR": "test_engine_umbrella_*.py",
}


@pytest.mark.parametrize(("variable", "pattern"), sorted(CASES.items()))
def test_the_engine_unit_tests_pass_in_their_own_environment(variable, pattern):
    python = os.environ.get(variable, "")
    if not (python and Path(python).is_file()):
        pytest.skip(f"set {variable} to run these engine tests")
    if not list(TESTS.glob(pattern)):
        pytest.skip(f"no {pattern} yet")
    finished = subprocess.run(
        [python, "-m", "unittest", "discover", "-s", str(TESTS), "-p", pattern],
        capture_output=True,
        text=True,
        timeout=3600,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PREFECT_SERVER_ANALYTICS_ENABLED": "false",
        },
        cwd=str(TESTS.parents[1]),
    )
    summary = re.search(r"Ran (\d+) tests? in", finished.stderr)
    assert finished.returncode == 0, finished.stderr[-3000:]
    assert summary and int(summary.group(1)) > 0
    assert "skipped" not in finished.stderr.splitlines()[-1], finished.stderr[-500:]
