"""The Dockerfiles name their environments the way the current interface reads them.

The images are built and run on a Linux ARM64 host (see `environments/docker/README.md`);
these checks only keep the files consistent with `alambiq.core.envs`: every variable a layer
sets is one the core reads, no layer sets a variable the earlier interface used, and every
engine environment that can run in a container has a layer that provides it.
"""

from __future__ import annotations

import re
from pathlib import Path

from alambiq.core import envs

DIRECTORY = Path(__file__).resolve().parents[2] / "environments/docker"
# Apple's GPU path is native macOS only.
NOT_IN_A_CONTAINER = {"boltz_mps"}


def layers() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(DIRECTORY.glob("Dockerfile*"))}


def variables(text: str) -> set[str]:
    return set(re.findall(r"\b(ALAMBIQ_[A-Z0-9_]+)=", text))


def test_every_variable_a_layer_sets_is_one_the_core_reads():
    known = {spec.variable for spec in envs.ENVIRONMENTS.values()}
    for name, text in layers().items():
        for variable in variables(text):
            assert variable in known | {"ALAMBIQ_EXTRAS", "ALAMBIQ_PREFECT_HOME"}, (
                name,
                variable,
            )


def test_no_layer_sets_a_variable_of_the_earlier_interface():
    old = re.compile(r"ALAMBIQ_(?!ENV_)[A-Z0-9_]*(PYTHON|JAVA|PREFIX|ROOT)=")
    for name, text in layers().items():
        assert not old.search(text), name


def test_every_containerisable_environment_has_a_layer():
    provided = set()
    for text in layers().values():
        provided |= variables(text)
    wanted = {
        spec.variable
        for key, spec in envs.ENVIRONMENTS.items()
        if key not in NOT_IN_A_CONTAINER
    }
    assert wanted <= provided, sorted(wanted - provided)


def test_each_layer_points_its_variable_at_a_python_inside_the_image():
    for name, text in layers().items():
        for variable, value in re.findall(
            r"\b(ALAMBIQ_ENV_[A-Z0-9_]+)=(\S+)", text.replace("\\\n", " ")
        ):
            assert value.endswith("/bin/python"), (name, variable, value)
