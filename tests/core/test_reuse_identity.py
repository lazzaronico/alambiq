"""A saved result is reused only while the engine and models behind it are the same.

The reuse key already follows the request and the contents of its input files.
These tests cover the rest of what a result depends on: the engine environment
a step ran in, and the pinned models or databases it read. An upgraded tool or a
changed model revision must make earlier results stale, while the same engine
and models must keep them.
"""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from alambiq.core import envs, registry, resources
from alambiq.core.context import StepContext
from alambiq.core.envs import EnvSpec
from alambiq.core.execution import execute
from alambiq.core.model import Campaign, Step


@dataclass(frozen=True, slots=True)
class Request:
    value: int


@dataclass(frozen=True, slots=True)
class Choice:
    model: str
    value: int = 1


@dataclass(frozen=True, slots=True)
class Result:
    doubled: int


def make_tool(directory: Path, version: str) -> Path:
    """A command named faketool that reports a chosen version."""
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / "faketool"
    tool.write_text(f'#!/bin/sh\necho "faketool {version}"\n')
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return directory


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """Capabilities that declare an engine environment, a resource, or neither."""
    calls: list[str] = []
    monkeypatch.setitem(
        envs.ENVIRONMENTS,
        "probe",
        EnvSpec(
            name="probe", recipe="tests", imports=("pytest",), executables=("faketool",)
        ),
    )
    monkeypatch.setenv("ALAMBIQ_ENV_PROBE", sys.executable)
    monkeypatch.setitem(
        envs.ENVIRONMENTS,
        "bare",
        EnvSpec(name="bare", recipe="tests", imports=("pytest",)),
    )
    monkeypatch.setenv("ALAMBIQ_ENV_BARE", sys.executable)
    with registry.isolated():

        @registry.capability(
            "test.engine",
            family="measure",
            summary="Runs in an engine environment",
            environment="probe",
        )
        def engine_step(request: Request, ctx: StepContext) -> Result:
            """Double a number, pretending an engine did it."""
            calls.append("engine")
            return Result(request.value * 2)

        @registry.capability("test.plain", family="measure", summary="Runs in process")
        def plain_step(request: Request, ctx: StepContext) -> Result:
            """Double a number in process."""
            calls.append("plain")
            return Result(request.value * 2)

        @registry.capability(
            "test.model",
            family="measure",
            summary="Reads a pinned model",
            resources=("thermompnn",),
        )
        def model_step(request: Request, ctx: StepContext) -> Result:
            """Double a number, pretending a model did it."""
            calls.append("model")
            return Result(request.value * 2)

        @registry.capability(
            "test.variant",
            family="measure",
            summary="Runs in whichever engine variant the request names",
            environment="probe",
            environment_for=lambda request: (
                "probe" if request.model == "tooled" else "bare"
            ),
        )
        def variant_step(request: Choice, ctx: StepContext) -> Result:
            """Double a number in the named variant."""
            calls.append(request.model)
            return Result(request.value * 2)

        @registry.capability(
            "test.choice",
            family="measure",
            summary="Offers two models",
            resources=("thermompnn", "esm3-small"),
            resources_for=lambda request: (request.model,),
        )
        def choice_step(request: Choice, ctx: StepContext) -> Result:
            """Double a number, with whichever model was named."""
            calls.append(request.model)
            return Result(request.value * 2)

        yield calls


def use_tool(monkeypatch, directory: Path) -> None:
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")


def run(capability: str, run_dir: Path, value: int = 2):
    return execute(
        Campaign(steps=(Step(id="a", capability=capability, params={"value": value}),)),
        run_dir=run_dir,
    )


class TestTheEnvironmentIsPartOfTheKey:
    def test_the_same_environment_reuses_the_result(
        self, engine, monkeypatch, tmp_path
    ):
        use_tool(monkeypatch, make_tool(tmp_path / "v1", "1.0.0"))
        first = run("test.engine", tmp_path / "run")
        again = run("test.engine", tmp_path / "run")
        assert not first.steps["a"].reused and again.steps["a"].reused
        assert engine == ["engine"]

    def test_an_upgraded_tool_makes_the_result_stale(
        self, engine, monkeypatch, tmp_path
    ):
        use_tool(monkeypatch, make_tool(tmp_path / "v1", "1.0.0"))
        run("test.engine", tmp_path / "run")
        use_tool(monkeypatch, make_tool(tmp_path / "v2", "2.0.0"))
        upgraded = run("test.engine", tmp_path / "run")
        assert not upgraded.steps["a"].reused
        assert engine == ["engine", "engine"]

    def test_returning_to_the_old_tool_finds_the_old_result_again(
        self, engine, monkeypatch, tmp_path
    ):
        v1, v2 = (
            make_tool(tmp_path / "v1", "1.0.0"),
            make_tool(tmp_path / "v2", "2.0.0"),
        )
        use_tool(monkeypatch, v1)
        run("test.engine", tmp_path / "run")
        use_tool(monkeypatch, v2)
        run("test.engine", tmp_path / "run")
        use_tool(monkeypatch, v1)
        back = run("test.engine", tmp_path / "run")
        assert back.steps["a"].reused
        assert engine == ["engine", "engine"]

    def test_a_step_that_runs_in_process_does_not_depend_on_any_environment(
        self, engine, monkeypatch, tmp_path
    ):
        use_tool(monkeypatch, make_tool(tmp_path / "v1", "1.0.0"))
        run("test.plain", tmp_path / "run")
        use_tool(monkeypatch, make_tool(tmp_path / "v2", "2.0.0"))
        again = run("test.plain", tmp_path / "run")
        assert again.steps["a"].reused

    def test_an_environment_that_is_not_configured_fails_the_step_with_its_recipe(
        self, engine, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("ALAMBIQ_ENV_PROBE")
        outcome = run("test.engine", tmp_path / "run")
        assert outcome.steps["a"].status == "failed"
        assert "ALAMBIQ_ENV_PROBE" in outcome.steps["a"].error
        assert engine == []


class TestARequestDependsOnlyOnTheEnvironmentItRunsIn:
    def variant(self, model: str, run_dir: Path):
        return execute(
            Campaign(
                steps=(
                    Step(
                        id="a",
                        capability="test.variant",
                        params={"model": model, "value": 2},
                    ),
                )
            ),
            run_dir=run_dir,
        )

    def test_upgrading_the_tool_of_the_variant_it_used_makes_it_stale(
        self, engine, monkeypatch, tmp_path
    ):
        use_tool(monkeypatch, make_tool(tmp_path / "v1", "1.0.0"))
        self.variant("tooled", tmp_path / "run")
        use_tool(monkeypatch, make_tool(tmp_path / "v2", "2.0.0"))
        assert not self.variant("tooled", tmp_path / "run").steps["a"].reused

    def test_upgrading_a_tool_the_other_variant_has_leaves_it_alone(
        self, engine, monkeypatch, tmp_path
    ):
        use_tool(monkeypatch, make_tool(tmp_path / "v1", "1.0.0"))
        self.variant("plain", tmp_path / "run")
        use_tool(monkeypatch, make_tool(tmp_path / "v2", "2.0.0"))
        assert self.variant("plain", tmp_path / "run").steps["a"].reused

    def test_the_capability_names_its_default_variant_and_the_one_a_request_picks(
        self, engine
    ):
        capability = registry.get("test.variant")
        assert capability.environment == "probe"
        assert capability.environment_used(Choice("tooled")) == "probe"
        assert capability.environment_used(Choice("plain")) == "bare"
        assert capability.environment_used() == "probe"


class TestWhatAnEnvironmentReports:
    def test_it_reports_package_versions_and_tool_versions(
        self, engine, monkeypatch, tmp_path
    ):
        use_tool(monkeypatch, make_tool(tmp_path / "v1", "1.0.0"))
        found = envs.spec("probe").identity()
        assert found["packages"]["pytest"] == pytest.__version__
        assert found["tools"]["faketool"] == "faketool 1.0.0"
        assert found["python"] == ".".join(map(str, sys.version_info[:3]))

    def test_a_package_the_environment_lacks_is_reported_as_absent(self, monkeypatch):
        monkeypatch.setitem(
            envs.ENVIRONMENTS,
            "probe",
            EnvSpec(
                name="probe", recipe="tests", imports=("no_such_package_anywhere",)
            ),
        )
        monkeypatch.setenv("ALAMBIQ_ENV_PROBE", sys.executable)
        assert envs.spec("probe").identity()["packages"] == {
            "no_such_package_anywhere": None
        }

    def test_a_missing_tool_is_reported_as_absent(self, monkeypatch):
        monkeypatch.setitem(
            envs.ENVIRONMENTS,
            "probe",
            EnvSpec(
                name="probe", recipe="tests", executables=("no-such-tool-anywhere",)
            ),
        )
        monkeypatch.setenv("ALAMBIQ_ENV_PROBE", sys.executable)
        assert envs.spec("probe").identity()["tools"] == {"no-such-tool-anywhere": None}

    @pytest.mark.skipif(
        not os.environ.get("ALAMBIQ_ENV_MMSEQS"),
        reason="set ALAMBIQ_ENV_MMSEQS to the MMseqs2 environment's Python",
    )
    def test_the_real_mmseqs_environment_reports_its_release(self):
        found = envs.spec("mmseqs").identity()
        # The release the platform's recipes pin. Homebrew prints it as 18-8cc5c and
        # conda as 18.8cc5c; both are release 18, commit 8cc5c.
        assert found["tools"]["mmseqs"] in {"18-8cc5c", "18.8cc5c"}


class TestPinnedResources:
    def test_each_pinned_resource_has_its_own_stable_identity(self):
        names = ("esmc-300m", "esm3-small", "thermompnn")
        found = [resources.identity(n) for n in names]
        assert len(set(found)) == 3
        assert found == [resources.identity(n) for n in names]

    def test_a_changed_file_digest_is_a_different_resource(self, monkeypatch):
        record = resources.spec("thermompnn")
        before = resources.identity("thermompnn")
        altered = {**record, "files": [dict(f) for f in record["files"]]}
        altered["files"][0]["sha256"] = "0" * 64
        monkeypatch.setattr(resources, "spec", lambda name: altered)
        assert resources.identity("thermompnn") != before

    def test_a_changed_revision_is_a_different_resource(self, monkeypatch):
        record = resources.spec("thermompnn")
        before = resources.identity("thermompnn")
        monkeypatch.setattr(
            resources, "spec", lambda name: {**record, "version": "other"}
        )
        assert resources.identity("thermompnn") != before

    def test_the_order_files_are_listed_in_does_not_matter(self, monkeypatch):
        record = resources.spec("thermompnn")
        before = resources.identity("thermompnn")
        shuffled = {**record, "files": list(reversed(record["files"]))}
        monkeypatch.setattr(resources, "spec", lambda name: shuffled)
        assert resources.identity("thermompnn") == before

    def test_an_unpinned_resource_is_refused_when_the_capability_is_defined(self):
        with registry.isolated(), pytest.raises(KeyError, match="no-such-model"):

            @registry.capability(
                "test.bad", family="measure", summary="x", resources=("no-such-model",)
            )
            def bad(request: Request, ctx: StepContext) -> Result:
                """Never registered."""
                return Result(0)


class TestAResourceIsPartOfTheKey:
    def test_the_same_model_reuses_the_result(self, engine, tmp_path):
        first = run("test.model", tmp_path / "run")
        again = run("test.model", tmp_path / "run")
        assert not first.steps["a"].reused and again.steps["a"].reused
        assert engine == ["model"]

    def test_a_changed_model_makes_the_result_stale(
        self, engine, monkeypatch, tmp_path
    ):
        run("test.model", tmp_path / "run")
        monkeypatch.setattr(resources, "identity", lambda name: "a-different-revision")
        changed = run("test.model", tmp_path / "run")
        assert not changed.steps["a"].reused
        assert engine == ["model", "model"]


class TestARequestDependsOnlyOnTheModelItNamed:
    def choose(self, model: str, run_dir: Path):
        return execute(
            Campaign(
                steps=(Step(id="a", capability="test.choice", params={"model": model}),)
            ),
            run_dir=run_dir,
        )

    def test_changing_one_model_leaves_results_that_used_the_other_alone(
        self, engine, monkeypatch, tmp_path
    ):
        self.choose("thermompnn", tmp_path / "run")
        self.choose("esm3-small", tmp_path / "run")
        real = resources.identity
        monkeypatch.setattr(
            resources,
            "identity",
            lambda name: "new" if name == "esm3-small" else real(name),
        )
        kept = self.choose("thermompnn", tmp_path / "run")
        stale = self.choose("esm3-small", tmp_path / "run")
        assert kept.steps["a"].reused and not stale.steps["a"].reused
        assert engine == ["thermompnn", "esm3-small", "esm3-small"]

    def test_the_capability_lists_every_model_it_can_read(self, engine):
        assert set(registry.get("test.choice").resources) == {
            "thermompnn",
            "esm3-small",
        }


class TestAnEnvironmentsOwnToolsComeFirst:
    """An engine looks for its tools as an activated environment would."""

    @pytest.fixture
    def environment(self, monkeypatch, tmp_path):
        host = make_tool(tmp_path / "host", "1.0.0")
        use_tool(monkeypatch, host)
        inside = tmp_path / "env" / "bin"
        make_tool(inside, "2.0.0")
        (inside / "python").symlink_to(sys.executable)
        monkeypatch.setitem(
            envs.ENVIRONMENTS,
            "probe",
            EnvSpec(
                name="probe",
                recipe="tests",
                imports=("pytest",),
                executables=("faketool",),
            ),
        )
        monkeypatch.setenv("ALAMBIQ_ENV_PROBE", str(inside / "python"))
        return inside

    def test_the_tool_the_environment_holds_is_the_one_reported(self, environment):
        assert envs.spec("probe").identity()["tools"]["faketool"] == "faketool 2.0.0"

    def test_the_engine_finds_the_environments_tool_not_the_hosts(
        self, environment, tmp_path
    ):
        found = envs.run_in_environment(
            "probe", "echo_probe:where", {"tools": ["faketool"]}, workdir=tmp_path
        )
        assert found["faketool"] == str(environment / "faketool")

    def test_a_tool_only_the_host_has_is_still_found_after_the_environments_own(
        self, environment, tmp_path
    ):
        (environment / "faketool").unlink()
        found = envs.run_in_environment(
            "probe", "echo_probe:where", {"tools": ["faketool"]}, workdir=tmp_path
        )
        assert found["faketool"] == str(tmp_path / "host" / "faketool")

    def test_a_check_looks_in_the_environment_too(self, environment, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        assert envs.spec("probe").check()["missing"] == []
