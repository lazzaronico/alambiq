"""The engine boundary.

A capability that needs a scientific engine hands the work to a module running
inside that engine's environment, across a plain JSON boundary. These tests
cover the boundary itself, and then one real search, so the contract is
checked against MMseqs2 rather than only against a stub.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from alambiq.core import envs

HAS_MMSEQS = shutil.which("mmseqs") is not None

# The engine environment the developer has configured, read before any test
# overrides the variable. Empty when none is set.
CONFIGURED_MMSEQS_ENV = os.environ.get("ALAMBIQ_ENV_MMSEQS", "")

QUERY = """>query
MKFVKRRIIALVTILMLSVTSLFALQPSAKAAEHNPVVMVHGIGGASFNFAGIKSYLVSQGWSRDKLYAVDFWDK
TGTNYNNGPVLSRFVQKVLDETGAKKVDIVAHSMGGANTLYYIKNLDGGNKVANVVTLGGANRLTTGKALPGTDP
NQKILYTSIYSSADMIVMNYLSRLDGARNVQIHGVGHIGLLYSSQVNSLIKEGLNGGGQNTN
"""

PANEL = """>identical
MKFVKRRIIALVTILMLSVTSLFALQPSAKAAEHNPVVMVHGIGGASFNFAGIKSYLVSQGWSRDKLYAVDFWDK
TGTNYNNGPVLSRFVQKVLDETGAKKVDIVAHSMGGANTLYYIKNLDGGNKVANVVTLGGANRLTTGKALPGTDP
NQKILYTSIYSSADMIVMNYLSRLDGARNVQIHGVGHIGLLYSSQVNSLIKEGLNGGGQNTN
>fragment
MKFVKRRIIALVTILMLSVTSLFALQPSAKAAEHNPVVMVHGIGGASFNFAGIKSYLVSQ
>unrelated
MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGY
TARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMG
"""


class TestEnvironmentDeclaration:
    def test_an_undeclared_environment_lists_the_declared_ones(self):
        with pytest.raises(KeyError, match="declared"):
            envs.spec("nonexistent")

    def test_the_variable_name_follows_the_environment_name(self):
        assert envs.spec("openmm").variable == "ALAMBIQ_ENV_OPENMM"

    def test_a_missing_variable_names_the_recipe(self, monkeypatch):
        monkeypatch.delenv("ALAMBIQ_ENV_OPENFE", raising=False)
        report = envs.spec("openfe").check()
        assert report["ready"] is False
        assert "ALAMBIQ_ENV_OPENFE" in report["reason"]
        assert "binding-free-energy" in report["reason"]

    def test_a_variable_pointing_nowhere_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setenv("ALAMBIQ_ENV_OPENFE", "/no/such/python")
        report = envs.spec("openfe").check()
        assert report["ready"] is False
        assert "not a file" in report["reason"]


class TestCheckReadsInstalledNames:
    def test_a_package_installed_under_a_different_name_than_its_import_is_found(
        self, monkeypatch
    ):
        # PyYAML is imported as yaml; the check names packages as they are installed.
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        spec = envs.EnvSpec(
            name="mmseqs", recipe="docs/install.md", imports=("PyYAML", "no-such-package")
        )
        report = spec.check()
        assert report["versions"]["PyYAML"]
        assert report["versions"]["no-such-package"] is None
        assert report["missing"] == ["no-such-package"]
        assert report["ready"] is False


class TestBoundary:
    def test_the_boundary_carries_plain_json_both_ways(self, tmp_path, monkeypatch):
        """A module in the engines package is reachable and round-trips data."""
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        result = envs.run_in_environment(
            "mmseqs", "echo_probe:run", {"value": 41}, workdir=tmp_path
        )
        assert result == {"value": 41, "seen": True}

    def test_a_failing_module_reports_the_engine_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        with pytest.raises(RuntimeError) as caught:
            envs.run_in_environment(
                "mmseqs", "echo_probe:fail", {}, workdir=tmp_path
            )
        # The engine's own message survives, so a failure is readable without
        # re-running it.
        assert "deliberate engine failure" in str(caught.value)

    def test_an_unknown_entry_point_fails_clearly(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        with pytest.raises(RuntimeError, match="no_such_module"):
            envs.run_in_environment("mmseqs", "no_such_module:run", {}, workdir=tmp_path)


@pytest.fixture(scope="module")
def bare_python(tmp_path_factory) -> Path:
    """An interpreter that cannot import Alambiq.

    Engine environments are built from their own recipes and do not contain
    Alambiq, so this is the case the launcher exists for. A virtual
    environment with no packages and no access to the parent's is the smallest
    faithful stand-in, and building one needs no network.
    """
    place = tmp_path_factory.mktemp("bare") / "env"
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(place)
    python = place / "bin" / "python"
    premise = subprocess.run(
        [str(python), "-I", "-c", "import alambiq"], capture_output=True, text=True
    )
    assert premise.returncode != 0, "the stand-in must not be able to import alambiq"
    return python


class TestEnvironmentWithoutAlambiq:
    """The boundary works in an interpreter that has never heard of Alambiq."""

    def test_json_round_trips_through_it(self, bare_python, tmp_path, monkeypatch):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", str(bare_python))
        result = envs.run_in_environment(
            "mmseqs", "echo_probe:run", {"value": 41}, workdir=tmp_path
        )
        assert result == {"value": 41, "seen": True}

    def test_a_failure_still_carries_the_engine_message(
        self, bare_python, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", str(bare_python))
        with pytest.raises(RuntimeError, match="deliberate engine failure"):
            envs.run_in_environment("mmseqs", "echo_probe:fail", {}, workdir=tmp_path)

    def test_an_unknown_function_is_named(self, bare_python, tmp_path, monkeypatch):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", str(bare_python))
        with pytest.raises(RuntimeError, match="no_such_function"):
            envs.run_in_environment(
                "mmseqs", "echo_probe:no_such_function", {}, workdir=tmp_path
            )

    @pytest.mark.parametrize("entry", ["../escape:run", "a/b:run", "_runner:main"])
    def test_a_module_outside_the_engines_directory_is_refused(
        self, entry, bare_python, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", str(bare_python))
        with pytest.raises(RuntimeError, match="not an engine module name"):
            envs.run_in_environment("mmseqs", entry, {}, workdir=tmp_path)

    @pytest.mark.parametrize("entry", ["no-colon", ":run", "echo_probe:"])
    def test_a_malformed_entry_is_refused_before_anything_runs(
        self, entry, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        with pytest.raises(ValueError, match="module:function"):
            envs.run_in_environment("mmseqs", entry, {}, workdir=tmp_path)

    def test_engine_modules_import_only_the_standard_library_at_top_level(self):
        """What lets the runner load them in an environment without Alambiq.

        Third-party engines are imported inside the functions that need them,
        so a module can be loaded to be inspected even where its engine is
        absent. A sibling module in the same directory is allowed.
        """
        siblings = {p.stem for p in envs.ENGINES_DIR.glob("*.py")}
        offenders = []
        for path in envs.ENGINES_DIR.glob("*.py"):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    roots = [(node.module or "").split(".")[0]]
                else:
                    continue
                for root in roots:
                    if root not in sys.stdlib_module_names and root not in siblings:
                        offenders.append(f"{path.name}: {root}")
        assert not offenders, offenders


@pytest.mark.skipif(not HAS_MMSEQS, reason="mmseqs is not on PATH")
class TestRealSearch:
    @pytest.fixture
    def panel(self, tmp_path: Path) -> tuple[Path, Path]:
        query, panel = tmp_path / "q.fasta", tmp_path / "p.fasta"
        query.write_text(QUERY)
        panel.write_text(PANEL)
        return query, panel

    def test_a_search_finds_the_relative_and_reports_the_engine(
        self, panel, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        from alambiq.api import step

        query, sequences = panel
        result = step(
            "protein.search",
            query=str(query),
            panel=str(sequences),
            min_identity=0.3,
            min_coverage=0.5,
            run_dir=tmp_path / "run",
        )
        assert result.panel_size == 3
        targets = [h.target for h in result.hits]
        assert "identical" in targets
        # The fragment covers too little of the query to pass the threshold,
        # and the unrelated sequence is not similar at all.
        assert "unrelated" not in targets
        assert result.hits[0].identity == pytest.approx(1.0, abs=1e-3)
        assert result.engine_versions.get("mmseqs")

    def test_thresholds_that_exclude_everything_say_so(
        self, panel, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        from alambiq.api import run

        query, sequences = panel
        outcome = run(
            {
                "id": "search",
                "steps": [
                    {
                        "id": "find",
                        "capability": "protein.search",
                        "params": {
                            "query": str(query),
                            "panel": str(sequences),
                            "max_evalue": 1e-300,
                        },
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        assert outcome.ok
        assert outcome.outputs("find")["hits"] == []
        assert "No panel sequence met the thresholds" in " ".join(outcome.steps["find"].notes)

    def test_the_search_runs_in_an_interpreter_without_alambiq(
        self, bare_python, panel, tmp_path, monkeypatch
    ):
        """The configuration a researcher actually has: engine and Alambiq apart."""
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", str(bare_python))
        from alambiq.api import step

        query, sequences = panel
        result = step(
            "protein.search",
            query=str(query),
            panel=str(sequences),
            run_dir=tmp_path / "run",
        )
        assert "identical" in [h.target for h in result.hits]
        assert result.engine_versions.get("mmseqs")

    @pytest.mark.skipif(
        not CONFIGURED_MMSEQS_ENV,
        reason="set ALAMBIQ_ENV_MMSEQS to the MMseqs2 environment's Python to run this",
    )
    def test_the_search_runs_in_the_configured_engine_environment(
        self, panel, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", CONFIGURED_MMSEQS_ENV)
        from alambiq.api import step

        query, sequences = panel
        result = step(
            "protein.search",
            query=str(query),
            panel=str(sequences),
            run_dir=tmp_path / "run",
        )
        assert result.hits[0].identity == pytest.approx(1.0, abs=1e-3)

    def test_a_higher_sensitivity_never_finds_fewer_relatives(
        self, panel, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        from alambiq.api import step

        query, sequences = panel

        def found(sensitivity: float) -> set[str]:
            result = step(
                "protein.search",
                query=str(query),
                panel=str(sequences),
                max_evalue=10.0,
                min_identity=0.0,
                min_coverage=0.0,
                sensitivity=sensitivity,
                run_dir=tmp_path / f"sensitivity-{sensitivity}",
            )
            return {hit.target for hit in result.hits}

        low, high = found(1.0), found(7.5)
        assert "identical" in low
        assert low <= high


    @pytest.mark.parametrize("sensitivity", [0.0, 0.9, 7.6, 8.0, float("nan")])
    def test_a_sensitivity_outside_1_to_7_5_is_refused_before_the_search(
        self, panel, tmp_path, monkeypatch, sensitivity
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        from alambiq.api import run

        query, sequences = panel
        outcome = run(
            {
                "id": "search",
                "steps": [
                    {
                        "id": "find",
                        "capability": "protein.search",
                        "params": {
                            "query": str(query),
                            "panel": str(sequences),
                            "sensitivity": sensitivity,
                        },
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        assert outcome.steps["find"].status == "failed"
        assert "sensitivity must be between 1 and 7.5" in outcome.steps["find"].error

    @pytest.mark.parametrize("sensitivity", [1.0, 7.5])
    def test_the_limits_of_the_sensitivity_interval_are_accepted(
        self, panel, tmp_path, monkeypatch, sensitivity
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
        from alambiq.api import step

        query, sequences = panel
        result = step(
            "protein.search",
            query=str(query),
            panel=str(sequences),
            sensitivity=sensitivity,
            run_dir=tmp_path / "run",
        )
        assert result.hits[0].identity == pytest.approx(1.0, abs=1e-3)


class TestHints:
    def test_a_tool_that_cannot_run_a_file_it_wrote_gets_the_container_remedy(self):
        from alambiq.engines.hints import exec_hint

        mmseqs = "Failed to execute /tmp/x/searchslicedtargetprofile.sh with error 13."
        java = "UnsatisfiedLinkError: /tmp/lib.so: failed to map segment from shared object"
        assert "--tmpfs /tmp:rw,exec" in exec_hint(mmseqs)
        assert "--tmpfs /tmp:rw,exec" in exec_hint(java)
        assert exec_hint("mmseqs: no such database") == ""
