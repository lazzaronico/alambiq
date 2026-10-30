"""Running scientific engines in their own environments.

Each engine (OpenMM, PySCF, MMseqs2) needs its own installation. A capability
that uses an engine passes the work to an engine module that runs inside the
engine environment. The request and the result cross the process boundary as
JSON, so an engine module imports only the standard library and its own engine.

Each engine environment is declared once in this module and is located through
one `ALAMBIQ_ENV_<NAME>` variable. Version expectations are part of the
environment recipe, not of this code.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENGINES_DIR = Path(__file__).resolve().parents[1] / "engines"
RUNNER = ENGINES_DIR / "_runner.py"


class EnvironmentMissing(RuntimeError):
    """An engine environment the host has not provided."""


@dataclass(frozen=True, slots=True)
class EnvSpec:
    """One engine environment: how to find it and what it should contain."""

    name: str
    recipe: str
    imports: tuple[str, ...] = ()
    executables: tuple[str, ...] = ()

    @property
    def variable(self) -> str:
        return f"ALAMBIQ_ENV_{self.name.upper().replace('-', '_')}"

    def interpreter(self) -> Path:
        """The Python that has this engine installed."""
        value = os.environ.get(self.variable)
        if not value:
            raise EnvironmentMissing(
                f"{self.variable} is not set. Set it to the path of the Python interpreter "
                f"of an environment built from {self.recipe}"
            )
        path = Path(value)
        if not path.is_file():
            raise EnvironmentMissing(
                f"{self.variable} is {value!r}, which is not a file. Set it to the path "
                "of a Python interpreter."
            )
        return path

    def search_path(self) -> str:
        """PATH as seen from inside this environment.

        The environment's own `bin` directory comes first, as it does when an
        environment is activated, so a tool that the engine looks for is the
        environment's tool and not another copy installed on the host.
        """
        return os.pathsep.join(
            [str(self.interpreter().parent), os.environ.get("PATH", "")]
        )

    def identity(self) -> dict[str, Any]:
        """What this environment is, for deciding whether a stored result applies.

        The interpreter's Python version, the version of every declared
        package, and the version of every declared tool. Two environments with
        the same identity are interchangeable for reuse; an upgrade of any one
        of them is not.
        """
        return _identity(
            self.name,
            str(self.interpreter()),
            self.imports,
            self.executables,
            self.search_path(),
        )

    def check(self) -> dict[str, Any]:
        """Report what this environment provides, without running science."""
        try:
            interpreter = self.interpreter()
        except EnvironmentMissing as missing:
            return {
                "name": self.name,
                "ready": False,
                "reason": str(missing),
                "recipe": self.recipe,
                "missing": list(self.imports) + list(self.executables),
            }
        # Packages are named as they are installed (PyYAML, openff-toolkit), which is not
        # always how they are imported, so ask the installed metadata first.
        program = (
            "import importlib.metadata as m, importlib.util as u, json\n"
            f"names={list(self.imports)!r}\n"
            "def version(n):\n"
            "    try:\n"
            "        return m.version(n)\n"
            "    except m.PackageNotFoundError:\n"
            "        found = u.find_spec(n.replace('-', '_')) is not None\n"
            "        return 'installed (version unknown)' if found else None\n"
            "print(json.dumps({n: version(n) for n in names}))"
        )
        try:
            finished = subprocess.run(
                [str(interpreter), "-I", "-c", program],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            found = json.loads(finished.stdout or "{}")
        except Exception as error:  # noqa: BLE001 - reported, not raised
            return {
                "name": self.name,
                "ready": False,
                "reason": str(error),
                "recipe": self.recipe,
                "missing": list(self.imports) + list(self.executables),
            }
        missing_imports = sorted(n for n, v in found.items() if v is None)
        missing_tools = sorted(
            e for e in self.executables if not shutil.which(e, path=self.search_path())
        )
        return {
            "name": self.name,
            "ready": not missing_imports and not missing_tools,
            "interpreter": str(interpreter),
            "versions": found,
            "missing": missing_imports + missing_tools,
            "recipe": self.recipe,
        }


def _tool_version(executable: str, search_path: str | None = None) -> str | None:
    """The first line a tool prints for `version`, or for `--version`."""
    path = shutil.which(executable, path=search_path)
    if path is None:
        return None
    for argument in ("version", "--version"):
        try:
            finished = subprocess.run(
                [path, argument], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = (finished.stdout or finished.stderr).strip()
        if finished.returncode == 0 and output:
            return output.splitlines()[0].strip()
    return "unknown"


_IDENTITY_PROBE = """
import importlib.metadata as metadata, json, sys

found = {}
for name in json.loads(sys.argv[1]):
    try:
        found[name] = metadata.version(name)
    except Exception:
        found[name] = None
print(json.dumps({"python": ".".join(map(str, sys.version_info[:3])), "packages": found}))
"""


def _identity(
    name: str,
    interpreter: str,
    imports: tuple[str, ...],
    executables: tuple[str, ...],
    search_path: str | None = None,
) -> dict[str, Any]:
    """Ask the environment what it contains, every time.

    The answer is not cached, so that an upgrade made during a session changes
    the identity of the environment at the next step.
    """
    finished = subprocess.run(
        [interpreter, "-I", "-c", _IDENTITY_PROBE, json.dumps(list(imports))],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if finished.returncode != 0:
        raise RuntimeError(
            f"Could not read the {name} environment at {interpreter}: "
            f"{finished.stderr.strip() or finished.stdout.strip()}"
        )
    reported = json.loads(finished.stdout)
    return {
        "environment": name,
        "python": reported["python"],
        "packages": reported["packages"],
        "tools": {tool: _tool_version(tool, search_path) for tool in executables},
    }


ENVIRONMENTS: dict[str, EnvSpec] = {
    "mmseqs": EnvSpec(
        name="mmseqs",
        recipe="docs/install.md",
        imports=(),
        executables=("mmseqs",),
    ),
    "esm": EnvSpec(
        name="esm",
        recipe="environments/protein-representations/README.md",
        imports=("esm", "torch", "transformers", "numpy"),
    ),
    "thermompnn": EnvSpec(
        name="thermompnn",
        recipe="environments/thermompnn/README.md",
        imports=("torch", "numpy"),
    ),
    "p2rank": EnvSpec(
        name="p2rank",
        recipe="environments/p2rank/README.md",
        imports=(),
        executables=("java",),
    ),
    "openmm": EnvSpec(
        name="openmm",
        recipe="environments/openmm/README.md",
        imports=("openmm", "numpy", "pdbfixer"),
    ),
    "boltz": EnvSpec(
        name="boltz",
        recipe="environments/boltz2/README.md",
        imports=("boltz", "torch", "PyYAML", "rdkit", "gemmi", "numpy"),
    ),
    "boltz_mps": EnvSpec(
        name="boltz_mps",
        recipe="environments/boltz2-mps/manifest.yaml",
        imports=("boltz", "torch", "PyYAML", "rdkit", "gemmi", "numpy"),
    ),
    "openff": EnvSpec(
        name="openff",
        recipe="environments/openff-complex-preparation/README.md",
        imports=(
            "openmm",
            "rdkit",
            "pdbfixer",
            "openff-toolkit",
            "openff-nagl-models",
            "openmmforcefields",
            "ParmEd",
            "numpy",
        ),
    ),
    "openfe": EnvSpec(
        name="openfe",
        recipe="environments/binding-free-energy/README.md",
        imports=(
            "openfe",
            "gufe",
            "openmm",
            "openmmtools",
            "pymbar",
            "openff-toolkit",
            "rdkit",
            "numpy",
        ),
    ),
    "pymbar": EnvSpec(
        name="pymbar",
        recipe="environments/binding-free-energy/README.md",
        imports=("pymbar", "numpy", "scipy"),
    ),
    "qmmm": EnvSpec(
        name="qmmm",
        recipe="environments/qmmm/README.md",
        imports=("pyscf", "openmm", "numpy"),
    ),
    "molecular_properties": EnvSpec(
        name="molecular_properties",
        recipe="environments/molecular-properties/README.md",
        imports=("pyscf", "rdkit", "pyberny", "numpy"),
    ),
    "thermostability": EnvSpec(
        name="thermostability",
        recipe="environments/thermostability/README.md",
        imports=("torch", "transformers", "adapters", "numpy"),
    ),
    "admet": EnvSpec(
        name="admet",
        recipe="environments/admet/README.md",
        imports=("admet-ai", "chemprop", "torch", "rdkit", "numpy"),
    ),
}


def spec(name: str) -> EnvSpec:
    try:
        return ENVIRONMENTS[name]
    except KeyError:
        known = ", ".join(sorted(ENVIRONMENTS))
        raise KeyError(
            f"No environment {name!r}; declared environments: {known}"
        ) from None


def run_in_environment(
    environment: str,
    entry: str,
    request: dict[str, Any],
    *,
    timeout: float = 3600,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """Run `module:function` inside an engine environment.

    The request is written as JSON, the module writes its result as JSON, and
    nothing else crosses the boundary. The runner is executed by path and loads
    the module by file, so the engine environment does not need Alambiq
    installed. When the module fails, its output is attached to the error.
    """
    module_name, separator, function_name = entry.partition(":")
    if not module_name or not separator or not function_name:
        raise ValueError(f"entry {entry!r} must be module:function")
    environment_spec = spec(environment)
    interpreter = environment_spec.interpreter()
    with tempfile.TemporaryDirectory(dir=workdir) as directory:
        place = Path(directory)
        (place / "request.json").write_text(json.dumps(request))
        finished = subprocess.run(
            [
                str(interpreter),
                "-I",
                str(RUNNER),
                module_name,
                function_name,
                str(place / "request.json"),
                str(place / "result.json"),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PATH": environment_spec.search_path()},
        )
        if finished.returncode != 0:
            refusal = place / "result.json.error"
            if refusal.is_file() and not os.environ.get("ALAMBIQ_DEBUG"):
                # The engine said why in a sentence; that is the whole report.
                raise RuntimeError(
                    f"{json.loads(refusal.read_text())['message']} "
                    f"(engine environment {environment}, set by "
                    f"{environment_spec.variable})"
                )
            raise RuntimeError(
                f"{entry} failed in the {environment} environment "
                f"(exit {finished.returncode}).\n{finished.stdout}\n{finished.stderr}"
            )
        output = place / "result.json"
        if not output.is_file():
            raise RuntimeError(f"{entry} wrote no result.\n{finished.stdout}")
        return json.loads(output.read_text())
