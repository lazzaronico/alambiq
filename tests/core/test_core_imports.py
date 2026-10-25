"""The core lists every capability without any optional package installed.

A researcher who has installed only the core must be able to run `alambiq capability list`
and read what each capability needs. A capability module that imports NumPy, SciPy or a
chemistry package at its top level would break the listing for everyone, so those imports
belong inside the functions that use them. The check runs the listing in a process in
which those packages cannot be imported.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
BLOCKED = [
    "numpy",
    "scipy",
    "sklearn",
    "rdkit",
    "gemmi",
    "mdtraj",
    "Bio",
    "pydna",
    "dnachisel",
    "openmm",
    "pdbfixer",
    "umap",
    "doranet",
    "equilibrator_api",
    "pandas",
    "matplotlib",
]


def test_the_listing_needs_no_optional_package():
    program = textwrap.dedent(
        f"""
        import sys
        blocked = {BLOCKED!r}
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked: " + name)
        sys.meta_path.insert(0, Blocker())
        sys.path.insert(0, {str(SRC)!r})
        from alambiq import api
        from alambiq.core import registry
        api._ready()
        ids = [c.id for c in registry.registered()]
        assert len(ids) >= 50, ids
        for c in registry.registered():
            assert c.doc and c.request_type and c.result_type
        print(len(ids))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr[-1500:]
    assert int(done.stdout.strip().splitlines()[-1]) >= 50


def test_the_workflow_runtime_keeps_its_state_apart_from_other_prefect_installations():
    program = (
        f"import os, sys; sys.path.insert(0, {str(SRC)!r}); import alambiq; "
        "print(os.environ['PREFECT_HOME'])"
    )

    def home(**environment):
        import os

        env = {k: v for k, v in os.environ.items() if not k.endswith("PREFECT_HOME")}
        env.update(environment)
        done = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, env=env
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    from importlib import metadata

    assert home().endswith(f"/.alambiq/prefect-{metadata.version('prefect')}")
    assert home(ALAMBIQ_PREFECT_HOME="/somewhere/else") == "/somewhere/else"
    assert home(PREFECT_HOME="/explicit", ALAMBIQ_PREFECT_HOME="/x") == "/explicit"


def test_the_prefect_home_is_created_with_its_parents(tmp_path):
    home = tmp_path / "a" / "b" / "prefect"
    program = f"import sys; sys.path.insert(0, {str(SRC)!r}); import alambiq"
    import os

    env = {k: v for k, v in os.environ.items() if not k.endswith("PREFECT_HOME")}
    env["PREFECT_HOME"] = str(home)
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env
    )
    assert done.returncode == 0, done.stderr
    assert home.is_dir()
