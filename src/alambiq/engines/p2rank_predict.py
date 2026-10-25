"""Predict ligand-binding pockets on one structure with P2Rank.

Runs in the p2rank environment, which supplies Java 17 to 23 on its PATH. Imports
the standard library only. P2Rank is a Java application, started through a small
launcher (`P2RankOffline.java`) that gives BioJava its bundled reduced chemistry
so nothing is downloaded. The caller has verified the P2Rank files against their
pinned record; this module runs the program and returns its two result tables as
text, along with the versions that ran.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from hints import exec_hint
except ImportError:  # imported as part of the package, not by the runner
    from alambiq.engines.hints import exec_hint

LAUNCHER = Path(__file__).resolve().parent / "P2RankOffline.java"


def _java_version(java: str) -> tuple[str, int]:
    finished = subprocess.run(
        [java, "-version"], capture_output=True, text=True, timeout=30
    )
    output = (finished.stderr or finished.stdout).strip()
    if finished.returncode != 0:
        raise RuntimeError(output[-4000:] or "The Java version probe failed without output")
    found = re.search(r'version\s+"(\d+(?:\.\d+)*)', output)
    if found is None:
        raise RuntimeError("The Java version probe returned a response that is not a version string")
    version = found.group(1)
    major = int(version.split(".", 1)[0])
    if not 17 <= major <= 23:
        raise RuntimeError(f"P2Rank requires Java 17-23; found {version}")
    return version, major


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run P2Rank on `structure` and return the predictions and residue tables."""
    java = shutil.which("java")
    if java is None:
        raise RuntimeError(
            f"java is not on PATH for {sys.executable}. "
            "Install it in that environment as described in docs/install.md"
        )
    java_version, _ = _java_version(java)
    installation = Path(request["installation"])
    classpath = os.pathsep.join(
        (str(installation / "bin/p2rank.jar"), str(installation / "bin/lib/*"))
    )
    structure = Path(request["structure"])
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "output"
        command = [
            java,
            "-Xmx2048m",
            "-cp",
            classpath,
            str(LAUNCHER),
            "predict",
            "-f",
            str(structure),
            "-c",
            request["configuration"],
            "-threads",
            "1",
            "-visualizations",
            "0",
            "-o",
            str(output),
        ]
        clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS")
        }
        finished = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=request.get("timeout", 600),
            env=clean,
        )
        if finished.returncode != 0:
            raise RuntimeError(
                f"P2Rank exited {finished.returncode}:\n{(finished.stderr or finished.stdout)[-4000:]}{exec_hint(finished.stderr or finished.stdout)}"
            )
        combined = f"{finished.stdout}\n{finished.stderr}"
        if "DownloadChemCompProvider" in combined or "files.rcsb.org" in combined:
            raise RuntimeError(
                "P2Rank attempted to download chemical component data at run time, "
                "which is not permitted"
            )
        predictions = output / f"{structure.name}_predictions.csv"
        residues = output / f"{structure.name}_residues.csv"
        if not predictions.is_file() or not residues.is_file():
            raise RuntimeError("P2Rank did not write the prediction table or the residue table")
        return {
            "predictions_csv": predictions.read_text(encoding="utf-8"),
            "residues_csv": residues.read_text(encoding="utf-8"),
            "versions": {"p2rank": request["version"], "java": java_version},
        }
