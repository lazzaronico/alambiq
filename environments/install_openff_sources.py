"""Install pinned upstream sources into an already provisioned Conda environment.

The upstream Conda OpenMMForceFields package includes AmberTools. This explicit
source installation uses the OpenFF-only dependencies declared in our recipes.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


def install(binding=False):
    prefix = Path(sys.prefix)
    if not (prefix / "conda-meta").is_dir():
        raise ValueError("Run with the Python from the new Conda environment")
    if list((prefix / "conda-meta").glob("ambertools-*.json")):
        raise ValueError("Use the OpenFF-only dependency recipe")
    records = json.loads(Path(__file__).with_name("openff-sources.json").read_text())
    selected = records if binding else records[:1]
    with tempfile.TemporaryDirectory(prefix="alambiq-openff-") as directory:
        for record in selected:
            target = Path(directory) / f"{record['name']}-{record['version']}.tar.gz"
            with urllib.request.urlopen(record["url"], timeout=120) as response:
                target.write_bytes(response.read())
            if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Source digest differs for {record['name']}")
            env = os.environ.copy()
            # GitHub archives omit Git metadata needed by setuptools-scm.
            if record["name"] == "openfe":
                env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENFE"] = record["version"]
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "--no-cache-dir",
                    str(target),
                ],
                env=env,
                check=True,
            )
            if importlib.metadata.version(record["name"]) != record["version"]:
                raise ValueError("Installed source-release version differs")
    (prefix / "alambiq-openff-sources.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", action="store_true", help="Also install OpenFE")
    install(parser.parse_args().binding)
