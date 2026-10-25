"""Fetch the Boltz-2 model files with Boltz's own downloader.

Runs in the Boltz environment. Boltz stores its weights and its molecule library in one
cache directory and downloads what is missing when a prediction starts. A Alambiq run
downloads nothing, so Alambiq calls the same function once, on request, into a
staging directory; the caller then checks every file against the pinned record before the
directory is put in place. Imports the standard library and Boltz only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

EXPECTED = ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols.tar", "mols")


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Download and unpack the model files into `request["target"]`."""
    from boltz.main import download_boltz2

    target = Path(request["target"])
    target.mkdir(parents=True, exist_ok=True)
    download_boltz2(target)
    missing = [name for name in EXPECTED if not (target / name).exists()]
    if missing:
        raise RuntimeError(
            "The Boltz downloader did not write these files: " + ", ".join(missing)
        )
    return {"fetched": sorted(p.name for p in target.iterdir())}
