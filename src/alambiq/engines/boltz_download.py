"""Fetch the Boltz-2 model files from the pinned revision and unpack the molecule library.

Runs in the Boltz environment. The record of the resource names the three files with their
addresses at one revision of the Hugging Face repository, their sizes and their SHA-256
digests. Each file is downloaded over HTTPS and refused unless it matches the record. Boltz's
own function then runs once, on a directory that already holds all three files: it downloads
nothing and unpacks `mols.tar` into `mols/`, as Boltz does for its own cache. The caller then
checks every file of the staging directory against the pinned record before the directory is put
in place. Imports the standard library and Boltz only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

CHUNK = 1 << 20
DOWNLOADED = ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols.tar")
EXPECTED = (*DOWNLOADED, "mols")


def download(
    url: str, destination: Path, size: int, sha256: str, opener=urlopen
) -> None:
    """Download `url` to `destination`, refusing a size or SHA-256 digest that differs."""
    if not url.startswith("https://"):
        raise RuntimeError(f"{url} is not an HTTPS address")
    digest = hashlib.sha256()
    count = 0
    request = Request(url, headers={"User-Agent": "Alambiq resource provisioning"})
    with opener(request, timeout=60) as response, destination.open("wb") as out:
        if not response.geturl().startswith("https://"):
            raise RuntimeError(f"{url} was redirected to an address that is not HTTPS")
        while chunk := response.read(CHUNK):
            count += len(chunk)
            if count > size:
                raise RuntimeError(f"{url} is larger than the recorded {size} bytes")
            digest.update(chunk)
            out.write(chunk)
    if count != size or digest.hexdigest() != sha256:
        raise RuntimeError(
            f"{destination.name} downloaded from {url} does not match its record "
            f"(size {count} of {size} bytes, SHA-256 {digest.hexdigest()} against {sha256}). "
            "The record pins one revision, and the file at this address is not the pinned one"
        )


def run(request: dict[str, Any], opener=urlopen) -> dict[str, Any]:
    """Download the pinned files into `request["target"]` and unpack the molecule library."""
    from boltz.main import download_boltz2

    sources = {s["name"]: s for s in request.get("sources", [])}
    missing = [name for name in DOWNLOADED if name not in sources]
    if missing:
        raise RuntimeError(
            "the record lists no source address for " + ", ".join(missing)
        )
    target = Path(request["target"])
    target.mkdir(parents=True, exist_ok=True)
    for name in DOWNLOADED:
        source = sources[name]
        download(
            source["url"], target / name, source["bytes"], source["sha256"], opener
        )
    download_boltz2(target)  # every file is present, so it only unpacks mols.tar
    missing = [name for name in EXPECTED if not (target / name).exists()]
    if missing:
        raise RuntimeError(
            "The Boltz downloader did not write these files: " + ", ".join(missing)
        )
    return {"fetched": sorted(p.name for p in target.iterdir())}
