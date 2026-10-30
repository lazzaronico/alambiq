"""Fetch the UniProtKB/Swiss-Prot source files and build the resource from them.

Runs inside the mmseqs environment. Imports the standard library and its sibling module
`swissprot_annotations` only. `alambiq resource provision` calls it once, on request, with a
staging directory; the caller checks every file of the staging directory against the pinned
record before the directory is put in place.

Each source file is downloaded over HTTPS and checked against the recorded size and MD5
digest. The FASTA file becomes an MMseqs2 database with its index, and the flat file becomes
the annotation table. Downloads and intermediate files are removed, so the target holds the
files of the record and nothing else.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

try:
    from hints import exec_hint
    from swissprot_annotations import build as build_table
except ImportError:  # imported as part of the package, not by the runner
    from alambiq.engines.hints import exec_hint
    from alambiq.engines.swissprot_annotations import build as build_table

CHUNK = 1 << 20
FASTA = "uniprot_sprot.fasta.gz"
FLAT_FILE = "uniprot_sprot.dat.gz"


def download(url: str, destination: Path, size: int, md5: str, opener=urlopen) -> None:
    """Download `url` to `destination`, refusing a size or MD5 digest that differs."""
    if not url.startswith("https://"):
        raise RuntimeError(f"{url} is not an HTTPS address")
    digest = hashlib.md5()  # noqa: S324 - UniProt publishes MD5 digests of its files
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
    if count != size or digest.hexdigest() != md5:
        raise RuntimeError(
            f"{destination.name} downloaded from {url} does not match its record "
            f"(size {count} of {size} bytes, MD5 {digest.hexdigest()} against {md5}). "
            "UniProt replaces the files at `current_release` when it publishes a new "
            "release, and this record pins one release"
        )


def _mmseqs(executable: str, place: Path, *arguments: str) -> None:
    finished = subprocess.run(
        [executable, *arguments],
        capture_output=True,
        text=True,
        cwd=place,
        timeout=7200,
    )
    if finished.returncode != 0:
        detail = (finished.stderr or finished.stdout).strip()
        raise RuntimeError(
            f"mmseqs {arguments[0]} failed (exit {finished.returncode}): "
            f"{detail[-2000:]}{exec_hint(detail)}"
        )


def run(request: dict[str, Any], opener=urlopen) -> dict[str, Any]:
    """Fetch the sources named in the request and build the resource in `target`."""
    executable = shutil.which("mmseqs")
    if executable is None:
        raise RuntimeError(
            f"mmseqs is not on PATH for {sys.executable}. "
            "Install it in that environment as described in docs/install.md"
        )
    target = Path(request["target"])
    sources = {s["name"]: s for s in request["sources"]}
    missing = [n for n in (FASTA, FLAT_FILE) if n not in sources]
    if missing:
        raise RuntimeError("the record lists no source file " + ", ".join(missing))
    threads = int(request.get("threads") or min(8, os.cpu_count() or 1))
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".work-") as directory:
        work = Path(directory)
        for source in sources.values():
            download(
                source["url"],
                work / source["name"],
                source["bytes"],
                source["md5"],
                opener,
            )
        fasta = work / "uniprot_sprot.fasta"
        with gzip.open(work / FASTA, "rb") as compressed, fasta.open("wb") as out:
            shutil.copyfileobj(compressed, out, CHUNK)
        (work / FASTA).unlink()
        _mmseqs(executable, work, "createdb", str(fasta), str(target / "swissprot-db"))
        (work / "tmp").mkdir()
        _mmseqs(
            executable, work, "createindex", str(target / "swissprot-db"), "tmp",
            "--threads", str(threads),
        )  # fmt: skip
        entries = build_table(work / FLAT_FILE, target / "annotations.tsv")
    return {"entries": entries, "files": sorted(p.name for p in target.iterdir())}


if __name__ == "__main__":  # pragma: no cover - exercised through the runner
    import json

    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
