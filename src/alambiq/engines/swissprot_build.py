"""Fetch the UniProtKB/Swiss-Prot source files and build the resource from them.

Runs inside the mmseqs environment. Imports the standard library and its sibling module
`swissprot_annotations` only. `alambiq resource provision` calls it once, on request, with a
staging directory; the caller checks every file of the staging directory against the pinned
record before the directory is put in place.

Each source is downloaded over HTTPS and checked against the recorded size and MD5 digest.
A source that is an archive lists its `members`; each member is extracted by file name and
checked against its own recorded size and MD5 digest, so a release archive is accepted only if
it holds the files that the record was built from. The FASTA file becomes an MMseqs2 database
with its index, and the flat file becomes the annotation table. Downloads and intermediate
files are removed, so the target holds the files of the record and nothing else.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
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
            "The record pins one release, and the file at this address is not it"
        )


def extract_members(
    archive: Path, members: list[dict[str, Any]], destination: Path
) -> None:
    """Extract the listed members of a tar archive by file name, each checked against its record.

    A member is found by the last component of its path inside the archive. Nothing else is
    extracted, and a member whose size or MD5 digest differs from the record is refused.
    """
    wanted = {member["name"]: member for member in members}
    found: set[str] = set()
    seen: list[str] = []
    with tarfile.open(archive, "r:*") as tar:
        for info in tar:
            if not info.isfile():
                continue
            name = Path(info.name).name
            seen.append(name)
            if name not in wanted or name in found:
                continue
            member = wanted[name]
            digest = hashlib.md5()  # noqa: S324 - UniProt publishes MD5 digests of its files
            count = 0
            source = tar.extractfile(info)
            assert source is not None
            with source, (destination / name).open("wb") as out:
                while chunk := source.read(CHUNK):
                    count += len(chunk)
                    digest.update(chunk)
                    out.write(chunk)
            if count != member["bytes"] or digest.hexdigest() != member["md5"]:
                raise RuntimeError(
                    f"{name} in {archive.name} does not match its record "
                    f"(size {count} of {member['bytes']} bytes, MD5 {digest.hexdigest()} "
                    f"against {member['md5']}). The archive does not hold the file that the "
                    "record was built from"
                )
            found.add(name)
    missing = sorted(set(wanted) - found)
    if missing:
        raise RuntimeError(
            f"{archive.name} holds no member named {', '.join(missing)}; "
            f"it holds {', '.join(seen[:20]) or 'no files'}"
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
    provided = {
        name
        for source in sources.values()
        for name in (
            [m["name"] for m in source["members"]]
            if "members" in source
            else [source["name"]]
        )
    }
    missing = [n for n in (FASTA, FLAT_FILE) if n not in provided]
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
            if "members" in source:
                extract_members(work / source["name"], source["members"], work)
                (work / source["name"]).unlink()
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
