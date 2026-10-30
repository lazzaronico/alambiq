"""Align variants to reference proteins with MMseqs2 and return the strongest hits.

Runs inside the mmseqs environment. Imports the standard library only. The references are
a FASTA file or a pinned MMseqs2 database. Every hit that meets the thresholds is
returned with its alignment strings, strongest first, at most `keep` per variant.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from hints import exec_hint
    from swissprot_annotations import read
except ImportError:  # imported as part of the package, not by the runner
    from alambiq.engines.hints import exec_hint
    from alambiq.engines.swissprot_annotations import read

CANDIDATES = 2000
"""The most candidate references per variant that pass the MMseqs2 prefilter."""
KEPT = 20
"""The most hits per variant returned, strongest first."""

COLUMNS = (
    "query",
    "target",
    "fident",
    "evalue",
    "bits",
    "qcov",
    "tcov",
    "alnlen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "qlen",
    "tlen",
    "qaln",
    "taln",
)


def _version(executable: str) -> str:
    try:
        finished = subprocess.run(
            [executable, "version"], capture_output=True, text=True, timeout=60
        )
        return (finished.stdout or finished.stderr).strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - reported as unknown
        return "unknown"


def _mmseqs(executable: str, place: Path, timeout: int, *arguments: str) -> None:
    finished = subprocess.run(
        [executable, *arguments],
        capture_output=True,
        text=True,
        cwd=place,
        timeout=timeout,
    )
    if finished.returncode != 0:
        detail = (finished.stderr or finished.stdout).strip()
        raise RuntimeError(
            f"mmseqs {arguments[0]} failed (exit {finished.returncode}): "
            f"{detail[-2000:]}{exec_hint(detail)}"
        )


def _accession(target: str, database: bool) -> str:
    """The accession of a UniProt FASTA identifier such as `sp|P12345|NAME_HUMAN`."""
    parts = target.split("|")
    return parts[1] if database and len(parts) == 3 else target


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Search the references with the variants and return the hits that meet the thresholds."""
    executable = shutil.which("mmseqs")
    if executable is None:
        raise RuntimeError(
            f"mmseqs is not on PATH for {sys.executable}. "
            "Install it in that environment as described in docs/install.md"
        )
    timeout = int(request.get("timeout", 7200))
    pinned = request.get("database") is not None
    with tempfile.TemporaryDirectory() as directory:
        place = Path(directory)
        _mmseqs(executable, place, timeout, "createdb", request["variants"], "qdb")
        if pinned:
            database = request["database"]
        else:
            _mmseqs(
                executable, place, timeout, "createdb", request["references"], "rdb"
            )
            database = "rdb"
        _mmseqs(
            executable, place, timeout, "search", "qdb", database, "res", "tmp",
            "--threads", str(int(request["threads"])),
            "-s", str(float(request["sensitivity"])),
            "-e", str(float(request["max_evalue"])),
            "--max-seqs", str(CANDIDATES),
            "-a",
        )  # fmt: skip
        _mmseqs(
            executable, place, timeout, "convertalis", "qdb", database, "res", "hits.tsv",
            "--format-output", ",".join(COLUMNS),
        )  # fmt: skip
        rows = [
            dict(zip(COLUMNS, line.split("\t"), strict=True))
            for line in (place / "hits.tsv").read_text().splitlines()
            if line.strip()
        ]

    minimum_identity = float(request["min_identity"])
    minimum_coverage = float(request["min_coverage"])
    maximum_evalue = float(request["max_evalue"])
    keep = int(request.get("keep", KEPT))
    hits: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if (
            float(row["fident"]) < minimum_identity
            or float(row["qcov"]) < minimum_coverage
            or float(row["evalue"]) > maximum_evalue
        ):
            continue
        hits.setdefault(row["query"], []).append(
            {
                "target": _accession(row["target"], pinned),
                "identity": float(row["fident"]),
                "coverage": float(row["qcov"]),
                "target_coverage": float(row["tcov"]),
                "evalue": float(row["evalue"]),
                "bits": float(row["bits"]),
                "alignment_length": int(row["alnlen"]),
                "variant_start": int(row["qstart"]),
                "variant_end": int(row["qend"]),
                "variant_length": int(row["qlen"]),
                "reference_start": int(row["tstart"]),
                "reference_end": int(row["tend"]),
                "reference_length": int(row["tlen"]),
                "variant_aligned": row["qaln"],
                "reference_aligned": row["taln"],
            }
        )
    for found in hits.values():
        found.sort(key=lambda h: (-h["bits"], h["target"]))
        del found[keep:]
    names: dict[str, dict[str, Any]] = {}
    if pinned:
        wanted = {h["target"] for found in hits.values() for h in found}
        names = read(Path(request["annotations"]), wanted)
        absent = sorted(wanted - set(names))
        if absent:
            raise RuntimeError(
                f"{len(absent)} hit accessions are absent from the annotation table, "
                f"for example {absent[0]}; the database and the table are not from one build"
            )
    for found in hits.values():
        for hit in found:
            hit["name"] = str(names.get(hit["target"], {}).get("name", ""))
    return {"hits": hits, "versions": {"mmseqs": _version(executable)}}


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
