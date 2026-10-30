"""Search a pinned reviewed-protein database with MMseqs2 and join the annotation table.

Runs inside the mmseqs environment. Imports the standard library only. Every hit that meets
the thresholds is returned with the annotations of its target.
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

COLUMNS = ("query", "target", "fident", "evalue", "bits", "qcov", "tcov")


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


def _accession(target: str) -> str:
    """The accession of a UniProt FASTA identifier such as `sp|P12345|NAME_HUMAN`."""
    parts = target.split("|")
    return parts[1] if len(parts) == 3 else target


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Search the database with the proteins and return the hits that meet the thresholds."""
    executable = shutil.which("mmseqs")
    if executable is None:
        raise RuntimeError(
            f"mmseqs is not on PATH for {sys.executable}. "
            "Install it in that environment as described in docs/install.md"
        )
    timeout = int(request.get("timeout", 7200))
    with tempfile.TemporaryDirectory() as directory:
        place = Path(directory)
        _mmseqs(executable, place, timeout, "createdb", request["proteins"], "qdb")
        _mmseqs(
            executable, place, timeout, "search", "qdb", request["database"], "res", "tmp",
            "--threads", str(int(request["threads"])),
            "-s", str(float(request["sensitivity"])),
            "-e", str(float(request["max_evalue"])),
            "-a",
        )  # fmt: skip
        _mmseqs(
            executable, place, timeout, "convertalis", "qdb", request["database"], "res",
            "hits.tsv", "--format-output", ",".join(COLUMNS),
        )  # fmt: skip
        rows = [
            dict(zip(COLUMNS, line.split("\t"), strict=True))
            for line in (place / "hits.tsv").read_text().splitlines()
            if line.strip()
        ]

    minimum_identity = float(request["min_identity"])
    minimum_coverage = float(request["min_coverage"])
    maximum_evalue = float(request["max_evalue"])
    hits = [
        {
            "query": row["query"],
            "target": _accession(row["target"]),
            "identity": float(row["fident"]),
            "coverage": float(row["qcov"]),
            "target_coverage": float(row["tcov"]),
            "evalue": float(row["evalue"]),
            "bits": float(row["bits"]),
        }
        for row in rows
        if float(row["fident"]) >= minimum_identity
        and float(row["qcov"]) >= minimum_coverage
        and float(row["evalue"]) <= maximum_evalue
    ]
    table = read(Path(request["annotations"]), {hit["target"] for hit in hits})
    missing = sorted({hit["target"] for hit in hits} - set(table))
    if missing:
        raise RuntimeError(
            f"{len(missing)} hit accessions are absent from the annotation table, "
            f"for example {missing[0]}; the database and the table are not from one build"
        )
    for hit in hits:
        hit["annotations"] = table[hit["target"]]
    return {"hits": hits, "versions": {"mmseqs": _version(executable)}}


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
