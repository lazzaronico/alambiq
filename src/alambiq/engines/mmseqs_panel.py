"""Align every query against every target of a small panel with MMseqs2.

Runs inside the mmseqs environment. Imports the standard library only. The
search is exhaustive and keeps the alignment of each pair, so nothing a pair
shares is missed for want of a prefilter hit; it is meant for panels of up to a
few thousand sequences.
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
except ImportError:  # imported as part of the package, not by the runner
    from alambiq.engines.hints import exec_hint

COLUMNS = (
    "query",
    "target",
    "evalue",
    "bits",
    "fident",
    "alnlen",
    "qstart",
    "tstart",
    "qcov",
    "tcov",
    "qaln",
    "taln",
)


def _count(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.startswith(">"))


def _version(executable: str) -> str:
    try:
        finished = subprocess.run(
            [executable, "version"], capture_output=True, text=True, timeout=60
        )
        return (finished.stdout or finished.stderr).strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - reported as unknown
        return "unknown"


def _mmseqs(executable: str, *arguments: str) -> None:
    finished = subprocess.run(
        [executable, *arguments], capture_output=True, text=True, timeout=3600
    )
    if finished.returncode != 0:
        detail = (finished.stderr or finished.stdout).strip()
        raise RuntimeError(
            f"mmseqs {arguments[0]} failed (exit {finished.returncode}): {detail[-2000:]}{exec_hint(detail)}"
        )


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Return every alignment of a query FASTA against a target FASTA."""
    executable = shutil.which("mmseqs")
    if executable is None:
        raise RuntimeError(
            f"mmseqs is not on PATH for {sys.executable}. "
            "Install it in that environment as described in docs/install.md"
        )
    queries = Path(request["queries"])
    targets = Path(request["targets"])
    target_count = _count(targets)

    with tempfile.TemporaryDirectory() as directory:
        place = Path(directory)
        query_db, target_db = place / "query-db", place / "target-db"
        result_db, found = place / "result-db", place / "alignments.tsv"
        _mmseqs(executable, "createdb", str(queries), str(query_db))
        _mmseqs(executable, "createdb", str(targets), str(target_db))
        _mmseqs(
            executable,
            "search",
            str(query_db),
            str(target_db),
            str(result_db),
            str(place / "tmp"),
            "--threads",
            "1",
            "-s",
            str(float(request.get("sensitivity", 7.5))),
            "-e",
            str(float(request.get("evalue", 10000.0))),
            "--max-seqs",
            str(target_count + 1),
            "--exhaustive-search",
            "1",
            "-a",
        )
        _mmseqs(
            executable,
            "convertalis",
            str(query_db),
            str(target_db),
            str(result_db),
            str(found),
            "--format-output",
            ",".join(COLUMNS),
        )
        lines = [line for line in found.read_text().splitlines() if line.strip()]

    rows = []
    for line in lines:
        values = dict(zip(COLUMNS, line.split("\t"), strict=True))
        rows.append(
            {
                "query": values["query"],
                "target": values["target"],
                "evalue": float(values["evalue"]),
                "bits": float(values["bits"]),
                "identity": float(values["fident"]),
                "alignment_length": int(values["alnlen"]),
                "query_start": int(values["qstart"]),
                "target_start": int(values["tstart"]),
                "query_coverage": float(values["qcov"]),
                "target_coverage": float(values["tcov"]),
                "query_alignment": values["qaln"],
                "target_alignment": values["taln"],
            }
        )
    return {"alignments": rows, "versions": {"mmseqs": _version(executable)}}


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    import sys

    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
