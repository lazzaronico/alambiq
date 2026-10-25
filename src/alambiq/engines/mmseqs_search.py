"""Search a sequence panel with MMseqs2.

Runs inside the mmseqs environment. Imports the standard library only.
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
    "fident",
    "alnlen",
    "evalue",
    "bits",
    "qcov",
)


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    letters: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(letters)))
            name = line[1:].split()[0] if len(line) > 1 else ""
            letters = []
        elif line.strip():
            letters.append(line.strip())
    if name is not None:
        records.append((name, "".join(letters)))
    return records


def _version(executable: str) -> str:
    try:
        finished = subprocess.run(
            [executable, "version"], capture_output=True, text=True, timeout=60
        )
        return (finished.stdout or finished.stderr).strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - reported as unknown
        return "unknown"


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Search the panel and return every hit that meets the thresholds."""
    executable = shutil.which("mmseqs")
    if executable is None:
        raise RuntimeError(
            f"mmseqs is not on PATH for {sys.executable}. "
            "Install it in that environment as described in docs/install.md"
        )

    query_path = Path(request["query"])
    panel_path = Path(request["panel"])
    query = _read_fasta(query_path)
    panel = _read_fasta(panel_path)
    if len(query) != 1:
        raise ValueError(f"The query FASTA holds {len(query)} sequences; exactly 1 is required")
    if not panel:
        raise ValueError("The panel FASTA holds no sequences; at least 1 is required")

    with tempfile.TemporaryDirectory() as directory:
        place = Path(directory)
        output = place / "hits.tsv"
        finished = subprocess.run(
            [
                executable,
                "easy-search",
                str(query_path),
                str(panel_path),
                str(output),
                str(place / "tmp"),
                "-s",
                str(request.get("sensitivity", 5.7)),
                "-e",
                str(request.get("max_evalue", 1e-3)),
                "--format-output",
                ",".join(COLUMNS),
                "--threads",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if finished.returncode != 0:
            raise RuntimeError(
                "mmseqs easy-search failed "
                f"(exit {finished.returncode}):\n{finished.stderr[-2000:]}{exec_hint(finished.stderr)}"
            )
        rows = [
            dict(zip(COLUMNS, line.split("\t"), strict=True))
            for line in output.read_text().splitlines()
            if line.strip()
        ]

    minimum_identity = float(request.get("min_identity", 0.0))
    minimum_coverage = float(request.get("min_coverage", 0.0))
    maximum_evalue = float(request.get("max_evalue", 1e-3))

    hits = []
    for row in rows:
        identity, coverage = float(row["fident"]), float(row["qcov"])
        evalue = float(row["evalue"])
        if (
            identity < minimum_identity
            or coverage < minimum_coverage
            or evalue > maximum_evalue
        ):
            continue
        hits.append(
            {
                "target": row["target"],
                "evalue": evalue,
                "identity": identity,
                "coverage": coverage,
                "bits": float(row["bits"]),
                "alignment_length": int(row["alnlen"]),
            }
        )
    hits.sort(key=lambda h: (h["evalue"], -h["bits"]))

    return {
        "query_id": query[0][0],
        "panel_size": len(panel),
        "considered": len(rows),
        "hits": hits,
        "versions": {"mmseqs": _version(executable)},
    }


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    import sys

    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
