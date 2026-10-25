"""Search one protein against a library of sequence profiles with MMseqs2.

Runs inside the mmseqs environment. Imports the standard library only. The
profiles are either a prebuilt MMseqs2 profile database or a Stockholm file of
alignments, built into profiles here with the construction the library uses.
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

FIELDS = "query,target,evalue,bits,qstart,qend,tstart,tend,qlen,tlen"
CONSTRUCTION = {"match_mode": 1, "match_ratio": 0.5, "filter_msa": 0, "msa_type": 2}


def _version(executable: str) -> str:
    try:
        finished = subprocess.run(
            [executable, "version"], capture_output=True, text=True, timeout=60
        )
        return (finished.stdout or finished.stderr).strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - reported as unknown
        return "unknown"


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
            f"mmseqs {arguments[0]} failed (exit {finished.returncode}): {detail[-2000:]}{exec_hint(detail)}"
        )


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Return every profile hit for the query, unfiltered by coverage."""
    executable = shutil.which("mmseqs")
    if executable is None:
        raise RuntimeError(
            f"mmseqs is not on PATH for {sys.executable}. "
            "Install it in that environment as described in docs/install.md"
        )
    threads = str(int(request.get("threads", 4)))

    with tempfile.TemporaryDirectory() as directory:
        place = Path(directory)
        _mmseqs(executable, place, "createdb", request["query"], "qdb", "--dbtype", "1")
        if request.get("stockholm"):
            _mmseqs(
                executable,
                place,
                "convertmsa",
                request["stockholm"],
                "msa",
                "--identifier-field",
                "1",
            )
            _mmseqs(
                executable, place, "msa2profile", "msa", "profiles",
                "--msa-type", str(CONSTRUCTION["msa_type"]),
                "--match-mode", str(CONSTRUCTION["match_mode"]),
                "--match-ratio", str(CONSTRUCTION["match_ratio"]),
                "--filter-msa", str(CONSTRUCTION["filter_msa"]),
                "--threads", threads,
            )  # fmt: skip
            target = "profiles"
        else:
            target = request["database"]
        _mmseqs(
            executable, place, "search", "qdb", target, "result", "tmp",
            "-k", "5",
            "-s", str(float(request.get("sensitivity", 7.5))),
            "-e", str(float(request.get("evalue", 0.001))),
            "--alt-ali", str(int(request.get("alternative_alignments", 0))),
            "--max-seqs", str(int(request["profile_count"])),
            "--threads", threads,
        )  # fmt: skip
        _mmseqs(
            executable, place, "convertalis", "qdb", target, "result", "hits.tsv",
            "--format-output", FIELDS, "--threads", threads,
        )  # fmt: skip
        hits = [
            line.split("\t")
            for line in (place / "hits.tsv").read_text().splitlines()
            if line.strip()
        ]
    return {
        "hits": hits,
        "versions": {"mmseqs": _version(executable)},
        "construction": CONSTRUCTION,
    }


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    import sys

    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
