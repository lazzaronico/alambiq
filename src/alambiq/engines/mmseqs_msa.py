"""Build an A3M alignment for one protein from an MMseqs2 sequence database.

Runs inside the mmseqs environment. Imports the standard library only. The
search is iterative, and the hits are written as an A3M with the query first.
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
        timeout=28800,
    )
    if finished.returncode != 0:
        detail = (finished.stderr or finished.stdout).strip()
        raise RuntimeError(
            f"mmseqs {arguments[0]} failed (exit {finished.returncode}): {detail[-2000:]}{exec_hint(detail)}"
        )


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Search the database with the query and write the alignment to `output`."""
    executable = shutil.which("mmseqs")
    if executable is None:
        raise RuntimeError(
            f"mmseqs is not on PATH for {sys.executable}. "
            "Install it in that environment as described in docs/install.md"
        )
    with tempfile.TemporaryDirectory() as directory:
        place = Path(directory)
        _mmseqs(executable, place, "createdb", request["query"], "qdb")
        _mmseqs(
            executable, place, "search", "qdb", request["database"], "res", "tmp",
            "--threads", str(int(request["threads"])),
            "-s", str(float(request["sensitivity"])),
            "-e", str(float(request["evalue"])),
            "--num-iterations", str(int(request["iterations"])),
            "-c", str(float(request["minimum_query_coverage"])),
            "--cov-mode", "0",
            "--max-seqs", str(int(request["maximum_sequences"])),
            "-a",
        )  # fmt: skip
        _mmseqs(
            executable, place, "result2msa", "qdb", request["database"], "res", "msa",
            "--msa-format-mode", "5",
        )  # fmt: skip
        (place / "unpacked").mkdir()
        _mmseqs(
            executable, place, "unpackdb", "msa", "unpacked",
            "--unpack-name-mode", "0", "--unpack-suffix", ".a3m",
        )  # fmt: skip
        found = sorted((place / "unpacked").glob("*.a3m"))
        if len(found) != 1:
            raise RuntimeError(
                f"MMseqs2 wrote {len(found)} A3M files; exactly 1 per query protein is required"
            )
        shutil.copyfile(found[0], request["output"])
    return {"versions": {"mmseqs": _version(executable)}}


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    import sys

    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
