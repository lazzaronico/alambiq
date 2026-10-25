"""Run the commands of docs/getting-started.md in a new environment and check what they say.

    python tools/check_getting_started.py

The repository is copied to a scratch directory (without Git history or local run data),
and the shell blocks of the guide's install and run sections are executed there, in order,
by a shell, exactly as written. The Docker section is left out: it needs a Docker daemon.
Afterwards the checks are the ones the guide promises: the version prints, the capability
list has the stated number of capabilities, the first run writes `report.html` as one
self-contained file, and the second run of the same command reports every step as reused.
The time each block took is printed, which is the guide's "about five minutes".

Exit status 1 means a command failed or a promise was not kept.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/getting-started.md"
SKIP = shutil.ignore_patterns(
    ".git",
    ".alambiq",
    ".venv",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
)


def blocks() -> list[str]:
    text = GUIDE.read_text()
    text = text.split("## Docker")[0].split("## Other measurements")[0]
    return re.findall(r"```bash\n(.*?)```", text, re.DOTALL)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="alambiq-start-"))
    try:
        shutil.copytree(ROOT, work / "repo", ignore=SKIP)
        repo = work / "repo"
        outputs, problems = [], []
        for block in blocks():
            started = time.monotonic()
            done = subprocess.run(
                ["bash", "-ec", block], cwd=repo, capture_output=True, text=True
            )
            seconds = time.monotonic() - started
            first = block.strip().splitlines()[0][:70]
            print(f"{seconds:6.1f} s  exit {done.returncode}  {first}")
            outputs.append(done.stdout + done.stderr)
            if done.returncode != 0:
                problems.append(f"failed: {first}\n{done.stderr[-800:]}")
        joined = "\n".join(outputs)
        report = repo / "report.html"
        if not report.is_file():
            problems.append("the first run wrote no report.html")
        elif re.search(r"(src|href)=[\"']https?://", report.read_text()):
            problems.append("report.html loads something from the network")
        if "0.1.0" not in joined:
            problems.append("--version did not print a version")
        listed = [
            ln for ln in joined.splitlines() if re.match(r"^[a-z]+\.[a-z_]+\s", ln)
        ]
        stated = re.search(r"has (\d+) capabilities", GUIDE.read_text())
        if stated and len(listed) != int(stated.group(1)):
            problems.append(
                f"the guide says {stated.group(1)} capabilities, the list has {len(listed)}"
            )
        if joined.count("reused") < 2:
            problems.append("the second run did not report both steps as reused")
        for problem in problems:
            print("FAIL", problem)
        print("the guide works" if not problems else f"{len(problems)} problems")
        return 1 if problems else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
