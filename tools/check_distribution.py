"""Build the wheel and the source distribution and inspect what they contain.

    python tools/check_distribution.py [--keep DIR]

The checks are the ones a release depends on:

* the wheel holds the package, its license and its notices; files from the tests, the
  examples, the documentation, planning material and local run directories are rejected;
* every file in either distribution is within the size limit, which excludes data and
  weights; the offline viewer bundle is the one larger file allowed in the wheel;
* the source distribution holds the examples, tests, documentation, environment recipes,
  tools, license and notices;
* files that look like a secret, a run record or a cache are rejected from both;
* the wheel installs into a new environment outside the checkout and reports the
  capabilities.

Exit status 1 means a check failed; each failure is printed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 2_000_000
ALLOWED_LARGE = {"alambiq/report/assets/3Dmol-min.js"}
FORBIDDEN_IN_WHEEL = ("tests/", "examples/", "docs/", "planning/", ".alambiq/", "tools/")
SUSPECT = ("run.json", ".env", "id_rsa", ".pem", "credentials", ".DS_Store", ".pyc")
REQUIRED_IN_WHEEL = (
    "alambiq/resources/THIRD_PARTY_NOTICES.md",
    "alambiq/report/assets/3Dmol-min.js",
    "alambiq/report/assets/LICENSE",
    "alambiq/report/assets/README.md",
)
REQUIRED_IN_SDIST = (
    "LICENSE",
    "LICENSING.md",
    "README.md",
    "CHANGELOG.md",
    "docs/getting-started.md",
    "examples/README.md",
    "examples/kinetics-from-measurements/campaign.yaml",
    "tests/core/test_examples.py",
    "tools/check_distribution.py",
    "environments",
)


def build(out: Path) -> tuple[Path, Path]:
    subprocess.run(
        ["uvx", "--from", "build", "pyproject-build", "--outdir", str(out), str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(out.glob("*.whl"))
    sdist = next(out.glob("*.tar.gz"))
    return wheel, sdist


def wheel_files(wheel: Path) -> dict[str, int]:
    with zipfile.ZipFile(wheel) as z:
        return {i.filename: i.file_size for i in z.infolist() if not i.is_dir()}


def sdist_files(sdist: Path) -> dict[str, int]:
    with tarfile.open(sdist) as t:
        root = t.getnames()[0].split("/")[0]
        return {
            m.name.removeprefix(root + "/"): m.size for m in t.getmembers() if m.isfile()
        }


def inspect(wheel: Path, sdist: Path) -> list[str]:
    problems = []
    inside = wheel_files(wheel)
    for name in REQUIRED_IN_WHEEL:
        if name not in inside:
            problems.append(f"wheel lacks {name}")
    if not any(n.endswith(".dist-info/licenses/LICENSE") for n in inside):
        problems.append("wheel lacks LICENSE in its metadata")
    for name in inside:
        if name.startswith(FORBIDDEN_IN_WHEEL):
            problems.append(f"wheel holds {name}")
    source = sdist_files(sdist)
    for name in REQUIRED_IN_SDIST:
        if not any(n == name or n.startswith(name + "/") for n in source):
            problems.append(f"sdist lacks {name}")
    for label, files in (("wheel", inside), ("sdist", source)):
        for name, size in files.items():
            if size > LIMIT and name not in ALLOWED_LARGE:
                problems.append(f"{label} holds {name} of {size:,} bytes")
            if any(s in name for s in SUSPECT):
                problems.append(f"{label} holds {name}")
            if name.startswith((".alambiq/", "build/", "dist/")):
                problems.append(f"{label} holds {name}")
    return problems


def install_and_run(wheel: Path, scratch: Path) -> list[str]:
    env = scratch / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    python = env / "bin" / "python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", str(wheel)],
        check=True,
        cwd=scratch,
    )
    out = subprocess.run(
        [str(env / "bin" / "alambiq"), "capability", "list"],
        capture_output=True,
        text=True,
        cwd=scratch,
    )
    lines = [line for line in out.stdout.splitlines() if "." in line.split(" ")[0]]
    problems = []
    if out.returncode != 0 or len(lines) < 50:
        problems.append(
            f"the installed wheel lists {len(lines)} capabilities "
            f"(exit {out.returncode}): {out.stderr[-300:]}"
        )
    return problems


def main() -> int:
    keep = None
    if "--keep" in sys.argv:
        keep = Path(sys.argv[sys.argv.index("--keep") + 1])
    scratch = Path(tempfile.mkdtemp(prefix="alambiq-dist-"))
    try:
        wheel, sdist = build(scratch / "out")
        print(f"wheel {wheel.name}: {sum(wheel_files(wheel).values()):,} bytes")
        print(f"sdist {sdist.name}: {sum(sdist_files(sdist).values()):,} bytes")
        problems = inspect(wheel, sdist) + install_and_run(wheel, scratch)
        if keep:
            shutil.copytree(scratch / "out", keep, dirs_exist_ok=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for problem in problems:
        print("FAIL", problem)
    print("distribution is sound" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
