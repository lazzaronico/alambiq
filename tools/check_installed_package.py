"""Run the repository tests against an installed wheel and record skips and import origins."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_imports(wheel):
    import alambiq

    package = Path(alambiq.__file__).resolve().parent
    if package.is_relative_to(ROOT / "src"):
        raise RuntimeError("Install the wheel: source/editable imports are active")
    distribution = importlib.metadata.distribution("alambiq")
    if distribution.locate_file("alambiq").resolve() != package:
        raise RuntimeError("Imported package and installed distribution disagree")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("alambiq/") and not name.endswith("/"):
                relative = Path(name).relative_to("alambiq")
                if not (package / relative).is_file():
                    raise RuntimeError(f"Installed wheel file missing: {relative}")
                if (package / relative).read_bytes() != archive.read(name):
                    raise RuntimeError(f"Installed wheel bytes changed: {relative}")
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == "alambiq" or name.startswith("alambiq."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(package):
                    raise RuntimeError(f"Source fallback: {name}")
                origins[name] = str(path.relative_to(package))
    return origins


class Recorder:
    """A pytest plugin that counts the tests that ran and keeps the skip reasons."""

    def __init__(self):
        self.run = 0
        self.skipped = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.skipped):
            self.run += 1
        if report.skipped:
            reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else ""
            self.skipped.append({"test": report.nodeid, "reason": str(reason)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--require-extras", action="store_true")
    args = parser.parse_args()
    verify_imports(args.wheel)
    required = [
        "numpy",
        "scipy",
        "sklearn",
        "rdkit",
        "doranet",
        "equilibrator_api",
        "gemmi",
        "Bio",
        "pydna",
        "dnachisel",
        "mdtraj",
        "openmm",
        "pdbfixer",
        "umap",
    ]
    if args.require_extras:
        for name in required:
            importlib.import_module(name)
    # Test modules import their siblings by package name; package origins stay verified.
    sys.path.insert(0, str(ROOT))
    import pytest

    outcomes = Recorder()
    status = pytest.main(
        [str(ROOT / "tests/core"), "-q", "-p", "no:cacheprovider"], plugins=[outcomes]
    )
    origins = verify_imports(args.wheel)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(
            {
                "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                "tests_run": outcomes.run,
                "successful": status == 0,
                "skipped": outcomes.skipped,
                "required_extras_imported": required if args.require_extras else [],
                "loaded_module_origins": origins,
            },
            indent=2,
        )
        + "\n"
    )
    return int(status != 0)


if __name__ == "__main__":
    raise SystemExit(main())
