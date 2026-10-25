"""Rights are stated for everything the wheel names, and no refused licence is installed.

* every pinned resource record states a licence from the accepted set and gives a size and a
  64-character SHA-256 for each file it names;
* the third-party notices name every resource and the bundled viewer, and the viewer's own
  licence files exist;
* nothing in the installed dependency closure is under a licence `LICENSING.md` refuses, and
  every package's licence can be read.

The last check reads the packages installed where the tests run, so it covers whatever
extras that environment has.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "src/alambiq/resources"
ACCEPTED = {"MIT", "MIT-with-third-party-notices", "CC0-1.0", "CC-BY-4.0", "Apache-2.0"}


def records():
    return sorted(RESOURCES.glob("*.json"))


@pytest.mark.parametrize("path", records(), ids=lambda p: p.stem)
def test_resource_record_states_licence_and_digests(path):
    record = json.loads(path.read_text())
    assert record["license"] in ACCEPTED, record["license"]
    assert record["files"]
    for entry in record["files"]:
        assert isinstance(entry["bytes"], int) and entry["bytes"] >= 0, entry["path"]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["path"]


def test_notices_name_every_resource_and_the_viewer():
    notices = (RESOURCES / "THIRD_PARTY_NOTICES.md").read_text()
    for path in records():
        assert f"`{path.name}`" in notices, path.name
    assert "3Dmol.js" in notices
    assets = ROOT / "src/alambiq/report/assets"
    for name in ("LICENSE", "README.md", "3Dmol-min.js.LICENSE.txt"):
        assert (assets / name).is_file(), name


def test_the_wheel_ships_no_data_the_notices_do_not_cover():
    shipped = {
        p.relative_to(RESOURCES).as_posix()
        for p in RESOURCES.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    expected = {"THIRD_PARTY_NOTICES.md", "__init__.py"} | {p.name for p in records()}
    assert shipped == expected


@pytest.mark.skipif(
    importlib.util.find_spec("packaging") is None, reason="packaging is not installed"
)
def test_no_installed_dependency_is_under_a_refused_licence():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import scan_licenses
    finally:
        sys.path.pop(0)
    rows = scan_licenses.scan()
    assert len(rows) > 10
    bad = [(n, v, t, s) for n, v, t, s in rows if s in ("refused", "unknown")]
    assert bad == []
