"""List the license of every installed package the dependencies of Alambiq require.

    python tools/scan_licenses.py [--all]

The dependencies are those of the installed `alambiq` distribution and its extras,
followed through their own requirements. Each package's license is read from its metadata
(the SPDX expression when it has one, else the license field, else the trove classifiers)
and sorted into allowed, needing a decision, or refused. Exit status 1 means a package is
refused or its license cannot be determined; those need a person to look at the package
and either replace it or record why it is acceptable in `LICENSING.md`.

The allowlist follows `LICENSING.md`: permissive terms are preferred, weak copyleft in
dependencies that are linked but not modified (MPL, LGPL) is allowed and named, and
strong copyleft and non-commercial terms are refused.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata

from packaging.requirements import Requirement

ALLOWED = re.compile(
    r"\b(MIT|BSD|Apache|ISC|PSF|Python Software Foundation|HPND|Zlib|Unlicense|"
    r"CC0|Public Domain|0BSD|BlueOak|Mulan|Historical Permission)\b",
    re.IGNORECASE,
)
WEAK = re.compile(r"(\bMPL|Mozilla|LGPL|Lesser General Public)", re.IGNORECASE)
REFUSED = re.compile(
    r"\b(AGPL|Affero|GPL|General Public License|Non-?commercial|SSPL|Commons Clause)\b",
    re.IGNORECASE,
)

# Packages whose metadata carries no usable license, with the license read from the project.
# Each entry names where it was read; add one only after reading it.
READ_BY_HAND: dict[str, str] = {
    # LICENSE file of 0.7.0: "Ptyprocess is under the ISC license".
    "ptyprocess": "ISC",
    # LICENSE file of 2.9.0.post0: Apache-2.0 for recent work, BSD-3-Clause for the rest.
    "python-dateutil": "Apache-2.0 and BSD-3-Clause",
    # LICENSE.txt of 1.3: "either GPL or GPLv2+, or the Artistic License". Used under the
    # Artistic License (OSI-approved); required by python-slugify, which Prefect requires.
    "text-unidecode": "Artistic License (chosen from a dual license with GPLv2+)",
    # Biopython's license file offers the Biopython License Agreement or BSD-3-Clause.
    "biopython": "Biopython License Agreement or BSD-3-Clause",
}


def licence_of(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression
    text = (meta.get("License") or "").strip()
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    if text and len(text) < 120 and "\n" not in text:
        return text
    if classifiers:
        return "; ".join(classifiers)
    if text:
        return text.splitlines()[0][:120]
    return ""


def closure(root: str = "alambiq") -> dict[str, metadata.Distribution]:
    found: dict[str, metadata.Distribution] = {}
    pending = [root]
    while pending:
        name = pending.pop()
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        key = dist.metadata["Name"].lower().replace("_", "-")
        if key in found:
            continue
        found[key] = dist
        for line in dist.requires or []:
            requirement = Requirement(line)
            if requirement.marker is not None:
                extras = [""] if key != root else [""] + list(
                    dist.metadata.get_all("Provides-Extra") or []
                )
                if not any(requirement.marker.evaluate({"extra": e}) for e in extras):
                    continue
            pending.append(requirement.name)
    return found


def classify(name: str, text: str) -> str:
    text = READ_BY_HAND.get(name, text)
    if not text:
        return "unknown"
    if "Artistic" in text:
        return "allowed"
    if REFUSED.search(text) and not WEAK.search(text):
        return "refused"
    if WEAK.search(text):
        return "weak"
    if ALLOWED.search(text):
        return "allowed"
    return "unknown"


def scan() -> list[tuple[str, str, str, str]]:
    rows = []
    for name, dist in sorted(closure().items()):
        text = licence_of(dist)
        text = READ_BY_HAND.get(name, text)
        rows.append((name, dist.version, text, classify(name, text)))
    return rows


def main() -> int:
    rows = scan()
    show_all = "--all" in sys.argv
    for name, version, text, state in rows:
        if show_all or state != "allowed":
            print(f"{state:8s} {name} {version}: {text or '(none declared)'}")
    counts = {s: sum(1 for r in rows if r[3] == s) for s in ("allowed", "weak", "unknown", "refused")}
    print(f"\n{len(rows)} packages: {counts}")
    return 1 if counts["unknown"] or counts["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
