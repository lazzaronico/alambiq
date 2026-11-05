"""Checks of the request for the protonation of a protein structure."""

from __future__ import annotations

from pathlib import Path

PH_RANGE = (0.0, 14.0)
SUFFIXES = (".pdb", ".ent")
NO_PKA = 99.0
"""PROPKA reports 99.99 for a cysteine in a disulfide bond: it does not titrate."""


def check_request(*, structure: str, ph: float, timeout_seconds: int) -> None:
    """Refuse a request the engine cannot carry out."""
    if Path(structure).suffix.lower() not in SUFFIXES:
        raise ValueError(
            f"The structure {structure!r} is not a PDB file; the extension must be .pdb or .ent"
        )
    if not PH_RANGE[0] <= ph <= PH_RANGE[1]:
        raise ValueError(f"ph is {ph}; it must lie between {PH_RANGE[0]:g} and {PH_RANGE[1]:g}")
    if timeout_seconds < 10:
        raise ValueError(f"timeout_seconds is {timeout_seconds}; it must be 10 or more")


def split_disulfides(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Titratable groups, and the cysteines that are bonded in disulfides."""
    groups, bonded = [], []
    for row in rows:
        if row["group"] == "CYS" and row["pka"] >= NO_PKA:
            bonded.append(f"{row['chain']}:CYS{row['number']}")
        else:
            groups.append(row)
    return groups, bonded


def describe_residue(chain: str, number: int, name: str) -> str:
    return f"{chain}:{name}{number}"


VARIANT_NAMES = frozenset({"ASP", "ASH", "GLU", "GLH", "LYS", "LYN", "CYS", "CYX", "HID", "HIE", "HIP"})
"""The residue names in which a protonated structure states a protonation, as OpenMM reads them."""


def variants_from_structure(path: str | Path, chain_id: str) -> dict[int, str]:
    """The protonation variant of each titratable residue of one chain, by residue number.

    The file is the protonated structure of `structure.assign_protonation`, whose residue
    names (`HID`, `ASH`, `LYN`, `CYX`) state the protonation, and whose numbers are those of
    the structure that was protonated.
    """
    found: dict[int, str] = {}
    chain_seen = False
    for line in Path(path).read_text().splitlines():
        if line[:6].strip() != "ATOM" or line[21] != chain_id:
            continue
        chain_seen = True
        if line[26].strip():
            raise ValueError(
                f"The protonated structure has an insertion code at residue {line[22:27].strip()}; "
                "residue numbers must be plain integers"
            )
        name = line[17:20].strip()
        if name in VARIANT_NAMES:
            found[int(line[22:26])] = name
    if not chain_seen:
        raise ValueError(f"The protonated structure {str(path)!r} has no chain {chain_id!r}")
    return found
