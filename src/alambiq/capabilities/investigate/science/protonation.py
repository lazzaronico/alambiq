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
