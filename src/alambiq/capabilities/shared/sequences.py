"""Sequence parsing shared by more than one family.

Pure functions over text. This module imports the standard library only, so a
capability's science can use it without reaching into the orchestration layer.
"""

from __future__ import annotations

import re

# One-letter residues, including the ambiguity codes a deposited sequence may
# carry, and the length a single supplied sequence may reach.
ALPHABET = re.compile(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+")
MAX_RESIDUES = 10_000


def parse_fasta(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">") or not lines[0][1:].strip():
        raise ValueError("Supply one named protein FASTA record")
    sequence = "".join(lines[1:]).upper()
    if not ALPHABET.fullmatch(sequence) or len(sequence) > MAX_RESIDUES:
        raise ValueError(
            "Expected one ungapped protein sequence of 1 to 10,000 residues, without stop symbols"
        )
    return lines[0][1:].split()[0], sequence


def parse_panel(text):
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">") and line[1:].strip():
            header = line[1:].strip()
            records.append(
                {"id": header.split()[0], "description": header, "sequence": ""}
            )
        elif not records:
            raise ValueError("FASTA sequence must follow a named header")
        else:
            records[-1]["sequence"] += line.upper()
    if not 1 <= len(records) <= 10000 or len({r["id"] for r in records}) != len(
        records
    ):
        raise ValueError("Supply 1 to 10,000 proteins with unique identifiers")
    if any(
        not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYX]{1,5000}", r["sequence"])
        for r in records
    ):
        raise ValueError(
            "Each protein needs 1 to 5,000 ungapped standard residues (X allowed)"
        )
    if sum(len(r["sequence"]) for r in records) > 1_000_000:
        raise ValueError(
            "The local-panel profile accepts at most 1,000,000 residues in total"
        )
    return records


RESIDUE_LABEL = re.compile(r"(-?\d+)([A-Za-z]?)")


def parse_residue(text: str) -> tuple[str, str]:
    """Split an author residue label such as `55`, `-3` or `55A` into number and insertion code."""
    found = RESIDUE_LABEL.fullmatch(text.strip())
    if found is None:
        raise ValueError(
            f"Residue {text!r} must be an author residue number with an optional insertion letter, such as 55 or 55A"
        )
    return found.group(1), found.group(2)
