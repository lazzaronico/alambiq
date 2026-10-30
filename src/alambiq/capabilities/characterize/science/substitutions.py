"""The differences between a variant and its reference, and variants written in notation.

The alignment is a pair of strings of equal length, with `-` for a gap, as MMseqs2 writes
them. Positions are residue numbers of the reference, counted from 1. A substitution is
written reference residue, position, variant residue (`Q21W`), and the substitutions of one
variant are joined with colons (`Q21W:Q51W`).
"""

from __future__ import annotations

import re
from typing import Any

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
MAX_SUBSTITUTIONS = 50
_TOKEN = re.compile(r"([A-Z])([0-9]+)([A-Z])")


def parse_variant(text: str, sequence: str) -> list[tuple[str, int, str]]:
    """The substitutions of a variant written `K183N:T209K`, checked against the sequence.

    Each substitution names the residue of the sequence at its position, a different
    residue of the 20 amino acids, and a position that no other substitution of the variant
    uses. At most `MAX_SUBSTITUTIONS` substitutions are accepted.
    """
    found: list[tuple[str, int, str]] = []
    for token in text.split(":"):
        match = _TOKEN.fullmatch(token.strip())
        if not match:
            raise ValueError(
                f"{text!r}: {token!r} is not a substitution; write the residue of the "
                "sequence, its position and the new residue, such as K183N, joined by colons"
            )
        wild, position, mutant = match.group(1), int(match.group(2)), match.group(3)
        if not 1 <= position <= len(sequence):
            raise ValueError(
                f"{text!r}: position {position} lies outside the sequence of {len(sequence)} residues"
            )
        if sequence[position - 1] != wild:
            raise ValueError(
                f"{text!r}: position {position} is {sequence[position - 1]} in the sequence, not {wild}"
            )
        if mutant not in CANONICAL or mutant == wild:
            raise ValueError(
                f"{text!r}: {mutant} must be one of the 20 amino acids and differ from {wild}"
            )
        found.append((wild, position, mutant))
    if len({position for _, position, _ in found}) != len(found):
        raise ValueError(f"{text!r}: two substitutions are at one position")
    if len(found) > MAX_SUBSTITUTIONS:
        raise ValueError(
            f"{text!r}: at most {MAX_SUBSTITUTIONS} substitutions are scored together"
        )
    return found


def call(
    variant_aligned: str,
    reference_aligned: str,
    *,
    variant_start: int,
    variant_end: int,
    variant_length: int,
    reference_start: int,
    reference_end: int,
    reference_length: int,
) -> dict[str, Any]:
    """The substitutions, insertions and deletions of an aligned variant.

    An insertion is variant residues between two reference residues, given as the
    position of the reference residue before them (0 before the first). A deletion is a run
    of reference residues that the variant lacks. The residues of the variant that the
    alignment does not cover, and those of the reference, are counted at each end.
    """
    if len(variant_aligned) != len(reference_aligned):
        raise ValueError("the two aligned strings differ in length")
    substitutions: list[dict[str, Any]] = []
    insertions: list[dict[str, Any]] = []
    deletions: list[dict[str, Any]] = []
    position = reference_start - 1
    inserted = ""
    deleted_from = 0
    deleted = ""

    def close() -> None:
        nonlocal inserted, deleted
        if inserted:
            insertions.append(
                {"after_position": position - len(deleted), "residues": inserted}
            )
            inserted = ""
        if deleted:
            deletions.append(
                {
                    "start": deleted_from,
                    "end": deleted_from + len(deleted) - 1,
                    "residues": deleted,
                }
            )
            deleted = ""

    for v, r in zip(variant_aligned, reference_aligned, strict=True):
        if v != "-" and r != "-":
            close()
            position += 1
            if v != r:
                substitutions.append(
                    {"reference": r, "position": position, "variant": v}
                )
        elif v != "-":
            if deleted:
                close()
            inserted += v
        else:
            if inserted:
                close()
            position += 1
            if not deleted:
                deleted_from = position
            deleted += r
    close()
    return {
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "unaligned_variant_start": variant_start - 1,
        "unaligned_variant_end": variant_length - variant_end,
        "unaligned_reference_start": reference_start - 1,
        "unaligned_reference_end": reference_length - reference_end,
    }


def count_differences(read: dict[str, Any]) -> int:
    """The residues that differ: substitutions, inserted and deleted residues, and the
    residues that the alignment leaves out at the ends of the variant and the reference."""
    return (
        len(read["substitutions"])
        + sum(len(i["residues"]) for i in read["insertions"])
        + sum(len(d["residues"]) for d in read["deletions"])
        + read["unaligned_variant_start"]
        + read["unaligned_variant_end"]
        + read["unaligned_reference_start"]
        + read["unaligned_reference_end"]
    )


def notation(substitutions: list[dict[str, Any]]) -> str:
    """The substitutions in standard notation joined by colons, such as `Q21W:Q51W`."""
    return ":".join(
        f"{s['reference']}{s['position']}{s['variant']}" for s in substitutions
    )
