"""Building a masked prompt and checking what a model wrote into it.

Pure functions over sequences. The model runs elsewhere; everything it returns
is checked here against what was asked, so a candidate is only accepted if it is
a standard-amino-acid sequence of the right length that changed nothing outside
the regions it was allowed to rewrite.
"""

from __future__ import annotations

import hashlib
import re

STANDARD = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")
MAX_RESIDUES = 512


def build_prompt(
    sequence: str, regions: list[tuple[int, int]], num_steps: int | None
) -> dict:
    """Validate the mutable regions and return the prompt and the positions masked."""
    if not 1 <= len(sequence) <= MAX_RESIDUES:
        raise ValueError(
            f"protein: the sequence has {len(sequence)} residues; 1–{MAX_RESIDUES} "
            "protein residues are required"
        )
    if not STANDARD.fullmatch(sequence):
        raise ValueError(
            "protein: the sequence contains a residue outside the 20 standard amino "
            "acids; completion requires an unambiguous standard-amino-acid input"
        )
    if not regions:
        raise ValueError(
            "regions: no region is given; at least one region with one-based "
            "inclusive positions is required"
        )
    positions: set[int] = set()
    for start, end in regions:
        if not 1 <= start <= end <= len(sequence):
            raise ValueError(
                f"regions: region {start}–{end} must lie within the input sequence "
                f"(1–{len(sequence)})"
            )
        current = set(range(start, end + 1))
        if current & positions:
            raise ValueError(
                f"regions: region {start}–{end} overlaps an earlier region; regions "
                "must not overlap"
            )
        positions |= current
    if len(positions) == len(sequence):
        raise ValueError(
            f"regions: the regions cover all {len(sequence)} residues; at least one "
            "residue outside the regions is required"
        )
    steps = min(8, len(positions)) if num_steps is None else num_steps
    if not 1 <= steps <= min(64, len(positions)):
        raise ValueError(
            f"num_steps is {steps}; it must be 1–64 and no greater than the number "
            f"of masked residues ({len(positions)})"
        )
    return {
        "prompt": "".join(
            "_" if i in positions else aa for i, aa in enumerate(sequence, 1)
        ),
        "mutable_positions": sorted(positions),
        "num_steps": steps,
    }


def check_candidate(
    generated: str | None, sequence: str, mutable: list[int], seed: int, identifier: str
) -> dict:
    """Accept a generated sequence only if it kept every fixed residue."""
    result = {
        "id": identifier,
        "seed": seed,
        "sequence": generated,
        "status": "rejected",
    }
    if not isinstance(generated, str) or not STANDARD.fullmatch(generated):
        return {
            **result,
            "reason": "Generated sequence is missing, masked or nonstandard",
        }
    if len(generated) != len(sequence):
        return {**result, "reason": "Generated length differs from input length"}
    changes = [
        {"position": i, "before": a, "after": b}
        for i, (a, b) in enumerate(zip(sequence, generated, strict=True), 1)
        if a != b
    ]
    if any(change["position"] not in mutable for change in changes):
        return {
            **result,
            "reason": "Generated sequence changed a fixed residue",
            "mutations": changes,
        }
    return {
        **result,
        "status": "completed",
        "mutations": changes,
        "unchanged_from_input": not changes,
        "sequence_sha256": hashlib.sha256(generated.encode()).hexdigest(),
    }
