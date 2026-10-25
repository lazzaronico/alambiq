"""Checking a Boltz-2 request and writing its inputs.

Pure functions. Boltz runs in its own environment, outside this module.
"""

from __future__ import annotations

import re
from typing import Any

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")
MAX_CHAINS = 8
MAX_RESIDUES = 2000
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,4}")
LIMITS = {
    "seed": (0, 2**32 - 1),
    "sampling_steps": (2, 1000),
    "diffusion_samples": (1, 20),
    "recycling_steps": (1, 20),
    "max_msa_seqs": (1, 32768),
    "num_subsampled_msa": (1, 8192),
    "threads": (1, 64),
}


def check_chains(proteins: list[dict[str, Any]], ligand: dict[str, Any] | None) -> None:
    """Refuse chains Boltz would misread or that exceed what this capability runs."""
    if not 1 <= len(proteins) <= MAX_CHAINS:
        raise ValueError(f"Supply 1–{MAX_CHAINS} protein chains")
    identifiers: set[str] = set()
    total = 0
    for chain in [*proteins, *([ligand] if ligand else [])]:
        if not IDENTIFIER.fullmatch(chain["id"]) or chain["id"] in identifiers:
            raise ValueError(
                "Chain ids must be unique and simple: a letter then letters, digits "
                "or `_`, up to 5 characters (Boltz-2 truncates longer names and then "
                "cannot find the chain)"
            )
        identifiers.add(chain["id"])
    for chain in proteins:
        sequence = chain["sequence"].upper()
        if not 1 <= len(sequence) <= MAX_RESIDUES or set(sequence) - AMINO_ACIDS:
            raise ValueError(
                f"{chain['id']}: a protein sequence has 1–{MAX_RESIDUES} letters from "
                "the 20 amino acids and X"
            )
        total += len(sequence)
    if total > MAX_RESIDUES:
        raise ValueError(f"At most {MAX_RESIDUES} protein residues in all are run here")
    if ligand is not None and (
        not 1 <= len(ligand["smiles"]) <= 4000
        or any(c.isspace() for c in ligand["smiles"])
    ):
        raise ValueError(
            "A ligand needs a SMILES of 1–4000 characters with no whitespace"
        )


def check_settings(**values: int) -> None:
    for name, value in values.items():
        low, high = LIMITS[name]
        if not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")


def normalize_a3m(text: str, query: str) -> tuple[str, int]:
    """Join wrapped rows and check them; keep order, gaps and insertions.

    Boltz reads one unwrapped line per sequence, with gaps as dashes; rows with
    dot padding are refused. The first row must be the query, ungapped. Returns
    the text and the number of rows.
    """
    records: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if not line[1:].strip():
                raise ValueError("An A3M header needs a name")
            records.append([line[1:], ""])
        elif not records:
            raise ValueError("An A3M sequence comes before its header")
        else:
            records[-1][1] += line
    if not records or records[0][1] != query:
        raise ValueError(
            "The first A3M row must be exactly the chain's ungapped sequence"
        )
    for _, sequence in records:
        if not sequence or set(sequence.upper()) - (AMINO_ACIDS | {"-"}):
            raise ValueError(
                "An A3M row holds amino-acid letters, lowercase insertions and `-` gaps; "
                "dot padding is not accepted"
            )
        if len("".join(c for c in sequence if not c.islower())) != len(query):
            raise ValueError(
                "Every A3M row must cover the same number of query columns"
            )
    return "".join(f">{name}\n{sequence}\n" for name, sequence in records), len(records)


def timeout_seconds(
    accelerator: str, diffusion_samples: int, sampling_steps: int, recycling_steps: int
) -> int:
    """The time ceiling for one run."""
    base = 14_400 if accelerator == "cpu" else 3_600
    return int(
        base
        * diffusion_samples
        * max(1, sampling_steps / 200)
        * max(1, recycling_steps / 3)
    )
