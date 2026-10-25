"""Reading and judging an A3M multiple sequence alignment.

Pure functions over text. An A3M holds the query first, then each hit; lower-case
letters are residues inserted relative to the query and do not add columns.
"""

from __future__ import annotations

import statistics
from typing import Any

QUERY_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYX-")
DIVERSE_IDENTITY = 0.90


def read_a3m(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    sequence: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(sequence)))
            identifier = line[1:].strip()
            sequence = []
        elif identifier is None:
            raise ValueError("A3M sequence data precedes the first header")
        else:
            sequence.append(line)
    if identifier is not None:
        records.append((identifier, "".join(sequence)))
    if not records:
        raise ValueError("A3M contains no sequence records")
    return records


def canonicalize(text: str) -> str:
    """An order- and wrapping-stable A3M with the query kept first."""
    records = read_a3m(text)
    identifiers = [name for name, _ in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("A3M record identifiers must be unique")
    query, *hits = records
    ordered = [query, *sorted(hits, key=lambda record: (record[0], record[1]))]
    return "".join(f">{name}\n{sequence}\n" for name, sequence in ordered)


def _columns(sequence: str) -> str:
    columns = "".join(c for c in sequence if not c.islower() and c != ".")
    if not columns or set(columns) - QUERY_RESIDUES:
        raise ValueError("A3M contains unsupported query-column characters")
    return columns


def assess(
    text: str,
    query: str,
    *,
    minimum_query_coverage: float,
    minimum_alignment_sequences: int,
    minimum_diverse_sequences: int,
) -> dict[str, Any]:
    """Count what an alignment holds and say whether it is deep enough to use."""
    records = read_a3m(text)
    alignments = [_columns(sequence) for _, sequence in records]
    if any(len(a) != len(query) for a in alignments):
        raise ValueError("A3M query-column length does not match the declared query")
    if alignments[0].replace("-", "") != query:
        raise ValueError("A3M first sequence does not reproduce the declared query")
    coverages = [sum(r != "-" for r in a) / len(query) for a in alignments]
    identities = [
        sum(o == e for o, e in zip(a, query, strict=True) if o != "-")
        / max(1, sum(r != "-" for r in a))
        for a in alignments
    ]
    unique = set(alignments)
    diverse = {
        a
        for a, coverage, identity in zip(
            alignments[1:], coverages[1:], identities[1:], strict=True
        )
        if coverage >= minimum_query_coverage and identity <= DIVERSE_IDENTITY
    }
    reasons = []
    if len(unique) < minimum_alignment_sequences:
        reasons.append("msa-insufficient-unique-sequences")
    if len(diverse) < minimum_diverse_sequences:
        reasons.append("msa-insufficient-diverse-sequences")
    median = float(statistics.median(coverages))
    if median < minimum_query_coverage:
        reasons.append("msa-insufficient-query-coverage")
    return {
        "sequence_count": len(records),
        "unique_sequence_count": len(unique),
        "diverse_sequence_count": len(diverse),
        "query_coverage_fraction": coverages[0],
        "median_query_coverage_fraction": median,
        "quality_passed": not reasons,
        "reason_codes": reasons,
    }
