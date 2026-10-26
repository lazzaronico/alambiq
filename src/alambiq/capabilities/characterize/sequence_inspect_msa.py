"""Describe a supplied multiple sequence alignment, column by column.

Returns, for each column of a supplied alignment, the residue counts, the
consensus, the gap fraction and the Shannon entropy in bits. The result also
counts rows, columns, duplicate rows and insertions, and lists the fully
conserved positions.

The method:

- The alignment is read as supplied. Every row has weight 1.
- The first row is the query. Its ungapped positions number the columns.
- Each column reports the number of rows with each residue.
- The consensus is the most common residue. Its fraction is the share of the
  non-gap, non-ambiguous rows of the column that have it.
- Each column reports the fraction of gaps and of ambiguous letters.
- The entropy of a column comes from the raw residue counts of the rows that
  are neither gaps nor ambiguous letters. Gaps and the ambiguous letters B, J,
  X and Z are excluded from the entropy and the consensus. Selenocysteine and
  pyrrolysine are included.
- An A3M file, an alignment format in which lowercase letters mark insertions
  relative to the query, keeps its inserted residues. Each attaches to the
  column that it follows.
- Optionally each row comes with the full protein of which it is a fragment.
  The alignment positions of the fragment then map back to its full sequence.
- Rows whose match columns equal those of an earlier row are flagged as
  duplicates (`duplicate_rows`) and not removed. `unique_rows` counts the
  distinct match-column strings, which `sequence.generate_msa` reports as
  `unique_sequences`.

Limits.

- Every row counts once. A column dominated by close relatives has low entropy
  whatever the number of independent lineages.
- A column with fewer informative rows has a less certain entropy.
- Conservation is a property of the supplied sequences, which the search that
  produced the alignment selected. It indicates functional importance only
  against that selection.
- An alignment with more than 200 rows, 5,000 columns or 250,000 aligned cells
  is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alambiq.capabilities.characterize.sequence_inspect_msa_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class FullSequence:
    """The whole protein from which an alignment row was cut.

    sequence:
        The full protein sequence.
    start:
        The one-based position in it at which the ungapped sequence of the row
        begins.
    """

    sequence: str
    start: int


@dataclass(frozen=True, slots=True)
class InspectMsaRequest:
    """The alignment to describe.

    alignment:
        A path to the alignment. The identifier of a row is its header text up
        to the first whitespace. Identifiers are unique. In both formats the
        first row is the query, with no gaps and no lowercase letters.
    format:
        `fasta`: aligned FASTA. `-` marks gaps. Lowercase letters in the rows
        after the first become uppercase.
        `a3m`: lowercase letters mark inserted residues and `.` pads them. The
        rows after the first can hold `-`.
    full_sequences:
        A mapping from row identifier to the full protein and the one-based
        position at which the row begins in it. Its identifiers are exactly
        the row identifiers of the alignment.
    """

    alignment: str
    format: Literal["fasta", "a3m"]
    full_sequences: dict[str, FullSequence] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Position:
    """One query position and how its column varies."""

    query_position: int
    entropy_bits: float | None
    consensus: str | None
    consensus_fraction: float | None
    gap_fraction: float


@dataclass(frozen=True, slots=True)
class InspectMsaResult:
    """Counts of rows, columns, duplicates and insertions, the fully conserved positions, the mean entropy and gap fraction, the 10 query positions with the highest entropy, and the stored record."""

    query_id: str
    rows: int
    columns: int
    unique_rows: int
    duplicate_rows: int
    inserted_residues: int
    fully_conserved_positions: list[int]
    """Query positions where every row has the same residue, with no gaps."""
    mean_entropy_bits: float | None
    mean_gap_fraction: float
    most_variable: list[Position]
    """The 10 query positions with the highest entropy."""
    record: str
    """The stored whole alignment, with the counts of every column."""


@capability(
    "sequence.inspect_msa",
    version="1.0",
    family="characterize",
    summary="Describe the conservation, gaps and redundancy of a supplied alignment",
    render=render,
)
def inspect_msa(request: InspectMsaRequest, ctx: StepContext) -> InspectMsaResult:
    """Read the alignment and summarize it by column and by row."""
    from alambiq.capabilities.characterize.science.alignment import inspect_alignment

    context = (
        {
            name: {"sequence": f.sequence, "start": f.start}
            for name, f in request.full_sequences.items()
        }
        if request.full_sequences
        else None
    )
    record = inspect_alignment(
        ctx.read_text(request.alignment), request.format, full_sequence_context=context
    )

    rows, columns = record["rows"], record["columns"]
    duplicates = sum(1 for r in rows if r["duplicate_of"] is not None)
    entropies = [c["entropy_bits"] for c in columns if c["entropy_bits"] is not None]
    on_query = [c for c in columns if c["query_position"] is not None]
    conserved = [
        c["query_position"]
        for c in on_query
        if c["informative_count"] == record["row_count"]
        and c["consensus_fraction"] == 1.0
    ]
    variable = sorted(
        (c for c in on_query if c["entropy_bits"] is not None),
        key=lambda c: (-c["entropy_bits"], c["query_position"]),
    )[:10]

    if record["row_count"] == 1:
        ctx.note("The alignment has one row.")
    if duplicates:
        ctx.note(
            f"{duplicates} of {record['row_count']} rows repeat an earlier row's "
            "match columns."
        )

    return InspectMsaResult(
        query_id=record["query_id"],
        rows=record["row_count"],
        columns=record["column_count"],
        unique_rows=record["unique_aligned_rows"],
        duplicate_rows=duplicates,
        inserted_residues=sum(
            len(i["sequence"]) for r in rows for i in r["insertions"]
        ),
        fully_conserved_positions=conserved,
        mean_entropy_bits=sum(entropies) / len(entropies) if entropies else None,
        mean_gap_fraction=sum(c["gap_fraction"] for c in columns) / len(columns),
        most_variable=[
            Position(
                query_position=c["query_position"],
                entropy_bits=c["entropy_bits"],
                consensus=c["consensus"],
                consensus_fraction=c["consensus_fraction"],
                gap_fraction=c["gap_fraction"],
            )
            for c in variable
        ],
        record=ctx.keep_json(record),
    )
