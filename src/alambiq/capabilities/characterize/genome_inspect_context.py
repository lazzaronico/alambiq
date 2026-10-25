"""Show the genomic features around chosen proteins, from a supplied GFF3.

For each anchor protein the capability finds the CDS feature whose chosen
attribute (`ID`, `protein_id` or `locus_tag`) equals the identifier of the
protein, then lists every CDS, tRNA, rRNA, ncRNA and tmRNA feature that
overlaps a window of `flank_bp` bases on each side, clipped at the bounds
declared for the contig. The proteins of the displayed features that are in the
supplied `proteins` FASTA are written to a FASTA of their own, which can be
passed to `protein.search` as the protein set; `sequence.generate_msa` accepts
at most 8 proteins per step. Coordinates are one-based and inclusive.

Limits. The capability reads the supplied files only: an identifier is matched
exactly against an attribute value, and the sequences are the supplied
proteins. An anchor that matches no CDS is reported as unmapped and has no
window; an anchor that matches more than one CDS is reported as ambiguous and
has one window per matching CDS feature. An anchor given in `candidates` whose
sequence differs from the sequence with the same identifier in `proteins` is
withheld, as is an anchor given in `candidates` that matches a CDS and is absent
from `proteins`. The bounds of a contig come from a `##sequence-region`
directive or the embedded `##FASTA` section of the GFF3. Without declared
bounds, the window start is limited to position 1, the end is not limited, and
the result reports clipping as unknown. Declared bounds can be the edge of a
contig that is part of a larger sequence rather than the end of the genome, and
a window of a circular contig that reaches the origin ends there and does not
wrap across it. Adjacency is positional; operon membership, regulation and
function are not read from it. A request with more than 50 anchors, 100
windows, 250 features in a window, 5,000 displayed feature occurrences or
10,000 GFF3 feature rows is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alambiq.capabilities.characterize.genome_inspect_context_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class ContextRequest:
    """The genome files and the anchors around which windows are placed.

    gff:
        A path to a GFF3 file for the genome.
    proteins:
        A path to a FASTA of 1 to 10,000 proteins of the genome, with the same
        identifiers as the values of the GFF3 attribute. Identifiers are
        unique. Each protein has 1 to 5,000 residues (the 20 amino acids and
        X); all proteins together have at most 1,000,000 residues. The output
        FASTA lists the proteins of displayed features that have an entry in
        this FASTA.
    genome_id:
        A name for the genome, 1 to 128 letters, digits, underscore, dot, colon and hyphen.
    protein_id_attribute:
        The CDS attribute that holds the protein identifier: `ID`, `protein_id`
        or `locus_tag`. A CDS feature matches an anchor when a value of this
        attribute equals the anchor identifier exactly.
    anchor_ids:
        The proteins on which to center a window, 1 to 50 unique identifiers
        without whitespace. Exactly one of this field and `candidates` is
        required.
    candidates:
        A path to a FASTA of 1 to 50 proteins that are the anchors, with the
        same sequence rules as `proteins`; the sequence of each candidate is
        also checked against the sequence with the same identifier in
        `proteins`.
    flank_bp:
        Number of bases in the window on each side of each anchor, 0 to
        100,000; default 5,000.
    """

    gff: str
    proteins: str
    genome_id: str
    protein_id_attribute: Literal["ID", "protein_id", "locus_tag"]
    anchor_ids: list[str] = field(default_factory=list)
    candidates: str | None = None
    flank_bp: int = 5000


@dataclass(frozen=True, slots=True)
class Anchor:
    """One anchor protein and whether it was placed in the GFF3."""

    id: str
    status: Literal[
        "mapped", "ambiguous", "unmapped", "sequence_mismatch", "sequence_unavailable"
    ]
    loci: int
    """Number of CDS features with its identifier."""


@dataclass(frozen=True, slots=True)
class Window:
    """The window displayed around one CDS feature that matches an anchor."""

    anchor_id: str
    contig: str
    start: int
    end: int
    features: int
    left_clipped: bool | None
    right_clipped: bool | None
    circular: bool


@dataclass(frozen=True, slots=True)
class ContextResult:
    """The location of each anchor and the features around it."""

    genome_id: str
    flank_bp: int
    anchors: list[Anchor]
    windows: list[Window]
    neighbourhood_proteins: int
    record: str
    """The stored full windows, with every feature."""
    neighbourhood_fasta: str | None
    """A FASTA of the proteins of the displayed features, or None if there are none."""


@capability(
    "genome.inspect_context",
    version="1.0",
    family="characterize",
    summary="Show the CDS and RNA features around chosen proteins from a supplied GFF3",
    render=render,
)
def inspect_context(request: ContextRequest, ctx: StepContext) -> ContextResult:
    """Place each anchor in the GFF3 and list the features around it."""
    from alambiq.capabilities.characterize.science.genome import (
        inspect_context as place,
    )

    if request.candidates is not None and request.anchor_ids:
        raise ValueError(
            "candidates and anchor_ids are both set; set a candidates FASTA or "
            "anchor_ids, not both"
        )
    candidates = (
        ctx.read_text(request.candidates) if request.candidates is not None else None
    )
    record = place(
        ctx.read_text(request.gff),
        ctx.read_text(request.proteins),
        {
            "genome_id": request.genome_id,
            "flank_bp": request.flank_bp,
            "protein_id_attribute": request.protein_id_attribute,
            "anchor_ids": None if candidates is not None else list(request.anchor_ids),
        },
        candidates=candidates,
    )

    anchors = [
        Anchor(id=a["id"], status=a["status"], loci=len(a["locus_keys"]))
        for a in record["anchors"]
    ]
    windows = [
        Window(
            anchor_id=w["anchor_id"],
            contig=w["seqid"],
            start=w["start"],
            end=w["end"],
            features=len(w["features"]),
            left_clipped=w["left_clipped"],
            right_clipped=w["right_clipped"],
            circular=w["circular"],
        )
        for w in record["neighbourhoods"]
    ]
    for status, meaning in (
        (
            "unmapped",
            "match no CDS feature by the chosen attribute",
        ),
        (
            "ambiguous",
            "match more than one CDS feature",
        ),
        (
            "sequence_mismatch",
            "have a sequence that differs from the sequence with the same "
            "identifier in the proteins FASTA",
        ),
        (
            "sequence_unavailable",
            "are not in the proteins FASTA",
        ),
    ):
        names = [a.id for a in anchors if a.status == status]
        if names:
            ctx.note(f"{len(names)} anchor(s) {meaning}: {', '.join(names)}.")
    clipped = [w for w in windows if w.left_clipped or w.right_clipped]
    if clipped:
        ctx.note(
            f"{len(clipped)} window(s) reach the edge of the contig bounds declared "
            "in the file."
        )

    proteins = record["neighbourhood_proteins"]
    fasta = "".join(f">{p['description']}\n{p['sequence']}\n" for p in proteins)
    return ContextResult(
        genome_id=record["genome_id"],
        flank_bp=request.flank_bp,
        anchors=anchors,
        windows=windows,
        neighbourhood_proteins=len(proteins),
        record=ctx.keep_json(record),
        neighbourhood_fasta=(
            str(ctx.keep_text(fasta, "text/x-fasta").path) if fasta else None
        ),
    )
