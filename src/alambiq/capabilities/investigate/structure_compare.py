"""Superpose one chain of a structure on another and report the RMSD of mapped alpha carbons.

The result gives the RMSD of the paired alpha carbons before and after the best rigid-body
fit, the distance of every pair, the rotation and translation, and the atoms of the mobile
chain in the reference frame as an mmCIF file.

Two stored `protein.inspect` records each supply a structure (deposited or predicted); the
request selects one chain of each. The fit superposes the alpha carbons of corresponding
residues with equal weights. Every pair with both alpha carbons observed enters the fit.

The correspondence between residues comes from one of these:

- the residue pairs of the request;
- the columns of the alignment of the request, with the reference row and the mobile row
  named;
- the query sequence, when the request gives neither. Both proteins must then have the same
  query sequence, and equal positions correspond.

A request supplies at most one of the residue pairs and the alignment.

Limits.

- The RMSD depends on which residues are paired. A pairing that includes a flexible loop or
  residues that are not equivalent can increase it. A pairing restricted to a well-conserved
  core can decrease it.
- Equal weights let large displacements of individual residues affect the whole fit.
- Only the alpha carbons of the selected model and chain enter the fit; ligands, waters and
  other chains do not.
- Residues missing an alpha carbon in either structure are listed and excluded. The fraction
  of each query that was fitted is reported with the RMSD.
- The coordinates are as deposited and carry their experimental uncertainty. The RMSD of a
  predicted structure includes the prediction error.
- The fit requires at least 3 fitted pairs that are not collinear. A request with more than
  10,000 pairs is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alambiq.capabilities.investigate.structure_compare_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class ResiduePair:
    """One pair of corresponding residues, by query position (one-based).

    reference, mobile:
        The residue's position in each protein's query sequence.
    """

    reference: int
    mobile: int


@dataclass(frozen=True, slots=True)
class CompareRequest:
    """The two structures, the chains, and how their residues correspond.

    reference:
        A path to a `protein.inspect` record whose structure is the fixed reference.
    mobile:
        A path to a `protein.inspect` record whose chain is moved onto the reference.
    reference_chain, mobile_chain:
        The `label_asym_id` of the chain in each record. The record must map that chain
        exactly to its query sequence (mapping `exact`).
    residue_pairs:
        An explicit correspondence of 1 to 10,000 pairs. The fit requires at least 3
        fitted pairs that are not collinear.
    alignment:
        A path to an alignment whose columns give the correspondence, instead of
        `residue_pairs`. Set `alignment_format`, `reference_row` and `mobile_row` with it.
    alignment_format:
        `fasta` or `a3m`.
    reference_row, mobile_row:
        The identifiers of the alignment rows for each protein.

    With neither pairs nor an alignment, both proteins must have the same query sequence.
    Residues are then paired by position in it.
    """

    reference: str
    mobile: str
    reference_chain: str
    mobile_chain: str
    residue_pairs: list[ResiduePair] = field(default_factory=list)
    alignment: str | None = None
    alignment_format: Literal["fasta", "a3m"] | None = None
    reference_row: str | None = None
    mobile_row: str | None = None


@dataclass(frozen=True, slots=True)
class PairDistance:
    """One pair's distance before and after the fit."""

    reference: int
    mobile: int
    reference_residue: str
    mobile_residue: str
    status: str
    """`fitted`, or which alpha carbon was missing."""
    distance_before_angstrom: float | None
    distance_after_angstrom: float | None


@dataclass(frozen=True, slots=True)
class CompareResult:
    """The fit, and how much of each protein it covers."""

    correspondence: str
    requested_pairs: int
    fitted_pairs: int
    rmsd_before_angstrom: float
    rmsd_after_angstrom: float
    reference_coverage: float
    """Fraction of the reference query that took part in the fit."""
    mobile_coverage: float
    rotation: list[list[float]]
    translation_angstrom: list[float]
    pairs: list[PairDistance]
    fitted_structure: str
    """The mobile chain's atoms in the reference frame, as an mmCIF file."""
    record: str
    gemmi_version: str


@capability(
    "structure.compare",
    version="1.0",
    family="investigate",
    summary="Superpose one structure chain on another by mapped alpha carbons",
    render=render,
)
def compare(request: CompareRequest, ctx: StepContext) -> CompareResult:
    """Fit the mobile chain onto the reference over the corresponding residues."""
    from alambiq.capabilities.investigate.science.comparison import compare_structures

    alignment = None
    if request.alignment is not None:
        alignment = {"text": ctx.read_text(request.alignment)}
    rows = None
    if request.reference_row is not None or request.mobile_row is not None:
        rows = {"reference": request.reference_row, "mobile": request.mobile_row}
    record, fitted = compare_structures(
        ctx.read_json(request.reference),
        ctx.read_json(request.mobile),
        reference_chain=request.reference_chain,
        mobile_chain=request.mobile_chain,
        residue_pairs=(
            [
                {"reference": p.reference, "mobile": p.mobile}
                for p in request.residue_pairs
            ]
            or None
        ),
        alignment=alignment,
        alignment_format=request.alignment_format,
        alignment_rows=rows,
    )

    unfitted = [p for p in record["pairs"] if p["status"] != "fitted"]
    if unfitted:
        ctx.note(
            f"{len(unfitted)} of {record['requested_pair_count']} pairs are left out of the "
            "fit because an alpha carbon is missing in one of the structures."
        )
    for side in ("reference", "mobile"):
        coverage = record[side]["fit_query_coverage"]
        if coverage < 0.5:
            ctx.note(f"Only {coverage:.0%} of the {side} query is included in the fit.")
    transform = record["transform"]
    return CompareResult(
        correspondence=record["correspondence"],
        requested_pairs=record["requested_pair_count"],
        fitted_pairs=record["fit_pair_count"],
        rmsd_before_angstrom=record["rmsd_before_angstrom"],
        rmsd_after_angstrom=record["rmsd_after_angstrom"],
        reference_coverage=record["reference"]["fit_query_coverage"],
        mobile_coverage=record["mobile"]["fit_query_coverage"],
        rotation=transform["rotation"],
        translation_angstrom=transform["translation_angstrom"],
        pairs=[
            PairDistance(
                reference=p["reference"],
                mobile=p["mobile"],
                reference_residue=p["reference_residue"],
                mobile_residue=p["mobile_residue"],
                status=p["status"],
                distance_before_angstrom=p["distance_before_angstrom"],
                distance_after_angstrom=p["distance_after_angstrom"],
            )
            for p in record["pairs"]
        ],
        fitted_structure=str(ctx.keep_text(fitted, "chemical/x-mmcif").path),
        record=ctx.keep_json(record),
        gemmi_version=record["gemmi_version"],
    )
