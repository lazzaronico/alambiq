"""Inspect one protein: its sequence, a UniProt entry and a structure.

Returns one stored record, numbered as the query sequence. The record holds the
query, the features of the UniProt entry, the chains and residues of the
structure, and the mapping state of each. Anything that does not map is stored
and marked withheld, with the reason.

The mapping:

- A UniProt feature maps onto the query when the query occurs exactly once in
  the entry sequence. The feature has exact start and end positions inside the
  query span and no `sequence` qualifier.
- Only `polypeptide(L)` entities of an mmCIF structure form chains.
- A chain maps when its polymer sequence contains the query exactly once or is
  contained in the query exactly once. A polymer sequence with X maps only when
  the `_entity_poly_seq` category resolves it to standard residues.
- A residue maps when its monomer agrees with the query at that position and
  the position lies inside the query span. Otherwise the reason is
  `outside-query-span`.
- The per-residue pLDDT declared in a ModelCIF structure is always reported.
- With `coordinate_annotations`, secondary structure (DSSP, 8 states) and
  solvent accessible surface area are computed per residue.

Limits.

- Mapping requires exact sequence agreement. A structure or UniProt entry that
  differs from the query by a mutation, a tag or a different construct is
  withheld whole.
- The structure is read in mmCIF format. Alternate locations other than "A" are
  dropped. A file with more than 100,000 `_atom_site` rows (all models
  together) is refused.
- A structure is one model of one state. Crystal packing and absent binding
  partners can change secondary structure and surface exposure relative to the
  free protein.
- An experimental structure has no confidence estimate. The pLDDT comes from a
  ModelCIF.
- The result records the version of MDTraj that computed the annotations. The
  version is not enforced.
- Any use of `structure` requires the `dossier` extra (gemmi 0.7.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alambiq.capabilities.characterize.protein_inspect_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class SourceNote:
    """The origin of a supplied file, stored with the dossier for attribution.

    source_url:
        Where the file was obtained.
    license:
        The license terms of the file.
    citation:
        The citation of the file.
    """

    source_url: str | None = None
    license: str | None = None
    citation: str | None = None


@dataclass(frozen=True, slots=True)
class InspectRequest:
    """What to inspect.

    fasta:
        A path to a FASTA file with one protein sequence. It defines the query.
    uniprot:
        A path to a UniProt JSON entry. With no `fasta`, its sequence is the
        query.
    structure:
        A path to an mmCIF structure to map onto the query. Needs the `dossier`
        extra (gemmi 0.7.5) in the Python environment that runs Alambiq.
    model_number:
        The model of the structure, numbered from 1; default 1.
        `protein.predict_stability_change` numbers models from 0 in
        `model_index`.
    coordinate_annotations:
        Compute secondary structure and surface area for each residue. Needs
        `structure` and the MDTraj package (extra `trajectory`) in the Python
        environment that runs Alambiq.
    source_notes:
        Attribution for each supplied file, by input name (`fasta`, `uniprot`,
        `structure`).
    """

    fasta: str | None = None
    uniprot: str | None = None
    structure: str | None = None
    model_number: int = 1
    coordinate_annotations: bool = False
    source_notes: dict[str, SourceNote] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConfidenceSummary:
    """A ModelCIF's declared per-residue pLDDT, on its 0 to 100 scale."""

    residues: int
    missing: int
    mean: float | None
    lowest: float | None
    highest: float | None


@dataclass(frozen=True, slots=True)
class InspectResult:
    """Counts of mapped features, chains and residues, the chain mappings, the pLDDT summary, optional secondary-structure and surface-area summaries, and the stored records."""

    identifier: str
    sequence_length: int
    sequence_sha256: str
    reference_features: int
    mapped_features: int
    chains: int
    mapped_chains: int
    chain_mappings: list[dict[str, Any]]
    """For each chain in file order: its `label_asym_id` (not the author chain), mapping state and reason."""
    mapped_residues: int
    observed_residues: int
    confidence: ConfidenceSummary | None
    secondary_structure: dict[str, int] | None
    """Observed residues per DSSP code; ` ` is coil. None unless requested."""
    mean_surface_area: float | None
    """Mean absolute surface area of observed residues, in square angstrom."""
    record: str
    """The stored whole dossier: features, chains, atoms and sources."""
    confidence_record: str | None
    annotation_record: str | None


_NAMES = ("fasta", "uniprot", "structure")


def _source(
    ctx: StepContext, name: str, path: str, notes: dict[str, SourceNote]
) -> dict:
    spec: dict[str, Any] = {"text": ctx.read_text(path), "path": str(ctx.resolve(path))}
    note = notes.get(name)
    if note is not None:
        if note.source_url:
            spec["source_url"] = note.source_url
        if note.license:
            spec["license"] = note.license
        if note.citation:
            spec["receipt"] = {"citation": note.citation}
    return spec


def _confidence_summary(record: dict[str, Any] | None) -> ConfidenceSummary | None:
    if record is None:
        return None
    values = [r["value"] for r in record["residues"] if r["value"] is not None]
    return ConfidenceSummary(
        residues=len(record["residues"]),
        missing=len(record["residues"]) - len(values),
        mean=sum(values) / len(values) if values else None,
        lowest=min(values) if values else None,
        highest=max(values) if values else None,
    )


@capability(
    "protein.inspect",
    version="1.0",
    family="characterize",
    summary="Map a UniProt entry and a structure onto one query sequence",
    render=render,
)
def inspect(request: InspectRequest, ctx: StepContext) -> InspectResult:
    """Build the dossier and report what mapped and what was withheld."""
    from alambiq.capabilities.characterize.science.annotation import (
        annotate_coordinates,
    )
    from alambiq.capabilities.characterize.science.confidence import inspect_confidence
    from alambiq.capabilities.characterize.science.dossier import build_dossier

    if request.fasta is None and request.uniprot is None:
        raise ValueError(
            "fasta and uniprot are both empty; give a fasta file, a UniProt record, or both"
        )
    if request.coordinate_annotations and request.structure is None:
        raise ValueError(
            "structure is empty while coordinate_annotations is true; coordinate "
            "annotations need a structure"
        )
    if request.model_number < 1:
        raise ValueError(
            f"model_number is {request.model_number}; it must be 1 or greater, "
            "because models are numbered from 1"
        )
    unknown = set(request.source_notes) - set(_NAMES)
    if unknown:
        raise ValueError(
            f"source_notes: {sorted(unknown)} are not inputs; the allowed names "
            "are fasta, uniprot and structure"
        )

    supplied = {
        "fasta": request.fasta,
        "uniprot": request.uniprot,
        "structure": request.structure,
    }
    inputs = {
        name: _source(ctx, name, path, request.source_notes)
        for name, path in supplied.items()
        if path is not None
    }
    dossier = build_dossier(inputs, request.model_number)
    confidence = inspect_confidence(dossier, request.model_number)
    annotation = (
        annotate_coordinates(dossier, request.model_number)
        if request.coordinate_annotations
        else None
    )

    features, chains = dossier["features"], dossier["chains"]
    mapped_features = sum(f["mapping"] == "exact" for f in features)
    mapped_chains = sum(c["mapping"] == "exact" for c in chains)
    residues = [r for c in chains for r in c["residues"]]
    mapped_residues = sum(r["mapping"] == "exact" for r in residues)

    if mapped_features < len(features):
        ctx.note(
            f"{len(features) - mapped_features} of {len(features)} reference features "
            "were withheld: they fall outside the query, or their ends are not exact."
        )
    if request.structure is not None and chains and not mapped_chains:
        ctx.note(
            "No chain of the structure maps onto the query, so no residue has a "
            "query position."
        )
    elif mapped_chains < len(chains):
        ctx.note(
            f"{len(chains) - mapped_chains} of {len(chains)} polymer chains were "
            "withheld because their sequence does not match the query exactly."
        )

    codes = None
    mean_area = None
    if annotation is not None:
        observed = [
            r for r in annotation["residues"] if r["coordinate_status"] == "observed"
        ]
        codes = {}
        for row in observed:
            codes[row["secondary_structure"]] = (
                codes.get(row["secondary_structure"], 0) + 1
            )
        areas = [
            r["sasa_angstrom2"] for r in observed if r["sasa_angstrom2"] is not None
        ]
        mean_area = sum(areas) / len(areas) if areas else None

    return InspectResult(
        identifier=dossier["identifier"],
        sequence_length=len(dossier["sequence"]),
        sequence_sha256=dossier["sequence_sha256"],
        reference_features=len(features),
        mapped_features=mapped_features,
        chains=len(chains),
        mapped_chains=mapped_chains,
        chain_mappings=[
            {
                "chain": c["label_asym_id"],
                "mapping": c["mapping"],
                "reason": c["reason"],
                "query_offset": c["query_offset"],
            }
            for c in chains
        ],
        mapped_residues=mapped_residues,
        observed_residues=len(residues),
        confidence=_confidence_summary(confidence),
        secondary_structure=codes,
        mean_surface_area=mean_area,
        record=ctx.keep_json(dossier),
        confidence_record=ctx.keep_json(confidence) if confidence is not None else None,
        annotation_record=ctx.keep_json(annotation) if annotation is not None else None,
    )
