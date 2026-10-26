"""Insert a coding sequence into a circular vector in silico and record the product.

The result reports the length of the circular product, the digest of its sequence, the designed primers with their melting temperatures, and the counts of retained and omitted vector features. The stored record also holds the sequence, an annotated GenBank file, and the reason for each omitted feature. The coding sequence of the product is translated and compared with the expected protein.

The capability models three steps:

- linearization of the vector with a named blunt-cutting restriction enzyme;
- amplification of the insert with designed primers;
- joining of the two by Gibson overlap.

Limits. Melting temperatures come from the nearest-neighbor model `pydna.tm.tm_default` of pydna 5.5.16. They describe the duplex of the annealing footprint with the template.

- The assembly is refused unless it yields exactly one circular sequence after duplicates are removed.
- The capability requires pydna 5.5.16 and Biopython 1.88, a vector with at most 500 features, and exact feature locations in the vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alambiq.capabilities.engineer.construct_assemble_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class AssembleRequest:
    """The vector, the insert and the assembly settings.

    vector:
        Path to the circular GenBank vector, 100 to 20,000 bases, each A, C, G or T.
    insert:
        Path to the insert, 60 to 6,000 bases, each A, C, G or T. By default, a FASTA
        file with a complete coding sequence that translates to `expected_protein`
        under genetic code 11. With `annotated_cassette`, a GenBank cassette. Its CDS
        must carry `transl_table` 1 or 11 and `codon_start` 1, and none of the
        qualifiers `transl_except`, `exception`, `ribosomal_slippage`, `partial`,
        `pseudo` or `pseudogene`.
    expected_protein:
        Path to the FASTA protein that the product must encode.
    annotated_cassette:
        Read the insert as a cassette: a linear GenBank record with 1 to 30 features,
        exactly one of them a CDS. All its features are annotated on the product.
    insert_features:
        At most 30 features to annotate on a FASTA insert. Each has a `label` (1 to 120
        characters) and one-based inclusive `start` and `end` within the insert. Not
        accepted with an annotated cassette.
    enzyme:
        The blunt-cutting restriction enzyme that linearizes the vector, as named in
        Biopython `Bio.Restriction`. The vector needs exactly one site for it. Default
        `SmaI`.
    overlap:
        Bases of Gibson overlap, 25 to 60. Default 35.
    primer_min:
        Minimum length of the annealing footprint, 18 to 35 bases. The footprint is the
        part of a primer that anneals to the insert. Default 18.
    target_tm:
        Target melting temperature of the primer design, 45 to 75 degrees Celsius. Default 60.
    """

    vector: str
    insert: str
    expected_protein: str
    annotated_cassette: bool = False
    insert_features: list[dict[str, Any]] = field(default_factory=list)
    enzyme: str = "SmaI"
    overlap: int = 35
    primer_min: int = 18
    target_tm: float = 60.0


@dataclass(frozen=True, slots=True)
class Primer:
    """One designed primer: the full oligo and the annealing footprint with its melting temperature."""

    id: str
    sequence: str
    footprint: str
    footprint_tm_degC: float


@dataclass(frozen=True, slots=True)
class AssembleResult:
    """The length, digest and primers of the circular product, and the counts of its retained and omitted features."""

    length: int
    protein_id: str
    sequence_sha256: str
    primers: list[Primer]
    features: int
    removed_vector_features: int
    record: str
    """Path to the stored record: the product, its GenBank file and the intermediates."""


@capability(
    "construct.assemble",
    version="1.0",
    family="engineer",
    summary="Join an insert into a circular vector and verify the product",
    render=render,
    presets={"blunt-cassette": {"annotated_cassette": True}},
)
def assemble(request: AssembleRequest, ctx: StepContext) -> AssembleResult:
    """Design the assembly and confirm that the product encodes the expected protein."""
    from alambiq.capabilities.engineer.science.assembly import assemble_construct

    record = assemble_construct(
        ctx.read_text(request.vector),
        ctx.read_text(request.insert),
        ctx.read_text(request.expected_protein),
        {
            "enzyme": request.enzyme,
            "overlap": request.overlap,
            "primer_min": request.primer_min,
            "target_tm": request.target_tm,
        },
        tuple(request.insert_features),
        insert_format="genbank" if request.annotated_cassette else "fasta",
    )

    if record["raw_circular_products"] != record["unique_circular_products"]:
        ctx.note(
            f"The assembly produced {record['raw_circular_products']} circular "
            f"products with {record['unique_circular_products']} distinct "
            "sequences."
        )
    if record["removed_vector_features"]:
        ctx.note(
            f"{len(record['removed_vector_features'])} vector features are not "
            "annotated on the product because they are outside the retained backbone."
        )
    return AssembleResult(
        length=record["length"],
        protein_id=record["protein_id"],
        sequence_sha256=record["sequence_sha256"],
        primers=[
            Primer(
                id=p["id"],
                sequence=p["sequence"],
                footprint=p["footprint"],
                footprint_tm_degC=p["footprint_tm_degC"],
            )
            for p in record["primers"]
        ],
        features=len(record["features"]),
        removed_vector_features=len(record["removed_vector_features"]),
        record=ctx.keep_json(record),
    )
