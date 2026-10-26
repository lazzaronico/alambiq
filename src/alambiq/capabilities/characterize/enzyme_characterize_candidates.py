"""Check whether candidates retain the residues that the literature associates
with function.

Returns a decision for each protein of the set from `enzyme.retrieve_candidates`:
whether it has the amino acids that the literature names at the aligned
positions. The positive and negative controls of that record are decided by the
same rule. The result reports the fraction of positive controls that advance and
the fraction of negative controls that are rejected.

The input is the ranked set, one reference protein and the residues of that
reference that published work associates with its function. These are the named
residues, in the request field `features`.

The method:

- MMseqs2 aligns each protein to the reference. The reference is part of the set
  and is aligned to itself.
- The named residues are read from the alignment.
- Each protein receives the first applicable decision below. The thresholds are
  fields of the request.

1. Reject, if it has no alignment to the reference. The reason code is
   `no-reference-alignment` alone, and the retrieval score is not tested.
2. Reject, if its retrieval score is below `minimum_retrieval_bits`. The score is
   the highest bit score of its alignments to a different reference in the
   `enzyme.retrieve_candidates` record.
3. Withhold, if the alignment covers less than `minimum_query_coverage` of the
   reference.
4. Withhold, if the fraction of named residues inside the alignment is below
   `minimum_mapped_feature_fraction`.
5. Reject, if the fraction of named residues that are the expected amino acid is
   below `minimum_conserved_feature_fraction`.
6. Advance.

Limits.

- The request supplies the named residues. The capability checks that the
  reference has them and records the citations as supplied.
- A match between the aligned residue and the named residue indicates that the
  candidate has the named amino acid at the aligned position. A residue can
  match by chance in a weak alignment.
- A named residue is transferred correctly when the alignment is correct at that
  position. Alignment accuracy decreases as sequence identity decreases.
- The result reports the sequence identity and the coverage of the alignment
  behind each decision.
- An insertion or deletion between two named residues can shift the aligned
  position of one of them.
- A decision changes when a threshold changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alambiq.capabilities.characterize.enzyme_characterize_candidates_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

CLASSES = {
    "positive-control",
    "hard-negative-control",
    "reference",
    "unresolved-candidate",
}


@dataclass(frozen=True, slots=True)
class Feature:
    """A residue of the reference that the literature associates with its function.

    position:
        One-based position in the reference sequence.
    residue:
        The amino acid of the reference at that position, as one uppercase
        letter. The capability checks it against the sequence.
    role:
        The function of the residue, as a short phrase.
    citation:
        The publication or database entry that establishes the role.
    """

    position: int
    residue: str
    role: str
    citation: str


@dataclass(frozen=True, slots=True)
class CharacterizeRequest:
    """The retrieved protein set, the reference to transfer from and the thresholds.

    retrieval:
        A path to the stored record of `enzyme.retrieve_candidates`.
    reference_id:
        The reference whose named residues are transferred. It is one of the
        references of the `retrieval` record.
    features:
        The named residues of the reference, which the literature associates
        with its function. At least one is required.
    minimum_retrieval_bits:
        A candidate whose retrieval score in the record is below this many bits
        is rejected. At least 0; default 100.0.
    minimum_query_coverage:
        The smallest fraction of the reference that the alignment covers for a
        valid transfer. Candidates below it are withheld. 0 to 1; default 0.8.
    minimum_mapped_feature_fraction:
        The smallest fraction of named residues inside the alignment and not on
        a gap. Candidates below it are withheld. 0 to 1; default 1.0.
    minimum_conserved_feature_fraction:
        The smallest fraction of named residues that are the expected residue.
        Candidates below it are rejected. 0 to 1; default 1.0.
    sensitivity, evalue:
        MMseqs2 sensitivity (`-s`), 1 to 7.5, default 7.5; and the largest
        expectation value reported, positive, default 10000.0.
    """

    retrieval: str
    reference_id: str
    features: list[Feature]
    minimum_retrieval_bits: float = 100.0
    minimum_query_coverage: float = 0.8
    minimum_mapped_feature_fraction: float = 1.0
    minimum_conserved_feature_fraction: float = 1.0
    sensitivity: float = 7.5
    evalue: float = 10000.0


@dataclass(frozen=True, slots=True)
class Assessment:
    """The decision for one protein, with its reasons. `query_coverage` is the fraction of the reference that the alignment covers. `identity` is the sequence identity of the alignment."""

    identifier: str
    evidence_class: str
    decision: str
    reason_codes: list[str]
    retrieval_bits: float
    identity: float
    query_coverage: float
    mapped_feature_fraction: float
    conserved_feature_fraction: float
    residues: list[dict[str, Any]]
    """For each named residue: the aligned position, the residue of the protein there and whether it matches."""


@dataclass(frozen=True, slots=True)
class CharacterizeResult:
    """The decision for every protein of the set, the fractions of controls decided as expected and the thresholds applied."""

    reference_id: str
    assessments: list[Assessment]
    advanced: list[str]
    withheld: list[str]
    rejected: list[str]
    positive_control_advance_rate: float
    negative_control_rejection_rate: float
    unresolved_candidates_advanced: int
    policy: dict[str, float]
    record: str
    engine_versions: dict[str, Any]


@capability(
    "enzyme.characterize_candidates",
    version="1.0",
    family="characterize",
    summary="Check candidates for the residues that the literature associates with function",
    render=render,
    environment="mmseqs",
)
def characterize(request: CharacterizeRequest, ctx: StepContext) -> CharacterizeResult:
    """Transfer the named residues of the reference onto every protein in the set and decide."""
    from alambiq.capabilities.characterize.science.enzyme import (
        assess_candidates,
        check_features,
        fasta,
    )
    from alambiq.capabilities.characterize.science.profiles import check_sensitivity

    retrieved = ctx.read_json(request.retrieval)
    panel = retrieved["ranked_panel"]
    if any(item["evidence_class"] not in CLASSES for item in panel):
        raise ValueError(
            "retrieval: the record holds an unrecognized evidence class; the "
            f"recognized classes are {', '.join(sorted(CLASSES))}"
        )
    sequences = {item["identifier"]: item["sequence"] for item in panel}
    if request.reference_id not in retrieved["references"]:
        raise ValueError(
            f"reference_id {request.reference_id!r} must be one of the references "
            "in the retrieval record"
        )
    features = [
        {
            "position": f.position,
            "residue": f.residue,
            "role": f.role,
            "citation": f.citation,
        }
        for f in request.features
    ]
    check_features(features, sequences[request.reference_id])
    for name in (
        "minimum_query_coverage",
        "minimum_mapped_feature_fraction",
        "minimum_conserved_feature_fraction",
    ):
        if not 0 <= getattr(request, name) <= 1:
            raise ValueError(f"{name} must be a number between 0 and 1")
    if request.minimum_retrieval_bits < 0:
        raise ValueError("minimum_retrieval_bits must not be negative")
    check_sensitivity(request.sensitivity)
    if request.evalue <= 0:
        raise ValueError("evalue must be positive")

    queries = ctx.workdir / "reference.fasta"
    targets = ctx.workdir / "panel.fasta"
    queries.write_text(fasta({request.reference_id: sequences[request.reference_id]}))
    targets.write_text(fasta(sequences))
    found = ctx.run_in_environment(
        "mmseqs",
        "mmseqs_panel:run",
        {
            "queries": str(queries),
            "targets": str(targets),
            "sensitivity": request.sensitivity,
            "evalue": request.evalue,
        },
    )

    policy = {
        "minimum_retrieval_bits": request.minimum_retrieval_bits,
        "minimum_query_coverage": request.minimum_query_coverage,
        "minimum_mapped_feature_fraction": request.minimum_mapped_feature_fraction,
        "minimum_conserved_feature_fraction": request.minimum_conserved_feature_fraction,
    }
    decided = assess_candidates(panel, features, found["alignments"], policy)
    by_decision = {
        kind: sorted(a["identifier"] for a in decided if a["decision"] == kind)
        for kind in ("advance", "withhold", "reject")
    }
    by_class: dict[str, list[dict[str, Any]]] = {}
    for a in decided:
        by_class.setdefault(a["evidence_class"], []).append(a)
    positives = by_class["positive-control"]
    negatives = by_class["hard-negative-control"]
    positive_rate = sum(a["decision"] == "advance" for a in positives) / len(positives)
    negative_rate = sum(a["decision"] == "reject" for a in negatives) / len(negatives)
    if positive_rate < 1.0:
        ctx.note(f"{positive_rate:.0%} of the positive controls advanced.")
    if negative_rate < 1.0:
        ctx.note(f"{negative_rate:.0%} of the hard negatives were rejected.")
    unresolved = by_class.get("unresolved-candidate", [])
    record = {
        "reference_id": request.reference_id,
        "features": features,
        "policy": policy,
        "assessments": decided,
        "engine_versions": found["versions"],
    }
    return CharacterizeResult(
        reference_id=request.reference_id,
        assessments=[
            Assessment(
                identifier=a["identifier"],
                evidence_class=a["evidence_class"],
                decision=a["decision"],
                reason_codes=a["reason_codes"],
                retrieval_bits=a["retrieval_bits"],
                identity=a["identity"],
                query_coverage=a["query_coverage"],
                mapped_feature_fraction=a["mapped_feature_fraction"],
                conserved_feature_fraction=a["conserved_feature_fraction"],
                residues=a["residue_transfers"],
            )
            for a in decided
        ],
        advanced=by_decision["advance"],
        withheld=by_decision["withhold"],
        rejected=by_decision["reject"],
        positive_control_advance_rate=round(positive_rate, 6),
        negative_control_rejection_rate=round(negative_rate, 6),
        unresolved_candidates_advanced=sum(
            a["decision"] == "advance" for a in unresolved
        ),
        policy=policy,
        record=ctx.keep_json(record),
        engine_versions=found["versions"],
    )
