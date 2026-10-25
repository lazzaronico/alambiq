"""Rank candidate proteins by their similarity to experimentally supported ones.

Every reference is aligned with MMseqs2 against every reference and candidate
(exhaustively, with every alignment retained), and each protein is scored by
the highest bit score of its alignments to a different reference. The ranking
is then judged against the named controls: positive controls, which are
references known to have the function, and negative controls, which are
candidates known not to have it (the field names `maximum_hard_negative_bits`
and `hard_negative_margin_bits` and the evidence class `hard-negative-control`
use the term hard negative). The result reports the area under the receiver
operating characteristic curve (AUROC) of the score for positive controls
against negative controls, and the AUROC of a score from sequence length alone
(the negative of the smallest length difference to another reference). The
difference between the two AUROC values is reported as
`control_auroc_lift_over_length`. A difference above 0 indicates that
similarity separates the controls better than size does.

What the result supports: how well sequence similarity to the supplied
references separates the supplied controls. The rank orders all references and
candidates by the highest bit score of their alignments to a different
reference.

Limits. The ranking depends on the references and the controls, which are
supplied in the request. The AUROC is the probability that a random positive
control outscores a random negative control, with ties counting half. For P
positive and N negative controls it takes values in steps of 1/(PN) without
tied scores and 1/(2PN) with ties; with 3 and 3 controls the steps are 1/9 and
1/18. An AUROC near 1 indicates that similarity to the references separates these
controls. For a candidate that is not a control, a high rank indicates
similarity to the references; homologs can share a fold without sharing a
reaction. Scores are comparable only between proteins ranked against the same
references. A references FASTA or a candidates FASTA with more than 2,000
proteins is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alambiq.capabilities.characterize.enzyme_retrieve_candidates_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class RetrieveRequest:
    """The references, the candidates, and the controls against which the ranking is judged.

    references:
        A path to a protein FASTA of 2 to 2,000 proteins that the request
        states to have experimental support for the function.
        Identifiers are unique and use letters, digits, underscore, dot and
        hyphen. Sequences use the letters A to Z.
    candidates:
        A path to a protein FASTA of 1 to 2,000 proteins to rank. Identifiers
        are unique and use the same characters as the identifiers of
        `references`. No identifier may also be a reference.
    positive_controls:
        References known to have the function, as asserted in the request. At
        least one is required.
    negative_controls:
        Candidates known not to have the function, as asserted in the request.
        A negative control with no alignment to a reference scores 0 bits and
        does not test the ranking. At least one is required.
    positive_min_bits:
        A positive control that scores at least this many bits counts toward
        `positive_control_recall`. At least 0; default 40.0. The value is
        independent of `minimum_retrieval_bits` of
        `enzyme.characterize_candidates`, whose default is 100.0 bits.
    sensitivity:
        MMseqs2 sensitivity (`-s`), 1 to 7.5; default 7.5. A higher value
        finds more distant relatives.
    evalue:
        The largest expectation value that the search reports (positive;
        default 10000.0). The default retains weak alignments, so that a
        protein with only a weak alignment receives a low bit score; a protein
        with no alignment receives 0 bits.
    """

    references: str
    candidates: str
    positive_controls: list[str]
    negative_controls: list[str]
    positive_min_bits: float = 40.0
    sensitivity: float = 7.5
    evalue: float = 10000.0


@dataclass(frozen=True, slots=True)
class Ranked:
    """The position of one protein in the ranking."""

    identifier: str
    rank: int
    evidence_class: str
    bits: float
    best_reference: str | None
    identity: float | None
    sequence_length: int


@dataclass(frozen=True, slots=True)
class RetrieveResult:
    """The ranking and the separation of the controls."""

    ranked: list[Ranked]
    reference_count: int
    positive_control_recall: float
    minimum_positive_bits: float
    maximum_hard_negative_bits: float
    hard_negative_margin_bits: float
    all_hard_negatives_below_positives: bool
    control_auroc: float
    length_only_baseline_auroc: float
    control_auroc_lift_over_length: float
    record: str
    """The stored ranking with sequences, which `enzyme.characterize_candidates` reads."""
    engine_versions: dict[str, Any]


@capability(
    "enzyme.retrieve_candidates",
    version="1.0",
    family="characterize",
    summary="Rank candidate proteins by similarity to experimentally supported references",
    render=render,
    environment="mmseqs",
)
def retrieve(request: RetrieveRequest, ctx: StepContext) -> RetrieveResult:
    """Align the references against the protein set and score it against the controls."""
    from alambiq.capabilities.characterize.science.enzyme import (
        check_panel,
        fasta,
        parse_proteins,
        score_panel,
    )
    from alambiq.capabilities.characterize.science.profiles import check_sensitivity

    references = parse_proteins(ctx.read_text(request.references), "references")
    candidates = parse_proteins(ctx.read_text(request.candidates), "candidates")
    check_panel(
        references, candidates, request.positive_controls, request.negative_controls
    )
    check_sensitivity(request.sensitivity)
    if request.evalue <= 0:
        raise ValueError("evalue must be positive")
    if request.positive_min_bits < 0:
        raise ValueError("positive_min_bits must not be negative")

    queries = ctx.workdir / "references.fasta"
    targets = ctx.workdir / "panel.fasta"
    queries.write_text(fasta(references))
    targets.write_text(fasta(references | candidates))
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

    scored = score_panel(
        references,
        candidates,
        found["alignments"],
        request.positive_controls,
        request.negative_controls,
        request.positive_min_bits,
    )
    if scored["positive_control_recall"] < 1.0:
        ctx.note(
            f"{scored['positive_control_recall']:.0%} of the positive controls "
            f"reached {request.positive_min_bits:g} bits."
        )
    if not scored["all_hard_negatives_below_positives"]:
        ctx.note(
            "At least one hard negative scored as high as the lowest positive control."
        )
    if scored["control_auroc_lift_over_length"] <= 0:
        ctx.note(
            "Sequence similarity separated the controls no better than protein "
            "length alone."
        )

    record = {
        "references": sorted(references),
        "positive_controls": sorted(request.positive_controls),
        "negative_controls": sorted(request.negative_controls),
        "ranked_panel": scored["ranked_panel"],
        "engine_versions": found["versions"],
    }
    return RetrieveResult(
        ranked=[
            Ranked(
                identifier=r["identifier"],
                rank=r["rank"],
                evidence_class=r["evidence_class"],
                bits=r["bits"],
                best_reference=r["best_reference"],
                identity=r["identity"],
                sequence_length=r["sequence_length"],
            )
            for r in scored["ranked_panel"]
        ],
        reference_count=scored["reference_count"],
        positive_control_recall=scored["positive_control_recall"],
        minimum_positive_bits=scored["minimum_positive_bits"],
        maximum_hard_negative_bits=scored["maximum_hard_negative_bits"],
        hard_negative_margin_bits=scored["hard_negative_margin_bits"],
        all_hard_negatives_below_positives=scored["all_hard_negatives_below_positives"],
        control_auroc=scored["control_auroc"],
        length_only_baseline_auroc=scored["length_only_baseline_auroc"],
        control_auroc_lift_over_length=scored["control_auroc_lift_over_length"],
        record=ctx.keep_json(record),
        engine_versions=found["versions"],
    )
