"""Name the substitutions that separate a variant from its closest reference protein.

Returns, for each variant, the closest reference in a supplied FASTA or in UniProtKB/
Swiss-Prot and the differences between them: substitutions in standard notation, insertions
and deletions. Positions are residue numbers of the reference. A variant that equals its
reference over its whole length has no differences.

The method:

- MMseqs2 aligns each variant to the references; up to 2,000 candidates per variant pass its
  prefilter, and the 20 with the highest bit score that meet the thresholds are compared.
- The closest reference is the one with the fewest differences from the variant: substitutions,
  inserted and deleted residues, and the residues that the alignment leaves out at the ends of
  the variant and of the reference. A tie goes to the higher bit score. The other qualifying
  references among the 20 are listed as alternatives.
- The substitutions, insertions and deletions are read from the alignment of the variant and
  its closest reference. A substitution is written reference residue, position, variant residue
  (`Q21W`), and the substitutions of one variant are joined with colons (`Q21W:Q51W`).
- The residues at either end that the alignment does not cover are counted for the variant
  and for the reference.

Limits.

- The closest reference is the most similar sequence in the source. It is the parent of the
  variant only when the source holds the parent.
- A variant that differs from every reference in many positions receives a long list in
  which engineered and natural differences are not distinguished.
- A variant with no reference that meets the thresholds is returned without a reference.
- The alignment is local. A difference at the first or last residues of the variant or of the
  reference can fall outside it; those residues are counted as unaligned at that end, and the
  variant is not marked identical. A variant that spans several domains can align to part of the
  reference.
- A variant of 10 to 5,000 residues from the 20 amino acids and X is aligned. A request with
  more than 2,000 variants or more than 2,000 references is refused.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from alambiq.capabilities.characterize.protein_call_substitutions_report import render
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

DATABASE = "swissprot-2026_03"
SEARCH_TIMEOUT_SECONDS = 2 * 3600


@dataclass(frozen=True, slots=True)
class CallRequest:
    """The variants, the source of the references and the thresholds of a reference.

    variants:
        A path to a FASTA of 1 to 2,000 variants, each 10 to 5,000 residues from the 20
        amino acids and X, with unique identifiers.
    references:
        A path to a FASTA of 1 to 2,000 reference proteins, each 10 to 5,000 residues.
        Exactly one of `references` and `database` is given.
    database:
        The pinned database of references. The only value is `swissprot-2026_03`
        (UniProtKB/Swiss-Prot release 2026_03).
    min_identity:
        A reference qualifies when the sequence identity over the aligned region is at least
        this fraction, 0 to 1; default 0.5.
    min_coverage:
        A reference qualifies when the alignment covers at least this fraction of the
        variant, 0 to 1; default 0.8.
    max_evalue:
        A reference qualifies when its expectation value is at most this value, above 0 and
        at most 1,000,000; default 1e-5.
    sensitivity:
        MMseqs2 sensitivity (`-s`), 1 to 7.5; default 5.7, the default of MMseqs2.
    threads:
        Threads used by MMseqs2, 1 to 64; default 8.
    """

    variants: str
    references: str | None = None
    database: Literal["swissprot-2026_03"] | None = None
    min_identity: float = 0.5
    min_coverage: float = 0.8
    max_evalue: float = 1e-5
    sensitivity: float = 5.7
    threads: int = 8


@dataclass(frozen=True, slots=True)
class Substitution:
    """One residue of the variant that differs from the reference."""

    reference: str
    position: int
    variant: str


@dataclass(frozen=True, slots=True)
class Insertion:
    """Residues of the variant between two residues of the reference."""

    after_position: int
    """The position of the reference residue before the insertion; 0 before the first."""
    residues: str


@dataclass(frozen=True, slots=True)
class Deletion:
    """A run of reference residues that the variant lacks."""

    start: int
    end: int
    residues: str


@dataclass(frozen=True, slots=True)
class Alternative:
    """A qualifying reference that is not the closest."""

    reference: str
    identity: float
    bits: float
    differences: int
    """The number of differing residues, counted as for the closest reference."""


@dataclass(frozen=True, slots=True)
class VariantCall:
    """The closest reference of one variant and the differences from it."""

    id: str
    length: int
    reference: str | None
    """The identifier of the closest reference; None when no reference qualifies."""
    reference_name: str
    """The recommended name of a Swiss-Prot reference; empty for a supplied reference."""
    identity: float | None
    coverage: float | None
    """The fraction of the variant that the alignment covers."""
    target_coverage: float | None
    evalue: float | None
    bits: float | None
    differences: int | None
    """The number of differing residues: substitutions, inserted and deleted residues, and
    the residues that the alignment leaves out at the ends of the variant and the reference."""
    notation: str
    """The substitutions joined with colons; empty without substitutions."""
    substitutions: list[Substitution]
    insertions: list[Insertion]
    deletions: list[Deletion]
    unaligned_variant_start: int
    unaligned_variant_end: int
    unaligned_reference_start: int
    unaligned_reference_end: int
    identical: bool
    """Whether the variant equals the reference over its whole length."""
    alternatives: list[Alternative]


@dataclass(frozen=True, slots=True)
class CallResult:
    """The call of every variant and the source of the references."""

    source: str
    """`database` and its logical identifier, or `references` and the number of sequences."""
    variants: list[VariantCall]
    matched: int
    """The variants with a closest reference."""
    unmatched: int
    thresholds: dict[str, Any]
    record: str
    engine_versions: dict[str, Any]


def _check(request: CallRequest) -> None:
    from alambiq.capabilities.characterize.science.profiles import check_sensitivity

    if (request.references is None) == (request.database is None):
        raise ValueError("exactly one of references and database is required")
    check_sensitivity(request.sensitivity)
    for name, value in (
        ("min_identity", request.min_identity),
        ("min_coverage", request.min_coverage),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not math.isfinite(request.max_evalue) or not 0 < request.max_evalue <= 1e6:
        raise ValueError("max_evalue must be above 0 and at most 1000000")
    if not 1 <= request.threads <= 64:
        raise ValueError("threads must be an integer from 1 to 64")


@capability(
    "protein.call_substitutions",
    version="1.0",
    family="characterize",
    summary="Name the substitutions that separate a variant from its closest reference protein",
    render=render,
    environment="mmseqs",
    resources=(DATABASE,),
    resources_for=lambda request: (DATABASE,) if request.database else (),
)
def call_substitutions(request: CallRequest, ctx: StepContext) -> CallResult:
    """Align each variant to the references and read its differences from the closest."""
    from alambiq.capabilities.characterize.science.enzyme import fasta, parse_proteins
    from alambiq.capabilities.characterize.science.profiles import check_query
    from alambiq.capabilities.characterize.science.substitutions import (
        call,
        count_differences,
        notation,
    )

    _check(request)
    variants = parse_proteins(ctx.read_text(request.variants), "variants")
    for name, sequence in variants.items():
        try:
            check_query(sequence)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from None
    query = ctx.workdir / "variants.fasta"
    query.write_text(fasta(variants))
    settings: dict[str, Any] = {
        "variants": str(query),
        "threads": request.threads,
        "sensitivity": request.sensitivity,
        "max_evalue": request.max_evalue,
        "min_identity": request.min_identity,
        "min_coverage": request.min_coverage,
        "timeout": SEARCH_TIMEOUT_SECONDS,
    }
    if request.database:
        root = resources.verify(DATABASE)
        pinned = resources.spec(DATABASE)
        settings["database"] = str(root / "swissprot-db")
        settings["annotations"] = str(root / "annotations.tsv")
        source = f"database {pinned['logical_id']}"
    else:
        references = parse_proteins(ctx.read_text(request.references), "references")
        for name, sequence in references.items():
            try:
                check_query(sequence)
            except ValueError as error:
                raise ValueError(f"reference {name}: {error}") from None
        reference_file = ctx.workdir / "references.fasta"
        reference_file.write_text(fasta(references))
        settings["references"] = str(reference_file)
        source = f"references {len(references)} sequences"
    found = ctx.run_in_environment(
        "mmseqs",
        "mmseqs_call:run",
        settings,
        timeout=SEARCH_TIMEOUT_SECONDS + 600,
    )

    calls: list[VariantCall] = []
    for name, sequence in variants.items():
        hits = found["hits"].get(name, [])
        if not hits:
            calls.append(
                VariantCall(
                    id=name,
                    length=len(sequence),
                    reference=None,
                    reference_name="",
                    identity=None,
                    coverage=None,
                    target_coverage=None,
                    evalue=None,
                    bits=None,
                    differences=None,
                    notation="",
                    substitutions=[],
                    insertions=[],
                    deletions=[],
                    unaligned_variant_start=0,
                    unaligned_variant_end=0,
                    unaligned_reference_start=0,
                    unaligned_reference_end=0,
                    identical=False,
                    alternatives=[],
                )
            )
            continue
        compared = []
        for hit in hits:
            read = call(
                hit["variant_aligned"],
                hit["reference_aligned"],
                variant_start=hit["variant_start"],
                variant_end=hit["variant_end"],
                variant_length=hit["variant_length"],
                reference_start=hit["reference_start"],
                reference_end=hit["reference_end"],
                reference_length=hit["reference_length"],
            )
            compared.append(
                (count_differences(read), -hit["bits"], hit["target"], hit, read)
            )
        compared.sort(key=lambda item: item[:3])
        distance, _, _, best, differences = compared[0]
        none_different = not (
            differences["substitutions"]
            or differences["insertions"]
            or differences["deletions"]
        )
        calls.append(
            VariantCall(
                id=name,
                length=len(sequence),
                reference=best["target"],
                reference_name=best["name"],
                identity=best["identity"],
                coverage=best["coverage"],
                target_coverage=best["target_coverage"],
                evalue=best["evalue"],
                bits=best["bits"],
                differences=distance,
                notation=notation(differences["substitutions"]),
                substitutions=[Substitution(**s) for s in differences["substitutions"]],
                insertions=[Insertion(**i) for i in differences["insertions"]],
                deletions=[Deletion(**d) for d in differences["deletions"]],
                unaligned_variant_start=differences["unaligned_variant_start"],
                unaligned_variant_end=differences["unaligned_variant_end"],
                unaligned_reference_start=differences["unaligned_reference_start"],
                unaligned_reference_end=differences["unaligned_reference_end"],
                identical=(
                    none_different
                    and differences["unaligned_variant_start"] == 0
                    and differences["unaligned_variant_end"] == 0
                    and differences["unaligned_reference_start"] == 0
                    and differences["unaligned_reference_end"] == 0
                ),
                alternatives=[
                    Alternative(
                        reference=h["target"],
                        identity=h["identity"],
                        bits=h["bits"],
                        differences=d,
                    )
                    for d, _, _, h, _ in compared[1:4]
                ],
            )
        )

    unmatched = [c for c in calls if c.reference is None]
    if unmatched:
        shown = ", ".join(c.id for c in unmatched[:5])
        more = f" and {len(unmatched) - 5} more" if len(unmatched) > 5 else ""
        ctx.note(
            f"{len(unmatched)} of {len(calls)} variants have no reference that meets the "
            f"thresholds ({shown}{more})."
        )
    thresholds = {
        "min_identity": request.min_identity,
        "min_coverage": request.min_coverage,
        "max_evalue": request.max_evalue,
        "sensitivity": request.sensitivity,
    }
    record = {
        "source": source,
        "settings": thresholds,
        "variants": [asdict(c) for c in calls],
        "engine_versions": found["versions"],
    }
    return CallResult(
        source=source,
        variants=calls,
        matched=len(calls) - len(unmatched),
        unmatched=len(unmatched),
        thresholds=thresholds,
        record=ctx.keep_json(record),
        engine_versions=found["versions"],
    )
