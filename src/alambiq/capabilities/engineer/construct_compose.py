"""Join ordered parts into one coding insert and check what it translates to.

Each part supplies an expected protein and, unless a codon table is given, the
coding DNA that encodes it. The parts are concatenated in the order given, a
stop codon is appended, and the complete insert is translated under genetic
code 11 and compared with the concatenated expected proteins. The first part
must begin with a start codon of genetic code 11. A part may declare that its
terminal stop is removed before joining. The span of every part in both the DNA
and the protein is recorded, one-based and inclusive.

When `codon_choices` is given, every part is encoded from the codon table:
exactly one codon per amino acid, each codon checked to encode its amino acid
under genetic code 11. A part then has no `coding_dna` (a part that has one is
refused) and `remove_terminal_stop` is not read.

Limits. The capability composes and verifies a sequence. The translation
check confirms that the DNA encodes the stated protein. The composed insert,
including the stop codon, is 60 to 6,000 bp. Codons come from the `coding_dna`
of each part or from `codon_choices`. `dna.optimize_coding` recodes
a coding sequence under constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from alambiq.capabilities.engineer.construct_compose_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class Part:
    """One ordered part of the insert.

    id, role:
        The name of the part (1 to 64 characters, unique within the insert), and its
        role: `protein`, `tag` or `linker`.
    expected_protein:
        Path to the FASTA protein that this part contributes.
    coding_dna:
        Path to the FASTA DNA that encodes the protein: whole codons, at most 6,000
        bases. Must be omitted when `codon_choices` is given; the capability
        refuses a part that has both.
    remove_terminal_stop:
        Remove the final stop codon of this part before joining it to the next
        part. When `codon_choices` is given, the value is not read: a part
        encoded from the codon table holds no stop codon, and the insert ends
        with `stop_codon` alone.
    """

    id: str
    role: Literal["protein", "tag", "linker"]
    expected_protein: str
    coding_dna: str | None = None
    remove_terminal_stop: bool = False


@dataclass(frozen=True, slots=True)
class ComposeRequest:
    """The ordered parts of the insert, its identifier and its stop codon.

    parts:
        The ordered parts, each with `id`, `role`, `expected_protein`, `coding_dna`
        and `remove_terminal_stop`. 2 to 20 parts, or 1 to 20 when `codon_choices`
        is given.
    identifier:
        The name of the composed insert: 1 to 64 characters, letters, digits, `_`,
        `.` and `-`, starting with a letter or digit.
    stop_codon:
        The stop codon appended to the insert: TAA, TAG or TGA.
    codon_choices:
        Path to a JSON object that maps each of the 20 standard amino acids, by
        one-letter code, to one uppercase ACGT codon. When given, every part is
        encoded from it, and a part that also has `coding_dna` is refused.
    """

    parts: list[Part]
    identifier: str
    stop_codon: Literal["TAA", "TAG", "TGA"]
    codon_choices: str | None = None


@dataclass(frozen=True, slots=True)
class PartSpan:
    """The span of one part in the insert: DNA and protein start and end, one-based and inclusive."""

    id: str
    role: str
    dna_start: int
    dna_end: int
    protein_start: int
    protein_end: int
    removed_stop_codon: str | None
    """The terminal stop codon removed before joining, or None if none was removed."""


@dataclass(frozen=True, slots=True)
class ComposeResult:
    """The composed insert and the span of each part in it."""

    identifier: str
    length_bp: int
    protein_length: int
    parts: list[PartSpan]
    encoded_from_codon_table: bool
    record: str
    """Path to the stored record: the insert, its DNA, its protein and the part sources."""
    dna_fasta: str = ""
    """The DNA of the insert as FASTA, for binding to `dna.optimize_coding` or `construct.assemble`."""
    protein_fasta: str = ""
    """The protein the insert encodes, as FASTA."""


def _fasta(ctx: StepContext, name: str, sequence: str, filename: str) -> str:
    path = ctx.workdir / filename
    path.write_text(f">{name}\n{sequence}\n")
    return str(ctx.keep(path, "text/x-fasta").path)


def _descriptor(ctx: StepContext, path: str) -> dict[str, Any]:
    """Return the source descriptor (text and resolved path) that the science module expects."""
    return {"text": ctx.read_text(path), "path": str(ctx.resolve(path))}


@capability(
    "construct.compose_coding",
    version="1.0",
    family="engineer",
    summary="Join ordered parts into a coding insert and verify its translation",
    render=render,
)
def compose_coding(request: ComposeRequest, ctx: StepContext) -> ComposeResult:
    """Compose the insert and confirm it translates to the expected protein."""
    from alambiq.capabilities.engineer.science.composition import compose

    parts = []
    for part in request.parts:
        supplied: dict[str, Any] = {
            "id": part.id,
            "role": part.role,
            "expected_protein": _descriptor(ctx, part.expected_protein),
        }
        if request.codon_choices is not None and part.coding_dna is not None:
            raise ValueError(
                f"Part {part.id} has coding_dna {part.coding_dna!r} together with "
                "codon_choices; supply coding_dna for every part, or codon_choices "
                "alone, not both."
            )
        if request.codon_choices is None:
            if part.coding_dna is None:
                raise ValueError(
                    f"Part {part.id} has no coding_dna. Set coding_dna, or set "
                    "codon_choices to encode the protein parts."
                )
            supplied["coding_dna"] = _descriptor(ctx, part.coding_dna)
            supplied["remove_terminal_stop"] = part.remove_terminal_stop
        parts.append(supplied)

    record = compose(
        parts,
        {"identifier": request.identifier, "stop_codon": request.stop_codon},
        codon_choices=(
            _descriptor(ctx, request.codon_choices)
            if request.codon_choices is not None
            else None
        ),
    )

    return ComposeResult(
        identifier=record["identifier"],
        length_bp=len(record["dna"]),
        protein_length=len(record["protein"]),
        parts=[
            PartSpan(
                id=p["id"],
                role=p["role"],
                dna_start=p["dna_start"],
                dna_end=p["dna_end"],
                protein_start=p["protein_start"],
                protein_end=p["protein_end"],
                removed_stop_codon=p["removed_stop_codon"],
            )
            for p in record["parts"]
        ],
        encoded_from_codon_table="encoding" in record,
        record=ctx.keep_json(record),
        dna_fasta=_fasta(ctx, request.identifier, record["dna"], "dna.fasta"),
        protein_fasta=_fasta(
            ctx, request.identifier, record["protein"], "protein.fasta"
        ),
    )
