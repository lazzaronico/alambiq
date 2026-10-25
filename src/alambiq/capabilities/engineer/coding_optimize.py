"""Recode a coding sequence without changing the protein it encodes.

The solver is DNA Chisel. It changes synonymous codons only, leaves the first
and last codon unchanged, removes named motifs and constrains GC content to
stated bounds. Every constraint is then re-checked on the output of the solver:
the translation under genetic code 11, the terminal codons, each avoided motif
on both strands, and the GC fraction of the whole sequence or, when `gc_window`
is set, of every window of that length. When the solver output fails any of
those checks, the capability raises an error instead of returning a sequence.

When the solver finds no solution, the result has status `failed` and the
stored record contains the rejected sequence under `rejected_candidate`.
`dna_fasta` is empty, so that a downstream capability cannot receive the
rejected sequence by a binding.

Limits. The coding sequence is 60 to 6,000 bases, including the terminal stop
codon. The avoided motifs and the GC bounds are supplied in the request; the host's
restriction sites, codon preferences and expression behavior enter only through
them. Synonymous recoding can change translation efficiency, mRNA folding and
regulatory elements; the effect of a recoded sequence on expression is
determined by experiment. The seed makes the search reproducible for the same
package versions, which the stored record lists. A search with status
`completed` returns a sequence that satisfies every constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alambiq.capabilities.engineer.coding_optimize_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class CodingDesignRequest:
    """The sequence to recode and the constraints on the result.

    coding_dna:
        Path to the FASTA coding sequence, 60 to 6,000 bases, each A, C, G or T: a
        complete coding sequence of whole codons from the start codon to one terminal
        stop codon, which translates under genetic code 11 with `cds=True`.
    expected_protein:
        Path to the FASTA protein that the sequence encodes, without the stop codon.
        The result must encode the same protein.
    seed:
        Integer seed of the search, 0 to 4,294,967,295; the same seed reproduces the
        reported sequence for the same package versions.
    avoid_patterns:
        Up to 20 unique literal ACGT motifs of 4 to 30 bases to remove, checked on
        both strands.
    gc_min, gc_max:
        Lower and upper bound of the GC fraction (0 to 1); 0 and 1 by default.
    gc_window:
        Window length in bases, 3 to 6,000 and at most the sequence length. When
        set, the GC bounds apply to every window of this length instead of to the
        sequence as a whole.
    """

    coding_dna: str
    expected_protein: str
    seed: int
    avoid_patterns: list[str] = field(default_factory=list)
    gc_min: float = 0.0
    gc_max: float = 1.0
    gc_window: int | None = None


@dataclass(frozen=True, slots=True)
class ConstraintState:
    """Whether a sequence meets the stated constraints, with the GC statistics and the counts of violations."""

    satisfied: bool
    gc_fraction: float
    minimum_window_gc: float
    maximum_window_gc: float
    motif_matches: int
    gc_violations: int


@dataclass(frozen=True, slots=True)
class CodingDesignResult:
    """The status, the number of changed codons, the constraint states before and after, and the paths of the stored record and of the recoded FASTA."""

    status: str
    """`completed`, or `failed` when the constraints could not all be met."""
    changed_codons: int
    before: ConstraintState
    after: ConstraintState
    record: str
    """Path to the stored record: the sequences, the per-codon changes and the solver log."""
    dna_fasta: str = ""
    """The recoded DNA as FASTA, empty when the constraints could not be met."""


def _dna_fasta(ctx: StepContext, record: dict) -> str:
    path = ctx.workdir / "recoded.fasta"
    path.write_text(f">recoded\n{record['dna_after']}\n")
    return str(ctx.keep(path, "text/x-fasta").path)


def _state(values: dict) -> ConstraintState:
    return ConstraintState(
        satisfied=values["satisfied"],
        gc_fraction=values["gc_fraction"],
        minimum_window_gc=values["minimum_window_gc"],
        maximum_window_gc=values["maximum_window_gc"],
        motif_matches=len(values["motif_matches"]),
        gc_violations=len(values["gc_violations"]),
    )


@capability(
    "dna.optimize_coding",
    version="1.0",
    family="engineer",
    summary="Recode a coding sequence synonymously under explicit constraints",
    render=render,
)
def optimize_coding(
    request: CodingDesignRequest, ctx: StepContext
) -> CodingDesignResult:
    """Recode the sequence and re-check every constraint on the result."""
    from alambiq.capabilities.engineer.science.coding import design
    from alambiq.capabilities.shared.sequences import parse_fasta

    dna = parse_fasta(ctx.read_text(request.coding_dna))[1]
    protein = parse_fasta(ctx.read_text(request.expected_protein))[1]
    record = design(
        dna,
        protein,
        {
            "seed": request.seed,
            "avoid_patterns": list(request.avoid_patterns),
            "gc_min": request.gc_min,
            "gc_max": request.gc_max,
            "gc_window": request.gc_window,
        },
    )

    if record["status"] != "completed":
        ctx.note(
            "The constraints could not all be met: "
            f"{record['error']}. The rejected candidate is stored for "
            "inspection under rejected_candidate; dna_fasta is empty."
        )
    return CodingDesignResult(
        status=record["status"],
        changed_codons=len(record.get("changed_codons", ())),
        before=_state(record["before"]),
        after=_state(record["after"]),
        record=ctx.keep_json(record),
        dna_fasta=_dna_fasta(ctx, record) if record["status"] == "completed" else "",
    )
