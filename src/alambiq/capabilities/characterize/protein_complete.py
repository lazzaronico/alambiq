"""Propose new residues for chosen regions of a protein, with ESM3-small.

Returns proposals for the residues in the requested regions, sampled from
ESM3-small and conditioned on the residues outside the regions. The proposals
are hypotheses to test experimentally, for example as the starting set of a
library.

The method:

- The residues in the regions are masked. The model predicts them over the
  requested number of unmasking steps, once for each seed. Each seed gives one
  proposal.
- The generation settings are `schedule` `cosine`, `strategy` `random` and
  temperature annealing. The record stores them under `generation`.
- Residues outside the regions are given to the model unmasked.
- Each proposal is then checked independently of the model. It has the same
  length, contains only the 20 standard amino acids and changes nothing outside
  the regions.
- A proposal that passes has status `completed` and counts as accepted. One
  that fails has status `rejected` and is stored with its reason.
- An accepted proposal that repeats an earlier one, or changes no residue, is
  flagged and stays in the FASTA output.

Limits.

- The model was trained on natural proteins, and a sampled change can abolish
  activity.
- The model input is the sequence. Active-site residues are preserved when they
  are outside the regions.
- Proposals from different seeds can be identical. Repeated proposals are
  flagged.
- In the repository test, a repeated request with the same seed, device and
  package versions returned the same proposals on one device.
- A protein of more than 512 residues, or a request for more than 8 proposals,
  is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from alambiq.capabilities.characterize.protein_complete_report import render
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

RESOURCE = "esm3-small"
MODEL = "biohub/esm3-sm-open-v1"


@dataclass(frozen=True, slots=True)
class Region:
    """A region of the protein in which the model may change residues.

    start:
        First position, one-based.
    end:
        Last position, one-based and inclusive.
    """

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CompleteRequest:
    """The protein, the regions to rewrite, and how to sample.

    protein:
        A path to a FASTA with one protein of 1 to 512 residues from the 20
        standard amino acids.
    regions:
        The regions in which the model may change residues, with one-based
        inclusive positions. Regions do not overlap, and at least one residue
        is outside them.
    seed:
        Required, 0 to 4,294,967,287. Proposal `i` (numbered from 0) uses seed
        `seed + i`.
    candidates:
        Number of proposals, 1 to 8; default 4.
    temperature:
        Initial sampling temperature, above 0 and at most 2; default 1.0. It is
        annealed over the unmasking steps. A higher value gives more varied
        proposals.
    top_p:
        Cumulative probability, above 0 and at most 1, of the most probable
        residues from which each residue is sampled; default 1.0.
    num_steps:
        Unmasking steps, 1 to 64 and at most the number of masked residues.
        The default is the smaller of 8 and the number of masked residues.
    device:
        `cpu`, `mps` or `cuda`; default `cpu`. An unavailable device is an
        error.
    threads:
        Host CPU threads used by the engine on every device, 1 to 32; default
        4.
    """

    protein: str
    regions: list[Region]
    seed: int
    candidates: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    num_steps: int | None = None
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    threads: int = 4


@dataclass(frozen=True, slots=True)
class Mutation:
    """One residue changed by a proposal."""

    position: int
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class Proposal:
    """One proposal and whether it passed the independent checks."""

    id: str
    seed: int
    status: Literal["completed", "rejected"]
    reason: str | None
    sequence: str | None
    mutations: list[Mutation]
    unchanged_from_input: bool | None
    duplicate_of: str | None


@dataclass(frozen=True, slots=True)
class CompleteResult:
    """The proposals, the prompt they came from and the model that generated them."""

    protein_id: str
    length: int
    prompt: str
    """The input with each masked position shown as an underscore."""
    masked: int
    proposals: list[Proposal]
    accepted: int
    fasta: str | None
    """A FASTA of the accepted proposals, or None. `protein.embed` and `protein.inspect` read it."""
    record: str
    model: str
    revision: str
    license: str
    device: str
    engine_versions: dict[str, Any]


@capability(
    "protein.complete",
    version="1.0",
    family="characterize",
    summary="Propose new residues for chosen regions of a protein",
    render=render,
    environment="esm",
    resources=(RESOURCE,),
)
def complete(request: CompleteRequest, ctx: StepContext) -> CompleteResult:
    """Mask the regions, sample proposals, and check each one."""
    from alambiq.capabilities.characterize.science.completion import (
        build_prompt,
        check_candidate,
    )
    from alambiq.capabilities.shared.sequences import parse_fasta

    name, sequence = parse_fasta(ctx.read_text(request.protein))
    built = build_prompt(
        sequence,
        [(region.start, region.end) for region in request.regions],
        request.num_steps,
    )
    for label, value, low, high in (
        ("candidates", request.candidates, 1, 8),
        ("threads", request.threads, 1, 32),
        ("seed", request.seed, 0, 2**32 - 9),
    ):
        if not low <= value <= high:
            raise ValueError(f"{label} must be an integer from {low} to {high}")
    if not 0 < request.temperature <= 2.0:
        raise ValueError("temperature must be above 0 and at most 2")
    if not 0 < request.top_p <= 1.0:
        raise ValueError("top_p must be above 0 and at most 1")

    root = resources.verify(RESOURCE)
    pinned = resources.spec(RESOURCE)
    seeds = [request.seed + i for i in range(request.candidates)]
    found = ctx.run_in_environment(
        "esm",
        "esm3_predict:complete",
        {
            "prompt": built["prompt"],
            "num_steps": built["num_steps"],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seeds": seeds,
            "model_root": str(root),
            "package_data": pinned["files"],
            "device": request.device,
            "threads": request.threads,
        },
        timeout=4 * 3600,
    )

    proposals: list[dict[str, Any]] = []
    for index, generation in enumerate(found["generations"]):
        checked = check_candidate(
            generation["sequence"],
            sequence,
            built["mutable_positions"],
            seeds[index],
            f"proposal-{index + 1}",
        )
        if "native_error" in generation:
            checked["reason"] = (
                f"{checked.get('reason', 'rejected')}: {generation['native_error']}"
            )
        checked["duplicate_of"] = next(
            (
                earlier["id"]
                for earlier in proposals
                if checked["status"] == "completed"
                and earlier.get("sequence_sha256") == checked["sequence_sha256"]
            ),
            None,
        )
        proposals.append(checked)

    accepted = [p for p in proposals if p["status"] == "completed"]
    if not accepted:
        ctx.note("No proposal passed the checks.")
    unchanged = [p["id"] for p in accepted if p["unchanged_from_input"]]
    if unchanged:
        ctx.note(
            f"{', '.join(unchanged)} changed no residue: the model returned the "
            "original residue at every masked position."
        )
    repeated = [p["id"] for p in accepted if p["duplicate_of"]]
    if repeated:
        ctx.note(f"{', '.join(repeated)} repeat an earlier proposal.")

    fasta = "".join(f">{p['id']}\n{p['sequence']}\n" for p in accepted)
    record = {
        "input": {
            "id": name,
            "sequence": sequence,
            "prompt": built["prompt"],
            "mutable_positions": built["mutable_positions"],
        },
        "model": MODEL,
        "revision": pinned["version"],
        "resource_identity": resources.identity(RESOURCE),
        "license": pinned["license"],
        "versions": found["versions"],
        "device": found["device"],
        "generation": found["generation"],
        "candidates": proposals,
    }
    return CompleteResult(
        protein_id=name,
        length=len(sequence),
        prompt=built["prompt"],
        masked=len(built["mutable_positions"]),
        proposals=[
            Proposal(
                id=p["id"],
                seed=p["seed"],
                status=p["status"],
                reason=p.get("reason"),
                sequence=p["sequence"],
                mutations=[Mutation(**m) for m in p.get("mutations", [])],
                unchanged_from_input=p.get("unchanged_from_input"),
                duplicate_of=p["duplicate_of"],
            )
            for p in proposals
        ],
        accepted=len(accepted),
        fasta=str(ctx.keep_text(fasta, "text/x-fasta").path) if fasta else None,
        record=ctx.keep_json(record),
        model=MODEL,
        revision=pinned["version"],
        license=pinned["license"],
        device=found["device"],
        engine_versions=found["versions"],
    )
