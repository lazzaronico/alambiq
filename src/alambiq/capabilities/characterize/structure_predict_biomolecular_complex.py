"""Predict the structure of 1 to 8 protein chains and an optional small-molecule ligand with Boltz-2.

The request supplies 1 to 8 protein chains by sequence, optionally an alignment
(A3M) for each chain, and optionally one ligand by SMILES (a line notation for a
molecule). Boltz-2 samples 1 to 20 structures by diffusion, each written as an
mmCIF file with the confidence estimates of the model: the overall confidence
score, predicted TM scores, the mean predicted local distance difference test
(pLDDT) and the predicted distance errors.

When a ligand is supplied, the capability reads the ligand molecule that Boltz
processed and refuses the prediction if its canonical SMILES differs from the
canonical SMILES of the supplied SMILES with hydrogens removed. It writes one
SDF file per sample, with the bond orders and charges of the processed ligand
and the atom positions of the prediction, assigned by atom name. A pose is the position of the ligand in one sample. The
capability computes the stereochemistry of each pose from its coordinates,
compares it with the canonical SMILES of the processed ligand and reports
whether they match.

What the result supports: a set of structures that the model proposes for the
sequences and the ligand, with the confidence of the model in each, and a ligand
pose with its intended bonds and charges, for use in the preparation of a
simulation.

Limits. A prediction is a hypothesis from a trained model. The confidence
scores are the model's estimates of its own accuracy. They rank the samples of
one prediction and are uncalibrated. A ligand pose from co-folding is a
predicted position of the ligand, and its confidence scores rank the samples of
one prediction. A chain without an alignment runs in
single-sequence mode, and the model is less accurate in that mode. A prediction
with fewer than 50 diffusion steps or fewer than 2 recycling passes is reported
with a note that the settings are below the standard setting of 200 steps and 3
passes and are useful only to test the pipeline; the settings of each
prediction are recorded in the result. In the repository test (CPU, 1 thread, 1
machine), a repeated request with the same seed returned the same first sample
and a different seed returned a different one. The request accepts protein chains and one ligand given by
SMILES. A request whose chains together have more than 2,000 residues is
refused. The model files total 6.2 GB (rounded) and are not distributed with
Alambiq. A step verifies the size of each file against the record and
recomputes its SHA-256 when no receipt of an earlier verification matches the
size and modification time of the file; it fails if a file is absent and does
not download it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.characterize.structure_predict_biomolecular_complex_report import (
    render,
)
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

MODEL = "boltz2-2.2.1"


@dataclass(frozen=True, slots=True)
class ProteinChain:
    """One protein chain.

    id:
        A name for the chain, unique among the chains and the ligand: a letter,
        then letters, digits or `_`, up to 5 characters in all.
    sequence:
        The amino-acid sequence, 1 to 2,000 letters from the 20 amino acids
        and X (X for an unknown residue).
    msa:
        A path to an A3M alignment for this chain whose first row is the
        sequence itself, ungapped. Every row covers the same number of query
        columns; `.` padding is not accepted. If it is omitted, the chain runs
        in single-sequence mode.
    """

    id: str
    sequence: str
    msa: str | None = None


@dataclass(frozen=True, slots=True)
class Ligand:
    """One small-molecule ligand.

    id:
        A name for the ligand, distinct from the names of the chains and of
        the same form as a chain `id` (a letter, then letters, digits or `_`,
        up to 5 characters in all).
    smiles:
        The ligand as SMILES, 1 to 4,000 characters, with no whitespace. The
        SMILES must specify the intended stereochemistry and charges.
    """

    id: str
    smiles: str


@dataclass(frozen=True, slots=True)
class PredictRequest:
    """The chains, the ligand and how Boltz-2 is run.

    proteins:
        1 to 8 chains (each with an id, a sequence and optionally an alignment
        path), at most 2,000 residues in all chains together.
    ligand:
        Zero or one ligand, given by an id and a SMILES.
    sampling_steps:
        Diffusion steps per sample, 2 to 1,000. The default, 200, is the
        standard setting of the model; a value below 50 is reported with a note
        that the settings are below the standard setting.
    diffusion_samples:
        Number of structures to sample, 1 to 20; default 1.
    recycling_steps:
        Number of recycling passes, 1 to 20; default 3. A recycling pass is
        one feedback of the output of the network trunk as its input.
    seed:
        The random seed, 0 to 4,294,967,295; default 0.
    accelerator:
        `cpu`, `gpu` or `mps`; default `cpu`. The field names the hardware
        that the protein capabilities name by their `device` field. `gpu`
        selects CUDA and corresponds to the value `cuda` of `device`. `mps` (Apple
        silicon) runs in the engine environment `boltz_mps`, which holds a Boltz
        build with a patch (Boltz pull request 527) that accepts `mps`. An
        unavailable accelerator is an error; there is no fallback to another
        accelerator.
    threads:
        Threads of the host CPU used by the engine, applied only when
        `accelerator` is `cpu`, 1 to 64; default 8.
    max_msa_seqs, num_subsampled_msa:
        Used when any chain has an alignment: for each chain, the largest
        number of alignment rows read, 1 to 32,768 (default 8192), and the
        number of rows subsampled for the model, 1 to 8,192 (default 1024).
    """

    proteins: list[ProteinChain]
    ligand: Ligand | None = None
    sampling_steps: int = 200
    diffusion_samples: int = 1
    recycling_steps: int = 3
    seed: int = 0
    accelerator: Literal["cpu", "gpu", "mps"] = "cpu"
    threads: int = 8
    max_msa_seqs: int = 8192
    num_subsampled_msa: int = 1024


@dataclass(frozen=True, slots=True)
class ChainSummary:
    """The settings used for one chain."""

    id: str
    residues: int
    alignment: str
    """`none (single sequence)`, or the number of A3M rows used."""


@dataclass(frozen=True, slots=True)
class LigandSummary:
    """The ligand as processed by Boltz."""

    id: str
    input_smiles: str
    canonical_smiles: str
    formal_charge: int
    atoms: int
    """Heavy atoms and the hydrogens of the processed molecule."""


@dataclass(frozen=True, slots=True)
class Sample:
    """One sampled structure and the confidence of the model in it."""

    index: int
    structure: str
    """The mmCIF file."""
    confidence_file: str
    confidence_score: float
    ptm: float | None
    iptm: float | None
    ligand_iptm: float | None
    protein_iptm: float | None
    complex_plddt: float | None
    complex_iplddt: float | None
    complex_pde: float | None
    complex_ipde: float | None
    chains_ptm: dict[str, float] | None
    ligand_pose: str | None
    """An SDF file: the intended bonds and charges of the ligand at the positions of this sample."""
    ligand_stereochemistry_matches: bool | None
    """Whether the pose has the stereochemistry specified by the SMILES; None without a ligand."""
    arrays: dict[str, str]
    """NPZ files of the per-residue pLDDT and the pairwise PAE and PDE."""


@dataclass(frozen=True, slots=True)
class PredictResult:
    """The sampled structures and the settings, versions and files of the prediction."""

    chains: list[ChainSummary]
    ligand: LigandSummary | None
    samples: list[Sample]
    ligand_chemistry: str | None
    """A JSON file with the graph and atom names of the processed ligand."""
    input_yaml: str
    log: str
    accelerator: str
    seed: int
    sampling_steps: int
    model: str
    versions: dict[str, str]
    seconds: float


def environment_for(request: PredictRequest) -> str:
    return "boltz_mps" if request.accelerator == "mps" else "boltz"


@capability(
    "structure.predict_biomolecular_complex",
    version="1.0",
    family="characterize",
    summary="Predict the structure of protein chains and an optional ligand with Boltz-2",
    render=render,
    environment="boltz",
    environment_for=environment_for,
    resources=(MODEL,),
)
def predict_biomolecular_complex(
    request: PredictRequest, ctx: StepContext
) -> PredictResult:
    """Write Boltz's inputs, run it, and collect and check the structures."""
    from alambiq.capabilities.characterize.science.boltz import (
        check_chains,
        check_settings,
        normalize_a3m,
        timeout_seconds,
    )

    ligand = (
        {"id": request.ligand.id, "smiles": request.ligand.smiles}
        if request.ligand
        else None
    )
    proteins = [
        {"id": p.id, "sequence": p.sequence.upper(), "msa": p.msa}
        for p in request.proteins
    ]
    check_chains(proteins, ligand)
    check_settings(
        seed=request.seed,
        sampling_steps=request.sampling_steps,
        diffusion_samples=request.diffusion_samples,
        recycling_steps=request.recycling_steps,
        max_msa_seqs=request.max_msa_seqs,
        num_subsampled_msa=request.num_subsampled_msa,
        threads=request.threads,
    )
    workdir = ctx.workdir / "boltz"
    workdir.mkdir()
    chains, prepared = [], []
    for index, protein in enumerate(proteins):
        rows = None
        path = None
        if protein["msa"] is not None:
            text, rows = normalize_a3m(
                ctx.read_text(protein["msa"]), protein["sequence"]
            )
            file = workdir / f"chain-{index:02d}.a3m"
            file.write_text(text, encoding="utf-8")
            path = str(file)
        prepared.append({**protein, "msa": path})
        chains.append(
            ChainSummary(
                id=protein["id"],
                residues=len(protein["sequence"]),
                alignment="none (single sequence)"
                if rows is None
                else f"{rows} A3M rows",
            )
        )
    if any(c.alignment.startswith("none") for c in chains):
        ctx.note("At least one chain ran from its sequence alone.")
    if request.sampling_steps < 50 or request.recycling_steps < 2:
        ctx.note(
            f"{request.sampling_steps} diffusion steps and {request.recycling_steps} "
            "recycling pass(es) are below the default settings (200 diffusion steps, 3 "
            "recycling passes)."
        )
    root = resources.verify(MODEL)
    found = ctx.run_in_environment(
        environment_for(request),
        "boltz_predict:run",
        {
            "proteins": prepared,
            "ligand": ligand,
            "model_root": str(root),
            "sampling_steps": request.sampling_steps,
            "diffusion_samples": request.diffusion_samples,
            "recycling_steps": request.recycling_steps,
            "seed": request.seed,
            "accelerator": request.accelerator,
            "threads": request.threads,
            "max_msa_seqs": request.max_msa_seqs,
            "num_subsampled_msa": request.num_subsampled_msa,
            "timeout": timeout_seconds(
                request.accelerator,
                request.diffusion_samples,
                request.sampling_steps,
                request.recycling_steps,
            ),
            "output": str(workdir / "output"),
        },
        timeout=timeout_seconds(
            request.accelerator,
            request.diffusion_samples,
            request.sampling_steps,
            request.recycling_steps,
        )
        + 600,
    )

    def keep(path: str, media: str) -> str:
        return str(ctx.keep(path, media).path)

    samples = []
    for sample in found["samples"]:
        confidence = sample["confidence"]
        samples.append(
            Sample(
                index=sample["index"],
                structure=keep(sample["structure"], "chemical/x-mmcif"),
                confidence_file=keep(sample["confidence_file"], "application/json"),
                confidence_score=confidence["confidence_score"],
                ptm=confidence["ptm"],
                iptm=confidence["iptm"],
                ligand_iptm=confidence["ligand_iptm"],
                protein_iptm=confidence["protein_iptm"],
                complex_plddt=confidence["complex_plddt"],
                complex_iplddt=confidence["complex_iplddt"],
                complex_pde=confidence["complex_pde"],
                complex_ipde=confidence["complex_ipde"],
                chains_ptm=sample["chains_ptm"],
                ligand_pose=keep(sample["ligand_sdf"], "chemical/x-mdl-sdfile")
                if "ligand_sdf" in sample
                else None,
                ligand_stereochemistry_matches=sample["ligand_pose"][
                    "stereochemistry_matches_input"
                ]
                if "ligand_pose" in sample
                else None,
                arrays={
                    kind: keep(path, "application/x-npz")
                    for kind, path in sample["arrays"].items()
                },
            )
        )
    chemistry = found["chemistry"]
    flipped = [s.index for s in samples if s.ligand_stereochemistry_matches is False]
    if flipped:
        ctx.note(
            f"In sample(s) {', '.join(map(str, flipped))} the ligand coordinates have "
            "a stereochemistry that differs from the input SMILES (stereochemistry "
            "read from the coordinates)."
        )
    return PredictResult(
        chains=chains,
        ligand=LigandSummary(
            id=chemistry["chain_id"],
            input_smiles=chemistry["input_smiles"],
            canonical_smiles=chemistry["canonical_isomeric_smiles"],
            formal_charge=chemistry["formal_charge"],
            atoms=len(chemistry["atoms"]),
        )
        if chemistry
        else None,
        samples=samples,
        ligand_chemistry=ctx.keep_json(chemistry) if chemistry else None,
        input_yaml=keep(found["input_yaml"], "application/yaml"),
        log=keep(found["log"], "text/plain"),
        accelerator=found["accelerator"],
        seed=request.seed,
        sampling_steps=request.sampling_steps,
        model=f"Boltz-2 {resources.spec(MODEL)['version']}",
        versions=found["versions"],
        seconds=found["seconds"],
    )
