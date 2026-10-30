"""Predict the structure of 1 to 8 protein chains and an optional small-molecule ligand with Boltz-2.

Returns 1 to 20 sampled structures for the protein chains and the ligand, each
with the confidence estimates of the model. A structure is a hypothesis. A
ligand pose has its intended bonds and charges, for use in the preparation of
a simulation.

The input:

- 1 to 8 protein chains by sequence, each with an optional A3M alignment.
- An optional ligand by SMILES, a line notation for a molecule.

The method:

- Boltz-2 samples structures by diffusion. Each is written as an mmCIF file.
- The confidence estimates are the overall confidence score, predicted TM
  scores, the mean predicted local distance difference test (pLDDT) and the
  predicted distance errors.
- With a ligand, the capability reads the ligand molecule that Boltz processed.
  It refuses the prediction when the canonical SMILES of that molecule differs
  from the canonical SMILES of the supplied SMILES with hydrogens removed.
- One SDF file per sample holds the bond orders and charges of the processed
  ligand and the atom positions of the prediction, matched by atom name. A pose
  is the position of the ligand in one sample.
- The capability computes the stereochemistry of each pose from its coordinates
  and reports whether it matches the canonical SMILES of the processed ligand.
- With `affinity`, Boltz first standardizes the ligand with the ChEMBL structure
  pipeline (largest fragment, neutralization); the structures and the estimate
  refer to the standardized molecule, and a changed molecule carries a note.
  The affinity module of Boltz-2 then estimates the binding of the ligand to
  the protein chains from the highest-ranked sample. It returns the
  log10 of an IC50 in µM (lower values mean stronger binding: 0 is 1 µM) and
  the probability that the ligand binds, from an ensemble of two models and
  from each model.

Limits.

- The confidence scores are the model's estimates of its own accuracy. They
  rank the samples of one prediction and are uncalibrated.
- A chain without an alignment runs in single-sequence mode, in which the model
  is less accurate.
- A prediction with fewer than 50 diffusion steps or fewer than 2 recycling
  passes carries a note. The standard setting is 200 steps and 3 passes, and
  the lower settings serve to test the pipeline. The result records the
  settings.
- In the repository test (CPU, 1 thread, 1 machine), a repeated request with the
  same seed returned the same first sample, and a different seed returned a
  different one.
- A request whose chains together have more than 2,000 residues is refused.
- Affinity is estimated for one small-molecule ligand and protein chains, for a
  ligand of at most 128 atoms (heavy atoms and the hydrogens that RDKit keeps).
  Boltz advises against ligands much larger than 56 atoms; such a ligand
  carries a note.
- Following the Boltz documentation, the binder probability separates binders
  from decoys, and the IC50 value compares molecules that are active.
- The model files total 6.2 GB (rounded) and are not distributed with Alambiq.
  A step compares the size of each file with the record. It recomputes the
  SHA-256 when no receipt of an earlier verification matches the size and
  modification time of the file. It fails if a file is absent and does not
  download it.
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
        A path to an A3M alignment for this chain. Its first row is the
        ungapped sequence itself. Every row covers the same number of query
        columns. `.` padding is refused. Without it, the chain runs in
        single-sequence mode.
    """

    id: str
    sequence: str
    msa: str | None = None


@dataclass(frozen=True, slots=True)
class Ligand:
    """One small-molecule ligand.

    id:
        A name for the ligand, in the form of a chain `id` and distinct from the
        chain names.
    smiles:
        The ligand as SMILES, 1 to 4,000 characters, with no whitespace. The
        SMILES must specify the intended stereochemistry and charges.
    """

    id: str
    smiles: str


@dataclass(frozen=True, slots=True)
class PredictRequest:
    """The chains, the ligand and the Boltz-2 settings.

    proteins:
        1 to 8 chains (each with an id, a sequence and optionally an alignment
        path), at most 2,000 residues in all chains together.
    ligand:
        Zero or one ligand, given by an id and a SMILES.
    sampling_steps:
        Diffusion steps per sample, 2 to 1,000. The default, 200, is the
        standard setting of the model. A value below 50 adds a note.
    diffusion_samples:
        Number of structures to sample, 1 to 20; default 1.
    recycling_steps:
        Number of recycling passes, 1 to 20; default 3. A pass feeds the output
        of the network trunk back as its input.
    seed:
        The random seed, 0 to 4,294,967,295; default 0.
    accelerator:
        `cpu`, `gpu` or `mps`; default `cpu`. The protein capabilities name this
        hardware by their `device` field. `gpu` selects CUDA, the value `cuda`
        of `device`. `mps` (Apple silicon) runs in the engine environment
        `boltz_mps`, which holds a Boltz build with a patch (Boltz pull request
        527) that accepts `mps`. An unavailable accelerator is an error.
    threads:
        Host CPU threads used by the engine, only when `accelerator` is `cpu`,
        1 to 64; default 8.
    max_msa_seqs, num_subsampled_msa:
        Used when any chain has an alignment. For each chain, the most alignment
        rows read, 1 to 32,768 (default 8192), and the rows subsampled for the
        model, 1 to 8,192 (default 1024).
    affinity:
        Whether to estimate the binding affinity of the ligand; requires a
        ligand. Default false.
    affinity_sampling_steps, affinity_samples:
        Diffusion steps (2 to 1,000; default 200) and samples (1 to 20; default
        5) of the affinity pass, the Boltz defaults.
    affinity_mw_correction:
        Whether Boltz adds its molecular-weight correction to the affinity
        value. Default false.
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
    affinity: bool = False
    affinity_sampling_steps: int = 200
    affinity_samples: int = 5
    affinity_mw_correction: bool = False


@dataclass(frozen=True, slots=True)
class AffinityEstimate:
    """One estimate of binding affinity."""

    log10_ic50_um: float
    """The log10 of the IC50 in µM; lower values mean stronger binding."""
    binder_probability: float
    """The probability that the ligand binds, from 0 to 1."""


@dataclass(frozen=True, slots=True)
class Affinity:
    """The affinity of the ligand estimated by Boltz-2, and the settings of the pass."""

    log10_ic50_um: float
    """The ensemble estimate: the log10 of the IC50 in µM."""
    binder_probability: float
    """The ensemble probability that the ligand binds."""
    models: list[AffinityEstimate]
    """The estimates of the two models of the ensemble."""
    sampling_steps: int
    samples: int
    mw_correction: bool
    ligand_atoms: int
    """Atoms of the ligand as Boltz counts them for affinity."""
    file: str
    """The JSON file that Boltz wrote."""


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
    """An SDF file with the intended bonds and charges of the ligand at the positions of this sample."""
    ligand_stereochemistry_matches: bool | None
    """Whether the pose has the stereochemistry of the SMILES; None without a ligand."""
    arrays: dict[str, str]
    """NPZ files of the per-residue pLDDT and the pairwise PAE and PDE."""


@dataclass(frozen=True, slots=True)
class PredictResult:
    """The sampled structures, and the settings, versions and files of the prediction."""

    chains: list[ChainSummary]
    ligand: LigandSummary | None
    samples: list[Sample]
    ligand_chemistry: str | None
    """A JSON file with the graph and atom names of the processed ligand."""
    affinity: Affinity | None
    """The affinity estimate; None unless `affinity` was requested."""
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
    version="1.1",
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
        check_affinity,
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
    check_affinity(request.affinity, ligand)
    check_settings(
        seed=request.seed,
        sampling_steps=request.sampling_steps,
        diffusion_samples=request.diffusion_samples,
        recycling_steps=request.recycling_steps,
        max_msa_seqs=request.max_msa_seqs,
        num_subsampled_msa=request.num_subsampled_msa,
        threads=request.threads,
        affinity_sampling_steps=request.affinity_sampling_steps,
        affinity_samples=request.affinity_samples,
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
    ceiling = timeout_seconds(
        request.accelerator,
        request.diffusion_samples,
        request.sampling_steps,
        request.recycling_steps,
        request.affinity_samples if request.affinity else 0,
        request.affinity_sampling_steps,
    )
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
            "affinity": request.affinity,
            "affinity_sampling_steps": request.affinity_sampling_steps,
            "affinity_samples": request.affinity_samples,
            "affinity_mw_correction": request.affinity_mw_correction,
            "timeout": ceiling,
            "output": str(workdir / "output"),
        },
        timeout=ceiling + 600,
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
    estimate = found.get("affinity")
    affinity = None
    if chemistry and chemistry.get("standardized"):
        ctx.note(
            f"For affinity, Boltz standardized the ligand {chemistry['input_smiles']} to "
            f"{chemistry['canonical_isomeric_smiles']}; the poses and the affinity refer to "
            "the standardized molecule."
        )
    if estimate is not None:
        if estimate["ligand_atoms"] > 56:
            ctx.note(
                f"The ligand has {estimate['ligand_atoms']} atoms; Boltz advises against "
                "affinity for ligands much larger than 56 atoms."
            )
        affinity = Affinity(
            log10_ic50_um=estimate["log10_ic50_um"],
            binder_probability=estimate["binder_probability"],
            models=[AffinityEstimate(**m) for m in estimate["models"]],
            sampling_steps=request.affinity_sampling_steps,
            samples=request.affinity_samples,
            mw_correction=request.affinity_mw_correction,
            ligand_atoms=estimate["ligand_atoms"],
            file=keep(estimate["file"], "application/json"),
        )
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
        affinity=affinity,
        input_yaml=keep(found["input_yaml"], "application/yaml"),
        log=keep(found["log"], "text/plain"),
        accelerator=found["accelerator"],
        seed=request.seed,
        sampling_steps=request.sampling_steps,
        model=f"Boltz-2 {resources.spec(MODEL)['version']}",
        versions=found["versions"],
        seconds=found["seconds"],
    )
