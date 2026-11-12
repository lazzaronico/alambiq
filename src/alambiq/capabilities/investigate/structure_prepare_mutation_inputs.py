"""Prepare the two legs of a thermodynamic cycle for one amino-acid substitution.

Returns, for one residue of one chain of a PDB file, the folded leg (the protein in a water
box) and the reference leg (a capped tripeptide in a smaller water box), each with a wild-type
state and a mutant state in which the side chain of the residue is replaced. The files of each
leg are bundled in one archive that `simulation.run_mutation_free_energy` reads.

The method:

- The chain is read from the first model. Other chains and heteroatoms are dropped and listed.
  The terminal OXT is added when it is missing; a chain with any other heavy atom missing, or
  with gaps in the sequence, is refused.
- Hydrogens are added at the stated pH, or in the protonation states of a protonated
  structure from `structure.assign_protonation` when the request gives one. The wild-type
  system is solvated in a dodecahedral TIP3P box (Jorgensen et al., 1983) with sodium and
  chloride ions at the ionic strength, neutralized, with ff14SB (Amber14; Maier et al.,
  2015) parameters. The random placement of ions and water orientations is seeded.
- The mutant state is the wild-type system with the side chain of the residue replaced by the
  standard template of the new residue (a PDBFixer template, one rotamer), protonated in the
  same way. Atoms of the backbone N, CA, C and O, and the atoms that follow from CA by bonds
  to atoms of the same name and element in both side chains, are mapped between the two
  states. The hydrogens of each mapped atom map in name order. The other atoms are present in
  one state only.
- The reference leg is the residue and its two neighbors on each side, with the outer two
  residues reduced to acetyl and N-methylamide caps, solvated in the same way.
- Both states of each leg have the same net charge.

Limits.

- The substitutable residues are Ala, Val, Ile, Leu, Met, Phe, Trp, Tyr, Ser, Thr, Asn, Gln and
  Cys. A charged residue, histidine, glycine and proline are not substituted. A cysteine in a
  disulfide bond is refused.
- The residue needs 2 residues on each side in the chain, joined by peptide bonds.
- The new side chain starts in one template conformation. Its conformations are sampled by the
  run that follows, and the sampling decides how far the estimate has converged.
- The unfolded state is represented by the capped tripeptide. A free energy of folding
  from this cycle reflects that choice of reference.
- Heteroatoms, cofactors and other chains are not part of the system, and a substitution
  near them changes in their absence.
- `identity_check` builds both states from the wild type, so that the free energy of the leg
  is exactly zero. It tests the pipeline that follows.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass

from alambiq.capabilities.investigate.structure_prepare_mutation_inputs_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class MutationRequest:
    """The structure, the residue, the substitution and the box settings.

    structure:
        A path to a PDB file (extension `.pdb` or `.ent`); the first model is read.
    chain_id:
        The chain of the residue, as in the PDB file.
    residue_number:
        The number of the residue in that chain, as in the PDB file.
    to:
        The new residue, as a one-letter or three-letter code.
    ph:
        The pH for the protonation of the residues, 0 to 14; default 7.
    protonated_structure:
        Optional. The protonated structure of `structure.assign_protonation`, made from this
        structure. Its residue names (`HID`, `ASH`, `LYN`, `CYX`) fix the protonation of each
        titratable residue of the chain in both states and both legs; the pH then decides only
        what the file leaves open. Without it Modeller assigns the protonation from the pH.
    folded_padding_nm, reference_padding_nm:
        The margin of the water box around the protein and around the tripeptide, 1 to 3 nm;
        default 1.0 and 1.2. The margin grows when the box would be narrower than twice the
        nonbonded cutoff plus 0.1 nm; the padding in the result is the one used.
    ionic_strength_molar:
        The salt concentration, 0 to 1 M; default 0.15.
    seed:
        The seed for the placement of ions and water, 0 to 2,147,483,647; default 1.
    identity_check:
        Whether to build the null substitution, in which the mutant state equals the wild type.
        The free energy of such a leg is zero.
    timeout_seconds:
        The longest permitted duration of the preparation, 60 s or more; default 3,600.
    """

    structure: str
    chain_id: str
    residue_number: int
    to: str
    ph: float = 7.0
    protonated_structure: str | None = None
    folded_padding_nm: float = 1.0
    reference_padding_nm: float = 1.2
    ionic_strength_molar: float = 0.15
    seed: int = 1
    identity_check: bool = False
    timeout_seconds: int = 3600


@dataclass(frozen=True, slots=True)
class LegSummary:
    """One prepared leg."""

    leg: str
    """`folded` or `reference`."""
    atoms: int
    core_atoms: int
    """Atoms of the substituted residue that map between the two states."""
    wild_type_unique_atoms: list[str]
    """Atoms of the wild-type residue that the mutant lacks."""
    mutant_unique_atoms: list[str]
    """Atoms of the mutant residue that the wild type lacks."""
    net_charge: float
    padding_nm: float


@dataclass(frozen=True, slots=True)
class MutationInputs:
    """The prepared legs and what was left out."""

    mutation: str
    """The wild-type residue, its number and the new residue, such as `ILE4VAL`."""
    chain_id: str
    residue_number: int
    wild_type: str
    mutant: str
    folded: LegSummary
    reference: LegSummary
    folded_inputs: str
    """An archive of the folded leg, for `simulation.run_mutation_free_energy`."""
    reference_inputs: str
    dropped_chains: list[str]
    dropped_heteroatoms: list[str]
    """Residue names of the heteroatom groups that are not part of the system."""
    structure_sha256: str
    versions: dict[str, str]
    seconds: float


def _summary(record: dict) -> LegSummary:
    return LegSummary(
        leg=record["leg"],
        atoms=record["atoms"],
        core_atoms=record["core_atoms"],
        wild_type_unique_atoms=record["wild_type_unique_atoms"],
        mutant_unique_atoms=record["mutant_unique_atoms"],
        net_charge=round(record["net_charge"], 6),
        padding_nm=record["padding_nm"],
    )


@capability(
    "structure.prepare_mutation_inputs",
    version="1.0",
    family="investigate",
    summary="Prepare the folded and reference legs for the free energy of one amino-acid substitution",
    render=render,
    environment="openfe",
)
def prepare_mutation_inputs(request: MutationRequest, ctx: StepContext) -> MutationInputs:
    """Check the request, build both legs, and bundle each as one archive."""
    from alambiq.capabilities.investigate.science.mutation import (
        check_preparation,
        residue_name,
    )

    target = residue_name(request.to)
    path = ctx.resolve(request.structure)
    check_preparation(
        structure=str(path),
        chain_id=request.chain_id,
        residue_number=request.residue_number,
        ph=request.ph,
        folded_padding_nm=request.folded_padding_nm,
        reference_padding_nm=request.reference_padding_nm,
        ionic_strength_molar=request.ionic_strength_molar,
        seed=request.seed,
        timeout_seconds=request.timeout_seconds,
    )
    variants = []
    if request.protonated_structure is not None:
        from alambiq.capabilities.investigate.science.protonation import (
            variants_from_structure,
        )

        variants = [
            {"residue_number": number, "variant": variant}
            for number, variant in sorted(
                variants_from_structure(ctx.resolve(request.protonated_structure), request.chain_id).items()
            )
        ]
    output = ctx.workdir / "mutation"
    found = ctx.run_in_environment(
        "openfe",
        "openmm_mutation:prepare",
        {
            "structure": str(path),
            "chain_id": request.chain_id,
            "residue_number": request.residue_number,
            "to": target,
            "ph": request.ph,
            "residue_variants": variants,
            "folded_padding_nm": request.folded_padding_nm,
            "reference_padding_nm": request.reference_padding_nm,
            "ionic_strength_molar": request.ionic_strength_molar,
            "seed": request.seed,
            "identity_check": request.identity_check,
            "output": str(output),
        },
        timeout=request.timeout_seconds,
    )
    archives = {}
    for leg in ("folded", "reference"):
        archive = ctx.workdir / f"{leg}-leg.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for member in sorted((output / leg).iterdir()):
                bundle.add(member, arcname=member.name, recursive=False)
        archives[leg] = str(ctx.keep(archive, "application/gzip").path)
    folded, reference = (found["legs"][leg] for leg in ("folded", "reference"))
    if found["dropped"]["other_chains"] or found["dropped"]["hetero"]:
        ctx.note(
            "Left out of the system: chains "
            f"{found['dropped']['other_chains'] or 'none'}, heteroatom groups "
            f"{found['dropped']['hetero'] or 'none'}."
        )
    if request.identity_check:
        ctx.note("identity_check is set: both states of each leg are the wild type.")
    return MutationInputs(
        mutation=found["mutation"],
        chain_id=request.chain_id,
        residue_number=request.residue_number,
        wild_type=folded["wild_type_residue"],
        mutant=folded["mutant_residue"],
        folded=_summary(folded),
        reference=_summary(reference),
        folded_inputs=archives["folded"],
        reference_inputs=archives["reference"],
        dropped_chains=found["dropped"]["other_chains"],
        dropped_heteroatoms=found["dropped"]["hetero"],
        structure_sha256=found["structure_sha256"],
        versions=found["versions"],
        seconds=found["seconds"],
    )
