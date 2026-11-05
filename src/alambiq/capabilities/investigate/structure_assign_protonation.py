"""Assign the protonation states of a protein at a pH and place its hydrogen atoms.

Returns the structure with hydrogen atoms in Amber residue names, the pKa that PROPKA predicts
for each titratable group, the residues that the optimization of the hydrogen-bond network
turned, and the heavy atoms that had to be added.

The method:

- The first model is read. One conformation is kept for each residue (the first alternate
  location named). Water, heteroatoms and the hydrogen atoms of the file are left out.
- PROPKA 3.5 predicts a pKa for each aspartate, glutamate, arginine, lysine, histidine,
  cysteine and tyrosine, and for the termini of each chain, from the structure.
- PDB2PQR 3.7 protonates each group whose pKa is above the pH, names it as Amber does (`ASH`,
  `GLH`, `LYN`, `HIP`, `CYX`), adds the hydrogen atoms, and removes bumps.
- It then optimizes the hydrogen-bond network. A histidine takes the tautomer (`HID` or
  `HIE`) that bonds best, and the amide of asparagine and glutamine and the ring of histidine
  may turn by 180 degrees. The optimizer records which residues it turned; the result lists
  them.
- PDB2PQR detects the disulfide bonds of the structure; their cysteines are named `CYX` and
  listed apart, because PROPKA gives them no pKa.

Limits.

- A pKa from PROPKA is an empirical estimate from the structure. A group whose state matters
  for an experiment, such as the histidine of an active site, is worth a second look; the
  result lists the pKa of each group so that this is possible.
- The state assigned at one pH holds for the whole simulation that follows; the simulation
  does not titrate.
- Residues with insertion codes, and ligands, metal ions and cofactors, are not handled. A
  ligand and its neighbors are protonated as if it were absent.
- Missing heavy atoms are rebuilt by PDB2PQR from templates. They are listed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from alambiq.capabilities.investigate.structure_assign_protonation_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class ProtonationRequest:
    """The structure, the pH and whether to optimize the hydrogen-bond network.

    structure:
        A path to a PDB file (extension `.pdb` or `.ent`); the first model is read.
    ph:
        The pH at which the groups are protonated, 0 to 14; default 7.
    optimize_hydrogen_network:
        Whether to optimize the hydrogen bonds, which also lets the side chains of asparagine,
        glutamine and histidine turn by 180 degrees and fixes the histidine tautomers.
        Without it the tautomer of each histidine follows a fixed rule and no side chain
        turns. Default true.
    timeout_seconds:
        The longest permitted duration, 10 s or more; default 600.
    """

    structure: str
    ph: float = 7.0
    optimize_hydrogen_network: bool = True
    timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class Group:
    """One titratable group with its predicted pKa."""

    group: str
    """`ASP`, `GLU`, `ARG`, `LYS`, `HIS`, `CYS`, `TYR`, `N+` or `C-`."""
    chain: str
    number: int
    pka: float
    model_pka: float
    """The pKa of the group free in water, as PROPKA takes it."""
    protonated: bool
    """Whether the pKa is above the pH, so that the group carries its proton."""
    name_in: str
    """The residue name in the file."""
    name_out: str
    """The residue name in the protonated structure, such as `HID` or `ASH`."""


@dataclass(frozen=True, slots=True)
class TurnedResidue:
    """A residue whose side chain the optimization turned by 180 degrees."""

    chain: str
    number: int
    name: str


@dataclass(frozen=True, slots=True)
class AddedAtoms:
    """Heavy atoms of one residue that the file lacked."""

    chain: str
    number: int
    name: str
    atoms: list[str]


@dataclass(frozen=True, slots=True)
class ProtonationResult:
    """The protonated structure and what was assigned."""

    protonated_structure: str
    """A PDB file with hydrogen atoms in Amber residue names."""
    ph: float
    atoms: int
    hydrogens: int
    groups: list[Group]
    disulfide_cysteines: list[str]
    """Cysteines in disulfide bonds, such as `A:CYS6`; they have no pKa and are named `CYX`."""
    turned: list[TurnedResidue]
    added_heavy_atoms: list[AddedAtoms]
    dropped_heteroatoms: dict[str, int]
    """The atoms of each heteroatom group left out, by residue name."""
    dropped_waters: int
    dropped_hydrogens: int
    dropped_alternate_atoms: int
    structure_sha256: str
    versions: dict[str, str]


@capability(
    "structure.assign_protonation",
    version="1.0",
    family="investigate",
    summary="Protonate a protein at a pH with PROPKA and PDB2PQR and optimize its hydrogen bonds",
    render=render,
    environment="protonation",
)
def assign_protonation(request: ProtonationRequest, ctx: StepContext) -> ProtonationResult:
    """Check the request, run PROPKA and PDB2PQR, and keep the protonated structure."""
    from alambiq.capabilities.investigate.science.protonation import (
        check_request,
        split_disulfides,
    )

    path = ctx.resolve(request.structure)
    check_request(
        structure=str(path), ph=request.ph, timeout_seconds=request.timeout_seconds
    )
    output = ctx.workdir / "protonation"
    found = ctx.run_in_environment(
        "protonation",
        "pdb2pqr_protonate:run",
        {
            "structure": str(path),
            "ph": request.ph,
            "optimize_hydrogen_network": request.optimize_hydrogen_network,
            "output": str(output),
            "timeout": request.timeout_seconds,
        },
        timeout=request.timeout_seconds + 60,
    )
    rows, bonded = split_disulfides(found["titration"])
    kept = ctx.keep(output / found["files"]["protonated"], "chemical/x-pdb")
    if found["added_heavy_atoms"]:
        ctx.note(
            f"{len(found['added_heavy_atoms'])} residue(s) lacked heavy atoms, which were rebuilt "
            "from templates; the result lists them."
        )
    if not request.optimize_hydrogen_network:
        ctx.note("The hydrogen-bond network was not optimized; no side chain was turned.")
    dropped = found["dropped"]
    if dropped["heteroatoms"]:
        ctx.note(
            "Heteroatom groups left out: "
            + ", ".join(f"{name} ({count} atoms)" for name, count in sorted(dropped["heteroatoms"].items()))
            + ". The residues around them were protonated as if they were absent."
        )
    return ProtonationResult(
        protonated_structure=str(kept.path),
        ph=request.ph,
        atoms=found["atoms"],
        hydrogens=found["hydrogens"],
        groups=[
            Group(
                group=row["group"],
                chain=row["chain"],
                number=row["number"],
                pka=row["pka"],
                model_pka=row["model_pka"],
                protonated=row["protonated_at_ph"],
                name_in=row["name_in"],
                name_out=row["name_out"],
            )
            for row in rows
        ],
        disulfide_cysteines=bonded,
        turned=[
            TurnedResidue(chain=t["chain"], number=t["number"], name=t["name"])
            for t in found["turned"]
        ],
        added_heavy_atoms=[
            AddedAtoms(chain=a["chain"], number=a["number"], name=a["name"], atoms=a["atoms"])
            for a in found["added_heavy_atoms"]
        ],
        dropped_heteroatoms=dropped["heteroatoms"],
        dropped_waters=dropped["waters"],
        dropped_hydrogens=dropped["input_hydrogens"],
        dropped_alternate_atoms=dropped["alternate_location_atoms"],
        structure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        versions=found["versions"],
    )
