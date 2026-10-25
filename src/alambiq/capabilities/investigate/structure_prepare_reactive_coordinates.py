"""Build a model of an enzyme with missing terminal residues and a bound substrate from deposited structures.

The capability builds the coordinates of an enzyme with its substrate bound, from four
inputs: a target structure (an assembly of one or more chains, with the termini that the
crystal did not resolve), a donor structure that contains those termini, a structure
with the substrate (or a close analogue) deposited, and the substrate as an explicit
molecule. The anchor residues of a terminus are the observed residues next to the gap
(`terminal_model.anchor_residues` of them). The capability copies the missing terminal
residues of each target chain from the donor chain with the lowest RMSD after
superposition on the anchor residues of the target, among the donor chains that meet all
of these conditions: the RMSD of the fit is below the declared maximum, the length of
the new peptide bond is in the declared range, and no modeled atom is closer than the
declared clash distance to another protein atom. It refuses the request if no donor
chain meets them. The substrate takes the coordinates of the deposited ligand through
the declared atom mapping and is transferred to the target by aligning an analogue in
the active site. A force field adds hydrogens to the
substrate with every heavy atom held fixed. Histidine protonation states are declared in
the request, and waters closer than the declared exclusion distance to a modeled atom
are removed.

The result gives heavy-atom coordinates of a protein-substrate complex, with hydrogens
on the substrate, in which every deposited atom is at its deposited position, every
modeled atom is traceable to a donor, and the substrate's chemistry is that of the
supplied molecule, with the geometric overlap of substrate and protein reported.

Limits. The modeled termini and the transferred substrate are geometric models. A
terminus takes the conformation of the donor, which can differ from the conformation in
the target at the same sequence, and a substrate positioned by aligning an analogue takes
the position of the analogue. The protonation of histidines is as declared. The protonation of the
other ionizable groups is set by the preparation capability that reads this result (for
example `structure.prepare_protein_ligand_system`). The hydrogens of the substrate come
from the geometry of a force field with all heavy atoms held fixed. Relaxation,
parameterization and dynamics are done by other capabilities. Only
canonical protein residues with all their heavy atoms are accepted (the terminal oxygen
excepted). A request with more than 4,000 residues or more than 64 modeled terminal
residues is refused.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from alambiq.capabilities.investigate.structure_prepare_reactive_coordinates_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class TargetChain:
    """One chain of the target assembly.

    source_chain:
        Its name in the built assembly: the structure's chain name followed by the number
        of the copy, so `A1` for the first copy of chain `A`.
    id:
        The one-character name it has in the prepared model.
    sequence:
        The complete canonical sequence, 4 to 2,000 letters.
    missing_residues:
        The one-based positions the crystal did not resolve; they must be terminal
        groups around one observed range.
    histidines:
        For every histidine, by one-based position as text, its state: `HID`, `HIE` or
        `HIP`.
    """

    source_chain: str
    id: str
    sequence: str
    missing_residues: list[int] = field(default_factory=list)
    histidines: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Target:
    """The assembly of the target structure.

    assembly_id:
        The biological assembly to build.
    model_number:
        The deposited model to use.
    chains:
        1 to 16 chains of the assembly.
    """

    assembly_id: str
    model_number: int
    chains: list[TargetChain]


@dataclass(frozen=True, slots=True)
class Donor:
    """The donor structure's chains.

    model_number:
        The deposited model to use.
    chains:
        The names of the donor chains that can supply termini; each must hold the exact
        complete target sequence.
    """

    model_number: int
    chains: list[str]


@dataclass(frozen=True, slots=True)
class TerminalModel:
    """What makes a donor terminus acceptable.

    anchor_residues:
        How many adjacent observed residues next to the gap are fitted, 1 to 10.
    max_fit_rmsd_angstrom:
        The largest backbone RMSD of that fit.
    junction_min_angstrom, junction_max_angstrom:
        The range the new peptide bond's length must fall in.
    clash_distance_angstrom:
        Modeled atoms may be no closer than this to other protein atoms.
    water_exclusion_angstrom:
        Waters closer than this to a modeled atom are removed.
    """

    anchor_residues: int = 3
    max_fit_rmsd_angstrom: float = 1.0
    junction_min_angstrom: float = 1.2
    junction_max_angstrom: float = 1.5
    clash_distance_angstrom: float = 2.0
    water_exclusion_angstrom: float = 2.5


@dataclass(frozen=True, slots=True)
class Substrate:
    """The substrate and where its coordinates come from.

    model_number, chain, residue_number, insertion_code, component, altloc:
        The deposited residue that supplies the substrate's coordinates: model, chain,
        number, insertion code (blank if none), component name, and alternate location
        (blank if none).
    source_form:
        A line of text describing the deposited construct, for the record.
    formal_charge:
        The substrate's net charge, which the molecule must have.
    source_atom_names:
        The deposited heavy atoms' names, in the order of the atoms of the molecule.
    anchor_component, anchor_atom_names:
        The residue name of the analogue in each target chain's active site and
        its heavy atoms in the same order, by which the substrate is transferred.
    hydrogen_max_iterations:
        The maximum number of force-field iterations to place the hydrogens, 1 to 10,000.
    """

    model_number: int
    chain: str
    residue_number: int
    component: str
    source_form: str
    formal_charge: int
    source_atom_names: list[str]
    anchor_component: str
    anchor_atom_names: list[str]
    insertion_code: str = ""
    altloc: str = ""
    hydrogen_max_iterations: int = 1000


@dataclass(frozen=True, slots=True)
class CoordinateRequest:
    """The four sources and the choices that combine them.

    structure:
        The target structure, mmCIF.
    donor_structure:
        The structure whose chains supply missing termini, mmCIF.
    substrate_source:
        The structure holding the deposited substrate and its component bond table, mmCIF.
    molecule:
        The substrate as a MOL file with its heavy atoms in the order of
        `substrate.source_atom_names`, explicit bonds and stereochemistry, no hydrogens
        required.
    target:
        The assembly, the model number and the chains of the target structure that
        receive termini.
    donor:
        The model number and the chains of the donor structure.
    substrate:
        The deposited residue that supplies the substrate coordinates, and the atom
        mapping to the molecule.
    terminal_model:
        The thresholds that a modeled terminus must meet.
    """

    structure: str
    donor_structure: str
    substrate_source: str
    molecule: str
    target: Target
    donor: Donor
    substrate: Substrate
    terminal_model: TerminalModel = field(default_factory=TerminalModel)


@dataclass(frozen=True, slots=True)
class ModeledTerminus:
    """One terminus that was modeled from a donor."""

    target_chain: str
    terminus: str
    """`N` or `C`."""
    modeled_residues: list[int]
    donor_chain: str
    backbone_fit_rmsd_angstrom: float
    junction_bond_angstrom: float
    candidates_considered: int


@dataclass(frozen=True, slots=True)
class PositionedSubstrate:
    """The substrate as placed in one target chain."""

    target_chain: str
    sdf: str
    heavy_atoms: int
    atoms: int
    """With hydrogens."""
    analogue_rmsd_angstrom: float
    protein_overlap_score_angstrom2: float
    """Summed squared shortfall below 2.2 A between substrate and protein heavy atoms."""
    canonical_smiles: str
    formal_charge: int


@dataclass(frozen=True, slots=True)
class CoordinateResult:
    """The modeled complex and every choice that built it."""

    protein_pdb: str
    display_structure: str
    sequence_fasta: str
    protein_residues: int
    protein_heavy_atoms: int
    water_count: int
    removed_waters: int
    modeled_termini: list[ModeledTerminus]
    substrates: list[PositionedSubstrate]
    manifest: str
    versions: dict[str, str]


@capability(
    "structure.prepare_reactive_coordinates",
    version="1.0",
    family="investigate",
    summary="Model missing termini from a donor and position a substrate in an enzyme",
    render=render,
)
def prepare_reactive_coordinates(
    request: CoordinateRequest, ctx: StepContext
) -> CoordinateResult:
    """Check the recipe, build the model and describe what was done."""
    from alambiq.capabilities.investigate.science.reactive import prepare

    recipe: dict[str, Any] = {
        "schema_version": "0.1",
        "target": asdict(request.target),
        "donor": asdict(request.donor),
        "terminal_model": asdict(request.terminal_model),
        "substrate": asdict(request.substrate),
    }
    output = ctx.workdir / "model"
    record = prepare(
        {
            "structure": ctx.resolve(request.structure),
            "donor": ctx.resolve(request.donor_structure),
            "substrate_source": ctx.resolve(request.substrate_source),
            "molecule": ctx.resolve(request.molecule),
        },
        recipe,
        output,
    )
    removed = len(record["removed_waters"])
    if removed:
        ctx.note(
            f"{removed} water(s) within {request.terminal_model.water_exclusion_angstrom} Å of "
            "a modeled residue are removed."
        )
    substrates = []
    for chain, row in record["ligands"].items():
        substrates.append(
            PositionedSubstrate(
                target_chain=chain,
                sdf=str(
                    ctx.keep(
                        output / f"ligand-{chain}.sdf", "chemical/x-mdl-sdfile"
                    ).path
                ),
                heavy_atoms=row["heavy_atom_count"],
                atoms=row["atom_count"],
                analogue_rmsd_angstrom=row["analogue_rmsd_angstrom"],
                protein_overlap_score_angstrom2=row["protein_geometric_overlap_score"],
                canonical_smiles=row["canonical_isomeric_smiles"],
                formal_charge=row["formal_charge"],
            )
        )
    return CoordinateResult(
        protein_pdb=str(ctx.keep(output / "protein.pdb", "chemical/x-pdb").path),
        display_structure=str(
            ctx.keep(output / "prepared.cif", "chemical/x-mmcif").path
        ),
        sequence_fasta=str(ctx.keep(output / "protein.fasta", "text/x-fasta").path),
        protein_residues=record["protein_residues"],
        protein_heavy_atoms=record["protein_heavy_atom_count"],
        water_count=record["water_count"],
        removed_waters=removed,
        modeled_termini=[
            ModeledTerminus(
                target_chain=t["target_chain"],
                terminus=t["terminus"],
                modeled_residues=t["modeled_residues"],
                donor_chain=t["donor_chain"],
                backbone_fit_rmsd_angstrom=t["backbone_fit_rmsd_angstrom"],
                junction_bond_angstrom=t["junction_bond_angstrom"],
                candidates_considered=len(t["candidates"]),
            )
            for t in record["terminal_templates"]
        ],
        substrates=substrates,
        manifest=ctx.keep_json(json.loads(json.dumps(record))),
        versions=record["environment_versions"],
    )
