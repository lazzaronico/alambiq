"""Prepare protein and ring-congener inputs for relative binding free energies.

From a deposited complex with a six-carbon aromatic ring in the binding site (benzene
in a lysozyme cavity, for example), the capability prepares one protein chain and a
series of neutral congeners that each contain that ring plus substituents. The protein
is the named chain, with heterogens and water removed, hydrogens added at the stated pH
and, if allowed in the request, missing terminal residues modeled; the capability checks
that every deposited heavy atom is at its deposited position. The capability places each
congener by superposing its ring on the deposited ring in each of the 12
symmetry-related orientations, relaxing the substituents with the ring held fixed, and
keeping the orientation with the lowest clash score of the substituents against the
protein (the clash score is the summed squared shortfall below 2.2 angstrom between
substituent atoms and protein heavy atoms). The first substituted congener fixes the
deposited ring atom on which its substituent lies. For every later congener only the
orientations that place the substituent on that ring atom are considered, and the one
with the lowest clash score is kept. A congener has heavy-atom substituents on exactly one ring position. The
capability assigns AshGC partial charges to each congener and writes it as an SDF that
`simulation.run_relative_free_energy` accepts. For each listed edge, the capability
proposes an atom mapping with the LOMAP mapper of OpenFE without element changes,
choosing the mapping with the smallest summed squared distance between mapped atoms.

The result gives protein and ligand files from which sampling starts with a pose
consistent with the deposited ring, with every placement choice and the atom mapping
recorded in the manifest (a JSON file).

Limits. The pose of a substituent is a geometric placement: the one with the lowest clash
score among the eligible orientations. A substituent that the deposited site cannot hold
is accepted and appears as a clash score in the manifest. The capability handles proteins
of one chain, one anchor ring and congeners whose only ring is a six-membered aromatic
carbon ring; the site retains that chain and the congener, and cofactors, ions,
structural waters and a second ligand are removed. The protonation is assigned at one pH
by OpenMM `Modeller.addHydrogens`, independent of the environment of the site. The
coordinates of terminal residues that the request allows to be modeled come from
modeling. Hydrogens are placed by an energy minimization whose result varies between
executions. Two executions of the capability agree exactly on the heavy atoms and the
ligands. Protein hydrogen positions differed by up to 0.063 angstrom between two
executions; the test of this capability bounds the difference at 0.5 angstrom. The atom
mapping is the output of the LOMAP mapper. `simulation.run_relative_free_energy` checks
the mapping for chemistry and connectivity. A request with more than 20 ligands or more
than 50 edges is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from alambiq.capabilities.investigate.simulation_analyze_binding_free_energy import (
    Edge,
)
from alambiq.capabilities.investigate.structure_prepare_relative_binding_inputs_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class Congener:
    """One ligand of the series.

    id:
        Identifier of the ligand: a letter, then letters, digits, `_` or `-`; used for
        its file.
    smiles:
        The ligand with explicit stereochemistry. The first ligand must be benzene
        (`c1ccccc1`), the deposited ring itself.
    embedding_seed:
        The seed for its three-dimensional embedding, 1 to 2^31 - 1.
    """

    id: str
    smiles: str
    embedding_seed: int = 1


@dataclass(frozen=True, slots=True)
class BindingInputsRequest:
    """The deposited complex, the series and the edges.

    structure:
        A path to the deposited complex, mmCIF or PDB.
    chain_id:
        The one protein chain to keep.
    anchor_residue_name:
        The residue name of the deposited ring, which must occur exactly once and
        hold only carbon atoms.
    anchor_atom_names:
        The six ring carbons' atom names, in order around the ring.
    ligands:
        2 to 20 ligands; the first is benzene.
    edges:
        1 to 50 directed edges between them.
    structure_format:
        `pdb` or `mmcif`; needed only when the file name does not say.
    ph:
        The pH for adding hydrogens, 0 to 14.
    complete_terminals:
        If true, missing terminal residues are modeled. Missing residues inside the
        chain are always refused.
    seed:
        The seed for modeling missing atoms and residues, 1 to 2^31 - 1.
    threads:
        CPU threads, 1 to 64.
    """

    structure: str
    chain_id: str
    anchor_residue_name: str
    anchor_atom_names: list[str]
    ligands: list[Congener]
    edges: list[Edge]
    structure_format: Literal["pdb", "mmcif"] | None = None
    ph: float = 7.0
    complete_terminals: bool = False
    seed: int = 1
    threads: int = 4


@dataclass(frozen=True, slots=True)
class PreparedLigand:
    """One placed, charged ligand."""

    id: str
    smiles: str
    sdf: str
    atoms: int
    """With hydrogens."""
    charge_sum_e: float
    selected_pose_index: int | None
    """Which of the 12 placements was retained; None for the anchor ring itself."""
    overlap_penalty_angstrom2: float | None
    """Summed squared shortfall below 2.2 A to protein heavy atoms, over substituent atoms."""
    minimum_substituent_distance_angstrom: float | None


@dataclass(frozen=True, slots=True)
class PreparedEdge:
    """An atom mapping for one edge."""

    id: str
    a: str
    b: str
    atom_mapping: list[list[int]]
    """Pairs of zero-based atom indices in the SDF of a and of b."""
    mapped_atoms: int
    rms_mapped_distance_angstrom: float


@dataclass(frozen=True, slots=True)
class BindingInputsResult:
    """The prepared files and every choice that made them."""

    protein_pdb: str
    protein_residues: int
    deposited_residues: int
    modeled_terminal_residues: list[dict[str, Any]]
    ph: float
    ligands: list[PreparedLigand]
    edges: list[PreparedEdge]
    manifest: str
    versions: dict[str, str]
    seconds: float


@capability(
    "structure.prepare_relative_binding_inputs",
    version="1.0",
    family="investigate",
    summary="Prepare protein and anchored ring-congener inputs for relative free energies",
    render=render,
    environment="openfe",
)
def prepare_relative_binding_inputs(
    request: BindingInputsRequest, ctx: StepContext
) -> BindingInputsResult:
    """Check the recipe, then prepare the protein, place the congeners and map the edges."""
    import json
    import math
    import re

    path = ctx.resolve(request.structure)
    fmt = request.structure_format
    if fmt is None:
        suffix = path.suffix.lower()
        if suffix in (".pdb", ".ent"):
            fmt = "pdb"
        elif suffix in (".cif", ".mmcif"):
            fmt = "mmcif"
        else:
            raise ValueError(
                f"The extension {path.suffix!r} of the structure file does not identify "
                "PDB or mmCIF; set structure_format"
            )
    if not request.chain_id.strip() or not request.anchor_residue_name.strip():
        raise ValueError("chain_id or anchor_residue_name is empty; both are required")
    names = request.anchor_atom_names
    if len(names) != 6 or len(set(names)) != 6 or any(not n.strip() for n in names):
        raise ValueError(
            f"anchor_atom_names has {len(names)} entries, repeats a name or contains an "
            "empty name; 6 distinct ring carbon atom names in ring order are required"
        )
    if not 2 <= len(request.ligands) <= 20:
        raise ValueError(
            f"ligands has {len(request.ligands)} entries; the benzene anchor and 1 to 19 "
            "congeners (2 to 20 entries) are required"
        )
    ids = [lig.id for lig in request.ligands]
    if len(set(ids)) != len(ids) or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", i) for i in ids
    ):
        raise ValueError(
            "A ligand id is repeated or does not match the pattern; ligand ids must be "
            "distinct and start with a letter, then letters, digits, `_` or `-` "
            "(32 characters at most)"
        )
    first = request.ligands[0]
    if first.smiles != "c1ccccc1":
        raise ValueError(
            f"The first ligand has the SMILES {first.smiles}; the first ligand is the "
            "deposited ring and must be benzene (c1ccccc1)"
        )
    for lig in request.ligands:
        if not lig.smiles.strip() or len(lig.smiles) > 500:
            raise ValueError(
                f"The SMILES of ligand {lig.id} is empty or longer than 500 characters; "
                "a SMILES of 1 to 500 characters is required"
            )
        if not 1 <= lig.embedding_seed < 2**31:
            raise ValueError(
                f"embedding_seed of ligand {lig.id} is {lig.embedding_seed}; an integer "
                "from 1 to 2^31 - 1 is required"
            )
    if not 1 <= len(request.edges) <= 50:
        raise ValueError(
            f"edges has {len(request.edges)} entries; 1 to 50 are required"
        )
    edge_ids = set()
    for edge in request.edges:
        if (
            edge.a not in ids
            or edge.b not in ids
            or edge.a == edge.b
            or edge.id in edge_ids
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", edge.id)
        ):
            raise ValueError(
                f"The edge {edge.id} ({edge.a} to {edge.b}) has a repeated id, an id that "
                "does not match the pattern, or ligands that are identical or not in the "
                "series; edges need distinct ids and two different ligands from the series"
            )
        edge_ids.add(edge.id)
    if not math.isfinite(request.ph) or not 0 <= request.ph <= 14:
        raise ValueError(f"ph is {request.ph}; a value from 0 to 14 is required")
    for name, value, low, high in (
        ("seed", request.seed, 1, 2**31 - 1),
        ("threads", request.threads, 1, 64),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; a value from {low} to {high} is required"
            )
    found = ctx.run_in_environment(
        "openfe",
        "openfe_prepare_binding:run",
        {
            "structure": str(path),
            "structure_format": fmt,
            "chain_id": request.chain_id,
            "anchor_residue_name": request.anchor_residue_name,
            "anchor_atom_names": names,
            "ligands": [
                {
                    "id": lig.id,
                    "smiles": lig.smiles,
                    "embedding_seed": lig.embedding_seed,
                }
                for lig in request.ligands
            ],
            "edges": [{"id": e.id, "a": e.a, "b": e.b} for e in request.edges],
            "ph": request.ph,
            "complete_terminals": request.complete_terminals,
            "seed": request.seed,
            "threads": request.threads,
            "output": str(ctx.workdir / "prepared"),
        },
        timeout=7200,
    )
    if found["modeled_terminal_residues"]:
        ctx.note(
            "Terminal residues are modeled where the structure has none: "
            + "; ".join(
                f"{', '.join(m['names'])} at insertion point {m['insertion_index']}"
                for m in found["modeled_terminal_residues"]
            )
        )
    ligands = []
    for lig in request.ligands:
        row = found["ligands"][lig.id]
        chosen = (
            row["candidate_poses"][row["selected_pose_index"]]
            if row["selected_pose_index"] is not None
            else None
        )
        ligands.append(
            PreparedLigand(
                id=lig.id,
                smiles=lig.smiles,
                sdf=str(ctx.keep(row["sdf"], "chemical/x-mdl-sdfile").path),
                atoms=row["atoms"],
                charge_sum_e=row["charge_sum_e"],
                selected_pose_index=row["selected_pose_index"],
                overlap_penalty_angstrom2=chosen["overlap_penalty_angstrom2"]
                if chosen
                else None,
                minimum_substituent_distance_angstrom=chosen[
                    "minimum_added_atom_distance_angstrom"
                ]
                if chosen
                else None,
            )
        )
        if chosen and chosen["overlap_penalty_angstrom2"] > 0:
            ctx.note(
                f"{lig.id}: the best placement still has substituent atoms closer than "
                f"2.2 Å to protein heavy atoms (nearest {chosen['minimum_added_atom_distance_angstrom']:.2f} Å)."
            )
    edges = [
        PreparedEdge(
            id=e["id"],
            a=e["a"],
            b=e["b"],
            atom_mapping=e["atom_mapping"],
            mapped_atoms=len(e["atom_mapping"]),
            rms_mapped_distance_angstrom=(
                sum(d * d for d in e["mapped_distances_angstrom"])
                / max(1, len(e["mapped_distances_angstrom"]))
            )
            ** 0.5,
        )
        for e in found["edges"]
    ]
    manifest = {k: v for k, v in found.items() if k not in ("protein_pdb",)}
    return BindingInputsResult(
        protein_pdb=str(ctx.keep(found["protein_pdb"], "chemical/x-pdb").path),
        protein_residues=found["completed_residues"],
        deposited_residues=found["deposited_residues"],
        modeled_terminal_residues=found["modeled_terminal_residues"],
        ph=found["ph"],
        ligands=ligands,
        edges=edges,
        manifest=ctx.keep_json(json.loads(json.dumps(manifest))),
        versions=found["versions"],
        seconds=found["seconds"],
    )
