"""Parameterize a modeled protein with positioned ligands, relax it, and optionally solvate it.

The inputs are a protein with the heavy atoms that it contains (for example the model that
`structure.prepare_reactive_coordinates` writes) and the ligands placed in it. Heavy
atoms in residues not listed in `modeled_residues` are called observed atoms; they are
held at their positions during relaxation. The capability builds
the system that a simulation or a QM/MM calculation reads. Terminal oxygens and
hydrogens are completed, histidines keep the tautomer named in the file, and the protein
is assigned Amber ff14SB parameters. Each ligand is assigned Sage 2.3.0 parameters and
AshGC charges. The unsolvated model is then relaxed in vacuum with every observed heavy
atom, the ligands and any retained crystal waters held at their positions, and the
residues in `modeled_residues` tied by springs to their placed positions. With
`solvent: tip3p`, a TIP3P box with neutralizing ions and salt is built around the
relaxed model, and the solvent and hydrogens are minimized with all solute heavy atoms
still held. The result is the Amber topology, the OpenMM system and the OpenMM state,
which `structure.prepare_finite_environment` and QM/MM calculations read. The result also
gives the residue number, first atom index and atom count of each ligand in the
topology.

The result is a fully parameterized, minimized model. With `solvent: tip3p` its net
charge is within 0.01 e of zero; with `solvent: none` its net charge is that of the
protein and the ligands. The observed heavy atoms, the ligands and the retained crystal
waters keep their supplied positions. The residues in `modeled_residues` move within
their position restraint, and the result reports the maximum displacement of their heavy
atoms.

Limits. The relaxation is a minimization that reduces steric clashes; the strain from
the positions of the held atoms remains. Modeled residues are tied to
their placed positions by springs, so they move only as far as the held atoms and the
spring constant allow. The ligand charges come from a graph neural network trained to
reproduce AM1-BCC and the protein charges from ff14SB, so they derive from different
methods. A ligand contains only the elements H, C, N, O, F, P, S, Cl, Br and I; it is
closed-shell, with explicit hydrogens and a single conformer. The parameterized
components are standard amino acids, crystal waters and non-covalent ligands. A protein
with missing residues, or with missing heavy atoms other than a terminal oxygen, is
refused. The protonation is fixed: histidines as named in the file, and the
other ionizable groups as OpenMM `Modeller.addHydrogens` assigns them at pH 7.0. A
system that `structure.prepare_finite_environment` reads requires an orthorhombic cell,
and of the box shapes only `cube` gives one. OpenMM adds the salt ions as whole ion
pairs, so the salt concentration of the finished box is approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from alambiq.capabilities.investigate.structure_prepare_protein_ligand_system_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

NET_CHARGE_TOLERANCE_E = 0.01


@dataclass(frozen=True, slots=True)
class ResidueRef:
    """One protein residue.

    chain, number:
        Its chain identifier in the protein PDB and its residue number there.
    """

    chain: str
    number: int


@dataclass(frozen=True, slots=True)
class LigandPlacement:
    """One ligand and where it is.

    chain:
        A chain identifier for the ligand: one character that no protein chain uses.
    residue_name:
        Its residue name, three characters, not an amino acid, water or ion name.
    pose:
        The SDF of the ligand at its place: one molecule, explicit hydrogens, one
        conformer.
    formal_charge:
        The charge the molecule must have; a mismatch is refused.
    """

    chain: str
    residue_name: str
    pose: str
    formal_charge: int


@dataclass(frozen=True, slots=True)
class SystemRequest:
    """The protein, its ligands, which residues were modeled, and the solvent.

    protein:
        The protein PDB: standard amino acids with all their heavy atoms (only a
        terminal oxygen may be missing), and optionally crystal waters.
    ligands:
        1 to 8 ligands, each with a distinct chain.
    modeled_residues:
        The residues whose coordinates come from modeling, not from an experiment. The
        relaxation restrains them with springs to their placed positions; every other
        protein heavy atom is held fixed.
    solvent:
        `tip3p` to add a water box with ions, or `none` to stop at the relaxed
        vacuum model.
    padding_nm:
        The distance from the solute to the box edge, 0.5 to 3. Ignored without solvent.
    ionic_strength_molar:
        The salt concentration in mol/L, in addition to the ions that neutralize the
        system, 0 to 1.
    box_shape:
        `cube`, `dodecahedron` or `octahedron`. `structure.prepare_finite_environment`
        requires `cube`.
    modeled_restraint_k_kj_mol_nm2:
        The spring constant tying modeled heavy atoms to where they were placed, 0 to
        10,000.
    minimizer_tolerance_kj_mol_nm:
        The force at which minimization stops, 0.1 to 100.
    minimization_max_iterations:
        The maximum number of minimizer iterations in each stage (the vacuum relaxation
        and the solvent minimization), 0 (until converged) to 100,000.
    seed:
        The seed for the random choices in completing and solvating, 1 to 2^31 - 1.
    threads:
        CPU threads, 1 to 64.
    """

    protein: str
    ligands: list[LigandPlacement]
    modeled_residues: list[ResidueRef]
    solvent: Literal["tip3p", "none"] = "tip3p"
    padding_nm: float = 1.0
    ionic_strength_molar: float = 0.15
    box_shape: Literal["cube", "dodecahedron", "octahedron"] = "cube"
    modeled_restraint_k_kj_mol_nm2: float = 100.0
    minimizer_tolerance_kj_mol_nm: float = 10.0
    minimization_max_iterations: int = 5000
    seed: int = 1
    threads: int = 4


@dataclass(frozen=True, slots=True)
class LigandSite:
    """Where one ligand is in the topology, in the one-based numbering other steps take."""

    chain: str
    residue_name: str
    residue_number: int
    first_atom: int
    atom_count: int
    formal_charge: int
    charge_sum_e: float


@dataclass(frozen=True, slots=True)
class SystemResult:
    """The prepared system's composition, relaxation and files."""

    periodic: bool
    atom_count: int
    solute_atoms: int
    """Protein, ligands and retained crystal waters: the atoms before any added solvent."""
    water_molecules: int
    sodium_ions: int
    chloride_ions: int
    net_charge_e: float
    observed_heavy_atoms_held: int
    modeled_heavy_atoms_relaxed: int
    maximum_modeled_displacement_angstrom: float
    vacuum_energy_before_kj_mol: float
    vacuum_energy_after_kj_mol: float
    solvent_energy_before_kj_mol: float
    solvent_energy_after_kj_mol: float
    rms_force_after_kj_mol_nm: float
    max_force_after_kj_mol_nm: float
    max_peptide_planarity_deviation_degrees: float | None
    inverted_alpha_residues: list[int]
    ligand_sites: list[LigandSite]
    system_xml: str
    state_xml: str
    topology_pdb: str
    amber_topology: str
    manifest: str
    force_field: dict[str, Any]
    versions: dict[str, str]
    box_volume_nm3: float | None = None


def _protein_chains(path: Path) -> set[str]:
    return {
        line[21]
        for line in path.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM")) and line[17:20].strip() != "HOH"
    }


@capability(
    "structure.prepare_protein_ligand_system",
    version="1.0",
    family="investigate",
    summary="Parameterize a modeled protein with ligands, relax it around held atoms, and solvate",
    render=render,
    environment="openff",
)
def prepare_protein_ligand_system(
    request: SystemRequest, ctx: StepContext
) -> SystemResult:
    """Check the request, build and relax the system, and describe what was done."""
    for name, value, low, high in (
        ("padding_nm", request.padding_nm, 0.5, 3.0),
        ("ionic_strength_molar", request.ionic_strength_molar, 0.0, 1.0),
        (
            "modeled_restraint_k_kj_mol_nm2",
            request.modeled_restraint_k_kj_mol_nm2,
            0.0,
            10_000.0,
        ),
        (
            "minimizer_tolerance_kj_mol_nm",
            request.minimizer_tolerance_kj_mol_nm,
            0.1,
            100.0,
        ),
        (
            "minimization_max_iterations",
            request.minimization_max_iterations,
            0,
            100_000,
        ),
        ("seed", request.seed, 1, 2**31 - 1),
        ("threads", request.threads, 1, 64),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; a value from {low} to {high} is required"
            )
    if not 1 <= len(request.ligands) <= 8:
        raise ValueError(
            f"ligands has {len(request.ligands)} entries; 1 to 8 are required"
        )
    protein = ctx.resolve(request.protein)
    chains = _protein_chains(protein)
    seen: set[str] = set()
    for ligand in request.ligands:
        if len(ligand.chain) != 1 or ligand.chain == " ":
            raise ValueError(
                f"The ligand chain {ligand.chain!r} is empty, longer than one character "
                "or a space; a single character other than a space is required"
            )
        if ligand.chain in chains or ligand.chain in seen:
            raise ValueError(
                f"Ligand chain {ligand.chain} is already used by the protein or by another "
                "ligand; a chain identifier unique to each ligand is required"
            )
        if len(ligand.residue_name) != 3 or not ligand.residue_name.isalnum():
            raise ValueError(
                f"The ligand residue name {ligand.residue_name!r} is not 3 letters or "
                "digits; exactly 3 letters or digits are required"
            )
        seen.add(ligand.chain)
    for residue in request.modeled_residues:
        if residue.chain not in chains:
            raise ValueError(
                f"Modeled residue {residue.chain}{residue.number} names chain "
                f"{residue.chain}, which is not in the protein file; a protein chain is required"
            )
    work = ctx.workdir / "system"
    work.mkdir()
    found = ctx.run_in_environment(
        "openff",
        "openff_prepare_system:run",
        {
            "protein_pdb": str(protein),
            "ligands": [
                {
                    "id": f"{x.residue_name}:{x.chain}",
                    "chain": x.chain,
                    "residue_name": x.residue_name,
                    "sdf": str(ctx.resolve(x.pose)),
                    "formal_charge": x.formal_charge,
                }
                for x in request.ligands
            ],
            "modeled_residues": [
                {"chain": r.chain, "number": r.number} for r in request.modeled_residues
            ],
            "solvent": request.solvent,
            "padding_nm": request.padding_nm,
            "ionic_strength_molar": request.ionic_strength_molar,
            "box_shape": request.box_shape,
            "modeled_restraint_k_kj_mol_nm2": request.modeled_restraint_k_kj_mol_nm2,
            "minimizer_tolerance_kj_mol_nm": request.minimizer_tolerance_kj_mol_nm,
            "minimization_max_iterations": request.minimization_max_iterations,
            "seed": request.seed,
            "threads": request.threads,
            "output": str(work / "output"),
        },
        timeout=7200,
    )
    if found["periodic"] and abs(found["net_charge_e"]) > NET_CHARGE_TOLERANCE_E:
        raise ValueError(
            f"The prepared system has net charge {found['net_charge_e']:.4f} e; a net "
            f"charge within {NET_CHARGE_TOLERANCE_E} e of zero is required"
        )
    geometry = found["protein_geometry"]
    output = work / "output"
    by_chain = {x.chain: x for x in request.ligands}
    sites = [
        LigandSite(
            chain=s["chain"],
            residue_name=s["residue_name"],
            residue_number=s["residue_number"],
            first_atom=s["first_atom"],
            atom_count=s["atom_count"],
            formal_charge=by_chain[s["chain"]].formal_charge,
            charge_sum_e=next(
                d["charge_sum_e"]
                for d in found["ligands"]
                if d["id"] == f"{s['residue_name']}:{s['chain']}"
            ),
        )
        for s in found["ligand_sites"]
    ]
    return SystemResult(
        periodic=found["periodic"],
        atom_count=found["atom_count"],
        solute_atoms=found["solute_atoms"],
        water_molecules=found["water_molecules"],
        sodium_ions=found["sodium_ions"],
        chloride_ions=found["chloride_ions"],
        net_charge_e=found["net_charge_e"],
        observed_heavy_atoms_held=found["observed_heavy_atoms_held"],
        modeled_heavy_atoms_relaxed=found["modeled_heavy_atoms_relaxed"],
        maximum_modeled_displacement_angstrom=found[
            "maximum_modeled_displacement_angstrom"
        ],
        vacuum_energy_before_kj_mol=found["vacuum_energy_before_kj_mol"],
        vacuum_energy_after_kj_mol=found["vacuum_energy_after_kj_mol"],
        solvent_energy_before_kj_mol=found["solvent_energy_before_kj_mol"],
        solvent_energy_after_kj_mol=found["solvent_energy_after_kj_mol"],
        rms_force_after_kj_mol_nm=found["forces_after"]["rms_component_kj_mol_nm"],
        max_force_after_kj_mol_nm=found["forces_after"]["maximum_atom_kj_mol_nm"],
        max_peptide_planarity_deviation_degrees=geometry[
            "maximum_peptide_planarity_deviation_degrees"
        ],
        inverted_alpha_residues=geometry["inverted_alpha_residues_one_based"],
        ligand_sites=sites,
        system_xml=str(ctx.keep(output / "system.xml", "application/xml").path),
        state_xml=str(ctx.keep(output / "state.xml", "application/xml").path),
        topology_pdb=str(ctx.keep(output / "topology.pdb", "chemical/x-pdb").path),
        amber_topology=str(
            ctx.keep(output / "complex.prmtop", "application/octet-stream").path
        ),
        manifest=str(ctx.keep(output / "preparation.json", "application/json").path),
        force_field=found["force_field"],
        versions=found["versions"],
        box_volume_nm3=found["cell"]["volume_nm3"] if found["cell"] else None,
    )
