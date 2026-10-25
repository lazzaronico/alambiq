"""Prepare a Boltz-predicted protein-ligand complex for molecular dynamics.

The inputs are the three files that `structure.predict_biomolecular_complex` writes for
one sample (one predicted structure of the complex): the predicted mmCIF, the pose of the ligand as an SDF, and the ligand
chemistry file (a JSON file with the canonical isomeric SMILES and the atom names of the
ligand). The protein must consist of standard amino acids with exactly the heavy atoms
of its residues and, if declared, the declared sequences. The capability identifies a
disulfide bond when two cysteine sulfur atoms are at most 2.5 angstrom apart, and
refuses a pair at a distance from 2.5 to 3.5 angstrom as ambiguous. The ligand pose
must be the one the SMILES specifies, including its handedness. The capability adds
hydrogens to the ligand and assigns Sage 2.3.0 parameters and AshGC partial charges. It
assigns Amber ff14SB parameters to the protein, adds the terminal oxygen and hydrogens,
and sets every histidine to the epsilon tautomer (HIE). It solvates the complex in a
TIP3P box with neutralizing ions and salt, builds the system with PME and constrained
hydrogen bonds, and minimizes it. The result is the serialized system, state and topology
that `simulation.run_molecular_dynamics` takes, with the energies, forces, charges and
composition.

The result is a minimized, fully parameterized model of the predicted complex, with a
net charge within 0.01 e of zero. The model is built from the positions of the
prediction and then minimized without restraints, so heavy atoms are displaced from
the predicted positions.

Limits. Every input is a prediction, and the pose is as accurate as the prediction. A
minimized energy that is finite and lower than the starting energy indicates that the
system can be built; the pose is that of the prediction. A predicted complex in a box can dissociate or rearrange during dynamics. The
ligand charges come from a graph neural network trained to reproduce AM1-BCC and the
protein charges from ff14SB, so ligand and protein electrostatics derive from
different methods. A ligand contains only the elements H, C, N, O, F, P, S, Cl, Br and I; it is
closed-shell and connected; its stereochemistry is fully specified; it has at most 999
atoms. A sample whose ligand has the opposite handedness from the SMILES is refused. The
protonation is fixed: histidine as HIE, and the other ionizable groups in the states that
OpenMM `Modeller.addHydrogens` assigns at pH 7.0, independent of the environment of the
binding site. The capability prepares the one sample given. The
capability refuses residues other than the 20 standard amino acids and one ligand
residue (`LIG1`), which excludes metals, cofactors, nucleic acids and modified
residues, and the topology has no covalent bond between the ligand and the protein. OpenMM
adds the salt ions as whole ion pairs, so the salt concentration of the finished box
is approximate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from alambiq.capabilities.investigate.structure_prepare_predicted_complex_for_dynamics_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

NET_CHARGE_TOLERANCE_E = 0.01


@dataclass(frozen=True, slots=True)
class ComplexRequest:
    """The predicted sample and how it is solvated and minimized.

    structure:
        The predicted mmCIF of one sample.
    ligand_pose:
        The SDF of the same sample's ligand pose, from the same prediction.
    ligand_chemistry:
        The ligand chemistry file (JSON) written for the same sample.
    protein_sequences:
        Optional. The sequences the protein chains must have, in the file's chain
        order; a mismatch is refused. Without it the sequences are read from the
        file.
    ligand_formal_charge:
        Optional. The charge the ligand must have; a mismatch is refused.
    padding_nm:
        The distance from the solute to the box edge, 0.5 to 3.
    ionic_strength_molar:
        The salt concentration in mol/L, in addition to the ions that neutralize the
        system, 0 to 1.
    box_shape:
        `dodecahedron`, `octahedron` or `cube`.
    minimize_iterations:
        The maximum number of minimizer iterations, 0 (until converged) to 100,000.
    seed:
        The seed for the random choices in completing and solvating, 1 to 2^31 - 1.
    threads:
        CPU threads for the minimization, 1 to 64.
    """

    structure: str
    ligand_pose: str
    ligand_chemistry: str
    protein_sequences: list[str] | None = None
    ligand_formal_charge: int | None = None
    padding_nm: float = 1.0
    ionic_strength_molar: float = 0.15
    box_shape: Literal["dodecahedron", "octahedron", "cube"] = "dodecahedron"
    minimize_iterations: int = 5000
    seed: int = 1
    threads: int = 4


@dataclass(frozen=True, slots=True)
class ComplexResult:
    """The prepared complex's composition, charges, energies and files."""

    protein_chains: int
    protein_residues: int
    ligand_heavy_atoms: int
    ligand_formal_charge: int
    ligand_charge_sum_e: float
    disulfides: list[str]
    """Cysteine pairs joined, by chain and residue number."""
    atom_count: int
    water_molecules: int
    sodium_ions: int
    chloride_ions: int
    net_charge_e: float
    potential_energy_before_kj_mol: float
    potential_energy_after_kj_mol: float
    rms_force_after_kj_mol_nm: float
    max_force_after_kj_mol_nm: float
    box_volume_nm3: float
    max_peptide_planarity_deviation_degrees: float | None
    inverted_alpha_residues: list[int]
    system_xml: str
    state_xml: str
    topology_pdb: str
    protein_pdb: str
    amber_topology: str
    """The Amber topology of the periodic system, which `structure.prepare_finite_environment` reads."""
    ligand_parameters: str
    """A JSON file with the ligand's atom names, charges and positions."""
    manifest: str
    force_field: dict[str, Any]
    versions: dict[str, str]


def _source_digest(sdf_text: str) -> str | None:
    match = re.search(
        r">\s*<ALAMBIQ_SOURCE_MODEL_SHA256>[^\n]*\n([0-9a-f]{64})", sdf_text
    )
    return match.group(1) if match else None


@capability(
    "structure.prepare_predicted_complex_for_dynamics",
    version="1.0",
    family="investigate",
    summary="Parameterize, solvate and minimize a predicted protein-ligand complex",
    render=render,
    environment="openff",
)
def prepare_predicted_complex(
    request: ComplexRequest, ctx: StepContext
) -> ComplexResult:
    """Read and check the prediction, then build and minimize the system."""
    from alambiq.capabilities.investigate.science.predicted_complex import (
        find_disulfides,
        read_prediction,
        write_protein_pdb,
    )

    for name, value, low, high in (
        ("padding_nm", request.padding_nm, 0.5, 3.0),
        ("ionic_strength_molar", request.ionic_strength_molar, 0.0, 1.0),
        ("minimize_iterations", request.minimize_iterations, 0, 100_000),
        ("seed", request.seed, 1, 2**31 - 1),
        ("threads", request.threads, 1, 64),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; a value from {low} to {high} is required"
            )
    structure = ctx.resolve(request.structure)
    chemistry = json.loads(ctx.read_text(request.ligand_chemistry))
    sdf_text = ctx.read_text(request.ligand_pose)
    model_digest = hashlib.sha256(structure.read_bytes()).hexdigest()
    if _source_digest(sdf_text) != model_digest:
        raise ValueError(
            "The source digest recorded in ligand_pose differs from the digest of "
            "structure; the pose of the same sample as structure is required"
        )
    chains, _ligand_chain, ligand_atoms, sequences = read_prediction(
        structure, request.protein_sequences
    )
    names = {a.name.strip() for a in ligand_atoms}
    if names != {a["name"] for a in chemistry["atoms"]}:
        raise ValueError(
            "The ligand atom names in structure differ from those in ligand_chemistry; "
            "the two sets must be equal"
        )
    bonds = find_disulfides(chains)
    work = ctx.workdir / "complex"
    work.mkdir()
    protein = work / "protein.pdb"
    write_protein_pdb(chains, bonds, protein)
    pose = work / "ligand_pose.sdf"
    pose.write_text(sdf_text, encoding="utf-8")
    found = ctx.run_in_environment(
        "openff",
        "openff_prepare_complex:run",
        {
            "protein_pdb": str(protein),
            "ligand_pose": str(pose),
            "ligand_chemistry": str(ctx.resolve(request.ligand_chemistry)),
            "ligand_formal_charge": request.ligand_formal_charge,
            "padding_nm": request.padding_nm,
            "ionic_strength_molar": request.ionic_strength_molar,
            "box_shape": request.box_shape,
            "minimize_iterations": request.minimize_iterations,
            "seed": request.seed,
            "threads": request.threads,
            "output": str(work / "output"),
        },
        timeout=3600,
    )
    if abs(found["net_charge_e"]) > NET_CHARGE_TOLERANCE_E:
        raise ValueError(
            f"The prepared system has net charge {found['net_charge_e']:.4f} e; a net "
            f"charge within {NET_CHARGE_TOLERANCE_E} e of zero is required"
        )
    geometry = found["protein_geometry"]
    if bonds:
        ctx.note(
            f"{len(bonds)} disulfide bond(s) are assigned from sulfur distances "
            "(SG-SG distance of at most 2.5 Å)."
        )
    output = work / "output"
    return ComplexResult(
        protein_chains=len(chains),
        protein_residues=sum(len(r) for _, r in chains),
        ligand_heavy_atoms=found["ligand_heavy_atoms"],
        ligand_formal_charge=found["ligand_formal_charge"],
        ligand_charge_sum_e=found["ligand_parameters"]["charge_sum_e"],
        disulfides=[f"{a[0]}:{a[1]}-{b[0]}:{b[1]}" for a, b, _ in bonds],
        atom_count=found["atom_count"],
        water_molecules=found["water_molecules"],
        sodium_ions=found["sodium_ions"],
        chloride_ions=found["chloride_ions"],
        net_charge_e=found["net_charge_e"],
        potential_energy_before_kj_mol=found["initial_potential_energy_kj_mol"],
        potential_energy_after_kj_mol=found["minimized_potential_energy_kj_mol"],
        rms_force_after_kj_mol_nm=found["forces_after"]["rms_component_kj_mol_nm"],
        max_force_after_kj_mol_nm=found["forces_after"]["maximum_atom_kj_mol_nm"],
        box_volume_nm3=found["cell"]["volume_nm3"],
        max_peptide_planarity_deviation_degrees=geometry[
            "maximum_peptide_planarity_deviation_degrees"
        ],
        inverted_alpha_residues=geometry["inverted_alpha_residues_one_based"],
        system_xml=str(ctx.keep(output / "system.xml", "application/xml").path),
        state_xml=str(ctx.keep(output / "state.xml", "application/xml").path),
        topology_pdb=str(ctx.keep(output / "topology.pdb", "chemical/x-pdb").path),
        protein_pdb=str(ctx.keep(protein, "chemical/x-pdb").path),
        amber_topology=str(
            ctx.keep(output / "complex.prmtop", "application/octet-stream").path
        ),
        ligand_parameters=ctx.keep_json(found["ligand_parameters"]),
        manifest=str(ctx.keep(output / "preparation.json", "application/json").path),
        force_field=found["force_field"],
        versions=found["versions"],
    )
