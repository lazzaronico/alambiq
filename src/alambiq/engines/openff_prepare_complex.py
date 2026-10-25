"""Prepare a protein with a positioned ligand for dynamics: Sage, AshGC and ff14SB.

Runs inside the OpenFF environment. The standard library is imported at the top;
everything else is loaded when a request runs. The caller has already read the
predicted complex and written the protein as a PDB; this engine takes that file and
the positioned ligand (an SDF with its intended chemistry), then:

1. checks the ligand pose has the stereochemistry the intended SMILES gives, adds
   its hydrogens at computed positions, and assigns AshGC partial charges with the
   Sage 2.3.0 force field;
2. completes the protein (terminal OXT and hydrogens, histidine as HIE), adds the
   ligand, solvates in TIP3P with neutralizing ions and salt;
3. writes the Amber-format parameters, builds the OpenMM system with PME and
   constrained hydrogen bonds, and minimises on the CPU.

Topologies are written with ParmEd. The force-field and charge-model files are checked against
recorded digests before they are used.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import openmm_common as diagnostics

FORCE_FIELD = "openff_unconstrained-2.3.0.offxml"
FORCE_FIELD_SHA256 = {
    FORCE_FIELD: "65b926cf38ae4bc2f6f9cac907c3cadba154cc84988dd10ab2c3895aaf2dd995",
    "openff-2.3.0.offxml": "7a0d7195a4b717e29fe14aa2cf1bc3203e87ce4aef9cc2ef10fb5145b7edcf63",
}
CHARGE_MODEL = "openff-gnn-am1bcc-1.0.0.pt"
CHARGE_MODEL_SHA256 = "7981e7f5b0b1e424c9e10a40d9e7606d96dcd3dd2b095cb4eeff6829f92238ee"
SALT_WATER_MOLAR = 55.4
SOLVENT = {"HOH", "WAT"}
SODIUM = {"NA", "Na+"}
CHLORIDE = {"CL", "Cl-"}
ALLOWED_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def digest(path: Any) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def versions() -> dict[str, str]:
    """Package versions, from conda's records where a wheel reports 0.0.0."""
    names = (
        "openmm",
        "rdkit",
        "pdbfixer",
        "openff-toolkit",
        "openff-interchange",
        "openff-nagl",
        "openff-nagl-models",
        "openff-forcefields",
        "openmmforcefields",
        "ParmEd",
    )
    records = {}
    for path in (Path(sys.prefix) / "conda-meta").glob("*.json"):
        record = json.loads(path.read_text())
        records[record["name"]] = record["version"]
    found = {}
    for name in names:
        try:
            reported = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            reported = "0.0.0"
        found[name] = records.get(name, reported) if reported == "0.0.0" else reported
    return found


def check_data_files() -> dict[str, Any]:
    """The force field and charge model must be the bytes this method was qualified with."""
    from openff.nagl_models import validate_nagl_model_path
    from openforcefields import get_forcefield_dirs_paths

    found = {}
    for name, expected in FORCE_FIELD_SHA256.items():
        paths = [
            Path(folder) / name
            for folder in get_forcefield_dirs_paths()
            if (Path(folder) / name).is_file()
        ]
        if len(paths) != 1 or digest(paths[0]) != expected:
            raise ValueError(
                f"The installed force field {name} differs from the one this method records"
            )
        found[name] = expected
    model = validate_nagl_model_path(CHARGE_MODEL)
    if digest(model) != CHARGE_MODEL_SHA256:
        raise ValueError(
            "The installed AshGC charge model differs from the one recorded"
        )
    return {"force_field_sha256": found, "charge_model_sha256": CHARGE_MODEL_SHA256}


def read_ligand(sdf: str, chemistry: dict[str, Any], expected_charge: int | None):
    """The ligand pose with explicit hydrogens, after checking it is the one intended."""
    from rdkit import Chem

    supplier = list(Chem.SDMolSupplier(sdf, removeHs=False))
    if len(supplier) != 1 or supplier[0] is None:
        raise ValueError("The ligand pose must be an SDF holding exactly one molecule")
    posed = supplier[0]
    if posed.GetNumConformers() != 1:
        raise ValueError("The ligand pose must have one set of coordinates")
    if len(Chem.GetMolFrags(posed)) != 1:
        raise ValueError("The ligand must be a single connected molecule")
    intended = chemistry["canonical_isomeric_smiles"]
    perceived = Chem.MolToSmiles(posed)
    if perceived != intended:
        plain = Chem.MolToSmiles(posed, isomericSmiles=False)
        if plain != Chem.MolToSmiles(
            Chem.MolFromSmiles(intended), isomericSmiles=False
        ):
            raise ValueError(
                "The ligand pose has different bonds from the intended ligand"
            )
        raise ValueError(
            "The ligand pose has different stereochemistry from the intended SMILES "
            f"(pose {perceived}, intended {intended}); a sample whose ligand was placed "
            "with the wrong handedness is not prepared"
        )
    charge = Chem.GetFormalCharge(posed)
    if expected_charge is not None and charge != expected_charge:
        raise ValueError(
            f"The ligand has formal charge {charge}, but {expected_charge} was declared"
        )
    unspecified = [
        info
        for info in Chem.FindPotentialStereo(posed)
        if str(info.specified) == "Unspecified"
    ]
    if unspecified:
        raise ValueError("The ligand has unspecified stereochemistry")
    heavy = Chem.RemoveHs(posed)
    with_hydrogens = Chem.AddHs(heavy, addCoords=True)
    return posed, with_hydrogens, charge


def parameterize(
    molecule: Any, directory: Path, residue_name: str = "LIG", chain_id: str = "L"
):
    """Keep the explicit chemistry, atom order and coordinates; assign AshGC charges."""
    import numpy as np
    from openff.nagl_models import _dynamic_fetch
    from openff.toolkit import Molecule
    from openff.toolkit.utils import RDKitToolkitWrapper, ToolkitRegistry
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
    from openff.units import unit
    from rdkit import Chem

    if (
        molecule.GetNumConformers() != 1
        or any(
            a.GetNumImplicitHs()
            or a.GetNumRadicalElectrons()
            or a.GetIsotope()
            or a.GetSymbol() not in ALLOWED_ELEMENTS
            for a in molecule.GetAtoms()
        )
        or not np.isfinite(molecule.GetConformer().GetPositions()).all()
    ):
        raise ValueError(
            "The ligand must be closed-shell with explicit hydrogens, elements from "
            "H C N O F P S Cl Br I, and finite coordinates"
        )
    registry = ToolkitRegistry([RDKitToolkitWrapper(), NAGLToolkitWrapper()])
    off = Molecule.from_rdkit(
        molecule, allow_undefined_stereo=False, hydrogens_are_explicit=True
    )
    if off.n_atoms > 999:
        raise ValueError("At most 999 atoms per ligand are supported")
    for index, atom in enumerate(off.atoms, 1):
        atom.name = f"{atom.symbol[0]}{index:03d}"
        atom.metadata.update(
            residue_name=residue_name, residue_number="1", chain_id=chain_id
        )
    # NAGL makes a cache directory even for an installed model; keep it temporary.
    previous = _dynamic_fetch.CACHE_DIR
    with tempfile.TemporaryDirectory(prefix="alambiq-nagl-") as cache:
        _dynamic_fetch.CACHE_DIR = Path(cache)
        try:
            off.assign_partial_charges(CHARGE_MODEL, toolkit_registry=registry)
        finally:
            _dynamic_fetch.CACHE_DIR = previous
    charges = off.partial_charges.m_as(unit.elementary_charge)
    if (
        not np.isfinite(charges).all()
        or abs(float(charges.sum()) - Chem.GetFormalCharge(molecule)) > 1e-6
        or not np.allclose(
            off.conformers[0].m_as(unit.angstrom),
            molecule.GetConformer().GetPositions(),
            rtol=0,
            atol=1e-10,
        )
    ):
        raise ValueError(
            "The AshGC charges or the source coordinates failed validation"
        )
    (directory / "ligand.json").write_text(off.to_json())
    return off, {
        "canonical_isomeric_smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule)),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "atom_count": off.n_atoms,
        "atom_names": [a.name for a in off.atoms],
        "atom_elements": [a.atomic_number for a in off.atoms],
        "positions_angstrom": off.conformers[0].m_as(unit.angstrom).tolist(),
        "partial_charges_e": charges.tolist(),
        "charge_sum_e": float(charges.sum()),
        "force_field": FORCE_FIELD,
        "charge_model": CHARGE_MODEL,
    }


def forcefield(molecules: list[Any], directory: Path):
    from openmm import app
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    generator = SMIRNOFFTemplateGenerator(molecules=molecules, forcefield=FORCE_FIELD)
    generator.debug_ffxml_filename = str(directory / "ligand-templates.xml")
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    ff.registerTemplateGenerator(generator.generator)
    return ff


def assemble(protein: Path, molecules: list[Any], directory: Path, seed: int):
    """Complete terminal OXT and hydrogens; keep every source heavy atom where it was."""
    import numpy as np
    from openff.units.openmm import to_openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer

    random.seed(seed)
    fixer = PDBFixer(filename=str(protein))
    original = [
        (a.name, a.residue.index, p.value_in_unit(unit.nanometer))
        for a, p in zip(fixer.topology.atoms(), fixer.positions, strict=True)
    ]
    fixer.findMissingResidues()
    if fixer.missingResidues:
        raise ValueError("Missing protein residues must be completed before assembly")
    fixer.findMissingAtoms()
    if fixer.missingAtoms or any(
        set(names) != {"OXT"} for names in fixer.missingTerminals.values()
    ):
        raise ValueError(
            "Missing protein heavy atoms must be completed before assembly; "
            "only a terminal OXT is added"
        )
    fixer.addMissingAtoms(seed=seed)
    # PDBFile canonicalises histidine names; keep the tautomer the PDB named.
    named = {}
    for line in Path(protein).read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            named[(line[21:22], line[22:26].strip(), line[26:27].strip())] = line[
                17:20
            ].strip()
    variants = []
    for residue in fixer.topology.residues():
        name = named.get(
            (residue.chain.id, residue.id, residue.insertionCode), residue.name
        )
        variants.append(name if name in {"HID", "HIE", "HIP"} else None)
    model = app.Modeller(fixer.topology, fixer.positions)
    model.topology.setPeriodicBoxVectors(None)
    protein_ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    chosen = model.addHydrogens(protein_ff, pH=7.0, variants=variants)
    observed = {
        (a.name, a.residue.index): p.value_in_unit(unit.nanometer)
        for a, p in zip(model.topology.atoms(), model.positions, strict=True)
    }
    if any(
        not np.allclose(observed[(name, residue)], position, rtol=0, atol=1e-6)
        for name, residue, position in original
    ):
        raise ValueError(
            "Completing the protein changed the source heavy-atom coordinates"
        )
    # Storage can round coordinates; restore the source values after the check.
    originals = {(name, residue): position for name, residue, position in original}
    for atom in model.topology.atoms():
        key = (atom.name, atom.residue.index)
        if key in originals:
            model.positions[atom.index] = originals[key] * unit.nanometer
    # Retained crystal waters follow the ligands, as the solvent would.
    waters = [r for r in model.topology.residues() if r.name in SOLVENT]
    water_model = app.Modeller(model.topology, model.positions)
    water_model.delete(
        [r for r in water_model.topology.residues() if r.name not in SOLVENT]
    )
    model.delete(waters)
    for molecule in molecules:
        model.add(molecule.to_topology().to_openmm(), to_openmm(molecule.conformers[0]))
    if water_model.topology.getNumAtoms():
        model.add(water_model.topology, water_model.positions)
    return model, forcefield(molecules, directory), chosen


def solvate(
    model: Any, ff: Any, padding_nm: float, shape: str, salt_molar: float, seed: int
):
    import numpy as np
    from openmm import unit

    random.seed(seed)
    count = model.topology.getNumAtoms()
    source = np.asarray(model.positions.value_in_unit(unit.nanometer))
    model.addSolvent(
        ff,
        model="tip3p",
        padding=padding_nm * unit.nanometer,
        boxShape=shape,
        ionicStrength=salt_molar * unit.molar,
        neutralize=True,
    )
    if not np.array_equal(
        source, np.asarray(model.positions.value_in_unit(unit.nanometer))[:count]
    ):
        raise ValueError("Solvation changed the positions of the solute atoms")


def export_parameters(model: Any, ff: Any, directory: Path) -> None:
    """Write a portable Amber topology with ParmEd."""
    import parmed
    from openmm import app, unit
    from parmed.geometry import box_vectors_to_lengths_and_angles

    system = ff.createSystem(
        model.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    structure = parmed.openmm.load_topology(model.topology, system, xyz=model.positions)
    lengths, angles = box_vectors_to_lengths_and_angles(
        *model.topology.getPeriodicBoxVectors()
    )
    structure.box = [
        *lengths.value_in_unit(unit.angstrom),
        *angles.value_in_unit(unit.degree),
    ]
    structure.save(str(directory / "complex.prmtop"), overwrite=False)
    structure.save(str(directory / "complex.inpcrd"), format="rst7", overwrite=False)


def build_and_minimize(directory: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import openmm
    from openmm import app, unit

    prmtop = app.AmberPrmtopFile(str(directory / "complex.prmtop"))
    inpcrd = app.AmberInpcrdFile(str(directory / "complex.inpcrd"))
    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(
        prmtop.topology,
        system,
        openmm.VerletIntegrator(1.0 * unit.femtosecond),
        platform,
        {"Threads": str(cfg["threads"])},
    )
    simulation.context.setPositions(inpcrd.positions)
    # Added hydrogens and waters need not satisfy the constraints, and projecting onto
    # them can raise the energy of a good start; compare like with like.
    simulation.context.applyConstraints(1e-8)
    before = simulation.context.getState(getEnergy=True, getForces=True)
    initial = float(before.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    openmm.LocalEnergyMinimizer.minimize(
        simulation.context, maxIterations=int(cfg["minimize_iterations"])
    )
    state = simulation.context.getState(
        getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=True
    )
    minimized = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    if not (math.isfinite(initial) and math.isfinite(minimized)):
        raise ValueError("The potential energy is not finite")
    if minimized > initial:
        raise ValueError(
            f"Minimization increased the potential energy from {initial:.6f} to "
            f"{minimized:.6f} kJ/mol"
        )
    (directory / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    (directory / "state.xml").write_text(openmm.XmlSerializer.serialize(state))
    with (directory / "topology.pdb").open("w") as handle:
        app.PDBFile.writeFile(
            prmtop.topology, state.getPositions(), handle, keepIds=True
        )
    nonbonded = next(
        f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    charge = sum(
        float(
            nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        )
        for i in range(system.getNumParticles())
    )
    residues = list(prmtop.topology.residues())
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return {
        "initial_potential_energy_kj_mol": initial,
        "minimized_potential_energy_kj_mol": minimized,
        "atom_count": system.getNumParticles(),
        "water_molecules": sum(r.name in SOLVENT for r in residues),
        "sodium_ions": sum(r.name in SODIUM for r in residues),
        "chloride_ions": sum(r.name in CHLORIDE for r in residues),
        "net_charge_e": charge,
        "forces_after": diagnostics.force_diagnostics(state, unit, prmtop.topology),
        "cell": diagnostics.cell_diagnostics(state, unit, 0.9),
        "protein_geometry": diagnostics.protein_geometry_diagnostics(
            prmtop.topology, np.asarray(positions), system
        ),
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check, parameterise, assemble, solvate and minimise; write the files."""
    from rdkit import Chem

    started = time.perf_counter()
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)
    data = check_data_files()
    chemistry = json.loads(Path(cfg["ligand_chemistry"]).read_text())
    posed, with_hydrogens, charge = read_ligand(
        cfg["ligand_pose"], chemistry, cfg.get("ligand_formal_charge")
    )
    molecule, record = parameterize(with_hydrogens, out)
    model, ff, variants = assemble(
        Path(cfg["protein_pdb"]), [molecule], out, cfg["seed"]
    )
    solvate(
        model,
        ff,
        float(cfg["padding_nm"]),
        cfg["box_shape"],
        float(cfg["ionic_strength_molar"]),
        cfg["seed"],
    )
    export_parameters(model, ff, out)
    summary = build_and_minimize(out, cfg)
    pairs = min(summary["sodium_ions"], summary["chloride_ions"])
    summary.update(
        {
            "ligand_heavy_atoms": Chem.RemoveHs(posed).GetNumAtoms(),
            "ligand_formal_charge": charge,
            "ligand_parameters": record,
            "hydrogen_variants": variants,
            "added_salt_pairs_at_least": pairs,
            "ionic_strength_molar": cfg["ionic_strength_molar"],
            "padding_nm": cfg["padding_nm"],
            "box_shape": cfg["box_shape"],
            "salt_note": (
                "Added NaCl pairs exclude neutralizing counterions and are rounded from the "
                f"replaceable water count using {SALT_WATER_MOLAR} M bulk water; the "
                "concentration in a finite cell is approximate"
            ),
            "force_field": {
                "protein": "amber14/protein.ff14SB",
                "water": "amber14/tip3p",
                "ligand": FORCE_FIELD,
                "charges": CHARGE_MODEL,
                **data,
            },
            "versions": versions(),
            "seed": cfg["seed"],
            "seconds": time.perf_counter() - started,
        }
    )
    (out / "preparation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return summary
