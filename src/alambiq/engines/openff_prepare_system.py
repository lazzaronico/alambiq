"""A protein with positioned ligands, parameterised and optionally solvated, with fixed observed atoms.

Runs inside the OpenFF environment. The caller supplies a protein PDB (heavy atoms, and
whatever crystal waters are kept), positioned ligands with explicit hydrogens, and which
residues were modeled rather than observed. Terminal oxygens and hydrogens are completed,
the ligands get Sage 2.3.0 parameters and AshGC charges, and the unsolvated model is relaxed
in vacuum with every observed heavy atom held exactly where it was and the modeled residues
tied loosely to where they were placed. If solvent is asked for, TIP3P with neutralizing
ions and salt is added around the relaxed model and the solvent and hydrogens alone are
minimized, with every solute heavy atom held. The exported system has normal masses and no
restraint; the check that no held atom moved is exact.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import openff_prepare_complex as complex_prep
import openmm_common as diagnostics


def _minimise(
    system: Any,
    positions: Any,
    held: set[int],
    springs: dict[int, list[float]],
    cfg: dict[str, Any],
):
    """Minimise with held atoms massless and optional positional springs.

    Returns the minimized positions (nm), and the energy before and after on the
    unmodified system.
    """
    import numpy as np
    import openmm
    from openmm import unit

    def energy(candidate: Any, coordinates: Any) -> float:
        context = openmm.Context(
            candidate,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("CPU"),
            {"Threads": str(cfg["threads"])},
        )
        context.setPositions(coordinates)
        value = context.getState(getEnergy=True).getPotentialEnergy()
        return float(value.value_in_unit(unit.kilojoule_per_mole))

    working = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    for index in held:
        working.setParticleMass(index, 0)
    if springs:
        spring = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        spring.addGlobalParameter("k", cfg["modeled_restraint_k_kj_mol_nm2"])
        for name in ("x0", "y0", "z0"):
            spring.addPerParticleParameter(name)
        for index, xyz in springs.items():
            spring.addParticle(index, xyz)
        working.addForce(spring)
    context = openmm.Context(
        working,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("CPU"),
        {"Threads": str(cfg["threads"])},
    )
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(
        context,
        cfg["minimizer_tolerance_kj_mol_nm"],
        cfg["minimization_max_iterations"],
    )
    moved = context.getState(getPositions=True).getPositions(asNumpy=True)
    final = np.array(moved.value_in_unit(unit.nanometer))
    start = np.array(positions.value_in_unit(unit.nanometer))
    before = energy(system, positions)
    after = energy(system, final * unit.nanometer)
    if not (
        math.isfinite(before) and math.isfinite(after) and np.isfinite(final).all()
    ):
        raise ValueError("The relaxation produced a non-finite energy or coordinate")
    if held:
        rows = sorted(held)
        if not np.array_equal(final[rows], start[rows]):
            raise ValueError("A held heavy atom moved during relaxation")
    return final, before, after


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Assemble, relax, optionally solvate, and write the system."""
    import numpy as np
    import openmm
    import parmed
    from openmm import app, unit
    from rdkit import Chem

    started = time.perf_counter()
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)
    data = complex_prep.check_data_files()
    molecules, details = [], []
    for position, ligand in enumerate(cfg["ligands"]):
        supplied = list(Chem.SDMolSupplier(ligand["sdf"], removeHs=False))
        if len(supplied) != 1 or supplied[0] is None:
            raise ValueError(
                f"{ligand['id']}: supply exactly one positioned SDF molecule"
            )
        charge = Chem.GetFormalCharge(supplied[0])
        if charge != ligand["formal_charge"]:
            raise ValueError(
                f"{ligand['id']}: the molecule has formal charge {charge}, but "
                f"{ligand['formal_charge']} was declared"
            )
        scratch = out / f"ligand-{position}-scratch"
        scratch.mkdir()
        molecule, record = complex_prep.parameterize(
            supplied[0], scratch, ligand["residue_name"], ligand["chain"]
        )
        molecules.append(molecule)
        details.append(
            {"id": ligand["id"], "residue_name": ligand["residue_name"], **record}
        )
    model, ff, variants = complex_prep.assemble(
        Path(cfg["protein_pdb"]), molecules, out, cfg["seed"]
    )
    sites = []
    for ligand in cfg["ligands"]:
        rows = [
            r
            for r in model.topology.residues()
            if r.chain.id == ligand["chain"] and r.name == ligand["residue_name"]
        ]
        if len(rows) != 1:
            raise ValueError(
                f"{ligand['id']}: expected one ligand residue {ligand['residue_name']} "
                f"in chain {ligand['chain']}, found {len(rows)}"
            )
        atoms = list(rows[0].atoms())
        sites.append(
            {
                "chain": ligand["chain"],
                "residue_name": ligand["residue_name"],
                "residue_number": rows[0].index + 1,
                "first_atom": atoms[0].index + 1,
                "atom_count": len(atoms),
            }
        )
    modeled = {(r["chain"], str(r["number"])) for r in cfg["modeled_residues"]}
    found = set()
    held: set[int] = set()
    modeled_heavy: set[int] = set()
    for atom in model.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        key = (atom.residue.chain.id, atom.residue.id)
        if key in modeled:
            modeled_heavy.add(atom.index)
            found.add(key)
        else:
            held.add(atom.index)
    if found != modeled:
        raise ValueError(
            "Declared modeled residues are not in the protein: "
            + ", ".join(f"{c}{n}" for c, n in sorted(modeled - found))
        )
    solute_atoms = model.topology.getNumAtoms()
    start = np.array(model.positions.value_in_unit(unit.nanometer))
    vacuum = ff.createSystem(
        model.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    relaxed, before, after = _minimise(
        vacuum,
        model.positions,
        held,
        {i: start[i].tolist() for i in modeled_heavy},
        cfg,
    )
    displacement = (
        float(
            np.linalg.norm(
                relaxed[sorted(modeled_heavy)] - start[sorted(modeled_heavy)], axis=1
            ).max()
            * 10
        )
        if modeled_heavy
        else 0.0
    )
    model.positions = relaxed * unit.nanometer
    periodic = cfg["solvent"] == "tip3p"
    if periodic:
        complex_prep.solvate(
            model,
            ff,
            cfg["padding_nm"],
            cfg["box_shape"],
            cfg["ionic_strength_molar"],
            cfg["seed"],
        )
        complex_prep.export_parameters(model, ff, out)
    else:
        structure = parmed.openmm.load_topology(
            model.topology, vacuum, xyz=model.positions
        )
        structure.save(str(out / "complex.prmtop"), overwrite=False)
        structure.save(str(out / "complex.inpcrd"), format="rst7", overwrite=False)
    prmtop = app.AmberPrmtopFile(str(out / "complex.prmtop"))
    inpcrd = app.AmberInpcrdFile(str(out / "complex.inpcrd"))
    system = prmtop.createSystem(
        nonbondedMethod=app.PME if periodic else app.NoCutoff,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=app.HBonds if periodic else None,
        rigidWater=periodic,
        removeCMMotion=False,
    )
    solute_heavy = {
        a.index
        for a in prmtop.topology.atoms()
        if a.index < solute_atoms and a.element is not None and a.element.symbol != "H"
    }
    # Held atoms are massless, which constraints cannot involve, so the solvent stage
    # uses flexible bonds; the exported system keeps its constraints.
    flexible = prmtop.createSystem(
        nonbondedMethod=app.PME if periodic else app.NoCutoff,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    final, before_final, after_final = _minimise(
        flexible, inpcrd.positions, solute_heavy, {}, cfg
    )
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("CPU"),
        {"Threads": str(cfg["threads"])},
    )
    context.setPositions(final * unit.nanometer)
    state = context.getState(
        getPositions=True, getEnergy=True, getForces=True, enforcePeriodicBox=False
    )
    (out / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    (out / "state.xml").write_text(openmm.XmlSerializer.serialize(state))
    with (out / "topology.pdb").open("w") as stream:
        app.PDBFile.writeFile(
            prmtop.topology, state.getPositions(), stream, keepIds=True
        )
    nonbonded = next(
        f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    net = sum(
        float(
            nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        )
        for i in range(system.getNumParticles())
    )
    residues = list(prmtop.topology.residues())
    summary: dict[str, Any] = {
        "periodic": periodic,
        "atom_count": system.getNumParticles(),
        "solute_atoms": solute_atoms,
        "observed_heavy_atoms_held": len(held),
        "modeled_heavy_atoms_relaxed": len(modeled_heavy),
        "maximum_modeled_displacement_angstrom": displacement,
        "vacuum_energy_before_kj_mol": before,
        "vacuum_energy_after_kj_mol": after,
        "solvent_energy_before_kj_mol": before_final,
        "solvent_energy_after_kj_mol": after_final,
        "water_molecules": sum(r.name in complex_prep.SOLVENT for r in residues),
        "sodium_ions": sum(r.name in complex_prep.SODIUM for r in residues),
        "chloride_ions": sum(r.name in complex_prep.CHLORIDE for r in residues),
        "net_charge_e": net,
        "forces_after": diagnostics.force_diagnostics(state, unit, prmtop.topology),
        "protein_geometry": diagnostics.protein_geometry_diagnostics(
            prmtop.topology, final, system
        ),
        "cell": diagnostics.cell_diagnostics(state, unit, 0.9) if periodic else None,
        "ligands": details,
        "ligand_sites": sites,
        "hydrogen_variants": variants,
        "force_field": {
            "protein": "amber14/protein.ff14SB",
            "water": "amber14/tip3p",
            "ligand": complex_prep.FORCE_FIELD,
            "charges": complex_prep.CHARGE_MODEL,
            **data,
        },
        "versions": complex_prep.versions(),
        "seed": cfg["seed"],
        "seconds": time.perf_counter() - started,
    }
    (out / "preparation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return summary
