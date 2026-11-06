"""Free energy of a single amino-acid substitution with OpenMM and the hybrid topology of OpenFE.

Runs inside the OpenFE environment. The standard library is imported at the top; OpenMM,
PDBFixer, openmmtools and OpenFE load when a request runs.

`prepare` reads one chain of a PDB file and writes two legs of the thermodynamic cycle: the folded
protein in a water box and a capped tripeptide of the residue and its two neighbors on each side
(Ace-X1-X2-X3-X4-X5-NMe, the mutated residue in the middle) in a smaller water box. Each leg has a
wild-type state and a mutant state in which the side chain of one residue is replaced. `run` builds
the hybrid system of a leg with the hybrid topology factory of OpenFE and samples it with replica
exchange over lambda windows, then analyzes it with MBAR.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

PACKAGES = ("openfe", "gufe", "openmm", "openmmtools", "pymbar", "pdbfixer")
FORCEFIELD = ("amber14-all.xml", "amber14/tip3p.xml")
CUTOFF_NM = 0.9
TIMESTEP_FS = 2.0
BACKBONE = ("N", "CA", "C", "O")
FLANK = 2
DISULFIDE_ANGSTROM = 2.5
CONTACT_BOND_ANGSTROM = 2.0
STANDARD = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
VARIANTS = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}
SUBSTITUTABLE = ("ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "SER", "THR", "ASN", "GLN", "CYS")


def digest(path: Any) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def versions() -> dict[str, str]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - depends on the install
            found[name] = "unknown"
    return found


# ---------------------------------------------------------------------------------------------
# Atom mapping between the side chains of two residues (standard library only)
# ---------------------------------------------------------------------------------------------


def plan_mapping(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> dict[str, Any]:
    """Which atoms of the old residue correspond to atoms of the new residue.

    Each residue is a list of atoms, {"name", "element", "bonds"} with `bonds` the names of the atoms
    bonded to it inside the residue. The backbone heavy atoms N, CA, C and O are mapped. The mapping grows
    from CA along bonds: a heavy atom maps when both residues have an atom of its name and element bonded
    to the mapped atom it was reached from. The hydrogens of a mapped heavy atom map in name order up to
    the smaller count; the rest of the hydrogens, and every atom not reached, are unique to one state.
    """
    by_old = {a["name"]: a for a in old}
    by_new = {a["name"]: a for a in new}
    mapped: dict[str, str] = {}
    queue = []
    for name in BACKBONE:
        if name in by_old and name in by_new and by_old[name]["element"] == by_new[name]["element"]:
            mapped[name] = name
    if "CA" not in mapped:
        raise ValueError("Both residues need the backbone atoms N, CA, C and O to be mapped")
    queue.append("CA")
    seen = {"CA"}
    while queue:
        name = queue.pop(0)
        for other in sorted(by_old[name]["bonds"]):
            if other in seen or by_old[other]["element"] == "H":
                continue
            seen.add(other)
            if (
                other in by_new
                and by_new[other]["element"] == by_old[other]["element"]
                and other in by_new[name]["bonds"]
            ):
                mapped[other] = other
                queue.append(other)

    def hydrogens(table, atom):
        return sorted(b for b in table[atom]["bonds"] if table[b]["element"] == "H")

    for name in list(mapped):
        pairs = zip(hydrogens(by_old, name), hydrogens(by_new, name), strict=False)
        for h_old, h_new in pairs:
            mapped[h_old] = h_new
    # Amide and alpha hydrogens of the backbone bond to atoms outside the residue's heavy side chain.
    old_unique = [a["name"] for a in old if a["name"] not in mapped]
    new_unique = [a["name"] for a in new if a["name"] not in set(mapped.values())]
    return {"core": mapped, "old_unique": old_unique, "new_unique": new_unique}


# ---------------------------------------------------------------------------------------------
# System preparation (OpenMM)
# ---------------------------------------------------------------------------------------------


def _forcefield():
    from openmm import app

    return app.ForceField(*FORCEFIELD)


def _canonical(name: str) -> str:
    return VARIANTS.get(name, name)


def read_chain(path: str, chain_id: str):
    """The standard residues of one chain of the first model, heavy atoms only."""
    from openmm import app

    pdb = app.PDBFile(path)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    drop, dropped = [], {"other_chains": set(), "hetero": set()}
    chains = [c.id for c in modeller.topology.chains()]
    if chain_id not in chains:
        raise ValueError(f"chain_id is {chain_id!r}; the file has the chains {', '.join(chains)}")
    for chain in modeller.topology.chains():
        for residue in chain.residues():
            standard = _canonical(residue.name) in STANDARD
            if chain.id != chain_id:
                dropped["other_chains"].add(chain.id)
                drop.extend(residue.atoms())
            elif not standard:
                dropped["hetero"].add(residue.name)
                drop.extend(residue.atoms())
            else:
                drop.extend(a for a in residue.atoms() if a.element is not None and a.element.symbol in ("H", "D"))
    modeller.delete(drop)
    return modeller, {k: sorted(v) for k, v in dropped.items()}


def complete_terminal_atoms(modeller):
    """Add the terminal OXT; refuse a chain with other heavy atoms missing."""
    import io

    import pdbfixer
    from openmm import app

    buffer = io.StringIO()
    app.PDBFile.writeFile(modeller.topology, modeller.positions, buffer, keepIds=True)
    buffer.seek(0)
    fixer = pdbfixer.PDBFixer(pdbfile=buffer)
    fixer.findMissingResidues()
    if fixer.missingResidues:
        raise ValueError("The chain has missing residues (gaps in the sequence); a continuous chain is required")
    fixer.findMissingAtoms()
    internal = {
        f"{res.name}{res.id}": [a.name for a in atoms]
        for res, atoms in fixer.missingAtoms.items()
    }
    if internal:
        raise ValueError(
            "The chain lacks heavy atoms of "
            + ", ".join(f"{k} ({', '.join(v)})" for k, v in sorted(internal.items()))
            + "; rebuild them first"
        )
    fixer.addMissingAtoms()
    return app.Modeller(fixer.topology, fixer.positions)


def residue_index(topology, residue_number: int) -> int:
    for residue in topology.residues():
        if int(residue.id) == residue_number and not residue.insertionCode.strip():
            return residue.index
    raise ValueError(f"residue_number is {residue_number}; no such residue in the chain")


def residue_atoms(topology, index: int):
    return list(next(r for r in topology.residues() if r.index == index).atoms())


def _heavy_neighbors_ok(modeller, index: int) -> None:
    """Both neighbors on each side exist and are bonded by peptide bonds."""
    import numpy as np

    residues = list(modeller.topology.residues())
    if index < FLANK or index + FLANK >= len(residues):
        raise ValueError(
            f"The residue has fewer than {FLANK} residues on one side; the reference tripeptide needs "
            f"{FLANK} residues on each side"
        )
    positions = np.array(modeller.positions.value_in_unit(modeller.positions.unit))
    for left, right in zip(residues[index - FLANK : index + FLANK], residues[index - FLANK + 1 : index + FLANK + 1], strict=True):
        c = next(a for a in left.atoms() if a.name == "C")
        n = next(a for a in right.atoms() if a.name == "N")
        if float(np.linalg.norm(positions[c.index] - positions[n.index])) * 10 > CONTACT_BOND_ANGSTROM:
            raise ValueError(
                f"Residues {left.id} and {right.id} are not bonded (C-N distance above "
                f"{CONTACT_BOND_ANGSTROM} Å); a continuous stretch of 5 residues is required"
            )


def _check_disulfide(modeller, index: int) -> None:
    import numpy as np

    residue = next(r for r in modeller.topology.residues() if r.index == index)
    if residue.name not in ("CYS", "CYX"):
        return
    positions = np.array(modeller.positions.value_in_unit(modeller.positions.unit))
    sulfurs = [a for r in modeller.topology.residues() if r.name in ("CYS", "CYX") for a in r.atoms() if a.name == "SG"]
    mine = next(a for a in residue.atoms() if a.name == "SG")
    for other in sulfurs:
        if other.index != mine.index and float(np.linalg.norm(positions[mine.index] - positions[other.index])) * 10 < DISULFIDE_ANGSTROM:
            raise ValueError("The cysteine is part of a disulfide bond; a disulfide is outside this capability")


def cap_fragment(modeller):
    """The first and last residue of a fragment become the acetyl and N-methylamide caps."""
    residues = list(modeller.topology.residues())
    first, last = residues[0], residues[-1]
    drop = [a for a in first.atoms() if a.name not in ("C", "O", "CA")]
    drop += [a for a in last.atoms() if a.name not in ("N", "CA")]
    modeller.delete(drop)
    residues = list(modeller.topology.residues())
    # Modeller reads the PDB names of the caps: CH3 for the acetyl methyl, C for the N-methylamide methyl.
    for residue, name, methyl in ((residues[0], "ACE", "CH3"), (residues[-1], "NME", "C")):
        residue.name = name
        for atom in residue.atoms():
            if atom.name == "CA":
                atom.name = methyl
    return modeller


def fragment(modeller, index: int):
    """The capped five-residue stretch around residue `index`, and the index of that residue in it."""
    from openmm import app

    keep = {r.index for r in modeller.topology.residues() if abs(r.index - index) <= FLANK}
    piece = app.Modeller(modeller.topology, modeller.positions)
    piece.delete([r for r in piece.topology.residues() if r.index not in keep])
    return cap_fragment(piece), FLANK


def seed_random(seed: int) -> None:
    """Seed the generators that OpenMM's Modeller uses for hydrogens, ions and water orientation."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy ships with OpenMM
        pass


def variants_for(topology, rows: dict[int, str]) -> list[str | None]:
    """The protonation variant of each residue of a solute, by residue number; None leaves it to the pH."""
    return [
        rows.get(int(residue.id)) if _canonical(residue.name) in STANDARD else None
        for residue in topology.residues()
    ]


def mutate(modeller, index: int, new_name: str, seed: int, ph: float, variants: list[str | None]):
    """The same solute with the residue replaced by `new_name` (PDBFixer template), protonated alike."""
    import io

    import pdbfixer
    from openmm import app

    buffer = io.StringIO()
    app.PDBFile.writeFile(modeller.topology, modeller.positions, buffer, keepIds=False)
    buffer.seek(0)
    fixer = pdbfixer.PDBFixer(pdbfile=buffer)
    residues = list(fixer.topology.residues())
    old = residues[index]
    chain = next(iter(fixer.topology.chains())).id
    fixer.applyMutations([f"{old.name}-{old.id}-{new_name}"], chain)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    mutant = app.Modeller(fixer.topology, fixer.positions)
    seed_random(seed)
    states = list(variants)
    states[index] = None
    mutant.addHydrogens(_forcefield(), pH=ph, variants=states)
    return mutant


def _records(topology, atoms) -> list[dict[str, Any]]:
    members = {a.index: a for a in atoms}
    names = {i: a.name for i, a in members.items()}
    bonds: dict[int, list[str]] = {i: [] for i in members}
    for a, b in topology.bonds():
        if a.index in members and b.index in members:
            bonds[a.index].append(names[b.index])
            bonds[b.index].append(names[a.index])
    return [{"name": a.name, "element": a.element.symbol, "bonds": bonds[a.index]} for a in atoms]


def graft(old_topology, old_positions, target: int, new_atoms, new_positions, new_topology, new_name: str, plan: dict[str, Any]):
    """The old system with the side chain of residue `target` replaced by the atoms of the new residue."""
    import numpy as np
    from openmm import app, unit

    positions = np.array(old_positions.value_in_unit(unit.nanometer))
    mutant = np.array(new_positions.value_in_unit(unit.nanometer))
    old_atoms = residue_atoms(old_topology, target)
    old_by_name = {a.name: a for a in old_atoms}
    new_by_name = {a.name: a for a in new_atoms}
    old_to_new_name = plan["core"]
    topology = app.Topology()
    topology.setPeriodicBoxVectors(old_topology.getPeriodicBoxVectors())
    old_to_new: dict[int, int] = {}
    new_positions_list = []
    new_index_of_new_atom: dict[int, int] = {}
    for chain in old_topology.chains():
        new_chain = topology.addChain(chain.id)
        for residue in chain.residues():
            if residue.index != target:
                added = topology.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
                for atom in residue.atoms():
                    created = topology.addAtom(atom.name, atom.element, added, atom.id)
                    old_to_new[atom.index] = created.index
                    new_positions_list.append(positions[atom.index])
                continue
            added = topology.addResidue(new_name, new_chain, residue.id, residue.insertionCode)
            for old_name, new_atom_name in old_to_new_name.items():
                new_atom = new_by_name[new_atom_name]
                created = topology.addAtom(new_atom.name, new_atom.element, added)
                old_to_new[old_by_name[old_name].index] = created.index
                new_index_of_new_atom[new_atom.index] = created.index
                new_positions_list.append(positions[old_by_name[old_name].index])
            for name in plan["new_unique"]:
                new_atom = new_by_name[name]
                created = topology.addAtom(new_atom.name, new_atom.element, added)
                new_index_of_new_atom[new_atom.index] = created.index
                new_positions_list.append(mutant[new_atom.index])
    new_bonds = set()
    for a, b in old_topology.bonds():
        in_a, in_b = a.residue.index == target, b.residue.index == target
        if in_a and in_b:
            continue
        if (in_a and a.index not in old_to_new) or (in_b and b.index not in old_to_new):
            continue
        new_bonds.add((old_to_new[a.index], old_to_new[b.index]))
    for a, b in new_topology.bonds():
        if a.index in new_index_of_new_atom and b.index in new_index_of_new_atom:
            new_bonds.add((new_index_of_new_atom[a.index], new_index_of_new_atom[b.index]))
    atoms = list(topology.atoms())
    for i, j in sorted(new_bonds):
        topology.addBond(atoms[i], atoms[j])
    core = {old_by_name[o].index: old_to_new[old_by_name[o].index] for o in old_to_new_name}
    return topology, unit.Quantity(np.array(new_positions_list), unit.nanometer), old_to_new, core


MIN_BOX_WIDTH_NM = 2 * CUTOFF_NM + 0.1
DODECAHEDRON_FACE_RATIO = 0.5**0.5
"""The nonbonded cutoff must not exceed half of the width of the box between two faces."""


def box_widths(vectors) -> list[float]:
    """The distance between each pair of opposite faces of a periodic box, in nm."""
    import numpy as np
    from openmm import unit

    a, b, c = (np.array([x.value_in_unit(unit.nanometer) for x in v]) for v in vectors)
    volume = abs(float(np.dot(a, np.cross(b, c))))
    return [volume / float(np.linalg.norm(np.cross(u, v))) for u, v in ((b, c), (c, a), (a, b))]


def solvate(
    modeller, padding_nm: float, ionic_strength_molar: float, seed: int, ph: float, variants: list[str | None]
):
    """The solute with hydrogens in a water box; the padding grows until the box is wide enough.

    Returns the solvated system and the padding that was used. A compact solute with the stated
    padding can give a box narrower than twice the cutoff, which OpenMM refuses.
    """
    from openmm import app, unit

    forcefield = _forcefield()
    seed_random(seed)
    modeller.addHydrogens(forcefield, pH=ph, variants=variants)
    padding = padding_nm
    for _ in range(6):
        trial = app.Modeller(modeller.topology, modeller.positions)
        seed_random(seed)
        trial.addSolvent(
            forcefield,
            model="tip3p",
            padding=padding * unit.nanometer,
            ionicStrength=ionic_strength_molar * unit.molar,
            neutralize=True,
            boxShape="dodecahedron",
        )
        width = min(box_widths(trial.topology.getPeriodicBoxVectors()))
        if width >= MIN_BOX_WIDTH_NM:
            return trial, padding
        # The edge of the box grows by about 1 nm per nm of padding, and the faces of the
        # dodecahedron that are closest lie 1/sqrt(2) of the edge apart.
        padding = round(padding + (MIN_BOX_WIDTH_NM - width) / DODECAHEDRON_FACE_RATIO * 1.05, 3)
    raise ValueError(f"No water box of at least {MIN_BOX_WIDTH_NM:.2f} nm between faces was found for this solute")


RELAX_ITERATIONS = 2000
RELAX_TOLERANCE_KJ_MOL_NM = 10.0


def relax(topology, positions):
    """Positions after an energy minimization of the solvated wild type on the CPU.

    Solvent placement leaves contacts of up to 2e5 kJ/mol/nm between water and ions, which
    the 100 steps of the sampler do not remove in a large box and which turn into NaN. The
    minimizer sums forces on several threads, so the relaxed coordinates agree between runs
    to about 0.01 nm; the solvent placement itself is fixed by the seed.
    """
    import openmm
    from openmm import unit

    context = openmm.Context(
        create_system(topology),
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("CPU"),
    )
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(
        context, RELAX_TOLERANCE_KJ_MOL_NM * unit.kilojoule_per_mole / unit.nanometer, RELAX_ITERATIONS
    )
    return context.getState(getPositions=True).getPositions()


def create_system(topology):
    from openmm import app, unit

    return _forcefield().createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=CUTOFF_NM * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )


def net_charge(system) -> float:
    import openmm
    from openmm import unit

    total = 0.0
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            total += sum(
                force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                for i in range(system.getNumParticles())
            )
    return total


def write_pdb(path: Path, topology, positions) -> None:
    from openmm import app

    with path.open("w") as stream:
        app.PDBFile.writeFile(topology, positions, stream, keepIds=False)


def build_leg(
    solute,
    target: int,
    new_name: str,
    *,
    identity: bool,
    padding_nm: float,
    ionic_strength_molar: float,
    seed: int,
    ph: float,
    directory: Path,
    leg: str,
    numbers: list[int],
    residue_variants: dict[int, str],
) -> dict[str, Any]:
    """Solvate the wild type, build the mutant state, write both states and the mapping."""
    from openmm import unit

    bare = solute
    variants = variants_for(bare.topology, residue_variants)
    mutant_solute = None if identity else mutate(bare, target, new_name, seed, ph, variants)
    wild, padding_nm = solvate(bare, padding_nm, ionic_strength_molar, seed, ph, variants)
    wild.positions = relax(wild.topology, wild.positions)
    old_topology, old_positions = wild.topology, wild.positions
    old_atoms = residue_atoms(old_topology, target)
    if identity:
        new_topology, new_positions = old_topology, old_positions
        old_to_new = {a.index: a.index for a in old_topology.atoms()}
        core = {a.index: a.index for a in old_atoms}
        plan = {"core": {a.name: a.name for a in old_atoms}, "old_unique": [], "new_unique": []}
        new_residue_name = old_atoms[0].residue.name
    else:
        new_atoms = residue_atoms(mutant_solute.topology, target)
        plan = plan_mapping(_records(old_topology, old_atoms), _records(mutant_solute.topology, new_atoms))
        new_topology, new_positions, old_to_new, core = graft(
            old_topology, old_positions, target, new_atoms, mutant_solute.positions, mutant_solute.topology, new_name, plan
        )
        new_residue_name = new_name
    systems = [create_system(old_topology), create_system(new_topology)]
    charges = [net_charge(s) for s in systems]
    if abs(charges[0] - charges[1]) > 1e-3:
        raise ValueError(
            f"The substitution changes the net charge from {charges[0]:+.3f} to {charges[1]:+.3f}; "
            "a charge-changing substitution is outside this capability"
        )
    directory.mkdir(parents=True, exist_ok=True)
    write_pdb(directory / "wild_type.pdb", old_topology, old_positions)
    write_pdb(directory / "mutant.pdb", new_topology, new_positions)
    record = {
        "schema_version": "0.1",
        "leg": leg,
        "wild_type_residue": old_atoms[0].residue.name,
        "mutant_residue": new_residue_name,
        "target_index": target,
        "solute_residue_numbers": numbers,
        "ph": ph,
        "residue_variants": {str(n): v for n, v in sorted(residue_variants.items()) if n in numbers},
        "atoms": systems[0].getNumParticles(),
        "core_atoms": len(plan["core"]),
        "wild_type_unique_atoms": plan["old_unique"],
        "mutant_unique_atoms": plan["new_unique"],
        "net_charge": charges[0],
        "padding_nm": padding_nm,
        "ionic_strength_molar": ionic_strength_molar,
        "seed": seed,
        "box_nm": [[v.value_in_unit(unit.nanometer) for v in row] for row in old_topology.getPeriodicBoxVectors()],
        "old_to_new": [[int(o), int(n)] for o, n in sorted(old_to_new.items())],
        "core_old_to_new": [[int(o), int(n)] for o, n in sorted(core.items())],
        "force_field": list(FORCEFIELD),
        "files": {"wild_type": "wild_type.pdb", "mutant": "mutant.pdb"},
        "sha256": {"wild_type": digest(directory / "wild_type.pdb"), "mutant": digest(directory / "mutant.pdb")},
    }
    (directory / "leg.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def prepare(cfg: dict[str, Any]) -> dict[str, Any]:
    """Write the folded leg and the reference leg of one substitution."""
    started = time.perf_counter()
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    modeller, dropped = read_chain(cfg["structure"], cfg["chain_id"])
    modeller = complete_terminal_atoms(modeller)
    index = residue_index(modeller.topology, cfg["residue_number"])
    residue = next(r for r in modeller.topology.residues() if r.index == index)
    wild = _canonical(residue.name)
    if wild not in SUBSTITUTABLE:
        raise ValueError(f"Residue {cfg['residue_number']} is {residue.name}; the substitutable residues are {', '.join(SUBSTITUTABLE)}")
    if not cfg["identity_check"] and cfg["to"] == wild:
        raise ValueError(f"The residue is already {wild}; set identity_check to run the null substitution")
    _check_disulfide(modeller, index)
    _heavy_neighbors_ok(modeller, index)
    numbers = [int(r.id) for r in modeller.topology.residues()]
    residue_variants = {int(row["residue_number"]): row["variant"] for row in cfg.get("residue_variants", [])}
    legs = {}
    folded = build_leg(
        modeller, index, cfg["to"], identity=cfg["identity_check"], padding_nm=cfg["folded_padding_nm"],
        ionic_strength_molar=cfg["ionic_strength_molar"], seed=cfg["seed"], ph=cfg["ph"], directory=output / "folded",
        leg="folded", numbers=numbers, residue_variants=residue_variants,
    )
    legs["folded"] = folded
    modeller, dropped_unused = read_chain(cfg["structure"], cfg["chain_id"])
    modeller = complete_terminal_atoms(modeller)
    piece, local = fragment(modeller, index)
    legs["reference"] = build_leg(
        piece, local, cfg["to"], identity=cfg["identity_check"], padding_nm=cfg["reference_padding_nm"],
        ionic_strength_molar=cfg["ionic_strength_molar"], seed=cfg["seed"], ph=cfg["ph"], directory=output / "reference",
        leg="reference", numbers=[int(r.id) for r in piece.topology.residues()],
        residue_variants=residue_variants,
    )
    return {
        "legs": legs,
        "dropped": dropped,
        "mutation": f"{wild}{cfg['residue_number']}{legs['folded']['mutant_residue']}",
        "structure_sha256": digest(cfg["structure"]),
        "versions": versions(),
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------------------------
# Sampling and analysis
# ---------------------------------------------------------------------------------------------


def load_leg(directory: Path):
    import numpy as np
    from openmm import app

    record = json.loads((directory / "leg.json").read_text())
    for key, name in record["files"].items():
        if digest(directory / name) != record["sha256"][key]:
            raise ValueError(f"{name} differs from the file that `prepare` wrote; the leg is not intact")
    wild = app.PDBFile(str(directory / record["files"]["wild_type"]))
    mutant = app.PDBFile(str(directory / record["files"]["mutant"]))
    return record, wild, mutant, np


def _platform(name: str, precision: str):
    import openmm

    platform = openmm.Platform.getPlatformByName(name)
    if name == "CUDA":
        platform.setPropertyDefaultValue("Precision", precision)
    return platform


def analysis_class():
    """OpenFE's MBAR analysis with the number of bootstrap samples of the free-energy error as a setting."""
    from openfe.protocols.openmm_utils.multistate_analysis import (
        MultistateEquilFEAnalysis,
    )

    class Analysis(MultistateEquilFEAnalysis):
        bootstraps = 1000

        def get_equil_free_energy(self):
            u_ln = self.analyzer._unbiased_decorrelated_u_ln
            n_l = self.analyzer._unbiased_decorrelated_N_l
            return self._get_free_energy(self.analyzer, u_ln, n_l, type(self).bootstraps, self.units)

    return Analysis


def run_replica(record, wild, mutant, cfg: dict[str, Any], directory: Path) -> dict[str, Any]:
    import numpy as np
    import openmm
    from openfe.protocols.openmm_rfe import _rfe_utils
    from openff.units import unit as ffunit
    from openmm import unit
    from openmmtools import cache, mcmc
    from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer

    temperature = cfg["temperature_kelvin"] * unit.kelvin
    systems = []
    for state in (wild, mutant):
        system = create_system(state.topology)
        system.addForce(openmm.MonteCarloBarostat(cfg["pressure_bar"] * unit.bar, temperature, 25))
        system.setDefaultPeriodicBoxVectors(*state.topology.getPeriodicBoxVectors())
        systems.append(system)
    factory = _rfe_utils.relative.HybridTopologyFactory(
        systems[0], wild.positions, wild.topology, systems[1], mutant.positions, mutant.topology,
        old_to_new_atom_map={o: n for o, n in record["old_to_new"]},
        old_to_new_core_atom_map={o: n for o, n in record["core_old_to_new"]},
        use_dispersion_correction=False,
        softcore_alpha=0.85,
        softcore_LJ_v2=True,
        softcore_LJ_v2_alpha=0.85,
        interpolate_old_and_new_14s=False,
    )
    steps = int(round(cfg["time_per_iteration_ps"] * 1000 / TIMESTEP_FS))
    production = int(round(cfg["production_ps"] / cfg["time_per_iteration_ps"]))
    equilibration = int(round(cfg["equilibration_ps"] / cfg["time_per_iteration_ps"]))
    directory.mkdir(parents=True, exist_ok=True)
    reporter = MultiStateReporter(
        storage=directory / "replica.nc",
        analysis_particle_indices=np.arange(factory.hybrid_system.getNumParticles()),
        checkpoint_interval=max(1, production // 10),
        checkpoint_storage="replica_checkpoint.nc",
        position_interval=0,
    )
    move = mcmc.LangevinDynamicsMove(
        timestep=TIMESTEP_FS * unit.femtosecond,
        collision_rate=1.0 / unit.picosecond,
        n_steps=steps,
        reassign_velocities=False,
        n_restart_attempts=20,
        constraint_tolerance=1e-6,
    )
    sampler = _rfe_utils.multistate.HybridRepexSampler(
        mcmc_moves=move,
        hybrid_system=factory.hybrid_system,
        hybrid_positions=factory.hybrid_positions,
        online_analysis_interval=None,
        number_of_iterations=production,
    )
    platform = _platform(cfg["platform"], cfg["precision"])
    sampler.setup(
        n_replicas=cfg["lambda_windows"],
        reporter=reporter,
        lambda_protocol=_rfe_utils.lambdaprotocol.LambdaProtocol(functions="default", windows=cfg["lambda_windows"]),
        temperature=temperature,
        endstates=False,
        minimization_platform=cfg["platform"],
        minimization_steps=100,
    )
    sampler.energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
    sampler.sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
    started = time.perf_counter()
    sampler.equilibrate(max(1, equilibration))
    sampler.run(n_iterations=production)
    sampling_seconds = time.perf_counter() - started
    reporter.close()
    reader = MultiStateReporter(
        storage=directory / "replica.nc", checkpoint_storage="replica_checkpoint.nc", open_mode="r"
    )
    try:
        analysis_type = analysis_class()
        analysis_type.bootstraps = cfg["bootstraps"]
        analysis = analysis_type(
            reader,
            sampling_method="repex",
            result_units=ffunit.kilocalorie_per_mole,
            forward_reverse_samples=cfg["forward_reverse_samples"],
        )
        analyzer = MultiStateSamplerAnalyzer(reader)
        effective = analyzer.mbar.compute_effective_sample_number().tolist()
        free_energy = float(analysis.free_energy.m)
        error = float(analysis.free_energy_error.m)
        if not (math.isfinite(free_energy) and math.isfinite(error)):
            raise ValueError(
                f"The free energy of the leg is not finite ({free_energy}, error {error}); "
                "the sampling did not produce usable energies"
            )
        overlap = np.asarray(analysis.free_energy_overlaps["matrix"]).tolist()
        fractional = analysis.forward_and_reverse_free_energies
        forward_reverse = None
        if fractional is not None:
            forward_reverse = {
                "fractions": np.asarray(fractional["fractions"]).tolist(),
                "forward_kcal_mol": np.asarray(fractional["forward_DGs"].to(ffunit.kilocalorie_per_mole).m).tolist(),
                "reverse_kcal_mol": np.asarray(fractional["reverse_DGs"].to(ffunit.kilocalorie_per_mole).m).tolist(),
                "forward_error_kcal_mol": np.asarray(fractional["forward_dDGs"].to(ffunit.kilocalorie_per_mole).m).tolist(),
                "reverse_error_kcal_mol": np.asarray(fractional["reverse_dDGs"].to(ffunit.kilocalorie_per_mole).m).tolist(),
            }
        analysis.close()
    finally:
        reader.close()
    return {
        "trajectory_sha256": digest(directory / "replica.nc"),
        "delta_g_kcal_mol": free_energy,
        "mbar_error_kcal_mol": error,
        "overlap_matrix": overlap,
        "effective_samples": effective,
        "forward_reverse": forward_reverse,
        "sampling_seconds": sampling_seconds,
        "atoms": factory.hybrid_system.getNumParticles(),
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Sample every replica of one leg and gather its record."""
    started = time.perf_counter()
    os.environ["OPENMM_CPU_THREADS"] = str(cfg["threads"])
    os.environ["OMP_NUM_THREADS"] = str(cfg["threads"])
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    record, wild, mutant, _ = load_leg(Path(cfg["leg_directory"]))
    replicas = []
    for number in range(cfg["replicas"]):
        found = run_replica(record, wild, mutant, cfg, output / f"replica-{number + 1}")
        found["id"] = f"replica-{number + 1}"
        replicas.append(found)
    method = {
        "versions": versions(),
        "force_field": record["force_field"],
        "timestep_fs": TIMESTEP_FS,
        "lambda_windows": cfg["lambda_windows"],
        "equilibration_ps": cfg["equilibration_ps"],
        "production_ps": cfg["production_ps"],
        "time_per_iteration_ps": cfg["time_per_iteration_ps"],
        "temperature_kelvin": cfg["temperature_kelvin"],
        "pressure_bar": cfg["pressure_bar"],
        "platform": cfg["platform"],
        "precision": cfg["precision"] if cfg["platform"] == "CUDA" else "native",
        "error_bootstraps": cfg["bootstraps"],
        "forward_reverse_samples": cfg["forward_reverse_samples"],
    }
    leg = {
        "schema_version": "0.1",
        "leg": record["leg"],
        "wild_type_residue": record["wild_type_residue"],
        "mutant_residue": record["mutant_residue"],
        "temperature_kelvin": cfg["temperature_kelvin"],
        "inputs_sha256": record["sha256"],
        "method": method,
        "replicas": replicas,
    }
    (output / "leg-result.json").write_text(json.dumps(leg, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"record": leg, "method": method, "seconds": time.perf_counter() - started}
