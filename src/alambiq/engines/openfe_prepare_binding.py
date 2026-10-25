"""Inputs for relative binding free energies of ring congeners anchored on a deposited ring.

Runs inside the OpenFE environment. The standard library is imported at the top;
everything else is loaded when a request runs.

The protein is the one chain of a deposited complex, with heterogens and water removed
and hydrogens added at the requested pH; its heavy-atom coordinates are checked to be
untouched. The deposited complex's own six-carbon ring (benzene) is the anchor: every
congener, a molecule with exactly one six-membered aromatic carbon ring and its
substituents, is placed by laying its ring on the anchor ring in each of the twelve
symmetry-related ways, optimising the substituents with the ring held fixed, and keeping
the placement whose substituents overlap the protein least. Every congener after the
first must substitute the same ring position as the first. Each placed congener is given
AshGC charges, and each requested edge an atom mapping from OpenFE's LOMAP mapper.
"""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any

import openff_prepare_complex as ashgc

CLASH_DISTANCE_ANGSTROM = 2.2
POSES = 12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ring_candidates(base: Any, anchor: Any, heavy: Any, embedding_seed: int):
    """Every way to lay the congener's ring on the anchor ring, scored against the protein."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign

    ring = list(base.GetRingInfo().AtomRings()[0])
    candidates, poses = [], []
    for direction in (-1, 1):
        for shift in range(6):
            candidate = Chem.Mol(base)
            if AllChem.EmbedMolecule(candidate, randomSeed=embedding_seed) != 0:
                raise ValueError(
                    "RDKit could not embed the congener in three dimensions"
                )
            if AllChem.MMFFOptimizeMolecule(candidate, maxIters=1000) != 0:
                raise ValueError("The congener's geometry did not converge in MMFF94")
            mapping = [(ring[(shift + direction * i) % 6], i) for i in range(6)]
            rdMolAlign.AlignMol(candidate, anchor, atomMap=mapping)
            for target, source in mapping:
                candidate.GetConformer().SetAtomPosition(
                    target, anchor.GetConformer().GetAtomPosition(source)
                )
            force = AllChem.MMFFGetMoleculeForceField(
                candidate, AllChem.MMFFGetMoleculeProperties(candidate)
            )
            for atom in ring:
                force.AddFixedPoint(atom)
            force.Initialize()
            if force.Minimize(maxIts=1000) != 0:
                raise ValueError(
                    "The substituents did not relax with the ring held fixed"
                )
            added = candidate.GetConformer().GetPositions()[
                [
                    a.GetIdx()
                    for a in candidate.GetAtoms()
                    if a.GetAtomicNum() > 1 and a.GetIdx() not in ring
                ]
            ]
            distances = (
                np.linalg.norm(added[:, None, :] - heavy[None, :, :], axis=2)
                if len(added)
                else np.full((1, 1), np.inf)
            )
            penalty = np.maximum(0, CLASH_DISTANCE_ANGSTROM - distances)
            substituted = [
                i
                for i in ring
                if any(
                    n.GetAtomicNum() > 1 and n.GetIdx() not in ring
                    for n in candidate.GetAtomWithIdx(i).GetNeighbors()
                )
            ]
            if len(substituted) != 1:
                raise ValueError(
                    "A congener must carry heavy-atom substituents on exactly one ring position"
                )
            candidates.append(
                {
                    "shift": shift,
                    "direction": direction,
                    "ring_mapping": mapping,
                    "substitution_anchor": dict(mapping)[substituted[0]],
                    "overlap_penalty_angstrom2": float((penalty**2).sum()),
                    "minimum_added_atom_distance_angstrom": float(distances.min()),
                }
            )
            poses.append(candidate)
    return candidates, poses


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepare the protein, place and charge the congeners, and map the edges."""
    import numpy as np
    import openfe
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    from rdkit import Chem
    from rdkit.Chem import AllChem

    started = time.perf_counter()
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    data = ashgc.check_data_files()
    platform = openmm.Platform.getPlatformByName("CPU")
    platform.setPropertyDefaultValue("Threads", str(cfg["threads"]))
    source = Path(cfg["structure"])
    if cfg["structure_format"] == "mmcif":
        with source.open() as stream:
            fixer = PDBFixer(pdbxfile=stream, platform=platform)
    else:
        fixer = PDBFixer(filename=str(source), platform=platform)

    anchors = [
        r for r in fixer.topology.residues() if r.name == cfg["anchor_residue_name"]
    ]
    if len(anchors) != 1:
        raise ValueError(
            f"The structure has {len(anchors)} residues named {cfg['anchor_residue_name']}; "
            "exactly one anchor ring is needed"
        )
    if not all(a.element.atomic_number == 6 for a in anchors[0].atoms()):
        raise ValueError("The anchor residue must consist of carbon atoms only")
    ring_positions = {
        a.name: np.asarray(fixer.positions[a.index].value_in_unit(unit.angstrom))
        for a in anchors[0].atoms()
    }
    if set(ring_positions) != set(cfg["anchor_atom_names"]):
        raise ValueError(
            f"The anchor residue's atoms {sorted(ring_positions)} are not the declared "
            f"{sorted(cfg['anchor_atom_names'])}"
        )
    chains = list(fixer.topology.chains())
    if cfg["chain_id"] not in [c.id for c in chains]:
        raise ValueError(f"The structure has no chain {cfg['chain_id']}")
    fixer.removeChains([i for i, c in enumerate(chains) if c.id != cfg["chain_id"]])
    fixer.removeHeterogens(keepWater=False)
    deposited = len(list(fixer.topology.residues()))
    before = {
        (a.residue.id, a.name): np.asarray(
            fixer.positions[a.index].value_in_unit(unit.nanometer)
        )
        for a in fixer.topology.atoms()
    }
    fixer.findMissingResidues()
    chain_residues = {
        i: len(list(c.residues())) for i, c in enumerate(fixer.topology.chains())
    }
    missing = fixer.missingResidues
    internal = [
        key for key in missing if key[1] not in (0, chain_residues.get(key[0], -1))
    ]
    if internal:
        raise ValueError(
            "The structure is missing residues inside the chain (insertion points "
            f"{sorted(internal)}); only missing terminal residues can be completed"
        )
    if missing and not cfg["complete_terminals"]:
        raise ValueError(
            "The structure is missing terminal residues "
            f"{ {k: list(v) for k, v in missing.items()} }; set complete_terminals to model them"
        )
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=seed)
    after = {
        (a.residue.id, a.name): np.asarray(
            fixer.positions[a.index].value_in_unit(unit.nanometer)
        )
        for a in fixer.topology.atoms()
    }
    if not all(np.array_equal(before[k], after[k]) for k in before):
        raise ValueError("Completing the structure moved a deposited heavy atom")
    model = app.Modeller(fixer.topology, fixer.positions)
    force_field = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    variants = model.addHydrogens(force_field, pH=float(cfg["ph"]), platform=platform)
    heavy_names = {
        (a.residue.id, a.name): np.asarray(
            model.positions[a.index].value_in_unit(unit.nanometer)
        )
        for a in model.topology.atoms()
        if a.element.symbol != "H"
    }
    if not all(np.array_equal(before[k], heavy_names[k]) for k in before):
        raise ValueError("Adding hydrogens moved a deposited heavy atom")
    protein = out / "protein.pdb"
    with protein.open("w") as stream:
        app.PDBFile.writeFile(model.topology, model.positions, stream, keepIds=True)
    roundtrip = app.PDBFile(str(protein))
    exported = {
        (a.residue.id, a.name): np.asarray(
            roundtrip.positions[a.index].value_in_unit(unit.nanometer)
        )
        for a in roundtrip.topology.atoms()
        if a.element.symbol != "H"
    }
    if not all(np.max(np.abs(before[k] - exported[k])) <= 5.1e-05 for k in before):
        raise ValueError("The written protein differs from the deposited heavy atoms")

    heavy = np.array(
        [
            pos.value_in_unit(unit.angstrom)
            for a, pos in zip(model.topology.atoms(), model.positions, strict=True)
            if a.element.symbol != "H"
        ]
    )
    anchor = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    if (
        AllChem.EmbedMolecule(
            anchor, randomSeed=int(cfg["ligands"][0]["embedding_seed"])
        )
        != 0
    ):
        raise ValueError("RDKit could not embed the anchor ring")
    for i in range(6):
        anchor.GetConformer().SetAtomPosition(
            i, ring_positions[cfg["anchor_atom_names"][i]]
        )
    force = AllChem.MMFFGetMoleculeForceField(
        anchor, AllChem.MMFFGetMoleculeProperties(anchor)
    )
    for i in range(6):
        force.AddFixedPoint(i)
    force.Initialize()
    if force.Minimize(maxIts=1000) != 0:
        raise ValueError("The anchor ring's hydrogens did not relax")

    ligands, components, substitution_anchor = {}, {}, None
    for index, ligand in enumerate(cfg["ligands"]):
        name = ligand["id"]
        if index == 0:
            molecule, selected, candidates = Chem.Mol(anchor), None, []
        else:
            base = Chem.MolFromSmiles(ligand["smiles"])
            if (
                base is None
                or Chem.GetFormalCharge(base) != 0
                or len(Chem.GetMolFrags(base)) != 1
            ):
                raise ValueError(
                    f"{name}: a congener must be one connected neutral molecule"
                )
            if any(
                str(info.specified) == "Unspecified"
                for info in Chem.FindPotentialStereo(base)
            ):
                raise ValueError(
                    f"{name}: the SMILES leaves stereochemistry unspecified"
                )
            rings = base.GetRingInfo().AtomRings()
            if (
                len(rings) != 1
                or len(rings[0]) != 6
                or not all(
                    base.GetAtomWithIdx(i).GetIsAromatic()
                    and base.GetAtomWithIdx(i).GetAtomicNum() == 6
                    for i in rings[0]
                )
            ):
                raise ValueError(
                    f"{name}: a congener has exactly one ring, six aromatic carbons"
                )
            base = Chem.AddHs(base)
            candidates, poses = ring_candidates(
                base, anchor, heavy, int(ligand["embedding_seed"])
            )
            eligible = [
                i
                for i in range(len(poses))
                if substitution_anchor is None
                or candidates[i]["substitution_anchor"] == substitution_anchor
            ]
            selected = min(
                eligible,
                key=lambda i: (
                    candidates[i]["overlap_penalty_angstrom2"],
                    -candidates[i]["minimum_added_atom_distance_angstrom"],
                    i,
                ),
            )
            substitution_anchor = candidates[selected]["substitution_anchor"]
            molecule = poses[selected]
        with_scratch = out / f"{name}-scratch"
        with_scratch.mkdir()
        off, charges = ashgc.parameterize(molecule, with_scratch)
        component = openfe.SmallMoleculeComponent.from_openff(off, name=name)
        target = out / f"{name}.sdf"
        target.write_text(component.to_sdf())
        components[name] = component
        ligands[name] = {
            "smiles": ligand["smiles"],
            "sdf": str(target),
            "sha256": sha(target),
            "atoms": off.n_atoms,
            "charge_sum_e": charges["charge_sum_e"],
            "partial_charges_e": charges["partial_charges_e"],
            "selected_pose_index": selected,
            "candidate_poses": candidates,
        }
    edges = []
    for edge in cfg["edges"]:
        a, b = edge["a"], edge["b"]
        mapper = openfe.LomapAtomMapper(element_change=False)
        mappings = list(mapper.suggest_mappings(components[a], components[b]))
        if not mappings:
            raise ValueError(f"LOMAP found no mapping for edge {edge['id']}")
        best = min(mappings, key=lambda m: float(np.sum(m.get_distances() ** 2)))
        edges.append(
            {
                "id": edge["id"],
                "a": a,
                "b": b,
                "atom_mapping": sorted(
                    [list(pair) for pair in best.componentA_to_componentB.items()]
                ),
                "mapped_distances_angstrom": best.get_distances().tolist(),
            }
        )

    residues = list(model.topology.residues())
    result = {
        "protein_pdb": str(protein),
        "protein_sha256": sha(protein),
        "deposited_residues": deposited,
        "completed_residues": len(residues),
        "modeled_terminal_residues": [
            {"insertion_index": key[1], "names": list(names)}
            for key, names in missing.items()
        ],
        "atom_count": len(list(model.topology.atoms())),
        "hydrogen_variants": variants,
        "ph": float(cfg["ph"]),
        "ligands": ligands,
        "edges": edges,
        "force_field": data,
        "versions": ashgc.versions(),
        "seed": seed,
        "seconds": time.perf_counter() - started,
    }
    return result
