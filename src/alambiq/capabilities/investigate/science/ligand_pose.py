"""The pose of a ligand along a trajectory against its pose in a reference structure.

Pure functions. MDTraj, NumPy and RDKit are imported where they are used.
"""

from __future__ import annotations

from typing import Any

MAX_FRAMES = 100_000
MAX_LIGAND_ATOMS = 200
MAX_MAPPINGS = 5_000
MAX_ELEMENTS = 20_000_000
FIT_CUTOFF = (4.0, 1000.0)
THRESHOLD = (0.1, 20.0)
CHUNK_FRAMES = (1, 64)
IMAGING = ("as-stored", "whole-molecules")


def check_pose(
    *,
    fit_cutoff_angstrom: float,
    threshold_angstrom: float,
    imaging: str,
    chunk_frames: int,
) -> None:
    if not FIT_CUTOFF[0] <= fit_cutoff_angstrom <= FIT_CUTOFF[1]:
        raise ValueError(
            f"fit_cutoff_angstrom must be from {FIT_CUTOFF[0]:g} to {FIT_CUTOFF[1]:g}"
        )
    if not THRESHOLD[0] <= threshold_angstrom <= THRESHOLD[1]:
        raise ValueError(
            f"threshold_angstrom must be from {THRESHOLD[0]:g} to {THRESHOLD[1]:g}"
        )
    if imaging not in IMAGING:
        raise ValueError("imaging is `as-stored` or `whole-molecules`")
    if not CHUNK_FRAMES[0] <= chunk_frames <= CHUNK_FRAMES[1]:
        raise ValueError(
            f"chunk_frames must be an integer from {CHUNK_FRAMES[0]} to {CHUNK_FRAMES[1]}"
        )


def _chain(residue) -> str:
    return residue.chain.chain_id or str(residue.chain.index)


def find_residue(topology, selector: dict[str, Any], where: str):
    """The one residue that matches the chain, number and name of the selector."""
    matches = [
        r
        for r in topology.residues
        if r.name == selector["residue_name"]
        and r.resSeq == selector["residue_number"]
        and _chain(r) == selector["chain_id"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{where}: residue {selector['residue_name']}{selector['residue_number']} of "
            f"chain {selector['chain_id']} matches {len(matches)} residues; it must match "
            "exactly one"
        )
    return matches[0]


def heavy_atoms(residue, where: str) -> list[int]:
    """Indices of the non-hydrogen atoms of a residue."""
    atoms = []
    for atom in residue.atoms:
        if atom.element is None:
            raise ValueError(f"{where}: atom {atom.name} has no element")
        if atom.element.symbol not in ("H", "D"):
            atoms.append(atom.index)
    if not 1 <= len(atoms) <= MAX_LIGAND_ATOMS:
        raise ValueError(
            f"{where}: the ligand has {len(atoms)} heavy atoms; 1 to {MAX_LIGAND_ATOMS} "
            "are supported"
        )
    return atoms


def ligand_graph(elements: list[str], xyz_angstrom, smiles: str, where: str):
    """The ligand as a molecule: bonds from the coordinates, bond orders from the SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError("RDKit cannot read the ligand SMILES")
    template = Chem.RemoveHs(template)
    if template.GetNumAtoms() != len(elements):
        raise ValueError(
            f"{where}: the ligand has {len(elements)} heavy atoms and the SMILES "
            f"{template.GetNumAtoms()}"
        )
    lines = [
        f"HETATM{i:5d} {element:<2s}{i % 100:02d} LIG A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        for i, (element, (x, y, z)) in enumerate(
            zip(elements, xyz_angstrom, strict=True), 1
        )
    ]
    pose = Chem.MolFromPDBBlock(
        "\n".join(lines) + "\nEND\n", removeHs=False, proximityBonding=True
    )
    if pose is None or pose.GetNumAtoms() != len(elements):
        raise ValueError(f"{where}: RDKit cannot read the ligand coordinates")
    try:
        graph = AllChem.AssignBondOrdersFromTemplate(template, pose)
    except ValueError as exc:
        raise ValueError(
            f"{where}: the SMILES does not match the heavy atoms and bonds of the ligand "
            f"({exc})"
        ) from exc
    if graph.GetNumBonds() != template.GetNumBonds():
        raise ValueError(
            f"{where}: the SMILES does not match the heavy atoms and bonds of the ligand "
            f"(it has {template.GetNumBonds()} bonds and the coordinates give "
            f"{graph.GetNumBonds()})"
        )
    return graph


def symmetric_pairings(top_graph, reference_graph):
    """Every way to pair the ligand atoms of the trajectory with those of the reference.

    Returns an array of shape (pairings, atoms): row k gives, for each atom i of the
    trajectory ligand, the atom of the reference ligand it is paired with. The rows
    are one graph isomorphism composed with each automorphism of the ligand.
    """
    import numpy as np

    match = reference_graph.GetSubstructMatch(top_graph)
    if len(match) != top_graph.GetNumAtoms():
        raise ValueError(
            "The ligand of the trajectory and the ligand of the reference are not the "
            "same molecule"
        )
    autos = top_graph.GetSubstructMatches(
        top_graph, uniquify=False, maxMatches=MAX_MAPPINGS + 1
    )
    if len(autos) > MAX_MAPPINGS:
        raise ValueError(
            f"The ligand has more than {MAX_MAPPINGS} symmetry-equivalent atom pairings"
        )
    iso = np.asarray(match)
    return np.stack([iso[np.asarray(a)] for a in autos])


def smallest_rmsd(xyz, reference_pairs):
    """The RMSD of each frame to the reference over the symmetric pairings.

    `xyz` has shape (frames, atoms, 3) and `reference_pairs` (pairings, atoms, 3), in
    the same unit. Returns the smallest RMSD of each frame, in that unit.
    """
    import numpy as np

    pairings, atoms, _ = reference_pairs.shape
    step = max(1, MAX_ELEMENTS // (pairings * atoms * 3))
    out = np.empty(len(xyz), dtype=np.float64)
    for start in range(0, len(xyz), step):
        block = xyz[start : start + step, None].astype(np.float64)
        squared = ((block - reference_pairs[None]) ** 2).sum(axis=3).mean(axis=2)
        out[start : start + step] = np.sqrt(squared.min(axis=1))
    return out


def nearest_image_shift(ligand_centroid, anchor_centroid, boxes):
    """For each frame, the lattice vector that puts the ligand nearest to the anchor.

    `boxes` has shape (frames, 3, 3), the rows being the cell vectors. The 27 cells
    around the original one are searched, which is exact for a ligand within one cell
    of the anchor.
    """
    import numpy as np

    steps = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    shifts = np.einsum("sa,fax->sfx", steps, boxes)
    distance = np.linalg.norm(
        ligand_centroid[None] + shifts - anchor_centroid[None], axis=2
    )
    best = distance.argmin(axis=0)
    return shifts[best, np.arange(len(best))]


def statistics(values) -> dict[str, float]:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def measure_pose(
    topology,
    trajectory,
    reference,
    *,
    ligand: dict[str, Any],
    reference_ligand: dict[str, Any],
    smiles: str,
    fit_cutoff_angstrom: float,
    threshold_angstrom: float,
    imaging: str,
    chunk_frames: int,
    chain_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Stream the trajectory once and measure the ligand pose of every frame."""
    import mdtraj as md
    import numpy as np

    top = md.load_pdb(str(topology)).topology
    with md.formats.DCDTrajectoryFile(str(trajectory)) as handle:
        count = len(handle)
    if not 1 <= count <= MAX_FRAMES:
        raise ValueError(
            f"The trajectory has {count} frames; 1 to {MAX_FRAMES} are supported"
        )
    reference_frame = md.load_pdb(str(reference))
    ref_top = reference_frame.topology

    lig_residue = find_residue(top, ligand, "ligand")
    ref_residue = find_residue(ref_top, reference_ligand, "reference_ligand")
    lig_atoms = heavy_atoms(lig_residue, "ligand")
    ref_atoms = heavy_atoms(ref_residue, "reference_ligand")
    ref_xyz = reference_frame.xyz[0] * 10.0

    # The pocket: alpha carbons of the reference residues near the reference ligand.
    ligand_xyz_ref = ref_xyz[ref_atoms]
    pocket: list[int] = []
    for residue in ref_top.residues:
        if not residue.is_protein:
            continue
        idx = [a.index for a in residue.atoms if a.element and a.element.symbol != "H"]
        if not idx:
            continue
        gaps = np.linalg.norm(
            ref_xyz[idx][:, None, :] - ligand_xyz_ref[None, :, :], axis=2
        )
        if gaps.min() <= fit_cutoff_angstrom:
            pocket += [a.index for a in residue.atoms if a.name == "CA"]
    if len(pocket) < 3:
        raise ValueError(
            f"{len(pocket)} alpha carbons lie within {fit_cutoff_angstrom:g} Å of the "
            "reference ligand; the fit needs at least 3. Raise fit_cutoff_angstrom"
        )
    index = {}
    for atom in top.atoms:
        key = (_chain(atom.residue), atom.residue.resSeq, atom.residue.name, atom.name)
        index[key] = -1 if key in index else atom.index
    fit = []
    chains = chain_map or {}
    for i in pocket:
        atom = ref_top.atom(i)
        chain = _chain(atom.residue)
        key = (
            chains.get(chain, chain),
            atom.residue.resSeq,
            atom.residue.name,
            atom.name,
        )
        found = index.get(key)
        if found is None or found < 0:
            raise ValueError(
                f"The reference alpha carbon {atom.residue.name}{atom.residue.resSeq} of "
                f"chain {chain} (topology chain {key[0]}) matches "
                f"{0 if found is None else 'more than one'} atoms "
                "of the topology; the pocket must match by exact chain, number and name"
            )
        fit.append(found)
    fit_ref = pocket
    ref_fit_xyz = reference_frame.xyz[0, fit_ref]
    if np.linalg.matrix_rank(ref_fit_xyz - ref_fit_xyz.mean(axis=0), tol=1e-6) < 2:
        raise ValueError(
            "The alpha carbons of the pocket are collinear in the reference"
        )

    # The graphs come from the reference and from the first frame of the trajectory.
    with md.formats.DCDTrajectoryFile(str(trajectory)) as handle:
        first = handle.read_as_traj(top, n_frames=1)
    elements = [top.atom(i).element.symbol for i in lig_atoms]
    ref_elements = [ref_top.atom(i).element.symbol for i in ref_atoms]
    top_graph = ligand_graph(elements, first.xyz[0, lig_atoms] * 10.0, smiles, "ligand")
    ref_graph = ligand_graph(ref_elements, ligand_xyz_ref, smiles, "reference_ligand")
    pairings = symmetric_pairings(top_graph, ref_graph)
    reference_pairs = ref_xyz[ref_atoms][pairings]
    reference_centroid = ligand_xyz_ref.mean(axis=0)

    rmsd = np.empty(count)
    centroid = np.empty(count)
    fit_rmsd = np.empty(count)
    offset = 0
    size = min(chunk_frames, max(1, 4_000_000 // top.n_atoms))
    with md.formats.DCDTrajectoryFile(str(trajectory)) as handle:
        while offset < count:
            try:
                chunk = handle.read_as_traj(top, n_frames=size)
            except ValueError as exc:
                raise ValueError(
                    f"The trajectory and the topology disagree about the atoms: {exc}"
                ) from exc
            if not len(chunk):
                break
            if not np.isfinite(chunk.xyz).all():
                raise ValueError("The trajectory has a non-finite coordinate")
            end = offset + chunk.n_frames
            if end > count:
                raise ValueError("The trajectory has more frames than it declared")
            if imaging == "whole-molecules":
                boxes = chunk.unitcell_vectors
                if (
                    boxes is None
                    or not np.isfinite(boxes).all()
                    or np.any(np.linalg.det(boxes) <= 0)
                ):
                    raise ValueError(
                        "Whole-molecule imaging requires a positive-volume cell in every "
                        "frame"
                    )
                if top.n_bonds:
                    chunk.make_molecules_whole(inplace=True)
                xyz = chunk.xyz.copy()
                shift = nearest_image_shift(
                    xyz[:, lig_atoms].mean(axis=1), xyz[:, fit].mean(axis=1), boxes
                )
                xyz[:, lig_atoms] += shift[:, None, :]
                chunk.xyz = xyz
            chunk.superpose(reference_frame, atom_indices=fit, ref_atom_indices=fit_ref)
            moved = chunk.xyz[:, lig_atoms] * 10.0
            rmsd[offset:end] = smallest_rmsd(moved, reference_pairs)
            centroid[offset:end] = np.linalg.norm(
                moved.mean(axis=1) - reference_centroid, axis=1
            )
            residual = chunk.xyz[:, fit] - ref_fit_xyz[None]
            fit_rmsd[offset:end] = (
                np.sqrt((residual**2).sum(axis=2).mean(axis=1)) * 10.0
            )
            offset = end
    if offset != count or not np.isfinite(rmsd).all():
        raise ValueError(
            f"The measurement read {offset} of {count} frames, or a value is non-finite"
        )
    inside = rmsd <= threshold_angstrom
    beyond = np.flatnonzero(~inside)
    return {
        "frames": count,
        "ligand": f"{lig_residue.name}{lig_residue.resSeq} (chain {_chain(lig_residue)})",
        "reference_ligand": (
            f"{ref_residue.name}{ref_residue.resSeq} (chain {_chain(ref_residue)})"
        ),
        "heavy_atoms": len(lig_atoms),
        "pairings": int(len(pairings)),
        "fit_atoms": len(fit),
        "rmsd": rmsd,
        "centroid_shift": centroid,
        "fit_rmsd": fit_rmsd,
        "frames_within": int(inside.sum()),
        "first_beyond": int(beyond[0]) if len(beyond) else None,
        "rdkit_version": _rdkit_version(),
    }


def _rdkit_version() -> str:
    from rdkit import rdBase

    return rdBase.rdkitVersion
