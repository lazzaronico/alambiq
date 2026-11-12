"""A rigid fit of one structure on another over all polymer chains, and the distances from a
ligand of the fitted structure to atoms of the reference.

Pure functions over structure text. Residues are paired by author residue number and residue
name within paired chains; chains are paired by the assignment with the smallest alpha-carbon
RMSD, or by the assignment the request gives.
"""

from __future__ import annotations

import itertools
from typing import Any

MAX_CHAINS = 6
MIN_PAIRS = 3
MAX_TARGETS = 20
BOND_TOLERANCE = 0.45


def _gemmi():
    try:
        import gemmi
    except ModuleNotFoundError as exc:
        if exc.name != "gemmi":
            raise
        raise ModuleNotFoundError(
            "Measuring a ligand site requires the optional dossier extra (Gemmi). "
            "Run python -m pip install 'alambiq[dossier]' in the Python environment "
            "that runs Alambiq.",
            name="gemmi",
        ) from exc
    return gemmi


def _numpy():
    try:
        import numpy
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        raise ModuleNotFoundError(
            "Measuring a ligand site requires NumPy. Run python -m pip install "
            "'alambiq[quantitative]' in the Python environment that runs Alambiq.",
            name="numpy",
        ) from exc
    return numpy


def read_structure(text: str, label: str):
    """The first model of an mmCIF or PDB text, by the content and not by a file name."""
    gemmi = _gemmi()
    head = text.lstrip()[:200]
    try:
        if head.startswith("data_") or "_atom_site." in text[:20000]:
            block = gemmi.cif.read_string(text).sole_block()
            structure = gemmi.make_structure_from_block(block)
        else:
            structure = gemmi.read_pdb_string(text)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} is neither an mmCIF nor a PDB file: {exc}") from exc
    if len(structure) == 0 or sum(len(c) for c in structure[0]) == 0:
        raise ValueError(f"{label} holds no atoms")
    return structure


def _residue_key(residue):
    return (residue.seqid.num, residue.seqid.icode.strip(), residue.name)


def alpha_carbons(structure) -> dict[str, dict[tuple, Any]]:
    """Alpha carbon positions of amino acid residues, by chain and residue."""
    gemmi = _gemmi()
    found: dict[str, dict[tuple, Any]] = {}
    for chain in structure[0]:
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info is None or not info.is_amino_acid():
                continue
            atom = residue.find_atom("CA", "*")
            if atom is None:
                continue
            found.setdefault(chain.name, {})[_residue_key(residue)] = atom.pos.tolist()
    return found


def kabsch(moving, fixed):
    """The rotation and the translation that best superpose `moving` on `fixed`."""
    np = _numpy()
    centre_moving, centre_fixed = moving.mean(0), fixed.mean(0)
    u, _, vt = np.linalg.svd((moving - centre_moving).T @ (fixed - centre_fixed))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    translation = centre_fixed - rotation @ centre_moving
    moved = moving @ rotation.T + translation
    rmsd = float(np.sqrt(((moved - fixed) ** 2).sum(1).mean()))
    return rotation, translation, rmsd


def _noncollinear(points) -> bool:
    np = _numpy()
    vectors = points - points[0]
    return bool(np.linalg.matrix_rank(vectors, tol=1e-6) >= 2)


def fit_chains(
    mobile, reference, chain_map: dict[str, str] | None = None, tolerance: float = 0.1
) -> dict:
    """The superposition of `mobile` on `reference` by alpha carbons of paired residues.

    Without `chain_map` every assignment of the mobile polymer chains to the reference
    polymer chains is tried, and the one with the smallest RMSD is kept. The assignments
    whose RMSD lies within `tolerance` of the smallest are returned as `equivalent`.
    """
    np = _numpy()
    m, r = alpha_carbons(mobile), alpha_carbons(reference)
    if not m or not r:
        side = "mobile" if not m else "reference"
        raise ValueError(
            f"The {side} structure has no amino acid residue with an alpha carbon"
        )
    if chain_map:
        for chain, other in chain_map.items():
            if chain not in m:
                raise ValueError(
                    f"chain_map names the mobile chain {chain}, which has no protein residues"
                )
            if other not in r:
                raise ValueError(
                    f"chain_map names the reference chain {other}, which has no protein residues"
                )
        if len(set(chain_map.values())) != len(chain_map):
            raise ValueError(
                "chain_map assigns two mobile chains to one reference chain"
            )
        mobile_chains = sorted(chain_map)
        assignments = [tuple(chain_map[c] for c in mobile_chains)]
    else:
        mobile_chains, references = sorted(m), sorted(r)
        if len(mobile_chains) != len(references):
            raise ValueError(
                f"The mobile structure has {len(mobile_chains)} protein chains and the reference "
                f"{len(references)}; give chain_map to say which chain corresponds to which"
            )
        if len(mobile_chains) > MAX_CHAINS:
            raise ValueError(
                f"More than {MAX_CHAINS} protein chains; give chain_map to say which chain "
                "corresponds to which"
            )
        assignments = list(itertools.permutations(references))

    pairs: dict[tuple[str, str], tuple] = {}
    for chain in mobile_chains:
        for other in {a[mobile_chains.index(chain)] for a in assignments}:
            keys = sorted(set(m[chain]) & set(r[other]))
            pairs[(chain, other)] = (
                np.array([m[chain][k] for k in keys]).reshape(-1, 3),
                np.array([r[other][k] for k in keys]).reshape(-1, 3),
            )

    scored = []
    for assignment in assignments:
        moving = np.concatenate(
            [pairs[(c, o)][0] for c, o in zip(mobile_chains, assignment, strict=True)]
        )
        fixed = np.concatenate(
            [pairs[(c, o)][1] for c, o in zip(mobile_chains, assignment, strict=True)]
        )
        if len(moving) < MIN_PAIRS or not _noncollinear(moving):
            continue
        rotation, translation, rmsd = kabsch(moving, fixed)
        scored.append((rmsd, assignment, rotation, translation, len(moving)))
    if not scored:
        raise ValueError(
            f"Fewer than {MIN_PAIRS} residues correspond, or they lie on a line. Residues "
            "correspond when they have the same author number and residue name in paired chains"
        )
    scored.sort(key=lambda row: row[0])
    rmsd, assignment, rotation, translation, count = scored[0]
    equivalent = [
        (dict(zip(mobile_chains, a, strict=True)), rot, tra)
        for value, a, rot, tra, _ in scored
        if value - rmsd <= tolerance
    ]
    return {
        "rmsd_angstrom": rmsd,
        "fitted_residues": count,
        "chain_assignment": dict(zip(mobile_chains, assignment, strict=True)),
        "assignments_tested": len(scored),
        "runner_up_rmsd_angstrom": scored[1][0] if len(scored) > 1 else None,
        "rotation": rotation,
        "translation": translation,
        "equivalent": equivalent,
        "mobile_residues": sum(len(m[c]) for c in mobile_chains),
        "reference_residues": sum(len(r[o]) for o in assignment),
    }


def find_ligand(
    structure,
    residue_name: str | None,
    chain_id: str | None,
    residue_number: int | None,
):
    """The one residue of the structure that matches the selector, and its chain name."""
    if residue_name is None and chain_id is None:
        raise ValueError("The ligand selector names a residue_name, a chain_id or both")
    found = []
    for chain in structure[0]:
        if chain_id is not None and chain.name != chain_id:
            continue
        for residue in chain:
            if residue_name is not None and residue.name != residue_name:
                continue
            if residue_number is not None and residue.seqid.num != residue_number:
                continue
            if residue.name == "HOH":
                continue
            found.append((chain.name, residue))
    if len(found) != 1:
        raise ValueError(
            f"The ligand selector matches {len(found)} residues in the mobile structure; "
            "it must match exactly one. Add chain_id or residue_number to narrow it"
        )
    return found[0]


def _heavy(residue):
    return [a for a in residue if a.element.name not in ("H", "D")]


def ligand_atoms(residue) -> list[dict]:
    """Name, element and bonded heavy atoms of a ligand, with bonds perceived from distances."""
    atoms = _heavy(residue)
    out = []
    for atom in atoms:
        bonded = []
        for other in atoms:
            if other is atom:
                continue
            reach = atom.element.covalent_r + other.element.covalent_r + BOND_TOLERANCE
            if atom.pos.dist(other.pos) <= reach:
                bonded.append(other.name)
        out.append(
            {
                "name": atom.name,
                "element": atom.element.name,
                "bonded_to": sorted(bonded),
            }
        )
    return out


def measure_site(
    mobile,
    reference,
    *,
    ligand: dict,
    targets: list[dict],
    chain_map: dict[str, str] | None = None,
    tolerance: float = 0.1,
) -> tuple[dict, Any]:
    """The fit, the ligand in the reference frame, and the distances to the targets.

    Returns the record and the mobile structure moved into the reference frame.
    """
    gemmi = _gemmi()
    np = _numpy()
    if not 0 <= tolerance <= 5:
        raise ValueError("equivalent_fit_angstrom is between 0 and 5 Å")
    if not 1 <= len(targets) <= MAX_TARGETS:
        raise ValueError(f"targets lists 1 to {MAX_TARGETS} reference atoms")
    labels = [t["label"] for t in targets]
    if len(set(labels)) != len(labels):
        raise ValueError("The labels of the targets are not distinct")
    fit = fit_chains(mobile, reference, chain_map, tolerance)
    chain_name, residue = find_ligand(
        mobile,
        ligand.get("residue_name"),
        ligand.get("chain_id"),
        ligand.get("residue_number"),
    )
    atoms = _heavy(residue)
    if not atoms:
        raise ValueError("The ligand has no heavy atoms")
    positions = {a.name: np.array(a.pos.tolist()) for a in atoms}
    sites = []
    for target in targets:
        found = []
        for chain in reference[0]:
            for ref in chain:
                if ref.name != target["residue_name"]:
                    continue
                atom = ref.find_atom(target["atom"], "*")
                if atom is not None:
                    found.append(
                        (chain.name, ref.seqid.num, np.array(atom.pos.tolist()))
                    )
        if not found:
            raise ValueError(
                f"The reference has no atom {target['atom']} in a residue named "
                f"{target['residue_name']} (target {target['label']})"
            )
        named = target.get("ligand_atom")
        if named is not None and named not in positions:
            raise ValueError(
                f"The ligand has no atom {named}; its atoms are {sorted(positions)}"
            )
        sites.append((target, found))

    def measure(rotation, translation):
        moved = {n: rotation @ v + translation for n, v in positions.items()}
        rows = []
        for target, found in sites:
            instances = []
            for chain_id, number, point in found:
                distances = {
                    n: float(np.linalg.norm(v - point)) for n, v in moved.items()
                }
                name = target.get("ligand_atom") or min(distances, key=distances.get)
                instances.append(
                    {
                        "chain_id": chain_id,
                        "residue_number": number,
                        "ligand_atom": name,
                        "distance_angstrom": distances[name],
                    }
                )
            instances.sort(key=lambda row: (row["distance_angstrom"], row["chain_id"]))
            rows.append(instances)
        return moved, rows

    rotation, translation = fit["rotation"], fit["translation"]
    moved, chosen = measure(rotation, translation)
    centroid = np.mean(list(moved.values()), axis=0)
    others = [measure(rot, tra)[1] for _, rot, tra in fit["equivalent"][1:]]
    measurements = []
    for index, (target, _) in enumerate(sites):
        nearest = [chosen[index][0]["distance_angstrom"]] + [
            rows[index][0]["distance_angstrom"] for rows in others
        ]
        measurements.append(
            {
                "label": target["label"],
                "residue_name": target["residue_name"],
                "atom": target["atom"],
                "ligand_atom_requested": target.get("ligand_atom"),
                "nearest": chosen[index][0],
                "instances": chosen[index],
                "nearest_range_angstrom": [min(nearest), max(nearest)],
            }
        )

    record = {
        "fit": {
            "rmsd_angstrom": fit["rmsd_angstrom"],
            "fitted_residues": fit["fitted_residues"],
            "mobile_residues": fit["mobile_residues"],
            "reference_residues": fit["reference_residues"],
            "chain_assignment": fit["chain_assignment"],
            "assignments_tested": fit["assignments_tested"],
            "equivalent_fits": len(fit["equivalent"]),
            "runner_up_rmsd_angstrom": fit["runner_up_rmsd_angstrom"],
            "rotation": [[float(x) for x in row] for row in rotation],
            "translation_angstrom": [float(x) for x in translation],
        },
        "ligand": {
            "chain_id": chain_name,
            "residue_number": residue.seqid.num,
            "residue_name": residue.name,
            "atoms": ligand_atoms(residue),
            "centroid_angstrom": [float(x) for x in centroid],
        },
        "measurements": measurements,
        "gemmi_version": gemmi.__version__,
    }
    transform = gemmi.Transform()
    transform.mat.fromlist([[float(x) for x in row] for row in rotation])
    transform.vec.fromlist([float(x) for x in translation])
    placed = mobile.clone()
    for model in placed:
        model.transform_pos_and_adp(transform)
    return record, placed


def structure_text(structure) -> str:
    return structure.make_mmcif_document().as_string()
