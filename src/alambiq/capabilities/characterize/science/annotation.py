"""Coordinate-derived residue annotations of one structure model.

Secondary structure is MDTraj's DSSP eight-state assignment on protein polymers
and surface area is Shrake-Rupley, absolute and per observed atom, in square
angstrom. Only observed heavy atoms count, water is excluded, and every other
retained partner occludes solvent. Positions the model does not observe are kept
and marked unobserved.

Limits. Secondary structure and surface area describe the one model supplied,
which may be a prediction or a crystal in a packing that differs from the
protein in solution; a residue that contacts a ligand or partner absent from
the file has a higher computed exposure than in the complex. The MDTraj version
that produced the numbers is recorded in the result, not enforced.
"""

import hashlib
from collections import OrderedDict
from importlib.metadata import version

from alambiq.capabilities.characterize.science.dossier import inspect_structure
from alambiq.capabilities.shared.sequences import parse_fasta

SETTINGS = {
    "probe_radius_angstrom": 1.4,
    "sphere_points": 960,
    "atom_selection": "Observed heavy atoms, excluding HOH/WAT/DOD water; all retained partners occlude solvent",
    "secondary_structure": "MDTraj DSSP eight-state assignments on protein polymers",
    "surface_area": "Shrake-Rupley absolute observed-atom area, angstrom squared",
    "chain_continuity": "Consecutive label positions with complete N/CA/C/O and C-N distance <= 2 angstrom",
    "coordinate_context": "Selected deposited model, blank/A alternatives, direct coordinates",
}
BACKBONE = {"N", "CA", "C", "O"}


def coordinate_trajectory(dossier):
    """Keep source residue identities beside a calculation-only topology."""
    import mdtraj as md
    import numpy as np

    groups = OrderedDict()
    for atom in dossier["atoms"]:
        if atom["elem"].upper() in {"H", "D"} or atom["resn"] in {"HOH", "WAT", "DOD"}:
            continue
        prop = atom["properties"]
        label = None if atom["hetflag"] else prop.get("label_seq_id")
        if not atom["hetflag"] and (type(label) is not int or label < 1):
            raise ValueError(
                "Coordinate annotation requires explicit polymer label_seq_id "
                f"for source atom {prop['source_atom_id']}; author numbering remains available for display"
            )
        key = (
            atom["chain"],
            label,
            prop["auth_residue"],
            prop["insertion_code"],
            atom["resn"],
        )
        groups.setdefault(key, []).append(atom)
    if not groups:
        raise ValueError("Coordinate annotation requires observed heavy atoms")
    topology, xyz, identities, previous, chain = md.Topology(), [], [], None, None
    polymer_keys, source_ids = set(), set()
    chain_order = {
        name: i for i, name in enumerate(dict.fromkeys(k[0] for k in groups))
    }
    ordered = sorted(
        groups.items(),
        key=lambda item: (chain_order[item[0][0]], item[0][1] or 1_000_000),
    )
    for (asym, label, author, insertion, name), atoms in ordered:
        names = {a["atom"] for a in atoms}
        if len(names) != len(atoms):
            raise ValueError(
                "Coordinate annotation requires distinct atom names per selected residue; inspect alternate locations"
            )
        if label is not None:
            if (asym, label) in polymer_keys:
                raise ValueError(
                    "Coordinate annotation requires one residue per polymer label position"
                )
            polymer_keys.add((asym, label))
        positions = {a["atom"]: np.array([a[k] for k in "xyz"]) for a in atoms}
        complete = label is not None and BACKBONE <= names
        continuous = (
            complete
            and previous is not None
            and previous["complete"]
            and asym == previous["asym"]
            and label == previous["label"] + 1
            and np.linalg.norm(previous["positions"]["C"] - positions["N"]) <= 2.0
        )
        if not continuous:
            chain = topology.add_chain(chain_id=asym)
        residue = topology.add_residue(name, chain, resSeq=label or len(identities) + 1)
        for atom in atoms:
            atom_id = atom["properties"]["source_atom_id"]
            if atom_id in source_ids:
                raise ValueError(
                    "Coordinate annotation requires distinct source atom identifiers"
                )
            source_ids.add(atom_id)
            try:
                element = md.element.get_by_symbol(atom["elem"])
            except KeyError as exc:
                raise ValueError(
                    f"MDTraj element required for source atom {atom_id}: {atom['elem']}"
                ) from exc
            topology.add_atom(atom["atom"], element, residue)
            xyz.append([atom[k] / 10 for k in "xyz"])
        identities.append(
            {
                "label_asym_id": asym,
                "label_seq_id": label,
                "auth_seq_id": author,
                "insertion_code": insertion,
                "residue_name": name,
                "source_atom_ids": [a["properties"]["source_atom_id"] for a in atoms],
                "backbone_status": "complete" if complete else "incomplete",
            }
        )
        previous = {
            "asym": asym,
            "label": label,
            "complete": complete,
            "positions": positions,
        }
    return md.Trajectory(np.asarray(xyz)[None], topology), identities


def annotate_coordinates(dossier, model_number):
    """Annotate the exact dossier model, preserving unobserved sequence positions."""
    import mdtraj as md
    import numpy as np

    mdtraj_version = version("mdtraj")
    sequence = parse_fasta(f">query\n{dossier['sequence']}")[1]
    if hashlib.sha256(sequence.encode()).hexdigest() != dossier["sequence_sha256"]:
        raise ValueError("Dossier sequence hash mismatch")
    text = dossier["source_texts"]["structure"]
    if (
        hashlib.sha256(text.encode()).hexdigest()
        != dossier["sources"]["structure"]["sha256"]
    ):
        raise ValueError("Input size or SHA-256 mismatch")
    chains, atoms = inspect_structure(text, sequence, model_number)
    if chains != dossier["chains"] or atoms != dossier["atoms"]:
        raise ValueError(
            "Dossier geometry and residue mapping must match the exact source model"
        )
    trajectory, identities = coordinate_trajectory(dossier)
    polymer_residues = [
        i for i, row in enumerate(identities) if row["label_seq_id"] is not None
    ]
    if not polymer_residues:
        raise ValueError(
            "Coordinate annotation requires observed protein-polymer heavy atoms"
        )
    polymer_set = set(polymer_residues)
    protein = trajectory.atom_slice(
        [a.index for a in trajectory.topology.atoms if a.residue.index in polymer_set]
    )
    secondary = dict(
        zip(
            polymer_residues, md.compute_dssp(protein, simplified=False)[0], strict=True
        )
    )
    try:
        areas = md.shrake_rupley(
            trajectory, probe_radius=0.14, n_sphere_points=960, mode="residue"
        )[0]
    except KeyError as exc:
        raise ValueError(
            f"MDTraj solvent-accessibility radius required for element {exc}"
        ) from exc
    if not np.isfinite(areas).all() or (areas < 0).any():
        raise ValueError(
            "Every observed residue requires finite nonnegative solvent area"
        )
    calculated = {}
    for index, (identity, area) in enumerate(zip(identities, areas, strict=True)):
        if identity["label_seq_id"] is not None:
            ss = secondary[index]
            calculated[(identity["label_asym_id"], identity["label_seq_id"])] = {
                **identity,
                "secondary_structure": None if ss == "NA" else str(ss),
                "sasa_angstrom2": float(area) * 100,
            }
    rows = []
    for chain in chains:
        observed = {r["label_seq_id"]: r for r in chain["residues"]}
        for position in range(1, len(chain["sequence"]) + 1):
            source = observed.get(position)
            qpos = (
                source["query_position"]
                if source
                else (
                    position + chain["query_offset"]
                    if chain["query_offset"] is not None
                    else None
                )
            )
            if qpos is not None and not 1 <= qpos <= len(sequence):
                qpos = None
            result = calculated.get((chain["label_asym_id"], position))
            rows.append(
                {
                    "label_asym_id": chain["label_asym_id"],
                    "label_seq_id": position,
                    "query_position": qpos,
                    "sequence_letter": chain["sequence"][position - 1],
                    "auth_seq_id": source["auth_seq_id"] if source else None,
                    "insertion_code": source["insertion_code"] if source else "",
                    "residue_name": source["residue_name"] if source else None,
                    "coordinate_status": "observed" if result else "unobserved",
                    "backbone_status": "unobserved",
                    "source_atom_ids": [],
                    "secondary_structure": None,
                    "sasa_angstrom2": None,
                    **(result or {}),
                }
            )
    return {
        "schema_version": "0.1",
        "mdtraj_version": mdtraj_version,
        "source_structure_sha256": dossier["sources"]["structure"]["sha256"],
        "query_sequence_sha256": dossier["sequence_sha256"],
        "model_number": model_number,
        "settings": dict(SETTINGS),
        "atom_count": trajectory.n_atoms,
        "partner_residue_count": sum(r["label_seq_id"] is None for r in identities),
        "residues": rows,
    }
