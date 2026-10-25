"""Geometry-defined IUPAC beta-methylene labels for scalar-coupling endpoints."""

# Independent implementation of the signed-volume convention described in
# Tejero et al., J Biomol NMR (2013), doi:10.1007/s10858-013-9753-7, Fig. 1.
# No PDBStat code or runtime is used. Assignment never reads measured couplings.
BETA_NEXT = dict.fromkeys(
    "ARG ASN ASP GLN GLU HIS HID HIE HIP LEU LYS MET PHE PRO TRP TYR".split(), "CG"
) | {"CYS": "SG", "CYX": "SG", "SER": "OG"}


def _signed_sides(trajectory, indices):
    import mdtraj as md
    import numpy as np

    pairs = [
        [indices["previous"], indices["parent"]],
        [indices["parent"], indices["next"]],
        [indices["parent"], indices["HB2"]],
        [indices["parent"], indices["HB3"]],
    ]
    vectors = md.compute_displacements(trajectory, pairs, periodic=True)
    normal = np.cross(vectors[:, 0], vectors[:, 1])
    denominators = np.linalg.norm(normal, axis=1)[:, None] * np.linalg.norm(
        vectors[:, 2:], axis=2
    )
    if not np.isfinite(vectors).all() or np.any(denominators < 1e-12):
        raise ValueError("IUPAC beta-methylene assignment has degenerate geometry")
    return np.einsum("fhi,fi->fh", vectors[:, 2:], normal) / denominators


def beta_methylene_assignments(trajectory, observations):
    """Map requested HB2/HB3 names to indices using one topology coordinate frame."""
    import numpy as np

    if len(trajectory) != 1:
        raise ValueError("IUPAC assignment requires one topology coordinate frame")
    requested = {
        (atom["chain_id"], atom["residue_number"], atom["residue_name"])
        for row in observations
        for atom in row["atoms"]
        if atom["atom_name"] in {"HB2", "HB3"}
    }
    topology = trajectory.topology
    adjacency = {atom.index: set() for atom in topology.atoms}
    for a, b in topology.bonds:
        adjacency[a.index].add(b.index)
        adjacency[b.index].add(a.index)
    overrides, assignments = {}, []
    for chain, number, name in sorted(requested):
        residues = [
            r
            for r in topology.residues
            if (r.chain.chain_id, r.resSeq, r.name) == (chain, number, name)
        ]
        if name not in BETA_NEXT or len(residues) != 1:
            raise ValueError(
                f"IUPAC beta-methylene assignment requires one supported residue: {chain}:{number}:{name}"
            )
        required = {
            "previous": "CA",
            "parent": "CB",
            "next": BETA_NEXT[name],
            "HB2": "HB2",
            "HB3": "HB3",
        }
        indices = {}
        for role, atom_name in required.items():
            matches = [a for a in residues[0].atoms if a.name == atom_name]
            expected_element = (
                "H"
                if role in {"HB2", "HB3"}
                else ("S" if atom_name == "SG" else "O" if atom_name == "OG" else "C")
            )
            if len(matches) != 1 or matches[0].element.symbol != expected_element:
                raise ValueError(
                    f"IUPAC beta-methylene assignment needs exact {atom_name}"
                )
            indices[role] = matches[0].index
        if adjacency[indices["parent"]] != {
            indices[role] for role in ("previous", "next", "HB2", "HB3")
        }:
            raise ValueError(
                "IUPAC beta-methylene assignment requires the four CB bonds"
            )
        sides = _signed_sides(trajectory, indices)[0]
        if not np.isfinite(sides).all() or sides[0] * sides[1] >= -1e-8:
            raise ValueError(
                "IUPAC beta hydrogens must occupy opposite nondegenerate sides"
            )
        swapped = bool(sides[0] < 0)
        original = dict(indices)
        if swapped:
            indices["HB2"], indices["HB3"] = indices["HB3"], indices["HB2"]
        for h in ("HB2", "HB3"):
            overrides[original[h]] = indices[h]
        assignments.append(
            {
                "chain_id": chain,
                "residue_number": number,
                "residue_name": name,
                "atom_indices": indices,
                "source_hydrogen_names": {
                    h: topology.atom(indices[h]).name for h in ("HB2", "HB3")
                },
                "swapped": swapped,
            }
        )
    validate_beta_methylene(trajectory, assignments)
    return overrides, assignments


def validate_beta_methylene(trajectory, assignments):
    """Every frame must retain the declared handedness, including across PBC."""
    import numpy as np

    for assignment in assignments:
        sides = _signed_sides(trajectory, assignment["atom_indices"])
        if (
            not np.isfinite(sides).all()
            or np.any(sides[:, 0] <= 1e-4)
            or np.any(sides[:, 1] >= -1e-4)
        ):
            raise ValueError(
                "IUPAC beta-methylene handedness changed or became degenerate in the trajectory"
            )
