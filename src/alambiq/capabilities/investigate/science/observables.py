"""Exact molecular selections and structural observables for streamed trajectories."""

import math
import re


def atom_identity(atom):
    return {
        "chain_id": atom.residue.chain.chain_id or str(atom.residue.chain.index),
        "residue_number": atom.residue.resSeq,
        "residue_name": atom.residue.name,
        "atom_name": atom.name,
    }


def validate_observables(rows):
    if not isinstance(rows, list) or len(rows) > 100:
        raise ValueError(
            "Supply at most 100 explicitly selected structural observables"
        )
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) - {"id", "kind", "atoms", "cutoff_angstrom"}
            or not {"id", "kind", "atoms"} <= set(row)
            or not isinstance(row["id"], str)
            or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", row["id"])
            or row["id"] in seen
            or row["kind"] not in {"distance", "angle", "contact"}
        ):
            raise ValueError(
                "Observables need distinct IDs, distance/angle/contact kinds and exact atoms"
            )
        seen.add(row["id"])
        atoms = row["atoms"]
        if not isinstance(atoms, list) or len(atoms) != (
            3 if row["kind"] == "angle" else 2
        ):
            raise ValueError(
                "Distance/contact needs two atoms; angle needs three ordered atoms"
            )
        for atom in atoms:
            if (
                not isinstance(atom, dict)
                or set(atom)
                != {"chain_id", "residue_number", "residue_name", "atom_name"}
                or type(atom["residue_number"]) is not int
                or not all(
                    isinstance(atom[k], str) and atom[k].strip()
                    for k in ("chain_id", "residue_name", "atom_name")
                )
            ):
                raise ValueError(
                    "Each atom needs exact chain, residue number/name and atom name"
                )
        if len({tuple(sorted(a.items())) for a in atoms}) != len(atoms):
            raise ValueError("An observable must select distinct atoms")
        cutoff = row.get("cutoff_angstrom")
        if row["kind"] == "contact":
            if (
                type(cutoff) not in (int, float)
                or not math.isfinite(cutoff)
                or not 0 < cutoff <= 30
            ):
                raise ValueError("Contacts require a finite cutoff_angstrom in (0,30]")
        elif cutoff is not None:
            raise ValueError("cutoff_angstrom applies only to contacts")
    return rows


def resolve_atoms(topology, identities):
    atoms = list(topology.atoms)
    result = []
    for identity in identities:
        matches = [a.index for a in atoms if atom_identity(a) == identity]
        if len(matches) != 1:
            raise ValueError(
                f"Exact atom selection has {len(matches)} matches: {identity}"
            )
        result.append(matches[0])
    return result


def molecular_observables(traj, rows, indices, periodic):
    import mdtraj as md
    import numpy as np

    columns = []
    for row, atoms in zip(rows, indices, strict=True):
        if row["kind"] == "angle":
            value = np.degrees(
                md.compute_angles(traj, [atoms], periodic=periodic)[:, 0]
            )
        else:
            value = md.compute_distances(traj, [atoms], periodic=periodic)[:, 0] * 10
            if row["kind"] == "contact":
                value = (value <= row["cutoff_angstrom"]).astype(float)
        columns.append(value)
    return columns
