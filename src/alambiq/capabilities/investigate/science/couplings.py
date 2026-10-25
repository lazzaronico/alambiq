"""Three-bond scalar couplings from a trajectory by a Karplus relation.

Each coupling is J(theta) = A cos^2(theta) + B cos(theta) + C, where theta is the
dihedral of the unique three-bond atom path between the two coupled atoms, in every
frame, with minimum-image geometry when the trajectory has a periodic cell.
"""

from __future__ import annotations

import math
from typing import Any

from alambiq.capabilities.investigate.science.coupling_naming import (
    beta_methylene_assignments,
    validate_beta_methylene,
)
from alambiq.capabilities.investigate.science.trajectory import (
    MAX_FRAMES,
    align_state,
    parse_state_data,
)

MAX_VALUES = 50_000_000


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def check_reference(reference: dict[str, Any]) -> None:
    """Refuse a measured-coupling reference that lacks what a comparison needs."""
    if (
        not _finite(reference["temperature_kelvin"])
        or reference["temperature_kelvin"] <= 0
    ):
        raise ValueError("The reference needs a positive temperature_kelvin")
    for key in ("title", "citation", "license", "conditions"):
        if not reference[key].strip():
            raise ValueError(f"The reference needs a non-empty {key}")
    rows = reference["observations"]
    if not 1 <= len(rows) <= 5000:
        raise ValueError("Supply 1–5000 scalar-coupling observations")
    seen: set[str] = set()
    for row in rows:
        if not row["id"].strip() or row["id"] in seen:
            raise ValueError("Observation ids must be unique and non-empty")
        seen.add(row["id"])
        if (
            not _finite(row["measured_hz"])
            or not _finite(row["reported_error_hz"])
            or row["reported_error_hz"] < 0
            or not row["error_description"].strip()
        ):
            raise ValueError(
                f"{row['id']}: couplings need finite Hz values, a non-negative reported "
                "error and a description of what that error means"
            )
        if not all(_finite(v) for v in row["karplus"].values()):
            raise ValueError(f"{row['id']}: the Karplus coefficients must be finite")
        if len(row["atoms"]) != 2:
            raise ValueError(f"{row['id']}: exactly two endpoint atoms are required")
        for atom in row["atoms"]:
            if not all(
                atom[k].strip() for k in ("chain_id", "residue_name", "atom_name")
            ):
                raise ValueError(
                    f"{row['id']}: each atom needs a chain, residue name and atom name"
                )


def _identity(atom) -> dict[str, Any]:
    return {
        "chain_id": atom.residue.chain.chain_id,
        "residue_number": atom.residue.resSeq,
        "residue_name": atom.residue.name,
        "atom_name": atom.name,
    }


def coupling_paths(topology, observations, atom_overrides=None):
    """Resolve endpoints through a unique simple three-bond topology path."""
    atoms = list(topology.atoms)
    adjacency: dict[int, set[int]] = {a.index: set() for a in atoms}
    for left, right in topology.bonds:
        adjacency[left.index].add(right.index)
        adjacency[right.index].add(left.index)
    result = []
    for row in observations:
        endpoints = []
        for identity in row["atoms"]:
            matches = [a.index for a in atoms if _identity(a) == identity]
            if len(matches) != 1:
                raise ValueError(
                    f"{row['id']}: exact endpoint has {len(matches)} topology matches: {identity}"
                )
            endpoints.append((atom_overrides or {}).get(matches[0], matches[0]))
        start, finish = endpoints
        paths = [
            [start, b, c, finish]
            for b in sorted(adjacency[start])
            for c in sorted(adjacency[b] - {start})
            if finish in adjacency[c] and len({start, b, c, finish}) == 4
        ]
        if len(paths) != 1:
            raise ValueError(
                f"{row['id']}: expected one three-bond coupling path; found {len(paths)}"
            )
        record = {
            "id": row["id"],
            "atom_indices": paths[0],
            "path": [_identity(atoms[i]) for i in paths[0]],
        }
        if atom_overrides:
            record["requested_atoms"] = row["atoms"]
        result.append(record)
    return result


def resolved_coupling_paths(topology_frame, observations, atom_naming: str):
    overrides, assignments = {}, []
    if atom_naming == "iupac-beta-methylene":
        overrides, assignments = beta_methylene_assignments(
            topology_frame, observations
        )
    elif atom_naming != "topology":
        raise ValueError("Select topology or iupac-beta-methylene atom naming")
    return coupling_paths(topology_frame.topology, observations, overrides), assignments


def coupling_values(angles_radians, observations):
    """Karplus couplings (Hz) for dihedral angles, one column per observation."""
    import numpy as np

    angles = np.asarray(angles_radians, dtype=float)
    if (
        angles.ndim != 2
        or angles.shape[1] != len(observations)
        or not np.isfinite(angles).all()
    ):
        raise ValueError(
            "Dihedral array requires finite radians, one column per observation"
        )
    coefficients = np.array(
        [[r["karplus"][k] for k in ("a_hz", "b_hz", "c_hz")] for r in observations]
    )
    cosine = np.cos(angles)
    return (
        coefficients[:, 0] * cosine**2
        + coefficients[:, 1] * cosine
        + coefficients[:, 2]
    )


def read_segment(
    topology_frame, paths, assignments, trajectory, state_text, observations
):
    """Stream one trajectory segment; return its couplings, times and state samples."""
    import mdtraj as md
    import numpy as np

    top = topology_frame.topology
    indices = np.asarray([r["atom_indices"] for r in paths])
    with md.formats.DCDTrajectoryFile(str(trajectory)) as handle:
        count = len(handle)
    if not 1 <= count <= MAX_FRAMES:
        raise ValueError(f"A segment holds 1–{MAX_FRAMES} frames")
    series = align_state(parse_state_data(state_text), count)
    predictions = []
    periodic = None
    with md.formats.DCDTrajectoryFile(str(trajectory)) as handle:
        while True:
            try:
                chunk = handle.read_as_traj(top, n_frames=100)
            except ValueError as exc:
                raise ValueError(
                    f"The trajectory and the topology disagree about the atoms: {exc}"
                ) from exc
            if not len(chunk):
                break
            boxes = chunk.unitcell_vectors
            if not np.isfinite(chunk.xyz).all():
                raise ValueError("The trajectory holds a non-finite coordinate")
            here = boxes is not None
            if periodic is not None and here != periodic:
                raise ValueError("Frames of a segment differ in whether they have a periodic cell")
            periodic = here
            if here and (
                not np.isfinite(boxes).all() or np.any(np.linalg.det(boxes) <= 0)
            ):
                raise ValueError("Every periodic cell must have a positive volume")
            validate_beta_methylene(chunk, assignments)
            local = md.compute_angles(
                chunk,
                np.concatenate((indices[:, :3], indices[:, 1:]), axis=0),
                periodic=here,
            )
            if np.any(np.abs(np.sin(local)) < 1e-6):
                raise ValueError(
                    "A coupling dihedral contains collinear or degenerate bonded geometry"
                )
            predictions.append(
                coupling_values(
                    md.compute_dihedrals(chunk, indices, periodic=here), observations
                )
            )
    values = np.concatenate(predictions)
    if len(values) != count:
        raise ValueError("The frames read differ from the frames counted")
    return values, series, bool(periodic)
