"""Clustering of the frames of a trajectory by pairwise RMSD with the method of Daura et al. (1999)."""

from __future__ import annotations

from typing import Any

SELECTIONS = {
    "alpha-carbons": "protein and name CA",
    "backbone": "protein and backbone",
    "protein-heavy-atoms": "protein and not element H",
}
MAX_FRAMES = 5000
MIN_ATOMS = 3
CHUNK_FRAMES = 64
LISTED_CLUSTERS = 50


def check_request(
    *,
    atoms: str,
    rmsd_cutoff: float,
    first_frame: int,
    stride: int,
    representatives: int,
    imaging: str,
) -> None:
    if atoms not in SELECTIONS:
        raise ValueError(
            f"atoms is {atoms!r}; one of {', '.join(SELECTIONS)} is required"
        )
    if imaging not in ("as-stored", "whole-molecules"):
        raise ValueError(
            f"imaging is {imaging!r}; imaging must be as-stored or whole-molecules"
        )
    for name, value, low, high in (
        ("rmsd_cutoff_angstrom", rmsd_cutoff, 0.05, 20.0),
        ("first_frame", first_frame, 0, 10_000_000),
        ("stride", stride, 1, 1_000_000),
        ("representatives", representatives, 0, 50),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; a value from {low} to {high} is required"
            )


def read_selected_frames(
    topology, trajectory, *, atoms: str, first_frame: int, stride: int, imaging: str
):
    """The coordinates (nm) of the selected atoms of every read frame, in a (frames, atoms, 3) array."""
    import mdtraj as md
    import numpy as np

    top = md.load_pdb(str(topology)).topology
    selected = top.select(SELECTIONS[atoms])
    if len(selected) < MIN_ATOMS:
        raise ValueError(
            f"The selection {atoms!r} matches {len(selected)} atom(s) of the topology; at least "
            f"{MIN_ATOMS} are required"
        )
    with md.formats.DCDTrajectoryFile(str(trajectory)) as handle:
        total = len(handle)
    available = max(0, -(-(total - first_frame) // stride))
    if available < 2:
        raise ValueError(
            f"The trajectory has {total} frames; first_frame {first_frame} and stride {stride} "
            f"leave {available}, and at least 2 are required"
        )
    if available > MAX_FRAMES:
        raise ValueError(
            f"first_frame {first_frame} and stride {stride} leave {available} frames; at most "
            f"{MAX_FRAMES} are clustered, so raise stride to at least {-(-(total - first_frame) // MAX_FRAMES)}"
        )
    blocks = []
    for chunk in md.iterload(
        str(trajectory),
        chunk=CHUNK_FRAMES * stride,
        top=top,
        stride=stride,
        skip=first_frame,
    ):
        if imaging == "whole-molecules":
            boxes = chunk.unitcell_vectors
            if (
                boxes is None
                or not np.isfinite(boxes).all()
                or np.any(np.linalg.det(boxes) <= 0)
            ):
                raise ValueError(
                    "imaging is whole-molecules and the trajectory has no periodic cell; "
                    "use as-stored or a trajectory with a cell"
                )
            chunk.make_molecules_whole(inplace=True)
        blocks.append(np.asarray(chunk.xyz[:, selected, :], dtype=np.float64))
    return top, np.concatenate(blocks), total


def rmsd_matrix(xyz) -> Any:
    """The pairwise RMSD in angstrom after the best rigid fit of each pair (frames x frames)."""
    import mdtraj as md
    import numpy as np

    frames, atoms, _ = xyz.shape
    shell_topology = md.Topology()
    residue = shell_topology.add_residue("X", shell_topology.add_chain())
    for _ in range(atoms):
        shell_topology.add_atom("CA", md.element.carbon, residue)
    shell = md.Trajectory(xyz.astype(np.float32), shell_topology)
    matrix = np.empty((frames, frames))
    for i in range(frames):
        matrix[i] = md.rmsd(shell, shell, i, parallel=True, precentered=False)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    return matrix * 10.0


def daura_clusters(matrix, cutoff: float) -> list[dict[str, Any]]:
    """Clusters ordered by size. The center is the unassigned frame with the most neighbors within
    the cutoff (the lowest frame index among ties); its neighbors form the cluster and leave the pool."""
    import numpy as np

    near = matrix <= cutoff
    counts = near.sum(axis=1).astype(np.int64)
    unassigned = np.ones(len(matrix), dtype=bool)
    clusters = []
    while unassigned.any():
        candidates = np.where(unassigned, counts, -1)
        center = int(np.argmax(candidates))
        members = np.flatnonzero(near[center] & unassigned)
        clusters.append({"center": center, "members": members.tolist()})
        unassigned[members] = False
        counts -= near[:, members].sum(axis=1)
    clusters.sort(key=lambda c: (-len(c["members"]), c["center"]))
    return clusters


def summarize(
    clusters: list[dict[str, Any]], matrix, first_frame: int, stride: int
) -> list[dict[str, Any]]:
    import numpy as np

    total = len(matrix)
    rows = []
    for index, cluster in enumerate(clusters, 1):
        distances = matrix[cluster["center"], cluster["members"]]
        rows.append(
            {
                "index": index,
                "size": len(cluster["members"]),
                "fraction": len(cluster["members"]) / total,
                "center_frame": first_frame + cluster["center"] * stride,
                "mean_rmsd_to_center": float(np.mean(distances)),
                "max_rmsd_to_center": float(np.max(distances)),
            }
        )
    return rows


def assignments(
    clusters: list[dict[str, Any]], frames: int, first_frame: int, stride: int
) -> str:
    label = [0] * frames
    for index, cluster in enumerate(clusters, 1):
        for member in cluster["members"]:
            label[member] = index
    return "frame,cluster\n" + "".join(
        f"{first_frame + i * stride},{c}\n" for i, c in enumerate(label)
    )
