"""Group the frames of a trajectory by structural similarity and write a representative of each group.

Returns clusters of frames that lie within an RMSD cutoff of a central frame, ordered by size.
Each cluster reports its size, the frame that is its center, and the mean and maximum RMSD of its
members to the center. The center structures of the largest clusters are written as PDB files, and
a table assigns every frame read to a cluster.

The method:

- The frames are read from a DCD trajectory with the topology PDB, from `first_frame` at the
  stated `stride`. The selected atoms are fitted pair by pair: the RMSD between two frames is
  the root-mean-square distance of the selected atoms after the best rigid-body superposition.
- Clustering follows Daura et al. (1999). The frame with the most neighbors within the cutoff
  is the center of the first cluster; its neighbors are its members. The members leave the
  pool, and the frame with the most remaining neighbors in the pool is the next center. A frame
  that has no neighbor within the cutoff forms a cluster of 1. Among equal counts, the frame
  with the lowest index is the center.
- A member lies within the cutoff of the center of its cluster. Two members of one cluster can
  lie farther apart than the cutoff.

Limits.

- The clusters depend on the atom selection and the cutoff. A change of either changes the
  count and the sizes.
- The population of a cluster counts saved frames, unweighted by time, from one trajectory. The
  populations of independent trajectories vary and describe the time sampled.
- The RMSD matrix holds every pair of frames. At most 5,000 frames are clustered; a longer
  trajectory needs a larger stride.
- Fitted atoms register the motion of the selected atoms. A rearrangement of other atoms,
  such as a ligand or a side chain outside the selection, does not separate frames.
- A molecule that crosses the box edge in the stored coordinates is broken across the edge
  unless `imaging: whole-molecules` is set, which requires a periodic cell.
- The written centers are frames as stored (with the imaging of the request), for the whole
  system of the topology.
- The result lists the 50 largest clusters; the table of assignments covers every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.investigate.simulation_cluster_trajectory_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class ClusterRequest:
    """The trajectory, the atoms compared, and the clustering cutoff.

    topology:
        The topology PDB of the dynamics.
    trajectory:
        The DCD trajectory of the simulation.
    atoms:
        The atoms whose coordinates are compared: `alpha-carbons`, `backbone` (N, CA, C, O) or
        `protein-heavy-atoms`; default `alpha-carbons`. Each needs at least 3 atoms in the topology.
    rmsd_cutoff_angstrom:
        Two frames are neighbors when their RMSD is at most this value, 0.05 to 20; default 2.0.
    first_frame:
        The zero-based index of the first frame read; default 0.
    stride:
        Every stride-th frame from `first_frame` is read, 1 or more; default 1. At most 5,000 frames
        are read.
    representatives:
        The number of largest clusters whose center structure is written as a PDB, 0 to 50;
        default 5.
    imaging:
        `as-stored` uses the coordinates as written; `whole-molecules` first makes every molecule
        whole across the box edge and requires a periodic cell; default `as-stored`.
    """

    topology: str
    trajectory: str
    atoms: Literal["alpha-carbons", "backbone", "protein-heavy-atoms"] = "alpha-carbons"
    rmsd_cutoff_angstrom: float = 2.0
    first_frame: int = 0
    stride: int = 1
    representatives: int = 5
    imaging: Literal["as-stored", "whole-molecules"] = "as-stored"


@dataclass(frozen=True, slots=True)
class Cluster:
    """One cluster of frames."""

    index: int
    """Rank by size, 1 for the largest."""
    size: int
    fraction: float
    """The share of the frames read."""
    center_frame: int
    """The zero-based index of the center frame in the trajectory."""
    mean_rmsd_to_center_angstrom: float
    max_rmsd_to_center_angstrom: float
    structure: str | None
    """A PDB file of the center frame; None beyond the first `representatives` clusters."""


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """The clusters of the frames read, and where the assignments are."""

    frames_in_file: int
    frames_read: int
    atoms_compared: int
    rmsd_cutoff_angstrom: float
    cluster_count: int
    singleton_count: int
    """Clusters of 1 frame."""
    clusters: list[Cluster]
    """The 50 largest clusters at most."""
    assignments: str
    """A CSV file with the columns `frame` (the index in the trajectory) and `cluster` (the rank)."""
    mdtraj_version: str


@capability(
    "simulation.cluster_trajectory",
    version="1.0",
    family="investigate",
    summary="Cluster trajectory frames by RMSD and write the center structure of the largest clusters",
    render=render,
)
def cluster_trajectory(request: ClusterRequest, ctx: StepContext) -> ClusterResult:
    """Read the frames, cluster them by pairwise RMSD, and write the centers."""
    try:
        import mdtraj
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ValueError(
            "Trajectory clustering needs MDTraj; install alambiq[trajectory]"
        ) from exc
    from alambiq.capabilities.investigate.science.clustering import (
        LISTED_CLUSTERS,
        assignments,
        check_request,
        daura_clusters,
        read_selected_frames,
        rmsd_matrix,
        summarize,
    )

    check_request(
        atoms=request.atoms,
        rmsd_cutoff=request.rmsd_cutoff_angstrom,
        first_frame=request.first_frame,
        stride=request.stride,
        representatives=request.representatives,
        imaging=request.imaging,
    )
    topology = ctx.resolve(request.topology)
    trajectory = ctx.resolve(request.trajectory)
    top, xyz, total = read_selected_frames(
        topology,
        trajectory,
        atoms=request.atoms,
        first_frame=request.first_frame,
        stride=request.stride,
        imaging=request.imaging,
    )
    matrix = rmsd_matrix(xyz)
    clusters = daura_clusters(matrix, request.rmsd_cutoff_angstrom)
    rows = summarize(clusters, matrix, request.first_frame, request.stride)
    table = ctx.keep_text(
        assignments(clusters, len(xyz), request.first_frame, request.stride), "text/csv"
    )
    listed = []
    for row in rows[:LISTED_CLUSTERS]:
        structure = None
        if row["index"] <= request.representatives:
            frame = mdtraj.load_frame(str(trajectory), row["center_frame"], top=top)
            if request.imaging == "whole-molecules":
                frame.make_molecules_whole(inplace=True)
            path = (
                ctx.workdir
                / f"cluster-{row['index']:03d}-frame-{row['center_frame']}.pdb"
            )
            frame.save_pdb(str(path))
            structure = str(ctx.keep(path, "chemical/x-pdb").path)
        listed.append(
            Cluster(
                index=row["index"],
                size=row["size"],
                fraction=row["fraction"],
                center_frame=row["center_frame"],
                mean_rmsd_to_center_angstrom=row["mean_rmsd_to_center"],
                max_rmsd_to_center_angstrom=row["max_rmsd_to_center"],
                structure=structure,
            )
        )
    singletons = sum(len(c["members"]) == 1 for c in clusters)
    if len(clusters) > LISTED_CLUSTERS:
        ctx.note(
            f"The trajectory forms {len(clusters)} clusters; the result lists the "
            f"{LISTED_CLUSTERS} largest, and the assignments file covers every frame."
        )
    if singletons > len(xyz) / 2:
        ctx.note(
            f"{singletons} of {len(xyz)} frames form clusters of 1; the cutoff of "
            f"{request.rmsd_cutoff_angstrom} Å may be small for this trajectory."
        )
    return ClusterResult(
        frames_in_file=total,
        frames_read=len(xyz),
        atoms_compared=xyz.shape[1],
        rmsd_cutoff_angstrom=request.rmsd_cutoff_angstrom,
        cluster_count=len(clusters),
        singleton_count=singletons,
        clusters=listed,
        assignments=str(table.path),
        mdtraj_version=f"mdtraj {mdtraj.__version__}",
    )
