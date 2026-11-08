"""Measure the pose of a ligand along a trajectory against a reference pose.

Returns, for every frame, the heavy-atom RMSD of the ligand to its pose in a reference
structure after the protein is fitted, the distance between the centroid of the ligand
and that of the reference ligand, and the RMSD of the fit atoms. The summary states how
many frames lie within a threshold of the reference pose and the first frame beyond it.

A trajectory from `simulation.run_molecular_dynamics`, with its topology, is read once from
start to end. The reference is a PDB with the ligand in the pose of interest, for example
the deposited complex, or the structure a prediction gave.

The method:

- The pocket is the alpha carbons of the reference residues that have a heavy atom within
  `fit_cutoff_angstrom` of a heavy atom of the reference ligand. Every frame is superposed
  onto the reference by those alpha carbons, which must match the topology by exact chain,
  residue number, residue name and atom name.
- The ligand is the same molecule in both files. Its heavy atoms are read from the
  residue selected in each file. Bonds are perceived from the coordinates of the reference
  and of the first frame, and bond orders are assigned from `smiles`.
- Atoms of the two files are paired by the molecular graph. The RMSD of a frame is the
  smallest over all pairings that the symmetry of the graph allows, so a benzene turned by
  60 degrees about its ring normal has an RMSD of 0.
- The RMSD is in Å, over heavy atoms, and the ligand is not fitted again after the protein
  fit.

Limits.

- The RMSD states how far the atoms lie from the reference pose. It does not state whether
  the reference pose is correct.
- Equivalent atoms are those with the same connectivity. The handedness of a stereocenter
  is read from the coordinates only through the RMSD itself.
- Bonds are perceived from coordinates, so the ligand of the reference and the ligand of
  the first frame need bond lengths of a normal structure, and the SMILES must match their
  heavy atoms and bonds.
- A ligand with more than 5,000 symmetry-equivalent pairings, or more than 200 heavy atoms,
  is refused.
- With `imaging: whole-molecules`, molecules are made whole across the box edge from the
  bonds of the topology, and the ligand is placed at the periodic image nearest to the
  fit atoms. A ligand without bonds in the topology that crosses the box edge is not made
  whole.
- A fraction of frames counts saved frames, unweighted by time.

The capability refuses a trajectory with more than 100,000 frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.investigate.simulation_measure_ligand_pose_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

PLOT_POINTS = 2000


@dataclass(frozen=True, slots=True)
class ResidueSelector:
    """One residue, selected by exact match.

    chain_id, residue_number, residue_name:
        As named in the structure file; the residue must match exactly once. A chain
        without a letter is named by its zero-based index.
    """

    chain_id: str
    residue_number: int
    residue_name: str


@dataclass(frozen=True, slots=True)
class PoseRequest:
    """The trajectory, the reference pose and the ligand.

    topology:
        The topology PDB of the dynamics.
    trajectory:
        The DCD trajectory of the simulation.
    reference_structure:
        A PDB that holds the ligand in the reference pose and the protein around it. The
        alpha carbons of the pocket must match those of the topology by exact chain, number,
        residue name and atom name.
    ligand:
        The ligand residue in the topology.
    reference_ligand:
        The same ligand residue in the reference structure.
    smiles:
        The ligand as SMILES, with no whitespace. It gives the bond orders, so that equivalent
        atoms are known, and must match the heavy atoms of the ligand in both files.
    state_data:
        Optional. The CSV of state data of the same simulation, which gives the time of each
        frame. It has either one row per frame or one additional row at the start of
        production. Without it the series are indexed by frame.
    fit_cutoff_angstrom:
        Residues of the reference with a heavy atom within this distance of a heavy atom of
        the ligand form the fit; 4 to 1,000 Å. A large value fits the whole protein.
        Default 10.
    threshold_angstrom:
        The RMSD below which the ligand counts as in the reference pose, 0.1 to 20 Å.
        Default 2.
    imaging:
        `as-stored` uses the coordinates as written. `whole-molecules` first makes each
        molecule whole across the box edge and places the ligand at the periodic image
        nearest to the fit atoms; it requires a periodic cell. Default `as-stored`.
    chunk_frames:
        The number of frames held in memory at once, 1 to 64. Default 16.
    """

    topology: str
    trajectory: str
    reference_structure: str
    ligand: ResidueSelector
    reference_ligand: ResidueSelector
    smiles: str
    state_data: str | None = None
    fit_cutoff_angstrom: float = 10.0
    threshold_angstrom: float = 2.0
    imaging: Literal["as-stored", "whole-molecules"] = "as-stored"
    chunk_frames: int = 16


@dataclass(frozen=True, slots=True)
class PoseStatistics:
    """A distance over all frames, in Å."""

    first: float
    last: float
    mean: float
    standard_deviation: float
    median: float
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class PoseResult:
    """The ligand pose in every frame, and its summary."""

    frames: int
    ligand: str
    reference_ligand: str
    heavy_atoms: int
    pairings: int
    """The symmetry-equivalent pairings of the ligand atoms that the RMSD minimizes over."""
    fit_atoms: int
    fit_cutoff_angstrom: float
    imaging: str
    rmsd: PoseStatistics
    """Heavy-atom RMSD to the reference pose."""
    centroid_shift: PoseStatistics
    """Distance between the centroid of the ligand and that of the reference ligand."""
    fit_rmsd_mean_angstrom: float
    """Mean over frames of the RMSD of the fit atoms after the fit."""
    threshold_angstrom: float
    frames_within_threshold: int
    fraction_within_threshold: float
    first_frame_beyond_threshold: int | None
    """Zero-based index of the first frame with an RMSD above the threshold; None if there is none."""
    first_time_ps: float | None
    last_time_ps: float | None
    plot_frame: list[int]
    plot_time_ps: list[float] | None
    plot_rmsd_angstrom: list[float]
    plot_centroid_shift_angstrom: list[float]
    arrays: str
    """An NPZ file: `frame`, `time_ps` (if state data was given), `rmsd_angstrom`, `centroid_shift_angstrom`, `fit_rmsd_angstrom`."""
    mdtraj_version: str
    rdkit_version: str


@capability(
    "simulation.measure_ligand_pose",
    version="1.0",
    family="investigate",
    summary="Measure the RMSD of a ligand to a reference pose along a trajectory",
    render=render,
)
def measure_ligand_pose(request: PoseRequest, ctx: StepContext) -> PoseResult:
    """Fit each frame by the pocket and measure the ligand against the reference pose."""
    try:
        import mdtraj
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ValueError(
            "Measuring a ligand pose needs MDTraj; install alambiq[trajectory]"
        ) from exc
    try:
        import rdkit  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ValueError(
            "Measuring a ligand pose needs RDKit; install alambiq[chemistry]"
        ) from exc
    import numpy as np

    from alambiq.capabilities.investigate.science.ligand_pose import (
        check_pose,
        measure_pose,
        statistics,
    )
    from alambiq.capabilities.investigate.science.trajectory import (
        align_state,
        parse_state_data,
    )

    check_pose(
        fit_cutoff_angstrom=request.fit_cutoff_angstrom,
        threshold_angstrom=request.threshold_angstrom,
        imaging=request.imaging,
        chunk_frames=request.chunk_frames,
    )
    if not request.smiles or any(c.isspace() for c in request.smiles):
        raise ValueError("smiles is a ligand SMILES with no whitespace")
    rows = parse_state_data(ctx.read_text(request.state_data)) if request.state_data else None
    found = measure_pose(
        ctx.resolve(request.topology),
        ctx.resolve(request.trajectory),
        ctx.resolve(request.reference_structure),
        ligand=_selector(request.ligand),
        reference_ligand=_selector(request.reference_ligand),
        smiles=request.smiles,
        fit_cutoff_angstrom=request.fit_cutoff_angstrom,
        threshold_angstrom=request.threshold_angstrom,
        imaging=request.imaging,
        chunk_frames=request.chunk_frames,
    )
    count = found["frames"]
    time = (
        np.asarray([r["time_ps"] for r in align_state(rows, count)])
        if rows is not None
        else None
    )
    arrays = ctx.workdir / "ligand_pose.npz"
    columns = {
        "frame": np.arange(count),
        "rmsd_angstrom": found["rmsd"],
        "centroid_shift_angstrom": found["centroid_shift"],
        "fit_rmsd_angstrom": found["fit_rmsd"],
    }
    if time is not None:
        columns["time_ps"] = time
    np.savez(arrays, **columns)
    kept = ctx.keep(str(arrays), "application/x-npz")
    plot = np.unique(np.linspace(0, count - 1, min(PLOT_POINTS, count), dtype=int))
    inside = found["frames_within"]
    return PoseResult(
        frames=count,
        ligand=found["ligand"],
        reference_ligand=found["reference_ligand"],
        heavy_atoms=found["heavy_atoms"],
        pairings=found["pairings"],
        fit_atoms=found["fit_atoms"],
        fit_cutoff_angstrom=request.fit_cutoff_angstrom,
        imaging=request.imaging,
        rmsd=PoseStatistics(**statistics(found["rmsd"])),
        centroid_shift=PoseStatistics(**statistics(found["centroid_shift"])),
        fit_rmsd_mean_angstrom=float(np.mean(found["fit_rmsd"])),
        threshold_angstrom=request.threshold_angstrom,
        frames_within_threshold=inside,
        fraction_within_threshold=inside / count,
        first_frame_beyond_threshold=found["first_beyond"],
        first_time_ps=float(time[0]) if time is not None else None,
        last_time_ps=float(time[-1]) if time is not None else None,
        plot_frame=[int(i) for i in plot],
        plot_time_ps=[float(time[i]) for i in plot] if time is not None else None,
        plot_rmsd_angstrom=[float(found["rmsd"][i]) for i in plot],
        plot_centroid_shift_angstrom=[float(found["centroid_shift"][i]) for i in plot],
        arrays=str(kept.path),
        mdtraj_version=mdtraj.__version__,
        rdkit_version=found["rdkit_version"],
    )


def _selector(selector: ResidueSelector) -> dict:
    return {
        "chain_id": selector.chain_id,
        "residue_number": selector.residue_number,
        "residue_name": selector.residue_name,
    }
