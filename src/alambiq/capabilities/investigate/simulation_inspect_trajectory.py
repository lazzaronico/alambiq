"""Fit the frames of a molecular dynamics trajectory to a reference and measure them.

A trajectory from `simulation.run_molecular_dynamics`, with its topology and state data,
is read once from start to end. Every frame is fitted onto a reference by the protein
alpha carbons (the reference is the first saved frame unless a structure is supplied).
The result gives the alpha-carbon RMSD and the radius of gyration of the alpha carbons
of every frame, the per-residue alpha-carbon fluctuation, and the distances, angles and
contacts between atoms that the request selects by exact atom identity (chain, residue
number, residue name and atom name). Optionally it gives the per-residue contact
fraction with a ligand and a check that the protein does not come closer to its own
periodic image than `protein_image_check.cutoff_nm`. The full per-frame values are
written to an array file. A details file holds 2 to 100 frames (`max_preview_frames`)
and 100 to 5,000 points of each series (`max_plot_frames`) for plotting.

The result describes the behavior of this trajectory, frame by frame, by the selected
measures, with the state-data averages (temperature, potential energy, volume).

Limits. A mean or a fluctuation from one trajectory describes the time sampled; the
variation between independent trajectories is estimated with
`simulation.compare_replica_structures`. An RMSD that rises and remains high shows that
the structure moved away from the reference. Fluctuations are taken about the mean of
this trajectory after fitting, so they reflect the time sampled. The fit and the RMSD
use alpha carbons, so they register backbone motion; a motion of side chains or ligands
that leaves the backbone unchanged is followed with the distances, angles and contacts
that the request defines. The reference is the first saved frame unless `reference_structure`
supplies one, for example the minimized starting structure. Distances, angles and
contacts use the stored coordinates unless `periodic_observables` (minimum-image
distances) is set. A molecule that crosses the box edge in the stored coordinates is
broken across the edge unless `imaging: whole-molecules` is set. The periodic image
check covers only the `protein_image_check.frames` evenly spaced saved frames. A contact
fraction counts saved frames, unweighted by time. The measures are structural
descriptors of the trajectory. A request with more than 100,000 frames or more than
250,000 atoms is refused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from alambiq.capabilities.investigate.simulation_inspect_trajectory_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

PLOT_POINTS = 200


@dataclass(frozen=True, slots=True)
class AtomSelector:
    """One atom, selected by exact match.

    chain_id, residue_number, residue_name, atom_name:
        As named in the topology file; the atom must match exactly once.
    """

    chain_id: str
    residue_number: int
    residue_name: str
    atom_name: str


@dataclass(frozen=True, slots=True)
class Observable:
    """A distance, angle or contact between selected atoms, measured in every frame.

    id:
        The lowercase name of the column (letters, digits, `_` or `-`).
    kind:
        `distance` (two atoms, in Å), `angle` (three atoms in order, the middle one
        the vertex, in degrees) or `contact` (two atoms, 1 when within the cutoff
        and 0 otherwise).
    atoms:
        Two atoms, or three for an angle.
    cutoff_angstrom:
        The contact cutoff, in angstrom; required for `contact` and refused otherwise.
    """

    id: str
    kind: Literal["distance", "angle", "contact"]
    atoms: list[AtomSelector]
    cutoff_angstrom: float | None = None


@dataclass(frozen=True, slots=True)
class LigandContacts:
    """Per-residue contact with one ligand residue.

    ligand_residue_index:
        The zero-based index of the ligand residue in the topology; it must be
        neither protein nor water and contain carbon.
    cutoff_angstrom:
        A residue is in contact in a frame when its nearest heavy atom is within
        this distance of a ligand heavy atom.
    periodic:
        Whether to measure with the minimum-image convention. The value is required
        because it changes the result for a ligand near the box edge.
    """

    ligand_residue_index: int
    cutoff_angstrom: float
    periodic: bool


@dataclass(frozen=True, slots=True)
class ImageCheck:
    """A check of the distance between the protein and its own periodic image.

    cutoff_nm:
        The distance below which the protein is judged to touch its image, in
        nanometers.
    frames:
        The number of evenly spaced saved frames to check (at most all of them).
    """

    cutoff_nm: float
    frames: int


@dataclass(frozen=True, slots=True)
class InspectRequest:
    """The trajectory, its topology and state data, and the measures.

    topology:
        The topology PDB of the dynamics.
    trajectory:
        The DCD trajectory of the simulation.
    state_data:
        The CSV of state data of the same simulation. It has either one row per frame
        or one additional row at the start of production.
    reference_structure:
        A PDB to fit onto instead of the first saved frame; its alpha carbons must
        match those of the topology by exact chain, number, residue name and atom name.
    observables:
        Up to 100 distances, angles or contacts to measure in every frame.
    ligand_contacts:
        Per-residue contact with a ligand residue.
    protein_image_check:
        Optional. The distance check between the protein and its own periodic image.
    imaging:
        `as-stored` uses the coordinates as written, so a molecule that crossed the box
        edge remains broken across the edge; `whole-molecules` first makes every molecule
        whole across the box edge and requires a periodic cell.
    periodic_observables:
        If true, the observables are measured with the minimum-image convention.
    display_atom_indices:
        Zero-based topology indices of the atoms stored for the coordinate preview, 1
        to 5,000 indices. The default is every non-water heavy atom; a solute with more
        than 5,000 of them requires an explicit list.
    max_preview_frames, max_plot_frames, chunk_frames:
        The number of frames stored as coordinates (2 to 100), the number of series
        points stored (100 to 5,000) and the number of frames held in memory at once
        (1 to 64).
    """

    topology: str
    trajectory: str
    state_data: str
    reference_structure: str | None = None
    observables: list[Observable] = field(default_factory=list)
    ligand_contacts: LigandContacts | None = None
    protein_image_check: ImageCheck | None = None
    imaging: Literal["as-stored", "whole-molecules"] = "as-stored"
    periodic_observables: bool = False
    display_atom_indices: list[int] | None = None
    max_preview_frames: int = 20
    max_plot_frames: int = 2000
    chunk_frames: int = 16


@dataclass(frozen=True, slots=True)
class ObservableSummary:
    """One measured column over all frames."""

    id: str
    unit: str
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    first: float
    last: float


@dataclass(frozen=True, slots=True)
class ContactResidue:
    """The contact of one protein residue with the ligand."""

    residue_index: int
    chain_id: str
    residue_number: int
    residue_name: str
    contact_fraction: float
    minimum_distance_angstrom: float


@dataclass(frozen=True, slots=True)
class ImageResult:
    """Whether the protein came near its own image in the frames checked."""

    cutoff_nm: float
    checked_frames: int
    total_frames: int
    within_cutoff_frames: int
    minimum_distance_nm: float
    passed: bool
    all_frames_checked: bool


@dataclass(frozen=True, slots=True)
class InspectResult:
    """The measures, the state averages and the location of the full arrays."""

    frames: int
    atoms: int
    fit_atoms: int
    first_time_ps: float
    last_time_ps: float
    reference: str
    imaging: str
    observables: list[ObservableSummary]
    """The built-in alpha-carbon RMSD and radius of gyration, then the observables of the request."""
    rmsf_mean_angstrom: float
    rmsf_max_angstrom: float
    rmsf_max_residue: str
    ligand_contacts: list[ContactResidue] | None
    """Residues in contact in at least one frame, in order of decreasing contact fraction (at most 30)."""
    image_check: ImageResult | None
    mean_temperature_kelvin: float
    mean_potential_kj_mol: float
    mean_volume_nm3: float | None
    plot_time_ps: list[float]
    plot_rmsd_angstrom: list[float]
    plot_radius_gyration_angstrom: list[float]
    arrays: str
    """An NPZ file: `values` (frames x columns), `observable_ids`, `units`, `steps`, `times_ps`."""
    details: str
    """A JSON file with the coordinate preview, series, fluctuations and method."""
    mdtraj_version: str


@capability(
    "simulation.inspect_trajectory",
    version="1.0",
    family="investigate",
    summary="Fit a trajectory and measure RMSD, fluctuations, distances and contacts",
    render=render,
)
def inspect(request: InspectRequest, ctx: StepContext) -> InspectResult:
    """Stream the trajectory once and summarize it by the chosen measures."""
    try:
        import mdtraj
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ValueError(
            "Trajectory inspection needs MDTraj; install alambiq[trajectory]"
        ) from exc
    from alambiq.capabilities.investigate.science.trajectory import (
        check_inspection,
        inspect_trajectory,
        parse_state_data,
    )

    observables = [asdict(o) for o in request.observables]
    for row in observables:
        if row["cutoff_angstrom"] is None:
            del row["cutoff_angstrom"]
    contacts = asdict(request.ligand_contacts) if request.ligand_contacts else None
    image = asdict(request.protein_image_check) if request.protein_image_check else None
    params = {
        "max_preview_frames": request.max_preview_frames,
        "max_plot_frames": request.max_plot_frames,
        "chunk_frames": request.chunk_frames,
        "imaging": request.imaging,
        "periodic_observables": request.periodic_observables,
        "observables": observables,
        "display_atom_indices": request.display_atom_indices,
        "ligand_contacts": contacts,
        "protein_image_check": image,
    }
    check_inspection(
        max_preview_frames=request.max_preview_frames,
        max_plot_frames=request.max_plot_frames,
        chunk_frames=request.chunk_frames,
        imaging=request.imaging,
        observables=observables,
        display_atom_indices=request.display_atom_indices,
        ligand_contacts=contacts,
        protein_image_check=image,
    )
    rows = parse_state_data(ctx.read_text(request.state_data))
    arrays_path = ctx.workdir / "observables.npz"
    data = inspect_trajectory(
        ctx.resolve(request.topology),
        ctx.resolve(request.trajectory),
        rows,
        params,
        reference=ctx.resolve(request.reference_structure)
        if request.reference_structure
        else None,
        array_output=arrays_path,
    )
    if request.imaging == "as-stored" and data["state"]["mean_volume_nm3"] is not None:
        ctx.note(
            "Coordinates are used as stored in a periodic system; a molecule that crosses "
            "the box edge produces a jump in RMSD."
        )

    stats = data["observables"]
    summaries = [
        ObservableSummary(
            id=name,
            unit=unit,
            mean=stats["mean"][i],
            standard_deviation=stats["standard_deviation"][i],
            minimum=stats["minimum"][i],
            maximum=stats["maximum"][i],
            first=stats["first"][i],
            last=stats["last"][i],
        )
        for i, (name, unit) in enumerate(zip(stats["ids"], stats["units"], strict=True))
    ]
    rmsf = data["ca_fluctuations"]["rmsf_angstrom"]
    worst = max(range(len(rmsf)), key=rmsf.__getitem__)
    atom = data["ca_fluctuations"]["atoms"][worst]
    contact_rows = None
    if "ligand_contacts" in data:
        contact_rows = sorted(
            (
                ContactResidue(
                    residue_index=r["residue_index"],
                    chain_id=r["chain_id"],
                    residue_number=r["residue_number"],
                    residue_name=r["residue_name"],
                    contact_fraction=r["contact_fraction"],
                    minimum_distance_angstrom=r["minimum_distance_angstrom"],
                )
                for r in data["ligand_contacts"]["residues"]
                if r["contact_frame_count"]
            ),
            key=lambda r: (-r.contact_fraction, r.residue_index),
        )[:30]
    image_result = None
    if "protein_image_check" in data:
        c = data["protein_image_check"]
        image_result = ImageResult(
            cutoff_nm=c["cutoff_nm"],
            checked_frames=c["checked_frame_count"],
            total_frames=c["total_frame_count"],
            within_cutoff_frames=c["within_cutoff_frame_count"],
            minimum_distance_nm=c["minimum_distance_nm"],
            passed=c["passed"],
            all_frames_checked=c["all_saved_frames_checked"],
        )
        if not image_result.passed:
            ctx.note(
                f"The protein came within {image_result.cutoff_nm} nm of its own image in "
                f"{image_result.within_cutoff_frames} of {image_result.checked_frames} "
                "checked frames; the box is too small for this motion."
            )
    series = data["series"]
    stride = max(1, -(-len(series) // PLOT_POINTS))
    picked = series[::stride]
    state = data["state"]
    return InspectResult(
        frames=data["frame_count"],
        atoms=data["atom_count"],
        fit_atoms=len(data["fit_atom_indices"]),
        first_time_ps=state["first_time_ps"],
        last_time_ps=state["last_time_ps"],
        reference="supplied structure"
        if request.reference_structure
        else "first saved frame",
        imaging=request.imaging,
        observables=summaries,
        rmsf_mean_angstrom=sum(rmsf) / len(rmsf),
        rmsf_max_angstrom=rmsf[worst],
        rmsf_max_residue=f"{atom['residue_name']}{atom['residue_number']} chain {atom['chain_id']}",
        ligand_contacts=contact_rows,
        image_check=image_result,
        mean_temperature_kelvin=state["mean_temperature_kelvin"],
        mean_potential_kj_mol=state["mean_potential_kj_mol"],
        mean_volume_nm3=state["mean_volume_nm3"],
        plot_time_ps=[r["time_ps"] for r in picked],
        plot_rmsd_angstrom=[r["ca_rmsd_angstrom"] for r in picked],
        plot_radius_gyration_angstrom=[
            r["ca_radius_gyration_angstrom"] for r in picked
        ],
        arrays=str(ctx.keep(arrays_path, "application/x-npz").path),
        details=ctx.keep_json(data),
        mdtraj_version=mdtraj.__version__,
    )
