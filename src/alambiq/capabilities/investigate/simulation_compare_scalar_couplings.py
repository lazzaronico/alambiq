"""Compute three-bond scalar couplings from replica trajectories and compare them with measurements.

Returns, for each measured coupling, the calculated mean, its error, its residual against
the measurement and the error reported for the measurement. It also returns the
root-mean-square and mean absolute residuals.

Each measured coupling names its two coupled atoms and the Karplus coefficients that
convert a dihedral angle into a coupling, J(theta) = A cos^2(theta) + B cos(theta) + C in
hertz. For every frame of every replica:

- the dihedral of the unique three-bond path between the two atoms is computed, with the
  minimum-image convention when the trajectory has a periodic cell;
- the dihedral is converted into a coupling.

A replica consists of 1 to 10 trajectory segments that follow one another and are joined
end to end. Every frame has equal weight within a replica, and replicas have equal weight
in the mean. The standard error is the sample standard deviation of the replica means
divided by the square root of the number of replicas.

When sampling checks are set, the split R-hat and effective sample size of each coupling
are also judged. The atoms HB2 and HB3 of beta-methylene groups can be assigned by
geometry (the IUPAC convention) instead of by the labels of the topology, which differ
between force fields.

Limits.

- A residual combines three contributions: the force field and its sampling, the Karplus
  relation, and the measurement.
- The Karplus coefficients are empirical and depend on the coupling and the residues. The
  standard error takes them as given, so their errors come in addition.
- The measured error is the error that the source reports, with the meaning that its
  authors give it. The ratio of the residual to the measured error is a scale for the
  residual.
- The standard error assumes independent replicas. With 2 replicas it rests on 1 degree of
  freedom. Replicas that agree can all remain in the same part of the conformational
  space.
- The couplings are averaged over the ensemble. Averaging the angles first gives different
  values.
- A difference of more than 5 K between the measurement temperature and the mean sampled
  temperature produces a note, and the comparison proceeds.

The capability refuses:

- a pair of atoms that has no unique three-bond path;
- with `iupac-beta-methylene`, a trajectory in which the assignment of HB2 and HB3
  changes between frames;
- more than 30 replicas, or more than 10 segments in a replica.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from alambiq.capabilities.investigate.simulation_compare_scalar_couplings_report import (
    render,
)
from alambiq.capabilities.investigate.simulation_inspect_trajectory import (
    AtomSelector,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

TEMPERATURE_TOLERANCE_KELVIN = 5.0


@dataclass(frozen=True, slots=True)
class Karplus:
    """The coefficients of J(theta) = A cos^2(theta) + B cos(theta) + C.

    a_hz, b_hz, c_hz:
        A, B and C, in hertz.
    """

    a_hz: float
    b_hz: float
    c_hz: float


@dataclass(frozen=True, slots=True)
class CouplingObservation:
    """One measured coupling.

    id:
        Identifier of the coupling, unique in the reference.
    atoms:
        The two coupled atoms; exactly one three-bond path must join them.
    karplus:
        The coefficients for this coupling.
    measured_hz:
        The measured coupling, in hertz.
    reported_error_hz:
        The error reported by the source for the measurement, in hertz.
    error_description:
        The definition of that error as stated by the source.
    """

    id: str
    atoms: list[AtomSelector]
    karplus: Karplus
    measured_hz: float
    reported_error_hz: float
    error_description: str


@dataclass(frozen=True, slots=True)
class CouplingReference:
    """The measurements and their source.

    title, citation, license:
        The description of the measurements, their publication and the license under
        which they are used. All are required.
    temperature_kelvin:
        The temperature of the measurement, in kelvin.
    conditions:
        The solution conditions, in words (pH, salt, solvent).
    observations:
        One to 5,000 measured couplings.
    """

    title: str
    citation: str
    license: str
    temperature_kelvin: float
    conditions: str
    observations: list[CouplingObservation]


@dataclass(frozen=True, slots=True)
class Segment:
    """One stretch of the trajectory of a replica.

    trajectory, state_data:
        The DCD and state-data CSV written by `simulation.run_molecular_dynamics` for
        the stretch.
    """

    trajectory: str
    state_data: str


@dataclass(frozen=True, slots=True)
class CouplingReplica:
    """One independent replica.

    id:
        Identifier of the replica, unique in the request.
    segments:
        1 to 10 stretches that follow one another, joined end to end.
    """

    id: str
    segments: list[Segment]


@dataclass(frozen=True, slots=True)
class SamplingChecks:
    """Thresholds on the sampling diagnostics of each coupling.

    split_rhat_max, combined_ess_min:
        A coupling passes when its split R-hat is at most the first value and its
        effective sample size, summed over replicas, is at least the second value.
    required_jointly_passing:
        The number of couplings that must pass for the sampling as a whole to pass.
    """

    split_rhat_max: float
    combined_ess_min: float
    required_jointly_passing: int


@dataclass(frozen=True, slots=True)
class CouplingRequest:
    """The topology, the replicas and the measurements to compare with.

    topology:
        The topology PDB of the dynamics.
    reference:
        The measured couplings.
    replicas:
        2 to 30 independent replicas with equal frame counts and sampling interval.
    atom_naming:
        `topology` takes the atom names as given in the topology. `iupac-beta-methylene`
        assigns the requested HB2 and HB3 hydrogens by geometry and checks that the
        assignment holds in every frame.
    block_counts:
        The numbers of equal blocks into which each replica is divided for the block
        errors: 1 to 6 values from 2 to 100; each must divide the frame count after
        trimming into blocks of at least 2 frames.
    checks:
        Thresholds on the sampling diagnostics (split R-hat and effective sample
        size), applied to each coupling.
    """

    topology: str
    reference: CouplingReference
    replicas: list[CouplingReplica]
    atom_naming: Literal["topology", "iupac-beta-methylene"] = "topology"
    block_counts: list[int] = field(default_factory=lambda: [5, 10, 20])
    checks: SamplingChecks | None = None


@dataclass(frozen=True, slots=True)
class CouplingRow:
    """One coupling: measured value, calculated value and sampling quality."""

    id: str
    measured_hz: float
    reported_error_hz: float
    calculated_hz: float
    replica_standard_error_hz: float
    residual_hz: float
    """Calculated minus measured."""
    residual_over_reported_error: float | None
    split_rhat: float | None
    effective_samples: float | None
    diagnostic: str
    sampling: str | None
    """`pass`, `fail` or `unassessed` against the checks; None without checks."""


@dataclass(frozen=True, slots=True)
class CouplingResult:
    """The calculated value and residual of each coupling, the summary residuals and the sampling diagnostics."""

    couplings: list[CouplingRow]
    rmse_hz: float
    mae_hz: float
    replicas: list[str]
    frames_per_replica: int
    trimmed_frames: int
    sampling_interval_ps: float
    mean_sampled_temperature_kelvin: float
    measured_temperature_kelvin: float
    sampling_status: str | None
    """`pass`, `fail` or `unassessed` for all couplings together, against the checks; None without checks."""
    atom_naming: str
    swapped_beta_hydrogens: int
    """The number of residues whose HB2 and HB3 are exchanged by the IUPAC assignment."""
    arrays: str
    """An NPZ: per replica, a frames-by-couplings array in hertz, with the coupling ids."""
    details: str
    """A JSON file with the atom paths, assignments, block errors and cumulative means."""
    mdtraj_version: str


@capability(
    "simulation.compare_scalar_couplings",
    version="1.0",
    family="investigate",
    summary="Compute scalar couplings from replica trajectories and compare with measurements",
    render=render,
)
def compare_scalar_couplings(
    request: CouplingRequest, ctx: StepContext
) -> CouplingResult:
    """Compute each coupling per frame, pool the replicas and compare with the measurements."""
    import re

    try:
        import mdtraj as md
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ValueError(
            "Scalar-coupling analysis needs MDTraj; install alambiq[trajectory]"
        ) from exc
    from alambiq.capabilities.investigate.science.couplings import (
        MAX_VALUES,
        check_reference,
        read_segment,
        resolved_coupling_paths,
    )
    from alambiq.capabilities.investigate.science.replicas import GRID, MIN_FRAMES
    from alambiq.capabilities.investigate.science.sampling import (
        check_sampling,
        sampling_statistics,
    )

    reference = asdict(request.reference)
    check_reference(reference)
    observations = reference["observations"]
    if not 2 <= len(request.replicas) <= 30:
        raise ValueError(
            f"replicas holds {len(request.replicas)} entries; 2–30 replicas are required"
        )
    ids = [r.id for r in request.replicas]
    if len(set(ids)) != len(ids) or any(
        not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", i) for i in ids
    ):
        raise ValueError(
            f"Replica ids {ids} are invalid. Replica ids must be unique and use lowercase "
            "letters, digits, `_` or `-`"
        )
    for r in request.replicas:
        if not 1 <= len(r.segments) <= 10:
            raise ValueError(
                f"Replica {r.id} has {len(r.segments)} segments; one to ten segments "
                "are required"
            )
    if not 1 <= len(request.block_counts) <= 6 or any(
        not 2 <= b <= 100 for b in request.block_counts
    ):
        raise ValueError(
            f"block_counts is {list(request.block_counts)}; 1–6 block counts between 2 "
            "and 100 are required"
        )
    checks = request.checks
    if checks is not None:
        if checks.split_rhat_max < 1 or checks.combined_ess_min <= 0:
            raise ValueError(
                f"checks.split_rhat_max is {checks.split_rhat_max:g} and "
                f"checks.combined_ess_min is {checks.combined_ess_min:g}; use a split "
                "R-hat limit of at least 1 and a positive effective size"
            )
        if not 1 <= checks.required_jointly_passing <= len(observations):
            raise ValueError(
                f"checks.required_jointly_passing is {checks.required_jointly_passing}; "
                f"it must be between 1 and the number of couplings ({len(observations)})"
            )

    topology_frame = md.load_pdb(
        str(ctx.resolve(request.topology)), standard_names=False
    )
    paths, assignments = resolved_coupling_paths(
        topology_frame, observations, request.atom_naming
    )
    per_replica, interval, temperatures, recorded = [], None, [], 0
    for replica in request.replicas:
        pieces = []
        for segment in replica.segments:
            values, series, periodic = read_segment(
                topology_frame,
                paths,
                assignments,
                ctx.resolve(segment.trajectory),
                ctx.read_text(segment.state_data),
                observations,
            )
            recorded += values.size
            if recorded > MAX_VALUES:
                raise ValueError(
                    f"At most {MAX_VALUES} coupling values can be read in one comparison"
                )
            times = np.array([r["time_ps"] for r in series])
            if len(times) > 1:
                step = float(np.median(np.diff(times)))
                if interval is None:
                    interval = step
                elif abs(step - interval) > 1e-6 * max(1.0, interval):
                    raise ValueError(
                        f"Replica {replica.id} is sampled every {step:g} ps against "
                        f"{interval:g} ps in earlier segments; all segments must be sampled "
                        "at the same interval"
                    )
            temperatures.extend(r["temperature_kelvin"] for r in series)
            pieces.append(values)
        per_replica.append(np.concatenate(pieces))
    lengths = {len(v) for v in per_replica}
    if len(lengths) != 1:
        raise ValueError(
            f"Replicas have {sorted(lengths)} frames; replicas must have equal numbers "
            "of frames"
        )
    total = lengths.pop()
    usable = total // GRID * GRID
    if usable < MIN_FRAMES:
        raise ValueError(
            f"Replicas have {total} frames ({usable} after trimming to a multiple of "
            f"{GRID}); each replica needs at least {MIN_FRAMES} frames"
        )
    for i in range(len(per_replica)):
        for j in range(i):
            if np.array_equal(per_replica[i], per_replica[j]):
                raise ValueError(
                    f"Replicas {ids[j]} and {ids[i]} hold identical samples"
                )
    bad = [b for b in request.block_counts if usable % b or usable // b < 2]
    if bad:
        raise ValueError(
            f"Block counts {bad} do not divide the {usable} frames per replica into "
            "blocks of at least two frames"
        )
    stack = np.stack([v[:usable] for v in per_replica])
    statistics = sampling_statistics(stack, request.block_counts)
    measured = np.array([o["measured_hz"] for o in observations])
    mean = np.asarray(statistics["mean"])
    residual = mean - measured
    names = [o["id"] for o in observations]
    judged = (
        check_sampling(
            statistics,
            names,
            split_rhat_max=checks.split_rhat_max,
            combined_ess_min=checks.combined_ess_min,
            required=checks.required_jointly_passing,
        )
        if checks
        else None
    )
    sampled = float(np.mean(temperatures))
    if abs(sampled - reference["temperature_kelvin"]) > TEMPERATURE_TOLERANCE_KELVIN:
        ctx.note(
            f"The sampled temperature averaged {sampled:.1f} K, but the couplings were "
            f"measured at {reference['temperature_kelvin']:g} K; the comparison mixes "
            "temperatures."
        )
    if usable != total:
        ctx.note(
            f"The last {total - usable} frames of each replica were left out so the frame "
            "count is a multiple of 20."
        )
    swapped = sum(a["swapped"] for a in assignments)
    arrays_path = ctx.workdir / "couplings.npz"
    np.savez_compressed(
        arrays_path,
        observation_ids=np.asarray(names),
        **{f"replica_{i}": per_replica[k] for k, i in enumerate(ids)},
    )
    rows = []
    for i, observation in enumerate(observations):
        error = observation["reported_error_hz"]
        rows.append(
            CouplingRow(
                id=observation["id"],
                measured_hz=observation["measured_hz"],
                reported_error_hz=error,
                calculated_hz=float(mean[i]),
                replica_standard_error_hz=statistics["replica_mean_standard_error"][i],
                residual_hz=float(residual[i]),
                residual_over_reported_error=float(residual[i] / error)
                if error > 0
                else None,
                split_rhat=statistics["split_rhat"][i],
                effective_samples=statistics["effective_samples"][i],
                diagnostic=statistics["diagnostics"][i]["status"],
                sampling=judged["quantities"][i]["status"] if judged else None,
            )
        )
    details = {
        "reference": reference,
        "atom_paths": paths,
        "atom_naming": {"policy": request.atom_naming, "assignments": assignments},
        "statistics": statistics,
        "sampling_checks": judged,
        "method": {
            "coupling": "A cos^2(theta) + B cos(theta) + C, with theta the dihedral of the unique three-bond atom path, per frame",
            "averaging": "Equal frame weights within a replica, equal replica weights in the mean",
            "uncertainty": "Standard deviation of the replica means over the square root of their number",
        },
    }
    return CouplingResult(
        couplings=rows,
        rmse_hz=float(np.sqrt(np.mean(residual**2))),
        mae_hz=float(np.mean(np.abs(residual))),
        replicas=ids,
        frames_per_replica=usable,
        trimmed_frames=total - usable,
        sampling_interval_ps=float(interval or 0.0),
        mean_sampled_temperature_kelvin=sampled,
        measured_temperature_kelvin=reference["temperature_kelvin"],
        sampling_status=judged["status"] if judged else None,
        atom_naming=request.atom_naming,
        swapped_beta_hydrogens=swapped,
        arrays=str(ctx.keep(arrays_path, "application/x-npz").path),
        details=ctx.keep_json(details),
        mdtraj_version=md.__version__,
    )
