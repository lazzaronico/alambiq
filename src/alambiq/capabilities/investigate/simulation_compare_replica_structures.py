"""Judge whether independent replicas determine the mean of named observables to a precision.

Returns, for each named observable, the mean over replicas, its standard error, convergence
diagnostics, and whether the replicas meet the requested precision. With a `tolerance`, it
also returns whether the mean agrees with the value on a reference structure.

Each replica is a trajectory already described by `simulation.inspect_trajectory`. Every
inspection has the same alpha carbons and observable definitions. With a `tolerance`,
every inspection is fitted to the same supplied `reference_structure`. The capability pools
the per-frame values of the observables named in the request. For each observable it
reports:

- the mean over replicas (the mean of the replica means);
- its standard error, computed from the spread of the replica means;
- the split R-hat, which is 1 when the halves agree. Each replica is split into two halves.
  R-hat is the square root of the ratio of the pooled variance estimate to the mean
  within-half variance;
- the effective sample size;
- the drift between the halves of each replica;
- the block-averaged errors.

An observable meets its precision when:

- the standard error of the replica means is at most `precision`;
- the split R-hat is at most `max_split_rhat`;
- the effective sample size is at least `min_effective_samples`;
- the drift between the halves of each replica is at most twice `precision`.

An observable agrees with the reference structure when the 95 % interval of its mean lies
within `tolerance` of the value measured on the reference structure.

The result also reports the first checkpoint, in frames and picoseconds per replica, at and
after which the precision criterion is met up to the end of the trajectory. Checkpoints are
40 frames per replica, the full length, and the multiples of 20 frames nearest below each
tenth of the length (at least 40 frames).

Limits.

- The observables are alpha-carbon RMSD, radius of gyration and the distances, angles and
  contacts defined in the `simulation.inspect_trajectory` request.
- The errors assume that the replicas are independent draws from one ensemble. This
  requires independent starting states and seeds.
- The interval uses a t quantile with n - 1 degrees of freedom for n replicas: 12.71 for
  2 replicas and 4.30 for 3 replicas. The interval is wide for a small number of replicas.
- R-hat and effective sample size that meet their thresholds show that the replicas agree
  with each other over the time sampled. Slower motions need longer trajectories. Replicas
  that all started from one structure can agree while remaining in one conformational
  state.
- The checkpoint values describe these samples.
- Every frame has the same weight, so the statistics apply to unbiased trajectories. A
  biased trajectory, such as one from metadynamics, needs frame weights.

The capability refuses:

- a periodic trajectory inspected with `imaging: as-stored`; use `imaging: whole-molecules`;
- a replica with fewer than 40 frames after the frames beyond the last multiple of 20 are
  excluded;
- fewer than 2 or more than 30 replicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alambiq.capabilities.investigate.simulation_compare_replica_structures_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class ReplicaSource:
    """The inspection of one replica.

    id:
        Identifier of the replica, unique in the request.
    details:
        The `details` file returned by `simulation.inspect_trajectory`.
    arrays:
        The `arrays` file of the same inspection.
    seed:
        The random seed of the replica, used to check that no two replicas share a
        seed. It is supplied for every replica or for none.
    """

    id: str
    details: str
    arrays: str
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class Criterion:
    """What to judge for one observable.

    observable:
        The id of the observable, such as `ca_rmsd_angstrom` or an observable named in
        the inspection.
    precision:
        The largest accepted standard error of the replica mean, in the unit of the
        observable; the halves of a replica must also not drift apart by more than
        twice this value.
    tolerance:
        Optional. The largest accepted distance between the replica mean and the
        value measured on the reference structure; the whole 95 % interval must lie
        within it. Requires every inspection to have been fitted to the same
        `reference_structure`.
    """

    observable: str
    precision: float
    tolerance: float | None = None


@dataclass(frozen=True, slots=True)
class ReplicaRequest:
    """The replicas and the criteria.

    replicas:
        2 to 30 inspections of independent replicas, with equal frame counts and the
        same sampling interval.
    criteria:
        The observables to judge and the precision and tolerance for each.
    max_split_rhat:
        The largest accepted split R-hat, between 1 and 1.2.
    min_effective_samples:
        The smallest accepted effective sample size, summed over replicas.
    block_counts:
        The numbers of equal blocks into which each replica is divided for the block
        errors (each must divide the frame count after trimming).
    """

    replicas: list[ReplicaSource]
    criteria: list[Criterion]
    max_split_rhat: float = 1.1
    min_effective_samples: float = 100.0
    block_counts: list[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass(frozen=True, slots=True)
class ObservableVerdict:
    """One observable across the replicas."""

    observable: str
    unit: str
    mean: float
    replica_standard_error: float
    ci95_low: float
    ci95_high: float
    split_rhat: float | None
    effective_samples: float | None
    diagnostic: str
    """`estimated`, or the reason R-hat and the effective size could not be estimated (a constant series)."""
    max_half_drift: float
    precision_target: float
    precision_passed: bool
    reference_value: float | None
    tolerance: float | None
    mean_offset: float | None
    """The replica mean minus the reference value."""
    agrees: bool | None


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The precision judged on the initial part of every replica."""

    frames_per_replica: int
    simulated_ps_per_replica: float
    precision_passed: bool


@dataclass(frozen=True, slots=True)
class ReplicaResult:
    """The precision and agreement of each observable, the replica diagnostics and the first sustained checkpoint."""

    replicas: list[str]
    frames_per_replica: int
    trimmed_frames: int
    sampling_interval_ps: float
    simulated_ps_per_replica: float
    total_simulated_ps: float
    observables: list[ObservableVerdict]
    precision_passed: bool
    """True if every observable met its precision."""
    agrees_with_reference: bool | None
    """True if every observable with a tolerance agreed; None if none had a tolerance."""
    first_sustained_checkpoint: Checkpoint | None
    prefix_checks: list[Checkpoint]
    rmsf_mean_angstrom: list[float]
    """Per alpha carbon, in the fit order of the inspections: the mean over replicas."""
    rmsf_standard_error_angstrom: list[float]
    details: str
    """A JSON file with the block errors, the cumulative means and the full diagnostics."""


@capability(
    "simulation.compare_replica_structures",
    version="1.0",
    family="investigate",
    summary="Judge the precision of replica means and their agreement with a reference structure",
    render=render,
)
def compare_replica_structures(
    request: ReplicaRequest, ctx: StepContext
) -> ReplicaResult:
    """Pool the observables of the replicas and judge precision and agreement."""
    import numpy as np

    from alambiq.capabilities.investigate.science.replicas import (
        check_criteria,
        compare_replicas,
    )

    if not 2 <= len(request.replicas) <= 30:
        raise ValueError(
            f"replicas holds {len(request.replicas)} entries; 2–30 replicas are required"
        )
    criteria = [
        {
            "observable": c.observable,
            "precision": c.precision,
            "tolerance": c.tolerance,
        }
        for c in request.criteria
    ]
    check_criteria(
        criteria,
        max_split_rhat=request.max_split_rhat,
        min_effective_samples=request.min_effective_samples,
        block_counts=request.block_counts,
    )
    replicas = []
    for source in request.replicas:
        details = ctx.read_json(source.details)
        with np.load(ctx.resolve(source.arrays), allow_pickle=False) as arrays:
            if arrays["observable_ids"].tolist() != details["observables"]["ids"]:
                raise ValueError(
                    f"{source.id}: observable_ids in arrays differ from observables in "
                    "details; supply the details and arrays of the same inspection"
                )
            replicas.append(
                {
                    "id": source.id,
                    "seed": source.seed,
                    "details": details,
                    "values": arrays["values"],
                    "times_ps": arrays["times_ps"],
                }
            )
    found = compare_replicas(
        replicas,
        criteria,
        max_split_rhat=request.max_split_rhat,
        min_effective_samples=request.min_effective_samples,
        block_counts=request.block_counts,
    )
    if found["trimmed_frames"]:
        ctx.note(
            f"The last {found['trimmed_frames']} frames of each replica were left out so "
            "the frame count is a multiple of 20."
        )
    if not found["precision_passed"]:
        ctx.note("At least one observable did not meet its precision.")
    return ReplicaResult(
        replicas=found["replicas"],
        frames_per_replica=found["frames_per_replica"],
        trimmed_frames=found["trimmed_frames"],
        sampling_interval_ps=found["sampling_interval_ps"],
        simulated_ps_per_replica=found["simulated_ps_per_replica"],
        total_simulated_ps=found["total_simulated_ps"],
        observables=[
            ObservableVerdict(
                observable=r["observable"],
                unit=r["unit"],
                mean=r["mean"],
                replica_standard_error=r["replica_standard_error"],
                ci95_low=r["ci95"][0],
                ci95_high=r["ci95"][1],
                split_rhat=r["split_rhat"],
                effective_samples=r["effective_samples"],
                diagnostic=r["diagnostic"],
                max_half_drift=r["max_half_drift"],
                precision_target=r["precision_target"],
                precision_passed=r["precision_passed"],
                reference_value=r["reference_value"],
                tolerance=r["tolerance"],
                mean_offset=r["mean_offset"],
                agrees=r["agrees"],
            )
            for r in found["observables"]
        ],
        precision_passed=found["precision_passed"],
        agrees_with_reference=found["agrees_with_reference"],
        first_sustained_checkpoint=Checkpoint(**found["first_sustained_checkpoint"])
        if found["first_sustained_checkpoint"]
        else None,
        prefix_checks=[Checkpoint(**p) for p in found["prefix_checks"]],
        rmsf_mean_angstrom=found["rmsf_mean_angstrom"],
        rmsf_standard_error_angstrom=found["rmsf_standard_error_angstrom"],
        details=ctx.keep_json(found),
    )
