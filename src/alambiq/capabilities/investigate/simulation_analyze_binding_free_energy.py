"""Combine relative free-energy legs into binding differences, cycle closures and checks.

Each directed edge A to B of a ligand series has two legs, one with the ligand bound to
the protein and one in solvent, each run as independent replicas by
`simulation.run_relative_free_energy`. The relative binding free energy of an edge is
the mean of the bound leg minus the mean of the solvent leg, in kcal/mol, with each
leg's mean taken with equal weight over its replicas. For every reported value the
result gives the sample standard deviation of the replica estimates, the error that MBAR
reports within each replica, the overlap between neighboring lambda windows, the
effective sample size per lambda window, and the difference between the forward and
reverse estimates at half of the data. Closed cycles of edges are summed, because a
thermodynamic cycle sums to zero, and a set of measured binding free energies
(`reference`) can be compared with the calculated differences, edge by edge and cycle by
cycle. The checks in the request convert these diagnostics into one pass, fail or
unassessed verdict per diagnostic and per replica, leg, edge or cycle, and one overall
status.

The result gives the calculated relative binding free energies of these edges under
this method, the reproducibility of the estimates across replicas, and whether the
sampling met the requested checks.

Limits. The spread across replicas measures the reproducibility of the sampling from
independent starts. The error of the force field, the charges and the atom mapping is
additional to the spread, so a reproducible result can differ from the true value. The
MBAR error assumes uncorrelated samples and understates the error when the samples are
correlated; it is reported separately from the replica spread. A cycle that closes shows internal
consistency. Cycle errors assume independent sampling of the edges, which a shared leg
or a shared starting structure violates. Every ligand must have the same SHA-256 digest
in every leg, the protein must have the same digest in every bound leg, the method and
the temperature must be the same throughout, and both legs of an edge must share one
atom mapping: the capability checks these conditions and refuses a mismatch. The result
assumes correct protonation of the binding site, correct ligand charge states and the
absence of slow protein rearrangements. Overlap, effective sample size and the
forward-reverse difference indicate convergence of the lambda path. The capability
detects ligand drift in the bound leg only when `structural_diagnostics` is supplied and
a check on the drift is set. A request with more than 100 edges or more than 100 cycles
is refused, and so is a leg with fewer than 2 or more than 50 replicas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from alambiq.capabilities.investigate.simulation_analyze_binding_free_energy_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class Edge:
    """One directed transformation of the ligand series.

    id:
        Identifier of the edge, unique in the series.
    a, b:
        The two ligands, by the names recorded in the legs; the edge goes from a to b.
    """

    id: str
    a: str
    b: str


@dataclass(frozen=True, slots=True)
class Cycle:
    """A closed cycle of edges, which must sum to zero.

    id:
        Identifier of the cycle.
    coefficients:
        For each edge in the cycle, 1 if the edge is traversed from a to b and -1 if
        from b to a. The signed edges must close over the ligands.
    """

    id: str
    coefficients: dict[str, int]


@dataclass(frozen=True, slots=True)
class ReferenceObservation:
    """A measured binding free energy for one ligand.

    ligand:
        The name of the ligand, as in the series.
    delta_g_kcal_mol:
        The measured binding free energy, in kcal/mol.
    reported_error_kcal_mol:
        The error reported by the source for the measurement, in kcal/mol.
    """

    ligand: str
    delta_g_kcal_mol: float
    reported_error_kcal_mol: float


@dataclass(frozen=True, slots=True)
class BindingReference:
    """Measured binding free energies to compare with the calculation.

    temperature_kelvin:
        The temperature of the measurements, in kelvin; it must equal the temperature
        of the legs.
    citation, license:
        The source of the numbers and the license under which they are used.
    conditions, error_description:
        The measurement conditions and the meaning of the reported errors, in words.
    observations:
        Exactly one per ligand of the series.
    """

    temperature_kelvin: float
    citation: str
    license: str
    conditions: str
    error_description: str
    observations: list[ReferenceObservation]


@dataclass(frozen=True, slots=True)
class BindingChecks:
    """Thresholds that convert diagnostics into a pass or a fail; each threshold is optional.

    maximum_edge_replica_sem_kcal_mol:
        The largest permitted standard error of an edge across replicas.
    maximum_leg_replica_sd_kcal_mol:
        The largest permitted standard deviation of the replica estimates of a leg.
    minimum_adjacent_overlap_each_replica:
        The smallest permitted overlap between neighboring lambda windows in every replica,
        from 0 to 1.
    minimum_effective_samples_each_state:
        The smallest permitted effective sample size of any lambda window in every replica.
    half_fraction_target, half_fraction_maximum_distance,
    maximum_half_forward_reverse_difference_kcal_mol:
        Together: at the stored fraction nearest the target (from 0 to 1, within the
        stated distance of the target) the forward and reverse estimates differ by at
        most this value.
    cycle_absolute_floor_kcal_mol, cycle_replica_sem_multiplier:
        Together: a cycle closes if its closure is within the larger of the floor and
        the multiplier times its replica standard error.
    maximum_ligand_com_drift_angstrom:
        The largest permitted drift of the center of mass of the ligand in a bound
        replica; requires the structural diagnostics.
    """

    maximum_edge_replica_sem_kcal_mol: float | None = None
    maximum_leg_replica_sd_kcal_mol: float | None = None
    minimum_adjacent_overlap_each_replica: float | None = None
    minimum_effective_samples_each_state: float | None = None
    half_fraction_target: float | None = None
    half_fraction_maximum_distance: float | None = None
    maximum_half_forward_reverse_difference_kcal_mol: float | None = None
    cycle_absolute_floor_kcal_mol: float | None = None
    cycle_replica_sem_multiplier: float | None = None
    maximum_ligand_com_drift_angstrom: float | None = None


@dataclass(frozen=True, slots=True)
class BindingRequest:
    """The series, its legs, and what to compare and check.

    title:
        Title of the series.
    edges:
        1 to 100 directed edges.
    legs:
        Paths to the leg records written by `simulation.run_relative_free_energy`:
        exactly one bound leg and one solvent leg for every edge.
    cycles:
        Closed cycles of edges, at most 100.
    reference:
        Measured binding free energies to compare with the calculation.
    checks:
        Thresholds for the diagnostics.
    structural_diagnostics:
        Maps the id of a bound replica to the path of its native structural file (an
        NPZ with `time_ps` and `ligand_COM_drift`), used to check the drift of the
        ligand.
    """

    title: str
    edges: list[Edge]
    legs: list[str]
    cycles: list[Cycle] = field(default_factory=list)
    reference: BindingReference | None = None
    checks: BindingChecks | None = None
    structural_diagnostics: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LegSummary:
    """One leg of one edge, over its replicas."""

    replicas: int
    mean_kcal_mol: float
    replica_sd_kcal_mol: float
    replica_sem_kcal_mol: float
    mbar_error_kcal_mol: float
    """The within-replica MBAR errors, added in quadrature and divided by the replica count."""
    minimum_adjacent_overlap: float
    minimum_effective_samples: float
    largest_half_forward_reverse_difference_kcal_mol: float | None
    """The largest absolute difference over the replicas at half the data; None if not stored."""


@dataclass(frozen=True, slots=True)
class EdgeResult:
    """The relative binding free energy of one edge."""

    id: str
    a: str
    b: str
    delta_delta_g_kcal_mol: float
    """Bound minus solvent."""
    replica_sem_kcal_mol: float
    mbar_error_kcal_mol: float
    bound: LegSummary
    solvent: LegSummary
    reference_delta_delta_g_kcal_mol: float | None
    """From the reference's ligand values; None without a reference."""


@dataclass(frozen=True, slots=True)
class CycleResult:
    """A cycle's closure."""

    id: str
    closure_kcal_mol: float
    replica_sem_kcal_mol: float
    mbar_error_kcal_mol: float
    reference_closure_kcal_mol: float | None
    reference_propagated_error_kcal_mol: float | None


@dataclass(frozen=True, slots=True)
class CheckRow:
    """One diagnostic against its limit."""

    diagnostic: str
    identity: str
    observed: float | None
    limit: float
    comparison: str
    status: str
    """`pass`, `fail`, or `unassessed` when the diagnostic is not available."""
    reason: str | None
    selected_fraction: float | None
    """For the half-data check: the stored fraction of the data that is read."""


@dataclass(frozen=True, slots=True)
class BindingResult:
    """The edges, cycles and checks, and the location of the full record."""

    title: str
    temperature_kelvin: float
    direction: str
    edges: list[EdgeResult]
    cycles: list[CycleResult]
    status: str | None
    """`pass`, `fail` or `unassessed` over all checks; None if none were set."""
    sampling_status: str | None
    structural_review_status: str | None
    checks: list[CheckRow]
    details: str
    """A JSON file with every replica, overlap matrix, forward-reverse series and identity."""


def _leg_summary(leg: dict[str, Any]) -> LegSummary:
    halves = [
        abs(r["half_forward_reverse_difference_kcal_mol"])
        for r in leg["replicas"]
        if r["half_forward_reverse_difference_kcal_mol"] is not None
    ]
    return LegSummary(
        replicas=leg["n_replicas"],
        mean_kcal_mol=leg["mean_kcal_mol"],
        replica_sd_kcal_mol=leg["replica_sd_kcal_mol"],
        replica_sem_kcal_mol=leg["replica_sem_kcal_mol"],
        mbar_error_kcal_mol=leg["mbar_propagated_error_kcal_mol"],
        minimum_adjacent_overlap=min(
            r["minimum_adjacent_overlap"] for r in leg["replicas"]
        ),
        minimum_effective_samples=min(
            r["minimum_effective_samples"] for r in leg["replicas"]
        ),
        largest_half_forward_reverse_difference_kcal_mol=max(halves)
        if halves
        else None,
    )


@capability(
    "simulation.analyze_binding_free_energy",
    version="1.0",
    family="investigate",
    summary="Combine relative free-energy legs into binding differences, cycles and checks",
    render=render,
)
def analyze_binding_free_energy(
    request: BindingRequest, ctx: StepContext
) -> BindingResult:
    """Combine the legs, close the cycles, compare with the reference and run the checks."""
    from alambiq.capabilities.investigate.science.binding import (
        analyze_binding_series,
        assess_binding,
        binding_check_settings,
        read_native_com_drift,
    )

    legs = [ctx.read_json(path) for path in request.legs]
    reference = None
    if request.reference is not None:
        reference = {
            "temperature_kelvin": request.reference.temperature_kelvin,
            "source": {
                "citation": request.reference.citation,
                "license": request.reference.license,
            },
            "conditions": request.reference.conditions,
            "error_description": request.reference.error_description,
            "observations": [asdict(o) for o in request.reference.observations],
        }
    series = {
        "schema_version": "0.1",
        "title": request.title,
        "edges": [asdict(e) for e in request.edges],
        "cycles": [asdict(c) for c in request.cycles],
        "reference": reference,
    }
    criteria = None
    if request.checks is not None:
        criteria = binding_check_settings(
            {k: v for k, v in asdict(request.checks).items() if v is not None}
        )
    data = analyze_binding_series(series, legs)
    bound = {r["id"]: r for e in data["edges"] for r in e["bound"]["replicas"]}
    unknown = sorted(set(request.structural_diagnostics) - set(bound))
    if unknown:
        raise ValueError(
            f"structural_diagnostics names replicas that are not bound replicas: {unknown}; "
            "each key must be the id of a bound replica."
        )
    diagnostics = {}
    for replica_id, path in request.structural_diagnostics.items():
        replica = bound[replica_id]
        diagnostics[replica_id] = {
            **read_native_com_drift(ctx.resolve(path), len(replica["overlap_matrix"])),
            "replica_id": replica_id,
            "trajectory_sha256": replica["trajectory_sha256"],
        }
    if diagnostics:
        data["structural_diagnostics"] = diagnostics
    assessment = None
    if criteria is not None:
        assessment = assess_binding(data, criteria, diagnostics)
        data["assessment"] = assessment
    reference_edges: dict[str, float] = {}
    reference_cycles: dict[str, dict[str, Any]] = {}
    if data["reference"] is not None:
        reference_edges = dict(
            zip(
                data["reference"]["edge_order"],
                data["reference"]["edge_means_kcal_mol"],
                strict=True,
            )
        )
        reference_cycles = {c["id"]: c for c in data["reference"]["cycles"]}
    edges = [
        EdgeResult(
            id=e["id"],
            a=e["a"],
            b=e["b"],
            delta_delta_g_kcal_mol=e["delta_delta_g_kcal_mol"],
            replica_sem_kcal_mol=e["replica_sem_kcal_mol"],
            mbar_error_kcal_mol=e["mbar_propagated_error_kcal_mol"],
            bound=_leg_summary(e["bound"]),
            solvent=_leg_summary(e["solvent"]),
            reference_delta_delta_g_kcal_mol=reference_edges.get(e["id"]),
        )
        for e in data["edges"]
    ]
    cycles = [
        CycleResult(
            id=c["id"],
            closure_kcal_mol=c["closure_kcal_mol"],
            replica_sem_kcal_mol=c["replica_sem_kcal_mol"],
            mbar_error_kcal_mol=c["mbar_propagated_error_kcal_mol"],
            reference_closure_kcal_mol=reference_cycles[c["id"]]["closure_kcal_mol"]
            if c["id"] in reference_cycles
            else None,
            reference_propagated_error_kcal_mol=reference_cycles[c["id"]][
                "propagated_reported_error_kcal_mol"
            ]
            if c["id"] in reference_cycles
            else None,
        )
        for c in data["cycles"]
    ]
    rows = [
        CheckRow(
            diagnostic=r["diagnostic"],
            identity=r["identity"],
            observed=r["observed"],
            limit=r["limit"],
            comparison=r["comparison"],
            status=r["status"],
            reason=r.get("reason"),
            selected_fraction=r.get("selected_fraction"),
        )
        for r in (assessment["checks"] if assessment else [])
    ]
    if assessment and assessment["status"] != "pass":
        ctx.note(
            f"The checks gave {assessment['status']}; the checks of the result list the rows "
            "that failed or could not be assessed."
        )
    return BindingResult(
        title=data["title"],
        temperature_kelvin=data["temperature_kelvin"],
        direction=data["direction"],
        edges=edges,
        cycles=cycles,
        status=assessment["status"] if assessment else None,
        sampling_status=assessment.get("sampling_status") if assessment else None,
        structural_review_status=assessment.get("structural_review_status")
        if assessment
        else None,
        checks=rows,
        details=ctx.keep_json(data),
    )
