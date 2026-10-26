"""Reconstruct a free-energy profile along a reaction coordinate from umbrella windows.

Returns the potential of mean force along one coordinate, the free-energy difference
between the product bins and the reactant bins, the height of the transition bins above a
reference bin, and the diagnostics of the sampling. A classical one-way flux through a
dividing surface gives a rate and an activation free energy when requested.

Each umbrella window is a QM/MM simulation (`simulation.run_qmmm_dynamics`) with a harmonic
bias on the same distance-difference coordinate, centered at its own value. A replica is a
complete set of windows run with independent seeds. For each replica the capability:

- decorrelates the samples of each window;
- combines the windows with MBAR against the unbiased thermodynamic state;
- takes the density of the coordinate in the bins of the request, which gives the
  potential of mean force and its asymptotic uncertainty, referenced to a bin in the
  reactant bins;
- derives the free-energy difference between the product bins and the reactant bins from
  their integrated probabilities;
- derives the maximum of the profile over the populated transition bins, relative to the
  reference bin.

Chronological blocks of each replica are analyzed separately. A block that cannot be
reconstructed is recorded as a failure.

The result also gives the spread of the independent replicas, the uncertainty of each
replica, the overlap between neighboring windows and the effective sample sizes. These
indicate whether the windows connect.

Limits.

- The profile is that of one chosen coordinate under one Hamiltonian, with the other
  restraints of the simulation. It holds when the slowly relaxing degrees of freedom
  orthogonal to the coordinate are sampled within each window.
- The windows must overlap. The analysis refuses windows when the graph of window pairs
  with an overlap above 1e-12 is not connected. The effective sample sizes, the agreement
  among replicas and the agreement between time blocks indicate whether the overlap
  suffices.
- The asymptotic uncertainty assumes decorrelated samples. The stride that decorrelates
  them is the larger of the statistical inefficiencies of the coordinate and of the
  harmonic bias energy, rounded up. The uncertainty is too small when a relaxation time of
  the system exceeds the length of the simulation.
- A QM/MM barrier at PBE/6-31G** includes the error of that functional and basis.
- The classical flux has a transmission factor of 1, a constant inverse mass for the
  coordinate and a finite-width surface estimate. Its rate is a transition-state estimate.

The capability refuses:

- fewer than 2 or more than 20 replicas;
- fewer than 2 or more than 100 windows per replica;
- more than 100,000 frames per replica or more than 1,000,000 frames in all;
- an umbrella window with fewer frames after `discard_frames` than 20 times the largest
  value of `block_counts`, because every time block needs at least 20 frames.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from alambiq.capabilities.investigate.simulation_analyze_reaction_free_energy_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

PROFILE = "pymbar.umbrella-distance-difference.v1"


@dataclass(frozen=True, slots=True)
class WindowRun:
    """One umbrella window of one replica, as `simulation.run_qmmm_dynamics` recorded it.

    window:
        Identifier of the umbrella window, identical in every replica; the windows are
        listed in order of increasing center.
    summary:
        The summary record of the simulation.
    table:
        The state table of the simulation.
    discard_frames:
        The number of leading production frames excluded as equilibration.
    """

    window: str
    summary: str
    table: str
    discard_frames: int = 0


@dataclass(frozen=True, slots=True)
class Replica:
    """One independent, complete set of umbrella windows.

    id:
        Identifier of the replica.
    windows:
        The windows of the replica, in order of increasing center.
    """

    id: str
    windows: list[WindowRun]


@dataclass(frozen=True, slots=True)
class Flux:
    """A classical one-way flux through a dividing surface.

    transition_bin:
        The zero-based bin that is the dividing surface; it must be one of the
        transition bins.
    inverse_mass_dalton:
        The constant inverse effective mass of the coordinate, in 1/dalton.
    """

    transition_bin: int
    inverse_mass_dalton: float


@dataclass(frozen=True, slots=True)
class ReactionRequest:
    """The replicas, and the bins that define the reactant, product and transition regions.

    title:
        Title of the analysis.
    replicas:
        2 to 20 replicas with the same windows.
    bin_edges_angstrom:
        The edges of 3 to 200 bins along the coordinate, strictly increasing, in
        angstrom.
    reference_bin:
        The zero-based bin that is the reference of the profile; it must be one of
        the reactant bins.
    reactant_bins, product_bins, transition_bins:
        Zero-based bins of each region, disjoint.
    block_counts:
        The numbers of chronological blocks into which each replica is divided: 1 to 4
        values, each from 2 to 10.
    transition_state_flux:
        Optional. A classical flux through one of the transition bins.
    """

    title: str
    replicas: list[Replica]
    bin_edges_angstrom: list[float]
    reference_bin: int
    reactant_bins: list[int]
    product_bins: list[int]
    transition_bins: list[int]
    block_counts: list[int]
    transition_state_flux: Flux | None = None


@dataclass(frozen=True, slots=True)
class ReplicaProfile:
    """One replica's summary."""

    id: str
    basin_delta_g_kj_mol: float | None
    """Product minus reactant basin free energy, None if a basin was unsampled."""
    peak_minus_reference_kj_mol: float | None
    minimum_adjacent_overlap: float
    minimum_state_effective_samples: float
    blocks_total: int
    blocks_failed: int


@dataclass(frozen=True, slots=True)
class FluxSummary:
    """The flux estimates over the replicas."""

    log_rate_per_second_mean: float | None
    log_rate_per_second_replica_sem: float | None
    activation_free_energy_kj_mol_mean: float | None
    activation_free_energy_kj_mol_replica_sem: float | None


@dataclass(frozen=True, slots=True)
class ReactionResult:
    """The profile, the product-minus-reactant free-energy difference and the sampling diagnostics."""

    title: str
    temperature_kelvin: float
    coordinate: str
    """The reaction coordinate, given as the two distances of which it is the difference."""
    bin_centres_angstrom: list[float]
    pmf_mean_kj_mol: list[float | None]
    """By bin, relative to the reference bin; None where any replica has an empty bin."""
    pmf_replica_sd_kj_mol: list[float | None]
    pmf_replica_sem_kj_mol: list[float | None]
    basin_delta_g_mean_kj_mol: float | None
    basin_delta_g_replica_sd_kj_mol: float | None
    basin_delta_g_replica_sem_kj_mol: float | None
    replicas: list[ReplicaProfile]
    flux: FluxSummary | None
    windows: int
    frames: int
    details: str
    """A JSON file with the profile, covariance, blocks and sampling record of every replica."""
    pymbar_version: str


@capability(
    "simulation.analyze_reaction_free_energy",
    version="1.0",
    family="investigate",
    summary="Reconstruct a free-energy profile along a reaction coordinate from umbrella windows",
    render=render,
    environment="pymbar",
)
def analyze_reaction_free_energy(
    request: ReactionRequest, ctx: StepContext
) -> ReactionResult:
    """Join the recorded windows, run the analysis and summarize it."""
    from alambiq.capabilities.investigate.science.umbrella import assemble_samples

    replicas = [
        {
            "id": r.id,
            "windows": [
                {
                    "window": w.window,
                    "summary": ctx.read_json(w.summary),
                    "table": ctx.resolve(w.table),
                    "discard_frames": w.discard_frames,
                }
                for w in r.windows
            ],
        }
        for r in request.replicas
    ]
    samples = assemble_samples(request.title, replicas)
    parameters: dict[str, Any] = {
        "bin_edges_angstrom": request.bin_edges_angstrom,
        "reference_bin": request.reference_bin,
        "reactant_bins": request.reactant_bins,
        "product_bins": request.product_bins,
        "transition_bins": request.transition_bins,
        "block_counts": request.block_counts,
    }
    if request.transition_state_flux is not None:
        parameters["transition_state_flux"] = {
            "transition_bin": request.transition_state_flux.transition_bin,
            "inverse_mass_dalton": request.transition_state_flux.inverse_mass_dalton,
        }
    found = ctx.run_in_environment(
        "pymbar",
        "umbrella_analysis:run",
        {"samples": samples, "parameters": parameters, "method_profile": PROFILE},
        timeout=3600,
    )
    edges = request.bin_edges_angstrom
    summary = found["summary"]
    profiles = []
    failed_total = 0
    for replica in found["replicas"]:
        blocks = [b for group in replica["time_blocks"] for b in group["blocks"]]
        failed = sum(b["error"] is not None for b in blocks)
        failed_total += failed
        profiles.append(
            ReplicaProfile(
                id=replica["id"],
                basin_delta_g_kj_mol=replica["basin_delta_g_kj_mol"],
                peak_minus_reference_kj_mol=replica["peak_minus_reference_kj_mol"],
                minimum_adjacent_overlap=_minimum_adjacent(replica["overlap_matrix"]),
                minimum_state_effective_samples=min(replica["state_effective_samples"]),
                blocks_total=len(blocks),
                blocks_failed=failed,
            )
        )
    if failed_total:
        ctx.note(
            f"{failed_total} chronological block(s) could not be reconstructed; the details "
            "give the reason for each."
        )
    flux = None
    if "transition_state_flux" in summary:
        block = summary["transition_state_flux"]
        flux = FluxSummary(
            log_rate_per_second_mean=block["log_rate_per_second"]["mean"],
            log_rate_per_second_replica_sem=block["log_rate_per_second"]["replica_sem"],
            activation_free_energy_kj_mol_mean=block["activation_free_energy_kj_mol"][
                "mean"
            ],
            activation_free_energy_kj_mol_replica_sem=block[
                "activation_free_energy_kj_mol"
            ]["replica_sem"],
        )
    return ReactionResult(
        title=request.title,
        temperature_kelvin=samples["temperature_kelvin"],
        coordinate=samples["coordinate"]["definition"],
        bin_centres_angstrom=[
            (a + b) / 2 for a, b in zip(edges[:-1], edges[1:], strict=True)
        ],
        pmf_mean_kj_mol=summary["pmf_mean_kj_mol"],
        pmf_replica_sd_kj_mol=summary["pmf_replica_sd_kj_mol"],
        pmf_replica_sem_kj_mol=summary["pmf_replica_sem_kj_mol"],
        basin_delta_g_mean_kj_mol=summary["basin_delta_g_mean_kj_mol"],
        basin_delta_g_replica_sd_kj_mol=summary["basin_delta_g_replica_sd_kj_mol"],
        basin_delta_g_replica_sem_kj_mol=summary["basin_delta_g_replica_sem_kj_mol"],
        replicas=profiles,
        flux=flux,
        windows=len(samples["windows"]),
        frames=sum(
            len(t["coordinate_angstrom"])
            for r in samples["replicas"]
            for t in r["windows"]
        ),
        details=ctx.keep_json(json.loads(json.dumps(found))),
        pymbar_version=found["method"]["pymbar_version"],
    )


def _minimum_adjacent(matrix: list[list[float]]) -> float:
    n = len(matrix)
    values = [matrix[i][i + 1] for i in range(n - 1)] + [
        matrix[i + 1][i] for i in range(n - 1)
    ]
    return float(min(values)) if values else math.nan
