"""Map a free-energy surface along one or two torsions with metadynamics.

Returns the free-energy surface along the chosen torsions, reweighted from the production
frames of one trajectory, with the effective sample size, the fraction of bins visited and
the difference between its two halves.

Well-tempered metadynamics adds Gaussian potential terms (hills) to the potential along a
periodic torsion, or along a pair of torsions, at the values that the simulation visits.
The simulation has three phases:

- Adaptation: hills are deposited, and the implied surface is recorded at 1/4, 1/2, 3/4
  and the end of the phase.
- Frozen equilibration: the bias is held constant.
- Production: sampling under the fixed bias, recording each torsion and the bias energy at
  every report.

Because the bias is fixed in production, each frame is reweighted by exp(+bias/kT). The
free-energy surface is -kT ln of the weighted histogram of the torsions. Free energies are
relative: the lowest bin is set to zero, and regions never visited are blank.

Limits.

- Only the torsions named in the request are biased, so the surface applies when those
  torsions capture the slow motion of the system. Any other slow motion (a ring pucker, a
  loop, solvent reorganization, a second torsion that couples to the first) is unbiased and
  can change the surface without changing these diagnostics.
- A barrier along an unbiased coordinate keeps its full height, and the simulation can fail
  to cross it.
- The surface is an estimate from one trajectory. The half-to-half difference checks the
  consistency of its two halves.
- The effective sample fraction is the effective sample size divided by the number of
  frames. Below 0.1, the frames with the largest weights dominate the surface, which has a
  large error, and a note is issued.
- The torsion atoms must form a bonded path in the topology.
- CUDA simulations require DisablePmeStream=true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from alambiq.capabilities.investigate.simulation_run_metadynamics_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class TorsionAtom:
    """One atom of a torsion, named by the labels of the topology.

    chain_id:
        The topology chain id (`A` for a prepared protein).
    residue:
        The residue number in the topology, with an insertion letter if any.
    atom_name:
        The name of the atom in its residue, such as `N` or `CA`.
    """

    chain_id: str
    residue: str
    atom_name: str


@dataclass(frozen=True, slots=True)
class Torsion:
    """A periodic torsion to bias.

    name:
        The identifier of the torsion, unique within the request.
    atoms:
        Four atoms forming a bonded three-bond path.
    width_radians:
        The Gaussian width, from 0.05 to pi, in radians.
    grid_width:
        The number of points on the bias grid, at least 16 and with at least 4
        intervals per Gaussian width.
    """

    name: str
    atoms: list[TorsionAtom]
    width_radians: float
    grid_width: int


@dataclass(frozen=True, slots=True)
class MetadynamicsRequest:
    """The system, the torsions and the length of each phase.

    system, state, topology:
        Paths to the serialized OpenMM system, the OpenMM state file and the topology PDB, unbiased,
        as produced by `structure.prepare_protein_system` or a dynamics run.
    torsions:
        1 or 2 torsions to bias.
    ensemble:
        `npt` or `nvt` for a periodic system, `implicit` for a non-periodic one.
    bias_factor, height_kj_mol, deposition_interval:
        The well-tempered bias factor (2 to 100), the initial Gaussian height in kJ/mol
        and the number of integration steps between depositions.
    adaptation_steps, frozen_equilibration_steps, production_steps:
        The length of each phase, in integration steps. Adaptation must be a multiple of the
        deposition interval and at least 4 deposition intervals long; production must
        be a multiple of the report interval.
    report_interval:
        The number of integration steps between production samples.
    histogram_bins:
        The number of bins per torsion for the reweighted surface: 8 to 180 for one
        torsion, 8 to 90 for two torsions.
    temperature_kelvin, friction_per_ps, timestep_fs, pressure_bar, barostat_interval:
        The settings of the thermostat, the timestep and (for `npt`) the barostat.
    platform, platform_properties, seed:
        The OpenMM platform, its properties (1 CPU thread by default) and the random
        seed.
    """

    system: str
    state: str
    topology: str
    torsions: list[Torsion]
    ensemble: Literal["npt", "nvt", "implicit"]
    adaptation_steps: int
    production_steps: int
    report_interval: int
    bias_factor: float = 10.0
    height_kj_mol: float = 1.0
    deposition_interval: int = 500
    frozen_equilibration_steps: int = 0
    histogram_bins: int = 72
    temperature_kelvin: float = 300.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 2.0
    pressure_bar: float = 1.0
    barostat_interval: int = 25
    platform: Literal["CPU", "Reference", "CUDA", "OpenCL"] = "CPU"
    platform_properties: dict[str, str] = field(default_factory=dict)
    seed: int = 1


@dataclass(frozen=True, slots=True)
class Surface:
    """The reweighted free-energy surface on a regular grid of torsion angles."""

    axes_radians: list[float]
    """Bin centers, identical for each torsion."""
    free_energy_kj_mol: list[Any]
    """One list of bins for a single torsion, a bins-by-bins grid for two; None where no sample falls."""


@dataclass(frozen=True, slots=True)
class MetadynamicsResult:
    """The surface and the indicators of its reliability."""

    torsions: list[str]
    surface: Surface
    samples: int
    effective_sample_size: float
    effective_sample_fraction: float
    bins_visited_fraction: float
    half_to_half_rms_kj_mol: float | None
    adaptation_settling_rms_kj_mol: float | None
    """RMS change of the adaptation estimate between the last two quarters, over low-energy points."""
    kt_kj_mol: float
    bias_factor: float
    samples_table: str
    final_state: str
    record: str
    platform: str
    versions: dict[str, Any]


@capability(
    "simulation.run_metadynamics",
    version="1.0",
    family="investigate",
    summary="Map a free-energy surface along one or two torsions with metadynamics",
    render=render,
    environment="openmm",
)
def run_metadynamics(
    request: MetadynamicsRequest, ctx: StepContext
) -> MetadynamicsResult:
    """Adapt the bias, freeze it, sample under it and reweight."""
    import numpy as np

    from alambiq.capabilities.investigate.science.simulation import (
        check_metadynamics,
        parse_residue,
        platform_properties,
    )

    torsions = []
    for row in request.torsions:
        atoms = []
        for atom in row.atoms:
            number, insertion = parse_residue(atom.residue)
            atoms.append(
                {
                    "chain_id": atom.chain_id,
                    "residue_id": number,
                    "insertion_code": insertion,
                    "atom_name": atom.atom_name,
                }
            )
        torsions.append(
            {
                "name": row.name,
                "atoms": atoms,
                "width_radians": row.width_radians,
                "grid_width": row.grid_width,
            }
        )
    check_metadynamics(
        ensemble=request.ensemble,
        bias_factor=request.bias_factor,
        height_kj_mol=request.height_kj_mol,
        deposition_interval=request.deposition_interval,
        adaptation_steps=request.adaptation_steps,
        frozen_equilibration_steps=request.frozen_equilibration_steps,
        production_steps=request.production_steps,
        report_interval=request.report_interval,
        histogram_bins=request.histogram_bins,
        torsions=torsions,
    )
    for name, value, low, high in (
        ("temperature_kelvin", request.temperature_kelvin, 1, 500),
        ("friction_per_ps", request.friction_per_ps, 0.01, 100),
        ("timestep_fs", request.timestep_fs, 0.1, 4),
        ("pressure_bar", request.pressure_bar, 0.0001, 10000),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value:g}; {name} must be between {low:g} and {high:g}"
            )
    if not 1 <= request.seed <= 2147483647:
        raise ValueError(
            f"seed is {request.seed}; seed must be an integer from 1 to 2147483647"
        )
    if not 1 <= request.barostat_interval <= 1000:
        raise ValueError(
            f"barostat_interval is {request.barostat_interval}; barostat_interval must "
            "be an integer from 1 to 1000"
        )

    output = ctx.workdir / "metadynamics"
    found = ctx.run_in_environment(
        "openmm",
        "openmm_metadynamics:run",
        {
            "system_xml": str(ctx.resolve(request.system)),
            "state_xml": str(ctx.resolve(request.state)),
            "topology_pdb": str(ctx.resolve(request.topology)),
            "torsions": torsions,
            "ensemble": request.ensemble,
            "bias_factor": request.bias_factor,
            "height_kj_mol": request.height_kj_mol,
            "deposition_interval": request.deposition_interval,
            "adaptation_steps": request.adaptation_steps,
            "frozen_equilibration_steps": request.frozen_equilibration_steps,
            "production_steps": request.production_steps,
            "report_interval": request.report_interval,
            "histogram_bins": request.histogram_bins,
            "temperature_kelvin": request.temperature_kelvin,
            "friction_per_ps": request.friction_per_ps,
            "timestep_fs": request.timestep_fs,
            "pressure_bar": request.pressure_bar,
            "barostat_interval": request.barostat_interval,
            "platform": request.platform,
            "platform_properties": platform_properties(
                request.platform, request.platform_properties
            ),
            "seed": request.seed,
            "output": str(output),
        },
        timeout=14 * 24 * 3600,
    )

    production = found["production"]
    surface = found["free_energy"]
    kt = found["kt_kj_mol"]
    settling = None
    estimates = found["adaptation_estimates"]
    if len(estimates) >= 2:
        last = np.array(estimates[-1]["free_energy_kj_mol"], dtype=float)
        before = np.array(estimates[-2]["free_energy_kj_mol"], dtype=float)
        mask = np.isfinite(last) & np.isfinite(before) & (last < 6 * kt)
        settling = (
            float(np.sqrt(np.mean((last[mask] - before[mask]) ** 2)))
            if mask.any()
            else None
        )
    if production["effective_sample_fraction"] < 0.1:
        ctx.note(
            f"The effective sample fraction of the production frames is "
            f"{production['effective_sample_fraction']:.0%} (below 10 %), so the "
            "reweighted surface rests on a small effective sample."
        )
    if surface["bins_visited_fraction"] < 0.5:
        ctx.note(
            f"Production visited {surface['bins_visited_fraction']:.0%} of the torsion grid "
            "(below 50 %). Blank regions were not sampled."
        )
    return MetadynamicsResult(
        torsions=[t["name"] for t in torsions],
        surface=Surface(
            axes_radians=surface["axes_radians"],
            free_energy_kj_mol=surface["reweighted_kj_mol"],
        ),
        samples=production["samples"],
        effective_sample_size=production["effective_sample_size"],
        effective_sample_fraction=production["effective_sample_fraction"],
        bins_visited_fraction=surface["bins_visited_fraction"],
        half_to_half_rms_kj_mol=surface["half_to_half_rms_kj_mol"],
        adaptation_settling_rms_kj_mol=settling,
        kt_kj_mol=kt,
        bias_factor=found["bias_factor"],
        samples_table=str(ctx.keep(output / "samples.csv", "text/csv").path),
        final_state=str(ctx.keep(output / "final_state.xml", "application/xml").path),
        record=ctx.keep_json(found),
        platform=found["platform"],
        versions=found["versions"],
    )
