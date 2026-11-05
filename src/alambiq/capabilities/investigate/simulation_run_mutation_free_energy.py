"""Run one leg of the free energy of an amino-acid substitution with replica exchange.

Returns the free-energy difference of the transformation from the wild-type to the mutant state
for each replica of the leg, with the sampling diagnostics of each replica. The folded leg and
the reference leg of the same substitution, analyzed together by
`simulation.analyze_stability_change`, give the change in the free energy of folding.

The capability builds the hybrid-topology system of the two states of the leg that
`structure.prepare_mutation_inputs` wrote and samples it. The hybrid system is the one that
OpenFE builds for relative free-energy calculations: atoms mapped between the states follow the
parameters of the wild type at lambda 0 and those of the mutant at lambda 1; atoms present in one
state only are decoupled with soft-core terms. The settings are:

- ff14SB (Amber14) with TIP3P, particle mesh Ewald with a 0.9 nm cutoff, bonds to hydrogen
  constrained, a 2 fs integration step;
- Langevin dynamics at 1 ps^-1 and a Monte Carlo barostat at the stated temperature and pressure;
- replica exchange over `lambda_windows` windows, one replica per window, one exchange
  attempt after each iteration of `time_per_iteration_ps`;
- MBAR analysis of the decorrelated samples, with a bootstrap estimate of the error (`bootstraps` samples).

Each of the `replicas` independent replicas starts from the prepared coordinates and runs the
whole set of lambda windows. Each reports its free-energy difference, the MBAR error, the
overlap between neighboring lambda windows, the effective sample size per window, and the
estimates from increasing forward and reverse fractions of the data.

Limits.

- The free energy depends on the force field, the template conformation of the new side chain
  and the sampling. The spread between replicas measures the reproducibility of the sampling from
  one preparation, and the spread across independent preparations is larger. The MBAR error
  assumes uncorrelated samples and is smaller than both.
- The protonation states are fixed at preparation.
- On CUDA, `precision` selects mixed or double precision. Double precision runs several times
  slower than mixed precision.
- The legs run as sampled: a short run reports small overlap and large spreads, and the notes
  say when the production length is below 2,500 ps per window.
- The capability refuses more than 10 replicas, more than 41 lambda windows and a leg
  whose files differ from those that were prepared.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.investigate.simulation_run_mutation_free_energy_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class MutationLegRequest:
    """The prepared leg and the sampling settings.

    inputs:
        A path to the archive of one leg from `structure.prepare_mutation_inputs`
        (`folded_inputs` or `reference_inputs`).
    temperature_kelvin, pressure_bar:
        The temperature, 250 to 400 K, and the pressure, 0.1 to 100 bar; default 300 K, 1 bar.
    replicas:
        The number of replicas, 1 to 10; default 1. The replicas of a leg start from the prepared
        coordinates and differ in the random velocities and exchange moves. An uncertainty needs
        independent preparations (several runs of `structure.prepare_mutation_inputs` with different
        seeds), each with its own run of this capability.
    lambda_windows:
        The number of lambda windows, 5 to 41; default 12.
    equilibration_ps, production_ps, time_per_iteration_ps:
        The equilibration length, the production length and the length of one replica-exchange
        iteration. The iteration length is a whole number of 2 fs steps; the other two lengths
        are whole numbers of iterations. Defaults 50 ps, 2,500 ps and 1 ps. The lengths are per
        lambda window.
    platform:
        `CPU` or `CUDA`; no fallback. Default `CPU`.
    precision:
        `mixed` or `double`; applies to `CUDA`. Default `mixed`.
    threads:
        The number of CPU threads, 1 to 128; default 8.
    timeout_seconds:
        The longest permitted duration of the simulation, 60 s or more.
    bootstraps:
        The number of bootstrap samples for the MBAR error of each replica, 0 to 1,000; default
        1,000. With 0, the error is the analytical MBAR error.
    forward_reverse_samples:
        The number of fractions of the data in the forward and reverse estimates, 2 to 20;
        default 10.
    keep_native_records:
        Store the trajectory and checkpoint files of every replica as one archive.
    """

    inputs: str
    temperature_kelvin: float = 300.0
    pressure_bar: float = 1.0
    replicas: int = 1
    lambda_windows: int = 12
    equilibration_ps: float = 50.0
    production_ps: float = 2500.0
    time_per_iteration_ps: float = 1.0
    platform: Literal["CPU", "CUDA"] = "CPU"
    precision: Literal["mixed", "double"] = "mixed"
    threads: int = 8
    timeout_seconds: int = 86400
    bootstraps: int = 1000
    forward_reverse_samples: int = 10
    keep_native_records: bool = True


@dataclass(frozen=True, slots=True)
class MutationRepeat:
    """One replica of the leg."""

    id: str
    delta_g_kcal_mol: float
    mbar_error_kcal_mol: float
    minimum_adjacent_overlap: float
    minimum_effective_samples: float
    sampling_seconds: float


@dataclass(frozen=True, slots=True)
class MutationLegResult:
    """The replicas of the leg and the location of its records."""

    leg: str
    mutation: str
    repeats: list[MutationRepeat]
    mean_delta_g_kcal_mol: float
    replica_sd_kcal_mol: float | None
    """None with a single replica."""
    leg_record: str
    """The JSON record that `simulation.analyze_stability_change` accepts as a leg."""
    native_records: str | None
    versions: dict[str, str]
    platform: str
    precision: str
    seconds: float


@capability(
    "simulation.run_mutation_free_energy",
    version="1.0",
    family="investigate",
    summary="Run one leg of the free energy of an amino-acid substitution with replica exchange",
    render=render,
    environment="openfe",
)
def run_mutation_free_energy(request: MutationLegRequest, ctx: StepContext) -> MutationLegResult:
    """Unpack the leg, sample every replica, and assemble the leg record."""
    from alambiq.capabilities.investigate.science.mutation import (
        check_sampling,
        leg_statistics,
    )

    check_sampling(
        replicas=request.replicas,
        lambda_windows=request.lambda_windows,
        temperature_kelvin=request.temperature_kelvin,
        pressure_bar=request.pressure_bar,
        equilibration_ps=request.equilibration_ps,
        production_ps=request.production_ps,
        time_per_iteration_ps=request.time_per_iteration_ps,
        threads=request.threads,
        timeout_seconds=request.timeout_seconds,
        bootstraps=request.bootstraps,
        forward_reverse_samples=request.forward_reverse_samples,
    )
    archive = ctx.resolve(request.inputs)
    leg_directory = ctx.workdir / "leg"
    leg_directory.mkdir()
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            names = bundle.getnames()
            if sorted(names) != ["leg.json", "mutant.pdb", "wild_type.pdb"]:
                raise ValueError(
                    f"inputs holds {names}; an archive of leg.json, mutant.pdb and wild_type.pdb is required"
                )
            bundle.extractall(leg_directory, filter="data")
    except tarfile.TarError as exc:
        raise ValueError(f"inputs is {request.inputs!r}, which is not a leg archive: {exc}") from exc
    if request.production_ps < 2500:
        ctx.note(
            f"{request.production_ps:g} ps of production per window is below 2500 ps. For one substitution, the "
            "convergence record of this method area shows a spread of 1.3 kcal/mol across independent "
            "preparations at 1000 ps and 0.3 kcal/mol at 2500 ps."
        )
    output = ctx.workdir / "mutation-leg"
    found = ctx.run_in_environment(
        "openfe",
        "openmm_mutation:run",
        {
            "leg_directory": str(leg_directory),
            "temperature_kelvin": request.temperature_kelvin,
            "pressure_bar": request.pressure_bar,
            "replicas": request.replicas,
            "lambda_windows": request.lambda_windows,
            "equilibration_ps": request.equilibration_ps,
            "production_ps": request.production_ps,
            "time_per_iteration_ps": request.time_per_iteration_ps,
            "platform": request.platform,
            "precision": request.precision,
            "threads": request.threads,
            "bootstraps": request.bootstraps,
            "forward_reverse_samples": request.forward_reverse_samples,
            "output": str(output),
        },
        timeout=request.timeout_seconds,
    )
    record = found["record"]
    statistics = leg_statistics(record)
    native = None
    if request.keep_native_records:
        bundle_path = ctx.workdir / "native-records.tar.gz"
        with tarfile.open(bundle_path, "w:gz") as bundle:
            for path in sorted(output.rglob("*.nc")):
                bundle.add(path, arcname=str(path.relative_to(output)), recursive=False)
        native = str(ctx.keep(bundle_path, "application/gzip").path)
    return MutationLegResult(
        leg=record["leg"],
        mutation=f"{record['wild_type_residue']}>{record['mutant_residue']}",
        repeats=[
            MutationRepeat(
                id=r["id"],
                delta_g_kcal_mol=r["delta_g_kcal_mol"],
                mbar_error_kcal_mol=r["mbar_error_kcal_mol"],
                minimum_adjacent_overlap=s["overlap"],
                minimum_effective_samples=float(min(r["effective_samples"])),
                sampling_seconds=r["sampling_seconds"],
            )
            for r, s in zip(record["replicas"], statistics["by_replica"], strict=True)
        ],
        mean_delta_g_kcal_mol=statistics["mean"],
        replica_sd_kcal_mol=statistics["sd"],
        leg_record=ctx.keep_json(record),
        native_records=native,
        versions=found["method"]["versions"],
        platform=request.platform,
        precision=found["method"]["precision"],
        seconds=found["seconds"],
    )
