"""Compute the hydration free energy of a small molecule by alchemical simulation.

Returns the free energy of transferring one neutral molecule from the gas phase into water,
with its spread over independent repeats and the sampling diagnostics of each leg.

The method applies the OpenFE absolute solvation protocol (version 1.12.0):

- RDKit embeds the molecule from its SMILES and relaxes it with MMFF94. A molecule with
  explicit stereochemistry keeps it.
- The molecule has AshGC partial charges (the graph-network model of the OpenFF project that
  reproduces AM1-BCC charges) and Sage 2.3.0 parameters. TIP3P water fills a dodecahedral box
  with 1.5 nm of padding.
- In water and in vacuum, the molecule is decoupled from its surroundings along 14 lambda
  windows: electrostatics first, then Lennard-Jones interactions. Hamiltonian replica exchange
  samples the windows, with a 4 fs step and hydrogen masses of 3 amu.
- MBAR gives the decoupling free energy of each leg. The hydration free energy of a repeat is
  the vacuum decoupling free energy minus the solvent decoupling free energy. A negative value
  means that the molecule prefers water.
- The solvent leg samples `production_ns` of production after `equilibration_ns` of
  equilibration; the vacuum leg samples one fifth of the production length after half of the
  equilibration length, the ratios of the OpenFE defaults (10 ns and 1 ns in the solvent).
- `repeats` independent repeats run the whole protocol. The result is their mean, and the
  spread between them is the uncertainty.

Limits.

- The free energy depends on the force field and the charges. `docs/methods/molecules.md`
  gives the checks that were made against measured hydration free energies.
- The molecule must be neutral, closed-shell and one connected species, from H, C, N, O, F,
  P, S, Cl, Br and I, with at most 50 heavy atoms. Ions, salts and zwitterions are refused.
- The standard state is the same concentration in both phases. The free energy is that of
  one conformer-averaged molecule in the force field; slow conformational changes of a large
  or flexible molecule need long sampling.
- The default length (10 ns of solvent production, 3 repeats) takes hours on a GPU. A shorter
  length gives a result with an unconverged error; the result notes a length below 5 ns, fewer
  than 3 repeats and a low overlap between neighboring lambda windows.
- The CUDA platform runs in double precision. The CPU platform is available, but the single
  run of it on an Apple M-series CPU stopped with a numerical failure in the solvent leg; the
  CUDA runs of the same molecule completed.
- The MBAR error assumes uncorrelated samples. The uncertainty of the result is the spread
  between repeats, combined from the two legs as OpenFE does, and is zero with one repeat.
- Each repeat uses an automatic random seed, so a repeated run agrees statistically.
"""

from __future__ import annotations

import math
import tarfile
from dataclasses import dataclass
from typing import Any, Literal

from alambiq.capabilities.investigate.molecule_compute_hydration_free_energy_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

ITERATION_PS = 2.5
MAX_HEAVY_ATOMS = 50
LOW_OVERLAP = 0.03
SHORT_PRODUCTION_NS = 5.0


@dataclass(frozen=True, slots=True)
class HydrationRequest:
    """The molecule and the sampling settings.

    smiles:
        One neutral molecule as SMILES, 1 to 500 characters with no whitespace.
    repeats:
        The number of independent repeats, 1 to 10; default 3.
    production_ns:
        The production length of the solvent leg in nanoseconds, 0.05 to 100; default 10. A
        whole number of 2.5 ps iterations. The vacuum leg uses one fifth.
    equilibration_ns:
        The equilibration length of the solvent leg in nanoseconds, 0.05 to 10; default 1. A
        whole number of 2.5 ps iterations. The vacuum leg uses half.
    temperature_kelvin, pressure_bar:
        The temperature, 250 to 400 K (default 298.15), and the pressure, 0.1 to 100 bar
        (default 1).
    seed:
        The random seed of the conformer embedding, 0 to 2,147,483,647; default 1.
    platform:
        `CUDA` (default) or `CPU`; no fallback.
    timeout_seconds:
        The longest permitted duration of the simulation in seconds, 60 to 2,592,000; default
        172,800.
    keep_native_records:
        Store the native OpenFE output files, trajectories and checkpoints as one archive;
        default false.
    """

    smiles: str
    repeats: int = 3
    production_ns: float = 10.0
    equilibration_ns: float = 1.0
    temperature_kelvin: float = 298.15
    pressure_bar: float = 1.0
    seed: int = 1
    platform: Literal["CPU", "CUDA"] = "CUDA"
    timeout_seconds: int = 172800
    keep_native_records: bool = False


@dataclass(frozen=True, slots=True)
class LegRepeat:
    """One repeat of one leg."""

    decoupling_kcal_mol: float
    """The free energy of decoupling the molecule from the medium of the leg."""
    mbar_error_kcal_mol: float
    minimum_adjacent_overlap: float
    """The smallest overlap between neighboring lambda windows; 0.03 or more is expected."""
    minimum_effective_samples: float
    """The smallest effective sample size over the lambda windows."""


@dataclass(frozen=True, slots=True)
class HydrationRepeat:
    """One repeat of the protocol: its solvent and vacuum legs and their difference."""

    hydration_free_energy_kcal_mol: float
    mbar_error_kcal_mol: float
    """The two MBAR errors of the legs added in quadrature."""
    solvent: LegRepeat
    vacuum: LegRepeat


@dataclass(frozen=True, slots=True)
class HydrationResult:
    """The hydration free energy with its repeats and the location of the records."""

    smiles: str
    canonical_smiles: str
    hydration_free_energy_kcal_mol: float
    """The vacuum decoupling free energy minus the solvent one, averaged over repeats."""
    uncertainty_kcal_mol: float | None
    """The spread between repeats, combined over the legs; none with one repeat."""
    mean_mbar_error_kcal_mol: float
    repeats: list[HydrationRepeat]
    force_field: str
    charge_model: str
    production_ns: float
    equilibration_ns: float
    temperature_kelvin: float
    pressure_bar: float
    platform: str
    record: str
    native_records: str | None
    versions: dict[str, str]
    seconds: float


def _check(request: HydrationRequest) -> None:
    import re

    if (
        not request.smiles
        or len(request.smiles) > 500
        or re.search(r"\s", request.smiles)
    ):
        raise ValueError("smiles must be 1 to 500 characters with no whitespace")
    for name, value, low, high in (
        ("repeats", request.repeats, 1, 10),
        ("seed", request.seed, 0, 2147483647),
        ("timeout_seconds", request.timeout_seconds, 60, 2592000),
    ):
        if not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    for name, value, low, high in (
        ("production_ns", request.production_ns, 0.05, 100.0),
        ("equilibration_ns", request.equilibration_ns, 0.05, 10.0),
        ("temperature_kelvin", request.temperature_kelvin, 250.0, 400.0),
        ("pressure_bar", request.pressure_bar, 0.1, 100.0),
    ):
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low:g} and {high:g}")
    for name, value in (
        ("production_ns", request.production_ns),
        ("equilibration_ns", request.equilibration_ns),
    ):
        iterations = value * 1000 / ITERATION_PS
        if abs(iterations - round(iterations)) > 1e-6:
            raise ValueError(
                f"{name} must be a whole number of {ITERATION_PS:g} ps iterations"
            )
    for name, value, divisor in (
        ("production_ns", request.production_ns, 5),
        ("equilibration_ns", request.equilibration_ns, 2),
    ):
        iterations = value * 1000 / divisor / ITERATION_PS
        if abs(iterations - round(iterations)) > 1e-6:
            raise ValueError(
                f"{name} divided by {divisor} (the vacuum leg) must be a whole number of "
                f"{ITERATION_PS:g} ps iterations"
            )


def _leg(entry: dict[str, Any]) -> LegRepeat:
    import numpy as np

    matrix = np.asarray(entry["overlap_matrix"])
    return LegRepeat(
        decoupling_kcal_mol=entry["decoupling_kcal_mol"],
        mbar_error_kcal_mol=entry["mbar_error_kcal_mol"],
        minimum_adjacent_overlap=float(
            min(np.diag(matrix, 1).min(), np.diag(matrix, -1).min())
        ),
        minimum_effective_samples=float(min(entry["effective_samples"])),
    )


@capability(
    "molecule.compute_hydration_free_energy",
    version="1.0",
    family="investigate",
    summary="Compute the hydration free energy of a small molecule by simulation",
    render=render,
    environment="openfe",
)
def compute_hydration_free_energy(
    request: HydrationRequest, ctx: StepContext
) -> HydrationResult:
    """Run the absolute solvation protocol and gather the repeats."""
    import numpy as np

    _check(request)
    output = ctx.workdir / "hydration"
    found = ctx.run_in_environment(
        "openfe",
        "openfe_hydration:run",
        {
            "smiles": request.smiles,
            "repeats": request.repeats,
            "production_ns": request.production_ns,
            "equilibration_ns": request.equilibration_ns,
            "temperature_kelvin": request.temperature_kelvin,
            "pressure_bar": request.pressure_bar,
            "seed": request.seed,
            "platform": request.platform,
            "output": str(output),
        },
        timeout=request.timeout_seconds,
    )
    solvent = [_leg(e) for e in found["legs"]["solvent"]]
    vacuum = [_leg(e) for e in found["legs"]["vacuum"]]
    if len(solvent) != request.repeats or len(vacuum) != request.repeats:
        raise RuntimeError("the native results do not close the requested repeat count")
    repeats = [
        HydrationRepeat(
            hydration_free_energy_kcal_mol=v.decoupling_kcal_mol
            - s.decoupling_kcal_mol,
            mbar_error_kcal_mol=math.hypot(
                v.mbar_error_kcal_mol, s.mbar_error_kcal_mol
            ),
            solvent=s,
            vacuum=v,
        )
        for s, v in zip(solvent, vacuum, strict=True)
    ]
    if request.production_ns < SHORT_PRODUCTION_NS or request.repeats < 3:
        ctx.note(
            f"{request.repeats} repeat(s) with {request.production_ns:g} ns of solvent "
            f"production are below the 3 repeats and {SHORT_PRODUCTION_NS:g} ns that this "
            "check expects for a converged free energy."
        )
    low = min(
        leg.minimum_adjacent_overlap for r in repeats for leg in (r.solvent, r.vacuum)
    )
    if low < LOW_OVERLAP:
        ctx.note(
            f"The overlap between neighboring lambda windows falls to {low:.3f}, below "
            f"{LOW_OVERLAP}; the MBAR estimate of such a leg is unreliable."
        )
    record = {
        "smiles": request.smiles,
        "canonical_smiles": found["canonical_smiles"],
        "production_ns": request.production_ns,
        "equilibration_ns": request.equilibration_ns,
        "temperature_kelvin": request.temperature_kelvin,
        "pressure_bar": request.pressure_bar,
        "seed": request.seed,
        "platform": request.platform,
        "legs": found["legs"],
        "charges": found["charges"],
        "versions": found["versions"],
        "settings": found["settings"],
        "estimate_kcal_mol": found["estimate_kcal_mol"],
        "uncertainty_kcal_mol": found["uncertainty_kcal_mol"],
    }
    native = None
    if request.keep_native_records:
        archive = ctx.workdir / "native-records.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    bundle.add(
                        path, arcname=str(path.relative_to(output)), recursive=False
                    )
        native = str(ctx.keep(archive, "application/gzip").path)
    values = [r.hydration_free_energy_kcal_mol for r in repeats]
    return HydrationResult(
        smiles=request.smiles,
        canonical_smiles=found["canonical_smiles"],
        hydration_free_energy_kcal_mol=float(np.mean(values)),
        uncertainty_kcal_mol=found["uncertainty_kcal_mol"]
        if request.repeats > 1
        else None,
        mean_mbar_error_kcal_mol=float(
            np.mean([r.mbar_error_kcal_mol for r in repeats])
        ),
        repeats=repeats,
        force_field="openff-2.3.0 (Sage 2.3.0), TIP3P",
        charge_model="openff-gnn-am1bcc-1.0.0 (AshGC)",
        production_ns=request.production_ns,
        equilibration_ns=request.equilibration_ns,
        temperature_kelvin=request.temperature_kelvin,
        pressure_bar=request.pressure_bar,
        platform=request.platform,
        record=ctx.keep_json(record),
        native_records=native,
        versions=found["versions"],
        seconds=found["seconds"],
    )
