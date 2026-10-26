"""Run one leg of a relative binding free-energy transformation with OpenFE.

Returns the free-energy difference of the transformation A to B for each replica of the
leg, with the sampling diagnostics of each replica. A bound leg and a solvent leg of the
same edge, analyzed together by `simulation.analyze_binding_free_energy`, give the relative
binding free energy.

The capability applies the OpenFE relative hybrid topology protocol to two positioned
neutral ligands, an atom mapping between them and, for the bound leg, the protein. Each
ligand is an SDF with AshGC partial charges. The protocol uses:

- Sage 2.3.0 for the ligands, ff14SB for the protein and TIP3P for the water;
- hydrogen masses of 1.0 amu (no hydrogen mass repartitioning);
- an integration step of 2 fs;
- a set of lambda windows that connects the two ligands.

Each of the `replicas` independent replicas samples the lambda windows with replica
exchange. Each replica gives:

- its free-energy difference for the transformation A to B;
- the error that MBAR reports for it;
- the overlap between neighboring lambda windows;
- the effective sample size per lambda window;
- the estimates from increasing forward and reverse fractions of the data.

Limits.

- The free energy depends on the force field, the charges and the atom mapping.
- The atom mapping checks cover element, isotope, aromaticity and charge retention,
  stereochemistry, ring completeness and connectivity of the core.
- The ligands are neutral and already positioned in the binding site.
- The protonation of the protein is set before the simulation. The protein starts from the
  supplied conformation and samples the conformations reachable within the simulated time.
- A binding free energy requires both legs of the edge.
- Overlap between lambda windows, the effective sample sizes and the agreement of the
  forward and reverse estimates indicate whether the sampling converged.
- The MBAR error assumes uncorrelated samples.
- Each replica uses an automatic random seed. Replicas are statistically independent, and
  a repeated run agrees statistically.
- The CUDA platform runs in double precision. The tests exercise it with a stand-in
  platform, without a GPU.

The request accepts 5 to 41 lambda windows and a production length of 0.04 ps to
200,000 ps. The capability refuses more than 10 replicas, more than 41 windows and more
than 500 mapped atoms. `simulation.analyze_binding_free_energy` refuses a leg with 1
replica.
"""

from __future__ import annotations

import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from alambiq.capabilities.investigate.simulation_analyze_binding_free_energy import (
    Edge,
)
from alambiq.capabilities.investigate.simulation_run_relative_free_energy_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class AtomPair:
    """One mapped atom of ligand A and its counterpart in ligand B.

    a, b:
        Zero-based atom indices in the SDF of ligand A and of ligand B.
    """

    a: int
    b: int


@dataclass(frozen=True, slots=True)
class RelativeRequest:
    """The leg: edge, ligands, mapping, protein and sampling settings.

    phase:
        The leg: `bound` (with the protein) or `solvent` (the ligands in water alone).
    edge:
        The directed edge: its `id` and the names `a` and `b` of ligand A and ligand B.
    ligand_a, ligand_b:
        Paths to the SDF of each ligand: one connected molecule of at most 500 atoms,
        neutral, with explicit hydrogens, specified stereochemistry, a 3D conformer, and
        AshGC charges in `atom.dprop.PartialCharge` that sum to zero.
    atom_mapping:
        The mapped atoms, 1 to 500 pairs. Mapped atoms must retain element, isotope,
        aromaticity and charge; stereocenters and double bonds must retain their
        orientation; every ring touched must be mapped whole; and the core must be
        connected.
    protein:
        For the bound leg, the path to the protein PDB with hydrogens; refused for
        the solvent leg.
    temperature_kelvin, pressure_bar:
        The temperature, 250 to 400 K, and the pressure, 0.1 to 100 bar.
    solvent_padding_nm, ionic_strength_molar:
        The margin of the water box (1.5 to 3 nm) and the added salt concentration (0
        to 1 M).
    replicas:
        The number of independent replicas, 1 to 10; each runs the whole set of lambda
        windows. A leg that
        `simulation.analyze_binding_free_energy` analyzes needs at least 2.
    lambda_windows:
        The number of lambda windows along the transformation, 5 to 41.
    equilibration_ps, production_ps, time_per_iteration_ps:
        The equilibration length, the production length and the length of one
        replica-exchange iteration. The iteration length must be a whole number of 2 fs
        integration steps; the equilibration and production lengths must be whole numbers of
        iterations.
    checkpoint_interval_ps, positions_interval_ps:
        The intervals at which checkpoints and positions are written, each a whole
        number of iterations and at most the production length.
    platform:
        `CPU` or `CUDA`; no fallback.
    threads:
        The number of CPU threads, 1 to 128.
    timeout_seconds:
        The longest permitted duration of the simulation, in seconds.
    keep_native_records:
        Store the native OpenFE output files, trajectories and checkpoints as one
        archive.
    """

    phase: Literal["bound", "solvent"]
    edge: Edge
    ligand_a: str
    ligand_b: str
    atom_mapping: list[AtomPair]
    protein: str | None = None
    temperature_kelvin: float = 298.15
    pressure_bar: float = 1.0
    solvent_padding_nm: float = 1.5
    ionic_strength_molar: float = 0.15
    replicas: int = 3
    lambda_windows: int = 11
    equilibration_ps: float = 1000.0
    production_ps: float = 5000.0
    time_per_iteration_ps: float = 1.0
    checkpoint_interval_ps: float = 100.0
    positions_interval_ps: float = 100.0
    platform: Literal["CPU", "CUDA"] = "CPU"
    threads: int = 8
    timeout_seconds: int = 86400
    keep_native_records: bool = True


@dataclass(frozen=True, slots=True)
class RepeatEstimate:
    """One replica of the leg."""

    id: str
    delta_g_kcal_mol: float
    mbar_error_kcal_mol: float
    minimum_adjacent_overlap: float
    minimum_effective_samples: float


@dataclass(frozen=True, slots=True)
class RelativeResult:
    """The replicas of the leg (field `repeats`) and the location of its records."""

    phase: str
    edge: Edge
    repeats: list[RepeatEstimate]
    mean_delta_g_kcal_mol: float
    replica_sd_kcal_mol: float | None
    """None with a single replica."""
    leg_record: str
    """The JSON record that `simulation.analyze_binding_free_energy` accepts as a leg."""
    provenance: str
    native_records: str | None
    structural_diagnostics: dict[str, str]
    """Per replica id, the structural NPZ of OpenFE (bound leg), in the form accepted by the analysis."""
    versions: dict[str, str]
    platform: str
    seconds: float


def _read_ligand(path: str):
    from rdkit import Chem

    rows = list(Chem.SDMolSupplier(path, removeHs=False))
    if len(rows) != 1 or rows[0] is None:
        found = f"{len(rows)} entries" if len(rows) != 1 else "one entry that cannot be read"
        raise ValueError(
            f"{path} holds {found}; each ligand SDF must contain exactly one molecule"
        )
    return rows[0]


@capability(
    "simulation.run_relative_free_energy",
    version="1.0",
    family="investigate",
    summary="Run one leg of a relative binding free-energy transformation with OpenFE",
    render=render,
    environment="openfe",
)
def run_relative_free_energy(
    request: RelativeRequest, ctx: StepContext
) -> RelativeResult:
    """Check the ligands and the mapping, run the replicas, and assemble the leg."""
    import numpy as np

    from alambiq.capabilities.investigate.science.rbfe import (
        check_settings,
        validate_ligands,
    )

    pairs = [[p.a, p.b] for p in request.atom_mapping]
    check_settings(
        replicas=request.replicas,
        lambda_windows=request.lambda_windows,
        threads=request.threads,
        timeout_seconds=request.timeout_seconds,
        temperature_kelvin=request.temperature_kelvin,
        pressure_bar=request.pressure_bar,
        ionic_strength_molar=request.ionic_strength_molar,
        solvent_padding_nm=request.solvent_padding_nm,
        equilibration_ps=request.equilibration_ps,
        production_ps=request.production_ps,
        time_per_iteration_ps=request.time_per_iteration_ps,
        checkpoint_interval_ps=request.checkpoint_interval_ps,
        positions_interval_ps=request.positions_interval_ps,
        atom_mapping=pairs,
    )
    edge = request.edge
    if edge.a == edge.b or not all(x.strip() for x in (edge.id, edge.a, edge.b)):
        raise ValueError(
            f"edge has id {edge.id!r} and ligand identities {edge.a!r} and {edge.b!r}; "
            "a named directed edge with two distinct ligand identities is required"
        )
    if request.phase == "bound" and request.protein is None:
        raise ValueError("phase is bound and protein is not set; the bound phase needs a protein")
    if request.phase == "solvent" and request.protein is not None:
        raise ValueError(
            "phase is solvent and protein is set; the solvent phase takes ligands only, "
            "so protein must be left out"
        )
    paths = {
        "ligand_a": str(ctx.resolve(request.ligand_a)),
        "ligand_b": str(ctx.resolve(request.ligand_b)),
    }
    validate_ligands(
        _read_ligand(paths["ligand_a"]), _read_ligand(paths["ligand_b"]), pairs
    )
    if request.protein is not None:
        paths["protein"] = str(ctx.resolve(request.protein))
    if request.production_ps < 1000 or request.replicas < 3:
        ctx.note(
            f"{request.replicas} replica(s) of {request.production_ps:g} ps are below the "
            "3 replicas and 1000 ps of production that this check expects for a "
            "converged free energy."
        )
    output = ctx.workdir / "rbfe"
    found = ctx.run_in_environment(
        "openfe",
        "openfe_rbfe:run",
        {
            "phase": request.phase,
            "edge": asdict(edge),
            "atom_mapping": pairs,
            "temperature_kelvin": request.temperature_kelvin,
            "pressure_bar": request.pressure_bar,
            "solvent_padding_nm": request.solvent_padding_nm,
            "ionic_strength_molar": request.ionic_strength_molar,
            "replicas": request.replicas,
            "lambda_windows": request.lambda_windows,
            "equilibration_ps": request.equilibration_ps,
            "production_ps": request.production_ps,
            "time_per_iteration_ps": request.time_per_iteration_ps,
            "checkpoint_interval_ps": request.checkpoint_interval_ps,
            "positions_interval_ps": request.positions_interval_ps,
            "platform": request.platform,
            "threads": request.threads,
            **paths,
            "output": str(output),
        },
        timeout=request.timeout_seconds,
    )
    record = found["record"]
    replicas = record["replicas"]
    values = [r["delta_g_kcal_mol"] for r in replicas]
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
    structural = {
        replica: str(ctx.keep(Path(path), "application/zip").path)
        for replica, path in found["structural"].items()
    }
    return RelativeResult(
        phase=request.phase,
        edge=edge,
        repeats=[
            RepeatEstimate(
                id=r["id"],
                delta_g_kcal_mol=r["delta_g_kcal_mol"],
                mbar_error_kcal_mol=r["mbar_error_kcal_mol"],
                minimum_adjacent_overlap=_minimum_adjacent(r["overlap_matrix"]),
                minimum_effective_samples=float(min(r["effective_samples"])),
            )
            for r in replicas
        ],
        mean_delta_g_kcal_mol=float(np.mean(values)),
        replica_sd_kcal_mol=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        leg_record=ctx.keep_json(record),
        provenance=str(ctx.keep(output / "provenance.json", "application/json").path),
        native_records=native,
        structural_diagnostics=structural,
        versions=found["method"]["versions"],
        platform=request.platform,
        seconds=found["seconds"],
    )


def _minimum_adjacent(matrix: list[list[float]]) -> float:
    import numpy as np

    array = np.asarray(matrix)
    return float(min(np.diag(array, 1).min(), np.diag(array, -1).min()))
