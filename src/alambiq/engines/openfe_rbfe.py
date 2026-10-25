"""One leg of an OpenFE relative binding free-energy transformation.

Runs inside the OpenFE environment. The standard library is imported at the top;
OpenFE, OpenMM and RDKit are loaded when a request runs. Two positioned neutral
ligands with precomputed AshGC charges (an SDF each), an atom mapping between them
and, for the bound phase, the protein are turned into OpenFE's relative hybrid
topology protocol with Sage 2.3.0 and ff14SB, run as independent repeats, and gathered
into one record per leg: each repeat's free-energy estimate, MBAR error, overlap
matrix, effective sample sizes per state and forward-and-reverse estimates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import openff_prepare_complex as ashgc

PACKAGES = (
    "openfe",
    "gufe",
    "openmm",
    "openmmtools",
    "pymbar",
    "openff-toolkit",
    "rdkit",
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path: Any) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGES}


@contextmanager
def qualified_precision(backend: str, *, compute: Any = None, platform: Any = None):
    """Select double precision on CUDA inside this process, around OpenFE's own choice.

    OpenFE 1.12 picks mixed precision in its platform helper. This wrapper keeps its
    platform, device and determinism selection and sets double precision before the
    minimization and sampling create their contexts. The receipt counts how many
    times the platform was selected, one per repeat.
    """
    if backend == "CPU":
        yield {"platform": "CPU", "precision": "native", "platform_selections": 0}
        return
    if backend != "CUDA":
        raise ValueError("Free-energy sampling runs on CPU or CUDA")
    if compute is None:
        from openfe.protocols.openmm_utils import omm_compute

        compute = omm_compute
    if platform is None:
        from openmm import Platform

        platform = Platform.getPlatformByName("CUDA")
    defaults = {
        key: platform.getPropertyDefaultValue(key)
        for key in platform.getPropertyNames()
        if key in {"Precision", "DeviceIndex", "DeterministicForces"}
    }
    original = compute.get_openmm_platform
    receipt = {
        "adapter": "openfe-1.12-cuda-double-v1",
        "platform": "CUDA",
        "precision": "double",
        "platform_selections": 0,
    }

    def select(*args: Any, **kwargs: Any) -> Any:
        selected = original(*args, **kwargs)
        if selected.getName() != "CUDA":
            raise ValueError("The CUDA setting needs a CUDA context")
        selected.setPropertyDefaultValue("Precision", "double")
        if selected.getPropertyDefaultValue("Precision") != "double":
            raise ValueError("CUDA double precision must be selected explicitly")
        receipt["platform_selections"] += 1
        return selected

    compute.get_openmm_platform = select
    try:
        yield receipt
    finally:
        compute.get_openmm_platform = original
        for key, value in defaults.items():
            platform.setPropertyDefaultValue(key, value)


def protocol_settings(p: dict[str, Any]) -> Any:
    from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
    from openff.units import unit

    s = RelativeHybridTopologyProtocol.default_settings()
    s.protocol_repeats = p["replicas"]
    s.engine_settings.compute_platform = p["platform"]
    s.forcefield_settings.forcefields = ["amber/ff14SB.xml", "amber/tip3p_standard.xml"]
    s.forcefield_settings.hydrogen_mass = 1.0
    s.forcefield_settings.small_molecule_forcefield = "openff-2.3.0"
    s.thermo_settings.temperature = p["temperature_kelvin"] * unit.kelvin
    s.thermo_settings.pressure = p["pressure_bar"] * unit.bar
    s.solvation_settings.solvent_padding = p["solvent_padding_nm"] * unit.nanometer
    s.integrator_settings.timestep = 2 * unit.femtosecond
    s.lambda_settings.lambda_windows = p["lambda_windows"]
    s.simulation_settings.n_replicas = p["lambda_windows"]
    s.simulation_settings.equilibration_length = p["equilibration_ps"] * unit.picosecond
    s.simulation_settings.production_length = p["production_ps"] * unit.picosecond
    s.simulation_settings.time_per_iteration = (
        p["time_per_iteration_ps"] * unit.picosecond
    )
    s.simulation_settings.real_time_analysis_interval = None
    s.output_settings.checkpoint_interval = (
        p["checkpoint_interval_ps"] * unit.picosecond
    )
    s.output_settings.positions_write_frequency = (
        p["positions_interval_ps"] * unit.picosecond
    )
    return s


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the protocol, sample every repeat, and gather the leg's record."""
    import numpy as np
    import openfe
    from gufe.protocols import execute_DAG
    from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
    from openff.units import unit
    from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer
    from rdkit import Chem

    started = time.perf_counter()
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s"
    )
    logging.getLogger("openmmtools.multistate.multistatesampler").setLevel(logging.INFO)
    count = str(cfg["threads"])
    os.environ["OPENMM_CPU_THREADS"] = count
    os.environ["OMP_NUM_THREADS"] = count
    p = cfg
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    found = versions()
    data = ashgc.check_data_files()
    molecules = []
    for key in ("ligand_a", "ligand_b"):
        rows = list(Chem.SDMolSupplier(cfg[key], removeHs=False))
        if len(rows) != 1 or rows[0] is None:
            raise ValueError("Each ligand SDF must contain exactly one molecule")
        molecules.append(rows[0])
    records = []
    for molecule in molecules:
        with tempfile.TemporaryDirectory(prefix="alambiq-rbfe-") as scratch:
            off, record = ashgc.parameterize(molecule, Path(scratch))
        supplied = np.asarray(
            molecule.GetProp("atom.dprop.PartialCharge").split(), dtype=float
        )
        if not np.allclose(supplied, off.partial_charges.m, rtol=0, atol=1e-6):
            raise ValueError(
                "The supplied partial charges are not the AshGC charges of this molecule; "
                "regenerate the ligands with structure.prepare_relative_binding_inputs"
            )
        records.append(record)
    edge = cfg["edge"]
    components = [
        openfe.SmallMoleculeComponent.from_rdkit(m, name=edge[key])
        for m, key in zip(molecules, ("a", "b"), strict=True)
    ]
    mapping = openfe.LigandAtomMapping(
        componentA=components[0],
        componentB=components[1],
        componentA_to_componentB={a: b for a, b in cfg["atom_mapping"]},
    )
    common: dict[str, Any] = {
        "solvent": openfe.SolventComponent(
            ion_concentration=p["ionic_strength_molar"] * unit.molar
        )
    }
    if p["phase"] == "bound":
        common["protein"] = openfe.ProteinComponent.from_pdb_file(cfg["protein"])
    settings = protocol_settings(p)
    protocol = RelativeHybridTopologyProtocol(settings)
    dag = protocol.create(
        stateA=openfe.ChemicalSystem({"ligand": components[0], **common}),
        stateB=openfe.ChemicalSystem({"ligand": components[1], **common}),
        mapping=mapping,
    )
    dag.to_json(output / "protocol.json")
    (output / "settings.json").write_text(settings.model_dump_json(indent=2) + "\n")
    with qualified_precision(p["platform"]) as precision:
        try:
            result = execute_DAG(
                dag,
                shared_basedir=output,
                scratch_basedir=output,
                keep_shared=True,
                keep_scratch=True,
                raise_error=False,
                n_retries=0,
            )
        finally:
            (output / "execution-precision.json").write_text(
                json.dumps(precision, indent=2, sort_keys=True) + "\n"
            )
    result.to_json(output / "native-result.json")
    if not result.ok():
        raise RuntimeError("The OpenFE protocol failed; the native records are retained")
    if p["platform"] == "CUDA" and precision["platform_selections"] != p["replicas"]:
        raise RuntimeError("Each repeat must select double precision on CUDA")
    gathered = protocol.gather([result])
    gathered.to_json(output / "gathered-result.json")

    def to_energy(value: Any) -> Any:
        return np.asarray(value.to(unit.kilocalorie_per_mole).m).tolist()

    replicas, structural = [], {}
    for repeat_id, units in gathered.data.items():
        if len(units) != 1:
            raise ValueError("Each repeat must have exactly one analysis result")
        r = units[0].outputs
        reporter = MultiStateReporter(
            r["trajectory"], checkpoint_storage=r["checkpoint"], open_mode="r"
        )
        try:
            analyzer = MultiStateSamplerAnalyzer(reporter)
            effective = analyzer.mbar.compute_effective_sample_number().tolist()
        finally:
            reporter.close()
        fractional = r["forward_and_reverse_energies"]
        if fractional is not None:
            fractional = {
                "fractions": fractional["fractions"].tolist(),
                **{
                    out: to_energy(fractional[key])
                    for out, key in [
                        ("forward_kcal_mol", "forward_DGs"),
                        ("reverse_kcal_mol", "reverse_DGs"),
                        ("forward_error_kcal_mol", "forward_dDGs"),
                        ("reverse_error_kcal_mol", "reverse_dDGs"),
                    ]
                },
            }
        replicas.append(
            {
                "id": str(repeat_id),
                "trajectory_sha256": file_digest(r["trajectory"]),
                "delta_g_kcal_mol": to_energy(r["unit_estimate"]),
                "mbar_error_kcal_mol": to_energy(r["unit_estimate_error"]),
                "overlap_matrix": r["unit_mbar_overlap"]["matrix"].tolist(),
                "effective_samples": effective,
                "forward_reverse": fractional,
            }
        )
        if r.get("structural_analysis"):
            structural[str(repeat_id)] = str(r["structural_analysis"])
    if len(replicas) != p["replicas"]:
        raise ValueError("The native results do not close the requested repeat count")
    method = {
        "versions": found,
        "settings": json.loads(settings.model_dump_json()),
        "ion_concentration_molar": p["ionic_strength_molar"],
        "execution_precision": {
            k: v for k, v in precision.items() if k != "platform_selections"
        },
        "small_molecule_force_field_sha256": data["force_field_sha256"],
        "charge_model_sha256": data["charge_model_sha256"],
    }
    record = {
        "schema_version": "0.1",
        "edge": edge,
        "phase": p["phase"],
        "temperature_kelvin": p["temperature_kelvin"],
        "method_sha256": digest(method),
        "mapping_sha256": digest(cfg["atom_mapping"]),
        "ligand_sha256": {
            "a": file_digest(cfg["ligand_a"]),
            "b": file_digest(cfg["ligand_b"]),
        },
        "protein_sha256": file_digest(cfg["protein"])
        if p["phase"] == "bound"
        else None,
        "replicas": replicas,
    }
    (output / "leg.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    provenance = {
        "method": method,
        "execution_precision": precision,
        "ligand_parameters": records,
        "randomness": "OpenFE UUID-tagged independent repeats; the integrators use automatic runtime seeds, so repeats differ at the bit level.",
        "mbar_error": "OpenFE native bootstrap MBAR error on decorrelated production energies",
        "seconds": time.perf_counter() - started,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return {
        "record": record,
        "method": method,
        "structural": structural,
        "seconds": provenance["seconds"],
    }
