"""The hydration free energy of one small molecule by alchemical simulation.

Runs inside the OpenFE environment. The standard library is imported at the top; OpenFE,
OpenMM and RDKit are loaded when a request runs. A neutral molecule given as SMILES is embedded
in 3D, given AshGC charges (the graph-network model of the OpenFF project that reproduces
AM1-BCC charges) and Sage 2.3.0 parameters, and run through the OpenFE absolute solvation
protocol: the molecule is decoupled from TIP3P water and, separately, from vacuum, each with
Hamiltonian replica exchange and MBAR. The hydration free energy of a repeat is the vacuum
decoupling free energy minus the solvent decoupling free energy.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import openfe_rbfe as rbfe
import openff_prepare_complex as ashgc

ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
MAX_HEAVY_ATOMS = 50


def molecule(smiles: str, seed: int):
    """A 3D conformer with explicit hydrogens of a neutral, closed-shell, single molecule."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError(f"{smiles!r} is not a SMILES string that RDKit can read")
    if "." in smiles or len(Chem.GetMolFrags(parsed)) != 1:
        raise ValueError(
            "smiles must be one molecule; a salt or mixture is not accepted"
        )
    if Chem.GetFormalCharge(parsed) != 0 or any(
        a.GetFormalCharge() for a in parsed.GetAtoms()
    ):
        # A zwitterion has no net charge, but its charged groups leave it outside the method.
        raise ValueError(
            "the molecule has charged atoms; the hydration free energy of ions and "
            "zwitterions is not supported"
        )
    if any(a.GetNumRadicalElectrons() for a in parsed.GetAtoms()):
        raise ValueError(
            "the molecule has unpaired electrons; only closed-shell molecules are accepted"
        )
    elements = sorted({a.GetSymbol() for a in parsed.GetAtoms()} - ELEMENTS)
    if elements:
        raise ValueError(f"elements {', '.join(elements)} are not supported")
    heavy = parsed.GetNumHeavyAtoms()
    if heavy > MAX_HEAVY_ATOMS:
        raise ValueError(
            f"the molecule has {heavy} heavy atoms; at most {MAX_HEAVY_ATOMS} are accepted"
        )
    centers = Chem.FindMolChiralCenters(
        parsed, includeUnassigned=True, useLegacyImplementation=False
    )
    if any(label == "?" for _, label in centers):
        raise ValueError(
            "the molecule has a stereocenter with no specified configuration"
        )
    mol = Chem.AddHs(parsed)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError("RDKit could not embed the molecule in 3D")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return mol, Chem.MolToSmiles(parsed)


def protocol_settings(p: dict[str, Any]) -> Any:
    from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol
    from openff.units import unit

    s = AbsoluteSolvationProtocol.default_settings()
    s.protocol_repeats = p["repeats"]
    for leg in ("solvent", "vacuum"):
        getattr(s, f"{leg}_engine_settings").compute_platform = p["platform"]
        getattr(
            s, f"{leg}_forcefield_settings"
        ).small_molecule_forcefield = "openff-2.3.0"
    s.thermo_settings.temperature = p["temperature_kelvin"] * unit.kelvin
    s.thermo_settings.pressure = p["pressure_bar"] * unit.bar
    solvent, vacuum = s.solvent_simulation_settings, s.vacuum_simulation_settings
    # OpenFE's defaults are 10 ns and 1 ns of solvent leg and 2 ns and 0.5 ns of vacuum leg.
    solvent.production_length = p["production_ns"] * unit.nanosecond
    solvent.equilibration_length = p["equilibration_ns"] * unit.nanosecond
    vacuum.production_length = p["production_ns"] / 5 * unit.nanosecond
    vacuum.equilibration_length = p["equilibration_ns"] / 2 * unit.nanosecond
    for leg in (solvent, vacuum):
        leg.real_time_analysis_interval = None
    return s


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the protocol, sample every repeat, and gather the legs."""
    import numpy as np
    import openfe
    from gufe.protocols import execute_DAG
    from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol
    from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer

    started = time.perf_counter()
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s"
    )
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    found = rbfe.versions()
    data = ashgc.check_data_files()
    mol, canonical = molecule(cfg["smiles"], cfg["seed"])
    with tempfile.TemporaryDirectory(prefix="alambiq-hydration-") as scratch:
        off, _ = ashgc.parameterize(mol, Path(scratch))
    charges = off.partial_charges.m
    mol.SetProp("atom.dprop.PartialCharge", " ".join(f"{x:.10f}" for x in charges))
    component = openfe.SmallMoleculeComponent.from_rdkit(mol, name="molecule")
    water = openfe.SolventComponent()
    settings = protocol_settings(cfg)
    protocol = AbsoluteSolvationProtocol(settings)
    dag = protocol.create(
        stateA=openfe.ChemicalSystem({"ligand": component, "solvent": water}),
        stateB=openfe.ChemicalSystem({"solvent": water}),
        mapping=None,
    )
    (output / "settings.json").write_text(settings.model_dump_json(indent=2) + "\n")
    with rbfe.qualified_precision(cfg["platform"]) as precision:
        result = execute_DAG(
            dag,
            shared_basedir=output,
            scratch_basedir=output,
            keep_shared=True,
            keep_scratch=True,
            raise_error=False,
            n_retries=0,
        )
    if not result.ok():
        failures = [
            f"{f.name}: {f.exception[0]}: {f.exception[1][0] if f.exception[1] else ''}"
            for f in result.protocol_unit_failures
        ]
        raise RuntimeError("the OpenFE protocol failed; " + "; ".join(failures)[:1500])
    if (
        cfg["platform"] == "CUDA"
        and precision["platform_selections"] < 2 * cfg["repeats"]
    ):
        raise RuntimeError(
            "each leg of each repeat must select double precision on CUDA"
        )
    gathered = protocol.gather([result])

    def kcal(value: Any) -> float:
        return float(value.to("kilocalorie_per_mole").m)

    legs: dict[str, list[dict[str, Any]]] = {}
    for leg in ("solvent", "vacuum"):
        legs[leg] = []
        for repeat_id, units in gathered.data[leg].items():
            if len(units) != 1:
                raise ValueError("each repeat must have exactly one analysis result")
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
                    "fractions": np.asarray(fractional["fractions"]).tolist(),
                    "forward_kcal_mol": [kcal(v) for v in fractional["forward_DGs"]],
                    "reverse_kcal_mol": [kcal(v) for v in fractional["reverse_DGs"]],
                }
            legs[leg].append(
                {
                    "id": str(repeat_id),
                    "decoupling_kcal_mol": kcal(r["unit_estimate"]),
                    "mbar_error_kcal_mol": kcal(r["unit_estimate_error"]),
                    "overlap_matrix": np.asarray(
                        r["unit_mbar_overlap"]["matrix"]
                    ).tolist(),
                    "effective_samples": effective,
                    "forward_reverse": fractional,
                }
            )
    seconds = time.perf_counter() - started
    record = {
        "smiles": cfg["smiles"],
        "canonical_smiles": canonical,
        "legs": legs,
        "versions": found,
        "settings": json.loads(settings.model_dump_json()),
        "small_molecule_force_field_sha256": data["force_field_sha256"],
        "charge_model_sha256": data["charge_model_sha256"],
        "charges": [float(x) for x in charges],
        "execution_precision": {
            k: v for k, v in precision.items() if k != "platform_selections"
        },
        "seconds": seconds,
    }
    (output / "record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return {
        "canonical_smiles": canonical,
        "legs": legs,
        "versions": found,
        "charges": [float(x) for x in charges],
        "settings": json.loads(settings.model_dump_json()),
        "seconds": seconds,
        "estimate_kcal_mol": kcal(gathered.get_estimate()),
        "uncertainty_kcal_mol": kcal(gathered.get_uncertainty()),
    }
