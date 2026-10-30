"""Boltz-2 structure prediction of a protein complex with an optional ligand.

Runs inside the Boltz environment. Writes Boltz's input file, runs its
command-line interface in a child process with the pinned model cache, reads the
structures and confidences it wrote, and exports the ligand's chemistry (bond
orders and charges from the SMILES, positions from each predicted structure).
PyYAML, RDKit and gemmi are imported when they are needed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

CONFIDENCE_FIELDS = (
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)


def _index(path: Path) -> int:
    match = re.search(r"_model_([0-9]+)\.", path.name)
    if match is None:
        raise ValueError(f"The name of the Boltz output file {path.name} has no model index")
    return int(match.group(1))


def _finite(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"Boltz output {field!r} is {value!r}; a finite number is required")
    return float(value)


def _command(cfg: dict[str, Any], input_path: Path, out: Path) -> list[str]:
    command = [
        sys.executable,
        "-I",
        "-c",
        "from boltz.main import cli; cli()",
        "predict",
        str(input_path),
        "--out_dir",
        str(out),
        "--cache",
        cfg["model_root"],
        "--model",
        "boltz2",
        "--output_format",
        "mmcif",
        "--sampling_steps",
        str(cfg["sampling_steps"]),
        "--diffusion_samples",
        str(cfg["diffusion_samples"]),
        "--recycling_steps",
        str(cfg["recycling_steps"]),
        "--accelerator",
        cfg["accelerator"],
        "--devices",
        "1",
        "--seed",
        str(cfg["seed"]),
        "--no_kernels",
    ]
    if any(p["msa"] for p in cfg["proteins"]):
        command += [
            "--max_msa_seqs",
            str(cfg["max_msa_seqs"]),
            "--subsample_msa",
            "--num_subsampled_msa",
            str(cfg["num_subsampled_msa"]),
        ]
    if cfg.get("affinity"):
        command += [
            "--sampling_steps_affinity",
            str(cfg["affinity_sampling_steps"]),
            "--diffusion_samples_affinity",
            str(cfg["affinity_samples"]),
        ]
        if cfg.get("affinity_mw_correction"):
            command.append("--affinity_mw_correction")
    # One to eight inputs gain nothing from data-loader workers, and in a container with the
    # default 64 MB of shared memory the workers can deadlock the whole prediction, so none are
    # started.
    command += ["--num_workers", "0"]
    return command


def _document(cfg: dict[str, Any]) -> dict[str, Any]:
    """Boltz's input: the chains, the ligand and, with affinity, the binder to score."""
    entities: list[dict[str, Any]] = [
        {
            "protein": {
                "id": p["id"],
                "sequence": p["sequence"],
                "msa": p["msa"] or "empty",
            }
        }
        for p in cfg["proteins"]
    ]
    if cfg.get("ligand"):
        entities.append({"ligand": cfg["ligand"]})
    document: dict[str, Any] = {"version": 1, "sequences": entities}
    if cfg.get("affinity"):
        document["properties"] = [{"affinity": {"binder": cfg["ligand"]["id"]}}]
    return document


AFFINITY_ATOMS = 128
AFFINITY_ADVISED_ATOMS = 56


def _affinity_atoms(smiles: str) -> int:
    """The ligand's atoms as Boltz counts them for affinity: RDKit RemoveHs keeps the rest."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit cannot read the ligand SMILES")
    count = Chem.RemoveHs(molecule).GetNumAtoms()
    if count > AFFINITY_ATOMS:
        raise ValueError(
            f"The ligand has {count} atoms; Boltz-2 estimates affinity for ligands of at "
            f"most {AFFINITY_ATOMS} atoms"
        )
    return count


def _affinity(path: Path) -> dict[str, Any]:
    """The ensemble estimate and the estimates of its two models."""
    values = json.loads(path.read_text(encoding="utf-8"))
    found: dict[str, Any] = {
        "log10_ic50_um": _finite(values["affinity_pred_value"], "affinity_pred_value"),
        "binder_probability": _finite(
            values["affinity_probability_binary"], "affinity_probability_binary"
        ),
        "models": [],
    }
    for index in (1, 2):
        if f"affinity_pred_value{index}" in values:
            found["models"].append(
                {
                    "log10_ic50_um": _finite(
                        values[f"affinity_pred_value{index}"],
                        f"affinity_pred_value{index}",
                    ),
                    "binder_probability": _finite(
                        values[f"affinity_probability_binary{index}"],
                        f"affinity_probability_binary{index}",
                    ),
                }
            )
    if not 0.0 <= found["binder_probability"] <= 1.0:
        raise ValueError("The Boltz binder probability lies outside 0 to 1")
    return found


def _check_device(accelerator: str) -> None:
    import torch

    if accelerator == "gpu" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable; set accelerator to cpu to run on the CPU"
        )
    if accelerator == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS is unavailable; set accelerator to cpu to run on the CPU"
        )


def _chemistry(
    native: Path, ligand: dict[str, Any], processed_smiles: str | None = None
) -> dict[str, Any]:
    """The ligand's graph as Boltz processed it, checked against the SMILES Boltz was to use.

    That is the input SMILES, or with affinity the SMILES after Boltz's standardization.
    """
    from rdkit import Chem

    data = native.read_bytes()
    # Boltz wrote this file moments ago in this run's private output directory.
    molecules = pickle.loads(data)
    if not isinstance(molecules, dict) or set(molecules) != {"LIG1"}:
        raise ValueError("Unexpected Boltz extra-molecule identities; only LIG1 is expected")
    molecule = Chem.Mol(molecules["LIG1"])
    Chem.SanitizeMol(molecule)
    expected = Chem.MolFromSmiles(processed_smiles or ligand["smiles"])
    canonical = Chem.MolToSmiles(molecule)
    if expected is None or canonical != Chem.MolToSmiles(Chem.RemoveHs(expected)):
        raise ValueError("The ligand processed by Boltz differs from the input SMILES")
    atoms = []
    for atom in molecule.GetAtoms():
        if not atom.HasProp("name") or not atom.GetProp("name").strip():
            raise ValueError("The ligand processed by Boltz has no atom names")
        atoms.append(
            {
                "index": atom.GetIdx(),
                "name": atom.GetProp("name"),
                "element": atom.GetSymbol(),
            }
        )
    if len({a["name"] for a in atoms}) != len(atoms):
        raise ValueError("The ligand atom names are not unique")
    # The processed conformer is a template, not the predicted pose.
    molecule.RemoveAllConformers()
    block = Chem.MolToMolBlock(molecule, forceV3000=True)
    restored = Chem.MolFromMolBlock(block, removeHs=False)
    if restored is None or Chem.MolToSmiles(restored) != canonical:
        raise ValueError("The ligand differs after serialization to a molecular block")
    return {
        "chain_id": ligand["id"],
        "component_id": "LIG1",
        "input_smiles": ligand["smiles"],
        "standardized": processed_smiles is not None
        and Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(ligand["smiles"])))
        != canonical,
        "canonical_isomeric_smiles": canonical,
        "formal_charge": Chem.GetFormalCharge(molecule),
        "atoms": atoms,
        "molblock": block,
        "processed_molecules_sha256": hashlib.sha256(data).hexdigest(),
    }


def _graph(molecule: Any) -> str:
    """The molecule's connectivity, bond orders and charges without stereochemistry."""
    from rdkit import Chem

    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def _positioned(chemistry: dict[str, Any], model: Path, output: Path) -> dict[str, Any]:
    """Place the ligand graph on a predicted structure by atom name; never by distance."""
    import gemmi
    from rdkit import Chem

    block = gemmi.cif.read_file(str(model)).sole_block()
    category = block.get_mmcif_category("_atom_site.")
    rows = [
        dict(zip(category, row, strict=True))
        for row in zip(*category.values(), strict=True)
    ]
    selected = [r for r in rows if r["label_asym_id"] == chemistry["chain_id"]]
    expected = {a["name"]: a for a in chemistry["atoms"]}
    if len(selected) != len(expected) or {r["label_atom_id"] for r in selected} != set(
        expected
    ):
        raise ValueError("The atom names of the predicted ligand differ from the atom names of the input ligand")
    molecule = Chem.MolFromMolBlock(chemistry["molblock"], removeHs=False)
    molecule.RemoveAllConformers()
    conformer = Chem.Conformer(len(expected))
    conformer.Set3D(True)
    mapping = []
    for row in selected:
        atom = expected[row["label_atom_id"]]
        if (
            row["label_comp_id"] != chemistry["component_id"]
            or row["type_symbol"].upper() != atom["element"].upper()
            or int(row.get("pdbx_PDB_model_num") or 1) != 1
            or row.get("label_alt_id") not in {None, False, "", "."}
        ):
            raise ValueError(
                "The component, element, model number or alternate location of a predicted ligand atom differs from the input ligand"
            )
        xyz = [float(row[f"Cartn_{axis}"]) for axis in "xyz"]
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError("The predicted ligand has a non-finite coordinate")
        conformer.SetAtomPosition(atom["index"], xyz)
        mapping.append({**atom, "source_atom_id": row["id"]})
    molecule.AddConformer(conformer)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    molecule.SetProp("ALAMBIQ_SOURCE_MODEL_SHA256", digest)
    molecule.SetProp(
        "ALAMBIQ_ATOM_MAP",
        json.dumps(sorted(mapping, key=lambda a: a["index"]), sort_keys=True),
    )
    with Chem.SDWriter(str(output)) as writer:
        writer.SetForceV3000(True)
        writer.write(molecule)
    restored = Chem.SDMolSupplier(str(output), removeHs=False)[0]
    if restored is None or _graph(restored) != _graph(molecule):
        raise ValueError("The bonds of the ligand differ between the prediction and the SDF file")
    # RDKit reads handedness from the coordinates, so this is the pose's, not the input's.
    perceived = Chem.MolToSmiles(restored)
    return {
        "source_model_sha256": digest,
        "atoms": sorted(mapping, key=lambda a: a["index"]),
        "perceived_smiles": perceived,
        "stereochemistry_matches_input": perceived
        == chemistry["canonical_isomeric_smiles"],
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Predict, then read and check everything Boltz wrote."""
    import yaml

    started = time.perf_counter()
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)
    _check_device(cfg["accelerator"])
    processed_smiles = None
    affinity_atoms = None
    if cfg.get("affinity"):
        # For affinity, Boltz parses the ligand after the ChEMBL standardization
        # (largest fragment, neutralization) of its own `standardize`.
        from boltz.data.parse.schema import standardize

        processed_smiles = standardize(cfg["ligand"]["smiles"])
        affinity_atoms = _affinity_atoms(processed_smiles)
    input_path = out.parent / "input.yaml"
    input_path.write_text(
        yaml.safe_dump(_document(cfg), sort_keys=False),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "BOLTZ_CACHE": cfg["model_root"],
        "NUMBA_CACHE_DIR": str(out.parent / ".numba-cache"),
    }
    if cfg["accelerator"] == "cpu":
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = str(cfg["threads"])
    if cfg["accelerator"] == "mps":
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    log = out.parent / "boltz.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            _command(cfg, input_path, out),
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=cfg["timeout"],
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Boltz failed (exit {completed.returncode}): "
            + log.read_text(encoding="utf-8", errors="replace")[-4000:]
        )
    predictions = out / "boltz_results_input" / "predictions" / "input"
    structures = sorted(predictions.glob("*_model_*.cif"), key=_index)
    confidences = sorted(predictions.glob("confidence_input_model_*.json"), key=_index)
    count = cfg["diffusion_samples"]
    if len(structures) != count or len(confidences) != count:
        raise ValueError(
            f"Boltz wrote {len(structures)} structures and {len(confidences)} confidence files; "
            f"{count} of each were requested"
        )
    chemistry = None
    if cfg.get("ligand"):
        chemistry = _chemistry(
            out / "boltz_results_input" / "processed" / "mols" / "input.pkl",
            cfg["ligand"],
            processed_smiles,
        )
    samples = []
    for position, (structure, confidence_path) in enumerate(
        zip(structures, confidences, strict=True)
    ):
        if _index(structure) != position or _index(confidence_path) != position:
            raise ValueError("Boltz sample indices are not consecutive from 0")
        confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
        sample: dict[str, Any] = {
            "index": position,
            "structure": str(structure),
            "confidence_file": str(confidence_path),
            "confidence": {
                key: _finite(confidence[key], key)
                if confidence.get(key) is not None
                else None
                for key in CONFIDENCE_FIELDS
            },
            "chains_ptm": confidence.get("chains_ptm"),
            "pair_chains_iptm": confidence.get("pair_chains_iptm"),
            "arrays": {
                kind: str(predictions / f"{kind}_input_model_{position}.npz")
                for kind in ("pae", "pde", "plddt")
                if (predictions / f"{kind}_input_model_{position}.npz").is_file()
            },
        }
        if sample["confidence"]["confidence_score"] is None:
            raise ValueError("The Boltz confidence file has no confidence_score")
        if chemistry is not None:
            sdf = out.parent / f"ligand_model_{position}.sdf"
            sample["ligand_sdf"] = str(sdf)
            sample["ligand_pose"] = _positioned(chemistry, structure, sdf)
        samples.append(sample)
    affinity = None
    if cfg.get("affinity"):
        affinity_path = predictions / "affinity_input.json"
        if not affinity_path.is_file():
            raise ValueError("Boltz wrote no affinity file although affinity was requested")
        affinity = {
            **_affinity(affinity_path),
            "file": str(affinity_path),
            "ligand_atoms": affinity_atoms,
        }
    import torch

    return {
        "samples": samples,
        "chemistry": chemistry,
        "affinity": affinity,
        "input_yaml": str(input_path),
        "log": str(log),
        "accelerator": cfg["accelerator"],
        "versions": {
            "boltz": version("boltz"),
            "torch": torch.__version__,
            "rdkit": version("rdkit"),
        },
        "seconds": time.perf_counter() - started,
    }
