"""Predict how single substitutions change a protein's folding stability.

Returns, for each requested substitution, a predicted change in folding free
energy in kcal/mol, mutant minus wild type. A positive value is destabilizing.
The score screens for substitutions that disrupt the fold of the selected
chain. A substitution to the wild-type residue is a zero-change control and
returns 0 kcal/mol.

The method:

- ThermoMPNN, a network trained on measured stability changes, reads the
  backbone of one chain.
- The structure is mapped by author residue number, with an optional insertion
  letter.
- The chain matches the supplied construct sequence residue for residue.
- Every residue has exactly one N, CA, C and O atom. Each atom has an occupancy
  above 0 and no alternate-location label.
- The chain has no break. A break is a distance from the C of one residue to
  the N of the next outside 1.0 to 1.8 angstrom.

Limits.

- Temperature, pH, ligands, oligomeric state and the function of the protein
  are not inputs.
- Only the backbone of the selected chain is read. Other chains, waters and
  side-chain coordinates are excluded. In mmCIF, non-polymer and branched
  residues of the selected chain are also excluded. The result stores the list
  of excluded items.
- In PDB, a non-water residue of the selected chain that is not a standard
  amino acid is refused.
- The capability runs on the CPU.
- A request with more than 20,000 substitutions, or a construct sequence of
  more than 1,024 residues, is refused.
- A substitution predicted as stabilizing can reduce function.

Evaluation. The repository benchmark covers one enzyme, a beta-glucosidase
(BglB), and one prepared structure (`environments/thermompnn/BGLB_BENCHMARK.md`).

- 128 substitutions were scored. 78 had a measured functional thermal midpoint,
  derived from the activity after a 30 min heat challenge. It is not an
  equilibrium folding free energy.
- The Spearman correlation of the negative predicted change with the midpoint
  was 0.161 (95% position-bootstrap interval -0.098 to 0.379). The criterion
  was a lower end above 0.
- Of the 10 highest-ranked of the 78 variants, 3 had a midpoint at least 1 °C
  above wild type. Random selection gives 1.28. The one-sided hypergeometric
  probability of 3 or more is 0.113. The criterion was at most 0.05.
- Both criteria were fixed before any prediction. Neither was met.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from alambiq.capabilities.characterize.protein_predict_stability_change_report import (
    render,
)
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

RESOURCE = "thermompnn"
SIGN = "Folding free-energy change: mutant minus wild type; positive destabilizing"


@dataclass(frozen=True, slots=True)
class Substitution:
    """One substitution, named by the residue numbering of the structure.

    residue:
        The author residue number, with an insertion letter if the structure
        has one: `55`, `-3` or `55A`.
    wild_type:
        The residue of the structure at that position, as one letter. It is
        checked against the structure.
    mutant:
        The residue to substitute, one of the 20 standard amino acids, as one
        letter. The same letter as `wild_type` is a zero-change control.
    """

    residue: str
    wild_type: str
    mutant: str


@dataclass(frozen=True, slots=True)
class StabilityRequest:
    """The structure, the construct sequence, the chain and the substitutions.

    structure:
        A path to a PDB or mmCIF file.
    sequence:
        A path to a FASTA with the complete construct sequence of the chain,
        2 to 1,024 residues from the 20 standard amino acids. The observed
        residues of the chain equal it exactly, so a structure with missing
        residues is refused.
    chain_id:
        The author chain to read (`auth_asym_id`). `protein.inspect` reports
        chains by `label_asym_id`, which can differ.
    mutations:
        1 to 20,000 distinct substitutions.
    structure_format:
        `pdb` or `mmcif`. By default the file extension decides: `.pdb` and
        `.ent` are PDB, `.cif` and `.mmcif` are mmCIF. Any other extension needs
        this field.
    model_index:
        The model of an ensemble to read, numbered from 0, 0 to 999; default
        0. `protein.inspect` numbers models from 1 in `model_number`.
    seed, threads:
        `seed`, 0 to 2,147,483,647 (default 0), seeds the random number
        generators of the engine. `threads` is the number of host CPU threads
        the engine uses, 1 to 32 (default 4). The engine runs on the CPU, so the
        request has no `device` or `accelerator` field.
    """

    structure: str
    sequence: str
    chain_id: str
    mutations: list[Substitution]
    structure_format: Literal["pdb", "mmcif"] | None = None
    model_index: int = 0
    seed: int = 0
    threads: int = 4


@dataclass(frozen=True, slots=True)
class Prediction:
    """The predicted change for one substitution."""

    residue: str
    residue_name: str
    wild_type: str
    mutant: str
    ddg_kcal_mol: float
    """Mutant minus wild type: positive means the substitution destabilizes."""


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Every prediction and the inputs that the model used."""

    chain_id: str
    length: int
    predictions: list[Prediction]
    destabilizing: int
    stabilizing: int
    unchanged: int
    sign_convention: str
    record: str
    model_context: dict[str, Any]
    model: str
    revision: str
    license: str
    engine_versions: dict[str, Any]


@capability(
    "protein.predict_stability_change",
    version="1.0",
    family="characterize",
    summary="Predict how single substitutions change folding stability",
    render=render,
    environment="thermompnn",
    resources=(RESOURCE,),
)
def predict_stability_change(
    request: StabilityRequest, ctx: StepContext
) -> StabilityResult:
    """Map the structure, score each substitution and record what was left out."""
    from alambiq.capabilities.characterize.science.stability import (
        normalize_structure,
        parse_residue,
    )
    from alambiq.capabilities.shared.sequences import parse_fasta

    path = ctx.resolve(request.structure)
    fmt = request.structure_format or (
        "pdb" if path.suffix.lower() in (".pdb", ".ent") else "mmcif"
    )
    if request.structure_format is None and path.suffix.lower() not in (
        ".pdb",
        ".ent",
        ".cif",
        ".mmcif",
    ):
        raise ValueError(
            f"structure_format is not set and the extension of {Path(request.structure).name} "
            "does not identify the format; set structure_format to pdb or mmcif"
        )
    for name, value, low, high in (
        ("model_index", request.model_index, 0, 999),
        ("seed", request.seed, 0, 2**31 - 1),
        ("threads", request.threads, 1, 32),
    ):
        if not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    _, sequence = parse_fasta(ctx.read_text(request.sequence))
    rows = []
    for item in request.mutations:
        number, insertion = parse_residue(item.residue)
        rows.append(
            {
                "residue_id": number,
                "insertion_code": insertion,
                "wild_type": item.wild_type,
                "mutant": item.mutant,
            }
        )
    structure_text = path.read_text()
    normalized, omitted = normalize_structure(
        structure_text, fmt, request.chain_id, request.model_index, sequence, rows
    )

    root = resources.verify(RESOURCE)
    pinned = resources.spec(RESOURCE)
    found = ctx.run_in_environment(
        "thermompnn",
        "thermompnn_predict:run",
        {
            **normalized,
            "model_root": str(root),
            "seed": request.seed,
            "threads": request.threads,
        },
        timeout=3600,
    )

    predictions = []
    for mutation, value in zip(
        normalized["mutations"], found["ddg_kcal_mol"], strict=True
    ):
        source = normalized["residues"][mutation["position"]]
        predictions.append(
            Prediction(
                residue=source["residue_id"] + source["insertion_code"],
                residue_name=source["residue_name"],
                wild_type=mutation["wild_type"],
                mutant=mutation["mutant"],
                ddg_kcal_mol=value,
            )
        )
    destabilizing = sum(p.ddg_kcal_mol > 0 for p in predictions)
    stabilizing = sum(p.ddg_kcal_mol < 0 for p in predictions)
    unchanged = len(predictions) - destabilizing - stabilizing
    if omitted["other_chains"] or omitted["selected_chain_omitted_residues"]:
        ctx.note(
            "The model read only the backbone of chain "
            f"{request.chain_id}. Other chains ({', '.join(omitted['other_chains']) or 'none'}) "
            f"and {len(omitted['selected_chain_omitted_residues'])} ligand or non-polymer "
            "residues in this chain are excluded from its input."
        )
    record = {
        "schema_version": "0.1",
        "model": "ThermoMPNN-default",
        "source_revision": pinned["version"],
        "resource_identity": resources.identity(RESOURCE),
        "license": pinned["license"],
        "sign_convention": SIGN,
        "prediction_scope": (
            "Single substitutions on a supplied complete monomer backbone; assay "
            "conditions, ligand effects, activity and kinetic rates are not inputs of this model"
        ),
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "source": {
            "file": Path(request.structure).name,
            "sha256": hashlib.sha256(structure_text.encode()).hexdigest(),
            "format": fmt,
            "chain_id": request.chain_id,
            "model_index": request.model_index,
        },
        "rows": [
            {
                **mutation,
                "source_residue": normalized["residues"][mutation["position"]],
                "predicted_ddg_kcal_mol": value,
            }
            for mutation, value in zip(
                normalized["mutations"], found["ddg_kcal_mol"], strict=True
            )
        ],
        "model_context": omitted,
        "runtime": found["runtime"] | {"versions": found["versions"]},
    }
    return StabilityResult(
        chain_id=request.chain_id,
        length=len(sequence),
        predictions=predictions,
        destabilizing=destabilizing,
        stabilizing=stabilizing,
        unchanged=unchanged,
        sign_convention=SIGN,
        record=ctx.keep_json(record),
        model_context=omitted,
        model="ThermoMPNN-default",
        revision=pinned["version"],
        license=pinned["license"],
        engine_versions=found["versions"],
    )
