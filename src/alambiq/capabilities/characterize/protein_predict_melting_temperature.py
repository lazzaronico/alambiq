"""Estimate the melting temperature of proteins from their sequences.

Returns, for each protein, the melting temperature in degrees Celsius that each of the three
replicas of TemBERTure Tm predicts, their mean and their spread. The estimate ranks proteins by
thermostability, for instance a panel of homologs or the variants of one protein.

The method:

- TemBERTure Tm (Rodella, Lazaridi and Lemmin, Bioinformatics Advances 2024, MIT) is a
  ProtBERT-BFD protein language model with an adapter and a regression head. Three replicas,
  trained with different random seeds, are published.
- Each sequence is tokenized one residue per token and truncated to 512 tokens, so the model
  reads at most the first 510 residues. The head returns a melting temperature in degrees
  Celsius.
- The result lists the three values, their mean and their standard deviation. A longer
  sequence is marked as truncated and carries a note.
- The model weights (ProtBERT-BFD and the adapters) are the pinned resource `temberture-tm`,
  which `alambiq resource provision temberture-tm --yes` downloads (about 1.7 GB).

Limits.

- The model learned from the melting temperatures of the Meltome Atlas, which covers 13
  organisms. The authors split their data into training, validation and test sets by
  clusters of sequences at 50 % identity. A protein that differs from those organisms, and a
  measurement under other conditions, can lie outside what the model learned.
- The authors report a Pearson correlation of 0.78 between predicted and measured melting
  temperatures on their test set, and describe the predictions as bimodal: concentrated below
  60 degrees Celsius and above 80 degrees Celsius, with few values in between. They also
  report that the replicas differ markedly. Read the spread of the three values; a large
  spread means an unreliable estimate.
- The estimate is a ranking aid. The model was trained on the sequences of whole proteomes,
  and no test of it on the variants of single proteins is part of this capability, so the
  estimate for one substitution is not validated.
- Only the first 510 residues are read. A multidomain protein or one with a signal peptide
  can melt at a temperature that its first 510 residues do not predict.
- A protein of 10 to 5,000 residues from the 20 amino acids and X is accepted, and a request
  of more than 1,000 proteins is refused.
- The estimate is of the sequence alone. Ligands, metals, the oligomeric state, the pH and
  the solvent are not inputs.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.characterize.protein_predict_melting_temperature_report import (
    render,
)
from alambiq.capabilities.characterize.science.enzyme import parse_proteins
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

RESOURCE = "temberture-tm"
MAX_PROTEINS = 1000
MAX_RESIDUES_READ = 510
RESIDUES = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]{10,5000}")
TIMEOUT_SECONDS = 4 * 3600
LARGE_SPREAD_C = 15.0


@dataclass(frozen=True, slots=True)
class MeltingRequest:
    """The proteins and the compute settings.

    proteins:
        A path to a FASTA of 1 to 1,000 proteins, each 10 to 5,000 residues from the 20 amino
        acids and X, with unique identifiers.
    batch_size:
        Sequences per forward pass, 1 to 256; default 16.
    device:
        `cpu` (default) or `cuda`; no fallback.
    threads:
        CPU threads used by PyTorch, 1 to 64; default 8.
    """

    proteins: str
    batch_size: int = 16
    device: Literal["cpu", "cuda"] = "cpu"
    threads: int = 8


@dataclass(frozen=True, slots=True)
class MeltingPrediction:
    """The melting temperature of one protein."""

    id: str
    length: int
    truncated: bool
    """True when the protein is longer than the 510 residues that the model reads."""
    melting_temperature_c: float
    """The mean of the three replicas."""
    replica_values_c: list[float]
    spread_c: float
    """The standard deviation of the three replica values."""


@dataclass(frozen=True, slots=True)
class MeltingResult:
    """The predictions in the order of the FASTA, and what ran."""

    proteins: list[MeltingPrediction]
    model: str
    resource_identity: str
    record: str
    device: str
    seconds: float
    engine_versions: dict[str, str]


def _check(request: MeltingRequest) -> None:
    for name, value, low, high in (
        ("batch_size", request.batch_size, 1, 256),
        ("threads", request.threads, 1, 64),
    ):
        if not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")


@capability(
    "protein.predict_melting_temperature",
    version="1.0",
    family="characterize",
    summary="Estimate melting temperatures of proteins from their sequences",
    render=render,
    environment="thermostability",
    resources=(RESOURCE,),
)
def predict_melting_temperature(
    request: MeltingRequest, ctx: StepContext
) -> MeltingResult:
    """Predict each protein with the three replicas and gather their values."""
    _check(request)
    proteins = parse_proteins(ctx.read_text(request.proteins), "proteins")
    if len(proteins) > MAX_PROTEINS:
        raise ValueError(
            f"proteins: the file holds {len(proteins)} proteins; at most {MAX_PROTEINS} "
            "are accepted"
        )
    for name, sequence in proteins.items():
        if not RESIDUES.fullmatch(sequence):
            raise ValueError(
                f"proteins: {name} must have 10 to 5,000 residues from the 20 amino "
                "acids and X"
            )
    root = resources.verify(RESOURCE)
    found = ctx.run_in_environment(
        "thermostability",
        "thermostability_predict:run",
        {
            "sequences": list(proteins.values()),
            "model_root": str(root),
            "batch_size": request.batch_size,
            "device": request.device,
            "threads": request.threads,
        },
        timeout=TIMEOUT_SECONDS,
    )
    predictions = []
    for (name, sequence), values in zip(
        proteins.items(), found["replica_values"], strict=True
    ):
        predictions.append(
            MeltingPrediction(
                id=name,
                length=len(sequence),
                truncated=len(sequence) > MAX_RESIDUES_READ,
                melting_temperature_c=statistics.fmean(values),
                replica_values_c=list(values),
                spread_c=statistics.stdev(values),
            )
        )
    truncated = sum(p.truncated for p in predictions)
    if truncated:
        ctx.note(
            f"{truncated} protein(s) are longer than {MAX_RESIDUES_READ} residues; the "
            f"model read only their first {MAX_RESIDUES_READ} residues."
        )
    wide = sum(p.spread_c > LARGE_SPREAD_C for p in predictions)
    if wide:
        ctx.note(
            f"{wide} protein(s) have a spread of more than {LARGE_SPREAD_C:g} degrees "
            "Celsius between the replicas; the estimate of such a protein is unreliable."
        )
    identity = resources.identity(RESOURCE)
    record = {
        "model": "TemBERTure Tm",
        "resource_identity": identity,
        "source_revision": resources.spec(RESOURCE)["version"],
        "proteins": len(predictions),
        "device": request.device,
        "max_tokens": 512,
        "versions": found["versions"],
        "predictions": [
            {
                "id": p.id,
                "length": p.length,
                "truncated": p.truncated,
                "replica_values_c": p.replica_values_c,
            }
            for p in predictions
        ],
    }
    if not all(math.isfinite(p.melting_temperature_c) for p in predictions):
        raise RuntimeError("a prediction is not a number")
    return MeltingResult(
        proteins=predictions,
        model="TemBERTure Tm (3 replicas)",
        resource_identity=identity,
        record=ctx.keep_json(record),
        device=request.device,
        seconds=found["seconds"],
        engine_versions=found["versions"],
    )
