"""Estimate ADMET and physicochemical properties of small molecules from their SMILES.

Returns 52 estimates per molecule: 11 physicochemical descriptors computed with RDKit
and 41 absorption, distribution, metabolism, excretion and toxicity endpoints predicted by the
ADMET-AI models. The result also lists each endpoint with its kind, units and the metrics that
ADMET-AI reports for it, and, when asked, the percentile of each estimate among the approved
drugs of DrugBank.

The method:

- RDKit reads every SMILES. A request that holds a SMILES that RDKit cannot read is refused and
  names the positions.
- ADMET-AI 2.0.1 (MIT) predicts each endpoint with an ensemble of 5 Chemprop message-passing
  networks that were trained on a dataset of the Therapeutics Data Commons. The estimate is the
  mean of the 5 networks. A classification endpoint is a probability from 0 to 1, and a
  regression endpoint has the units of the endpoint table.
- The 11 descriptors (molecular weight, logP, hydrogen bond acceptors and donors, Lipinski
  count, QED, stereo centers, TPSA and the PAINS, BRENK and NIH alert counts) are computed, not
  predicted.
- With `drugbank_percentiles`, each estimate is also placed among the values of the same
  endpoint for about 2,800 approved drugs that ADMET-AI ships.
- The same molecule written twice is predicted once. The results follow the order of the
  request.

Limits.

- The estimates come from models trained on public datasets of 278 to 13,130 molecules. They
  are reliable for molecules that resemble those datasets, and the training sets are drug-like.
  A metabolite, a sugar or a small industrial chemical can lie outside, and the model gives no
  warning when it does.
- A classification output is a probability from the model, not a measured fraction, and an
  endpoint with a low reported AUROC (the table gives it) separates the classes poorly.
- Some regression endpoints are predicted with low accuracy: the reported R squared of the half
  life and the volume of distribution is negative. Read the reported metrics in the endpoint
  table before using an estimate of such an endpoint.
- Each molecule is read as one structure with the charge and stereochemistry of its SMILES. A
  salt, a mixture or an unusual protonation state is predicted as written, because ADMET-AI does
  not standardize molecules.
- The endpoint `HydrationFreeEnergy_FreeSolv` is a model trained on the FreeSolv database. It
  is an estimate with a reported mean absolute error of about 0.8 kcal/mol, not a simulation.
- The numbers come from the neural networks of ADMET-AI. A different ADMET-AI version can give
  different values; the result states the versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alambiq.capabilities.investigate.molecule_predict_admet_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

MAX_MOLECULES = 10000
MAX_SMILES_LENGTH = 1000
TIMEOUT_SECONDS = 2 * 3600


@dataclass(frozen=True, slots=True)
class AdmetRequest:
    """The molecules to estimate.

    smiles:
        1 to 10,000 molecules as SMILES, each 1 to 1,000 characters with no whitespace.
    drugbank_percentiles:
        Add the percentile of every estimate among the approved drugs of DrugBank; default
        false.
    threads:
        Threads used by PyTorch, 1 to 64; default 8.
    """

    smiles: list[str]
    drugbank_percentiles: bool = False
    threads: int = 8


@dataclass(frozen=True, slots=True)
class AdmetEndpoint:
    """One estimated quantity and what ADMET-AI reports about its models."""

    id: str
    name: str
    category: str
    """Physicochemical, Absorption, Distribution, Metabolism, Excretion or Toxicity."""
    kind: str
    """`descriptor` (computed), `classification` (a probability) or `regression`."""
    units: str
    species: str
    training_molecules: int | None
    """The molecules in the training set; none for a descriptor."""
    reported_auroc: float | None = None
    reported_auprc: float | None = None
    reported_r2: float | None = None
    reported_mae: float | None = None
    """The metrics of the ADMET-AI table; AUROC and AUPRC for a classification endpoint, R
    squared and the mean absolute error (in the units of the endpoint) for a regression one."""


@dataclass(frozen=True, slots=True)
class AdmetMolecule:
    """The estimates for one molecule of the request."""

    smiles: str
    canonical_smiles: str
    values: dict[str, float]
    """The estimate of every endpoint, by endpoint id."""
    drugbank_percentiles: dict[str, float] | None = None
    """The percentile among approved drugs, 0 to 100, by endpoint id."""


@dataclass(frozen=True, slots=True)
class AdmetResult:
    """The estimates of every molecule and the table of endpoints."""

    molecules: list[AdmetMolecule]
    endpoints: list[AdmetEndpoint]
    distinct_molecules: int
    seconds: float
    record: str
    engine_versions: dict[str, Any]


def _check(request: AdmetRequest) -> None:
    import re

    if not 1 <= len(request.smiles) <= MAX_MOLECULES:
        raise ValueError(f"smiles must hold 1 to {MAX_MOLECULES} molecules")
    for position, text in enumerate(request.smiles, start=1):
        if (
            not isinstance(text, str)
            or not text
            or len(text) > MAX_SMILES_LENGTH
            or re.search(r"\s", text)
        ):
            raise ValueError(
                f"smiles {position} must be 1 to {MAX_SMILES_LENGTH} characters with no whitespace"
            )
    if not 1 <= request.threads <= 64:
        raise ValueError("threads must be an integer from 1 to 64")


@capability(
    "molecule.predict_admet",
    version="1.0",
    family="investigate",
    summary="Estimate ADMET and physicochemical properties of small molecules",
    render=render,
    environment="admet",
)
def predict_admet(request: AdmetRequest, ctx: StepContext) -> AdmetResult:
    """Estimate the 52 ADMET-AI endpoints for each molecule."""
    _check(request)
    found = ctx.run_in_environment(
        "admet",
        "admet_predict:run",
        {"smiles": list(request.smiles), "threads": request.threads},
        timeout=TIMEOUT_SECONDS,
    )
    molecules = [
        AdmetMolecule(
            smiles=given,
            canonical_smiles=canonical,
            values=values,
            drugbank_percentiles=percentiles if request.drugbank_percentiles else None,
        )
        for given, canonical, values, percentiles in zip(
            request.smiles,
            found["canonical_smiles"],
            found["values"],
            found["percentiles"],
            strict=True,
        )
    ]
    endpoints = [AdmetEndpoint(**e) for e in found["endpoints"]]
    distinct = len(set(found["canonical_smiles"]))
    record = {
        "molecules": len(molecules),
        "distinct_molecules": distinct,
        "drugbank_percentiles": request.drugbank_percentiles,
        "threads": request.threads,
        "versions": found["versions"],
    }
    return AdmetResult(
        molecules=molecules,
        endpoints=endpoints,
        distinct_molecules=distinct,
        seconds=found["seconds"]["prediction"],
        record=ctx.keep_json(record),
        engine_versions=found["versions"],
    )
