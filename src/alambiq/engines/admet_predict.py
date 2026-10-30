"""ADMET and physicochemical estimates for small molecules from their SMILES.

Runs inside the admet environment (ADMET-AI, Chemprop, PyTorch, RDKit). Imports the standard
library and those packages only. ADMET-AI ships its 5 models per ensemble, its endpoint table
and the DrugBank reference set inside the package, so the engine reads no other file. Every
SMILES is checked with RDKit first; the request is refused if any of them cannot be read.
"""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path
from typing import Any


def reported(row: dict[str, str], key: str) -> float | None:
    return float(row[key]) if row[key] else None


def endpoint_table() -> list[dict[str, Any]]:
    """The endpoints that ADMET-AI predicts, with the metrics of its own endpoint table."""
    import csv

    import admet_ai

    path = Path(admet_ai.__file__).parent / "resources" / "data" / "admet.csv"
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            descriptor = row["category"] == "Physicochemical"

            rows.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "category": row["category"],
                    "kind": "descriptor" if descriptor else row["task_type"],
                    "units": "" if row["units"] == "-" else row["units"],
                    "species": "" if row["species"] == "-" else row["species"],
                    "training_molecules": int(row["size"]) if row["size"] else None,
                    "reported_auroc": reported(row, "AUROC"),
                    "reported_auprc": reported(row, "AUPRC"),
                    "reported_r2": reported(row, "R^2"),
                    "reported_mae": reported(row, "MAE"),
                }
            )
    return rows


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Predict every endpoint for each molecule and return them in the order of the request."""
    import importlib.metadata as metadata

    import torch
    from rdkit import Chem

    started = time.time()
    torch.set_num_threads(int(request["threads"]))
    given = list(request["smiles"])
    canonical: list[str] = []
    unreadable = []
    for position, text in enumerate(given):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            unreadable.append(position + 1)
            continue
        canonical.append(Chem.MolToSmiles(molecule))
    if unreadable:
        shown = ", ".join(f"{p} ({given[p - 1]})" for p in unreadable[:5])
        more = f" and {len(unreadable) - 5} more" if len(unreadable) > 5 else ""
        raise ValueError(
            f"{len(unreadable)} SMILES cannot be read by RDKit: position {shown}{more}"
        )

    from admet_ai import ADMETModel

    unique = list(dict.fromkeys(canonical))
    # ADMET-AI prints progress bars and warnings; the result is the only output that matters.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        model = ADMETModel(num_workers=0)
        frame = model.predict(smiles=unique)
    suffix = "_drugbank_approved_percentile"
    values: dict[str, dict[str, float]] = {}
    percentiles: dict[str, dict[str, float]] = {}
    for key, row in frame.iterrows():
        values[key] = {
            name: float(value)
            for name, value in row.items()
            if not name.endswith(suffix)
        }
        percentiles[key] = {
            name[: -len(suffix)]: float(value)
            for name, value in row.items()
            if name.endswith(suffix)
        }
    return {
        "canonical_smiles": canonical,
        "values": [values[c] for c in canonical],
        "percentiles": [percentiles[c] for c in canonical],
        "endpoints": endpoint_table(),
        "versions": {
            name: metadata.version(name)
            for name in ("admet-ai", "chemprop", "torch", "rdkit")
        },
        "seconds": {"prediction": time.time() - started},
    }
