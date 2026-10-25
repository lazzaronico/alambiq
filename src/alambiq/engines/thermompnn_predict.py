"""Predict the stability change of single substitutions with ThermoMPNN.

Runs inside the thermompnn environment. The standard library is all that is
imported at the top; torch and numpy load when a request runs. The caller
verifies the model and source files against their pinned record, so this module
trusts the directory it is given and reports the package versions that ran. The
request is the normalized one `science/stability.py` builds: one chain's
backbone coordinates, its sequence, and substitutions by zero-based position.
"""

from __future__ import annotations

import importlib.metadata
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PACKAGES = ("torch", "numpy")


class _Configuration(dict):
    """The upstream inference constructor uses only mapping and attribute access."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Score every substitution on the supplied backbone."""
    started = time.perf_counter()
    import numpy as np
    import torch

    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    source = Path(request["model_root"]) / "model"
    # The resource directory is read-only in spirit: compiled caches must not
    # appear beside the files whose digests were verified.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from transfer_model import TransferModel

    torch.set_num_threads(int(request["threads"]))
    torch.manual_seed(int(request["seed"]))
    np.random.seed(int(request["seed"]))
    random.seed(int(request["seed"]))
    cfg = _Configuration(
        platform=_Configuration(thermompnn_dir=str(source)),
        model=_Configuration(
            hidden_dims=[64, 32],
            subtract_mut=True,
            num_final_layers=2,
            freeze_weights=True,
            load_pretrained=True,
            lightattn=True,
        ),
    )
    model = TransferModel(cfg)
    # The checkpoint is a pickle; the caller has verified its digest against the pin.
    checkpoint = torch.load(
        source / "models/thermoMPNN_default.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(
        {
            k.removeprefix("model."): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        },
        strict=True,
    )
    model.eval()

    sequence = request["sequence"]
    coordinates = request["coordinates_angstrom"]
    pdb = [
        {
            "name": "query",
            "num_of_chains": 1,
            "seq": sequence,
            "seq_chain_A": sequence,
            "coords_chain_A": {
                f"{atom}_chain_A": [residue[i] for residue in coordinates]
                for i, atom in enumerate(("N", "CA", "C", "O"))
            },
        }
    ]
    mutations = [
        SimpleNamespace(
            position=row["position"], wildtype=row["wild_type"], mutation=row["mutant"]
        )
        for row in request["mutations"]
    ]
    load_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    with torch.inference_mode():
        predictions, _ = model(pdb, mutations)
    values = []
    for prediction in predictions:
        value = float(prediction["ddG"].item())
        if not math.isfinite(value):
            raise ValueError("ThermoMPNN returned a non-finite prediction")
        values.append(value)
    if len(values) != len(request["mutations"]):
        raise ValueError(
            f"ThermoMPNN returned {len(values)} predictions for "
            f"{len(request['mutations'])} mutations"
        )
    return {
        "ddg_kcal_mol": values,
        "versions": versions,
        "runtime": {
            "device": "cpu",
            "threads": int(request["threads"]),
            "seed": int(request["seed"]),
            "load_seconds": load_seconds,
            "inference_seconds": time.perf_counter() - inference_started,
        },
    }
