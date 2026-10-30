"""Melting temperatures of proteins from their sequences with TemBERTure Tm.

Runs inside the thermostability environment (PyTorch, Transformers, Adapters). The standard
library is all that is imported at the top; the libraries load when a request runs. The caller
verifies the ProtBERT-BFD files and the adapters against their pinned record, so this module
trusts the directory it is given. Each of the three TemBERTure Tm replicas is a ProtBERT-BFD
encoder with an adapter and a regression head. A sequence is tokenized one residue per token,
truncated to 512 tokens, and the head returns the melting temperature in degrees Celsius.
"""

from __future__ import annotations

import importlib.metadata
import math
import time
from pathlib import Path
from typing import Any

PACKAGES = ("torch", "transformers", "adapters", "numpy")
REPLICAS = (1, 2, 3)
MAX_TOKENS = 512


def load_replica(root: Path, replica: int, device: str):
    """The ProtBERT-BFD encoder with the adapter and the regression head of one replica."""
    from adapters import BertAdapterModel

    folder = root / "adapters" / f"replica{replica}"
    model = BertAdapterModel.from_pretrained(str(root / "prot_bert_bfd"))
    name = model.load_adapter(str(folder / "AdapterBERT_adapter"), with_head=True)
    model.load_head(str(folder / "AdapterBERT_head_adapter"))
    model.set_active_adapters([name])
    model.eval()
    return model.to(device)


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Predict every sequence with each replica and return the values in request order."""
    started = time.perf_counter()
    import torch
    from transformers import BertTokenizer

    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    torch.set_num_threads(int(request["threads"]))
    device = request["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device is cuda and PyTorch sees no CUDA device")
    root = Path(request["model_root"])
    tokenizer = BertTokenizer.from_pretrained(
        str(root / "prot_bert_bfd"), do_lower_case=False
    )
    texts = [" ".join(sequence) for sequence in request["sequences"]]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    batch = int(request["batch_size"])
    values: list[list[float]] = [[] for _ in texts]
    for replica in REPLICAS:
        model = load_replica(root, replica, device)
        predicted: dict[int, float] = {}
        with torch.inference_mode():
            for start in range(0, len(order), batch):
                chosen = order[start : start + batch]
                encoded = tokenizer(
                    [texts[i] for i in chosen],
                    truncation=True,
                    padding=True,
                    max_length=MAX_TOKENS,
                    return_tensors="pt",
                ).to(device)
                output = model(**encoded).logits.reshape(-1).tolist()
                for index, value in zip(chosen, output, strict=True):
                    predicted[index] = float(value)
        for index in range(len(texts)):
            values[index].append(predicted[index])
        del model
    if any(not math.isfinite(v) for row in values for v in row):
        raise RuntimeError("a replica returned a value that is not a number")
    return {
        "replica_values": values,
        "versions": versions,
        "device": device,
        "seconds": time.perf_counter() - started,
    }
