"""Embed proteins with ESM C 300M.

Runs inside the esm environment. The standard library is all that is imported at
the top; torch, numpy and the model package are imported when a request runs.
The checkpoint's files are verified by the caller against the pinned record, so
this module trusts the directory it is given and reports the package versions
that actually ran.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

DIMENSION = 960
PACKAGES = ("esm", "torch", "transformers", "numpy")
RESIDUES = re.compile(r"[ACDEFGHIKLMNPQRSTVWYXBZUO]{1,1024}")


def residue_vectors(hidden, token_ids, sequence, tokenizer):
    """Check the residue-to-token correspondence before dropping the boundary tokens."""
    import numpy as np

    expected = (
        [tokenizer.cls_token_id]
        + tokenizer.convert_tokens_to_ids(list(sequence))
        + [tokenizer.eos_token_id]
    )
    if list(token_ids) != expected or hidden.shape[0] != len(expected):
        raise ValueError("The tokenizer output does not contain exactly 1 token per residue")
    vectors = np.asarray(hidden[1:-1], dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("The residue embeddings have an unexpected shape or non-finite values")
    return vectors


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Embed every protein and write the per-residue vectors beside the result."""
    import numpy as np
    import torch
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

    sequences: dict[str, str] = request["sequences"]
    device = request["device"]
    threads = int(request["threads"])
    for name, sequence in sequences.items():
        if not RESIDUES.fullmatch(sequence):
            raise ValueError(f"{name}: found {len(sequence)} characters; expected 1–1024 ungapped protein residues")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; no implicit device fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no implicit device fallback")
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"device is {device!r}; it must be cpu, mps or cuda")

    model_root = Path(request["model_root"])
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    tokenizer = EsmcTokenizer.from_pretrained(str(model_root), local_files_only=True)
    model = EsmcForMaskedLM.from_pretrained(
        str(model_root),
        device=device,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()

    matrices: dict[str, Any] = {}
    pooled: dict[str, list[float]] = {}
    for index, (name, sequence) in enumerate(sequences.items()):
        tokens = tokenizer(sequence, return_tensors="pt")
        token_ids = tokens["input_ids"][0].tolist()
        with torch.inference_mode():
            result = model(**{k: v.to(device) for k, v in tokens.items()})
        hidden = result.last_hidden_state[0].float().cpu().numpy()
        vectors = residue_vectors(hidden, token_ids, sequence, tokenizer)
        if vectors.shape[1] != DIMENSION:
            raise ValueError(f"ESM C 300M vectors have {DIMENSION} dimensions; found {vectors.shape[1]}")
        matrices[f"protein_{index}"] = vectors
        pooled[name] = vectors.mean(axis=0).tolist()
    np.savez_compressed(request["residues_output"], **matrices)
    return {
        "pooled": pooled,
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
        "device": device,
        "dimension": DIMENSION,
    }


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    import sys

    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
