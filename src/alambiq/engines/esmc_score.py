"""Score every substitution at chosen positions with ESM C 300M, by masked marginals.

Runs inside the esm environment. For each position the residue is replaced by the mask
token, the model is run on the whole sequence, and the log-probability it gives each of
the twenty canonical residues at that position is recorded. The log-likelihood ratio of a
substitution is then the mutant's log-probability minus the wild type's at the same masked
position, which is the masked-marginal score of Meier and colleagues (2021). The standard
library is imported at the top; torch and the model package are imported when a request
runs.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

PACKAGES = ("esm", "torch", "transformers", "numpy")
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
RESIDUES = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{1,1024}")


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Mask each requested position in turn and return the canonical log-probabilities."""
    import torch
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

    sequence: str = request["sequence"]
    positions: list[int] = request["positions"]
    device = request["device"]
    if not RESIDUES.fullmatch(sequence):
        raise ValueError(
            f"found {len(sequence)} characters; expected 1–1024 ungapped canonical protein residues"
        )
    if any(not 0 <= p < len(sequence) for p in positions):
        raise ValueError(
            f"A position lies outside the sequence; positions run from 0 to {len(sequence) - 1}"
        )
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; no implicit device fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no implicit device fallback")
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"device is {device!r}; it must be cpu, mps or cuda")

    model_root = Path(request["model_root"])
    torch.set_num_threads(int(request["threads"]))
    torch.manual_seed(0)
    tokenizer = EsmcTokenizer.from_pretrained(str(model_root), local_files_only=True)
    model = EsmcForMaskedLM.from_pretrained(
        str(model_root),
        device=device,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()

    encoded = tokenizer(sequence, return_tensors="pt")
    ids = encoded["input_ids"][0]
    expected = (
        [tokenizer.cls_token_id]
        + tokenizer.convert_tokens_to_ids(list(sequence))
        + [tokenizer.eos_token_id]
    )
    if ids.tolist() != expected:
        raise ValueError("The tokenizer output does not contain exactly 1 token per residue")
    canonical_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(CANONICAL)), device=device
    )
    mask_id = tokenizer.mask_token_id
    batch = int(request["batch_size"])
    log_probs: dict[str, list[float]] = {}
    for start in range(0, len(positions), batch):
        chunk = positions[start : start + batch]
        masked = ids.unsqueeze(0).repeat(len(chunk), 1)
        for row, position in enumerate(chunk):
            masked[row, position + 1] = mask_id
        with torch.inference_mode():
            logits = model(
                input_ids=masked.to(device),
                attention_mask=torch.ones_like(masked).to(device),
            ).logits.float()
        for row, position in enumerate(chunk):
            scores = torch.log_softmax(logits[row, position + 1], dim=-1)
            values = scores[canonical_ids].cpu().tolist()
            if not all(v == v and abs(v) != float("inf") for v in values):
                raise ValueError("The model output contains non-finite values")
            log_probs[str(position)] = values
    return {
        "alphabet": CANONICAL,
        "log_probs": log_probs,
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
        "device": device,
    }


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    import sys

    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
