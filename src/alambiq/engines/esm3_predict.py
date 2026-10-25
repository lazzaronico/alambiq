"""Run ESM3-small: embeddings, predicted tracks, and masked completion.

Runs inside the esm environment. The standard library is all that is imported at
the top; torch, numpy and the model package load when a request runs. The caller
verifies the checkpoint against its pinned record before calling; this module
checks the five data files the ESM package ships and the function decoder reads
against the digests it is given, because those live in the environment, not in
the checkpoint, and reports the package versions that ran.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import random
from pathlib import Path
from typing import Any

PACKAGES = ("esm", "torch", "transformers", "numpy")
SS8 = "GHITEBSC"
PACKAGE_DATA = {
    "entry_list_safety_29026.list",
    "ParentChildTreeFile.txt",
    "keyword_vocabulary_safety_filtered_58641.txt",
    "keyword_idf_safety_filtered_58641.npy",
    "interpro_29026_to_keywords_58641.csv",
}


def _check_package_data(expected: list[dict[str, Any]], data: Path) -> None:
    for item in expected:
        name = Path(item["path"]).name
        if name not in PACKAGE_DATA:
            continue
        path = data / name
        if path.is_symlink() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"{name} in the ESM package differs from the pinned file")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(
                    f"{name} in the ESM package differs from the pinned file"
                )


def load_runtime(request: dict[str, Any]):
    device, threads = request["device"], int(request["threads"])
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"device is {device!r}; it must be cpu, mps or cuda")
    model_root = Path(request["model_root"]).resolve()
    # Upstream's local-infrastructure branch resolves checkpoint paths from the
    # working directory. This runs in a worker of its own, never the caller's process.
    os.environ.update(
        {
            "INFRA_PROVIDER": "alambiq-local",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        }
    )
    os.chdir(model_root)
    import esm
    import torch
    from esm.pretrained import load_local_model

    _check_package_data(request["package_data"], Path(esm.__file__).parent / "data")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; no implicit device fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no implicit device fallback")
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = load_local_model("esm3_sm_open_v1", device="cpu").float().to(device).eval()
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    return model, versions


def _encode(model, sequence: str):
    from esm.sdk.api import ESMProtein

    protein = model.encode(ESMProtein(sequence=sequence))
    tokenizer = model.tokenizers.sequence
    expected = (
        [tokenizer.cls_token_id]
        + tokenizer.convert_tokens_to_ids(list(sequence))
        + [tokenizer.eos_token_id]
    )
    if protein.sequence.tolist() != expected:
        raise ValueError("The sequence tokenizer changed the residue identities")
    return protein


def embed(request: dict[str, Any]) -> dict[str, Any]:
    """Residue vectors for each protein, one at a time, and their means."""
    import numpy as np
    import torch
    from esm.sdk.api import LogitsConfig

    model, versions = load_runtime(request)
    matrices: dict[str, Any] = {}
    pooled: dict[str, list[float]] = {}
    with torch.inference_mode():
        for index, (name, sequence) in enumerate(request["sequences"].items()):
            prediction = model.logits(
                _encode(model, sequence), LogitsConfig(return_embeddings=True)
            )
            vectors = prediction.embeddings[0, 1:-1].float().cpu().numpy()
            if vectors.shape != (len(sequence), 1536) or not np.isfinite(vectors).all():
                raise ValueError("The ESM3 residue embeddings have an unexpected shape or non-finite values")
            matrices[f"protein_{index}"] = vectors
            pooled[name] = vectors.mean(axis=0).tolist()
    np.savez_compressed(request["residues_output"], **matrices)
    return {
        "pooled": pooled,
        "versions": versions,
        "device": request["device"],
        "dimension": 1536,
    }


def annotate(request: dict[str, Any]) -> dict[str, Any]:
    """Predicted InterPro regions, function keywords and secondary structure."""
    import numpy as np
    import torch
    from esm.sdk.api import LogitsConfig
    from esm.utils.sampling import sample_function_logits

    sequence: str = request["sequence"]
    model, versions = load_runtime(request)
    with torch.inference_mode():
        prediction = model.logits(
            _encode(model, sequence),
            LogitsConfig(
                function=True, secondary_structure=True, return_embeddings=True
            ),
        )
        vectors = prediction.embeddings[0, 1:-1].float().cpu().numpy()
        if vectors.shape != (len(sequence), 1536) or not np.isfinite(vectors).all():
            raise ValueError("The ESM3 residue embeddings have an unexpected shape or non-finite values")
        function_logits = prediction.logits.function
        ss_logits = prediction.logits.secondary_structure
        if (
            not torch.isfinite(function_logits).all()
            or not torch.isfinite(ss_logits).all()
        ):
            raise ValueError("The model output logits contain non-finite values")
        tokens, _ = sample_function_logits(
            function_logits,
            model.tokenizers.function,
            temperature=1.0,
            top_p=1.0,
            p_none_threshold=0.05,
        )
        tokens = tokens[0]
        tokens[[0, -1]] = model.tokenizers.function.pad_token_id
        decoder = model.get_function_decoder().float().eval()
        decoded = decoder.decode(
            tokens,
            model.tokenizers.function,
            annotation_threshold=0.1,
            keywords_threshold=0.5,
            annotation_min_length=5,
            annotation_gap_merge_max=3,
        )

        def regions(items, kind):
            out = []
            for item in items:
                if not 1 <= item.start <= item.end <= len(sequence):
                    raise ValueError(
                        "A decoded annotation lies outside the sequence coordinates (1-based)"
                    )
                out.append(
                    {
                        "label": item.label,
                        "start": item.start,
                        "end": item.end,
                        "kind": kind,
                    }
                )
            return out

        annotations = regions(decoded["interpro_annotations"], "interpro")
        for annotation, native in zip(
            annotations, decoded["interpro_annotations"], strict=True
        ):
            annotation["description"] = model.tokenizers.function.format_annotation(
                native
            )
        annotations += regions(decoded["function_keywords"], "keyword")

        probabilities = torch.softmax(ss_logits[0, 1:-1, 3:11], dim=-1).cpu().numpy()
        if (
            probabilities.shape != (len(sequence), 8)
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        ):
            raise ValueError("The secondary-structure probabilities have an unexpected shape, non-finite values or rows that do not sum to 1")
        interpro = torch.sigmoid(decoded["interpro_logits"])[1:-1].cpu().numpy()
        keyword = torch.sigmoid(decoded["keyword_logits"])[1:-1].cpu().numpy()
        if not np.isfinite(interpro).all() or not np.isfinite(keyword).all():
            raise ValueError("The function-decoder probabilities contain non-finite values")
        np.savez_compressed(
            request["arrays_output"],
            interpro_probability=interpro,
            keyword_probability=keyword,
            secondary_structure_probability=probabilities,
        )
    return {
        "annotations": annotations,
        "secondary_structure": "".join(SS8[i] for i in probabilities.argmax(axis=1)),
        "secondary_structure_max_probability": probabilities.max(axis=1).tolist(),
        "array_columns": {
            "interpro_probability": list(decoder.interpro_ids),
            "keyword_probability": list(decoder.keywords_vocabulary),
            "secondary_structure_probability": list(SS8),
        },
        "versions": versions,
        "device": request["device"],
    }


def complete(request: dict[str, Any]) -> dict[str, Any]:
    """Fill the masked positions of a prompt, once per seed, and return what was written."""
    import attr
    import numpy as np
    import torch
    from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig
    from esm.utils.decoding import decode_sequence

    prompt: str = request["prompt"]
    model, versions = load_runtime(request)
    generation = {
        "track": "sequence",
        "schedule": "cosine",
        "strategy": "random",
        "temperature_annealing": True,
        "num_steps": int(request["num_steps"]),
        "temperature": float(request["temperature"]),
        "top_p": float(request["top_p"]),
    }
    tokenizer = model.tokenizers.sequence
    generated_sequences = []
    for seed in request["seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        with torch.inference_mode():
            encoded = model.encode(ESMProtein(sequence=prompt))
            expected = (
                [tokenizer.cls_token_id]
                + [
                    tokenizer.mask_token_id
                    if aa == "_"
                    else tokenizer.convert_tokens_to_ids(aa)
                    for aa in prompt
                ]
                + [tokenizer.eos_token_id]
            )
            if encoded.sequence.tolist() != expected:
                raise ValueError("The tokenizer changed the residue identities of the masked prompt")
            generated = model.generate(encoded, GenerationConfig(**generation))
            if isinstance(generated, ESMProteinError):
                generated_sequences.append(
                    {"seed": seed, "sequence": None, "native_error": str(generated)}
                )
            else:
                generated_sequences.append(
                    {
                        "seed": seed,
                        "sequence": decode_sequence(generated.sequence, tokenizer),
                    }
                )
    return {
        "generations": generated_sequences,
        "generation": attr.asdict(GenerationConfig(**generation)),
        "versions": versions,
        "device": request["device"],
    }
