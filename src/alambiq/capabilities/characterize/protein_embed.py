"""Represent each protein as a vector with a protein language model.

Returns one vector per protein from a protein language model, and the
per-residue vectors. The vectors describe each sequence in the space of one
model, for comparing sequences, clustering proteins or as features of a fitted
model.

Each protein is run through the model on its own (batch size 1, float32, fixed
seed). The model gives one vector per residue from its last hidden state. The
vector of a protein is the mean over its residues, excluding the start and end
tokens.

- ESM C 300M: 960 values, taken after the final normalization.
- ESM3-small: 1,536 values, taken before the final normalization.

The record stores both the pooled and the per-residue vectors, with the model,
its pinned revision, its license and the package versions.

Limits.

- Vectors from different models, or from the same model at a different
  revision, layer or pooling, are in different spaces. One record holds the
  vectors of one model and names it.
- Distances between vectors compare learned representations and have no
  evolutionary time scale.
- Mean pooling discards residue order.
- A protein longer than 1,024 residues (ESM C) or 512 residues (ESM3-small) is
  refused. So is a request with more than 64 proteins.
- The models were trained on natural proteins. Designed sequences can lie in
  regions that the training data cover poorly.
- The model runs on the CPU unless a device is requested. Results on different
  devices can differ numerically.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from alambiq.capabilities.characterize.protein_embed_report import render
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

MODELS = {
    "esmc-300m": {
        "resource": "esmc-300m",
        "name": "biohub/ESMC-300M",
        "entry": "esmc_embed:run",
        "model_dir": "model",
        "weights": "model/model.safetensors",
        "layer": "last-hidden-state-post-norm",
        "max_residues": 1024,
    },
    "esm3-small": {
        "resource": "esm3-small",
        "name": "biohub/esm3-sm-open-v1",
        "entry": "esm3_predict:embed",
        "model_dir": "",
        "weights": "data/weights/esm3_sm_open_v1.pth",
        "layer": "last-hidden-state-pre-final-norm",
        "max_residues": 512,
    },
}
MAX_PROTEINS = 64


@dataclass(frozen=True, slots=True)
class EmbedRequest:
    """The proteins to embed, the model, and where it runs.

    proteins:
        A path to a protein FASTA of 1 to 64 proteins. Each protein has 1 to
        1,024 residues (512 for ESM3-small) and a unique identifier of letters,
        digits, underscore, dot and hyphen. Residues are the 20 amino acids and
        B, X, Z, U and O.
    model:
        `esmc-300m` or `esm3-small`; default `esmc-300m`. The result records
        the model.
    device:
        `cpu`, `mps` or `cuda`; default `cpu`. An unavailable device is an
        error.
    threads:
        Host CPU threads used by the engine on every device, 1 to 32; default
        4.
    """

    proteins: str
    model: Literal["esmc-300m", "esm3-small"] = "esmc-300m"
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    threads: int = 4


@dataclass(frozen=True, slots=True)
class Embedded:
    """One embedded protein."""

    id: str
    length: int
    sequence_sha256: str


@dataclass(frozen=True, slots=True)
class EmbedResult:
    """The model, revision, layer and pooling that define the vector space, and the files with the vectors."""

    model: str
    revision: str
    license: str
    dimension: int
    layer: str
    pooling: str
    proteins: list[Embedded]
    record: str
    """The pooled vectors with the model name, revision, layer and pooling, as other capabilities read them."""
    residue_embeddings: str
    """A NumPy archive with one array of per-residue vectors per protein."""
    device: str
    engine_versions: dict[str, Any]


@capability(
    "protein.embed",
    version="1.0",
    family="characterize",
    summary="Represent each protein as a vector with a protein language model",
    render=render,
    environment="esm",
    resources=("esmc-300m", "esm3-small"),
    resources_for=lambda request: (MODELS[request.model]["resource"],),
)
def embed(request: EmbedRequest, ctx: StepContext) -> EmbedResult:
    """Embed every protein and store the vectors with the model name, revision, layer and pooling."""
    from alambiq.capabilities.characterize.science.enzyme import parse_proteins

    model = MODELS[request.model]
    proteins = parse_proteins(ctx.read_text(request.proteins), "proteins")
    if len(proteins) > MAX_PROTEINS:
        raise ValueError(f"At most {MAX_PROTEINS} proteins are embedded in one step")
    alphabet = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    limit = model["max_residues"]
    for name, sequence in proteins.items():
        if len(sequence) > limit or not set(sequence) <= alphabet:
            raise ValueError(
                f"{name}: expected 1–{limit} ungapped protein residues for {request.model}"
            )
    if not 1 <= request.threads <= 32:
        raise ValueError("threads must be an integer from 1 to 32")

    root = resources.verify(model["resource"])
    pinned = resources.spec(model["resource"])
    residues = ctx.workdir / "residue-embeddings.npz"
    found = ctx.run_in_environment(
        "esm",
        model["entry"],
        {
            "sequences": proteins,
            "model_root": str(root / model["model_dir"]),
            "package_data": pinned["files"],
            "residues_output": str(residues),
            "device": request.device,
            "threads": request.threads,
        },
        timeout=4 * 3600,
    )

    weights = next(f for f in pinned["files"] if f["path"] == model["weights"])
    records = [
        {
            "id": name,
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "residue_array": f"protein_{index}",
            "pooled_vector": found["pooled"][name],
        }
        for index, (name, sequence) in enumerate(proteins.items())
    ]
    stored_residues = ctx.keep(residues, "application/x-npz")
    summary = {
        "model": model["name"],
        "revision": pinned["version"],
        "weights_sha256": weights["sha256"],
        "resource_identity": resources.identity(model["resource"]),
        "license": pinned["license"],
        "versions": found["versions"],
        "device": found["device"],
        "dtype": "float32",
        "input_tracks": ["sequence"],
        "layer": model["layer"],
        "pooling": "mean-of-residues",
        "special_tokens_excluded": True,
        "dimension": found["dimension"],
        "batch_size": 1,
        "threads": request.threads,
        "seed": 0,
        "residue_embeddings": str(stored_residues.path),
        "proteins": records,
    }
    return EmbedResult(
        model=model["name"],
        revision=pinned["version"],
        license=pinned["license"],
        dimension=found["dimension"],
        layer=summary["layer"],
        pooling=summary["pooling"],
        proteins=[
            Embedded(
                id=r["id"],
                length=len(r["sequence"]),
                sequence_sha256=r["sequence_sha256"],
            )
            for r in records
        ],
        record=ctx.keep_json(summary),
        residue_embeddings=str(stored_residues.path),
        device=found["device"],
        engine_versions=found["versions"],
    )
