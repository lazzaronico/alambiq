"""Predict InterPro regions, function keywords and secondary structure for one protein with ESM3-small.

Returns three predictions from the sequence alone (ESM3 calls them tracks):

- InterPro regions, each a family or domain label with a start and end.
- Function keywords with their regions.
- An 8-class secondary-structure label for every residue (G, H, I, T, E, B, S,
  C).

The predictions are hypotheses to check against structure. A Pfam-derived label
can be compared with a search by `protein.match_profiles`.

The method:

- The output layers of ESM3-small give the three tracks.
- Regions are decoded from the model output at fixed thresholds. The `decoding`
  field of the result lists them: `p_none_threshold` 0.05, `interpro_threshold`
  0.1, `keyword_threshold` 0.5, `minimum_span` 5, `merge_gap` 3.
- Region positions are one-based and inclusive.
- The full probability arrays are stored, so other thresholds can be applied.

Limits.

- The reported maximum probabilities are the model's own, with an uncalibrated
  confidence.
- A predicted InterPro region can have a wrong label, start or end.
- The regions come from several InterPro member databases. A search with
  `protein.match_profiles` covers Pfam 38.2 or supplied profiles.
- Other decoding thresholds give other regions.
- A protein of more than 512 residues is refused.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from alambiq.capabilities.characterize.protein_predict_tracks_report import render
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

RESOURCE = "esm3-small"
MODEL = "biohub/esm3-sm-open-v1"
MAX_RESIDUES = 512
DECODING = {
    "function_temperature": 1.0,
    "function_top_p": 1.0,
    "p_none_threshold": 0.05,
    "interpro_threshold": 0.1,
    "keyword_threshold": 0.5,
    "minimum_span": 5,
    "merge_gap": 3,
    "secondary_structure": "argmax-over-eight-residue-classes",
}


@dataclass(frozen=True, slots=True)
class TracksRequest:
    """The protein and where the model runs.

    protein:
        A path to a FASTA with one protein of 1 to 512 residues.
    device:
        `cpu`, `mps` or `cuda`; default `cpu`. An unavailable device is an
        error.
    threads:
        Host CPU threads used by the engine on every device, 1 to 32; default
        4.
    """

    protein: str
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    threads: int = 4


@dataclass(frozen=True, slots=True)
class Region:
    """One predicted region."""

    kind: Literal["interpro", "keyword"]
    label: str
    description: str | None
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TracksResult:
    """The predicted regions and secondary structure, and the model that made them."""

    protein_id: str
    length: int
    sequence_sha256: str
    secondary_structure: str
    """One letter per residue from G, H, I, T, E, B, S, C."""
    secondary_counts: dict[str, int]
    regions: list[Region]
    record: str
    arrays: str
    """A NumPy archive of the probability arrays that the labels were decoded from."""
    model: str
    revision: str
    license: str
    decoding: dict[str, Any]
    device: str
    engine_versions: dict[str, Any]


@capability(
    "protein.predict_tracks",
    version="1.0",
    family="characterize",
    summary="Predict secondary structure and function regions for one protein",
    render=render,
    environment="esm",
    resources=(RESOURCE,),
)
def predict_tracks(request: TracksRequest, ctx: StepContext) -> TracksResult:
    """Run the model on the protein and decode its outputs."""
    from alambiq.capabilities.shared.sequences import parse_fasta

    name, sequence = parse_fasta(ctx.read_text(request.protein))
    if len(sequence) > MAX_RESIDUES:
        raise ValueError(
            f"protein: the sequence has {len(sequence)} residues; 1–{MAX_RESIDUES} "
            "protein residues are required"
        )
    if not 1 <= request.threads <= 32:
        raise ValueError("threads must be an integer from 1 to 32")

    root = resources.verify(RESOURCE)
    pinned = resources.spec(RESOURCE)
    arrays = ctx.workdir / "track-arrays.npz"
    found = ctx.run_in_environment(
        "esm",
        "esm3_predict:annotate",
        {
            "sequence": sequence,
            "model_root": str(root),
            "package_data": pinned["files"],
            "arrays_output": str(arrays),
            "device": request.device,
            "threads": request.threads,
        },
        timeout=3600,
    )
    labels = found["secondary_structure"]
    if len(labels) != len(sequence):
        raise ValueError(
            "The secondary-structure track has "
            f"{len(labels)} labels for a sequence of {len(sequence)} residues; the "
            "numbers must be equal"
        )
    counts = {letter: labels.count(letter) for letter in "GHITEBSC" if letter in labels}
    regions = [
        Region(
            kind=item["kind"],
            label=item["label"],
            description=item.get("description"),
            start=item["start"],
            end=item["end"],
        )
        for item in found["annotations"]
    ]
    if not regions:
        ctx.note(
            "The model predicted no InterPro or keyword regions at the decoding "
            "thresholds."
        )
    stored = ctx.keep(arrays, "application/x-npz")
    record = {
        "schema_version": "0.1",
        "identifier": name,
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "coordinate_system": "one-based-inclusive",
        "model": MODEL,
        "revision": pinned["version"],
        "resource_identity": resources.identity(RESOURCE),
        "license": pinned["license"],
        "input_tracks": ["sequence"],
        "origin": "model-predicted",
        "versions": found["versions"],
        "device": found["device"],
        "decoding": DECODING,
        "annotations": found["annotations"],
        "secondary_structure": {
            "alphabet": "GHITEBSC",
            "labels": labels,
            "max_probability": found["secondary_structure_max_probability"],
        },
        "array_columns": found["array_columns"],
        "arrays": str(stored.path),
    }
    return TracksResult(
        protein_id=name,
        length=len(sequence),
        sequence_sha256=record["sequence_sha256"],
        secondary_structure=labels,
        secondary_counts=counts,
        regions=regions,
        record=ctx.keep_json(record),
        arrays=str(stored.path),
        model=MODEL,
        revision=pinned["version"],
        license=pinned["license"],
        decoding=DECODING,
        device=found["device"],
        engine_versions=found["versions"],
    )
