"""Project proteins from one embedding space to two dimensions and propose
representatives.

Returns a two-dimensional layout of the proteins, the nearest neighbor of each
protein, and representatives. It reads stored vectors and runs no model. The
result shows which proteins the model places close together and a subset of
proteins far apart in the vector space to inspect or test first.

The method:

- The layout is PCA or a seeded UMAP.
- The nearest neighbor is found by cosine distance.
- A hierarchical clustering uses average linkage.
- The first representative is the medoid, the protein with the smallest mean
  cosine distance to all proteins. Each next one is the protein farthest, by
  cosine distance, from the nearest chosen protein.
- Selection stops when that distance is 1e-12 or less. Fewer representatives
  than requested can result.

Distances and representatives are computed from the full vectors.

Limits.

- The result describes the space of one model. Records that differ in model,
  revision, layer, pooling, input tracks or dimension are refused.
- A request with more than 500 proteins or more than 16 records is refused.
- A two-dimensional layout discards the variance in the other dimensions. PCA
  axes are the directions of greatest variance, and the explained fractions are
  reported.
- UMAP preserves local neighbor relations. Distances between clusters, cluster
  sizes and gaps are properties of the layout. The cosine distances of the full
  vectors are the distances of the space.
- A UMAP layout depends on `seed`, `n_neighbors`, `min_dist` and `n_epochs`. A
  PCA layout is deterministic.
- A cluster in a layout is a cluster of the vector space when the cosine
  distances within it are smaller than those between it and the other proteins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alambiq.capabilities.characterize.protein_explore_embeddings_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class ExploreRequest:
    """The stored embeddings and how to lay them out.

    embeddings:
        A path to a record stored by `protein.embed`.
    more_embeddings:
        Paths to up to 15 further `protein.embed` records, analyzed together
        with the first. This allows more than the 64 proteins of one record.
        All records agree in model, revision, layer, pooling, input tracks and
        dimension. Protein identifiers are unique across them.
    projection:
        `pca` or `umap`; default `pca`. `umap` needs umap-learn 0.5.12, which
        the `embedding-analysis` extra installs.
    representatives:
        The most representatives to propose, 1 to 64; default 5. Fewer are
        proposed when there are fewer proteins or when every remaining protein
        is within a cosine distance of 1e-12 of a chosen one.
    seed:
        UMAP only. Random seed, 0 to 2,147,483,647; default 0.
    n_neighbors:
        UMAP only. Neighbor count, 2 to 499 and smaller than the number of
        proteins; default 5.
    min_dist:
        UMAP only. Minimum distance between points in the layout, 0 to 1;
        default 0.1.
    n_epochs:
        UMAP only. Number of optimization epochs, 10 to 2,000; default 500.
    """

    embeddings: str
    more_embeddings: list[str] = field(default_factory=list)
    projection: Literal["pca", "umap"] = "pca"
    representatives: int = 5
    seed: int = 0
    n_neighbors: int = 5
    min_dist: float = 0.1
    n_epochs: int = 500


@dataclass(frozen=True, slots=True)
class Point:
    """The position of one protein in the two-dimensional layout."""

    id: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Neighbour:
    """The protein nearest to this one, by cosine distance in the full space."""

    id: str
    neighbour: str
    cosine_distance: float


@dataclass(frozen=True, slots=True)
class ExploreResult:
    """The layout, the neighbors and the representatives (`recommended`, in order of selection)."""

    model: str
    revision: str
    layer: str
    pooling: str
    proteins: int
    projection: str
    points: list[Point]
    explained_variance_ratio: list[float] | None
    nearest: list[Neighbour]
    recommended: list[str]
    record: str
    versions: dict[str, str]


@capability(
    "protein.explore_embeddings",
    version="1.0",
    family="characterize",
    summary="Project protein embeddings to two dimensions and propose representatives",
    render=render,
)
def explore_embeddings(request: ExploreRequest, ctx: StepContext) -> ExploreResult:
    """Merge the records, refuse mixed spaces, and explore the proteins."""
    from alambiq.capabilities.characterize.science.embeddings import (
        explore,
        merge_records,
        validate_embeddings,
    )

    paths = [request.embeddings, *request.more_embeddings]
    if len(paths) > 16:
        raise ValueError("At most 16 embedding records are analyzed together")
    data = merge_records([ctx.read_json(path) for path in paths])
    proteins = validate_embeddings(data)
    options: dict[str, int | float] = {}
    if request.projection == "umap":
        options = {
            "seed": request.seed,
            "n_neighbors": request.n_neighbors,
            "min_dist": request.min_dist,
            "n_epochs": request.n_epochs,
        }
        if request.n_neighbors >= len(proteins):
            raise ValueError(
                f"n_neighbors must be smaller than the panel ({len(proteins)} proteins)"
            )
    result = explore(
        data,
        request.representatives,
        projection=request.projection,
        **options,
    )
    if request.projection == "pca" and (
        result["explained_variance_ratio"]
        and sum(result["explained_variance_ratio"]) < 0.5
    ):
        ctx.note(
            "The two principal components explain "
            f"{sum(result['explained_variance_ratio']):.0%} of the variance."
        )
    if len(proteins) < 5:
        ctx.note(f"The panel has {len(proteins)} proteins, fewer than 5.")
    return ExploreResult(
        model=data["model"],
        revision=data["revision"],
        layer=data["layer"],
        pooling=data["pooling"],
        proteins=len(proteins),
        projection=result["projection"],
        points=[
            Point(id=p["id"], x=c[0], y=c[1])
            for p, c in zip(proteins, result["coordinates"], strict=True)
        ],
        explained_variance_ratio=result["explained_variance_ratio"],
        nearest=[Neighbour(**n) for n in result["nearest_neighbours"]],
        recommended=result["recommended_ids"],
        record=ctx.keep_json(result),
        versions=result["versions"],
    )
