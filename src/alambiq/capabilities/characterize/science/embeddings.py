"""Exploring one explicitly identified representation space, without re-running a model.

Pure functions over stored vectors. PCA and the cosine analyses use numpy, scipy and
scikit-learn; UMAP additionally needs umap-learn, which is imported only when a
UMAP layout is asked for.
"""

import hashlib
import importlib.metadata
import math
import re


def validate_embeddings(data):
    if not isinstance(data, dict):
        raise ValueError("Embedding input must be a record")
    for key in ("model", "revision", "layer", "pooling"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"Embedding metadata needs {key}")
    tracks = data.get("input_tracks")
    if (
        not isinstance(tracks, list)
        or not tracks
        or not all(isinstance(x, str) and x for x in tracks)
    ):
        raise ValueError("Declare the representation's input tracks")
    dimension = data.get("dimension")
    if type(dimension) is not int or not 1 <= dimension <= 4096:
        raise ValueError("Embedding dimension must be between 1 and 4096")
    proteins = data.get("proteins")
    if not isinstance(proteins, list) or not 1 <= len(proteins) <= 500:
        raise ValueError("Supply between 1 and 500 saved protein embedding records")
    identifiers = set()
    for protein in proteins:
        if (
            not isinstance(protein, dict)
            or not isinstance(protein.get("id"), str)
            or not protein["id"].strip()
            or any(c.isspace() for c in protein["id"])
        ):
            raise ValueError("Each embedding needs a protein identifier")
        if protein["id"] in identifiers:
            raise ValueError("Protein identifiers must be unique")
        identifiers.add(protein["id"])
        sequence = protein.get("sequence")
        if not isinstance(sequence, str) or not re.fullmatch(
            r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]{1,10000}", sequence
        ):
            raise ValueError(
                "Each embedding needs its ungapped uppercase source protein sequence"
            )
        if (
            "sequence_sha256" in protein
            and protein["sequence_sha256"]
            != hashlib.sha256(sequence.encode()).hexdigest()
        ):
            raise ValueError("Embedding sequence SHA-256 mismatch")
        for key in (
            "model",
            "revision",
            "layer",
            "pooling",
            "input_tracks",
            "dimension",
        ):
            if key in protein and protein[key] != data[key]:
                raise ValueError(
                    "Do not mix different representation spaces in one panel"
                )
        vector = protein.get("pooled_vector")
        if (
            not isinstance(vector, list)
            or len(vector) != dimension
            or not all(type(v) in (int, float) and math.isfinite(v) for v in vector)
        ):
            raise ValueError("Vector dimensions and finite numerical values must agree")
        if math.hypot(*vector) == 0 or not math.isfinite(math.hypot(*vector)):
            raise ValueError("Cosine exploration needs a finite nonzero vector norm")
    return proteins


def exploration_parameters(supplied, projection="pca"):
    allowed = {"representatives"}
    values = {"representatives": 5, **supplied}
    if projection == "umap":
        allowed |= {"seed", "n_neighbors", "min_dist", "n_epochs"}
        values = {"n_neighbors": 5, "min_dist": 0.1, "n_epochs": 500, **values}
        for key, lo, hi in [
            ("seed", 0, 2147483647),
            ("n_neighbors", 2, 499),
            ("n_epochs", 10, 2000),
        ]:
            if type(values.get(key)) is not int or not lo <= values[key] <= hi:
                raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
        if (
            type(values["min_dist"]) not in (int, float)
            or not math.isfinite(values["min_dist"])
            or not 0 <= values["min_dist"] <= 1
        ):
            raise ValueError("min_dist must be finite in [0, 1]")
    elif projection != "pca":
        raise ValueError("Unknown projection")
    if set(values) - allowed:
        raise ValueError("Unsupported parameters for this projection")
    if (
        type(values["representatives"]) is not int
        or not 1 <= values["representatives"] <= 64
    ):
        raise ValueError("representatives must be an integer from 1 to 64")
    return values


def explore(data, representatives=5, *, projection="pca", **options):
    import numpy as np
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_distances

    parameters = exploration_parameters(
        {"representatives": representatives, **options}, projection
    )
    proteins = validate_embeddings(data)
    matrix = np.asarray([p["pooled_vector"] for p in proteins], dtype=np.float64)
    n = len(proteins)
    coords = np.zeros((n, 2))
    explained = np.zeros(2)
    if n > 1 and np.any(matrix != matrix[0]):
        pca = PCA(n_components=min(2, n, matrix.shape[1]), svd_solver="full")
        projected = pca.fit_transform(matrix)
        coords[:, : projected.shape[1]] = projected
        explained[: len(pca.explained_variance_ratio_)] = pca.explained_variance_ratio_
    if not np.isfinite(coords).all() or not np.isfinite(explained).all():
        raise ValueError("PCA produced non-finite coordinates or explained variance")
    unit = matrix / np.asarray([math.hypot(*v) for v in matrix])[:, None]
    distances = cosine_distances(unit)
    distances = np.maximum(0, (distances + distances.T) / 2)
    np.fill_diagonal(distances, 0)
    nearest = []
    for i, protein in enumerate(proteins):
        candidates = sorted(
            (float(distances[i, j]), proteins[j]["id"], j) for j in range(n) if j != i
        )
        if candidates:
            distance, identifier, _ = candidates[0]
            nearest.append(
                {
                    "id": protein["id"],
                    "neighbour": identifier,
                    "cosine_distance": distance,
                }
            )
    # Start at the medoid, then maximize distance from the selected set. This
    # operates in the original representation, never the display projection.
    first = min(range(n), key=lambda i: (float(distances[i].mean()), proteins[i]["id"]))
    selected = [first]
    while len(selected) < min(representatives, n):
        remaining = [i for i in range(n) if i not in selected]
        candidate = min(
            remaining,
            key=lambda i: (-float(distances[i, selected].min()), proteins[i]["id"]),
        )
        if float(distances[candidate, selected].min()) <= 1e-12:
            break
        selected.append(candidate)
    hierarchy = (
        linkage(squareform(distances, checks=True), method="average").tolist()
        if n > 1
        else []
    )
    result = {
        "schema_version": "0.1",
        "representation": {k: v for k, v in data.items() if k != "proteins"},
        "proteins": proteins,
        "coordinates": coords.tolist(),
        "explained_variance_ratio": explained.tolist(),
        "projection": "PCA; centered pooled vectors; no feature scaling; full SVD",
        "distance": "cosine in the original pooled representation",
        "nearest_neighbours": nearest,
        "hierarchy": hierarchy,
        "hierarchy_method": "average-linkage cosine similarity hierarchy",
        "recommended_ids": [proteins[i]["id"] for i in selected],
        "selection_method": "medoid then farthest-first cosine distance; identical directions not repeated",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn")
        },
    }
    if projection == "umap":
        import umap

        if importlib.metadata.version("umap-learn") != "0.5.12":
            raise ValueError("This profile requires umap-learn 0.5.12")
        if not 2 <= parameters["n_neighbors"] < n:
            raise ValueError(
                "UMAP requires at least three proteins and n_neighbors smaller than the panel"
            )
        if np.max(distances) <= 1e-12:
            raise ValueError(
                "All cosine directions are identical; use PCA/sequence inspection instead of a UMAP layout"
            )
        effective = {
            "n_neighbors": parameters["n_neighbors"],
            "min_dist": parameters["min_dist"],
            "n_epochs": parameters["n_epochs"],
            "random_state": parameters["seed"],
            "metric": "cosine",
            "n_components": 2,
            "init": "random",
            "n_jobs": 1,
            "spread": 1.0,
        }
        projected = umap.UMAP(**effective).fit_transform(matrix)
        if projected.shape != (n, 2) or not np.isfinite(projected).all():
            raise ValueError("UMAP produced incomplete or non-finite coordinates")
        result["projection_options"] = [
            {
                "id": "pca",
                "coordinates": coords.tolist(),
                "explained_variance_ratio": explained.tolist(),
            },
            {"id": "umap", "coordinates": projected.tolist(), "parameters": effective},
        ]
        result["coordinates"] = projected.tolist()
        result["explained_variance_ratio"] = None
        result["projection"] = (
            "UMAP; cosine pooled vectors; seeded random initialization; one worker"
        )
        result["default_projection"] = "umap"
        result["versions"].update(
            {
                name: importlib.metadata.version(name)
                for name in ("umap-learn", "numba", "pynndescent")
            }
        )
    return result


IDENTITY = ("model", "revision", "layer", "pooling", "input_tracks", "dimension")


def merge_records(records):
    """Combine stored embedding records of one representation space into one panel.

    Vectors are comparable within one model, revision, layer and pooling, so
    records that differ in any of them are refused and the difference is named.
    """
    if not records:
        raise ValueError("Supply at least one embedding record")
    first = records[0]
    for index, record in enumerate(records[1:], start=2):
        differing = [key for key in IDENTITY if record.get(key) != first.get(key)]
        if differing:
            details = "; ".join(
                f"{key}: {first.get(key)!r} against {record.get(key)!r}"
                for key in differing
            )
            raise ValueError(
                f"Embedding record {index} is in a different representation space "
                f"({details}). Vectors are comparable within one model, revision, "
                "layer and pooling."
            )
    proteins = [p for record in records for p in record["proteins"]]
    names = [p["id"] for p in proteins]
    if len(names) != len(set(names)):
        raise ValueError(
            "Protein identifiers must be unique across the embedding records"
        )
    return {**{k: v for k, v in first.items() if k != "proteins"}, "proteins": proteins}
