"""Exact nearest translated-copy distance for a whole, connected protein."""

from itertools import product


def nearest_image(xyz, box):
    """Return the closest atom pair over all nonzero lattice translations (nm)."""
    import numpy as np
    from scipy.spatial import cKDTree

    xyz, box = np.asarray(xyz, dtype=float), np.asarray(box, dtype=float)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not 1 <= len(xyz) <= 50000
        or box.shape != (3, 3)
        or not np.isfinite(xyz).all()
        or not np.isfinite(box).all()
        or np.linalg.det(box) <= 0
    ):
        raise ValueError(
            "Image distances require finite coordinates and a positive-volume cell"
        )
    # A particle and its image along the shortest basis vector give an upper
    # bound. For any closer pair d = xi - xj - n B, reciprocal column norms
    # bound each integer n_k. This covers skew cells without a fixed 27-cell guess.
    best = float(np.linalg.norm(box, axis=1).min())
    reciprocal = np.linalg.inv(box)
    bounds = np.floor(
        np.ptp(xyz @ reciprocal, axis=0)
        + best * np.linalg.norm(reciprocal, axis=0)
        + 1e-7
    )
    if np.prod(2 * bounds + 1) > 10000:
        raise ValueError(
            "Cell/solute geometry requires more than 10000 lattice translations"
        )
    ranges = [range(-int(n), int(n) + 1) for n in bounds]
    direction = int(np.argmin(np.linalg.norm(box, axis=1)))
    best_shift = [int(i == direction) for i in range(3)]
    pair = [0, 0]
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    tree = cKDTree(xyz)
    for translation in product(*ranges):
        # Opposite images have identical distances with the atom pair reversed.
        if next((v for v in translation if v), 0) <= 0:
            continue
        shift = np.asarray(translation) @ box
        lower = np.linalg.norm(
            np.maximum(0, np.maximum(low - high - shift, low + shift - high))
        )
        if lower >= best:
            continue
        distances, indices = tree.query(xyz + shift, distance_upper_bound=best)
        index = int(np.argmin(distances))
        if distances[index] < best:
            best = float(distances[index])
            pair = [int(indices[index]), index]
            best_shift = list(translation)
    return {
        "minimum_distance_nm": best,
        "atom_indices": pair,
        "lattice_translation": best_shift,
    }


def protein_atoms(topology):
    """Limit the image interpretation to one covalently connected protein."""
    indices = topology.select("protein").tolist()
    if not 3 <= len(indices) <= 50000:
        raise ValueError("Protein image inspection supports 3–50000 protein atoms")
    neighbors = {i: set() for i in indices}
    for a, b in topology.bonds:
        if a.index in neighbors and b.index in neighbors:
            neighbors[a.index].add(b.index)
            neighbors[b.index].add(a.index)
    seen, pending = set(), [indices[0]]
    while pending:
        index = pending.pop()
        if index not in seen:
            seen.add(index)
            pending.extend(neighbors[index] - seen)
    if len(seen) != len(indices):
        raise ValueError(
            "Protein image inspection requires one covalently connected protein"
        )
    return indices


def inspect_frames(chunk, indices, frame_indices, offset, series, cutoff):
    """Measure before any fitting, making the selected molecule whole first."""
    import numpy as np

    positions = [i for i in frame_indices if offset <= i < offset + len(chunk)]
    if not positions:
        return []
    selected = chunk[[int(i - offset) for i in positions]].atom_slice(indices)
    selected.make_molecules_whole(inplace=True)
    bonds = np.asarray([(a.index, b.index) for a, b in selected.topology.bonds])
    # Whole-molecule reconstruction should retain ordinary protein bond lengths.
    if np.any(
        np.linalg.norm(
            selected.xyz[:, bonds[:, 0]] - selected.xyz[:, bonds[:, 1]], axis=2
        )
        > 0.25
    ):
        raise ValueError("Protein image inspection found a bond longer than 0.25 nm")
    records = []
    for i, frame_index in enumerate(positions):
        nearest = nearest_image(selected.xyz[i], selected.unitcell_vectors[i])
        nearest["atom_indices"] = [indices[j] for j in nearest["atom_indices"]]
        records.append(
            {
                **nearest,
                "frame_index": int(frame_index),
                "step": series[frame_index]["step"],
                "time_ps": series[frame_index]["time_ps"],
                "within_cutoff": nearest["minimum_distance_nm"] <= cutoff,
            }
        )
    return records
