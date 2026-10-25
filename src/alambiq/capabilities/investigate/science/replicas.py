"""Do independent replicas agree with a reference structure, and are they converged?

Each replica contributes the per-frame values of the same structural observables,
as `simulation.inspect_trajectory` writes them. The replicas are independent samples
of one ensemble, so the standard error of a mean is the spread of the replica means
over the square root of their number; a frame-by-frame error would treat correlated
frames as independent and understate it.
"""

from __future__ import annotations

import math
from typing import Any

from alambiq.capabilities.investigate.science.sampling import sampling_statistics

MIN_FRAMES = 40
GRID = 20


def check_criteria(
    criteria: list[dict[str, Any]],
    *,
    max_split_rhat: float,
    min_effective_samples: float,
    block_counts: list[int],
) -> None:
    """Refuse criteria that cannot be judged, before any file is read."""
    if not 1 <= len(criteria) <= 100:
        raise ValueError("Supply 1–100 criteria, one per observable judged")
    names = [c["observable"] for c in criteria]
    if len(set(names)) != len(names):
        raise ValueError("Each observable may have only one criterion")
    for c in criteria:
        if not (math.isfinite(c["precision"]) and c["precision"] > 0):
            raise ValueError(
                f"{c['observable']}: precision must be positive and finite"
            )
        tolerance = c["tolerance"]
        if tolerance is not None and not (math.isfinite(tolerance) and tolerance > 0):
            raise ValueError(
                f"{c['observable']}: tolerance must be positive and finite"
            )
    if not (math.isfinite(max_split_rhat) and 1 <= max_split_rhat <= 1.2):
        raise ValueError("max_split_rhat must be between 1 and 1.2")
    if not (
        math.isfinite(min_effective_samples) and 40 <= min_effective_samples <= 1e7
    ):
        raise ValueError("min_effective_samples must be between 40 and 10,000,000")
    if (
        not 1 <= len(block_counts) <= 6
        or len(set(block_counts)) != len(block_counts)
        or any(not 2 <= b <= 100 for b in block_counts)
    ):
        raise ValueError("Choose 1–6 distinct block counts between 2 and 100")


def _precision_pass(statistics, indices, criteria, max_split_rhat, min_effective):
    return all(
        statistics["diagnostics"][i]["status"] == "estimated"
        and statistics["replica_mean_standard_error"][i] <= criteria[name]["precision"]
        and statistics["split_rhat"][i] <= max_split_rhat
        and statistics["effective_samples"][i] >= min_effective
        and max(abs(row[i]) for row in statistics["second_minus_first_half"])
        <= 2 * criteria[name]["precision"]
        for name, i in indices.items()
    )


def compare_replicas(
    replicas: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
    *,
    max_split_rhat: float,
    min_effective_samples: float,
    block_counts: list[int],
) -> dict[str, Any]:
    """Judge convergence to the stated precision and agreement with the reference.

    Each replica is a dict with `id`, `seed` (or None), `details` (the inspection
    record), `values` (frames by observables) and `times_ps`.
    """
    import numpy as np
    from scipy.stats import t

    check_criteria(
        criteria,
        max_split_rhat=max_split_rhat,
        min_effective_samples=min_effective_samples,
        block_counts=block_counts,
    )
    if len(replicas) < 2:
        raise ValueError("Convergence between replicas needs at least two replicas")
    if len({r["id"] for r in replicas}) != len(replicas):
        raise ValueError("Replica ids must be unique")
    common = replicas[0]["details"]
    ids = common["observables"]["ids"]
    by_name = {c["observable"]: c for c in criteria}
    missing = sorted(set(by_name) - set(ids))
    if missing:
        raise ValueError(f"The inspections do not record the observables {missing}")
    wants_reference = any(c["tolerance"] is not None for c in criteria)
    reference = common["reference_values"]
    for r in replicas:
        d = r["details"]
        if d["ca_fluctuations"]["atoms"] != common["ca_fluctuations"]["atoms"] or any(
            d["observables"][k] != common["observables"][k]
            for k in ("ids", "units", "selections")
        ):
            raise ValueError(
                "The replicas were inspected with different protein atoms or observable definitions"
            )
        if (
            d["parameters"]["imaging"] == "as-stored"
            and d["state"]["mean_volume_nm3"] is not None
        ):
            raise ValueError(
                f"{r['id']}: a periodic trajectory must be inspected with imaging "
                "whole-molecules before replicas are compared"
            )
        if wants_reference and (
            d["reference_values"]["source"] != "supplied structure"
            or d["reference_values"]["values"] != reference["values"]
        ):
            raise ValueError(
                f"{r['id']}: agreement with a reference needs every inspection to have "
                "been fitted to the same supplied reference_structure"
            )
        values = np.asarray(r["values"])
        if (
            values.shape != (d["frame_count"], len(ids))
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"{r['id']}: the observable array does not match its inspection"
            )
    seeds = [r["seed"] for r in replicas]
    if any(s is not None for s in seeds):
        if any(s is None for s in seeds):
            raise ValueError("Give a seed for every replica, or for none")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Independent replicas need distinct seeds")
    frames = {len(r["values"]) for r in replicas}
    if len(frames) != 1:
        raise ValueError("Replicas must have equal numbers of frames")
    n_all = frames.pop()
    trimmed = n_all - n_all // GRID * GRID
    n = n_all - trimmed
    if n < MIN_FRAMES:
        raise ValueError(
            f"Each replica needs at least {MIN_FRAMES} frames (after trimming to a "
            f"multiple of {GRID}); these have {n_all}"
        )
    stack = np.stack([np.asarray(r["values"])[:n] for r in replicas])
    for i in range(len(replicas)):
        for j in range(i):
            if np.array_equal(stack[i], stack[j]):
                raise ValueError(
                    f"Replicas {replicas[j]['id']} and {replicas[i]['id']} hold identical samples"
                )
    axes = [np.asarray(r["times_ps"])[:n] - r["times_ps"][0] for r in replicas]
    step = axes[0][1] - axes[0][0]
    if (
        step <= 0
        or any(not np.allclose(a, axes[0], rtol=1e-7, atol=1e-7) for a in axes)
        or not np.allclose(np.diff(axes[0]), step, rtol=1e-6, atol=1e-7)
    ):
        raise ValueError("Replicas must be sampled at the same uniform interval")
    bad = [b for b in block_counts if n % b or n // b < 2]
    if bad:
        raise ValueError(
            f"Block counts {bad} do not divide the {n} frames per replica into blocks "
            "of at least two frames"
        )
    indices = {name: ids.index(name) for name in by_name}
    stats = sampling_statistics(stack, block_counts)
    count = len(replicas)
    quantile = float(t.ppf(0.975, count - 1))
    rows = []
    for name, i in indices.items():
        criterion = by_name[name]
        mean = stats["mean"][i]
        se = stats["replica_mean_standard_error"][i]
        low, high = mean - quantile * se, mean + quantile * se
        target = (
            reference["values"][name] if criterion["tolerance"] is not None else None
        )
        diagnostic = stats["diagnostics"][i]["status"]
        precise = (
            diagnostic == "estimated"
            and se <= criterion["precision"]
            and stats["split_rhat"][i] <= max_split_rhat
            and stats["effective_samples"][i] >= min_effective_samples
            and max(abs(row[i]) for row in stats["second_minus_first_half"])
            <= 2 * criterion["precision"]
        )
        rows.append(
            {
                "observable": name,
                "unit": common["observables"]["units"][i],
                "mean": mean,
                "replica_means": [row[i] for row in stats["replica_means"]],
                "replica_standard_error": se,
                "ci95": [low, high],
                "split_rhat": stats["split_rhat"][i],
                "effective_samples": stats["effective_samples"][i],
                "diagnostic": diagnostic,
                "max_half_drift": max(
                    abs(row[i]) for row in stats["second_minus_first_half"]
                ),
                "precision_target": criterion["precision"],
                "precision_passed": precise,
                "reference_value": target,
                "tolerance": criterion["tolerance"],
                "mean_offset": None if target is None else mean - target,
                "agrees": None
                if target is None
                else bool(
                    low >= target - criterion["tolerance"]
                    and high <= target + criterion["tolerance"]
                ),
            }
        )
    checkpoints = sorted(
        {MIN_FRAMES, n}
        | {max(MIN_FRAMES, int(n * k / 10) // GRID * GRID) for k in range(1, 10)}
    )
    prefixes = []
    for frames_used in checkpoints:
        passed = _precision_pass(
            sampling_statistics(stack[:, :frames_used], [2]),
            indices,
            by_name,
            max_split_rhat,
            min_effective_samples,
        )
        prefixes.append(
            {
                "frames_per_replica": frames_used,
                "simulated_ps_per_replica": float(axes[0][frames_used - 1] + step),
                "precision_passed": passed,
            }
        )
    first = next(
        (
            p
            for k, p in enumerate(prefixes)
            if all(q["precision_passed"] for q in prefixes[k:])
        ),
        None,
    )
    rmsf = np.asarray(
        [r["details"]["ca_fluctuations"]["rmsf_angstrom"] for r in replicas]
    )
    return {
        "replicas": [r["id"] for r in replicas],
        "frames_per_replica": n,
        "trimmed_frames": trimmed,
        "sampling_interval_ps": float(step),
        "simulated_ps_per_replica": float(axes[0][-1] + step),
        "total_simulated_ps": float((axes[0][-1] + step) * count),
        "observables": rows,
        "precision_passed": all(r["precision_passed"] for r in rows),
        "agrees_with_reference": (
            all(r["agrees"] for r in rows if r["agrees"] is not None)
            if wants_reference
            else None
        ),
        "reference_source": reference["source"] if wants_reference else None,
        "first_sustained_checkpoint": first,
        "prefix_checks": prefixes,
        "rmsf_mean_angstrom": rmsf.mean(axis=0).tolist(),
        "rmsf_standard_error_angstrom": (
            rmsf.std(axis=0, ddof=1) / np.sqrt(count)
        ).tolist(),
        "statistics": stats,
        "limits": {
            "max_split_rhat": max_split_rhat,
            "min_effective_samples": min_effective_samples,
            "block_counts": block_counts,
        },
    }
