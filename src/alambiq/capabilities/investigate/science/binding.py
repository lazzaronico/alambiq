"""Combining relative free-energy legs into edges, closed cycles and checks.

Pure functions over the records `simulation.run_relative_free_energy` writes. A leg
is one phase (protein-bound or solvent) of one directed edge, with several independent
replicas. The relative binding free energy of an edge is the bound leg's mean minus the
solvent leg's mean, with the replica spread and the propagated MBAR errors kept apart.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import zipfile
from pathlib import Path


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _number(value, name, minimum=None):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(
            f"{name} requires a finite number"
            + (f" >= {minimum}" if minimum is not None else "")
        )
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a nonempty string")
    return value


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Identity requires a lowercase SHA-256 digest")
    return value


def _fields(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f"{name} requires exactly these fields: {keys}")


def _array(value, name, shape=None, minimum=None):
    import numpy as np

    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} requires a numeric array") from exc
    if (
        not np.isfinite(array).all()
        or (shape is not None and array.shape != shape)
        or (minimum is not None and (array < minimum).any())
    ):
        raise ValueError(f"{name} has incompatible shape or numerical values")
    return array


def _replica(row):
    import numpy as np

    _fields(
        row,
        "id trajectory_sha256 delta_g_kcal_mol mbar_error_kcal_mol overlap_matrix effective_samples forward_reverse",
        "Replica",
    )
    _text(row["id"], "Replica identity")
    _digest(row["trajectory_sha256"])
    _number(row["delta_g_kcal_mol"], "Replica free energy")
    _number(row["mbar_error_kcal_mol"], "MBAR uncertainty", 0)
    overlap = _array(row["overlap_matrix"], "Overlap", minimum=0)
    if (
        overlap.ndim != 2
        or overlap.shape[0] != overlap.shape[1]
        or not 2 <= len(overlap) <= 100
        or (overlap > 1 + 1e-8).any()
        or not np.allclose(overlap.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            "Overlap requires a square row-stochastic matrix for 2–100 states"
        )
    effective = _array(
        row["effective_samples"], "Effective samples", (len(overlap),), minimum=0
    )
    half_difference = None
    analysis = row["forward_reverse"]
    if analysis is not None:
        _fields(
            analysis,
            "fractions forward_kcal_mol reverse_kcal_mol forward_error_kcal_mol reverse_error_kcal_mol",
            "Fractional analysis",
        )
        fractions = _array(analysis["fractions"], "Fractions")
        if (
            fractions.ndim != 1
            or not 2 <= len(fractions) <= 100
            or (fractions <= 0).any()
            or (fractions > 1).any()
            or not (np.diff(fractions) >= 0).all()
            or fractions[-1] != 1
        ):
            raise ValueError(
                "Fractions require nondecreasing positive sample fractions ending at one"
            )
        for key in (
            "forward_kcal_mol",
            "reverse_kcal_mol",
            "forward_error_kcal_mol",
            "reverse_error_kcal_mol",
        ):
            _array(
                analysis[key],
                key,
                fractions.shape,
                minimum=0 if "error" in key else None,
            )
        for key in ("forward_kcal_mol", "reverse_kcal_mol"):
            if any(
                not math.isclose(
                    value, row["delta_g_kcal_mol"], rel_tol=1e-6, abs_tol=1e-6
                )
                for fraction, value in zip(fractions, analysis[key], strict=True)
                if fraction == 1
            ):
                raise ValueError(
                    "Full fractional estimates must reproduce the replica estimate"
                )
        if 0.5 in fractions:
            index = list(fractions).index(0.5)
            half_difference = (
                analysis["forward_kcal_mol"][index]
                - analysis["reverse_kcal_mol"][index]
            )
    return {
        **copy.deepcopy(row),
        "minimum_adjacent_overlap": float(
            min(np.diag(overlap, 1).min(), np.diag(overlap, -1).min())
        ),
        "minimum_effective_samples": float(effective.min()),
        "half_forward_reverse_difference_kcal_mol": half_difference,
    }


def _leg(leg, identities, trajectories):
    import numpy as np

    replicas = leg["replicas"]
    if not isinstance(replicas, list) or not 2 <= len(replicas) <= 50:
        raise ValueError(
            "Binding comparison requires at least two independent replicas per leg (maximum 50)"
        )
    rows = []
    for replica in replicas:
        row = _replica(replica)
        if row["id"] in identities or row["trajectory_sha256"] in trajectories:
            raise ValueError(
                "Every leg requires independent replica and trajectory identities"
            )
        identities.add(row["id"])
        trajectories.add(row["trajectory_sha256"])
        rows.append(row)
    n = len(rows)
    sd = float(np.std([r["delta_g_kcal_mol"] for r in rows], ddof=1))
    return {
        "n_replicas": n,
        "mean_kcal_mol": float(np.mean([r["delta_g_kcal_mol"] for r in rows])),
        "replica_sd_kcal_mol": sd,
        "replica_sem_kcal_mol": sd / math.sqrt(n),
        "mbar_propagated_error_kcal_mol": math.sqrt(
            sum(r["mbar_error_kcal_mol"] ** 2 for r in rows)
        )
        / n,
        "replicas": rows,
    }


def _reference(reference, nodes, edges, incidence, cycles, temperature):
    import numpy as np

    if reference is None:
        return None
    _fields(
        reference,
        "temperature_kelvin source conditions error_description observations",
        "Binding reference",
    )
    reference_temperature = _number(
        reference["temperature_kelvin"], "Reference temperature", 0
    )
    if reference_temperature != temperature:
        raise ValueError(
            f"The reference temperature is {reference_temperature:g} K and the series "
            f"temperature is {temperature:g} K; reference comparison requires matched "
            "temperature"
        )
    _fields(reference["source"], "citation license", "Reference source")
    for key in ("citation", "license"):
        _text(reference["source"][key], key)
    for key in ("conditions", "error_description"):
        _text(reference[key], key)
    observations = reference["observations"]
    if not isinstance(observations, list) or len(observations) != len(nodes):
        raise ValueError("Reference requires one observation per series ligand")
    by_id = {}
    for row in observations:
        _fields(
            row,
            "ligand delta_g_kcal_mol reported_error_kcal_mol",
            "Reference observation",
        )
        key = _text(row["ligand"], "Reference ligand")
        if key in by_id or key not in nodes:
            raise ValueError(
                "Reference ligand identities must match the series exactly"
            )
        by_id[key] = (
            _number(row["delta_g_kcal_mol"], "Reference free energy"),
            _number(row["reported_error_kcal_mol"], "Reported reference error", 0),
        )
    values, errors = np.array([by_id[k] for k in nodes]).T
    means = incidence @ values
    covariance = (incidence * errors**2) @ incidence.T
    cycle_results = []
    for cycle in cycles:
        weights = np.array([cycle["coefficients"].get(e["id"], 0) for e in edges])
        # Evaluate through node coefficients so exact cycle cancellation retains zero.
        node_weights = weights @ incidence
        cycle_results.append(
            {
                "id": cycle["id"],
                "closure_kcal_mol": float(node_weights @ values),
                "propagated_reported_error_kcal_mol": float(
                    np.linalg.norm(node_weights * errors)
                ),
            }
        )
    return {
        "source": copy.deepcopy(reference),
        "edge_order": [e["id"] for e in edges],
        "edge_means_kcal_mol": means.tolist(),
        "edge_covariance_kcal2_mol2": covariance.tolist(),
        "covariance_assumption": "Independent ligand-level reported errors; shared-ligand covariance propagated by signed incidence.",
        "cycles": cycle_results,
    }


def analyze_binding_series(series, legs):
    """Combine separately sampled A→B bound and solvent legs, with explicit cycles."""
    import numpy as np

    _fields(series, "schema_version title edges cycles reference", "Binding series")
    if series["schema_version"] != "0.1":
        raise ValueError("Binding series requires schema version 0.1")
    _text(series["title"], "Title")
    edges = series["edges"]
    if not isinstance(edges, list) or not 1 <= len(edges) <= 100:
        raise ValueError("Supply 1–100 directed ligand edges")
    ids, pairs, nodes = set(), set(), set()
    for edge in edges:
        _fields(edge, "id a b", "Edge")
        for key in edge:
            _text(edge[key], "Edge " + key)
        pair = frozenset((edge["a"], edge["b"]))
        if edge["id"] in ids or len(pair) != 2 or pair in pairs:
            raise ValueError(
                "Edges require distinct IDs and distinct undirected ligand pairs"
            )
        ids.add(edge["id"])
        pairs.add(pair)
        nodes.update(pair)
    nodes = sorted(nodes)
    incidence = np.zeros((len(edges), len(nodes)), dtype=int)
    for i, edge in enumerate(edges):
        incidence[i, nodes.index(edge["a"])] = -1
        incidence[i, nodes.index(edge["b"])] = 1
    cycles, cycle_ids = series["cycles"], set()
    if not isinstance(cycles, list) or len(cycles) > 100:
        raise ValueError("Supply a list of at most 100 named cycles")
    for cycle in cycles:
        _fields(cycle, "id coefficients", "Cycle")
        _text(cycle["id"], "Cycle ID")
        c = cycle["coefficients"]
        if (
            cycle["id"] in cycle_ids
            or not isinstance(c, dict)
            or not c
            or c.keys() - ids
            or any(type(v) is not int or v not in (-1, 1) for v in c.values())
        ):
            raise ValueError(
                "Cycles require distinct IDs and signed edge coefficients ±1"
            )
        cycle_ids.add(cycle["id"])
        if (np.array([c.get(e["id"], 0) for e in edges]) @ incidence).any():
            raise ValueError("Cycle must be closed over ligand identities")
    if not isinstance(legs, list) or len(legs) != 2 * len(edges):
        raise ValueError(
            "Supply exactly one bound and one solvent record for each edge"
        )
    by_key, ligand_identity, proteins, methods, temperatures = (
        {},
        {},
        set(),
        set(),
        set(),
    )
    for leg in legs:
        _fields(
            leg,
            "schema_version edge phase temperature_kelvin method_sha256 mapping_sha256 ligand_sha256 protein_sha256 replicas",
            "Free-energy leg",
        )
        if (
            leg["schema_version"] != "0.1"
            or leg["edge"] not in edges
            or leg["phase"] not in {"bound", "solvent"}
        ):
            raise ValueError(
                "Leg schema, edge and phase must match the requested series"
            )
        key = (leg["edge"]["id"], leg["phase"])
        if key in by_key:
            raise ValueError(
                "Supply exactly one bound and one solvent record for each edge"
            )
        by_key[key] = leg
        methods.add(_digest(leg["method_sha256"]))
        temperatures.add(_number(leg["temperature_kelvin"], "Temperature", 1))
        _digest(leg["mapping_sha256"])
        _fields(leg["ligand_sha256"], "a b", "Ligand identities")
        for endpoint, value in leg["ligand_sha256"].items():
            ligand = leg["edge"][endpoint]
            value = _digest(value)
            if ligand in ligand_identity and ligand_identity[ligand] != value:
                raise ValueError(
                    "Every ligand node must preserve the same exact molecular input across legs/edges"
                )
            ligand_identity[ligand] = value
        if leg["phase"] == "bound":
            proteins.add(_digest(leg["protein_sha256"]))
        elif leg["protein_sha256"] is not None:
            raise ValueError("Solvent leg protein identity must be null")
    if len(methods) != 1 or len(temperatures) != 1:
        raise ValueError("All legs require the same exact method and temperature")
    if len(proteins) != 1:
        raise ValueError("All bound legs require the same exact protein input")
    identities, trajectories, results = set(), set(), []
    for edge in edges:
        bound, solvent = [
            by_key.get((edge["id"], phase)) for phase in ("bound", "solvent")
        ]
        if bound is None or solvent is None:
            raise ValueError(
                "Supply exactly one bound and one solvent record for each edge"
            )
        if bound["mapping_sha256"] != solvent["mapping_sha256"]:
            raise ValueError("Both thermodynamic legs require identical atom mapping")
        b, s = [_leg(leg, identities, trajectories) for leg in (bound, solvent)]
        results.append(
            {
                **edge,
                "bound": b,
                "solvent": s,
                "delta_delta_g_kcal_mol": b["mean_kcal_mol"] - s["mean_kcal_mol"],
                "replica_sem_kcal_mol": math.hypot(
                    b["replica_sem_kcal_mol"], s["replica_sem_kcal_mol"]
                ),
                "mbar_propagated_error_kcal_mol": math.hypot(
                    b["mbar_propagated_error_kcal_mol"],
                    s["mbar_propagated_error_kcal_mol"],
                ),
            }
        )
    cycle_results = []
    for cycle in cycles:
        selected = [
            (cycle["coefficients"][r["id"]], r)
            for r in results
            if r["id"] in cycle["coefficients"]
        ]
        cycle_results.append(
            {
                **copy.deepcopy(cycle),
                "closure_kcal_mol": sum(
                    c * r["delta_delta_g_kcal_mol"] for c, r in selected
                ),
                **{
                    key: math.sqrt(sum(r[key] ** 2 for _, r in selected))
                    for key in (
                        "replica_sem_kcal_mol",
                        "mbar_propagated_error_kcal_mol",
                    )
                },
            }
        )
    temperature = next(iter(temperatures))
    return {
        "schema_version": "0.1",
        "title": series["title"],
        "temperature_kelvin": temperature,
        "direction": "ΔΔG_bind(B−A) = ΔG_bound(A→B) − ΔG_solvent(A→B)",
        "uncertainty": "Equal-weight replica means; sample SD and SEM across independent replicas; propagated within-replica MBAR errors displayed separately. Cycle uncertainties assume independently sampled edges.",
        "effective_samples": "Per-state MBAR importance-weight effective sample sizes after the native analysis decorrelation procedure.",
        "method_sha256": next(iter(methods)),
        "protein_sha256": next(iter(proteins)),
        "ligand_sha256": ligand_identity,
        "edges": results,
        "cycles": cycle_results,
        "reference": _reference(
            series["reference"], nodes, edges, incidence, cycles, temperature
        ),
    }


BINDING_CHECKS = {
    "maximum_edge_replica_sem_kcal_mol",
    "maximum_leg_replica_sd_kcal_mol",
    "minimum_adjacent_overlap_each_replica",
    "minimum_effective_samples_each_state",
    "half_fraction_target",
    "half_fraction_maximum_distance",
    "maximum_half_forward_reverse_difference_kcal_mol",
    "cycle_absolute_floor_kcal_mol",
    "cycle_replica_sem_multiplier",
    "maximum_ligand_com_drift_angstrom",
}


def binding_check_settings(criteria):
    if (
        not isinstance(criteria, dict)
        or not criteria
        or set(criteria) - BINDING_CHECKS
        or any(not _finite(v) or v < 0 for v in criteria.values())
    ):
        raise ValueError(
            "Supply named binding checks with finite nonnegative thresholds"
        )
    for group in (
        {
            "half_fraction_target",
            "half_fraction_maximum_distance",
            "maximum_half_forward_reverse_difference_kcal_mol",
        },
        {"cycle_absolute_floor_kcal_mol", "cycle_replica_sem_multiplier"},
    ):
        if set(criteria) & group and not group <= criteria.keys():
            raise ValueError("Supply the complete fractional or cycle check settings")
    if (
        "half_fraction_target" in criteria
        and not 0 < criteria["half_fraction_target"] < 1
        or criteria.get("half_fraction_maximum_distance", 0) > 0.5
        or criteria.get("minimum_adjacent_overlap_each_replica", 0) > 1
    ):
        raise ValueError(
            "Fractions and overlap thresholds must retain their probability ranges"
        )
    return copy.deepcopy(criteria)


def _check(name, identity, value, limit, lower=False, reason=None):
    if value is not None and not _finite(value):
        raise ValueError(
            "A checked diagnostic must be finite numeric data or explicit null"
        )
    status = (
        "unassessed"
        if value is None
        else "pass"
        if (value >= limit if lower else abs(value) <= limit)
        else "fail"
    )
    row = dict(
        diagnostic=name,
        identity=identity,
        observed=value,
        limit=limit,
        comparison=">=" if lower else "absolute <=",
        status=status,
    )
    if reason:
        row["reason"] = reason
    return row


def _status(rows):
    statuses = {r["status"] for r in rows}
    return (
        "fail"
        if "fail" in statuses
        else "unassessed"
        if "unassessed" in statuses or not rows
        else "pass"
    )


def read_native_com_drift(path, state_count):
    """Read OpenFE's native structural NPZ; retain exact distance/time identity."""
    import numpy as np

    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > 200_000_000:
                raise ValueError("Structural NPZ expanded data exceeds 200 MB")
    except zipfile.BadZipFile as exc:
        raise ValueError("Supply a native structural NPZ archive") from exc
    with np.load(path, allow_pickle=False) as arrays:
        times = arrays["time_ps"].copy()
        drift = arrays["ligand_COM_drift"].copy()
    if (
        times.ndim != 1
        or not 2 <= len(times) <= 1_000_000
        or drift.shape != (state_count, len(times))
        or any(
            a.dtype.kind not in "fiu" or not np.isfinite(a).all() or (a < 0).any()
            for a in (times, drift)
        )
        or not (np.diff(times) > 0).all()
    ):
        raise ValueError(
            "Native COM drift requires nonnegative finite state-by-time angstrom data and increasing ps times"
        )
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(
        source_sha256=sha,
        distance_unit="angstrom",
        time_unit="picosecond",
        state_count=state_count,
        frames_per_state=len(times),
        time_start_ps=float(times[0]),
        time_end_ps=float(times[-1]),
        per_state_maximum_angstrom=drift.max(axis=1).astype(float).tolist(),
        maximum_drift_angstrom=float(drift.max()),
    )


def assess_binding(data, criteria, structural_diagnostics=None):
    criteria = binding_check_settings(criteria)
    structural = structural_diagnostics or {}
    checks, review = [], []
    for edge in data["edges"]:
        if "maximum_edge_replica_sem_kcal_mol" in criteria:
            checks.append(
                _check(
                    "edge_replica_sem",
                    edge["id"],
                    edge["replica_sem_kcal_mol"],
                    criteria["maximum_edge_replica_sem_kcal_mol"],
                )
            )
        for phase in ("bound", "solvent"):
            leg, identity = edge[phase], edge["id"] + "/" + phase
            if "maximum_leg_replica_sd_kcal_mol" in criteria:
                checks.append(
                    _check(
                        "leg_replica_sd",
                        identity,
                        leg["replica_sd_kcal_mol"],
                        criteria["maximum_leg_replica_sd_kcal_mol"],
                    )
                )
            for replica in leg["replicas"]:
                key = identity + "/" + replica["id"]
                for name, field, setting in (
                    (
                        "overlap",
                        "minimum_adjacent_overlap",
                        "minimum_adjacent_overlap_each_replica",
                    ),
                    (
                        "effective_samples",
                        "minimum_effective_samples",
                        "minimum_effective_samples_each_state",
                    ),
                ):
                    if setting in criteria:
                        checks.append(
                            _check(name, key, replica[field], criteria[setting], True)
                        )
                if "half_fraction_target" in criteria:
                    fractional = replica["forward_reverse"]
                    fraction, difference = None, None
                    if fractional is not None and fractional["fractions"]:
                        index = min(
                            range(len(fractional["fractions"])),
                            key=lambda i: abs(
                                fractional["fractions"][i]
                                - criteria["half_fraction_target"]
                            ),
                        )
                        fraction = fractional["fractions"][index]
                        if (
                            abs(fraction - criteria["half_fraction_target"])
                            <= criteria["half_fraction_maximum_distance"] + 1e-12
                        ):
                            difference = (
                                fractional["forward_kcal_mol"][index]
                                - fractional["reverse_kcal_mol"][index]
                            )
                    row = _check(
                        "half_forward_reverse",
                        key,
                        difference,
                        criteria["maximum_half_forward_reverse_difference_kcal_mol"],
                        reason="Native fraction must lie within the declared target tolerance"
                        if difference is None
                        else None,
                    )
                    row["selected_fraction"] = fraction
                    checks.append(row)
                if phase == "bound" and "maximum_ligand_com_drift_angstrom" in criteria:
                    diagnostic = structural.get(replica["id"])
                    row = _check(
                        "ligand_com_drift",
                        key,
                        diagnostic["maximum_drift_angstrom"] if diagnostic else None,
                        criteria["maximum_ligand_com_drift_angstrom"],
                        reason="Native structural NPZ was not supplied"
                        if not diagnostic
                        else None,
                    )
                    row["purpose"] = "structural-review"
                    if diagnostic:
                        row["source_sha256"] = diagnostic["source_sha256"]
                    review.append(row)
    if "cycle_absolute_floor_kcal_mol" in criteria:
        for cycle in data["cycles"]:
            limit = max(
                criteria["cycle_absolute_floor_kcal_mol"],
                criteria["cycle_replica_sem_multiplier"]
                * cycle["replica_sem_kcal_mol"],
            )
            checks.append(
                _check("cycle_closure", cycle["id"], cycle["closure_kcal_mol"], limit)
            )
        if not data["cycles"]:
            checks.append(
                _check(
                    "cycle_closure",
                    "series",
                    None,
                    criteria["cycle_absolute_floor_kcal_mol"],
                    reason="No closed cycle was supplied",
                )
            )
    result = dict(
        schema_version="0.1",
        criteria=criteria,
        checks=checks + review,
        status=_status(checks + review),
        interpretation="Checks report diagnostics. Structural excursions require researcher review. Sampling agreement and experimental agreement are assessed separately.",
    )
    if checks:
        result["sampling_status"] = _status(checks)
    if review:
        result["structural_review_status"] = (
            "review-required"
            if any(r["status"] == "fail" for r in review)
            else _status(review)
        )
    return result
