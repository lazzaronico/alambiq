"""Checks of a residue-substitution request and of the legs of its thermodynamic cycle."""

from __future__ import annotations

import math
from typing import Any

ONE_LETTER = {
    "A": "ALA", "V": "VAL", "I": "ILE", "L": "LEU", "M": "MET", "F": "PHE", "W": "TRP",
    "Y": "TYR", "S": "SER", "T": "THR", "N": "ASN", "Q": "GLN", "C": "CYS",
}  # fmt: skip
SUBSTITUTABLE = tuple(ONE_LETTER.values())
TIMESTEP_PS = 0.002


def residue_name(value: str) -> str:
    """The three-letter name of a substitutable residue, from a one- or three-letter code."""
    text = value.strip().upper()
    name = ONE_LETTER.get(text, text)
    if name not in SUBSTITUTABLE:
        raise ValueError(
            f"to is {value!r}; a residue from {', '.join(SUBSTITUTABLE)} (or its one-letter code) is required"
        )
    return name


def check_preparation(
    *,
    structure: str,
    chain_id: str,
    residue_number: int,
    ph: float,
    folded_padding_nm: float,
    reference_padding_nm: float,
    ionic_strength_molar: float,
    seed: int,
    timeout_seconds: int,
) -> None:
    if not structure.lower().endswith((".pdb", ".ent")):
        raise ValueError(
            f"structure is {structure!r}; a PDB file with the extension .pdb or .ent is required"
        )
    if not chain_id.strip():
        raise ValueError("chain_id is empty; one chain identifier is required")
    for name, value, low, high in (
        ("residue_number", residue_number, -999, 99999),
        ("seed", seed, 0, 2**31 - 1),
        ("timeout_seconds", timeout_seconds, 60, 1_000_000),
    ):
        if not isinstance(value, int) or not low <= value <= high:
            raise ValueError(
                f"{name} requires an integer in [{low}, {high}]; got {value!r}"
            )
    for name, value, low, high in (
        ("ph", ph, 0.0, 14.0),
        ("folded_padding_nm", folded_padding_nm, 1.0, 3.0),
        ("reference_padding_nm", reference_padding_nm, 1.0, 3.0),
        ("ionic_strength_molar", ionic_strength_molar, 0.0, 1.0),
    ):
        if not (
            isinstance(value, (int, float))
            and math.isfinite(value)
            and low <= value <= high
        ):
            raise ValueError(
                f"{name} requires a finite value in [{low}, {high}]; got {value!r}"
            )


def _whole(value: float, unit: float) -> bool:
    ratio = value / unit
    return abs(ratio - round(ratio)) < 1e-6


def check_sampling(
    *,
    replicas: int,
    lambda_windows: int,
    temperature_kelvin: float,
    pressure_bar: float,
    equilibration_ps: float,
    production_ps: float,
    time_per_iteration_ps: float,
    threads: int,
    timeout_seconds: int,
    bootstraps: int = 1000,
    forward_reverse_samples: int = 10,
) -> None:
    for name, value, low, high in (
        ("bootstraps", bootstraps, 0, 1000),
        ("forward_reverse_samples", forward_reverse_samples, 2, 20),
        ("replicas", replicas, 1, 10),
        ("lambda_windows", lambda_windows, 5, 41),
        ("threads", threads, 1, 128),
        ("timeout_seconds", timeout_seconds, 60, 10_000_000),
    ):
        if not isinstance(value, int) or not low <= value <= high:
            raise ValueError(
                f"{name} requires an integer in [{low}, {high}]; got {value!r}"
            )
    for name, value, low, high in (
        ("temperature_kelvin", temperature_kelvin, 250.0, 400.0),
        ("pressure_bar", pressure_bar, 0.1, 100.0),
        ("equilibration_ps", equilibration_ps, 0.0, 1e6),
        ("production_ps", production_ps, 0.04, 1e6),
        ("time_per_iteration_ps", time_per_iteration_ps, TIMESTEP_PS, 100.0),
    ):
        if not (
            isinstance(value, (int, float))
            and math.isfinite(value)
            and low <= value <= high
        ):
            raise ValueError(
                f"{name} requires a finite value in [{low:g}, {high:g}]; got {value!r}"
            )
    if not _whole(time_per_iteration_ps, TIMESTEP_PS):
        raise ValueError(
            f"time_per_iteration_ps is {time_per_iteration_ps}; it must be a whole number of 2 fs steps"
        )
    for name, value in (
        ("production_ps", production_ps),
        ("equilibration_ps", equilibration_ps),
    ):
        if not _whole(value, time_per_iteration_ps):
            raise ValueError(
                f"{name} is {value}; it must be a whole number of iterations of {time_per_iteration_ps} ps"
            )


def leg_statistics(leg: dict[str, Any]) -> dict[str, Any]:
    """Mean, spread and diagnostics of the replicas of one leg record."""
    import statistics

    replicas = leg["replicas"]
    values = [r["delta_g_kcal_mol"] for r in replicas]
    errors = [r["mbar_error_kcal_mol"] for r in replicas]
    overlaps = []
    for r in replicas:
        matrix = r["overlap_matrix"]
        overlaps.append(
            min(min(matrix[i][i + 1], matrix[i + 1][i]) for i in range(len(matrix) - 1))
        )
    gaps = []
    for r in replicas:
        series = r.get("forward_reverse")
        if series:
            half = min(
                range(len(series["fractions"])),
                key=lambda i: abs(series["fractions"][i] - 0.5),
            )
            gaps.append(
                abs(series["forward_kcal_mol"][half] - series["reverse_kcal_mol"][half])
            )
    return {
        "replicas": len(replicas),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "mbar_rms": math.sqrt(statistics.fmean(e * e for e in errors)),
        "minimum_overlap": min(overlaps),
        "minimum_effective_samples": float(
            min(min(r["effective_samples"]) for r in replicas)
        ),
        "forward_reverse_gap": max(gaps) if gaps else None,
        "by_replica": [
            {"id": r["id"], "delta_g": v, "error": e, "overlap": o}
            for r, v, e, o in zip(replicas, values, errors, overlaps, strict=True)
        ],
    }


def combine_preparations(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The legs of independent preparations of one kind, combined.

    Each record is one preparation with one or more replicas. The mean of a leg is the mean of the
    preparation means; the spread across preparations needs 2 or more.
    """
    import statistics

    parts = [leg_statistics(r) for r in records]
    means = [p["mean"] for p in parts]
    replicas = sum(p["replicas"] for p in parts)
    within = [p["sd"] for p in parts if p["sd"] is not None]
    gaps = [
        p["forward_reverse_gap"] for p in parts if p["forward_reverse_gap"] is not None
    ]
    return {
        "preparations": len(parts),
        "replicas": replicas,
        "mean": statistics.fmean(means),
        "preparation_sd": statistics.stdev(means) if len(means) > 1 else None,
        "replica_sd": max(within) if within else None,
        "mbar_rms": math.sqrt(statistics.fmean(p["mbar_rms"] ** 2 for p in parts)),
        "mbar_error": math.sqrt(sum(p["mbar_rms"] ** 2 / p["replicas"] for p in parts))
        / len(parts),
        "minimum_overlap": min(p["minimum_overlap"] for p in parts),
        "minimum_effective_samples": min(p["minimum_effective_samples"] for p in parts),
        "forward_reverse_gap": max(gaps) if gaps else None,
        "by_preparation": parts,
    }
