"""Checks and conversions for QM/MM requests, before any file is read."""

from __future__ import annotations

import math
from typing import Any


def check_qmmm_settings(
    *,
    grid: int,
    gaussian_radius_angstrom: float,
    threads: int,
    memory_mb: int,
    classical_coulomb_ratio: float,
    timeout_seconds: int,
) -> None:
    """Refuse electronic settings PySCF cannot honour."""
    for name, value, low, high in (
        ("grid", grid, 0, 9),
        ("threads", threads, 1, 1024),
        ("memory_mb", memory_mb, 256, 10**7),
        ("timeout_seconds", timeout_seconds, 1, 604800),
    ):
        if not low <= value <= high:
            raise ValueError(f"{name} requires an integer in [{low}, {high}]")
    for name, value in (
        ("gaussian_radius_angstrom", gaussian_radius_angstrom),
        ("classical_coulomb_ratio", classical_coulomb_ratio),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} requires a positive finite number")


def region_dict(region: Any) -> dict[str, Any]:
    """The quantum region as the engine reads it."""
    link = None
    if region.link is not None:
        link = {
            "quantum_parent": region.link.quantum_parent,
            "classical_parent": region.link.classical_parent,
            "alpha": region.link.alpha,
            "cancelled_coupling_atoms": list(region.link.cancelled_coupling_atoms),
        }
    return {"atoms": list(region.atoms), "charge": region.charge, "link": link}


def umbrella_dict(umbrella: Any) -> dict[str, Any] | None:
    if umbrella is None:
        return None
    return {
        "pairs": [list(p) for p in umbrella.pairs],
        "center_angstrom": umbrella.center_angstrom,
        "coefficient_kj_mol_angstrom2": umbrella.coefficient_kj_mol_angstrom2,
    }


def check_sampling(sampling: dict[str, Any]) -> None:
    """Refuse sampling choices the integrator or the bookkeeping cannot honour."""
    for name in ("temperature_kelvin", "friction_per_ps", "timestep_fs"):
        value = sampling[name]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} requires a positive finite number")
    for name, low in (
        ("seed", 1),
        ("equilibration_steps", 0),
        ("production_steps", 1),
        ("report_interval", 1),
    ):
        if not low <= sampling[name] <= 2147483647:
            raise ValueError(f"{name} requires an integer in [{low}, 2147483647]")
    if sampling["production_steps"] % sampling["report_interval"]:
        raise ValueError("Production steps must contain complete reporting intervals")
    umbrella = sampling["umbrella"]
    if umbrella is not None:
        pairs = umbrella["pairs"]
        if len(pairs) != 2 or any(len(p) != 2 or p[0] == p[1] for p in pairs):
            raise ValueError(
                "Two ordered one-based atom pairs define the distance difference"
            )
        if set(pairs[0]) == set(pairs[1]):
            raise ValueError("The two distances require distinct physical atom pairs")
        if not math.isfinite(umbrella["center_angstrom"]):
            raise ValueError("A finite umbrella center is required")
        k = umbrella["coefficient_kj_mol_angstrom2"]
        if not math.isfinite(k) or k <= 0:
            raise ValueError("The umbrella coefficient must be positive")
