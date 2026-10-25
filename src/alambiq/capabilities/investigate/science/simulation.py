"""Shared validation for the OpenMM capabilities.

Pure functions. Nothing here imports OpenMM; the engines do, in their own
environment.
"""

from __future__ import annotations

from alambiq.capabilities.shared.sequences import parse_residue  # noqa: F401


def platform_properties(platform: str, supplied: dict[str, str]) -> dict[str, str]:
    """The properties to give an OpenMM platform: yours, or reproducible defaults.

    The CPU platform defaults to one thread, because with more the order of
    floating-point sums varies from run to run and the same seed gives a
    different trajectory (checked, not assumed). CUDA defaults to mixed
    precision with deterministic forces, OpenCL to mixed precision, and the
    reference platform takes none.
    """
    if supplied:
        return dict(supplied)
    if platform == "CPU":
        return {"Threads": "1"}
    if platform == "CUDA":
        return {"Precision": "mixed", "DeterministicForces": "true"}
    if platform == "OpenCL":
        return {"Precision": "mixed"}
    return {}


def check_dynamics(
    *,
    ensemble: str,
    temperature_kelvin: float,
    friction_per_ps: float,
    timestep_fs: float,
    pressure_bar: float,
    barostat_interval: int,
    minimize_iterations: int,
    thermalize_ps: float,
    release_ps: float,
    restraint_kj_mol_nm2: float,
    readiness_window_ps: float,
    max_readiness_windows: int,
    production_ps: float,
    report_interval: int,
    seed: int,
) -> None:
    """Refuse a dynamics request whose numbers or phases cannot mean what they say."""
    for name, value, low, high in (
        ("temperature_kelvin", temperature_kelvin, 1, 500),
        ("friction_per_ps", friction_per_ps, 0.01, 100),
        ("timestep_fs", timestep_fs, 0.1, 4),
        ("pressure_bar", pressure_bar, 0.0001, 10000),
        ("restraint_kj_mol_nm2", restraint_kj_mol_nm2, 0, 100000),
        ("thermalize_ps", thermalize_ps, 0, 1000),
        ("release_ps", release_ps, 0, 2000),
        ("readiness_window_ps", readiness_window_ps, 0, 2000),
        ("production_ps", production_ps, 0, 100000),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value:g}; {name} must be between {low:g} and {high:g}"
            )
    for name, value, low, high in (
        ("barostat_interval", barostat_interval, 1, 1000),
        ("minimize_iterations", minimize_iterations, 0, 100000),
        ("max_readiness_windows", max_readiness_windows, 1, 10),
        ("report_interval", report_interval, 1, 1_000_000),
        ("seed", seed, 1, 2147483647),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; {name} must be an integer from {low} to {high}"
            )
    if 0 < readiness_window_ps < 20:
        raise ValueError(
            f"readiness_window_ps is {readiness_window_ps:g}; readiness_window_ps must "
            "be 0 (no check) or at least 20"
        )
    if ensemble == "nve" and (thermalize_ps or release_ps or readiness_window_ps):
        raise ValueError(
            "The nve ensemble has no thermostat to equilibrate with: set thermalize_ps, "
            "release_ps and readiness_window_ps to 0"
        )
    if ensemble != "npt" and release_ps:
        raise ValueError(
            "release_ps applies only to the npt ensemble, where the box relaxes"
        )
    if production_ps == 0 and not (thermalize_ps or release_ps or readiness_window_ps):
        raise ValueError("Nothing to run: every phase is zero length")


def check_metadynamics(
    *,
    ensemble: str,
    bias_factor: float,
    height_kj_mol: float,
    deposition_interval: int,
    adaptation_steps: int,
    frozen_equilibration_steps: int,
    production_steps: int,
    report_interval: int,
    histogram_bins: int,
    torsions: list[dict],
) -> None:
    """Refuse a metadynamics request whose phases or collective variables cannot mean what they say."""
    import math

    if not 2 <= bias_factor <= 100:
        raise ValueError(
            f"bias_factor is {bias_factor:g}; bias_factor must be between 2 and 100"
        )
    if not 0.001 <= height_kj_mol <= 20:
        raise ValueError(
            f"height_kj_mol is {height_kj_mol:g}; height_kj_mol must be between 0.001 "
            "and 20"
        )
    for name, value, low, high in (
        ("deposition_interval", deposition_interval, 1, 1_000_000),
        ("adaptation_steps", adaptation_steps, 1, 500_000_000),
        ("frozen_equilibration_steps", frozen_equilibration_steps, 0, 500_000_000),
        ("production_steps", production_steps, 1, 5_000_000_000),
        ("report_interval", report_interval, 1, 1_000_000),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; {name} must be an integer from {low} to {high}"
            )
    if (
        adaptation_steps % deposition_interval
        or production_steps % report_interval
        or adaptation_steps < 4 * deposition_interval
    ):
        raise ValueError(
            f"adaptation_steps is {adaptation_steps}, deposition_interval {deposition_interval}, "
            f"production_steps {production_steps} and report_interval {report_interval}; "
            "adaptation_steps must be a multiple of deposition_interval and at least four "
            "deposition intervals, and production_steps a multiple of report_interval"
        )
    if production_steps // report_interval > 2_000_000:
        raise ValueError(
            f"production_steps and report_interval give {production_steps // report_interval} "
            "production samples; at most 2,000,000 production samples are recorded"
        )
    if adaptation_steps // deposition_interval > 1_000_000:
        raise ValueError(
            f"adaptation_steps and deposition_interval give "
            f"{adaptation_steps // deposition_interval} Gaussians; at most 1,000,000 "
            "Gaussians may be deposited"
        )
    if not 1 <= len(torsions) <= 2:
        raise ValueError("Select one or two periodic torsions")
    top = 180 if len(torsions) == 1 else 90
    if not 8 <= histogram_bins <= top:
        raise ValueError(
            f"histogram_bins must be an integer from 8 to {top} for {len(torsions)} torsion(s)"
        )
    names, paths = set(), set()
    for row in torsions:
        name = row["name"]
        if not name.isidentifier() or len(name) > 64 or name in names:
            raise ValueError(
                "Torsion names must be unique identifiers of at most 64 characters"
            )
        names.add(name)
        identities = [
            (a["chain_id"], a["residue_id"], a["insertion_code"], a["atom_name"])
            for a in row["atoms"]
        ]
        if len(row["atoms"]) != 4:
            raise ValueError("Each torsion requires four atoms")
        if any(not i[1].strip() or not i[3].strip() for i in identities):
            raise ValueError("Atom name and residue must be populated")
        canonical = min(tuple(identities), tuple(reversed(identities)))
        if len(set(identities)) != 4 or canonical in paths:
            raise ValueError("Torsions require four distinct atoms and distinct paths")
        paths.add(canonical)
        width, grid = row["width_radians"], row["grid_width"]
        if not (math.isfinite(width) and 0.05 <= width <= math.pi):
            raise ValueError("width_radians must be between 0.05 and pi")
        if (
            not 16 <= grid <= (1024 if len(torsions) == 1 else 256)
            or 2 * math.pi / (grid - 1) > width / 4
        ):
            raise ValueError(
                "grid_width must be at least 16 (at most 1024, or 256 for two torsions), with at "
                "least four intervals per Gaussian width"
            )
