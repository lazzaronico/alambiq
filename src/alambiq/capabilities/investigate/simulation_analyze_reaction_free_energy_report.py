"""How a reaction free-energy profile appears in the dossier."""

from __future__ import annotations

import math

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, table


def _profile(centres, values, errors) -> str:
    """The mean profile with its replica standard errors as a band."""
    points = [
        (x, v, 0.0 if e is None else e)
        for x, v, e in zip(centres, values, errors, strict=True)
        if v is not None
    ]
    if len(points) < 2:
        return ""
    xs = [p[0] for p in points]
    return line_chart(
        [
            Series(
                "Mean of replicas",
                xs,
                [p[1] for p in points],
                mode="both",
                band=([p[1] - p[2] for p in points], [p[1] + p[2] for p in points]),
            )
        ],
        x_label="reaction coordinate (Å)",
        y_label="free energy (kJ/mol)",
        title="Free-energy profile, band is one standard error across replicas",
        height=300,
    )


def _number(value) -> str | float:
    return (
        ""
        if value is None or (isinstance(value, float) and math.isnan(value))
        else value
    )


def render(result) -> str:
    replicas = table(
        [
            "Replica",
            "Basin ΔG (kJ/mol)",
            "Peak − reference",
            "Min overlap",
            "Min effective n",
            "Blocks failed / total",
        ],
        [
            [
                r.id,
                _number(r.basin_delta_g_kj_mol),
                _number(r.peak_minus_reference_kj_mol),
                r.minimum_adjacent_overlap,
                r.minimum_state_effective_samples,
                f"{r.blocks_failed} / {r.blocks_total}",
            ]
            for r in result.replicas
        ],
        numeric={1, 2, 3, 4},
    )
    basin = (
        "not available (a basin was unsampled)"
        if result.basin_delta_g_mean_kj_mol is None
        else f"{result.basin_delta_g_mean_kj_mol:.2f} kJ/mol, replica SEM "
        f"{_number(result.basin_delta_g_replica_sem_kj_mol)}"
    )
    parts = [
        f"<p>{esc(result.title)}: {result.windows} windows, {result.frames} frames at "
        f"{result.temperature_kelvin:g} K along {esc(result.coordinate)}. Product minus reactant "
        f"basin: {esc(basin)}. PyMBAR {esc(result.pymbar_version)}.</p>",
        _profile(
            result.bin_centres_angstrom,
            result.pmf_mean_kj_mol,
            result.pmf_replica_sem_kj_mol,
        ),
        replicas,
    ]
    if result.flux is not None:
        parts.append(
            f"<p>Classical transition-state flux: activation free energy "
            f"{_number(result.flux.activation_free_energy_kj_mol_mean)} kJ/mol "
            f"(replica SEM {_number(result.flux.activation_free_energy_kj_mol_replica_sem)}), with "
            "transmission factor 1.</p>"
        )
    return "".join(parts)
