"""How a metadynamics run appears in the dossier."""

from __future__ import annotations

import math

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, table


def _line(axes, values) -> str:
    """Free energy along one torsion; gaps where nothing was sampled."""
    points = [(a, v) for a, v in zip(axes, values, strict=True) if v is not None]
    if len(points) < 2:
        return ""
    return line_chart(
        [Series("Free energy", [p[0] for p in points], [p[1] for p in points])],
        x_label="torsion (radians)",
        y_label="free energy (kJ/mol)",
        title="Free energy along the torsion",
        x_range=(-math.pi, math.pi),
        height=300,
    )


def render(result) -> str:
    one = len(result.torsions) == 1
    figure = (
        _line(result.surface.axes_radians, result.surface.free_energy_kj_mol)
        if one
        else ""
    )
    kt = result.kt_kj_mol
    settle = (
        "not available"
        if result.adaptation_settling_rms_kj_mol is None
        else f"{result.adaptation_settling_rms_kj_mol:.2f} kJ/mol"
    )
    half = (
        "not available"
        if result.half_to_half_rms_kj_mol is None
        else f"{result.half_to_half_rms_kj_mol:.2f} kJ/mol"
    )
    rows = [
        ["Production samples", result.samples],
        ["Effective sample size", result.effective_sample_size],
        ["Effective fraction", result.effective_sample_fraction],
        ["Grid visited", result.bins_visited_fraction],
        ["Half-to-half RMS difference", half],
        ["Adaptation settling (last quarter)", settle],
        ["kT (kJ/mol)", kt],
        ["Bias factor", result.bias_factor],
    ]
    return "".join(
        [
            f"<p>Torsions {esc(', '.join(result.torsions))}; {esc(result.platform)}, OpenMM {esc(result.versions.get('openmm', ''))}.</p>",
            figure,
            table(["Quantity", "Value"], rows, numeric={1}),
        ]
    )
