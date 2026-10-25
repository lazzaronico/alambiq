"""How an embedding exploration appears in the dossier."""

from __future__ import annotations

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, number, table


def _scatter(result) -> str:
    chosen = set(result.recommended)
    groups = (
        ("Recommended", [p for p in result.points if p.id in chosen], "#0072b2", 5.5),
        (
            "Other proteins",
            [p for p in result.points if p.id not in chosen],
            "#999999",
            3.5,
        ),
    )
    return line_chart(
        [
            Series(
                label,
                [p.x for p in points],
                [p.y for p in points],
                mode="points",
                color=color,
                labels=[p.id for p in points],
                radius=radius,
            )
            for label, points, color, radius in groups
            if points
        ],
        x_label="first axis",
        y_label="second axis",
        title="Two-dimensional layout of the panel",
    )


def render(result) -> str:
    explained = (
        " · ".join(number(v, 3) for v in result.explained_variance_ratio)
        if result.explained_variance_ratio
        else "not defined for this layout"
    )
    return "".join(
        [
            f"<p>{result.proteins} proteins in the {esc(result.model)} space "
            f"({esc(result.layer)}, {esc(result.pooling)}). Layout: {esc(result.projection)}. "
            f"Variance explained by each axis: {esc(explained)}.</p>",
            _scatter(result),
            f"<p>Representatives (highlighted): {esc(', '.join(result.recommended))}.</p>",
            table(
                ["Protein", "Nearest neighbor", "Cosine distance"],
                [[n.id, n.neighbour, n.cosine_distance] for n in result.nearest],
                numeric={2},
            ),
        ]
    )
