"""How a fit appears in the dossier."""

from __future__ import annotations

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, number, read_record, table


def _interval(pair) -> str:
    if not pair:
        return "—"
    return f"{number(pair[0])} to {number(pair[1])}"


def _chart(result) -> str:
    """Observations, the fitted curve and its two 95 % intervals, from the stored record."""
    record = read_record(result.record)
    if not record or not record.get("observations"):
        return ""
    request = record.get("request", {})
    observed = record["observations"]
    series = [
        Series(
            "Observations",
            [o["x"] for o in observed],
            [o["y"] for o in observed],
            mode="points",
            color="currentColor",
        )
    ]
    grid = [p for p in record.get("predictions", []) if p.get("y") is not None]
    if len(grid) > 1:
        x = [p["x"] for p in grid]
        for key, label, color in (
            (
                "prediction_interval_95",
                "95 % interval of a new measurement",
                "#e69f00",
            ),
            ("mean_interval_95", "95 % interval of the curve", "#56b4e9"),
        ):
            if all(p.get(key) for p in grid):
                series.append(
                    Series(
                        label,
                        x,
                        [p["y"] for p in grid],
                        mode="band",
                        color=color,
                        band=([p[key][0] for p in grid], [p[key][1] for p in grid]),
                    )
                )
        series.append(
            Series("Fitted curve", x, [p["y"] for p in grid], color="#0072b2")
        )
    return line_chart(
        series,
        x_label=f"{request.get('x_quantity', 'x')} ({request.get('x_unit', '')})",
        y_label=f"{request.get('metric', 'y')} ({request.get('y_unit', '')})",
        title="Measured values and the fitted curve",
    )


def render(result) -> str:
    rows = [
        [
            f"{p.name} (fixed)" if p.fixed else p.name,
            p.value,
            p.standard_error,
            _interval(p.interval_95),
            p.unit or "—",
        ]
        for p in result.parameters
    ]
    termination = result.termination.replace("`", "")
    parts = [
        _chart(result),
        table(
            ["Coefficient", "Value", "Standard error", "95 % interval", "Unit"],
            rows,
            numeric={1, 2},
        ),
        '<p class="sub">'
        f"{result.observations_used} observations fitted, "
        f"{result.observations_excluded} excluded · "
        f"{esc(result.model.replace('_', '-'))} model · "
        f"{esc(result.uncertainty_method.replace('_', ' '))} uncertainty · "
        f"solver termination: {esc(termination)}</p>",
    ]
    if result.predictions:
        parts.append(
            table(
                [
                    "x",
                    "Predicted",
                    "95 % interval of the curve",
                    "95 % interval of a new measurement",
                    "",
                ],
                [
                    [
                        p.x,
                        p.y,
                        _interval(p.mean_interval_95),
                        _interval(p.prediction_interval_95),
                        "outside the measured range: extrapolated"
                        if p.extrapolation
                        else "",
                    ]
                    for p in result.predictions
                ],
                numeric={0, 1},
            )
        )
    if result.derived:
        parts.append(
            table(
                [
                    "Derived quantity",
                    "Value",
                    "Standard error",
                    "95 % interval",
                    "Unit",
                ],
                [
                    [
                        d.name.replace("_", " "),
                        d.value,
                        d.standard_error,
                        _interval(d.interval_95),
                        d.unit,
                    ]
                    for d in result.derived
                ],
                numeric={1, 2},
            )
        )
    return "".join(parts)
