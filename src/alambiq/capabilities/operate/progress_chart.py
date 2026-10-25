"""Measured and predicted values against time, one chart per quantity."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from alambiq.report.charts import Series, line_chart

MAX_CHARTS = 6


def progress_charts(rows: list[dict[str, Any]]) -> str:
    """Charts of observation rows that contain `time_s`, `value` and `predicted`.

    Rows are grouped by experiment, quantity and target; each group is one chart with the
    measured values as points and the predictions at the same times joined by a line.
    """
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "time_s" in row and "predicted" in row:
            key = (row.get("experiment_id", ""), row["quantity"], row["target"])
            groups[key].append(row)
    out = []
    for (experiment, quantity, target), members in list(groups.items())[:MAX_CHARTS]:
        members.sort(key=lambda r: r["time_s"])
        times = [r["time_s"] for r in members]
        name = " · ".join(str(part) for part in (experiment, quantity, target) if part)
        unit = members[0].get("unit", "")
        out.append(
            line_chart(
                [
                    Series(
                        "Predicted",
                        times,
                        [r["predicted"] for r in members],
                        mode="both",
                    ),
                    Series(
                        "Measured",
                        times,
                        [r["value"] for r in members],
                        mode="points",
                        color="currentColor",
                    ),
                ],
                x_label="time (s)",
                y_label=f"{target} ({unit})" if unit else str(target),
                title=name,
                height=300,
            )
        )
    if len(groups) > MAX_CHARTS:
        out.append(
            f'<p class="sub">{len(groups) - MAX_CHARTS} more quantities are in the '
            "stored record.</p>"
        )
    return "".join(out)
