"""How a trajectory appears in the dossier."""

from __future__ import annotations

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, number, read_record, table


def time_course(trajectory: list[dict], title: str) -> str:
    """Concentrations of each species against time, from recorded states."""
    if len(trajectory) < 2:
        return ""
    names = list(trajectory[0]["concentrations_M"])[:8]
    t = [state["time_s"] for state in trajectory]
    series = [
        Series(name, t, [state["concentrations_M"][name] for state in trajectory])
        for name in names
    ]
    return line_chart(
        series, x_label="time (s)", y_label="concentration (M)", title=title
    )


def render(result) -> str:
    rows = [
        [name, value] for name, value in sorted(result.final_concentrations_M.items())
    ]
    record = read_record(result.record)
    scaled = (
        table(
            ["Rate parameter", "Dependence", "Multiplier at the stated conditions"],
            [[m.target, m.kind, m.multiplier] for m in result.condition_multipliers],
            numeric={2},
        )
        if result.condition_multipliers
        else ""
    )
    chart = (
        time_course(record["trajectory"], "Concentrations over time") if record else ""
    )
    return "".join(
        [
            chart,
            table(["Species", "Final concentration (M)"], rows, numeric={1}),
            scaled,
            '<p class="sub">'
            f"Operation {esc(result.operation_id)} · {result.recorded_states} "
            f"recorded states over {number(result.duration_s)} s · "
            f"{result.events} events · largest species-ledger balance error "
            f"{number(result.maximum_balance_error_M)} M</p>",
        ]
    )
