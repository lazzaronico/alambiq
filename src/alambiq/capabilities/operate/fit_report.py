"""How an operating fit appears in the dossier."""

from __future__ import annotations

from alambiq.capabilities.operate.progress_chart import progress_charts
from alambiq.report.page import number, read_record, table


def render(result) -> str:
    rows = [
        [
            p.name,
            p.value,
            p.unit,
            f"{number(p.lower)} to {number(p.upper)}",
            f"{number(p.interval[0])} to {number(p.interval[1])}"
            if p.interval
            else "—",
            "at bound" if p.at_bound else "",
        ]
        for p in result.parameters
    ]
    record = read_record(result.record)
    return "".join(
        [
            progress_charts(record["observations"]) if record else "",
            table(
                ["Parameter", "Value", "Unit", "Bounds", "Local 95 % interval", ""],
                rows,
                numeric={1},
            ),
            '<p class="sub">'
            f"{result.fitted} observations fitted, standardized RMSE "
            f"{number(result.fit_error)} · {result.evaluated} held out, "
            f"standardized RMSE {number(result.evaluation_error)} · rank "
            f"{result.rank} · scaled condition number "
            f"{number(result.scaled_condition_number)} · "
            f"{result.converged_starts} of {result.starts} starts converged</p>",
        ]
    )
