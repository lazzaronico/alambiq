"""How a comparison with measurements appears in the dossier."""

from __future__ import annotations

from alambiq.capabilities.operate.progress_chart import progress_charts
from alambiq.report.page import esc, read_record, table

HEADERS = ["Partition", "Quantity", "Target", "Unit", "Observations", "RMSE"]


def _rows(groups) -> list[list]:
    return [
        [g.partition, g.quantity, g.target, g.unit, g.count, g.rmse] for g in groups
    ]


def render(result) -> str:
    record = read_record(result.record)
    return "".join(
        [
            progress_charts(record["observations"]) if record else "",
            table(HEADERS, _rows(result.errors), numeric={4, 5}),
        ]
    )


def render_ensemble(result) -> str:
    parts = [
        '<p class="sub">Pointwise median over '
        f"{result.members} members, with {result.observations} observations each.</p>",
        table(HEADERS, _rows(result.errors), numeric={4, 5}),
    ]
    for name, groups in result.member_errors.items():
        parts.append(f'<p class="sub">Member {esc(name)}</p>')
        parts.append(table(HEADERS, _rows(groups), numeric={4, 5}))
    return "".join(parts)
