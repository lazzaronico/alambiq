"""How a selected batch appears in the dossier."""

from __future__ import annotations

from alambiq.report.charts import bar_chart
from alambiq.report.page import esc, number, read_record, table


def _chart(result) -> str:
    """The value the strategy gave each chosen candidate when it was chosen."""
    record = read_record(result.record)
    if not record or not record.get("selected"):
        return ""
    values = [c.get("selection_value") for c in record["selected"]]
    if not any(values):
        return ""
    strategy = record.get("request", {}).get("strategy", "selection")
    return bar_chart(
        [(c["sample_id"], c["selection_value"], True) for c in record["selected"]],
        x_label=f"value at selection ({strategy})",
        title="Chosen candidates in the order they were chosen",
    )


def _chosen(result) -> str:
    """The batch, with a cost column only when costs were supplied, in their unit."""
    if result.cost_unit is None:
        return table(
            ["Candidate", "Group"], [[c.sample_id, c.group_id] for c in result.selected]
        )
    return table(
        ["Candidate", "Group", f"Cost ({result.cost_unit})"],
        [[c.sample_id, c.group_id, c.cost] for c in result.selected],
        numeric={2},
    )


def render(result) -> str:
    parts = [
        _chart(result),
        _chosen(result),
        '<p class="sub">'
        f"{len(result.selected)} of {result.requested} requested candidates chosen"
        + (
            f" · total cost {number(result.total_cost)} {esc(result.cost_unit or '')}"
            if result.total_cost is not None
            else ""
        )
        + f" · {result.rejected} candidates not chosen</p>",
    ]
    if result.reasons:
        parts.append(
            table(
                ["Reason not chosen", "Candidates"],
                [[r.replace("-", " "), n] for r, n in result.reasons.items()],
                numeric={1},
            )
        )
    return "".join(parts)
