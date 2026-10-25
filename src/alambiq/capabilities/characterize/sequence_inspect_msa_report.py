"""How an alignment description appears in the dossier."""

from __future__ import annotations

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, number, read_record, table
from alambiq.report.viewers import MAX_ALIGNMENT_ROWS, alignment_view


def _views(result) -> str:
    """Conservation along the alignment, then the aligned rows themselves."""
    record = read_record(result.record)
    if not record or not record.get("rows"):
        return ""
    columns = record.get("columns", [])
    parts = []
    if len(columns) > 1:
        parts.append(
            line_chart(
                [
                    Series(
                        "Entropy",
                        [c["column"] for c in columns],
                        [max(c["entropy_bits"], 0.0) for c in columns],
                    ),
                    Series(
                        "Gap fraction",
                        [c["column"] for c in columns],
                        [c["gap_fraction"] for c in columns],
                        color="#d55e00",
                        dashed=True,
                    ),
                ],
                x_label="alignment column",
                y_label="entropy (bits) · gap fraction",
                title="Variability along the alignment",
                height=260,
            )
        )
    rows = record["rows"]
    parts.append(
        alignment_view(
            [(r["id"], r["aligned"]) for r in rows[:MAX_ALIGNMENT_ROWS]],
            reference=0,
        )
    )
    if len(rows) > MAX_ALIGNMENT_ROWS:
        parts.append(
            f'<p class="sub">The first {MAX_ALIGNMENT_ROWS} of {len(rows)} rows are '
            "shown; blue marks a residue equal to the query residue, orange a "
            "residue that differs from it.</p>"
        )
    return "".join(parts)


def render(result) -> str:
    parts = [
        f'<p class="sub">Query {esc(result.query_id)} · {result.rows} rows '
        f"({result.unique_rows} distinct) · {result.columns} columns · "
        f"mean gap fraction {number(result.mean_gap_fraction)}"
        + (
            f" · {result.inserted_residues} inserted residues"
            if result.inserted_residues
            else ""
        )
        + "</p>",
        f"<p>{len(result.fully_conserved_positions)} query positions are identical in "
        "every row, with no gaps"
        + (
            ": "
            + ", ".join(map(str, result.fully_conserved_positions[:40]))
            + (" …" if len(result.fully_conserved_positions) > 40 else "")
            if result.fully_conserved_positions
            else ""
        )
        + ".</p>",
    ]
    if result.most_variable:
        parts.append(
            table(
                [
                    "Query position",
                    "Entropy (bits)",
                    "Consensus residue",
                    "Consensus share",
                    "Gaps",
                ],
                [
                    [
                        p.query_position,
                        p.entropy_bits,
                        p.consensus,
                        p.consensus_fraction,
                        p.gap_fraction,
                    ]
                    for p in result.most_variable
                ],
                numeric={0, 1, 3, 4},
            )
        )
    parts.append(_views(result))
    return "".join(parts)
