"""How scored substitutions appear in the dossier."""

from __future__ import annotations

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, number, table


def render(result) -> str:
    scored = result.positions
    chart = (
        line_chart(
            [
                Series(
                    "Wild-type log-probability",
                    [p.position for p in scored],
                    [p.wild_type_log_prob for p in scored],
                ),
                Series(
                    "Most probable residue",
                    [p.position for p in scored],
                    [p.top_log_prob for p in scored],
                    color="#999999",
                    dashed=True,
                ),
            ],
            x_label="position",
            y_label="log-probability (nats)",
            title="Log-probability of the wild type at each position",
            height=300,
        )
        if len(scored) > 1
        else ""
    )
    unsure = sorted(scored, key=lambda p: -p.entropy_bits)[:10]
    return "".join(
        [
            f"<p>{esc(result.protein_id)}: {result.length} residues, "
            f"{result.positions_scored} positions and {result.substitutions} substitutions "
            f"scored with {esc(result.model)}. The most probable residue equals the wild "
            f"type at {number(result.wild_type_is_top * 100, 3)} % of positions.</p>",
            chart,
            table(
                [
                    "Position",
                    "Wild type",
                    "Wild-type log-probability (nats)",
                    "Most probable residue",
                    "Entropy (bits)",
                ],
                [
                    [
                        p.position,
                        p.wild_type,
                        p.wild_type_log_prob,
                        p.top_residue,
                        p.entropy_bits,
                    ]
                    for p in unsure
                ],
                numeric={0, 2, 4},
            ),
            f'<p class="sub">The table lists the {len(unsure)} positions with the '
            "highest entropy.</p>",
        ]
    )
