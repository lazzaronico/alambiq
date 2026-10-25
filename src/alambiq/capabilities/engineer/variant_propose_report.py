"""How proposed substitutions appear in the dossier."""

from __future__ import annotations

from alambiq.report.charts import bar_chart
from alambiq.report.page import esc, table


def render(result) -> str:
    proposals = result.proposals
    chart = bar_chart(
        [(p.variant, p.llr, True) for p in proposals],
        x_label="log-likelihood ratio (nats)",
        title="Proposed substitutions in rank order",
    )
    rows = [
        [
            p.rank,
            p.variant,
            p.llr,
            "" if p.ddg_kcal_mol is None else p.ddg_kcal_mol,
            "" if p.combined_rank is None else p.combined_rank,
            "" if p.homolog_count is None else p.homolog_count,
        ]
        for p in proposals
    ]
    excluded = ", ".join(
        f"{n} {esc(reason.replace('-', ' '))}" for reason, n in result.excluded.items()
    )
    return "".join(
        [
            f"<p>{len(proposals)} proposals for {esc(result.protein_id)} ranked by "
            f"{esc(result.ranking)}, from {result.candidates} candidates that passed the "
            f"filters{f' (excluded: {excluded})' if excluded else ''}.</p>",
            chart,
            table(
                [
                    "Rank",
                    "Variant",
                    "Log-likelihood ratio",
                    "Predicted ΔΔG (kcal/mol)",
                    "Combined rank",
                    "Homologs with it (count)",
                ],
                rows,
                numeric={0, 2, 3, 4, 5},
            ),
        ]
    )
