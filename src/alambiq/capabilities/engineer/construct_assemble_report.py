"""How an assembled construct appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    return "".join(
        [
            table(
                ["Primer", "Sequence", "Annealing footprint", "Footprint Tm (°C)"],
                [
                    [p.id, p.sequence, p.footprint, p.footprint_tm_degC]
                    for p in result.primers
                ],
                numeric={3},
            ),
            '<p class="sub">'
            f"{result.length} bp circular product encoding "
            f"{esc(result.protein_id)} · {result.features} annotated features · "
            f"{result.removed_vector_features} vector features not annotated on the "
            f"product · SHA-256 of the sequence, first 16 characters: "
            f"{esc(result.sequence_sha256[:16])}</p>",
        ]
    )
