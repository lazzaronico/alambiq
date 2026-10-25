"""How a genomic neighborhood appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    parts = [
        table(
            ["Anchor", "Status", "Loci"],
            [[a.id, a.status.replace("_", " "), a.loci] for a in result.anchors],
            numeric={2},
        )
    ]
    if result.windows:
        parts.append(
            table(
                ["Anchor", "Contig", "Window", "Features", "Clipped at"],
                [
                    [
                        w.anchor_id,
                        w.contig,
                        f"{w.start:,}–{w.end:,}",
                        w.features,
                        ", ".join(
                            side
                            for side, hit in (
                                ("left", w.left_clipped),
                                ("right", w.right_clipped),
                            )
                            if hit
                        )
                        or "—",
                    ]
                    for w in result.windows
                ],
                numeric={3},
            )
        )
    parts.append(
        f'<p class="sub">{esc(result.genome_id)} · ±{result.flank_bp:,} bp · '
        f"{result.neighbourhood_proteins} proteins in the displayed features. "
        "Coordinates are one-based and inclusive.</p>"
    )
    return "".join(parts)
