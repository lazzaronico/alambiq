"""How predicted tracks appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    wrapped = "\n".join(
        result.secondary_structure[i : i + 60]
        for i in range(0, len(result.secondary_structure), 60)
    )
    counts = ", ".join(f"{k} {v}" for k, v in result.secondary_counts.items())
    parts = [
        f"<p>{esc(result.protein_id)}, {result.length} residues, predicted from the "
        f"sequence alone by {esc(result.model)} (revision "
        f"<code>{esc(result.revision[:12])}</code>) on {esc(result.device)}.</p>",
        f"<p>Secondary structure, eight classes (G H I T E B S C), 60 per line. "
        f"Counts: {esc(counts)}.</p><pre>{esc(wrapped)}</pre>",
    ]
    if result.regions:
        parts.append(
            table(
                ["Kind", "Label", "Description", "Start", "End"],
                [
                    [r.kind, r.label, r.description or "", r.start, r.end]
                    for r in result.regions
                ],
                numeric={3, 4},
            )
        )
    return "".join(parts)
