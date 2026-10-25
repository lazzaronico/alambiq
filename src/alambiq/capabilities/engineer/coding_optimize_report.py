"""How a recoded sequence appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import table


def render(result) -> str:
    rows = [
        [
            name,
            "yes" if state.satisfied else "no",
            state.gc_fraction,
            state.minimum_window_gc,
            state.maximum_window_gc,
            state.motif_matches,
            state.gc_violations,
        ]
        for name, state in (("Before", result.before), ("After", result.after))
    ]
    return "".join(
        [
            table(
                [
                    "Sequence",
                    "Constraints met",
                    "GC fraction",
                    "Lowest window GC (fraction)",
                    "Highest window GC (fraction)",
                    "Motif matches",
                    "GC violations",
                ],
                rows,
                numeric={2, 3, 4, 5, 6},
            ),
            '<p class="sub">'
            f"{result.status} · {result.changed_codons} codons changed</p>",
        ]
    )
