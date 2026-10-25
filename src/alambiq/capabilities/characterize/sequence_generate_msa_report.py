"""How generated alignments appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    rows = [
        [
            c.id,
            c.length,
            c.decision,
            ", ".join(c.reason_codes),
            c.sequences,
            c.unique_sequences,
            c.diverse_sequences,
            c.query_coverage,
            c.median_query_coverage,
        ]
        for c in result.chains
    ]
    return "".join(
        [
            f"<p>{result.generated} alignments generated and {result.withheld} withheld, "
            f"searched against UniRef50 {esc(result.database_version)} "
            f"(<code>{esc(result.database_identity[:12])}</code>) with "
            f"{esc(result.engine_versions.get('mmseqs', 'MMseqs2'))}.</p>",
            table(
                [
                    "Protein",
                    "Length (residues)",
                    "Decision",
                    "Reasons",
                    "Rows",
                    "Unique",
                    "Diverse",
                    "Query covered",
                    "Median row covered",
                ],
                rows,
                numeric={1, 4, 5, 6, 7, 8},
            ),
        ]
    )
