"""How transferred annotations appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def _terms(terms, limit: int = 3) -> str:
    shown = [f"{t.term} ({t.name})" if t.name else t.term for t in terms[:limit]]
    more = f" +{len(terms) - limit}" if len(terms) > limit else ""
    return ", ".join(shown) + more


def render(result) -> str:
    rows = [
        [
            p.id,
            p.length,
            len(p.sources),
            f"{p.sources[0].accession} ({p.sources[0].name})" if p.sources else "",
            p.sources[0].identity if p.sources else "",
            _terms(p.ec),
            _terms(p.go_function),
            _terms(p.location),
        ]
        for p in result.proteins
    ]
    return "".join(
        [
            f"<p>{result.annotated} of {len(result.proteins)} proteins annotated from "
            f"Swiss-Prot {esc(result.database_version)} "
            f"(<code>{esc(result.database_identity[:12])}</code>) with "
            f"{esc(result.engine_versions.get('mmseqs', 'MMseqs2'))}.</p>",
            table(
                [
                    "Protein",
                    "Length (residues)",
                    "Sources",
                    "Closest source",
                    "Identity",
                    "EC numbers",
                    "Molecular function",
                    "Location",
                ],
                rows,
                numeric={1, 2, 4},
            ),
        ]
    )
