"""How ADMET estimates appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table

SHOWN = 8


def render(result) -> str:
    shown = result.molecules[:SHOWN]
    parts = [
        f"<p>{len(result.molecules)} molecules ({result.distinct_molecules} distinct), "
        f"{len(result.endpoints)} estimates each, from ADMET-AI "
        f"{esc(result.engine_versions['admet-ai'])} in {number(result.seconds, 3)} s.</p>"
    ]
    headers = ["Endpoint", "Units"] + [esc(m.canonical_smiles) for m in shown]
    rows = []
    for endpoint in result.endpoints:
        kind = " (probability)" if endpoint.kind == "classification" else ""
        rows.append(
            [endpoint.name + kind, endpoint.units]
            + [m.values[endpoint.id] for m in shown]
        )
    parts.append(table(headers, rows, numeric=set(range(2, 2 + len(shown)))))
    if len(result.molecules) > SHOWN:
        parts.append(
            f'<p class="sub">The first {SHOWN} of {len(result.molecules)} molecules are '
            "shown; the result holds all of them.</p>"
        )
    return "".join(parts)
