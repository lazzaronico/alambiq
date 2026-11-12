"""How a protonation appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table

SHOWN = 30


def _shown(
    title: str, headers: list[str], rows: list[list], numeric: set[int] | None = None
) -> str:
    if not rows:
        return f"<h3>{esc(title)}</h3><p>None.</p>"
    more = (
        f'<p class="sub">The first {SHOWN} of {len(rows)} are shown; the result lists all.</p>'
        if len(rows) > SHOWN
        else ""
    )
    return (
        f"<h3>{esc(title)} ({len(rows)})</h3>"
        + table(headers, rows[:SHOWN], numeric=numeric)
        + more
    )


def render(result) -> str:
    changed = [g for g in result.groups if g.name_in != g.name_out]
    others = ", ".join(
        f"{name} ({count} atoms)"
        for name, count in sorted(result.dropped_heteroatoms.items())
    )
    return "".join(
        [
            f"<p>Protonated at pH {result.ph:g}: {result.atoms} atoms, {result.hydrogens} of them "
            f"hydrogen. Left out: {result.dropped_waters} water atoms, {result.dropped_hydrogens} "
            f"hydrogen atoms of the file, {result.dropped_alternate_atoms} atoms of other "
            f"alternate locations, heteroatoms: {esc(others) or 'none'}.</p>",
            _shown(
                "Groups that changed their residue name",
                ["Group", "Residue", "pKa", "In the file", "Protonated structure"],
                [
                    [g.group, f"{g.chain}:{g.number}", g.pka, g.name_in, g.name_out]
                    for g in changed
                ],
                numeric={2},
            ),
            _shown(
                "Side chains turned by 180 degrees",
                ["Residue"],
                [[f"{t.chain}:{t.name}{t.number}"] for t in result.turned],
            ),
            _shown(
                "Heavy atoms added",
                ["Residue", "Atoms"],
                [
                    [f"{a.chain}:{a.name}{a.number}", ", ".join(a.atoms)]
                    for a in result.added_heavy_atoms
                ],
            ),
            _shown(
                "Titratable groups",
                ["Group", "Residue", "pKa", "pKa in water", "Protonated"],
                [
                    [
                        g.group,
                        f"{g.chain}:{g.name_in}{g.number}",
                        g.pka,
                        g.model_pka,
                        "yes" if g.protonated else "no",
                    ]
                    for g in result.groups
                ],
                numeric={2, 3},
            ),
            f"<p>Disulfide cysteines: {esc(', '.join(result.disulfide_cysteines)) or 'none'}.</p>",
            f'<p class="sub">PDB2PQR {esc(result.versions.get("pdb2pqr", ""))}, PROPKA '
            f"{esc(result.versions.get('propka', ''))}; structure SHA-256 {esc(result.structure_sha256)}</p>",
        ]
    )
