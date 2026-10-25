"""How ranked pockets appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    rows = [
        [
            p.rank,
            p.score,
            p.probability,
            p.sas_points,
            p.surface_atoms,
            f"({p.center_angstrom[0]:.1f}, {p.center_angstrom[1]:.1f}, {p.center_angstrom[2]:.1f})",
            len(p.residue_ids),
            "" if p.query_positions is None else ", ".join(map(str, p.query_positions)),
        ]
        for p in result.pockets
    ]
    return "".join(
        [
            f"<p>{len(result.pockets)} pockets on a {esc(result.structure_kind)} structure "
            f"(P2Rank {esc(result.versions.get('p2rank', ''))}, configuration "
            f"{esc(result.configuration)}, Java {esc(result.versions.get('java', ''))}).</p>",
            table(
                [
                    "Rank",
                    "Score",
                    "Probability",
                    "Surface points",
                    "Surface atoms",
                    "Center (Å)",
                    "Residues",
                    "Query positions",
                ],
                rows,
                numeric={0, 1, 2, 3, 4, 6},
            ),
        ]
    )
