"""How a route search appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    smiles = {m.chemical_id: m.canonical_smiles for m in result.molecules}
    by_id = {r.id: r for r in result.reactions}

    def side(ids):
        return " + ".join(smiles.get(i, i) for i in ids)

    parts = [
        f"<p>Target: <strong>{esc(result.target_status)}</strong>. "
        f"{len(result.routes)} routes, {len(result.reactions)} generated reactions, "
        f"{len(result.molecules)} molecules; {result.rejected_reactions} candidate "
        f"reactions rejected (unbalanced or beyond a bound); "
        f"{result.recipes_evaluated} operator applications.</p>"
    ]
    if result.limits_reached:
        parts.append(
            '<div class="note">Bounds reached, so the search stopped early: '
            f"{esc(', '.join(result.limits_reached))}.</div>"
        )
    if result.routes:
        rows = []
        for route in result.routes:
            chain = " → ".join(
                by_id[s.reaction_id].operator_name
                if s.reaction_id in by_id
                else s.reaction_id
                for s in route.steps
            )
            rows.append(
                [
                    route.id[:18],
                    len(route.steps),
                    chain or "(no steps)",
                    route.target_amount,
                ]
            )
        parts.append(
            table(
                ["Route", "Steps", "Operators in order", "Target amount (feedstock units)"],
                rows,
                numeric={1, 3},
            )
        )
    if result.reactions:
        parts.append(
            table(
                ["Operator", "Reactants", "Products", "Requires", "Enzyme references"],
                [
                    [
                        r.operator_name,
                        side(r.reactants),
                        side(r.products),
                        "; ".join(r.catalyst_requirements),
                        r.enzyme_references,
                    ]
                    for r in result.reactions
                ],
                numeric={4},
            )
        )
    if result.versions:
        parts.append(
            '<p class="sub">Backend: '
            f"{esc(', '.join(f'{k} {v}' for k, v in result.versions.items()))}.</p>"
        )
    return "".join(parts)
