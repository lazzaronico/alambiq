"""How an inspected reaction appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    balance = result.balance
    if balance.balanced:
        verdict = "The equation balances in elements, isotopes and charge."
    else:
        residual = ", ".join(
            f"{k} {v}" for k, v in {**balance.elements, **balance.isotopes}.items()
        )
        verdict = (
            "The equation does not balance (products minus reactants): "
            f"{residual or 'no element or isotope residual'}; charge {balance.charge}."
        )
    drawings = "".join(
        f'<figure style="display:inline-block;margin:4px">{s.drawing_svg}'
        f'<figcaption class="sub">{esc(s.id)} ({esc(s.side)}, {esc(s.role)})'
        "</figcaption></figure>"
        for s in result.species
    )
    parts = [
        f"<p>{esc(verdict)} Net transformation <code>{esc(result.reaction_id[:20])}</code>.</p>",
        table(
            [
                "Species",
                "Side",
                "Role",
                "Coefficient",
                "Formula",
                "Charge (e)",
                "Canonical SMILES",
            ],
            [
                [
                    s.id,
                    s.side,
                    s.role,
                    s.coefficient,
                    s.formula,
                    s.formal_charge,
                    s.canonical_smiles,
                ]
                for s in result.species
            ],
            numeric={3, 5},
        ),
        drawings,
        f"<p>Atom maps: {esc(result.mapping.status)}.</p>",
    ]
    if result.mapping.bond_changes:
        parts.append(
            table(
                ["Atoms (map numbers)", "Bond order before", "Bond order after"],
                [
                    [
                        f"{c.atoms[0]}–{c.atoms[1]}",
                        "none" if c.reactant_order is None else c.reactant_order,
                        "none" if c.product_order is None else c.product_order,
                    ]
                    for c in result.mapping.bond_changes
                ],
                numeric={1, 2},
            )
        )
    if result.diagnostics:
        parts.append(
            table(
                ["Diagnostic", "Species", "Detail"],
                [
                    [d.code, ", ".join(d.participants), d.message]
                    for d in result.diagnostics
                ],
            )
        )
    return "".join(parts)
