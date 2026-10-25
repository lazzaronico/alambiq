"""How thermodynamic estimates appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def _cell(value):
    return "—" if value is None else value


def render(result) -> str:
    c = result.conditions
    parts = [
        f"<p>pH {c.pH:g}, pMg {c.pMg:g}, ionic strength {c.ionic_strength_M:g} M, "
        f"{c.temperature_K:g} K (RT = {result.rt_kj_mol:.4f} kJ/mol).</p>",
        table(
            [
                "Reaction",
                "Status",
                "ΔG′° (kJ/mol)",
                "ΔG′ (kJ/mol)",
                "Model sd (kJ/mol)",
                "ln Q",
                "ln K′",
            ],
            [
                [
                    r.id,
                    r.status,
                    _cell(r.dg0_prime_kj_mol),
                    _cell(r.dg_prime_kj_mol),
                    _cell(r.model_sd_kj_mol),
                    _cell(r.ln_q),
                    _cell(r.ln_k_prime),
                ]
                for r in result.reactions
            ],
            numeric={2, 3, 4, 5, 6},
        ),
    ]
    unresolved = [r for r in result.reactions if r.status != "estimated"]
    if unresolved:
        parts.append(
            table(
                ["Reaction", "Reason for missing estimate"],
                [[r.id, "; ".join(r.notes) or r.status] for r in unresolved],
            )
        )
    route = result.route
    if route.status == "estimated":
        parts.append(
            f"<p>Route: ΔG′° {route.dg0_prime_kj_mol:.3f}, ΔG′ {route.dg_prime_kj_mol:.3f} "
            f"kJ/mol, model sd {route.model_sd_kj_mol:.3f} (with the covariance between steps).</p>"
        )
    else:
        parts.append(
            "<p>Route unresolved: no estimate for "
            f"{esc(', '.join(route.unresolved_steps))}.</p>"
        )
    return "".join(parts)
