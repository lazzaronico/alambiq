"""How stability predictions appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    rows = [
        [
            f"{p.wild_type}{p.residue}{p.mutant}",
            p.residue_name,
            p.ddg_kcal_mol,
        ]
        for p in result.predictions
    ]
    return "".join(
        [
            f"<p>{len(result.predictions)} substitutions on chain {esc(result.chain_id)} "
            f"({result.length} residues): {result.destabilizing} predicted destabilizing, "
            f"{result.stabilizing} stabilizing, {result.unchanged} unchanged, by "
            f"{esc(result.model)} (revision <code>{esc(result.revision[:12])}</code>).</p>",
            table(
                ["Substitution", "Residue", "Predicted ΔΔG (kcal/mol)"],
                rows,
                numeric={2},
            ),
            f'<p class="sub">{esc(result.sign_convention)}.</p>',
        ]
    )
