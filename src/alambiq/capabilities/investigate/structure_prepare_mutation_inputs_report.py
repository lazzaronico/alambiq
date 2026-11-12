"""How the preparation of a substitution appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    rows = [
        [
            leg.leg,
            leg.atoms,
            leg.core_atoms,
            ", ".join(leg.wild_type_unique_atoms) or "none",
            ", ".join(leg.mutant_unique_atoms) or "none",
            f"{leg.net_charge:+.3f}",
            leg.padding_nm,
        ]
        for leg in (result.folded, result.reference)
    ]
    left_out = (
        ", ".join([*result.dropped_chains, *result.dropped_heteroatoms]) or "nothing"
    )
    return (
        f"<p>{esc(result.wild_type)}{result.residue_number} to {esc(result.mutant)} in chain "
        f"{esc(result.chain_id)} ({esc(result.mutation)}). Left out of the system: {esc(left_out)}.</p>"
        + table(
            [
                "Leg",
                "Atoms",
                "Core atoms",
                "Wild-type only",
                "Mutant only",
                "Net charge",
                "Padding (nm)",
            ],
            rows,
            numeric={1, 2, 5, 6},
        )
        + f'<p class="sub">Structure SHA-256 {esc(result.structure_sha256)}</p>'
    )
