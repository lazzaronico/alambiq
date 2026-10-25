"""How a composed insert appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    rows = [
        [
            p.id,
            p.role,
            f"{p.dna_start}–{p.dna_end}",
            f"{p.protein_start}–{p.protein_end}",
            p.removed_stop_codon or "",
        ]
        for p in result.parts
    ]
    return "".join(
        [
            table(["Part", "Role", "Bases", "Residues", "Stop codon removed"], rows),
            '<p class="sub">'
            f"{esc(result.identifier)}: {result.length_bp} bp encoding "
            f"{result.protein_length} residues"
            + (
                ", encoded with the supplied codon table"
                if result.encoded_from_codon_table
                else ""
            )
            + ". Coordinates are one-based and inclusive.</p>",
        ]
    )
