"""How a candidate ranking appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table


def render(result) -> str:
    rows = [
        [
            r.rank,
            r.identifier,
            r.evidence_class.replace("-", " "),
            r.bits,
            r.best_reference or "none",
            "" if r.identity is None else r.identity,
            r.sequence_length,
        ]
        for r in result.ranked
    ]
    return "".join(
        [
            "<p>Control check: fraction of positive controls recovered "
            f"{number(result.positive_control_recall)}; lowest positive control "
            f"{number(result.minimum_positive_bits)} bits, highest hard negative "
            f"{number(result.maximum_hard_negative_bits)} bits "
            f"(margin {number(result.hard_negative_margin_bits)} bits). "
            f"AUROC {number(result.control_auroc)} for the score and "
            f"{number(result.length_only_baseline_auroc)} for protein length alone.</p>",
            table(
                [
                    "Rank",
                    "Protein",
                    "Class",
                    "Bits",
                    "Best reference",
                    "Identity",
                    "Length (residues)",
                ],
                rows,
                numeric={0, 3, 5, 6},
            ),
            '<p class="sub">'
            f"{esc(result.engine_versions.get('mmseqs', 'MMseqs2'))}</p>",
        ]
    )
