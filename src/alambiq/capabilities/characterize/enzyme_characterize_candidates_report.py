"""How candidate decisions appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table


def render(result) -> str:
    rows = [
        [
            a.identifier,
            a.evidence_class.replace("-", " "),
            a.decision,
            ", ".join(a.reason_codes),
            a.retrieval_bits,
            a.identity,
            a.query_coverage,
            a.conserved_feature_fraction,
        ]
        for a in result.assessments
    ]
    policy = " · ".join(
        f"{k.replace('_', ' ')} {number(v)}" for k, v in result.policy.items()
    )
    detail = "".join(
        f"<details><summary>{esc(a.identifier)}: residues</summary>"
        + table(
            [
                "Reference position",
                "Expected",
                "Role",
                "Protein position",
                "Observed",
                "Match",
            ],
            [
                [
                    r["reference_position"],
                    r["expected_residue"],
                    r["role"],
                    "" if r["target_position"] is None else r["target_position"],
                    r["observed_residue"] or "gap",
                    "yes" if r["conserved"] else "no",
                ]
                for r in a.residues
            ],
            numeric={0, 3},
        )
        + "</details>"
        for a in result.assessments
    )
    return "".join(
        [
            f"<p>{len(result.advanced)} advanced, {len(result.withheld)} withheld and "
            f"{len(result.rejected)} rejected against reference "
            f"{esc(result.reference_id)}. Fraction of positive controls advanced: "
            f"{number(result.positive_control_advance_rate)}; fraction of hard "
            f"negatives rejected: "
            f"{number(result.negative_control_rejection_rate)}.</p>",
            table(
                [
                    "Protein",
                    "Class",
                    "Decision",
                    "Reasons",
                    "Retrieval bits",
                    "Identity",
                    "Coverage",
                    "Features conserved",
                ],
                rows,
                numeric={4, 5, 6, 7},
            ),
            f'<p class="sub">{esc(policy)}</p>',
            detail,
        ]
    )
