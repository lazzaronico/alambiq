"""How a structure comparison appears in the dossier."""

from __future__ import annotations

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def _profile(result) -> str:
    """Distance between each fitted pair, before and after the superposition."""
    pairs = [p for p in result.pairs if p.distance_after_angstrom is not None]
    if len(pairs) < 2:
        return ""
    index = list(range(1, len(pairs) + 1))
    return line_chart(
        [
            Series(
                "Before the fit",
                index,
                [p.distance_before_angstrom for p in pairs],
                color="#999999",
                dashed=True,
            ),
            Series("After the fit", index, [p.distance_after_angstrom for p in pairs]),
        ],
        x_label="fitted pair",
        y_label="distance (Å)",
        title="Distance between paired residues",
        height=280,
    )


def render(result) -> str:
    worst = sorted(
        (p for p in result.pairs if p.distance_after_angstrom is not None),
        key=lambda p: -p.distance_after_angstrom,
    )[:10]
    parts = [
        f"<p>{result.fitted_pairs} of {result.requested_pairs} residue pairs fitted "
        f"({esc(result.correspondence)}). RMSD of alpha carbons "
        f"<strong>{result.rmsd_after_angstrom:.3f} Å</strong> after the fit, "
        f"{result.rmsd_before_angstrom:.3f} Å before. The fit covers "
        f"{result.reference_coverage:.0%} of the reference and "
        f"{result.mobile_coverage:.0%} of the mobile query.</p>",
        _profile(result),
        structure_file(
            result.fitted_structure,
            caption="The mobile structure after superposition onto the reference.",
        ),
        "<p>Largest distances after the fit:</p>",
        table(
            ["Reference", "Mobile", "Residues", "After (Å)", "Before (Å)"],
            [
                [
                    p.reference,
                    p.mobile,
                    f"{p.reference_residue}/{p.mobile_residue}",
                    p.distance_after_angstrom,
                    p.distance_before_angstrom,
                ]
                for p in worst
            ],
            numeric={0, 1, 3, 4},
        ),
        f'<p class="sub">{esc(result.gemmi_version)}</p>',
    ]
    return "".join(parts)
