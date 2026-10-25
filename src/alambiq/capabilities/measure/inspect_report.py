"""How a measurement record appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import table


def render(result) -> str:
    rows = [
        ["Measured", result.measured],
        ["Missing", result.missing],
        ["Failed", result.failed],
        ["Censored at a limit", result.censored],
        ["Total readings", result.measurements],
    ]
    unit = result.independent_unit or "none declared"
    samples, conditions = len(result.samples), len(result.conditions)
    return "".join(
        [
            table(["Reading", "Count"], rows, numeric={1}),
            f'<p class="sub">{samples} sample{"" if samples == 1 else "s"} across '
            f"{conditions} condition{'' if conditions == 1 else 's'} · "
            f"independent unit: {unit}</p>",
        ]
    )
