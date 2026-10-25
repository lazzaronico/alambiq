"""How a propagated uncertainty appears in the dossier."""

from __future__ import annotations

from alambiq.report.charts import Series, line_chart
from alambiq.report.page import esc, table


def _chart(result) -> str:
    """For each species, the median and the outer quantiles against time."""
    out = []
    for name in dict.fromkeys(s.species for s in result.spreads):
        rows = [s for s in result.spreads if s.species == name]
        if len(rows) < 2:
            continue
        levels = sorted(rows[0].quantiles_M, key=float)
        low, high = levels[0], levels[-1]
        mid = min(levels, key=lambda q: abs(float(q) - 0.5))
        t = [r.time_s for r in rows]
        out.append(
            line_chart(
                [
                    Series(
                        f"{float(low):g} to {float(high):g} quantiles",
                        t,
                        [r.quantiles_M[mid] for r in rows],
                        mode="band",
                        band=(
                            [r.quantiles_M[low] for r in rows],
                            [r.quantiles_M[high] for r in rows],
                        ),
                        color="#56b4e9",
                    ),
                    Series("Nominal prediction", t, [r.nominal_M for r in rows]),
                ],
                x_label="time (s)",
                y_label=f"{name} (M)",
                title=f"Predicted {name} with the uncertainty of the fit",
                height=300,
            )
        )
    return "".join(out)


def render(result) -> str:
    levels = sorted(result.spreads[0].quantiles_M, key=float) if result.spreads else []
    rows = [
        [s.species, s.time_s, s.nominal_M, s.mean_M, s.standard_deviation_M]
        + [s.quantiles_M[q] for q in levels]
        for s in result.spreads
    ]
    coefficients = [
        [c.name, c.value, c.unit, c.draw_mean, c.draw_standard_deviation]
        for c in result.coefficients
    ]
    return "".join(
        [
            _chart(result),
            table(
                ["Coefficient", "Fitted", "Unit", "Draw mean", "Draw SD"],
                coefficients,
                numeric={1, 3, 4},
            ),
            table(
                ["Species", "Time (s)", "Nominal (M)", "Mean (M)", "SD (M)"]
                + [f"{float(q):g} quantile (M)" for q in levels],
                rows,
                numeric=set(range(1, 5 + len(levels))),
            ),
            f'<p class="sub">{result.samples} coefficient sets drawn from a '
            f"{esc(result.distribution)} distribution of the fit; "
            f"{result.discarded_draws} draws discarded</p>",
        ]
    )
