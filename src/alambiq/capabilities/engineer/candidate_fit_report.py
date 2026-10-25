"""How a candidate fit appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import number, read_record, table


def render(result) -> str:
    unit = ((read_record(result.record) or {}).get("request") or {}).get("unit")
    suffix = f" ({unit})" if unit else ""
    rows = [
        [
            "This model",
            result.model.n,
            result.model.mae,
            result.model.rmse,
            result.model.r_squared,
        ],
        [
            "Training mean",
            result.baseline.n,
            result.baseline.mae,
            result.baseline.rmse,
            result.baseline.r_squared,
        ],
    ]
    return "".join(
        [
            table(
                [
                    "Predictor",
                    "Candidates",
                    f"Mean absolute error{suffix}",
                    f"RMSE{suffix}",
                    "R²",
                ],
                rows,
                numeric={1, 2, 3, 4},
            ),
            '<p class="sub">'
            f"Fitted on {result.trained_on} candidates; evaluated on "
            f"{result.evaluated_on} held-out candidates in "
            f"{result.evaluation_groups} groups · fraction of evaluation values "
            f"inside the 95 % predictive interval "
            f"{number(result.interval_95_coverage)}, mean interval width "
            f"{number(result.mean_interval_95_width)}{suffix} · "
            f"{result.excluded} measurements excluded</p>",
            *(
                [
                    '<p class="sub">'
                    f"{result.indistinguishable_groups} sets of candidates have "
                    "identical features.</p>"
                ]
                if result.indistinguishable_groups
                else []
            ),
        ]
    )
