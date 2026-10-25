"""How a panel evaluation appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table

METHODS = (
    "training-mean",
    "nearest-reference",
    "sequence-ridge",
    "reaction-ridge",
    "joint-ridge",
)


def render(result) -> str:
    endpoint = result.endpoint
    counts = ", ".join(f"{k} {v}" for k, v in sorted(result.observation_counts.items()))
    parts = [
        f"<p>Panel {esc(result.panel_id)}: endpoint {esc(endpoint.get('metric', ''))} "
        f"({esc(endpoint.get('unit', ''))}). Observations: {esc(counts)}.</p>"
    ]
    for split in result.splits:
        parts.append(
            f"<h3>{esc(split.id)}: {esc(split.novelty)} novelty</h3>"
            f"<p>{split.training_groups} training and {split.evaluation_groups} evaluation "
            "pair groups.</p>"
        )
        parts.append(
            table(
                ["Predictor", "n", "MAE", "RMSE", "Bias", "Spearman ρ"],
                [
                    [
                        m,
                        split.metrics[m].n,
                        split.metrics[m].mae,
                        split.metrics[m].rmse,
                        split.metrics[m].bias,
                        "—"
                        if split.metrics[m].spearman_rho is None
                        else split.metrics[m].spearman_rho,
                    ]
                    for m in METHODS
                ],
                numeric={1, 2, 3, 4, 5},
            )
        )
    return "".join(parts)
