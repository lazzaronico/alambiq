"""Fit measured candidate properties and check the fit on held-out candidates.

The result reports the error of a Bayesian ridge regression on the evaluation partition. The stored record holds a prediction with a 95 % predictive interval for every candidate.

A candidate panel is a JSON file (`schema_version` 0.1) with 1 to 500 protein candidates. Each candidate has:

- a unique `sample_id`;
- a `sequence` of 1 to 10,000 standard amino acids, and its digest `sequence_sha256`;
- a `group_id`, which assigns related candidates to one group;
- a `partition`: `train`, `evaluation` or `pool`. Pool candidates are unmeasured.

A `sample_id` and a `group_id` have at most 200 characters and no whitespace.

The model is fitted on the training partition alone, on standardized features. The features are the sequence length and the 20 amino-acid fractions, or supplied pooled embeddings. The stored record includes predictions for the pool partition. The result reports these measures on the evaluation partition, for the model and for a baseline that predicts the training mean:

- the mean absolute error;
- the root-mean-square error;
- R squared.

It also reports the coverage of the 95 % predictive interval of the model.

The panel supplies the partitions. The fit refuses a panel in which a group or an identical sequence of the evaluation partition also appears in another partition. It also refuses measured values for pool candidates.

Limits. The model is linear in the features, so candidates with identical features cannot be separated. The result counts the sets of such candidates (`indistinguishable_groups`) and the stored record lists their identifiers.

- The predictive interval assumes Gaussian residuals with constant variance. Its coverage is measured on the evaluation partition and reported.
- The training partition needs at least 8 measured candidates from at least 3 groups. The evaluation partition needs at least 4 measured candidates from at least 2 groups.
- With fewer than 30 evaluation observations, the result carries a warning that the estimates of performance and interval coverage rest on a small sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.engineer.candidate_fit_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class CandidateFitRequest:
    """The candidate panel, the measurements and the model settings of a candidate fit.

    panel:
        Path to the candidate panel, a JSON file with the fields in the capability description.
    assay:
        Path to a stored `assay.inspect` record that holds the measurements.
    metric, condition_id, unit:
        The measured quantity, the condition and the unit. Only measurements with
        this metric and condition are fitted. A selected measurement in another unit
        is an error.
    observation_basis:
        `supplied-values` requires exactly one measured record per candidate, with no
        independent unit assigned. A candidate with no measured record is excluded,
        and the stored record lists the reason. `independent-means` fits the mean over
        the independent units of each candidate (see `independent_unit` in
        `assay.inspect`). Default `supplied-values`.
    features:
        `sequence-composition-v1` uses the sequence length and the 20 amino-acid
        fractions. `pooled-embeddings` uses the supplied vectors. Default
        `sequence-composition-v1`.
    embeddings:
        Path to a stored embedding record whose identifiers and sequences match
        those of the candidate panel. Required for `pooled-embeddings` features
        and accepted only with them.
    """

    panel: str
    assay: str
    metric: str
    condition_id: str
    unit: str
    observation_basis: Literal["supplied-values", "independent-means"] = (
        "supplied-values"
    )
    features: Literal["sequence-composition-v1", "pooled-embeddings"] = (
        "sequence-composition-v1"
    )
    embeddings: str | None = None


@dataclass(frozen=True, slots=True)
class Performance:
    """Prediction error on one set of candidates: count, mean absolute error, root-mean-square error and R squared."""

    n: int
    mae: float
    rmse: float
    r_squared: float | None


@dataclass(frozen=True, slots=True)
class CandidateFitResult:
    """The fit, its performance on the evaluation partition, and the number of sets of candidates that it cannot separate."""

    trained_on: int
    evaluated_on: int
    evaluation_groups: int
    model: Performance
    baseline: Performance
    """The same measures for a prediction equal to the training mean for every candidate."""
    interval_95_coverage: float
    """Fraction of evaluation values inside their 95 % predictive interval."""
    mean_interval_95_width: float
    excluded: int
    indistinguishable_groups: int
    record: str
    """Path to the stored fit record: every prediction and coefficient."""


def _performance(values: dict) -> Performance:
    return Performance(
        n=values["n"],
        mae=values["mae"],
        rmse=values["rmse"],
        r_squared=values["r_squared"],
    )


@capability(
    "candidate.fit",
    version="1.0",
    family="engineer",
    summary="Fit measured candidate properties and check the fit on held-out candidates",
    render=render,
    presets={
        "sequence-composition": {"features": "sequence-composition-v1"},
        "embeddings": {"features": "pooled-embeddings"},
    },
)
def fit(request: CandidateFitRequest, ctx: StepContext) -> CandidateFitResult:
    """Fit on the training partition and measure the fit on the evaluation partition."""
    from alambiq.capabilities.engineer.science.candidates import fit_candidates

    parameters = {
        "metric": request.metric,
        "condition_id": request.condition_id,
        "unit": request.unit,
        "observation_basis": request.observation_basis,
        "features": request.features,
    }
    record = fit_candidates(
        ctx.read_json(request.panel),
        ctx.read_json(request.assay),
        parameters,
        embeddings=(
            ctx.read_json(request.embeddings)
            if request.embeddings is not None
            else None
        ),
    )

    for message in record["diagnostics"]:
        ctx.note(message)
    evaluation = record["evaluation"]
    aliases = record["source"]["representation"]["indistinguishable_candidates"]
    return CandidateFitResult(
        trained_on=len(record["training"]),
        evaluated_on=evaluation["model"]["n"],
        evaluation_groups=evaluation["groups"],
        model=_performance(evaluation["model"]),
        baseline=_performance(evaluation["baseline"]),
        interval_95_coverage=evaluation["model"]["interval_95_coverage"],
        mean_interval_95_width=evaluation["model"]["mean_interval_95_width"],
        excluded=len(record["excluded"]),
        indistinguishable_groups=len(aliases),
        record=ctx.keep_json(record),
    )
