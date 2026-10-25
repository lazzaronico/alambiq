"""Fit measured candidate properties and check the fit on held-out candidates.

A candidate panel is a JSON file (`schema_version` 0.1) that lists 1 to 500
protein candidates. Each candidate has a unique `sample_id`, a `sequence` of
1 to 10,000 standard amino acids, the `sequence_sha256` digest of that
sequence, a `group_id` and a `partition`. The `sample_id` and the `group_id`
contain no whitespace and have at most 200 characters. A group is a set of
related candidates that share a `group_id`. The partition is `train`,
`evaluation` or `pool`; these are the training, evaluation and pool
partitions. Candidates in the pool partition are unmeasured.

The model is Bayesian ridge regression on standardized features, fitted on
the training partition alone. Features are either the sequence length and the
20 amino-acid fractions, or supplied pooled embeddings. The stored record
contains a prediction and a 95 % predictive interval for every candidate,
including the pool partition. On the evaluation partition the result reports the mean
absolute error, the root-mean-square error and R squared of the model and of a
baseline that predicts the training mean, and the coverage of the 95 %
predictive interval of the model.

The partitions are supplied in the candidate panel. The fit refuses a
candidate panel in which a group or an identical sequence of the evaluation
partition also appears in another partition, so that no evaluation candidate
shares a group or a sequence with a training or pool candidate. It also
refuses measured values for pool candidates, because the pool partition is
unmeasured.

Limits. The model is linear in the chosen features, so candidates with
identical features cannot be separated; the result counts the sets of such
candidates (`indistinguishable_groups`) and the stored record lists their
identifiers. The predictive
interval assumes Gaussian residuals with constant variance; its coverage is
measured separately on the evaluation partition and reported. The training
partition must contain at least 8 measured candidates from at least 3 groups,
and the evaluation partition at least 4 measured candidates from at least 2
groups. Below 30 evaluation observations the result carries a warning that the
estimates of performance and interval coverage rest on a small sample.
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
        Path to the candidate panel, a JSON file with the fields listed in the capability description.
    assay:
        Path to a stored `assay.inspect` record that holds the measurements.
    metric, condition_id, unit:
        The measured quantity, the condition and the unit. Only measurements with
        this metric and condition are fitted; a selected measurement in another
        unit is an error.
    observation_basis:
        `supplied-values` requires exactly one measured record per candidate,
        with no independent unit assigned; a candidate with no measured record is
        excluded and the stored record lists the reason. `independent-means` fits
        the mean over the independent units of each candidate (see
        `independent_unit` in `assay.inspect`). Default `supplied-values`.
    features:
        `sequence-composition-v1` uses the sequence length and the 20 amino-acid
        fractions. `pooled-embeddings` uses the supplied vectors instead. Default
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
    """Prediction error on one set of candidates: count, mean absolute error, root mean squared error and R squared."""

    n: int
    mae: float
    rmse: float
    r_squared: float | None


@dataclass(frozen=True, slots=True)
class CandidateFitResult:
    """The fit, its performance on the evaluation partition and the number of sets of candidates it could not separate."""

    trained_on: int
    evaluated_on: int
    evaluation_groups: int
    model: Performance
    baseline: Performance
    """The same measure for a prediction equal to the training mean for every candidate."""
    interval_95_coverage: float
    """Fraction of evaluation values inside their 95 % predictive interval."""
    mean_interval_95_width: float
    excluded: int
    indistinguishable_groups: int
    record: str
    """Path to the stored fit record, which contains every prediction and coefficient."""


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
