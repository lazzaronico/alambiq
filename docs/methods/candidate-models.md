# Candidate models

A candidate model predicts a measured protein property. It is used to choose the next
candidates to make. `candidate.fit` fits the model and `candidate.select` chooses
candidates.

The model is Bayesian ridge regression on standardized features, fitted on the training
candidates alone, with Gaussian predictive intervals.

A panel is a set of protein candidates. Each candidate has a sample id, a sequence, a
group id that assigns related candidates to one group, and a partition label.

## Panel partitions

The partition label of a candidate is `train`, `evaluation` or `pool`. The researcher
assigns candidates to partitions and groups.

- **Training** fits the model and the feature standardization.
- **Evaluation** measures error and interval coverage.
- **Pool** holds the candidates that have not been measured and are eligible to be made
  next.

Alambiq refuses a panel unless every group and every sequence in the evaluation
partition appears in neither training nor the pool. A group can appear in both training
and the pool.

Pool candidates carry no measured outcomes, because a pool prediction made from the
measured value of the same candidate is not a prediction. Alambiq refuses a panel that
supplies one.

A fit requires at least 8 measured training candidates in 3 groups and at least 4
measured evaluation candidates in 2 groups. Failed, missing and censored measurements are
excluded from the target values (the measured property) with their reasons. Only measured
values count toward these minimums.

The researcher chooses evaluation groups to match the intended use of the model. Holding
out random candidates tests interpolation among variants whose positions appear in
training. Holding out whole mutation positions or whole families tests generalization to
positions or families absent from training. The result records the partition used.

## Model performance

The training-mean baseline is the RMSE on the evaluation set of a predictor that returns
the mean of the training responses for every candidate. A model improves on the baseline
when its evaluation RMSE is lower than the baseline RMSE, whatever its R². Otherwise the
result adds a diagnostic. The result reports both errors.

The result also reports the evaluation error, the residuals and the observed coverage of
the 95 % predictive interval. The observed coverage on the evaluation set is the fraction
of evaluation candidates whose measured response lies inside its 95 % predictive
interval. The predictive standard deviation comes from the fitted model, so the interval
depends on the model's assumptions. When the observed coverage is below 95 %, the result
adds a diagnostic, and the observed coverage, not the nominal level, describes the
reliability of the interval.

Evaluation sets with fewer than 30 candidates are flagged in the result, because the
estimates of error and interval coverage are based on a small sample. The number of
evaluation groups, not the number of variants, is the number of independent tests of
generalization.

## Features

Sequence composition (length and the 20 amino-acid fractions) is the default feature set
(`sequence-composition-v1`). Distinct variants at different positions can have identical
composition, and the result lists these sets of candidates. A representation that cannot
distinguish two candidates predicts the same value for both.

A single component of a pooled embedding from a protein language model has no assigned
meaning. The vector as a whole and the distances between vectors are used. Vectors are
comparable within one model, revision, layer and pooling. `candidate.fit` and
`candidate.select` each read one embedding record and do not check it against the record
used in another step. Merging several records into one panel refuses records that differ
in model, revision, layer, pooling, input tracks or dimension.

Measured responses are excluded from feature construction. A feature derived from the
quantity being predicted gives low error only on candidates whose response is known.

## Batch selection

`candidate.select` selects from the pool only. It selects greedily: one candidate at a
time, the one with the highest selection value at each step, in an order fixed by a seed.
The request states the constraints: an eligibility list, a cap per group, and a budget
against per-candidate costs.

`strategy` selects one of:

- `random`: takes the eligible pool candidates in the order of a shuffle fixed by the
  seed; requires no fitted model.
- `diverse`: selects the candidate farthest from the nearest candidate already chosen,
  without reference to predicted performance. The distance is set by `diversity`:
  `sequence` (the fraction of positions that differ, which requires proteins of equal
  length) or `embedding-cosine`.
- `highest-predicted`, `lowest-predicted` and `uncertain`: rank candidates by the
  predicted response or by its predictive standard deviation. They require a fitted model.
- `highest-predicted-diverse`: combines the predicted response with the distance from the
  candidates already chosen. It requires a fitted model and `diversity`.
- `pareto`: ranks candidates by non-dominated fronts over 2 to 5 fitted objectives, each
  maximized or minimized; within a front, the objectives scaled 0 to 1 are added with
  equal weight.

A ranking from a fitted model is limited by the evaluation error of that model, which the
result reports with the baseline and the coverage.

Every candidate that is not chosen is recorded with the reason. A batch that cannot be
filled is reported as unfilled. The recorded reasons distinguish a batch truncated by the
budget from a batch truncated by group caps.

A seeded selection returns the same batch for the same inputs.

## Measurement rounds

A round is one cycle of fitting, selection and measurement. When new measurements are
available, the researcher moves the measured candidates from the pool to training in the
panel and refits the model. The evaluation partition and the observation basis
(`observation_basis`) stay unchanged across rounds. Moving the evaluation candidates of
one round to training in the next is a change of the panel and is recorded in it. To test
the loop on a published dataset, exclude the responses of unselected candidates from the
model inputs.

## Limits

The model estimates a linear association between the standardized features and the
response of the training candidates.

Predictions for the pool assume that pool candidates and training candidates have the
same relation between features and response. The evaluation partition tests this
assumption for the variation present between the training and evaluation groups.

Selection uses the supplied predictions and constraints (the eligibility list, the cap per
group and the budget against per-candidate costs). Synthesis feasibility, expression,
toxicity and laboratory capacity enter through the eligibility list and the costs that
the researcher supplies. A named candidate is a proposal for the researcher to decide on.
