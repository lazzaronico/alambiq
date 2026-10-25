# Measurements

Reading of a measurement table by `assay.inspect`, fitting of a response model, and
projection of a fit onto a batch target. The capabilities are `assay.inspect`,
`assay.fit` and `assay.balance`. `alambiq capability show <id>` prints the request, result
and limits of each.

## Measured rows

A row holds a value as the source recorded it. The value can be a raw instrument reading
or an endpoint the source already computed. Alambiq stores both alike, so the
description of the condition states which applies.

A value is measured, missing, failed or censored. The last three states are stored
separately from measured values:

- A **missing** value has no entry in the value column.
- A **censored** value is known only relative to a stated limit; it is recorded with its
  status and the limit. Calculated means exclude it.
- A **failed** measurement was attempted and produced no value. It remains in the table
  with its status.

Roles (sample, reference, blank, control) come from the design of the originating study
and are stored as given. Conditions are analyzed separately: two conditions are two
experiments until the researcher states otherwise.

## Replicates and independence

The replicate structure determines which quantities the data can estimate, so
`assay.inspect` takes it as a declared input.

An independent unit is the smallest item that is prepared independently, for example a
separate preparation, culture or animal. Independent units are the replicates. Technical
readings are repeated values from one independent unit, for example repeated reads of a
well or lanes of one library. Technical readings are averaged within their unit first,
and each unit's mean has equal weight. A set of 20 reads of one preparation is one
observation.

The mean and standard deviation of the unit means of a group are reported only when at
least 2 independent units were observed and every observed value of the group has an
`independent_id`. For published summary values without replicate identities, the
identities are left blank. A summary row is one value without a measured replicate
precision.

An error reported by the source is stored as reported and shown beside the data. The
means and intervals that Alambiq computes come from the observed values, and the
reported error is not used as a weight. A standard deviation, a confidence limit and a
standard error are different quantities; the column `reported_standard_error` holds a
standard error.

## Response models

`assay.fit` fits one of five models, selected with `model`: `linear`, `michaelis_menten`
(the Michaelis-Menten initial-rate relation, the default) and three models of how an
enzyme responds to conditions:

- `arrhenius`: a rate constant against temperature; reports the rate constant at the
  reference temperature, the activation energy in kJ/mol and the pre-exponential factor.
- `first_order_decay`: activity remaining against time; reports a rate constant and a
  half-life.
- `ph_bell`: a bell-shaped profile over pH; reports two pKa values, the optimum pH and
  the activity at the optimum.

The researcher chooses the model from the relation the experiment expresses. A
Michaelis-Menten curve fitted to points that do not approach saturation returns a Km whose
standard error is large relative to Km, which means the data do not determine it. A fitted
Km outside the measured concentration range produces a diagnostic in the result for the
same reason: the data do not bracket half-saturation.

Every model requires at least 4 included observations at 3 distinct values of the
independent quantity. Included observations are those not listed in `exclude`; under
`independent-means`, each pair of an independent unit and a condition counts as 1
observation, the mean of that unit's technical readings in that condition.

Michaelis-Menten reports Km and Vmax; kcat requires an independently established
concentration of active enzyme, which a table of rates does not contain. `assay.fit`
fits a response against one independent quantity (`x_quantity`) with one of the five
models; Michaelis-Menten applies to a response that rises toward saturation. A whole
progress curve is fitted with `operation.fit`.

The three condition models (`arrhenius`, `first_order_decay` and `ph_bell`) report
quantities derived from the coefficients, such as the half-life, with the uncertainty
propagated by the delta method. Arrhenius states its rate constant at a reference
temperature. By default the reference temperature is the mean of the measured
temperatures. The bell-shaped profile describes two ionizations and applies to data
without a plateau, a shoulder or a third ionizing group. If a
step other than the chemical conversion limits the rate, the fitted activation energy is
that of the limiting step. The interval at an extrapolated temperature or pH widens with
the distance from the measured range.

A condition quantity other than `x_quantity` that differs between the included
conditions, or is given for only some of them, is outside every model. The fit refuses such
data unless the quantity is named in `pool_over` or the conditions are restricted with
`condition_ids`. A pooled quantity is listed in a diagnostic of the result with the
statement that its effect is not modeled.

## Uncertainties

The fit stores three uncertainties; a fourth is obtained separately. They are different
quantities.

- **Coefficient standard error** quantifies how precisely the data determine a
  coefficient. The result also gives a 95 % interval for each coefficient.
- **Mean-response interval**: the 95 % interval for the expected response at a given
  value of the independent quantity.
- **Future-observation interval**: the 95 % interval for one new measurement at that
  value. It is wider than the mean-response interval.
- **Measured predictive error** compares predictions with observations excluded from the
  fit. Of the four, it is the only one that is a measurement and not a consequence of the
  model's assumptions. The researcher obtains it by excluding
  observations with `exclude`, fitting the remainder, and comparing the prediction with the
  excluded observations.

The coefficient standard error and the intervals are conditional on the model form being
correct; the measured predictive error applies to the range of the excluded observations.

Intervals are computed from the residuals by default (`uncertainty: residual`), which
assumes independent residuals. `uncertainty: cluster_bootstrap` resamples whole independent
units; it is valid when the data have replicate structure and those units are the sampling
units. It requires the `independent-means` basis, at least 4 independent units, a `seed` and
`resamples` from 100 to 2,000 (default 1,000); its intervals are the 2.5 % and 97.5 %
percentiles of the resamples. The bootstrap gives coefficient and mean-response intervals; a
fit with `uncertainty: cluster_bootstrap` has no future-observation interval, which the
residual method provides. If any resample fails, the intervals are withheld.

A prediction outside the measured range has `extrapolation: true` in the stored record,
and the result carries a note that gives the number of such predictions and the measured
range. The intervals at such a prediction assume that the model form holds outside the
range.

## Projection onto a batch target

`assay.balance` computes the conversion that the product target requires and the
volumetric activity that achieves that conversion in the stated time, by integrating
irreversible single-substrate depletion at constant activity and volume.

The inputs are of two kinds. The fitted coefficients are estimates from the measurements.
The batch (substrate, volume, duration, product target, and the formation, recovery and
accepted fractions) is a scenario chosen by the researcher. Overrides of the scenario
inputs (the `sensitivity` scenarios) give the sensitivity of the result to those inputs;
how well the kinetics were measured is reported by the fit that supplies the
coefficients.

A fit used by a balance states its rate basis: the species the assay measured, the
substrate it was measured against, and the moles of measured species formed per mole of
substrate consumed, including any reporter yield. Activities are comparable when their
bases match and their units are in one dimension.

## Limits

`assay.inspect` reads the given values and refuses a table that lacks a required column,
has a duplicate `measurement_id`, has a `status` inconsistent with `value`, or exceeds
10,000 rows (the requirements are in `alambiq capability show assay.inspect`). The design
of the experiment, the choice of controls and the computation of endpoints are taken as
stated in the table. Values are used as recorded, without blank subtraction, control
normalization, outlier removal or significance testing. `assay.inspect` converts units
only on request and only within one dimension: concentration, rate, time, temperature,
mass concentration or specific activity. A calibration, a molecular weight or an
active-enzyme amount changes the basis of a number and is a separate input supplied by the
researcher.

A fit is conditional on the recorded experiment and the chosen model. Its intervals
assume the model is correct and, by default, that residuals are independent.

The batch balance assumes that the kinetics of the assay hold at the batch conditions:
irreversible depletion at constant activity and constant volume, with product inhibition,
enzyme decay and transport limits absent. Its result is the arithmetic consequence of the
fit and the scenario. The required activity is the Vmax at which the integrated depletion
reaches the required conversion at the stated duration. A scenario has a `limitation`
message and no required Vmax when the accepted fraction is 0, when the target exceeds the
accepted product at full conversion (conversion above 1), or when the conversion is
exactly 1 and Km is positive (infinite time).
