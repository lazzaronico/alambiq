# Operation from progress curves

**Question.** What time course does a fitted rate law predict for a hydrogel-assembled
enzyme, given measured progress curves?

The data are the progress of acetophenone reduction by free and hydrogel-assembled TbSADH
(Chen et al., 2023). `operation.fit` fits an effective first-order law and a saturation law
to the time points of each preparation from 60 to 240 s. It evaluates each fit on the time
points from 300 to 480 s. `operation.simulate` then simulates the fitted hydrogel
operation.

```bash
alambiq run examples/operation-from-progress-curves/campaign.yaml --report report.html
```

**Needs.** The `quantitative` extra.

**What to look at.** The standardized RMSE is the root-mean-square of the residuals divided
by the observation scale of 0.01 (one percentage point of conversion). The report lists it
for the fitted points (60 to 240 s) and for the evaluation points (300 to 480 s):

- free enzyme, first-order law: 0.87 and 1.44; saturation law: 0.87 and 1.44, with Km at
  the upper bound of 1 M, so Km takes the bound value;
- hydrogel, first-order law: 1.95 and 5.48; saturation law: 1.00 and 1.85, with Km 6.2 mM
  and capacity 0.050 mM/s (`capacity_M_s`, see
  [operation formats](../../docs/operation-formats.md)).

The report notes that Km of the free-enzyme saturation fit equals the upper bound. For the
hydrogel saturation fit it gives rank 2 (the number of parameters) and a scaled condition
number of 59. For the free-enzyme saturation fit the scaled condition number is 2.8e4. The
simulation of the fitted hydrogel operation ends at 480 s with 0.224 mM acetophenone and
4.776 mM converted substrate (sum 5 mM).

**Limits.** Each saturation fit has 4 fitted points and 2 parameters. A fitted value is an
estimate when the rank equals the number of parameters and the parameter is inside its
bounds (`docs/methods/operation.md`). The scaled condition number and the profiles show how
well the data determine the parameters. Each definition file sets `uncertainty` to
`descriptive`, so the fits report the diagnostics and the profiles. The fits describe one
assay at fixed conditions (5 mM substrate, 1 mg/mL enzyme, pH 7.4, 30 °C), and predictions
apply to these conditions. The fit uses the means of the triplicates without the source
error values, with a fixed residual scale of one percentage point.

**Data.** Chen et al., *Nature Communications* (2023), doi:10.1038/s41467-023-37921-y,
Fig. 3d1 source data, CC BY 4.0. `source-measurements.json` stores the cells; the fit files
contain the derived means as conversion fractions of the initial 5 mM acetophenone.
