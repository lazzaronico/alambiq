# Measured variant rounds

**Question.** Given measurements of variants, which variants does each selection strategy
choose for the next batch?

The data are published thermostability (T50) measurements of *Bacillus subtilis* lipase A
variants. `assay.inspect` reads the 185 measured rows. `candidate.fit` fits a Bayesian ridge
regression on the sequence length and the 20 amino-acid fractions of each variant (the
composition features). It fits the 82 training variants and evaluates the model on 103
held-out variants. `candidate.select` runs 3 strategies on the pool of unmeasured variants:
highest predicted value, highest predictive uncertainty and random. Each strategy selects 8
variants. The two model-based strategies select at most one variant per position.

```bash
alambiq run examples/measured-variant-rounds/campaign.yaml --report report.html
```

**Needs.** The `quantitative` extra.

**What to look at.** The model has a larger error than the baseline of predicting the
training mean. The evaluation RMSE is 2.93 °C for the model and 2.39 °C for the training
mean. The 95 % intervals of the model cover 81.6 % of the evaluation points (nominal
coverage 95 %). The `model` step and the two model-based selection steps
(`highest-predicted` and `most-uncertain`) state this result in a note. The positions are
partitioned among training, evaluation and pool, so the evaluation set contains positions
that are absent from the training set. A model on composition features has no per-position
parameter.

**Limits.** Variants with the same amino-acid counts have identical composition features:
268 of the 384 variants of the panel share their counts with at least one other variant, in
104 groups of identical counts. Variants with identical counts receive identical predictions
and identical predictive standard deviations, and the seed resolves the ties between them.
T50 is a thermostability measure, so selection by T50 selects for thermostability.

**Data.** ProteinGym 1.3, ESTA_BACSU_Nutschel_2020 (MIT); DOI 10.1021/acs.jcim.9b00954.
Positions use the 212-residue precursor numbering. `panel.json` lists the 384 variants of
the panel. A hash of the position of each variant sets its partition (train, evaluation or
pool), independent of the measurements. The 185 measured variants are the 82 training and
103 evaluation variants. The 199 pool variants have no measurement.
