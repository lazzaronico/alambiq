# Connected investigation

**Question.** Starting from a protein, which unmeasured variant has the highest predicted
activity, and how does the simulated operation of that variant compare with a generated
progress curve?

The campaign has 10 steps:

| Step | Capability | Role |
| --- | --- | --- |
| `first-look` | `protein.inspect` | reads the sequence of hen egg-white lysozyme |
| `propose` | `variant.propose` | selects 24 substitutions by language-model score, excluding the catalytic residues Glu35 and Asp52, with at most one substitution per position |
| `round` | `assay.inspect` | reads the returned round: the generated activities of 18 of the 24 variants |
| `model` | `candidate.fit` | fits activity on the sequence length and the 20 amino-acid fractions (the composition features) and checks it on held-out variants |
| `next-batch` | `candidate.select` | selects the unmeasured variant with the best predicted activity |
| `kinetics-intake`, `kinetics` | `assay.inspect`, `assay.fit` | Michaelis-Menten constants of variant S100R, the variant that `next-batch` selects, from rates at 10 substrate concentrations |
| `define` | `operation.define` | converts the fitted constants into a saturation rate law (maximum rate and scale) for a 0.5 L reactor that contains 10 mg of enzyme |
| `simulate` | `operation.simulate` | integrates the reaction over 1 h |
| `compare` | `operation.compare` | compares the simulation with a generated progress curve |

```bash
alambiq run examples/connected-investigation/campaign.yaml --report report.html
```

**Needs.** The `quantitative` extra. No step requires an engine environment.

`scores.json` is the stored result of `protein.score_substitutions` (ESM C 300M), run once
on the lysozyme sequence. The campaign reads it as an input and does not run that
capability.

## Data provenance

The sequence is that of hen egg-white lysozyme. ESM C 300M computes the language-model
scores. `prepare.py` generates the returned round, the rates and the progress curve from a
stated model with fixed seeds, so that the fitted values can be compared with the generating
values in `truth.json`:

- activity = 1 + 0.15 × (change in the number of arginines) + noise, with a Gaussian noise
  standard deviation of 0.015;
- the variant the model selects (S100R) has a Vmax of 1.15 × 180 µM/min = 207 µM/min (1.15
  is the activity of S100R and 180 µM/min the base Vmax) and a Km of 2.5 mM, generated with
  3 % noise;
- the progress curve is the integrated Michaelis-Menten solution for those constants, with
  1.5 % noise.

The arginine rule is a synthetic rule that a model on the composition features can learn.
To use other data, replace `returned.csv`, `rates.csv` and `progress.json` with files of
the same layout. Then update `panel.json` (sample ids and partitions) and the `sample_id` of
the `kinetics` step to the variant that `next-batch` selects, and rerun `propose` to
select the variants of a new round.

**What to look at.** Each step reports the following.

- **`propose`**: the ranking uses the language-model score and no structure. See
  [proposing variants](../../docs/methods/variant-proposal.md) for the rank correlation of
  that score with measured variants of real enzymes, including the amine transaminase
  lineage for which the ranking is inverted.
- **`model`**: the root-mean-square error on the held-out variants and, for comparison, the
  error of predicting the training mean. The report states that 2 candidates have identical
  features. The stored record names them (K97R and K116R, which have the same amino-acid
  counts), and the model assigns them the same prediction.
- **`kinetics`**: the fitted Km and Vmax with their 95 % intervals. Both intervals contain
  the generating values (207 µM/min and 2.5 mM).
- **`define`**: the capacity of the channel `hydrolysis` (see
  [operation formats](../../docs/operation-formats.md)), computed from Vmax, the enzyme
  concentration of the assay (20 mg/L) and the 10 mg of enzyme in 0.5 L. The template
  declares no dependence of the rate on temperature or pH, so the simulation applies the
  fitted rate at the assay temperature and pH. The campaign assumes that these equal the
  temperature and pH of the reactor.
- **`compare`**: the root-mean-square difference between the simulated and the generated
  substrate concentration, in M.

**Limits.** In an experimental investigation the proposed variants are measured, and the
steps after `propose` use those measurements. In this example the measurements are
generated. The language-model score ranks sequences by similarity to natural proteins. The
activity of a variant is known only from the generated round, which follows the arginine
rule. The model on the composition features is fitted to 12 variants and evaluated on 6
held-out variants; `model` reports the held-out error.
