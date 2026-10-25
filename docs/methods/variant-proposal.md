# Variant proposal

Shortlisting of substitutions of a protein to build and measure, and a benchmark of the
ranking against measured variants. The capabilities are `protein.score_substitutions`,
`variant.propose`, `protein.predict_stability_change` (which supplies the stability
evidence) and `candidate.select` (which chooses among candidates of a stored panel). The first three are
applied in sequence: `protein.score_substitutions` scores every substitution with a
protein language model; `variant.propose` forms a shortlist by language-model score;
`protein.predict_stability_change` predicts the stability of the shortlist from a
structure; `variant.propose` ranks the shortlist again with `ranking: combined`.
`candidate.select` is independent of this sequence.

## Scoring and ranking

`protein.score_substitutions` runs ESM C 300M once per position with that residue hidden and
records the log-probability that the model assigns to each canonical residue there. The
log-likelihood ratio of a substitution is the log-probability of the mutant minus that of the
wild type at the same masked position (the masked-marginal score of Meier and colleagues,
2021). A positive value means that the model assigns the mutant a higher probability than the
residue the protein has. The score evaluates one position at a time, with the rest of the
sequence as given; it applies to single substitutions. The model was trained on natural
sequences, so the score ranks substitutions by their fit to natural variation.

`protein.predict_stability_change` takes a backbone and predicts, for each substitution named
in the request, the change in folding free energy; a positive value is destabilizing. Its
limits are given in [Stability of substitutions](embeddings-and-prediction.md#stability-of-substitutions);
its benchmark on a beta-glucosidase did not meet the 2 criteria set for it: the lower end
of the 95 % interval of the Spearman correlation had to exceed 0, and the enrichment among
the 10 highest-ranked variants had to exceed the random expectation at a one-sided
hypergeometric probability of at most 0.05
(`environments/thermompnn/BGLB_BENCHMARK.md`).

`variant.propose` ranks substitutions by the language-model score from
`protein.score_substitutions` and, optionally, by the predicted stability change from a
stored `protein.predict_stability_change` result. Optionally it also counts, from an
alignment of homologs, how many homologs carry the mutant residue at each position. The
request selects the ranking: the language-model score, the predicted stability, or the mean
of the percentile ranks of a candidate on both. The combined rank is the mean of the two
percentile ranks, with fixed equal weights. The request sets these filters: a minimum
language-model score, a maximum destabilization, a restriction to residues that the
homologs show, a maximum number per position, and the positions to consider or exclude.
Catalytic residues are excluded by listing their positions as excluded. The result counts
each excluded candidate under the reason for its exclusion.

`variant.propose` returns the shortlist as a request for `protein.predict_stability_change`.
The steps of the sequence are: shortlist by language-model score (one model pass per
position), predict the stability of the shortlist (this requires a structure and an engine
environment), then call `variant.propose` again with `ranking: combined`.

## Benchmark on measured variants

The benchmark in `tools/benchmark_variant_proposal.py` scores measured variants with
`protein.score_substitutions` and compares the ranking with the measurements. It was run
once, without tuning, and the results are in
[`variant-proposal-benchmark.json`](variant-proposal-benchmark.json). Two baselines are
compared with the model: a random ranking (the mean and standard deviation over 1,000
seeded shuffles) and the BLOSUM62 score of the substitution, which prefers substitutions
to residues similar to the replaced residue and does not use a model. The last column gives
how many of the top tenth of variants by score have a measurement in the best quarter.
For n variants, the top tenth is the max(5, round(0.1 n)) variants with the highest scores
(this count is the denominator). A hit is a variant whose measurement is at or above the
ceil(0.25 n)-th highest measurement of the dataset. A random ranking is expected to find
the best quarter in 25 % of the top tenth; the column gives the observed share as a
multiple of 25 %, so a random ranking scores 1.0x.

| Dataset | Variants | Method | Spearman (95 % interval) | Top tenth: best quarter found |
| --- | ---: | --- | --- | --- |
| Bacillus subtilis Lipase A, T50 (Nutschel et al. 2020) | 2,172 | ESM C 300M log-likelihood ratio | +0.37 (+0.33 to +0.40) | 98 of 217 (1.8x chance) |
| Bacillus subtilis Lipase A, T50 (Nutschel et al. 2020) | 2,172 | BLOSUM62 score of the substitution | +0.11 (+0.07 to +0.15) | 81 of 217 (1.5x chance) |
| Bacillus subtilis Lipase A, T50 (Nutschel et al. 2020) | 2,172 | random ranking | -0.00 (SD 0.02) | 1.0x chance |
| Human glucokinase, activity (Gersing et al. 2023, 2024) | 190 | ESM C 300M log-likelihood ratio | +0.58 (+0.48 to +0.66) | 11 of 19 (2.3x chance) |
| Human glucokinase, activity (Gersing et al. 2023, 2024) | 190 | BLOSUM62 score of the substitution | +0.21 (+0.08 to +0.34) | 6 of 19 (1.3x chance) |
| Human glucokinase, activity (Gersing et al. 2023, 2024) | 190 | random ranking | +0.00 (SD 0.07) | 1.0x chance |
| Human glucokinase, abundance (Gersing et al. 2023, 2024) | 186 | ESM C 300M log-likelihood ratio | +0.37 (+0.24 to +0.50) | 8 of 19 (1.7x chance) |
| Human glucokinase, abundance (Gersing et al. 2023, 2024) | 186 | BLOSUM62 score of the substitution | +0.18 (+0.04 to +0.31) | 4 of 19 (0.8x chance) |
| Human glucokinase, abundance (Gersing et al. 2023, 2024) | 186 | random ranking | -0.00 (SD 0.07) | 1.0x chance |
| Amine transaminase lineage, specific activity (Wu et al. 2022) | 27 | ESM C 300M log-likelihood ratio | -0.42 (-0.73 to -0.01) | 0 of 5 (0.0x chance) |
| Amine transaminase lineage, specific activity (Wu et al. 2022) | 27 | BLOSUM62 score of the substitution | -0.64 (-0.84 to -0.29) | 0 of 5 (0.0x chance) |
| Amine transaminase lineage, specific activity (Wu et al. 2022) | 27 | random ranking | +0.00 (SD 0.19) | 1.0x chance |

The results are:

- On the three sets of single substitutions of a natural enzyme, the language-model score
  has a higher Spearman correlation with the measurements than either baseline. Its
  intervals exclude zero and exclude the point estimate of the BLOSUM62 correlation. The
  correlations are 0.37 for thermostability of Lipase A, and 0.58 and 0.37 for the activity
  and abundance of glucokinase. The top tenth by score contains a measurement in the best
  quarter 1.7 to 2.3 times as often as chance.
- On the engineered transaminase lineage the ranking is inverted: the correlation is negative
  for the language-model score (Spearman -0.42) and more negative for the BLOSUM62 baseline
  (-0.64); no variant in the top 5 by score has a measurement in the best quarter. The set
  has 27 variants, with the score of a variant summed over its 1 to 5 substitutions
  (an additive score).
- The benchmark uses sequences without structures, so `protein.predict_stability_change`
  is not part of it. Its own benchmark on a beta-glucosidase is described in
  [Stability of substitutions](embeddings-and-prediction.md#stability-of-substitutions).

The results apply to the proteins and properties of the datasets in the table.

## Choosing among candidates

`candidate.select` picks candidates greedily from the `pool` partition of the panel named
in the request, with the strategies `random`, `diverse` (by distance), `highest-predicted`,
`lowest-predicted` and `uncertain` (by the prediction or uncertainty of a fitted model).
Two further strategies combine several criteria. `highest-predicted-diverse` selects, at each
step, the candidate with the largest value of the predicted response (scaled 0 to 1 over the
pool) plus `diversity_weight` times the distance to the nearest candidate already chosen. A
weight of 0 gives the ranking by predicted response; as the weight grows, the selection
approaches the farthest-point selection. `pareto` takes whole non-dominated fronts over
several fitted objectives, such as activity and stability, and within a front orders
candidates by the sum of their objectives scaled to 0 to 1, which weights the objectives
equally; a different weighting selects different candidates within a front. Both strategies
are reproducible with a given seed and select one candidate at a time.
