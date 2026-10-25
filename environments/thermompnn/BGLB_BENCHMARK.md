# BglB functional-stability benchmark

The benchmark applies 2 criteria that were fixed before any prediction, on 2026-10-04, in the
[protocol](bglb-protocol.json). The score is the negative of the predicted ΔΔG (positive ΔΔG
denotes destabilization), so a larger score ranks a variant as more stable.

1. The lower end of the 95% interval of the Spearman correlation between the score and the
   functional thermal midpoint of the 78 variants with a measured midpoint must exceed 0.
   The interval comes from a bootstrap over mutated positions: 2,000 replicates, each
   resampling positions with replacement and keeping all variants at a drawn position.
2. Among the 10 highest-ranked of these 78 variants, the number with a midpoint at least
   1 °C above wild type (39.93 °C) must exceed the random expectation, with a one-sided
   hypergeometric probability of at most 0.05.

ThermoMPNN met neither criterion. The Spearman correlation is 0.161, with a position-bootstrap
95% interval of −0.098 to 0.379. The 10 highest-ranked of the 78 variants contain 3 with a
midpoint at least 1 °C above wild type. If 10 of the 78 variants have such a midpoint, a
random choice of 10 variants contains 1.28 of them on average (10 × 10 / 78). The
probability of 3 or more is 0.113 (hypergeometric).

Timings were measured on Apple Silicon CPU.

| Measurement | Wall time (seconds) |
| --- | ---: |
| Engine: structure validation, resource verification, worker startup, model loading and inference for 128 variants plus the unchanged control | 1.326 |
| Planning, campaign execution and collection | 5.292 |
| First use: runtime setup, model provisioning and the first complete benchmark process (summed phases; see the [first-use measurement](README.md#first-use-time)) | 57.201 |
| Second benchmark process | 10.090 |
| Third benchmark process | 8.345 |

The first two rows were measured with the model files already installed. They were measured
with an earlier version of the script, which timed planning, execution and collection; the
current script times the execution of the campaign. The first-use row was measured with
Python, Alambiq and the experimental inputs already available. The second and third
benchmark processes reproduced the original result exactly.

Among the 10 highest-ranked of all 128 variants, 9 have a measured midpoint and 1 has none.
Of these 10 variants, only N220Y has a midpoint at least 1 °C above wild type and a catalytic efficiency of at
least half of wild type. The experiment (Carlin et al.) measured the midpoints and the
catalytic efficiencies; the model did not use them. ThermoMPNN predicts changes in folding
stability from backbone coordinates.

## Data sources and structure

Carlin et al. (2017),
[Thermal stability and kinetic constants for 129 variants of a family 1 glycoside hydrolase](https://doi.org/10.1371/journal.pone.0176255),
reports BglB expression, thermal and kinetic measurements. The
[CC BY 4.0 supplementary deposit](https://doi.org/10.6084/m9.figshare.5029253) provides the
experimental table and the prepared coordinates used here. The supplied table contains
128 mutants and the wild type. The article describes 129 mutants and 79 mutant thermal
midpoints; the table has 78. Every supplied row is used; the absent variant is omitted.
50 of the 128 variants have no measured midpoint; they include all 37 variants that do not
express. The result table contains all 128 variants, with missing and censored values
marked. The functional midpoint is the fitted midpoint reported by the authors for the
activity measured after a 30 min heat challenge at temperatures from 30 to 50 °C (pNPG
activity readout). The functional midpoint is an activity-based measure of heat
inactivation, whereas ThermoMPNN predicts the change in folding stability.

In the author-prepared apo model (445 residues), the wild-type residue and the position of
each of the 128 mutations match. The structure 2O9P in the PDB uses different numbering,
lacks the unresolved N-terminal residues, and contains engineered background substitutions,
including a mismatch at H373R. The benchmark reads only the prepared coordinate file from the
supplementary archive and runs `protein.predict_stability_change`.

MMseqs2 18-8cc5c, run without prefilter, found no local alignment with an E-value ≤ 0.001
between the BglB sequences (the deposited entity, the observed construct and the
author-prepared construct) and any of the 431 reference sequences. The reference sequences
represent the MegaScale training and validation sets, the FireProt proteins evaluated
earlier, and the FireProt training and validation sets as reported. The search covers sequence
similarity to the reference sets; the structure pretraining of ProteinMPNN and fold similarity
lie outside it. The benchmark covers one enzyme and one prepared structure.

## Reproduction

Prepare the [ThermoMPNN environment and model resource](README.md), and install the Alambiq
extras `dossier` and `quantitative`. Download these 2 files without modification into one
directory:

- [S1 Table.csv](https://ndownloader.figshare.com/files/8503931), saved as `S1-Table.csv`.
- [S1 Code.zip](https://ndownloader.figshare.com/files/8504015), saved as `S1-Code.zip`.

Run from the repository root with a Python that has NumPy and SciPy installed. The second
argument is a new directory, here `.alambiq/bglb-check` below the current directory:

```bash
python environments/thermompnn/benchmark_bglb.py /absolute/path/to/sources .alambiq/bglb-check
```

`benchmark_bglb.py` checks the SHA-256 of the table, the archive, the coordinate file and the
protocol, creates the output directory, runs `protein.predict_stability_change` as a
one-step campaign with `alambiq.api.run`, and writes the campaign (`campaign.json`), the
run directory (`run/`, with the stored artifacts and the run record), the result
(`result.json`), the analysis with the timings (`analysis.json`) and the 2,000 bootstrap
values (`bootstrap.json`). For the earlier version of the script, the verification file
records that a second execution gave identical predictions, metrics and bootstrap values,
and that the Spearman correlation (average ranks for ties) and the enrichment probability
(sum over integer combinations) recomputed with separate code agreed with the reported
values. The current script was checked by its argument parser and by a run on a synthetic
table and archive of the same layout built from the GB3 structure; the figures of this
document come from the earlier version.

The [protocol](bglb-protocol.json) states the endpoint, the discrepancy between the sources,
the assay conditions, the exclusions, the uncertainty method and the decision criteria; it
was fixed before the run. The [verification file](verification.bglb-functional-stability.json)
contains all predictions and observations, the SHA-256 of all inputs and the run records of
the first and the second execution. The benchmark evaluates ranking by functional thermal
midpoint.
