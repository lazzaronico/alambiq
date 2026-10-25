# ThermoMPNN inference environment

ThermoMPNN predicts changes in folding stability for single substitutions from the backbone
of a selected chain. The result is in kcal/mol; positive values denote destabilization. The
inputs are the substitution and the backbone coordinates of the selected chain; the
prediction applies to that chain in isolation. See the
[capability guide](../../docs/methods/embeddings-and-prediction.md#stability-of-substitutions).

The `protein.predict_stability_change` capability uses the ThermoMPNN checkpoint and inference
source of the authors at commit `370f76ec62bd929f7425e311d8df04a0d094990f`. The
[resource record](../../src/alambiq/resources/thermompnn.json) lists a SHA-256 value for
each of the 10 files: the source files, the 2 checkpoint files
`model/models/thermoMPNN_default.pt` and `model/vanilla_model_weights/v_48_020.pt`, and the
upstream MIT notices. `alambiq resource provision thermompnn --yes` stores them in the
resource store, outside Git.

Create the environment with Mamba:

```bash
mamba env create --prefix /absolute/path/to/alambiq-thermompnn \
  --file environments/thermompnn/environment.yaml
```

Install the `dossier` extra (Gemmi, which reads the structure and maps residue numbers) in
the core environment, provision the model, and set `ALAMBIQ_ENV_THERMOMPNN` to the Python of
the new environment:

```bash
python -m pip install '.[dossier]'
alambiq resource show thermompnn
alambiq resource provision thermompnn --yes
export ALAMBIQ_ENV_THERMOMPNN=/absolute/path/to/alambiq-thermompnn/bin/python
alambiq env check thermompnn
```

`ALAMBIQ_RESOURCE_HOME` sets the location of the resource store. To use a copy of the
resource outside the store, set `ALAMBIQ_RESOURCE_THERMOMPNN` to its directory and run
`alambiq resource verify thermompnn`. Before it runs, an inference checks the SHA-256 of all
model and code files against the resource record. The request
fields `threads` and `seed` set the number of CPU threads and the random seed.
`alambiq capability show protein.predict_stability_change` lists the full request.

## Verification

The [Apple Silicon verification file](verification.osx-arm64.json) records Python 3.11.14,
Torch 2.8.0 and NumPy 1.26.4. The [package list](requirements.osx-arm64.txt) contains
10 installed Python distributions and pins Torch 2.13.0. With Torch 2.13.0, the 47 capability
tests passed on 2026-10-25, and the predictions for 64 substitutions on GB3 differed from the
Torch 2.8.0 predictions by at most 2e-6 kcal/mol. The inference path imports Torch, NumPy and their
dependencies only. Gemmi is installed in the Alambiq core environment through the `dossier`
extra. The file covers Apple Silicon CPU. The engine runs on the CPU. The Docker image ran
the capability tests on Linux ARM64 with Torch 2.13.0+cpu, installed from the PyTorch CPU wheel
index (47 passed; see the [Docker notes](../docker/README.md)).

On Apple Silicon CPU, a campaign on GB3 with 2 substitutions (T55A and the unchanged control
T55T) completed. The prediction for T55T was 0 kcal/mol. The engine took 0.935 s, including
input validation, asset checks, worker startup, model loading and inference.

## Comparison with the authors' code


In the identical-backbone comparison, the upstream ThermoMPNN code and Alambiq were run on
identical complete backbone arrays for 64 proteins. The predictions for all 1,999 distinct substitutions were identical (maximum
difference 0.0 kcal/mol, tolerance 1e-5 kcal/mol). The earlier FireProt benchmark has
2,578 rows and 89 proteins; the raw-input comparison of these 64 proteins had 2,018 rows.
The verification file lists the 25 excluded source structures with their reasons: 12 with
incomplete backbone atoms, 7 with a peptide-chain break, 5 that fail to parse and 1 whose
substitution does not match the mapped wild-type residue.

The raw-input comparison, made earlier, differed for 5 of the 64 proteins (35 experimental
rows), by at most 0.61225 kcal/mol. The cause: the upstream parser inserts padding residues at
author residue numbers that are skipped in the structure, whereas Alambiq uses the supplied
complete construct sequence and stores the author numbering separately. The verification
file records the differences, the diagnosis and the comparison with identical backbones. The
predictions of the earlier benchmark are unchanged. The checks show equal predictions for
equal inputs.

A further comparison read the same structures from mmCIF files with the author chain parts
selected. The 64 inputs and the 2 GB3 predictions were identical to those of the
identical-backbone comparison. The engine run took 1.095 s. The model input comprises
polymer residues; nonpolymer residues (ligands) are listed as omitted. A structure with
modified polymer residues needs explicit preparation before prediction.

## BenchStab eligibility

The [BenchStab eligibility audit](verification.benchstab-eligibility.json) checked all 289
deposited measurements (36 proteins) before any prediction was made. It excluded measurements
of proteins with any local alignment (MMseqs2, E-value ≤ 0.001, no minimum identity or
coverage) to the MegaScale training or validation sets or to the proteins evaluated earlier,
and measurements without a complete backbone or a mapped mutation. The audit retained 2
measurements: one phosphotriesterase substitution and one twitchin-domain substitution. An accuracy
estimate or a test of shortlist enrichment requires more than 2 measurements. A
protein without a sequence match can still share a fold with the training data. The source
is [Velecký et al., BenchStab dataset](https://doi.org/10.5281/zenodo.10637728), CC BY 4.0;
deposited structures use CC0 1.0.

## BglB benchmark

The [BglB functional-stability benchmark](BGLB_BENCHMARK.md) predicted 128 single
substitutions on the prepared apo model of the authors of the BglB data. The Spearman
correlation with the 78 measured thermal midpoints and the enrichment of variants with a
higher midpoint among the 10 highest-ranked variants both failed the criteria that were fixed
before the run. The 128-variant campaign took 5.292 s on Apple Silicon CPU.

## First-use time

An [Apple Silicon setup measurement](verification.first-use.osx-arm64.json) started with an
empty ThermoMPNN environment and an empty model store. It downloaded the 10 pinned inference
packages without the pip cache and provisioned all 10 model, source and notice files
(45,688,340 bytes) with `alambiq resource provision`. Python 3.11.14, pip 25.2, Alambiq
and the experimental inputs were already available. The measurement started with an empty
Prefect home (the state directory of Prefect, the library that runs campaigns). The times
include the effects of the caches of the operating system and of the network.

| Phase | Wall time (seconds) |
| --- | ---: |
| Create isolated environment | 0.080 |
| Download and install pinned packages | 28.036 |
| Record runtime inventory | 0.129 |
| Download and verify model resource | 12.026 |
| First complete BglB benchmark process | 16.930 |
| Sum through the first benchmark | 57.201 |

A benchmark process includes imports, input verification, the campaign, 2,000 bootstrap
replicates, saved results and shutdown. 2 further processes that used the installed
resources took 10.090 and 8.345 seconds. The campaign took 6.923 and 5.700 seconds in these
processes and 12.847 seconds in the first process; these durations are included in the
process times. The table lists the phases; the overhead between the phases and the integrity
audit lie outside the phase times in the table.

The 3 runs gave the same 129 predictions and the same bootstrap values as the original
benchmark run. The times are those of one setup and two repeats on one Apple Silicon machine.

Reference: [Dieckhaus et al., PNAS 2024](https://doi.org/10.1073/pnas.2314853121).
The reference data of the FireProt comparison are from
[FireProtDB + PDB, Zenodo 8169289](https://doi.org/10.5281/zenodo.8169289), CC BY 4.0.
