# Changelog

## Unreleased

- `molecule.predict_admet` estimates 52 physicochemical and ADMET endpoints for 1 to 10,000
  molecules given as SMILES with ADMET-AI 2.0.1, with the units and the reported metrics of each
  endpoint and, on request, the percentile among DrugBank approved drugs. The engine environment
  is `ALAMBIQ_ENV_ADMET`.
- `molecule.compute_quantum_properties` computes the frontier orbital energies, thermochemistry,
  vertical excitations, Mulliken charges and dipole moment of a small molecule given as SMILES
  with density-functional theory. The molecule can be charged, in a higher spin state, and in
  water (IEF-PCM). The engine environment is `ALAMBIQ_ENV_MOLECULAR_PROPERTIES`.
- `protein.call_substitutions` names the substitutions, insertions and deletions that separate a
  variant from its closest reference in a supplied FASTA or in Swiss-Prot.
- `protein.score_substitutions` (version 1.1) scores variants of several substitutions, written
  `K183N:T209K`, as the sum of the scores of their substitutions.
- `alambiq resource provision swissprot-2026_03 --yes` downloads the two UniProt files of the
  Swiss-Prot resource, checks them against their recorded size and MD5 digest, and builds the
  database and the annotation table. Without `--yes` it states the download and the build and
  does nothing.
- `protein.annotate_function` lists the EC numbers, Gene Ontology terms and subcellular
  locations that the closest reviewed relatives of a protein in UniProtKB/Swiss-Prot carry,
  with the support and identity of each. The Swiss-Prot resource is built locally.
- `structure.predict_biomolecular_complex` (version 1.1) estimates binding affinity with
  `affinity: true`: Boltz-2's log10 IC50 in µM and binder probability, for the ensemble and
  each of its two models.

## 0.1.2

- The package description is "Alambiq is software that runs computational methods for
  biotechnology as capabilities. A campaign lists steps, and each step runs one capability."
- The README, the documentation pages, the example READMEs and the text that `alambiq capability show`
  prints open with the core idea and use shorter sentences.
- Some notes and error messages of the capabilities are reworded. The message for a missing
  `dossier` extra gives the PyPI install command.

## 0.1.1

- The `alambiq` command stops the temporary Prefect server when it ends, including after a
  termination request, a hang-up or a server that did not finish starting.
- The package description is "Capabilities for biotechnology R&D, run from YAML or Python".
- The notes of `assay.inspect` use the singular for one reading, and the solver line of an
  `assay.fit` report has no backticks.
- Pinned versions of `hydra-core`, `requests`, `filelock` and `torch` in the Boltz-2 Apple GPU,
  Boltz-2 CUDA and ThermoMPNN environments are updated and verified; the Docker images of
  Boltz-2 and ThermoMPNN are rebuilt and tested.

## 0.1.0

Alambiq is organized around capabilities that have declared request and result types.

### Capabilities and runs

- Capabilities in families (characterize, investigate, engineer, measure,
  operate). Each capability is a function with a declared request type, a declared result
  type and documented limits, and has tests in `tests/core`. `alambiq capability list`
  prints the capabilities; `alambiq capability show ID` prints the request, result,
  method and limits of one capability.
- A campaign is a list of steps; each step has `params` and `bind` (see
  [Concepts](docs/concepts.md)). Prefect executes the campaign. Results are stored in a
  content-addressed artifact store. A step is reused when its key is unchanged
  (see [Reuse](docs/concepts.md#reuse)).
- Docker images for the engine environments (MMseqs2, P2Rank, ThermoMPNN, pymbar, QM/MM,
  OpenMM CUDA, OpenFF, ESM, Boltz-2, OpenFE). An engine environment is built from a plain
  Python image and copied onto the core image, so a change to the core does not rebuild
  it. `Dockerfile.tests` adds the test runner (pytest) to any image, so that the repository
  tests, mounted at run time, run against the engines of that image. The images were built
  for linux/arm64 on a Linux ARM64 host with an NVIDIA GB10 GPU on 2026-10-25. The
  environment check of each image and the examples named in the receipt were run there, and
  the engine tests were run there with 20 QM/MM engine tests skipped
  (`environments/docker/verification.linux-arm64-spark-20261025.json`). The
  Apple GPU environment of Boltz-2 is available only as a native environment.
- Engines run in engine environments (separate Python environments with pinned packages,
  named by `ALAMBIQ_ENV_<NAME>`) and are started by a runner that imports only the Python
  standard library. Models and databases are pinned resources: each is installed before
  a run and verified before use.
- Every run writes a run record (`run.json`); a copy of every run, including the latest, is
  stored under `history/` in the run directory. `alambiq history RUN_DIR` lists the runs, and
  `alambiq replay RUN_DIR STEP --out DIR` writes one step of a recorded run as a
  campaign that runs it again.

### Reports

- One HTML file per run, with no external references. Charts are inline SVG that records
  the plotted x and y values. An offline structure viewer (3Dmol.js, BSD-3-Clause) is
  added when a structure is shown. Alignments are shown as aligned rows in blocks. When the run
  directory holds at least 2 recorded runs of the campaign, the report ends with a table of
  those runs.

### Methods

- `assay.fit` fits Arrhenius, first-order decay and pH-profile models in addition to the
  linear and Michaelis-Menten models, with derived quantities and their uncertainty.
- `operation.define` converts measured kinetics into an operating definition (see
  [Operation file formats](docs/operation-formats.md)), with temperature and pH
  dependence, a stated operating state (`state`: temperature and pH) and, for a reversible
  pair, the Haldane relation to the equilibrium constant. `operation.propagate`
  propagates the uncertainty of a fitted Michaelis-Menten rate to the predicted
  concentrations.
- `protein.score_substitutions` scores substitutions with ESM C by masked marginals.
  `variant.propose` ranks them by the language-model score, the predicted change in folding
  free energy or the mean of both percentile ranks, with filters that include the presence
  of the mutant residue in homologs. A benchmark on 4
  measured datasets (3 of single substitutions of natural enzymes and 1 of an engineered
  lineage) is in [the methods page](docs/methods/variant-proposal.md). The Spearman
  correlation between the language-model score and the measurements is +0.37 for the
  thermostability of Bacillus subtilis Lipase A, +0.58 for the activity of human
  glucokinase, +0.37 for the abundance of human glucokinase and -0.42 (95 % interval -0.73
  to -0.01) for an amine transaminase lineage.
- `candidate.select` has 7 strategies, among them `highest-predicted-diverse`, which
  weights predicted response against diversity, and `pareto`, which takes non-dominated
  fronts over 2 to 5 fitted objectives.

### Examples and documentation

- Runnable examples with asserted headline values, methods pages and an installation
  guide.
