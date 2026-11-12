# Changelog

## 1.0.0

Alambiq 1.0.0 is the first stable release. The catalog holds 69 capabilities in five families
(investigate 33, characterize 19, operate 8, engineer 6, measure 3), run from one request format in a
campaign written in YAML or Python. Each capability declares its request, its result and its limits, carries
its own version (the rule is in [Concepts](docs/concepts.md)), and has tests that run every request field.
The version of a capability is part of the reuse key of a step, so a campaign saved with an earlier
capability version recomputes the steps whose capability changed.

### New in 1.0.0 since 0.1.2

Structures and complexes

- `structure.predict_biomolecular_complex` predicts a complex with Boltz-2 and, with `affinity: true`, estimates
  the binding affinity (Boltz-2 log10 IC50 in µM, pIC50 and binder probability, for the ensemble and for each
  model). It takes `cofactors`, components named by their Chemical Component Dictionary code (for example `NAP`
  and `ZN`) with a number of copies; each copy is a chain of the prediction, listed in the result. Chains and the
  ligand need no `id` (the next unused capital letter is taken). A ligand written as a salt runs as its largest
  component, and the result lists the others. A protein of identical chains runs with one shared alignment, and
  identical chains that name different alignments are refused before the run.
- `structure.measure_ligand_site` superposes a structure that holds a ligand, for example a predicted complex,
  on a reference over all protein chains and measures the distance from the ligand to chosen atoms of the
  reference (a catalytic metal, the reactive carbon of a cofactor). Chains are paired by the assignment with the
  smallest alpha-carbon RMSD, or by `chain_map`; assignments within `equivalent_fit_angstrom` of the smallest are
  equivalent fits, and the result gives the range of each nearest distance over them. It returns the nearest
  reference atom to each ligand atom (`contacts`, so that a ligand that overlaps a cofactor is visible), and, for
  `own_targets`, distances to atoms of the mobile structure itself (`own_measurements`, `own_contacts`).
- `structure.fetch` downloads a structure from the RCSB Protein Data Bank or the AlphaFold Protein Structure
  Database, in mmCIF or PDB format, and stores it with its URL, retrieval time, size and SHA-256. It connects to
  the network only when `allow_network` is true.
- `structure.check` reports the heavy atoms missing from standard residues, alternate locations, atoms of zero
  occupancy, nonstandard amino acids, chain breaks, cis and twisted peptide bonds, alpha carbons off the L
  arrangement, close contacts, and the ligands, metal ions, waters and hydrogens of one model.
- `structure.assign_protonation` protonates a protein at a pH with PROPKA 3.5 and PDB2PQR 3.7, in Amber residue
  names, and reports the pKa of every titratable group, the histidine tautomers and the side chains that the
  hydrogen-bond optimization turned. It runs in the protonation environment (`ALAMBIQ_ENV_PROTONATION`).
- `protein.inspect` returns the query `sequence`, so that a binding can pass it to a later step.

Dynamics and free energy

- `structure.prepare_predicted_complex_for_dynamics`, `structure.prepare_protein_ligand_system` and
  `structure.prepare_protein_system` give the same minimized structure for the same request: the CPU sums the
  reciprocal-space forces of PME in an order that varies from call to call, so the preparation sets
  `DeterministicForces` and `threads` defaults to 1 (with more threads the structure still differs from run to
  run). `simulation.run_molecular_dynamics` and `simulation.run_metadynamics` add `DeterministicForces` to
  their CPU defaults. With these defaults two complete 10 ns campaigns of the T4 lysozyme example wrote the same
  prepared system and the same 1,000 frames.
- The complex preparations use the periodic box of the coordinate file (a prmtop stores one box angle, which put
  a rhombic dodecahedron on another lattice and stopped dynamics on CUDA with a NaN coordinate), and refuse a
  minimized start whose largest force exceeds 1e5 kJ/mol/nm. `structure.prepare_predicted_complex_for_dynamics`
  returns `ligand_residue` and `ligand_residue_index`, which `simulation.measure_ligand_pose` and
  `simulation.inspect_trajectory` take, so a campaign names the ligand once.
- `simulation.measure_ligand_pose` measures the heavy-atom RMSD of a ligand to a reference pose in every frame
  of a trajectory (pocket-fitted, symmetry-aware), with the centroid shift and the fraction of frames within a
  threshold. `simulation.cluster_trajectory` clusters frames by pairwise RMSD with the method of Daura et al.
  (1999) and writes the center structures of the largest clusters.
- `structure.prepare_mutation_inputs`, `simulation.run_mutation_free_energy` and
  `simulation.analyze_stability_change` estimate the change in the folding free energy of a single substitution
  (the folded protein and a capped tripeptide are each transformed with the OpenFE hybrid-topology factory and
  replica exchange, and the two legs are subtracted). The preparation widens the water margin of a compact
  solute, minimizes the wild type, takes the protonation from `structure.assign_protonation` and protonates the
  mutant at the stated pH; a leg whose free energy is not finite is refused.
- `molecule.compute_hydration_free_energy` computes the hydration free energy of a neutral small molecule with
  the OpenFE absolute solvation protocol (Sage 2.3.0, AshGC charges, TIP3P), with repeats, MBAR errors and the
  overlap between lambda windows. It runs in the OpenFE environment, on CUDA by default.

Proteins, sequences and molecules

- `protein.predict_melting_temperature` (three TemBERTure Tm replicas, mean and spread), `protein.annotate_function`
  (EC numbers, GO terms and subcellular locations of the closest reviewed Swiss-Prot relatives, with support and
  identity), `protein.call_substitutions` (substitutions, insertions and deletions against the closest reference,
  with a `reference_file` for scoring) and `protein.score_substitutions` (variants of several substitutions, as
  the sum of their scores; the limits state that the scores are uncalibrated log-likelihood ratios).
- `molecule.predict_admet` estimates 52 physicochemical and ADMET endpoints for up to 10,000 molecules with
  ADMET-AI 2.0.1. `molecule.compute_quantum_properties` computes frontier orbital energies, thermochemistry,
  vertical excitations, Mulliken charges and the dipole moment of a small molecule with density-functional
  theory, charged, in a higher spin state and in water (IEF-PCM).

Resources, environments and documentation

- Pinned resources fail closed: a file whose size or digest differs from its record is refused. Their fetch paths
  are versioned. The Swiss-Prot resource is fetched from the release archive
  `previous_releases/release-2026_03/knowledgebase/uniprot_sprot-only2026_03.tar.gz` (the addresses under
  `current_release` serve the newest release); each member is extracted by file name and checked against its
  own size and MD5 digest. The Boltz-2 resource is fetched from the pinned Hugging Face revision of its record
  (address, size and SHA-256 digest of each file), and Boltz's own function only unpacks the molecule library.
  `alambiq resource provision NAME --yes` states what it downloads before it does.
- The capability text, the environment READMEs and the methods pages cite the publications of the upstream
  methods, and `LICENSING.md` states the licenses of the optional packages `doranet`, `pydna` and `dnachisel`.
  The source is formatted with `ruff format`.

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
