# Changelog

## Unreleased

- `structure.predict_biomolecular_complex` 1.4 takes `cofactors`: components named by their Chemical Component Dictionary code (for example `NAP` and `ZN`) with a number of copies. Each copy is a chain of the prediction, listed in the result, and the ligand is placed with the cofactors present. A request without cofactors gives the input it gave before.
- `structure.measure_ligand_site` 1.1 returns `contacts`, the nearest heavy atom of the reference to each ligand atom (waters excluded), so that a ligand that overlaps a cofactor or the protein is visible. 1.0 was not released.
- `structure.measure_ligand_site` superposes a structure that holds a ligand, for example a predicted complex, on a reference over all protein chains and measures the distance from the ligand to chosen atoms of the reference (a catalytic metal, the reactive carbon of a cofactor). Chains are paired by the assignment with the smallest alpha-carbon RMSD, or by `chain_map`. Assignments that fit within `equivalent_fit_angstrom` of the smallest RMSD are equivalent fits, and the result gives the range of each nearest distance over them. The result lists the ligand atoms with the atoms they are bonded to, the distance to every reference residue that has the target atom, and the ligand in the reference frame as an mmCIF file. The TbSADH example uses it in place of a script.
- `structure.predict_biomolecular_complex` 1.3 runs a protein of identical chains (a homo-oligomer) with an alignment. It wrote one copy of the alignment per chain, and Boltz-2 refuses chains of one sequence that name different alignment files (`All proteins with the same sequence must share the same MSA!`), so such a request could not run. Identical alignments are now written once, and identical chains with different alignments are refused before the run, with the chains named.
- `structure.predict_biomolecular_complex` 1.2 runs a ligand SMILES with several components (a salt, written with a `.`) as its largest component. Before, any such SMILES stopped with an RDKit failed precondition (`getNumImplicitHs() called without preceding call to calcImplicitValence()`), raised by Boltz's own standardization, which gives an unsanitized molecule to RDKit's largest-fragment chooser; this happened with RDKit 2026.03.5 in the pinned environment as well. The result lists the other components in `ligand.other_components`, `ligand.input_smiles` keeps the SMILES as written, and a note names what was left out.
- `structure.prepare_predicted_complex_for_dynamics` 1.2, `structure.prepare_protein_ligand_system` 1.1 and `structure.prepare_protein_system` 1.1 give the same minimized structure for the same request. The CPU platform sums the reciprocal-space forces of PME in an order that varies from call to call, even with `Threads` set to 1 (the PME kernel does not read that property); the minimizer then ended at a different structure on every run (on a 100,000-atom system, twelve force evaluations at one set of positions gave twelve different arrays). The preparation now sets `DeterministicForces`, and `threads` defaults to 1; with more threads the structure still differs from run to run. `simulation.run_molecular_dynamics` 1.1 and `simulation.run_metadynamics` 1.1 add `DeterministicForces` to the CPU defaults for the same reason. A capability version change changes the reuse key, so saved campaigns recompute these steps.
- `protein.inspect` 1.1 returns `sequence`, the query sequence, so that a binding can pass it to a later step (for example `proteins.0.sequence: dossier.sequence` in a complex prediction).
- `structure.prepare_predicted_complex_for_dynamics` 1.1 returns `ligand_residue` (chain, residue number and name of the ligand in `topology_pdb`) and `ligand_residue_index` (its zero-based index). `simulation.measure_ligand_pose` takes the first as `ligand` and `simulation.inspect_trajectory` the second as `ligand_residue_index`, so a campaign names the ligand once. The T4 lysozyme example binds both.
- `protein.call_substitutions` 1.1 returns `reference_file` for each variant, a FASTA file with the sequence of its closest reference. The file is the `protein` and `notation` the variant of a request to `protein.score_substitutions`, so a variant known only by its sequence can be scored without retyping the reference.
- `structure.prepare_predicted_complex_for_dynamics` and `structure.prepare_protein_ligand_system` use the
  periodic box of the coordinate file. A prmtop stores one box angle, so a rhombic dodecahedron read from it
  was another lattice than the one the solvent was placed in: waters lay on their own periodic images, the
  energy before minimization was about 1e13 to 1e23 kJ/mol, and dynamics on a CUDA platform stopped with a NaN
  coordinate. The complex preparation now also refuses a minimized start whose largest force exceeds 1e5
  kJ/mol/nm.
- `simulation.measure_ligand_pose` measures the heavy-atom RMSD of a ligand to a reference pose in
  every frame of a trajectory, after fitting the protein by the pocket alpha carbons, with the
  smallest RMSD over the symmetric atom pairings, the centroid shift and the fraction of frames
  within a threshold.
- `docs/concepts.md` states when the version of a capability changes and what a saved campaign
  does after a capability changes.
- `structure.prepare_mutation_inputs` widens the water margin of a leg when a compact solute would
  give a box narrower than twice the nonbonded cutoff, which OpenMM refuses; the result reports the
  padding used.
- `structure.prepare_mutation_inputs` minimizes the solvated wild type before it builds the two
  states. Without it a 38,000-atom system started with forces of 2e5 kJ/mol/nm between water and ions
  and the sampler stopped with NaN. `simulation.run_mutation_free_energy` now refuses a leg whose free
  energy is not finite, with a message that says so.
- `structure.prepare_protein_system` and `structure.prepare_mutation_inputs` take an optional
  `protonated_structure`, the output of `structure.assign_protonation`, and use its residue names
  as the protonation of the chain. The mutant state of `structure.prepare_mutation_inputs` is now
  protonated at the stated pH (it used pH 7 for every request).
- `structure.assign_protonation` protonates a protein at a pH with PROPKA 3.5 and PDB2PQR 3.7, in
  Amber residue names. It reports the pKa of every titratable group, the histidine tautomers, the
  asparagine, glutamine and histidine side chains that the hydrogen-bond optimization turned by
  180 degrees, and the heavy atoms it rebuilt. It runs in a protonation engine environment
  (`ALAMBIQ_ENV_PROTONATION`).
- `structure.prepare_mutation_inputs`, `simulation.run_mutation_free_energy` and
  `simulation.analyze_stability_change` estimate the change in the folding free energy of a single
  amino-acid substitution. The folded protein and a capped tripeptide are each transformed from the
  wild-type to the mutant residue with the hybrid-topology factory of OpenFE and replica exchange,
  and the two legs are subtracted. They run in the OpenFE engine environment (`ALAMBIQ_ENV_OPENFE`).
- `structure.fetch` downloads a structure file from the RCSB Protein Data Bank or the AlphaFold
  Protein Structure Database, in mmCIF or PDB format, and stores it with its URL, retrieval time,
  size and SHA-256. It connects to the network only when `allow_network` is true.
- `simulation.cluster_trajectory` clusters the frames of a trajectory by pairwise RMSD with the
  method of Daura et al. (1999), reports the size and center frame of each cluster, assigns every
  frame to a cluster, and writes the center structures of the largest clusters as PDB files. It
  needs the `trajectory` extra.
- `structure.check` reports, for one model of a PDB or mmCIF file, the heavy atoms missing from
  standard residues, alternate locations, atoms of zero occupancy, nonstandard amino acids,
  chain breaks, cis and twisted peptide bonds, alpha carbons off the L arrangement, close
  contacts, and the ligands, metal ions, waters and hydrogens. It runs in the core with the
  `dossier` extra.
- The chains and the ligand of `structure.predict_biomolecular_complex` need no `id`: a chain or
  ligand without one takes the next unused capital letter. The fields `sequence` (chain) and `smiles`
  (ligand) now come first in the request schema.
- `protein.score_substitutions` states in its limits that the request names the wild-type sequence
  and that the scores are log-likelihood ratios without a calibration to an assay scale.
- The affinity of `structure.predict_biomolecular_complex` also gives pIC50 (6 minus the log10 of the
  IC50 in µM) for the ensemble and for each model.
- `protein.predict_melting_temperature` estimates the melting temperature of proteins from their
  sequences with the three replicas of TemBERTure Tm, and returns the mean and the spread of the
  replicas. The model is the pinned resource `temberture-tm` (about 1.7 GB), which
  `alambiq resource provision temberture-tm --yes` downloads. The engine environment is
  `ALAMBIQ_ENV_THERMOSTABILITY`.
- `molecule.compute_hydration_free_energy` computes the hydration free energy of a neutral small
  molecule given as SMILES with the OpenFE absolute solvation protocol (Sage 2.3.0, AshGC
  charges, TIP3P), with repeats, MBAR errors and the overlap between lambda windows. It runs in
  the OpenFE environment, on CUDA by default.
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
