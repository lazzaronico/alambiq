# Relative binding free energy

OpenFE 1.12.0 provides hybrid-topology preparation, Hamiltonian replica exchange and MBAR
analysis. The force-field set is Sage 2.3.0, AshGC 1.0.0, ff14SB and TIP3P (identifier
`openfe.sage-2.3.0-ashgc-neutral-relative-binding-v1` in the verification files). An input
contains atom mappings, ligand poses and ligand partial charges. Before it samples, `simulation.run_relative_free_energy` checks the
supplied partial charges against the AshGC charges that the pinned model computes for the
molecule. Inputs made with AM1-BCC charges fail this check and must be regenerated.

Install in a fresh environment from the repository root:

```sh
mamba env create --prefix /path/to/openfe-env \
  --file environments/binding-free-energy/environment.yaml
/path/to/openfe-env/bin/python environments/install_openff_sources.py --binding
export ALAMBIQ_ENV_OPENFE=/path/to/openfe-env/bin/python
alambiq env check openfe
```

`ALAMBIQ_ENV_OPENFE` is the engine environment of
`structure.prepare_relative_binding_inputs` and `simulation.run_relative_free_energy`. The
second capability runs one leg, bound or solvent. The command
`alambiq capability show simulation.run_relative_free_energy` lists its request, including
`platform` (`CPU` or `CUDA`) and `threads`. On the Mac CPU run recorded below, the bound
step sets `threads: 8` in its `params`; with 2 threads the bound step timed out at the limit
of 1 h. `simulation.analyze_binding_free_energy` combines legs into binding differences and
cycles. It runs in the core.

The locks of this environment contain PyMBAR 4.2.0, NumPy and SciPy, which
`alambiq env check pymbar` requires. That the Python of this environment satisfies
`alambiq env check pymbar` follows from the package lists in `locks/`, so
`ALAMBIQ_ENV_PYMBAR` can name it for `simulation.analyze_reaction_free_energy`. The
[reaction-free-energy](../reaction-free-energy/README.md) recipe is a smaller environment
for the same capability, and the capability is verified there.

The recipe installs OpenMM 8.4 and OpenFF Toolkit 0.18 without AmberTools. OpenFE and
OpenMMForceFields 0.15.1 are installed from MIT-licensed upstream archives, with hash
verification and without dependency resolution. Alambiq supplies explicit mappings and its
own result views, so the recipe omits the upstream test packages and PyGraphviz. CPU-only
PyTorch infers the charges; GPU dynamics use OpenMM. The installer writes the manifest
entries it installed (archive hash, commit and release version) to
`alambiq-openff-sources.json` in the prefix. The GitHub archive of OpenFE lacks version
metadata, so the installer sets the setuptools-scm version variable for it after it has
verified the archive hash. The installed files are the upstream files.

The [OpenFF preparation environment](../openff-complex-preparation/README.md) uses OpenMM 8.6
and OpenFF Toolkit 0.19. Both environments use the same Sage force-field files and the same
AshGC model file. The [source manifest](../openff-sources.json) and the package inventories in
`locks/` identify the packages of this environment.

Set `platform` to `CPU` or `CUDA`. On Linux, `cuda-version=13.0` can be added to the recipe
for compatible drivers. The OpenFE CUDA adapter in Alambiq selects double precision. The
[Docker layer](../docker/Dockerfile.openfe) uses the same recipe and source installer.
The configurations in which execution is verified are those listed in the next section.

By default (`keep_native_records`), each run stores the setup, trajectory, checkpoint and
analysis files of OpenFE. The recorded runs check software integration. A result reports the
overlap between neighboring windows, the effective sample sizes, the forward and reverse
estimates and the spread between independent repeats; these quantities indicate whether
sampling has converged.

## Verification files

- The [Docker CPU file of 2026-10-09](verification.openff-docker-linux-aarch64-20261009.json)
  records an installation of the 348 packages of the Linux ARM64 lock with a Alambiq wheel
  of 231 package files. The container ran with no network, a read-only root file system,
  user ID 1000 and 8 CPU threads. Both legs and the comparison completed. All 4 repeats
  contain 50 production iterations and finite checkpoints. A separate MBAR implementation
  reproduces the endpoint estimates to within 3.32e-10 kcal/mol, inside the tolerance of
  1e-5 kcal/mol recorded in the file. A second run with engine execution disabled reused all
  stored step results, and the restored report is byte-identical to the original report.
  Each repeat samples 1 ps. The file records the observed uncertainties, overlaps and
  effective sample counts.
- The [Linux ARM64 file of 2026-10-09](verification.openff-linux-aarch64-20261009.json)
  records a CUDA double-precision solvent and bound campaign with 2 independent repeats per
  leg and 5 states, and a second run with engine execution disabled that reused all stored
  step results. All 4 repeats contain 50 production iterations and finite full-system
  checkpoints. A separate MBAR implementation reproduces the endpoint estimates to within
  2.69e-10 kcal/mol, and the overlap matrices and effective sample counts to within the
  tolerances of the file. The file describes the overlap of this short run as near-disconnected. The first
  check, which used a SciPy BFGS solver, differed from the final solution by up to
  1.48e-5 kcal/mol, above the tolerance of 1e-5 kcal/mol. A 50-digit Newton refinement of
  the same equations met every tolerance. The file records the failed first check, and the
  data and tolerances are unchanged. The restored report is byte-identical to the original
  report. The 1 ps example checks software integration. Its minimum effective sample count
  is 4.51.
- The [Mac CPU file of 2026-10-08](verification.openff-osx-arm64-20261008.json) records a
  completed 3-step benzene-to-toluene campaign with 2 independent repeats per leg and 5
  lambda states. The solvent-leg results of the first attempt are reused. The bound step
  sets `threads: 8`. The physical settings and the limit of 1 h on engine runtime are those
  of the first attempt. The bound leg and the final comparison took about 39 minutes. The
  first attempt, which was stopped, and a later attempt with 2 threads, which timed out, are
  recorded in the history of the run record. Reading the OpenFE output files confirms 50
  production iterations per repeat, the assignment of every replica to a state in every
  iteration, and the full-system checkpoints. Coordinates, box vectors and volumes are
  stored every 10 iterations. A separate MBAR implementation and a SciPy solver reproduce
  the endpoint estimates to within 1.03e-8 kcal/mol, and the overlap matrices and effective
  sample counts, on the decorrelated samples that OpenFE selects. The bootstrap errors are
  those computed by OpenFE. A second run with engine execution disabled restores the report
  byte for byte. The case has 0.2 ps equilibration and 1 ps production, a minimum effective
  sample count of 4.65 and a standard deviation of 2.57 kcal/mol between the bound repeats.
  The run checks the integration on CPU. The campaign has 1 edge; a production cycle has 3
  edges.
- The [file of 2026-10-03](verification.openff-20261003.json) records a CPU solvent
  leg with an installed wheel on macOS and a CUDA double-precision solvent and bound campaign
  on the Spark host (Linux ARM64 with an NVIDIA GB10 GPU), with 2 repeats and a final
  comparison. The GPU run took about 519 s. A CPU run of the bound leg, started in addition, was
  stopped during minimization after the GPU run finished and has no result. The 519 s is the
  time of one GPU campaign; the stopped CPU run has no time.
- The [runtime-simplification file](verification.runtime-simplification-20261003.json)
  records input preparation and inspection on Apple Silicon, and a second 2-leg CUDA
  campaign, after the optional graph rendering and the upstream test dependencies were
  removed. The resolved Conda environments contain 307 packages on Mac (329 before the
  removal) and 348 on Spark (397 before). The Mac solve also changed CPU PyTorch and its
  dependencies; the file lists all changes. The scientific package versions and builds on
  Spark are unchanged. Compressed package downloads decreased by about 13 MB (Mac) and
  about 29 MB (Spark). The check took about 600 s; the earlier run of the 2-leg CUDA campaign
  took about 519 s.

The [Apple Silicon](locks/osx-arm64.packages.json) and
[Linux ARM64](locks/linux-aarch64.cuda13.packages.json) inventories list the name, version,
build and license of each Conda package. The hash-verified source installs are recorded in
the [source manifest](../openff-sources.json) and in `alambiq-openff-sources.json` in the
prefix.

## Linux container

Build the Alambiq core image with the `quantitative` extra, then the OpenFE layer. The
layer copies in the OpenMM engine of the `alambiq:openmm` image (the default
`OPENMM_IMAGE`), so build that image first (see the [Docker notes](../docker/README.md)):

```sh
docker build -f environments/docker/Dockerfile \
  --build-arg ALAMBIQ_EXTRAS=quantitative -t alambiq:binding-core .
docker build -f environments/docker/Dockerfile.openfe \
  --build-arg BASE_IMAGE=alambiq:binding-core -t alambiq:openfe .
```

For the recorded Linux ARM64 dependency set, add
`--build-arg OPENFE_SPEC=environments/binding-free-energy/locks/linux-aarch64.cuda13.explicit.txt`
to the second build. The layer runs the same hash-verified source installer for the recipe
and for the explicit lock. Set `platform` in the step and provide GPU access for CUDA. The
verification file of 2026-10-03 predates the container runs. Container runs are recorded in
the Docker CPU file of 2026-10-09 and in the run of 2026-10-23 below.

The layer sets `ALAMBIQ_ENV_OPENFE` and `ALAMBIQ_ENV_PYMBAR`, and copies in the OpenMM
engine of the `alambiq:openmm` image (set as `ALAMBIQ_ENV_OPENMM`), because the protein is
prepared with OpenMM 8.6 and the OpenMM of this environment is 8.4. On 2026-10-23, on the Spark host with the GPU visible, 141 tests passed and none were
skipped (see the [Docker notes](../docker/README.md)).

## Licenses

OpenFE, OpenFF Toolkit, NAGL and OpenMMForceFields use MIT terms. Sage parameters and NAGL
model assets use CC BY 4.0. OpenMM GPU platforms, MDTraj and ParmEd are under LGPL terms, and
the Python of Conda includes GPL readline. The locks omit graph rendering or its
GPL hicolor icon theme. Redistribution of an environment or image requires the notices of
each package; the package inventories list the licenses.

## Historical methods

Runs made with GAFF/AM1-BCC parameters or with Sage 2.2.1 used different force-field
parameters and charge models. Their results apply to those parameters and charge models. All
verification files in this directory record the Sage 2.3.0/AshGC force-field set.
