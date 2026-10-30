# Installation

Alambiq has three parts: the core, the engine environments and the resources. The core is
the `alambiq` Python package and requires Python 3.11 or newer. Engine environments provide
the scientific software that some capabilities need. Resources are the models and databases
that some capabilities declare.

Install the core and run `examples/kinetics-from-measurements`. Then install the engine
environments and resources that the capabilities in use require.

## Docker and native installation

Engine environments come from Docker images or from native builds.

- Docker: one image per engine environment. Use this route where Docker runs Linux
  containers. The images are built and tested for linux/arm64. The
  [Docker notes](../environments/docker/README.md) list the images, the tests run in each
  and the run command, which uses `--read-only` and `--network none`.
- Native: build each engine environment from its recipe in `environments/`. The table
  under Engine environments lists the recipes. Use this route where Docker does not run
  Linux containers.

`ALAMBIQ_ENV_BOLTZ_MPS` (Apple GPU) is available only as a native environment. Docker
images set the `ALAMBIQ_ENV_<NAME>` variables listed in the table inside the image. A
native installation sets them in the shell.

## Core

The core requires Prefect and PyYAML. Install the package from PyPI. The `quantitative`
extra is described below:

```bash
pipx install "alambiq[quantitative]"
```

The examples are in the repository. To run one from a clone, install the core in a virtual
environment:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[quantitative]"
.venv/bin/alambiq run examples/kinetics-from-measurements/campaign.yaml --report report.html
```

The last command runs the example and writes `report.html`. `alambiq capability list`
prints every capability, and `alambiq capability show ID` prints the request fields,
result, method and limits of one capability.

Extras are optional dependency groups of the core package. Install one or several in a
quoted argument, for example `pip install -e ".[quantitative,chemistry]"`. The table lists
the extras and the capabilities that import their packages.

| Extra | Adds | Used by |
| --- | --- | --- |
| `quantitative` | NumPy, SciPy, scikit-learn | `assay.*`, `operation.*`, `candidate.*`, `dna.optimize_coding`, `protein.explore_embeddings`, `protein.inspect`, `reaction.analyze_enzyme_panel`, `reaction.estimate_thermodynamics`, `simulation.analyze_binding_free_energy`, `simulation.compare_replica_structures`, `simulation.compare_scalar_couplings`, `simulation.inspect_trajectory`, `simulation.run_metadynamics`, `simulation.run_qmmm_dynamics`, `simulation.run_relative_free_energy`, `structure.prepare_reactive_coordinates`, `structure.prepare_relative_binding_inputs` |
| `chemistry` | RDKit | `reaction.*` (including molecular drawings), `simulation.run_relative_free_energy`, `structure.prepare_reactive_coordinates` |
| `routes` | DORAnet, RDKit | `reaction.explore_routes` |
| `thermodynamics` | eQuilibrator and its cache, RDKit | `reaction.estimate_thermodynamics` |
| `dossier` | Gemmi | reading structure files, for example in `protein.inspect`, `structure.compare` and `structure.detect_binding_pockets` |
| `construct` | Pydna, Biopython | `construct.*` |
| `coding-design` | DNA Chisel, Biopython | `dna.optimize_coding` |
| `trajectory` | MDTraj | `simulation.inspect_trajectory`, `simulation.compare_scalar_couplings`, `protein.inspect` (coordinate annotations) |
| `openmm`, `protein-repair` | OpenMM; OpenMM and PDBFixer | tests of the OpenMM-based engines, which import OpenMM in the Python environment of the core |
| `embedding-analysis` | scikit-learn, UMAP | `protein.explore_embeddings` |

## Engine environments

An engine runs in its own engine environment with its own pinned packages. Its
dependencies may conflict with those of the core without affecting it.

Set `ALAMBIQ_ENV_<NAME>` to the Python executable (`bin/python`) of an environment built
from its recipe. `alambiq env check` reports which engine environments are ready. For each
that is not, it reports the reason (for an unset variable, the variable to set) and the
recipe.

| Variable | Used by | Recipe |
| --- | --- | --- |
| `ALAMBIQ_ENV_MMSEQS` | `protein.search`, `protein.match_profiles`, `protein.annotate_function`, `protein.call_substitutions`, `sequence.generate_msa`, `enzyme.*` | below |
| `ALAMBIQ_ENV_ESM` | `protein.embed`, `protein.predict_tracks`, `protein.complete`, `protein.score_substitutions` | [protein-representations](../environments/protein-representations/README.md) |
| `ALAMBIQ_ENV_THERMOMPNN` | `protein.predict_stability_change` | [thermompnn](../environments/thermompnn/README.md) |
| `ALAMBIQ_ENV_P2RANK` | `structure.detect_binding_pockets` | [p2rank](../environments/p2rank/README.md) |
| `ALAMBIQ_ENV_OPENMM` | `structure.prepare_protein_system`, `simulation.run_molecular_dynamics`, `simulation.run_metadynamics` | [openmm](../environments/openmm/README.md) |
| `ALAMBIQ_ENV_BOLTZ` | `structure.predict_biomolecular_complex` (CPU and CUDA) | [boltz2](../environments/boltz2/README.md) |
| `ALAMBIQ_ENV_BOLTZ_MPS` | `structure.predict_biomolecular_complex` (Apple GPU) | [boltz2-mps](../environments/boltz2-mps/manifest.yaml) |
| `ALAMBIQ_ENV_OPENFF` | `structure.prepare_predicted_complex_for_dynamics`, `structure.prepare_protein_ligand_system`, `structure.prepare_finite_environment` | [openff-complex-preparation](../environments/openff-complex-preparation/README.md) |
| `ALAMBIQ_ENV_OPENFE` | `simulation.run_relative_free_energy`, `structure.prepare_relative_binding_inputs` | [binding-free-energy](../environments/binding-free-energy/README.md) |
| `ALAMBIQ_ENV_PYMBAR` | `simulation.analyze_reaction_free_energy` | [binding-free-energy](../environments/binding-free-energy/README.md) or [reaction-free-energy](../environments/reaction-free-energy/README.md) |
| `ALAMBIQ_ENV_MOLECULAR_PROPERTIES` | `molecule.compute_quantum_properties` | [molecular-properties](../environments/molecular-properties/README.md) |
| `ALAMBIQ_ENV_ADMET` | `molecule.predict_admet` | [admet](../environments/admet/README.md) |
| `ALAMBIQ_ENV_QMMM` | `simulation.evaluate_qmmm_state`, `simulation.run_qmmm_dynamics` | [qmmm](../environments/qmmm/README.md) |

Create the MMseqs2 environment with the pinned release, conda package version `18.8cc5c`
(release 18-8cc5c):

```bash
mamba create --prefix "$HOME/.alambiq/environments/mmseqs2" \
  --override-channels -c conda-forge -c bioconda python=3.11 mmseqs2=18.8cc5c
export ALAMBIQ_ENV_MMSEQS="$HOME/.alambiq/environments/mmseqs2/bin/python"
```

The key of a step includes the identity of its engine environment and of each resource it
read (see [Reuse](concepts.md#reuse)). A rerun in the same run directory computes a step
again when the Python version, a declared package version or a declared tool version of
its engine environment changes.

## Resources

A resource is a model or a sequence database. Each file is pinned by its SHA-256 digest
and verified before use. No resource ships with the package, which carries only the
record of each one, and a run never downloads anything. Install the resources a campaign
reads before it runs, with `alambiq resource provision`.

```bash
alambiq resource list
alambiq resource show uniref50-2026_02
alambiq resource verify esmc-300m
alambiq resource provision thermompnn --yes
```

- `list` shows the known resources and which are installed.
- `show` prints the provider, license, size and location of a resource and, where its
  specification has a `build` field, how the resource is obtained.
- `verify` checks the installed files against the digests.
- `provision` downloads the files of a resource, or fetches them with the downloader named
  in the specification, and verifies them. It refuses a resource that is built locally from
  commands in its `build` field.

`provision` downloads a resource only when its specification in `src/alambiq/resources/`
lists a download URL for every file or a `fetch` method, and only with `--yes`. Without
`--yes` it prints what it would download, what it would build, the license and the hosts, and
does nothing. Boltz-2 is fetched by Boltz's downloader inside the Boltz environment.
Swiss-Prot is fetched and built in the mmseqs environment: `provision` downloads the two
UniProt files, checks their size and MD5 digest, builds the MMseqs2 database and the
annotation table, and verifies the result against the record. UniProt replaces the files at
its `current_release` address when it publishes a new release, and the download then fails
its check. UniRef50 and Pfam are built locally with the commands in the `build` field of
their specifications.

Resources are stored in `~/.alambiq/resources`, or in `ALAMBIQ_RESOURCE_HOME`. Set
`ALAMBIQ_RESOURCE_<NAME>` (the resource name in capitals, punctuation as underscores) to
use a resource stored elsewhere.

## Run directories

A run writes to the directory given by `--run-dir`, or to a new temporary directory
without it. The run directory holds the content-addressed artifact store, the step cache
used for reuse and the run record (`run.json`). Stored results and the run record write the
location of the artifact store as the token `{store}`. A run directory can therefore be
moved or copied to another machine, and its stored results remain usable.

## Prefect state

Prefect stores its state in `~/.alambiq/prefect-VERSION`, where VERSION is the installed
Prefect version. The state is separate from any other Prefect installation and from other
Prefect versions on the machine. `ALAMBIQ_PREFECT_HOME` replaces this location.
`PREFECT_HOME`, when set, takes precedence over both.

## Platforms

All steps of a campaign run on the machine that starts the run. On a cluster, run a
campaign in a single-node job.

The core is tested on macOS (Apple silicon) and Linux x86_64. CI installs the wheel with
every pip-installable extra and runs the tests on ubuntu-latest (Linux x86_64) and macos-14
(Apple silicon) with Python 3.11 and 3.13.

The verification files in the directories under `environments/` record runs on two
platforms.

- macOS Apple silicon (osx-arm64): binding-free-energy, openff-complex-preparation,
  openmm, protein-representations, qmmm and thermompnn.
- Linux ARM64 (linux-aarch64): binding-free-energy, boltz2, openff-complex-preparation,
  openmm, p2rank, protein-representations, qmmm and reaction-free-energy.

Docker images for every engine environment except the Apple GPU path were built for
linux/arm64 on 2026-10-23 and tested on one Linux ARM64 host with an NVIDIA GPU (see
[the Docker notes](../environments/docker/README.md)).
