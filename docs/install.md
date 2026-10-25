# Installation

Alambiq consists of the core, the engine environments and the resources. The core is the
`alambiq` Python package with its required dependencies, Prefect and PyYAML. It requires
Python 3.11 or newer and is tested on macOS (Apple silicon) and Linux x86_64 (see
Platforms). Engine environments provide the scientific software that some capabilities
need, and some of these capabilities declare resources (models and databases). Install
the core, run `examples/kinetics-from-measurements`, then install the engine
environments and resources that the capabilities in use require.

## Docker and native installation

Two routes provide the engine environments.

- Docker: one image per engine environment. The documented run command uses
  `--read-only` and `--network none`. The
  [Docker notes](../environments/docker/README.md) list the images, the tests run in each
  and the run command. Use this route where Docker runs Linux containers; the images are
  built and tested for linux/arm64.
- Native: build each engine environment from its recipe in `environments/`; the table
  under Engine environments lists the recipes. Use this route where Docker does not run
  Linux containers.

The routes differ in one respect: `ALAMBIQ_ENV_BOLTZ_MPS` (Apple GPU) is available
only as a native environment. Docker images set the `ALAMBIQ_ENV_<NAME>` variables listed in the table
inside the image; a native installation sets them in the shell.

## Core

Install the package from PyPI. The `quantitative` extra is described below:

```bash
pipx install "alambiq[quantitative]"
```

The examples are in the repository. To run one from a clone, install the core in a
virtual environment:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[quantitative]"
.venv/bin/alambiq run examples/kinetics-from-measurements/campaign.yaml --report report.html
```

The last command runs the example and writes `report.html`. `alambiq capability list`
prints every capability, and `alambiq capability show ID` prints the request fields,
result, method and limits of one capability.

Extras are optional dependency groups of the core package. One or several extras are
installed in a quoted argument, for example `pip install -e ".[quantitative,chemistry]"`.
The table lists the extras and the capabilities that import their packages.

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

An engine runs in its own engine environment with its own pinned packages, so the
dependencies of an engine may conflict with those of the core without affecting it. Set
`ALAMBIQ_ENV_<NAME>` to the Python executable (`bin/python`) of an environment built from
its recipe. `alambiq env check` reports which engine environments are ready and, for each
that is not, the reason (for an unset variable, the variable to set) and the recipe.

| Variable | Used by | Recipe |
| --- | --- | --- |
| `ALAMBIQ_ENV_MMSEQS` | `protein.search`, `protein.match_profiles`, `sequence.generate_msa`, `enzyme.*` | below |
| `ALAMBIQ_ENV_ESM` | `protein.embed`, `protein.predict_tracks`, `protein.complete`, `protein.score_substitutions` | [protein-representations](../environments/protein-representations/README.md) |
| `ALAMBIQ_ENV_THERMOMPNN` | `protein.predict_stability_change` | [thermompnn](../environments/thermompnn/README.md) |
| `ALAMBIQ_ENV_P2RANK` | `structure.detect_binding_pockets` | [p2rank](../environments/p2rank/README.md) |
| `ALAMBIQ_ENV_OPENMM` | `structure.prepare_protein_system`, `simulation.run_molecular_dynamics`, `simulation.run_metadynamics` | [openmm](../environments/openmm/README.md) |
| `ALAMBIQ_ENV_BOLTZ` | `structure.predict_biomolecular_complex` (CPU and CUDA) | [boltz2](../environments/boltz2/README.md) |
| `ALAMBIQ_ENV_BOLTZ_MPS` | `structure.predict_biomolecular_complex` (Apple GPU) | [boltz2-mps](../environments/boltz2-mps/manifest.yaml) |
| `ALAMBIQ_ENV_OPENFF` | `structure.prepare_predicted_complex_for_dynamics`, `structure.prepare_protein_ligand_system`, `structure.prepare_finite_environment` | [openff-complex-preparation](../environments/openff-complex-preparation/README.md) |
| `ALAMBIQ_ENV_OPENFE` | `simulation.run_relative_free_energy`, `structure.prepare_relative_binding_inputs` | [binding-free-energy](../environments/binding-free-energy/README.md) |
| `ALAMBIQ_ENV_PYMBAR` | `simulation.analyze_reaction_free_energy` | [binding-free-energy](../environments/binding-free-energy/README.md) or [reaction-free-energy](../environments/reaction-free-energy/README.md) |
| `ALAMBIQ_ENV_QMMM` | `simulation.evaluate_qmmm_state`, `simulation.run_qmmm_dynamics` | [qmmm](../environments/qmmm/README.md) |

Create the MMseqs2 environment with the pinned release (conda package version `18.8cc5c`,
which is release 18-8cc5c):

```bash
mamba create --prefix "$HOME/.alambiq/environments/mmseqs2" \
  --override-channels -c conda-forge -c bioconda python=3.11 mmseqs2=18.8cc5c
export ALAMBIQ_ENV_MMSEQS="$HOME/.alambiq/environments/mmseqs2/bin/python"
```

The key of a step includes the identity of its engine environment and of each resource it
read (see [Reuse](concepts.md#reuse)). A rerun in the same run directory computes a step
again when the Python version, a declared package version or a declared tool version of
its engine environment has changed.

## Resources

Resources (models and sequence databases) are pinned by the SHA-256 digest of each file
and verified before use. They are installed before a run, with
`alambiq resource provision` or, for the locally built databases, with the commands in the
`build` field of the specification, as described below.

```bash
alambiq resource list
alambiq resource show uniref50-2026_02
alambiq resource verify esmc-300m
alambiq resource provision thermompnn --yes
```

`list` shows the known resources and which are installed. `show` prints the provider,
license, size and location of a resource and, where its specification has a `build` field,
how the resource is obtained. `verify` checks the installed files against the digests.
`provision` downloads the files of a resource, or fetches them with the downloader named
in the specification, and verifies them. It refuses a resource that is built locally.

Resources are stored in `~/.alambiq/resources`, or in `ALAMBIQ_RESOURCE_HOME`. Set
`ALAMBIQ_RESOURCE_<NAME>` (the resource name in capitals, punctuation as underscores) to
use a resource stored elsewhere. `alambiq resource provision NAME` downloads a resource
only when its specification in `src/alambiq/resources/` lists a download URL for every
file or a `fetch` method, and only with `--yes`; without `--yes` it prints the size, the
license and the hosts and downloads nothing. UniRef50 and Pfam are built locally with the
commands in the `build` field of their specifications. Boltz-2 is fetched by Boltz's
downloader inside the Boltz environment. Every resource is verified against the digests
in its specification.

## Run directories

A run writes to the directory given by `--run-dir`; without `--run-dir`, it writes to a
new temporary directory. The run directory holds the content-addressed artifact store, the
step cache used for reuse and the run record (`run.json`). Stored results and the run
record write the location of the artifact store as the token `{store}`, so a run
directory can be moved or copied to another machine and its stored results remain usable.

## Prefect state

Prefect stores its state in `~/.alambiq/prefect-VERSION`, where VERSION is the installed
Prefect version, so the state is separate from any other Prefect installation and from
other Prefect versions on the machine. `ALAMBIQ_PREFECT_HOME` replaces this location.
`PREFECT_HOME`, when set, takes precedence over both.

## Platforms

CI installs the wheel with every pip-installable extra and runs the tests on
ubuntu-latest (Linux x86_64) and macos-14 (Apple silicon) with Python 3.11 and 3.13. The
verification files in the directories under `environments/` record runs on macOS Apple
silicon (osx-arm64) for binding-free-energy, openff-complex-preparation, openmm,
protein-representations, qmmm and thermompnn, and on Linux ARM64 (linux-aarch64) for
binding-free-energy, boltz2, openff-complex-preparation, openmm, p2rank,
protein-representations, qmmm and reaction-free-energy. Docker images for every
engine environment except the Apple GPU path were built for linux/arm64 on 2026-10-23 and
tested on one Linux ARM64 host with an NVIDIA GPU (see
[the Docker notes](../environments/docker/README.md)).
All steps of a campaign run on the machine that starts the run; on a cluster, run a
campaign in a single-node job.
