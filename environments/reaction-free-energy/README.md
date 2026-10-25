# Reaction free-energy analysis

PyMBAR 4.2.0 computes a free-energy profile along a reaction coordinate from umbrella
windows supplied in the request. A replica is a complete set of umbrella windows run with
independent seeds; the request lists the replicas. The environment is CPU-only. Create it
from the repository root:

```sh
mamba env create --prefix /path/to/pymbar-env --file environments/reaction-free-energy/environment.yaml
export ALAMBIQ_ENV_PYMBAR=/path/to/pymbar-env/bin/python
alambiq env check pymbar
```

`ALAMBIQ_ENV_PYMBAR` is the engine environment of `simulation.analyze_reaction_free_energy`.
The command `alambiq capability show simulation.analyze_reaction_free_energy` names the
environment.

Use `mamba env create` to solve the YAML recipe afresh. Use `mamba create` with an explicit
lock to install exactly the recorded package set. For the recorded Apple Silicon package
set, use:

```sh
mamba create --prefix /path/to/pymbar-env \
  --file environments/reaction-free-energy/locks/osx-arm64.explicit.txt
```

The 34-package environment contains Python 3.11.16, PyMBAR 4.2.0, NumPy 2.4.6 and SciPy
1.17.1. Its package-set SHA-256 is
`cab44aacce8ebb2224d7a7e51a49c166d4936e50462e1345ec862305da239be5`. The
[package inventory](locks/osx-arm64.packages.json) lists the source URL, hash and license of
each package.

For the recorded Linux ARM64 package set, use:

```sh
mamba create --prefix /path/to/pymbar-env \
  --file environments/reaction-free-energy/locks/linux-aarch64.explicit.txt
```

This 39-package environment contains Python 3.11.16, PyMBAR 4.2.0, NumPy 2.4.6 and SciPy
1.17.1. Its package-set SHA-256 is
`acdb8ca2c52980cbc9c864218ce092b3a7f988f73a405da33152fcea5b2527ea`. The
[package inventory](locks/linux-aarch64.packages.json) lists the URL, checksum and license
of each package.

The [verification file](verification.linux-aarch64.json) records 5 analysis cases on Apple
Silicon, on Linux ARM64 outside Docker and in a Docker image. The cases are the double-well,
paired-energy and classical-flux examples and 2 saved molecular-window cases, including the
assembly of the complete summary and state tables. The file records the identifiers
`pymbar.umbrella-distance-difference.v1` and `pymbar.umbrella-target-reweighting.v1` of the
two methods. For each case, a Python script and a YAML campaign give identical scientific
output, a second run reuses the stored step results and produces the same saved report, and
the saved results and their artifacts reconstruct independently after transfer. The file
states a tolerance of 1e-7 (absolute and relative) for the comparison of complete numerical
records.

## Linux container

The [Docker layer](../docker/Dockerfile.reaction-free-energy) installs the CPU recipe
`environment.yaml` under `/opt/pymbar`. Build the Alambiq core image first, then add this
analysis environment:

```sh
docker build -f environments/docker/Dockerfile -t alambiq:core .
docker build -f environments/docker/Dockerfile.reaction-free-energy \
  --build-arg BASE_IMAGE=alambiq:core -t alambiq:reaction-analysis .
```

The layer sets `ALAMBIQ_ENV_PYMBAR` to `/opt/pymbar/bin/python`. On 2026-10-23, the
environment check passed and 32 tests passed and 2 were skipped in the image (see the
[Docker notes](../docker/README.md)). The container run in the verification file was made
from source commit `42984ad`. It ran the 5 cases with no network, a read-only root file
system and user ID 1000. The Linux environment outside Docker and the container environment
have the same package-set SHA-256,
`acdb8ca2c52980cbc9c864218ce092b3a7f988f73a405da33152fcea5b2527ea`. The 5 cases also passed
on a core image built without optional extras; NumPy, SciPy, Gemmi, MDTraj, RDKit and
scikit-learn are absent from its Python, and the numerical libraries come from the analysis
environment.

The [analysis guide](../../docs/methods/binding-and-qmmm.md#reaction-free-energy) defines the
harmonic coefficient convention, the density and basin observables, the decorrelation and
the uncertainty. PyMBAR is licensed under MIT. Alambiq is licensed under Apache-2.0.
