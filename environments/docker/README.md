# Docker

The Dockerfiles in this directory define an image of the core and one image for each engine
environment except `boltz_mps`. Each engine image contains the core and the engine
environment and sets the variable `ALAMBIQ_ENV_<NAME>` of that environment (the OpenFE
image also sets `ALAMBIQ_ENV_PYMBAR` and `ALAMBIQ_ENV_OPENMM`). The Mamba recipes in the other `environments/` folders
build the same engine environments outside Docker, for machines that cannot run Linux
containers, such as macOS.

The images build from the repository root. The core image installs from `uv.lock`, as an
installation outside Docker does; each engine layer installs from the recipe of its
environment. The entry point of an image is `alambiq`, so `docker run IMAGE run CAMPAIGN ...`
is equivalent to `alambiq run`.

## Verification

Every image was built on 2026-10-25 on the Spark host: Linux ARM64 with an NVIDIA GB10 GPU,
driver 580.159.03 and Docker 29.2.1. For each image, `alambiq env check` returned ready, and
the repository tests that use that engine environment were run in the image. The
verification file with the image sizes, the test counts and the skipped tests is
[verification.linux-arm64-spark-20261025.json](verification.linux-arm64-spark-20261025.json).

| Layer | Dockerfile | Environment variable | Checks run in the image |
| --- | --- | --- | --- |
| core | `Dockerfile` | none | the examples that need no engine: all completed, with a read-only root file system and no network |
| MMseqs2 | `Dockerfile.mmseqs` | `ALAMBIQ_ENV_MMSEQS` | 166 passed, 4 skipped; 3 examples |
| P2Rank | `Dockerfile.p2rank` | `ALAMBIQ_ENV_P2RANK` | 21 passed; 1 example |
| ThermoMPNN | `Dockerfile.thermompnn` | `ALAMBIQ_ENV_THERMOMPNN` | 47 passed |
| pymbar | `Dockerfile.reaction-free-energy` | `ALAMBIQ_ENV_PYMBAR` | 32 passed, 2 skipped |
| QM/MM | `Dockerfile.qmmm` | `ALAMBIQ_ENV_QMMM` | 65 passed, 20 skipped |
| OpenMM (CUDA 13) | `Dockerfile.openmm` | `ALAMBIQ_ENV_OPENMM` | 165 passed; the GB3 ensemble example (protein GB3, PDB 1P7E) with 3 replicas of 20 ps on the GPU |
| OpenFF complex preparation | `Dockerfile.complex` | `ALAMBIQ_ENV_OPENFF` | 72 passed, 2 skipped |
| ESM C and ESM3 | `Dockerfile.models` | `ALAMBIQ_ENV_ESM` | 93 passed |
| Boltz-2 | `Dockerfile.boltz` | `ALAMBIQ_ENV_BOLTZ` | 56 passed, 1 skipped (the Apple GPU case) on the GPU, with the `boltz2-2.2.1` resource mounted read-only |
| OpenFE | `Dockerfile.openfe` | `ALAMBIQ_ENV_OPENFE`, `ALAMBIQ_ENV_PYMBAR`, `ALAMBIQ_ENV_OPENMM` | 141 passed on the GPU |

Tested platform: Linux ARM64. Boltz on `mps` runs outside Docker.

## Build

Build the core once with the extras that the campaigns to be run use, then add the layers
for the engines to be used on top of it:

```bash
docker build -f environments/docker/Dockerfile \
  --build-arg 'ALAMBIQ_EXTRAS=quantitative dossier trajectory chemistry routes thermodynamics construct coding-design embedding-analysis' \
  -t alambiq:core .
docker build -f environments/docker/Dockerfile.mmseqs --build-arg BASE_IMAGE=alambiq:core -t alambiq:mmseqs .
docker build -f environments/docker/Dockerfile.p2rank --build-arg BASE_IMAGE=alambiq:core -t alambiq:p2rank .
```

- The extras are those in `pyproject.toml`. An extra that `pyproject.toml` does not define
  fails the locked install.
- Each layer takes the core image as `--build-arg BASE_IMAGE=<tag>`.
- The `models` and `boltz` layers require a core built on Python 3.12
  (`--build-arg PYTHON_IMAGE=python:3.12-slim-bookworm`); their default base images are
  `alambiq:models-base` and `alambiq:boltz-base`. The recorded Python 3.12 core used the
  extras `quantitative dossier chemistry embedding-analysis`; the `chemistry` extra enables
  the Boltz tests that import RDKit.
- The `openfe` layer also copies in the OpenMM engine, because a relative-binding campaign
  prepares the protein with OpenMM 8.6 and the OpenFE environment contains OpenMM 8.4.
  Build `alambiq:openmm` first, or pass `--build-arg OPENMM_IMAGE=...`.
- The engine environment of a layer is built in a stage that starts from a plain Python
  image. Rebuilding the core therefore reuses the cached engine stage.

Check the environment of an image before using it:

```bash
docker run --rm alambiq:mmseqs env check mmseqs
```

## Run

```bash
mkdir -p results
docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,mode=1777,size=2g \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work:ro" -v "$PWD/results:/results" \
  alambiq:core run examples/kinetics-from-measurements/campaign.yaml \
  --run-dir /results/kinetics --report /results/kinetics.html
```

The repository is mounted read-only at `/work`, the working directory, so campaign and input
paths are relative to it. Results persist only in the mounted output directory, which must
exist and belong to the invoking user. Run concurrent campaigns in separate output
directories. A run needs no network unless the capability reads an external source, so the
example uses `--network none`.

The MMseqs2 and P2Rank images need an executable temporary directory. Replace the `--tmpfs`
option of the example with `--tmpfs /tmp:rw,exec,nosuid,nodev,mode=1777,size=4g` for them.
MMseqs2 writes and runs helper scripts in `/tmp`, and Java (P2Rank) maps a library that it
unpacks into `/tmp`. Without `exec`, the runs fail with
`Failed to execute ... searchslicedtargetprofile.sh` and `failed to map segment from shared object`.
The root file system stays read-only.

Pass the GPU with `--device nvidia.com/gpu=all` on a host whose NVIDIA container toolkit is
configured through CDI (the Spark host). The step must select the GPU in its `params`:
`platform: CUDA` for the OpenMM and OpenFE capabilities, `accelerator: gpu` for
`structure.predict_biomolecular_complex`, `device: cuda` for `protein.embed`.

## Models and databases

Models and databases are resources (`alambiq resource list`) stored outside the images. A
resource is verified against its pinned file digests before a run, and a run never downloads
a resource. Mount the store read-only and point the image at it:

```bash
-e ALAMBIQ_RESOURCE_HOME=/resources -v "$HOME/.alambiq/resources:/resources:ro"
```

Alambiq looks for a resource at `<store>/sha256/<content address>` unless
`ALAMBIQ_RESOURCE_<NAME>` names another directory. Mount the store, or one resource
directory, so that the directory is at `/resources/sha256/<digest>`. Alambiq records the
files it has hashed in a receipt file, by default under `<store>/verified`. A read-only
store cannot hold the receipt, so every run hashes all files again. To avoid this, set
`ALAMBIQ_RESOURCE_RECEIPTS` to a writable directory.

To fetch a resource, run `provision` once with the network and a writable store:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e ALAMBIQ_RESOURCE_HOME=/resources \
  -v "$HOME/.alambiq/resources:/resources" alambiq:boltz resource provision boltz2-2.2.1 --yes
```

`provision` requires `--yes`; without it, the command reports the download and stops. It
downloads into a staging directory, verifies every file against the pinned digests and then
moves the directory into the store. Boltz-2 is fetched by the downloader of Boltz, which
runs in the Boltz environment, so use the `boltz` image for it.

## Scratch space

Trajectory analysis writes intermediate files to temporary storage. The `/tmp` of the
container in the example above is a tmpfs of 2 GB, which is backed by memory. For an
analysis whose intermediate files exceed that size, mount a disk-backed scratch directory
and set `TMPDIR` to it:

```bash
-v "$PWD/scratch:/scratch" -e TMPDIR=/scratch
```

The result mount holds the artifacts and the run record. The scratch directory is a host
directory and remains after the container exits; delete it when its content is not needed.

## Repository tests in an image

`Dockerfile.tests` installs `pytest` in an image so that the repository tests of an engine
run in that image:

```bash
docker build -f environments/docker/Dockerfile.tests \
  --build-arg BASE_IMAGE=alambiq:openmm -t alambiq:openmm-tests .
docker run --rm --device nvidia.com/gpu=all --tmpfs /tmp:rw,exec,mode=1777 \
  --user "$(id -u):$(id -g)" -v "$PWD:/work:ro" -w /work alambiq:openmm-tests \
  tests/core/test_simulation_run_molecular_dynamics.py
```

The tests import the installed package, so a changed source requires a rebuild of the image.
Low-level engine tests (20 in the QM/MM set) skip because they import the packages of the
engine in the test process, which in an engine image is the Python of the core. The
capability tests run the engines through their own interpreters and pass.

## Outside the images

- Apple GPU: Boltz on `mps` runs outside Docker. See [boltz2-mps](../boltz2-mps/manifest.yaml).
- Other CUDA drivers: the CUDA layers install CUDA 13 components and were run with
  driver 580.159.03.
