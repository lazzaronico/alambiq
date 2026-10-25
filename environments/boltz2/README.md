# Boltz runtime environments

Boltz runs in its own engine environment, not in the Alambiq core environment. The model
files are a resource that is provisioned with `alambiq resource provision` and verified
before each run. A prediction request reads the model files from the verified resource and
takes the MSA of each protein from the request (`empty` when none is given).

The [boltz2-mps](../boltz2-mps/manifest.yaml) environment (experimental, Apple MPS)
installs Boltz 2.2.1 from the source of upstream pull request 527, which adds `mps` to the
accepted accelerator values of the Boltz command line. The CPU and CUDA environments
install the official Boltz 2.2.1 release.

## CPU environment

`environment.yaml` installs Python 3.11 and `boltz==2.2.1` from conda-forge and PyPI:

```bash
mamba env create -f environments/boltz2/environment.yaml
```

The packages of the installation made on macOS ARM64 are listed in the `osx-arm64.*` files
in [locks](locks/), and `manifest.yaml` records the checks of that installation.

## Linux ARM64 recipe

The recipe installs the CUDA environment on Linux ARM64 (the Spark host: Linux ARM64 with an
NVIDIA GB10 GPU). It was run on Linux ARM64 only; the Docker image is described in the
[Docker notes](../docker/README.md). The results of the runs and the installation limits
are in the [Spark Boltz verification file](verification.linux-aarch64.json).

```bash
mamba env create -f environments/boltz2/environment.linux-aarch64.yaml
mamba run -n alambiq-boltz2-spark python -m pip install \
  -r environments/boltz2/requirements.torch-cu130.txt
mamba run -n alambiq-boltz2-spark python -m pip install \
  -r environments/boltz2/requirements.txt
```

The recipe installs Boltz without its optional CUDA 12 kernel extra.

To use the environment, set `ALAMBIQ_ENV_BOLTZ` to the `bin/python` of the environment
created above (for the experimental MPS environment, set `ALAMBIQ_ENV_BOLTZ_MPS`
instead) and check it:

```bash
export ALAMBIQ_ENV_BOLTZ=/absolute/path/to/boltz-env/bin/python
alambiq env check boltz
alambiq resource show boltz2-2.2.1
alambiq resource verify boltz2-2.2.1
```

Provision the model files (6.20 GB in 3 files, from the sources that
`alambiq resource show boltz2-2.2.1` lists):

```bash
alambiq resource provision boltz2-2.2.1 --yes
```

The command requires `--yes`; without it, the command reports what it would download and
stops. It runs the downloader of Boltz in the Boltz environment into a staging folder, then
verifies every file against the pinned checksums before it moves the folder into the
resource store. To use existing files, point `ALAMBIQ_RESOURCE_BOLTZ2_2_1` at the folder
and run `alambiq resource verify boltz2-2.2.1`. The Alambiq core extras and the OpenMM
environment are separate from the Boltz environment.

The capability is `structure.predict_biomolecular_complex`. The command
`alambiq capability show structure.predict_biomolecular_complex` names the environment and
the resource it needs. Its `accelerator` option selects the device:

- `cpu` (the default) and `gpu` use the `boltz` environment. `mps` uses the `boltz_mps`
  environment.
- `gpu` fails if CUDA is unavailable, and `mps` fails if MPS is unavailable; the
  run uses the requested device.
- Every accelerator runs on one device (`--devices 1`), with custom kernels disabled
  (`--no_kernels`) and single-process data loading (`--num_workers 0`).

The example `examples/pockets-and-complex/complex.yaml` predicts benzene in T4 lysozyme
L99A:

```bash
alambiq run examples/pockets-and-complex/complex.yaml --report report.html
```

The [Docker verification file](../docker/verification.linux-arm64-spark-20261025.json)
records a run of it in the Boltz image on the GPU with 20 diffusion steps instead of the 200
`sampling_steps` of the shipped file; that run took 65 s.

A prediction checks that the requested device is available and verifies the model resource.
The result records the versions of Boltz, Torch and RDKit. The engine starts the Boltz
command with the Python that runs the engine, in isolated mode (`-I`).

The [portability verification file](verification.portable-runtime.json) records a second
installation on the Spark host, with the same conda package-set SHA-256 as the first. It
completed a prediction, the preparation of the predicted complex and the inspection of both.
The package inventories list the packages of the installations that were made.

The wheel `nvidia-cusparselt-cu13` 0.8.0, a dependency of PyTorch, has an ARM64 file name
but declares the tag `manylinux2014_sbsa` in its metadata. `pip check` reports the wheel as
unsupported on Linux ARM64. The prediction on the GPU completed with this report present
(see the [Spark Boltz verification file](verification.linux-aarch64.json)). The wheel
metadata is unchanged.
