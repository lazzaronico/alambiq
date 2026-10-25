# QM/MM engine environment

The [Mamba recipe](environment.yaml) installs Python 3.12, OpenMM 8.6.0 and PySCF 2.14.0 for
the CPU PBE/6-31G** evaluation (identifier `pyscf.molecular-pbe-gaussian.v1` in the
verification files). Create the environment separately
from the Alambiq core environment, then set `ALAMBIQ_ENV_QMMM` to its Python executable.

```bash
mamba env create --prefix /path/to/qmmm-env --file environments/qmmm/environment.yaml
export ALAMBIQ_ENV_QMMM=/path/to/qmmm-env/bin/python
alambiq env check qmmm
```

`ALAMBIQ_ENV_QMMM` is the engine environment of `simulation.evaluate_qmmm_state` and
`simulation.run_qmmm_dynamics` (`alambiq capability show` names the environment of each
capability). They take a prepared molecular state, for instance from
`structure.prepare_finite_environment` in the OpenFF environment.

The [Apple Silicon explicit lock](osx-arm64.explicit.txt) lists the 69 packages of the
recorded installation, and the [package records](osx-arm64.packages.json) list the source,
hash and license of each package. This command installs the packages of the lock:

```sh
mamba create --prefix /path/to/qmmm-env --file environments/qmmm/osx-arm64.explicit.txt
```

## Verification files for the CPU environment

- The [finite-enzyme evaluation on Linux ARM64](verification.openff-preparation-linux-aarch64-20261008.json)
  records a CPU evaluation of the Sage/AshGC finite preparation (8,677 physical atoms, with
  the 24 atoms of chorismate in the quantum region and 118 electrons). The PBE/6-31G** SCF
  converges in 40 cycles. A separate implementation reconstructs the force partition, the
  classical and boundary contributions, the electronic density and population, and
  regenerates the report. All 4 preparation stages are reused, and in the second run engine
  execution is disabled. The evaluation covers a single geometry.
- The [same preparation on a Mac](verification.openff-preparation-osx-arm64-20261008.json)
  gives the same evaluation: forces, density and population are reconstructed by a separate
  implementation, the preparation stages are reused, and the regenerated report is
  byte-identical to the original report. The SCF converges in 40 cycles on both hosts. The
  electronic energies of the Linux and Mac runs differ by 2.27e-13 hartree, and the forces on
  all 8,677 atoms by at most 1.83e-8 kJ/mol/nm. These values are for the same single
  geometry.
- The same Mac file records an offline installation in a new prefix of the 69 packages of the
  Apple Silicon lock, from cached archives, made with the command
  `mamba create --offline --prefix /path/to/qmmm-env --file environments/qmmm/osx-arm64.explicit.txt`.
  The numerical test suite (35 tests of the earlier test layout; see "Installation check")
  passed 33 tests and skipped 2 comparisons that need CUDA. The 5-atom state-to-trajectory
  campaign (3 atoms in the quantum region, 2 equilibration steps, 8 production steps) ran in
  the new prefix, and a second run with engine execution disabled restored a report that is
  byte-identical to the original report.

## Linux CPU installation and Docker

The [Linux ARM64 CPU lock](linux-aarch64-cpu.explicit.txt) recreates the recorded
74-package environment. Its [package records](linux-aarch64-cpu.packages.json) list the
URL, checksum and license of each package:

```sh
mamba create --prefix /path/to/qmmm-env --file environments/qmmm/linux-aarch64-cpu.explicit.txt
export ALAMBIQ_ENV_QMMM=/path/to/qmmm-env/bin/python
```

The [Docker layer](../docker/Dockerfile.qmmm) installs the CPU environment in `/opt/qmmm` in
an image that also contains the core. Build from the repository root:

```sh
docker build -f environments/docker/Dockerfile -t alambiq:core .
docker build -f environments/docker/Dockerfile.qmmm \
  --build-arg BASE_IMAGE=alambiq:core -t alambiq:qmmm .
```

The layer sets `ALAMBIQ_ENV_QMMM` to `/opt/qmmm/bin/python`. On 2026-10-23, the
environment check passed and 65 tests passed in the image; 20 low-level engine tests were
skipped because they import PySCF in the test process itself (see the
[Docker notes](../docker/README.md)).

The [CPU container file](verification.cpu-container.json) was made from source commit
`768a953`. It records the 5-atom water and ion example on Linux ARM64 outside Docker and in Docker:
state evaluation, density-bound thermal sampling (4 production frames, 11 electronic
calls), the individual Python capabilities, the reuse of exported results and the command
line. Between the two platforms, positions differ by at most 4.44e-16 Å and potentials by at
most 4.07e-10 kJ/mol, within the limits of 1e-8 Å and 1e-5 kJ/mol. A separate implementation
restored 9 runs and reproduced the saved reports. The runs outside Docker and in Docker have the
same conda package-set SHA-256 (`b4ad7bac6cba647140ce44d49756d5b250546518437a3bd8d97e7a2a732d26fd`),
and the lock recreates the 74 packages. The layer was built on a core image without optional
extras.

## Linux ARM64 CUDA installation

The capabilities `simulation.evaluate_qmmm_state` and
`simulation.run_qmmm_dynamics` request the CPU backend (`backend: cpu`), and the present
engine rejects any other backend value. The procedure below installs the environment
recorded in [verification.cuda-install.json](verification.cuda-install.json), which was
run with the request interface of source commit `2cb9936`.

The [explicit Conda lock](linux-aarch64-cuda13.explicit.txt) and its
[package records](linux-aarch64-cuda13.packages.json) list the 87 Conda packages. The
installer adds GPU4PySCF 1.8.1, CuPy CUDA13x 14.0.1 and cuda-pathfinder 1.8.2 with pip. The
[verification file](verification.cuda-install.json) lists the hashes of these distributions
and of the wheel built for GPU4PySCF.

The [source installer](install_cuda.py) builds the GPU4PySCF extension for CUDA 13.0 and
CUDA architecture 121 (GB10), with LibXC built from source. The prerequisites are the CUDA
toolkit with `nvcc`, GNU C/C++ at `/usr/bin/gcc` and `/usr/bin/g++`, CMake, Make, Git and a
Fortran compiler. The recorded build used GNU C/C++ 13.3 and GNU Fortran 15.3. Create a
separate scientific environment and select a fresh build directory whose path contains no
whitespace. The second `mamba create` command creates a build environment that provides
`gfortran`:

```bash
mamba create --prefix /path/to/qmmm-cuda --file environments/qmmm/linux-aarch64-cuda13.explicit.txt
mamba create --prefix /path/to/qmmm-build -c conda-forge gfortran_linux-aarch64=15.3.0
/path/to/qmmm-cuda/bin/python environments/qmmm/install_cuda.py \
  --prefix /path/to/qmmm-cuda \
  --work /path/to/qmmm-cuda-build \
  --cuda /usr/local/cuda-13.0 \
  --fortran /path/to/qmmm-build/bin/aarch64-conda-linux-gnu-gfortran
export ALAMBIQ_ENV_QMMM=/path/to/qmmm-cuda/bin/python
export CUDA_PATH=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/path/to/qmmm-cuda/lib:$CUDA_PATH/lib64
alambiq env check qmmm
```

The installer performs these steps:

1. It checks the prerequisites, the fresh build directory, and that the environment contains
   Python 3.12, OpenMM 8.6, PySCF 2.14.0 and no GPU4PySCF.
2. It checks that the CUDA toolkit is version 13.0.
3. It downloads the GPU4PySCF source and the CuPy and cuda-pathfinder wheels, and verifies
   the SHA-256 of each file. The build downloads the CUTLASS and LibXC sources, so setup
   needs network access.
4. It builds the GPU4PySCF wheel with 4 parallel jobs, and verifies the commits of CUTLASS
   and LibXC.
5. It installs the three wheels without dependency resolution, from the verified files.
6. It checks the imports in isolated mode and a GPU calculation.

The logs, the built wheel, the sources, the compiler identity and the installation records
remain in the build directory, including after a failure.

The [verification file](verification.cuda-install.json) records a fresh installation on
Linux ARM64 with a GB10 GPU, made from source commit `2cb9936`. All 35 tests of the earlier
test layout passed. The
tests are in the modules `test_qmmm_native`, `test_qmmm_pyscf`, `test_qmmm_runner`,
`test_qmmm_coupling` and `test_qmmm_dynamics_runner` of the earlier test layout. A CUDA
state-to-sampling campaign for the 5-atom example completed. The same campaign was run from
separate Python calls and with reuse of exported results. A separate implementation
reproduced the saved artifacts and the report after relocation. The campaign also completed
on Apple CPU.

The engine imports `openmm`, `pyscf`, `cupy` and `gpu4pyscf` under the selected Python in
isolated mode. `CUDA_PATH` and the library path give the locations of the CUDA toolkit and of
the shared libraries of the environment. The source installer reproduces the Linux ARM64
CUDA 13.0 build recorded above. The CPU Mamba and Docker recipes have the verification files
described above.

## Installation check

`alambiq env check qmmm` confirms that `pyscf`, `openmm` and `numpy` import in the selected
environment. The test counts above (35 tests) come from the test layout of the dates in the
verification files; the module names of that layout no longer exist. The engine tests are
now `tests/core/test_engine_qmmm_*.py`. The Docker image ran the current tests with
65 passed and 20 skipped (see "Linux CPU installation and Docker"). The earlier suite
covered:

- energy and force comparisons between CPU and CUDA for neutral and charged systems,
- independent derivatives,
- the mapping of prepared atoms and charges,
- thermal sampling,
- error handling.

A CPU environment skipped the CUDA comparisons; the CUDA environment ran all 35 tests.

## Component notices

The installed OpenMM, PySCF and GPU4PySCF distributions include their upstream license
notices. PySCF and GPU4PySCF are licensed under Apache-2.0. Redistribution of a built wheel
or environment requires the notices of CUTLASS, LibXC and the CUDA components (CUDA: NVIDIA
terms). The
[method guide](../../docs/methods/binding-and-qmmm.md#evaluating-and-sampling-qmmm) describes
the quantum region, the classical partition, the Gaussian embedding, the convergence
settings and the recorded units.
