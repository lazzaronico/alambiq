# P2Rank runtime

P2Rank 2.5.1 is a Java application. It runs outside the Alambiq core environment and adds
no Python or GPU dependency to it. The engine requires Java 17 to 23 and records the Java
version it finds. `alambiq resource verify p2rank-2.5.1` verifies the P2Rank files. The
capability is `structure.detect_binding_pockets`.

## Linux ARM64 installation

The [recipe](environment.yaml) was installed with Mamba on the Spark host (the NVIDIA Spark:
Linux ARM64 with a GB10 GPU). It installed conda-forge OpenJDK 17.0.18. The output of
`java -version` reports `17.0.18-internal+0-adhoc.rattler.src`. The
[package inventory](linux-aarch64.packages.json) lists the URL, version, size, SHA-256 and
license of each of the 59 packages. It records one installation. The recipe pins only
`openjdk=17`, so a later solve can select other package versions.

```bash
mamba env create --prefix /absolute/path/to/p2rank-java17 \
  --file environments/p2rank/environment.yaml
```

The recorded installation contains no Python interpreter (none of its 59 packages is one),
so `/absolute/path/to/p2rank-java17/bin/python` does not exist. The engine module runs under
the Python named by `ALAMBIQ_ENV_P2RANK`, and it starts the first `java` on the `PATH` of
that process. That `PATH` consists of the directory of the named Python followed by the
`PATH` of the host process. `ALAMBIQ_ENV_P2RANK` can therefore name any Python (for
example the Python of the Alambiq core environment) if `java` is found on that `PATH`,
either beside that Python or in a directory on the host `PATH`. To use the Java prefix of
the recipe, put its `bin` directory first on the host `PATH`:

```bash
export PATH=/absolute/path/to/p2rank-java17/bin:$PATH
export ALAMBIQ_ENV_P2RANK=/absolute/path/to/python
alambiq env check p2rank
```

`alambiq env check p2rank` reports whether `java` is found. The
[Docker layer](../docker/Dockerfile.p2rank) uses this arrangement: `ALAMBIQ_ENV_P2RANK` is
the core Python, and the Java prefix is first on `PATH`.

Size: the package downloads total 221,571,201 bytes (221.6 MB; `package_download_bytes` in
the inventory). The installed Java directory occupies 520,007,015 bytes (520.0 MB;
`java_directory_bytes` in the [verification file](verification.linux-aarch64.json)).

Platforms: the Linux recipe was run on Linux ARM64.

A complete package inventory requires package metadata with SHA-256 fields. A clean
`CONDA_PKGS_DIRS` gives complete metadata. Running the engine needs a Java version from 17 to
23 and the P2Rank resource.

## P2Rank resource

The pinned resource is `p2rank-2.5.1` (600 files, 304.6 MB). It is unpacked from the
upstream release archive named in the resource record. The archive is
`https://github.com/rdk/p2rank/releases/download/2.5.1/p2rank_2.5.1.tar.gz` (SHA-256
`d243f2d9036ac053fefb9407b5fe1c85f4fe077c519fd975ac585e995feab274`). Unpack it into the
location of the resource, which `alambiq resource show p2rank-2.5.1` prints along with
whether the resource is installed. The resource record lists its files as `p2rank/...`, so
the unpacked directory `p2rank/` lies directly inside the location. The default location is the content-addressed directory
under `~/.alambiq/resources` (or `ALAMBIQ_RESOURCE_HOME`). Set
`ALAMBIQ_RESOURCE_P2RANK_2_5_1` to use another location. Verify the resource and the
environment with:

```bash
alambiq resource verify p2rank-2.5.1
alambiq env check p2rank
```

If the resource is missing or fails verification, the run stops with an error.

## Docker layer

The [Docker layer](../docker/Dockerfile.p2rank) installs Java from this recipe and uses a
read-only external P2Rank resource. The 21 P2Rank tests and
`examples/pockets-and-complex/campaign.yaml` passed in it (2026-10-23). Run the image with an executable temporary directory (the JNI
library is unpacked there), as the [Docker notes](../docker/README.md#run) describe.

## Example campaign

The example `examples/pockets-and-complex/campaign.yaml` runs P2Rank on the 1A3N
hemoglobin structure. It needs the `dossier` extra, which installs Gemmi to parse the
structure:

```bash
alambiq run examples/pockets-and-complex/campaign.yaml --report report.html
```

The `structure_kind` option is `experimental` or `predicted`. `experimental` selects the
`default` configuration of P2Rank. `predicted` selects the `alphafold` configuration and
omits the experimental B-factor feature. The command
`alambiq capability show structure.detect_binding_pockets` lists the options.

## Moving the resource

To move the resource to another machine, unpack the original archive again (see "P2Rank
resource"), or copy the regular files only (no macOS AppleDouble files or extended attributes) and run
`alambiq resource verify p2rank-2.5.1`. A tar copy that contained macOS metadata failed
verification. A copy without metadata matched the original resource record.

## Verification

The [verification file](verification.linux-aarch64.json) records runs with `structure_kind`
`experimental` (15 pockets) and `predicted` (2 pockets), the integrity check of 12 stored
artifacts per run, and browser checks of the report without network requests. The saved
Boltz prediction of the `predicted` run was produced on a Mac.
