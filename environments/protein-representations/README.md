# Protein representation engine environment

This engine environment installs the packages that run the ESM C and ESM3 models.
`protein.explore_embeddings` needs only the `quantitative` extra. Runs on macOS Apple
Silicon (CPU, MPS) and on the Spark host (Linux ARM64 with an NVIDIA GB10 GPU: CPU, CUDA)
are recorded in the verification files linked below. The [Docker image](../docker/README.md)
for ESM C and ESM3 installs the pinned packages of this recipe.

## Installation

From the repository root, choose a new environment prefix:

```bash
mamba env create --prefix /absolute/path/to/alambiq-esmc \
  --file environments/protein-representations/environment.yaml --yes
export ALAMBIQ_ENV_ESM=/absolute/path/to/alambiq-esmc/bin/python
alambiq env check esm
```

Run Alambiq from the core environment. The engine runs under the Python named by
`ALAMBIQ_ENV_ESM`. The recipe pins the model packages and the numerical packages; Mamba and
pip resolve all other dependencies. The package inventories in [locks](locks/) list the
packages of the installation on macOS ARM64, and the Spark
[verification file](verification.accelerators.linux-aarch64.json) lists those of the
installation on the Spark host. They apply to these two platforms. `ALAMBIQ_ENV_ESM` is
the one setting for both models. `alambiq env check esm` reports whether `esm`, `torch`,
`transformers` and `numpy` import in the environment.

On the Spark host, `pip check` reports one error: the wheel
`nvidia-cusparselt-cu13==0.8.0` has a platform tag that does not match Linux ARM64. CUDA
inference ran on the Spark host, and the wheel metadata is unchanged. The
[verification file](verification.accelerators.linux-aarch64.json) contains the
diagnostic, the package list and the results.

## Model resources

`protein.predict_tracks` (predicted annotations) and `protein.complete` (completion) need
the resource `esm3-small` (about 4.15 GB). Provision it with
`alambiq resource provision esm3-small --yes`. ESM3 runs in the same engine environment as
ESM C. The resource `esm3-small` holds a checkpoint and function decoder weights. To store
it outside the default store, set `ALAMBIQ_RESOURCE_ESM3_SMALL`. ESM3 was run on CPU and
CUDA on the Spark host (see the verification file above).

The commands for `esmc-300m` work in an installed package and in a source checkout:

```bash
alambiq resource list
alambiq resource show esmc-300m
alambiq resource provision esmc-300m --yes
alambiq resource verify esmc-300m
```

`show` is read-only. `provision` downloads about 1.33 GB of pinned model, configuration and
tokenizer files and the upstream notices. Without `--yes` it reports the files and their
sources and stops. It verifies each file before it adds the resource to the store. `verify`
rechecks an installed resource against its pinned file list. Only `provision` downloads.

The default resource store is `~/.alambiq/resources`. Set `ALAMBIQ_RESOURCE_HOME` before
both provisioning and analysis to choose another store, or set `ALAMBIQ_RESOURCE_ESMC_300M`
to the directory of an existing copy. The directory of the `esmc-300m` resource in the store
is named by its content address,
`sha256/ea536daaab01a513a6d0d124d330581569b3074d4a7295331a386097ad3ff15b`.

Every file is checked against its expected size and SHA-256. A download whose files fail the
check adds nothing to the store, and the command can be run again. `alambiq resource show`
also prints an identity, which differs from the content address. The identity is computed
from the version and the path and SHA-256 of each of the 8 model and notice files.

## Running ESM C embeddings

The `device` option of `protein.embed` is `cpu` (default), `mps` or `cuda`. The capability
uses the requested device. `mps` was run on Apple Silicon and `cuda` on the Spark host. The
capability accepts a FASTA file with 1 to 64 proteins of 1 to 1,024 residues each
(512 residues for `esm3-small`) and embeds one protein at a time. The verification files
record the inputs that were run. `alambiq capability show protein.embed` lists the options,
the environment and the resources.

The example `examples/representative-panel/embed.yaml` runs `protein.embed` on the 6-protein
panel with the `esmc-300m` resource. The campaign `examples/representative-panel/campaign.yaml`
then explores the stored vectors with `protein.explore_embeddings`, which needs only the
`quantitative` extra, and runs without the model:

```bash
alambiq run examples/representative-panel/embed.yaml --report embed-report.html
alambiq run examples/representative-panel/campaign.yaml --report report.html
```

The report contains controls to export selected sequences and vectors. ESM C returns
embeddings; `protein.predict_tracks` runs ESM3-small and returns predicted InterPro
regions, function keywords and secondary structure.

The [portable Mac verification](verification.portable.osx-arm64.json) records an
installation in a new prefix, runs on CPU and MPS, and the 6-protein campaign with a freshly
downloaded resource. The maximum difference between the pooled vectors and those of the
original environment is 0. The selection and export controls of the report were exercised in
a browser. For a 142-residue sequence, the CPU and MPS embeddings differ by at most 2.22e-6
per value.
