# Getting started

Python 3.11 or newer is required. Capabilities that run scientific software, for example
sequence search (MMseqs2), protein language models (ESM) and molecular dynamics (OpenMM),
need an engine environment, which is installed as described in
[Installation](install.md).

## Install

From the root of the repository, create a virtual environment and install the
[core](install.md) with the `quantitative` [extra](install.md#core) (NumPy, SciPy and scikit-learn):

```bash
python -m venv .venv
.venv/bin/pip install -e ".[quantitative]"
```

Check the installation:

```bash
.venv/bin/alambiq --version
.venv/bin/alambiq capability list
```

The list shows the family of each capability. `alambiq capability show assay.fit`
prints the request fields, result, method and limits of one capability.

## Run an example

```bash
.venv/bin/alambiq run examples/kinetics-from-measurements/campaign.yaml --report report.html
```

The command prints one line per step, then `report:` with the path of the report file and
`artifacts:` with the path of the [run directory](concepts.md#runs):

```text
intake                   completed    0.02s
fit                      completed    0.70s
report: report.html
artifacts: /tmp/alambiq-...
```

Open `report.html` in a browser. The report is one HTML file with no external references;
it can be moved and opened without a network connection. Each step has a section with its
result, a chart if the capability defines one, and the notes recorded by the capability.

The rates in this example are generated without noise from Vmax = 180 µM/min and
Km = 2.5 mM, and the fit returns these values to a relative tolerance of 1e-5 (the
tolerance used in `tests/core/test_examples.py`). The rates are synthetic. A successful run shows that the core and the `quantitative` extra are installed.

## Repeated runs

Without `--run-dir`, a run uses a new temporary directory, so no result is reused. Run the
following command twice with the same `--run-dir`:

```bash
.venv/bin/alambiq run examples/kinetics-from-measurements/campaign.yaml --run-dir runs/kinetics
.venv/bin/alambiq run examples/kinetics-from-measurements/campaign.yaml --run-dir runs/kinetics
```

The second run reports every step as `reused`: the key of each step (capability version,
request, and the name and SHA-256 digest of each input file) is unchanged, so the stored
results are used (see [Reuse](concepts.md#reuse)). After any change to
`examples/kinetics-from-measurements/rates.csv`, `intake` is computed again, because the
digest of the file is part of its key, and `fit` is computed again when the record that
`intake` writes has changed. `alambiq history runs/kinetics` lists the runs recorded in
the run directory.

## Docker

The same example runs in a container with a read-only filesystem and no network access:

```bash
docker build -f environments/docker/Dockerfile --build-arg ALAMBIQ_EXTRAS=quantitative -t alambiq:core .
mkdir -p results
docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,mode=1777,size=2g \
  --user "$(id -u):$(id -g)" -v "$PWD:/work:ro" -v "$PWD/results:/results" \
  alambiq:core run examples/kinetics-from-measurements/campaign.yaml \
  --run-dir /results/kinetics --report /results/kinetics.html
```

Engine environments are available as images; the
[Docker notes](../environments/docker/README.md) list the images and the run command.

## Adaptation of the example

Copy the example directory, replace `rates.csv` with a table of the same columns, and
describe the conditions in `campaign.yaml`. If the table is invalid, the run stops at the
step that detects the error, with a message that names the reading, column or unit at
fault. The [measurements page](methods/measurements.md) describes the layout and the
choice of model. The [examples](../examples/README.md) include a protein analysis
(`protein-first-look`), a reaction transformation (`transformation-and-routes`), measured
rounds of variants (`measured-variant-rounds`) and a construct design
(`variant-to-construct`).

## Related documentation

- [Concepts](concepts.md): definitions of capability, family, campaign, step, binding, preset, request, result, record, run, run record, reuse, engine environment, resource and report.
- [Methods](methods/README.md): choosing and reading the capabilities of each method area.
- [Installation](install.md): the engine environments and resources.
- [Python interface](python.md): running campaigns from Python.
