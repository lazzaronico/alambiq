# Alambiq

Alambiq is software that runs computational methods for biotechnology as capabilities.
A campaign, written in YAML or Python, lists steps, and each step runs one capability.
A capability that runs scientific software does so in a pinned engine environment, and one
that needs models or databases declares them as pinned resources. A report is one HTML file;
`alambiq run --report PATH` writes it. The terms are defined in
[Concepts](https://github.com/lazzaronico/alambiq/blob/main/docs/concepts.md).

Version `0.1.1`. The code, documentation and configuration written for
Alambiq are [Apache-2.0](https://github.com/lazzaronico/alambiq/blob/main/LICENSE). Optional software, models and datasets are licensed
under their own terms ([LICENSING.md](https://github.com/lazzaronico/alambiq/blob/main/LICENSING.md)).

## Capabilities

Capabilities are grouped in families (`alambiq capability list` shows the family of
each):

- **Characterize:** protein inspection, search, alignment, profiles, embeddings,
  substitution scoring, stability-change prediction, secondary-structure and
  function-region prediction, protein completion, enzyme candidate retrieval, structure
  prediction, genomic context.
- **Investigate:** reactions and routes, thermodynamics, enzyme-panel evaluation of
  activity prediction, structure comparison and preparation, binding pockets, molecular
  dynamics, metadynamics, trajectory and replica analysis, relative binding free energy,
  QM/MM and reaction free energy.
- **Engineer:** candidate fitting and selection, variant proposal, construct design
  and assembly, coding-sequence recoding.
- **Measure:** measurement intake, fits, and the conversion and activity that a batch
  product target requires.
- **Operate:** definition of rate parameters from measured kinetics, reaction
  simulation, fitting, uncertainty propagation, comparison with measurements, output
  accounting and comparison of input costs.

```bash
alambiq capability list
alambiq capability show assay.fit
```

The first command prints every capability. The second prints the request fields, the
result, the method and the limits of `assay.fit`.

## Reports

`alambiq run --report PATH` writes a report: one HTML file that lists each step with its
result. The first image shows the report of
[kinetics-from-measurements](https://github.com/lazzaronico/alambiq/blob/main/examples/kinetics-from-measurements/README.md): a table of the
readings and a Michaelis-Menten fit with the interval of the curve. The second image shows
the search and alignment steps of the report of
[lanmodulin-comparison](https://github.com/lazzaronico/alambiq/blob/main/examples/lanmodulin-comparison/README.md).

![Report of kinetics-from-measurements: the intake table of 10 readings and the fitted curve with its coefficients](https://raw.githubusercontent.com/lazzaronico/alambiq/main/docs/images/report-kinetics.png)

![Report of lanmodulin-comparison: the search hits and the alignment with conservation per position](https://raw.githubusercontent.com/lazzaronico/alambiq/main/docs/images/report-lanmodulin.png)

## Installation

Install the package from PyPI (Python 3.11 or newer). The `quantitative` extra installs
NumPy, SciPy and scikit-learn:

```bash
pipx install "alambiq[quantitative]"
```

The examples are in the repository. To run one from a clone, install the
[core](https://github.com/lazzaronico/alambiq/blob/main/docs/install.md) in a virtual environment:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[quantitative]"
.venv/bin/alambiq run examples/kinetics-from-measurements/campaign.yaml --report report.html
```

The example fits a Michaelis-Menten curve to a table of 10 initial rates and writes a
report.

Each engine environment contains one of the following scientific packages and the Alambiq
engines that call it: MMseqs2, P2Rank, ThermoMPNN, ESM, OpenMM, Boltz-2, OpenFF, OpenFE,
pymbar, PySCF (QM/MM). Each engine environment except the Apple GPU environment of Boltz-2 is available as a
[Docker image](https://github.com/lazzaronico/alambiq/blob/main/environments/docker/README.md), and each is available as a native environment
built from a recipe. The [installation guide](https://github.com/lazzaronico/alambiq/blob/main/docs/install.md) describes both routes and the
pinned resources. `alambiq env check` reports which engine environments are ready, and
`alambiq resource list` reports which resources are installed.

## Documentation

- [Index](https://github.com/lazzaronico/alambiq/blob/main/docs/README.md): the pages by subject.
- [Getting started](https://github.com/lazzaronico/alambiq/blob/main/docs/getting-started.md): installation of the core and the first run.
- [Installation](https://github.com/lazzaronico/alambiq/blob/main/docs/install.md): extras, engine environments, resources, run
  directories and platforms.
- [Concepts](https://github.com/lazzaronico/alambiq/blob/main/docs/concepts.md): capabilities, families, campaigns, steps, bindings,
  presets, requests, results, records, runs, run records, reuse, engine environments,
  resources and reports.
- [Python interface](https://github.com/lazzaronico/alambiq/blob/main/docs/python.md): campaigns and capabilities from Python.
- [Examples](https://github.com/lazzaronico/alambiq/blob/main/examples/README.md): worked examples.
- [Methods](https://github.com/lazzaronico/alambiq/blob/main/docs/methods/README.md): choosing and reading the capabilities of each method
  area.
- [Reports](https://github.com/lazzaronico/alambiq/blob/main/docs/reports.md): the contents of a report.
- [Licensing](https://github.com/lazzaronico/alambiq/blob/main/LICENSING.md): the licenses of Alambiq and of optional software, models and
  datasets.

## Repository layout

- `src/alambiq/core`: the campaign runner.
- `src/alambiq/capabilities`: the capabilities by family, with their scientific
  computation in functions that tests call directly.
- `src/alambiq/engines`: the engines that run inside engine environments.
- `src/alambiq/report`: the report writer.
- `environments/`: the recipes and verification files of the engine environments, and
  the Dockerfiles in `environments/docker`.
- `examples/`: the runnable examples.
- `tests/core`: the test suite (`python -m pytest tests/core`).

Models, engine environments and databases are installed as described in
[docs/install.md](https://github.com/lazzaronico/alambiq/blob/main/docs/install.md).
