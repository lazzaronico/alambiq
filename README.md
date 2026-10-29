<table>
<tr>
<td><img src="https://raw.githubusercontent.com/lazzaronico/alambiq/main/docs/images/mascot.webp" width="180" alt="Albi, the Alambiq mascot: a robot shaped like an alembic flask"></td>
<td>

# Alambiq

Alambiq is software that runs computational methods for biotechnology as capabilities.
A campaign lists steps, and each step runs one capability. The terms are defined in
[Concepts](https://github.com/lazzaronico/alambiq/blob/main/docs/concepts.md).

Version `0.1.2`. Alambiq is [Apache-2.0](https://github.com/lazzaronico/alambiq/blob/main/LICENSE). Optional software, models and datasets
keep their own licenses ([LICENSING.md](https://github.com/lazzaronico/alambiq/blob/main/LICENSING.md)).

</td>
</tr>
</table>

## Capabilities

Capabilities belong to five families:

- **Characterize:** protein search and alignment, embeddings, stability change, structure
  prediction, genomic context and other descriptions of proteins and enzymes.
- **Investigate:** reactions and thermodynamics, structure preparation, binding pockets,
  molecular dynamics, free energies and QM/MM.
- **Engineer:** candidate selection, variant proposal, construct design and assembly,
  coding-sequence recoding.
- **Measure:** measurement intake and fits.
- **Operate:** reaction simulation from measured kinetics, uncertainty, comparison with
  measurements, output accounting and input costs.

```bash
alambiq capability list
alambiq capability show assay.fit
```

The first command prints every capability with its family. The second prints the request
fields, the result, the method and the limits of `assay.fit`.

## Reports

`alambiq run --report PATH` writes a report: one HTML file with the result of each step.
The images show the reports of
[kinetics-from-measurements](https://github.com/lazzaronico/alambiq/blob/main/examples/kinetics-from-measurements/README.md) and
[lanmodulin-comparison](https://github.com/lazzaronico/alambiq/blob/main/examples/lanmodulin-comparison/README.md).

![Report of kinetics-from-measurements: the intake table of 10 readings and the fitted curve with its coefficients](https://raw.githubusercontent.com/lazzaronico/alambiq/main/docs/images/report-kinetics.png)

![Report of lanmodulin-comparison: the search hits and the alignment with conservation per position](https://raw.githubusercontent.com/lazzaronico/alambiq/main/docs/images/report-lanmodulin.png)

## Installation

Install the package from PyPI (Python 3.11 or newer). The `quantitative` extra adds NumPy,
SciPy and scikit-learn:

```bash
pipx install "alambiq[quantitative]"
```

The examples are in the repository. To run one, install the [core](https://github.com/lazzaronico/alambiq/blob/main/docs/install.md) from
a clone in a virtual environment:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[quantitative]"
.venv/bin/alambiq run examples/kinetics-from-measurements/campaign.yaml --report report.html
```

The example fits a Michaelis-Menten curve to 10 initial rates and writes a report.

Capabilities that run scientific software (MMseqs2, P2Rank, ThermoMPNN, ESM, OpenMM,
Boltz-2, OpenFF, OpenFE, pymbar, PySCF) use engine environments. Each is built natively from
a recipe, and each except the Apple GPU environment of Boltz-2 is also a
[Docker image](https://github.com/lazzaronico/alambiq/blob/main/environments/docker/README.md). `alambiq env check` shows which engine
environments are ready, and `alambiq resource list` shows which resources are installed. The
[installation guide](https://github.com/lazzaronico/alambiq/blob/main/docs/install.md) describes both routes.

## Documentation

- [Index](https://github.com/lazzaronico/alambiq/blob/main/docs/README.md): the pages by subject.
- [Getting started](https://github.com/lazzaronico/alambiq/blob/main/docs/getting-started.md): the core and the first run.
- [Installation](https://github.com/lazzaronico/alambiq/blob/main/docs/install.md): extras, engine environments, resources and platforms.
- [Concepts](https://github.com/lazzaronico/alambiq/blob/main/docs/concepts.md): the terms.
- [Python interface](https://github.com/lazzaronico/alambiq/blob/main/docs/python.md): campaigns and capabilities from Python.
- [Examples](https://github.com/lazzaronico/alambiq/blob/main/examples/README.md): worked examples.
- [Methods](https://github.com/lazzaronico/alambiq/blob/main/docs/methods/README.md): the capabilities of each method area.
- [Reports](https://github.com/lazzaronico/alambiq/blob/main/docs/reports.md): the contents of a report.
- [Licensing](https://github.com/lazzaronico/alambiq/blob/main/LICENSING.md): the licenses of Alambiq and of optional software.

## Repository layout

- `src/alambiq/core`: the campaign runner.
- `src/alambiq/capabilities`: the capabilities by family.
- `src/alambiq/engines`: the code that runs inside engine environments.
- `src/alambiq/report`: the report writer.
- `environments/`: the recipes and verification files of the engine environments, and the
  Dockerfiles in `environments/docker`.
- `examples/`: the runnable examples.
- `tests/core`: the test suite (`python -m pytest tests/core`).
