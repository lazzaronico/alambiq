# Developer scripts

Scripts for checking a release and for reproducing a benchmark. The scripts are run from a
checkout of the repository.
`.github/workflows/ci.yml` runs `check_distribution.py`, `check_getting_started.py`,
`scan_licenses.py` and `check_installed_package.py`. Run a script with
`python tools/NAME.py`. The arguments of `check_distribution.py` (`--keep DIR`) and
`scan_licenses.py` (`--all`) are in their docstrings; `python tools/NAME.py --help` lists
those of `check_installed_package.py` and `benchmark_variant_proposal.py`.

- `check_distribution.py`: builds the wheel and the source distribution and checks their
  contents against the lists in the script. The wheel contains the package, the license and the notices;
  the check rejects tests, examples, documentation, planning material, tools and local run
  directories in the wheel, and secret-like files, run records and caches in either
  distribution. Every file in either distribution is within the size limit stated in the
  script (2,000,000 bytes), except the offline viewer bundle in the wheel. The source
  distribution contains the examples, tests, documentation, environment recipes, tools,
  license and notices. The script then installs the wheel into a new environment outside
  the checkout and lists its capabilities.
- `check_getting_started.py`: runs the commands of `docs/getting-started.md` in a copy of the
  repository with a new virtual environment, and checks the version, the number of
  capabilities stated in that guide, that the report is self-contained and that a second
  run reuses every step.
- `scan_licenses.py`: classifies the license of every dependency of the installed `alambiq`
  distribution and its extras, followed through their requirements, as allowed, weak
  copyleft, unknown or refused. It lists the dependencies that are not classed as allowed,
  and every dependency with `--all`, followed by a count line. The exit status is 1 when a
  license is refused or cannot be determined.
- `check_installed_package.py --wheel WHEEL --receipt FILE`: runs the tests against the
  installed wheel and writes to FILE which tests were skipped and the file each module was
  imported from. With `--require-extras` it also imports numpy, scipy, sklearn, rdkit,
  doranet, equilibrator_api, gemmi, Bio, pydna, dnachisel, mdtraj, openmm, pdbfixer and
  umap. CI runs it with `--require-extras`.
- `benchmark_variant_proposal.py`: reproduces the benchmark in
  [Benchmark on measured variants](../docs/methods/variant-proposal.md#benchmark-on-measured-variants).
  It requires `--studies` (the path to a checkout of the `alambiq-studies` repository, which
  holds the glucokinase data and the lipase sequence manifest), `--out` (the JSON file to
  write; the file in the repository is `docs/methods/variant-proposal-benchmark.json`) and
  the environment variable `ALAMBIQ_ENV_ESM` (the Python of the `esm` engine environment).
  The lipase measurements are read from `tests/fixtures/variant-benchmark/`. The
  transaminase data are read from the Git history of this repository at the commit given
  by `--transaminase-commit` (default `39f0515`). `--only` restricts the run to the listed
  dataset ids.
