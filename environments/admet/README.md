# ADMET-AI engine environment

The [Mamba recipe](environment.yaml) installs Python 3.12 and ADMET-AI 2.0.1 from PyPI, which
brings Chemprop, PyTorch, RDKit, Lightning and pandas. ADMET-AI is MIT licensed and its
wheel contains the trained models, the table of endpoints and the DrugBank reference set, so
the environment needs no other download. Create it separately from the Alambiq core
environment, then set `ALAMBIQ_ENV_ADMET` to its Python executable.

```bash
mamba env create --prefix /path/to/admet-env --file environments/admet/environment.yaml
export ALAMBIQ_ENV_ADMET=/path/to/admet-env/bin/python
alambiq env check admet
```

`ALAMBIQ_ENV_ADMET` is the engine environment of `molecule.predict_admet`. The environment
takes about 1.4 GB on disk, most of it PyTorch.

The [Apple Silicon requirements list](requirements.osx-arm64.txt) records the installed
packages: ADMET-AI 2.0.1, Chemprop 2.3.1, PyTorch 2.15.0, RDKit 2026.3.6, NumPy 2.5.4 and
pandas 3.1.0.

## Check

The package holds the predictions that ADMET-AI made for 2,845 approved drugs. Run on the first
200 of them in this environment, the engine reproduces every endpoint to within 3e-5. The
tests of `molecule.predict_admet` repeat this check.
