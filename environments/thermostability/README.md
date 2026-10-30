# TemBERTure engine environment

The [Mamba recipe](environment.yaml) installs Python 3.11, PyTorch, Transformers, Adapters and
NumPy. `protein.predict_melting_temperature` runs TemBERTure Tm in it. Create the environment
separately from the Alambiq core environment, provision the model resource, and set
`ALAMBIQ_ENV_THERMOSTABILITY` to the Python executable:

```bash
mamba env create --prefix /path/to/thermostability-env --file environments/thermostability/environment.yaml
export ALAMBIQ_ENV_THERMOSTABILITY=/path/to/thermostability-env/bin/python
alambiq env check thermostability
alambiq resource show temberture-tm
alambiq resource provision temberture-tm --yes
```

The [resource record](../../src/alambiq/resources/temberture-tm.json) pins 17 files of about
1.7 GB: the ProtBERT-BFD base model (Rostlab/prot_bert_bfd, commit
`6c5c8a55a52ff08a664dfd584aa1773f125a0487`, downloaded from Hugging Face) and the adapter and
head of each of the three TemBERTure Tm replicas (ibmm-unibe-ch/TemBERTure, commit
`3c17a7a2d2a8365b187f5dec4eb1ed5db6d37f41`, downloaded from GitHub). `provision` states the
download and does nothing without `--yes`. Before each prediction the capability checks every
file against the record. `ALAMBIQ_RESOURCE_HOME` sets the location of the resource store.

The [requirements list](requirements.osx-arm64.txt) records the packages of the Apple Silicon
installation: PyTorch 2.15.0, Transformers 4.57.6, Adapters 1.3.0 and NumPy 2.4.6. The
TemBERTure repository pins Adapters 0.1.1 and PyTorch 1.10.2; the capability runs the files
of the authors in newer versions, and its tests check that they load and predict
deterministically.
