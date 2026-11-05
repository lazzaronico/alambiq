# Protonation engine environment

The [Mamba recipe](environment.yaml) installs Python 3.11, PDB2PQR 3.7.1 and PROPKA 3.5.1.
`structure.assign_protonation` runs both in it. The environment is pure Python and installs
in seconds. Create it separately from the Alambiq core environment and set
`ALAMBIQ_ENV_PROTONATION` to the Python executable:

```bash
mamba env create --prefix /path/to/protonation-env --file environments/protonation/environment.yaml
export ALAMBIQ_ENV_PROTONATION=/path/to/protonation-env/bin/python
alambiq env check protonation
```

The [requirements list](requirements.osx-arm64.txt) records the packages of the Apple Silicon
installation. PDB2PQR pins docutils below 0.18; NumPy is the only numerical dependency.

## Licensing

- PDB2PQR 3.7.1: BSD 3-Clause (Nielsen and Baker; Battelle Memorial Institute).
- PROPKA 3.5.1: LGPL 2.1. It runs as a separate process in this environment and Alambiq does
  not import it. Distributing the environment is subject to the terms of the LGPL.
- docutils 0.17.1: public domain, Python and 2-clause BSD licenses; its distribution also
  contains files under the GPL 3 that PDB2PQR does not use.
- NumPy, requests, certifi, urllib3, idna, charset-normalizer and mmcif-pdbx: permissive
  licenses.

## References

- Jurrus E, et al. Improvements to the APBS biomolecular solvation software suite. Protein
  Sci 27, 112-128 (2018).
- Dolinsky TJ, et al. PDB2PQR: expanding and upgrading automated preparation of biomolecular
  structures for molecular simulations. Nucleic Acids Res 35, W522-W525 (2007).
- Olsson MHM, et al. PROPKA3: consistent treatment of internal and surface residues in
  empirical pKa predictions. J Chem Theory Comput 7, 525-537 (2011).
