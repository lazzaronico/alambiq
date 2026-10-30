# Molecular properties engine environment

The [Mamba recipe](environment.yaml) installs Python 3.12, PySCF 2.14.0, RDKit and PyBerny.
PySCF computes the density-functional energy, the optimized geometry, the Hessian and the
excitations, RDKit builds the starting 3D structure from SMILES, and PyBerny drives the
geometry optimization. Create the environment separately from the Alambiq core environment,
then set `ALAMBIQ_ENV_MOLECULAR_PROPERTIES` to its Python executable.

```bash
mamba env create --prefix /path/to/molecular-properties-env --file environments/molecular-properties/environment.yaml
export ALAMBIQ_ENV_MOLECULAR_PROPERTIES=/path/to/molecular-properties-env/bin/python
alambiq env check molecular_properties
```

`ALAMBIQ_ENV_MOLECULAR_PROPERTIES` is the engine environment of
`molecule.compute_quantum_properties`. The environment holds no model weights or data files.

The [Apple Silicon explicit lock](osx-arm64.explicit.txt) lists the 131 packages of the recorded
installation:

```sh
mamba create --prefix /path/to/molecular-properties-env --file environments/molecular-properties/osx-arm64.explicit.txt
```

It records PySCF 2.14.0, RDKit 2026.9.1, PyBerny 0.7.0, NumPy 2.5.4 and SciPy 1.18.1.

## Settings that the capability fixes

- The basis functions are Cartesian, the convention of Gaussian for the Pople bases. With
  spherical d functions, water at B3LYP/6-31G* is 1.9 millihartree higher.
- The integration grid is level 4. At PySCF's default level 3 the gradients are noisy enough that
  the geometry optimization of acetic acid does not converge in 60 steps, and at level 4 it
  converges in 4.
- The temperature is a request field, and the pressure of the thermochemistry is 101,325 Pa.

## Check

Water at B3LYP/6-31G* in this environment has an energy of -76.40895 hartree, an ideal-gas heat
capacity at constant volume of 5.997 cal/(mol K) at 298.15 K and a harmonic zero-point energy of
13.28 kcal/mol. The tests of `molecule.compute_quantum_properties` check these values.
