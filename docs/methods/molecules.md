# Molecules

Three capabilities cover small molecules given as SMILES. `molecule.compute_quantum_properties`
computes the electronic properties of one molecule with density-functional theory.
`molecule.predict_admet` estimates absorption, distribution, metabolism, excretion and
toxicity endpoints of thousands of molecules with trained models.
`molecule.compute_hydration_free_energy` computes the free energy of transferring one
molecule from the gas phase into water by alchemical simulation.

## Quantum-chemical properties

`molecule.compute_quantum_properties` computes frontier orbital energies, thermochemistry,
vertical excitations, and the dipole moment with atomic charges. The molecule can be charged,
in a spin state above the singlet, and in water.

### Method

RDKit embeds 10 conformers from the request's `seed` and keeps the one of lowest MMFF94
energy. PySCF optimizes it with PyBerny at the functional and basis of the request, on an
integration grid of level 4, and the optimization must converge: a geometry that has not
converged returns no properties. The basis functions are Cartesian, the convention of
Gaussian for the Pople bases. The functionals are B3LYP, PBE0, PBE and CAM-B3LYP; the bases
are 6-31G*, 6-31G(2df,p), def2-SVP and def2-TZVP.

The properties come from the optimized geometry:

- Orbitals: the energies of the highest occupied and lowest unoccupied orbitals and their gap.
- Charges: Mulliken charges and the dipole moment about the center of mass. For an open-shell
  molecule the result adds the spin population of each atom.
- Thermochemistry: the analytic Hessian with the harmonic oscillator, rigid rotor and ideal
  gas approximations at the requested temperature and 101,325 Pa. The result gives zero-point
  energy, enthalpy, Gibbs energy, entropy and both heat capacities. Modes with an imaginary
  frequency are counted and left out.
- Excitations: vertical excitation energies of the lowest singlet or spin-conserving states
  from time-dependent density-functional theory, with wavelengths and oscillator strengths.

A molecule with `multiplicity` 1 uses restricted Kohn-Sham theory. A higher multiplicity uses
unrestricted theory and reports S squared against S(S+1) of a pure spin state; a difference
above 10 % of S(S+1) adds a note. The charge is the formal charge of the SMILES. A radical in
the SMILES requires a multiplicity above 1, and a multiplicity that the electron count cannot
have is refused.

### Water

With `solvent` set to `water`, the optimization and every property use the IEF-PCM
polarizable continuum with a relative permittivity of 78.3553. The result adds the
polarization energy: the energy in the continuum minus the gas-phase energy at the same
geometry.

That energy is the electrostatic part of solvation only. It has no cavity or dispersion term,
so it is not a hydration free energy. `tools/validate_solvent_polarization.py` compares it
with the hydration free energies of FreeSolv for six molecules at B3LYP/6-31G*
([solvent-polarization.json](solvent-polarization.json)):

| Molecule | Polarization energy (kcal/mol) | Experiment (kcal/mol) | Difference (kcal/mol) |
|---|---:|---:|---:|
| water | -6.53 | -6.32 | -0.21 |
| methanol | -4.50 | -5.11 | +0.61 |
| ethanol | -4.37 | -5.01 | +0.64 |
| acetone | -4.94 | -3.85 | -1.09 |
| methane | -0.30 | 1.99 | -2.29 |
| benzene | -2.48 | -0.86 | -1.62 |

Polar molecules come within about 1 kcal/mol of the experimental value. Nonpolar molecules
do not: benzene differs by 1.6 kcal/mol, and methane has a negative polarization energy
against a positive hydration free energy, because the cavity and dispersion terms that decide
their solvation are absent. Use the polarization energy to compare the
electrostatic response of molecules and to relax a geometry or an excited state in water. A
hydration free energy needs a method that includes the cavity and dispersion terms.

### Checks

Water at B3LYP/6-31G* has an energy of -76.40895 hartree, the value of Gaussian with
Cartesian d functions, a zero-point energy of 13.28 kcal/mol and a heat capacity at constant
volume of 5.997 cal/(mol K) at 298.15 K. The tests check these values, the rotational symmetry
numbers of methane (12), benzene (12), carbon dioxide (2) and acetic acid (1), the spin
expectation values of the hydroxyl radical, methyl radical and triplet oxygen, and the refusal
of impossible spin states and of molecules above the size limit.

### Limits and run time

- A molecule of 1 to 12 heavy atoms from H, C, N, O, F, P, S, Cl and Br is accepted.
- The geometry is the lowest-energy conformer of 10, which need not be the global minimum.
  Properties of a flexible molecule depend on the conformer.
- Vibrations are harmonic and internal rotations count as vibrations, which affects the
  entropy and heat capacity of flexible molecules.
- The symmetry number comes from the point group that PySCF detects at the optimized geometry
  with a tolerance of 0.003 angstrom.
- Excitation energies are vertical. B3LYP and PBE describe charge-transfer and Rydberg states
  poorly. In water the continuum is equilibrated with the ground state.
- Mulliken charges depend strongly on the basis set.
- An unrestricted calculation can converge to a state other than the lowest of its
  multiplicity.

The run times below are for all four properties at B3LYP/6-31G* in the gas phase, on an
Apple M-series laptop with 8 threads. The time grows steeply with the number of atoms and
with the optimization steps that a molecule needs.

| Molecule | Heavy atoms | Optimization steps | Optimization | Thermochemistry | Excitations | Total |
|---|---:|---:|---:|---:|---:|---:|
| acetic acid | 4 | 4 | 8 s | 20 s | 11 s | 44 s |
| phenol | 7 | 4 | 20 s | 73 s | 53 s | 150 s |
| acetanilide | 10 | 23 | 229 s | 236 s | 134 s | 606 s |
| paracetamol | 11 | 24 | 317 s | 328 s | 167 s | 822 s |

## ADMET estimates

`molecule.predict_admet` returns 52 estimates for each of 1 to 10,000 molecules: 11
physicochemical descriptors that RDKit computes (molecular weight, logP, hydrogen bond
acceptors and donors, the Lipinski count, QED, stereo centers, TPSA, and the PAINS, BRENK
and NIH alert counts) and 41 endpoints that ADMET-AI predicts. Of the predicted endpoints, 31
are classifications, whose estimate is a probability from 0 to 1, and 10 are regressions in
the units of the endpoint table. They cover absorption (intestinal absorption, bioavailability,
solubility, lipophilicity, hydration free energy, permeability and P-glycoprotein
inhibition), distribution (blood-brain barrier, plasma protein binding and volume of
distribution), metabolism (five CYP inhibition endpoints and three substrate endpoints),
excretion (half life and two clearances) and toxicity (hERG, ClinTox, Ames mutagenicity, DILI,
carcinogenicity, acute toxicity LD50, skin reaction and the 12 Tox21 assays).

ADMET-AI 2.0.1 (MIT) predicts each endpoint with the mean of 5 Chemprop message-passing
networks. The networks were trained on datasets of the Therapeutics Data Commons, from 278
molecules (carcinogenicity) to 13,130 (CYP2D6 inhibition). The result holds an endpoint table
with the kind, units, species and training size of each endpoint and the metrics that
ADMET-AI reports for it: AUROC and AUPRC for classifications, R squared and the mean absolute
error for regressions. With `drugbank_percentiles`, each estimate is also placed among the
values of about 2,800 approved drugs from DrugBank, which ADMET-AI ships.

Equivalent SMILES of one molecule are predicted once. A SMILES that RDKit cannot read refuses
the whole request and the message names the positions.

### Checks

The ADMET-AI package stores its own predictions for 2,845 approved drugs. The tests run the
capability on the first 20 of them and require every endpoint to agree within 0.001; the
largest difference observed over the first 200 is 3e-5. The test also checks the RDKit
descriptors of aspirin and that every classification output lies between 0 and 1.

### Limits and run time

- The training sets are public datasets of mostly drug-like molecules. A metabolite, a sugar
  or a small industrial chemical can lie outside what the networks have seen, and the models
  give no warning when it does.
- The reported metrics are those of ADMET-AI's own table, not a test of this capability on
  other data. They differ by endpoint: the reported AUROC ranges from 0.63 for CYP2C9
  substrate to 0.99 for intestinal absorption, and the reported R squared of half life
  (-2.39) and volume of distribution (-1.21) is negative, which means that the estimate
  is a poorer predictor than the mean of the test values.
- A salt, a mixture or an unusual protonation state is predicted as written, because
  ADMET-AI does not standardize molecules.
- The hydration free energy is a model trained on FreeSolv, with a reported mean absolute error
  of 0.78 kcal/mol; it is an estimate, not a simulation.
- The engine takes about 25 s for 2,845 molecules, including the loading of the models, on an
  Apple M-series laptop. The environment is set by `ALAMBIQ_ENV_ADMET`.

## Hydration free energy

`molecule.compute_hydration_free_energy` returns the hydration free energy of one neutral
molecule with the OpenFE absolute solvation protocol. A negative value means that the molecule
prefers water. The free energy is from the gas phase to water at the same concentration.

RDKit embeds the molecule from its SMILES and relaxes it with MMFF94. The molecule has AshGC
partial charges, the graph-network model of the OpenFF project that reproduces AM1-BCC charges,
and Sage 2.3.0 parameters; the water is TIP3P in a dodecahedral box with 1.5 nm of padding. In
water and in vacuum, 14 lambda windows decouple the molecule from its surroundings, with the
electrostatics first and the Lennard-Jones interactions after, and Hamiltonian replica exchange
samples them. MBAR gives the free energy of each leg, and the hydration free energy of a repeat
is the vacuum value minus the solvent value. The result is the mean of `repeats` independent
repeats. Its uncertainty is the spread between the repeats, with the legs combined as OpenFE
combines them, and the result also gives the MBAR error, the smallest overlap between
neighboring windows and the smallest effective sample size of each leg.

The solvent leg samples `production_ns` after `equilibration_ns`; the vacuum leg samples
one fifth and half of those lengths, the ratios of the OpenFE defaults. The defaults
are 10 ns and 1 ns with 3 repeats, which takes hours on a GPU: a run of methane with 0.05 ns
of production and 0.05 ns of equilibration took 160 s on an NVIDIA GB10. A run shorter than 5 ns of
production or with fewer than 3 repeats carries a note, as does an overlap below 0.03.

The molecule must be neutral, closed-shell and one connected species from H, C, N, O, F, P,
S, Cl, Br and I, with at most 50 heavy atoms and specified stereochemistry. Ions, salts and
zwitterions are refused.

### Accuracy

The accuracy of the capability against measured hydration free energies is being checked; this
section holds the result when the check completes.

### Platforms

The capability runs on CUDA by default and the OpenFE environment (`ALAMBIQ_ENV_OPENFE`) is the
one of `simulation.run_relative_free_energy`. On the CPU of an Apple M-series laptop, the solvent
leg of one run of methane stopped with a numerical failure (NaN energies) in the first
replica-exchange iteration, with both short and default non-alchemical equilibration. The same
molecule completed on CUDA. The CPU platform is therefore available but not verified. A box of 2,661 water atoms ran at 64
ns per day on that CPU with a 2 fs step, so a default run would take days. On the CPU, the vacuum leg runs on one thread, as OpenFE sets it, because a tiny system
runs about ten times slower with eight threads.
