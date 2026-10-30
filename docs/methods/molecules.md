# Molecules

`molecule.compute_quantum_properties` computes the electronic properties of one small
molecule with density-functional theory: frontier orbital energies, thermochemistry, vertical
excitations, and the dipole moment with atomic charges. The molecule is a SMILES. It can be
charged, in a spin state above the singlet, and in water.

## Method

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

## Water

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

## Checks

Water at B3LYP/6-31G* has an energy of -76.40895 hartree, the value of Gaussian with
Cartesian d functions, a zero-point energy of 13.28 kcal/mol and a heat capacity at constant
volume of 5.997 cal/(mol K) at 298.15 K. The tests check these values, the rotational symmetry
numbers of methane (12), benzene (12), carbon dioxide (2) and acetic acid (1), the spin
expectation values of the hydroxyl radical, methyl radical and triplet oxygen, and the refusal
of impossible spin states and of molecules above the size limit.

## Limits and run time

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
