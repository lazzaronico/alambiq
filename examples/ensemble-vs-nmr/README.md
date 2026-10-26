# Ensemble versus NMR

**Question.** How do scalar couplings computed from a simulated ensemble differ from
measured scalar couplings?

The protein is GB3 (PDB 1P7E). `structure.prepare_protein_system` prepares the protein.
Three steps of `simulation.run_molecular_dynamics` run one replica each, with different
seeds. `simulation.compare_scalar_couplings` computes the 225 published three-bond
couplings from every frame and compares them with the measurements.

```bash
export ALAMBIQ_ENV_OPENMM=/path/to/openmm/bin/python   # see docs/install.md
alambiq run examples/ensemble-vs-nmr/campaign.yaml --report report.html
```

The campaign uses the default platform, CPU, with the default of 1 thread per replica.
`platform: CUDA` in the request of `simulation.run_molecular_dynamics` selects a GPU. The
solvent is implicit (GBn2), and each replica runs 20 ps after 10 ps of thermalization.

**Needs.** The OpenMM environment (`ALAMBIQ_ENV_OPENMM`) and the `trajectory` and
`dossier` extras.

**What to look at.** On the CPU platform with 1 thread per replica, the 20 ps run gave an
RMSE of 0.87 Hz and a mean absolute error of 0.57 Hz over the 225 couplings. With more
threads or on another platform the trajectories differ and the results agree in
distribution. The assignment of HB2 and HB3 by geometry
(`atom_naming: iupac-beta-methylene`) exchanged the labels of the beta-methylene hydrogens
of 8 residues. The report shows the residual of each coupling against the experimental error
reported for it, together with the split R-hat and the effective sample size of the
coupling.

**Limits.** A run of 20 ps samples the neighborhood of the starting structure, and agreement
between replicas applies to that neighborhood. A residual combines contributions from the
force field, the sampling, the Karplus relation and the measurement. The results apply to
implicit solvent (GBn2) and replicas of 20 ps.

**Data.** `1P7E.cif` is wwPDB (CC0). The couplings are from Vögeli, Olsson, Riek and
Güntert (2015), doi:10.1016/j.dib.2015.08.020, CC BY 4.0, from the primary `GB3.cco`
file of the supplementary data. The campaign restates each coupling with its atoms,
Karplus coefficients, value and reported error.
