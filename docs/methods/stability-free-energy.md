# Free energy of an amino-acid substitution

The capabilities are `structure.prepare_mutation_inputs`, `simulation.run_mutation_free_energy` and
`simulation.analyze_stability_change`. They estimate the change in the free energy of folding,
ΔΔG(fold), when one residue of a folded protein is replaced by another. A positive value means that the
substitution raises the folding free energy and destabilizes the folded state; a negative value
stabilizes it. The value is the result of a calculation under a force field and a sampling protocol. The
diagnostics that accompany it (overlap, replica spread, forward and reverse estimates) describe the
sampling.

## Thermodynamic cycle

The substitution is carried out as an alchemical transformation, from the wild-type residue to the mutant
residue, in two environments:

- the folded leg, the protein in a water box;
- the reference leg, a tripeptide of the residue and its two neighbors on each side, in a smaller water
  box. The outer two residues are capped as acetyl and N-methylamide, so that the termini carry no charge.

ΔΔG(fold) is the free energy of the transformation in the folded leg minus that in the reference leg. The
reference tripeptide stands for the unfolded state, so the value depends on that choice.

## Preparation

`structure.prepare_mutation_inputs` reads the first model of a PDB file and takes one chain. It adds
hydrogens at the stated pH, solvates the wild type in a dodecahedral TIP3P box with sodium and chloride
ions at the ionic strength, and parameterizes it with ff14SB. It builds the mutant state by replacing the
side chain with the standard template of the new residue. The backbone atoms N, CA, C and O, and the atoms
that follow from CA by bonds to atoms of the same name and element in both side chains, are mapped between
the two states. The hydrogens of each mapped atom map in name order. The remaining atoms are present in one
state only. Both states carry the same net charge. A request seeds the random placement of ions and
orientations, so the same seed gives the same system.

The substitutable residues are Ala, Val, Ile, Leu, Met, Phe, Trp, Tyr, Ser, Thr, Asn, Gln and Cys. The
residue needs two residues on each side. The new side chain starts in one conformation, the template, and
the sampling that follows determines how far the free energy has converged.

`identity_check` builds both states from the wild type. Such a leg has the same Hamiltonian in both states,
so its free energy is zero with a vanishing error, and the overlap between any two windows is the
reciprocal of the number of windows. This is a control for the pipeline that follows.

## Sampling

`simulation.run_mutation_free_energy` builds the hybrid system of one leg with the hybrid-topology factory
of OpenFE and samples it by replica exchange over lambda windows, at the stated temperature and pressure,
with a 2 fs step. Each of the independent replicas gives a free energy with its MBAR error, the overlap
between neighboring windows, the effective sample size per window, and the estimates from increasing
forward and reverse fractions of the data. `platform` and `precision` select the hardware; on CUDA, double
precision runs several times slower than mixed precision.

## Reading the result

`simulation.analyze_stability_change` subtracts the means of the two legs and reports two uncertainties:
the standard error from the spread of the replicas, and the error from MBAR. The MBAR error assumes
uncorrelated samples and is smaller than the spread between independent replicas when the sampling has not
converged. The overlap, the effective sample sizes and the agreement of the forward and reverse estimates
indicate how far the lambda path has converged; the checks of the request turn each into a verdict.

A run of a few hundred picoseconds per window tests the machinery. A free energy that is stated with its
uncertainty needs several replicas of a longer run, and the replica spread decides how many.
