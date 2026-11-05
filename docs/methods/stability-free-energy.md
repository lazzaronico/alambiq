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
orientations, so the same seed gives the same atoms in the same order. The solvated wild type is then
minimized on the CPU (2,000 iterations) before the states are built, since the placement leaves contacts
between water and ions that a short minimization inside the sampler does not remove in a large box.
The minimization sums forces on several threads, so the coordinates of the solvent differ slightly
between runs.

The substitutable residues are Ala, Val, Ile, Leu, Met, Phe, Trp, Tyr, Ser, Thr, Asn, Gln and Cys. The
residue needs two residues on each side. The new side chain starts in one conformation, the template, and
the sampling that follows determines how far the free energy has converged.

`identity_check` builds both states from the wild type. Such a leg has the same Hamiltonian in both states,
so its free energy is zero with a vanishing error, and the overlap between any two windows is the
reciprocal of the number of windows. This is a control for the pipeline that follows.

## Sampling

`simulation.run_mutation_free_energy` builds the hybrid system of one leg with the hybrid-topology factory
of OpenFE and samples it by replica exchange over lambda windows, at the stated temperature and pressure,
with a 2 fs step. Each replica gives a free energy with its MBAR error, the overlap between neighboring
windows, the effective sample size per window, and the estimates from increasing forward and reverse
fractions of the data. The replicas of a run start from the prepared coordinates and differ in the random
velocities and exchange moves. `platform` and `precision` select the hardware; on CUDA, double precision
runs several times slower than mixed precision.

## Independent preparations

An uncertainty needs independent preparations: several runs of `structure.prepare_mutation_inputs` with
different seeds, each followed by a run of its two legs. The seed changes the placement of ions and
water and the starting conformation of the new side chain. `simulation.analyze_stability_change` takes the
records of the legs of all preparations. The mean of a leg is the mean of its preparation means. The
standard error of ΔΔG(fold) combines the standard deviations of the preparation means of the two legs,
and needs 2 or more preparations in each leg. The replicas within one preparation give a smaller spread,
because they share their starting coordinates, and the MBAR error is smaller still.

## Convergence of one substitution

[mutation-free-energy-convergence.json](mutation-free-energy-convergence.json) records Ile4Val in the
miniprotein Trp-cage (1L2Y) from 3 independent preparations at each of two lengths, 12 lambda windows, a
capped tripeptide reference, mixed precision on one GPU. ΔΔG(fold) differed between preparations with a
standard deviation of 1.33 kcal/mol at 1,000 ps per window (values from -1.09 to +1.54) and 0.26 kcal/mol at
2,500 ps per window (values from -0.27 to +0.20). The MBAR error of a leg was 0.04 to 0.10 kcal/mol in every
run, about one tenth of the spread of a single leg across preparations (0.56 to 0.84 kcal/mol at 1,000 ps and
0.3 to 0.6 kcal/mol at 2,500 ps). In a comparison of the tripeptide leg at 500 ps per window, double precision
gave a mean 0.18 kcal/mol below mixed precision with a standard error of 0.16 kcal/mol from 2 runs of each. The
folded leg ran at about 670 ns per day (aggregate over the 12 replicas) in mixed precision and 61 ns per day in
double precision on an NVIDIA GB10. These values describe one small protein and one substitution, and the
record states no comparison with a measured stability.

## Reading the result

`simulation.analyze_stability_change` subtracts the means of the two legs and reports the standard error
from the spread across preparations and the error from MBAR. The overlap, the effective sample sizes and
the agreement of the forward and reverse estimates indicate how far the lambda path has converged; the
checks of the request turn each into a verdict. The defaults of the run (2,500 ps per window, one replica)
follow the record above; a run of 200 ps per window tests the machinery.
